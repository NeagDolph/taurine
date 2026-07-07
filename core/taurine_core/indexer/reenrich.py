from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from taurine_core import config
from taurine_core.ports import LLMPort
from taurine_core.schemas import ReenrichStatus

from .enrich import enrich_note_with_status
from .profile import build_user_card
from .sync import _embedding_units


@dataclass(frozen=True)
class ReenrichReport:
    total: int
    changed: int
    failures: int
    profile_failed: bool = False


async def reenrich_all(
    index: object,
    llm: LLMPort,
    progress: Callable[[ReenrichStatus], None],
    source: str | None = None,
) -> ReenrichReport:
    """Re-enrich every note, or only one connector's notes when `source` is set.

    Scoped runs skip the profile phase: the user card is corpus-wide and only
    worth regenerating on a full pass.
    """
    notes = [
        note
        for note in index.notes()  # type: ignore[attr-defined]
        if source is None or note.source == source
    ]
    total = len(notes)

    profile_failed = False
    if source is None:
        progress(ReenrichStatus(running=True, phase="profile", done=0, total=total))
        old_card = index.user_card()  # type: ignore[attr-defined]
        try:
            user_card = await build_user_card(index, llm)
            index.set_user_card(user_card)  # type: ignore[attr-defined]
        except Exception:
            profile_failed = True
            user_card = old_card
    else:
        user_card = index.user_card()  # type: ignore[attr-defined]

    progress(ReenrichStatus(running=True, phase="notes", done=0, total=total))

    semaphore = asyncio.Semaphore(config.REENRICH_CONCURRENCY)
    changed_note_ids: list[str] = []
    done = 0
    failures = 0

    async def one(note: object) -> None:
        nonlocal done, failures
        async with semaphore:
            enrichment, failed = await enrich_note_with_status(
                note,  # type: ignore[arg-type]
                llm,
                user_card=user_card,
            )
        if failed:
            failures += 1
        else:
            index.cache_enrichment(  # type: ignore[attr-defined]
                note.note_id,
                note.content_hash,
                enrichment,
            )
            index.upsert_note(  # type: ignore[attr-defined]
                note,
                enrichment,
                _embedding_units(note, enrichment),  # type: ignore[arg-type]
            )
            changed_note_ids.append(note.note_id)
        done += 1
        progress(ReenrichStatus(running=True, phase="notes", done=done, total=total))

    await asyncio.gather(*(one(note) for note in notes))

    progress(ReenrichStatus(running=True, phase="finalizing", done=done, total=total))
    for note_id in changed_note_ids:
        index.rebuild_for(note_id)  # type: ignore[attr-defined]
    index.load()  # type: ignore[attr-defined]
    progress(ReenrichStatus(running=False, phase="idle", done=done, total=total))

    return ReenrichReport(
        total=total,
        changed=len(changed_note_ids),
        failures=failures,
        profile_failed=profile_failed,
    )
