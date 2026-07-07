from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from taurine_core.ports import LLMPort
from taurine_core.schemas import Enrichment, NoteRecord, ReenrichStatus

from .batch_enrich import enrich_all_batched
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
    disabled = getattr(index, "disabled_sources", set)()
    notes = [
        note
        for note in index.notes()  # type: ignore[attr-defined]
        if (source is None or note.source == source) and note.source not in disabled
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

    changed_note_ids: list[str] = []
    done = 0
    failures = 0

    typed_notes: list[NoteRecord] = notes  # type: ignore[assignment]
    notes_by_id: dict[str, NoteRecord] = {note.note_id: note for note in typed_notes}

    def on_note_done(note_id: str, enrichment: Enrichment, failed: bool) -> None:
        nonlocal done, failures
        note = notes_by_id[note_id]
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

    await enrich_all_batched(
        typed_notes,
        llm,
        user_card,
        _embed_texts,
        on_note_done,
    )

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


def _embed_texts(texts: list[str]) -> np.ndarray:
    from . import sync

    return sync.embed_texts(texts)
