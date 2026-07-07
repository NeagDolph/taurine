from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import ConnectorSyncReport, Enrichment, NoteRecord

from .embed import chunk_text, embed_texts
from .enrich import enrich_note_with_status
from .notes import extract_notes
from .store import EmbeddingInput, Index

T = TypeVar("T")
NotesSource = Callable[[], Iterable[NoteRecord]] | Iterable[NoteRecord]


@dataclass(frozen=True)
class SyncReport:
    added: int
    changed: int
    removed: int
    enrich_failures: int


def sync_once(index: Index, notes_source: NotesSource, llm: LLMPort) -> SyncReport:
    report = sync_source(index, "apple_notes", _notes_from(notes_source), llm)
    return SyncReport(
        added=report.added,
        changed=report.changed,
        removed=report.removed,
        enrich_failures=report.enrich_failures,
    )


SyncProgress = Callable[[str, int, int], None]
_LOG = logging.getLogger("taurine.session")


def sync_source(
    index: Index,
    source_id: str,
    records: Iterable[NoteRecord],
    llm: LLMPort,
    progress: SyncProgress | None = None,
) -> ConnectorSyncReport:
    def report(phase: str, done: int, total: int) -> None:
        if progress is None:
            return
        try:
            progress(phase, done, total)
        except Exception:
            pass

    incoming = [note.model_copy(update={"source": source_id}) for note in records]
    incoming_by_id = {note.note_id: note for note in incoming}
    existing_by_id = {note.note_id: note for note in index.notes(source_id)}

    removed_ids = sorted(set(existing_by_id) - set(incoming_by_id))
    for note_id in removed_ids:
        index.delete_note(note_id)

    pending: list[NoteRecord] = []
    added = 0
    changed = 0
    for note in incoming:
        existing = existing_by_id.get(note.note_id)
        if existing is not None and existing.content_hash == note.content_hash:
            continue
        if existing is None:
            added += 1
        else:
            changed += 1
        pending.append(note)

    _LOG.info(
        "sync %s: %d extracted, %d to index (%d new, %d changed), %d removed",
        source_id, len(incoming), len(pending), added, changed, len(removed_ids),
    )
    report("enriching", 0, len(pending))

    user_card = index.user_card()
    enrich_failures = 0
    cached: dict[str, Enrichment] = {}
    needs_llm: list[NoteRecord] = []
    for note in pending:
        hit = index.cached_enrichment(note.note_id, note.content_hash)
        if hit is not None:
            cached[note.note_id] = hit
        else:
            needs_llm.append(note)

    fresh: dict[str, tuple[Enrichment, bool]] = {}
    if needs_llm:
        fresh = _run_async(
            _enrich_concurrently(
                needs_llm,
                llm,
                user_card,
                source_id,
                lambda done: report("enriching", done, len(pending)),
                already_done=len(cached),
            )
        )

    rebuilt: list[str] = []
    for position, note in enumerate(pending, start=1):
        if note.note_id in cached:
            enrichment = cached[note.note_id]
        else:
            enrichment, failed = fresh[note.note_id]
            enrich_failures += int(failed)
            if not failed:  # fallback enrichments must retry on the next sync
                index.cache_enrichment(note.note_id, note.content_hash, enrichment)
        index.upsert_note(note, enrichment, _embedding_units(note, enrichment))
        rebuilt.append(note.note_id)
        report("indexing", position, len(pending))

    report("finalizing", len(pending), len(pending))
    for note_id in rebuilt:
        index.rebuild_for(note_id)
    index.load()
    _LOG.info(
        "sync %s: done — +%d new, ~%d changed, -%d removed, %d enrich failures",
        source_id, added, changed, len(removed_ids), enrich_failures,
    )
    return ConnectorSyncReport(
        connector_id=source_id,
        added=added,
        changed=changed,
        removed=len(removed_ids),
        enrich_failures=enrich_failures,
    )


async def _enrich_concurrently(
    notes: list[NoteRecord],
    llm: LLMPort,
    user_card: str | None,
    source_id: str,
    on_progress: Callable[[int], None],
    already_done: int,
) -> dict[str, tuple[Enrichment, bool]]:
    from taurine_core import config

    semaphore = asyncio.Semaphore(config.REENRICH_CONCURRENCY)
    done = already_done
    lock = asyncio.Lock()
    results: dict[str, tuple[Enrichment, bool]] = {}

    async def one(note: NoteRecord) -> None:
        nonlocal done
        async with semaphore:
            result = await enrich_note_with_status(note, llm, user_card=user_card)
        results[note.note_id] = result
        async with lock:
            done += 1
            current = done
        on_progress(current)
        if current % 5 == 0 or current == already_done + len(notes):
            _LOG.info("sync %s: enriched %d/%d", source_id, current, already_done + len(notes))

    await asyncio.gather(*(one(note) for note in notes))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync Apple Notes into the Taurine index.")
    parser.add_argument("--db", type=Path, default=Path("taurine.db"))
    args = parser.parse_args(argv)

    index = Index(args.db)
    report = sync_once(index, extract_notes, _cli_llm())
    print(report)
    return 0


def _embedding_units(note: NoteRecord, enrichment: Enrichment) -> list[EmbeddingInput]:
    texts: list[str] = []
    kinds: list[str] = []
    chunks = chunk_text(note.text) or [note.text or note.title]
    for chunk in chunks:
        if chunk.strip():
            texts.append(chunk)
            kinds.append("chunk")
    if enrichment.gist.strip():
        texts.append(enrichment.gist)
        kinds.append("gist")
    for utterance in enrichment.utterances:
        if utterance.strip():
            texts.append(utterance)
            kinds.append("utterance")

    vectors = embed_texts(texts)
    units: list[EmbeddingInput] = []
    for kind, text, vector in zip(kinds, texts, vectors, strict=True):
        units.append((kind, text, np.asarray(vector, dtype=np.float32)))  # type: ignore[arg-type]
    return units


def _notes_from(source: NotesSource) -> Iterable[NoteRecord]:
    return source() if callable(source) else source


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[T] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - defensive event-loop bridge
            error.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


class _UnavailableLLM:
    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        del model, system, user, timeout_s
        raise LLMError("taurine_core.llm.OpenRouterClient is unavailable")


def _cli_llm() -> LLMPort:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        return _UnavailableLLM()
    from taurine_core.llm import OpenRouterClient

    return OpenRouterClient(api_key)


if __name__ == "__main__":
    raise SystemExit(main())
