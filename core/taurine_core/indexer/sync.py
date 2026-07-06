from __future__ import annotations

import argparse
import asyncio
import threading
from collections.abc import Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import Enrichment, NoteRecord

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
    incoming = list(_notes_from(notes_source))
    incoming_by_id = {note.note_id: note for note in incoming}
    existing_by_id = {note.note_id: note for note in index.notes()}

    removed_ids = sorted(set(existing_by_id) - set(incoming_by_id))
    for note_id in removed_ids:
        index.delete_note(note_id)

    added = 0
    changed = 0
    enrich_failures = 0
    rebuilt: list[str] = []

    for note in incoming:
        existing = existing_by_id.get(note.note_id)
        if existing is not None and existing.content_hash == note.content_hash:
            continue

        if existing is None:
            added += 1
        else:
            changed += 1

        enrichment = index.cached_enrichment(note.note_id, note.content_hash)
        if enrichment is None:
            enrichment, failed = _run_async(enrich_note_with_status(note, llm))
            enrich_failures += int(failed)
            index.cache_enrichment(note.note_id, note.content_hash, enrichment)

        index.upsert_note(note, enrichment, _embedding_units(note, enrichment))
        rebuilt.append(note.note_id)

    for note_id in rebuilt:
        index.rebuild_for(note_id)
    index.load()
    return SyncReport(
        added=added,
        changed=changed,
        removed=len(removed_ids),
        enrich_failures=enrich_failures,
    )


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
    async def complete_json(self, model: str, system: str, user: str) -> dict:
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
