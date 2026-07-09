from __future__ import annotations

import argparse
import asyncio
import logging
import re
import threading
from collections.abc import Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import ConnectorSyncReport, Enrichment, Extraction, NoteRecord

from .embed import chunk_text, embed_texts
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
    llm: LLMPort | None,
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
        has_valid_extraction = (
            index.cached_extraction(note.note_id, note.content_hash) is not None
        )
        if (
            existing is not None
            and existing.content_hash == note.content_hash
            and has_valid_extraction
        ):
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
    report("extracting", 0, len(pending))
    enrich_failures = 0
    cached: dict[str, Extraction] = {}
    needs_extract: list[NoteRecord] = []
    for note in pending:
        hit = index.cached_extraction(note.note_id, note.content_hash)
        if hit is not None and not hit.failed:
            cached[note.note_id] = hit
        else:
            needs_extract.append(note)

    fresh: dict[str, Extraction] = {}
    if needs_extract:
        from . import extract

        fresh = extract.extract_notes(needs_extract)
    report("extracting", len(pending), len(pending))

    rebuilt: list[str] = []
    enrichments: dict[str, Enrichment] = {}
    for position, note in enumerate(pending, start=1):
        if note.note_id in cached:
            extraction = cached[note.note_id]
        else:
            extraction = fresh.get(note.note_id, Extraction(failed=True))
            enrich_failures += int(extraction.failed)
            if extraction.failed:
                _drop_cached_extraction(index, note)
            else:
                index.cache_extraction(note.note_id, note.content_hash, extraction)
        enrichment = _enrichment_from_extraction(note, extraction)
        enrichments[note.note_id] = enrichment
        index.upsert_note(note, enrichment, _embedding_units(note, enrichment))
        rebuilt.append(note.note_id)
        report("indexing", position, len(pending))

    report("finalizing", len(pending), len(pending))
    for note_id in rebuilt:
        index.rebuild_for(note_id)
    index.load()
    from .rank import compute_quality_ranks, compute_value_scores, refresh_fact_embeddings

    compute_quality_ranks(index)
    value_scores = compute_value_scores(index)
    if llm is not None:
        _run_async(_apply_slim_enrichment(index, pending, enrichments, llm, value_scores))
        for note_id in rebuilt:
            index.rebuild_for(note_id)
        index.load()
        compute_quality_ranks(index)
        compute_value_scores(index)
    if llm is not None:
        try:
            from .adjudicate import adjudicate_graph

            adjudication = _run_async(
                adjudicate_graph(
                    index,
                    llm,
                    lambda done, total: report("adjudicating", done, total),
                )
            )
            _LOG.info(
                "adjudicate: %d judged, %d admitted, %d vetoed",
                adjudication.judged,
                adjudication.admitted,
                adjudication.vetoed,
            )
        except Exception:
            _LOG.exception("adjudicate failed; continuing sync")
    refresh_fact_embeddings(index)
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
    for utterance in enrichment.utterances:
        if utterance.strip():
            texts.append(utterance)
            kinds.append("utterance")

    vectors = embed_texts(texts)
    units: list[EmbeddingInput] = []
    for kind, text, vector in zip(kinds, texts, vectors, strict=True):
        units.append((kind, text, np.asarray(vector, dtype=np.float32)))  # type: ignore[arg-type]
    return units


def _utterance_embedding_units(enrichment: Enrichment) -> list[EmbeddingInput]:
    texts = [utterance for utterance in enrichment.utterances if utterance.strip()]
    if not texts:
        return []
    vectors = embed_texts(texts)
    units: list[EmbeddingInput] = []
    for text, vector in zip(texts, vectors, strict=True):
        units.append(("utterance", text, np.asarray(vector, dtype=np.float32)))
    return units


async def _apply_slim_enrichment(
    index: Index,
    changed: list[NoteRecord],
    enrichments: dict[str, Enrichment],
    llm: LLMPort,
    value_scores: dict[str, float],
) -> None:
    from .slim_enrich import select_slim_notes, slim_enrich_batch

    selected = select_slim_notes(changed, value_scores)
    if not selected:
        return
    slim = await slim_enrich_batch(selected, llm, index.user_card())
    for note in selected:
        base = enrichments.get(note.note_id)
        if base is None:
            continue
        slim_item = slim.get(note.note_id)
        if slim_item is None:
            continue
        merged = base.model_copy(
            update={
                "aliases": [*base.aliases, *slim_item.aliases],
                "utterances": slim_item.utterances,
            }
        )
        enrichments[note.note_id] = merged
        index.apply_slim_enrichment(
            note.note_id,
            merged,
            _utterance_embedding_units(merged),
        )


def _enrichment_from_extraction(note: NoteRecord, extraction: Extraction) -> Enrichment:
    return Enrichment(
        aliases=_heuristic_aliases(note, extraction),
        entities=extraction.entities,
        relations=extraction.relations,
        extraction_failed=extraction.failed,
    )


def _heuristic_aliases(note: NoteRecord, extraction: Extraction) -> list[str]:
    aliases: list[str] = []
    for entity in extraction.entities:
        surface = entity.surface.strip()
        if not surface or not _alias_allowed(surface):
            continue
        if re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", note.title, re.IGNORECASE):
            aliases.append(surface)
    return _dedupe(aliases)


def _alias_allowed(value: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    if not normalized:
        return False
    words = normalized.split()
    compact_len = sum(len(word) for word in words)
    if compact_len < config.ALIAS_MIN_COMPACT_LEN:
        return False
    return not (words and all(word in config.ALIAS_STOPWORDS for word in words))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _drop_cached_extraction(index: Index, note: NoteRecord) -> None:
    with index._conn:  # type: ignore[attr-defined]
        index._conn.execute(  # type: ignore[attr-defined]
            """
            DELETE FROM enrich_cache
            WHERE note_id = ? AND content_hash = ?
            """,
            (note.note_id, note.content_hash),
        )


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
