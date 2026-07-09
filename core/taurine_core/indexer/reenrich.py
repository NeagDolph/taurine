from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from taurine_core.ports import LLMPort
from taurine_core.schemas import ReenrichStatus

from .sync import (
    _apply_slim_enrichment,
    _drop_cached_extraction,
    _embedding_units,
    _enrichment_from_extraction,
)

_LOG = logging.getLogger("taurine.session")


@dataclass(frozen=True)
class ReenrichReport:
    total: int
    changed: int
    failures: int


async def reenrich_all(
    index: object,
    llm: LLMPort,
    progress: Callable[[ReenrichStatus], None],
    source: str | None = None,
) -> ReenrichReport:
    """Re-extract every note, or only one connector's notes when `source` is set."""
    disabled = getattr(index, "disabled_sources", set)()
    notes = [
        note
        for note in index.notes()  # type: ignore[attr-defined]
        if (source is None or note.source == source) and note.source not in disabled
    ]
    total = len(notes)

    progress(ReenrichStatus(running=True, phase="notes", done=0, total=total))

    changed_note_ids: list[str] = []
    changed_notes = []
    enrichments = {}
    done = 0
    failures = 0

    from . import extract

    extractions = extract.extract_notes(notes)  # type: ignore[arg-type]
    for note in notes:
        extraction = extractions.get(note.note_id)
        if extraction is None or extraction.failed:
            failures += 1
            _drop_cached_extraction(index, note)  # type: ignore[arg-type]
        else:
            index.cache_extraction(  # type: ignore[attr-defined]
                note.note_id,
                note.content_hash,
                extraction,
            )
            enrichment = _enrichment_from_extraction(note, extraction)
            enrichments[note.note_id] = enrichment
            index.upsert_note(  # type: ignore[attr-defined]
                note,
                enrichment,
                _embedding_units(note, enrichment),  # type: ignore[arg-type]
            )
            changed_note_ids.append(note.note_id)
            changed_notes.append(note)
        done += 1
        progress(ReenrichStatus(running=True, phase="notes", done=done, total=total))

    progress(ReenrichStatus(running=True, phase="finalizing", done=done, total=total))
    for note_id in changed_note_ids:
        index.rebuild_for(note_id)  # type: ignore[attr-defined]
    index.load()  # type: ignore[attr-defined]
    from .rank import compute_quality_ranks, compute_value_scores, refresh_fact_embeddings

    compute_quality_ranks(index)  # type: ignore[arg-type]
    value_scores = compute_value_scores(index)  # type: ignore[arg-type]
    await _apply_slim_enrichment(  # type: ignore[arg-type]
        index,
        changed_notes,
        enrichments,
        llm,
        value_scores,
    )
    for note_id in changed_note_ids:
        index.rebuild_for(note_id)  # type: ignore[attr-defined]
    index.load()  # type: ignore[attr-defined]
    compute_quality_ranks(index)  # type: ignore[arg-type]
    compute_value_scores(index)  # type: ignore[arg-type]
    if llm is not None:
        try:
            from .adjudicate import adjudicate_graph

            adjudication = await adjudicate_graph(
                index,
                llm,
                lambda adjudicated, adjudication_total: progress(
                    ReenrichStatus(
                        running=True,
                        phase="finalizing",
                        done=adjudicated,
                        total=adjudication_total,
                    )
                ),
            )
            _LOG.info(
                "adjudicate: %d judged, %d admitted, %d vetoed",
                adjudication.judged,
                adjudication.admitted,
                adjudication.vetoed,
            )
        except Exception:
            _LOG.exception("adjudicate failed; continuing reenrich")
    refresh_fact_embeddings(index)  # type: ignore[arg-type]
    progress(ReenrichStatus(running=False, phase="idle", done=done, total=total))

    return ReenrichReport(
        total=total,
        changed=len(changed_note_ids),
        failures=failures,
    )
