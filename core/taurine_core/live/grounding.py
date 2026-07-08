"""Grounding engine: seeds → PPR → prior blend → grounding set (spec §4.3–4.5).

Owned state (per session):
- meeting seed accumulator (decayed sum of fused top-K across passes — the
  "entire transcription so far" as a PPR personalization vector),
- previous PPR vector (warm start),
- cached NoteGraphMatrix, rebuilt when ``index.graph_version()`` changes,
- stable short-id registries: note_id → "N#" and (src, dst) → "F#"
  (assigned once per session, never reused — citations must stay valid
  across summary revisions).

No LLM calls; target < 50 ms per pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .. import config
from ..indexer.store import Index
from ..schemas import GroundedFact, GroundingItem


@dataclass
class GroundingUpdate:
    """One pass's output, consumed by the trigger and the summarizer."""

    items: list[GroundingItem] = field(default_factory=list)   # top GROUNDING_TOP_K by blended score
    facts: list[GroundedFact] = field(default_factory=list)    # ≤ GROUNDING_FACTS_MAX, stable fact_ids
    revision: int = 0                                          # bumps when membership/order changes
    ppr_top: list[tuple[str, float]] = field(default_factory=list)  # for the map's retrieval event


class GroundingEngine:
    def __init__(self, index: Index) -> None:
        self.index = index
        raise NotImplementedError  # Unit H

    def on_pass(
        self,
        window_text: str,
        fused: Sequence[tuple[str, float]],
        exact_hits: Sequence[tuple[str, str]],
        now: float,
    ) -> GroundingUpdate:
        """Advance the accumulator and produce the current grounding set.

        ``fused``: RRF-fused (note_id, score), best first, from the heads
        (dense window, GLiNER spans, BM25, exact-as-list). ``exact_hits``:
        (note_id, alias) C1 matches this window (max-strength seeds; recorded
        in GroundingWhy.exact_alias).

        Steps (spec §4.3–4.5): decay+add seeds → PPR (warm start; matrix
        rebuilt on graph_version change) → blend
        ``rel = (1−β)·norm(fused) + β·norm(ppr)`` then
        ``final = rel × recency × source_rank × exp(QUALITY_LAMBDA·quality_rank)``
        → hysteresis vs previous set (challenger > incumbent × GROUNDING_HYSTERESIS)
        → top-K items + fact selection (edge facts within the set or set→top-5
        seeds, plus this pass's fact_dense_search hits).
        """
        raise NotImplementedError  # Unit H
