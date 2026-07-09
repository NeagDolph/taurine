"""Stage A — latent context pack for typed prompts (spec chat §1.2).

Reuses the live pipeline's machinery with chat posture:
- heads: dense(message, CHAT_EMBED_QUERY_PROMPT), BM25, exact aliases at
  rank 1, fact_dense_search, GLiNER entity spans as transient PPR seeds
  (typed text has no ASR fuzzy snapping);
- RRF fusion (RRF_K), per-chat seed accumulator (CHAT_SEED_DECAY per turn),
  PPR (warm-started per chat, matrix cached on graph_version), blend with
  the same recency/source/quality priors;
- pack: CHAT_CONTEXT_NOTES notes + CHAT_CONTEXT_FACTS facts, NO hysteresis
  (continuity lives in the accumulator, not set stickiness).

Per-chat N#/F# registries and the accumulator persist via
Index.chat_registry / set_chat_registry so citations in old messages keep
resolving after restart.

Implementation note (Stage A refactor): extract the shared head-running +
RRF from AmbientRetriever into a reusable helper, and parameterize
GroundingEngine (top_k, facts_max, seed_decay, hysteresis: float | None,
query_prompt) rather than duplicating it. No LLM calls in this module.
"""
from __future__ import annotations

from .. import config
from ..indexer.store import Index
from ..live.ambient import run_head_fusion
from ..live.exact import ExactCollector
from ..live.grounding import GroundingEngine
from ..ports import SpanExtractorPort
from ..schemas import Chunk, GroundedFact, GroundingItem


class ChatRetriever:
    """One instance per chat_id; state restored from the chat registry."""

    def __init__(
        self,
        index: Index,
        chat_id: str,
        *,
        span_extractor: SpanExtractorPort | None = None,
    ) -> None:
        self.index = index
        self.chat_id = chat_id
        self._span_extractor = span_extractor
        self._exact = ExactCollector(index)
        self._grounding = GroundingEngine(
            index,
            top_k=config.CHAT_CONTEXT_NOTES,
            facts_max=config.CHAT_CONTEXT_FACTS,
            seed_decay=config.CHAT_SEED_DECAY,
            hysteresis=None,
            query_prompt=config.CHAT_EMBED_QUERY_PROMPT,
        )
        registry = index.chat_registry(chat_id)
        if isinstance(registry, dict) and registry:
            self._grounding.restore_state(registry)

    def context_pack(
        self, message: str, *, now: float
    ) -> tuple[list[GroundingItem], list[GroundedFact]]:
        """Advance the per-turn accumulator and return the Stage-A pack."""
        update = self._run(message, now=now, decay_seeds=True)
        return update.items, update.facts

    def search(
        self, query: str, k: int, *, now: float
    ) -> list[GroundingItem]:
        """The search_notes tool body: full latent pipeline on a
        model-formulated query, seeded by accumulator ∪ this query.
        New notes join the chat's citation registry."""
        if k <= 0:
            return []
        update = self._run(query, now=now, decay_seeds=False)
        return update.items[:k]

    def resolve(self, ref: str) -> str | None:
        """short_id ("N3") or raw note_id → note_id, if known to this chat."""
        for note_id, short_id in self._grounding.note_short_ids().items():
            if short_id == ref:
                return note_id
        if self.index.note(ref) is not None:
            return ref
        return None

    def save(self) -> None:
        """Persist registries + accumulator to Index.set_chat_registry."""
        self.index.set_chat_registry(self.chat_id, self._grounding.export_state())

    def _run(self, text: str, *, now: float, decay_seeds: bool):
        exact_hits = self._exact_hits(text, now)
        gliner_spans = self._extract_spans(text)
        heads = run_head_fusion(
            self.index,
            text,
            exact_hits=exact_hits,
            query_prompt=config.CHAT_EMBED_QUERY_PROMPT,
        )
        return self._grounding.on_pass(
            text,
            heads.fused,
            exact_hits,
            now,
            head_hits=heads.head_hits,
            gliner_spans=gliner_spans,
            decay_seeds=decay_seeds,
        )

    def _extract_spans(self, text: str) -> list[tuple[str, float, str]]:
        if self._span_extractor is None:
            return []
        try:
            return self._span_extractor.extract(text)
        except Exception:
            return []

    def _exact_hits(self, text: str, now: float) -> list[tuple[str, str]]:
        chunk = Chunk(channel=0, text=text, ts=now, final=True)
        hits: list[tuple[str, str]] = []
        seen: set[str] = set()
        for note_id, alias, _is_title in self._exact.on_chunk(chunk):
            if note_id in seen:
                continue
            seen.add(note_id)
            hits.append((note_id, alias))
        return hits
