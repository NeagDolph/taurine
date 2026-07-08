"""Stage A — latent context pack for typed prompts (spec chat §1.2).

Reuses the live pipeline's machinery with chat posture:
- heads: dense(message, CHAT_EMBED_QUERY_PROMPT), BM25, exact aliases at
  rank 1, fact_dense_search — NO GLiNER, NO fuzzy snapping (typed text has
  no ASR drift);
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

from ..indexer.store import Index
from ..schemas import GroundedFact, GroundingItem


class ChatRetriever:
    """One instance per chat_id; state restored from the chat registry."""

    def __init__(self, index: Index, chat_id: str) -> None:
        self.index = index
        self.chat_id = chat_id
        raise NotImplementedError  # Stage A

    def context_pack(
        self, message: str, *, now: float
    ) -> tuple[list[GroundingItem], list[GroundedFact]]:
        """Advance the per-turn accumulator and return the Stage-A pack."""
        raise NotImplementedError  # Stage A

    def search(
        self, query: str, k: int, *, now: float
    ) -> list[GroundingItem]:
        """The search_notes tool body: full latent pipeline on a
        model-formulated query, seeded by accumulator ∪ this query.
        New notes join the chat's citation registry."""
        raise NotImplementedError  # Stage A

    def resolve(self, ref: str) -> str | None:
        """short_id ("N3") or raw note_id → note_id, if known to this chat."""
        raise NotImplementedError  # Stage A

    def save(self) -> None:
        """Persist registries + accumulator to Index.set_chat_registry."""
        raise NotImplementedError  # Stage A
