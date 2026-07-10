"""Bounded local reranking of exact source passages."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Sequence

from taurine_core import config
from taurine_core.models_local import (
    LocalModelUnavailable,
    ModelDeadlineExceeded,
    PairScorer,
)
from taurine_core.retrieval.types import RetrievalMode


logger = logging.getLogger(__name__)

PASSAGE_RERANK_INSTRUCTION = (
    "Determine whether the exact source passage contains evidence relevant to "
    "answering the query. Judge the passage text, not its retrieval hint."
)


@dataclass(frozen=True)
class PassageRerankCandidate:
    passage_id: str
    note_id: str
    title: str
    source: str
    text: str
    # The candidate's rerank-input tier score (calibrated head score for
    # direct hits, recognition score for triple support, PPR mass / path
    # weight for expansions). Kept separate from the component scores below,
    # which are never max()-mixed across families (H6).
    recall_score: float
    hint: str = ""
    fused_score: float = 0.0
    ppr_mass: float = 0.0
    triple_support: float = 0.0
    path_weight: float = 0.0

    @property
    def rerank_document(self) -> str:
        parts = [
            f"Title: {self.title}",
            f"Source: {self.source}",
            f"Passage:\n{self.text}",
        ]
        if self.hint:
            parts.append(f"Retrieval hint: {self.hint}")
        return "\n".join(parts)


@dataclass(frozen=True)
class PassageRerankDecision:
    candidate: PassageRerankCandidate
    score: float
    accepted: bool
    rank: int
    reason: str


@dataclass(frozen=True)
class PassageRerankResult:
    decisions: tuple[PassageRerankDecision, ...]
    accepted: tuple[PassageRerankDecision, ...]
    degraded: bool
    reason: str


class LocalPassageReranker:
    """Scores a mode-bounded recall pool and fails closed on local-model faults."""

    def __init__(
        self,
        scorer: PairScorer,
        *,
        mode: RetrievalMode,
        acceptance_floor: float,
    ) -> None:
        self.scorer = scorer
        self.mode = mode
        self.acceptance_floor = float(acceptance_floor)
        if not 0.0 <= self.acceptance_floor <= 1.0:
            raise ValueError("passage rerank acceptance floor must be within [0, 1]")

    def rerank(
        self,
        query: str,
        candidates: Sequence[PassageRerankCandidate],
        *,
        deadline_at: float | None = None,
    ) -> PassageRerankResult:
        bounded = tuple(
            sorted(
                candidates,
                key=lambda row: (-_safe_recall(row.recall_score), row.passage_id),
            )[: self._candidate_cap]
        )
        if not bounded:
            return PassageRerankResult((), (), False, "no_passage_candidates")
        documents = tuple(candidate.rerank_document for candidate in bounded)
        try:
            scores = tuple(
                float(score)
                for score in self.scorer.score_pairs(
                    query,
                    documents,
                    instruction=PASSAGE_RERANK_INSTRUCTION,
                    deadline_at=deadline_at,
                )
            )
        except ModelDeadlineExceeded:
            return _degraded(bounded, "local_reranker_deadline")
        except LocalModelUnavailable:
            return _degraded(bounded, "local_reranker_unavailable")
        except Exception:
            # Fail-closed to the degraded state, but keep a debug trail so a
            # programming error cannot masquerade as "reranker unavailable".
            logger.debug("passage reranker raised", exc_info=True)
            return _degraded(bounded, "local_reranker_unavailable")
        if len(scores) != len(bounded) or any(not math.isfinite(score) for score in scores):
            return _degraded(bounded, "local_reranker_invalid_output")

        ordered = sorted(
            zip(bounded, scores, strict=True),
            key=lambda row: (-row[1], row[0].passage_id),
        )
        decisions = tuple(
            PassageRerankDecision(
                candidate=candidate,
                score=min(1.0, max(0.0, score)),
                accepted=score >= self.acceptance_floor,
                rank=rank,
                reason=(
                    "accepted_passage_rerank"
                    if score >= self.acceptance_floor
                    else "below_passage_rerank_floor"
                ),
            )
            for rank, (candidate, score) in enumerate(ordered, 1)
        )
        return PassageRerankResult(
            decisions=decisions,
            accepted=tuple(row for row in decisions if row.accepted),
            degraded=False,
            reason="reranked" if decisions else "no_passage_candidates",
        )

    @property
    def _candidate_cap(self) -> int:
        return (
            config.LIVE_RERANK_CANDIDATES
            if self.mode == "live"
            else config.CHAT_RERANK_CANDIDATES
        )


def _degraded(
    candidates: Sequence[PassageRerankCandidate],
    reason: Literal[
        "local_reranker_deadline",
        "local_reranker_unavailable",
        "local_reranker_invalid_output",
    ],
) -> PassageRerankResult:
    decisions = tuple(
        PassageRerankDecision(
            candidate=candidate,
            score=0.0,
            accepted=False,
            rank=rank,
            reason=reason,
        )
        for rank, candidate in enumerate(candidates, 1)
    )
    return PassageRerankResult(decisions, (), True, reason)


def _safe_recall(score: float) -> float:
    value = float(score)
    return value if math.isfinite(value) else float("-inf")
