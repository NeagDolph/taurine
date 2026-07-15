"""Bounded local reranking of exact source passages."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Literal, Sequence

import httpx

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


class RerankError(RuntimeError):
    """Sanitized failure from the OpenRouter passage-rerank backend."""


class OpenRouterRerankClient:
    """Cohere-shaped /rerank client; also speaks to api.cohere.com directly.

    OpenRouter's /rerank proxy and Cohere's native v2/rerank share request
    and response schemas, so one client serves both — base_url/provider
    select the host. Scores are bit-identical (same underlying model), so
    the calibrated floors remain valid either way.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str,
        base_url: str | None = None,
        provider: str = "OpenRouter",
        key_env: str = "OPEN_ROUTER_API_KEY",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise RerankError(f"{key_env} is required for {provider} reranking")
        self._api_key = api_key
        self.model_id = str(model_id)
        self._base_url = str(base_url or config.OPENROUTER_BASE_URL)
        self._provider = provider
        self._sleep = sleep
        self._client = httpx.Client(
            transport=transport,
            timeout=float(config.RERANK_API_TIMEOUT_S),
        )

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> tuple[float, ...]:
        del instruction
        rows = tuple(str(document) for document in documents)
        if not rows:
            return ()
        attempts = max(1, int(config.RERANK_API_ATTEMPTS))
        for attempt in range(attempts):
            if deadline_at is not None and time.monotonic() >= deadline_at:
                raise ModelDeadlineExceeded(f"{self._provider} reranking deadline exceeded")
            try:
                response = self._client.post(
                    f"{self._base_url}/rerank",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self.model_id,
                        "query": query,
                        "documents": list(rows),
                        "top_n": len(rows),
                    },
                )
            except httpx.TransportError as exc:
                if attempt + 1 >= attempts:
                    raise RerankError(
                        f"{self._provider} reranking transport failed "
                        f"after {attempts} attempts"
                    ) from exc
                self._backoff(attempt)
                continue
            status = int(response.status_code)
            if 200 <= status < 300:
                return self._parse(response, len(rows))
            retryable = status == 429 or 500 <= status < 600
            if retryable and attempt + 1 < attempts:
                self._backoff(attempt)
                continue
            raise RerankError(f"{self._provider} reranking returned HTTP {status}")
        raise AssertionError("rerank retry loop exhausted without returning or raising")

    def _backoff(self, attempt: int) -> None:
        self._sleep(float(config.RERANK_API_BACKOFF_S) * (2**attempt))

    @staticmethod
    def _parse(response: httpx.Response, count: int) -> tuple[float, ...]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RerankError("OpenRouter reranking returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RerankError("OpenRouter reranking returned invalid results")
        results = payload["results"]
        if len(results) != count or any(not isinstance(row, dict) for row in results):
            raise RerankError("OpenRouter reranking returned the wrong row count")
        try:
            by_index = {
                int(row["index"]): float(row["relevance_score"])
                for row in results
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError("OpenRouter reranking returned invalid rows") from exc
        if set(by_index) != set(range(count)) or len(by_index) != count:
            raise RerankError("OpenRouter reranking returned invalid row indexes")
        scores = tuple(by_index[index] for index in range(count))
        if any(not math.isfinite(score) for score in scores):
            raise RerankError("OpenRouter reranking returned non-finite scores")
        return scores


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
    model_id: str = config.RERANK_MODEL
    floor_source: Literal["artifact", "api_override", "none"] = "artifact"
    evidence_floor_override: float | None = None


class _InvalidRerankOutput(ValueError):
    pass


class LocalPassageReranker:
    """Scores a mode-bounded recall pool and fails closed on local-model faults."""

    def __init__(
        self,
        scorer: PairScorer,
        *,
        mode: RetrievalMode,
        acceptance_floor: float,
        model_id: str = config.RERANK_MODEL,
        evidence_floor_override: float | None = None,
        fallback_scorer: PairScorer | None = None,
        fallback_acceptance_floor: float | None = None,
        fallback_model_id: str = config.RERANK_MODEL,
    ) -> None:
        self.scorer = scorer
        self.mode = mode
        self.acceptance_floor = float(acceptance_floor)
        self.model_id = str(model_id)
        self.evidence_floor_override = (
            None
            if evidence_floor_override is None
            else float(evidence_floor_override)
        )
        self.fallback_scorer = fallback_scorer
        self.fallback_acceptance_floor = (
            None
            if fallback_acceptance_floor is None
            else float(fallback_acceptance_floor)
        )
        self.fallback_model_id = str(fallback_model_id)
        if not 0.0 <= self.acceptance_floor <= 1.0:
            raise ValueError("passage rerank acceptance floor must be within [0, 1]")
        if self.evidence_floor_override is not None and not (
            0.0 <= self.evidence_floor_override <= 1.0
        ):
            raise ValueError("passage evidence floor override must be within [0, 1]")
        if (self.fallback_scorer is None) != (self.fallback_acceptance_floor is None):
            raise ValueError("fallback scorer and acceptance floor must be supplied together")
        if self.fallback_acceptance_floor is not None and not (
            0.0 <= self.fallback_acceptance_floor <= 1.0
        ):
            raise ValueError("fallback rerank acceptance floor must be within [0, 1]")

    def rerank(
        self,
        query: str,
        candidates: Sequence[PassageRerankCandidate],
        *,
        deadline_at: float | None = None,
    ) -> PassageRerankResult:
        ordered_candidates = tuple(
            sorted(
                candidates,
                key=lambda row: (-_safe_recall(row.recall_score), row.passage_id),
            )
        )
        bounded = ordered_candidates[: self._candidate_cap]
        unscored = ordered_candidates[self._candidate_cap :]
        if not bounded:
            return PassageRerankResult(
                (), (), False, "no_passage_candidates", "", "none", None
            )
        documents = tuple(candidate.rerank_document for candidate in bounded)
        try:
            scores = _score_documents(
                self.scorer,
                query,
                documents,
                deadline_at=deadline_at,
            )
        except Exception as primary_exc:  # noqa: BLE001 - fallback boundary
            if self.fallback_scorer is None:
                if not isinstance(primary_exc, (ModelDeadlineExceeded, LocalModelUnavailable)):
                    # Keep a debug trail so programming errors cannot
                    # masquerade as an unavailable local model.
                    logger.debug("passage reranker raised", exc_info=True)
                return _with_unscored_tail(
                    _degraded(bounded, _failure_reason(primary_exc)), unscored
                )
            logger.warning(
                "cloud passage reranker failed (%s); trying local Qwen fallback",
                type(primary_exc).__name__,
            )
            try:
                scores = _score_documents(
                    self.fallback_scorer,
                    query,
                    documents,
                    deadline_at=deadline_at,
                )
            except Exception as fallback_exc:  # noqa: BLE001 - final fail-open boundary
                logger.warning(
                    "no passage reranker succeeded; local fallback failed (%s)",
                    type(fallback_exc).__name__,
                )
                return _with_unscored_tail(
                    _degraded(bounded, _failure_reason(fallback_exc)), unscored
                )
            return _ranked_result(
                bounded,
                scores,
                unscored,
                acceptance_floor=float(self.fallback_acceptance_floor),
                model_id=self.fallback_model_id,
                floor_source="artifact",
                evidence_floor_override=None,
                reason="reranked_with_local_fallback",
            )

        return _ranked_result(
            bounded,
            scores,
            unscored,
            acceptance_floor=self.acceptance_floor,
            model_id=self.model_id,
            floor_source=(
                "api_override"
                if self.evidence_floor_override is not None
                else "artifact"
            ),
            evidence_floor_override=self.evidence_floor_override,
            reason="reranked",
        )

    @property
    def _candidate_cap(self) -> int:
        return max(0, int(config.RERANK_MAX_CANDIDATES))


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
    return PassageRerankResult(decisions, (), True, reason, "", "none", None)


def _failure_reason(
    error: Exception,
) -> Literal[
    "local_reranker_deadline",
    "local_reranker_unavailable",
    "local_reranker_invalid_output",
]:
    if isinstance(error, ModelDeadlineExceeded):
        return "local_reranker_deadline"
    if isinstance(error, _InvalidRerankOutput):
        return "local_reranker_invalid_output"
    return "local_reranker_unavailable"


def _score_documents(
    scorer: PairScorer,
    query: str,
    documents: tuple[str, ...],
    *,
    deadline_at: float | None,
) -> tuple[float, ...]:
    scores = tuple(
        float(score)
        for score in scorer.score_pairs(
            query,
            documents,
            instruction=PASSAGE_RERANK_INSTRUCTION,
            deadline_at=deadline_at,
        )
    )
    if len(scores) != len(documents) or any(
        not math.isfinite(score) for score in scores
    ):
        raise _InvalidRerankOutput("passage reranker returned invalid scores")
    return scores


def _ranked_result(
    candidates: tuple[PassageRerankCandidate, ...],
    scores: tuple[float, ...],
    unscored: tuple[PassageRerankCandidate, ...],
    *,
    acceptance_floor: float,
    model_id: str,
    floor_source: Literal["artifact", "api_override"],
    evidence_floor_override: float | None,
    reason: str,
) -> PassageRerankResult:
    ordered = sorted(
        zip(candidates, scores, strict=True),
        key=lambda row: (-row[1], row[0].passage_id),
    )
    decisions = tuple(
        PassageRerankDecision(
            candidate=candidate,
            score=min(1.0, max(0.0, score)),
            accepted=score >= acceptance_floor,
            rank=rank,
            reason=(
                "accepted_passage_rerank"
                if score >= acceptance_floor
                else "below_passage_rerank_floor"
            ),
        )
        for rank, (candidate, score) in enumerate(ordered, 1)
    )
    return PassageRerankResult(
        decisions=(*decisions, *_unscored_decisions(unscored, len(decisions))),
        accepted=tuple(row for row in decisions if row.accepted),
        degraded=False,
        reason=reason,
        model_id=model_id,
        floor_source=floor_source,
        evidence_floor_override=evidence_floor_override,
    )


def _with_unscored_tail(
    result: PassageRerankResult,
    candidates: Sequence[PassageRerankCandidate],
) -> PassageRerankResult:
    return PassageRerankResult(
        decisions=(
            *result.decisions,
            *_unscored_decisions(candidates, len(result.decisions)),
        ),
        accepted=result.accepted,
        degraded=result.degraded,
        reason=result.reason,
        model_id=result.model_id,
        floor_source=result.floor_source,
        evidence_floor_override=result.evidence_floor_override,
    )


def _unscored_decisions(
    candidates: Sequence[PassageRerankCandidate],
    rank_offset: int,
) -> tuple[PassageRerankDecision, ...]:
    return tuple(
        PassageRerankDecision(
            candidate=candidate,
            score=0.0,
            accepted=False,
            rank=rank,
            reason="outside_rerank_candidate_cap",
        )
        for rank, candidate in enumerate(candidates, rank_offset + 1)
    )


def _safe_recall(score: float) -> float:
    value = float(score)
    return value if math.isfinite(value) else float("-inf")
