"""HippoRAG-style query-to-triple recognition and entity reset seeding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic
from typing import Protocol, Sequence

import numpy as np

from taurine_core import config
from taurine_core.indexer.facts import FactStatus
from taurine_core.models_local import (
    LocalModelUnavailable,
    ModelDeadlineExceeded,
    PairScorer,
)
from taurine_core.retrieval.calibration import HeadCalibrator
from taurine_core.retrieval.snapshot import IndexSnapshot
from taurine_core.retrieval.types import RetrievalMode, RetrievalQuery, TemporalIntent


LogicalFactKey = tuple[str, str, str]
TRIPLE_RECOGNITION_INSTRUCTION = (
    "Determine whether the canonical fact triple and its exact source assertion "
    "are relevant to the query and satisfy the query's entity and temporal intent."
)


@dataclass(frozen=True)
class FactEmbeddingHit:
    fact_id: str
    fact_ids: tuple[str, ...]
    logical_key: LogicalFactKey
    contexts: tuple[FactContext, ...]
    embedding_score: float
    calibrated_score: float
    accepted: bool
    rank: int


@dataclass(frozen=True)
class FactContext:
    fact_id: str
    subject_entity_id: str
    subject: str
    predicate: str
    predicate_description: str
    object_entity_id: str
    object: str
    source_passage_id: str
    source_note_id: str
    assertion_text: str
    observed_at: float
    valid_from: float | None
    valid_to: float | None
    status: FactStatus
    recognition_document: str
    logical_key: LogicalFactKey


@dataclass(frozen=True)
class TripleRecognitionDecision:
    logical_key: LogicalFactKey
    fact_ids: tuple[str, ...]
    embedding_score: float
    recognition_score: float
    accepted: bool
    reason: str
    used_fallback: bool = False


@dataclass(frozen=True)
class RecognizedTriple:
    logical_key: LogicalFactKey
    fact_ids: tuple[str, ...]
    source_passage_ids: tuple[str, ...]
    source_note_ids: tuple[str, ...]
    subject_entity_id: str
    subject: str
    predicate: str
    predicate_description: str
    object_entity_id: str
    object: str
    score: float
    contexts: tuple[FactContext, ...]


@dataclass(frozen=True)
class TripleRecognitionResult:
    triples: tuple[RecognizedTriple, ...]
    decisions: tuple[TripleRecognitionDecision, ...]
    entity_seed_weights: dict[str, float]
    degraded: bool
    reason: str

    @property
    def accepted_fact_ids(self) -> tuple[str, ...]:
        return tuple(
            fact_id for triple in self.triples for fact_id in triple.fact_ids
        )


class FactContextStore(Protocol):
    def fact_contexts(
        self, fact_ids: tuple[str, ...]
    ) -> tuple[FactContext, ...]: ...


class BatchedTripleAcceptor(Protocol):
    """Optional chat/tool ambiguity adjudicator; returns accepted fact IDs only."""

    def accept_fact_ids(
        self,
        query: RetrievalQuery,
        candidates: tuple[RecognizedTriple, ...],
        *,
        deadline_at: float | None = None,
    ) -> set[str]: ...


class FactRecallIndex:
    """Cosine recall over canonical fact embeddings in one immutable snapshot."""

    def __init__(
        self,
        snapshot: IndexSnapshot,
        store: FactContextStore,
        calibrator: HeadCalibrator,
    ) -> None:
        self.snapshot = snapshot
        self.store = store
        self.calibrator = calibrator

    def recall(
        self,
        query_vector: np.ndarray,
        query: RetrievalQuery,
        k: int,
    ) -> tuple[FactEmbeddingHit, ...]:
        if k <= 0 or not self.snapshot.fact_ids:
            return ()
        matrix = np.asarray(self.snapshot.fact_matrix, dtype=np.float32)
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"query vector must be one-dimensional, got {vector.shape}")
        if matrix.ndim != 2 or matrix.shape[0] != len(self.snapshot.fact_ids):
            raise ValueError("snapshot fact IDs and matrix rows do not align")
        if matrix.shape[1] != vector.shape[0]:
            raise ValueError(
                f"query vector dimension {vector.shape[0]} does not match "
                f"fact dimension {matrix.shape[1]}"
            )
        query_norm = float(np.linalg.norm(vector))
        if not math.isfinite(query_norm) or query_norm <= 0.0:
            return ()
        row_norms = np.linalg.norm(matrix, axis=1)
        valid = row_norms > 0
        scores = np.full(len(self.snapshot.fact_ids), -1.0, dtype=np.float64)
        scores[valid] = (
            matrix[valid] @ (vector / query_norm) / row_norms[valid]
        ).astype(np.float64)
        index_by_id = {
            fact_id: index for index, fact_id in enumerate(self.snapshot.fact_ids)
        }
        grouped: dict[LogicalFactKey, list[FactContext]] = {}
        for context in self.store.fact_contexts(self.snapshot.fact_ids):
            if context.fact_id not in index_by_id:
                continue
            if _temporal_status_reason(context, query.temporal_intent, now=query.now):
                continue
            grouped.setdefault(context.logical_key, []).append(context)
        group_rows: list[tuple[float, LogicalFactKey, tuple[FactContext, ...]]] = []
        for logical_key, contexts in grouped.items():
            ordered_contexts = tuple(sorted(contexts, key=lambda row: row.fact_id))
            raw = max(float(scores[index_by_id[row.fact_id]]) for row in ordered_contexts)
            group_rows.append((raw, logical_key, ordered_contexts))
        group_rows.sort(key=lambda row: (-row[0], row[1]))
        hits: list[FactEmbeddingHit] = []
        for rank, (group_score, logical_key, contexts) in enumerate(
            group_rows[: min(int(k), len(group_rows))], 1
        ):
            raw = float(np.clip(group_score, -1.0, 1.0))
            calibrated = self.calibrator.calibrate("triple", raw)
            representative = min(
                contexts,
                key=lambda row: (
                    -float(scores[index_by_id[row.fact_id]]),
                    row.fact_id,
                ),
            )
            hits.append(
                FactEmbeddingHit(
                    fact_id=representative.fact_id,
                    fact_ids=tuple(row.fact_id for row in contexts),
                    logical_key=logical_key,
                    contexts=contexts,
                    embedding_score=raw,
                    calibrated_score=calibrated.probability,
                    accepted=calibrated.accepted,
                    rank=rank,
                )
            )
        return tuple(hits)


class TripleRecognizer:
    """Groups fact provenance, recognizes logical triples, and seeds endpoints."""

    def __init__(
        self,
        recall_index: FactRecallIndex,
        store: FactContextStore,
        scorer: PairScorer,
        *,
        mode: RetrievalMode,
        recognition_floor: float,
        fallback_embedding_floor: float,
        ambiguity_low: float | None = None,
        ambiguity_high: float | None = None,
        ambiguity_acceptor: BatchedTripleAcceptor | None = None,
    ) -> None:
        self.recall_index = recall_index
        self.store = store
        self.scorer = scorer
        self.mode = mode
        self.recognition_floor = float(recognition_floor)
        self.fallback_embedding_floor = float(fallback_embedding_floor)
        self.ambiguity_low = None if ambiguity_low is None else float(ambiguity_low)
        self.ambiguity_high = None if ambiguity_high is None else float(ambiguity_high)
        self.ambiguity_acceptor = ambiguity_acceptor
        for name, value in (
            ("recognition", self.recognition_floor),
            ("fallback embedding", self.fallback_embedding_floor),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} floor must be within [0, 1]")
        if (self.ambiguity_low is None) != (self.ambiguity_high is None):
            raise ValueError("ambiguity low and high must be supplied together")
        if self.mode in {"chat", "tool"} and self.ambiguity_low is None:
            raise ValueError("chat/tool ambiguity band must be supplied from calibration")
        if self.ambiguity_low is not None and not (
            0.0 <= self.ambiguity_low < self.recognition_floor < self.ambiguity_high <= 1.0
        ):
            raise ValueError("ambiguity band must contain the recognition floor")

    def recognize(
        self,
        query: RetrievalQuery,
        query_vector: np.ndarray,
        resolved_entity_ids: set[str],
        required_entity_ids: set[str],
        deadline_at: float | None = None,
    ) -> TripleRecognitionResult:
        recalled = self.recall_index.recall(
            query_vector, query, k=config.TRIPLE_RECALL_K
        )
        hits = tuple(hit for hit in recalled if hit.accepted)
        if not hits:
            return TripleRecognitionResult((), (), {}, False, "no_fact_candidates")
        preliminary: list[TripleRecognitionDecision] = []
        groups: list[_FactGroup] = []
        for hit in hits:
            context = hit.contexts[0]
            reason = _entity_compatibility_reason(context, required_entity_ids)
            if reason is not None:
                preliminary.append(
                    TripleRecognitionDecision(
                        hit.logical_key,
                        hit.fact_ids,
                        hit.calibrated_score,
                        0.0,
                        False,
                        reason,
                    )
                )
                continue
            groups.append(_FactGroup.from_hit(hit))
        if not groups:
            return TripleRecognitionResult(
                (), tuple(preliminary), {}, False, "no_compatible_fact_candidates"
            )

        ordered_groups = tuple(
            sorted(
                groups,
                key=lambda group: (-group.embedding_score, group.logical_key),
            )
        )
        try:
            if deadline_at is not None and monotonic() >= deadline_at:
                raise ModelDeadlineExceeded("triple recognition deadline exceeded")
            recognition_scores = tuple(
                float(score)
                for score in self.scorer.score_pairs(
                    query.text,
                    tuple(group.recognition_document for group in ordered_groups),
                    instruction=TRIPLE_RECOGNITION_INSTRUCTION,
                    deadline_at=deadline_at,
                )
            )
            if len(recognition_scores) != len(ordered_groups) or any(
                not math.isfinite(score) for score in recognition_scores
            ):
                return self._fallback(
                    ordered_groups,
                    preliminary,
                    resolved_entity_ids,
                    "local_reranker_invalid_output_fallback",
                )
        except ModelDeadlineExceeded:
            return self._fallback(
                ordered_groups,
                preliminary,
                resolved_entity_ids,
                "local_reranker_deadline_fallback",
            )
        except LocalModelUnavailable:
            return self._fallback(
                ordered_groups,
                preliminary,
                resolved_entity_ids,
                "local_reranker_unavailable_fallback",
            )
        except Exception:
            return self._fallback(
                ordered_groups,
                preliminary,
                resolved_entity_ids,
                "local_reranker_unavailable_fallback",
            )

        scored = tuple(
            (group, min(1.0, max(0.0, raw_score)))
            for group, raw_score in zip(ordered_groups, recognition_scores, strict=True)
        )
        ambiguity_overrides: dict[LogicalFactKey, bool] = {}
        ambiguous = tuple(
            group.recognized(score)
            for group, score in scored
            if self._is_ambiguous(score)
        )
        if ambiguous and self.ambiguity_acceptor is not None and self.mode in {"chat", "tool"}:
            try:
                accepted_ids = set(
                    self.ambiguity_acceptor.accept_fact_ids(
                        query, ambiguous, deadline_at=deadline_at
                    )
                )
            except Exception:
                accepted_ids = set()
                ambiguity_succeeded = False
            else:
                ambiguity_succeeded = True
            if ambiguity_succeeded:
                ambiguity_overrides = {
                    candidate.logical_key: bool(set(candidate.fact_ids) & accepted_ids)
                    for candidate in ambiguous
                }

        decisions = list(preliminary)
        recognized: list[RecognizedTriple] = []
        for group, score in scored:
            local_accepted = score >= self.recognition_floor
            if group.logical_key in ambiguity_overrides:
                accepted = ambiguity_overrides[group.logical_key]
                decision_reason = (
                    "accepted_batched_ambiguity_recognition"
                    if accepted
                    else "rejected_batched_ambiguity_recognition"
                )
            else:
                accepted = local_accepted
                decision_reason = (
                    "accepted_local_triple_recognition"
                    if accepted
                    else "below_triple_recognition_floor"
                )
            decisions.append(
                TripleRecognitionDecision(
                    logical_key=group.logical_key,
                    fact_ids=group.fact_ids,
                    embedding_score=group.embedding_score,
                    recognition_score=score,
                    accepted=accepted,
                    reason=decision_reason,
                )
            )
            if accepted:
                recognized.append(group.recognized(score))
        recognized.sort(key=lambda row: (-row.score, row.logical_key))
        selected = tuple(recognized[: config.TRIPLE_SEED_K])
        return TripleRecognitionResult(
            triples=selected,
            decisions=tuple(decisions),
            entity_seed_weights=_seed_weights(selected),
            degraded=False,
            reason="recognized_triples" if selected else "no_recognized_triples",
        )

    def _is_ambiguous(self, score: float) -> bool:
        return bool(
            self.ambiguity_low is not None
            and self.ambiguity_high is not None
            and self.ambiguity_low <= score <= self.ambiguity_high
        )

    def _fallback(
        self,
        groups: Sequence["_FactGroup"],
        preliminary: Sequence[TripleRecognitionDecision],
        resolved_entity_ids: set[str],
        reason: str,
    ) -> TripleRecognitionResult:
        decisions = list(preliminary)
        accepted: list[RecognizedTriple] = []
        for group in groups:
            has_overlap = bool(group.endpoint_ids & resolved_entity_ids)
            passes_floor = group.embedding_score >= self.fallback_embedding_floor
            keep = passes_floor and has_overlap
            if not passes_floor:
                decision_reason = "fallback_below_strict_embedding_floor"
            elif not has_overlap:
                decision_reason = "fallback_requires_resolved_endpoint_overlap"
            else:
                decision_reason = "accepted_strict_embedding_entity_fallback"
            decisions.append(
                TripleRecognitionDecision(
                    group.logical_key,
                    group.fact_ids,
                    group.embedding_score,
                    0.0,
                    keep,
                    decision_reason,
                    used_fallback=True,
                )
            )
            if keep:
                accepted.append(group.recognized(group.embedding_score))
        accepted.sort(key=lambda row: (-row.score, row.logical_key))
        selected = tuple(accepted[: config.TRIPLE_SEED_K])
        return TripleRecognitionResult(
            selected,
            tuple(decisions),
            _seed_weights(selected),
            True,
            reason,
        )


@dataclass(frozen=True)
class _FactGroup:
    logical_key: LogicalFactKey
    contexts: tuple[FactContext, ...]
    embedding_score: float
    recognition_document: str

    @classmethod
    def from_hit(cls, hit: FactEmbeddingHit) -> "_FactGroup":
        ordered = hit.contexts
        first = ordered[0]
        documents = tuple(dict.fromkeys(row.recognition_document for row in ordered))
        document = documents[0]
        if len(documents) > 1:
            document = "\n\n".join(
                f"Provenance {index}:\n{source}"
                for index, source in enumerate(documents, 1)
            )
        return cls(
            logical_key=first.logical_key,
            contexts=ordered,
            embedding_score=hit.calibrated_score,
            recognition_document=document,
        )

    @property
    def fact_ids(self) -> tuple[str, ...]:
        return tuple(row.fact_id for row in self.contexts)

    @property
    def endpoint_ids(self) -> set[str]:
        first = self.contexts[0]
        return {first.subject_entity_id, first.object_entity_id}

    def recognized(self, score: float) -> RecognizedTriple:
        first = self.contexts[0]
        return RecognizedTriple(
            logical_key=self.logical_key,
            fact_ids=self.fact_ids,
            source_passage_ids=tuple(
                sorted({row.source_passage_id for row in self.contexts})
            ),
            source_note_ids=tuple(sorted({row.source_note_id for row in self.contexts})),
            subject_entity_id=first.subject_entity_id,
            subject=first.subject,
            predicate=first.predicate,
            predicate_description=first.predicate_description,
            object_entity_id=first.object_entity_id,
            object=first.object,
            score=float(score),
            contexts=self.contexts,
        )


def _entity_compatibility_reason(
    context: FactContext,
    required_entity_ids: set[str],
) -> str | None:
    endpoints = {context.subject_entity_id, context.object_entity_id}
    if required_entity_ids and not required_entity_ids <= endpoints:
        return "required_entity_mismatch"
    return None


def _temporal_status_reason(
    context: FactContext,
    intent: TemporalIntent | None,
    *,
    now: float,
) -> str | None:
    kind = intent.kind if intent is not None else "current"
    if context.status == "candidate":
        return "incompatible_fact_status"
    if kind == "any":
        return None
    if context.status == "conflicted":
        return "incompatible_fact_status"
    if kind == "historical":
        return None if context.status in {"historical", "superseded"} else "not_historical"
    at = intent.at if intent is not None and intent.at is not None else now
    if kind == "current" and context.status != "active":
        return "not_current"
    if kind == "at_time" and context.status not in {
        "active",
        "historical",
        "superseded",
    }:
        return "not_valid_at_time"
    if context.valid_from is not None and at < context.valid_from:
        return "not_yet_valid"
    if context.valid_to is not None and at >= context.valid_to:
        return "no_longer_valid"
    return None


def _seed_weights(triples: Sequence[RecognizedTriple]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for triple in triples:
        for entity_id in {triple.subject_entity_id, triple.object_entity_id}:
            weights[entity_id] = weights.get(entity_id, 0.0) + triple.score
    return dict(sorted(weights.items()))
