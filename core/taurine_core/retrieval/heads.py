"""Calibrated, independently abstaining retrieval recall heads."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Literal, Protocol, Sequence

import numpy as np

from taurine_core import config
from taurine_core.indexer.knowledge_store import BM25PassageHit
from taurine_core.retrieval.calibration import HeadCalibrator
from taurine_core.retrieval.snapshot import IndexSnapshot
from taurine_core.retrieval.types import HeadHit, TargetKind


@dataclass(frozen=True)
class RawHeadCandidate:
    target_kind: TargetKind
    target_id: str
    raw_score: float
    reason: str
    entity_etype: str | None = None


@dataclass(frozen=True)
class HeadResult:
    accepted: tuple[HeadHit, ...]
    rejected: tuple[HeadHit, ...]
    status: Literal["ok", "unavailable"] = "ok"
    reason: str = ""

    @property
    def all_hits(self) -> tuple[HeadHit, ...]:
        return self.accepted + self.rejected


class BM25StorePort(Protocol):
    def search_passages_bm25(
        self, query: str, k: int
    ) -> tuple[BM25PassageHit, ...]: ...


class DensePassageHead:
    def __init__(self, snapshot: IndexSnapshot, calibrator: HeadCalibrator) -> None:
        self.snapshot = snapshot
        self.calibrator = calibrator

    def search(self, query_vector: np.ndarray, *, k: int) -> HeadResult:
        if k <= 0 or not self.snapshot.passage_ids:
            return HeadResult(accepted=(), rejected=())
        query = np.asarray(query_vector, dtype=np.float32)
        matrix = self.snapshot.passage_matrix
        if query.ndim != 1:
            raise ValueError(f"query vector must be one-dimensional, got {query.shape}")
        if matrix.ndim != 2 or matrix.shape[0] != len(self.snapshot.passage_ids):
            raise ValueError("snapshot passage IDs and matrix rows do not align")
        if query.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"query vector dimension {query.shape[0]} does not match "
                f"passage dimension {matrix.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm <= 0.0:
            return HeadResult(accepted=(), rejected=())
        normalized_query = query / norm
        raw_scores = np.asarray(matrix @ normalized_query, dtype=np.float64)
        ordered = sorted(
            range(len(self.snapshot.passage_ids)),
            key=lambda index: (
                -float(raw_scores[index]),
                self.snapshot.passage_ids[index],
            ),
        )[:k]
        candidates = tuple(
            RawHeadCandidate(
                target_kind="passage",
                target_id=self.snapshot.passage_ids[index],
                raw_score=float(np.clip(raw_scores[index], -1.0, 1.0)),
                reason="passage_embedding_cosine",
            )
            for index in ordered
        )
        return _calibrate_candidates("dense", candidates, self.calibrator)


class BM25PassageHead:
    def __init__(self, store: BM25StorePort, calibrator: HeadCalibrator) -> None:
        self.store = store
        self.calibrator = calibrator

    def search(self, query: str, *, k: int) -> HeadResult:
        rows = self.store.search_passages_bm25(query, k)
        accepted: list[HeadHit] = []
        rejected: list[HeadHit] = []
        for rank, row in enumerate(rows, 1):
            calibrated = self.calibrator.calibrate("bm25", row.raw_score)
            coverage_ok = (
                math.isfinite(row.token_coverage)
                and row.token_coverage >= self.calibrator.bm25_min_token_coverage
            )
            if not coverage_ok:
                reason = "insufficient_query_token_coverage"
            elif not calibrated.accepted:
                reason = "below_calibrated_acceptance_floor"
            else:
                reason = "accepted_bm25_passage"
            hit = HeadHit(
                target_kind="passage",
                target_id=row.passage_id,
                raw_score=float(row.raw_score),
                calibrated_score=calibrated.probability,
                rank=rank,
                head="bm25",
                reason=reason,
            )
            (accepted if coverage_ok and calibrated.accepted else rejected).append(hit)
        return HeadResult(accepted=tuple(accepted), rejected=tuple(rejected))


class ExactAliasHead:
    def __init__(self, snapshot: IndexSnapshot, calibrator: HeadCalibrator) -> None:
        self.snapshot = snapshot
        self.calibrator = calibrator

    def search(self, query: str) -> HeadResult:
        matches: list[tuple[int, int, str, tuple[str, ...]]] = []
        for alias, target_ids in self.snapshot.alias_index.items():
            surface = alias.strip()
            if not surface or not target_ids or not query_alias_allowed(surface):
                continue
            if len(surface) <= 3:
                raw_variants = tuple(
                    (raw_surface, raw_target_ids)
                    for raw_surface, raw_target_ids in self.snapshot.alias_case_index.items()
                    if raw_surface.strip().casefold() == surface.casefold()
                ) or ((surface.upper(), target_ids),)
                for raw_surface, raw_target_ids in raw_variants:
                    raw_surface = raw_surface.strip()
                    match = re.search(
                        rf"(?<!\w){re.escape(raw_surface)}(?!\w)", query
                    )
                    if match is not None:
                        matches.append(
                            (
                                match.start(),
                                -len(raw_surface),
                                surface.casefold(),
                                raw_target_ids,
                            )
                        )
                continue
            match = re.search(
                rf"(?<!\w){re.escape(surface)}(?!\w)",
                query,
                flags=re.IGNORECASE,
            )
            if match is not None:
                matches.append(
                    (match.start(), -len(surface), surface.casefold(), target_ids)
                )
        matches.sort()

        accepted: list[HeadHit] = []
        rejected: list[HeadHit] = []
        credited: set[str] = set()
        rank = 0
        for _, _, surface, target_ids in matches:
            unique_targets = tuple(sorted(set(target_ids)))
            if len(unique_targets) != 1:
                for target_id in unique_targets:
                    if target_id in credited:
                        continue
                    rank += 1
                    rejected.append(
                        _make_hit(
                            "exact",
                            RawHeadCandidate("entity", target_id, 1.0, surface),
                            self.calibrator,
                            rank,
                            forced_reason="ambiguous_exact_alias",
                            forced_rejected=True,
                        )[0]
                    )
                continue
            target_id = unique_targets[0]
            if target_id in credited:
                continue
            credited.add(target_id)
            rank += 1
            hit, is_accepted = _make_hit(
                "exact",
                RawHeadCandidate(
                    "entity",
                    target_id,
                    1.0,
                    surface,
                    self.snapshot.entity_types.get(target_id),
                ),
                self.calibrator,
                rank,
                accepted_reason="accepted_unambiguous_exact_alias",
            )
            (accepted if is_accepted else rejected).append(hit)
        return HeadResult(accepted=tuple(accepted), rejected=tuple(rejected))


EntityLookup = Callable[[str, str | None, str], Sequence[RawHeadCandidate] | object]


class QueryEntityHead:
    def __init__(self, resolver: EntityLookup, calibrator: HeadCalibrator) -> None:
        self.resolver = resolver
        self.calibrator = calibrator

    def search(
        self,
        query: str,
        spans: Sequence[tuple[str, str | None]],
    ) -> HeadResult:
        accepted: list[HeadHit] = []
        rejected: list[HeadHit] = []
        seen: set[str] = set()
        rank = 0
        for surface, etype in spans:
            if not query_alias_allowed(surface):
                continue
            resolution = self.resolver(surface, etype, query)
            candidates, ambiguous = _resolution_candidates(resolution)
            for candidate in candidates:
                if candidate.target_id in seen:
                    continue
                seen.add(candidate.target_id)
                rank += 1
                if candidate.entity_etype is None and etype is not None:
                    candidate = replace(candidate, entity_etype=etype)
                hit, is_accepted = _make_hit(
                    "entity",
                    candidate,
                    self.calibrator,
                    rank,
                    accepted_reason="accepted_unambiguous_entity_resolution",
                    forced_reason=(
                        "ambiguous_entity_resolution" if ambiguous else None
                    ),
                    forced_rejected=ambiguous,
                )
                (accepted if is_accepted else rejected).append(hit)
        return HeadResult(accepted=tuple(accepted), rejected=tuple(rejected))


RawRecallHook = Callable[[str, int], Sequence[RawHeadCandidate]]


class TripleHeadHook:
    """Calibration wrapper only; Task 13 supplies triple recognition semantics."""

    def __init__(self, recall: RawRecallHook, calibrator: HeadCalibrator) -> None:
        self.recall = recall
        self.calibrator = calibrator

    def search(self, query: str, *, k: int) -> HeadResult:
        candidates = tuple(self.recall(query, k))
        if any(candidate.target_kind != "fact" for candidate in candidates):
            raise ValueError("triple recall hooks may return fact targets only")
        return _calibrate_candidates("triple", candidates, self.calibrator)


class UtteranceHead:
    """Optional calibrated recall hook; utterances are navigation, never evidence."""

    def __init__(self, recall: RawRecallHook, calibrator: HeadCalibrator) -> None:
        self.recall = recall
        self.calibrator = calibrator

    def search(self, query: str, *, k: int) -> HeadResult:
        # There is intentionally no default utterance curve in the artifact. A retained
        # utterance ablation must add one before this optional head can be enabled.
        return _calibrate_candidates(
            "utterance", tuple(self.recall(query, k)), self.calibrator
        )


def _calibrate_candidates(
    head: str,
    candidates: Sequence[RawHeadCandidate],
    calibrator: HeadCalibrator,
) -> HeadResult:
    accepted: list[HeadHit] = []
    rejected: list[HeadHit] = []
    for rank, candidate in enumerate(candidates, 1):
        hit, is_accepted = _make_hit(head, candidate, calibrator, rank)
        (accepted if is_accepted else rejected).append(hit)
    return HeadResult(accepted=tuple(accepted), rejected=tuple(rejected))


def _make_hit(
    head: str,
    candidate: RawHeadCandidate,
    calibrator: HeadCalibrator,
    rank: int,
    *,
    accepted_reason: str | None = None,
    forced_reason: str | None = None,
    forced_rejected: bool = False,
) -> tuple[HeadHit, bool]:
    calibrated = calibrator.calibrate(head, candidate.raw_score)
    is_accepted = calibrated.accepted and not forced_rejected
    if forced_reason is not None:
        reason = forced_reason
    elif is_accepted:
        reason = accepted_reason or candidate.reason or f"accepted_{head}"
    else:
        reason = "below_calibrated_acceptance_floor"
    return (
        HeadHit(
            target_kind=candidate.target_kind,
            target_id=candidate.target_id,
            raw_score=float(candidate.raw_score),
            calibrated_score=calibrated.probability,
            rank=rank,
            head=head,
            reason=reason,
            entity_etype=candidate.entity_etype,
        ),
        is_accepted,
    )


def query_alias_allowed(surface: str, *, alias: str | None = None) -> bool:
    """Apply query-only alias guards without changing stored alias tables."""

    query_surface = surface.strip()
    if not query_surface or _is_query_alias_stopword(query_surface):
        return False
    if alias is None:
        return True
    stored_alias = alias.strip()
    if not stored_alias or _is_query_alias_stopword(stored_alias):
        return False
    return len(stored_alias) > 3 or query_surface == stored_alias


def _is_query_alias_stopword(surface: str) -> bool:
    tokens = re.findall(r"\w+", surface.casefold())
    return len(tokens) == 1 and tokens[0] in config.QUERY_ALIAS_STOPWORDS


def _resolution_candidates(
    resolution: Sequence[RawHeadCandidate] | object,
) -> tuple[tuple[RawHeadCandidate, ...], bool]:
    if isinstance(resolution, Sequence):
        candidates = tuple(resolution)
        return candidates, len(candidates) != 1

    status = getattr(resolution, "status", "missing")
    source_candidates = getattr(resolution, "candidates", ())
    candidates = tuple(
        candidate
        if isinstance(candidate, RawHeadCandidate)
        else RawHeadCandidate(
            target_kind="entity",
            target_id=str(candidate.entity_id),
            raw_score=float(candidate.score),
            reason=str(candidate.reason),
        )
        for candidate in source_candidates
    )
    return candidates, status != "resolved" or len(candidates) != 1
