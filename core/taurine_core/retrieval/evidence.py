"""Confidence-preserving candidate fusion and abstained pack helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from taurine_core.retrieval.types import (
    EvidencePack,
    HeadHit,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
    TargetKind,
)


@dataclass(frozen=True)
class FusedHit:
    target_kind: TargetKind
    target_id: str
    score: float
    heads: tuple[str, ...]
    hits: tuple[HeadHit, ...]


def weighted_score_aware_rrf(
    accepted_hits: Sequence[HeadHit],
    *,
    head_weights: Mapping[str, float],
    rrf_k: int,
) -> dict[tuple[TargetKind, str], FusedHit]:
    """Fuse accepted hits without converting facts into endpoint bonuses.

    Duplicate ``(head, target)`` rows are credited once. The strongest calibrated
    row wins, with rank and target ID providing stable tie breaks.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    deduplicated: dict[tuple[str, TargetKind, str], HeadHit] = {}
    for hit in accepted_hits:
        if hit.rank < 1:
            raise ValueError("head hit ranks must start at one")
        if not math.isfinite(hit.calibrated_score) or hit.calibrated_score < 0.0:
            continue
        key = (hit.head, hit.target_kind, hit.target_id)
        current = deduplicated.get(key)
        if current is None or (-hit.calibrated_score, hit.rank, hit.target_id) < (
            -current.calibrated_score,
            current.rank,
            current.target_id,
        ):
            deduplicated[key] = hit

    grouped: dict[tuple[TargetKind, str], list[HeadHit]] = {}
    for hit in deduplicated.values():
        grouped.setdefault((hit.target_kind, hit.target_id), []).append(hit)

    fused: dict[tuple[TargetKind, str], FusedHit] = {}
    for key, rows in grouped.items():
        ordered = tuple(
            sorted(rows, key=lambda hit: (hit.head, hit.rank, hit.target_id))
        )
        score = sum(
            max(0.0, float(head_weights.get(hit.head, 0.0)))
            * min(1.0, float(hit.calibrated_score))
            / (rrf_k + hit.rank)
            for hit in ordered
        )
        if score <= 0.0:
            continue
        fused[key] = FusedHit(
            target_kind=key[0],
            target_id=key[1],
            score=score,
            heads=tuple(hit.head for hit in ordered),
            hits=ordered,
        )
    return dict(
        sorted(
            fused.items(),
            key=lambda item: (-item[1].score, item[0][0], item[0][1]),
        )
    )


def empty_pack(query: RetrievalQuery, decision: RetrievalDecision) -> EvidencePack:
    if decision.accepted:
        raise ValueError("an accepted decision cannot produce an empty abstention pack")
    return EvidencePack(
        decision=decision,
        query=query,
        passages=(),
        facts=(),
        paths=(),
        notes=(),
        trace=RetrievalTrace(
            original_query=query.text,
            rewritten_query=query.text,
        ),
    )
