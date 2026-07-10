"""Versioned score calibration and conservative retrieval abstention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from taurine_core.evaluation.graph_retrieval import (
    CalibrationArtifact,
    CalibrationCurve,
    ModeCalibration,
)
from taurine_core.retrieval.types import HeadHit, RetrievalDecision, RetrievalMode


_SUPPORTED_ARTIFACT_VERSION = "1"


class CalibrationMismatch(ValueError):
    """Raised before ranking when runtime identity and calibration differ."""


@dataclass(frozen=True)
class CalibratedValue:
    raw_score: float
    probability: float
    acceptance_floor: float
    accepted: bool


class HeadCalibrator:
    """A compatibility-checked view over one mode's monotonic curves."""

    def __init__(
        self,
        artifact: CalibrationArtifact,
        *,
        mode: RetrievalMode,
        artifact_hash: str,
        config_hash: str,
        model_manifest: Mapping[str, str],
    ) -> None:
        if artifact.artifact_version != _SUPPORTED_ARTIFACT_VERSION:
            raise CalibrationMismatch(
                f"unsupported calibration artifact version {artifact.artifact_version!r}"
            )
        if artifact_hash != artifact.hash:
            raise CalibrationMismatch(
                f"calibration artifact hash mismatch: expected {artifact.hash!r}, "
                f"got {artifact_hash!r}"
            )
        if config_hash != artifact.config_hash:
            raise CalibrationMismatch(
                f"calibration config hash mismatch: expected {artifact.config_hash!r}, "
                f"got {config_hash!r}"
            )
        runtime_manifest = dict(model_manifest)
        if runtime_manifest != artifact.model_manifest:
            names = ", ".join(
                sorted(set(runtime_manifest) ^ set(artifact.model_manifest))
                or sorted(
                    name
                    for name in runtime_manifest
                    if runtime_manifest[name] != artifact.model_manifest.get(name)
                )
            )
            raise CalibrationMismatch(f"calibration model manifest mismatch: {names}")
        self.artifact = artifact
        self.mode = mode
        self.mode_calibration = artifact.modes[mode]

    def probability(self, head: str, raw_score: float) -> float:
        curve = self._curve(head)
        return interpolate_curve(curve, raw_score)

    def calibrate(self, head: str, raw_score: float) -> CalibratedValue:
        curve = self._curve(head)
        probability = interpolate_curve(curve, raw_score)
        return CalibratedValue(
            raw_score=float(raw_score),
            probability=probability,
            acceptance_floor=curve.acceptance_floor,
            accepted=probability >= curve.acceptance_floor,
        )

    def accepts(self, head: str, raw_score: float) -> bool:
        return self.calibrate(head, raw_score).accepted

    def acceptance_floor(self, head: str) -> float:
        """The head's calibrated acceptance floor; the one place floors are read."""

        return float(self._curve(head).acceptance_floor)

    @property
    def bm25_min_token_coverage(self) -> float:
        return self.mode_calibration.bm25_min_token_coverage

    @property
    def head_weights(self) -> Mapping[str, float]:
        return self.mode_calibration.head_weights

    @property
    def rrf_k(self) -> int:
        return self.mode_calibration.rrf_k

    @property
    def mmr_lambda(self) -> float:
        return self.mode_calibration.mmr_lambda

    @property
    def source_diversity_penalty(self) -> float:
        return self.mode_calibration.source_diversity_penalty

    @property
    def note_support_bonus(self) -> float:
        return self.mode_calibration.note_support_bonus

    @property
    def note_support_bonus_cap(self) -> float:
        return self.mode_calibration.note_support_bonus_cap

    def _curve(self, head: str) -> CalibrationCurve:
        try:
            return self.mode_calibration.heads[head]
        except KeyError as exc:
            raise KeyError(
                f"no {self.mode!r} calibration curve for head {head!r}"
            ) from exc


def interpolate_curve(curve: CalibrationCurve, raw_score: float) -> float:
    """Clamp and linearly interpolate an already-validated monotonic curve."""

    raw = float(raw_score)
    if not math.isfinite(raw):
        return 0.0
    knots = curve.knots
    if raw <= knots[0][0]:
        return float(knots[0][1])
    if raw >= knots[-1][0]:
        return float(knots[-1][1])
    for (left_raw, left_probability), (right_raw, right_probability) in zip(
        knots, knots[1:], strict=False
    ):
        if raw <= right_raw:
            position = (raw - left_raw) / (right_raw - left_raw)
            return float(
                left_probability + position * (right_probability - left_probability)
            )
    return float(knots[-1][1])


class AbstentionPolicy:
    """The explicit conservative rule table used before and after expansion."""

    def __init__(self, artifact: CalibrationArtifact, *, mode: RetrievalMode) -> None:
        self.mode = mode
        self.rules: ModeCalibration = artifact.modes[mode]

    def pre_ppr(
        self,
        accepted_hits: Sequence[HeadHit],
        rejected_hits: Sequence[HeadHit] = (),
        *,
        unavailable_heads: tuple[str, ...] = (),
    ) -> RetrievalDecision:
        qualified, disqualified = self._qualified(accepted_hits)
        if unavailable_heads:
            # §17.3: with one or more heads unavailable, cross-head
            # corroboration may be missing rather than absent, so acceptance
            # requires the strictest disjunct — an accepted hit at ≥1.1× its
            # own head's acceptance floor (capped at 1.0).
            strict: list[HeadHit] = []
            lenient: list[HeadHit] = []
            for hit in qualified:
                curve = self.rules.heads.get(hit.head)
                floor = min(1.0, 1.1 * curve.acceptance_floor) if curve else 1.0
                if _safe_probability(hit.calibrated_score) >= floor:
                    strict.append(hit)
                else:
                    lenient.append(hit)
            qualified, disqualified = tuple(strict), (*disqualified, *lenient)
        heads = _head_names(qualified)
        rejected = _head_names((*rejected_hits, *disqualified))
        if not qualified:
            return RetrievalDecision(
                accepted=False,
                confidence=0.0,
                reason=(
                    "no_head_cleared_degraded_strict_floor"
                    if unavailable_heads
                    else "no_head_cleared_acceptance_floor"
                ),
                accepted_heads=(),
                rejected_heads=rejected,
            )
        top = max(_safe_probability(hit.calibrated_score) for hit in qualified)
        if any(hit.head in {"exact", "entity"} for hit in qualified):
            reason = "accepted_high_precision_head"
        elif any(hit.head == "triple" for hit in qualified):
            reason = "accepted_recognized_triple"
        else:
            reason = "accepted_recall_head"
        return RetrievalDecision(
            accepted=True,
            confidence=top,
            reason=reason,
            accepted_heads=heads,
            rejected_heads=rejected,
        )

    def post_rerank(
        self,
        accepted_hits: Sequence[HeadHit],
        evidence_scores: Sequence[float],
        rejected_hits: Sequence[HeadHit] = (),
        *,
        recognized_triple_confidence: float = 0.0,
        evidence_target_ids: Sequence[str] | None = None,
    ) -> RetrievalDecision:
        qualified, disqualified = self._qualified(accepted_hits)
        heads = _head_names(qualified)
        rejected = _head_names((*rejected_hits, *disqualified))
        finite_scores = tuple(
            _safe_probability(score)
            for score in evidence_scores
            if math.isfinite(float(score))
        )
        top = max(finite_scores, default=0.0)
        if top < self.rules.evidence_floor:
            return RetrievalDecision(
                accepted=False,
                confidence=top,
                reason="no_evidence_above_floor",
                accepted_heads=heads,
                rejected_heads=rejected,
            )

        independent_heads = set(heads)
        high_precision = bool(independent_heads & {"exact", "entity"})
        if evidence_target_ids is None:
            has_agreement = (
                len(independent_heads) >= self.rules.minimum_independent_heads
            )
        else:
            selected = set(evidence_target_ids)
            heads_by_target: dict[str, set[str]] = {}
            for hit in qualified:
                if hit.target_kind == "passage" and hit.target_id in selected:
                    heads_by_target.setdefault(hit.target_id, set()).add(hit.head)
            has_agreement = any(
                len(target_heads) >= self.rules.minimum_independent_heads
                for target_heads in heads_by_target.values()
            )
        if has_agreement:
            reason = "accepted_independent_head_agreement"
        elif high_precision:
            reason = "accepted_high_precision_head"
        elif top >= self.rules.single_head_override:
            reason = "accepted_single_head_override"
        else:
            return RetrievalDecision(
                accepted=False,
                confidence=top,
                reason="single_head_without_override",
                accepted_heads=heads,
                rejected_heads=rejected,
            )

        confidence = max(
            top,
            min(1.0, float(recognized_triple_confidence)),
            max(
                (_safe_probability(hit.calibrated_score) for hit in qualified),
                default=0.0,
            ),
        )
        return RetrievalDecision(
            accepted=True,
            confidence=confidence,
            reason=reason,
            accepted_heads=heads,
            rejected_heads=rejected,
        )

    def _qualified(
        self, hits: Sequence[HeadHit]
    ) -> tuple[tuple[HeadHit, ...], tuple[HeadHit, ...]]:
        qualified: list[HeadHit] = []
        rejected: list[HeadHit] = []
        for hit in hits:
            curve = self.rules.heads.get(hit.head)
            rejected_reason = hit.reason.startswith(
                ("below_", "insufficient_", "ambiguous_", "rejected_")
            )
            if (
                curve is not None
                and not rejected_reason
                and _safe_probability(hit.calibrated_score) >= curve.acceptance_floor
            ):
                qualified.append(hit)
            else:
                rejected.append(hit)
        return tuple(qualified), tuple(rejected)


def _head_names(hits: Sequence[HeadHit]) -> tuple[str, ...]:
    return tuple(sorted({hit.head for hit in hits}))


def _safe_probability(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))
