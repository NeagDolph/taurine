from __future__ import annotations

from pathlib import Path

import pytest

from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.retrieval.calibration import (
    AbstentionPolicy,
    CalibrationMismatch,
    HeadCalibrator,
)
from taurine_core.retrieval.evidence import empty_pack, weighted_score_aware_rrf
from taurine_core.retrieval.types import HeadHit, RetrievalQuery


_ARTIFACT = (
    Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
)


def test_curve_interpolates_clamps_and_applies_acceptance_floor() -> None:
    artifact = load_calibration(_ARTIFACT)
    calibrator = HeadCalibrator(
        artifact,
        mode="chat",
        artifact_hash=artifact.hash,
        config_hash=artifact.config_hash,
        model_manifest=artifact.model_manifest,
    )

    assert calibrator.probability("dense", -9.0) == pytest.approx(0.0)
    assert calibrator.probability("dense", 9.0) == pytest.approx(1.0)
    assert calibrator.probability("dense", 0.67) == pytest.approx(0.75)
    assert calibrator.accepts("dense", 0.67)
    assert not calibrator.accepts("dense", 0.58)
    assert calibrator.rrf_k == 60
    assert calibrator.head_weights["dense"] == pytest.approx(1.0)
    assert calibrator.mmr_lambda == pytest.approx(0.2)
    assert calibrator.note_support_bonus == pytest.approx(0.05)
    assert calibrator.note_support_bonus_cap == pytest.approx(0.1)
    assert calibrator.source_diversity_penalty == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"artifact_hash": "sha256:other"}, "artifact hash"),
        ({"config_hash": "sha256:other"}, "config hash"),
        ({"model_manifest": {"embedding": "wrong"}}, "model manifest"),
    ],
)
def test_calibration_mismatch_fails_closed(
    overrides: dict[str, object], message: str
) -> None:
    artifact = load_calibration(_ARTIFACT)
    kwargs: dict[str, object] = {
        "artifact_hash": artifact.hash,
        "config_hash": artifact.config_hash,
        "model_manifest": artifact.model_manifest,
    }
    kwargs.update(overrides)

    with pytest.raises(CalibrationMismatch, match=message):
        HeadCalibrator(artifact, mode="live", **kwargs)  # type: ignore[arg-type]


def test_calibration_rejects_incomplete_runtime_model_manifest() -> None:
    artifact = load_calibration(_ARTIFACT)

    with pytest.raises(CalibrationMismatch, match="model manifest"):
        HeadCalibrator(
            artifact,
            mode="live",
            artifact_hash=artifact.hash,
            config_hash=artifact.config_hash,
            model_manifest={"embedding": artifact.model_manifest["embedding"]},
        )


def test_weighted_score_aware_rrf_uses_scores_and_credits_exact_once() -> None:
    hits = (
        _hit("dense", "p1", 0.8, 1),
        _hit("bm25", "p1", 0.7, 2),
        _hit("exact", "e1", 0.95, 1, target_kind="entity"),
        _hit("exact", "e1", 0.90, 2, target_kind="entity"),
        _hit("triple", "f1", 0.9, 1, target_kind="fact"),
    )

    fused = weighted_score_aware_rrf(
        hits,
        head_weights={"dense": 2.0, "bm25": 1.0, "exact": 1.0, "triple": 1.0},
        rrf_k=60,
    )

    assert fused[("passage", "p1")].score == pytest.approx(2.0 * 0.8 / 61 + 0.7 / 62)
    assert fused[("entity", "e1")].score == pytest.approx(0.95 / 61)
    assert fused[("fact", "f1")].score == pytest.approx(0.9 / 61)
    assert ("passage", "f1") not in fused


def test_pre_and_post_abstention_return_explicit_empty_outcomes() -> None:
    artifact = load_calibration(_ARTIFACT)
    policy = AbstentionPolicy(artifact, mode="chat")
    query = RetrievalQuery(text="uh well blargle", mode="chat", now=1.0)

    pre = policy.pre_ppr(())
    assert not pre.accepted
    assert pre.reason == "no_head_cleared_acceptance_floor"
    pack = empty_pack(query, pre)
    assert not pack.decision.accepted
    assert pack.passages == pack.facts == pack.paths == pack.notes == ()

    unqualified = policy.pre_ppr((_hit("dense", "p0", 0.1, 1),))
    assert not unqualified.accepted
    assert unqualified.reason == "no_head_cleared_acceptance_floor"

    one_head = (_hit("dense", "p1", 0.8, 1),)
    post = policy.post_rerank(one_head, (0.80,))
    assert not post.accepted
    assert post.reason == "single_head_without_override"

    accepted = policy.post_rerank(
        one_head + (_hit("bm25", "p1", 0.7, 1),),
        (0.80,),
    )
    assert accepted.accepted
    assert accepted.reason == "accepted_independent_head_agreement"


def test_post_abstention_requires_evidence_floor_and_allows_exact_override() -> None:
    artifact = load_calibration(_ARTIFACT)
    policy = AbstentionPolicy(artifact, mode="live")

    low = policy.post_rerank((_hit("exact", "e1", 1.0, 1, "entity"),), (0.70,))
    assert not low.accepted
    assert low.reason == "no_evidence_above_floor"

    high_precision = policy.post_rerank(
        (_hit("exact", "e1", 1.0, 1, "entity"),),
        (0.80,),
    )
    assert high_precision.accepted
    assert high_precision.reason == "accepted_high_precision_head"


def test_post_abstention_requires_independent_heads_to_agree_on_selected_evidence() -> None:
    artifact = load_calibration(_ARTIFACT)
    policy = AbstentionPolicy(artifact, mode="chat")
    hits = (
        _hit("dense", "p1", 0.8, 1),
        _hit("bm25", "p2", 0.8, 1),
    )

    rejected = policy.post_rerank(
        hits,
        (0.8,),
        evidence_target_ids=("p1",),
    )
    accepted = policy.post_rerank(
        hits + (_hit("bm25", "p1", 0.8, 2),),
        (0.8,),
        evidence_target_ids=("p1",),
    )

    assert not rejected.accepted
    assert rejected.reason == "single_head_without_override"
    assert accepted.accepted
    assert accepted.reason == "accepted_independent_head_agreement"


def _hit(
    head: str,
    target_id: str,
    score: float,
    rank: int,
    target_kind: str = "passage",
) -> HeadHit:
    return HeadHit(
        target_kind=target_kind,  # type: ignore[arg-type]
        target_id=target_id,
        raw_score=score,
        calibrated_score=score,
        rank=rank,
        head=head,
        reason="accepted",
    )
