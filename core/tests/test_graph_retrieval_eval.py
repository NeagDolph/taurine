from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from taurine_core.evaluation.graph_retrieval import (
    EvaluationExample,
    citation_coverage,
    load_calibration,
    load_jsonl,
    mrr,
    ndcg_at,
    recall_at,
    support_set_recall,
    validate_split_isolation,
)


def test_retrieval_metrics_match_hand_calculated_rankings() -> None:
    ranking = ["p1", "p2", "p3", "p4"]
    relevant = {"p2", "p4"}

    assert recall_at(ranking, relevant, 2) == pytest.approx(0.5)
    assert recall_at(ranking, relevant, 4) == pytest.approx(1.0)
    assert ndcg_at(ranking, relevant, 4) == pytest.approx(0.6509209298)
    assert mrr(ranking, relevant) == pytest.approx(0.5)
    assert support_set_recall(ranking[:2], relevant) == pytest.approx(0.5)
    assert support_set_recall(ranking, relevant) == pytest.approx(1.0)


def test_metrics_have_explicit_empty_set_semantics() -> None:
    assert recall_at(["p1"], set(), 5) == 0.0
    assert ndcg_at(["p1"], set(), 5) == 0.0
    assert mrr(["p1"], set()) == 0.0
    assert support_set_recall(["p1"], set()) == 0.0
    assert citation_coverage(0, 0) == 1.0
    assert citation_coverage(1, 0) == 0.0
    assert citation_coverage(4, 3) == pytest.approx(0.75)


def test_jsonl_loader_rejects_invalid_lines_and_duplicate_ids(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"example_id":"a"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid JSON on line 2"):
        load_jsonl(invalid)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join(
            [
                json.dumps(_example("same", "train", ["n1"])),
                json.dumps(_example("same", "held_out", ["n2"])),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate example_id"):
        load_jsonl(duplicate)


def test_split_isolation_rejects_source_note_leakage() -> None:
    rows = (
        EvaluationExample.model_validate(_example("a", "train", ["n1", "n2"])),
        EvaluationExample.model_validate(_example("b", "calibration", ["n3"])),
        EvaluationExample.model_validate(_example("c", "held_out", ["n1"])),
    )

    with pytest.raises(ValueError, match=r"n1.*held_out.*train|n1.*train.*held_out"):
        validate_split_isolation(rows)


def test_checked_in_calibration_artifact_is_strict_and_complete() -> None:
    path = Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
    artifact = load_calibration(path)

    assert artifact.artifact_version == "1"
    assert artifact.hash
    assert artifact.extraction.entity_thresholds["person"] == pytest.approx(0.5)
    assert artifact.extraction.relation_thresholds["responsible_for"] == pytest.approx(0.45)
    assert artifact.extraction.promotion_thresholds["responsible_for"] == pytest.approx(0.7)
    assert set(artifact.modes) == {"live", "chat", "tool"}
    for mode in artifact.modes.values():
        assert {"dense", "bm25", "exact", "entity", "triple"} <= set(mode.heads)
        assert 0.0 <= mode.evidence_floor <= 1.0
        assert 0.0 <= mode.single_head_override <= 1.0
        assert 0.0 <= mode.bm25_min_token_coverage <= 1.0
    assert 0.0 <= artifact.nli.entailment_floor <= 1.0


def test_calibration_rejects_non_monotonic_curves(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "artifact_version": "1",
                "hash": "sha256:bad",
                "model_manifest": {"embedding": "fake"},
                "config_hash": "sha256:config",
                "extraction": {
                    "entity_thresholds": {"person": 0.5},
                    "relation_thresholds": {"responsible_for": 0.45},
                    "promotion_thresholds": {"responsible_for": 0.7},
                },
                "modes": {
                    mode: {
                        "heads": {
                            head: {
                                "knots": [[0.0, 0.8], [1.0, 0.2]],
                                "acceptance_floor": 0.5,
                                "sample_count": 2,
                                "trained_at": "2026-07-09",
                                "held_out": {},
                            }
                            for head in ("dense", "bm25", "exact", "entity", "triple")
                        },
                        "evidence_floor": 0.5,
                        "single_head_override": 0.9,
                        "minimum_independent_heads": 2,
                    }
                    for mode in ("live", "chat", "tool")
                },
                "triple_recognition": {
                    "live_floor": 0.8,
                    "chat_floor": 0.7,
                    "tool_floor": 0.68,
                    "live_fallback_embedding_floor": 0.9,
                    "chat_fallback_embedding_floor": 0.85,
                    "tool_fallback_embedding_floor": 0.83,
                    "chat_ambiguity_low": 0.6,
                    "chat_ambiguity_high": 0.8,
                    "tool_ambiguity_low": 0.58,
                    "tool_ambiguity_high": 0.78,
                },
                "rerank": {"live_floor": 0.7, "chat_floor": 0.6},
                "nli": {"entailment_floor": 0.8},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="monotonic"):
        load_calibration(path)


def _example(example_id: str, split: str, source_note_ids: list[str]) -> dict:
    return {
        "example_id": example_id,
        "split": split,
        "source_note_ids": source_note_ids,
        "kind": "retrieval_chat",
        "input": {"query": "example"},
        "labels": {"passage_ids": ["p1"]},
    }
