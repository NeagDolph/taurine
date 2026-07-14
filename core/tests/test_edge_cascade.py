from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from taurine_core.indexer.edge_adjudication import SemanticPair


def _pair(index: int, *, size: int = 40) -> SemanticPair:
    return SemanticPair(
        src_passage_id=f"pa-{index}",
        dst_passage_id=f"pb-{index}",
        src_title=f"Alpha {index}",
        dst_title=f"Beta {index}",
        src_text="a" * size,
        dst_text="b" * size,
        src_content_hash=f"ha-{index}",
        dst_content_hash=f"hb-{index}",
    )


def test_flash_judge_uses_production_prompt_payload_and_fails_open_per_batch() -> None:
    from scripts.eval_edge_cascade import judge_flash_pairs
    from taurine_core.indexer.edge_adjudication import (
        _REASON_SYSTEM_PROMPT,
        _reason_payload,
    )

    pairs = [_pair(0), _pair(1), _pair(2)]
    calls: list[tuple[str, dict]] = []

    def judge(system_prompt: str, payload: dict) -> dict:
        calls.append((system_prompt, payload))
        if len(calls) == 2:
            raise RuntimeError("provider failed")
        return {"pairs": {"0": "same project", "1": "NONE"}}

    artifact = judge_flash_pairs(pairs, judge=judge, batch_size=2, concurrency=1)

    assert calls == [
        (_REASON_SYSTEM_PROMPT, _reason_payload(pairs[:2])),
        (_REASON_SYSTEM_PROMPT, _reason_payload(pairs[2:])),
    ]
    assert artifact["failed_batches"] == 1
    assert [row["verdict"] for row in artifact["verdicts"]] == [
        "keep",
        "drop",
        "keep",
    ]
    assert artifact["verdicts"][2]["fail_open"] is True
    assert artifact["batches"][0]["raw_response"] == {
        "pairs": {"0": "same project", "1": "NONE"}
    }


def test_flash_cache_reuse_never_calls_judge(tmp_path: Path) -> None:
    from scripts.eval_edge_cascade import load_or_judge_flash

    pairs = [_pair(0), _pair(1)]
    cache_path = tmp_path / "flash.json"
    first = load_or_judge_flash(
        pairs,
        cache_path=cache_path,
        judge=lambda system, payload: {"pairs": {"0": "NONE", "1": "related"}},
        batch_size=2,
        concurrency=1,
    )

    def forbidden(system: str, payload: dict) -> dict:
        raise AssertionError("valid Flash cache must make reruns free")

    second = load_or_judge_flash(
        pairs,
        cache_path=cache_path,
        judge=forbidden,
        batch_size=2,
        concurrency=1,
    )

    assert second == first
    assert json.loads(cache_path.read_text()) == first


def test_reranker_features_are_directional_and_bound_passage_text() -> None:
    from scripts.eval_edge_cascade import reranker_features

    pair = SemanticPair(
        **{
            **_pair(0).__dict__,
            "src_text": "a" * 1200,
            "dst_text": "b" * 1300,
        }
    )

    class Scorer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...], str]] = []

        def score_pairs(self, query, documents, *, instruction, deadline_at=None):
            self.calls.append((query, tuple(documents), instruction))
            return (0.8 if query.startswith("a") else 0.2,)

    scorer = Scorer()
    result = reranker_features(pair, scorer)

    assert result.src_to_dst == 0.8
    assert result.dst_to_src == 0.2
    assert result.failed_directions == 0
    assert [call[0] for call in scorer.calls] == ["a" * 1000, "b" * 1000]
    assert "b" * 1000 in scorer.calls[0][1][0]
    assert "b" * 1001 not in scorer.calls[0][1][0]
    assert "a" * 1000 in scorer.calls[1][1][0]
    assert "a" * 1001 not in scorer.calls[1][1][0]


def test_reranker_features_fail_open_per_direction_and_without_scorer() -> None:
    from scripts.eval_edge_cascade import reranker_features

    class PartialScorer:
        def score_pairs(self, query, documents, *, instruction, deadline_at=None):
            if query.startswith("a"):
                raise RuntimeError("one direction failed")
            return (0.3,)

    partial = reranker_features(_pair(0), PartialScorer())
    missing = reranker_features(_pair(0), None)

    assert (partial.src_to_dst, partial.dst_to_src) == (0.0, 0.3)
    assert partial.failed_directions == 1
    assert (missing.src_to_dst, missing.dst_to_src) == (0.0, 0.0)
    assert missing.failed_directions == 2


def test_cohens_kappa_handles_agreement_beyond_chance() -> None:
    from scripts.eval_edge_cascade import cohens_kappa

    assert cohens_kappa([True, True, False, False], [True, False, False, False]) == 0.5
    assert cohens_kappa([True, False], [True, False]) == 1.0


def test_cascade_sweep_requires_real_precise_auto_drops_and_maximizes_resolution() -> None:
    from scripts.eval_edge_cascade import cascade_metrics, select_cascade_bands

    labels = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.02, 0.08, 0.50, 0.70, 0.90, 0.99])

    selected = select_cascade_bands(labels, probabilities)
    metrics = cascade_metrics(
        labels,
        probabilities,
        lo=float(selected["lo"]),
        hi=float(selected["hi"]),
    )

    assert metrics["auto_drop_count"] > 0
    assert metrics["auto_drop_precision"] >= 0.95
    assert metrics["overall_false_drop_rate"] <= 0.05
    assert metrics["auto_resolved_count"] == 6
    assert metrics["middle_band_size"] == 0


def test_edge_cascade_script_direct_invocation_can_import_core_package() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/eval_edge_cascade.py", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--knowledge" in result.stdout
    assert "--flash-cache" in result.stdout
