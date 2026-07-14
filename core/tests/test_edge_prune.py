from __future__ import annotations

from pathlib import Path

import numpy as np
import subprocess
import sys

from tests.conftest import make_note
from tests.test_edge_adjudication import Tokenizer, constant_embedder
from taurine_core import config
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.indexer.edge_adjudication import SemanticPair
from taurine_core.indexer.knowledge_store import KnowledgeStore


def _pair(src: str = "pa", dst: str = "pb") -> SemanticPair:
    return SemanticPair(
        src_passage_id=src,
        dst_passage_id=dst,
        src_title="Alpha",
        dst_title="Beta",
        src_text="Adi plans the Saturday potluck.",
        dst_text="Adi will bring a grill to the potluck.",
        src_content_hash="ha",
        dst_content_hash="hb",
    )


def _features(cosine: float, *, shared: tuple[str, ...] = ()):
    from taurine_core.indexer.edge_prune import EdgeFeatures

    return EdgeFeatures(
        cosine=cosine,
        shared_entity_count=len(shared),
        min_shared_entity_idf=2.0 if shared else 0.0,
        same_source=0.0,
        title_jaccard=0.0,
        length_ratio=0.9,
        shared_entities=shared,
        src_source="gmail",
        dst_source="imessage",
    )


def test_decide_keeps_and_drops_from_embedded_logistic_score(monkeypatch) -> None:
    from taurine_core.indexer import edge_prune

    low = _pair("pa", "pb")
    high = _pair("pc", "pd")
    monkeypatch.setattr(edge_prune, "INTERCEPT", -5.0)
    monkeypatch.setattr(
        edge_prune,
        "WEIGHTS",
        {name: (10.0 if name == "cosine" else 0.0) for name in edge_prune.FEATURE_NAMES},
    )
    monkeypatch.setattr(edge_prune, "DECISION_THRESHOLD", 0.5)

    outcome = edge_prune.decide(
        [low, high],
        {low.pair: _features(0.1), high.pair: _features(0.9, shared=("Adi",))},
    )

    assert outcome.dropped == frozenset({low.pair})
    assert set(outcome.reasons) == {high.pair}
    assert outcome.reasoned_count == 1
    assert outcome.dropped_count == 1


def test_deterministic_reasons_prefer_shared_entities_then_sources() -> None:
    from taurine_core.indexer.edge_prune import deterministic_reason, is_template_reason

    pair = _pair()
    shared = deterministic_reason(pair, _features(0.8, shared=("Adi", "Saturday potluck")))
    sources = deterministic_reason(pair, _features(0.8))

    assert shared == "Both mention Adi and Saturday potluck"
    assert sources == "Semantically similar passages from gmail and imessage"
    assert is_template_reason(shared) is True
    assert is_template_reason(sources) is True
    assert is_template_reason("Both plan the same Saturday potluck") is False


def test_build_local_mode_skips_llm_and_persists_template_reason(
    tmp_path: Path, monkeypatch
) -> None:
    from taurine_core.indexer import edge_prune

    monkeypatch.setattr(config, "SEMANTIC_PRUNE_MODE", "local")
    monkeypatch.setattr(edge_prune, "INTERCEPT", 10.0)
    monkeypatch.setattr(
        edge_prune,
        "WEIGHTS",
        {name: 0.0 for name in edge_prune.FEATURE_NAMES},
    )

    def forbidden_judge(system_prompt: str, payload: dict) -> dict:
        raise AssertionError("local semantic pruning must not call the LLM")

    notes = [
        make_note("a", "Alpha", "Potluck planning with Adi on Saturday."),
        make_note("b", "Beta", "Bring the grill for the Saturday potluck."),
    ]
    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=constant_embedder,
            embedding_dim=16,
            edge_adjudicator=forbidden_judge,
        )
    ).build(notes, target, replace_derived=True)

    assert report.semantic_reasoned_edge_count == 1
    store = KnowledgeStore(target, read_only=True)
    try:
        (row,) = store._conn.execute(
            "SELECT reason FROM passage_semantic_edges"
        ).fetchall()
        assert row["reason"].startswith("Semantically similar passages from")
    finally:
        store.close()


def test_fit_metrics_report_agreement_and_false_drops() -> None:
    from scripts.fit_edge_prune import classification_metrics

    metrics = classification_metrics(
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.1, 0.8, 0.2]),
        threshold=0.5,
    )

    assert metrics["drop_agreement"] == 0.5
    assert metrics["false_drop_rate"] == 0.5
    assert metrics["predicted_drops"] == 2


def test_fit_threshold_maximizes_agreement_under_false_drop_cap() -> None:
    from scripts.fit_edge_prune import select_threshold

    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.4, 0.2])

    threshold, metrics = select_threshold(
        labels,
        probabilities,
        thresholds=(0.3, 0.5, 0.85),
        false_drop_cap=0.05,
    )

    assert threshold == 0.5
    assert metrics["drop_agreement"] == 1.0
    assert metrics["false_drop_rate"] == 0.0


def test_fit_script_direct_invocation_can_import_core_package() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/fit_edge_prune.py", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--knowledge" in result.stdout
