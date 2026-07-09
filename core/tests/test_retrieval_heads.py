from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from taurine_core.indexer.base_graph import materialize_base_graph
from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.knowledge_store import BM25PassageHit, KnowledgeStore
from taurine_core.retrieval.calibration import HeadCalibrator
from taurine_core.retrieval.heads import (
    BM25PassageHead,
    DensePassageHead,
    ExactAliasHead,
    QueryEntityHead,
    RawHeadCandidate,
    TripleHeadHook,
)
from taurine_core.retrieval.snapshot import IndexSnapshot


_ARTIFACT = (
    Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
)


@pytest.fixture
def calibrator() -> HeadCalibrator:
    artifact = load_calibration(_ARTIFACT)
    return HeadCalibrator(
        artifact,
        mode="chat",
        artifact_hash=artifact.hash,
        config_hash=artifact.config_hash,
        model_manifest=artifact.model_manifest,
    )


def test_dense_passage_head_calibrates_and_can_reject_every_candidate(
    calibrator: HeadCalibrator, tmp_path: Path
) -> None:
    snapshot = _snapshot(tmp_path)
    accepted = DensePassageHead(snapshot, calibrator).search(
        np.array([1.0, 0.0], dtype=np.float32), k=5
    )
    null = DensePassageHead(snapshot, calibrator).search(
        np.array([0.0, -1.0], dtype=np.float32), k=5
    )

    assert [hit.target_id for hit in accepted.accepted] == ["p1"]
    assert accepted.accepted[0].raw_score == pytest.approx(1.0)
    assert {hit.target_id for hit in accepted.rejected} == {"p2"}
    assert null.accepted == ()
    assert {hit.target_id for hit in null.rejected} == {"p1", "p2"}


def test_bm25_requires_artifact_token_coverage_floor(
    calibrator: HeadCalibrator,
) -> None:
    class Store:
        def search_passages_bm25(
            self, query: str, k: int
        ) -> tuple[BM25PassageHit, ...]:
            assert query == "maya launch"
            assert k == 5
            return (
                BM25PassageHit("p1", 10.0, 1.0),
                BM25PassageHit("p2", 10.0, 0.25),
            )

    result = BM25PassageHead(Store(), calibrator).search("maya launch", k=5)

    assert [hit.target_id for hit in result.accepted] == ["p1"]
    assert result.rejected[0].target_id == "p2"
    assert result.rejected[0].reason == "insufficient_query_token_coverage"


def test_exact_alias_uses_word_boundaries_and_credits_each_entity_once(
    calibrator: HeadCalibrator, tmp_path: Path
) -> None:
    head = ExactAliasHead(_snapshot(tmp_path), calibrator)

    result = head.search("Ask Maya about Maya Tung and Maya Labs, not Mayan history")

    assert [(hit.target_id, hit.rank) for hit in result.accepted] == [
        ("e-maya", 1),
        ("e-maya-labs", 2),
    ]
    assert head.search("A Mayan calendar").accepted == ()


def test_query_entity_head_accepts_only_unambiguous_resolutions(
    calibrator: HeadCalibrator,
) -> None:
    def resolve(
        surface: str, etype: str | None, context: str
    ) -> tuple[RawHeadCandidate, ...]:
        assert context == "Maya owns the launch"
        if surface == "Maya":
            return (RawHeadCandidate("entity", "e-maya", 1.0, "exact_alias"),)
        return (
            RawHeadCandidate("entity", "e-a", 0.9, "ambiguous"),
            RawHeadCandidate("entity", "e-b", 0.9, "ambiguous"),
        )

    result = QueryEntityHead(resolve, calibrator).search(
        "Maya owns the launch", (("Maya", "person"), ("Launch", "project"))
    )

    assert [hit.target_id for hit in result.accepted] == ["e-maya"]
    assert {hit.target_id for hit in result.rejected} == {"e-a", "e-b"}
    assert all(hit.reason == "ambiguous_entity_resolution" for hit in result.rejected)


def test_triple_hook_keeps_fact_targets_and_never_adds_endpoint_passages(
    calibrator: HeadCalibrator,
) -> None:
    hook = TripleHeadHook(
        lambda query, k: (
            RawHeadCandidate("fact", "f1", 0.9, "canonical_fact_embedding"),
        ),
        calibrator,
    )

    result = hook.search("Who owns the launch?", k=20)

    assert [(hit.target_kind, hit.target_id) for hit in result.accepted] == [
        ("fact", "f1")
    ]


def test_store_exposes_deterministic_alias_index_for_snapshot_loading(
    tmp_path: Path,
) -> None:
    store = KnowledgeStore(tmp_path / "aliases.db")
    try:
        for entity_id in ("e2", "e1"):
            store.insert_entity(
                entity_id=entity_id,
                canonical=entity_id,
                normalized=entity_id,
                etype="person",
                df=1,
                traversal_eligible=True,
                ambiguity_key=entity_id,
                build_id="build-1",
            )
        store.insert_alias("e2", "MAYA", "mention", 0.9)
        store.insert_alias("e1", "Maya", "mention", 1.0)

        assert store.alias_index() == {"maya": ("e1", "e2")}
    finally:
        store.close()


def _snapshot(tmp_path: Path) -> IndexSnapshot:
    return IndexSnapshot(
        build_id="build-1",
        database_path=tmp_path / "knowledge.db",
        calibration_hash="sha256:initial-conservative-2026-07-09",
        passage_ids=("p1", "p2"),
        passage_matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        fact_ids=(),
        fact_matrix=np.empty((0, 2), dtype=np.float32),
        alias_index={
            "maya": ("e-maya",),
            "maya tung": ("e-maya",),
            "maya labs": ("e-maya-labs",),
        },
        base_graph=materialize_base_graph(
            build_id="build-1",
            node_ids=("p1", "p2"),
            node_kinds=("passage", "passage"),
            edges=(),
        ),
        projection=(),
        quality_ranks={},
    )
