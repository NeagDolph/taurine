from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import monotonic

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.base_graph import BaseGraph
from taurine_core.retrieval.calibration import HeadCalibrator
from taurine_core.retrieval.snapshot import IndexSnapshot
from taurine_core.retrieval.triples import (
    FactContext,
    FactRecallIndex,
    TripleRecognizer,
)
from taurine_core.retrieval.types import RetrievalQuery, TemporalIntent
from taurine_core.models_local import ModelDeadlineExceeded


_ARTIFACT = (
    Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
)


class ContextStore:
    def __init__(self, contexts: tuple[FactContext, ...]) -> None:
        self.contexts = contexts
        self.requested: tuple[str, ...] = ()

    def fact_contexts(self, fact_ids: tuple[str, ...]) -> tuple[FactContext, ...]:
        self.requested = fact_ids
        wanted = set(fact_ids)
        return tuple(row for row in self.contexts if row.fact_id in wanted)


class FakeAmbiguityAcceptor:
    def __init__(self, accepted: set[str] | Exception) -> None:
        self.accepted = accepted
        self.calls: list[tuple[RetrievalQuery, tuple[object, ...], float | None]] = []

    def accept_fact_ids(
        self,
        query: RetrievalQuery,
        candidates: tuple[object, ...],
        *,
        deadline_at: float | None = None,
    ) -> set[str]:
        self.calls.append((query, candidates, deadline_at))
        if isinstance(self.accepted, Exception):
            raise self.accepted
        return self.accepted


class FakeScorer:
    def __init__(self, scores: tuple[float, ...] | Exception) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...], str, float | None]] = []

    def score_pairs(
        self,
        query: str,
        documents: tuple[str, ...],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> tuple[float, ...]:
        self.calls.append((query, documents, instruction, deadline_at))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


@pytest.fixture
def artifact():
    return load_calibration(_ARTIFACT)


def _calibrator(mode: str = "chat") -> HeadCalibrator:
    artifact = load_calibration(_ARTIFACT)
    return HeadCalibrator(
        artifact,
        mode=mode,  # type: ignore[arg-type]
        artifact_hash=artifact.hash,
        config_hash=artifact.config_hash,
        model_manifest=artifact.model_manifest,
    )


def _recognition_kwargs(artifact, mode: str) -> dict[str, float]:
    calibration = artifact.triple_recognition
    if mode == "live":
        return {
            "recognition_floor": calibration.live_floor,
            "fallback_embedding_floor": calibration.live_fallback_embedding_floor,
        }
    if mode == "tool":
        return {
            "recognition_floor": calibration.tool_floor,
            "fallback_embedding_floor": calibration.tool_fallback_embedding_floor,
            "ambiguity_low": calibration.tool_ambiguity_low,
            "ambiguity_high": calibration.tool_ambiguity_high,
        }
    return {
        "recognition_floor": calibration.chat_floor,
        "fallback_embedding_floor": calibration.chat_fallback_embedding_floor,
        "ambiguity_low": calibration.chat_ambiguity_low,
        "ambiguity_high": calibration.chat_ambiguity_high,
    }


def _snapshot(tmp_path: Path) -> IndexSnapshot:
    return IndexSnapshot(
        build_id="build-1",
        database_path=tmp_path / "knowledge.db",
        calibration_hash="sha256:initial-conservative-2026-07-09",
        passage_ids=(),
        passage_matrix=np.empty((0, 2), dtype=np.float32),
        fact_ids=("f1", "f2", "f3"),
        fact_matrix=np.array(
            [[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]], dtype=np.float32
        ),
        alias_index={},
        base_graph=BaseGraph(
            build_id="build-1",
            node_ids=(),
            node_kinds=(),
            row_by_id={},
            matrix=csr_matrix((0, 0), dtype=np.float64),
            dangling_rows=(),
            edges=(),
        ),
        projection=(),
        quality_ranks={},
    )


def _context(
    fact_id: str,
    *,
    passage_id: str,
    status: str = "active",
    valid_from: float | None = None,
    valid_to: float | None = None,
) -> FactContext:
    return FactContext(
        fact_id=fact_id,
        subject_entity_id="e-maya",
        subject="Maya",
        predicate="owns_or_has",
        predicate_description="owns or has",
        object_entity_id="e-launch",
        object="Launch",
        source_passage_id=passage_id,
        source_note_id="n1",
        assertion_text=f"Maya owns Launch ({fact_id}).",
        observed_at=100.0,
        valid_from=valid_from,
        valid_to=valid_to,
        status=status,  # type: ignore[arg-type]
        recognition_document=(
            "Triple: Maya — owns_or_has (owns or has) — Launch\n"
            f"Assertion: Maya owns Launch ({fact_id})."
        ),
        logical_key=("e-maya", "owns_or_has", "e-launch"),
    )


def test_fact_recall_is_cosine_ranked_calibrated_and_bounded(tmp_path: Path) -> None:
    second = replace(
        _context("f2", passage_id="p2"),
        object_entity_id="e-second",
        object="Second",
        logical_key=("e-maya", "owns_or_has", "e-second"),
    )
    recall = FactRecallIndex(
        _snapshot(tmp_path),
        ContextStore((_context("f1", passage_id="p1"), second)),
        _calibrator(),
    )

    hits = recall.recall(
        np.array([1.0, 0.0], dtype=np.float32),
        RetrievalQuery("Who owns Launch?", "chat", now=200.0),
        k=2,
    )

    assert [(hit.fact_id, hit.rank) for hit in hits] == [("f1", 1), ("f2", 2)]
    assert hits[0].embedding_score == pytest.approx(1.0)
    assert hits[0].calibrated_score == pytest.approx(1.0)
    assert hits[0].accepted


def test_fact_recall_filters_temporal_status_before_top_k(tmp_path: Path) -> None:
    fact_ids = tuple(f"old-{index:02d}" for index in range(21)) + ("active",)
    snapshot = replace(
        _snapshot(tmp_path),
        fact_ids=fact_ids,
        fact_matrix=np.array(
            [[1.0, index / 1000.0] for index in range(21)] + [[0.8, 0.6]],
            dtype=np.float32,
        ),
    )
    contexts = tuple(
        replace(
            _context(fact_id, passage_id=f"p-{index}"),
            status="superseded",
            object_entity_id=f"e-old-{index}",
            object=f"Old {index}",
            logical_key=("e-maya", "owns_or_has", f"e-old-{index}"),
        )
        for index, fact_id in enumerate(fact_ids[:-1])
    ) + (_context("active", passage_id="p-active"),)

    hits = FactRecallIndex(snapshot, ContextStore(contexts), _calibrator()).recall(
        np.array([1.0, 0.0], dtype=np.float32),
        RetrievalQuery("current owner", "chat", now=200.0),
        k=1,
    )

    assert [hit.fact_ids for hit in hits] == [("active",)]


def test_fact_recall_tops_by_logical_triple_and_keeps_all_compatible_provenance(
    tmp_path: Path,
) -> None:
    contexts = (_context("f1", passage_id="p1"), _context("f2", passage_id="p2"))
    hits = FactRecallIndex(
        _snapshot(tmp_path), ContextStore(contexts), _calibrator()
    ).recall(
        np.array([1.0, 0.0], dtype=np.float32),
        RetrievalQuery("Who owns Launch?", "chat", now=200.0),
        k=1,
    )

    assert len(hits) == 1
    assert hits[0].fact_ids == ("f1", "f2")
    assert [row.source_passage_id for row in hits[0].contexts] == ["p1", "p2"]


def test_recognizer_groups_duplicate_logical_facts_and_seeds_endpoints_once(
    tmp_path: Path, artifact
) -> None:
    contexts = (_context("f1", passage_id="p1"), _context("f2", passage_id="p2"))
    store = ContextStore(contexts)
    scorer = FakeScorer((0.91,))
    recognizer = TripleRecognizer(
        FactRecallIndex(_snapshot(tmp_path), store, _calibrator()),
        store,
        scorer,
        mode="chat",
        **_recognition_kwargs(artifact, "chat"),
    )

    result = recognizer.recognize(
        RetrievalQuery("Who owns Launch?", "chat", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-launch"},
        required_entity_ids={"e-launch"},
    )

    assert not result.degraded
    assert len(result.triples) == 1
    assert result.triples[0].fact_ids == ("f1", "f2")
    assert result.triples[0].source_passage_ids == ("p1", "p2")
    assert result.entity_seed_weights == {
        "e-maya": pytest.approx(0.91),
        "e-launch": pytest.approx(0.91),
    }
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0][1]) == 1
    assert "canonical fact triple" in scorer.calls[0][2]


@pytest.mark.parametrize(
    ("intent", "status", "valid_from", "valid_to", "accepted"),
    [
        (None, "active", None, None, True),
        (None, "superseded", None, 150.0, False),
        (TemporalIntent("historical"), "superseded", None, 150.0, True),
        (TemporalIntent("current"), "active", 250.0, None, False),
        (TemporalIntent("at_time", 120.0), "superseded", 50.0, 150.0, True),
        (TemporalIntent("any"), "conflicted", None, None, True),
    ],
)
def test_recognizer_applies_strict_temporal_and_status_compatibility(
    tmp_path: Path,
    artifact,
    intent: TemporalIntent | None,
    status: str,
    valid_from: float | None,
    valid_to: float | None,
    accepted: bool,
) -> None:
    context = _context(
        "f1",
        passage_id="p1",
        status=status,
        valid_from=valid_from,
        valid_to=valid_to,
    )
    store = ContextStore((context,))
    recognizer = TripleRecognizer(
        FactRecallIndex(_snapshot(tmp_path), store, _calibrator()),
        store,
        FakeScorer((0.95,)),
        mode="chat",
        **_recognition_kwargs(artifact, "chat"),
    )

    result = recognizer.recognize(
        RetrievalQuery(
            "Who owns Launch?", "chat", now=200.0, temporal_intent=intent
        ),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-launch"},
        required_entity_ids={"e-launch"},
    )

    assert bool(result.triples) is accepted


def test_recognizer_rejects_required_entity_mismatch(tmp_path: Path, artifact) -> None:
    store = ContextStore((_context("f1", passage_id="p1"),))
    recognizer = TripleRecognizer(
        FactRecallIndex(_snapshot(tmp_path), store, _calibrator()),
        store,
        FakeScorer((0.99,)),
        mode="chat",
        **_recognition_kwargs(artifact, "chat"),
    )

    result = recognizer.recognize(
        RetrievalQuery("Where is Jordan?", "chat", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-jordan"},
        required_entity_ids={"e-jordan"},
    )

    assert result.triples == ()
    assert result.decisions[0].reason == "required_entity_mismatch"


def test_rejected_embedding_head_hits_never_reach_local_recognition(
    tmp_path: Path, artifact
) -> None:
    rejected = replace(
        _context("f3", passage_id="p3"),
        object_entity_id="e-unrelated",
        object="Unrelated",
        logical_key=("e-maya", "owns_or_has", "e-unrelated"),
        recognition_document="Triple: an unrelated rejected candidate",
    )
    store = ContextStore((_context("f1", passage_id="p1"), rejected))
    scorer = FakeScorer((0.95,))
    result = TripleRecognizer(
        FactRecallIndex(_snapshot(tmp_path), store, _calibrator()),
        store,
        scorer,
        mode="chat",
        **_recognition_kwargs(artifact, "chat"),
    ).recognize(
        RetrievalQuery("Who owns Launch?", "chat", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-launch"},
        required_entity_ids=set(),
    )

    assert result.accepted_fact_ids == ("f1",)
    assert "unrelated rejected candidate" not in scorer.calls[0][1][0]
    # Context/status filtering intentionally happens over the full snapshot before
    # calibrated logical-group top-k; the rejected group never reaches the scorer.
    assert store.requested == ("f1", "f2", "f3")


def test_recognizer_accepts_at_most_five_logical_triples(
    tmp_path: Path, artifact
) -> None:
    fact_ids = tuple(f"f{index}" for index in range(6))
    snapshot = replace(
        _snapshot(tmp_path),
        fact_ids=fact_ids,
        fact_matrix=np.array(
            [[1.0, index / 100.0] for index in range(6)], dtype=np.float32
        ),
    )
    contexts = tuple(
        replace(
            _context(fact_id, passage_id=f"p{index}"),
            object_entity_id=f"e-object-{index}",
            object=f"Object {index}",
            logical_key=("e-maya", "owns_or_has", f"e-object-{index}"),
        )
        for index, fact_id in enumerate(fact_ids)
    )
    store = ContextStore(contexts)
    result = TripleRecognizer(
        FactRecallIndex(snapshot, store, _calibrator()),
        store,
        FakeScorer((0.99, 0.98, 0.97, 0.96, 0.95, 0.94)),
        mode="chat",
        **_recognition_kwargs(artifact, "chat"),
    ).recognize(
        RetrievalQuery("What does Maya own?", "chat", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-maya"},
        required_entity_ids={"e-maya"},
    )

    assert len(result.triples) == 5
    assert len(result.accepted_fact_ids) == 5


def test_deadline_failure_uses_strict_embedding_and_endpoint_overlap_only(
    tmp_path: Path, artifact
) -> None:
    store = ContextStore((_context("f1", passage_id="p1"), _context("f3", passage_id="p3")))
    recognizer = TripleRecognizer(
        FactRecallIndex(_snapshot(tmp_path), store, _calibrator("live")),
        store,
        FakeScorer(ModelDeadlineExceeded("late")),
        mode="live",
        **_recognition_kwargs(artifact, "live"),
    )

    accepted = recognizer.recognize(
        RetrievalQuery("Who owns Launch?", "live", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-launch"},
        required_entity_ids=set(),
        deadline_at=monotonic() + 1.0,
    )
    rejected = recognizer.recognize(
        RetrievalQuery("unrelated", "live", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids=set(),
        required_entity_ids=set(),
    )

    assert accepted.degraded
    assert accepted.reason == "local_reranker_deadline_fallback"
    assert [triple.logical_key for triple in accepted.triples] == [
        ("e-maya", "owns_or_has", "e-launch")
    ]
    assert rejected.triples == ()
    assert rejected.degraded


def test_fallback_embedding_floor_is_separate_and_stricter_than_recognition_floor(
    tmp_path: Path, artifact
) -> None:
    snapshot = replace(
        _snapshot(tmp_path),
        fact_ids=("f1",),
        fact_matrix=np.array([[0.75, np.sqrt(1.0 - 0.75**2)]], dtype=np.float32),
    )
    store = ContextStore((_context("f1", passage_id="p1"),))
    result = TripleRecognizer(
        FactRecallIndex(snapshot, store, _calibrator()),
        store,
        FakeScorer(ModelDeadlineExceeded("late")),
        mode="chat",
        **_recognition_kwargs(artifact, "chat"),
    ).recognize(
        RetrievalQuery("Who owns Launch?", "chat", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-launch"},
        required_entity_ids=set(),
    )

    assert artifact.triple_recognition.chat_fallback_embedding_floor > (
        artifact.triple_recognition.chat_floor
    )
    assert result.triples == ()
    assert result.decisions[0].reason == "fallback_below_strict_embedding_floor"


@pytest.mark.parametrize("mode", ["chat", "tool"])
def test_chat_and_tool_use_one_batched_acceptor_for_ambiguity_band(
    tmp_path: Path, artifact, mode: str
) -> None:
    kwargs = _recognition_kwargs(artifact, mode)
    ambiguous_score = (kwargs["ambiguity_low"] + kwargs["recognition_floor"]) / 2
    store = ContextStore(
        (_context("f1", passage_id="p1"), _context("f2", passage_id="p2"))
    )
    acceptor = FakeAmbiguityAcceptor({"f1"})
    result = TripleRecognizer(
        FactRecallIndex(_snapshot(tmp_path), store, _calibrator(mode)),
        store,
        FakeScorer((ambiguous_score,)),
        mode=mode,  # type: ignore[arg-type]
        ambiguity_acceptor=acceptor,
        **kwargs,
    ).recognize(
        RetrievalQuery("Who owns Launch?", mode, now=200.0),  # type: ignore[arg-type]
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-launch"},
        required_entity_ids=set(),
    )

    assert result.accepted_fact_ids == ("f1", "f2")
    assert len(acceptor.calls) == 1
    candidate = acceptor.calls[0][1][0]
    assert getattr(candidate, "fact_ids") == ("f1", "f2")
    assert next(row for row in result.decisions if row.accepted).reason == (
        "accepted_batched_ambiguity_recognition"
    )


def test_ambiguity_acceptor_failure_keeps_local_decisions(
    tmp_path: Path, artifact
) -> None:
    kwargs = _recognition_kwargs(artifact, "chat")
    ambiguous_score = (kwargs["ambiguity_low"] + kwargs["recognition_floor"]) / 2
    store = ContextStore((_context("f1", passage_id="p1"),))
    result = TripleRecognizer(
        FactRecallIndex(_snapshot(tmp_path), store, _calibrator()),
        store,
        FakeScorer((ambiguous_score,)),
        mode="chat",
        ambiguity_acceptor=FakeAmbiguityAcceptor(RuntimeError("unavailable")),
        **kwargs,
    ).recognize(
        RetrievalQuery("Who owns Launch?", "chat", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids={"e-launch"},
        required_entity_ids=set(),
    )

    assert result.triples == ()
    assert result.reason == "no_recognized_triples"
    assert any(row.reason == "below_triple_recognition_floor" for row in result.decisions)


def test_no_fact_embeddings_is_an_explicit_empty_fallback(tmp_path: Path, artifact) -> None:
    snapshot = replace(
        _snapshot(tmp_path),
        fact_ids=(),
        fact_matrix=np.empty((0, 2), dtype=np.float32),
    )
    store = ContextStore(())
    result = TripleRecognizer(
        FactRecallIndex(snapshot, store, _calibrator()),
        store,
        FakeScorer(()),
        mode="chat",
        **_recognition_kwargs(artifact, "chat"),
    ).recognize(
        RetrievalQuery("anything", "chat", now=200.0),
        np.array([1.0, 0.0], dtype=np.float32),
        resolved_entity_ids=set(),
        required_entity_ids=set(),
    )

    assert result.triples == ()
    assert result.entity_seed_weights == {}
    assert result.reason == "no_fact_candidates"
