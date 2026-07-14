from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.base_graph import BaseEdge, materialize_base_graph
from taurine_core.indexer.knowledge_store import BM25PassageHit, KnowledgeStore
from taurine_core.indexer.passages import PassageDraft
from taurine_core.retrieval.calibration import AbstentionPolicy, HeadCalibrator
from taurine_core.retrieval.evidence import (
    SharedRetrievalEngine,
    _assemble_facts,
    _mmr_select,
    _temporal_retrieval_graph,
)
from taurine_core.retrieval import evidence as evidence_module
from taurine_core.retrieval.heads import RawHeadCandidate
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.rerank import LocalPassageReranker
from taurine_core.retrieval.rerank import PassageRerankCandidate, PassageRerankDecision
from taurine_core.retrieval.snapshot import IndexSnapshot
from taurine_core.retrieval.triples import (
    FactContext,
    RecognizedTriple,
    TripleRecognitionDecision,
    TripleRecognitionResult,
)
from taurine_core.retrieval.types import RetrievalQuery, TemporalIntent
from taurine_core.schemas import NoteRecord


_ARTIFACT_PATH = (
    Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
)


@dataclass(frozen=True)
class PassageRow:
    passage_id: str
    note_id: str
    title: str
    source: str
    text: str
    char_start: int
    char_end: int
    created: float
    modified: float


@dataclass(frozen=True)
class NoteRow:
    note_id: str
    title: str
    source: str
    modified: float


@dataclass(frozen=True)
class EntityRow:
    entity_id: str
    canonical: str
    etype: str
    traversal_eligible: bool
    build_id: str = "build-1"


class FakeStore:
    def __init__(
        self,
        passages: Sequence[PassageRow],
        facts: Sequence[FactContext] = (),
        *,
        bm25: Sequence[BM25PassageHit] = (),
        entities: Sequence[EntityRow] = (),
        entity_passages: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._passages = {row.passage_id: row for row in passages}
        self._facts = tuple(facts)
        self._bm25 = tuple(bm25)
        self._entities = {row.entity_id: row for row in entities}
        self._entity_passages = dict(entity_passages or {})
        self.passage_reads: list[tuple[str, ...]] = []

    def entity_contexts(
        self, entity_ids: Sequence[str], *, build_id: str | None = None
    ) -> tuple[EntityRow, ...]:
        del build_id
        return tuple(
            self._entities[entity_id]
            for entity_id in sorted(set(entity_ids))
            if entity_id in self._entities
        )

    def passage_ids_for_entities(
        self, entity_ids: Sequence[str], *, build_id: str | None = None
    ) -> tuple[str, ...]:
        del build_id
        collected = {
            passage_id
            for entity_id in entity_ids
            for passage_id in self._entity_passages.get(entity_id, ())
        }
        return tuple(sorted(collected))

    def search_passages_bm25(self, query: str, k: int) -> tuple[BM25PassageHit, ...]:
        del query
        return self._bm25[:k]

    def passage_contexts(self, passage_ids: Sequence[str]) -> tuple[PassageRow, ...]:
        requested = tuple(passage_ids)
        self.passage_reads.append(requested)
        return tuple(self._passages[row_id] for row_id in requested if row_id in self._passages)

    def note_contexts(self, note_ids: Sequence[str]) -> tuple[NoteRow, ...]:
        wanted = set(note_ids)
        by_note: dict[str, NoteRow] = {}
        for row in self._passages.values():
            if row.note_id in wanted:
                by_note[row.note_id] = NoteRow(row.note_id, row.title, row.source, row.modified)
        return tuple(by_note[note_id] for note_id in sorted(by_note))

    def fact_contexts(self, fact_ids: tuple[str, ...] | None = None) -> tuple[FactContext, ...]:
        if fact_ids is None:
            return self._facts
        wanted = set(fact_ids)
        return tuple(row for row in self._facts if row.fact_id in wanted)


class FakeScorer:
    def __init__(self, scores: Sequence[float] | Exception) -> None:
        self.scores = scores

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> tuple[float, ...]:
        del query, instruction, deadline_at
        if isinstance(self.scores, Exception):
            raise self.scores
        return tuple(self.scores)[: len(documents)]


class ExplodingProjection:
    def __iter__(self):
        raise AssertionError("retrieval must never iterate the note projection")

    def __len__(self) -> int:
        raise AssertionError("retrieval must never inspect the note projection")

    def __bool__(self) -> bool:
        raise AssertionError("retrieval must never inspect the note projection")


class FakeTripleRecognizer:
    def __init__(self, result: TripleRecognitionResult) -> None:
        self.result = result
        self.calls: list[tuple[set[str], set[str]]] = []

    def recognize(
        self,
        query: RetrievalQuery,
        query_vector: np.ndarray,
        resolved_entity_ids: set[str],
        required_entity_ids: set[str],
        deadline_at: float | None = None,
    ) -> TripleRecognitionResult:
        del query, query_vector, deadline_at
        self.calls.append((set(resolved_entity_ids), set(required_entity_ids)))
        return self.result


def _pinned_artifact():
    """Real artifact with FIXED floors/curves for floor-semantics tests.

    The shipping artifact is data-fitted per corpus (self-calibration), so
    tests that assert acceptance/abstention boundaries pin the values their
    fixtures were designed around instead of tracking the fitted data.
    """
    artifact = load_calibration(_ARTIFACT_PATH)
    data = artifact.model_dump(mode="json")
    for mode_name in ("live", "chat", "tool"):
        mode_cfg = data["modes"][mode_name]
        mode_cfg["heads"]["dense"] = {
            **mode_cfg["heads"]["dense"],
            "knots": [[-1.0, 0.0], [0.45, 0.05], [0.62, 0.55], [0.78, 0.95], [1.0, 1.0]],
            "acceptance_floor": 0.7 if mode_name == "live" else 0.62,
        }
        mode_cfg["evidence_floor"] = 0.75 if mode_name == "live" else 0.65
    data["rerank"]["live_floor"] = 0.76
    data["rerank"]["chat_floor"] = 0.66
    from taurine_core.evaluation.graph_retrieval import (
        CalibrationArtifact,
        compute_artifact_hash,
    )

    pinned = CalibrationArtifact.model_validate(data)
    data["hash"] = compute_artifact_hash(pinned)
    return CalibrationArtifact.model_validate(data)


def _calibrator(mode: str) -> tuple[HeadCalibrator, AbstentionPolicy]:
    artifact = _pinned_artifact()
    calibrator = HeadCalibrator(
        artifact,
        mode=mode,  # type: ignore[arg-type]
        artifact_hash=artifact.hash,
        config_hash=artifact.config_hash,
        model_manifest=artifact.model_manifest,
    )
    return calibrator, AbstentionPolicy(artifact, mode=mode)  # type: ignore[arg-type]


def _snapshot(tmp_path: Path, passages: Sequence[PassageRow]) -> IndexSnapshot:
    vectors = {
        "p1": (1.0, 0.0, 0.0),
        "p2": (0.96, 0.1, 0.0),
        "p3": (0.1, 0.98, 0.0),
        "p4": (0.0, 0.1, 0.99),
    }
    passage_ids = tuple(row.passage_id for row in passages)
    matrix = np.asarray([vectors.get(row_id, (0.0, 0.0, 1.0)) for row_id in passage_ids], dtype=np.float32)
    node_ids = passage_ids + ("e-maya", "e-launch")
    node_kinds = ("passage",) * len(passage_ids) + ("entity", "entity")
    edges = (
        BaseEdge("p1", "e-maya", "contains", 0.9),
        BaseEdge("e-maya", "p1", "contains", 0.9),
        BaseEdge("e-maya", "e-launch", "fact", 0.9, "forward", ("f1",)),
        BaseEdge("e-launch", "e-maya", "fact", 0.45, "reverse", ("f1",)),
        BaseEdge("e-launch", "p3", "contains", 0.9),
        BaseEdge("p3", "e-launch", "contains", 0.9),
    )
    return IndexSnapshot(
        build_id="build-1",
        database_path=tmp_path / "knowledge.db",
        calibration_hash="calibration-hash",
        passage_ids=passage_ids,
        passage_matrix=matrix,
        fact_ids=(),
        fact_matrix=np.empty((0, 3), dtype=np.float32),
        alias_index={"maya": ("e-maya",)},
        base_graph=materialize_base_graph(
            build_id="build-1", node_ids=node_ids, node_kinds=node_kinds, edges=edges
        ),
        projection=ExplodingProjection(),  # type: ignore[arg-type]
        quality_ranks={},
    )


def _passages() -> tuple[PassageRow, ...]:
    return (
        PassageRow("p1", "n1", "Maya", "apple_notes", "Maya owns Launch.", 0, 17, 1.0, 4.0),
        PassageRow("p2", "n1", "Maya", "apple_notes", "Launch ships Friday.", 18, 38, 1.0, 4.0),
        PassageRow("p3", "n2", "Launch", "obsidian", "Launch depends on review.", 0, 25, 2.0, 5.0),
        PassageRow("p4", "n3", "Decision", "gmail", "Maya approved the plan.", 0, 23, 3.0, 6.0),
    )


def _fact() -> FactContext:
    return FactContext(
        fact_id="f1",
        subject_entity_id="e-maya",
        subject="Maya",
        predicate="owns_or_has",
        predicate_description="owns or has",
        object_entity_id="e-launch",
        object="Launch",
        source_passage_id="p4",
        source_note_id="n3",
        assertion_text="Maya approved the plan.",
        observed_at=6.0,
        valid_from=None,
        valid_to=None,
        status="active",
        recognition_document="Triple: Maya owns Launch",
        logical_key=("e-maya", "owns_or_has", "e-launch"),
    )


def _recognized() -> TripleRecognitionResult:
    context = _fact()
    triple = RecognizedTriple(
        logical_key=context.logical_key,
        fact_ids=(context.fact_id,),
        source_passage_ids=(context.source_passage_id,),
        source_note_ids=(context.source_note_id,),
        subject_entity_id=context.subject_entity_id,
        subject=context.subject,
        predicate=context.predicate,
        predicate_description=context.predicate_description,
        object_entity_id=context.object_entity_id,
        object=context.object,
        score=0.94,
        contexts=(context,),
    )
    decision = TripleRecognitionDecision(context.logical_key, ("f1",), 0.94, 0.94, True, "accepted")
    return TripleRecognitionResult((triple,), (decision,), {"e-maya": 0.94, "e-launch": 0.94}, False, "recognized_triples")


def _engine(
    tmp_path: Path,
    *,
    mode: str = "chat",
    scorer: Sequence[float] | Exception = (0.96, 0.91, 0.88, 0.86),
    triple_result: TripleRecognitionResult | None = None,
) -> tuple[SharedRetrievalEngine, EvidenceRegistry, FakeStore]:
    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator(mode)
    registry = EvidenceRegistry()
    store = FakeStore(
        rows,
        (_fact(),),
        bm25=(BM25PassageHit("p2", 20.0, 1.0),),
    )
    reranker = LocalPassageReranker(
        FakeScorer(scorer),
        mode=mode,  # type: ignore[arg-type]
        acceptance_floor=0.65 if mode != "live" else 0.75,
    )
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=registry,
        reranker=reranker,
        entity_resolver=lambda surface, etype, query: (
            RawHeadCandidate("entity", "e-launch", 0.99, "resolved"),
        ),
        triple_recognizer=(
            FakeTripleRecognizer(triple_result or _recognized())
            if triple_result is not None or mode == "chat"
            else None
        ),
    )
    return engine, registry, store


def test_shared_engine_assembles_registered_passage_fact_path_and_note_evidence(tmp_path: Path) -> None:
    engine, registry, _store = _engine(tmp_path)

    pack = engine.retrieve(
        RetrievalQuery("What does Maya own?", "chat", now=10.0),
        entity_spans=(("Launch", "project"),),
    )

    assert pack.decision.accepted
    assert {row.passage_id for row in pack.passages} >= {"p1", "p3", "p4"}
    assert len(pack.facts) == 1
    assert pack.facts[0].fact_ids == ("f1",)
    assert pack.facts[0].source_evidence_ids
    fact_record = registry.resolve(pack.facts[0].fact_ref)
    assert fact_record is not None
    assert fact_record.payload["source_evidence_ids"] == list(pack.facts[0].source_evidence_ids)
    assert fact_record.payload["status"] == "active"
    assert fact_record.payload["subject_entity_id"] == "e-maya"
    assert fact_record.payload["object_entity_id"] == "e-launch"
    assert pack.paths and any(path.fact_ids == ("f1",) for path in pack.paths)
    assert {row.note_id for row in pack.notes} == {row.note_id for row in pack.passages}
    assert registry.refs("read") == pack.trace.read_ids
    assert all(ref[:1] in {"N", "E", "F", "P"} for ref in pack.trace.read_ids)
    assert pack.trace.active_build_id == "build-1"
    assert pack.trace.calibration_hash == "calibration-hash"
    assert pack.trace.teleport
    assert pack.trace.ppr["iterations"] > 0
    assert pack.trace.triple_decisions
    assert pack.trace.rerank
    assert set(pack.trace.latency_ms) >= {"query_embedding", "heads", "ppr", "paths", "rerank", "mmr", "pack"}
    assert all(row[0].startswith("p") for row in pack.trace.ppr["top_passages"])
    assert pack.trace.paths


def test_shared_engine_uses_whole_corpus_dense_reset_and_never_reads_projection(tmp_path: Path) -> None:
    engine, _registry, store = _engine(tmp_path)

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert pack.trace.debug["dense_reset_considered"] == len(_passages())
    assert set(pack.trace.teleport) >= {"p1", "p2"}
    assert store.passage_reads


def _rerank_decision(passage_id: str, *, score: float) -> PassageRerankDecision:
    return PassageRerankDecision(
        candidate=PassageRerankCandidate(
            passage_id=passage_id,
            note_id=f"n-{passage_id}",
            title=passage_id,
            source="obsidian",
            text="text",
            recall_score=score,
        ),
        score=score,
        accepted=True,
        rank=1,
        reason="accepted_passage_rerank",
    )


def test_near_duplicate_evidence_suppression_keeps_higher_rerank_score() -> None:
    snapshot = type(
        "Snapshot",
        (),
        {
            "passage_ids": ("lower", "higher", "distinct"),
            "passage_matrix": np.asarray(
                ((1.0, 0.0), (0.9999, 0.01), (0.0, 1.0)), dtype=np.float32
            ),
        },
    )()
    selected = (
        _rerank_decision("lower", score=0.80),
        _rerank_decision("higher", score=0.95),
        _rerank_decision("distinct", score=0.85),
    )

    kept, suppressed = evidence_module._suppress_near_duplicate_passages(
        selected, snapshot
    )

    assert {row.candidate.passage_id for row in kept} == {"higher", "distinct"}
    assert suppressed == 1


def test_near_duplicate_evidence_suppression_fails_open_without_embeddings() -> None:
    snapshot = type(
        "Snapshot",
        (),
        {
            "passage_ids": ("available",),
            "passage_matrix": np.asarray(((1.0, 0.0),), dtype=np.float32),
        },
    )()
    selected = (
        _rerank_decision("available", score=0.95),
        _rerank_decision("missing", score=0.90),
    )

    kept, suppressed = evidence_module._suppress_near_duplicate_passages(
        selected, snapshot
    )

    assert kept == selected
    assert suppressed == 0


def test_final_evidence_pack_suppresses_near_duplicate_passages(tmp_path: Path) -> None:
    engine, _registry, _store = _engine(tmp_path)

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    passage_ids = {row.passage_id for row in pack.passages}
    assert "p1" in passage_ids
    assert "p2" not in passage_ids
    assert pack.trace.debug["near_duplicate_passages_suppressed"] == 1


def test_mmr_enforces_two_passages_per_note_and_mode_budget(tmp_path: Path) -> None:
    rows = tuple(
        PassageRow(f"x{index}", "same" if index < 4 else f"n{index}", f"T{index}", "obsidian", f"text {index}", 0, 6, 1.0, 2.0)
        for index in range(14)
    )
    snapshot = IndexSnapshot(
        build_id="build-1",
        database_path=tmp_path / "knowledge.db",
        calibration_hash="calibration-hash",
        passage_ids=tuple(row.passage_id for row in rows),
        passage_matrix=np.eye(14, dtype=np.float32),
        fact_ids=(),
        fact_matrix=np.empty((0, 14), dtype=np.float32),
        alias_index={},
        base_graph=materialize_base_graph(
            build_id="build-1",
            node_ids=tuple(row.passage_id for row in rows),
            node_kinds=("passage",) * len(rows),
            edges=(),
        ),
        projection=(object(),),
        quality_ranks={},
    )
    calibrator, abstention = _calibrator("chat")
    registry = EvidenceRegistry()
    engine = SharedRetrievalEngine(
        snapshot,
        FakeStore(rows, bm25=tuple(BM25PassageHit(row.passage_id, 20.0, 1.0) for row in rows)),
        calibrator,
        abstention,
        query_embedder=lambda text: np.ones(14, dtype=np.float32),
        registry=registry,
        reranker=LocalPassageReranker(FakeScorer((0.99,) * 14), mode="chat", acceptance_floor=0.65),
    )

    pack = engine.retrieve(RetrievalQuery("all", "chat", now=10.0))

    assert len(pack.passages) == 12
    assert sum(row.note_id == "same" for row in pack.passages) == 2
    same_note = next(row for row in pack.notes if row.note_id == "same")
    assert same_note.score <= max(row.rerank_score for row in pack.passages if row.note_id == "same") + 0.1
    assert pack.trace.mmr_exclusions


def test_degraded_reranker_falls_back_only_to_calibrated_direct_passages(tmp_path: Path) -> None:
    from taurine_core.models_local import LocalModelUnavailable

    engine, _registry, _store = _engine(
        tmp_path,
        scorer=LocalModelUnavailable("missing"),
        triple_result=TripleRecognitionResult((), (), {}, False, "none"),
    )

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert pack.decision.accepted
    assert {row.passage_id for row in pack.passages} <= {"p1", "p2", "p3"}
    assert pack.trace.debug["rerank_degraded"] is True
    assert pack.trace.debug["rerank_fallback"] == "calibrated_direct_only"


def test_post_rerank_abstention_returns_an_explicit_empty_pack(tmp_path: Path) -> None:
    engine, registry, _store = _engine(
        tmp_path,
        scorer=(0.2, 0.2, 0.2, 0.2),
        triple_result=TripleRecognitionResult((), (), {}, False, "none"),
    )

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert not pack.decision.accepted
    assert pack.decision.reason == "no_evidence_above_floor"
    assert pack.passages == pack.facts == pack.paths == pack.notes == ()
    assert pack.trace.selected_ids == ()
    assert registry.refs("read") == ()
    assert pack.trace.rerank


def test_trace_read_and_cited_ids_are_current_pack_only(tmp_path: Path) -> None:
    engine, registry, _store = _engine(tmp_path)
    old = registry.register("passage", "old-passage", {"passage_id": "old-passage"})
    registry.mark_cited(old)

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert old not in pack.trace.read_ids
    assert old not in pack.trace.cited_ids
    assert set(pack.trace.read_ids) == set(pack.trace.selected_ids)


def test_mmr_applies_bounded_provider_diversity_penalty(tmp_path: Path) -> None:
    rows = (
        PassageRerankDecision(
            PassageRerankCandidate("p1", "n1", "One", "gmail", "one", 0.9),
            0.90,
            True,
            1,
            "accepted",
        ),
        PassageRerankDecision(
            PassageRerankCandidate("p2", "n2", "Two", "gmail", "two", 0.9),
            0.89,
            True,
            2,
            "accepted",
        ),
        PassageRerankDecision(
            PassageRerankCandidate("p3", "n3", "Three", "obsidian", "three", 0.9),
            0.87,
            True,
            3,
            "accepted",
        ),
    )
    snapshot = IndexSnapshot(
        "build-1",
        tmp_path / "knowledge.db",
        "calibration",
        ("p1", "p2", "p3"),
        np.eye(3, dtype=np.float32),
        (),
        np.empty((0, 3), dtype=np.float32),
        {},
        materialize_base_graph(
            build_id="build-1",
            node_ids=("p1", "p2", "p3"),
            node_kinds=("passage", "passage", "passage"),
            edges=(),
        ),
        (),
        {},
    )

    selected, _excluded = _mmr_select(
        rows,
        snapshot,
        source_diversity_penalty=0.05,
        limit=2,
    )

    assert [row.candidate.passage_id for row in selected] == ["p1", "p3"]


def test_fact_assembly_keeps_temporal_variants_and_nonfabricated_path_confidence() -> None:
    registry = EvidenceRegistry()
    e1 = registry.register("passage", "p1", {"passage_id": "p1"})
    e2 = registry.register("passage", "p2", {"passage_id": "p2"})
    active = replace(_fact(), fact_id="fa", source_passage_id="p1")
    historical = replace(
        _fact(),
        fact_id="fh",
        source_passage_id="p2",
        status="superseded",
        valid_from=1.0,
        valid_to=5.0,
    )
    path = type(
        "Path",
        (),
        {"fact_ids": ("fa", "fh")},
    )()

    facts = _assemble_facts(
        registry,
        TripleRecognitionResult((), (), {}, False, "none"),
        (active, historical),
        {"p1": e1, "p2": e2},
        (path,),
        build_id="build-1",
        supplied_passage_ids={"p1", "p2"},
    )

    assert [(row.status, row.valid_from, row.valid_to) for row in facts] == [
        ("active", None, None),
        ("superseded", 1.0, 5.0),
    ]
    assert all(row.confidence == 0.0 for row in facts)


def test_historical_query_uses_temporal_fact_overlay_not_current_base_edges(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, _passages())
    historical = replace(
        _fact(),
        status="superseded",
        valid_from=1.0,
        valid_to=5.0,
    )

    current_graph, _ = _temporal_retrieval_graph(
        snapshot,
        (historical,),
        RetrievalQuery("current", "chat", now=10.0, temporal_intent=TemporalIntent("current")),
    )
    historical_graph, selected = _temporal_retrieval_graph(
        snapshot,
        (historical,),
        RetrievalQuery("before", "chat", now=10.0, temporal_intent=TemporalIntent("historical")),
    )

    assert not any(edge.kind == "fact" for edge in current_graph.edges)
    assert selected == {"f1"}
    assert any(edge.kind == "fact" and edge.fact_ids == ("f1",) for edge in historical_graph.edges)


def test_fact_registry_payload_is_stable_across_turn_dependent_support() -> None:
    registry = EvidenceRegistry()
    e1 = registry.register("passage", "p1", {"passage_id": "p1"})
    e2 = registry.register("passage", "p2", {"passage_id": "p2"})
    contexts = (
        replace(_fact(), fact_id="f1", source_passage_id="p1"),
        replace(_fact(), fact_id="f2", source_passage_id="p2"),
    )
    path = type("Path", (), {"fact_ids": ("f1", "f2")})()

    first = _assemble_facts(
        registry,
        TripleRecognitionResult((), (), {}, False, "none"),
        contexts,
        {"p1": e1, "p2": e2},
        (path,),
        build_id="build-1",
        supplied_passage_ids={"p1"},
    )
    second = _assemble_facts(
        registry,
        TripleRecognitionResult((), (), {}, False, "none"),
        contexts,
        {"p1": e1, "p2": e2},
        (path,),
        build_id="build-1",
        supplied_passage_ids={"p2"},
    )

    assert first[0].fact_ref == second[0].fact_ref
    payload = registry.resolve(first[0].fact_ref)
    assert payload is not None
    assert payload.payload["source_evidence_ids"] == [e1, e2]


def test_path_fact_source_passages_enter_rerank_union(tmp_path: Path) -> None:
    engine, _registry, _store = _engine(
        tmp_path,
        triple_result=TripleRecognitionResult((), (), {}, False, "none"),
    )

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert "p4" in {str(row["passage_id"]) for row in pack.trace.rerank}


def test_recognized_triple_endpoints_are_explanatory_path_seeds(tmp_path: Path) -> None:
    engine, _registry, _store = _engine(tmp_path)
    graph = materialize_base_graph(
        build_id="build-1",
        node_ids=("p1", "p2", "p3", "p4", "e-maya", "e-launch"),
        node_kinds=("passage", "passage", "passage", "passage", "entity", "entity"),
        edges=(
            BaseEdge("e-launch", "p3", "contains", 0.9),
            BaseEdge("p3", "e-launch", "contains", 0.9),
        ),
    )
    engine.snapshot = replace(engine.snapshot, base_graph=graph, alias_index={})

    pack = engine.retrieve(RetrievalQuery("ownership", "chat", now=10.0))

    assert any(path.node_ids[0] == "e-launch" for path in pack.paths)


def test_dense_rejected_trace_is_bounded_while_reset_considers_whole_corpus(tmp_path: Path) -> None:
    rows = tuple(
        PassageRow(f"p{index}", f"n{index}", f"T{index}", "obsidian", f"text {index}", 0, 6, 1.0, 2.0)
        for index in range(75)
    )
    matrix = np.asarray(
        [(1.0, 0.0) if index == 0 else (0.0, 1.0) for index in range(75)],
        dtype=np.float32,
    )
    snapshot = IndexSnapshot(
        "build-1",
        tmp_path / "knowledge.db",
        "calibration",
        tuple(row.passage_id for row in rows),
        matrix,
        (),
        np.empty((0, 2), dtype=np.float32),
        {},
        materialize_base_graph(
            build_id="build-1",
            node_ids=tuple(row.passage_id for row in rows),
            node_kinds=("passage",) * len(rows),
            edges=(),
        ),
        ExplodingProjection(),  # type: ignore[arg-type]
        {},
    )
    calibrator, abstention = _calibrator("chat")
    engine = SharedRetrievalEngine(
        snapshot,
        FakeStore(rows, bm25=(BM25PassageHit("p0", 20.0, 1.0),)),
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0], dtype=np.float32),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(FakeScorer((0.99,)), mode="chat", acceptance_floor=0.65),
    )

    pack = engine.retrieve(RetrievalQuery("query", "chat", now=10.0))

    dense_rejected = [row for row in pack.trace.rejected_hits if row.head == "dense"]
    assert len(dense_rejected) <= 60
    assert pack.trace.debug["dense_reset_considered"] == 75


def test_knowledge_store_exposes_typed_passage_and_note_contexts(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        note = NoteRecord(
            note_id="n1",
            title="Roadmap",
            source="obsidian",
            text="Alpha beta.",
            content_hash="hash",
            created=1.0,
            modified=2.0,
        )
        store.write_note(note, build_id="build-1", extraction_status="complete")
        store.write_passages(
            "n1",
            (PassageDraft("p1", 0, 0, 11, 3, "Alpha beta.", "phash"),),
            created=1.0,
            modified=2.0,
            build_id="build-1",
        )

        passages = store.passage_contexts(("p1",))
        notes = store.note_contexts(("n1",))

        assert passages[0].text == "Alpha beta."
        assert passages[0].title == "Roadmap"
        assert passages[0].char_start == 0
        assert notes[0].source == "obsidian"
        assert store.passage_contexts(("missing",)) == ()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# H-B2 — per-head failure isolation (§17.3)
# ---------------------------------------------------------------------------


class ExplodingBM25Store(FakeStore):
    def search_passages_bm25(self, query: str, k: int):
        raise RuntimeError("fts index corrupted")


def test_failing_bm25_head_degrades_to_dense_only_and_names_the_dead_head(
    tmp_path: Path,
) -> None:
    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator("chat")
    registry = EvidenceRegistry()
    store = ExplodingBM25Store(rows, (_fact(),))
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=registry,
        reranker=LocalPassageReranker(
            FakeScorer((0.96, 0.91, 0.88, 0.86)), mode="chat", acceptance_floor=0.65
        ),
    )

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert pack.decision.accepted
    assert "bm25" in pack.trace.debug["unavailable_heads"]
    assert "RuntimeError" in pack.trace.debug["unavailable_heads"]["bm25"]
    assert any(row.passage_id == "p1" for row in pack.passages)


def test_degraded_head_set_requires_strict_floor_for_acceptance(
    tmp_path: Path,
) -> None:
    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator("chat")
    store = ExplodingBM25Store(rows)
    # Dense cosine ≈ 0.625 calibrates to ≈ 0.65: above the normal 0.62 floor
    # but below the degraded strict floor of 1.1 × 0.62 = 0.682.
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array(
            [0.625, -0.78, 0.0], dtype=np.float32
        ),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            FakeScorer((0.9,) * 4), mode="chat", acceptance_floor=0.65
        ),
    )

    # No alias appears in the query, so only the dense head can qualify.
    pack = engine.retrieve(RetrievalQuery("Who owns the launch?", "chat", now=10.0))

    assert not pack.decision.accepted
    assert pack.decision.reason == "no_head_cleared_degraded_strict_floor"


def test_totally_failed_head_set_abstains_with_explicit_reason(tmp_path: Path) -> None:
    class ExplodingAliasIndex(dict):
        def items(self):
            raise RuntimeError("alias index corrupted")

    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    object.__setattr__(snapshot, "alias_index", ExplodingAliasIndex())
    calibrator, abstention = _calibrator("chat")
    engine = SharedRetrievalEngine(
        snapshot,
        ExplodingBM25Store(rows),
        calibrator,
        abstention,
        # Wrong dimensionality kills the dense head inside its own search.
        query_embedder=lambda text: np.array([1.0, 0.0], dtype=np.float32),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            FakeScorer((0.9,) * 4), mode="chat", acceptance_floor=0.65
        ),
    )

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert not pack.decision.accepted
    assert pack.decision.reason == "all_heads_unavailable"
    assert set(pack.trace.debug["unavailable_heads"]) == {"dense", "bm25", "exact"}


# ---------------------------------------------------------------------------
# H6 — tiered rerank-cap ordering
# ---------------------------------------------------------------------------


def test_direct_dense_hits_are_never_evicted_from_the_rerank_cap_by_expansions() -> None:
    from taurine_core.retrieval.evidence import _order_rerank_candidates
    from taurine_core.retrieval.ppr import PPRResult
    from taurine_core.retrieval.types import HeadHit

    direct_ids = tuple(f"d{index:02d}" for index in range(40))
    expansion_ids = tuple(f"x{index:02d}" for index in range(20))
    passage_by_id = {
        passage_id: PassageRow(
            passage_id, f"n-{passage_id}", passage_id, "obsidian", "text", 0, 4, 1.0, 2.0
        )
        for passage_id in (*direct_ids, *expansion_ids)
    }
    accepted_hits = tuple(
        HeadHit("passage", passage_id, 0.9, 0.9 - index * 0.001, index + 1, "dense", "ok")
        for index, passage_id in enumerate(direct_ids)
    )
    ppr = PPRResult(
        accepted=True,
        reason="converged",
        build_id="build-1",
        ranks={passage_id: 0.9 for passage_id in expansion_ids},
        teleport={},
        passage_mass=1.0,
        entity_mass=0.0,
        iterations=1,
        residual=0.0,
        contributions={},
    )
    path_support = {passage_id: 0.95 for passage_id in expansion_ids}

    candidates = _order_rerank_candidates(
        set(passage_by_id),
        passage_by_id,
        accepted_hits,
        {},
        ppr,
        {},
        path_support,
        {},
        cap=30,
    )

    chosen = [row.passage_id for row in candidates]
    assert len(chosen) == 30
    assert all(passage_id.startswith("d") for passage_id in chosen)
    assert chosen == sorted(chosen, key=lambda pid: direct_ids.index(pid))
    assert all(
        row.recall_score == pytest.approx(0.9 - direct_ids.index(row.passage_id) * 0.001)
        for row in candidates
    )


def test_rerank_cap_orders_triple_support_before_expansions() -> None:
    from taurine_core.retrieval.evidence import _order_rerank_candidates
    from taurine_core.retrieval.ppr import PPRResult
    from taurine_core.retrieval.types import HeadHit

    passage_by_id = {
        passage_id: PassageRow(
            passage_id, f"n-{passage_id}", passage_id, "obsidian", "text", 0, 4, 1.0, 2.0
        )
        for passage_id in ("direct", "support", "expanded")
    }
    ppr = PPRResult(
        accepted=True,
        reason="converged",
        build_id="build-1",
        ranks={"expanded": 0.99, "support": 0.98},
        teleport={},
        passage_mass=1.0,
        entity_mass=0.0,
        iterations=1,
        residual=0.0,
        contributions={},
    )

    candidates = _order_rerank_candidates(
        {"direct", "support", "expanded"},
        passage_by_id,
        (HeadHit("passage", "direct", 0.7, 0.66, 1, "dense", "ok"),),
        {},
        ppr,
        {"support": 0.8},
        {},
        {},
        cap=2,
    )

    assert [row.passage_id for row in candidates] == ["direct", "support"]
    assert candidates[0].recall_score == pytest.approx(0.66)
    assert candidates[1].triple_support == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# M2 — temporal coherence and conflicted-fact presentation
# ---------------------------------------------------------------------------


def test_recognized_historical_triple_can_render_a_path(tmp_path: Path) -> None:
    historical = replace(_fact(), status="superseded", valid_from=1.0, valid_to=5.0)
    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator("chat")
    registry = EvidenceRegistry()
    store = FakeStore(rows, (historical,), bm25=(BM25PassageHit("p2", 20.0, 1.0),))
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=registry,
        reranker=LocalPassageReranker(
            FakeScorer((0.96, 0.91, 0.88, 0.86)), mode="chat", acceptance_floor=0.65
        ),
        entity_resolver=lambda surface, etype, query: (
            RawHeadCandidate("entity", "e-launch", 0.99, "resolved"),
        ),
        triple_recognizer=FakeTripleRecognizer(_recognized()),
    )

    pack = engine.retrieve(
        RetrievalQuery(
            "Who owned the launch before?",
            "chat",
            now=10.0,
            temporal_intent=TemporalIntent("historical"),
        ),
        entity_spans=(("Launch", "project"),),
    )

    assert pack.decision.accepted
    assert any(path.fact_ids == ("f1",) for path in pack.paths)
    assert any(row.status == "superseded" for row in pack.facts)


def test_conflicted_fact_is_presented_but_never_traversed_when_both_endpoints_seed(
    tmp_path: Path,
) -> None:
    conflicted = replace(_fact(), status="conflicted", source_passage_id="p1")
    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator("chat")
    registry = EvidenceRegistry()
    store = FakeStore(rows, (conflicted,), bm25=(BM25PassageHit("p2", 20.0, 1.0),))
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=registry,
        reranker=LocalPassageReranker(
            FakeScorer((0.96, 0.91, 0.88, 0.86)), mode="chat", acceptance_floor=0.65
        ),
        entity_resolver=lambda surface, etype, query: (
            RawHeadCandidate("entity", "e-launch", 0.99, "resolved"),
        ),
    )

    # "Maya" resolves via the exact head; "Launch" via the entity span:
    # both endpoints of the conflicted pair are query-seeded.
    pack = engine.retrieve(
        RetrievalQuery("What do Maya and Launch disagree on?", "chat", now=10.0),
        entity_spans=(("Launch", "project"),),
    )

    assert pack.decision.accepted
    conflicted_facts = [row for row in pack.facts if row.status == "conflicted"]
    assert conflicted_facts and conflicted_facts[0].fact_ids == ("f1",)
    assert all("f1" not in path.fact_ids for path in pack.paths)

    unseeded = SharedRetrievalEngine(
        snapshot,
        FakeStore(rows, (conflicted,), bm25=(BM25PassageHit("p2", 20.0, 1.0),)),
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            FakeScorer((0.96, 0.91, 0.88, 0.86)), mode="chat", acceptance_floor=0.65
        ),
    ).retrieve(RetrievalQuery("What does Maya own?", "chat", now=10.0))
    assert all(row.status != "conflicted" for row in unseeded.facts)


def test_temporal_admission_downweights_history_under_any_intent(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, _passages())
    active = _fact()
    superseded = replace(_fact(), fact_id="f2", status="superseded")

    any_graph, selected = _temporal_retrieval_graph(
        snapshot,
        (active, superseded),
        RetrievalQuery("any", "chat", now=10.0, temporal_intent=TemporalIntent("any")),
    )

    assert selected == {"f1", "f2"}
    fact_edges = [edge for edge in any_graph.edges if edge.kind == "fact"]
    assert fact_edges


# ---------------------------------------------------------------------------
# M7 — entity reset must not double-count into passage reset
# ---------------------------------------------------------------------------


def test_resolved_entity_mass_enters_teleport_only_via_entity_reset(
    tmp_path: Path,
) -> None:
    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator("chat")
    # p4 mentions e-maya but has no dense/bm25 hit of its own: before the M7
    # fix it inherited the strongest entity weight straight into the teleport.
    store = FakeStore(
        rows,
        bm25=(BM25PassageHit("p2", 20.0, 1.0),),
        entity_passages={"e-maya": ("p4",)},
    )
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            FakeScorer((0.96, 0.91, 0.88, 0.86)), mode="chat", acceptance_floor=0.65
        ),
    )

    pack = engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert pack.decision.accepted
    assert "p4" not in pack.trace.teleport
    assert "e-maya" in pack.trace.teleport


# ---------------------------------------------------------------------------
# M13 (query half) — explicit hub seeding temporarily enables traversal
# ---------------------------------------------------------------------------


def _hub_snapshot(tmp_path: Path, rows: Sequence[PassageRow]) -> IndexSnapshot:
    passage_ids = tuple(row.passage_id for row in rows)
    vectors = {"p1": (1.0, 0.0, 0.0)}
    matrix = np.asarray(
        [vectors.get(row_id, (0.0, 1.0, 0.0)) for row_id in passage_ids],
        dtype=np.float32,
    )
    node_ids = passage_ids + ("e-date",)
    node_kinds = ("passage",) * len(passage_ids) + ("entity",)
    return IndexSnapshot(
        build_id="build-1",
        database_path=tmp_path / "knowledge.db",
        calibration_hash="calibration-hash",
        passage_ids=passage_ids,
        passage_matrix=matrix,
        fact_ids=(),
        fact_matrix=np.empty((0, 3), dtype=np.float32),
        alias_index={},
        base_graph=materialize_base_graph(
            build_id="build-1", node_ids=node_ids, node_kinds=node_kinds, edges=()
        ),
        projection=(),
        quality_ranks={},
    )


def _hub_engine(tmp_path: Path) -> SharedRetrievalEngine:
    rows = (
        PassageRow("p1", "n1", "Plan", "obsidian", "The plan is set.", 0, 16, 1.0, 2.0),
        PassageRow("p2", "n2", "Dates", "obsidian", "Due July 18.", 0, 12, 1.0, 2.0),
    )
    snapshot = _hub_snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator("chat")
    store = FakeStore(
        rows,
        entities=(EntityRow("e-date", "July 18", "date_or_deadline", False),),
        entity_passages={"e-date": ("p2",)},
    )
    return SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            FakeScorer((0.96, 0.91)), mode="chat", acceptance_floor=0.65
        ),
        entity_resolver=lambda surface, etype, query: (
            RawHeadCandidate("entity", "e-date", 1.0, "july 18"),
        ),
    )


def test_explicitly_seeded_hub_entity_becomes_traversable(tmp_path: Path) -> None:
    engine = _hub_engine(tmp_path)

    pack = engine.retrieve(
        RetrievalQuery("What is due on July 18?", "chat", now=10.0),
        entity_spans=(("July 18", "date_or_deadline"),),
    )

    assert pack.decision.accepted
    assert any(row.passage_id == "p2" for row in pack.passages)
    assert "e-date" in pack.trace.teleport


def test_unseeded_hub_entity_stays_dangling(tmp_path: Path) -> None:
    engine = _hub_engine(tmp_path)

    pack = engine.retrieve(RetrievalQuery("What is the plan?", "chat", now=10.0))

    assert all(row.passage_id != "p2" for row in pack.passages)
    assert "e-date" not in pack.trace.teleport


# ---------------------------------------------------------------------------
# M15 — hub path penalty is wired into the engine's path finder
# ---------------------------------------------------------------------------


def test_engine_passes_snapshot_hub_entities_to_the_path_finder(
    tmp_path: Path, monkeypatch
) -> None:
    import taurine_core.retrieval.evidence as evidence_module

    captured: dict[str, frozenset[str]] = {}
    real_finder = evidence_module.EvidencePathFinder

    def capturing_finder(graph, *, fact_status, hub_entity_ids=frozenset(), note_by_passage=None):
        captured["hub_entity_ids"] = frozenset(hub_entity_ids)
        return real_finder(
            graph,
            fact_status=fact_status,
            hub_entity_ids=hub_entity_ids,
            note_by_passage=note_by_passage,
        )

    monkeypatch.setattr(evidence_module, "EvidencePathFinder", capturing_finder)
    rows = _passages()
    snapshot = _snapshot(tmp_path, rows)
    calibrator, abstention = _calibrator("chat")
    store = FakeStore(
        rows,
        bm25=(BM25PassageHit("p2", 20.0, 1.0),),
        entities=(
            EntityRow("e-maya", "Maya", "person", True),
            EntityRow("e-launch", "Launch", "concept", False),
        ),
    )
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        abstention,
        query_embedder=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            FakeScorer((0.96, 0.91, 0.88, 0.86)), mode="chat", acceptance_floor=0.65
        ),
    )

    engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))

    assert captured["hub_entity_ids"] == {"e-launch"}


# ---------------------------------------------------------------------------
# §0.1 — chat/tool fallback lane
# ---------------------------------------------------------------------------


def _fallback_rows() -> tuple[PassageRow, ...]:
    return tuple(
        PassageRow(
            f"f{index}", f"note-{index}", f"T{index}", "obsidian",
            f"low confidence text {index}", 0, 20, 1.0, 2.0,
        )
        for index in range(5)
    ) + (
        PassageRow(
            "f5", "note-0", "T0b", "obsidian", "same note extra", 0, 15, 1.0, 2.0
        ),
    )


def _fallback_engine(
    tmp_path: Path, mode: str, *, fallback_on_abstain: bool, cosine: float = 0.60
) -> tuple[SharedRetrievalEngine, EvidenceRegistry]:
    rows = _fallback_rows()
    passage_ids = tuple(row.passage_id for row in rows)
    # Every passage shares the query direction with the requested cosine, so
    # under the PINNED chat curve each calibrates below the 0.62 acceptance
    # floor but (for 0.60 -> ~0.49) above the 0.75 × floor fallback floor.
    matrix = np.zeros((len(rows), 8), dtype=np.float32)
    for index in range(len(rows)):
        matrix[index, 0] = cosine
        matrix[index, index + 1] = float(np.sqrt(1.0 - cosine * cosine))
    snapshot = IndexSnapshot(
        build_id="build-1",
        database_path=tmp_path / "knowledge.db",
        calibration_hash="calibration-hash",
        passage_ids=passage_ids,
        passage_matrix=matrix,
        fact_ids=(),
        fact_matrix=np.empty((0, 8), dtype=np.float32),
        alias_index={},
        base_graph=materialize_base_graph(
            build_id="build-1",
            node_ids=passage_ids,
            node_kinds=("passage",) * len(rows),
            edges=(),
        ),
        projection=(),
        quality_ranks={},
    )
    calibrator, abstention = _calibrator(mode)
    registry = EvidenceRegistry()
    engine = SharedRetrievalEngine(
        snapshot,
        FakeStore(rows),
        calibrator,
        abstention,
        query_embedder=lambda text: np.array(
            [1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
        ),
        registry=registry,
        reranker=LocalPassageReranker(
            FakeScorer((0.9,) * len(rows)),
            mode=mode,  # type: ignore[arg-type]
            acceptance_floor=0.65,
        ),
        fallback_on_abstain=fallback_on_abstain,
    )
    return engine, registry


def test_chat_abstention_with_near_floor_candidates_yields_fallback_pack(
    tmp_path: Path,
) -> None:
    engine, registry = _fallback_engine(tmp_path, "chat", fallback_on_abstain=True)

    pack = engine.retrieve(RetrievalQuery("vaguely related", "chat", now=10.0))

    assert not pack.decision.accepted
    assert pack.decision.fallback is True
    assert pack.decision.reason == "fallback_low_confidence"
    assert len(pack.passages) == 4
    assert pack.facts == () and pack.paths == ()
    note_ids = [row.note_id for row in pack.passages]
    assert len(set(note_ids)) == len(note_ids)
    assert 0.0 < pack.decision.confidence < 0.62
    # Fallback passages are registered and marked read like normal evidence.
    for row in pack.passages:
        record = registry.resolve(row.evidence_id)
        assert record is not None and "read" in record.states
    assert set(pack.trace.read_ids) >= {row.evidence_id for row in pack.passages}


def test_live_mode_same_inputs_returns_strict_empty_pack(tmp_path: Path) -> None:
    engine, registry = _fallback_engine(tmp_path, "live", fallback_on_abstain=False)

    pack = engine.retrieve(RetrievalQuery("vaguely related", "live", now=10.0))

    assert not pack.decision.accepted
    assert pack.decision.fallback is False
    assert pack.passages == ()
    assert registry.refs("read") == ()


def test_nothing_above_fallback_floor_returns_normal_empty_pack(
    tmp_path: Path,
) -> None:
    engine, registry = _fallback_engine(
        tmp_path, "chat", fallback_on_abstain=True, cosine=0.30
    )

    pack = engine.retrieve(RetrievalQuery("vaguely related", "chat", now=10.0))

    assert not pack.decision.accepted
    assert pack.decision.fallback is False
    assert pack.passages == ()
    assert registry.refs("read") == ()


# ---------------------------------------------------------------------------
# L11 — triple head hit hygiene
# ---------------------------------------------------------------------------


def test_triple_head_hits_keep_only_retained_triples_with_clean_ranks() -> None:
    from taurine_core.retrieval.evidence import _triple_head_hits

    def _decision(key_suffix: str, embedding: float, recognition: float, accepted: bool, reason: str):
        return TripleRecognitionDecision(
            ("e-a", "owns_or_has", f"e-{key_suffix}"),
            (f"fact-{key_suffix}",),
            embedding,
            recognition,
            accepted,
            reason,
        )

    def _triple(key_suffix: str, score: float) -> RecognizedTriple:
        context = replace(
            _fact(),
            fact_id=f"fact-{key_suffix}",
            object_entity_id=f"e-{key_suffix}",
            logical_key=("e-a", "owns_or_has", f"e-{key_suffix}"),
        )
        return RecognizedTriple(
            logical_key=context.logical_key,
            fact_ids=(context.fact_id,),
            source_passage_ids=("p1",),
            source_note_ids=("n1",),
            subject_entity_id="e-a",
            subject="A",
            predicate="owns_or_has",
            predicate_description="owns or has",
            object_entity_id=context.object_entity_id,
            object=key_suffix,
            score=score,
            contexts=(context,),
        )

    result = TripleRecognitionResult(
        triples=(_triple("kept1", 0.95), _triple("kept2", 0.90)),
        decisions=(
            # Preliminary rejection: must not offset ranks.
            _decision("prelim", 0.88, 0.0, False, "required_entity_mismatch"),
            _decision("kept1", 0.81, 0.95, True, "accepted_local_triple_recognition"),
            _decision("kept2", 0.85, 0.90, True, "accepted_local_triple_recognition"),
            # Accepted beyond the seed cap: not retained, so not an accepted hit.
            _decision("overflow", 0.80, 0.89, True, "accepted_local_triple_recognition"),
            _decision("scoredout", 0.79, 0.30, False, "below_triple_recognition_floor"),
        ),
        entity_seed_weights={},
        degraded=False,
        reason="recognized_triples",
    )

    hits = _triple_head_hits(result)

    assert [(hit.target_id, hit.rank) for hit in hits.accepted] == [
        ("fact-kept1", 1),
        ("fact-kept2", 2),
    ]
    # Recognition score is the calibrated score; embedding calibrated is raw.
    assert hits.accepted[0].calibrated_score == pytest.approx(0.95)
    assert hits.accepted[0].raw_score == pytest.approx(0.81)
    rejected_ids = [hit.target_id for hit in hits.rejected]
    assert set(rejected_ids) == {"fact-prelim", "fact-overflow", "fact-scoredout"}
    assert min(hit.rank for hit in hits.rejected) == 3


# ---------------------------------------------------------------------------
# L12 — MMR tie-break prefix inversion
# ---------------------------------------------------------------------------


def test_reverse_key_makes_smallest_id_win_for_prefix_pairs() -> None:
    from taurine_core.retrieval.evidence import _reverse_key

    assert max(["ab", "a"], key=_reverse_key) == "a"
    assert max(["b", "ab"], key=_reverse_key) == "ab"
    assert max(["p10", "p1"], key=_reverse_key) == "p1"


# ---------------------------------------------------------------------------
# L29 — per-query temporal graph caching
# ---------------------------------------------------------------------------


def test_temporal_graph_is_cached_per_intent_and_activation(
    tmp_path: Path, monkeypatch
) -> None:
    import taurine_core.retrieval.evidence as evidence_module

    calls: list[tuple[str, ...]] = []
    real = evidence_module._temporal_retrieval_graph

    def counting(snapshot, fact_rows, query, *, hub_activation=None):
        intent = query.temporal_intent
        calls.append((intent.kind if intent else "current",))
        return real(snapshot, fact_rows, query, hub_activation=hub_activation)

    monkeypatch.setattr(evidence_module, "_temporal_retrieval_graph", counting)
    engine, _registry, _store = _engine(tmp_path)

    engine.retrieve(RetrievalQuery("Maya", "chat", now=10.0))
    engine.retrieve(RetrievalQuery("Maya owns what", "chat", now=10.0))
    assert len(calls) == 1

    engine.retrieve(
        RetrievalQuery(
            "Maya then", "chat", now=10.0, temporal_intent=TemporalIntent("any")
        )
    )
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# C2 prep — shared payload helpers must match the engine's registered shapes
# ---------------------------------------------------------------------------


def test_payload_helpers_match_engine_registered_shapes(tmp_path: Path) -> None:
    from taurine_core.retrieval.evidence_payloads import (
        fact_group_payload,
        fact_source_identity,
        passage_payload,
    )

    engine, registry, _store = _engine(tmp_path)
    pack = engine.retrieve(
        RetrievalQuery("What does Maya own?", "chat", now=10.0),
        entity_spans=(("Launch", "project"),),
    )

    passage = pack.passages[0]
    record = registry.resolve(passage.evidence_id)
    assert record is not None
    assert dict(record.payload) == passage_payload(passage)
    assert set(record.payload) == {
        "passage_id", "note_id", "text", "char_start", "char_end",
    }

    fact = pack.facts[0]
    fact_record = registry.resolve(fact.fact_ref)
    assert fact_record is not None
    context = _fact()
    identity = fact_source_identity(
        "build-1", context.logical_key, "active", None, None
    )
    assert registry.ref_for("fact", identity) == fact.fact_ref
    core = fact_group_payload(context.logical_key, "active", None, None, (context,))
    for key, value in core.items():
        assert fact_record.payload[key] == value


def test_neighborhood_fact_ids_pick_top_anchored_facts_and_skip_unactivated_hubs() -> None:
    from types import SimpleNamespace

    from taurine_core.retrieval.evidence import _neighborhood_fact_ids

    def fact(fact_id, subject, object_, confidence):
        return SimpleNamespace(
            fact_id=fact_id,
            subject_entity_id=subject,
            object_entity_id=object_,
            confidence=confidence,
        )

    rows = [
        fact("f-uses-1", "e-project", "e-docker", 0.9),
        fact("f-uses-2", "e-other", "e-docker", 0.8),
        fact("f-low", "e-docker", "e-thing", 0.4),
        fact("f-owner", "e-owner", "e-org", 0.99),  # hub, not activated
        fact("f-inactive", "e-docker", "e-old", 0.95),  # not traversable
    ]
    picked = _neighborhood_fact_ids(
        rows,
        resolved_entity_ids={"e-docker", "e-owner"},
        traversable_fact_ids={"f-uses-1", "f-uses-2", "f-low", "f-owner"},
        hub_ids=frozenset({"e-owner"}),
        activated_hubs=frozenset(),
        cap=2,
    )
    assert picked == {"f-uses-1", "f-uses-2"}
    # activating the hub admits its facts
    picked_hub = _neighborhood_fact_ids(
        rows,
        resolved_entity_ids={"e-owner"},
        traversable_fact_ids={"f-owner"},
        hub_ids=frozenset({"e-owner"}),
        activated_hubs=frozenset({"e-owner"}),
        cap=3,
    )
    assert picked_hub == {"f-owner"}


def test_candidate_hint_names_the_anchoring_query_entity() -> None:
    from taurine_core.retrieval.evidence import _candidate_hint

    hint = _candidate_hint(
        "p1", {}, {}, {}, {}, entity_anchor={"p1": ("Docker", "Git")}
    )
    assert "mentions the query entity: Docker, Git" in hint
    assert _candidate_hint("p2", {}, {}, {}, {}, entity_anchor={"p1": ("Docker",)}) == ""
