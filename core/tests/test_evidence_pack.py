from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

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


class FakeStore:
    def __init__(
        self,
        passages: Sequence[PassageRow],
        facts: Sequence[FactContext] = (),
        *,
        bm25: Sequence[BM25PassageHit] = (),
    ) -> None:
        self._passages = {row.passage_id: row for row in passages}
        self._facts = tuple(facts)
        self._bm25 = tuple(bm25)
        self.passage_reads: list[tuple[str, ...]] = []

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


def _calibrator(mode: str) -> tuple[HeadCalibrator, AbstentionPolicy]:
    artifact = load_calibration(_ARTIFACT_PATH)
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
