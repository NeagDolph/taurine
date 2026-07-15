from __future__ import annotations

from types import SimpleNamespace

from taurine_core.dates import timestamp_date
from taurine_core.chat.retrieval import ChatRetriever
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.types import (
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
)


class FakeIndex:
    def __init__(self) -> None:
        self.registries: dict[str, dict[str, object]] = {}

    def chat_registry(self, chat_id: str) -> dict[str, object]:
        return self.registries.get(chat_id, {})

    def set_chat_registry(self, chat_id: str, state: dict[str, object]) -> None:
        self.registries[chat_id] = state


class FakeStore:
    def __init__(self) -> None:
        self.passages = (
            SimpleNamespace(
                passage_id="p1", note_id="n1", title="Roadmap", source="obsidian",
                text="The roadmap launches on July 18.", char_start=20, char_end=52,
                created=1.0, modified=2.0,
            ),
            SimpleNamespace(
                passage_id="p2", note_id="n1", title="Roadmap", source="obsidian",
                text="Armaan owns the review.", char_start=80, char_end=104,
                created=1.0, modified=2.0,
            ),
            SimpleNamespace(
                passage_id="p3", note_id="n2", title="API", source="obsidian",
                text="The API review is blocked.", char_start=0, char_end=26,
                created=1.0, modified=2.0,
            ),
        )

    def passage_contexts(self, passage_ids):
        wanted = set(passage_ids)
        return tuple(row for row in self.passages if row.passage_id in wanted)

    def note_contexts(self, note_ids):
        return tuple(
            SimpleNamespace(note_id="n1", title="Roadmap", source="obsidian", modified=2.0)
            for note_id in note_ids if note_id == "n1"
        )


class FakeEngine:
    def __init__(self, registry: EvidenceRegistry) -> None:
        self.registry = registry
        self.store = FakeStore()
        self.snapshot = SimpleNamespace(
            build_id="build-1",
            passage_ids=("p1", "p2", "p3"),
            base_graph=SimpleNamespace(edges=(
                SimpleNamespace(src_id="p1", dst_id="p3", kind="semantic", fact_ids=()),
                SimpleNamespace(src_id="e-a", dst_id="e-b", kind="fact:depends_on", fact_ids=("f1",)),
                SimpleNamespace(src_id="e-b", dst_id="p3", kind="contains", fact_ids=()),
            )),
        )
        self.calls: list[tuple[RetrievalQuery, dict[str, float] | None, tuple[tuple[str, str | None], ...]]] = []
        self.passage_limits: list[int | None] = []

    def retrieve(self, query: RetrievalQuery, *, session_reset=None, entity_spans=(), passage_limit=None, **_kwargs) -> EvidencePack:
        self.calls.append((query, dict(session_reset) if session_reset else None, tuple(entity_spans)))
        self.passage_limits.append(passage_limit)
        if query.text == "nothing":
            return EvidencePack.abstained(query, "below_threshold")
        note_ref = self.registry.register("note", "n1", {"note_id": "n1"})
        evidence_ref = self.registry.register(
            "passage", "p1",
            {"passage_id": "p1", "note_id": "n1", "text": "The roadmap launches on July 18.", "char_start": 20, "char_end": 52},
        )
        self.registry.mark_read(note_ref)
        self.registry.mark_read(evidence_ref)
        passage = EvidencePassage(
            evidence_id=evidence_ref, passage_id="p1", note_id="n1", title="Roadmap",
            source="obsidian", text="The roadmap launches on July 18.", char_start=20,
            char_end=52, created=1.0, modified=2.0, retrieval_score=.8, rerank_score=.9,
        )
        note = EvidenceNote(
            note_ref=note_ref, note_id="n1", title="Roadmap", source="obsidian",
            modified=2.0, score=.9, evidence_ids=(evidence_ref,),
        )
        return EvidencePack(
            decision=RetrievalDecision(True, .9, "accepted", ("dense", "bm25")),
            query=query,
            passages=(passage,), facts=(), paths=(), notes=(note,),
            trace=RetrievalTrace(
                original_query=query.text, rewritten_query=query.text,
                selected_ids=(note_ref, evidence_ref),
                read_ids=(note_ref, evidence_ref), active_build_id="build-1",
            ),
        )


def _make(index: FakeIndex, chat_id: str = "chat-1") -> tuple[ChatRetriever, list[FakeEngine]]:
    engines: list[FakeEngine] = []

    def factory(registry: EvidenceRegistry) -> FakeEngine:
        engine = FakeEngine(registry)
        engines.append(engine)
        return engine

    return ChatRetriever(
        index,
        chat_id,
        engine_factory=factory,
        tool_engine_factory=factory,
    ), engines  # type: ignore[arg-type]


def test_stage_a_rewrites_dependent_query_and_keeps_original_in_trace() -> None:
    retriever, engines = _make(FakeIndex())

    pack = retriever.context_pack(
        "When does it launch?",
        now=100.0,
        conversation_context=("Tell me about the roadmap", "It is the Mermaid project"),
    )

    assert pack.query.text == "When does it launch?"
    assert pack.trace.original_query == "When does it launch?"
    assert pack.trace.rewritten_query == "Tell me about the roadmap\nIt is the Mermaid project\nWhen does it launch?"
    assert engines[-1].calls[0][0].text == pack.trace.rewritten_query
    assert engines[-1].calls[0][0].conversation_context == (
        "Tell me about the roadmap", "It is the Mermaid project",
    )


def test_independent_topic_does_not_rewrite_or_retain_old_reset() -> None:
    retriever, engines = _make(FakeIndex())
    retriever.context_pack("Tell me about Mermaid", now=100.0)

    pack = retriever.context_pack(
        "What is the garden plan?",
        now=101.0,
        conversation_context=("Tell me about Mermaid",),
    )

    assert pack.trace.rewritten_query == "What is the garden plan?"
    assert engines[-1].calls[-1][1] is None


def test_explicit_what_about_topic_pivot_drops_previous_topic() -> None:
    retriever, engines = _make(FakeIndex())
    retriever.context_pack("Tell me about Mermaid", now=100.0)

    pack = retriever.context_pack(
        "What about the garden plan?",
        now=101.0,
        conversation_context=("Tell me about Mermaid",),
    )

    assert pack.trace.rewritten_query == "What about the garden plan?"
    assert engines[-1].calls[-1][1] is None


def test_seed_notes_bias_exactly_one_accepted_turn_and_ignore_unknown_ids() -> None:
    retriever, engines = _make(FakeIndex())
    retriever.seed_notes(["n1", "missing-note", " "])

    retriever.context_pack("Tell me about the roadmap", now=100.0)
    assert engines[-1].calls[0][1] == {"p1": 1.0, "p2": 1.0}
    retriever.commit_turn()

    pack = retriever.context_pack(
        "What is the garden plan?",
        now=101.0,
        conversation_context=("Tell me about the roadmap",),
    )
    assert pack.trace.rewritten_query == "What is the garden plan?"
    assert engines[-1].calls[-1][1] is None  # seeds cleared after the first turn


def test_seed_notes_persist_via_registry_until_the_first_turn() -> None:
    index = FakeIndex()
    seeded, _ = _make(index)
    seeded.seed_notes(["n2"])
    seeded.save()

    restored, engines = _make(index)
    restored.context_pack("Roadmap", now=100.0)

    assert engines[-1].calls[0][1] == {"p3": 1.0}
    restored.commit_turn()
    restored.save()
    fresh, fresh_engines = _make(index)
    fresh.context_pack("Roadmap", now=101.0)
    assert fresh_engines[-1].calls[0][1] is None


def test_chat_engine_seed_chat_writes_seed_reset_for_the_first_turn() -> None:
    from taurine_core.chat.engine import ChatEngine

    index = FakeIndex()
    engines: list[FakeEngine] = []

    def engine_factory(registry: EvidenceRegistry) -> FakeEngine:
        engine = FakeEngine(registry)
        engines.append(engine)
        return engine

    def retriever_factory(chat_id: str) -> ChatRetriever:
        return ChatRetriever(
            index,
            chat_id,
            engine_factory=engine_factory,
            tool_engine_factory=engine_factory,
        )  # type: ignore[arg-type]

    chat_engine = ChatEngine(index, None, None, retriever_factory=retriever_factory)
    chat_engine.seed_chat("chat-seeded", ["n1", "missing"])

    first_turn = retriever_factory("chat-seeded")
    first_turn.context_pack("Roadmap", now=100.0)
    assert engines[-1].calls[0][1] == {"p1": 1.0, "p2": 1.0}


def test_tool_search_uses_tool_mode_without_advancing_accepted_turn_state() -> None:
    retriever, engines = _make(FakeIndex())
    retriever.context_pack("Roadmap", now=100.0)
    before = retriever.snapshot()

    pack = retriever.search_evidence("Armaan review", k=4, temporal="current", now=101.0)

    assert pack.query.mode == "tool"
    assert pack.query.temporal_intent is not None
    assert pack.query.temporal_intent.kind == "current"
    assert retriever.snapshot()["accepted_reset"] == before["accepted_reset"]
    assert engines[-1].calls[-1][1] == {"p1": .9}
    assert engines[-1].passage_limits[-1] == 4
    assert engines[0] is not engines[-1]


def test_speculative_tool_search_discard_restores_pre_turn_snapshot() -> None:
    retriever, _engines = _make(FakeIndex())
    baseline = retriever.snapshot()

    retriever.begin_turn()
    retriever.search_evidence("Roadmap", k=4, temporal="any", now=101.0)
    retriever.discard_turn()

    assert retriever.snapshot() == baseline


def test_begin_turn_clears_the_prior_turn_audit_pack() -> None:
    retriever, _engines = _make(FakeIndex())
    retriever.context_pack("Roadmap", now=100.0)
    retriever.commit_turn()

    retriever.begin_turn()
    current = retriever.search_evidence("Armaan review", k=4, temporal="any", now=101.0)

    assert retriever.turn_pack().query.text == current.query.text


def test_accepted_tool_search_promotes_abstained_stage_a_for_turn_audit() -> None:
    retriever, _ = _make(FakeIndex())
    stage_a = retriever.context_pack("nothing", now=100.0)
    assert stage_a.decision.accepted is False

    tool_pack = retriever.search_evidence("Roadmap", k=4, temporal="any", now=101.0)
    audit = retriever.turn_pack()

    assert tool_pack.decision.accepted is True
    assert audit is not None and audit.decision.accepted is True
    assert audit.trace.original_query == "nothing"
    assert audit.trace.debug["stage_a_abstention"] == "below_threshold"
    assert audit.trace.debug["tool_query"] == "Roadmap"


def test_registry_persists_and_regeneration_restores_pre_turn_snapshot() -> None:
    index = FakeIndex()
    retriever, _engines = _make(index)
    first = retriever.context_pack("Roadmap", now=100.0)
    retriever.commit_turn()
    retriever.save()
    first_ref = first.passages[0].evidence_id

    restored, _ = _make(index)
    restored.restore_pre_turn()
    regenerated = restored.context_pack("Roadmap", now=101.0)
    restored.commit_turn()

    assert regenerated.passages[0].evidence_id == first_ref
    assert restored.snapshot()["accepted_turn"] == 1


def test_typed_query_gliner_spans_seed_shared_entity_head() -> None:
    class Spans:
        def extract(self, text: str):
            assert text == "What is Armaan reviewing?"
            return [("Armaan", .97, "person")]

    index = FakeIndex()
    engines: list[FakeEngine] = []

    def factory(registry: EvidenceRegistry) -> FakeEngine:
        engine = FakeEngine(registry)
        engines.append(engine)
        return engine

    retriever = ChatRetriever(
        index,
        "chat-spans",
        engine_factory=factory,
        tool_engine_factory=factory,
        span_extractor=Spans(),
    )  # type: ignore[arg-type]

    retriever.context_pack("What is Armaan reviewing?", now=100.0)

    assert engines[-1].calls[-1][2] == (("Armaan", "person"),)


def test_read_note_is_query_relevant_registered_and_paginated() -> None:
    retriever, _ = _make(FakeIndex())
    pack = retriever.context_pack("Roadmap", now=100.0)
    note_ref = pack.notes[0].note_ref

    page = retriever.read_note(note_ref, query="Armaan review", max_chars=30)

    assert page["note"] == note_ref
    assert page["truncated"] is True
    assert page["next_cursor"]
    assert page["passages"][0]["text"] == "Armaan owns the review."
    assert page["passages"][0]["passage_date"] == timestamp_date(2.0)
    assert page["passages"][0]["evidence_id"].startswith("E")
    assert page["passages"][0]["evidence_id"] in retriever.registry.refs("read")
    record = retriever.registry.resolve(page["passages"][0]["evidence_id"])
    assert record is not None
    assert record.payload["text"] == page["passages"][0]["text"]

    turn_pack = retriever.turn_pack()
    assert turn_pack is not None
    assert {row.evidence_id for row in turn_pack.passages} == {"E1", page["passages"][0]["evidence_id"]}


def test_read_note_pages_one_oversized_passage_without_registering_hidden_text() -> None:
    retriever, _ = _make(FakeIndex())
    pack = retriever.context_pack("Roadmap", now=100.0)
    note_ref = pack.notes[0].note_ref

    first = retriever.read_note(note_ref, query="launch", max_chars=10)
    second = retriever.read_note(
        note_ref,
        query="launch",
        cursor=str(first["next_cursor"]),
        max_chars=10,
    )

    assert first["passages"][0]["text"] == "The roadma"
    assert second["passages"][0]["text"] == "p launches"
    first_record = retriever.registry.resolve(first["passages"][0]["evidence_id"])
    second_record = retriever.registry.resolve(second["passages"][0]["evidence_id"])
    assert first_record is not None and first_record.payload["text"] == "The roadma"
    assert second_record is not None and second_record.payload["text"] == "p launches"


def test_turn_audit_does_not_leak_evidence_read_in_prior_turns() -> None:
    retriever, _ = _make(FakeIndex())
    old_ref = retriever.registry.register(
        "passage", "old-passage",
        {"passage_id": "old-passage", "note_id": "old-note", "text": "old", "char_start": 0, "char_end": 3},
    )
    retriever.registry.mark_read(old_ref)

    retriever.context_pack("Roadmap", now=100.0)
    audit = retriever.turn_pack()

    assert audit is not None
    assert old_ref not in audit.trace.read_ids
    assert old_ref not in audit.trace.selected_ids


def test_accumulator_advances_only_on_commit_and_discard_restores_pre_turn() -> None:
    """H7: failed turns roll back wholesale; only commit_turn advances state."""

    retriever, engines = _make(FakeIndex())
    retriever.seed_notes(["n1"])
    baseline = retriever.snapshot()

    retriever.context_pack("Roadmap", now=100.0)
    retriever.discard_turn()
    assert retriever.snapshot() == baseline  # registry, seeds, counters intact

    # the retry after a failed turn is still seed-biased
    retriever.context_pack("Roadmap", now=101.0)
    assert engines[-1].calls[-1][1] == {"p1": 1.0, "p2": 1.0}
    retriever.commit_turn()
    after = retriever.snapshot()
    assert after["accepted_turn"] == 1
    assert after["seed_reset"] == {}
    assert after["accepted_reset"] == {"p1": 0.9}


def test_turn_pack_preserves_stage_a_selected_ids_and_tracks_reads_separately() -> None:
    """M5: retrieved (Stage A candidates) and consulted (read) stay distinct."""

    retriever, _ = _make(FakeIndex())
    pack = retriever.context_pack("Roadmap", now=100.0)
    page = retriever.read_note(pack.notes[0].note_ref, query="Armaan review", max_chars=30)

    audit = retriever.turn_pack()
    assert audit is not None
    assert audit.trace.selected_ids == pack.trace.selected_ids
    tool_ref = str(page["passages"][0]["evidence_id"])
    assert tool_ref not in audit.trace.selected_ids
    assert tool_ref in audit.trace.read_ids
    assert set(pack.trace.read_ids) < set(audit.trace.read_ids)


def test_evidence_sufficient_reflects_calibrated_turn_acceptance() -> None:
    """M11: sufficiency is the turn base's accepted decision, tool packs included."""

    retriever, _ = _make(FakeIndex())
    assert retriever.evidence_sufficient() is False
    stage_a = retriever.context_pack("nothing", now=100.0)
    assert stage_a.decision.accepted is False
    assert retriever.evidence_sufficient() is False

    retriever.search_evidence("Roadmap", k=4, temporal="any", now=101.0)
    assert retriever.evidence_sufficient() is True


def test_expose_fact_support_marks_bounded_support_refs_read() -> None:
    """M6: ≤1 non-selected support passage per fact, ≤4 per pack, all read."""

    from taurine_core.retrieval.types import EvidenceFact

    retriever, _ = _make(FakeIndex())
    registry = retriever.registry
    selected = registry.register(
        "passage", "p1",
        {"passage_id": "p1", "note_id": "n1", "text": "The roadmap launches on July 18.", "char_start": 20, "char_end": 52},
    )
    registry.mark_read(selected)
    support_a = registry.register(
        "passage", "p2",
        {"passage_id": "p2", "note_id": "n1", "text": "Armaan owns the review.", "char_start": 80, "char_end": 104},
    )
    support_b = registry.register(
        "passage", "p3",
        {"passage_id": "p3", "note_id": "n2", "text": "The API review is blocked.", "char_start": 0, "char_end": 26},
    )

    def fact(ref: str, sources: tuple[str, ...]) -> EvidenceFact:
        return EvidenceFact(
            fact_ref=ref, fact_ids=("f-" + ref,), subject="A", predicate="relates_to",
            object="B", direction="forward", status="active", observed_at=1.0,
            valid_from=None, valid_to=None, confidence=.9, source_evidence_ids=sources,
        )

    passage = EvidencePassage(
        evidence_id=selected, passage_id="p1", note_id="n1", title="Roadmap",
        source="obsidian", text="The roadmap launches on July 18.", char_start=20,
        char_end=52, created=1.0, modified=2.0, retrieval_score=.8, rerank_score=.9,
        passage_date="2026-07-18",
    )
    pack = EvidencePack(
        decision=RetrievalDecision(True, .9, "accepted"),
        query=RetrievalQuery("q", "chat", 1.0),
        passages=(passage,),
        facts=(
            fact("F1", (selected, support_a, support_b)),
            fact("F2", (support_b,)),
        ),
        paths=(), notes=(),
        trace=RetrievalTrace(original_query="q"),
    )

    exposed = retriever.expose_fact_support(pack)

    # one support per fact; the already-rendered ref is not re-exposed
    assert exposed["F1"][0] == support_a
    assert exposed["F2"][0] == support_b
    assert exposed["F1"][1] == "Roadmap"  # title resolved from the store
    assert exposed["F1"][2] == timestamp_date(2.0)
    assert exposed["F1"][3] == "Armaan owns the review."
    for ref in (support_a, support_b):
        record = registry.resolve(ref)
        assert record is not None and "read" in record.states
    audit_reads = retriever._turn_read_ids
    assert support_a in audit_reads and support_b in audit_reads

    # global cap: at most 4 extra passages per pack
    many = tuple(
        fact(f"F{index}", (registry.register(
            "passage", f"px{index}",
            {"passage_id": f"px{index}", "note_id": f"nx{index}", "text": f"text {index}", "char_start": 0, "char_end": 6},
        ),))
        for index in range(10, 20)
    )
    capped = retriever.expose_fact_support(
        EvidencePack(
            decision=RetrievalDecision(True, .9, "accepted"),
            query=RetrievalQuery("q", "chat", 1.0),
            passages=(), facts=many, paths=(), notes=(),
            trace=RetrievalTrace(original_query="q"),
        )
    )
    assert len(capped) == 4


def test_note_and_directed_fact_expansion_use_registered_graph_node_ids() -> None:
    retriever, _ = _make(FakeIndex())
    pack = retriever.context_pack("Roadmap", now=100.0)

    note_result = retriever.expand_evidence(
        pack.notes[0].note_ref,
        direction="outgoing",
        max_hops=1,
        limit=4,
    )
    assert note_result["results"][0]["passage_date"] == timestamp_date(2.0)
    fact_ref = retriever.registry.register(
        "fact",
        "build-1\x1fe-a\x1fdepends_on\x1fe-b",
        {
            "logical_key": ["A", "depends_on", "B"],
            "subject_entity_id": "e-a",
            "object_entity_id": "e-b",
            "subject": "A",
            "predicate": "depends_on",
            "object": "B",
            "direction": "forward",
            "status": "active",
            "observed_at": 1.0,
        },
    )
    retriever.registry.mark_read(fact_ref)
    fact_result = retriever.expand_evidence(
        fact_ref,
        direction="outgoing",
        max_hops=2,
        limit=4,
    )

    assert note_result["results"][0]["text"] == "The API review is blocked."
    assert fact_result["results"][0]["text"] == "The API review is blocked."
    assert fact_result["paths"][0]["edge_kinds"] == ["fact:depends_on", "contains"]


# ── C2: real-registry integration over a BuildCoordinator-built DB ──────


def _built_chat_retriever(tmp_path, notes, *, extractor=None, query_vector=None):
    from pathlib import Path

    from tests.conftest import make_note  # noqa: F401 - used by callers
    from tests.test_graph_retrieval_integration import (
        RelevantScorer,
        Tokenizer,
        embedder,
    )
    from taurine_core.evaluation.graph_retrieval import load_calibration
    from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
    from taurine_core.retrieval.calibration import AbstentionPolicy, HeadCalibrator
    from taurine_core.retrieval.evidence import SharedRetrievalEngine
    from taurine_core.retrieval.rerank import LocalPassageReranker
    from taurine_core.retrieval.snapshot import load_index_snapshot

    database = tmp_path / "taurine-knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build(list(notes), database, replace_derived=True)
    snapshot = load_index_snapshot(database)
    store = snapshot.open_knowledge_store()
    artifact = load_calibration(
        Path(__file__).resolve().parents[1]
        / "evals"
        / "graph_retrieval"
        / "calibration-v1.json"
    )

    def engine_factory(registry: EvidenceRegistry) -> SharedRetrievalEngine:
        return SharedRetrievalEngine(
            snapshot,
            store,
            HeadCalibrator(
                artifact,
                mode="chat",
                artifact_hash=artifact.hash,
                config_hash=artifact.config_hash,
                model_manifest=artifact.model_manifest,
            ),
            AbstentionPolicy(artifact, mode="chat"),
            query_embedder=lambda _text: (
                query_vector if query_vector is not None else snapshot.passage_matrix[0]
            ),
            registry=registry,
            reranker=LocalPassageReranker(
                RelevantScorer(),  # type: ignore[arg-type]
                mode="chat",
                acceptance_floor=artifact.rerank.chat_floor,
            ),
        )

    retriever = ChatRetriever(
        FakeIndex(),
        "chat-int",
        engine_factory=engine_factory,
        tool_engine_factory=engine_factory,
    )
    return retriever, snapshot, store


def test_read_note_full_passages_reuse_stage_a_refs_on_a_real_build(tmp_path) -> None:
    """C2: 'read the note you just cited' must reuse the engine's E# refs."""

    from taurine_core import config
    from tests.conftest import make_note

    retriever, _snapshot, store = _built_chat_retriever(
        tmp_path,
        [make_note("launch", "Launch plan", "The launch is scheduled for July 18.")],
    )
    try:
        pack = retriever.context_pack("When is the launch?", now=2.0)
        assert pack.decision.accepted
        stage_a_ref = pack.passages[0].evidence_id

        page = retriever.read_note(
            pack.notes[0].note_ref,
            query="launch schedule",
            max_chars=config.CHAT_NOTE_READ_MAX_CHARS,
        )

        refs = [str(row["evidence_id"]) for row in page["passages"]]
        assert stage_a_ref in refs
        for ref in refs:
            record = retriever.registry.resolve(ref)
            assert record is not None
            assert ":excerpt:" not in record.source_id
            assert "excerpt" not in record.payload
    finally:
        store.close()


def test_expand_evidence_reuses_engine_grouped_fact_ref_on_a_real_build(tmp_path) -> None:
    """C2: a multi-provenance fact expands against the engine's grouped ref."""

    import numpy as np

    from tests.conftest import make_note
    from taurine_core.retrieval.evidence_payloads import (
        FACT_APPEND_FIELDS,
        fact_group_payload,
        fact_source_identity,
    )
    from taurine_core.schemas import ExtractedMention, Extraction, ExtractedRelation

    def asserted(note_id: str) -> Extraction:
        del note_id
        return Extraction(
            mentions=[
                ExtractedMention(surface="Maya", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                ExtractedMention(surface="Launch", etype="project", confidence=0.9, char_start=10, char_end=16, extractor="fake", schema_id="s1"),
            ],
            fact_candidates=[
                ExtractedRelation(head="Maya", predicate="owns_or_has", tail="Launch", confidence=0.9, subject_start=0, subject_end=4, object_start=10, object_end=16, assertion_start=0, assertion_end=17, extraction_method="fake", schema_id="s1", status="active")
            ],
            schema_id="s1",
        )

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": asserted("n1"),
            "n2": asserted("n2"),
            "other": Extraction(
                mentions=[
                    ExtractedMention(surface="Launch", etype="project", confidence=0.9, char_start=0, char_end=6, extractor="fake", schema_id="s1"),
                ],
                schema_id="s1",
            ),
        }

    retriever, snapshot, store = _built_chat_retriever(
        tmp_path,
        [
            make_note("n1", "Ownership", "Maya owns Launch."),
            make_note("n2", "Ownership again", "Maya owns Launch."),
            make_note("other", "Shipping", "Launch shipped smoothly."),
        ],
        extractor=extractor,
        query_vector=np.zeros(16, dtype=np.float32),
    )
    try:
        contexts = tuple(store.fact_contexts())
        assert len(contexts) == 2, "fixture must produce a multi-provenance fact"
        assert len({row.logical_key for row in contexts}) == 1
        ordered = tuple(sorted(contexts, key=lambda row: str(row.fact_id)))
        first = ordered[0]

        # Simulate the engine's Stage A grouped registration byte-for-byte
        # (the shared helpers ARE the engine's payload builders).
        registry = retriever.registry
        identity = fact_source_identity(
            snapshot.build_id, first.logical_key, str(first.status),
            first.valid_from, first.valid_to,
        )
        engine_ref = registry.register(
            "fact",
            identity,
            fact_group_payload(
                first.logical_key, str(first.status),
                first.valid_from, first.valid_to, ordered,
            ),
        )
        registry.merge_payload(
            engine_ref,
            {"fact_ids": [str(row.fact_id) for row in ordered], "source_evidence_ids": []},
            append_list_fields=FACT_APPEND_FIELDS,
        )
        registry.mark_read(engine_ref)

        result = retriever.expand_evidence(
            engine_ref, direction="outgoing", max_hops=2, limit=8
        )

        traversed = [
            row for row in result["results"] if engine_ref in row["fact_refs"]
        ]
        assert traversed, result
        record = registry.resolve(engine_ref)
        assert record is not None
        assert set(record.payload["fact_ids"]) == {str(row.fact_id) for row in ordered}
        assert record.payload["observed_at"] == max(
            float(row.observed_at) for row in ordered
        )
        # both provenance passages are exposed as support and marked read
        support_ids = {row["evidence_id"] for row in result["supporting_passages"]}
        assert len(support_ids) == 2
    finally:
        store.close()
