from __future__ import annotations

from types import SimpleNamespace

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
            trace=RetrievalTrace(original_query=query.text, rewritten_query=query.text, read_ids=(note_ref, evidence_ref), active_build_id="build-1"),
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
    retriever.save()
    first_ref = first.passages[0].evidence_id

    restored, _ = _make(index)
    restored.restore_pre_turn()
    regenerated = restored.context_pack("Roadmap", now=101.0)

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


def test_note_and_directed_fact_expansion_use_registered_graph_node_ids() -> None:
    retriever, _ = _make(FakeIndex())
    pack = retriever.context_pack("Roadmap", now=100.0)

    note_result = retriever.expand_evidence(
        pack.notes[0].note_ref,
        direction="outgoing",
        max_hops=1,
        limit=4,
    )
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
