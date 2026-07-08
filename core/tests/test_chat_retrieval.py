from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from taurine_core import config
from taurine_core.chat.retrieval import ChatRetriever
from taurine_core.indexer import store as store_module
from taurine_core.live.grounding import GroundingEngine
from taurine_core.schemas import Enrichment, GraphEdge, GraphNode, GraphResponse, NoteRecord

from tests.fakes import note


class _CallableNotes(dict[str, NoteRecord]):
    def __call__(self, source: str | None = None) -> list[NoteRecord]:
        rows = list(self.values())
        if source is not None:
            rows = [row for row in rows if row.source == source]
        return rows


def _vec(axis: int) -> np.ndarray:
    values = np.zeros(config.EMBED_DIM, dtype=np.float32)
    values[axis] = 1.0
    return values


def _enrichment(gist: str) -> Enrichment:
    return Enrichment(
        gist=gist,
        utterances=[gist],
        aliases=[],
        topics=["work"],
        entities=[],
    )


@dataclass
class ChatFakeIndex:
    notes: dict[str, NoteRecord] = field(default_factory=dict)
    dense_by_query: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    bm25_by_query: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    fact_by_query: dict[str, list[tuple[str, str, float, str]]] = field(
        default_factory=dict
    )
    alias_rows: list[tuple[str, str, bool]] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    registries: dict[str, dict[str, Any]] = field(default_factory=dict)
    dense_calls: list[tuple[str, int, str | None]] = field(default_factory=list)
    fact_calls: list[tuple[str, int, str | None]] = field(default_factory=list)
    version: int = 1

    def __post_init__(self) -> None:
        self.notes = _CallableNotes(self.notes)

    def dense_search(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, float]]:
        self.dense_calls.append((text, k, prompt))
        return self.dense_by_query.get(text, [])[:k]

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.bm25_by_query.get(text, [])[:k]

    def fact_dense_search(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, str, float, str]]:
        self.fact_calls.append((text, k, prompt))
        return self.fact_by_query.get(text, [])[:k]

    def note(self, note_id: str) -> NoteRecord | None:
        return self.notes.get(note_id)

    def aliases(self) -> list[tuple[str, str, bool]]:
        return self.alias_rows

    def graph_version(self) -> int:
        return self.version

    def graph(self) -> GraphResponse:
        active = set(self.notes)
        degree = {note_id: 0 for note_id in active}
        edges: list[GraphEdge] = []
        for edge in self.edges:
            if edge.src not in active or edge.dst not in active:
                continue
            degree[edge.src] += 1
            degree[edge.dst] += 1
            edges.append(edge)
        return GraphResponse(
            nodes=[
                GraphNode(
                    note_id=row.note_id,
                    title=row.title,
                    degree=degree.get(row.note_id, 0),
                    source=row.source,
                )
                for row in self.notes.values()
            ],
            edges=edges,
        )

    def chat_registry(self, chat_id: str) -> dict:
        return self.registries.get(chat_id, {})

    def set_chat_registry(self, chat_id: str, registry: dict) -> None:
        self.registries[chat_id] = registry


def test_context_pack_respects_chat_sizes_and_has_no_hysteresis(monkeypatch) -> None:
    monkeypatch.setattr(config, "CHAT_CONTEXT_NOTES", 1)
    monkeypatch.setattr(config, "CHAT_CONTEXT_FACTS", 1)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    now = 1_000.0
    index = ChatFakeIndex(
        notes={
            "incumbent": note("incumbent", "Incumbent"),
            "challenger": note("challenger", "Challenger").model_copy(
                update={"quality_rank": 0.05}
            ),
            "fact-a": note("fact-a", "Fact A"),
            "fact-b": note("fact-b", "Fact B"),
            "fact-c": note("fact-c", "Fact C"),
        },
        dense_by_query={
            "first": [("incumbent", 1.0)],
            "second": [("incumbent", 1.0), ("challenger", 0.9)],
        },
        bm25_by_query={
            "second": [("challenger", 1.0), ("incumbent", 0.9)],
        },
        fact_by_query={
            "second": [
                ("fact-a", "fact-b", 1.0, "first fact"),
                ("fact-b", "fact-c", 0.9, "second fact"),
            ],
        },
    )
    retriever = ChatRetriever(index, "chat-a")  # type: ignore[arg-type]

    first_items, _first_facts = retriever.context_pack("first", now=now)
    second_items, second_facts = retriever.context_pack("second", now=now)

    assert [item.note_id for item in first_items] == ["incumbent"]
    assert [item.note_id for item in second_items] == ["challenger"]
    assert len(second_items) == config.CHAT_CONTEXT_NOTES
    assert len(second_facts) == config.CHAT_CONTEXT_FACTS


def test_seed_decay_is_per_turn_and_search_does_not_decay_mid_turn(monkeypatch) -> None:
    monkeypatch.setattr(config, "CHAT_CONTEXT_NOTES", 4)
    monkeypatch.setattr(config, "CHAT_CONTEXT_FACTS", 2)
    monkeypatch.setattr(config, "CHAT_SEED_DECAY", 0.25)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    index = ChatFakeIndex(
        notes={
            "alpha": note("alpha", "Alpha"),
            "beta": note("beta", "Beta"),
            "gamma": note("gamma", "Gamma"),
        },
        dense_by_query={
            "turn-one": [("alpha", 1.0)],
            "turn-two": [("beta", 1.0)],
            "tool-query": [("gamma", 1.0)],
        },
    )
    retriever = ChatRetriever(index, "chat-seeds")  # type: ignore[arg-type]

    retriever.context_pack("turn-one", now=1_000.0)
    first_alpha_seed = retriever._grounding._seeds["alpha"]
    retriever.context_pack("turn-two", now=1_001.0)

    assert retriever._grounding._seeds["alpha"] == pytest.approx(
        first_alpha_seed * config.CHAT_SEED_DECAY
    )
    alpha_after_turn = retriever._grounding._seeds["alpha"]

    retriever.search("tool-query", 2, now=1_002.0)

    assert retriever._grounding._seeds["alpha"] == pytest.approx(alpha_after_turn)
    assert "gamma" in retriever._grounding._seeds


def test_chat_uses_chat_query_prompt_and_store_default_stays_live_prompt(
    store,
    monkeypatch,
) -> None:
    index = ChatFakeIndex(
        notes={"alpha": note("alpha", "Alpha")},
        dense_by_query={"chat prompt": [("alpha", 1.0)]},
    )
    retriever = ChatRetriever(index, "chat-prompt")  # type: ignore[arg-type]

    retriever.context_pack("chat prompt", now=1_000.0)

    assert index.dense_calls == [
        ("chat prompt", config.HEAD_TOP_K, config.CHAT_EMBED_QUERY_PROMPT)
    ]
    assert index.fact_calls == [
        ("chat prompt", config.CHAT_CONTEXT_FACTS, config.CHAT_EMBED_QUERY_PROMPT)
    ]

    store.upsert_note(
        note("stored-alpha", "Stored Alpha"),
        _enrichment("stored alpha gist"),
        [("gist", "stored alpha gist", _vec(0))],
    )
    prompts: list[str | None] = []

    def fake_embed_queries(texts: list[str], *, prompt: str | None = None) -> np.ndarray:
        prompts.append(prompt)
        return np.vstack([_vec(0) for _text in texts])

    monkeypatch.setattr(store_module, "embed_queries", fake_embed_queries)

    store.dense_search("live prompt", 1)
    store.dense_search("chat prompt", 1, prompt=config.CHAT_EMBED_QUERY_PROMPT)

    # the default path passes NO prompt kwarg (None): embed.embed_queries owns
    # the None -> EMBED_QUERY_PROMPT resolution, so prompt-less fakes keep
    # working and real TypeErrors are never masked (Stage A review finding)
    assert prompts == [None, config.CHAT_EMBED_QUERY_PROMPT]


def test_exact_hits_are_rank_one_and_fact_hits_credit_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(config, "CHAT_CONTEXT_NOTES", 4)
    monkeypatch.setattr(config, "CHAT_CONTEXT_FACTS", 3)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    index = ChatFakeIndex(
        notes={
            "mermaid": note("mermaid", "Mermaid Roadmap"),
            "fact-a": note("fact-a", "Fact A"),
            "fact-b": note("fact-b", "Fact B"),
        },
        alias_rows=[("mermaid roadmap", "mermaid", True)],
        fact_by_query={
            "What happened with Mermaid Roadmap?": [
                ("fact-a", "fact-b", 0.8, "Fact A depends on Fact B")
            ]
        },
    )
    retriever = ChatRetriever(index, "chat-exact")  # type: ignore[arg-type]

    items, facts = retriever.context_pack(
        "What happened with Mermaid Roadmap?",
        now=1_000.0,
    )
    by_id = {item.note_id: item for item in items}

    assert by_id["mermaid"].why.exact_alias == "mermaid roadmap"
    assert "exact" in by_id["mermaid"].why.heads
    assert by_id["fact-a"].why.heads == ["fact"]
    assert by_id["fact-b"].why.heads == ["fact"]
    assert [(fact.src, fact.dst, fact.text) for fact in facts] == [
        ("fact-a", "fact-b", "Fact A depends on Fact B")
    ]


def test_registry_ids_survive_turns_save_restore_and_fact_ids_are_never_reused(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "CHAT_CONTEXT_NOTES", 2)
    monkeypatch.setattr(config, "CHAT_CONTEXT_FACTS", 1)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    index = ChatFakeIndex(
        notes={
            "alpha": note("alpha", "Alpha"),
            "beta": note("beta", "Beta"),
            "gamma": note("gamma", "Gamma"),
        },
        dense_by_query={
            "ab": [("alpha", 1.0), ("beta", 1.0)],
            "ag": [("alpha", 1.0), ("gamma", 1.0)],
        },
        fact_by_query={
            "ab": [("alpha", "beta", 1.0, "AB fact")],
            "ag": [("alpha", "gamma", 1.0, "AG fact")],
        },
    )
    retriever = ChatRetriever(index, "chat-registry")  # type: ignore[arg-type]

    ab_items, ab_facts = retriever.context_pack("ab", now=1_000.0)
    ag_items, ag_facts = retriever.context_pack("ag", now=1_001.0)
    retriever.save()
    restored = ChatRetriever(index, "chat-registry")  # type: ignore[arg-type]
    restored_items, restored_facts = restored.context_pack("ab", now=1_002.0)

    assert [(item.note_id, item.short_id) for item in ab_items] == [
        ("alpha", "N1"),
        ("beta", "N2"),
    ]
    assert next(item.short_id for item in ag_items if item.note_id == "alpha") == "N1"
    assert [(fact.src, fact.dst, fact.fact_id) for fact in ab_facts] == [
        ("alpha", "beta", "F1")
    ]
    assert [(fact.src, fact.dst, fact.fact_id) for fact in ag_facts] == [
        ("alpha", "gamma", "F2")
    ]
    assert [(item.note_id, item.short_id) for item in restored_items] == [
        ("alpha", "N1"),
        ("beta", "N2"),
    ]
    assert [(fact.src, fact.dst, fact.fact_id) for fact in restored_facts] == [
        ("alpha", "beta", "F1")
    ]


def test_resolve_accepts_registry_refs_raw_note_ids_and_unknowns(monkeypatch) -> None:
    monkeypatch.setattr(config, "CHAT_CONTEXT_NOTES", 1)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    index = ChatFakeIndex(
        notes={
            "alpha": note("alpha", "Alpha"),
            "beta": note("beta", "Beta"),
        },
        dense_by_query={"alpha question": [("alpha", 1.0)]},
    )
    retriever = ChatRetriever(index, "chat-resolve")  # type: ignore[arg-type]

    retriever.context_pack("alpha question", now=1_000.0)

    assert retriever.resolve("N1") == "alpha"
    assert retriever.resolve("beta") == "beta"
    assert retriever.resolve("N404") is None
    assert retriever.resolve("missing") is None


def test_grounding_engine_parameters_and_state_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    index = ChatFakeIndex(
        notes={
            "alpha": note("alpha", "Alpha"),
            "beta": note("beta", "Beta"),
        },
        fact_by_query={"window": [("alpha", "beta", 1.0, "AB fact")]},
    )
    engine = GroundingEngine(
        index,  # type: ignore[arg-type]
        top_k=1,
        facts_max=1,
        seed_decay=0.5,
        hysteresis=None,
    )

    first = engine.on_pass("window", [("alpha", 1.0), ("beta", 1.0)], [], 1_000.0)
    note_short_ids = engine.note_short_ids()
    fact_short_ids = engine.fact_short_ids()
    assert note_short_ids == {"alpha": "N1"}
    assert fact_short_ids == {("alpha", "beta"): "F1"}
    note_short_ids["beta"] = "N404"
    fact_short_ids[("beta", "gamma")] = "F404"
    assert engine.note_short_ids() == {"alpha": "N1"}
    assert engine.fact_short_ids() == {("alpha", "beta"): "F1"}
    state = engine.export_state()
    restored = GroundingEngine(
        index,  # type: ignore[arg-type]
        top_k=1,
        facts_max=1,
        seed_decay=0.5,
        hysteresis=None,
    )
    restored.restore_state(state)
    second = restored.on_pass("window", [("beta", 1.0), ("alpha", 1.0)], [], 1_001.0)

    assert len(first.items) == 1
    assert len(first.facts) == 1
    assert first.items[0].short_id == "N1"
    assert second.items[0].short_id == "N2"
    assert second.facts[0].fact_id == "F1"
    assert restored._seeds["alpha"] == pytest.approx(
        state["seeds"]["alpha"] * 0.5 + 1.0
    )
