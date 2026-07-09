from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from tests.conftest import entity, fixture_units, make_note, unit_vec
from tests.fakes import note
from taurine_core import config
from taurine_core.chat.retrieval import ChatRetriever
from taurine_core.indexer import graph
from taurine_core.live.grounding import GroundingEngine
from taurine_core.schemas import Enrichment, GraphNode, GraphResponse, NoteRecord


def _basis(index: int) -> np.ndarray:
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    vector[index % config.EMBED_DIM] = 1.0
    return vector


def _hash_embedding(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=config.EMBED_DIM).astype(np.float32)
    values /= np.linalg.norm(values)
    return values.astype(np.float32)


@pytest.fixture
def term_embedding_stub(monkeypatch: pytest.MonkeyPatch):
    vectors = {
        "billing flow": _basis(1),
        "Shop LC": _basis(2),
        "ShopLC": _basis(2),
        "Acme": _basis(3),
        "Jane Doe": _basis(4),
    }

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        return np.vstack([vectors.get(text, _hash_embedding(text)) for text in texts])

    monkeypatch.setattr(graph, "embed_texts", fake_embed_texts, raising=False)


def _install(
    store,
    note_id: str,
    *,
    entities: list = None,
    vector: np.ndarray | None = None,
) -> None:
    enrich = Enrichment(
        gist=f"{note_id} gist",
        utterances=[],
        aliases=[],
        entities=entities or [],
        relations=[],
        extraction_failed=False,
    )
    store.upsert_note(
        make_note(note_id, note_id.title(), f"{note_id} body"),
        enrich,
        fixture_units(vector if vector is not None else unit_vec(abs(hash(note_id))), enrich),
    )


def _active_entity_rows(store) -> dict[tuple[str, str], tuple[int, int]]:
    return {
        (canonical, etype): (entity_id, df)
        for entity_id, canonical, etype, df in store.entity_rows()
        if df > 0
    }


def test_entity_by_surface_resolves_synonym_members_and_case_rules(
    store,
    term_embedding_stub,
) -> None:
    _install(
        store,
        "alpha",
        entities=[
            entity("Billing-Flow", "concept", conf=0.9),
            entity("Shop LC", "project", conf=0.8),
            entity("Jane Doe", "person", conf=0.7),
        ],
        vector=_basis(11),
    )
    _install(
        store,
        "beta",
        entities=[
            entity("billing_flow", "concept", conf=0.75),
            entity("ShopLC", "project", conf=0.65),
        ],
        vector=_basis(12),
    )

    store.rebuild_graph()

    rows = _active_entity_rows(store)
    billing_id = rows[("billing flow", "concept")][0]
    shop_id = rows[("ShopLC", "project")][0]
    jane_id = rows[("Jane Doe", "person")][0]

    assert store.entity_by_surface("billing flow") == billing_id
    assert store.entity_by_surface("Billing Flow") == billing_id
    assert store.entity_by_surface("Billing-Flow") == billing_id
    assert store.entity_by_surface("ShopLC") == shop_id
    assert store.entity_by_surface("Shop LC") == shop_id
    assert store.entity_by_surface("shop lc") is None
    assert store.entity_by_surface("Jane Doe") == jane_id
    assert store.entity_by_surface("jane doe") is None
    assert store.entity_by_surface("Missing") is None


def test_grounding_entity_span_seeds_containing_note_without_accumulating(
    store,
    term_embedding_stub,
) -> None:
    _install(store, "alpha", entities=[entity("Acme", "organization")], vector=_basis(21))
    _install(store, "beta", entities=[], vector=_basis(22))
    _install(store, "gamma", entities=[], vector=_basis(23))
    store.rebuild_graph()

    without_entity = GroundingEngine(store, top_k=3, facts_max=0, hysteresis=None)
    without_update = without_entity.on_pass("Where is Acme?", [], [], 1_000.0)

    with_entity = GroundingEngine(store, top_k=3, facts_max=0, hysteresis=None)
    with_update = with_entity.on_pass(
        "Where is Acme?",
        [],
        [],
        1_000.0,
        gliner_spans=[("Acme", 0.5, "organization")],
    )

    assert without_update.ppr_top == []
    assert [item.note_id for item in with_update.items] == ["alpha"]
    assert with_update.ppr_top == [("alpha", pytest.approx(1.0))]
    assert with_entity._seeds == {}


class _CallableNotes(dict[str, NoteRecord]):
    def __call__(self, source: str | None = None) -> list[NoteRecord]:
        rows = list(self.values())
        if source is not None:
            rows = [row for row in rows if row.source == source]
        return rows


@dataclass
class _EntitySeedIndex:
    notes: dict[str, NoteRecord] = field(default_factory=dict)
    entity_lookup: dict[str, int] = field(default_factory=dict)
    registries: dict[str, dict[str, Any]] = field(default_factory=dict)
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
        del text, k, prompt
        return []

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        del text, k
        return []

    def fact_dense_search(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, str, float, str]]:
        del text, k, prompt
        return []

    def note(self, note_id: str) -> NoteRecord | None:
        return self.notes.get(note_id)

    def aliases(self) -> list[tuple[str, str, bool]]:
        return []

    def graph_version(self) -> int:
        return self.version

    def graph(self) -> GraphResponse:
        return GraphResponse(
            nodes=[
                GraphNode(
                    note_id=row.note_id,
                    title=row.title,
                    degree=0,
                    source=row.source,
                )
                for row in self.notes.values()
            ],
            edges=[],
        )

    def entity_rows(self) -> list[tuple[int, str, str, int]]:
        return [(1, "Acme", "organization", 1)]

    def note_entity_rows(self) -> list[tuple[str, int, float, int]]:
        return [("alpha", 1, 1.0, 1)]

    def relation_rows(self) -> list[tuple[str, int, str, int, float]]:
        return []

    def entity_by_surface(self, surface: str) -> int | None:
        return self.entity_lookup.get(surface)

    def chat_registry(self, chat_id: str) -> dict:
        return self.registries.get(chat_id, {})

    def set_chat_registry(self, chat_id: str, registry: dict) -> None:
        self.registries[chat_id] = registry


class _FakeSpanExtractor:
    def __init__(self, spans: list[tuple[str, float, str]]) -> None:
        self.spans = spans

    def extract(self, text: str) -> list[tuple[str, float, str]]:
        del text
        return list(self.spans)


def test_chat_retrieval_entity_span_seeds_ppr_and_absent_surface_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "CHAT_CONTEXT_NOTES", 3)
    monkeypatch.setattr(config, "CHAT_CONTEXT_FACTS", 0)
    index = _EntitySeedIndex(
        notes={
            "alpha": note("alpha", "Alpha"),
            "beta": note("beta", "Beta"),
            "gamma": note("gamma", "Gamma"),
        },
        entity_lookup={"Acme": 1},
    )

    retriever = ChatRetriever(
        index,  # type: ignore[arg-type]
        "chat-entity",
        span_extractor=_FakeSpanExtractor([("Acme", 0.75, "organization")]),
    )
    items, _facts = retriever.context_pack("Tell me about Acme", now=1_000.0)

    absent = ChatRetriever(
        index,  # type: ignore[arg-type]
        "chat-absent",
        span_extractor=_FakeSpanExtractor([("Missing", 0.75, "organization")]),
    )
    absent_items, absent_facts = absent.context_pack("Tell me about Missing", now=1_000.0)

    assert [item.note_id for item in items] == ["alpha"]
    assert retriever._grounding._seeds == {}
    assert absent_items == []
    assert absent_facts == []
