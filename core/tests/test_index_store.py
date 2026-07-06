from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core.schemas import GraphEdge


def test_upsert_note_updates_fts_aliases_and_cache(store):
    note = make_note("alpha", "Alpha Plan", "Body mentions invoice migration and rollout.")
    enrich = enrichment(
        "Alpha invoice rollout plan.",
        aliases=["Invoice rollout", "Alpha"],
        topics=["billing", "planning"],
        entities=["Stripe"],
    )
    store.cache_enrichment(note.note_id, note.content_hash, enrich)
    store.upsert_note(note, enrich, fixture_units(unit_vec(1), enrich))

    assert store.note("alpha") is not None
    assert store.bm25_search("invoice", 5) == [("alpha", store.bm25_search("invoice", 5)[0][1])]
    assert ("alpha plan", "alpha", True) in store.aliases()
    assert ("invoice rollout", "alpha", False) in store.aliases()
    assert store.cached_enrichment(note.note_id, note.content_hash) == enrich


def test_dense_search_uses_per_note_max_unit_score(store, monkeypatch):
    note_a = make_note("alpha", "Alpha", "Alpha text")
    note_b = make_note("beta", "Beta", "Beta text")
    enrich_a = enrichment("Alpha gist", topics=["alpha", "work"])
    enrich_b = enrichment("Beta gist", topics=["beta", "work"])
    query_vec = unit_vec(7)
    weaker = unit_vec(8)
    store.upsert_note(
        note_a,
        enrich_a,
        [("chunk", "weak alpha", weaker), ("gist", "strong alpha", query_vec)],
    )
    store.upsert_note(note_b, enrich_b, [("gist", "beta", unit_vec(9))])

    monkeypatch.setattr(
        "taurine_core.indexer.store.embed_texts",
        lambda texts: np.array([query_vec for _ in texts], dtype=np.float32),
    )

    top_note_id, top_score = store.dense_search("alpha query", 2)[0]
    assert top_note_id == "alpha"
    assert top_score == pytest.approx(1.0)


def test_delete_note_removes_fts_embeddings_aliases_and_edges(store):
    note = make_note("alpha", "Alpha", "Find me before delete")
    enrich = enrichment("Alpha gist", aliases=["alpha alias"], topics=["alpha", "work"])
    store.upsert_note(note, enrich, fixture_units(unit_vec(2), enrich))
    store.replace_edges([GraphEdge(src="alpha", dst="zeta", type="semantic", weight=0.7)])

    store.delete_note("alpha")

    assert store.note("alpha") is None
    assert store.bm25_search("delete", 5) == []
    assert store.dense_search("anything", 5) == []
    assert store.aliases() == []
    assert store.neighbors("alpha") == []


def test_best_chunk_returns_nearest_chunk_text(store, monkeypatch):
    note = make_note("alpha", "Alpha", "Alpha body")
    enrich = enrichment("Alpha gist", topics=["alpha", "work"])
    query_vec = unit_vec(12)
    store.upsert_note(
        note,
        enrich,
        [
            ("chunk", "unrelated chunk", unit_vec(13)),
            ("chunk", "matching chunk", query_vec),
            ("gist", "Alpha gist", unit_vec(14)),
        ],
    )
    monkeypatch.setattr(
        "taurine_core.indexer.store.embed_texts",
        lambda texts: np.array([query_vec for _ in texts], dtype=np.float32),
    )

    assert store.best_chunk("alpha", "matching query") == "matching chunk"


def test_doc_frequency_is_case_insensitive_cached_and_quotes_punctuation(store):
    alpha = make_note("alpha", "Alpha: Plan", "Launch alpha:plan details.")
    beta = make_note("beta", "Beta", "Alpha only.")
    enrich_alpha = enrichment("Alpha gist", topics=["alpha", "work"])
    enrich_beta = enrichment("Beta gist", topics=["beta", "work"])
    store.upsert_note(alpha, enrich_alpha, fixture_units(unit_vec(31), enrich_alpha))
    store.upsert_note(beta, enrich_beta, fixture_units(unit_vec(32), enrich_beta))

    assert store.doc_frequency("ALPHA") == 2
    assert store.doc_frequency("alpha") == 2
    assert store.doc_frequency("alpha:plan") == 1
