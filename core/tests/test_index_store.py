from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from tests.conftest import enrichment, fixture_units, make_note, shifted_vec, unit_vec
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


def test_best_chunk_considers_utterance_units_and_prefixes_spoken_reference(store, monkeypatch):
    note = make_note("alpha", "Alpha", "Alpha body")
    enrich = enrichment("Alpha gist", utterances=["matching spoken sentence"])
    query_vec = unit_vec(15)
    store.upsert_note(
        note,
        enrich,
        [
            ("chunk", "unrelated chunk", unit_vec(16)),
            ("gist", "unrelated gist", unit_vec(17)),
            ("utterance", "matching spoken sentence", query_vec),
        ],
    )
    monkeypatch.setattr(
        "taurine_core.indexer.store.embed_texts",
        lambda texts: np.array([query_vec for _ in texts], dtype=np.float32),
    )

    assert (
        store.best_chunk("alpha", "matching query")
        == "(spoken reference) matching spoken sentence"
    )


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


def test_note_source_round_trips_counts_and_graph_nodes(store):
    obsidian = make_note("obsidian:vault/alpha.md", "Alpha", "Alpha body").model_copy(
        update={"source": "obsidian"}
    )
    apple = make_note("x-coredata://beta", "Beta", "Beta body").model_copy(
        update={"source": "apple_notes"}
    )
    enrich_obsidian = enrichment("Alpha gist", topics=["alpha", "work"])
    enrich_apple = enrichment("Beta gist", topics=["beta", "work"])

    store.upsert_note(obsidian, enrich_obsidian, fixture_units(unit_vec(41), enrich_obsidian))
    store.upsert_note(apple, enrich_apple, fixture_units(unit_vec(42), enrich_apple))

    assert store.note(obsidian.note_id).source == "obsidian"
    assert {note.note_id: note.source for note in store.notes()} == {
        obsidian.note_id: "obsidian",
        apple.note_id: "apple_notes",
    }
    assert store.note_count_by_source() == {"apple_notes": 1, "obsidian": 1}
    assert {node.note_id: node.source for node in store.graph().nodes} == {
        obsidian.note_id: "obsidian",
        apple.note_id: "apple_notes",
    }


def test_store_migrates_existing_notes_without_source_column(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE notes (
            note_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            folder TEXT NOT NULL,
            modified REAL NOT NULL,
            content_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            gist TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO notes(note_id, title, folder, modified, content_hash, text, gist)
        VALUES ('legacy', 'Legacy', '', 1.0, 'hash', 'Legacy text', 'Legacy gist')
        """
    )
    conn.commit()
    conn.close()

    from taurine_core.indexer.store import Index

    index = Index(db_path)

    columns = {
        row["name"] for row in index._conn.execute("PRAGMA table_info(notes)").fetchall()
    }
    assert "source" in columns
    assert index.note("legacy").source == "apple_notes"
    assert index.note_count_by_source() == {"apple_notes": 1}


def test_preview_resolves_neighbors_sorted_by_weight(store):
    alpha = make_note("alpha", "Alpha", "Alpha body").model_copy(update={"source": "obsidian"})
    beta = make_note("beta", "Beta", "Beta body").model_copy(update={"source": "apple_notes"})
    gamma = make_note("gamma", "Gamma", "Gamma body").model_copy(update={"source": "codex_mem"})
    enrich_alpha = enrichment("Alpha gist", topics=["alpha", "work"])
    enrich_beta = enrichment("Beta gist", topics=["beta", "work"])
    enrich_gamma = enrichment("Gamma gist", topics=["gamma", "work"])
    store.upsert_note(alpha, enrich_alpha, fixture_units(unit_vec(51), enrich_alpha))
    store.upsert_note(beta, enrich_beta, fixture_units(unit_vec(52), enrich_beta))
    store.upsert_note(gamma, enrich_gamma, fixture_units(unit_vec(53), enrich_gamma))
    store.replace_edges(
        [
            GraphEdge(src="alpha", dst="beta", type="topic", weight=0.4),
            GraphEdge(src="alpha", dst="gamma", type="semantic", weight=0.8),
        ]
    )

    preview = store.preview("alpha")

    assert preview is not None
    assert preview.title == "Alpha"
    assert preview.source == "obsidian"
    assert preview.gist == "Alpha gist"
    assert preview.text == "Alpha body"
    assert [neighbor.note_id for neighbor in preview.neighbors] == ["gamma", "beta"]
    assert [neighbor.title for neighbor in preview.neighbors] == ["Gamma", "Beta"]
    assert [neighbor.source for neighbor in preview.neighbors] == ["codex_mem", "apple_notes"]
    assert [neighbor.weight for neighbor in preview.neighbors] == [pytest.approx(0.8), pytest.approx(0.4)]
    assert store.preview("missing") is None


def test_disabled_source_drops_out_of_active_index_but_keeps_rows(store):
    kept = make_note("keep:one", "Kept Note", "Shared topic body about billing").model_copy(
        update={"source": "keep"}
    )
    gone = make_note("gone:one", "Gone Note", "Shared topic body about billing").model_copy(
        update={"source": "gone"}
    )
    enrich = enrichment("Shared billing topic.", aliases=["billing sync"])
    store.upsert_note(kept, enrich, fixture_units(unit_vec(101), enrich))
    store.upsert_note(
        gone, enrich, fixture_units(shifted_vec(unit_vec(101), 202, noise=0.02), enrich)
    )
    store.rebuild_graph()

    store.set_source_enabled("gone", False)

    assert store.disabled_sources() == {"gone"}
    graph = store.graph()
    assert {node.note_id for node in graph.nodes} == {"keep:one"}
    assert all("gone:one" not in (edge.src, edge.dst) for edge in graph.edges)
    assert all(note_id != "gone:one" for note_id, _ in store.dense_search("billing topic", 10))
    assert all(note_id != "gone:one" for note_id, _ in store.bm25_search("billing", 10))
    assert all(note_id != "gone:one" for _, note_id, _ in store.aliases())
    assert store.status()[0] == 1
    # rows stay stored: counts by source and the note itself survive
    assert store.note_count_by_source() == {"keep": 1, "gone": 1}
    assert store.note("gone:one") is not None

    store.set_source_enabled("gone", True)
    assert store.disabled_sources() == set()
    assert {node.note_id for node in store.graph().nodes} == {"keep:one", "gone:one"}
    assert store.status()[0] == 2
