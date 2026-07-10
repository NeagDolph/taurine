from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from tests.conftest import make_note, unit_vec
from taurine_core import config


def _seed_alias(store, alias: str, note_id: str, *, is_title: bool = False) -> None:
    with store._conn:
        store._conn.execute(
            "INSERT OR REPLACE INTO aliases(alias_lowercase, note_id, is_title) VALUES (?, ?, ?)",
            (alias, note_id, int(is_title)),
        )


def _seed_embedding(store, note_id: str, text: str, seed: int) -> None:
    vector = unit_vec(seed).astype(np.float32)
    with store._conn:
        store._conn.execute(
            "INSERT INTO embeddings(note_id, kind, text, vec) VALUES (?, 'chunk', ?, ?)",
            (note_id, text, vector.tobytes()),
        )


def _seed_edge(store, src: str, dst: str, edge_type: str, weight: float) -> None:
    with store._conn:
        store._conn.execute(
            """
            INSERT OR REPLACE INTO edges(src, dst, type, weight, shared_json, fact, fact_provenance)
            VALUES (?, ?, ?, ?, '[]', '', '')
            """,
            (src, dst, edge_type, float(weight)),
        )


def test_replace_source_records_uses_clamped_first_non_empty_note_line_as_display_line(store):
    note = make_note(
        "alpha",
        "Alpha Title",
        "\n\n" + ("alpha " * 80) + "\nsecond line",
    )

    store.replace_source_records("apple_notes", [note])

    stored = store.note("alpha")
    assert stored.gist.startswith("alpha alpha")
    assert len(stored.gist) <= config.GIST_MAX_CHARS
    assert not stored.gist.endswith(" ")


def test_delete_note_removes_fts_embeddings_aliases_edges_and_recounts_entity_df(store):
    note = make_note("alpha", "Alpha", "Find me before delete")
    other = make_note("beta", "Beta", "Beta body")
    store.replace_source_records("apple_notes", [note, other])
    _seed_alias(store, "alpha alias", "alpha")
    _seed_embedding(store, "alpha", "alpha chunk", 2)
    _seed_edge(store, "alpha", "beta", "semantic", 0.7)
    with store._conn:
        store._conn.execute(
            "INSERT INTO entities(entity_id, canonical, etype, df) VALUES (1, 'Acme', 'ORG', 2)"
        )
        store._conn.execute(
            "INSERT INTO note_entities(note_id, entity_id, conf, count) VALUES ('alpha', 1, 0.9, 1)"
        )
        store._conn.execute(
            "INSERT INTO note_entities(note_id, entity_id, conf, count) VALUES ('beta', 1, 0.8, 1)"
        )

    store.delete_note("alpha")

    assert store.note("alpha") is None
    assert (
        store._conn.execute(
            "SELECT COUNT(*) AS c FROM notes_fts WHERE note_id = 'alpha'"
        ).fetchone()["c"]
        == 0
    )
    assert (
        store._conn.execute(
            "SELECT COUNT(*) AS c FROM embeddings WHERE note_id = 'alpha'"
        ).fetchone()["c"]
        == 0
    )
    assert store.aliases() == []
    assert all(
        "alpha" not in (edge.src, edge.dst) for edge in store.graph().edges
    )
    # entity df is recomputed from the surviving provenance rows
    assert (
        store._conn.execute("SELECT df FROM entities WHERE entity_id = 1").fetchone()["df"]
        == 1
    )


def test_note_source_round_trips_counts_and_graph_nodes(store):
    obsidian = make_note("obsidian:vault/alpha.md", "Alpha", "Alpha body")
    apple = make_note("x-coredata://beta", "Beta", "Beta body")

    store.replace_source_records("obsidian", [obsidian])
    store.replace_source_records("apple_notes", [apple])

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


def test_note_created_round_trips_and_zero_falls_back_to_modified(store):
    explicit = make_note("explicit", "Explicit", "Explicit body", modified=200.0).model_copy(
        update={"created": 100.0}
    )
    fallback = make_note("fallback", "Fallback", "Fallback body", modified=300.0)

    store.replace_source_records("apple_notes", [explicit, fallback])

    assert store.note("explicit").created == pytest.approx(100.0)
    assert store.note("fallback").created == pytest.approx(300.0)


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
    assert "created" in columns
    assert "quality_rank" in columns
    assert index.note("legacy").source == "apple_notes"
    assert index.note("legacy").created == pytest.approx(1.0)
    assert index.note("legacy").quality_rank == pytest.approx(0.0)
    assert index.note_count_by_source() == {"apple_notes": 1}


def test_entity_schema_migration_is_idempotent_for_existing_db(tmp_path) -> None:
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

    first = Index(db_path)
    first.close()
    second = Index(db_path)

    note_columns = {
        row["name"] for row in second._conn.execute("PRAGMA table_info(notes)").fetchall()
    }
    entity_tables = {
        row["name"]
        for row in second._conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name IN ('entities', 'note_entities', 'entity_relations')
            """
        ).fetchall()
    }

    assert {"ingest_salience", "value_score"}.issubset(note_columns)
    assert entity_tables == {"entities", "note_entities", "entity_relations"}


def test_store_migrates_legacy_chat_messages_columns(tmp_path):
    db_path = tmp_path / "legacy-chats.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chats (
            chat_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created REAL NOT NULL,
            updated REAL NOT NULL,
            registry_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content TEXT NOT NULL,
            thinking TEXT NOT NULL DEFAULT '',
            sources_json TEXT NOT NULL DEFAULT '[]',
            facts_json TEXT NOT NULL DEFAULT '[]',
            activity_json TEXT NOT NULL DEFAULT '[]',
            usage_json TEXT NOT NULL DEFAULT '{}',
            cancelled INTEGER NOT NULL DEFAULT 0,
            created REAL NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO chats(chat_id, created, updated) VALUES ('chat-legacy', 1.0, 1.0)"
    )
    conn.execute(
        """
        INSERT INTO chat_messages(chat_id, role, content, created)
        VALUES ('chat-legacy', 'user', 'hello', 1.0)
        """
    )
    conn.commit()
    conn.close()

    from taurine_core.indexer.store import Index

    index = Index(db_path)

    columns = {
        row["name"]
        for row in index._conn.execute("PRAGMA table_info(chat_messages)").fetchall()
    }
    assert {
        "evidence_json",
        "retrieved_ids_json",
        "consulted_ids_json",
        "cited_ids_json",
        "original_query",
        "rewritten_query",
        "retrieval_decision",
        "retrieval_confidence",
        "active_build_id",
    } <= columns
    messages = index.chat_messages("chat-legacy")
    assert len(messages) == 1
    assert messages[0].content == "hello"
    assert messages[0].evidence is None


def test_preview_resolves_neighbors_sorted_by_weight(store):
    alpha = make_note("alpha", "Alpha", "Alpha body")
    beta = make_note("beta", "Beta", "Beta body")
    gamma = make_note("gamma", "Gamma", "Gamma body")
    store.replace_source_records("obsidian", [alpha])
    store.replace_source_records("apple_notes", [beta])
    store.replace_source_records("codex_mem", [gamma])
    _seed_edge(store, "alpha", "beta", "topic", 0.4)
    _seed_edge(store, "alpha", "gamma", "semantic", 0.8)

    preview = store.preview("alpha")

    assert preview is not None
    assert preview.title == "Alpha"
    assert preview.source == "obsidian"
    assert preview.gist == "Alpha body"
    assert preview.text == "Alpha body"
    assert [neighbor.note_id for neighbor in preview.neighbors] == ["gamma", "beta"]
    assert [neighbor.title for neighbor in preview.neighbors] == ["Gamma", "Beta"]
    assert [neighbor.source for neighbor in preview.neighbors] == ["codex_mem", "apple_notes"]
    assert [neighbor.weight for neighbor in preview.neighbors] == [
        pytest.approx(0.8),
        pytest.approx(0.4),
    ]
    assert store.preview("missing") is None


def test_disabled_source_drops_out_of_active_index_but_keeps_rows(store):
    kept = make_note("keep:one", "Kept Note", "Shared topic body about billing")
    gone = make_note("gone:one", "Gone Note", "Shared topic body about billing")
    store.replace_source_records("keep", [kept])
    store.replace_source_records("gone", [gone])
    _seed_alias(store, "kept note", "keep:one", is_title=True)
    _seed_alias(store, "gone note", "gone:one", is_title=True)
    _seed_edge(store, "keep:one", "gone:one", "semantic", 0.9)

    store.set_source_enabled("gone", False)

    assert store.disabled_sources() == {"gone"}
    graph = store.graph()
    assert {node.note_id for node in graph.nodes} == {"keep:one"}
    assert all("gone:one" not in (edge.src, edge.dst) for edge in graph.edges)
    assert all(note_id != "gone:one" for _, note_id, _ in store.aliases())
    assert store.status()[0] == 1
    # rows stay stored: counts by source and the note itself survive
    assert store.note_count_by_source() == {"keep": 1, "gone": 1}
    assert store.note("gone:one") is not None

    store.set_source_enabled("gone", True)
    assert store.disabled_sources() == set()
    assert {node.note_id for node in store.graph().nodes} == {"keep:one", "gone:one"}
    assert store.status()[0] == 2
