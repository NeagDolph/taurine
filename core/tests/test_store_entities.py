from __future__ import annotations

import sqlite3

import pytest

from tests.conftest import entity, enrichment, fixture_units, make_note, unit_vec
from taurine_core.schemas import Enrichment


def _entities_by_key(store) -> dict[tuple[str, str], tuple[int, int]]:
    return {
        (canonical, etype): (entity_id, df)
        for entity_id, canonical, etype, df in store.entity_rows()
    }


def _install_note(store, note_id: str, seed: int) -> None:
    enrich = enrichment(f"{note_id} gist", topics=["store"])
    store.upsert_note(
        make_note(note_id, note_id.title(), f"{note_id} body"),
        enrich,
        fixture_units(unit_vec(seed), enrich),
    )


def test_replace_note_extraction_round_trips_entities_and_relations(store) -> None:
    store.replace_note_extraction(
        "alpha",
        [
            ("Jane Doe", "PERSON", 0.91, 2),
            ("Acme", "ORG", 0.87, 1),
        ],
        [("Jane Doe", "works_at", "Acme", 0.76)],
    )

    entities = _entities_by_key(store)
    jane_id, jane_df = entities[("Jane Doe", "PERSON")]
    acme_id, acme_df = entities[("Acme", "ORG")]

    assert jane_df == 1
    assert acme_df == 1
    assert store.note_entity_rows() == [
        ("alpha", jane_id, pytest.approx(0.91), 2),
        ("alpha", acme_id, pytest.approx(0.87), 1),
    ]
    assert store.relation_rows() == [
        ("alpha", jane_id, "works_at", acme_id, pytest.approx(0.76))
    ]


def test_replace_note_extraction_recounts_df_on_replace_and_delete(store) -> None:
    store.replace_note_extraction(
        "alpha",
        [
            ("Acme", "ORG", 0.9, 1),
            ("Roadmap", "TOPIC", 0.8, 1),
        ],
        [],
    )
    store.replace_note_extraction("beta", [("Acme", "ORG", 0.7, 1)], [])

    assert _entities_by_key(store)[("Acme", "ORG")][1] == 2
    assert _entities_by_key(store)[("Roadmap", "TOPIC")][1] == 1

    store.replace_note_extraction("alpha", [("Jane Doe", "PERSON", 0.95, 1)], [])

    entities = _entities_by_key(store)
    assert entities[("Acme", "ORG")][1] == 1
    assert entities[("Roadmap", "TOPIC")][1] == 0
    assert entities[("Jane Doe", "PERSON")][1] == 1

    store.delete_note("beta")

    assert _entities_by_key(store)[("Acme", "ORG")][1] == 0
    assert store.note_entity_rows() == [("alpha", entities[("Jane Doe", "PERSON")][0], 0.95, 1)]


def test_relations_between_returns_max_conf_in_both_note_orientations(store) -> None:
    extraction_entities = [
        ("Jane Doe", "PERSON", 0.9, 1),
        ("Acme", "ORG", 0.8, 1),
    ]
    store.replace_note_extraction(
        "alpha",
        extraction_entities,
        [("Jane Doe", "works_at", "Acme", 0.42)],
    )
    store.replace_note_extraction(
        "beta",
        extraction_entities,
        [("Jane Doe", "works_at", "Acme", 0.88)],
    )

    assert store.relations_between("alpha", "beta") == [
        ("Jane Doe", "works_at", "Acme", pytest.approx(0.88))
    ]
    assert store.relations_between("beta", "alpha") == [
        ("Jane Doe", "works_at", "Acme", pytest.approx(0.88))
    ]


def test_relations_between_scoped_requires_relation_provenance_in_pair(store) -> None:
    extraction_entities = [
        ("Jane Doe", "PERSON", 0.9, 1),
        ("Acme", "ORG", 0.8, 1),
    ]
    store.replace_note_extraction("alpha", extraction_entities, [])
    store.replace_note_extraction("beta", extraction_entities, [])
    store.replace_note_extraction(
        "gamma",
        extraction_entities,
        [("Jane Doe", "works_at", "Acme", 0.88)],
    )

    assert store.relations_between("alpha", "beta") == [
        ("Jane Doe", "works_at", "Acme", pytest.approx(0.88))
    ]
    assert store.relations_between_scoped("alpha", "beta") == []

    store.replace_note_extraction(
        "alpha",
        extraction_entities,
        [("Jane Doe", "works_at", "Acme", 0.77)],
    )

    assert store.relations_between_scoped("alpha", "beta") == [
        ("Jane Doe", "works_at", "Acme", pytest.approx(0.77))
    ]
    assert store.relations_between_scoped("beta", "alpha") == [
        ("Jane Doe", "works_at", "Acme", pytest.approx(0.77))
    ]


def test_entity_surfaces_for_orders_by_confidence_then_entity_id_and_limits(store) -> None:
    enrich = Enrichment(
        gist="Entity gist",
        entities=[
            entity("Beta", "concept", conf=0.80),
            entity("Alpha", "concept", conf=0.95),
            entity("Gamma", "concept", conf=0.95),
            entity("Delta", "concept", conf=0.70),
        ],
    )
    store.upsert_note(
        make_note("entity-note", "Entity Note", "Entity body"),
        enrich,
        fixture_units(unit_vec(12), enrich),
    )

    assert store.entity_surfaces_for("entity-note", limit=2) == ["Alpha", "Gamma"]
    assert store.entity_surfaces_for("entity-note") == ["Alpha", "Gamma", "Beta"]
    assert store.entity_surfaces_for("missing") == []


def test_ingest_salience_and_value_scores_round_trip(store) -> None:
    _install_note(store, "alpha", 10)
    _install_note(store, "beta", 11)

    store.set_ingest_salience({"alpha": 1.75, "missing": 9.0})
    store.set_value_scores({"alpha": 0.25, "missing": 9.0})

    salience_rows = store._conn.execute(
        "SELECT note_id, ingest_salience FROM notes ORDER BY note_id"
    ).fetchall()
    assert {row["note_id"]: float(row["ingest_salience"]) for row in salience_rows} == {
        "alpha": pytest.approx(1.75),
        "beta": pytest.approx(1.0),
    }
    assert store.value_scores() == {
        "alpha": pytest.approx(0.25),
        "beta": pytest.approx(0.0),
    }

    store.set_value_scores({"beta": 0.8})

    assert store.value_scores() == {
        "alpha": pytest.approx(0.0),
        "beta": pytest.approx(0.8),
    }


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
    assert second.value_scores() == {"legacy": pytest.approx(0.0)}
