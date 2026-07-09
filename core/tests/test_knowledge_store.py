from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from taurine_core.indexer.knowledge_store import (
    KnowledgeSchemaError,
    KnowledgeStore,
)
from taurine_core.indexer.passages import PassageDraft
from taurine_core.schemas import NoteRecord


REQUIRED_TABLES = {
    "notes",
    "passages",
    "passages_fts",
    "passage_embeddings",
    "entities",
    "entity_aliases",
    "entity_mentions",
    "facts",
    "predicate_proposals",
    "fact_embeddings",
    "passage_semantic_edges",
    "passage_entity_edges",
    "note_projection_edges",
    "index_builds",
}


def test_fresh_store_creates_complete_schema_and_build_record(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        assert REQUIRED_TABLES <= store.table_names()
        store.begin_build(
            "build-1",
            source_snapshot="sha256:sources",
            model_manifest={"embedding": "fake"},
            config_manifest={"passages": "v1"},
            calibration_hash="sha256:calibration",
            started_at=10.0,
        )
        row = store.build("build-1")
        assert row is not None
        assert row.status == "building"
        assert row.health == "healthy"
        assert row.source_snapshot == "sha256:sources"
    finally:
        store.close()


def test_note_delete_cascades_derived_rows_and_prunes_orphan_entity(
    tmp_path: Path,
) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        note = _note()
        store.write_note(note, build_id="build-1", extraction_status="complete")
        store.write_passages(
            note.note_id,
            (_passage(),),
            created=note.created,
            modified=note.modified,
            build_id="build-1",
        )
        store.insert_entity(
            entity_id="entity-1",
            canonical="Maya",
            normalized="maya",
            etype="person",
            df=1,
            traversal_eligible=True,
            ambiguity_key="maya",
            build_id="build-1",
        )
        store.insert_alias("entity-1", "Maya", "extracted", 0.95)
        store.insert_mention(
            mention_id="mention-1",
            entity_id="entity-1",
            passage_id="passage-1",
            note_id=note.note_id,
            surface="Maya",
            etype="person",
            char_start=0,
            char_end=4,
            confidence=0.95,
            extractor="fake",
            schema_id="schema-1",
            build_id="build-1",
        )

        store.delete_note(note.note_id)

        assert store.counts()["notes"] == 0
        assert store.counts()["passages"] == 0
        assert store.counts()["entity_mentions"] == 0
        assert store.counts()["entities"] == 0
        assert store.search_passages_bm25("Maya", 5) == ()
    finally:
        store.close()


def test_invariants_report_orphan_entities_and_prune_repairs_them(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        store.insert_entity(
            entity_id="orphan",
            canonical="Unused",
            normalized="unused",
            etype="concept",
            df=0,
            traversal_eligible=False,
            ambiguity_key="unused",
            build_id="build-1",
        )

        report = store.validate_invariants()
        assert report.ok is False
        assert report.orphan_entities == 1

        assert store.prune_orphan_entities() == 1
        assert store.validate_invariants().ok is True
    finally:
        store.close()


def test_read_only_store_reads_but_rejects_writes(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.db"
    writable = KnowledgeStore(path)
    writable.write_note(_note(), build_id="build-1", extraction_status="pending")
    writable.close()

    store = KnowledgeStore(path, read_only=True)
    try:
        assert [row.note_id for row in store.source_notes()] == ["note-1"]
        with pytest.raises(PermissionError, match="read-only"):
            store.write_note(_note(), build_id="build-2", extraction_status="complete")
    finally:
        store.close()


def test_legacy_schema_is_rejected_instead_of_migrated(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE notes(note_id TEXT PRIMARY KEY, text TEXT NOT NULL)")
    connection.commit()
    connection.close()

    with pytest.raises(KnowledgeSchemaError, match="delete and regenerate"):
        KnowledgeStore(path)

    columns = {
        row[1]
        for row in sqlite3.connect(path).execute("PRAGMA table_info(notes)").fetchall()
    }
    assert columns == {"note_id", "text"}


def _note() -> NoteRecord:
    return NoteRecord(
        note_id="note-1",
        title="Launch",
        source="apple_notes",
        created=1.0,
        modified=2.0,
        content_hash="note-hash",
        text="Maya owns the launch checklist.",
    )


def _passage() -> PassageDraft:
    return PassageDraft(
        passage_id="passage-1",
        ordinal=0,
        char_start=0,
        char_end=31,
        token_count=5,
        text="Maya owns the launch checklist.",
        content_hash="passage-hash",
    )
