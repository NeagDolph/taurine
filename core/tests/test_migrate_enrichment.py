from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import make_note, unit_vec
from taurine_core import config
from taurine_core.indexer.store import Index
from taurine_core.schemas import Enrichment, EntityMention, Extraction, RelationTriple


def _install_old_shape_fixture(db_path: Path) -> None:
    index = Index(db_path)
    notes = [
        make_note(
            "alpha",
            "Alpha Launch",
            "\n  Alpha launch display line\nThe body mentions Jane Doe and Taurine.",
            modified=10.0,
        ),
        make_note(
            "beta",
            "Beta Followup",
            "Beta first line\nA note that will fail extraction.",
            modified=20.0,
        ),
    ]
    for offset, note in enumerate(notes, start=1):
        vector = unit_vec(offset)
        index.upsert_note(
            note,
            Enrichment(gist=f"LLM era gist {note.note_id}"),
            [
                ("chunk", f"chunk {note.note_id}", vector),
                ("gist", f"old gist {note.note_id}", vector),
                ("utterance", f"old utterance {note.note_id}", vector),
            ],
        )
    with index._conn:
        index._conn.execute(
            "UPDATE notes SET gist = 'LLM-generated summary for alpha' WHERE note_id = 'alpha'"
        )
        index._conn.execute(
            "UPDATE notes SET gist = 'LLM-generated summary for beta' WHERE note_id = 'beta'"
        )
    index.close()


def _rows(db_path: Path, sql: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(db_path)
    try:
        return [tuple(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _snapshot(db_path: Path) -> dict[str, list[tuple[Any, ...]]]:
    return {
        "notes": _rows(
            db_path,
            "SELECT note_id, title, folder, modified, created, content_hash, text, gist, source, quality_rank FROM notes ORDER BY note_id",
        ),
        "embeddings": _rows(
            db_path,
            "SELECT note_id, kind, text, vec FROM embeddings ORDER BY note_id, kind, text",
        ),
        "entities": _rows(db_path, "SELECT canonical, etype, df FROM entities ORDER BY canonical"),
        "note_entities": _rows(
            db_path,
            "SELECT note_id, entity_id, conf, count FROM note_entities ORDER BY note_id, entity_id",
        ),
        "relations": _rows(
            db_path,
            "SELECT note_id, src_entity, rel_type, dst_entity, conf FROM entity_relations ORDER BY note_id, src_entity, rel_type, dst_entity",
        ),
        "cache": _rows(
            db_path,
            "SELECT note_id, content_hash, enrichment_json FROM enrich_cache ORDER BY note_id",
        ),
        "edges": _rows(db_path, "SELECT src, dst, type, weight, fact FROM edges ORDER BY src, dst"),
        "judgments": _rows(
            db_path,
            "SELECT src, dst, sig, related, fact FROM edge_judgments ORDER BY src, dst",
        ),
    }


def _patch_non_extraction_work(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import migrate_enrichment

    calls: list[str] = []

    def fake_rebuild_graph(index: Index) -> None:
        calls.append("rebuild")
        assert all(kind != "gist" for _row_id, _note_id, kind, _text in index._embedding_rows)

    monkeypatch.setattr(migrate_enrichment.Index, "rebuild_graph", fake_rebuild_graph)
    monkeypatch.setattr(
        migrate_enrichment.rank,
        "compute_quality_ranks",
        lambda _index: calls.append("quality") or {},
    )
    monkeypatch.setattr(
        migrate_enrichment.rank,
        "compute_value_scores",
        lambda _index: calls.append("value") or {},
    )
    monkeypatch.setattr(
        migrate_enrichment.rank,
        "refresh_fact_embeddings",
        lambda _index: calls.append("facts"),
    )
    return calls


def test_migrate_deletes_gist_embeddings_recomputes_display_lines_and_backfills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import migrate_enrichment

    db_path = tmp_path / "old-shape.db"
    _install_old_shape_fixture(db_path)
    calls = _patch_non_extraction_work(monkeypatch)

    def fake_extract_notes(notes):
        assert [note.note_id for note in notes] == ["alpha", "beta"]
        return {
            "alpha": Extraction(
                entities=[
                    EntityMention(surface="Jane Doe", etype="person", conf=0.91, mentions=1),
                    EntityMention(surface="Taurine", etype="project", conf=0.87, mentions=1),
                ],
                relations=[
                    RelationTriple(
                        head="Jane Doe",
                        rel_type="works_at",
                        tail="Taurine",
                        conf=0.76,
                    )
                ],
            ),
            "beta": Extraction(failed=True),
        }

    monkeypatch.setattr(config, "GLINER_INGEST_BATCH", 10)
    monkeypatch.setattr(migrate_enrichment.extract, "extract_notes", fake_extract_notes)

    report = migrate_enrichment.migrate(db_path, llm=None, dry_run=False)

    assert report == migrate_enrichment.MigrationReport(
        gist_rows_deleted=2,
        utterance_rows_deleted=0,
        notes_backfilled=1,
        extraction_failures=1,
        judgments_written=0,
    )
    assert calls == ["rebuild", "quality", "value"]
    assert _rows(db_path, "SELECT kind, COUNT(*) FROM embeddings GROUP BY kind ORDER BY kind") == [
        ("chunk", 2),
        ("utterance", 2),
    ]
    assert _rows(db_path, "SELECT note_id, gist FROM notes ORDER BY note_id") == [
        ("alpha", "Alpha launch display line"),
        ("beta", "Beta first line"),
    ]

    index = Index(db_path)
    assert [(canonical, etype, df) for _id, canonical, etype, df in index.entity_rows()] == [
        ("Jane Doe", "person", 1),
        ("Taurine", "project", 1),
    ]
    entities_by_name = {
        canonical: entity_id for entity_id, canonical, _etype, _df in index.entity_rows()
    }
    assert index.note_entity_rows() == [
        ("alpha", entities_by_name["Jane Doe"], pytest.approx(0.91), 1),
        ("alpha", entities_by_name["Taurine"], pytest.approx(0.87), 1),
    ]
    assert index.relation_rows() == [
        (
            "alpha",
            entities_by_name["Jane Doe"],
            "works_at",
            entities_by_name["Taurine"],
            pytest.approx(0.76),
        )
    ]
    cached = index.cached_extraction("alpha", index.note("alpha").content_hash)  # type: ignore[union-attr]
    assert cached is not None
    assert cached.entities[0].surface == "Jane Doe"
    assert index.cached_extraction("beta", index.note("beta").content_hash) is None  # type: ignore[union-attr]


def test_dry_run_mutates_nothing_and_reports_read_only_estimates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import migrate_enrichment

    db_path = tmp_path / "dry-run.db"
    _install_old_shape_fixture(db_path)
    before = _snapshot(db_path)

    monkeypatch.setattr(
        migrate_enrichment.extract,
        "extract_notes",
        lambda _notes: pytest.fail("dry-run should not run GLiNER extraction"),
    )
    monkeypatch.setattr(
        migrate_enrichment.Index,
        "rebuild_graph",
        lambda _index: pytest.fail("dry-run should not rebuild the graph"),
    )
    monkeypatch.setattr(
        migrate_enrichment.rank,
        "compute_quality_ranks",
        lambda _index: pytest.fail("dry-run should not write quality ranks"),
    )

    report = migrate_enrichment.migrate(db_path, llm=None, dry_run=True)

    assert report == migrate_enrichment.MigrationReport(
        gist_rows_deleted=2,
        utterance_rows_deleted=0,
        notes_backfilled=2,
        extraction_failures=0,
        judgments_written=0,
    )
    assert _snapshot(db_path) == before


def test_dry_run_does_not_initialize_missing_schema_tables(tmp_path: Path) -> None:
    import migrate_enrichment

    db_path = tmp_path / "legacy-dry-run.db"
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
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            vec BLOB NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO notes(note_id, title, folder, modified, content_hash, text, gist)
        VALUES ('legacy', 'Legacy', '', 1.0, 'hash', 'Legacy body', 'LLM legacy gist')
        """
    )
    conn.execute(
        "INSERT INTO embeddings(note_id, kind, text, vec) VALUES (?, ?, ?, ?)",
        ("legacy", "gist", "old gist", unit_vec(9).tobytes()),
    )
    conn.commit()
    conn.close()
    before_tables = _rows(
        db_path,
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual') ORDER BY name",
    )
    before_columns = _rows(db_path, "PRAGMA table_info(notes)")

    report = migrate_enrichment.migrate(db_path, llm=None, dry_run=True)

    assert report == migrate_enrichment.MigrationReport(
        gist_rows_deleted=1,
        utterance_rows_deleted=0,
        notes_backfilled=1,
        extraction_failures=0,
        judgments_written=0,
    )
    assert (
        _rows(
            db_path,
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual') ORDER BY name",
        )
        == before_tables
    )
    assert _rows(db_path, "PRAGMA table_info(notes)") == before_columns


def test_migrate_deletes_utterance_embeddings_when_slim_utterances_are_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import migrate_enrichment

    db_path = tmp_path / "no-slim.db"
    _install_old_shape_fixture(db_path)
    _patch_non_extraction_work(monkeypatch)
    monkeypatch.setattr(config, "SLIM_UTTERANCES", False)
    monkeypatch.setattr(
        migrate_enrichment.extract,
        "extract_notes",
        lambda notes: {note.note_id: Extraction() for note in notes},
    )

    report = migrate_enrichment.migrate(db_path, llm=None, dry_run=False)

    assert report.gist_rows_deleted == 2
    assert report.utterance_rows_deleted == 2
    assert _rows(db_path, "SELECT kind, COUNT(*) FROM embeddings GROUP BY kind ORDER BY kind") == [
        ("chunk", 2)
    ]


def test_llm_step_runs_only_with_llm_and_counts_new_judgments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import migrate_enrichment

    db_path = tmp_path / "llm.db"
    _install_old_shape_fixture(db_path)
    calls = _patch_non_extraction_work(monkeypatch)
    monkeypatch.setattr(
        migrate_enrichment.extract,
        "extract_notes",
        lambda notes: {note.note_id: Extraction() for note in notes},
    )

    async def fake_adjudicate_graph(index: Index, llm: object, progress=None):
        assert llm is fake_llm
        if progress is not None:
            progress(1, 1)
        index.set_edge_judgment("alpha", "beta", "sig-1", True, "Alpha and beta are related.")
        return object()

    fake_llm = object()
    monkeypatch.setattr(
        migrate_enrichment.adjudicate,
        "adjudicate_graph",
        fake_adjudicate_graph,
    )

    report = migrate_enrichment.migrate(db_path, llm=fake_llm, dry_run=False)

    assert report.judgments_written == 1
    assert calls == ["rebuild", "quality", "value", "facts"]
    assert _rows(db_path, "SELECT src, dst, sig, related, fact FROM edge_judgments") == [
        ("alpha", "beta", "sig-1", 1, "Alpha and beta are related.")
    ]
