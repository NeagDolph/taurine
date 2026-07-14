from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from taurine_core.evaluation import quality_bench
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.schemas import ExtractedRelation, Extraction, NoteRecord


def _note(note_id: str, text: str) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=f"Note {note_id}",
        modified=10.0,
        created=5.0,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
        source="apple_notes",
    )


def _relation(
    head: str, predicate: str, tail: str, start: int, end: int
) -> ExtractedRelation:
    return ExtractedRelation(
        head=head,
        predicate=predicate,
        tail=tail,
        confidence=0.9,
        subject_start=start,
        subject_end=start + len(head),
        object_start=end - len(tail),
        object_end=end,
        assertion_start=start,
        assertion_end=end,
        extraction_method="fixture",
        schema_id="fixture-v1",
        status="active",
    )


def _fixture_knowledge_db(path: Path) -> None:
    store = KnowledgeStore(path)
    try:
        store._conn.executescript(
            """
            INSERT INTO index_builds(
                build_id, started_at, completed_at, status, health,
                source_snapshot, model_manifest, config_manifest,
                calibration_hash, graph_hash
            ) VALUES (
                'build-fixture', 1, 2, 'complete', 'healthy', 'snapshot',
                '{}', '{}', 'calibration', 'graph'
            );
            INSERT INTO entities(
                entity_id, canonical, normalized, etype, df,
                traversal_eligible, traversal_reason, ambiguity_key, build_id
            ) VALUES
                ('e-alice', 'Alice Smith', 'alice smith', 'person', 1, 1, '', '', 'build-fixture'),
                ('e-acme', 'Acme Corporation', 'acme corporation', 'organization', 1, 1, '', '', 'build-fixture'),
                ('e-bob', 'Bob', 'bob', 'person', 1, 1, '', '', 'build-fixture');
            INSERT INTO entity_aliases(entity_id, surface, surface_cf, origin, confidence) VALUES
                ('e-alice', 'Alice', 'alice', 'resolved', 1),
                ('e-acme', 'Acme', 'acme', 'resolved', 1),
                ('e-bob', 'Bob', 'bob', 'resolved', 1);
            """
        )
        notes = [
            _note("n1", "Alice works at Acme."),
            _note("n2", "Bob uses Taurine."),
            _note("n3", "No fact three."),
            _note("n4", "No fact four."),
            _note("n5", "No fact five."),
            _note("n6", "No fact six."),
        ]
        for note in notes:
            store.write_note(
                note,
                build_id="build-fixture",
                extraction_status="complete",
                extraction_error="",
                salience=1.0,
            )
        store._conn.executescript(
            """
            INSERT INTO passages(
                passage_id, note_id, ordinal, char_start, char_end, token_count,
                text, content_hash, created, modified, build_id
            ) VALUES
                ('p1', 'n1', 0, 0, 20, 4, 'Alice works at Acme.', 'p1', 5, 10, 'build-fixture');
            INSERT INTO facts(
                fact_id, subject_entity_id, predicate, object_entity_id,
                source_passage_id, source_note_id, assertion_start, assertion_end,
                subject_start, subject_end, object_start, object_end, confidence,
                extraction_method, schema_id, observed_at, status, build_id,
                adjudication_score, adjudication_quote, adjudicated
            ) VALUES (
                'f1', 'e-alice', 'works_at', 'e-acme', 'p1', 'n1',
                0, 20, 0, 5, 15, 19, 0.9, 'gliner2_batch', 'fixture-v1',
                10, 'active', 'build-fixture', 0.95, 'Alice works at Acme', 1
            );
            """
        )
        store._conn.commit()
    finally:
        store.close()


def test_score_extractions_matches_silver_facts_through_aliases_and_counts_novel() -> (
    None
):
    silver = [
        quality_bench.SilverFact(
            note_id="n1",
            subject_entity_id="e-alice",
            predicate="works_at",
            object_entity_id="e-acme",
        )
    ]
    aliases = {
        "alice": frozenset({"e-alice"}),
        "alice smith": frozenset({"e-alice"}),
        "acme": frozenset({"e-acme"}),
        "acme corporation": frozenset({"e-acme"}),
        "bob": frozenset({"e-bob"}),
    }
    extractions = {
        "n1": Extraction(
            fact_candidates=[_relation("Alice", "WORKS-AT", "Acme", 0, 20)]
        ),
        "n2": Extraction(fact_candidates=[_relation("Bob", "uses", "Taurine", 0, 17)]),
    }

    metrics = quality_bench.score_extractions(extractions, silver, aliases)

    assert metrics == {
        "notes": 2,
        "failed_notes": 0,
        "mentions": 0,
        "extracted_facts": 2,
        "silver_facts": 1,
        "silver_facts_matched": 1,
        "silver_fact_recall": 1.0,
        "new_fact_yield": 1,
    }


def test_write_report_uses_stable_schema_and_caller_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    payload = quality_bench.write_report(
        path,
        variant="relex",
        metrics={"silver_fact_recall": 0.75},
        cost={"wall_time_seconds": 12.5},
        timestamp_source="2026-07-13T12:00:00Z",
    )

    assert payload == {
        "variant": "relex",
        "metrics": {"silver_fact_recall": 0.75},
        "cost": {"wall_time_seconds": 12.5},
        "timestamp_source": "2026-07-13T12:00:00Z",
    }
    assert json.loads(path.read_text()) == payload


def test_extraction_cli_runs_each_backend_on_read_only_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "knowledge.db"
    _fixture_knowledge_db(database)
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_dispatch(notes, *, tokenizer=None, progress=None, backend=None):
        del tokenizer, progress
        calls.append((backend, tuple(note.note_id for note in notes)))
        return {
            note.note_id: Extraction(
                fact_candidates=(
                    [_relation("Alice", "works_at", "Acme", 0, 20)]
                    if note.note_id == "n1"
                    else []
                )
            )
            for note in notes
        }

    monkeypatch.setattr(quality_bench, "extract_notes_dispatch", fake_dispatch)
    output = tmp_path / "bench.json"

    exit_code = quality_bench.main(
        [
            "extraction",
            "--db",
            str(database),
            "--backends",
            "gliner2,relex",
            "--sample",
            "6",
            "--seed",
            "7",
            "--timestamp-source",
            "fixture-time",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert {backend for backend, _ in calls} == {"gliner2", "relex"}
    assert all(len(note_ids) == 6 for _, note_ids in calls)
    assert not output.exists()
    for backend in ("gliner2", "relex"):
        report_path = tmp_path / f"bench-{backend}.json"
        report = json.loads(report_path.read_text())
        assert report["variant"] == backend
        assert report["timestamp_source"] == "fixture-time"
        assert report["metrics"]["silver_fact_recall"] == 1.0
        assert report["metrics"]["new_fact_yield"] == 0
        assert report["cost"]["wall_time_seconds"] >= 0.0

    # Opening in read-only mode is part of the CLI contract: no SQLite sidecars.
    assert not (tmp_path / "knowledge.db-wal").exists()
    assert not (tmp_path / "knowledge.db-shm").exists()
