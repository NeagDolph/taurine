"""One-time enrichment-streamlining migration."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from taurine_core import config
from taurine_core.ports import LLMPort
from taurine_core.schemas import Extraction, NoteRecord

from taurine_core.indexer import adjudicate, extract, rank
from taurine_core.indexer.store import Index, _display_line
from taurine_core.indexer.sync import _cli_llm, _run_async


@dataclass(frozen=True)
class MigrationReport:
    gist_rows_deleted: int
    utterance_rows_deleted: int
    notes_backfilled: int
    extraction_failures: int
    judgments_written: int


def migrate(db_path: Path, *, llm: LLMPort | None, dry_run: bool) -> MigrationReport:
    if dry_run:
        return _dry_run(db_path, llm=llm)

    index = Index(db_path)
    try:
        return _migrate_index(index, llm=llm, dry_run=dry_run)
    finally:
        index.close()


def _migrate_index(index: Index, *, llm: LLMPort | None, dry_run: bool) -> MigrationReport:
    active_notes = _active_notes(index)

    print("step 1/5: remove stale enrichment embeddings", flush=True)
    gist_rows_deleted = _embedding_count(index, "gist")
    utterance_rows_deleted = (
        _embedding_count(index, "utterance") if not config.SLIM_UTTERANCES else 0
    )
    if dry_run:
        print(
            "dry-run: would delete "
            f"{gist_rows_deleted} gist rows and {utterance_rows_deleted} utterance rows",
            flush=True,
        )
    else:
        with index._conn:  # type: ignore[attr-defined]
            index._conn.execute(  # type: ignore[attr-defined]
                "DELETE FROM embeddings WHERE kind = 'gist'"
            )
            if not config.SLIM_UTTERANCES:
                index._conn.execute(  # type: ignore[attr-defined]
                    "DELETE FROM embeddings WHERE kind = 'utterance'"
                )
        index.load()

    print("step 2/5: recompute display lines", flush=True)
    if dry_run:
        print(f"dry-run: would recompute {len(active_notes)} display lines", flush=True)
    else:
        _recompute_display_lines(index, active_notes)

    print("step 3/5: backfill GLiNER extractions", flush=True)
    if dry_run:
        notes_backfilled = len(active_notes)
        extraction_failures = 0
        print(
            "dry-run: extraction is not run; "
            f"reporting {notes_backfilled} notes that would be attempted "
            "and 0 measured failures",
            flush=True,
        )
    else:
        notes_backfilled, extraction_failures = _backfill_extractions(index, active_notes)

    print("step 4/5: rebuild graph and ranks", flush=True)
    if dry_run:
        print("dry-run: would rebuild graph, quality ranks, and value scores", flush=True)
    else:
        index.rebuild_graph()
        rank.compute_quality_ranks(index)
        rank.compute_value_scores(index)

    judgments_written = 0
    print("step 5/5: adjudicate graph facts", flush=True)
    if llm is None:
        print("llm disabled: skipping adjudication", flush=True)
    elif dry_run:
        print("dry-run: would run one adjudication pass", flush=True)
    else:
        judgments_written = _adjudicate_once(index, llm)

    report = MigrationReport(
        gist_rows_deleted=gist_rows_deleted,
        utterance_rows_deleted=utterance_rows_deleted,
        notes_backfilled=notes_backfilled,
        extraction_failures=extraction_failures,
        judgments_written=judgments_written,
    )
    print(report, flush=True)
    return report


def _active_notes(index: Index) -> list[NoteRecord]:
    disabled_sources = index.disabled_sources()
    return [note for note in index.notes() if note.source not in disabled_sources]


def _dry_run(db_path: Path, *, llm: LLMPort | None) -> MigrationReport:
    conn = _readonly_connection(db_path)
    try:
        gist_rows_deleted = _readonly_embedding_count(conn, "gist")
        utterance_rows_deleted = (
            _readonly_embedding_count(conn, "utterance")
            if not config.SLIM_UTTERANCES
            else 0
        )
        active_note_count = _readonly_active_note_count(conn)
    finally:
        conn.close()

    print("step 1/5: remove stale enrichment embeddings", flush=True)
    print(
        "dry-run: would delete "
        f"{gist_rows_deleted} gist rows and {utterance_rows_deleted} utterance rows",
        flush=True,
    )
    print("step 2/5: recompute display lines", flush=True)
    print(f"dry-run: would recompute {active_note_count} display lines", flush=True)
    print("step 3/5: backfill GLiNER extractions", flush=True)
    print(
        "dry-run: extraction is not run; "
        f"reporting {active_note_count} notes that would be attempted "
        "and 0 measured failures",
        flush=True,
    )
    print("step 4/5: rebuild graph and ranks", flush=True)
    print("dry-run: would rebuild graph, quality ranks, and value scores", flush=True)
    print("step 5/5: adjudicate graph facts", flush=True)
    if llm is None:
        print("llm disabled: skipping adjudication", flush=True)
    else:
        print("dry-run: would run one adjudication pass", flush=True)

    report = MigrationReport(
        gist_rows_deleted=gist_rows_deleted,
        utterance_rows_deleted=utterance_rows_deleted,
        notes_backfilled=active_note_count,
        extraction_failures=0,
        judgments_written=0,
    )
    print(report, flush=True)
    return report


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _readonly_embedding_count(conn: sqlite3.Connection, kind: str) -> int:
    if not _readonly_table_exists(conn, "embeddings"):
        return 0
    columns = _readonly_columns(conn, "embeddings")
    if "kind" not in columns:
        return 0
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM embeddings WHERE kind = ?",
        (kind,),
    ).fetchone()
    return int(row["count"]) if row is not None else 0


def _readonly_active_note_count(conn: sqlite3.Connection) -> int:
    if not _readonly_table_exists(conn, "notes"):
        return 0
    disabled_sources = _readonly_disabled_sources(conn)
    columns = _readonly_columns(conn, "notes")
    if "source" not in columns:
        return 0 if "apple_notes" in disabled_sources else _readonly_note_count(conn)
    if not disabled_sources:
        return _readonly_note_count(conn)
    placeholders = ",".join("?" for _source in disabled_sources)
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM notes WHERE source NOT IN ({placeholders})",
        tuple(sorted(disabled_sources)),
    ).fetchone()
    return int(row["count"]) if row is not None else 0


def _readonly_note_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM notes").fetchone()
    return int(row["count"]) if row is not None else 0


def _readonly_disabled_sources(conn: sqlite3.Connection) -> set[str]:
    if not _readonly_table_exists(conn, "meta"):
        return set()
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'disabled_sources'"
    ).fetchone()
    if row is None:
        return set()
    try:
        value = json.loads(row["value"])
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in value} if isinstance(value, list) else set()


def _readonly_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type IN ('table', 'virtual') AND name = ?
        """,
        (table,),
    ).fetchone()
    return row is not None


def _readonly_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _embedding_count(index: Index, kind: str) -> int:
    row = index._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS count FROM embeddings WHERE kind = ?",
        (kind,),
    ).fetchone()
    return int(row["count"]) if row is not None else 0


def _recompute_display_lines(index: Index, notes: Sequence[NoteRecord]) -> None:
    with index._conn:  # type: ignore[attr-defined]
        for note in notes:
            index._conn.execute(  # type: ignore[attr-defined]
                "UPDATE notes SET gist = ? WHERE note_id = ?",
                (_display_line(note), note.note_id),
            )


def _backfill_extractions(
    index: Index,
    notes: Sequence[NoteRecord],
) -> tuple[int, int]:
    batch_size = max(1, int(config.GLINER_INGEST_BATCH))
    notes_backfilled = 0
    extraction_failures = 0
    total = len(notes)
    for offset in range(0, total, batch_size):
        batch = list(notes[offset : offset + batch_size])
        results = _extract_batch(batch)
        for note in batch:
            extraction = results.get(note.note_id, Extraction(failed=True))
            if extraction.failed:
                extraction_failures += 1
                continue
            index.replace_note_extraction(
                note.note_id,
                [
                    (entity.surface, entity.etype, entity.conf, entity.mentions)
                    for entity in extraction.entities
                ],
                [
                    (relation.head, relation.rel_type, relation.tail, relation.conf)
                    for relation in extraction.relations
                ],
            )
            index.cache_extraction(note.note_id, note.content_hash, extraction)
            notes_backfilled += 1
        print(
            f"extracted {min(offset + batch_size, total)}/{total}, "
            f"failures={extraction_failures}",
            flush=True,
        )
    return notes_backfilled, extraction_failures


def _extract_batch(notes: list[NoteRecord]) -> dict[str, Extraction]:
    try:
        return extract.extract_notes(notes)
    except Exception:
        return {note.note_id: Extraction(failed=True) for note in notes}


def _adjudicate_once(index: Index, llm: LLMPort) -> int:
    before = len(index.edge_judgments())

    def progress(done: int, total: int) -> None:
        print(f"adjudicated {done}/{total}", flush=True)

    _run_async(adjudicate.adjudicate_graph(index, llm, progress))
    after = len(index.edge_judgments())
    rank.refresh_fact_embeddings(index)
    return max(0, after - before)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the one-time enrichment-streamlining migration."
    )
    parser.add_argument("--db", type=Path, default=Path("taurine.db"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run the budgeted graph adjudication pass after deterministic rebuilds.",
    )
    args = parser.parse_args(argv)

    migrate(args.db, llm=_cli_llm() if args.llm else None, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
