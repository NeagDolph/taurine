"""Fresh-only SQLite store for Taurine's disposable derived knowledge index."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import quote

import numpy as np

from taurine_core.indexer.base_graph import (
    BaseEdge,
    BaseGraph,
    FactGraphInput,
    grouped_fact_edges,
    materialize_base_graph,
)
from taurine_core.indexer.embed import FactEmbedding, PassageEmbedding
from taurine_core.indexer.facts import PREDICATES, FactRecord
from taurine_core.indexer.passages import PassageDraft
from taurine_core.indexer.projection import ProjectionEdge
from taurine_core.schemas import NoteRecord


class KnowledgeSchemaError(RuntimeError):
    """Raised when a disposable index has an incompatible schema."""


@dataclass(frozen=True)
class BuildRecord:
    build_id: str
    started_at: float
    completed_at: float | None
    status: str
    health: str
    source_snapshot: str
    calibration_hash: str
    graph_hash: str


@dataclass(frozen=True)
class StoredNote:
    note_id: str
    source: str
    title: str
    folder: str
    text: str
    content_hash: str
    created: float
    modified: float
    display_line: str
    extraction_status: str
    extraction_error: str
    build_id: str


@dataclass(frozen=True)
class IntegrityReport:
    foreign_key_errors: int
    orphan_entities: int
    stale_fact_embeddings: int
    missing_fact_embeddings: int
    self_facts: int

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.foreign_key_errors,
                self.orphan_entities,
                self.stale_fact_embeddings,
                self.missing_fact_embeddings,
                self.self_facts,
            )
        )


@dataclass(frozen=True)
class BM25PassageHit:
    passage_id: str
    raw_score: float
    token_coverage: float


@dataclass(frozen=True)
class FactContext:
    fact_id: str
    subject_entity_id: str
    subject: str
    predicate: str
    predicate_description: str
    object_entity_id: str
    object: str
    source_passage_id: str
    source_note_id: str
    assertion_start: int
    assertion_end: int
    assertion: str
    observed_at: float
    valid_from: float | None
    valid_to: float | None
    invalidated_at: float | None
    superseded_by: str | None
    status: str
    build_id: str

    @property
    def assertion_text(self) -> str:
        return self.assertion

    @property
    def canonical_text(self) -> str:
        return (
            f"{self.subject} — {self.predicate} ({self.predicate_description}) — "
            f"{self.object}"
        )

    @property
    def logical_key(self) -> tuple[str, str, str]:
        return (self.subject_entity_id, self.predicate, self.object_entity_id)

    @property
    def recognition_document(self) -> str:
        return (
            f"Triple: {self.canonical_text}\n"
            f"Assertion: {self.assertion}\n"
            f"Observed: {self.observed_at:g}\n"
            f"Status: {self.status}"
        )


class KnowledgeStore:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        existed = self.path.exists()
        if self.read_only:
            if not existed:
                raise FileNotFoundError(self.path)
            uri = f"file:{quote(str(self.path.resolve()), safe='/')}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

        if existed and "notes" in self.table_names():
            self._assert_current_schema()
        elif self.read_only:
            raise KnowledgeSchemaError(
                "knowledge index schema is missing; delete and regenerate the derived index"
            )
        else:
            self._init_schema()
            self._assert_current_schema()

    def close(self) -> None:
        self._conn.close()

    def table_names(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        return {str(row["name"]) for row in rows if not str(row["name"]).startswith("sqlite_")}

    def begin_build(
        self,
        build_id: str,
        *,
        source_snapshot: str,
        model_manifest: dict[str, Any],
        config_manifest: dict[str, Any],
        calibration_hash: str,
        started_at: float,
    ) -> None:
        self._writable()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO index_builds(
                    build_id, started_at, status, health, source_snapshot,
                    model_manifest, config_manifest, calibration_hash
                ) VALUES (?, ?, 'building', 'healthy', ?, ?, ?, ?)
                """,
                (
                    build_id,
                    float(started_at),
                    source_snapshot,
                    json.dumps(model_manifest, sort_keys=True, separators=(",", ":")),
                    json.dumps(config_manifest, sort_keys=True, separators=(",", ":")),
                    calibration_hash,
                ),
            )

    def build(self, build_id: str) -> BuildRecord | None:
        row = self._conn.execute(
            "SELECT * FROM index_builds WHERE build_id = ?", (build_id,)
        ).fetchone()
        if row is None:
            return None
        return BuildRecord(
            build_id=row["build_id"],
            started_at=float(row["started_at"]),
            completed_at=(
                float(row["completed_at"]) if row["completed_at"] is not None else None
            ),
            status=row["status"],
            health=row["health"],
            source_snapshot=row["source_snapshot"],
            calibration_hash=row["calibration_hash"],
            graph_hash=row["graph_hash"],
        )

    def completed_build(self, build_id: str | None = None) -> BuildRecord | None:
        if build_id is None:
            row = self._conn.execute(
                """
                SELECT build_id FROM index_builds
                WHERE status = 'complete'
                ORDER BY completed_at DESC, build_id DESC
                LIMIT 1
                """
            ).fetchone()
            return None if row is None else self.build(str(row["build_id"]))
        record = self.build(build_id)
        return record if record is not None and record.status == "complete" else None

    def complete_build(
        self,
        build_id: str,
        *,
        completed_at: float,
        health: Literal["healthy", "degraded"],
        graph_hash: str,
        base_edge_count: int,
        projection_count: int,
    ) -> None:
        self._writable()
        counts = self.counts()
        with self._conn:
            self._conn.execute(
                """
                UPDATE index_builds SET
                    completed_at = ?, status = 'complete', health = ?,
                    passage_count = ?, entity_count = ?, fact_count = ?,
                    base_edge_count = ?, projection_count = ?, graph_hash = ?, error = ''
                WHERE build_id = ?
                """,
                (
                    float(completed_at),
                    health,
                    counts["passages"],
                    counts["entities"],
                    counts["facts"],
                    int(base_edge_count),
                    int(projection_count),
                    graph_hash,
                    build_id,
                ),
            )

    def write_note(
        self,
        note: NoteRecord,
        *,
        build_id: str,
        extraction_status: Literal["pending", "complete", "failed"],
        extraction_error: str = "",
        ingest_salience: float = 1.0,
        quality_rank: float = 0.0,
        value_score: float = 0.0,
    ) -> None:
        self._writable()
        display_line = _display_line(note.text)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO notes(
                    note_id, source, title, folder, text, content_hash,
                    created, modified, display_line, ingest_salience,
                    quality_rank, value_score, extraction_status,
                    extraction_error, build_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    source=excluded.source, title=excluded.title, folder=excluded.folder,
                    text=excluded.text, content_hash=excluded.content_hash,
                    created=excluded.created, modified=excluded.modified,
                    display_line=excluded.display_line, ingest_salience=excluded.ingest_salience,
                    quality_rank=excluded.quality_rank, value_score=excluded.value_score,
                    extraction_status=excluded.extraction_status,
                    extraction_error=excluded.extraction_error, build_id=excluded.build_id
                """,
                (
                    note.note_id,
                    note.source,
                    note.title,
                    note.folder,
                    note.text,
                    note.content_hash,
                    float(note.created or note.modified),
                    float(note.modified),
                    display_line,
                    float(ingest_salience),
                    float(quality_rank),
                    float(value_score),
                    extraction_status,
                    extraction_error,
                    build_id,
                ),
            )

    def source_notes(self) -> tuple[StoredNote, ...]:
        rows = self._conn.execute("SELECT * FROM notes ORDER BY note_id").fetchall()
        return tuple(
            StoredNote(
                note_id=row["note_id"],
                source=row["source"],
                title=row["title"],
                folder=row["folder"],
                text=row["text"],
                content_hash=row["content_hash"],
                created=float(row["created"]),
                modified=float(row["modified"]),
                display_line=row["display_line"],
                extraction_status=row["extraction_status"],
                extraction_error=row["extraction_error"],
                build_id=row["build_id"],
            )
            for row in rows
        )

    def write_passages(
        self,
        note_id: str,
        passages: tuple[PassageDraft, ...],
        *,
        created: float,
        modified: float,
        build_id: str,
    ) -> None:
        self._writable()
        note = self._conn.execute(
            "SELECT title FROM notes WHERE note_id = ?", (note_id,)
        ).fetchone()
        if note is None:
            raise KeyError(note_id)
        with self._conn:
            self._conn.execute("DELETE FROM passages_fts WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM passages WHERE note_id = ?", (note_id,))
            for passage in passages:
                self._conn.execute(
                    """
                    INSERT INTO passages(
                        passage_id, note_id, ordinal, char_start, char_end,
                        token_count, text, content_hash, created, modified, build_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        passage.passage_id,
                        note_id,
                        passage.ordinal,
                        passage.char_start,
                        passage.char_end,
                        passage.token_count,
                        passage.text,
                        passage.content_hash,
                        float(created),
                        float(modified),
                        build_id,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO passages_fts(passage_id, note_id, title, body) VALUES (?, ?, ?, ?)",
                    (passage.passage_id, note_id, note["title"], passage.text),
                )

    def write_passage_embeddings(
        self, embeddings: tuple[PassageEmbedding, ...]
    ) -> None:
        self._writable()
        with self._conn:
            for embedding in embeddings:
                vector = np.asarray(embedding.vector, dtype=np.float32)
                if vector.shape != (embedding.dim,):
                    raise ValueError(
                        f"passage {embedding.passage_id} vector has shape {vector.shape}, "
                        f"expected {(embedding.dim,)}"
                    )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO passage_embeddings(
                        passage_id, model_id, dim, vec, build_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        embedding.passage_id,
                        embedding.model_id,
                        embedding.dim,
                        vector.tobytes(),
                        embedding.build_id,
                    ),
                )

    def passage_embeddings(
        self, *, build_id: str | None = None
    ) -> tuple[PassageEmbedding, ...]:
        if build_id is None:
            rows = self._conn.execute(
                "SELECT passage_id, model_id, dim, vec, build_id "
                "FROM passage_embeddings ORDER BY passage_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT passage_id, model_id, dim, vec, build_id "
                "FROM passage_embeddings WHERE build_id = ? ORDER BY passage_id",
                (build_id,),
            ).fetchall()
        return tuple(
            PassageEmbedding(
                passage_id=row["passage_id"],
                model_id=row["model_id"],
                dim=int(row["dim"]),
                vector=np.frombuffer(row["vec"], dtype=np.float32).copy(),
                build_id=row["build_id"],
            )
            for row in rows
        )

    def write_fact_embeddings(self, embeddings: Sequence[FactEmbedding]) -> None:
        self._writable()
        ordered = sorted(embeddings, key=lambda row: row.fact_id)
        if len({row.fact_id for row in ordered}) != len(ordered):
            raise ValueError("duplicate fact_id in embedding write")
        if len({row.model_id for row in ordered}) > 1:
            raise ValueError("fact embedding batch must use one model_id")
        if len({row.dim for row in ordered}) > 1:
            raise ValueError("fact embedding batch must use one dimension")
        with self._conn:
            for embedding in ordered:
                fact = self._conn.execute(
                    "SELECT build_id FROM facts WHERE fact_id = ?",
                    (embedding.fact_id,),
                ).fetchone()
                if fact is None:
                    raise ValueError(f"fact embedding references missing fact {embedding.fact_id}")
                fact_build_id = str(fact["build_id"])
                if embedding.build_id != fact_build_id:
                    raise ValueError(
                        f"fact {embedding.fact_id} embedding build {embedding.build_id} "
                        f"does not match fact build {fact_build_id}"
                    )
                vector = np.asarray(embedding.vector, dtype=np.float32)
                if vector.shape != (embedding.dim,):
                    raise ValueError(
                        f"fact {embedding.fact_id} vector has shape {vector.shape}, "
                        f"expected {(embedding.dim,)}"
                    )
                if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 0:
                    raise ValueError(
                        f"fact {embedding.fact_id} vector must be finite and nonzero"
                    )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO fact_embeddings(
                        fact_id, model_id, text, dim, vec, build_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        embedding.fact_id,
                        embedding.model_id,
                        embedding.text,
                        embedding.dim,
                        vector.tobytes(),
                        embedding.build_id,
                    ),
                )

    def fact_embeddings(
        self, *, build_id: str | None = None
    ) -> tuple[FactEmbedding, ...]:
        if build_id is None:
            rows = self._conn.execute(
                "SELECT fact_id, model_id, text, dim, vec, build_id "
                "FROM fact_embeddings ORDER BY fact_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT fact_id, model_id, text, dim, vec, build_id "
                "FROM fact_embeddings WHERE build_id = ? ORDER BY fact_id",
                (build_id,),
            ).fetchall()
        embeddings: list[FactEmbedding] = []
        for row in rows:
            vector = np.frombuffer(row["vec"], dtype=np.float32).copy()
            dim = int(row["dim"])
            if vector.shape != (dim,):
                raise ValueError(
                    f"stored fact {row['fact_id']} vector has shape {vector.shape}, "
                    f"expected {(dim,)}"
                )
            embeddings.append(
                FactEmbedding(
                    fact_id=row["fact_id"],
                    model_id=row["model_id"],
                    text=row["text"],
                    dim=dim,
                    vector=vector,
                    build_id=row["build_id"],
                )
            )
        return tuple(embeddings)

    def write_semantic_edges(
        self,
        rows: Sequence[tuple[str, str, float, float]],
        *,
        build_id: str,
    ) -> None:
        self._writable()
        with self._conn:
            for src, dst, cosine, weight in rows:
                left, right = sorted((src, dst))
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO passage_semantic_edges(
                        src_passage_id, dst_passage_id, cosine, weight, build_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (left, right, float(cosine), float(weight), build_id),
                )

    def write_passage_entity_edges(
        self,
        rows: Sequence[tuple[str, str, float, int, float, float]],
        *,
        build_id: str,
    ) -> None:
        self._writable()
        with self._conn:
            for passage_id, entity_id, confidence, mentions, idf, weight in sorted(
                rows, key=lambda row: (row[0], row[1])
            ):
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO passage_entity_edges(
                        passage_id, entity_id, confidence, mentions, idf, weight, build_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        passage_id,
                        entity_id,
                        float(confidence),
                        int(mentions),
                        float(idf),
                        float(weight),
                        build_id,
                    ),
                )

    def write_projection(
        self,
        rows: Sequence[ProjectionEdge],
        *,
        build_id: str,
    ) -> None:
        self._writable()
        with self._conn:
            for row in rows:
                self._conn.execute(
                    """
                    INSERT INTO note_projection_edges(
                        src_note_id, dst_note_id, edge_type, weight,
                        shared, fact_ids, mechanisms, build_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.src_note_id,
                        row.dst_note_id,
                        row.edge_type,
                        row.weight,
                        json.dumps(row.shared),
                        json.dumps(row.fact_ids),
                        json.dumps(row.mechanisms, sort_keys=True),
                        build_id,
                    ),
                )

    def projection(self, build_id: str) -> tuple[ProjectionEdge, ...]:
        rows = self._conn.execute(
            """
            SELECT src_note_id, dst_note_id, edge_type, weight,
                   shared, fact_ids, mechanisms
            FROM note_projection_edges
            WHERE build_id = ?
            ORDER BY src_note_id, dst_note_id
            """,
            (build_id,),
        ).fetchall()
        return tuple(
            ProjectionEdge(
                src_note_id=row["src_note_id"],
                dst_note_id=row["dst_note_id"],
                edge_type=row["edge_type"],
                weight=float(row["weight"]),
                shared=tuple(json.loads(row["shared"])),
                fact_ids=tuple(json.loads(row["fact_ids"])),
                mechanisms={
                    str(key): float(value)
                    for key, value in json.loads(row["mechanisms"]).items()
                },
            )
            for row in rows
        )

    def quality_ranks(self, build_id: str) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT note_id, quality_rank FROM notes WHERE build_id = ? ORDER BY note_id",
            (build_id,),
        ).fetchall()
        return {str(row["note_id"]): float(row["quality_rank"]) for row in rows}

    def search_passages_bm25(self, query: str, k: int) -> tuple[BM25PassageHit, ...]:
        if not query.strip() or k <= 0:
            return ()
        terms = _query_terms(query)
        if not terms:
            return ()
        try:
            rows = self._conn.execute(
                """
                SELECT f.passage_id, -bm25(passages_fts) AS score,
                       p.text, n.title
                FROM passages_fts f
                JOIN passages p ON p.passage_id = f.passage_id
                JOIN notes n ON n.note_id = p.note_id
                WHERE passages_fts MATCH ?
                ORDER BY bm25(passages_fts), f.passage_id
                LIMIT ?
                """,
                (" OR ".join(f'"{term}"' for term in terms), int(k)),
            ).fetchall()
        except sqlite3.OperationalError:
            return ()
        return tuple(
            BM25PassageHit(
                passage_id=row["passage_id"],
                raw_score=float(row["score"]),
                token_coverage=_token_coverage(terms, f"{row['title']} {row['text']}"),
            )
            for row in rows
        )

    def insert_entity(
        self,
        *,
        entity_id: str,
        canonical: str,
        normalized: str,
        etype: str,
        df: int,
        traversal_eligible: bool,
        ambiguity_key: str,
        build_id: str,
    ) -> None:
        self._writable()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO entities(
                    entity_id, canonical, normalized, etype, df,
                    traversal_eligible, ambiguity_key, build_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    canonical,
                    normalized,
                    etype,
                    int(df),
                    int(traversal_eligible),
                    ambiguity_key,
                    build_id,
                ),
            )

    def insert_alias(
        self,
        entity_id: str,
        surface: str,
        origin: str,
        confidence: float,
    ) -> None:
        self._writable()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO entity_aliases(entity_id, surface, surface_cf, origin, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entity_id, surface, surface.casefold(), origin, float(confidence)),
            )

    def alias_index(self, *, build_id: str | None = None) -> dict[str, tuple[str, ...]]:
        """Return the immutable-snapshot input keyed by case-folded surface."""

        if build_id is None:
            rows = self._conn.execute(
                "SELECT surface_cf, entity_id FROM entity_aliases "
                "ORDER BY surface_cf, entity_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT a.surface_cf, a.entity_id
                FROM entity_aliases a
                JOIN entities e ON e.entity_id = a.entity_id
                WHERE e.build_id = ?
                ORDER BY a.surface_cf, a.entity_id
                """,
                (build_id,),
            ).fetchall()
        grouped: dict[str, list[str]] = {}
        for row in rows:
            target_ids = grouped.setdefault(str(row["surface_cf"]), [])
            entity_id = str(row["entity_id"])
            if not target_ids or target_ids[-1] != entity_id:
                target_ids.append(entity_id)
        return {surface: tuple(target_ids) for surface, target_ids in grouped.items()}

    def insert_mention(
        self,
        *,
        mention_id: str,
        entity_id: str,
        passage_id: str,
        note_id: str,
        surface: str,
        etype: str,
        char_start: int,
        char_end: int,
        confidence: float,
        extractor: str,
        schema_id: str,
        build_id: str,
    ) -> None:
        self._writable()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO entity_mentions(
                    mention_id, entity_id, passage_id, note_id, surface, etype,
                    char_start, char_end, confidence, extractor, schema_id, build_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mention_id,
                    entity_id,
                    passage_id,
                    note_id,
                    surface,
                    etype,
                    int(char_start),
                    int(char_end),
                    float(confidence),
                    extractor,
                    schema_id,
                    build_id,
                ),
            )

    def insert_fact(self, fact: FactRecord) -> None:
        self._writable()
        with self._conn:
            self._conn.execute(
                """INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    fact.fact_id, fact.subject_entity_id, fact.predicate, fact.object_entity_id,
                    fact.source_passage_id, fact.source_note_id, fact.assertion_start,
                    fact.assertion_end, fact.subject_start, fact.subject_end, fact.object_start,
                    fact.object_end, fact.confidence, fact.extraction_method, fact.schema_id,
                    fact.observed_at, fact.valid_from, fact.valid_to, fact.invalidated_at,
                    fact.superseded_by, fact.status, fact.build_id,
                ),
            )

    def fact_contexts(
        self,
        fact_ids: tuple[str, ...] | None = None,
        *,
        include_candidates: bool = False,
        build_id: str | None = None,
    ) -> tuple[FactContext, ...]:
        if fact_ids == ():
            return ()
        clauses: list[str] = []
        parameters: list[Any] = []
        if not include_candidates:
            clauses.append("f.status <> 'candidate'")
        if build_id is not None:
            clauses.append("f.build_id = ?")
            parameters.append(build_id)
        if fact_ids is not None:
            clauses.append(f"f.fact_id IN ({','.join('?' for _ in fact_ids)})")
            parameters.extend(fact_ids)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT f.*, se.canonical AS subject_canonical,
                   oe.canonical AS object_canonical,
                   p.char_start AS passage_start, p.char_end AS passage_end,
                   p.text AS passage_text, n.text AS note_text
            FROM facts f
            JOIN entities se ON se.entity_id = f.subject_entity_id
            JOIN entities oe ON oe.entity_id = f.object_entity_id
            JOIN passages p ON p.passage_id = f.source_passage_id
            JOIN notes n ON n.note_id = f.source_note_id
            {where}
            ORDER BY f.fact_id
            """,
            tuple(parameters),
        ).fetchall()
        contexts: list[FactContext] = []
        for row in rows:
            start = int(row["assertion_start"])
            end = int(row["assertion_end"])
            passage_start = int(row["passage_start"])
            passage_end = int(row["passage_end"])
            if not passage_start <= start < end <= passage_end:
                raise ValueError(
                    f"fact {row['fact_id']} assertion [{start}, {end}) is outside "
                    f"passage [{passage_start}, {passage_end})"
                )
            local_start = start - passage_start
            local_end = end - passage_start
            assertion = str(row["passage_text"])[local_start:local_end]
            note_assertion = str(row["note_text"])[start:end]
            if assertion != note_assertion:
                raise ValueError(
                    f"fact {row['fact_id']} assertion does not map exactly from note to passage"
                )
            predicate = str(row["predicate"])
            spec = PREDICATES.get(predicate)
            if spec is None:
                raise ValueError(f"fact {row['fact_id']} has unregistered predicate {predicate}")
            contexts.append(
                FactContext(
                    fact_id=row["fact_id"],
                    subject_entity_id=row["subject_entity_id"],
                    subject=row["subject_canonical"],
                    predicate=predicate,
                    predicate_description=spec.description,
                    object_entity_id=row["object_entity_id"],
                    object=row["object_canonical"],
                    source_passage_id=row["source_passage_id"],
                    source_note_id=row["source_note_id"],
                    assertion_start=start,
                    assertion_end=end,
                    assertion=assertion,
                    observed_at=float(row["observed_at"]),
                    valid_from=_optional_float(row["valid_from"]),
                    valid_to=_optional_float(row["valid_to"]),
                    invalidated_at=_optional_float(row["invalidated_at"]),
                    superseded_by=row["superseded_by"],
                    status=row["status"],
                    build_id=row["build_id"],
                )
            )
        return tuple(contexts)

    def base_graph(self, build_id: str) -> BaseGraph:
        node_rows = self._conn.execute(
            """
            SELECT passage_id AS node_id, 'passage' AS kind FROM passages
            WHERE build_id = ?
            UNION ALL
            SELECT entity_id AS node_id, 'entity' AS kind FROM entities
            WHERE build_id = ?
            """,
            (build_id, build_id),
        ).fetchall()
        edges: list[BaseEdge] = []
        for row in self._conn.execute(
            """
            SELECT src_passage_id, dst_passage_id, cosine, weight
            FROM passage_semantic_edges WHERE build_id = ?
            ORDER BY src_passage_id, dst_passage_id
            """,
            (build_id,),
        ).fetchall():
            metadata = {"cosine": float(row["cosine"])}
            edges.extend(
                (
                    BaseEdge(row["src_passage_id"], row["dst_passage_id"], "semantic", float(row["weight"]), metadata=metadata),
                    BaseEdge(row["dst_passage_id"], row["src_passage_id"], "semantic", float(row["weight"]), metadata=metadata),
                )
            )
        for row in self._conn.execute(
            """
            SELECT passage_id, entity_id, idf, mentions, weight
            FROM passage_entity_edges WHERE build_id = ?
            ORDER BY passage_id, entity_id
            """,
            (build_id,),
        ).fetchall():
            metadata = {"idf": float(row["idf"]), "mentions": int(row["mentions"])}
            edges.extend(
                (
                    BaseEdge(row["passage_id"], row["entity_id"], "contains", float(row["weight"]), metadata=metadata),
                    BaseEdge(row["entity_id"], row["passage_id"], "contains", float(row["weight"]), metadata=metadata),
                )
            )
        fact_rows = self._conn.execute(
            """
            SELECT fact_id, subject_entity_id, predicate, object_entity_id,
                   source_note_id, confidence, status
            FROM facts
            WHERE build_id = ?
            ORDER BY fact_id
            """,
            (build_id,),
        ).fetchall()
        edges.extend(
            grouped_fact_edges(
                tuple(
                    FactGraphInput(
                        fact_id=str(row["fact_id"]),
                        subject_entity_id=str(row["subject_entity_id"]),
                        predicate=str(row["predicate"]),
                        object_entity_id=str(row["object_entity_id"]),
                        source_note_id=str(row["source_note_id"]),
                        confidence=float(row["confidence"]),
                        status=str(row["status"]),
                    )
                    for row in fact_rows
                )
            )
        )
        return materialize_base_graph(
            build_id=build_id,
            node_ids=tuple(str(row["node_id"]) for row in node_rows),
            node_kinds=tuple(str(row["kind"]) for row in node_rows),
            edges=edges,
        )

    def delete_note(self, note_id: str) -> None:
        self._writable()
        with self._conn:
            self._conn.execute("DELETE FROM passages_fts WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
        self.prune_orphan_entities()

    def counts(self) -> dict[str, int]:
        names = (
            "notes",
            "passages",
            "passage_embeddings",
            "entities",
            "entity_mentions",
            "facts",
            "fact_embeddings",
            "passage_semantic_edges",
            "passage_entity_edges",
            "note_projection_edges",
        )
        return {
            name: int(self._conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
            for name in names
        }

    def prune_orphan_entities(self) -> int:
        self._writable()
        with self._conn:
            cursor = self._conn.execute(
                """
                DELETE FROM entities
                WHERE NOT EXISTS (
                    SELECT 1 FROM entity_mentions m WHERE m.entity_id = entities.entity_id
                )
                """
            )
        return max(0, int(cursor.rowcount))

    def validate_invariants(self) -> IntegrityReport:
        foreign_key_errors = len(self._conn.execute("PRAGMA foreign_key_check").fetchall())
        orphan_entities = int(
            self._conn.execute(
                """
                SELECT count(*) FROM entities e
                WHERE NOT EXISTS (
                    SELECT 1 FROM entity_mentions m WHERE m.entity_id = e.entity_id
                )
                """
            ).fetchone()[0]
        )
        stale_fact_embeddings = int(
            self._conn.execute(
                """
                SELECT count(*) FROM fact_embeddings fe
                LEFT JOIN facts f ON f.fact_id = fe.fact_id
                WHERE f.fact_id IS NULL
                """
            ).fetchone()[0]
        )
        missing_fact_embeddings = int(
            self._conn.execute(
                """
                SELECT count(*) FROM facts f
                LEFT JOIN fact_embeddings fe ON fe.fact_id = f.fact_id
                WHERE f.status <> 'candidate' AND fe.fact_id IS NULL
                """
            ).fetchone()[0]
        )
        self_facts = int(
            self._conn.execute(
                "SELECT count(*) FROM facts WHERE subject_entity_id = object_entity_id"
            ).fetchone()[0]
        )
        return IntegrityReport(
            foreign_key_errors=foreign_key_errors,
            orphan_entities=orphan_entities,
            stale_fact_embeddings=stale_fact_embeddings,
            missing_fact_embeddings=missing_fact_embeddings,
            self_facts=self_facts,
        )

    def _writable(self) -> None:
        if self.read_only:
            raise PermissionError("knowledge store is read-only")

    def _assert_current_schema(self) -> None:
        required = {
            "note_id",
            "source",
            "title",
            "folder",
            "text",
            "content_hash",
            "created",
            "modified",
            "display_line",
            "ingest_salience",
            "quality_rank",
            "value_score",
            "extraction_status",
            "extraction_error",
            "build_id",
        }
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if not required <= columns:
            raise KnowledgeSchemaError(
                "incompatible derived knowledge schema; delete and regenerate the knowledge index"
            )
        missing_tables = _REQUIRED_TABLES - self.table_names()
        if missing_tables:
            raise KnowledgeSchemaError(
                "incomplete derived knowledge schema; delete and regenerate the knowledge index"
            )

    def _init_schema(self) -> None:
        self._writable()
        with self._conn:
            self._conn.executescript(_SCHEMA)


_REQUIRED_TABLES = {
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


_SCHEMA = """
CREATE TABLE notes (
    note_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    folder TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created REAL NOT NULL,
    modified REAL NOT NULL,
    display_line TEXT NOT NULL,
    ingest_salience REAL NOT NULL DEFAULT 1.0,
    quality_rank REAL NOT NULL DEFAULT 0.0,
    value_score REAL NOT NULL DEFAULT 0.0,
    extraction_status TEXT NOT NULL CHECK (extraction_status IN ('pending', 'complete', 'failed')),
    extraction_error TEXT NOT NULL DEFAULT '',
    build_id TEXT NOT NULL
);

CREATE TABLE passages (
    passage_id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created REAL NOT NULL,
    modified REAL NOT NULL,
    build_id TEXT NOT NULL,
    UNIQUE(note_id, ordinal),
    CHECK(char_start >= 0 AND char_end > char_start),
    CHECK(length(text) > 0)
);
CREATE INDEX passages_note_idx ON passages(note_id, ordinal);
CREATE VIRTUAL TABLE passages_fts USING fts5(
    passage_id UNINDEXED,
    note_id UNINDEXED,
    title,
    body,
    tokenize='unicode61'
);

CREATE TABLE passage_embeddings (
    passage_id TEXT PRIMARY KEY REFERENCES passages(passage_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    build_id TEXT NOT NULL
);

CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    canonical TEXT NOT NULL,
    normalized TEXT NOT NULL,
    etype TEXT NOT NULL,
    df INTEGER NOT NULL,
    traversal_eligible INTEGER NOT NULL,
    ambiguity_key TEXT NOT NULL,
    build_id TEXT NOT NULL,
    UNIQUE(etype, normalized, ambiguity_key)
);

CREATE TABLE entity_aliases (
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    surface TEXT NOT NULL,
    surface_cf TEXT NOT NULL,
    origin TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY(entity_id, surface, origin)
);
CREATE INDEX entity_alias_cf_idx ON entity_aliases(surface_cf);

CREATE TABLE entity_mentions (
    mention_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    surface TEXT NOT NULL,
    etype TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    confidence REAL NOT NULL,
    extractor TEXT NOT NULL,
    schema_id TEXT NOT NULL,
    build_id TEXT NOT NULL,
    CHECK(char_start >= 0 AND char_end > char_start)
);
CREATE INDEX entity_mentions_entity_idx ON entity_mentions(entity_id);
CREATE INDEX entity_mentions_passage_idx ON entity_mentions(passage_id);
CREATE INDEX entity_mentions_note_idx ON entity_mentions(note_id);

CREATE TABLE facts (
    fact_id TEXT PRIMARY KEY,
    subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    source_passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    source_note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    assertion_start INTEGER NOT NULL,
    assertion_end INTEGER NOT NULL,
    subject_start INTEGER NOT NULL,
    subject_end INTEGER NOT NULL,
    object_start INTEGER NOT NULL,
    object_end INTEGER NOT NULL,
    confidence REAL NOT NULL,
    extraction_method TEXT NOT NULL,
    schema_id TEXT NOT NULL,
    observed_at REAL NOT NULL,
    valid_from REAL,
    valid_to REAL,
    invalidated_at REAL,
    superseded_by TEXT REFERENCES facts(fact_id),
    status TEXT NOT NULL CHECK (status IN ('active', 'historical', 'superseded', 'conflicted', 'candidate')),
    build_id TEXT NOT NULL,
    CHECK(subject_entity_id <> object_entity_id),
    CHECK(assertion_start >= 0 AND assertion_end > assertion_start)
);
CREATE INDEX facts_subject_idx ON facts(subject_entity_id, predicate, status);
CREATE INDEX facts_object_idx ON facts(object_entity_id, predicate, status);
CREATE INDEX facts_passage_idx ON facts(source_passage_id);
CREATE INDEX facts_temporal_idx ON facts(observed_at, valid_from, valid_to, status);

CREATE TABLE predicate_proposals (
    proposal_id TEXT PRIMARY KEY,
    proposed_predicate TEXT NOT NULL,
    description TEXT NOT NULL,
    subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    object_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    source_passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    assertion_start INTEGER NOT NULL,
    assertion_end INTEGER NOT NULL,
    confidence REAL NOT NULL,
    extraction_method TEXT NOT NULL,
    build_id TEXT NOT NULL
);

CREATE TABLE fact_embeddings (
    fact_id TEXT PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    text TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    build_id TEXT NOT NULL
);

CREATE TABLE passage_semantic_edges (
    src_passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    dst_passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    cosine REAL NOT NULL,
    weight REAL NOT NULL,
    build_id TEXT NOT NULL,
    PRIMARY KEY(src_passage_id, dst_passage_id),
    CHECK(src_passage_id < dst_passage_id)
);

CREATE TABLE passage_entity_edges (
    passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    confidence REAL NOT NULL,
    mentions INTEGER NOT NULL,
    idf REAL NOT NULL,
    weight REAL NOT NULL,
    build_id TEXT NOT NULL,
    PRIMARY KEY(passage_id, entity_id)
);

CREATE TABLE note_projection_edges (
    src_note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    dst_note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    shared TEXT NOT NULL DEFAULT '[]',
    fact_ids TEXT NOT NULL DEFAULT '[]',
    mechanisms TEXT NOT NULL DEFAULT '{}',
    build_id TEXT NOT NULL,
    PRIMARY KEY(src_note_id, dst_note_id),
    CHECK(src_note_id < dst_note_id)
);

CREATE TABLE index_builds (
    build_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    completed_at REAL,
    status TEXT NOT NULL CHECK (status IN ('building', 'complete', 'failed')),
    health TEXT NOT NULL CHECK (health IN ('healthy', 'degraded')),
    source_snapshot TEXT NOT NULL,
    model_manifest TEXT NOT NULL,
    config_manifest TEXT NOT NULL,
    calibration_hash TEXT NOT NULL,
    passage_count INTEGER NOT NULL DEFAULT 0,
    entity_count INTEGER NOT NULL DEFAULT 0,
    fact_count INTEGER NOT NULL DEFAULT 0,
    base_edge_count INTEGER NOT NULL DEFAULT 0,
    projection_count INTEGER NOT NULL DEFAULT 0,
    graph_hash TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT ''
);
"""


def _display_line(text: str) -> str:
    return next((line.strip()[:220] for line in text.splitlines() if line.strip()), "")


def _query_terms(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.casefold() for match in re.findall(r"\w+", text)))


def _token_coverage(terms: tuple[str, ...], text: str) -> float:
    if not terms:
        return 0.0
    haystack = set(re.findall(r"\w+", text.casefold()))
    return sum(term in haystack for term in terms) / len(terms)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
