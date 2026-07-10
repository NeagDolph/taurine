from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import numpy as np

from taurine_core import config
from taurine_core.schemas import (
    ChatActivityStep,
    ChatMessage,
    ChatMeta,
    ChatUsage,
    EvidencePackPayload,
    Enrichment,
    EntityMention,
    Extraction,
    GraphEdge,
    GraphNode,
    GraphResponse,
    NeighborRef,
    NotePreview,
    NoteRecord,
    RelationTriple,
)

if TYPE_CHECKING:
    from taurine_core.retrieval.snapshot import IndexSnapshot, RetrievalSnapshotPort

EmbeddingKind = Literal["chunk", "gist", "utterance"]
EmbeddingInput = tuple[EmbeddingKind, str, np.ndarray]


class _Rows:
    """Fully-materialized query result, safe to consume from any thread."""

    def __init__(self, rows: list[sqlite3.Row]) -> None:
        self._rows = rows

    def fetchall(self) -> list[sqlite3.Row]:
        return self._rows

    def fetchone(self) -> sqlite3.Row | None:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _ThreadSafeConnection:
    """Serializes sqlite access across threads.

    The server runs sync/connector/reenrich work in asyncio.to_thread pools;
    a bare sqlite3 connection is bound to its creating thread. This wrapper
    opens with check_same_thread=False and guards every operation with an
    RLock, materializing read results inside the lock so cursors never
    cross threads.
    """

    _READ_PREFIXES = ("SELECT", "PRAGMA", "WITH", "EXPLAIN")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, sql: str, params: Any = ()) -> Any:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            if sql.lstrip()[:8].upper().startswith(self._READ_PREFIXES):
                return _Rows(cursor.fetchall())
            return cursor

    def executemany(self, sql: str, seq: Any) -> Any:
        with self._lock:
            return self._conn.executemany(sql, seq)

    def executescript(self, script: str) -> Any:
        with self._lock:
            return self._conn.executescript(script)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> _ThreadSafeConnection:
        self._lock.acquire()
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return self._conn.__exit__(exc_type, exc, tb)
        finally:
            self._lock.release()

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._conn.row_factory = value


def embed_texts(texts: list[str]) -> np.ndarray:
    from .embed import embed_texts as real_embed_texts

    return real_embed_texts(texts)


def embed_queries(texts: list[str], *, prompt: str | None = None) -> np.ndarray:
    from .embed import embed_queries as real_embed_queries

    return real_embed_queries(texts, prompt=prompt)


def _embed_queries_with_prompt(texts: list[str], prompt: str | None) -> np.ndarray:
    # None → the default spoken-conversation prompt, called WITHOUT the kwarg
    # so prompt-less test fakes keep working and a real TypeError is never
    # masked (review finding, Stage A)
    if prompt is None:
        return embed_queries(texts)
    return embed_queries(texts, prompt=prompt)


class Index:
    def __init__(
        self,
        db_path: str | Path,
        *,
        read_only: bool = False,
        knowledge_snapshots: "RetrievalSnapshotPort | None" = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._read_only = bool(read_only)
        self._knowledge_snapshots = knowledge_snapshots
        if self._read_only:
            if not self.db_path.exists():
                raise FileNotFoundError(self.db_path)
            uri = f"file:{quote(str(self.db_path.resolve()), safe='/')}?mode=ro"
            raw = sqlite3.connect(uri, check_same_thread=False, uri=True)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            raw = sqlite3.connect(self.db_path, check_same_thread=False)
        raw.row_factory = sqlite3.Row
        self._conn = _ThreadSafeConnection(raw)
        self._conn.execute("PRAGMA foreign_keys = ON")
        # rows+matrix live in ONE tuple assigned atomically: readers on the
        # live thread must never see rows from one load and the matrix from
        # another (zip(strict=True) in dense_search would raise)
        self._dense_state: tuple[list[tuple[int, str, EmbeddingKind, str]], np.ndarray] = (
            [],
            np.empty((0, config.EMBED_DIM), dtype=np.float32),
        )
        self._fact_state: tuple[list[tuple[str, str, str]], np.ndarray] = (
            [],
            np.empty((0, config.EMBED_DIM), dtype=np.float32),
        )
        self._entity_surface_state: dict[str, int] = {}
        self._doc_frequency_cache: dict[str, int] = {}
        if not self._read_only:
            self._init_schema()
        self._disabled_sources = self._load_disabled_sources()
        self.load()

    def load(self) -> None:
        clause, params = self._active_sources_clause("n.source")
        rows = self._conn.execute(
            f"""
            SELECT e.id, e.note_id, e.kind, e.text, e.vec
            FROM embeddings e JOIN notes n ON n.note_id = e.note_id
            WHERE 1=1{clause}
            ORDER BY e.id
            """,
            params,
        ).fetchall()
        vectors: list[np.ndarray] = []
        embedding_rows: list[tuple[int, str, EmbeddingKind, str]] = []
        for row in rows:
            vector = np.frombuffer(row["vec"], dtype=np.float32).copy()
            if vector.shape != (config.EMBED_DIM,):
                continue
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            vectors.append(vector.astype(np.float32))
            embedding_rows.append((int(row["id"]), row["note_id"], row["kind"], row["text"]))
        self._dense_state = (
            embedding_rows,
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.empty((0, config.EMBED_DIM), dtype=np.float32),
        )
        self.load_facts()
        self.load_entity_surfaces()

    @property
    def _embedding_rows(self) -> list[tuple[int, str, EmbeddingKind, str]]:
        return self._dense_state[0]

    @property
    def _matrix(self) -> np.ndarray:
        return self._dense_state[1]

    def load_facts(self) -> None:
        """(Re)load the in-memory fact-embedding matrix (active endpoints only)."""
        src_clause, src_params = self._active_sources_clause("n1.source")
        dst_clause, dst_params = self._active_sources_clause("n2.source")
        rows = self._conn.execute(
            f"""
            SELECT f.src, f.dst, f.text, f.vec
            FROM fact_embeddings f
            JOIN notes n1 ON n1.note_id = f.src
            JOIN notes n2 ON n2.note_id = f.dst
            WHERE 1=1{src_clause}{dst_clause}
            ORDER BY f.src, f.dst
            """,
            (*src_params, *dst_params),
        ).fetchall()
        vectors: list[np.ndarray] = []
        fact_rows: list[tuple[str, str, str]] = []
        for row in rows:
            vector = np.frombuffer(row["vec"], dtype=np.float32).copy()
            if vector.shape != (config.EMBED_DIM,):
                continue
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            vectors.append(vector.astype(np.float32))
            fact_rows.append((row["src"], row["dst"], row["text"]))
        self._fact_state = (
            fact_rows,
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.empty((0, config.EMBED_DIM), dtype=np.float32),
        )

    def close(self) -> None:
        self._conn.close()

    def retrieval_snapshot(self) -> "IndexSnapshot":
        if self._knowledge_snapshots is None:
            raise RuntimeError("no active derived knowledge snapshot")
        return self._knowledge_snapshots.get()

    def active_snapshot_id(self) -> str:
        if self._knowledge_snapshots is None:
            return ""
        return self._knowledge_snapshots.get().build_id

    def upsert_note(
        self,
        note: NoteRecord,
        enrichment: Enrichment,
        embedding_units: Sequence[EmbeddingInput],
    ) -> None:
        stored = note.model_copy(update={"gist": _display_line(note)})
        with self._conn:
            # quality_rank is deliberately absent: it is owned by the rank pass
            # (set_quality_ranks) and must survive note upserts untouched.
            self._conn.execute(
                """
                INSERT INTO notes(note_id, title, folder, modified, created, content_hash, text, gist, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    title=excluded.title,
                    folder=excluded.folder,
                    modified=excluded.modified,
                    created=excluded.created,
                    content_hash=excluded.content_hash,
                    text=excluded.text,
                    gist=excluded.gist,
                    source=excluded.source
                """,
                (
                    stored.note_id,
                    stored.title,
                    stored.folder,
                    stored.modified,
                    stored.created or stored.modified,
                    stored.content_hash,
                    stored.text,
                    stored.gist,
                    stored.source,
                ),
            )
            self._conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (stored.note_id,))
            self._conn.execute(
                "INSERT INTO notes_fts(note_id, title, body) VALUES (?, ?, ?)",
                (stored.note_id, stored.title, stored.text),
            )
            self._conn.execute("DELETE FROM aliases WHERE note_id = ?", (stored.note_id,))
            for alias, is_title in self._aliases_for(stored.title, enrichment):
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO aliases(alias_lowercase, note_id, is_title)
                    VALUES (?, ?, ?)
                    """,
                    (alias, stored.note_id, int(is_title)),
                )
            self._conn.execute("DELETE FROM embeddings WHERE note_id = ?", (stored.note_id,))
            for kind, text, vector in embedding_units:
                self._insert_embedding(stored.note_id, kind, text, vector)
            if not enrichment.extraction_failed:
                self._replace_note_extraction_rows(
                    stored.note_id,
                    [
                        (
                            entity.surface,
                            entity.etype,
                            entity.conf,
                            entity.mentions,
                        )
                        for entity in enrichment.entities
                    ],
                    [
                        (
                            relation.head,
                            relation.rel_type,
                            relation.tail,
                            relation.conf,
                        )
                        for relation in enrichment.relations
                    ],
                )
            self._set_last_sync(time.time())
        self._doc_frequency_cache.clear()
        self.load()

    def replace_source_records(
        self,
        source_id: str,
        records: Sequence[NoteRecord],
    ) -> tuple[int, int, int]:
        """Refresh the product/source cache without building a legacy graph."""

        incoming = {
            note.note_id: note.model_copy(update={"source": source_id})
            for note in records
        }
        existing = {note.note_id: note for note in self.notes(source_id)}
        removed = sorted(set(existing) - set(incoming))
        added = sorted(set(incoming) - set(existing))
        changed = sorted(
            note_id
            for note_id in set(incoming) & set(existing)
            if incoming[note_id].content_hash != existing[note_id].content_hash
        )
        replaced = (*removed, *changed)
        with self._conn:
            for note_id in replaced:
                self._conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM aliases WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM note_entities WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM entity_relations WHERE note_id = ?", (note_id,))
                self._conn.execute(
                    "DELETE FROM edges WHERE src = ? OR dst = ?", (note_id, note_id)
                )
            for note_id in (*added, *changed):
                note = incoming[note_id]
                self._conn.execute(
                    """
                    INSERT INTO notes(
                        note_id, title, folder, modified, created, content_hash,
                        text, gist, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note.note_id,
                        note.title,
                        note.folder,
                        note.modified,
                        note.created or note.modified,
                        note.content_hash,
                        note.text,
                        _display_line(note),
                        source_id,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO notes_fts(note_id, title, body) VALUES (?, ?, ?)",
                    (note.note_id, note.title, note.text),
                )
            self._set_last_sync(time.time())
        self._doc_frequency_cache.clear()
        self.load()
        return len(added), len(changed), len(removed)

    def apply_slim_enrichment(
        self,
        note_id: str,
        enrichment: Enrichment,
        utterance_units: Sequence[EmbeddingInput],
    ) -> None:
        with self._conn:
            row = self._conn.execute(
                "SELECT title, content_hash FROM notes WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if row is None:
                return
            self._conn.execute("DELETE FROM aliases WHERE note_id = ?", (note_id,))
            for alias, is_title in self._aliases_for(row["title"], enrichment):
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO aliases(alias_lowercase, note_id, is_title)
                    VALUES (?, ?, ?)
                    """,
                    (alias, note_id, int(is_title)),
                )
            self._conn.execute(
                "DELETE FROM embeddings WHERE note_id = ? AND kind = 'utterance'",
                (note_id,),
            )
            for kind, text, vector in utterance_units:
                if kind != "utterance":
                    raise ValueError(f"slim enrichment only accepts utterance units: {kind}")
                self._insert_embedding(note_id, kind, text, vector)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO enrich_cache(note_id, content_hash, enrichment_json)
                VALUES (?, ?, ?)
                """,
                (
                    note_id,
                    row["content_hash"],
                    Extraction(
                        entities=enrichment.entities,
                        relations=enrichment.relations,
                        failed=enrichment.extraction_failed,
                    ).model_dump_json(),
                ),
            )
            self._set_last_sync(time.time())
        self.load()

    def delete_note(self, note_id: str) -> None:
        with self._conn:
            touched_entity_ids = self._entity_ids_touched_by_note(note_id)
            self._conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM aliases WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM note_entities WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM entity_relations WHERE note_id = ?", (note_id,))
            self._conn.execute(
                "DELETE FROM edges WHERE src = ? OR dst = ?",
                (note_id, note_id),
            )
            self._recompute_entity_df(touched_entity_ids)
            self._set_last_sync(time.time())
        self._doc_frequency_cache.clear()
        self.load()

    def cache_extraction(
        self, note_id: str, content_hash: str, extraction: Extraction
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO enrich_cache(note_id, content_hash, enrichment_json)
                VALUES (?, ?, ?)
                """,
                (note_id, content_hash, extraction.model_dump_json()),
            )

    def cached_extraction(self, note_id: str, content_hash: str) -> Extraction | None:
        row = self._conn.execute(
            """
            SELECT enrichment_json FROM enrich_cache
            WHERE note_id = ? AND content_hash = ?
            """,
            (note_id, content_hash),
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["enrichment_json"])
            if not isinstance(data, dict):
                return None
            if set(data) - set(Extraction.model_fields):
                return None
            return Extraction.model_validate(data, strict=False)
        except Exception:
            return None

    def cache_enrichment(
        self, note_id: str, content_hash: str, enrichment: Enrichment
    ) -> None:
        self.cache_extraction(
            note_id,
            content_hash,
            Extraction(
                entities=enrichment.entities,
                relations=enrichment.relations,
                failed=enrichment.extraction_failed,
            ),
        )

    def cached_enrichment(self, note_id: str, content_hash: str) -> Enrichment | None:
        extraction = self.cached_extraction(note_id, content_hash)
        if extraction is None:
            return None
        return Enrichment(
            entities=extraction.entities,
            relations=extraction.relations,
            extraction_failed=extraction.failed,
        )

    def set_user_card(self, text: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO meta(key, value, updated)
                VALUES ('user_card', ?, ?)
                """,
                (text, time.time()),
            )

    def user_card(self) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'user_card'"
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_source_enabled(self, source: str, enabled: bool) -> None:
        """Ledger switch: disabled sources keep their rows but drop out of the
        active index (graph, search, aliases, counts) until re-enabled."""
        disabled = set(self._disabled_sources)
        if enabled:
            disabled.discard(source)
        else:
            disabled.add(source)
        if disabled == self._disabled_sources:
            return
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO meta(key, value, updated)
                VALUES ('disabled_sources', ?, ?)
                """,
                (json.dumps(sorted(disabled)), time.time()),
            )
        self._disabled_sources = disabled
        self._doc_frequency_cache.clear()
        self.load()

    def disabled_sources(self) -> set[str]:
        return set(self._disabled_sources)

    def _load_disabled_sources(self) -> set[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'disabled_sources'"
        ).fetchone()
        if row is None:
            return set()
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError:
            return set()
        return {str(item) for item in value} if isinstance(value, list) else set()

    def _active_sources_clause(self, column: str) -> tuple[str, tuple]:
        if not self._disabled_sources:
            return "", ()
        placeholders = ",".join("?" for _ in self._disabled_sources)
        return f" AND {column} NOT IN ({placeholders})", tuple(sorted(self._disabled_sources))

    def connector_state(self, connector_id: str) -> dict:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (_connector_meta_key(connector_id),),
        ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def set_connector_state(self, connector_id: str, state: dict) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO meta(key, value, updated)
                VALUES (?, ?, ?)
                """,
                (_connector_meta_key(connector_id), json.dumps(state, sort_keys=True), time.time()),
            )

    def dense_search(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, float]]:
        return [
            (note_id, score)
            for note_id, score, _kind in self.dense_search_with_kinds(
                text,
                k,
                prompt=prompt,
            )
        ]

    def dense_search_with_kinds(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, float, EmbeddingKind]]:
        embedding_rows, matrix = self._dense_snapshot()
        if k <= 0 or matrix.size == 0:
            return []
        query = _embed_queries_with_prompt([text], prompt)[0].astype(np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        scores = matrix @ query
        per_note: dict[str, tuple[float, EmbeddingKind, int]] = {}
        for index, ((_, note_id, kind, _), score) in enumerate(
            zip(embedding_rows, scores, strict=True)
        ):
            current = per_note.get(note_id)
            value = float(score)
            if current is None or value > current[0]:
                per_note[note_id] = (value, kind, index)
        ordered = sorted(
            per_note.items(),
            key=lambda item: (-item[1][0], item[1][2], item[0]),
        )[:k]
        return [(note_id, score, kind) for note_id, (score, kind, _index) in ordered]

    def mask_embedding_kind(self, kind: EmbeddingKind) -> int:
        if kind not in {"chunk", "gist", "utterance"}:
            raise ValueError(f"unknown embedding kind: {kind}")
        rows, matrix = self._dense_state
        keep_indexes = [index for index, row in enumerate(rows) if row[2] != kind]
        removed = len(rows) - len(keep_indexes)
        if removed == 0:
            return 0
        kept_rows = [rows[index] for index in keep_indexes]
        kept_matrix = (
            matrix[keep_indexes].astype(np.float32, copy=True)
            if keep_indexes
            else np.empty((0, config.EMBED_DIM), dtype=np.float32)
        )
        self._dense_state = (kept_rows, kept_matrix)
        return removed

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = self._fts_query(text)
        if not query:
            return []
        clause, params = self._active_sources_clause("source")
        try:
            rows = self._conn.execute(
                f"""
                SELECT note_id, -bm25(notes_fts) AS score
                FROM notes_fts
                WHERE notes_fts MATCH ?
                AND note_id IN (SELECT note_id FROM notes WHERE 1=1{clause})
                ORDER BY score DESC
                LIMIT ?
                """,
                (query, *params, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(row["note_id"], float(row["score"])) for row in rows]

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        src_clause, src_params = self._active_sources_clause("n1.source")
        dst_clause, dst_params = self._active_sources_clause("n2.source")
        rows = self._conn.execute(
            f"""
            SELECT src, dst, weight FROM edges
            JOIN notes n1 ON n1.note_id = edges.src
            JOIN notes n2 ON n2.note_id = edges.dst
            WHERE (src = ? OR dst = ?){src_clause}{dst_clause}
            ORDER BY weight DESC
            """,
            (note_id, note_id, *src_params, *dst_params),
        ).fetchall()
        return [
            (row["dst"] if row["src"] == note_id else row["src"], float(row["weight"]))
            for row in rows
        ]

    def neighbor_edges(self, note_id: str) -> list[tuple[str, float, str]]:
        """(other_id, weight, fact) for active neighbors — the fact-aware
        variant of neighbors(), used where edge facts are surfaced."""
        src_clause, src_params = self._active_sources_clause("n1.source")
        dst_clause, dst_params = self._active_sources_clause("n2.source")
        rows = self._conn.execute(
            f"""
            SELECT src, dst, weight, fact FROM edges
            JOIN notes n1 ON n1.note_id = edges.src
            JOIN notes n2 ON n2.note_id = edges.dst
            WHERE (src = ? OR dst = ?){src_clause}{dst_clause}
            ORDER BY weight DESC
            """,
            (note_id, note_id, *src_params, *dst_params),
        ).fetchall()
        return [
            (
                row["dst"] if row["src"] == note_id else row["src"],
                float(row["weight"]),
                row["fact"],
            )
            for row in rows
        ]

    def fact_dense_search(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, str, float, str]]:
        """Dense search over adjudicated edge facts: (src, dst, score, fact_text).

        HippoRAG-2-style query-to-fact matching; callers credit both endpoint
        notes (spec 2026-07-08 §1.3/§4.1)."""
        rows, matrix = self._fact_state
        if k <= 0 or matrix.size == 0:
            return []
        query = _embed_queries_with_prompt([text], prompt)[0].astype(np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        scores = matrix @ query
        order = np.argsort(scores)[::-1][:k]
        return [(rows[i][0], rows[i][1], float(scores[i]), rows[i][2]) for i in order]

    def fact_embedding_hashes(self) -> dict[tuple[str, str], str]:
        """(src, dst) → fact_hash for every stored fact embedding (content-addressed refresh)."""
        rows = self._conn.execute(
            "SELECT src, dst, fact_hash FROM fact_embeddings"
        ).fetchall()
        return {(row["src"], row["dst"]): row["fact_hash"] for row in rows}

    def upsert_fact_embedding(
        self, src: str, dst: str, fact_hash: str, text: str, vector: np.ndarray
    ) -> None:
        """Store one fact embedding (src < dst). Caller batches, then load_facts()."""
        values = np.asarray(vector, dtype=np.float32)
        if values.shape != (config.EMBED_DIM,):
            raise ValueError(f"fact embedding must have shape ({config.EMBED_DIM},)")
        norm = float(np.linalg.norm(values))
        if norm > 0:
            values = values / norm
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO fact_embeddings(src, dst, fact_hash, text, vec)
                VALUES (?, ?, ?, ?, ?)
                """,
                (src, dst, fact_hash, text, values.astype(np.float32).tobytes()),
            )

    def prune_fact_embeddings(self, valid_keys: set[tuple[str, str]]) -> None:
        """Drop fact embeddings whose (src, dst) is no longer a fact-carrying edge."""
        stored = self.fact_embedding_hashes()
        stale = [key for key in stored if key not in valid_keys]
        if not stale:
            return
        with self._conn:
            for src, dst in stale:
                self._conn.execute(
                    "DELETE FROM fact_embeddings WHERE src = ? AND dst = ?", (src, dst)
                )

    def set_quality_ranks(self, ranks: dict[str, float]) -> None:
        """Write the rebuild-time quality ranks (log-median transformed, clipped)."""
        with self._conn:
            self._conn.execute("UPDATE notes SET quality_rank = 0.0")
            self._conn.executemany(
                "UPDATE notes SET quality_rank = ? WHERE note_id = ?",
                [(float(rank), note_id) for note_id, rank in ranks.items()],
            )

    def quality_ranks(self) -> dict[str, float]:
        rows = self._conn.execute("SELECT note_id, quality_rank FROM notes").fetchall()
        return {row["note_id"]: float(row["quality_rank"]) for row in rows}

    def replace_note_extraction(
        self,
        note_id: str,
        entities: list[tuple[str, str, float, int]],
        relations: list[tuple[str, str, str, float]],
    ) -> None:
        """Replace GLiNER extraction rows for one note and refresh entity df."""
        with self._conn:
            self._replace_note_extraction_rows(note_id, entities, relations)

    def entity_rows(self) -> list[tuple[int, str, str, int]]:
        rows = self._conn.execute(
            "SELECT entity_id, canonical, etype, df FROM entities ORDER BY entity_id"
        ).fetchall()
        return [
            (int(row["entity_id"]), row["canonical"], row["etype"], int(row["df"]))
            for row in rows
        ]

    def note_entity_rows(self) -> list[tuple[str, int, float, int]]:
        rows = self._conn.execute(
            """
            SELECT note_id, entity_id, conf, count
            FROM note_entities ORDER BY note_id, entity_id
            """
        ).fetchall()
        return [
            (row["note_id"], int(row["entity_id"]), float(row["conf"]), int(row["count"]))
            for row in rows
        ]

    def relation_rows(self) -> list[tuple[str, int, str, int, float]]:
        rows = self._conn.execute(
            """
            SELECT note_id, src_entity, rel_type, dst_entity, conf
            FROM entity_relations ORDER BY note_id, src_entity, rel_type, dst_entity
            """
        ).fetchall()
        return [
            (
                row["note_id"],
                int(row["src_entity"]),
                row["rel_type"],
                int(row["dst_entity"]),
                float(row["conf"]),
            )
            for row in rows
        ]

    def replace_entity_surfaces(
        self, rows: Sequence[tuple[int, str]],
    ) -> None:
        """Replace the query-time entity surface lookup table."""
        with self._conn:
            self._conn.execute("DELETE FROM entity_surfaces")
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO entity_surfaces(entity_id, surface)
                VALUES (?, ?)
                """,
                [
                    (int(entity_id), surface.strip())
                    for entity_id, surface in rows
                    if surface.strip()
                ],
            )
        self.load_entity_surfaces()

    def load_entity_surfaces(self) -> None:
        rows = self._conn.execute(
            """
            SELECT entity_id, surface
            FROM entity_surfaces
            ORDER BY entity_id, surface
            """
        ).fetchall()
        surface_state: dict[str, int] = {}
        for row in rows:
            surface = str(row["surface"]).strip()
            if not surface:
                continue
            surface_state.setdefault(surface, int(row["entity_id"]))
        self._entity_surface_state = surface_state

    def entity_by_surface(self, surface: str) -> int | None:
        cleaned = surface.strip()
        if not cleaned:
            return None
        for candidate in _entity_surface_candidates(cleaned):
            entity_id = self._entity_surface_state.get(candidate)
            if entity_id is not None:
                return entity_id
        return None

    def relations_between(
        self, src_note: str, dst_note: str
    ) -> list[tuple[str, str, str, float]]:
        rows = self._conn.execute(
            """
            SELECT src.canonical AS head_canonical,
                   er.rel_type AS rel_type,
                   dst.canonical AS tail_canonical,
                   MAX(er.conf) AS conf
            FROM entity_relations er
            JOIN entities src ON src.entity_id = er.src_entity
            JOIN entities dst ON dst.entity_id = er.dst_entity
            WHERE (
                EXISTS (
                    SELECT 1 FROM note_entities ne
                    WHERE ne.note_id = ? AND ne.entity_id = er.src_entity
                )
                AND EXISTS (
                    SELECT 1 FROM note_entities ne
                    WHERE ne.note_id = ? AND ne.entity_id = er.dst_entity
                )
            ) OR (
                EXISTS (
                    SELECT 1 FROM note_entities ne
                    WHERE ne.note_id = ? AND ne.entity_id = er.src_entity
                )
                AND EXISTS (
                    SELECT 1 FROM note_entities ne
                    WHERE ne.note_id = ? AND ne.entity_id = er.dst_entity
                )
            )
            GROUP BY er.src_entity, er.rel_type, er.dst_entity
            ORDER BY head_canonical, rel_type, tail_canonical
            """,
            (src_note, dst_note, dst_note, src_note),
        ).fetchall()
        return [
            (
                row["head_canonical"],
                row["rel_type"],
                row["tail_canonical"],
                float(row["conf"]),
            )
            for row in rows
        ]

    def relations_between_scoped(
        self, src_note: str, dst_note: str
    ) -> list[tuple[str, str, str, float]]:
        rows = self._conn.execute(
            """
            SELECT src.canonical AS head_canonical,
                   er.rel_type AS rel_type,
                   dst.canonical AS tail_canonical,
                   MAX(er.conf) AS conf
            FROM entity_relations er
            JOIN entities src ON src.entity_id = er.src_entity
            JOIN entities dst ON dst.entity_id = er.dst_entity
            WHERE er.note_id IN (?, ?)
              AND (
                (
                    EXISTS (
                        SELECT 1 FROM note_entities ne
                        WHERE ne.note_id = ? AND ne.entity_id = er.src_entity
                    )
                    AND EXISTS (
                        SELECT 1 FROM note_entities ne
                        WHERE ne.note_id = ? AND ne.entity_id = er.dst_entity
                    )
                ) OR (
                    EXISTS (
                        SELECT 1 FROM note_entities ne
                        WHERE ne.note_id = ? AND ne.entity_id = er.src_entity
                    )
                    AND EXISTS (
                        SELECT 1 FROM note_entities ne
                        WHERE ne.note_id = ? AND ne.entity_id = er.dst_entity
                    )
                )
              )
            GROUP BY er.src_entity, er.rel_type, er.dst_entity
            ORDER BY head_canonical, rel_type, tail_canonical
            """,
            (src_note, dst_note, src_note, dst_note, dst_note, src_note),
        ).fetchall()
        return [
            (
                row["head_canonical"],
                row["rel_type"],
                row["tail_canonical"],
                float(row["conf"]),
            )
            for row in rows
        ]

    def set_ingest_salience(self, scores: dict[str, float]) -> None:
        with self._conn:
            self._conn.execute("UPDATE notes SET ingest_salience = 1.0")
            self._conn.executemany(
                "UPDATE notes SET ingest_salience = ? WHERE note_id = ?",
                [(float(score), note_id) for note_id, score in scores.items()],
            )

    def ingest_salience(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT note_id, ingest_salience FROM notes"
        ).fetchall()
        return {row["note_id"]: float(row["ingest_salience"]) for row in rows}

    def set_value_scores(self, scores: dict[str, float]) -> None:
        with self._conn:
            self._conn.execute("UPDATE notes SET value_score = 0.0")
            self._conn.executemany(
                "UPDATE notes SET value_score = ? WHERE note_id = ?",
                [(float(score), note_id) for note_id, score in scores.items()],
            )

    def value_scores(self) -> dict[str, float]:
        rows = self._conn.execute("SELECT note_id, value_score FROM notes").fetchall()
        return {row["note_id"]: float(row["value_score"]) for row in rows}

    def graph_version(self) -> int:
        """Monotonic counter bumped by replace_edges — lets live consumers know
        when to rebuild cached traversal matrices."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'graph_version'"
        ).fetchone()
        return int(row["value"]) if row is not None else 0

    # ── chat persistence (spec 2026-07-08-chat-knowledge-base §1.5) ────
    def create_chat(self, chat_id: str, *, now: float) -> ChatMeta:
        with self._conn:
            self._conn.execute(
                "INSERT INTO chats(chat_id, title, created, updated) VALUES (?, '', ?, ?)",
                (chat_id, now, now),
            )
        return ChatMeta(chat_id=chat_id, created=now, updated=now)

    def chat(self, chat_id: str) -> ChatMeta | None:
        row = self._conn.execute(
            """
            SELECT c.chat_id, c.title, c.created, c.updated,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = c.chat_id) AS message_count
            FROM chats c WHERE c.chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        return ChatMeta(**dict(row)) if row is not None else None

    def chats(self) -> list[ChatMeta]:
        rows = self._conn.execute(
            """
            SELECT c.chat_id, c.title, c.created, c.updated,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = c.chat_id) AS message_count
            FROM chats c ORDER BY c.updated DESC
            """
        ).fetchall()
        return [ChatMeta(**dict(row)) for row in rows]

    def rename_chat(self, chat_id: str, title: str, *, now: float) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE chats SET title = ?, updated = ? WHERE chat_id = ?",
                (title, now, chat_id),
            )

    def delete_chat(self, chat_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
            self._conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))

    def chat_registry(self, chat_id: str) -> dict:
        """Per-chat retrieval state: N#/F# registries + seed accumulator."""
        row = self._conn.execute(
            "SELECT registry_json FROM chats WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return json.loads(row["registry_json"]) if row is not None else {}

    def set_chat_registry(self, chat_id: str, registry: dict) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE chats SET registry_json = ? WHERE chat_id = ?",
                (json.dumps(registry), chat_id),
            )

    def add_chat_message(self, message: ChatMessage) -> int:
        """Persist a message; returns its rowid. Assistant rows may start
        empty (created at ack, finished via finish_chat_message)."""
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO chat_messages(chat_id, role, content, thinking, evidence_json, retrieved_ids_json,
                                          consulted_ids_json, cited_ids_json, original_query,
                                          rewritten_query, retrieval_decision, retrieval_confidence,
                                          active_build_id, activity_json,
                                          usage_json, cancelled, created)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.chat_id,
                    message.role,
                    message.content,
                    message.thinking,
                    message.evidence.model_dump_json() if message.evidence is not None else "null",
                    json.dumps(message.retrieved_ids),
                    json.dumps(message.consulted_ids),
                    json.dumps(message.cited_ids),
                    message.original_query,
                    message.rewritten_query,
                    message.retrieval_decision,
                    message.retrieval_confidence,
                    message.active_build_id,
                    json.dumps([item.model_dump(mode="json") for item in message.activity]),
                    message.usage.model_dump_json(),
                    int(message.cancelled),
                    message.created,
                ),
            )
            self._conn.execute(
                "UPDATE chats SET updated = ? WHERE chat_id = ?",
                (message.created, message.chat_id),
            )
            return int(cursor.lastrowid)

    def finish_chat_message(self, message: ChatMessage) -> None:
        """Overwrite the streamed-into assistant row (matched by message.id)."""
        with self._conn:
            self._conn.execute(
                """
                UPDATE chat_messages SET content = ?, thinking = ?, evidence_json = ?, retrieved_ids_json = ?,
                    consulted_ids_json = ?, cited_ids_json = ?, original_query = ?,
                    rewritten_query = ?, retrieval_decision = ?, retrieval_confidence = ?,
                    active_build_id = ?, activity_json = ?,
                    usage_json = ?, cancelled = ?
                WHERE id = ?
                """,
                (
                    message.content,
                    message.thinking,
                    message.evidence.model_dump_json() if message.evidence is not None else "null",
                    json.dumps(message.retrieved_ids),
                    json.dumps(message.consulted_ids),
                    json.dumps(message.cited_ids),
                    message.original_query,
                    message.rewritten_query,
                    message.retrieval_decision,
                    message.retrieval_confidence,
                    message.active_build_id,
                    json.dumps([item.model_dump(mode="json") for item in message.activity]),
                    message.usage.model_dump_json(),
                    int(message.cancelled),
                    message.id,
                ),
            )

    def chat_messages(self, chat_id: str) -> list[ChatMessage]:
        rows = self._conn.execute(
            """
            SELECT id, chat_id, role, content, thinking, evidence_json,
                   retrieved_ids_json, consulted_ids_json, cited_ids_json,
                   original_query, rewritten_query, retrieval_decision,
                   retrieval_confidence, active_build_id,
                   activity_json, usage_json, cancelled, created
            FROM chat_messages WHERE chat_id = ? ORDER BY id
            """,
            (chat_id,),
        ).fetchall()
        return [
            ChatMessage(
                id=int(row["id"]),
                chat_id=row["chat_id"],
                role=row["role"],
                content=row["content"],
                thinking=row["thinking"],
                evidence=(
                    EvidencePackPayload(**json.loads(row["evidence_json"]))
                    if row["evidence_json"] and row["evidence_json"] != "null"
                    else None
                ),
                retrieved_ids=list(json.loads(row["retrieved_ids_json"])),
                consulted_ids=list(json.loads(row["consulted_ids_json"])),
                cited_ids=list(json.loads(row["cited_ids_json"])),
                original_query=str(row["original_query"]),
                rewritten_query=str(row["rewritten_query"]),
                retrieval_decision=str(row["retrieval_decision"]),
                retrieval_confidence=float(row["retrieval_confidence"]),
                active_build_id=str(row["active_build_id"]),
                activity=[ChatActivityStep(**item) for item in json.loads(row["activity_json"])],
                usage=ChatUsage(**json.loads(row["usage_json"] or "{}")),
                cancelled=bool(row["cancelled"]),
                created=float(row["created"]),
            )
            for row in rows
        ]

    def delete_chat_message(self, message_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chat_messages WHERE id = ?", (message_id,))

    def term_vectors(self, terms: Sequence[str]) -> dict[str, np.ndarray]:
        """Cached embeddings for term strings (synonym clustering)."""
        found: dict[str, np.ndarray] = {}
        for term in terms:
            row = self._conn.execute(
                "SELECT vec FROM term_vectors WHERE term = ?", (term,)
            ).fetchone()
            if row is None:
                continue
            vector = np.frombuffer(row["vec"], dtype=np.float32).copy()
            if vector.shape == (config.EMBED_DIM,):
                found[term] = vector
        return found

    def set_term_vectors(self, vectors: dict[str, np.ndarray]) -> None:
        with self._conn:
            for term, vector in vectors.items():
                values = np.asarray(vector, dtype=np.float32)
                if values.shape != (config.EMBED_DIM,):
                    raise ValueError(f"term vector must have shape ({config.EMBED_DIM},)")
                self._conn.execute(
                    "INSERT OR REPLACE INTO term_vectors(term, vec) VALUES (?, ?)",
                    (term, values.tobytes()),
                )

    def edge_judgments(self) -> dict[tuple[str, str], tuple[str, bool, str]]:
        """(src, dst) -> (sig, related, fact) for every cached adjudication."""
        rows = self._conn.execute(
            "SELECT src, dst, sig, related, fact FROM edge_judgments"
        ).fetchall()
        return {
            (row["src"], row["dst"]): (row["sig"], bool(row["related"]), row["fact"])
            for row in rows
        }

    def set_edge_judgment(
        self, src: str, dst: str, sig: str, related: bool, fact: str
    ) -> None:
        src, dst = sorted((src, dst))
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO edge_judgments(src, dst, sig, related, fact, judged_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (src, dst, sig, int(related), fact, time.time()),
            )

    def prune_edge_judgments(self, valid_note_ids: set[str]) -> None:
        """Drop judgments referencing deleted notes."""
        rows = self._conn.execute("SELECT src, dst FROM edge_judgments").fetchall()
        stale = [
            (row["src"], row["dst"])
            for row in rows
            if row["src"] not in valid_note_ids or row["dst"] not in valid_note_ids
        ]
        if not stale:
            return
        with self._conn:
            for src, dst in stale:
                self._conn.execute(
                    "DELETE FROM edge_judgments WHERE src = ? AND dst = ?", (src, dst)
                )

    def note(self, note_id: str) -> NoteRecord | None:
        row = self._conn.execute(
            """
            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
                   source, quality_rank
            FROM notes WHERE note_id = ?
            """,
            (note_id,),
        ).fetchone()
        if row is None:
            return None
        return NoteRecord(**dict(row))

    def notes(self, source: str | None = None) -> list[NoteRecord]:
        if source is not None:
            rows = self._conn.execute(
                """
                SELECT note_id, title, folder, modified, created, content_hash, text, gist,
                       source, quality_rank
                FROM notes WHERE source = ? ORDER BY note_id
                """,
                (source,),
            ).fetchall()
            return [NoteRecord(**dict(row)) for row in rows]
        rows = self._conn.execute(
            """
            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
                   source, quality_rank
            FROM notes ORDER BY note_id
            """
        ).fetchall()
        return [NoteRecord(**dict(row)) for row in rows]

    def note_ids(self, source: str | None = None) -> set[str]:
        if source is not None:
            rows = self._conn.execute(
                "SELECT note_id FROM notes WHERE source = ?",
                (source,),
            ).fetchall()
            return {row["note_id"] for row in rows}
        rows = self._conn.execute("SELECT note_id FROM notes").fetchall()
        return {row["note_id"] for row in rows}

    def note_count_by_source(self) -> dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT source, COUNT(*) AS count
            FROM notes GROUP BY source ORDER BY source
            """
        ).fetchall()
        return {row["source"]: int(row["count"]) for row in rows}

    def preview(self, note_id: str) -> NotePreview | None:
        note = self.note(note_id)
        if note is None:
            return None
        neighbors: list[NeighborRef] = []
        for neighbor_id, weight, fact in self.neighbor_edges(note_id):
            neighbor = self.note(neighbor_id)
            if neighbor is None:
                continue
            neighbors.append(
                NeighborRef(
                    note_id=neighbor.note_id,
                    title=neighbor.title,
                    source=neighbor.source,
                    weight=weight,
                    fact=fact,
                )
            )
        neighbors.sort(key=lambda item: item.weight, reverse=True)
        return NotePreview(
            note_id=note.note_id,
            title=note.title,
            source=note.source,
            gist=note.gist,
            text=note.text,
            neighbors=neighbors,
        )

    def best_chunk(self, note_id: str, query_text: str) -> str:
        """Embedding unit of the note most similar to query_text for rerank docs.

        The method name is historical: candidates may be chunks, legacy stored
        summary rows, or synthetic utterances. Utterance winners are labeled as
        spoken references so rerank documents read naturally.
        """
        embedding_rows, matrix = self._dense_state
        rows = [
            (row_id, kind, text)
            for row_id, row_note_id, kind, text in embedding_rows
            if row_note_id == note_id
        ]
        if not rows:
            note = self.note(note_id)
            return note.text if note is not None else ""
        if matrix.size == 0:
            return _rerank_evidence_text(rows[0][1], rows[0][2])
        query = embed_queries([query_text])[0].astype(np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return _rerank_evidence_text(rows[0][1], rows[0][2])
        query = query / norm
        row_ids = {row_id: (kind, text) for row_id, kind, text in rows}
        best_kind = rows[0][1]
        best_text = rows[0][2]
        best_score = -1.0
        for index, (row_id, _, _, _) in enumerate(embedding_rows):
            if row_id not in row_ids:
                continue
            score = float(matrix[index] @ query)
            if score > best_score:
                best_score = score
                best_kind, best_text = row_ids[row_id]
        return _rerank_evidence_text(best_kind, best_text)

    def aliases(self) -> list[tuple[str, str, bool]]:
        clause, params = self._active_sources_clause("n.source")
        rows = self._conn.execute(
            f"""
            SELECT a.alias_lowercase AS alias_lowercase, a.note_id AS note_id, a.is_title AS is_title
            FROM aliases a JOIN notes n ON n.note_id = a.note_id
            WHERE 1=1{clause}
            ORDER BY a.note_id, a.is_title DESC, a.alias_lowercase
            """,
            params,
        ).fetchall()
        return [
            (row["alias_lowercase"], row["note_id"], bool(row["is_title"]))
            for row in rows
        ]

    def doc_frequency(self, term: str) -> int:
        key = term.strip().lower()
        if not key:
            return 0
        cached = self._doc_frequency_cache.get(key)
        if cached is not None:
            return cached
        escaped = key.replace('"', '""')
        query = f'"{escaped}"'
        clause, params = self._active_sources_clause("source")
        try:
            row = self._conn.execute(
                f"""
                SELECT COUNT(DISTINCT note_id) AS count
                FROM notes_fts
                WHERE notes_fts MATCH ?
                AND note_id IN (SELECT note_id FROM notes WHERE 1=1{clause})
                """,
                (query, *params),
            ).fetchone()
        except sqlite3.OperationalError:
            count = 0
        else:
            count = int(row["count"]) if row is not None else 0
        self._doc_frequency_cache[key] = count
        return count

    def graph(self) -> GraphResponse:
        active_notes = [
            note for note in self.notes() if note.source not in self._disabled_sources
        ]
        active_ids = {note.note_id for note in active_notes}
        degree: dict[str, int] = defaultdict(int)
        edge_rows = self._conn.execute(
            """
            SELECT src, dst, type, weight, shared_json, fact, fact_provenance
            FROM edges ORDER BY weight DESC, src, dst
            """
        ).fetchall()
        edges: list[GraphEdge] = []
        for row in edge_rows:
            if row["src"] not in active_ids or row["dst"] not in active_ids:
                continue
            degree[row["src"]] += 1
            degree[row["dst"]] += 1
            edges.append(
                GraphEdge(
                    src=row["src"],
                    dst=row["dst"],
                    type=row["type"],
                    weight=float(row["weight"]),
                    shared=json.loads(row["shared_json"]),
                    fact=row["fact"],
                    fact_provenance=row["fact_provenance"],
                )
            )
        nodes = [
            GraphNode(
                note_id=note.note_id,
                title=note.title,
                degree=degree[note.note_id],
                source=note.source,
                gist=note.gist,
            )
            for note in active_notes
        ]
        return GraphResponse(nodes=nodes, edges=edges)

    def status(self) -> tuple[int, float | None]:
        clause, params = self._active_sources_clause("source")
        row = self._conn.execute(
            f"SELECT COUNT(*) AS count FROM notes WHERE 1=1{clause}", params
        ).fetchone()
        metadata = self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_sync_ts'"
        ).fetchone()
        last_sync = float(metadata["value"]) if metadata is not None else None
        return int(row["count"]), last_sync

    def replace_edges(
        self, edges: Sequence[GraphEdge], *, incident_note_id: str | None = None
    ) -> None:
        with self._conn:
            if incident_note_id is None:
                self._conn.execute("DELETE FROM edges")
            else:
                self._conn.execute(
                    "DELETE FROM edges WHERE src = ? OR dst = ?",
                    (incident_note_id, incident_note_id),
                )
            for edge in edges:
                src, dst = sorted((edge.src, edge.dst))
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO edges(
                        src, dst, type, weight, shared_json, fact, fact_provenance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        src,
                        dst,
                        edge.type,
                        float(edge.weight),
                        json.dumps(sorted(set(edge.shared))),
                        edge.fact,
                        edge.fact_provenance,
                    ),
                )
            self._conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('graph_version', '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                """
            )

    def _dense_snapshot(self) -> tuple[list[tuple[int, str, EmbeddingKind, str]], np.ndarray]:
        return self._dense_state

    def _embedding_rows_raw(
        self, *, stale_only: bool = False
    ) -> list[tuple[int, str, EmbeddingKind, str]]:
        where = "WHERE length(vec) != ?" if stale_only else ""
        params = (config.EMBED_DIM * 4,) if stale_only else ()
        rows = self._conn.execute(
            f"""
            SELECT id, note_id, kind, text
            FROM embeddings
            {where}
            ORDER BY id
            """,
            params,
        ).fetchall()
        return [
            (int(row["id"]), row["note_id"], row["kind"], row["text"])
            for row in rows
        ]

    def update_embedding_vec(self, row_id: int, vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype=np.float32)
        if values.shape != (config.EMBED_DIM,):
            raise ValueError(f"embedding must have shape ({config.EMBED_DIM},)")
        norm = float(np.linalg.norm(values))
        if norm > 0:
            values = values / norm
        with self._conn:
            self._conn.execute(
                "UPDATE embeddings SET vec = ? WHERE id = ?",
                (values.astype(np.float32).tobytes(), int(row_id)),
            )

    def embedding_vectors(self, note_id: str, kind: EmbeddingKind | None = None) -> list[np.ndarray]:
        rows = self._conn.execute(
            """
            SELECT kind, vec FROM embeddings
            WHERE note_id = ?
            ORDER BY id
            """,
            (note_id,),
        ).fetchall()
        vectors: list[np.ndarray] = []
        for row in rows:
            if kind is not None and row["kind"] != kind:
                continue
            vector = np.frombuffer(row["vec"], dtype=np.float32).copy()
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            vectors.append(vector.astype(np.float32))
        return vectors

    def entity_surfaces_for(self, note_id: str, limit: int = 3) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT e.canonical
            FROM note_entities ne
            JOIN entities e ON e.entity_id = ne.entity_id
            WHERE ne.note_id = ?
            ORDER BY ne.conf DESC, e.entity_id ASC
            LIMIT ?
            """,
            (note_id, max(0, int(limit))),
        ).fetchall()
        return [row["canonical"] for row in rows]

    def enrichment_for(self, note_id: str) -> Enrichment | None:
        note = self.note(note_id)
        if note is None:
            return None
        alias_rows = self._conn.execute(
            "SELECT alias_lowercase FROM aliases WHERE note_id = ? AND is_title = 0",
            (note_id,),
        ).fetchall()
        entity_rows = self._conn.execute(
            """
            SELECT e.canonical, e.etype, ne.conf, ne.count
            FROM note_entities ne
            JOIN entities e ON e.entity_id = ne.entity_id
            WHERE ne.note_id = ?
            ORDER BY e.entity_id
            """,
            (note_id,),
        ).fetchall()
        relation_rows = self._conn.execute(
            """
            SELECT src.canonical AS head,
                   er.rel_type AS rel_type,
                   dst.canonical AS tail,
                   er.conf AS conf
            FROM entity_relations er
            JOIN entities src ON src.entity_id = er.src_entity
            JOIN entities dst ON dst.entity_id = er.dst_entity
            WHERE er.note_id = ?
            ORDER BY er.rowid
            """,
            (note_id,),
        ).fetchall()
        cached = self.cached_extraction(note_id, note.content_hash)
        return Enrichment(
            gist=note.gist,
            utterances=[],
            aliases=[row["alias_lowercase"] for row in alias_rows],
            entities=[
                EntityMention(
                    surface=row["canonical"],
                    etype=row["etype"],
                    conf=float(row["conf"]),
                    mentions=int(row["count"]),
                )
                for row in entity_rows
            ],
            relations=[
                RelationTriple(
                    head=row["head"],
                    rel_type=row["rel_type"],
                    tail=row["tail"],
                    conf=float(row["conf"]),
                )
                for row in relation_rows
            ],
            extraction_failed=bool(cached.failed) if cached is not None else False,
        )

    def rebuild_graph(self) -> None:
        from .graph import rebuild_all

        rebuild_all(self)

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    folder TEXT NOT NULL,
                    modified REAL NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    gist TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'apple_notes'
                );

                CREATE TABLE IF NOT EXISTS aliases (
                    alias_lowercase TEXT NOT NULL,
                    note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
                    is_title INTEGER NOT NULL,
                    PRIMARY KEY(alias_lowercase, note_id)
                );

                CREATE TABLE IF NOT EXISTS enrich_cache (
                    note_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    enrichment_json TEXT NOT NULL,
                    PRIMARY KEY(note_id, content_hash)
                );

                CREATE TABLE IF NOT EXISTS entities(
                    entity_id INTEGER PRIMARY KEY,
                    canonical TEXT NOT NULL,
                    etype TEXT NOT NULL,
                    df INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(canonical, etype)
                );

                CREATE TABLE IF NOT EXISTS entity_surfaces(
                    entity_id INTEGER NOT NULL,
                    surface TEXT NOT NULL,
                    PRIMARY KEY(entity_id, surface)
                );

                CREATE TABLE IF NOT EXISTS note_entities(
                    note_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    conf REAL NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(note_id, entity_id)
                );

                CREATE TABLE IF NOT EXISTS entity_relations(
                    note_id TEXT NOT NULL,
                    src_entity INTEGER NOT NULL,
                    rel_type TEXT NOT NULL,
                    dst_entity INTEGER NOT NULL,
                    conf REAL NOT NULL,
                    PRIMARY KEY(note_id, src_entity, rel_type, dst_entity)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('chunk', 'gist', 'utterance')),
                    text TEXT NOT NULL,
                    vec BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    shared_json TEXT NOT NULL,
                    fact TEXT NOT NULL DEFAULT '',
                    fact_provenance TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(src, dst)
                );

                CREATE TABLE IF NOT EXISTS term_vectors (
                    term TEXT PRIMARY KEY,
                    vec BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fact_embeddings (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    fact_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    vec BLOB NOT NULL,
                    PRIMARY KEY(src, dst)
                );

                CREATE TABLE IF NOT EXISTS edge_judgments (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    sig TEXT NOT NULL,
                    related INTEGER NOT NULL,
                    fact TEXT NOT NULL DEFAULT '',
                    judged_at REAL NOT NULL,
                    PRIMARY KEY(src, dst)
                );

                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    registry_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    thinking TEXT NOT NULL DEFAULT '',
                    evidence_json TEXT NOT NULL DEFAULT 'null',
                    retrieved_ids_json TEXT NOT NULL DEFAULT '[]',
                    consulted_ids_json TEXT NOT NULL DEFAULT '[]',
                    cited_ids_json TEXT NOT NULL DEFAULT '[]',
                    original_query TEXT NOT NULL DEFAULT '',
                    rewritten_query TEXT NOT NULL DEFAULT '',
                    retrieval_decision TEXT NOT NULL DEFAULT '',
                    retrieval_confidence REAL NOT NULL DEFAULT 0,
                    active_build_id TEXT NOT NULL DEFAULT '',
                    activity_json TEXT NOT NULL DEFAULT '[]',
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    cancelled INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated REAL NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
                USING fts5(note_id UNINDEXED, title, body);
                """
            )
            self._ensure_notes_source_column()
            self._ensure_edges_fact_column()
            self._ensure_edges_fact_provenance_column()
            self._ensure_notes_created_column()
            self._ensure_notes_quality_rank_column()
            self._ensure_notes_value_columns()

    def _ensure_notes_created_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "created" in columns:
            return
        self._conn.execute("ALTER TABLE notes ADD COLUMN created REAL NOT NULL DEFAULT 0")
        self._conn.execute("UPDATE notes SET created = modified WHERE created = 0")

    def _ensure_notes_quality_rank_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "quality_rank" in columns:
            return
        self._conn.execute(
            "ALTER TABLE notes ADD COLUMN quality_rank REAL NOT NULL DEFAULT 0.0"
        )

    def _ensure_notes_value_columns(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "ingest_salience" not in columns:
            self._conn.execute("ALTER TABLE notes ADD COLUMN ingest_salience REAL DEFAULT 1.0")
        if "value_score" not in columns:
            self._conn.execute("ALTER TABLE notes ADD COLUMN value_score REAL DEFAULT 0.0")

    def _ensure_notes_source_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "source" in columns:
            return
        self._conn.execute(
            "ALTER TABLE notes ADD COLUMN source TEXT NOT NULL DEFAULT 'apple_notes'"
        )

    def _ensure_edges_fact_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "fact" in columns:
            return
        self._conn.execute("ALTER TABLE edges ADD COLUMN fact TEXT NOT NULL DEFAULT ''")

    def _ensure_edges_fact_provenance_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "fact_provenance" in columns:
            return
        self._conn.execute(
            "ALTER TABLE edges ADD COLUMN fact_provenance TEXT NOT NULL DEFAULT ''"
        )

    def _replace_note_extraction_rows(
        self,
        note_id: str,
        entities: list[tuple[str, str, float, int]],
        relations: list[tuple[str, str, str, float]],
    ) -> None:
        touched_entity_ids = self._entity_ids_touched_by_note(note_id)
        self._conn.execute("DELETE FROM note_entities WHERE note_id = ?", (note_id,))
        self._conn.execute("DELETE FROM entity_relations WHERE note_id = ?", (note_id,))

        canonical_entity_ids: dict[str, set[int]] = defaultdict(set)
        for canonical, etype, conf, count in entities:
            if not canonical.strip() or not etype.strip():
                continue
            entity_id = self._upsert_entity(canonical, etype)
            touched_entity_ids.add(entity_id)
            canonical_entity_ids[canonical].add(entity_id)
            self._conn.execute(
                """
                INSERT INTO note_entities(note_id, entity_id, conf, count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(note_id, entity_id) DO UPDATE SET
                    conf = excluded.conf,
                    count = excluded.count
                """,
                (note_id, entity_id, float(conf), int(count)),
            )

        for src_canonical, rel_type, dst_canonical, conf in relations:
            src_entity = self._resolve_entity_id(src_canonical, canonical_entity_ids)
            dst_entity = self._resolve_entity_id(dst_canonical, canonical_entity_ids)
            if src_entity is None or dst_entity is None:
                continue
            touched_entity_ids.update((src_entity, dst_entity))
            self._conn.execute(
                """
                INSERT INTO entity_relations(note_id, src_entity, rel_type, dst_entity, conf)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(note_id, src_entity, rel_type, dst_entity) DO UPDATE SET
                    conf = MAX(entity_relations.conf, excluded.conf)
                """,
                (note_id, src_entity, rel_type, dst_entity, float(conf)),
            )

        self._recompute_entity_df(touched_entity_ids)

    def _upsert_entity(self, canonical: str, etype: str) -> int:
        self._conn.execute(
            "INSERT OR IGNORE INTO entities(canonical, etype) VALUES (?, ?)",
            (canonical, etype),
        )
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE canonical = ? AND etype = ?",
            (canonical, etype),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to upsert entity")
        return int(row["entity_id"])

    def _resolve_entity_id(
        self, canonical: str, preferred_ids: dict[str, set[int]]
    ) -> int | None:
        ids = preferred_ids.get(canonical, set())
        if len(ids) == 1:
            return next(iter(ids))
        rows = self._conn.execute(
            "SELECT entity_id FROM entities WHERE canonical = ? ORDER BY entity_id",
            (canonical,),
        ).fetchall()
        if len(rows) != 1:
            return None
        return int(rows[0]["entity_id"])

    def _entity_ids_touched_by_note(self, note_id: str) -> set[int]:
        rows = self._conn.execute(
            """
            SELECT entity_id FROM note_entities WHERE note_id = ?
            UNION
            SELECT src_entity AS entity_id FROM entity_relations WHERE note_id = ?
            UNION
            SELECT dst_entity AS entity_id FROM entity_relations WHERE note_id = ?
            """,
            (note_id, note_id, note_id),
        ).fetchall()
        return {int(row["entity_id"]) for row in rows}

    def _recompute_entity_df(self, entity_ids: set[int]) -> None:
        for entity_id in sorted(entity_ids):
            self._conn.execute(
                """
                UPDATE entities
                SET df = (
                    SELECT COUNT(DISTINCT note_id)
                    FROM note_entities
                    WHERE entity_id = ?
                )
                WHERE entity_id = ?
                """,
                (entity_id, entity_id),
            )

    def _insert_embedding(
        self, note_id: str, kind: EmbeddingKind, text: str, vector: np.ndarray
    ) -> None:
        if kind not in {"chunk", "gist", "utterance"}:
            raise ValueError(f"unsupported embedding kind: {kind}")
        values = np.asarray(vector, dtype=np.float32)
        if values.shape != (config.EMBED_DIM,):
            raise ValueError(f"embedding must have shape ({config.EMBED_DIM},)")
        norm = float(np.linalg.norm(values))
        if norm > 0:
            values = values / norm
        self._conn.execute(
            "INSERT INTO embeddings(note_id, kind, text, vec) VALUES (?, ?, ?, ?)",
            (note_id, kind, text, values.astype(np.float32).tobytes()),
        )

    def _set_last_sync(self, timestamp: float) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO metadata(key, value)
            VALUES ('last_sync_ts', ?)
            """,
            (str(timestamp),),
        )

    @staticmethod
    def _aliases_for(title: str, enrichment: Enrichment) -> list[tuple[str, bool]]:
        aliases: dict[str, bool] = {_normalize_alias(title): True}
        for alias in enrichment.aliases:
            normalized = _normalize_alias(alias)
            if normalized:
                aliases[normalized] = aliases.get(normalized, False)
        return [(alias, is_title) for alias, is_title in aliases.items() if alias]

    @staticmethod
    def _fts_query(text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", text)
        return " OR ".join(f'"{token}"' for token in tokens)


def _normalize_alias(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _entity_surface_candidates(surface: str) -> tuple[str, ...]:
    normalized = _normalize_entity_surface(surface, lowercase=False)
    lowered = surface.lower()
    lowered_normalized = _normalize_entity_surface(surface, lowercase=True)
    return tuple(
        dict.fromkeys(
            candidate
            for candidate in (surface, normalized, lowered, lowered_normalized)
            if candidate
        )
    )


def _normalize_entity_surface(surface: str, *, lowercase: bool) -> str:
    normalized = re.sub(r"[-_]+", " ", surface.strip())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower() if lowercase else normalized


def _display_line(note: NoteRecord) -> str:
    for line in note.text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return _clamp_display_line(cleaned)
    return _clamp_display_line(note.title)


def _clamp_display_line(value: str) -> str:
    max_chars = config.GIST_MAX_CHARS
    if len(value) <= max_chars:
        return value
    clipped = value[:max_chars]
    if clipped[-1].isspace() or value[max_chars : max_chars + 1].isspace():
        return clipped.rstrip()
    head = clipped.rsplit(" ", 1)[0].rstrip()
    return head or clipped.rstrip()


def _rerank_evidence_text(kind: EmbeddingKind, text: str) -> str:
    if kind == "utterance":
        return f"(spoken reference) {text}"
    return text


def _connector_meta_key(connector_id: str) -> str:
    return f"connector:{connector_id}"
