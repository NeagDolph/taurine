from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from taurine_core import config
from taurine_core.schemas import (
    Enrichment,
    GraphEdge,
    GraphNode,
    GraphResponse,
    NeighborRef,
    NotePreview,
    NoteRecord,
)

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


class Index:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(self.db_path, check_same_thread=False)
        raw.row_factory = sqlite3.Row
        self._conn = _ThreadSafeConnection(raw)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._embedding_rows: list[tuple[int, str, EmbeddingKind, str]] = []
        self._matrix = np.empty((0, config.EMBED_DIM), dtype=np.float32)
        self._doc_frequency_cache: dict[str, int] = {}
        self._init_schema()
        self.load()

    def load(self) -> None:
        rows = self._conn.execute(
            "SELECT id, note_id, kind, text, vec FROM embeddings ORDER BY id"
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
        self._embedding_rows = embedding_rows
        self._matrix = (
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.empty((0, config.EMBED_DIM), dtype=np.float32)
        )

    def close(self) -> None:
        self._conn.close()

    def upsert_note(
        self,
        note: NoteRecord,
        enrichment: Enrichment,
        embedding_units: Sequence[EmbeddingInput],
    ) -> None:
        stored = note.model_copy(update={"gist": enrichment.gist})
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO notes(note_id, title, folder, modified, content_hash, text, gist, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    title=excluded.title,
                    folder=excluded.folder,
                    modified=excluded.modified,
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
            self._conn.execute(
                """
                INSERT OR REPLACE INTO enrich_cache(note_id, content_hash, enrichment_json)
                VALUES (?, ?, ?)
                """,
                (stored.note_id, stored.content_hash, enrichment.model_dump_json()),
            )
            self._set_last_sync(time.time())
        self._doc_frequency_cache.clear()
        self.load()

    def delete_note(self, note_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM aliases WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
            self._conn.execute(
                "DELETE FROM edges WHERE src = ? OR dst = ?",
                (note_id, note_id),
            )
            self._set_last_sync(time.time())
        self._doc_frequency_cache.clear()
        self.load()

    def cache_enrichment(
        self, note_id: str, content_hash: str, enrichment: Enrichment
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO enrich_cache(note_id, content_hash, enrichment_json)
                VALUES (?, ?, ?)
                """,
                (note_id, content_hash, enrichment.model_dump_json()),
            )

    def cached_enrichment(self, note_id: str, content_hash: str) -> Enrichment | None:
        row = self._conn.execute(
            """
            SELECT enrichment_json FROM enrich_cache
            WHERE note_id = ? AND content_hash = ?
            """,
            (note_id, content_hash),
        ).fetchone()
        if row is None:
            return None
        return Enrichment.model_validate_json(row["enrichment_json"])

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

    def dense_search(self, text: str, k: int) -> list[tuple[str, float]]:
        if k <= 0 or self._matrix.size == 0:
            return []
        query = embed_texts([text])[0].astype(np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        scores = self._matrix @ query
        per_note: dict[str, float] = {}
        for (_, note_id, _, _), score in zip(self._embedding_rows, scores, strict=True):
            per_note[note_id] = max(per_note.get(note_id, -1.0), float(score))
        return sorted(per_note.items(), key=lambda item: item[1], reverse=True)[:k]

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = self._fts_query(text)
        if not query:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT note_id, -bm25(notes_fts) AS score
                FROM notes_fts
                WHERE notes_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(row["note_id"], float(row["score"])) for row in rows]

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            """
            SELECT src, dst, weight FROM edges
            WHERE src = ? OR dst = ?
            ORDER BY weight DESC
            """,
            (note_id, note_id),
        ).fetchall()
        return [
            (row["dst"] if row["src"] == note_id else row["src"], float(row["weight"]))
            for row in rows
        ]

    def note(self, note_id: str) -> NoteRecord | None:
        row = self._conn.execute(
            """
            SELECT note_id, title, folder, modified, content_hash, text, gist, source
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
                SELECT note_id, title, folder, modified, content_hash, text, gist, source
                FROM notes WHERE source = ? ORDER BY note_id
                """,
                (source,),
            ).fetchall()
            return [NoteRecord(**dict(row)) for row in rows]
        rows = self._conn.execute(
            """
            SELECT note_id, title, folder, modified, content_hash, text, gist, source
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
        for neighbor_id, weight in self.neighbors(note_id):
            neighbor = self.note(neighbor_id)
            if neighbor is None:
                continue
            neighbors.append(
                NeighborRef(
                    note_id=neighbor.note_id,
                    title=neighbor.title,
                    source=neighbor.source,
                    weight=weight,
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

        The method name is historical: candidates may be chunks, gists, or
        synthetic utterances. Utterance winners are labeled as spoken references
        so rerank documents read naturally.
        """
        rows = [
            (row_id, kind, text)
            for row_id, row_note_id, kind, text in self._embedding_rows
            if row_note_id == note_id
        ]
        if not rows:
            note = self.note(note_id)
            return note.text if note is not None else ""
        if self._matrix.size == 0:
            return _rerank_evidence_text(rows[0][1], rows[0][2])
        query = embed_texts([query_text])[0].astype(np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return _rerank_evidence_text(rows[0][1], rows[0][2])
        query = query / norm
        row_ids = {row_id: (kind, text) for row_id, kind, text in rows}
        best_kind = rows[0][1]
        best_text = rows[0][2]
        best_score = -1.0
        for index, (row_id, _, _, _) in enumerate(self._embedding_rows):
            if row_id not in row_ids:
                continue
            score = float(self._matrix[index] @ query)
            if score > best_score:
                best_score = score
                best_kind, best_text = row_ids[row_id]
        return _rerank_evidence_text(best_kind, best_text)

    def aliases(self) -> list[tuple[str, str, bool]]:
        rows = self._conn.execute(
            """
            SELECT alias_lowercase, note_id, is_title
            FROM aliases ORDER BY note_id, is_title DESC, alias_lowercase
            """
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
        try:
            row = self._conn.execute(
                """
                SELECT COUNT(DISTINCT note_id) AS count
                FROM notes_fts
                WHERE notes_fts MATCH ?
                """,
                (query,),
            ).fetchone()
        except sqlite3.OperationalError:
            count = 0
        else:
            count = int(row["count"]) if row is not None else 0
        self._doc_frequency_cache[key] = count
        return count

    def graph(self) -> GraphResponse:
        degree: dict[str, int] = defaultdict(int)
        edge_rows = self._conn.execute(
            """
            SELECT src, dst, type, weight, shared_json
            FROM edges ORDER BY weight DESC, src, dst
            """
        ).fetchall()
        edges: list[GraphEdge] = []
        for row in edge_rows:
            degree[row["src"]] += 1
            degree[row["dst"]] += 1
            edges.append(
                GraphEdge(
                    src=row["src"],
                    dst=row["dst"],
                    type=row["type"],
                    weight=float(row["weight"]),
                    shared=json.loads(row["shared_json"]),
                )
            )
        nodes = [
            GraphNode(
                note_id=note.note_id,
                title=note.title,
                degree=degree[note.note_id],
                source=note.source,
            )
            for note in self.notes()
        ]
        return GraphResponse(nodes=nodes, edges=edges)

    def status(self) -> tuple[int, float | None]:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM notes").fetchone()
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
                    INSERT OR REPLACE INTO edges(src, dst, type, weight, shared_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        src,
                        dst,
                        edge.type,
                        float(edge.weight),
                        json.dumps(sorted(set(edge.shared))),
                    ),
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

    def enrichment_for(self, note_id: str) -> Enrichment | None:
        note = self.note(note_id)
        if note is None:
            return None
        row = self._conn.execute(
            """
            SELECT enrichment_json FROM enrich_cache
            WHERE note_id = ? AND content_hash = ?
            """,
            (note_id, note.content_hash),
        ).fetchone()
        if row is not None:
            return Enrichment.model_validate_json(row["enrichment_json"])
        alias_rows = self._conn.execute(
            "SELECT alias_lowercase FROM aliases WHERE note_id = ? AND is_title = 0",
            (note_id,),
        ).fetchall()
        return Enrichment(
            gist=note.gist,
            utterances=[],
            aliases=[row["alias_lowercase"] for row in alias_rows],
            topics=[],
            entities=[],
        )

    def rebuild_graph(self) -> None:
        from .graph import rebuild_all

        rebuild_all(self)

    def rebuild_for(self, note_id: str) -> None:
        from .graph import rebuild_for

        rebuild_for(self, note_id)

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
                    PRIMARY KEY(src, dst)
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

    def _ensure_notes_source_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "source" in columns:
            return
        self._conn.execute(
            "ALTER TABLE notes ADD COLUMN source TEXT NOT NULL DEFAULT 'apple_notes'"
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


def _rerank_evidence_text(kind: EmbeddingKind, text: str) -> str:
    if kind == "utterance":
        return f"(spoken reference) {text}"
    return text


def _connector_meta_key(connector_id: str) -> str:
    return f"connector:{connector_id}"
