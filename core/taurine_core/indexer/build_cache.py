"""Persistent content-keyed caches for expensive build inputs.

Passage/fact embeddings and GLiNER extractions are pure functions of
(content, model/schema), so clean rebuilds can reuse them across syncs. The
cache lives beside — never inside — the knowledge index, which stays a
deterministic, wholly replaceable artifact. Deleting the cache file is always
safe; the next sync just recomputes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from taurine_core import config
from taurine_core.indexer import extract
from taurine_core.indexer.facts import PREDICATES
from taurine_core.schemas import Extraction, NoteRecord

_EXTRACTION_CACHE_VERSION = "extraction-cache-v2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS extraction_cache (
    key TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS context_cache (
    key TEXT PRIMARY KEY,
    passage_id TEXT NOT NULL,
    context TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS adjudication_cache (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL
);
"""


class BuildCache:
    """SQLite-backed caches shared by every build in this process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self.stats: dict[str, int] = {"cache_corrupt": 0}
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- embeddings ---------------------------------------------------------

    def caching_embedder(
        self,
        model_id: str,
        inner: Callable[[list[str]], np.ndarray],
    ) -> Callable[[list[str]], np.ndarray]:
        """Wrap an embedder; identical texts are embedded at most once."""

        def embed(texts: list[str]) -> np.ndarray:
            if not texts:
                return inner(texts)
            keys = [_embedding_key(model_id, text) for text in texts]
            cached = self._embedding_rows(keys)
            missing = [index for index, key in enumerate(keys) if key not in cached]
            fresh: np.ndarray | None = None
            if missing:
                fresh = np.asarray(
                    inner([texts[index] for index in missing]), dtype=np.float32
                )
                self._store_embeddings(
                    [(keys[index], model_id) for index in missing], fresh
                )
            dim = (
                int(fresh.shape[1])
                if fresh is not None and fresh.ndim == 2
                else len(next(iter(cached.values())))
            )
            matrix = np.empty((len(texts), dim), dtype=np.float32)
            fresh_position = 0
            for index, key in enumerate(keys):
                row = cached.get(key)
                if row is not None:
                    matrix[index] = row
                else:
                    assert fresh is not None
                    matrix[index] = fresh[fresh_position]
                    fresh_position += 1
            return matrix

        return embed

    def _embedding_rows(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        rows: dict[str, np.ndarray] = {}
        corrupt: list[str] = []
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = list(dict.fromkeys(keys[start : start + 500]))
                placeholders = ",".join("?" for _ in chunk)
                for key, dim, blob in self._conn.execute(
                    f"SELECT key, dim, vec FROM embedding_cache WHERE key IN ({placeholders})",
                    chunk,
                ):
                    # a corrupt row is a cache miss, never a build failure (L8)
                    if int(dim) <= 0 or len(blob) != int(dim) * 4:
                        corrupt.append(key)
                        continue
                    rows[key] = np.frombuffer(blob, dtype=np.float32, count=int(dim))
        self._drop_corrupt("embedding_cache", corrupt)
        return rows

    def _drop_corrupt(self, table: str, keys: Sequence[str]) -> None:
        if not keys:
            return
        assert table in {"embedding_cache", "extraction_cache", "adjudication_cache"}
        with self._lock, self._conn:
            self._conn.executemany(
                f"DELETE FROM {table} WHERE key = ?", [(key,) for key in keys]
            )
        self.stats["cache_corrupt"] += len(keys)

    def _store_embeddings(
        self,
        keyed: Sequence[tuple[str, str]],
        matrix: np.ndarray,
    ) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != len(keyed):
            return  # malformed embedder output is the builder's problem, not the cache's
        now = time.time()
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (key, model_id, dim, vec, created)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        key,
                        model_id,
                        int(matrix.shape[1]),
                        np.ascontiguousarray(matrix[index], dtype=np.float32).tobytes(),
                        now,
                    )
                    for index, (key, model_id) in enumerate(keyed)
                ],
            )

    # -- extractions --------------------------------------------------------

    def caching_extractor(
        self,
        inner: Callable[..., dict[str, Extraction]],
    ) -> Callable[..., dict[str, Extraction]]:
        """Wrap extract_notes; unchanged notes reuse their stored extraction."""

        def extract(
            notes: list[NoteRecord],
            *,
            tokenizer: Any = None,
            progress: Callable[[int, int], None] | None = None,
        ) -> dict[str, Extraction]:
            if not notes:
                return {}
            fingerprint = _schema_fingerprint()
            keys = {note.note_id: _extraction_key(note, fingerprint) for note in notes}
            cached = self._extraction_rows(list(keys.values()))
            results: dict[str, Extraction] = {}
            missing: list[NoteRecord] = []
            corrupt: list[str] = []
            for note in notes:
                payload = cached.get(keys[note.note_id])
                if payload is None:
                    missing.append(note)
                    continue
                try:
                    results[note.note_id] = Extraction.model_validate_json(payload)
                except Exception:
                    # a corrupt row is a cache miss, never a build failure (L8)
                    corrupt.append(keys[note.note_id])
                    missing.append(note)
            self._drop_corrupt("extraction_cache", corrupt)
            hits = len(results)
            if progress is not None:
                progress(hits, len(notes))
            if missing:
                inner_progress = (
                    None
                    if progress is None
                    else lambda done, total: progress(hits + done, len(notes))
                )
                fresh = inner(missing, tokenizer=tokenizer, progress=inner_progress)
                results.update(fresh)
                self._store_extractions(
                    [
                        (keys[note.note_id], note.note_id, extraction)
                        for note in missing
                        if (extraction := fresh.get(note.note_id)) is not None
                        and not extraction.failed
                    ]
                )
            return {note.note_id: results[note.note_id] for note in notes if note.note_id in results}

        return extract

    def _extraction_rows(self, keys: Sequence[str]) -> dict[str, str]:
        rows: dict[str, str] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = list(dict.fromkeys(keys[start : start + 500]))
                placeholders = ",".join("?" for _ in chunk)
                for key, payload in self._conn.execute(
                    f"SELECT key, payload FROM extraction_cache WHERE key IN ({placeholders})",
                    chunk,
                ):
                    rows[key] = payload
        return rows

    def _store_extractions(
        self,
        rows: Sequence[tuple[str, str, Extraction]],
    ) -> None:
        if not rows:
            return
        now = time.time()
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO extraction_cache (key, note_id, payload, created)"
                " VALUES (?, ?, ?, ?)",
                [
                    (key, note_id, extraction.model_dump_json(), now)
                    for key, note_id, extraction in rows
                ],
            )


    # -- passage contexts ----------------------------------------------------

    def cached_contexts(self, keys: Sequence[str]) -> dict[str, str]:
        """Situating-prefix cache rows keyed by the context_enrich key family
        (prompt version, model, passage content hash, window hash)."""

        rows: dict[str, str] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = list(dict.fromkeys(keys[start : start + 500]))
                placeholders = ",".join("?" for _ in chunk)
                for key, context in self._conn.execute(
                    f"SELECT key, context FROM context_cache WHERE key IN ({placeholders})",
                    chunk,
                ):
                    rows[key] = str(context)
        return rows

    def store_contexts(self, rows: Sequence[tuple[str, str, str]]) -> None:
        """Store (key, passage_id, context) rows; empty contexts are cached
        too — a model that declined to situate a passage stays declined."""

        if not rows:
            return
        now = time.time()
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO context_cache (key, passage_id, context, created)"
                " VALUES (?, ?, ?, ?)",
                [(key, passage_id, context, now) for key, passage_id, context in rows],
            )


    # -- edge adjudication verdicts -------------------------------------------

    def cached_judgments(self, keys: Sequence[str]) -> dict[str, dict]:
        """Grounded-judge verdict rows keyed by the edge_adjudication key
        families (prompt version, model, edge identity, content hashes)."""

        rows: dict[str, dict] = {}
        corrupt: list[str] = []
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = list(dict.fromkeys(keys[start : start + 500]))
                placeholders = ",".join("?" for _ in chunk)
                for key, payload in self._conn.execute(
                    f"SELECT key, payload FROM adjudication_cache WHERE key IN ({placeholders})",
                    chunk,
                ):
                    try:
                        parsed = json.loads(payload)
                    except Exception:
                        parsed = None
                    # a corrupt row is a cache miss, never a build failure (L8)
                    if not isinstance(parsed, dict):
                        corrupt.append(key)
                        continue
                    rows[key] = parsed
        self._drop_corrupt("adjudication_cache", corrupt)
        return rows

    def store_judgments(self, rows: Sequence[tuple[str, str, dict]]) -> None:
        """Store (key, kind, verdict) rows; only code-verified verdicts should
        reach the cache so replays never resurrect malformed judgments."""

        if not rows:
            return
        now = time.time()
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO adjudication_cache (key, kind, payload, created)"
                " VALUES (?, ?, ?, ?)",
                [
                    (key, kind, json.dumps(payload, sort_keys=True), now)
                    for key, kind, payload in rows
                ],
            )


def _embedding_key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\x1f{text}".encode("utf-8")).hexdigest()


def _schema_fingerprint() -> str:
    """Everything that shapes an extraction result must invalidate the cache
    (H5): the rendered GLiNER schema request (entity types + descriptions and
    the per-family relation prompts), chunking knobs, the extractor model id,
    and the calibrated per-type threshold tables baked into accepted rows."""
    calibration = extract._calibration().extraction
    payload = {
        "extractor": extract._EXTRACTOR_ID,
        "entities": dict(config.INGEST_ENTITY_TYPES),
        "relations": {
            family: extract._relation_descriptions(family)
            for family in ("conversation", "email", "written")
        },
        "predicates": {name: spec.description for name, spec in sorted(PREDICATES.items())},
        "chunking": {
            "chunk_size": int(config.GLINER_CHUNK_SIZE),
            "chunk_overlap": int(config.GLINER_CHUNK_OVERLAP),
            "long_chunk_tokens": int(config.GLINER_LONG_CHUNK_TOKENS),
            "long_overlap_tokens": int(config.GLINER_LONG_OVERLAP_TOKENS),
        },
        "thresholds": {
            "entity": dict(sorted(calibration.entity_thresholds.items())),
            "relation": dict(sorted(calibration.relation_thresholds.items())),
            "promotion": dict(sorted(calibration.promotion_thresholds.items())),
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _extraction_key(note: NoteRecord, schema_fingerprint: str) -> str:
    identity = "\x1f".join(
        (
            _EXTRACTION_CACHE_VERSION,
            schema_fingerprint,
            note.source,
            note.content_hash,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
