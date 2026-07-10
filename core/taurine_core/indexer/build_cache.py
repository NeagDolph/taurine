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
from taurine_core.indexer.facts import PREDICATES
from taurine_core.schemas import Extraction, NoteRecord

_EXTRACTION_CACHE_VERSION = "extraction-cache-v1"

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
"""


class BuildCache:
    """SQLite-backed caches shared by every build in this process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
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
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = list(dict.fromkeys(keys[start : start + 500]))
                placeholders = ",".join("?" for _ in chunk)
                for key, dim, blob in self._conn.execute(
                    f"SELECT key, dim, vec FROM embedding_cache WHERE key IN ({placeholders})",
                    chunk,
                ):
                    rows[key] = np.frombuffer(blob, dtype=np.float32, count=int(dim))
        return rows

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
            keys = {note.note_id: _extraction_key(note) for note in notes}
            cached = self._extraction_rows(list(keys.values()))
            results: dict[str, Extraction] = {}
            missing: list[NoteRecord] = []
            for note in notes:
                payload = cached.get(keys[note.note_id])
                if payload is None:
                    missing.append(note)
                    continue
                results[note.note_id] = Extraction.model_validate_json(payload)
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


def _embedding_key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\x1f{text}".encode("utf-8")).hexdigest()


def _schema_fingerprint() -> str:
    """Entity types and predicates drive extraction output; changing either
    must invalidate cached extractions."""
    payload = {
        "entities": dict(config.INGEST_ENTITY_TYPES),
        "predicates": {name: spec.description for name, spec in sorted(PREDICATES.items())},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _extraction_key(note: NoteRecord) -> str:
    identity = "\x1f".join(
        (
            _EXTRACTION_CACHE_VERSION,
            _schema_fingerprint(),
            note.source,
            note.content_hash,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
