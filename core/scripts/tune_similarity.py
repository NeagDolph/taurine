from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taurine_core import config
from taurine_core.indexer.embed import embed_queries

SAMPLE_QUERIES = [
    "the recommendation engine reranker",
    "dinner with my parents",
    "roadmap planning for next quarter",
    "notes about billing polish",
    "what did we decide in standup",
]

PERCENTILES = [50, 75, 90, 95, 97.5, 99]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect embedding similarity ranges.")
    parser.add_argument("--db", type=Path, default=Path("taurine.db"))
    args = parser.parse_args()

    conn = _connect_readonly(args.db)
    try:
        disabled_sources = _disabled_sources(conn)
        note_vectors = _note_vectors(conn, disabled_sources)
        print(f"notes with representative embeddings: {len(note_vectors)}")
        _print_percentiles(note_vectors)
        _print_query_scores(conn, disabled_sources)
    finally:
        conn.close()
    return 0


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _disabled_sources(conn: sqlite3.Connection) -> set[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = 'disabled_sources'").fetchone()
    if row is None:
        return set()
    try:
        value = json.loads(row["value"])
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in value} if isinstance(value, list) else set()


def _active_clause(disabled_sources: set[str], column: str) -> tuple[str, tuple[str, ...]]:
    if not disabled_sources:
        return "", ()
    placeholders = ",".join("?" for _ in disabled_sources)
    return f" AND {column} NOT IN ({placeholders})", tuple(sorted(disabled_sources))


def _note_vectors(
    conn: sqlite3.Connection,
    disabled_sources: set[str],
) -> dict[str, np.ndarray]:
    clause, params = _active_clause(disabled_sources, "n.source")
    rows = conn.execute(
        f"""
        SELECT e.note_id, e.kind, e.vec
        FROM embeddings e
        JOIN notes n ON n.note_id = e.note_id
        WHERE length(e.vec) = ?{clause}
        ORDER BY e.note_id, CASE WHEN e.kind = 'gist' THEN 0 ELSE 1 END, e.id
        """,
        (config.EMBED_DIM * 4, *params),
    ).fetchall()

    vectors: dict[str, np.ndarray] = {}
    selected_kind: dict[str, str] = {}
    for row in rows:
        note_id = str(row["note_id"])
        kind = str(row["kind"])
        if note_id in vectors and selected_kind[note_id] == "gist":
            continue
        vector = _vector(row["vec"])
        if vector is None:
            continue
        vectors[note_id] = vector
        selected_kind[note_id] = kind
    return vectors


def _all_embedding_rows(
    conn: sqlite3.Connection,
    disabled_sources: set[str],
) -> list[tuple[str, np.ndarray]]:
    clause, params = _active_clause(disabled_sources, "n.source")
    rows = conn.execute(
        f"""
        SELECT e.note_id, e.vec
        FROM embeddings e
        JOIN notes n ON n.note_id = e.note_id
        WHERE length(e.vec) = ?{clause}
        ORDER BY e.id
        """,
        (config.EMBED_DIM * 4, *params),
    ).fetchall()
    vectors: list[tuple[str, np.ndarray]] = []
    for row in rows:
        vector = _vector(row["vec"])
        if vector is not None:
            vectors.append((str(row["note_id"]), vector))
    return vectors


def _vector(blob: bytes) -> np.ndarray | None:
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    if vector.shape != (config.EMBED_DIM,):
        return None
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return None
    return (vector / norm).astype(np.float32)


def _print_percentiles(note_vectors: dict[str, np.ndarray]) -> None:
    values = list(note_vectors.values())
    if len(values) < 2:
        print("pairwise percentiles: need at least two notes")
        return
    sims: list[float] = []
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            sims.append(float(left @ right))
    percentile_values = np.percentile(np.asarray(sims, dtype=np.float32), PERCENTILES)
    print("pairwise cosine percentiles:")
    for percentile, value in zip(PERCENTILES, percentile_values, strict=True):
        print(f"  p{percentile:g}: {value:.4f}")


def _print_query_scores(conn: sqlite3.Connection, disabled_sources: set[str]) -> None:
    embeddings = _all_embedding_rows(conn, disabled_sources)
    if not embeddings:
        print("query scores: no valid embeddings")
        return

    print("sample dense_search scores:")
    for query in SAMPLE_QUERIES:
        query_vec = embed_queries([query])[0].astype(np.float32)
        norm = float(np.linalg.norm(query_vec))
        if norm == 0:
            print(f"  {query!r}: zero query vector")
            continue
        query_vec = query_vec / norm
        per_note: dict[str, float] = {}
        for note_id, vector in embeddings:
            score = float(vector @ query_vec)
            per_note[note_id] = max(per_note.get(note_id, -1.0), score)
        top = sorted(per_note.items(), key=lambda item: item[1], reverse=True)[:10]
        print(f"  {query!r}")
        for note_id, score in top:
            print(f"    {score:.4f}  {note_id}")


if __name__ == "__main__":
    raise SystemExit(main())
