from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from taurine_core import config

from .embed import embed_texts
from .store import EmbeddingKind, Index

_BATCH_SIZE = 64
_LOG_EVERY = 200
_LOG = logging.getLogger("taurine.session")

EmbeddingRow = tuple[int, str, EmbeddingKind, str]


def needs_reembed(index: Index) -> bool:
    row = index._conn.execute(
        "SELECT COUNT(*) AS count FROM embeddings WHERE length(vec) != ?",
        (config.EMBED_DIM * 4,),
    ).fetchone()
    return row is not None and int(row["count"]) > 0


def reembed_all(
    index: Index,
    progress: Callable[[int, int], None] | None = None,
    *,
    stale_only: bool = True,
) -> int:
    """Re-encode embedding rows. With stale_only (the default) only rows whose
    stored vector has the wrong byte length are touched — every row commits
    individually, so an interrupted migration resumes where it stopped."""
    rows = index._embedding_rows_raw(stale_only=stale_only)
    total = len(rows)
    done = 0
    next_log = _LOG_EVERY

    for batch in _batches(rows, _BATCH_SIZE):
        vectors = embed_texts([row[3] for row in batch])
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(batch), config.EMBED_DIM):
            raise ValueError(
                f"embedding model returned {matrix.shape}, expected {(len(batch), config.EMBED_DIM)}"
            )
        for (row_id, _note_id, _kind, _text), vector in zip(batch, matrix, strict=True):
            index.update_embedding_vec(row_id, vector)
        done += len(batch)
        if progress is not None:
            progress(done, total)
        if done >= next_log or done == total:
            _LOG.info("reembed: %d/%d embeddings", done, total)
            while next_log <= done:
                next_log += _LOG_EVERY

    index.rebuild_graph()
    index.load()
    return total


def _batches(rows: Sequence[EmbeddingRow], size: int) -> list[Sequence[EmbeddingRow]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-embed stale Taurine embeddings.")
    parser.add_argument("--db", type=Path, default=Path("taurine.db"))
    parser.add_argument(
        "--all",
        action="store_true",
        help="re-encode every row, not just stale-dimension ones",
    )
    args = parser.parse_args(argv)

    index = Index(args.db)
    try:
        count = reembed_all(
            index,
            progress=lambda done, total: print(
                f"reembed: {done}/{total} embeddings",
                flush=True,
            ),
            stale_only=not args.all,
        )
    finally:
        index.close()
    print(f"reembedded {count} embeddings", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
