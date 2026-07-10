from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from taurine_core import config

if TYPE_CHECKING:
    from taurine_core.indexer.projection import ProjectionEdge


def label_propagation(note_ids: Sequence[str], matrix: csr_matrix) -> dict[str, int]:
    """Deterministic synchronous label propagation over the note graph.

    ``matrix`` row/column order must match ``note_ids`` order.
    """
    size = len(note_ids)
    if size == 0:
        return {}

    base = matrix.tocsr()
    adjacency = ((base + base.T) > 0).astype(np.int8).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    row_of = {note_id: row for row, note_id in enumerate(note_ids)}
    labels = np.arange(size, dtype=np.int64)
    order = [row_of[note_id] for note_id in sorted(note_ids)]

    for _ in range(max(0, int(config.COMMUNITY_LP_ITERS))):
        updated = labels.copy()
        for row in order:
            start = adjacency.indptr[row]
            end = adjacency.indptr[row + 1]
            neighbors = adjacency.indices[start:end]
            if neighbors.size == 0:
                continue
            neighbor_labels = labels[neighbors]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            updated[row] = int(values[counts == counts.max()][0])
        if np.array_equal(updated, labels):
            labels = updated
            break
        labels = updated

    return {note_id: int(labels[row]) for note_id, row in row_of.items()}


def projection_communities(edges: Sequence["ProjectionEdge"]) -> dict[str, int]:
    """Deterministic label propagation over the note projection.

    Adjacency comes straight from the projection edges (binary, symmetric);
    labels are row indices over the sorted note ids, so ids are stable for a
    given build. Notes without projection edges are absent (community -1).
    """
    note_ids = sorted(
        {note_id for edge in edges for note_id in (edge.src_note_id, edge.dst_note_id)}
    )
    if not note_ids:
        return {}
    row_of = {note_id: row for row, note_id in enumerate(note_ids)}
    rows: list[int] = []
    cols: list[int] = []
    for edge in edges:
        left, right = row_of[edge.src_note_id], row_of[edge.dst_note_id]
        if left == right:
            continue
        rows.extend((left, right))
        cols.extend((right, left))
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(note_ids), len(note_ids)),
    )
    return label_propagation(note_ids, matrix)
