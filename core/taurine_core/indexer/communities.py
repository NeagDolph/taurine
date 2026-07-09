from __future__ import annotations

import numpy as np

from taurine_core import config

from .rank import NoteGraphMatrix


def label_propagation(matrix: NoteGraphMatrix) -> dict[str, int]:
    """Deterministic synchronous label propagation over the note graph."""
    size = len(matrix.note_ids)
    if size == 0:
        return {}

    base = matrix.matrix.tocsr()
    adjacency = ((base + base.T) > 0).astype(np.int8).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    labels = np.arange(size, dtype=np.int64)
    order = [matrix.row_of[note_id] for note_id in sorted(matrix.note_ids)]

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

    return {
        note_id: int(labels[row])
        for note_id, row in matrix.row_of.items()
    }
