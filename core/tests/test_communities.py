from __future__ import annotations

from scipy.sparse import csr_matrix

from taurine_core.indexer.rank import NoteGraphMatrix


def _matrix(note_ids: list[str], edges: list[tuple[str, str]]) -> NoteGraphMatrix:
    row_of = {note_id: offset for offset, note_id in enumerate(note_ids)}
    rows: list[int] = []
    cols: list[int] = []
    for left, right in edges:
        left_row = row_of[left]
        right_row = row_of[right]
        rows.extend((left_row, right_row))
        cols.extend((right_row, left_row))
    data = [1.0] * len(rows)
    return NoteGraphMatrix(
        note_ids,
        csr_matrix((data, (rows, cols)), shape=(len(note_ids), len(note_ids))),
    )


def test_label_propagation_splits_dumbbell_and_is_deterministic() -> None:
    left = [f"left-{index}" for index in range(4)]
    right = [f"right-{index}" for index in range(4)]
    clique_edges = [
        (group[i], group[j])
        for group in (left, right)
        for i in range(len(group))
        for j in range(i + 1, len(group))
    ]
    graph_matrix = _matrix(
        # Deliberately not sorted: iteration order must not depend on matrix row order.
        ["right-3", "left-2", "right-2", "left-0", "right-0", "left-3", "right-1", "left-1"],
        [*clique_edges, ("left-3", "right-0")],
    )

    from taurine_core.indexer.communities import label_propagation

    first = label_propagation(graph_matrix)
    second = label_propagation(graph_matrix)

    assert first == second
    assert len({first[note_id] for note_id in left}) == 1
    assert len({first[note_id] for note_id in right}) == 1
    assert first["left-3"] != first["right-0"]
    assert len(set(first.values())) == 2


def test_label_propagation_empty_graph_returns_empty_mapping() -> None:
    from taurine_core.indexer.communities import label_propagation

    assert label_propagation(_matrix([], [])) == {}
