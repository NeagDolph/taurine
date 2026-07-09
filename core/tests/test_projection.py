from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from taurine_core.indexer.base_graph import BaseEdge, BaseGraph
from taurine_core.indexer.projection import ProjectionPassage, build_projection


def test_projection_aggregates_mechanisms_and_fact_provenance() -> None:
    graph = _graph(
        (
            BaseEdge("p1", "p2", "semantic", 0.7),
            BaseEdge("p2", "p1", "semantic", 0.7),
            BaseEdge("p1", "e1", "contains", 0.8),
            BaseEdge("e1", "p1", "contains", 0.8),
            BaseEdge("p2", "e1", "contains", 0.9),
            BaseEdge("e1", "p2", "contains", 0.9),
            BaseEdge(
                "e1",
                "e2",
                "fact",
                0.4,
                direction="forward",
                fact_ids=("f1",),
                metadata={"predicate": "responsible_for"},
            ),
            BaseEdge("p2", "e2", "contains", 0.8),
            BaseEdge("e2", "p2", "contains", 0.8),
        )
    )

    rows = build_projection(
        graph,
        passages=(ProjectionPassage("p1", "n1"), ProjectionPassage("p2", "n2")),
        entity_labels={"e1": "Maya", "e2": "Launch"},
        max_degree=3,
    )

    assert len(rows) == 1
    edge = rows[0]
    assert (edge.src_note_id, edge.dst_note_id) == ("n1", "n2")
    assert edge.edge_type == "mixed"
    assert set(edge.mechanisms) == {"semantic", "entity", "fact"}
    assert edge.shared == ("Maya",)
    assert edge.fact_ids == ("f1",)
    assert 0.7 < edge.weight <= 1.0


def test_projection_is_deterministic_degree_capped_and_spans_when_possible() -> None:
    edges = tuple(
        BaseEdge(f"p{left}", f"p{right}", "semantic", weight)
        for left, right, weight in (
            (1, 2, 0.9),
            (2, 1, 0.9),
            (2, 3, 0.8),
            (3, 2, 0.8),
            (3, 4, 0.7),
            (4, 3, 0.7),
            (1, 4, 0.6),
            (4, 1, 0.6),
            (1, 3, 0.5),
            (3, 1, 0.5),
        )
    )
    passages = tuple(ProjectionPassage(f"p{i}", f"n{i}") for i in range(1, 5))

    first = build_projection(_graph(edges), passages, {}, max_degree=2)
    second = build_projection(_graph(tuple(reversed(edges))), tuple(reversed(passages)), {}, max_degree=2)

    assert first == second
    degree = {f"n{i}": 0 for i in range(1, 5)}
    adjacency = {f"n{i}": set() for i in range(1, 5)}
    for edge in first:
        degree[edge.src_note_id] += 1
        degree[edge.dst_note_id] += 1
        adjacency[edge.src_note_id].add(edge.dst_note_id)
        adjacency[edge.dst_note_id].add(edge.src_note_id)
    assert max(degree.values()) <= 2
    seen = {"n1"}
    frontier = ["n1"]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    assert seen == set(adjacency)


def test_projection_edges_are_not_inserted_into_base_graph() -> None:
    graph = _graph((BaseEdge("p1", "p2", "semantic", 0.9),))
    original_matrix = graph.matrix.copy()

    rows = build_projection(
        graph,
        (ProjectionPassage("p1", "n1"), ProjectionPassage("p2", "n2")),
        {},
        max_degree=1,
    )

    assert rows
    assert (graph.matrix != original_matrix).nnz == 0
    assert graph.node_ids == ("p1", "p2", "e1", "e2")


def _graph(edges: tuple[BaseEdge, ...]) -> BaseGraph:
    return BaseGraph(
        build_id="build-1",
        node_ids=("p1", "p2", "e1", "e2"),
        node_kinds=("passage", "passage", "entity", "entity"),
        row_by_id={"p1": 0, "p2": 1, "e1": 2, "e2": 3},
        matrix=csr_matrix(np.zeros((4, 4), dtype=np.float64)),
        dangling_rows=(0, 1, 2, 3),
        edges=edges,
    )
