from __future__ import annotations

import numpy as np
import pytest
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


def test_projection_fact_mechanism_uses_provenance_for_hub_endpoints() -> None:
    """L1: a fact whose object is traversal-ineligible (date/pronoun hub, so
    no contains edges) still projects via its provenance source notes."""
    graph = _graph(
        (
            BaseEdge("p1", "e1", "contains", 0.6),
            BaseEdge("e1", "p1", "contains", 0.6),
            # e2 (a date) has NO contains edges; the fact was asserted in n2
            BaseEdge(
                "e1",
                "e2",
                "fact",
                0.4,
                direction="forward",
                fact_ids=("f9",),
                metadata={"predicate": "scheduled_for", "source_note_ids": ("n2",)},
            ),
        )
    )

    rows = build_projection(
        graph,
        passages=(ProjectionPassage("p1", "n1"), ProjectionPassage("p2", "n2")),
        entity_labels={"e1": "Launch", "e2": "July 18"},
        max_degree=3,
    )

    assert len(rows) == 1
    edge = rows[0]
    assert (edge.src_note_id, edge.dst_note_id) == ("n1", "n2")
    assert "fact" in edge.mechanisms
    assert edge.fact_ids == ("f9",)


def test_projection_dates_only_facts_contribute_no_edge() -> None:
    """L1: a fact with BOTH endpoints ineligible still adds nothing."""
    graph = _graph(
        (
            BaseEdge(
                "e1",
                "e2",
                "fact",
                0.4,
                direction="forward",
                fact_ids=("f9",),
                metadata={"predicate": "scheduled_for", "source_note_ids": ("n2",)},
            ),
        )
    )

    rows = build_projection(
        graph,
        passages=(ProjectionPassage("p1", "n1"), ProjectionPassage("p2", "n2")),
        entity_labels={},
        max_degree=3,
    )

    assert rows == ()


def test_spanning_forest_ignores_degree_cap_so_hub_notes_still_connect() -> None:
    """L2: a hub note bridging more components than max_degree keeps the map
    connected; the cap applies only to the fill phase."""
    leaf_count = 14
    node_ids = ["hub"] + [f"p{index}" for index in range(leaf_count)]
    edges = []
    for index in range(leaf_count):
        weight = 0.9 - index * 0.01
        edges.append(BaseEdge("hub", f"p{index}", "semantic", weight))
        edges.append(BaseEdge(f"p{index}", "hub", "semantic", weight))
    graph = BaseGraph(
        build_id="build-1",
        node_ids=tuple(node_ids),
        node_kinds=tuple(["passage"] * len(node_ids)),
        row_by_id={node_id: index for index, node_id in enumerate(node_ids)},
        matrix=csr_matrix(np.zeros((len(node_ids),) * 2, dtype=np.float64)),
        dangling_rows=tuple(range(len(node_ids))),
        edges=tuple(edges),
    )
    passages = tuple(
        ProjectionPassage(node_id, f"note-{node_id}") for node_id in node_ids
    )

    rows = build_projection(graph, passages, {}, max_degree=12)

    assert len(rows) == leaf_count  # every leaf is bridged despite the cap
    hub_degree = sum(
        1 for row in rows if "note-hub" in (row.src_note_id, row.dst_note_id)
    )
    assert hub_degree == leaf_count > 12


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


def test_mechanism_aggregation_is_strongest_plus_bounded_support() -> None:
    """Six weak shared-entity links must not noisy-or to ~1.0: the pair scores
    strongest + a small capped bonus, so one strong specific link outranks a
    pile of weak incidental ones."""
    edges = tuple(
        BaseEdge(f"p1", "e" + str(i), "contains", 0.4)
        for i in range(6)
    ) + tuple(
        BaseEdge(f"p2", "e" + str(i), "contains", 0.4)
        for i in range(6)
    )
    rows = build_projection(
        _graph(edges),
        passages=(ProjectionPassage("p1", "n1"), ProjectionPassage("p2", "n2")),
        entity_labels={f"e{i}": f"E{i}" for i in range(6)},
        max_degree=12,
    )
    assert len(rows) == 1
    entity_weight = rows[0].mechanisms["entity"]
    # noisy-or of six 0.16s would be ~0.65; strongest+support = 0.16 + 5*0.05 capped
    assert entity_weight == pytest.approx(0.4 * 0.4 + 0.15, abs=1e-6)
    assert rows[0].weight < 0.5


def test_fill_phase_prunes_low_value_edges_but_forest_keeps_connectivity(monkeypatch) -> None:
    from taurine_core import config

    monkeypatch.setattr(config, "PROJECTION_FILL_MIN_WEIGHT", 0.5)
    # chain n1-n2-n3 with weak links (forest must keep them attached), plus a
    # weak extra mesh edge n1-n3 that the fill floor must prune
    edges = tuple(
        BaseEdge(src, dst, "semantic", weight)
        for src, dst, weight in (
            ("p1", "p2", 0.45), ("p2", "p1", 0.45),
            ("p2", "p3", 0.44), ("p3", "p2", 0.44),
            ("p1", "p3", 0.40), ("p3", "p1", 0.40),
        )
    )
    passages = tuple(ProjectionPassage(f"p{i}", f"n{i}") for i in (1, 2, 3))
    rows = build_projection(_graph(edges), passages, {}, max_degree=12)
    pairs = {(row.src_note_id, row.dst_note_id) for row in rows}
    # forest edges survive below the floor (connectivity), mesh edge pruned
    assert ("n1", "n2") in pairs and ("n2", "n3") in pairs
    assert ("n1", "n3") not in pairs
