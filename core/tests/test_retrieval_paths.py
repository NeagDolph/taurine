from __future__ import annotations

import pytest

from taurine_core.indexer.base_graph import BaseEdge, materialize_base_graph
from taurine_core.retrieval.paths import EvidencePathFinder


def test_paths_preserve_direction_fact_provenance_and_strongest_weight() -> None:
    graph = _graph(
        (
            BaseEdge("e1", "e2", "fact", 0.8, "forward", ("f1",)),
            BaseEdge("e2", "p2", "contains", 0.9),
            BaseEdge("e1", "p1", "contains", 0.4),
            BaseEdge("p1", "p2", "semantic", 0.5),
        )
    )
    finder = EvidencePathFinder(graph, fact_status={"f1": "active"})

    paths = finder.find(seed_ids=("e1",), target_id="p2")

    assert paths[0].node_ids == ("e1", "e2", "p2")
    assert paths[0].edge_kinds == ("fact:forward", "contains:symmetric")
    assert paths[0].fact_ids == ("f1",)
    assert paths[0].weight == pytest.approx(0.72)


def test_paths_reject_inactive_fact_edges_and_enforce_hop_cap() -> None:
    graph = _graph(
        (
            BaseEdge("e1", "e2", "fact", 0.9, "forward", ("f-old",)),
            BaseEdge("e2", "p2", "contains", 0.9),
            BaseEdge("e1", "p1", "contains", 0.9),
            BaseEdge("p1", "e3", "contains", 0.9),
            BaseEdge("e3", "p3", "contains", 0.9),
            BaseEdge("p3", "e2", "contains", 0.9),
        )
    )
    finder = EvidencePathFinder(graph, fact_status={"f-old": "superseded"})

    assert finder.find(seed_ids=("e1",), target_id="p2", max_edges=3) == ()
    assert finder.find(seed_ids=("e1",), target_id="p2", max_edges=5)


def test_paths_penalize_hubs_and_repeated_note_passages() -> None:
    graph = _graph(
        (
            BaseEdge("e1", "hub", "fact", 0.95, "forward", ("f-hub",)),
            BaseEdge("hub", "p2", "contains", 0.95),
            BaseEdge("e1", "specific", "fact", 0.8, "forward", ("f-specific",)),
            BaseEdge("specific", "p2", "contains", 0.9),
        )
    )
    finder = EvidencePathFinder(
        graph,
        fact_status={"f-hub": "active", "f-specific": "active"},
        hub_entity_ids={"hub"},
    )

    paths = finder.find(seed_ids=("e1",), target_id="p2")

    assert paths[0].fact_ids == ("f-specific",)


def _graph(edges: tuple[BaseEdge, ...]):
    nodes = tuple(sorted({node for edge in edges for node in (edge.src_id, edge.dst_id)}))
    return materialize_base_graph(
        build_id="build-1",
        node_ids=nodes,
        node_kinds=tuple("passage" if node.startswith("p") else "entity" for node in nodes),
        edges=edges,
    )
