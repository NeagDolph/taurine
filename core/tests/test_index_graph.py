from __future__ import annotations

from taurine_core.schemas import GraphEdge


def edges_between(index, left: str, right: str) -> GraphEdge | None:
    wanted = frozenset((left, right))
    for edge in index.graph().edges:
        if frozenset((edge.src, edge.dst)) == wanted:
            return edge
    return None


def degree_of(index, note_id: str) -> int:
    return sum(1 for edge in index.graph().edges if note_id in (edge.src, edge.dst))


def test_semantic_edge_from_similar_gists(fixture_index):
    # notes "roadmap-q3" and "roadmap-q4" have near-identical gists in the fixture
    edges = {(e.src, e.dst): e for e in fixture_index.graph().edges}
    assert any("roadmap-q3" in k and "roadmap-q4" in k for k in edges)


def test_common_topic_is_damped(fixture_index):
    # topic "work" is on >half the fixture notes -> alone it must not clear GRAPH_MIN_SHARED_W
    e = edges_between(fixture_index, "standup-notes", "grocery-list")
    assert e is None


def test_degree_cap(fixture_index_hub):
    # hub note connected to 20 candidates keeps only GRAPH_MAX_DEGREE strongest
    assert degree_of(fixture_index_hub, "hub") <= 12


def test_incremental_rebuild_touches_only_incident_edges(fixture_index):
    before = {frozenset((e.src, e.dst)) for e in fixture_index.graph().edges
              if "roadmap-q3" not in (e.src, e.dst)}
    fixture_index.rebuild_for("roadmap-q3")
    after = {frozenset((e.src, e.dst)) for e in fixture_index.graph().edges
             if "roadmap-q3" not in (e.src, e.dst)}
    assert before == after


def test_parallel_mechanisms_merge_into_mixed_edge(fixture_index):
    edge = edges_between(fixture_index, "roadmap-q3", "roadmap-q4")

    assert edge is not None
    assert edge.type == "mixed"
    assert "roadmap" in edge.shared
    assert "Taurine" in edge.shared


def test_neighbors_are_sorted_by_edge_weight(fixture_index):
    neighbors = fixture_index.neighbors("roadmap-q3")

    assert neighbors
    assert neighbors == sorted(neighbors, key=lambda item: item[1], reverse=True)
