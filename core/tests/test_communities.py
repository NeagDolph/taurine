from __future__ import annotations

from scipy.sparse import csr_matrix


def _matrix(
    note_ids: list[str], edges: list[tuple[str, str]]
) -> tuple[list[str], csr_matrix]:
    row_of = {note_id: offset for offset, note_id in enumerate(note_ids)}
    rows: list[int] = []
    cols: list[int] = []
    for left, right in edges:
        left_row = row_of[left]
        right_row = row_of[right]
        rows.extend((left_row, right_row))
        cols.extend((right_row, left_row))
    data = [1.0] * len(rows)
    return note_ids, csr_matrix(
        (data, (rows, cols)), shape=(len(note_ids), len(note_ids))
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
    note_ids, matrix = _matrix(
        # Deliberately not sorted: iteration order must not depend on matrix row order.
        ["right-3", "left-2", "right-2", "left-0", "right-0", "left-3", "right-1", "left-1"],
        [*clique_edges, ("left-3", "right-0")],
    )

    from taurine_core.indexer.communities import label_propagation

    first = label_propagation(note_ids, matrix)
    second = label_propagation(note_ids, matrix)

    assert first == second
    assert len({first[note_id] for note_id in left}) == 1
    assert len({first[note_id] for note_id in right}) == 1
    assert first["left-3"] != first["right-0"]
    assert len(set(first.values())) == 2


def test_label_propagation_empty_graph_returns_empty_mapping() -> None:
    from taurine_core.indexer.communities import label_propagation

    assert label_propagation(*_matrix([], [])) == {}


def _projection_edge(src: str, dst: str):
    from taurine_core.indexer.projection import ProjectionEdge

    return ProjectionEdge(
        src_note_id=src,
        dst_note_id=dst,
        edge_type="semantic",
        weight=0.8,
        shared=(),
        fact_ids=(),
        mechanisms={"semantic": 0.8},
    )


def test_projection_communities_split_clusters_deterministically() -> None:
    from taurine_core.indexer.communities import projection_communities

    left = [f"left-{index}" for index in range(4)]
    right = [f"right-{index}" for index in range(4)]
    edges = tuple(
        _projection_edge(group[i], group[j])
        for group in (left, right)
        for i in range(len(group))
        for j in range(i + 1, len(group))
    )

    first = projection_communities(edges)
    second = projection_communities(edges)

    assert first == second
    assert set(first) == set(left) | set(right)  # only projected notes get labels
    assert len({first[note_id] for note_id in left}) == 1
    assert len({first[note_id] for note_id in right}) == 1
    assert first[left[0]] != first[right[0]]


def test_projection_communities_empty_projection_returns_empty_mapping() -> None:
    from taurine_core.indexer.communities import projection_communities

    assert projection_communities(()) == {}


# ── build-time weighted clustering (cluster_projection) ────────────────


def _weighted_edge(src: str, dst: str, mechanisms: dict[str, float]):
    from taurine_core.indexer.communities import cluster_weight
    from taurine_core.indexer.projection import ProjectionEdge

    weight = min(1.0, cluster_weight(mechanisms))
    return ProjectionEdge(
        src_note_id=src,
        dst_note_id=dst,
        edge_type="mixed",
        weight=weight,
        shared=(),
        fact_ids=(),
        mechanisms=mechanisms,
    )


def _clique(names: list[str], mechanisms: dict[str, float]) -> list:
    return [
        _weighted_edge(names[i], names[j], mechanisms)
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]


def test_cluster_weight_desaturates_entity_mechanism() -> None:
    from taurine_core.indexer.communities import cluster_weight

    # saturated entity noisy-or contributes at most 0.3
    assert cluster_weight({"entity": 1.0}) == 0.3
    assert cluster_weight({"entity": 5.0}) == 0.3  # defensive cap
    semantic = cluster_weight({"semantic": 0.71})
    entity = cluster_weight({"entity": 0.99})
    assert semantic > entity
    combined = cluster_weight({"semantic": 0.5, "fact": 0.4, "entity": 1.0})
    assert abs(combined - (0.5 + 0.7 * 0.4 + 0.3)) < 1e-9


def test_cluster_projection_splits_semantic_cliques_and_is_order_independent() -> None:
    from taurine_core.indexer.communities import cluster_projection

    left = [f"left-{index}" for index in range(5)]
    right = [f"right-{index}" for index in range(5)]
    edges = [
        *_clique(left, {"semantic": 0.8}),
        *_clique(right, {"semantic": 0.8}),
        _weighted_edge("left-0", "right-0", {"semantic": 0.2}),
    ]

    first = cluster_projection(edges)
    second = cluster_projection(list(reversed(edges)))

    assert first == second  # input order must not matter
    assert first == cluster_projection(edges)  # double-run equality
    assert len({first[note_id] for note_id in left}) == 1
    assert len({first[note_id] for note_id in right}) == 1
    assert first[left[0]] != first[right[0]]
    # contiguous size-ordered ids
    assert set(first.values()) == {0, 1}


def test_cluster_projection_ignores_entity_only_saturated_blob() -> None:
    """A register blob connected only by saturated entity edges (weight 0.3
    each) must not swallow a genuinely semantic cluster across a weak link."""
    from taurine_core.indexer.communities import cluster_projection

    semantic = [f"sem-{index}" for index in range(4)]
    blob = [f"blob-{index}" for index in range(4)]
    edges = [
        *_clique(semantic, {"semantic": 0.9}),
        *_clique(blob, {"entity": 1.0}),
        # bridge is entity-only: strongly discounted
        _weighted_edge("sem-0", "blob-0", {"entity": 1.0}),
    ]

    labels = cluster_projection(edges)
    assert len({labels[n] for n in semantic}) == 1
    assert len({labels[n] for n in blob}) == 1
    assert labels["sem-0"] != labels["blob-0"]


def test_cluster_projection_drops_below_min_weight_edges() -> None:
    from taurine_core.indexer.communities import cluster_projection

    trio = ["a-0", "a-1", "a-2"]
    edges = [
        *_clique(trio, {"semantic": 0.9}),
        # entity 0.4 → 0.12 combined < CLUSTER_MIN_WEIGHT: ignored, so the
        # pendant pair has no clustering edges at all → unclustered
        _weighted_edge("a-0", "b-0", {"entity": 0.4}),
        _weighted_edge("b-0", "b-1", {"entity": 0.4}),
    ]

    labels = cluster_projection(edges)
    assert len({labels[n] for n in trio}) == 1
    assert labels[trio[0]] >= 0
    assert labels["b-0"] == -1
    assert labels["b-1"] == -1


def test_cluster_projection_merges_tiny_clusters_into_best_neighbor() -> None:
    from taurine_core.indexer.communities import cluster_projection

    big = [f"big-{index}" for index in range(6)]
    edges = [
        *_clique(big, {"semantic": 0.9}),
        # a two-note satellite attached to the big cluster: below
        # CLUSTER_MIN_SIZE, must merge into it
        _weighted_edge("tiny-0", "tiny-1", {"semantic": 0.9}),
        _weighted_edge("tiny-0", "big-0", {"semantic": 0.5}),
    ]

    labels = cluster_projection(edges)
    assert labels["tiny-0"] == labels["big-0"]
    assert labels["tiny-1"] == labels["big-0"]


def test_cluster_projection_splits_oversized_clusters() -> None:
    from taurine_core import config
    from taurine_core.indexer.communities import cluster_projection

    # Two dense 24-note cliques joined by a single weak edge: modularity at
    # resolution 1.0 may keep them together (48 > CLUSTER_MAX_SIZE), so the
    # recursive re-cluster must separate them.
    left = [f"l-{index:02d}" for index in range(24)]
    right = [f"r-{index:02d}" for index in range(24)]
    edges = [
        *_clique(left, {"semantic": 0.9}),
        *_clique(right, {"semantic": 0.9}),
        _weighted_edge("l-00", "r-00", {"semantic": 0.9}),
    ]

    labels = cluster_projection(edges)
    sizes: dict[int, int] = {}
    for value in labels.values():
        if value >= 0:
            sizes[value] = sizes.get(value, 0) + 1
    assert all(size <= config.CLUSTER_MAX_SIZE for size in sizes.values())
    assert len({labels[n] for n in left}) == 1
    assert len({labels[n] for n in right}) == 1
    assert labels["l-00"] != labels["r-00"]


def test_cluster_projection_empty_and_ids_are_size_ordered() -> None:
    from taurine_core.indexer.communities import cluster_projection

    assert cluster_projection(()) == {}

    small = ["s-0", "s-1", "s-2"]
    large = [f"g-{index}" for index in range(5)]
    edges = [*_clique(large, {"semantic": 0.9}), *_clique(small, {"semantic": 0.9})]
    labels = cluster_projection(edges)
    assert {labels[n] for n in large} == {0}  # largest cluster gets id 0
    assert {labels[n] for n in small} == {1}


def test_fallback_clusters_remaps_legacy_labels_contiguously() -> None:
    from taurine_core.indexer.communities import fallback_clusters

    small = ["s-0", "s-1", "s-2"]
    large = [f"g-{index}" for index in range(4)]
    edges = tuple(
        _projection_edge(group[i], group[j])
        for group in (large, small)
        for i in range(len(group))
        for j in range(i + 1, len(group))
    )

    membership, records = fallback_clusters(edges)
    assert {membership[n] for n in large} == {0}
    assert {membership[n] for n in small} == {1}
    assert [record.cluster_id for record in records] == [0, 1]
    assert records[0].size == 4
    assert records[0].name == "Cluster 1 (4 notes)"
    assert records[1].name == "Cluster 2 (3 notes)"
    assert records[0].label_source == "fallback"
