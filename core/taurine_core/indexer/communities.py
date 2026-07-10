from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from taurine_core import config

if TYPE_CHECKING:
    from taurine_core.indexer.projection import ProjectionEdge


@dataclass(frozen=True)
class ClusterRecord:
    """One persisted (or fallback-synthesized) note cluster."""

    cluster_id: int
    size: int
    name: str = ""
    description: str = ""
    label_source: str = "fallback"
    top_entities: tuple[str, ...] = ()


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


def fallback_clusters(
    edges: Sequence["ProjectionEdge"],
) -> tuple[dict[str, int], tuple[ClusterRecord, ...]]:
    """Legacy-DB fallback: label propagation remapped to the cluster contract.

    Used when the active database predates the persisted cluster tables. The
    raw label-propagation ids are arbitrary row indices; remap them to
    contiguous size-ordered ints and synthesize fallback-named records so the
    /graph clusters contract always holds.
    """
    raw = projection_communities(edges)
    groups: dict[int, list[str]] = {}
    for note_id, label in raw.items():
        groups.setdefault(label, []).append(note_id)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    membership: dict[str, int] = {}
    records: list[ClusterRecord] = []
    for cluster_id, members in enumerate(ordered):
        for note_id in members:
            membership[note_id] = cluster_id
        records.append(
            ClusterRecord(
                cluster_id=cluster_id,
                size=len(members),
                name=f"Cluster {cluster_id + 1} ({len(members)} notes)",
                description="",
                label_source="fallback",
            )
        )
    return membership, tuple(records)


# ── build-time weighted clustering ─────────────────────────────────────


def cluster_weight(mechanisms: Mapping[str, float]) -> float:
    """Clustering edge weight from the projection's mechanism map.

    The entity mechanism noisy-ors mention products across every shared
    entity, so on real corpora most entity components saturate at ~1.0 and
    stop discriminating; it is capped at 1.0 and strongly down-weighted while
    semantic and fact components keep their resolution.
    """
    return (
        config.CLUSTER_W_SEMANTIC * max(0.0, float(mechanisms.get("semantic", 0.0)))
        + config.CLUSTER_W_FACT * max(0.0, float(mechanisms.get("fact", 0.0)))
        + config.CLUSTER_W_ENTITY
        * min(1.0, max(0.0, float(mechanisms.get("entity", 0.0))))
    )


def cluster_projection(
    edges: Sequence["ProjectionEdge"],
    *,
    resolution: float | None = None,
) -> dict[str, int]:
    """Deterministic weighted greedy-modularity clusters over the projection.

    Louvain-style: single-level local moves followed by aggregation, repeated
    until stable. Determinism (spec §9.3): nodes are visited in sorted
    note-id order, ties break by (gain desc, smallest target community id),
    and every reduction iterates sorted structures — the result is identical
    for any input edge order.

    Post-processing: clusters above CLUSTER_MAX_SIZE are recursively
    re-clustered at resolution×1.6 (max 2 levels); clusters below
    CLUSTER_MIN_SIZE merge into the neighbor with the highest inter-cluster
    weight (or drop to -1 with no neighbor). Final ids are contiguous 0..N-1
    ordered by size desc; notes without a surviving cluster map to -1.
    """
    base_resolution = (
        float(config.CLUSTER_RESOLUTION) if resolution is None else float(resolution)
    )
    note_ids = sorted(
        {note_id for edge in edges for note_id in (edge.src_note_id, edge.dst_note_id)}
    )
    if not note_ids:
        return {}
    row_of = {note_id: row for row, note_id in enumerate(note_ids)}
    adjacency: list[dict[int, float]] = [{} for _ in note_ids]
    pair_weights: dict[tuple[int, int], float] = {}
    for edge in edges:
        weight = cluster_weight(edge.mechanisms)
        if weight < float(config.CLUSTER_MIN_WEIGHT):
            continue
        left, right = row_of[edge.src_note_id], row_of[edge.dst_note_id]
        if left == right:
            continue
        pair = (left, right) if left < right else (right, left)
        pair_weights[pair] = pair_weights.get(pair, 0.0) + weight
    for (left, right), weight in sorted(pair_weights.items()):
        adjacency[left][right] = weight
        adjacency[right][left] = weight

    membership = _louvain(adjacency, base_resolution)
    membership = _split_oversized(membership, adjacency, base_resolution, depth=0)
    membership = _merge_small(membership, adjacency)

    groups: dict[int, list[int]] = {}
    for node, label in enumerate(membership):
        if label >= 0:
            groups.setdefault(label, []).append(node)
    ordered = sorted(
        groups.values(), key=lambda members: (-len(members), note_ids[members[0]])
    )
    result = {note_id: -1 for note_id in note_ids}
    for cluster_id, members in enumerate(ordered):
        for node in members:
            result[note_ids[node]] = cluster_id
    return result


_GAIN_EPSILON = 1e-12
_MAX_PASSES = 12
_MAX_LEVELS = 12


def _louvain(adjacency: Sequence[Mapping[int, float]], resolution: float) -> list[int]:
    """Deterministic multi-level greedy modularity over a symmetric graph.

    Returns a community label per node (labels are arbitrary ints, later
    normalized by the caller).
    """
    count = len(adjacency)
    membership = list(range(count))
    graph: list[dict[int, float]] = [dict(neighbors) for neighbors in adjacency]
    self_loops = [0.0] * count
    for _level in range(_MAX_LEVELS):
        community = _one_level(graph, self_loops, resolution)
        relabel: dict[int, int] = {}
        for node in range(len(graph)):
            label = community[node]
            if label not in relabel:
                relabel[label] = len(relabel)
        community = [relabel[label] for label in community]
        membership = [community[label] for label in membership]
        merged = len(relabel)
        if merged == len(graph):
            break
        aggregated: list[dict[int, float]] = [{} for _ in range(merged)]
        aggregated_loops = [0.0] * merged
        for node, neighbors in enumerate(graph):
            source = community[node]
            aggregated_loops[source] += self_loops[node]
            for other, weight in sorted(neighbors.items()):
                target = community[other]
                if target == source:
                    if other > node:  # each intra edge appears twice; count once
                        aggregated_loops[source] += weight
                else:
                    aggregated[source][target] = (
                        aggregated[source].get(target, 0.0) + weight
                    )
        graph = aggregated
        self_loops = aggregated_loops
    return membership


def _one_level(
    graph: Sequence[Mapping[int, float]],
    self_loops: Sequence[float],
    resolution: float,
) -> list[int]:
    count = len(graph)
    degrees = [
        sum(graph[node].values()) + 2.0 * self_loops[node] for node in range(count)
    ]
    total = sum(degrees)  # == 2m
    community = list(range(count))
    if total <= 0.0:
        return community
    community_total = degrees.copy()
    for _pass in range(_MAX_PASSES):
        moved = False
        for node in range(count):
            current = community[node]
            community_total[current] -= degrees[node]
            links: dict[int, float] = {}
            for other, weight in graph[node].items():
                target = community[other]
                links[target] = links.get(target, 0.0) + weight
            best_label = current
            best_gain = (
                links.get(current, 0.0)
                - resolution * degrees[node] * community_total[current] / total
            )
            for label in sorted(links):
                gain = (
                    links[label]
                    - resolution * degrees[node] * community_total[label] / total
                )
                if gain > best_gain + _GAIN_EPSILON or (
                    abs(gain - best_gain) <= _GAIN_EPSILON and label < best_label
                ):
                    best_label, best_gain = label, gain
            if best_label != current:
                moved = True
                community[node] = best_label
            community_total[best_label] += degrees[node]
        if not moved:
            break
    return community


def _split_oversized(
    membership: list[int],
    adjacency: Sequence[Mapping[int, float]],
    resolution: float,
    *,
    depth: int,
) -> list[int]:
    if depth >= 2:
        return membership
    groups: dict[int, list[int]] = {}
    for node, label in enumerate(membership):
        groups.setdefault(label, []).append(node)
    next_label = max(groups) + 1
    for label in sorted(groups):
        members = groups[label]
        if len(members) <= int(config.CLUSTER_MAX_SIZE):
            continue
        index_of = {node: index for index, node in enumerate(members)}
        sub_adjacency = [
            {
                index_of[other]: weight
                for other, weight in adjacency[node].items()
                if other in index_of
            }
            for node in members
        ]
        sub_resolution = resolution * 1.6
        sub_membership = _louvain(sub_adjacency, sub_resolution)
        if len(set(sub_membership)) <= 1:
            continue  # indivisible at the sharper resolution: keep as is
        sub_membership = _split_oversized(
            sub_membership, sub_adjacency, sub_resolution, depth=depth + 1
        )
        remap: dict[int, int] = {}
        for position, sub_label in enumerate(sub_membership):
            if sub_label not in remap:
                remap[sub_label] = next_label
                next_label += 1
            membership[members[position]] = remap[sub_label]
    return membership


def _merge_small(
    membership: list[int],
    adjacency: Sequence[Mapping[int, float]],
) -> list[int]:
    groups: dict[int, list[int]] = {}
    for node, label in enumerate(membership):
        groups.setdefault(label, []).append(node)
    # smallest first so chains of fragments coalesce into their shared hub
    order = sorted(groups, key=lambda label: (len(groups[label]), groups[label][0]))
    for label in order:
        members = [node for node in range(len(membership)) if membership[node] == label]
        if not members or len(members) >= int(config.CLUSTER_MIN_SIZE):
            continue
        inter: dict[int, float] = {}
        for node in members:
            for other, weight in adjacency[node].items():
                target = membership[other]
                if target != label and target >= 0:
                    inter[target] = inter.get(target, 0.0) + weight
        if not inter:
            for node in members:
                membership[node] = -1
            continue
        target = min(inter, key=lambda candidate: (-inter[candidate], candidate))
        for node in members:
            membership[node] = target
    return membership
