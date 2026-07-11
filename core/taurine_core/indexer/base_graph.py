"""Uncapped passage/entity retrieval base graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from taurine_core import config
from taurine_core.indexer.facts import FactRecord, PREDICATES, adjudication_multiplier


@dataclass(frozen=True)
class PassageGraphInput:
    passage_id: str
    note_id: str
    ordinal: int
    vector: np.ndarray


@dataclass(frozen=True)
class EntityGraphInput:
    entity_id: str
    canonical: str
    etype: str
    df: int
    traversal_eligible: bool


@dataclass(frozen=True)
class PassageEntityInput:
    passage_id: str
    entity_id: str
    confidence: float
    mentions: int
    df: int


@dataclass(frozen=True)
class FactGraphInput:
    fact_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    source_note_id: str
    confidence: float
    status: str
    # facts.adjudication_multiplier output: mid-band adjudication scores
    # multiply this fact's edge weight (edge-quality item 1)
    adjudication_multiplier: float = 1.0


@dataclass(frozen=True)
class BaseEdge:
    src_id: str
    dst_id: str
    kind: str
    weight: float
    direction: str = "symmetric"
    fact_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaseGraph:
    build_id: str
    node_ids: tuple[str, ...]
    node_kinds: tuple[str, ...]
    row_by_id: Mapping[str, int]
    matrix: csr_matrix
    dangling_rows: tuple[int, ...]
    edges: tuple[BaseEdge, ...]


class BaseGraphBuilder:
    def __init__(
        self,
        *,
        build_id: str,
        semantic_k: int,
        semantic_floor: float,
        contains_scale: float = config.CONTAINS_WEIGHT_SCALE,
    ) -> None:
        self.build_id = build_id
        self.semantic_k = max(0, int(semantic_k))
        self.semantic_floor = float(semantic_floor)
        self.contains_scale = max(1e-9, float(contains_scale))

    def build(
        self,
        passages: Sequence[PassageGraphInput],
        entities: Sequence[EntityGraphInput],
        contains: Sequence[PassageEntityInput],
        facts: Sequence[FactRecord],
        *,
        activate_entities: set[str] | None = None,
    ) -> BaseGraph:
        passage_rows = sorted(passages, key=lambda row: row.passage_id)
        entity_rows = sorted(entities, key=lambda row: row.entity_id)
        node_ids = tuple(
            [row.passage_id for row in passage_rows]
            + [row.entity_id for row in entity_rows]
        )
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("base graph node IDs must be globally unique")
        node_kinds = tuple(
            ["passage"] * len(passage_rows) + ["entity"] * len(entity_rows)
        )
        row_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
        activated = activate_entities or set()
        entity_by_id = {row.entity_id: row for row in entity_rows}
        passage_by_id = {row.passage_id: row for row in passage_rows}
        note_count = max(1, len({row.note_id for row in passage_rows}))
        edges: list[BaseEdge] = []

        for edge in sorted(contains, key=lambda row: (row.passage_id, row.entity_id)):
            entity = entity_by_id.get(edge.entity_id)
            if edge.passage_id not in passage_by_id or entity is None:
                continue
            if not entity.traversal_eligible and entity.entity_id not in activated:
                continue
            idf = math.log((note_count + 1) / (max(0, edge.df) + 1)) + 1.0
            count_bonus = 1.0 + math.log(1.0 + min(max(1, edge.mentions), 3))
            # Frequency surprise (edge-quality item 3): an entity appearing in
            # more notes than DF_SOFT_CAP co-occurs with any one passage's
            # other entities roughly at chance, so its contribution decays
            # logarithmically past the cap. Chosen over a true per-pair NPMI
            # because pairwise co-occurrence counts are sparse/noisy at this
            # corpus size and note-cluster NPMI would be circular (clusters
            # are computed downstream of this graph); df beyond a soft cap is
            # a deterministic, monotone proxy for "no more than chance".
            surprise = 1.0 / (1.0 + math.log1p(max(0, edge.df - config.DF_SOFT_CAP)))
            # H2: divide by the calibrated scale before clipping so idf rarity
            # actually discriminates — without it every production edge
            # saturates at exactly 1.0 and column normalization erases rarity.
            weight = _clip(
                idf * edge.confidence * count_bonus * surprise / self.contains_scale
            )
            metadata = {"idf": idf, "mentions": edge.mentions, "surprise": surprise}
            edges.extend(
                (
                    BaseEdge(edge.passage_id, edge.entity_id, "contains", weight, metadata=metadata),
                    BaseEdge(edge.entity_id, edge.passage_id, "contains", weight, metadata=metadata),
                )
            )

        edges.extend(self._semantic_edges(passage_rows))
        graph_facts = (
            FactGraphInput(
                fact_id=fact.fact_id,
                subject_entity_id=fact.subject_entity_id,
                predicate=fact.predicate,
                object_entity_id=fact.object_entity_id,
                source_note_id=fact.source_note_id,
                confidence=fact.confidence,
                status=fact.status,
                adjudication_multiplier=adjudication_multiplier(
                    fact.adjudicated, fact.adjudication_score
                ),
            )
            for fact in facts
            if fact.subject_entity_id in row_by_id
            and fact.object_entity_id in row_by_id
        )
        edges.extend(grouped_fact_edges(tuple(graph_facts)))

        edges.sort(
            key=lambda row: (
                row.src_id,
                row.dst_id,
                row.kind,
                row.direction,
                row.fact_ids,
            )
        )
        matrix = _matrix(node_ids, node_kinds, row_by_id, edges)
        strengths = np.asarray(matrix.sum(axis=0)).ravel()
        dangling = tuple(int(index) for index in np.flatnonzero(strengths == 0.0))
        return BaseGraph(
            build_id=self.build_id,
            node_ids=node_ids,
            node_kinds=node_kinds,
            row_by_id=row_by_id,
            matrix=matrix,
            dangling_rows=dangling,
            edges=tuple(edges),
        )

    def _semantic_edges(
        self, passages: list[PassageGraphInput]
    ) -> list[BaseEdge]:
        if self.semantic_k <= 0 or len(passages) < 2:
            return []
        dimensions = {np.asarray(row.vector).shape for row in passages}
        if len(dimensions) != 1:
            raise ValueError("passage vectors must share one dimension")
        matrix = np.vstack([np.asarray(row.vector, dtype=np.float32) for row in passages])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("base graph passage vectors must be nonzero")
        matrix = matrix / norms
        similarities = matrix @ matrix.T
        # Mutual top-k (edge-quality item 5a): a pair becomes an edge only when
        # BOTH endpoints select each other — one-sided selections (register-
        # similar hubs dragging in everything above the floor) are dropped.
        directed: set[tuple[int, int]] = set()
        for left, passage in enumerate(passages):
            candidates: list[tuple[float, int]] = []
            for right, other in enumerate(passages):
                if left == right:
                    continue
                if passage.note_id == other.note_id and abs(passage.ordinal - other.ordinal) <= 1:
                    continue
                score = float(similarities[left, right])
                if score >= self.semantic_floor:
                    candidates.append((score, right))
            candidates.sort(key=lambda row: (-row[0], passages[row[1]].passage_id))
            for _, right in candidates[: self.semantic_k]:
                directed.add((left, right))
        selected = {
            (left, right)
            for left, right in directed
            if left < right and (right, left) in directed
        }
        edges: list[BaseEdge] = []
        for left, right in sorted(selected):
            score = float(similarities[left, right])
            edges.extend(
                (
                    BaseEdge(passages[left].passage_id, passages[right].passage_id, "semantic", score),
                    BaseEdge(passages[right].passage_id, passages[left].passage_id, "semantic", score),
                )
            )
        return edges


def grouped_fact_edges(facts: Sequence[FactGraphInput]) -> tuple[BaseEdge, ...]:
    """Build one directed edge pair per logical triple and retain its provenance."""

    groups: dict[tuple[str, str, str], list[FactGraphInput]] = {}
    for fact in facts:
        if fact.status != "active":
            continue
        key = (fact.subject_entity_id, fact.predicate, fact.object_entity_id)
        groups.setdefault(key, []).append(fact)

    edges: list[BaseEdge] = []
    for (subject_id, predicate, object_id), support in sorted(groups.items()):
        spec = PREDICATES[predicate]
        confidence_by_source: dict[str, float] = {}
        for fact in support:
            confidence_by_source[fact.source_note_id] = max(
                confidence_by_source.get(fact.source_note_id, 0.0),
                # edge-quality item 1: a mid-band adjudication score multiplies
                # the fact's contribution to its edge weight
                _clip(fact.confidence) * _clip(fact.adjudication_multiplier),
            )
        unsupported = math.prod(
            1.0 - confidence_by_source[source_id]
            for source_id in sorted(confidence_by_source)
        )
        support_confidence = 1.0 - unsupported
        forward_weight = spec.base_weight * support_confidence
        fact_ids = tuple(sorted({fact.fact_id for fact in support}))
        metadata = {
            "predicate": predicate,
            "source_note_ids": tuple(sorted(confidence_by_source)),
            "support_confidence": support_confidence,
        }
        edges.extend(
            (
                BaseEdge(
                    subject_id,
                    object_id,
                    "fact",
                    forward_weight,
                    direction="forward",
                    fact_ids=fact_ids,
                    metadata=metadata,
                ),
                BaseEdge(
                    object_id,
                    subject_id,
                    "fact",
                    forward_weight * spec.reverse_traversal_weight,
                    direction="reverse",
                    fact_ids=fact_ids,
                    metadata=metadata,
                ),
            )
        )
    return tuple(edges)


def _matrix(
    node_ids: tuple[str, ...],
    node_kinds: tuple[str, ...],
    row_by_id: Mapping[str, int],
    edges: list[BaseEdge],
) -> csr_matrix:
    if not node_ids:
        return csr_matrix((0, 0), dtype=np.float64)
    weights: dict[tuple[int, int], float] = {}
    for edge in edges:
        src = row_by_id.get(edge.src_id)
        dst = row_by_id.get(edge.dst_id)
        if src is None or dst is None or edge.weight <= 0:
            continue
        weights[(dst, src)] = weights.get((dst, src), 0.0) + float(edge.weight)
    if not weights:
        return csr_matrix((len(node_ids), len(node_ids)), dtype=np.float64)
    _dampen_entity_hubs(weights, node_kinds)
    rows, columns, data = zip(
        *((row, column, weight) for (row, column), weight in sorted(weights.items())),
        strict=True,
    )
    matrix = coo_matrix(
        (data, (rows, columns)),
        shape=(len(node_ids), len(node_ids)),
        dtype=np.float64,
    ).tocsr()
    strengths = np.asarray(matrix.sum(axis=0)).ravel()
    inverse = np.zeros_like(strengths)
    nonzero = strengths > 0
    inverse[nonzero] = 1.0 / strengths[nonzero]
    return (matrix @ diags(inverse)).tocsr()


def _dampen_entity_hubs(
    weights: dict[tuple[int, int], float],
    node_kinds: tuple[str, ...],
) -> None:
    """Cap random-walk mass through entity mega-hubs (edge-quality item 3).

    An entity column with more than HUB_DEGREE_CAP out-neighbors has its
    outgoing weights scaled by HUB_DEGREE_CAP/degree BEFORE column
    normalization, with the removed mass parked on a self-loop sized so the
    column still normalizes to exactly 1 (column-stochastic invariant).
    After normalization the hub forwards only cap/degree of arriving mass and
    retains the rest on itself, where PPR damping decays it — instead of
    spraying it across (e.g.) the owner entity's 80+ passages. Uniform
    scaling preserves the relative ordering of the hub's neighbors. This is
    the ONE dampening site: every matrix (build-time and the per-query
    overlay in retrieval/evidence.py) is materialized from raw edge rows
    through this function, so mass is never dampened twice.
    """

    cap = int(config.HUB_DEGREE_CAP)
    if cap <= 0:
        return
    by_column: dict[int, list[tuple[int, int]]] = {}
    for key in weights:
        row, column = key
        if row != column and node_kinds[column] == "entity":
            by_column.setdefault(column, []).append(key)
    for column, keys in sorted(by_column.items()):
        if len(keys) <= cap:
            continue
        factor = cap / len(keys)
        outgoing = sum(weights[key] for key in keys)
        for key in keys:
            weights[key] *= factor
        weights[(column, column)] = (
            weights.get((column, column), 0.0) + (1.0 - factor) * outgoing
        )


def materialize_base_graph(
    *,
    build_id: str,
    node_ids: Sequence[str],
    node_kinds: Sequence[str],
    edges: Sequence[BaseEdge],
) -> BaseGraph:
    """Materialize normalized sparse state from deterministically ordered graph rows."""

    paired = sorted(zip(node_ids, node_kinds, strict=True), key=lambda row: row[0])
    passages = [row for row in paired if row[1] == "passage"]
    entities = [row for row in paired if row[1] == "entity"]
    if len(passages) + len(entities) != len(paired):
        raise ValueError("base graph node kinds must be passage or entity")
    ordered = passages + entities
    ordered_ids = tuple(row[0] for row in ordered)
    ordered_kinds = tuple(row[1] for row in ordered)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("base graph node IDs must be globally unique")
    row_by_id = {node_id: index for index, node_id in enumerate(ordered_ids)}
    ordered_edges = tuple(
        sorted(
            edges,
            key=lambda row: (
                row.src_id,
                row.dst_id,
                row.kind,
                row.direction,
                row.fact_ids,
            ),
        )
    )
    matrix = _matrix(ordered_ids, ordered_kinds, row_by_id, list(ordered_edges))
    strengths = np.asarray(matrix.sum(axis=0)).ravel()
    dangling = tuple(int(index) for index in np.flatnonzero(strengths == 0.0))
    return BaseGraph(
        build_id=build_id,
        node_ids=ordered_ids,
        node_kinds=ordered_kinds,
        row_by_id=row_by_id,
        matrix=matrix,
        dangling_rows=dangling,
        edges=ordered_edges,
    )


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
