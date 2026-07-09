"""Uncapped passage/entity retrieval base graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from taurine_core.indexer.facts import FactRecord, PREDICATES


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
    ) -> None:
        self.build_id = build_id
        self.semantic_k = max(0, int(semantic_k))
        self.semantic_floor = float(semantic_floor)

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
            weight = _clip(idf * edge.confidence * count_bonus)
            metadata = {"idf": idf, "mentions": edge.mentions}
            edges.extend(
                (
                    BaseEdge(edge.passage_id, edge.entity_id, "contains", weight, metadata=metadata),
                    BaseEdge(edge.entity_id, edge.passage_id, "contains", weight, metadata=metadata),
                )
            )

        edges.extend(self._semantic_edges(passage_rows))
        for fact in sorted(facts, key=lambda row: row.fact_id):
            if fact.status != "active":
                continue
            if (
                fact.subject_entity_id not in row_by_id
                or fact.object_entity_id not in row_by_id
            ):
                continue
            spec = PREDICATES[fact.predicate]
            temporal_weight = 1.0
            forward_weight = spec.base_weight * fact.confidence * temporal_weight
            reverse_weight = forward_weight * spec.reverse_traversal_weight
            edges.append(
                BaseEdge(
                    fact.subject_entity_id,
                    fact.object_entity_id,
                    "fact",
                    forward_weight,
                    direction="forward",
                    fact_ids=(fact.fact_id,),
                    metadata={"predicate": fact.predicate},
                )
            )
            edges.append(
                BaseEdge(
                    fact.object_entity_id,
                    fact.subject_entity_id,
                    "fact",
                    reverse_weight,
                    direction="reverse",
                    fact_ids=(fact.fact_id,),
                    metadata={"predicate": fact.predicate},
                )
            )

        edges.sort(
            key=lambda row: (
                row.src_id,
                row.dst_id,
                row.kind,
                row.direction,
                row.fact_ids,
            )
        )
        matrix = _matrix(node_ids, row_by_id, edges)
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
        selected: set[tuple[int, int]] = set()
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
                selected.add((min(left, right), max(left, right)))
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


def _matrix(
    node_ids: tuple[str, ...],
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


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
