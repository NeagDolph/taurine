"""Bounded, direction-preserving explanatory paths over the retrieval graph."""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from taurine_core.indexer.base_graph import BaseEdge, BaseGraph


@dataclass(frozen=True)
class PathCandidate:
    path_key: str
    node_ids: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    fact_ids: tuple[str, ...]
    weight: float


class EvidencePathFinder:
    def __init__(
        self,
        graph: BaseGraph,
        *,
        fact_status: Mapping[str, str],
        hub_entity_ids: set[str] | frozenset[str] = frozenset(),
        note_by_passage: Mapping[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.fact_status = dict(fact_status)
        self.hub_entity_ids = frozenset(hub_entity_ids)
        self.note_by_passage = dict(note_by_passage or {})
        adjacency: dict[str, list[BaseEdge]] = {}
        for edge in graph.edges:
            adjacency.setdefault(edge.src_id, []).append(edge)
        self._adjacency = {
            node_id: tuple(
                sorted(
                    edges,
                    key=lambda row: (
                        row.dst_id,
                        row.kind,
                        row.direction,
                        row.fact_ids,
                    ),
                )
            )
            for node_id, edges in adjacency.items()
        }

    def find(
        self,
        *,
        seed_ids: Sequence[str],
        target_id: str,
        max_edges: int = 4,
        max_paths: int = 2,
    ) -> tuple[PathCandidate, ...]:
        if max_edges < 1 or max_paths < 1 or target_id not in self.graph.row_by_id:
            return ()
        heap: list[
            tuple[
                float,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []
        for seed_id in sorted(set(seed_ids)):
            if seed_id not in self.graph.row_by_id or seed_id == target_id:
                continue
            notes = _passage_notes((seed_id,), self.note_by_passage)
            heapq.heappush(heap, (0.0, (seed_id,), (), (), notes))

        results: list[PathCandidate] = []
        explanations: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        while heap and len(results) < max_paths:
            cost, nodes, edge_kinds, fact_ids, passage_notes = heapq.heappop(heap)
            current = nodes[-1]
            if current == target_id:
                explanation = (edge_kinds, fact_ids)
                if explanation in explanations:
                    continue
                explanations.add(explanation)
                weight = math.exp(-cost)
                results.append(
                    PathCandidate(
                        path_key=_path_key(nodes, edge_kinds, fact_ids),
                        node_ids=nodes,
                        edge_kinds=edge_kinds,
                        fact_ids=fact_ids,
                        weight=weight,
                    )
                )
                continue
            if len(edge_kinds) >= max_edges:
                continue
            for edge in self._adjacency.get(current, ()):
                if edge.dst_id in nodes:
                    continue
                usable_fact_ids = self._usable_fact_ids(edge)
                if usable_fact_ids is None:
                    continue
                multiplier = min(1.0, max(float(edge.weight), 1e-12))
                if edge.dst_id in self.hub_entity_ids:
                    multiplier *= 0.5
                note_id = self.note_by_passage.get(edge.dst_id)
                if note_id and note_id in passage_notes:
                    multiplier *= 0.7
                next_notes = (
                    passage_notes
                    if not note_id
                    else tuple(sorted(set(passage_notes) | {note_id}))
                )
                kind = f"{edge.kind}:{edge.direction}"
                heapq.heappush(
                    heap,
                    (
                        cost - math.log(max(multiplier, 1e-12)),
                        nodes + (edge.dst_id,),
                        edge_kinds + (kind,),
                        fact_ids + usable_fact_ids,
                        next_notes,
                    ),
                )
        return tuple(results)

    def _usable_fact_ids(self, edge: BaseEdge) -> tuple[str, ...] | None:
        if edge.kind != "fact":
            return ()
        active = tuple(
            fact_id
            for fact_id in edge.fact_ids
            if self.fact_status.get(fact_id) == "active"
        )
        return active or None


def _path_key(
    nodes: tuple[str, ...],
    edge_kinds: tuple[str, ...],
    fact_ids: tuple[str, ...],
) -> str:
    payload = "\x1f".join((*nodes, *edge_kinds, *fact_ids))
    return "path-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _passage_notes(
    node_ids: Sequence[str], note_by_passage: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                note_by_passage[node_id]
                for node_id in node_ids
                if node_id in note_by_passage
            }
        )
    )
