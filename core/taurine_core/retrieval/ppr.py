"""Typed reset composition and dangling-correct personalized PageRank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from taurine_core import config
from taurine_core.indexer.base_graph import BaseGraph


@dataclass(frozen=True)
class PPRResult:
    accepted: bool
    reason: str
    build_id: str
    ranks: dict[str, float]
    teleport: dict[str, float]
    passage_mass: float
    entity_mass: float
    iterations: int
    residual: float
    contributions: dict[str, tuple[tuple[str, float], ...]]


class TypedPPR:
    def __init__(
        self,
        graph: BaseGraph,
        *,
        damping: float = config.PPR_DAMPING,
        tolerance: float = config.PPR_TOL,
        max_iterations: int = config.PPR_MAX_ITERS,
    ) -> None:
        self.graph = graph
        self.damping = float(damping)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    def run(
        self,
        *,
        passage_reset: Mapping[str, float] | None = None,
        recognized_entity_reset: Mapping[str, float] | None = None,
        query_entity_reset: Mapping[str, float] | None = None,
        session_reset: Mapping[str, float] | None = None,
        warm_start: PPRResult | None = None,
    ) -> PPRResult:
        teleport = np.zeros(len(self.graph.node_ids), dtype=np.float64)
        self._add(teleport, passage_reset, config.PASSAGE_RESET_W)
        self._add(teleport, recognized_entity_reset, config.ENTITY_RESET_W)
        self._add(teleport, query_entity_reset, config.QUERY_ENTITY_RESET_W)
        self._add(teleport, session_reset, config.SESSION_RESET_W)
        total = float(teleport.sum())
        if total <= 0:
            return PPRResult(
                accepted=False,
                reason="empty_reset",
                build_id=self.graph.build_id,
                ranks={},
                teleport={},
                passage_mass=0.0,
                entity_mass=0.0,
                iterations=0,
                residual=0.0,
                contributions={},
            )
        teleport /= total
        rank = teleport.copy()
        if warm_start is not None and warm_start.build_id == self.graph.build_id:
            candidate = np.array(
                [warm_start.ranks.get(node_id, 0.0) for node_id in self.graph.node_ids],
                dtype=np.float64,
            )
            if candidate.sum() > 0:
                rank = candidate / candidate.sum()
        dangling = np.asarray(self.graph.dangling_rows, dtype=np.int64)
        residual = float("inf")
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            dangling_mass = float(rank[dangling].sum()) if dangling.size else 0.0
            next_rank = self.damping * (
                self.graph.matrix @ rank + dangling_mass * teleport
            ) + (1.0 - self.damping) * teleport
            next_total = float(next_rank.sum())
            if next_total > 0:
                next_rank /= next_total
            residual = float(np.abs(next_rank - rank).sum())
            rank = next_rank
            if residual <= self.tolerance:
                break
        ranks = {
            node_id: float(rank[index])
            for index, node_id in enumerate(self.graph.node_ids)
        }
        teleport_rows = {
            node_id: float(teleport[index])
            for index, node_id in enumerate(self.graph.node_ids)
            if teleport[index] > 0
        }
        passage_mass = sum(
            rank[index]
            for index, kind in enumerate(self.graph.node_kinds)
            if kind == "passage"
        )
        entity_mass = sum(
            rank[index]
            for index, kind in enumerate(self.graph.node_kinds)
            if kind == "entity"
        )
        return PPRResult(
            accepted=True,
            reason="converged" if residual <= self.tolerance else "max_iterations",
            build_id=self.graph.build_id,
            ranks=ranks,
            teleport=teleport_rows,
            passage_mass=float(passage_mass),
            entity_mass=float(entity_mass),
            iterations=iterations,
            residual=residual,
            contributions={},
        )

    def _add(
        self,
        target: np.ndarray,
        values: Mapping[str, float] | None,
        weight: float,
    ) -> None:
        for node_id, value in (values or {}).items():
            row = self.graph.row_by_id.get(node_id)
            if row is not None and value > 0:
                target[row] += float(weight) * float(value)
