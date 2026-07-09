"""Immutable, atomically replaceable retrieval index snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from taurine_core.indexer.knowledge_store import KnowledgeStore


@dataclass(frozen=True)
class IndexSnapshot:
    build_id: str
    database_path: Path
    calibration_hash: str
    passage_ids: tuple[str, ...]
    passage_matrix: np.ndarray
    fact_ids: tuple[str, ...]
    fact_matrix: np.ndarray
    alias_index: Mapping[str, tuple[str, ...]]
    base_graph: csr_matrix
    projection: tuple[Any, ...]
    quality_ranks: Mapping[str, float]

    def __post_init__(self) -> None:
        passage_matrix = np.asarray(self.passage_matrix, dtype=np.float32).copy()
        fact_matrix = np.asarray(self.fact_matrix, dtype=np.float32).copy()
        passage_matrix.flags.writeable = False
        fact_matrix.flags.writeable = False
        graph = self.base_graph.copy().tocsr()
        graph.data.flags.writeable = False
        graph.indices.flags.writeable = False
        graph.indptr.flags.writeable = False
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "passage_matrix", passage_matrix)
        object.__setattr__(self, "fact_matrix", fact_matrix)
        object.__setattr__(self, "base_graph", graph)
        object.__setattr__(self, "alias_index", MappingProxyType(dict(self.alias_index)))
        object.__setattr__(self, "quality_ranks", MappingProxyType(dict(self.quality_ranks)))

    def open_knowledge_store(self) -> "KnowledgeStore":
        """Open this generation's immutable DB for lexical/public row reads."""

        from taurine_core.indexer.knowledge_store import KnowledgeStore

        return KnowledgeStore(self.database_path, read_only=True)


class RetrievalSnapshotPort(Protocol):
    def get(self) -> IndexSnapshot: ...


class ActiveSnapshot:
    """One locked reference; readers never combine fields from two builds."""

    def __init__(self, snapshot: IndexSnapshot) -> None:
        self._snapshot = snapshot
        self._lock = RLock()

    def get(self) -> IndexSnapshot:
        with self._lock:
            return self._snapshot

    def activate(self, snapshot: IndexSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
