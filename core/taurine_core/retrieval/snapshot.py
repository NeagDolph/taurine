"""Immutable, atomically replaceable retrieval index snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np

from taurine_core import config
from taurine_core.indexer.base_graph import BaseGraph

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
    base_graph: BaseGraph
    projection: tuple[Any, ...]
    quality_ranks: Mapping[str, float]
    model_manifest: Mapping[str, str] = field(default_factory=dict)
    config_manifest: Mapping[str, str] = field(default_factory=dict)
    # Persisted note clusters (or the legacy label-propagation fallback when
    # the database predates the cluster tables). Additive: existing
    # constructors default to "no clusters".
    clusters: tuple[Any, ...] = ()
    note_clusters: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        passage_matrix = np.asarray(self.passage_matrix, dtype=np.float32).copy()
        fact_matrix = np.asarray(self.fact_matrix, dtype=np.float32).copy()
        if passage_matrix.ndim != 2 or passage_matrix.shape[0] != len(self.passage_ids):
            raise ValueError("snapshot passage IDs and matrix rows do not align")
        if fact_matrix.ndim != 2 or fact_matrix.shape[0] != len(self.fact_ids):
            raise ValueError("snapshot fact IDs and matrix rows do not align")
        passage_matrix.flags.writeable = False
        fact_matrix.flags.writeable = False
        graph_matrix = self.base_graph.matrix.copy().tocsr()
        graph_matrix.data.flags.writeable = False
        graph_matrix.indices.flags.writeable = False
        graph_matrix.indptr.flags.writeable = False
        graph = replace(
            self.base_graph,
            row_by_id=MappingProxyType(dict(self.base_graph.row_by_id)),
            matrix=graph_matrix,
            edges=tuple(
                replace(edge, metadata=MappingProxyType(dict(edge.metadata)))
                for edge in self.base_graph.edges
            ),
        )
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "passage_matrix", passage_matrix)
        object.__setattr__(self, "fact_matrix", fact_matrix)
        object.__setattr__(self, "base_graph", graph)
        object.__setattr__(self, "alias_index", MappingProxyType(dict(self.alias_index)))
        object.__setattr__(self, "quality_ranks", MappingProxyType(dict(self.quality_ranks)))
        object.__setattr__(self, "model_manifest", MappingProxyType(dict(self.model_manifest)))
        object.__setattr__(self, "config_manifest", MappingProxyType(dict(self.config_manifest)))
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(
            self,
            "note_clusters",
            MappingProxyType(
                {str(key): int(value) for key, value in dict(self.note_clusters).items()}
            ),
        )

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


def load_index_snapshot(
    database_path: str | Path,
    *,
    build_id: str | None = None,
) -> IndexSnapshot:
    """Load one complete immutable generation from its normalized SQLite rows."""

    from taurine_core.indexer.knowledge_store import KnowledgeStore

    path = Path(database_path)
    store = KnowledgeStore(path, read_only=True)
    try:
        build = store.completed_build(build_id)
        if build is None:
            requested = build_id or "latest"
            raise ValueError(f"complete knowledge build not found: {requested}")
        passage_ids, passage_matrix = _embedding_matrix(
            store.passage_embeddings(build_id=build.build_id),
            id_attribute="passage_id",
            kind="passage",
        )
        fact_ids, fact_matrix = _embedding_matrix(
            store.fact_embeddings(build_id=build.build_id),
            id_attribute="fact_id",
            kind="fact",
        )
        if passage_matrix.shape[1] and fact_matrix.shape[1]:
            if passage_matrix.shape[1] != fact_matrix.shape[1]:
                raise ValueError("passage and fact embedding dimensions do not match")
        elif passage_matrix.shape[1]:
            fact_matrix = np.empty((0, passage_matrix.shape[1]), dtype=np.float32)
        elif fact_matrix.shape[1]:
            passage_matrix = np.empty((0, fact_matrix.shape[1]), dtype=np.float32)
        else:
            dimension = config.embed_dim()
            passage_matrix = np.empty((0, dimension), dtype=np.float32)
            fact_matrix = np.empty((0, dimension), dtype=np.float32)
        projection = store.projection(build.build_id)
        if store.has_cluster_tables():
            clusters = store.clusters(build.build_id)
            note_clusters: Mapping[str, int] = store.note_clusters(build.build_id)
        else:
            # Pre-cluster production database: keep serving via the legacy
            # activation-time label propagation, remapped to the contiguous
            # cluster-id contract with fallback names.
            from taurine_core.indexer.communities import fallback_clusters

            note_clusters, clusters = fallback_clusters(projection)
        return IndexSnapshot(
            build_id=build.build_id,
            database_path=path,
            calibration_hash=build.calibration_hash,
            passage_ids=passage_ids,
            passage_matrix=passage_matrix,
            fact_ids=fact_ids,
            fact_matrix=fact_matrix,
            alias_index=store.alias_index(build_id=build.build_id),
            base_graph=store.base_graph(build.build_id),
            projection=projection,
            quality_ranks=store.quality_ranks(build.build_id),
            model_manifest=build.model_manifest,
            config_manifest=build.config_manifest,
            clusters=clusters,
            note_clusters=note_clusters,
        )
    finally:
        store.close()


def load_snapshot(
    database_path: str | Path,
    *,
    build_id: str | None = None,
) -> IndexSnapshot:
    """Concise public alias for the generation loader."""

    return load_index_snapshot(database_path, build_id=build_id)


def _embedding_matrix(
    rows: tuple[Any, ...],
    *,
    id_attribute: str,
    kind: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    ordered = sorted(rows, key=lambda row: str(getattr(row, id_attribute)))
    ids = tuple(str(getattr(row, id_attribute)) for row in ordered)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {kind} embedding ID")
    if not ordered:
        return (), np.empty((0, 0), dtype=np.float32)
    dimensions = {int(row.dim) for row in ordered}
    if len(dimensions) != 1:
        raise ValueError(f"{kind} embeddings have inconsistent dimensions")
    dimension = dimensions.pop()
    matrix = np.vstack(
        [np.asarray(row.vector, dtype=np.float32) for row in ordered]
    ).astype(np.float32, copy=False)
    expected = (len(ordered), dimension)
    if matrix.shape != expected:
        raise ValueError(f"{kind} embedding matrix has shape {matrix.shape}, expected {expected}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{kind} embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"{kind} embedding matrix contains a zero-norm row")
    return ids, matrix / norms
