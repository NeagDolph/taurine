from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from taurine_core.indexer.store import Index
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.schemas import NoteRecord
from taurine_core.retrieval.snapshot import ActiveSnapshot, IndexSnapshot


def test_index_snapshot_freezes_arrays_and_mappings(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "build-1", 1.0)

    assert snapshot.passage_matrix.flags.writeable is False
    assert snapshot.fact_matrix.flags.writeable is False
    assert isinstance(snapshot.alias_index, MappingProxyType)
    assert isinstance(snapshot.quality_ranks, MappingProxyType)
    with pytest.raises(ValueError, match="read-only"):
        snapshot.passage_matrix[0, 0] = 9.0
    with pytest.raises(TypeError):
        snapshot.alias_index["new"] = ("e2",)  # type: ignore[index]


def test_active_snapshot_swaps_whole_generation_atomically(tmp_path: Path) -> None:
    first = _snapshot(tmp_path, "build-1", 1.0)
    second = _snapshot(tmp_path, "build-2", 2.0)
    active = ActiveSnapshot(first)

    before = active.get()
    active.activate(second)
    after = active.get()

    assert before is first
    assert before.build_id == "build-1"
    assert float(before.passage_matrix[0, 0]) == 1.0
    assert after is second
    assert after.build_id == "build-2"
    assert float(after.passage_matrix[0, 0]) == 2.0
    assert after.calibration_hash == "cal-build-2"


def test_product_index_exposes_active_retrieval_snapshot_port(tmp_path: Path) -> None:
    first = _snapshot(tmp_path, "build-1", 1.0)
    second = _snapshot(tmp_path, "build-2", 2.0)
    active = ActiveSnapshot(first)
    index = Index(tmp_path / "product.db", knowledge_snapshots=active)
    try:
        assert index.active_snapshot_id() == "build-1"
        assert index.retrieval_snapshot() is first
        active.activate(second)
        assert index.active_snapshot_id() == "build-2"
        assert index.retrieval_snapshot() is second
    finally:
        index.close()


def test_snapshot_opens_its_generation_read_only_for_lexical_heads(tmp_path: Path) -> None:
    path = tmp_path / "build-1.db"
    writable = KnowledgeStore(path)
    try:
        writable.write_note(
            NoteRecord(
                note_id="n1",
                title="Maya launch",
                source="apple_notes",
                text="Maya owns the launch checklist.",
                content_hash="h1",
                created=1.0,
                modified=2.0,
            ),
            build_id="build-1",
            extraction_status="complete",
        )
    finally:
        writable.close()

    snapshot = _snapshot(tmp_path, "build-1", 1.0)
    store = snapshot.open_knowledge_store()
    try:
        assert store.read_only
        assert store.source_notes()[0].note_id == "n1"
        with pytest.raises(PermissionError):
            store.delete_note("n1")
    finally:
        store.close()


def _snapshot(tmp_path: Path, build_id: str, value: float) -> IndexSnapshot:
    return IndexSnapshot(
        build_id=build_id,
        database_path=tmp_path / f"{build_id}.db",
        calibration_hash=f"cal-{build_id}",
        passage_ids=(f"passage-{build_id}",),
        passage_matrix=np.array([[value]], dtype=np.float32),
        fact_ids=(f"fact-{build_id}",),
        fact_matrix=np.array([[value]], dtype=np.float32),
        alias_index={"maya": ("entity-1",)},
        base_graph=csr_matrix(np.array([[1.0]], dtype=np.float64)),
        projection=(),
        quality_ranks={"note-1": value},
    )
