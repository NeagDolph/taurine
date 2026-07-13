"""Cluster naming, build-time persistence, snapshot serving, and the
pre-cluster legacy fallback."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np

from tests.conftest import make_note
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.indexer.cluster_labels import (
    ClusterSummary,
    build_cluster_summaries,
    fallback_label,
)
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.retrieval.snapshot import load_index_snapshot


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def _grouped_embed(texts: list[str]) -> np.ndarray:
    """Deterministic 16-dim embedder: 'billing' texts cluster on axis 0,
    'travel' texts on axis 1, everything else pseudo-random."""

    rows = []
    for text in texts:
        lowered = text.lower()
        row = np.zeros(16, dtype=np.float32)
        noise = np.frombuffer(
            hashlib.sha256(text.encode()).digest()[:14], dtype=np.uint8
        ).astype(np.float32)
        if "billing" in lowered:
            row[0] = 100.0
            row[2:] = noise / 255.0
        elif "travel" in lowered:
            row[1] = 100.0
            row[2:] = noise / 255.0
        else:
            row[2:] = noise + 1.0
        rows.append(row / np.linalg.norm(row))
    return np.vstack(rows)


def _cluster_notes():
    billing = [
        make_note(
            f"bill-{index}",
            f"Billing {index}",
            f"billing invoice reconciliation pass number {index} for the billing rollout",
            modified=float(1000 + index),
        )
        for index in range(4)
    ]
    travel = [
        make_note(
            f"trip-{index}",
            f"Travel {index}",
            f"travel packing list and flights for the tokyo travel plan {index}",
            modified=float(2000 + index),
        )
        for index in range(3)
    ]
    return billing + travel


# ── summaries + fallback names ─────────────────────────────────────────


def test_build_cluster_summaries_orders_entities_titles_and_sources() -> None:
    membership = {"n1": 0, "n2": 0, "n3": 0, "iso": -1}
    summaries = build_cluster_summaries(
        membership,
        titles={"n1": "Alpha", "n2": "Beta", "n3": "Gamma", "iso": "Iso"},
        display_lines={"n1": "alpha line", "n2": "beta line", "n3": "", "iso": "x"},
        sources={"n1": "imessage", "n2": "imessage", "n3": "apple_notes", "iso": "gmail"},
        value_scores={"n1": 0.2, "n2": 0.9, "n3": 0.5},
        note_entities={"n1": ["Ernest Chae", "Adi Yadhav"], "n2": ["Ernest Chae"]},
    )

    assert len(summaries) == 1  # -1 members never form a cluster
    summary = summaries[0]
    assert summary.cluster_id == 0
    assert summary.size == 3
    assert summary.top_entities == ("Ernest Chae", "Adi Yadhav")
    # ranked by value score desc, then note id; title + display line ≤90 chars
    assert summary.note_lines == ("Beta — beta line", "Gamma", "Alpha — alpha line")
    assert summary.source_mix == "mostly imessage"
    assert summary.dominant_source == "imessage"


def test_fallback_label_prefers_entities_then_source() -> None:
    named = ClusterSummary(
        cluster_id=0,
        size=9,
        top_entities=("Ernest Chae", "Adi Yadhav", "Maya", "Noor"),
        note_lines=(),
        source_mix="mostly imessage",
        dominant_source="imessage",
    )
    name, description = fallback_label(named)
    assert name == "Ernest Chae, Adi Yadhav +2"
    assert "9 related notes" in description

    unnamed = ClusterSummary(
        cluster_id=1,
        size=4,
        top_entities=(),
        note_lines=(),
        source_mix="mostly imessage",
        dominant_source="imessage",
    )
    assert fallback_label(unnamed)[0] == "Group Chats"


# ── build persistence + snapshot serving ───────────────────────────────


def _build(tmp_path: Path):
    database = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=_grouped_embed,
            embedding_dim=16,
        )
    ).build(_cluster_notes(), database, replace_derived=True)
    return database, report


def test_build_persists_fallback_labeled_clusters_and_membership(
    tmp_path: Path,
) -> None:
    database, report = _build(tmp_path)

    assert report.cluster_count == 2
    assert sum(report.cluster_size_histogram.values()) == 2
    assert "clusters" in report.stages

    store = KnowledgeStore(database, read_only=True)
    try:
        assert store.has_cluster_tables()
        records = store.clusters(report.build_id)
        membership = store.note_clusters(report.build_id)
    finally:
        store.close()
    assert [record.cluster_id for record in records] == [0, 1]
    assert records[0].size == 4  # size-ordered: billing (4) before travel (3)
    assert all(record.name for record in records)
    assert all(record.label_source == "fallback" for record in records)
    assert set(membership) == {note.note_id for note in _cluster_notes()}
    assert {membership[f"bill-{i}"] for i in range(4)} == {0}
    assert {membership[f"trip-{i}"] for i in range(3)} == {1}

    snapshot = load_index_snapshot(database)
    assert dict(snapshot.note_clusters) == membership
    assert all(row.name for row in snapshot.clusters)
    assert all(row.label_source == "fallback" for row in snapshot.clusters)


def test_build_without_labeler_uses_fallback_names(tmp_path: Path) -> None:
    database, report = _build(tmp_path)
    assert report.cluster_count == 2
    snapshot = load_index_snapshot(database)
    assert all(row.label_source == "fallback" for row in snapshot.clusters)


def test_snapshot_falls_back_when_cluster_tables_missing(tmp_path: Path) -> None:
    """The current production DB predates the cluster tables: simulate by
    dropping them and assert the snapshot (and runtime boot) still serve the
    cluster contract via remapped label propagation."""

    database, report = _build(tmp_path)
    with sqlite3.connect(database) as conn:
        conn.execute("DROP TABLE note_clusters")
        conn.execute("DROP TABLE clusters")

    store = KnowledgeStore(database, read_only=True)  # schema still accepted
    try:
        assert not store.has_cluster_tables()
        assert store.clusters(report.build_id) == ()
        assert store.note_clusters(report.build_id) == {}
    finally:
        store.close()

    snapshot = load_index_snapshot(database)
    assert snapshot.clusters  # fallback records synthesized
    assert all(row.label_source == "fallback" for row in snapshot.clusters)
    assert snapshot.clusters[0].name.startswith("Cluster 1 (")
    labels = dict(snapshot.note_clusters)
    assert set(labels.values()) <= {-1} | set(range(len(snapshot.clusters)))


def test_runtime_boots_against_pre_cluster_database(tmp_path: Path, monkeypatch) -> None:
    from taurine_core.models_local import LocalModelUnavailable
    from taurine_core.retrieval.runtime import RetrievalRuntime

    database, _report = _build(tmp_path)
    with sqlite3.connect(database) as conn:
        conn.execute("DROP TABLE note_clusters")
        conn.execute("DROP TABLE clusters")
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker",
        lambda: (_ for _ in ()).throw(LocalModelUnavailable("missing")),
    )

    runtime = RetrievalRuntime(database)
    try:
        labels = runtime.community_labels()
        assert labels  # remapped contiguous ids
        assert all(value >= -1 for value in labels.values())
    finally:
        runtime.close()


def test_graph_endpoint_serves_persisted_clusters(tmp_path: Path) -> None:
    """A snapshot with persisted clusters drives GraphNode.community and the
    clusters list end to end."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from taurine_core.schemas import SyncReport
    from taurine_core.server import Runtime, create_app

    database, _report = _build(tmp_path)
    snapshot = load_index_snapshot(database)
    runtime = Runtime(
        index=None,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        knowledge_runtime=SimpleNamespace(snapshot=snapshot),
    )

    with TestClient(create_app(runtime)) as client:
        payload = client.get("/graph").json()

    clusters = {cluster["id"]: cluster for cluster in payload["clusters"]}
    assert clusters[0]["name"] == snapshot.clusters[0].name
    assert clusters[0]["description"] == snapshot.clusters[0].description
    assert clusters[0]["size"] == 4
    nodes = {node["note_id"]: node for node in payload["nodes"]}
    assert {nodes[f"bill-{index}"]["community"] for index in range(4)} == {0}
    assert {nodes[f"trip-{index}"]["community"] for index in range(3)} == {1}
