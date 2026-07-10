"""Graph-view endpoints over the knowledge runtime: /graph, /graph/local, /facts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from taurine_core.indexer.communities import fallback_clusters
from taurine_core.indexer.facts import FactRecord
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.passages import PassageDraft
from taurine_core.indexer.projection import ProjectionEdge
from taurine_core.schemas import SyncReport
from taurine_core.retrieval.runtime import RetrievalRuntime
from taurine_core.schemas import NoteRecord
from taurine_core.server import Runtime, create_app

BUILD_ID = "build-1"


@dataclass(frozen=True)
class FakeSnapshot:
    database_path: Path
    projection: tuple[ProjectionEdge, ...]

    def open_knowledge_store(self) -> KnowledgeStore:
        return KnowledgeStore(self.database_path, read_only=True)


class FakeKnowledgeRuntime:
    def __init__(self, snapshot: FakeSnapshot) -> None:
        self.snapshot = snapshot


def _note(note_id: str, title: str, text: str, *, folder: str, ordinal: int) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=title,
        folder=folder,
        created=100.0 + ordinal,
        modified=200.0 + ordinal,
        content_hash=f"hash-{note_id}",
        text=text,
        source="obsidian",
    )


def _passage(note_id: str, text: str) -> PassageDraft:
    return PassageDraft(
        passage_id=f"passage-{note_id}",
        ordinal=0,
        char_start=0,
        char_end=len(text),
        token_count=5,
        text=text,
        content_hash=f"passage-hash-{note_id}",
    )


def _edge(src: str, dst: str, mechanisms: dict[str, float], **kwargs: Any) -> ProjectionEdge:
    weight = kwargs.pop("weight", 0.8)
    return ProjectionEdge(
        src_note_id=src,
        dst_note_id=dst,
        edge_type=kwargs.pop("edge_type", "semantic"),
        weight=weight,
        shared=kwargs.pop("shared", ()),
        fact_ids=kwargs.pop("fact_ids", ()),
        mechanisms=mechanisms,
    )


def _entity(store: KnowledgeStore, entity_id: str, canonical: str, *, eligible: bool) -> None:
    store.insert_entity(
        entity_id=entity_id,
        canonical=canonical,
        normalized=canonical.casefold(),
        etype="concept",
        df=1,
        traversal_eligible=eligible,
        ambiguity_key=canonical.casefold(),
        build_id=BUILD_ID,
    )


def _mention(
    store: KnowledgeStore,
    mention_id: str,
    entity_id: str,
    note_id: str,
) -> None:
    store.insert_mention(
        mention_id=mention_id,
        entity_id=entity_id,
        passage_id=f"passage-{note_id}",
        note_id=note_id,
        surface="surface",
        etype="concept",
        char_start=0,
        char_end=4,
        confidence=0.9,
        extractor="fake",
        schema_id="schema-1",
        build_id=BUILD_ID,
    )


def _build_knowledge_db(path: Path) -> str:
    """Two projection clusters, one isolated note, entities and one fact."""

    store = KnowledgeStore(path)
    try:
        texts = {
            "alpha": "Billing works with Taurine on the launch.",
            "beta": "Beta covers the tagging system in detail.",
            "gamma": "Gamma continues the billing rollout.",
            "delta": "Delta starts the travel journal.",
            "epsilon": "Epsilon plans the packing list.",
            "zeta": "Zeta books the flights.",
            "omega": "Omega stands alone without edges.",
        }
        for ordinal, (note_id, text) in enumerate(sorted(texts.items())):
            note = _note(note_id, note_id.title(), text, folder=f"Folder-{note_id}", ordinal=ordinal)
            store.write_note(
                note,
                build_id=BUILD_ID,
                extraction_status="complete",
                value_score=0.5 + ordinal,
            )
            store.write_passages(
                note_id,
                (_passage(note_id, text),),
                created=note.created,
                modified=note.modified,
                build_id=BUILD_ID,
            )

        # alpha: Billing (2 mentions) outranks Taurine (1); Grocery is not
        # traversal-eligible and must never surface.
        _entity(store, "e-billing", "Billing", eligible=True)
        _entity(store, "e-taurine", "Taurine", eligible=True)
        _entity(store, "e-grocery", "Grocery", eligible=False)
        _mention(store, "m-1", "e-billing", "alpha")
        _mention(store, "m-2", "e-billing", "alpha")
        _mention(store, "m-3", "e-taurine", "alpha")
        _mention(store, "m-4", "e-grocery", "alpha")
        _mention(store, "m-5", "e-grocery", "alpha")
        # beta: ten eligible single-mention tags — only the first 8 survive.
        for index in range(10):
            _entity(store, f"e-tag-{index:02d}", f"Tag{index:02d}", eligible=True)
            _mention(store, f"m-tag-{index:02d}", f"e-tag-{index:02d}", "beta")

        fact = FactRecord(
            fact_id="fact-graph-1",
            subject_entity_id="e-billing",
            subject="Billing",
            predicate="works_with",
            object_entity_id="e-taurine",
            object="Taurine",
            source_passage_id="passage-alpha",
            source_note_id="alpha",
            assertion_start=0,
            assertion_end=10,
            subject_start=0,
            subject_end=7,
            object_start=19,
            object_end=26,
            confidence=0.92,
            extraction_method="fake",
            schema_id="schema-1",
            observed_at=42.0,
            valid_from=None,
            valid_to=None,
            invalidated_at=None,
            superseded_by=None,
            status="active",
            build_id=BUILD_ID,
        )
        store.insert_fact(fact)
        return fact.fact_id
    finally:
        store.close()


PROJECTION = (
    _edge(
        "alpha",
        "beta",
        {"semantic": 0.5, "fact": 0.9},
        edge_type="mixed",
        shared=("Billing",),
        fact_ids=("fact-graph-1",),
        weight=0.95,
    ),
    _edge("alpha", "gamma", {"semantic": 0.8}),
    _edge("beta", "gamma", {"semantic": 0.7}),
    _edge("delta", "epsilon", {"entity": 0.6}, edge_type="entity"),
    _edge("delta", "zeta", {"entity": 0.6}, edge_type="entity"),
    _edge("epsilon", "zeta", {"entity": 0.6}, edge_type="entity"),
)


def _runtime(tmp_path: Path) -> tuple[Runtime, str]:
    fact_id = _build_knowledge_db(tmp_path / "knowledge.db")
    knowledge = FakeKnowledgeRuntime(
        FakeSnapshot(tmp_path / "knowledge.db", PROJECTION)
    )
    return (
        Runtime(
            index=SimpleNamespace(),
            sync_once=lambda: SyncReport(0, 0, 0, 0),
            open_note=lambda note_id: None,
            make_live_session=None,
            make_asr_client=lambda on_chunk, on_status, index_status: None,
            knowledge_runtime=knowledge,
        ),
        fact_id,
    )


def test_graph_payload_carries_metadata_communities_entities_and_mechanisms(
    tmp_path: Path,
) -> None:
    runtime, _fact_id = _runtime(tmp_path)

    with TestClient(create_app(runtime)) as client:
        payload = client.get("/graph").json()

        nodes = {node["note_id"]: node for node in payload["nodes"]}
        assert set(nodes) == {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "omega"}
        alpha = nodes["alpha"]
        assert alpha["title"] == "Alpha"
        assert alpha["folder"] == "Folder-alpha"
        assert alpha["created"] == 100.0
        assert alpha["modified"] == 200.0
        assert alpha["value_score"] == 0.5
        assert alpha["degree"] == 2
        assert alpha["gist"] == "Billing works with Taurine on the launch."
        # by mention count, traversal-eligible only
        assert alpha["entities"] == ["Billing", "Taurine"]
        # cap at 8, deterministic tie-break by canonical label
        assert nodes["beta"]["entities"] == [f"Tag{index:02d}" for index in range(8)]

        # label propagation over the projection: two clusters, isolated = -1
        cluster_one = {nodes[nid]["community"] for nid in ("alpha", "beta", "gamma")}
        cluster_two = {nodes[nid]["community"] for nid in ("delta", "epsilon", "zeta")}
        assert len(cluster_one) == 1 and len(cluster_two) == 1
        assert cluster_one != cluster_two
        assert -1 not in cluster_one | cluster_two
        assert nodes["omega"]["community"] == -1
        # snapshot has no persisted clusters: contiguous remapped fallback
        assert cluster_one | cluster_two == {0, 1}
        clusters = {cluster["id"]: cluster for cluster in payload["clusters"]}
        assert set(clusters) == {0, 1}
        assert clusters[0]["size"] == 3
        assert clusters[0]["name"] == "Cluster 1 (3 notes)"
        assert set(clusters[0]) == {"id", "name", "description", "size"}

        edges = {(edge["src"], edge["dst"]): edge for edge in payload["edges"]}
        assert edges[("alpha", "beta")]["mechanisms"] == {"semantic": 0.5, "fact": 0.9}
        assert edges[("alpha", "beta")]["fact_ids"] == ["fact-graph-1"]
        assert edges[("alpha", "beta")]["fact_provenance"] == "directed_facts"
        assert edges[("alpha", "gamma")]["mechanisms"] == {"semantic": 0.8}

        # deterministic across requests (community ids included)
        assert client.get("/graph").json() == payload


def test_graph_local_bfs_depth_membership_and_unknown_center(tmp_path: Path) -> None:
    runtime, _fact_id = _runtime(tmp_path)

    with TestClient(create_app(runtime)) as client:
        depth_one = client.get("/graph/local", params={"center": "alpha", "depth": 1}).json()
        assert depth_one["center"] == "alpha"
        assert depth_one["depth"] == 1
        assert [node["note_id"] for node in depth_one["nodes"]] == ["alpha", "beta", "gamma"]
        # beta–gamma is between two hop-1 nodes: both endpoints in set → included
        assert [(edge["src"], edge["dst"]) for edge in depth_one["edges"]] == [
            ("alpha", "beta"),
            ("alpha", "gamma"),
            ("beta", "gamma"),
        ]
        assert depth_one["nodes"][0]["entities"] == ["Billing", "Taurine"]

        default_depth = client.get("/graph/local", params={"center": "delta"}).json()
        assert default_depth["depth"] == 2
        assert [node["note_id"] for node in default_depth["nodes"]] == [
            "delta",
            "epsilon",
            "zeta",
        ]

        # local responses carry only clusters with a member in the result set
        assert len(depth_one["clusters"]) == 1
        assert depth_one["clusters"][0]["id"] == depth_one["nodes"][0]["community"]

        isolated = client.get("/graph/local", params={"center": "omega", "depth": 3}).json()
        assert [node["note_id"] for node in isolated["nodes"]] == ["omega"]
        assert isolated["edges"] == []
        assert isolated["clusters"] == []

        assert client.get("/graph/local", params={"center": "missing"}).status_code == 404
        # out-of-range depths clamp into 1..3
        clamped = client.get("/graph/local", params={"center": "alpha", "depth": 9}).json()
        assert clamped["depth"] == 3


def test_facts_endpoint_resolves_labels_and_drops_unknown_ids(tmp_path: Path) -> None:
    runtime, fact_id = _runtime(tmp_path)

    with TestClient(create_app(runtime)) as client:
        payload = client.get(
            "/facts", params={"ids": f"missing-1,{fact_id},missing-2,{fact_id}"}
        ).json()
        assert payload == {
            "facts": [
                {
                    "fact_id": fact_id,
                    "subject": "Billing",
                    "predicate": "works_with",
                    "object": "Taurine",
                    "source_note_id": "alpha",
                    "confidence": 0.92,
                    "observed_at": 42.0,
                }
            ]
        }

        assert client.get("/facts").json() == {"facts": []}
        assert client.get("/facts", params={"ids": " , ,"}).json() == {"facts": []}

        # the 50-id cap truncates before resolution
        over_cap = ",".join([f"missing-{index}" for index in range(50)] + [fact_id])
        assert client.get("/facts", params={"ids": over_cap}).json() == {"facts": []}


def test_retrieval_runtime_serves_snapshot_clusters_with_legacy_fallback(
    tmp_path: Path,
) -> None:
    """community_labels serves the snapshot's persisted note_clusters when
    present and remaps legacy label propagation for snapshots without them."""

    legacy = FakeSnapshot(tmp_path / "active.db", PROJECTION[:3])
    persisted = SimpleNamespace(
        projection=PROJECTION[3:],
        note_clusters={"delta": 0, "epsilon": 0, "zeta": 0, "omega": -1},
    )

    runtime = RetrievalRuntime.__new__(RetrievalRuntime)
    runtime._lock = RLock()
    runtime._snapshot = legacy  # type: ignore[assignment]

    assert runtime.community_labels() == fallback_clusters(PROJECTION[:3])[0]
    assert runtime.community_labels(legacy) == fallback_clusters(PROJECTION[:3])[0]  # type: ignore[arg-type]
    assert runtime.community_labels(persisted) == {  # type: ignore[arg-type]
        "delta": 0,
        "epsilon": 0,
        "zeta": 0,
        "omega": -1,
    }
