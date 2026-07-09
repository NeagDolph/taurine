from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import entity, make_note
from taurine_core import config
from taurine_core.indexer import graph
from taurine_core.schemas import Enrichment, GraphEdge, RelationTriple


def _basis(index: int) -> np.ndarray:
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    vector[index % config.EMBED_DIM] = 1.0
    return vector


def _hash_embedding(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=config.EMBED_DIM).astype(np.float32)
    values /= np.linalg.norm(values)
    return values.astype(np.float32)


def _install(
    store,
    note_id: str,
    *,
    entities: list = None,
    relations: list[RelationTriple] = None,
    vector: np.ndarray | None = None,
) -> None:
    enrich = Enrichment(
        gist=f"{note_id} gist",
        utterances=[],
        aliases=[],
        entities=entities or [],
        relations=relations or [],
        extraction_failed=False,
    )
    note = make_note(note_id, note_id.title(), f"{note_id} body")
    chunk_vector = vector if vector is not None else _basis(abs(hash(note_id)))
    store.upsert_note(note, enrich, [("chunk", note.text, chunk_vector)])


@pytest.fixture
def term_embedding_stub(monkeypatch: pytest.MonkeyPatch):
    base_vectors = {
        "billing flow": _basis(101),
        "Shop LC": _basis(102),
        "ShopLC": _basis(102),
    }
    calls: list[tuple[str, ...]] = []

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        calls.append(tuple(texts))
        return np.vstack([base_vectors.get(text, _hash_embedding(text)) for text in texts])

    monkeypatch.setattr(graph, "embed_texts", fake_embed_texts, raising=False)
    return calls


def _active_entity_rows(store) -> dict[tuple[str, str], tuple[int, int]]:
    return {
        (canonical, etype): (entity_id, df)
        for entity_id, canonical, etype, df in store.entity_rows()
        if df > 0
    }


def _edge_between(store, left: str, right: str) -> GraphEdge | None:
    wanted = frozenset((left, right))
    for edge in store.graph().edges:
        if frozenset((edge.src, edge.dst)) == wanted:
            return edge
    return None


def test_rebuild_canonicalizes_entities_into_store_with_df(store, term_embedding_stub) -> None:
    _install(
        store,
        "alpha",
        entities=[
            entity("Billing-Flow", "concept", conf=0.9),
            entity("Shop LC", "project", conf=0.8),
        ],
        vector=_basis(1),
    )
    _install(
        store,
        "beta",
        entities=[
            entity("billing_flow", "concept", conf=0.7),
            entity("ShopLC", "project", conf=0.6),
        ],
        vector=_basis(2),
    )

    store.rebuild_graph()

    active = _active_entity_rows(store)
    assert active[("billing flow", "concept")][1] == 2
    assert active[("ShopLC", "project")][1] == 2
    assert {item.surface for item in store.enrichment_for("alpha").entities} == {
        "billing flow",
        "ShopLC",
    }
    assert len(term_embedding_stub) == 1


def test_relation_triples_create_note_candidate_with_catchall_downweight(
    store, term_embedding_stub
) -> None:
    _install(store, "beta", entities=[entity("Acme", "organization")], vector=_basis(2))
    _install(
        store,
        "alpha",
        entities=[entity("Jane Doe", "person")],
        relations=[RelationTriple(head="Jane Doe", rel_type="related_to", tail="Acme", conf=0.8)],
        vector=_basis(1),
    )

    store.rebuild_graph()

    edge = _edge_between(store, "alpha", "beta")
    expected = config.REL_EDGE_W * 0.8 * config.REL_CATCHALL_FACTOR
    assert edge is not None
    assert edge.type == "entity"
    assert edge.weight == pytest.approx(expected)
    assert edge.shared == ["Acme", "Jane Doe"]


def test_hetero_matrix_contains_note_entity_relation_edges_and_normalizes(store) -> None:
    from taurine_core.indexer.rank import HeteroGraphMatrix

    _install(store, "beta", entities=[entity("Jane Doe", "person", conf=0.5)], vector=_basis(2))
    _install(
        store,
        "alpha",
        entities=[entity("Acme", "organization", conf=0.8)],
        relations=[RelationTriple(head="Acme", rel_type="works_with", tail="Jane Doe", conf=0.6)],
        vector=_basis(1),
    )
    store.replace_edges([GraphEdge(src="alpha", dst="beta", type="semantic", weight=0.4)])
    entities = _active_entity_rows(store)
    acme_id = entities[("Acme", "organization")][0]
    jane_id = entities[("Jane Doe", "person")][0]

    matrix = HeteroGraphMatrix.build(store)
    dense = matrix.matrix.toarray()
    alpha = matrix.row_of["note:alpha"]
    beta = matrix.row_of["note:beta"]
    acme = matrix.row_of[f"ent:{acme_id}"]
    jane = matrix.row_of[f"ent:{jane_id}"]
    alpha_contains = graph._term_weight(1) * 0.8
    beta_contains = graph._term_weight(1) * 0.5
    rel_weight = config.REL_EDGE_W * 0.6

    assert matrix.note_ids == ["alpha", "beta"]
    assert matrix.entity_ids == [jane_id, acme_id]
    assert dense.shape == (4, 4)
    assert dense[beta, alpha] == pytest.approx(0.4 / (0.4 + alpha_contains))
    assert dense[acme, alpha] == pytest.approx(alpha_contains / (0.4 + alpha_contains))
    assert dense[alpha, acme] == pytest.approx(alpha_contains / (alpha_contains + rel_weight))
    assert dense[jane, acme] == pytest.approx(rel_weight / (alpha_contains + rel_weight))
    assert dense[beta, jane] == pytest.approx(beta_contains / (beta_contains + rel_weight))
    assert np.allclose(np.asarray(matrix.matrix.sum(axis=0)).ravel(), 1.0)


def test_hetero_ppr_accepts_entity_seeds_and_renormalizes_note_mass(store) -> None:
    from taurine_core.indexer.rank import HeteroGraphMatrix

    _install(store, "alpha", entities=[entity("Acme", "organization")], vector=_basis(1))
    _install(store, "beta", entities=[], vector=_basis(2))
    acme_id = _active_entity_rows(store)[("Acme", "organization")][0]
    matrix = HeteroGraphMatrix.build(store)

    masses, raw = matrix.ppr({}, entity_seeds={acme_id: 1.0}, damping=0.5, tol=1e-12)

    assert raw.shape == (len(matrix.note_ids) + len(matrix.entity_ids),)
    assert masses["alpha"] == pytest.approx(1.0)
    assert "beta" not in masses
    assert sum(masses.values()) == pytest.approx(1.0)


def test_index_graph_projection_contains_only_note_nodes(store, term_embedding_stub) -> None:
    _install(store, "alpha", entities=[entity("Acme", "organization")], vector=_basis(1))
    _install(store, "beta", entities=[entity("Acme", "organization")], vector=_basis(2))

    store.rebuild_graph()

    payload = store.graph()
    assert {node.note_id for node in payload.nodes} == {"alpha", "beta"}
    assert all(not node.note_id.startswith("ent:") for node in payload.nodes)
    assert all(edge.src in {"alpha", "beta"} and edge.dst in {"alpha", "beta"} for edge in payload.edges)
