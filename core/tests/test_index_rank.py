from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tests.conftest import entity, make_note
from taurine_core import config
from taurine_core.indexer import graph
from taurine_core.schemas import Enrichment, GraphEdge, NoteRecord


def _basis(index: int) -> np.ndarray:
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    vector[index % config.EMBED_DIM] = 1.0
    return vector


def _install(
    index: object,
    note_id: str,
    *,
    source: str = "apple_notes",
    modified: float = 1_000.0,
    vector: np.ndarray | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
) -> NoteRecord:
    typed_entities = [entity(topic, "concept") for topic in (topics or [])]
    typed_entities.extend(entity(item) for item in (entities or []))
    enrich = Enrichment(
        gist=f"{note_id} gist",
        utterances=[f"Discuss {note_id}."],
        aliases=[],
        entities=typed_entities,
        relations=[],
        extraction_failed=False,
    )
    note = make_note(note_id, note_id.title(), f"{note_id} body", modified=modified)
    note = note.model_copy(update={"source": source})
    index.upsert_note(  # type: ignore[attr-defined]
        note,
        enrich,
        [("chunk", note.text, vector if vector is not None else _basis(abs(hash(note_id))))],
    )
    return note


def _hash_embedding(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=config.EMBED_DIM).astype(np.float32)
    values /= np.linalg.norm(values)
    return values


def test_matrix_build_boosts_weights_and_column_normalizes(store) -> None:
    from taurine_core.indexer.rank import NoteGraphMatrix

    for note_id in ("alpha", "beta", "gamma"):
        _install(store, note_id)
    store.replace_edges(
        [
            GraphEdge(
                src="alpha",
                dst="beta",
                type="topic",
                weight=0.4,
                fact="Alpha and Beta share an adjudicated topic.",
            ),
            GraphEdge(src="beta", dst="gamma", type="semantic", weight=0.5),
        ]
    )

    graph_matrix = NoteGraphMatrix.build(store)

    assert isinstance(graph_matrix.matrix, csr_matrix)
    alpha = graph_matrix.row_of["alpha"]
    beta = graph_matrix.row_of["beta"]
    gamma = graph_matrix.row_of["gamma"]
    matrix = graph_matrix.matrix.toarray()
    boosted = min(1.0, 0.4 * config.FACT_EDGE_BOOST * config.SHARED_EDGE_BOOST)

    assert matrix[beta, alpha] == pytest.approx(1.0)
    assert matrix[alpha, beta] == pytest.approx(boosted / (boosted + 0.5))
    assert matrix[gamma, beta] == pytest.approx(0.5 / (boosted + 0.5))
    assert matrix[beta, gamma] == pytest.approx(1.0)
    assert np.allclose(np.asarray(graph_matrix.matrix.sum(axis=0)).ravel(), 1.0)


def test_matrix_build_for_quality_drops_same_source_duplicate_edges_only(store) -> None:
    from taurine_core.indexer.rank import NoteGraphMatrix

    duplicate_vec = _basis(1)
    _install(store, "same-a", source="obsidian", vector=duplicate_vec)
    _install(store, "same-b", source="obsidian", vector=duplicate_vec)
    _install(store, "cross-a", source="obsidian", vector=duplicate_vec)
    _install(store, "cross-b", source="apple_notes", vector=duplicate_vec)
    store.replace_edges(
        [
            GraphEdge(src="same-a", dst="same-b", type="semantic", weight=0.8),
            GraphEdge(src="cross-a", dst="cross-b", type="semantic", weight=0.8),
        ]
    )

    traversal = NoteGraphMatrix.build(store)
    quality = NoteGraphMatrix.build(store, for_quality=True)

    assert traversal.matrix[traversal.row_of["same-a"], traversal.row_of["same-b"]] > 0
    assert quality.matrix[quality.row_of["same-a"], quality.row_of["same-b"]] == 0
    assert quality.matrix[quality.row_of["cross-a"], quality.row_of["cross-b"]] > 0


def test_ppr_normalizes_seeds_ignores_unknowns_and_supports_warm_start(store) -> None:
    from taurine_core.indexer.rank import NoteGraphMatrix

    for note_id in ("alpha", "beta", "gamma"):
        _install(store, note_id)
    store.replace_edges(
        [
            GraphEdge(src="alpha", dst="beta", type="semantic", weight=1.0),
            GraphEdge(src="beta", dst="gamma", type="semantic", weight=1.0),
        ]
    )
    graph_matrix = NoteGraphMatrix.build(store)

    mass, raw = graph_matrix.ppr({"alpha": 2.0, "missing": 99.0}, damping=0.5, tol=1e-12)
    warm_mass, warm_raw = graph_matrix.ppr(
        {"alpha": 1.0},
        damping=0.5,
        tol=1e-12,
        warm_start=np.ones(len(graph_matrix.note_ids), dtype=np.float32),
    )

    assert mass["alpha"] == pytest.approx(7.0 / 12.0)
    assert mass["beta"] == pytest.approx(1.0 / 3.0)
    assert mass["gamma"] == pytest.approx(1.0 / 12.0)
    assert mass["alpha"] > mass["beta"] > mass["gamma"]
    assert np.allclose(raw, warm_raw, atol=1e-8)
    assert warm_mass == pytest.approx(mass)


def test_ppr_returns_zero_vector_for_empty_or_unknown_seed_sets(store) -> None:
    from taurine_core.indexer.rank import NoteGraphMatrix

    empty = NoteGraphMatrix.build(store)
    empty_mass, empty_raw = empty.ppr({"missing": 1.0})
    assert empty_mass == {}
    assert empty_raw.shape == (0,)

    _install(store, "alpha")
    graph_matrix = NoteGraphMatrix.build(store)
    mass, raw = graph_matrix.ppr({"missing": 1.0})
    assert mass == {}
    assert raw.shape == (1,)
    assert np.count_nonzero(raw) == 0


def test_compute_quality_ranks_uses_source_recency_teleport_and_persists(store) -> None:
    from taurine_core.indexer.rank import compute_quality_ranks

    now = 10_000_000.0
    _install(store, "fresh-obsidian", source="obsidian", modified=now)
    _install(store, "fresh-imessage", source="imessage", modified=now)
    _install(
        store,
        "old-obsidian",
        source="obsidian",
        modified=now - 365.0 * 24.0 * 60.0 * 60.0,
    )

    ranks = compute_quality_ranks(store, now=now)

    assert ranks["fresh-obsidian"] > ranks["fresh-imessage"] > ranks["old-obsidian"]
    assert all(-config.QUALITY_CLIP <= value <= config.QUALITY_CLIP for value in ranks.values())
    assert ranks["fresh-imessage"] == pytest.approx(0.0, abs=0.05)
    assert store.quality_ranks() == pytest.approx(ranks)


def test_compute_quality_ranks_keeps_isolated_notes_finite(store) -> None:
    from taurine_core.indexer.rank import compute_quality_ranks

    now = 20_000_000.0
    _install(store, "alpha", source="obsidian", modified=now)
    _install(store, "beta", source="obsidian", modified=now)
    _install(
        store,
        "isolated",
        source="codex_mem",
        modified=now - 365.0 * 24.0 * 60.0 * 60.0,
    )
    store.replace_edges([GraphEdge(src="alpha", dst="beta", type="semantic", weight=1.0)])

    ranks = compute_quality_ranks(store, now=now)

    assert set(ranks) == {"alpha", "beta", "isolated"}
    assert math.isfinite(ranks["isolated"])
    assert ranks["alpha"] > ranks["isolated"]
    assert ranks["beta"] > ranks["isolated"]


def test_refresh_fact_embeddings_is_content_addressed_and_prunes(store, monkeypatch) -> None:
    from taurine_core.indexer.rank import refresh_fact_embeddings
    from taurine_core.indexer import store as store_module

    for note_id in ("alpha", "beta"):
        _install(store, note_id)
    first_fact = "Alpha and Beta share the launch timeline."
    second_fact = "Alpha and Beta now share the revised launch timeline."
    embed_calls: list[tuple[str, ...]] = []

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        embed_calls.append(tuple(texts))
        return np.vstack([_hash_embedding(text) for text in texts])

    monkeypatch.setattr(store_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        store_module,
        "embed_queries",
        lambda texts: np.vstack([_hash_embedding(first_fact) for _ in texts]),
    )

    store.replace_edges(
        [GraphEdge(src="alpha", dst="beta", type="semantic", weight=0.8, fact=first_fact)]
    )
    refresh_fact_embeddings(store)
    refresh_fact_embeddings(store)

    assert embed_calls == [(first_fact,)]
    assert store.fact_dense_search(first_fact, 1)[0][:2] == ("alpha", "beta")

    store.replace_edges(
        [GraphEdge(src="alpha", dst="beta", type="semantic", weight=0.8, fact=second_fact)]
    )
    refresh_fact_embeddings(store)
    assert embed_calls[-1] == (second_fact,)

    store.replace_edges([])
    refresh_fact_embeddings(store)
    assert store.fact_embedding_hashes() == {}
    assert store.fact_dense_search(first_fact, 1) == []


def test_explicit_rank_and_fact_refresh_after_graph_rebuild(store, monkeypatch) -> None:
    from taurine_core.indexer import store as store_module
    from taurine_core.indexer.rank import compute_quality_ranks, refresh_fact_embeddings

    now = 30_000_000.0
    _install(
        store,
        "alpha",
        source="obsidian",
        modified=now,
        vector=_basis(1),
        entities=["Launch Project"],
    )
    _install(
        store,
        "beta",
        source="obsidian",
        modified=now,
        vector=_basis(2),
        entities=["Launch Project"],
    )
    _install(
        store,
        "old-memory",
        source="codex_mem",
        modified=now - 365.0 * 24.0 * 60.0 * 60.0,
        vector=_basis(3),
        entities=["Unrelated"],
    )
    fact = "Alpha and Beta are both part of the Launch Project."
    store.set_edge_judgment("alpha", "beta", graph.pair_sig(store, "alpha", "beta"), True, fact)

    monkeypatch.setattr(
        graph,
        "embed_texts",
        lambda texts: np.vstack([_basis(i + 100) for i, _ in enumerate(texts)]),
    )
    monkeypatch.setattr(
        store_module,
        "embed_texts",
        lambda texts: np.vstack([_hash_embedding(text) for text in texts]),
    )
    monkeypatch.setattr(
        store_module,
        "embed_queries",
        lambda texts: np.vstack([_hash_embedding(fact) for _ in texts]),
    )

    store.rebuild_graph()
    compute_quality_ranks(store, now=now)
    refresh_fact_embeddings(store)

    ranks = store.quality_ranks()
    assert any(abs(value) > 0 for value in ranks.values())
    assert store.fact_embedding_hashes()
    assert store.fact_dense_search(fact, 1)[0][:2] == ("alpha", "beta")


def test_every_connector_has_explicit_source_priors() -> None:
    # SOURCE_RANK/RECENCY_HALF_LIFE_DAYS keys must match real connector ids;
    # a typo silently hands the connector the default prior (2026-07-08 bug:
    # "claude_memories" vs actual "claude_mem")
    from taurine_core.connectors.catalog import CONNECTORS
    from taurine_core.indexer import rank

    for connector in CONNECTORS:
        assert connector.connector_id in config.SOURCE_RANK
        assert connector.connector_id in config.RECENCY_HALF_LIFE_DAYS
    assert rank.source_rank("claude_mem") == config.SOURCE_RANK["claude_mem"] == 0.70
