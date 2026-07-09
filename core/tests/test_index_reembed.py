from __future__ import annotations

import numpy as np

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core import config


def _vector_lengths(index: object) -> list[int]:
    rows = index._conn.execute("SELECT length(vec) AS size FROM embeddings ORDER BY id").fetchall()
    return [int(row["size"]) for row in rows]


def test_reembed_all_updates_legacy_vectors_rebuilds_graph_and_reloads(store, monkeypatch):
    from taurine_core.indexer import rank, reembed

    alpha = make_note("alpha", "Alpha", "Alpha legacy text")
    beta = make_note("beta", "Beta", "Beta current text")
    alpha_enrich = enrichment("Alpha migration gist", topics=["reembed"])
    beta_enrich = enrichment("Beta migration gist", topics=["reembed"])
    store.upsert_note(alpha, alpha_enrich, [])
    store.upsert_note(beta, beta_enrich, fixture_units(unit_vec(2), beta_enrich))

    legacy_vec = np.ones(384, dtype=np.float32).tobytes()
    with store._conn:
        store._conn.execute(
            "INSERT INTO embeddings(note_id, kind, text, vec) VALUES (?, ?, ?, ?)",
            ("alpha", "chunk", alpha.text, legacy_vec),
        )
    store.load()

    assert reembed.needs_reembed(store) is True
    assert 384 * 4 in _vector_lengths(store)

    encoded_texts: list[str] = []

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        encoded_texts.extend(texts)
        rows = np.ones((len(texts), config.EMBED_DIM), dtype=np.float32)
        return rows

    progress: list[tuple[int, int]] = []
    monkeypatch.setattr(reembed, "embed_texts", fake_embed_texts)

    rank_calls: list[str] = []

    def fake_compute_quality_ranks(index):
        rank_calls.append("quality")
        index.set_quality_ranks({"alpha": 1.25, "beta": -0.25})
        return {"alpha": 1.25, "beta": -0.25}

    def fake_refresh_fact_embeddings(index):
        del index
        rank_calls.append("facts")

    monkeypatch.setattr(rank, "compute_quality_ranks", fake_compute_quality_ranks)
    monkeypatch.setattr(rank, "refresh_fact_embeddings", fake_refresh_fact_embeddings)

    count = reembed.reembed_all(store, progress=lambda done, total: progress.append((done, total)))

    # stale-only default: current-dim rows are untouched, so an interrupted
    # migration resumes instead of restarting
    assert count == 1
    assert encoded_texts == [alpha.text]
    assert _vector_lengths(store) == [config.EMBED_DIM * 4] * 3
    assert reembed.needs_reembed(store) is False
    assert store._matrix.shape == (3, config.EMBED_DIM)
    assert progress[-1] == (1, 1)
    assert any({edge.src, edge.dst} == {"alpha", "beta"} for edge in store.graph().edges)
    assert rank_calls == ["quality", "facts"]
    assert store.quality_ranks() == {"alpha": 1.25, "beta": -0.25}

    encoded_texts.clear()
    assert reembed.reembed_all(store, stale_only=False) == 3
    assert len(encoded_texts) == 3
