from __future__ import annotations

import numpy as np

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core import config


def _vector_lengths(index: object) -> list[int]:
    rows = index._conn.execute("SELECT length(vec) AS size FROM embeddings ORDER BY id").fetchall()
    return [int(row["size"]) for row in rows]


def test_reembed_all_updates_legacy_vectors_rebuilds_graph_and_reloads(store, monkeypatch):
    from taurine_core.indexer import reembed

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
            ("alpha", "gist", alpha_enrich.gist, legacy_vec),
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

    count = reembed.reembed_all(store, progress=lambda done, total: progress.append((done, total)))

    assert count == 4
    assert encoded_texts == [
        beta_enrich.gist,
        beta_enrich.gist,
        beta_enrich.utterances[0],
        alpha_enrich.gist,
    ]
    assert _vector_lengths(store) == [config.EMBED_DIM * 4] * 4
    assert reembed.needs_reembed(store) is False
    assert store._matrix.shape == (4, config.EMBED_DIM)
    assert progress[-1] == (4, 4)
    assert any({edge.src, edge.dst} == {"alpha", "beta"} for edge in store.graph().edges)
