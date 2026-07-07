from __future__ import annotations

import os

import numpy as np
import pytest

from taurine_core import config


def test_embed_texts_normalizes_rows_and_loads_model_once(monkeypatch):
    from taurine_core.indexer import embed

    calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            calls.append(model_name)

        def encode(
            self,
            texts: list[str],
            *,
            convert_to_numpy: bool,
            show_progress_bar: bool,
        ) -> np.ndarray:
            assert convert_to_numpy is True
            assert show_progress_bar is False
            rows = []
            for index, _ in enumerate(texts, start=1):
                row = np.zeros(config.EMBED_DIM, dtype=np.float32)
                row[0] = float(index)
                row[1] = float(index + 1)
                rows.append(row)
            return np.vstack(rows)

    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embed, "_MODEL", None)

    first = embed.embed_texts(["alpha", "beta"])
    second = embed.embed_texts(["gamma"])

    from taurine_core.models_local import model_ref

    assert calls == [model_ref(config.EMBED_MODEL)]
    assert first.shape == (2, config.EMBED_DIM)
    assert second.shape == (1, config.EMBED_DIM)
    assert first.dtype == np.float32
    assert np.linalg.norm(first, axis=1) == pytest.approx([1.0, 1.0])
    assert np.linalg.norm(second, axis=1) == pytest.approx([1.0])


def test_embed_texts_empty_input_returns_empty_matrix(monkeypatch):
    from taurine_core.indexer import embed

    monkeypatch.setattr(embed, "_MODEL", object())

    result = embed.embed_texts([])

    assert result.shape == (0, config.EMBED_DIM)
    assert result.dtype == np.float32


def test_chunk_text_packs_on_sentence_boundaries(monkeypatch):
    from taurine_core.indexer import embed

    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 6)

    chunks = embed.chunk_text(
        "Alpha beta gamma. Delta epsilon. Zeta eta theta iota. Kappa."
    )

    assert chunks == [
        "Alpha beta gamma. Delta epsilon.",
        "Zeta eta theta iota. Kappa.",
    ]


def test_chunk_text_splits_overlong_sentence(monkeypatch):
    from taurine_core.indexer import embed

    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 3)

    assert embed.chunk_text("one two three four five") == [
        "one two three",
        "four five",
    ]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW is unset")
def test_real_embedding_model_smoke():
    from taurine_core.indexer.embed import embed_texts

    result = embed_texts(["roadmap planning"])

    assert result.shape == (1, config.EMBED_DIM)
    assert np.linalg.norm(result[0]) == pytest.approx(1.0, abs=1e-5)
