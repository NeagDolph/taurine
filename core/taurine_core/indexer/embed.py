from __future__ import annotations

import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from taurine_core import config

_MODEL: Any | None = None


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, config.EMBED_DIM), dtype=np.float32)
    model = _get_model()
    encoded = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(encoded, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape != (len(texts), config.EMBED_DIM):
        raise ValueError(
            f"embedding model returned {matrix.shape}, expected {(len(texts), config.EMBED_DIM)}"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix.astype(np.float32)


def chunk_text(text: str) -> list[str]:
    max_tokens = max(1, int(config.CHUNK_MAX_TOKENS))
    sentences = _sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        token_count = _token_count(sentence)
        if token_count > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.extend(_split_overlong(sentence, max_tokens))
            continue
        if current and current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = token_count
        else:
            current.append(sentence)
            current_tokens += token_count

    if current:
        chunks.append(" ".join(current))
    return chunks


def _get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(config.EMBED_MODEL)
    return _MODEL


def _sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]


def _token_count(text: str) -> int:
    return len(text.split())


def _split_overlong(sentence: str, max_tokens: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[index : index + max_tokens]) for index in range(0, len(words), max_tokens)]
