from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from taurine_core import config
from taurine_core.models_local import model_ref
from taurine_core.indexer.passages import PassageDraft
from taurine_core.indexer.facts import PREDICATES

_MODEL: Any | None = None


@dataclass(frozen=True)
class PassageEmbedding:
    passage_id: str
    model_id: str
    dim: int
    vector: np.ndarray
    build_id: str


class FactEmbeddingSource(Protocol):
    fact_id: str
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class FactEmbedding:
    fact_id: str
    model_id: str
    text: str
    dim: int
    vector: np.ndarray
    build_id: str


def embed_facts(
    facts: Sequence[FactEmbeddingSource],
    *,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    model_id: str,
    build_id: str,
    dim: int = config.EMBED_DIM,
) -> tuple[FactEmbedding, ...]:
    """Embed deterministic canonical triples, never generated relationship prose."""

    ordered = sorted(facts, key=lambda row: row.fact_id)
    if not ordered:
        return ()
    if len({row.fact_id for row in ordered}) != len(ordered):
        raise ValueError("duplicate fact_id in embedding batch")
    texts: list[str] = []
    for fact in ordered:
        predicate = PREDICATES.get(fact.predicate)
        if predicate is None:
            raise ValueError(f"unregistered predicate: {fact.predicate}")
        texts.append(
            f"{fact.subject} — {fact.predicate} ({predicate.description}) — {fact.object}"
        )
    matrix = np.asarray((embedder or embed_texts)(texts), dtype=np.float32)
    expected = (len(ordered), int(dim))
    if matrix.shape != expected:
        raise ValueError(f"fact embedder returned {matrix.shape}, expected {expected}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("fact embedder returned a non-finite vector")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("fact embedder returned a zero-norm vector")
    matrix = matrix / norms
    return tuple(
        FactEmbedding(
            fact_id=fact.fact_id,
            model_id=model_id,
            text=texts[index],
            dim=int(dim),
            vector=matrix[index].copy(),
            build_id=build_id,
        )
        for index, fact in enumerate(ordered)
    )


def embed_passages(
    passages: Sequence[PassageDraft],
    *,
    title: str,
    source: str,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    model_id: str,
    build_id: str,
    dim: int = config.EMBED_DIM,
) -> tuple[PassageEmbedding, ...]:
    """Embed exact passages with deterministic document metadata."""
    ordered = sorted(passages, key=lambda row: row.passage_id)
    if not ordered:
        return ()
    if len({row.passage_id for row in ordered}) != len(ordered):
        raise ValueError("duplicate passage_id in embedding batch")
    texts = [
        f"Title: {title}\nSource: {source}\nPassage:\n{row.text}"
        for row in ordered
    ]
    matrix = np.asarray((embedder or embed_texts)(texts), dtype=np.float32)
    expected = (len(ordered), int(dim))
    if matrix.shape != expected:
        raise ValueError(f"passage embedder returned {matrix.shape}, expected {expected}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("passage embedder returned a zero-norm vector")
    matrix = matrix / norms
    return tuple(
        PassageEmbedding(
            passage_id=passage.passage_id,
            model_id=model_id,
            dim=int(dim),
            vector=matrix[index].copy(),
            build_id=build_id,
        )
        for index, passage in enumerate(ordered)
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    return _embed(texts)


def embed_queries(texts: list[str], *, prompt: str | None = None) -> np.ndarray:
    query_prompt = config.EMBED_QUERY_PROMPT if prompt is None else prompt
    return _embed(texts, prompt=query_prompt)


def _embed(texts: list[str], *, prompt: str | None = None) -> np.ndarray:
    if not texts:
        return np.empty((0, config.EMBED_DIM), dtype=np.float32)
    model = _get_model()
    kwargs: dict[str, Any] = {
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if prompt is not None:
        kwargs["prompt"] = prompt
    encoded = model.encode(texts, **kwargs)
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
        ref = model_ref(config.EMBED_MODEL)
        # a bundled path must load with zero hub traffic (no update pings)
        _MODEL = SentenceTransformer(ref, local_files_only=ref != config.EMBED_MODEL)
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
