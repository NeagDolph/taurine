from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from taurine_core import config
from taurine_core.models_local import model_ref
from taurine_core.indexer.passages import PassageDraft
from taurine_core.indexer.facts import PREDICATES

_MODEL: Any | None = None


class EmbeddingError(RuntimeError):
    """Sanitized failure from the selected embedding backend."""


class OpenRouterEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        *,
        model_id: str,
        dim: int,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise EmbeddingError("OPEN_ROUTER_API_KEY is required for OpenRouter embeddings")
        self._api_key = api_key
        self.model_id = str(model_id)
        self.dim = int(dim)
        self._sleep = sleep
        self._client = httpx.Client(
            transport=transport,
            timeout=float(config.EMBED_API_TIMEOUT_S),
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        matrices = [self._embed_batch(batch) for batch in self._batches(texts)]
        return np.vstack(matrices).astype(np.float32, copy=False)

    @staticmethod
    def _batches(texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        batch: list[str] = []
        batch_chars = 0
        max_texts = max(1, int(config.EMBED_API_MAX_TEXTS))
        max_chars = max(1, int(config.EMBED_API_MAX_CHARS))
        for text in texts:
            text_chars = len(text)
            if text_chars > max_chars:
                raise EmbeddingError(
                    f"one embedding input exceeds the {max_chars} characters request limit"
                )
            if batch and (
                len(batch) >= max_texts or batch_chars + text_chars > max_chars
            ):
                batches.append(batch)
                batch = []
                batch_chars = 0
            batch.append(text)
            batch_chars += text_chars
        if batch:
            batches.append(batch)
        return batches

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        attempts = max(1, int(config.EMBED_API_ATTEMPTS))
        for attempt in range(attempts):
            try:
                response = self._client.post(
                    f"{config.OPENROUTER_BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.model_id, "input": texts},
                )
            except httpx.TransportError as exc:
                if attempt + 1 >= attempts:
                    raise EmbeddingError(
                        f"OpenRouter embeddings transport failed after {attempts} attempts"
                    ) from exc
                self._backoff(attempt)
                continue
            status = int(response.status_code)
            if 200 <= status < 300:
                return self._parse(response, len(texts))
            retryable = status == 429 or 500 <= status < 600
            if retryable and attempt + 1 < attempts:
                self._backoff(attempt)
                continue
            raise EmbeddingError(f"OpenRouter embeddings returned HTTP {status}")
        raise AssertionError("embedding retry loop exhausted without returning or raising")

    def _backoff(self, attempt: int) -> None:
        self._sleep(float(config.EMBED_API_BACKOFF_S) * (2**attempt))

    @staticmethod
    def _indexed_data(payload: Any, count: int) -> list[Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise EmbeddingError("OpenRouter embeddings returned invalid data")
        rows = payload["data"]
        if len(rows) != count or any(not isinstance(row, dict) for row in rows):
            raise EmbeddingError("OpenRouter embeddings returned the wrong row count")
        try:
            by_index = {int(row["index"]): row["embedding"] for row in rows}
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("OpenRouter embeddings returned invalid row indexes") from exc
        if set(by_index) != set(range(count)) or len(by_index) != count:
            raise EmbeddingError("OpenRouter embeddings returned invalid row indexes")
        return [by_index[index] for index in range(count)]

    def _parse(self, response: httpx.Response, count: int) -> np.ndarray:
        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbeddingError("OpenRouter embeddings returned invalid JSON") from exc
        try:
            matrix = np.asarray(self._indexed_data(payload, count), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("OpenRouter embeddings returned invalid vectors") from exc
        expected = (count, self.dim)
        if matrix.shape != expected:
            raise EmbeddingError(
                f"OpenRouter embeddings returned shape {matrix.shape}, expected {expected}"
            )
        if not np.all(np.isfinite(matrix)):
            raise EmbeddingError("OpenRouter embeddings returned non-finite vectors")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise EmbeddingError("OpenRouter embeddings returned zero-norm vectors")
        return (matrix / norms).astype(np.float32, copy=False)


_OPENROUTER_CLIENT: OpenRouterEmbeddingClient | None = None


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
    dim: int | None = None,
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
    resolved_dim = config.embed_dim() if dim is None else int(dim)
    matrix = np.asarray((embedder or embed_texts)(texts), dtype=np.float32)
    expected = (len(ordered), resolved_dim)
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
            dim=resolved_dim,
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
    contexts: Mapping[str, str] | None = None,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    model_id: str,
    build_id: str,
    dim: int | None = None,
) -> tuple[PassageEmbedding, ...]:
    """Embed exact passages with deterministic document metadata.

    ``contexts`` carries optional situating prefixes (contextual retrieval);
    a non-empty context joins the frame — and therefore the content-keyed
    embedding cache key — while the stored passage text stays untouched.
    """
    ordered = sorted(passages, key=lambda row: row.passage_id)
    if not ordered:
        return ()
    if len({row.passage_id for row in ordered}) != len(ordered):
        raise ValueError("duplicate passage_id in embedding batch")
    by_passage = contexts or {}
    texts = [
        (
            f"Title: {title}\nSource: {source}\nContext: {context}\nPassage:\n{row.text}"
            if (context := by_passage.get(row.passage_id, ""))
            else f"Title: {title}\nSource: {source}\nPassage:\n{row.text}"
        )
        for row in ordered
    ]
    resolved_dim = config.embed_dim() if dim is None else int(dim)
    matrix = np.asarray((embedder or embed_texts)(texts), dtype=np.float32)
    expected = (len(ordered), resolved_dim)
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
            dim=resolved_dim,
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
        return np.empty((0, config.embed_dim()), dtype=np.float32)
    if config.EMBED_BACKEND == "openrouter":
        framed = texts if prompt is None else [prompt + text for text in texts]
        return _get_openrouter_client().embed(framed)
    if config.EMBED_BACKEND != "local":
        config.embed_dim()  # raises the shared unsupported-backend error
    return _embed_local(texts, prompt=prompt)


def _embed_local(texts: list[str], *, prompt: str | None = None) -> np.ndarray:
    model = _get_model()
    kwargs: dict[str, Any] = {
        "convert_to_numpy": True,
        "show_progress_bar": False,
        # Bound the per-forward-pass working set. A large caller batch (e.g. the
        # ~7k unique entity surfaces resolved in one build call) otherwise hands
        # sentence-transformers one huge list; an explicit batch caps peak
        # device memory regardless of caller batch size.
        "batch_size": max(1, int(config.EMBED_BATCH_SIZE)),
    }
    if prompt is not None:
        kwargs["prompt"] = prompt
    encoded = model.encode(texts, **kwargs)
    matrix = np.asarray(encoded, dtype=np.float32)
    # Release cached Metal blocks after a large job so the allocator does not
    # carry a build's worth of intermediate tensors forward.
    if len(texts) >= max(1, int(config.EMBED_BATCH_SIZE)):
        _release_device_cache()
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    expected = (len(texts), config.embed_dim())
    if matrix.shape != expected:
        raise ValueError(
            f"embedding model returned {matrix.shape}, expected {expected}"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix.astype(np.float32)


def _get_openrouter_client() -> OpenRouterEmbeddingClient:
    global _OPENROUTER_CLIENT
    if _OPENROUTER_CLIENT is None:
        import os

        _OPENROUTER_CLIENT = OpenRouterEmbeddingClient(
            os.environ.get("OPEN_ROUTER_API_KEY", ""),
            model_id=config.embed_model_id(),
            dim=config.embed_dim(),
        )
    return _OPENROUTER_CLIENT


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


def _release_device_cache() -> None:
    """Best-effort release of the MPS/CUDA caching allocator between big jobs."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
