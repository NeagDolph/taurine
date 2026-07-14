from __future__ import annotations

import os

import httpx
import numpy as np
import pytest

from taurine_core import config


def test_embedding_backend_helpers_resolve_model_and_dimension(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BACKEND", "local")
    assert config.embed_model_id() == config.EMBED_MODEL
    assert config.embed_dim() == 1024

    monkeypatch.setattr(config, "EMBED_BACKEND", "openrouter")
    assert config.embed_model_id() == config.EMBED_API_MODEL
    assert config.embed_dim() == 4096

    monkeypatch.setattr(config, "EMBED_BACKEND", "unknown")
    with pytest.raises(ValueError, match="unsupported embedding backend"):
        config.embed_model_id()
    with pytest.raises(ValueError, match="unsupported embedding backend"):
        config.embed_dim()


def test_embed_texts_and_queries_normalize_rows_and_load_model_once(monkeypatch):
    from taurine_core.indexer import embed

    monkeypatch.setattr(config, "EMBED_BACKEND", "local")
    calls: list[str] = []
    encode_calls: list[tuple[tuple[str, ...], str | None]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, local_files_only: bool = False) -> None:
            del local_files_only
            calls.append(model_name)

        def encode(
            self,
            texts: list[str],
            *,
            convert_to_numpy: bool,
            show_progress_bar: bool,
            prompt: str | None = None,
            batch_size: int | None = None,
        ) -> np.ndarray:
            assert convert_to_numpy is True
            assert show_progress_bar is False
            encode_calls.append((tuple(texts), prompt))
            rows = []
            for index, _ in enumerate(texts, start=1):
                row = np.zeros(config.embed_dim(), dtype=np.float32)
                row[0] = float(index)
                row[1] = float(index + 1)
                rows.append(row)
            return np.vstack(rows)

    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embed, "_MODEL", None)

    first = embed.embed_texts(["alpha", "beta"])
    query = embed.embed_queries(["gamma"])
    second = embed.embed_texts(["delta"])

    from taurine_core.models_local import model_ref

    assert calls == [model_ref(config.EMBED_MODEL)]
    assert encode_calls == [
        (("alpha", "beta"), None),
        (("gamma",), config.EMBED_QUERY_PROMPT),
        (("delta",), None),
    ]
    assert first.shape == (2, config.embed_dim())
    assert query.shape == (1, config.embed_dim())
    assert second.shape == (1, config.embed_dim())
    assert first.dtype == np.float32
    assert query.dtype == np.float32
    assert np.linalg.norm(first, axis=1) == pytest.approx([1.0, 1.0])
    assert np.linalg.norm(query, axis=1) == pytest.approx([1.0])
    assert np.linalg.norm(second, axis=1) == pytest.approx([1.0])


def test_embed_texts_empty_input_returns_empty_matrix(monkeypatch):
    from taurine_core.indexer import embed

    monkeypatch.setattr(config, "EMBED_BACKEND", "local")
    monkeypatch.setattr(embed, "_MODEL", object())

    result = embed.embed_texts([])
    query_result = embed.embed_queries([])

    assert result.shape == (0, config.embed_dim())
    assert query_result.shape == (0, config.embed_dim())
    assert result.dtype == np.float32
    assert query_result.dtype == np.float32


def test_openrouter_client_sends_bare_documents_and_normalizes_indexed_rows():
    from taurine_core.indexer.embed import OpenRouterEmbeddingClient

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "emb-1",
                "object": "list",
                "model": "qwen/qwen3-embedding-8b",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 3.0, 4.0]},
                    {"object": "embedding", "index": 0, "embedding": [2.0, 0.0, 0.0]},
                ],
            },
        )

    client = OpenRouterEmbeddingClient(
        "secret-key",
        model_id="qwen/qwen3-embedding-8b",
        dim=3,
        transport=httpx.MockTransport(handler),
    )
    result = client.embed(["document one", "document two"])

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer secret-key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.read()
    assert __import__("json").loads(request.content) == {
        "model": "qwen/qwen3-embedding-8b",
        "input": ["document one", "document two"],
    }
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])


def test_openrouter_dispatch_prefixes_queries_but_not_documents(monkeypatch):
    from taurine_core.indexer import embed

    calls: list[list[str]] = []

    class FakeClient:
        def embed(self, texts: list[str]) -> np.ndarray:
            calls.append(list(texts))
            rows = np.ones((len(texts), config.embed_dim()), dtype=np.float32)
            return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    monkeypatch.setattr(config, "EMBED_BACKEND", "openrouter")
    monkeypatch.setattr(embed, "_get_openrouter_client", lambda: FakeClient())

    documents = embed.embed_texts(["bare document"])
    queries = embed.embed_queries(["find this"])

    assert calls == [
        ["bare document"],
        [config.EMBED_QUERY_PROMPT + "find this"],
    ]
    assert documents.shape == (1, 4096)
    assert queries.shape == (1, 4096)


def _embedding_response(request: httpx.Request, count: int, *, dim: int = 2) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "id": "emb-test",
            "object": "list",
            "model": "test-model",
            "usage": {"prompt_tokens": count, "total_tokens": count},
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [float(index + 1)] + [1.0] * (dim - 1),
                }
                for index in range(count)
            ],
        },
    )


def test_openrouter_client_batches_by_text_count_and_character_budget():
    from taurine_core.indexer.embed import OpenRouterEmbeddingClient

    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = __import__("json").loads(request.content)["input"]
        batches.append(inputs)
        assert len(inputs) <= 64
        assert sum(len(text) for text in inputs) <= 100_000
        return _embedding_response(request, len(inputs))

    client = OpenRouterEmbeddingClient(
        "secret",
        model_id="test-model",
        dim=2,
        transport=httpx.MockTransport(handler),
    )
    texts = ["x"] * 65 + ["a" * 60_000, "b" * 40_000, "c"]

    result = client.embed(texts)

    assert [len(batch) for batch in batches] == [64, 2, 2]
    assert result.shape == (68, 2)
    assert np.linalg.norm(result, axis=1) == pytest.approx(np.ones(68))


def test_openrouter_client_rejects_one_text_over_character_budget_without_request():
    from taurine_core.indexer.embed import EmbeddingError, OpenRouterEmbeddingClient

    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _embedding_response(request, 1)

    client = OpenRouterEmbeddingClient(
        "secret",
        model_id="test-model",
        dim=2,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(EmbeddingError, match="100000 characters"):
        client.embed(["x" * 100_001])
    assert called is False


@pytest.mark.parametrize("status", [429, 500, 503])
def test_openrouter_client_retries_retryable_http_statuses(status: int):
    from taurine_core.indexer.embed import OpenRouterEmbeddingClient

    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(status, request=request, json={"error": {"message": "busy"}})
        return _embedding_response(request, 1)

    client = OpenRouterEmbeddingClient(
        "secret",
        model_id="test-model",
        dim=2,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    assert client.embed(["text"]).shape == (1, 2)
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
def test_openrouter_client_retries_timeout_and_transport_errors(error_type):
    from taurine_core.indexer.embed import OpenRouterEmbeddingClient

    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise error_type("temporary embedding transport failure", request=request)
        return _embedding_response(request, 1)

    client = OpenRouterEmbeddingClient(
        "secret",
        model_id="test-model",
        dim=2,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    assert client.embed(["text"]).shape == (1, 2)
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_openrouter_client_does_not_retry_non_retryable_http_status():
    from taurine_core.indexer.embed import EmbeddingError, OpenRouterEmbeddingClient

    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request, json={"error": {"message": "bad input"}})

    client = OpenRouterEmbeddingClient(
        "secret",
        model_id="test-model",
        dim=2,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    with pytest.raises(EmbeddingError, match="HTTP 400"):
        client.embed(["text"])
    assert attempts == 1
    assert sleeps == []


def test_openrouter_client_exhausts_three_attempts_with_sanitized_error():
    from taurine_core.indexer.embed import EmbeddingError, OpenRouterEmbeddingClient

    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, request=request, text="secret provider detail")

    client = OpenRouterEmbeddingClient(
        "secret",
        model_id="test-model",
        dim=2,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    with pytest.raises(EmbeddingError, match="HTTP 500") as raised:
        client.embed(["text"])
    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": []}, "wrong row count"),
        (
            {"data": [{"object": "embedding", "index": 1, "embedding": [1.0, 0.0]}]},
            "invalid row indexes",
        ),
        (
            {"data": [{"object": "embedding", "index": 0, "embedding": [1.0]}]},
            "returned shape",
        ),
        (
            {"data": [{"object": "embedding", "index": 0, "embedding": [float("nan"), 1.0]}]},
            "non-finite",
        ),
        (
            {"data": [{"object": "embedding", "index": 0, "embedding": [0.0, 0.0]}]},
            "zero-norm",
        ),
    ],
)
def test_openrouter_client_rejects_invalid_success_payloads(payload, message):
    from taurine_core.indexer.embed import EmbeddingError, OpenRouterEmbeddingClient

    client = OpenRouterEmbeddingClient(
        "secret",
        model_id="test-model",
        dim=2,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
                content=__import__("json").dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
        ),
    )

    with pytest.raises(EmbeddingError, match=message):
        client.embed(["text"])


def test_openrouter_client_requires_api_key():
    from taurine_core.indexer.embed import EmbeddingError, OpenRouterEmbeddingClient

    with pytest.raises(EmbeddingError, match="OPEN_ROUTER_API_KEY"):
        OpenRouterEmbeddingClient("", model_id="test-model", dim=2)


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
def test_real_local_embedding_model_smoke(monkeypatch):
    from taurine_core.indexer.embed import embed_queries, embed_texts

    monkeypatch.setattr(config, "EMBED_BACKEND", "local")
    result = embed_texts(["roadmap planning"])
    query = embed_queries(["roadmap planning"])

    assert result.shape == (1, config.embed_dim())
    assert query.shape == (1, config.embed_dim())
    assert np.linalg.norm(result[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(query[0]) == pytest.approx(1.0, abs=1e-5)
