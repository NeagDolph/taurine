from __future__ import annotations

import json

import httpx
import pytest

from taurine_core import config
from taurine_core.live.rerank import CohereReranker
from taurine_core.ports import RerankError


async def test_posts_expected_payload_and_maps_result_indices() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            },
        )

    reranker = CohereReranker(
        "test-key",
        transport=httpx.MockTransport(handler),
    )

    ranked = await reranker.rerank(
        "what changed in planning?",
        [("n0", "doc zero"), ("n1", "doc one")],
    )

    assert ranked == [("n1", 0.91), ("n0", 0.42)]
    [request] = seen
    assert str(request.url) == "https://api.cohere.com/v2/rerank"
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload == {
        "model": config.RERANK_MODEL,
        "query": "what changed in planning?",
        "documents": ["doc zero", "doc one"],
    }


async def test_http_error_raises_rerank_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(500, json={"message": "unavailable"})

    reranker = CohereReranker("test-key", transport=httpx.MockTransport(handler))

    with pytest.raises(RerankError):
        await reranker.rerank("q", [("n0", "doc")])


async def test_transport_timeout_raises_rerank_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    reranker = CohereReranker("test-key", transport=httpx.MockTransport(handler))

    with pytest.raises(RerankError):
        await reranker.rerank("q", [("n0", "doc")])


async def test_malformed_response_raises_rerank_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"results": [{"index": 99, "relevance_score": 0.1}]},
        )

    reranker = CohereReranker("test-key", transport=httpx.MockTransport(handler))

    with pytest.raises(RerankError):
        await reranker.rerank("q", [("n0", "doc")])


async def test_empty_docs_short_circuits_without_request() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request should not be sent")

    reranker = CohereReranker("test-key", transport=httpx.MockTransport(handler))

    assert await reranker.rerank("q", []) == []
