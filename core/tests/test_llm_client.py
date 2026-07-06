import json

import httpx
import pytest

from taurine_core import config
from taurine_core.llm import OpenRouterClient
from taurine_core.ports import LLMError


@pytest.mark.asyncio
async def test_complete_json_posts_openrouter_json_object_request() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"picks": [{"note_id": "n1", "why": "now"}]})}}
                ]
            },
        )

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    result = await client.complete_json("model-a", "system text", "user text")

    assert result == {"picks": [{"note_id": "n1", "why": "now"}]}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"{config.OPENROUTER_BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-key"
    body = json.loads(request.content)
    assert body["model"] == "model-a"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]


@pytest.mark.asyncio
async def test_complete_json_raises_llm_error_on_http_failure() -> None:
    client = OpenRouterClient(
        "secret-key",
        transport=httpx.MockTransport(lambda _request: httpx.Response(500, json={"error": "bad"})),
    )

    with pytest.raises(LLMError):
        await client.complete_json("model-a", "system", "user")


@pytest.mark.asyncio
async def test_complete_json_raises_llm_error_on_unparseable_content() -> None:
    client = OpenRouterClient(
        "secret-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})
        ),
    )

    with pytest.raises(LLMError):
        await client.complete_json("model-a", "system", "user")
