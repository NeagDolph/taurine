import json

import httpx
import pytest

from taurine_core import config
from taurine_core.llm import OpenRouterClient, _parse_json_content
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
    assert "reasoning" not in body  # unknown model: no reasoning parameter sent
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]


@pytest.mark.asyncio
async def test_complete_json_posts_strict_json_schema_when_supplied() -> None:
    requests: list[httpx.Request] = []
    schema = {
        "title": "live_summary",
        "type": "object",
        "properties": {"no_update": {"type": "boolean"}},
        "required": ["no_update"],
        "additionalProperties": False,
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"no_update": true}'}}]},
        )

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    result = await client.complete_json(
        "model-a", "system text", "user text", schema=schema
    )

    assert result == {"no_update": True}
    body = json.loads(requests[0].content)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "live_summary",
            "strict": True,
            "schema": schema,
        },
    }
    assert body["provider"] == {"require_parameters": True}


def test_parse_json_content_takes_first_object_before_gemini_trailing_garbage() -> None:
    content = '{"ok": true}\n"gemini-2.5"}\n.5-pro-…'

    assert _parse_json_content(content) == {"ok": True}


@pytest.mark.asyncio
async def test_complete_json_accepts_per_call_timeout() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": true}"}}]})

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    assert await client.complete_json("model-a", "system", "user", timeout_s=123.0) == {"ok": True}
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_complete_tools_posts_openrouter_tools_request_and_returns_message() -> None:
    requests: list[httpx.Request] = []
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_note",
                "parameters": {
                    "type": "object",
                    "properties": {"note_id": {"type": "string"}},
                    "required": ["note_id"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "consider the evidence"}]
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_note", "arguments": "{\"note_id\":\"n1\"}"},
            }
        ],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": assistant_message}]})

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    result = await client.complete_tools("model-a", "system text", messages, tools)

    assert result == assistant_message
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"{config.OPENROUTER_BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-key"
    body = json.loads(request.content)
    assert body["model"] == "model-a"
    assert body["messages"] == [{"role": "system", "content": "system text"}, *messages]
    assert body["tools"] == tools
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_reasoning_parameter_is_per_model() -> None:
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    await client.complete_json("z-ai/glm-5.2", "system", "user")
    await client.complete_json("deepseek/deepseek-v4-flash", "system", "user")
    await client.complete_json("deepseek/deepseek-chat", "system", "user")
    await client.complete_json("google/gemini-3.5-flash", "system", "user")

    assert bodies[0]["reasoning"] == {"enabled": False}
    assert bodies[1]["reasoning"] == {"enabled": False}
    assert "reasoning" not in bodies[2]  # deepseek providers reject the parameter
    assert bodies[3]["reasoning"] == {"effort": "medium"}


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
