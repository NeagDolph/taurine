from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from taurine_core import config
from taurine_core.llm import OpenRouterClient, StreamEvent
from taurine_core.ports import LLMError


class SSEStream(httpx.AsyncByteStream):
    def __init__(self, frames: list[str]) -> None:
        self._frames = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame.encode()


def _frame(payload: dict[str, Any] | str) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {body}\n\n"


async def _collect(client: OpenRouterClient, **kwargs: Any) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for event in client.stream_chat(
        kwargs.pop("model", "model-a"),
        kwargs.pop("messages", [{"role": "user", "content": "hello"}]),
        timeout_s=kwargs.pop("timeout_s", 5.0),
        **kwargs,
    ):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_stream_chat_parses_reasoning_content_usage_done_and_reasoning_override() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            stream=SSEStream(
                [
                    ": keep-alive\n\n",
                    _frame({"choices": [{"delta": {"reasoning": "think "}}]}),
                    _frame({"choices": [{"delta": {"reasoning_details": [{"text": "ignored"}]}}]}),
                    _frame({"choices": [{"delta": {"content": "answer"}}]}),
                    _frame(
                        {
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 5,
                                "completion_tokens_details": {"reasoning_tokens": 2},
                            },
                        }
                    ),
                    _frame("[DONE]"),
                ]
            ),
        )

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    events = await _collect(
        client,
        reasoning={"enabled": True},
        tools=[{"type": "function", "function": {"name": "search_notes"}}],
    )

    assert [(event.kind, event.text, event.usage) for event in events] == [
        ("reasoning", "think ", None),
        ("reasoning", "ignored", None),
        ("content", "answer", None),
        ("usage", "", {"prompt": 3, "completion": 5, "reasoning": 2}),
        ("done", "", None),
    ]
    body = json.loads(requests[0].content)
    assert body["stream"] is True
    assert body["provider"] == {"require_parameters": True}
    assert body["reasoning"] == {"enabled": True}
    assert body["tools"] == [{"type": "function", "function": {"name": "search_notes"}}]


@pytest.mark.asyncio
async def test_stream_chat_uses_model_reasoning_when_override_is_none() -> None:
    bodies: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, stream=SSEStream([_frame("[DONE]")]))

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    await _collect(client, model="z-ai/glm-5.2", reasoning=None)

    assert bodies[0]["reasoning"] == config.LLM_REASONING_BY_MODEL["z-ai/glm-5.2"]


@pytest.mark.asyncio
async def test_stream_chat_accumulates_interleaved_tool_call_fragments_in_index_order() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            stream=SSEStream(
                [
                    _frame(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 1,
                                                "id": "call-b",
                                                "function": {
                                                    "name": "search_notes",
                                                    "arguments": '{"query":"road',
                                                },
                                            },
                                            {
                                                "index": 0,
                                                "id": "call-a",
                                                "function": {
                                                    "name": "read_note",
                                                    "arguments": '{"note":"',
                                                },
                                            },
                                        ]
                                    }
                                }
                            ]
                        }
                    ),
                    _frame(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {"index": 0, "function": {"arguments": 'N1"}'}},
                                            {"index": 1, "function": {"arguments": 'map","k":2}'}},
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ]
                        }
                    ),
                    _frame("[DONE]"),
                ]
            ),
        )

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    events = await _collect(client)

    assert [event.tool_call for event in events if event.kind == "tool_call"] == [
        {"id": "call-a", "name": "read_note", "arguments": {"note": "N1"}},
        {"id": "call-b", "name": "search_notes", "arguments": {"query": "roadmap", "k": 2}},
    ]
    assert events[-1].kind == "done"


@pytest.mark.asyncio
async def test_stream_chat_malformed_tool_json_yields_empty_arguments() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            stream=SSEStream(
                [
                    _frame(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call-a",
                                                "function": {"name": "read_note", "arguments": "{bad"},
                                            }
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ]
                        }
                    ),
                    _frame("[DONE]"),
                ]
            ),
        )

    client = OpenRouterClient("secret-key", transport=httpx.MockTransport(handler))

    events = await _collect(client)

    assert [event.tool_call for event in events if event.kind == "tool_call"] == [
        {"id": "call-a", "name": "read_note", "arguments": {}}
    ]


@pytest.mark.asyncio
async def test_stream_chat_provider_error_frame_raises_llm_error() -> None:
    client = OpenRouterClient(
        "secret-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                stream=SSEStream([_frame({"error": {"message": "provider failed"}})]),
            )
        ),
    )

    with pytest.raises(LLMError, match="provider failed"):
        await _collect(client)
