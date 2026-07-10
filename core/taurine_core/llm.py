from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from . import config
from .ports import LLMError


@dataclass
class StreamEvent:
    """One event from a streamed chat completion (spec chat §1.4).

    kind: "reasoning" (thinking delta) | "content" (answer delta) |
    "tool_call" (complete call, arguments parsed) | "usage" | "done".
    """

    kind: str
    text: str = ""                                # reasoning/content deltas
    tool_call: dict[str, Any] | None = None       # {"id", "name", "arguments": dict}
    usage: dict[str, Any] | None = None            # {"prompt", "completion", "reasoning"}


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._transport = transport

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        timeout_s: float | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            # only route to providers that honor response_format
            "provider": {"require_parameters": True},
            **_reasoning_for(model),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with httpx.AsyncClient(
                timeout=timeout_s if timeout_s is not None else config.LLM_TIMEOUT_S,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{config.OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = _parse_json_content(content) if isinstance(content, str) else content
            if not isinstance(parsed, dict):
                raise TypeError("OpenRouter content was not a JSON object")
            return parsed
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(str(exc)) from exc

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
        timeout_s: float,
    ) -> AsyncIterator[StreamEvent]:
        """SSE-streamed chat completion (spec chat §1.4).

        ``reasoning`` overrides LLM_REASONING_BY_MODEL for this call (chat
        enables GLM-5.2 thinking; enrichment keeps it disabled). Yields
        StreamEvents: reasoning/content deltas as they arrive, tool_call
        events only once a call's argument fragments are complete, one usage
        event when the provider reports it, then done. Cancellation = the
        caller cancelling the consuming task; the stream context closes
        cleanly. Raises LLMError on connect/HTTP/mid-stream provider errors.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "provider": {"require_parameters": True},
            **({"reasoning": dict(reasoning)} if reasoning is not None else _reasoning_for(model)),
        }
        if tools is not None:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self._api_key}"}
        tool_calls: dict[int, dict[str, Any]] = {}
        emitted_tools = False

        async def flush_tool_calls() -> AsyncIterator[StreamEvent]:
            nonlocal emitted_tools
            if emitted_tools:
                return
            emitted_tools = True
            for index in sorted(tool_calls):
                call = tool_calls[index]
                args_text = "".join(call.get("arguments_parts", []))
                try:
                    parsed_args = json.loads(args_text) if args_text else {}
                    if not isinstance(parsed_args, dict):
                        parsed_args = {}
                except json.JSONDecodeError:
                    parsed_args = {}
                yield StreamEvent(
                    kind="tool_call",
                    tool_call={
                        "id": call.get("id", ""),
                        "name": call.get("name", ""),
                        "arguments": parsed_args,
                    },
                )

        try:
            async with httpx.AsyncClient(
                timeout=timeout_s,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{config.OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise LLMError(str(exc)) from exc
                        if isinstance(chunk, dict) and "error" in chunk:
                            error = chunk.get("error")
                            if isinstance(error, dict):
                                detail = error.get("message") or error.get("detail") or error
                            else:
                                detail = error
                            raise LLMError(str(detail))
                        if not isinstance(chunk, dict):
                            continue

                        usage = chunk.get("usage")
                        choices = chunk.get("choices") or []
                        for choice in choices:
                            if not isinstance(choice, dict):
                                continue
                            delta = choice.get("delta") or {}
                            if not isinstance(delta, dict):
                                delta = {}
                            reasoning_text = delta.get("reasoning")
                            if isinstance(reasoning_text, str) and reasoning_text:
                                yield StreamEvent(kind="reasoning", text=reasoning_text)
                            elif isinstance(delta.get("reasoning_details"), list):
                                for detail in delta["reasoning_details"]:
                                    if isinstance(detail, dict) and isinstance(detail.get("text"), str) and detail["text"]:
                                        yield StreamEvent(kind="reasoning", text=detail["text"])
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                yield StreamEvent(kind="content", text=content)
                            for fragment in delta.get("tool_calls") or []:
                                if not isinstance(fragment, dict):
                                    continue
                                index = int(fragment.get("index") or 0)
                                call = tool_calls.setdefault(index, {"arguments_parts": []})
                                if fragment.get("id"):
                                    call["id"] = fragment["id"]
                                function = fragment.get("function") or {}
                                if isinstance(function, dict):
                                    if function.get("name"):
                                        call["name"] = function["name"]
                                    if isinstance(function.get("arguments"), str):
                                        call["arguments_parts"].append(function["arguments"])
                            if choice.get("finish_reason") == "tool_calls":
                                async for event in flush_tool_calls():
                                    yield event
                        if isinstance(usage, dict):
                            details = usage.get("completion_tokens_details") or {}
                            if not isinstance(details, dict):
                                details = {}
                            yield StreamEvent(
                                kind="usage",
                                usage={
                                    "prompt": int(usage.get("prompt_tokens") or 0),
                                    "completion": int(usage.get("completion_tokens") or 0),
                                    "reasoning": int(details.get("reasoning_tokens") or 0),
                                },
                            )
            async for event in flush_tool_calls():
                yield event
            yield StreamEvent(kind="done")
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc

    async def complete_tools(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                *messages,
            ],
            "tools": tools,
            # only route to providers that honor tool parameters
            "provider": {"require_parameters": True},
            **_reasoning_for(model),
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with httpx.AsyncClient(
                timeout=config.LLM_TIMEOUT_S,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{config.OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError("OpenRouter message was not a JSON object")
            return message
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(str(exc)) from exc


def _reasoning_for(model: str) -> dict[str, Any]:
    reasoning = config.LLM_REASONING_BY_MODEL.get(model)
    return {"reasoning": dict(reasoning)} if reasoning else {}


def _parse_json_content(content: str) -> Any:
    """Parse model output that may wrap JSON in markdown fences or prose."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])
