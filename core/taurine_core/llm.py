from __future__ import annotations

import json
from typing import Any

import httpx

from . import config
from .ports import LLMError


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
