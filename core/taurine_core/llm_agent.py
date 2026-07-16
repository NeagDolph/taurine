"""Small non-streaming tool-agent loop with one shared wall-clock budget."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any


async def run_tool_agent(
    llm: Any,
    *,
    model: str,
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    execute_tool: Callable[[str, dict[str, Any]], Awaitable[str]],
    max_tool_calls: int,
    timeout_s: float,
    schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Run bounded tool rounds, then request strict JSON over the full history.

    ``schema`` is an optional, backward-compatible extension to the plan
    signature so Gemini can use the measured-clean ``json_schema`` mode.
    """

    deadline = time.monotonic() + timeout_s
    history: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    tool_count = 0

    while tool_count < max(0, max_tool_calls):
        remaining = _remaining(deadline)
        async with asyncio.timeout(remaining):
            assistant = await llm.complete_tools(
                model,
                system,
                history[1:],
                tools,
                timeout_s=remaining,
            )
        if not isinstance(assistant, dict):
            raise TypeError("tool completion was not a message object")
        history.append(dict(assistant))
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            break

        for call in calls:
            call_id, name, arguments = _parse_tool_call(call)
            if tool_count >= max_tool_calls:
                result = json.dumps({"error": "tool call budget exhausted"})
            else:
                remaining = _remaining(deadline)
                async with asyncio.timeout(remaining):
                    result = await execute_tool(name, arguments)
                tool_count += 1
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": str(result),
                }
            )
        if tool_count >= max_tool_calls:
            break

    history.append(
        {
            "role": "user",
            "content": "Now return the final JSON result exactly as instructed.",
        }
    )
    remaining = _remaining(deadline)
    async with asyncio.timeout(remaining):
        return await llm.complete_json_messages(
            model,
            history,
            timeout_s=remaining,
            schema=schema,
            max_tokens=max_tokens,
        )


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("tool agent timed out")
    return remaining


def _parse_tool_call(call: Any) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(call, dict):
        return "", "", {}
    function = call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments")
    else:
        name = str(call.get("name") or "")
        raw_arguments = call.get("arguments")
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed = {}
        arguments = parsed if isinstance(parsed, dict) else {}
    else:
        arguments = {}
    return str(call.get("id") or ""), name, arguments
