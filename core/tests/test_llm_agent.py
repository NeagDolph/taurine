from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from taurine_core.llm_agent import run_tool_agent


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


class FakeLLM:
    def __init__(self, tool_responses: list[dict[str, Any]], final: dict[str, Any]) -> None:
        self.tool_responses = list(tool_responses)
        self.final = final
        self.tool_calls: list[dict[str, Any]] = []
        self.final_calls: list[dict[str, Any]] = []

    async def complete_tools(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.tool_calls.append(
            {
                "model": model,
                "system": system,
                "messages": list(messages),
                "tools": tools,
                "timeout_s": timeout_s,
            }
        )
        return self.tool_responses.pop(0)

    async def complete_json_messages(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.final_calls.append(
            {
                "model": model,
                "messages": list(messages),
                "timeout_s": timeout_s,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "schema": schema,
            }
        )
        return self.final


async def test_tool_loop_executes_calls_and_finalizes_over_full_history() -> None:
    schema = {
        "title": "answer",
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "search_evidence", {"query": "roadmap"})],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c2", "read_note", {"note": "N1", "query": "owner"})],
            },
            {"role": "assistant", "content": "I have enough evidence."},
        ],
        {"answer": "Maya owns it."},
    )
    executed: list[tuple[str, dict[str, Any]]] = []

    async def execute(name: str, arguments: dict[str, Any]) -> str:
        executed.append((name, arguments))
        return json.dumps({"tool": name})

    result = await run_tool_agent(
        llm,
        model="model-a",
        system="system",
        user="user",
        tools=[{"type": "function"}],
        execute_tool=execute,
        max_tool_calls=6,
        timeout_s=10,
        schema=schema,
    )

    assert result == {"answer": "Maya owns it."}
    assert executed == [
        ("search_evidence", {"query": "roadmap"}),
        ("read_note", {"note": "N1", "query": "owner"}),
    ]
    assert len(llm.tool_calls) == 3
    assert all(0 < call["timeout_s"] <= 10 for call in llm.tool_calls)
    final = llm.final_calls[0]
    assert final["schema"] == schema
    assert final["messages"][0] == {"role": "system", "content": "system"}
    assert final["messages"][-1] == {
        "role": "user",
        "content": "Now return the final JSON result exactly as instructed.",
    }
    assert [message["role"] for message in final["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert final["messages"][3]["tool_call_id"] == "c1"
    assert final["messages"][5]["tool_call_id"] == "c2"


async def test_tool_call_cap_returns_budget_errors_without_executing_extra_calls() -> None:
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("c1", "search_evidence", {"query": "one"}),
                    _tool_call("c2", "search_evidence", {"query": "two"}),
                ],
            }
        ],
        {"answer": "done"},
    )
    executed: list[str] = []

    async def execute(name: str, arguments: dict[str, Any]) -> str:
        del arguments
        executed.append(name)
        return "{}"

    await run_tool_agent(
        llm,
        model="model-a",
        system="system",
        user="user",
        tools=[],
        execute_tool=execute,
        max_tool_calls=1,
        timeout_s=10,
    )

    assert executed == ["search_evidence"]
    assert len(llm.tool_calls) == 1
    tool_messages = [
        row for row in llm.final_calls[0]["messages"] if row["role"] == "tool"
    ]
    assert len(tool_messages) == 2
    assert "budget exhausted" in tool_messages[1]["content"]


async def test_timeout_covers_hanging_tool_execution() -> None:
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "read_note", {"note": "N1"})],
            }
        ],
        {"answer": "unused"},
    )

    async def execute(name: str, arguments: dict[str, Any]) -> str:
        del name, arguments
        await asyncio.Event().wait()
        return "unreachable"

    with pytest.raises(TimeoutError):
        await run_tool_agent(
            llm,
            model="model-a",
            system="system",
            user="user",
            tools=[],
            execute_tool=execute,
            max_tool_calls=1,
            timeout_s=0.01,
        )


async def test_zero_budget_raises_before_calling_llm() -> None:
    llm = FakeLLM([], {"answer": "unused"})

    with pytest.raises(TimeoutError):
        await run_tool_agent(
            llm,
            model="model-a",
            system="system",
            user="user",
            tools=[],
            execute_tool=lambda name, arguments: asyncio.sleep(0, result="{}"),
            max_tool_calls=1,
            timeout_s=0,
        )

    assert llm.tool_calls == []
    assert llm.final_calls == []
