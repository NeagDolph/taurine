from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from taurine_core import config
from taurine_core.chat import engine as engine_module
from taurine_core.chat.engine import ChatEngine
from taurine_core.llm import StreamEvent
from taurine_core.ports import LLMError
from taurine_core.schemas import ChatActivityStep, ChatMessage, GroundedFact, GroundingItem


class MemoryLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def log(self, kind: str, **payload: Any) -> None:
        self.records.append((kind, payload))


class FakeRetriever:
    instances: list["FakeRetriever"] = []

    def __init__(self, index: Any, chat_id: str) -> None:
        del index
        self.chat_id = chat_id
        self.saved = 0
        FakeRetriever.instances.append(self)

    def context_pack(self, message: str, *, now: float) -> tuple[list[GroundingItem], list[GroundedFact]]:
        del message, now
        return (
            [
                GroundingItem(
                    note_id="n1",
                    short_id="N1",
                    title="Roadmap",
                    gist="Roadmap gist",
                    score=0.9,
                )
            ],
            [GroundedFact(fact_id="F1", src="n1", dst="n2", text="Roadmap relates to Beta.")],
        )

    def resolve(self, ref: str) -> str | None:
        return {"N1": "n1", "n1": "n1"}.get(ref)

    def save(self) -> None:
        self.saved += 1


@dataclass
class FakeLLM:
    streams: list[Any]
    title: str | Exception = "Project Answers"
    calls: list[dict[str, Any]] = field(default_factory=list)
    stream_started: asyncio.Event | None = None
    release_stream: asyncio.Event | None = None

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
        timeout_s: float,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "reasoning": reasoning,
                "timeout_s": timeout_s,
            }
        )
        stream = self.streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream
        for event in stream:
            if event == "wait":
                if self.stream_started is not None:
                    self.stream_started.set()
                if self.release_stream is not None:
                    await self.release_stream.wait()
                continue
            if isinstance(event, BaseException):
                raise event
            yield event

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        del model, system, user, timeout_s
        if isinstance(self.title, Exception):
            raise self.title
        return {"title": self.title}


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRetriever.instances = []
    monkeypatch.setattr(engine_module, "ChatRetriever", FakeRetriever, raising=False)


async def _capture(event: Any, events: list[dict[str, Any]]) -> None:
    events.append(event.model_dump(mode="json") if hasattr(event, "model_dump") else event)


@pytest.mark.asyncio
async def test_turn_event_order_tool_round_trip_budget_no_tools_and_citation_strip(
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "CHAT_MAX_TOOL_CALLS", 1)
    tool_step = ChatActivityStep(kind="search", label='Searched "roadmap" → 1 notes', note_ids=["n2"])
    monkeypatch.setattr(
        engine_module,
        "execute_tool",
        lambda name, arguments, retriever, index, now: (
            '{"rows":[{"short_id":"N2","title":"Beta","source":"apple_notes","age_days":0.0,"score":0.8,"gist":"Beta gist"}]}',
            tool_step,
        ),
    )
    llm = FakeLLM(
        streams=[
            [
                StreamEvent(kind="reasoning", text="thinking"),
                StreamEvent(kind="tool_call", tool_call={"id": "call-1", "name": "search_notes", "arguments": {"query": "roadmap"}}),
                StreamEvent(kind="usage", usage={"prompt": 2, "completion": 3, "reasoning": 4}),
                StreamEvent(kind="done"),
            ],
            [
                StreamEvent(kind="reasoning", text="again"),
                StreamEvent(kind="content", text="Answer [N1] [N404] [F1] [F404]"),
                StreamEvent(kind="usage", usage={"prompt": 5, "completion": 7, "reasoning": 11}),
                StreamEvent(kind="done"),
            ],
        ]
    )
    logger = MemoryLogger()
    engine = ChatEngine(store, llm, logger)
    events: list[dict[str, Any]] = []

    await engine.send("", "What changed?", lambda event: _capture(event, events))

    assert [event["type"] for event in events[:8]] == [
        "chat_meta",
        "chat_ack",
        "chat_retrieval",
        "chat_thinking",
        "chat_activity",
        "chat_activity",
        "chat_thinking",
        "chat_delta",
    ]
    assert events[-2]["type"] == "chat_done"
    assert events[-1]["type"] == "chat_meta"
    done = events[-2]["message"]
    assert done["content"] == "Answer [N1]  [F1] "
    assert done["usage"]["prompt"] == 7
    assert done["usage"]["completion"] == 10
    assert done["usage"]["reasoning"] == 15
    assert done["usage"]["ttft_ms"] >= 0
    assert done["usage"]["total_ms"] >= done["usage"]["ttft_ms"]
    assert [call["model"] for call in llm.calls] == [config.CHAT_MODEL, config.CHAT_MODEL]
    assert llm.calls[0]["tools"] is not None
    assert llm.calls[1]["tools"] is None
    assert llm.calls[1]["messages"][-2]["tool_calls"][0]["id"] == "call-1"
    assert llm.calls[1]["messages"][-1]["role"] == "tool"
    assert store.chat_messages(done["chat_id"])[1].content == "Answer [N1]  [F1] "
    assert ("chat_retrieval", {"chat_id": done["chat_id"], "query": "What changed?", "top": [{"id": "N1", "score": 0.9}], "facts": ["F1"]}) in logger.records
    assert any(kind == "chat_turn" and payload["usage"]["prompt"] == 7 for kind, payload in logger.records)


@pytest.mark.asyncio
async def test_busy_rejection_for_same_chat_and_global_cap(
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "CHAT_MAX_CONCURRENT_TURNS", 1)
    started = asyncio.Event()
    release = asyncio.Event()
    llm = FakeLLM(
        streams=[
            ["wait", StreamEvent(kind="content", text="done"), StreamEvent(kind="done")],
        ],
        stream_started=started,
        release_stream=release,
    )
    engine = ChatEngine(store, llm, MemoryLogger())
    first_events: list[dict[str, Any]] = []
    first = asyncio.create_task(engine.send("", "first", lambda event: _capture(event, first_events)))
    await started.wait()
    chat_id = next(event["chat"]["chat_id"] for event in first_events if event["type"] == "chat_meta")

    same_chat_events: list[dict[str, Any]] = []
    other_chat_events: list[dict[str, Any]] = []
    await engine.send(chat_id, "second", lambda event: _capture(event, same_chat_events))
    await engine.send("", "other", lambda event: _capture(event, other_chat_events))

    release.set()
    await first
    assert same_chat_events == [{"type": "chat_error", "chat_id": chat_id, "message_id": None, "code": "busy", "detail": ""}]
    assert other_chat_events == [{"type": "chat_error", "chat_id": "", "message_id": None, "code": "busy", "detail": ""}]


@pytest.mark.asyncio
async def test_cancel_persists_partial_content_with_cancelled_true(store: Any) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    llm = FakeLLM(
        streams=[
            [
                StreamEvent(kind="reasoning", text="partial thought"),
                StreamEvent(kind="content", text="partial answer"),
                "wait",
                StreamEvent(kind="done"),
            ]
        ],
        stream_started=started,
        release_stream=release,
    )
    engine = ChatEngine(store, llm, MemoryLogger())
    events: list[dict[str, Any]] = []
    task = asyncio.create_task(engine.send("", "cancel me", lambda event: _capture(event, events)))
    await started.wait()
    chat_id = next(event["chat"]["chat_id"] for event in events if event["type"] == "chat_meta")

    await engine.cancel(chat_id)
    await task

    done = [event for event in events if event["type"] == "chat_done"][-1]
    assert done["message"]["content"] == "partial answer"
    assert done["message"]["thinking"] == "partial thought"
    assert done["message"]["cancelled"] is True
    assert store.chat_messages(chat_id)[1].cancelled is True


@pytest.mark.asyncio
async def test_regenerate_deletes_last_assistant_and_reruns_last_user(store: Any) -> None:
    first_llm = FakeLLM(streams=[[StreamEvent(kind="content", text="first"), StreamEvent(kind="done")]])
    engine = ChatEngine(store, first_llm, MemoryLogger())
    events: list[dict[str, Any]] = []
    await engine.send("", "question", lambda event: _capture(event, events))
    chat_id = [event for event in events if event["type"] == "chat_done"][-1]["chat_id"]

    engine.llm = FakeLLM(streams=[[StreamEvent(kind="content", text="second"), StreamEvent(kind="done")]])
    regen_events: list[dict[str, Any]] = []
    await engine.regenerate(chat_id, lambda event: _capture(event, regen_events))

    messages = store.chat_messages(chat_id)
    assert [(message.role, message.content) for message in messages] == [
        ("user", "question"),
        ("assistant", "second"),
    ]
    assert [event["type"] for event in regen_events[:3]] == ["chat_ack", "chat_retrieval", "chat_delta"]


@pytest.mark.asyncio
async def test_fallback_only_for_first_stream_open_failure(store: Any) -> None:
    llm = FakeLLM(
        streams=[
            LLMError("primary failed before events"),
            [StreamEvent(kind="content", text="fallback answer"), StreamEvent(kind="done")],
        ]
    )
    events: list[dict[str, Any]] = []

    await ChatEngine(store, llm, MemoryLogger()).send("", "question", lambda event: _capture(event, events))

    assert [call["model"] for call in llm.calls] == [config.CHAT_MODEL, config.CHAT_FALLBACK_MODEL]
    assert [event["type"] for event in events if event["type"] == "chat_error"] == []
    assert [event for event in events if event["type"] == "chat_done"][-1]["message"]["content"] == "fallback answer"


@pytest.mark.asyncio
async def test_midstream_llm_error_emits_error_and_persists_partial_without_fallback(store: Any) -> None:
    llm = FakeLLM(
        streams=[
            [
                StreamEvent(kind="content", text="partial"),
                LLMError("provider failed midstream"),
            ],
        ]
    )
    events: list[dict[str, Any]] = []

    await ChatEngine(store, llm, MemoryLogger()).send("", "question", lambda event: _capture(event, events))

    assert [call["model"] for call in llm.calls] == [config.CHAT_MODEL]
    error = [event for event in events if event["type"] == "chat_error"][-1]
    done = [event for event in events if event["type"] == "chat_done"][-1]
    assert error["code"] == "llm"
    assert done["message"]["content"] == "partial"
    assert done["message"]["cancelled"] is False


@pytest.mark.asyncio
async def test_auto_title_failure_falls_back_to_first_40_chars(store: Any) -> None:
    llm = FakeLLM(
        streams=[[StreamEvent(kind="content", text="answer"), StreamEvent(kind="done")]],
        title=RuntimeError("title unavailable"),
    )
    engine = ChatEngine(store, llm, MemoryLogger())
    events: list[dict[str, Any]] = []
    text = "abcdefghijklmnopqrstuvwxyz0123456789 extra"

    await engine.send("", text, lambda event: _capture(event, events))

    titled = events[-1]["chat"]
    assert titled["title"] == "abcdefghijklmnopqrstuvwxyz0123456789 ext"
    assert store.chat(titled["chat_id"]).title == "abcdefghijklmnopqrstuvwxyz0123456789 ext"
