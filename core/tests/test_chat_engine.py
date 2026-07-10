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
from taurine_core.schemas import ChatActivityStep, ChatMessage
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
)
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.validate import EvidenceValidator


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

    def context_pack(
        self,
        message: str,
        *,
        now: float,
        conversation_context: tuple[str, ...] = (),
        rewritten_query: str | None = None,
    ) -> EvidencePack:
        del conversation_context, rewritten_query
        return EvidencePack(
            decision=RetrievalDecision(True, .9, "accepted", ("dense",)),
            query=RetrievalQuery(message, "chat", now),
            passages=(EvidencePassage(
                "E1", "p1", "n1", "Roadmap", "obsidian", "Roadmap gist", 0, 12,
                now, now, .8, .9,
            ),),
            facts=(EvidenceFact(
                "F1", ("f1",), "Roadmap", "relates_to", "Beta", "forward",
                "active", now, None, None, .9, ("E1",),
            ),),
            paths=(),
            notes=(EvidenceNote("N1", "n1", "Roadmap", "obsidian", now, .9, ("E1",)),),
            trace=RetrievalTrace(
                original_query=message,
                rewritten_query=message,
                selected_ids=("N1", "E1", "F1"),
                read_ids=("N1", "E1", "F1"),
                active_build_id="build-test",
            ),
        )

    def resolve(self, ref: str) -> str | None:
        return {"N1": "n1", "n1": "n1"}.get(ref)

    def save(self) -> None:
        self.saved += 1


class EvidenceRetriever(FakeRetriever):
    def __init__(self, index: Any, chat_id: str) -> None:
        super().__init__(index, chat_id)
        self.registry = EvidenceRegistry()
        self.registry.register("note", "n1", {"note_id": "n1"})
        self.registry.register(
            "passage",
            "p1",
            {"passage_id": "p1", "note_id": "n1", "text": "The launch is July 18.", "char_start": 0, "char_end": 22},
        )
        for ref in ("N1", "E1"):
            self.registry.mark_read(ref)

    def context_pack(
        self,
        message: str,
        *,
        now: float,
        conversation_context: tuple[str, ...] = (),
        rewritten_query: str | None = None,
    ) -> EvidencePack:
        del conversation_context, rewritten_query
        return EvidencePack(
            decision=RetrievalDecision(True, .9, "accepted", ("dense",)),
            query=RetrievalQuery(message, "chat", now),
            passages=(EvidencePassage(
                "E1", "p1", "n1", "Roadmap", "obsidian", "The launch is July 18.", 0, 22,
                now, now, .8, .9,
            ),),
            facts=(), paths=(),
            notes=(EvidenceNote("N1", "n1", "Roadmap", "obsidian", now, .9, ("E1",)),),
            trace=RetrievalTrace(
                original_query=message, rewritten_query=message,
                selected_ids=("N1", "E1"), read_ids=("N1", "E1"), active_build_id="build-test",
            ),
        )


class FixedNLI:
    def __init__(self, score: float) -> None:
        self.value = score

    def score(self, premise: str, hypothesis: str) -> float:
        del premise, hypothesis
        return self.value


@dataclass
class FakeLLM:
    streams: list[Any]
    title: str | Exception = "Project Answers"
    calls: list[dict[str, Any]] = field(default_factory=list)
    stream_started: asyncio.Event | None = None
    release_stream: asyncio.Event | None = None
    title_started: asyncio.Event | None = None
    release_title: asyncio.Event | None = None

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
        if self.title_started is not None:
            self.title_started.set()
        if self.release_title is not None:
            await self.release_title.wait()
        if isinstance(self.title, Exception):
            raise self.title
        return {"title": self.title}


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRetriever.instances = []
    monkeypatch.setattr(engine_module, "ChatRetriever", FakeRetriever, raising=False)


async def _capture(event: Any, events: list[dict[str, Any]]) -> None:
    events.append(event.model_dump(mode="json") if hasattr(event, "model_dump") else event)


def _terminal_types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events if event["type"] in {"chat_done", "chat_error"}]


@pytest.mark.asyncio
async def test_chat_injects_latest_user_profile_as_uncitable_system_context(store: Any) -> None:
    store.set_user_card("Neil builds Taurine and works on live knowledge systems.")
    llm = FakeLLM(
        streams=[
            [
                StreamEvent(kind="content", text="Grounded answer [N1]"),
                StreamEvent(kind="done"),
            ]
        ]
    )

    await ChatEngine(store, llm, MemoryLogger()).send(
        "",
        "What am I working on?",
        lambda _event: _capture(_event, []),
    )

    system = llm.calls[0]["messages"][0]["content"]
    assert (
        "<user_profile>\nNeil builds Taurine and works on live knowledge systems.\n"
        "</user_profile>"
    ) in system
    assert "background context" in system
    assert "not citable evidence" in system


@pytest.mark.asyncio
async def test_chat_omits_user_profile_block_when_card_is_missing(store: Any) -> None:
    llm = FakeLLM(
        streams=[[StreamEvent(kind="content", text="Answer"), StreamEvent(kind="done")]]
    )

    await ChatEngine(store, llm, MemoryLogger()).send(
        "",
        "Question",
        lambda _event: _capture(_event, []),
    )

    assert "<user_profile>" not in llm.calls[0]["messages"][0]["content"]


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
        lambda name, arguments, retriever, now: (
            '{"passages":[{"evidence_id":"E2","note_id":"n2","title":"Beta","text":"Beta gist"}]}',
            tool_step,
        ),
    )
    llm = FakeLLM(
        streams=[
            [
                StreamEvent(kind="reasoning", text="thinking"),
                StreamEvent(kind="tool_call", tool_call={"id": "call-1", "name": "search_evidence", "arguments": {"query": "roadmap"}}),
                StreamEvent(kind="usage", usage={"prompt": 2, "completion": 3, "reasoning": 4}),
                StreamEvent(kind="done"),
            ],
            [
                StreamEvent(kind="reasoning", text="again"),
                StreamEvent(kind="content", text="Answer [E1] [F1]"),
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
    assert _terminal_types(events) == ["chat_done"]
    assert done["content"] == "Answer [E1] [F1]"
    assert done["usage"]["prompt"] == 7
    assert done["usage"]["completion"] == 10
    assert done["usage"]["reasoning"] == 15
    assert done["usage"]["ttft_ms"] >= 0
    assert done["usage"]["total_ms"] >= done["usage"]["ttft_ms"]
    assert done["activity"][0]["tool_name"] == "search_evidence"
    assert done["activity"][0]["arguments"] == {"query": "roadmap"}
    assert done["activity"][0]["evidence_ids"] == ["E2"]
    assert done["activity"][0]["latency_ms"] >= 0
    assert [call["model"] for call in llm.calls] == [config.CHAT_MODEL, config.CHAT_MODEL]
    assert llm.calls[0]["tools"] is not None
    assert llm.calls[1]["tools"] is None
    assert llm.calls[1]["messages"][-3]["tool_calls"][0]["id"] == "call-1"
    assert llm.calls[1]["messages"][-2]["role"] == "tool"
    # the forced no-tools round tells the model why tools vanished (GLM
    # otherwise emits its native tool markup as plain text)
    budget_notice = llm.calls[1]["messages"][-1]
    assert budget_notice["role"] == "user"
    assert budget_notice["content"] != "What changed?"
    assert store.chat_messages(done["chat_id"])[1].content == "Answer [E1] [F1]"
    assert any(
        kind == "chat_retrieval"
        and payload["original_query"] == "What changed?"
        and payload["active_build_id"] == "build-test"
        for kind, payload in logger.records
    )
    assert any(kind == "chat_turn" and payload["usage"]["prompt"] == 7 for kind, payload in logger.records)


@pytest.mark.asyncio
async def test_pre_tool_narration_is_separated_from_the_answer(
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine_module,
        "execute_tool",
        lambda name, arguments, retriever, now: (
            '{"passages":[]}',
            ChatActivityStep(kind="search", label="Searched", note_ids=[]),
        ),
    )
    llm = FakeLLM(
        streams=[
            [
                StreamEvent(kind="content", text="Let me look that up for full detail."),
                StreamEvent(kind="tool_call", tool_call={"id": "c1", "name": "search_evidence", "arguments": {"query": "x"}}),
                StreamEvent(kind="done"),
            ],
            [
                StreamEvent(kind="content", text="Here's the answer."),
                StreamEvent(kind="done"),
            ],
        ]
    )
    engine = ChatEngine(store, llm, MemoryLogger())
    events: list[dict[str, Any]] = []

    await engine.send("", "State?", lambda event: _capture(event, events))

    done = next(event for event in events if event["type"] == "chat_done")
    # the round-1 narration and the round-2 answer must not run together
    assert done["message"]["content"] == "Let me look that up for full detail.\n\nHere's the answer."
    # persisted content equals the concatenation of streamed deltas
    streamed = "".join(event["delta"] for event in events if event["type"] == "chat_delta")
    assert streamed == done["message"]["content"]


@pytest.mark.asyncio
async def test_tool_call_remnants_are_stripped_from_final_content(store: Any) -> None:
    llm = FakeLLM(
        streams=[
            [
                StreamEvent(
                    kind="content",
                    text=(
                        "Here is the state [N1].<tool_call>read_note"
                        "<arg_key>note</arg_key><arg_value>N16</arg_value></tool_call>"
                    ),
                ),
                StreamEvent(kind="content", text="\n<tool_call>unterminated leak"),
                StreamEvent(kind="done"),
            ],
        ]
    )
    engine = ChatEngine(store, llm, MemoryLogger())
    events: list[dict[str, Any]] = []

    await engine.send("", "State of things?", lambda event: _capture(event, events))

    done = next(event for event in events if event["type"] == "chat_done")
    assert done["message"]["content"] == "Here is the state [N1]."
    assert _terminal_types(events) == ["chat_done"]


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
    with pytest.raises(asyncio.CancelledError):
        await task

    done = [event for event in events if event["type"] == "chat_done"][-1]
    assert _terminal_types(events) == ["chat_done"]
    assert done["message"]["content"] == "partial answer"
    assert done["message"]["thinking"] == "partial thought"
    assert done["message"]["cancelled"] is True
    assert store.chat_messages(chat_id)[1].cancelled is True


@pytest.mark.asyncio
async def test_cancel_during_success_finish_does_not_emit_cancelled_second_terminal(store: Any) -> None:
    title_started = asyncio.Event()
    release_title = asyncio.Event()
    llm = FakeLLM(
        streams=[[StreamEvent(kind="content", text="complete answer"), StreamEvent(kind="done")]],
        title_started=title_started,
        release_title=release_title,
    )
    engine = ChatEngine(store, llm, MemoryLogger())
    events: list[dict[str, Any]] = []
    task = asyncio.create_task(engine.send("", "title me", lambda event: _capture(event, events)))
    await title_started.wait()
    chat_id = next(event["chat"]["chat_id"] for event in events if event["type"] == "chat_meta")

    cancel_task = asyncio.create_task(engine.cancel(chat_id))
    await asyncio.sleep(0)
    release_title.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancel_task

    done_events = [event for event in events if event["type"] == "chat_done"]
    assert len(done_events) == 1
    assert _terminal_types(events) == ["chat_done"]
    assert done_events[0]["message"]["content"] == "complete answer"
    assert done_events[0]["message"]["cancelled"] is False
    assert store.chat_messages(chat_id)[1].cancelled is False


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
    chat_id = next(event["chat"]["chat_id"] for event in events if event["type"] == "chat_meta")
    assert error["code"] == "llm"
    assert _terminal_types(events) == ["chat_error"]
    persisted = store.chat_messages(chat_id)[1]
    assert persisted.content == "partial"
    assert persisted.cancelled is False


@pytest.mark.asyncio
async def test_context_pack_exception_emits_error_persists_empty_partial_and_does_not_raise(
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RaisingRetriever(FakeRetriever):
        def context_pack(
            self, message: str, *, now: float, conversation_context: tuple[str, ...] = (),
            rewritten_query: str | None = None,
        ) -> EvidencePack:
            del message, now, conversation_context, rewritten_query
            raise RuntimeError("context exploded")

    monkeypatch.setattr(engine_module, "ChatRetriever", RaisingRetriever, raising=False)
    logger = MemoryLogger()
    events: list[dict[str, Any]] = []

    await ChatEngine(store, FakeLLM(streams=[]), logger).send(
        "",
        "question",
        lambda event: _capture(event, events),
    )

    chat_id = next(event["chat"]["chat_id"] for event in events if event["type"] == "chat_meta")
    error = [event for event in events if event["type"] == "chat_error"][-1]
    assert error["code"] == "llm"
    assert error["detail"] == "context exploded"
    assert _terminal_types(events) == ["chat_error"]
    persisted = store.chat_messages(chat_id)[1]
    assert persisted.content == ""
    assert persisted.cancelled is False
    assert any(
        kind == "chat_error" and payload["detail"] == "context exploded"
        for kind, payload in logger.records
    )


@pytest.mark.asyncio
async def test_final_answer_is_nli_validated_and_persists_cited_vs_consulted(
    store: Any,
) -> None:
    llm = FakeLLM(
        streams=[[StreamEvent(kind="content", text="The launch is July 18 [E1]."), StreamEvent(kind="done")]]
    )
    engine = ChatEngine(
        store,
        llm,
        MemoryLogger(),
        retriever_factory=lambda chat_id: EvidenceRetriever(store, chat_id),
        validator_factory=lambda registry: EvidenceValidator(
            registry, FixedNLI(1.0), entailment_floor=.8
        ),
    )
    events: list[dict[str, Any]] = []

    await engine.send("", "When is launch?", lambda event: _capture(event, events))

    done = next(event for event in events if event["type"] == "chat_done")["message"]
    assert done["content"] == "The launch is July 18 [E1]."
    assert done["cited_ids"] == ["E1"]
    assert done["consulted_ids"] == ["N1", "E1"]
    assert done["retrieved_ids"] == ["N1", "E1"]
    assert done["evidence"]["passages"][0]["text"] == "The launch is July 18."
    persisted = store.chat_messages(done["chat_id"])[1]
    assert persisted.cited_ids == ["E1"]
    assert persisted.consulted_ids == ["N1", "E1"]
    assert persisted.evidence is not None
    assert persisted.evidence.passages[0].evidence_id == "E1"
    assert persisted.original_query == "When is launch?"
    assert persisted.rewritten_query == "When is launch?"
    assert persisted.retrieval_decision == "accepted"
    assert persisted.active_build_id == "build-test"


@pytest.mark.asyncio
async def test_failed_chat_entailment_returns_source_excerpt_not_unsupported_claim(
    store: Any,
) -> None:
    llm = FakeLLM(
        streams=[[StreamEvent(kind="content", text="The launch is tomorrow [E1]."), StreamEvent(kind="done")]]
    )
    engine = ChatEngine(
        store,
        llm,
        MemoryLogger(),
        retriever_factory=lambda chat_id: EvidenceRetriever(store, chat_id),
        validator_factory=lambda registry: EvidenceValidator(
            registry, FixedNLI(0.0), entailment_floor=.8
        ),
    )
    events: list[dict[str, Any]] = []

    await engine.send("", "When is launch?", lambda event: _capture(event, events))

    done = next(event for event in events if event["type"] == "chat_done")["message"]
    assert "tomorrow" not in done["content"]
    assert "The launch is July 18." in done["content"]
    assert done["cited_ids"] == ["E1"]


@pytest.mark.asyncio
async def test_chat_runs_one_constrained_repair_and_revalidates_it(store: Any) -> None:
    class ConditionalNLI:
        def score(self, premise: str, hypothesis: str) -> float:
            del premise
            return 1.0 if "July 18" in hypothesis else 0.0

    class RepairLLM(FakeLLM):
        repair_calls = 0

        async def complete_json(self, model, system, user, timeout_s=None):
            del model, user, timeout_s
            if "Repair the draft" in system:
                self.repair_calls += 1
                return {"answer": "The launch is July 18 [E1]."}
            return {"title": "Launch date"}

    llm = RepairLLM(
        streams=[[StreamEvent(kind="content", text="The launch is tomorrow [E1]."), StreamEvent(kind="done")]]
    )
    engine = ChatEngine(
        store,
        llm,
        MemoryLogger(),
        retriever_factory=lambda chat_id: EvidenceRetriever(store, chat_id),
        validator_factory=lambda registry: EvidenceValidator(
            registry, ConditionalNLI(), entailment_floor=.8
        ),
    )
    events: list[dict[str, Any]] = []

    await engine.send("", "When is launch?", lambda event: _capture(event, events))

    done = next(event for event in events if event["type"] == "chat_done")["message"]
    assert done["content"] == "The launch is July 18 [E1]."
    assert done["cited_ids"] == ["E1"]
    assert llm.repair_calls == 1


@pytest.mark.asyncio
async def test_dependent_query_uses_one_structured_flash_rewrite(store: Any) -> None:
    rewrites: list[str | None] = []

    class RewriteRetriever(EvidenceRetriever):
        def is_dependent_query(self, message: str, context: tuple[str, ...]) -> bool:
            return message == "When does it launch?" and bool(context)

        def fallback_rewrite(self, message: str, context: tuple[str, ...]) -> str:
            return "\n".join((*context, message))

        def context_pack(self, message: str, *, now: float, conversation_context=(), rewritten_query=None):
            rewrites.append(rewritten_query)
            return super().context_pack(
                message,
                now=now,
                conversation_context=conversation_context,
                rewritten_query=rewritten_query,
            )

    class RewriteLLM(FakeLLM):
        rewrite_calls = 0

        async def complete_json(self, model, system, user, timeout_s=None):
            del user, timeout_s
            if "self-contained retrieval query" in system:
                self.rewrite_calls += 1
                assert model == config.SUMMARY_MODEL
                return {"query": "When does the Mermaid roadmap launch?"}
            return {"title": "Mermaid launch"}

    store.create_chat("chat-rewrite", now=1.0)
    store.add_chat_message(ChatMessage(chat_id="chat-rewrite", role="user", content="Tell me about the Mermaid roadmap", created=1.0))
    store.add_chat_message(ChatMessage(chat_id="chat-rewrite", role="assistant", content="It is documented.", created=2.0))
    llm = RewriteLLM(
        streams=[[StreamEvent(kind="content", text="The launch is July 18 [E1]."), StreamEvent(kind="done")]]
    )
    engine = ChatEngine(
        store,
        llm,
        MemoryLogger(),
        retriever_factory=lambda chat_id: RewriteRetriever(store, chat_id),
        validator_factory=lambda registry: EvidenceValidator(registry, FixedNLI(1.0), entailment_floor=.8),
    )

    await engine.send("chat-rewrite", "When does it launch?", lambda _event: _capture(_event, []))

    assert rewrites == ["When does the Mermaid roadmap launch?"]
    assert llm.rewrite_calls == 1


@pytest.mark.asyncio
async def test_regeneration_restores_pre_turn_retrieval_state(store: Any) -> None:
    store.create_chat("chat-regen-state", now=1.0)
    store.add_chat_message(ChatMessage(chat_id="chat-regen-state", role="user", content="Question", created=1.0))
    store.add_chat_message(ChatMessage(chat_id="chat-regen-state", role="assistant", content="Old", created=2.0))
    restored: list[str] = []

    class RegenerationRetriever(FakeRetriever):
        def restore_pre_turn(self) -> None:
            restored.append(self.chat_id)

    llm = FakeLLM(streams=[[StreamEvent(kind="content", text="New"), StreamEvent(kind="done")]])
    engine = ChatEngine(
        store,
        llm,
        MemoryLogger(),
        retriever_factory=lambda chat_id: RegenerationRetriever(store, chat_id),
    )

    await engine.regenerate("chat-regen-state", lambda _event: _capture(_event, []))

    assert restored == ["chat-regen-state"]


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
