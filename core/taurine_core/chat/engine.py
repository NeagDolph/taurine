"""Stage B — the streamed GLM-5.2 thinking turn loop (spec chat §1.3–1.4).

One ChatEngine per server; turns are per-chat single-flight with
CHAT_MAX_CONCURRENT_TURNS across chats. A turn:

1. persist the user message; create the empty assistant row (→ chat_ack)
2. Stage-A context pack (→ chat_retrieval)
3. stream_chat(CHAT_MODEL, reasoning=CHAT_REASONING, tools=TOOL_DEFINITIONS):
   reasoning deltas → chat_thinking, content deltas → chat_delta; on
   tool_call: emit chat_activity(running), execute, emit chat_activity(done),
   append tool message, re-open the stream — until no tool calls or
   CHAT_MAX_TOOL_CALLS
4. validate citations (lenient: strip unknown [N#]/[F#], never reject),
   resolve sources, finish_chat_message, save retriever state (→ chat_done)
5. auto-title after the first completed exchange (one flash complete_json;
   fallback: first 40 chars) (→ chat_meta)

Cancellation (chat_cancel): cancel the turn task; partial content persists
with cancelled=True (→ chat_done with cancelled). Fallback model only on
stream-OPEN failure; mid-stream errors → chat_error. Session-log record
kinds: chat_turn, chat_retrieval.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine
from uuid import uuid4

from .. import config
from ..indexer.store import Index
from ..llm import OpenRouterClient, StreamEvent
from ..ports import LLMError
from ..schemas import (
    ChatAckEvent,
    ChatActivityEvent,
    ChatActivityStep,
    ChatDeltaEvent,
    ChatDoneEvent,
    ChatErrorEvent,
    ChatMessage,
    ChatMeta,
    ChatMetaEvent,
    ChatRetrievalEvent,
    ChatThinkingEvent,
    ChatUsage,
    GroundedFact,
    GroundingItem,
)
from .retrieval import ChatRetriever
from .tools import TOOL_DEFINITIONS, execute_tool

# async callback the server provides: emits one ws event dict/model to the client
EmitFn = Callable[[Any], Coroutine[Any, Any, None]]

_CITATION_RE = re.compile(r"\[(N|F)\d+\]")


@dataclass
class _Reservation:
    chat_id: str
    created: ChatMeta | None = None
    error: ChatErrorEvent | None = None


@dataclass
class _StreamOutcome:
    tool_calls: list[dict[str, Any]]


class _StreamFailure(Exception):
    def __init__(self, error: LLMError, *, opened: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.opened = opened


class ChatEngine:
    def __init__(self, index: Index, llm: OpenRouterClient, logger: Any) -> None:
        self.index = index
        self.llm = llm
        self.logger = logger
        self._lock = asyncio.Lock()
        self._active: dict[str, asyncio.Task[Any]] = {}
        self._active_count = 0

    async def send(self, chat_id: str, text: str, emit: EmitFn) -> None:
        """Handle chat_send ("" chat_id → create). Runs the full turn;
        busy/limit violations emit chat_error(code="busy") and return."""
        reservation = await self._reserve(chat_id)
        if reservation.error is not None:
            await emit(reservation.error)
            return
        if reservation.created is not None:
            await emit(ChatMetaEvent(chat=reservation.created))
        try:
            await self._run_turn(
                reservation.chat_id,
                text,
                emit,
                created_meta=reservation.created,
                persist_user=True,
            )
        finally:
            await self._release(reservation.chat_id)

    async def cancel(self, chat_id: str) -> None:
        async with self._lock:
            task = self._active.get(chat_id)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def regenerate(self, chat_id: str, emit: EmitFn) -> None:
        """Delete the last assistant message and re-run the last user one."""
        if self.index.chat(chat_id) is None:
            await emit(ChatErrorEvent(chat_id=chat_id, code="llm", detail="chat not found"))
            return
        reservation = await self._reserve(chat_id)
        if reservation.error is not None:
            await emit(reservation.error)
            return
        try:
            messages = self.index.chat_messages(chat_id)
            last_user = next((message for message in reversed(messages) if message.role == "user"), None)
            last_assistant = next((message for message in reversed(messages) if message.role == "assistant"), None)
            if last_user is None:
                await emit(ChatErrorEvent(chat_id=chat_id, code="llm", detail="no user message to regenerate"))
                return
            if last_assistant is not None:
                self.index.delete_chat_message(last_assistant.id)
            await self._run_turn(
                chat_id,
                last_user.content,
                emit,
                created_meta=None,
                persist_user=False,
                user_message_id=last_user.id,
            )
        finally:
            await self._release(chat_id)

    async def _reserve(self, chat_id: str) -> _Reservation:
        async with self._lock:
            if chat_id and chat_id in self._active:
                return _Reservation(chat_id, error=ChatErrorEvent(chat_id=chat_id, code="busy"))
            if self._active_count >= config.CHAT_MAX_CONCURRENT_TURNS:
                return _Reservation(chat_id, error=ChatErrorEvent(chat_id=chat_id, code="busy"))
            created: ChatMeta | None = None
            if not chat_id:
                now = time.time()
                while True:
                    chat_id = f"chat-{uuid4().hex[:8]}"
                    if self.index.chat(chat_id) is None:
                        break
                created = self.index.create_chat(chat_id, now=now)
            elif self.index.chat(chat_id) is None:
                return _Reservation(chat_id, error=ChatErrorEvent(chat_id=chat_id, code="llm", detail="chat not found"))
            task = asyncio.current_task()
            if task is None:
                return _Reservation(chat_id, error=ChatErrorEvent(chat_id=chat_id, code="llm", detail="missing task"))
            self._active[chat_id] = task
            self._active_count += 1
            return _Reservation(chat_id, created=created)

    async def _release(self, chat_id: str) -> None:
        async with self._lock:
            self._active.pop(chat_id, None)
            self._active_count = max(0, self._active_count - 1)

    async def _run_turn(
        self,
        chat_id: str,
        text: str,
        emit: EmitFn,
        *,
        created_meta: ChatMeta | None,
        persist_user: bool,
        user_message_id: int | None = None,
    ) -> None:
        turn_started = time.monotonic()
        wall_now = time.time()
        if persist_user:
            user_message_id = self.index.add_chat_message(
                ChatMessage(chat_id=chat_id, role="user", content=text, created=wall_now)
            )
        assert user_message_id is not None
        assistant_created = time.time()
        assistant_id = self.index.add_chat_message(
            ChatMessage(chat_id=chat_id, role="assistant", content="", created=assistant_created)
        )
        await emit(
            ChatAckEvent(
                chat_id=chat_id,
                message_id=assistant_id,
                user_message_id=user_message_id,
                title=created_meta.title if created_meta is not None else "",
            )
        )

        retriever = ChatRetriever(self.index, chat_id)
        context_items: list[GroundingItem] = []
        context_facts: list[GroundedFact] = []
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        activity: list[ChatActivityStep] = []
        usage = {"prompt": 0, "completion": 0, "reasoning": 0}
        ttft_ms: float | None = None
        tool_sources: dict[str, GroundingItem] = {}
        tool_summaries: list[dict[str, Any]] = []
        stripped_citations = 0
        model = config.CHAT_MODEL

        def _set_ttft(value: float) -> None:
            nonlocal ttft_ms
            if ttft_ms is None:
                ttft_ms = value

        async def finish(*, cancelled: bool) -> None:
            nonlocal stripped_citations
            final_content, sources, facts, stripped_citations = _finalize_grounding(
                "".join(content_parts),
                context_items,
                context_facts,
                tool_sources,
                retriever,
            )
            total_ms = (time.monotonic() - turn_started) * 1000.0
            message = ChatMessage(
                id=assistant_id,
                chat_id=chat_id,
                role="assistant",
                content=final_content,
                thinking="".join(thinking_parts),
                sources=sources,
                facts=facts,
                activity=activity,
                usage=ChatUsage(
                    prompt=usage["prompt"],
                    completion=usage["completion"],
                    reasoning=usage["reasoning"],
                    ttft_ms=ttft_ms or 0.0,
                    total_ms=total_ms,
                ),
                cancelled=cancelled,
                created=assistant_created,
            )
            self.index.finish_chat_message(message)
            retriever.save()
            self.logger.log(
                "chat_turn",
                chat_id=chat_id,
                message_id=assistant_id,
                model=model,
                retrieval={"sources": len(context_items), "facts": len(context_facts)},
                tools=tool_summaries,
                usage=message.usage.model_dump(mode="json"),
                ttft_ms=message.usage.ttft_ms,
                total_ms=total_ms,
                stripped_citations=stripped_citations,
                cancelled=cancelled,
            )
            await emit(ChatDoneEvent(chat_id=chat_id, message_id=assistant_id, message=message))
            if not cancelled:
                await self._maybe_title(chat_id, text, emit)

        try:
            context_items, context_facts = retriever.context_pack(text, now=wall_now)
            self.logger.log(
                "chat_retrieval",
                chat_id=chat_id,
                query=text,
                top=[{"id": item.short_id, "score": item.score} for item in context_items],
                facts=[fact.fact_id for fact in context_facts],
            )
            await emit(
                ChatRetrievalEvent(
                    chat_id=chat_id,
                    message_id=assistant_id,
                    sources=context_items,
                    facts=context_facts,
                )
            )
            messages = self._build_messages(
                chat_id,
                text,
                context_items,
                context_facts,
                current_user_id=user_message_id,
                assistant_id=assistant_id,
            )
            round_index = 0
            tool_count = 0
            while True:
                tools = TOOL_DEFINITIONS if tool_count < config.CHAT_MAX_TOOL_CALLS else None
                try:
                    outcome = await self._stream_round(
                        model,
                        messages,
                        tools,
                        emit,
                        chat_id=chat_id,
                        message_id=assistant_id,
                        content_parts=content_parts,
                        thinking_parts=thinking_parts,
                        usage=usage,
                        turn_started=turn_started,
                        ttft_ref=lambda value: _set_ttft(value),
                    )
                except _StreamFailure as exc:
                    if round_index == 0 and not exc.opened and model != config.CHAT_FALLBACK_MODEL:
                        model = config.CHAT_FALLBACK_MODEL
                        continue
                    await emit(
                        ChatErrorEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            code="llm",
                            detail=str(exc.error),
                        )
                    )
                    await finish(cancelled=False)
                    return
                except TimeoutError as exc:
                    await emit(
                        ChatErrorEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            code="timeout",
                            detail=str(exc),
                        )
                    )
                    await finish(cancelled=False)
                    return

                tool_calls = outcome.tool_calls
                if not tool_calls or tools is None:
                    await finish(cancelled=False)
                    return
                tool_messages: list[dict[str, Any]] = []
                assistant_tool_calls: list[dict[str, Any]] = []
                for call in tool_calls:
                    if tool_count >= config.CHAT_MAX_TOOL_CALLS:
                        break
                    step_index = len(activity)
                    running = ChatActivityStep(
                        kind=_tool_kind(call.get("name", "")),
                        label=_running_label(call),
                        status="running",
                    )
                    await emit(
                        ChatActivityEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            step_index=step_index,
                            step=running,
                        )
                    )
                    started = time.monotonic()
                    result_json, done_step = execute_tool(
                        str(call.get("name") or ""),
                        call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                        retriever,
                        self.index,
                        now=time.time(),
                    )
                    latency_ms = (time.monotonic() - started) * 1000.0
                    done_step.status = "done"
                    activity.append(done_step)
                    tool_summaries.append(
                        {
                            "name": call.get("name", ""),
                            "label": done_step.label,
                            "latency_ms": latency_ms,
                            "note_ids": done_step.note_ids,
                        }
                    )
                    _capture_tool_sources(result_json, done_step, tool_sources)
                    await emit(
                        ChatActivityEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            step_index=step_index,
                            step=done_step,
                        )
                    )
                    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                    assistant_tool_calls.append(
                        {
                            "id": call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": call.get("name", ""),
                                "arguments": json.dumps(arguments, separators=(",", ":"), ensure_ascii=False),
                            },
                        }
                    )
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "name": call.get("name", ""),
                            "content": result_json,
                        }
                    )
                    tool_count += 1
                if assistant_tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": assistant_tool_calls,
                        }
                    )
                    messages.extend(tool_messages)
                round_index += 1
        except asyncio.CancelledError:
            await finish(cancelled=True)
            return

    async def _stream_round(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        emit: EmitFn,
        *,
        chat_id: str,
        message_id: int,
        content_parts: list[str],
        thinking_parts: list[str],
        usage: dict[str, int],
        turn_started: float,
        ttft_ref: Callable[[float], None],
    ) -> _StreamOutcome:
        opened = False
        tool_calls: list[dict[str, Any]] = []
        try:
            async with asyncio.timeout(config.CHAT_TIMEOUT_S):
                async for event in self.llm.stream_chat(
                    model,
                    messages,
                    tools=tools,
                    reasoning=config.CHAT_REASONING,
                    timeout_s=config.CHAT_TIMEOUT_S,
                ):
                    opened = True
                    if event.kind == "reasoning":
                        if event.text:
                            ttft_ref((time.monotonic() - turn_started) * 1000.0)
                            thinking_parts.append(event.text)
                            await emit(ChatThinkingEvent(chat_id=chat_id, message_id=message_id, delta=event.text))
                    elif event.kind == "content":
                        if event.text:
                            ttft_ref((time.monotonic() - turn_started) * 1000.0)
                            content_parts.append(event.text)
                            await emit(ChatDeltaEvent(chat_id=chat_id, message_id=message_id, delta=event.text))
                    elif event.kind == "tool_call" and event.tool_call is not None:
                        tool_calls.append(event.tool_call)
                    elif event.kind == "usage" and event.usage is not None:
                        usage["prompt"] += int(event.usage.get("prompt") or 0)
                        usage["completion"] += int(event.usage.get("completion") or 0)
                        usage["reasoning"] += int(event.usage.get("reasoning") or 0)
                    elif event.kind == "done":
                        break
        except LLMError as exc:
            raise _StreamFailure(exc, opened=opened) from exc
        return _StreamOutcome(tool_calls=tool_calls)

    def _build_messages(
        self,
        chat_id: str,
        user_text: str,
        context_items: list[GroundingItem],
        context_facts: list[GroundedFact],
        *,
        current_user_id: int,
        assistant_id: int,
    ) -> list[dict[str, Any]]:
        history = [
            message
            for message in self.index.chat_messages(chat_id)
            if message.id not in {current_user_id, assistant_id}
        ]
        history = history[-config.CHAT_HISTORY_MAX_TURNS * 2 :]
        messages: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt()}]
        for message in history:
            if message.role == "assistant" and not message.content:
                continue
            messages.append({"role": message.role, "content": message.content})
        messages.append(
            {
                "role": "user",
                "content": f"<context>\n{_grounding_block(context_items, context_facts)}\n</context>\n\n{user_text}",
            }
        )
        return messages

    async def _maybe_title(self, chat_id: str, first_user_text: str, emit: EmitFn) -> None:
        chat = self.index.chat(chat_id)
        if chat is None or chat.title:
            return
        messages = self.index.chat_messages(chat_id)
        if sum(1 for message in messages if message.role == "user") != 1:
            return
        if sum(1 for message in messages if message.role == "assistant") != 1:
            return
        try:
            result = await self.llm.complete_json(
                config.SUMMARY_MODEL,
                "Return JSON with a concise title for this chat.",
                f"Write a title of at most {config.CHAT_TITLE_MAX_WORDS} words for this message: {first_user_text}",
                timeout_s=config.SUMMARY_TIMEOUT_S,
            )
            title = str(result.get("title") or "").strip()
        except Exception:
            title = ""
        if not title:
            title = first_user_text.strip()[:40]
        words = title.split()
        if len(words) > config.CHAT_TITLE_MAX_WORDS:
            title = " ".join(words[: config.CHAT_TITLE_MAX_WORDS])
        title = title[:120].strip()
        self.index.rename_chat(chat_id, title, now=time.time())
        updated = self.index.chat(chat_id)
        if updated is not None:
            await emit(ChatMetaEvent(chat=updated))


def _system_prompt() -> str:
    return (
        "You answer questions over the user's personal notes. Cite only [N#] and [F#] ids "
        "that appear in the supplied context block or tool results. Use tools when the "
        "context is insufficient. Be direct and write markdown."
    )


def _grounding_block(items: list[GroundingItem], facts: list[GroundedFact]) -> str:
    note_ids = {item.note_id: item.short_id for item in items}
    lines: list[str] = []
    for item in items:
        lines.append(
            f'[{item.short_id}] "{item.title}" '
            f"({item.source}, age {_age_label(item.age_days)}, score {item.score:.2f}) "
            f"— {item.gist}"
        )
    for fact in facts:
        src = note_ids.get(fact.src, fact.src)
        dst = note_ids.get(fact.dst, fact.dst)
        lines.append(f'[{fact.fact_id}] ({src}<->{dst}) "{fact.text}"')
    return "\n".join(lines) if lines else "none"


def _age_label(age_days: float) -> str:
    if age_days < 1.0:
        return "today"
    if age_days < 60.0:
        return f"{int(round(age_days))}d"
    return f"{int(round(age_days / 30.0))}mo"


def _tool_kind(name: str) -> str:
    if name == "read_note":
        return "read"
    if name == "expand_note":
        return "expand"
    return "search"


def _running_label(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "tool")
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if name == "search_notes":
        return f'Searching "{arguments.get("query", "")}"'
    if name == "read_note":
        return f'Reading {arguments.get("note", "")}'
    if name == "expand_note":
        return f'Expanding {arguments.get("note", "")}'
    return f"Running {name}"


def _capture_tool_sources(
    result_json: str,
    step: ChatActivityStep,
    target: dict[str, GroundingItem],
) -> None:
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        short_id = str(row.get("short_id") or "")
        if not short_id:
            continue
        note_id = step.note_ids[index] if index < len(step.note_ids) else short_id
        target[short_id] = GroundingItem(
            note_id=note_id,
            short_id=short_id,
            title=str(row.get("title") or ""),
            gist=str(row.get("gist") or ""),
            source=str(row.get("source") or "apple_notes"),
            age_days=float(row.get("age_days") or 0.0),
            score=float(row.get("score") or 0.0),
        )


def _finalize_grounding(
    content: str,
    context_items: list[GroundingItem],
    context_facts: list[GroundedFact],
    tool_sources: dict[str, GroundingItem],
    retriever: Any,
) -> tuple[str, list[GroundingItem], list[GroundedFact], int]:
    valid_ids = {item.short_id for item in context_items}
    valid_ids |= {fact.fact_id for fact in context_facts}
    valid_ids |= set(tool_sources)
    valid_ids |= _registry_ids(retriever)
    cited: list[str] = []
    stripped = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal stripped
        ref = match.group(0)[1:-1]
        if ref in valid_ids:
            cited.append(ref)
            return match.group(0)
        stripped += 1
        return ""

    stripped_content = _CITATION_RE.sub(replace, content)
    item_by_id = {item.short_id: item for item in context_items}
    item_by_id.update(tool_sources)
    fact_by_id = {fact.fact_id: fact for fact in context_facts}
    source_ids = [ref for ref in cited if ref.startswith("N") and ref in item_by_id]
    source_ids.extend(item.short_id for item in context_items if item.short_id not in source_ids)
    fact_ids = [ref for ref in cited if ref.startswith("F") and ref in fact_by_id]
    fact_ids.extend(fact.fact_id for fact in context_facts if fact.fact_id not in fact_ids)
    sources = [item_by_id[ref] for ref in _unique(source_ids)[: config.CHAT_CONTEXT_NOTES] if ref in item_by_id]
    facts = [fact_by_id[ref] for ref in _unique(fact_ids)[: config.CHAT_CONTEXT_FACTS] if ref in fact_by_id]
    return stripped_content, sources, facts, stripped


def _registry_ids(retriever: Any) -> set[str]:
    grounding = getattr(retriever, "_grounding", None)
    ids: set[str] = set()
    note_ids = getattr(grounding, "_note_short_ids", {})
    if isinstance(note_ids, dict):
        ids |= {str(value) for value in note_ids.values()}
    fact_ids = getattr(grounding, "_fact_short_ids", {})
    if isinstance(fact_ids, dict):
        ids |= {str(value) for value in fact_ids.values()}
    return ids


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
