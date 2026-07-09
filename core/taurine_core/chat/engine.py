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
# strip variant consumes one adjacent leading space so removals don't leave
# double spaces — a global space-collapse would mangle code indentation and
# markdown hard breaks (re-review finding)
_CITATION_STRIP_RE = re.compile(r" ?\[(N|F)\d+\]")
# GLM-5.2 leaks its native tool-call markup as plain text when it wants a
# tool on the forced no-tools round (live smoke 2026-07-08) — never render it
_TOOL_CALL_REMNANT_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL)


def _strip_tool_call_remnants(content: str) -> str:
    return _TOOL_CALL_REMNANT_RE.sub("", content).rstrip()


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
        persisted_message: ChatMessage | None = None
        success_finish_task: asyncio.Task[None] | None = None

        def _set_ttft(value: float) -> None:
            nonlocal ttft_ms
            if ttft_ms is None:
                ttft_ms = value

        def log_turn_error(exc: BaseException) -> None:
            try:
                self.logger.log(
                    "chat_error",
                    chat_id=chat_id,
                    message_id=assistant_id,
                    code="llm",
                    detail=str(exc),
                )
            except Exception:
                pass

        def persist_turn(*, cancelled: bool) -> ChatMessage:
            nonlocal stripped_citations, persisted_message
            if persisted_message is not None:
                return persisted_message
            final_content, sources, facts, stripped_citations = _finalize_grounding(
                _strip_tool_call_remnants("".join(content_parts)),
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
            persisted_message = message
            return message

        async def finish_done(*, cancelled: bool) -> None:
            message = persist_turn(cancelled=cancelled)
            await emit(ChatDoneEvent(chat_id=chat_id, message_id=assistant_id, message=message))
            if not cancelled:
                # post-done tail: chat_done is already out, so nothing here may
                # escape to the catch-all (chat_error after chat_done would
                # break §2.1 terminal exclusivity)
                try:
                    await self._maybe_title(chat_id, text, emit)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    try:
                        self.logger.log("chat_turn", chat_id=chat_id, title_error=str(exc))
                    except Exception:
                        pass

        async def finish_success() -> None:
            nonlocal success_finish_task
            success_finish_task = asyncio.create_task(finish_done(cancelled=False))
            try:
                await asyncio.shield(success_finish_task)
            except asyncio.CancelledError:
                await asyncio.shield(success_finish_task)
                raise

        async def finish_error(event: ChatErrorEvent) -> None:
            try:
                persist_turn(cancelled=False)
            except Exception as exc:
                log_turn_error(exc)
            await emit(event)

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
            budget_notice_sent = False
            while True:
                tools = TOOL_DEFINITIONS if tool_count < config.CHAT_MAX_TOOL_CALLS else None
                if tools is None and not budget_notice_sent:
                    # GLM keeps trying to call tools in plain text when they
                    # vanish silently (live smoke 2026-07-08) — tell it why
                    budget_notice_sent = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "(Tool budget exhausted — tools are no longer "
                                "available. Answer now using only the "
                                "information you already have.)"
                            ),
                        }
                    )
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
                    await finish_error(
                        ChatErrorEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            code="llm",
                            detail=str(exc.error),
                        )
                    )
                    return
                except TimeoutError as exc:
                    await finish_error(
                        ChatErrorEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            code="timeout",
                            detail=str(exc),
                        )
                    )
                    return

                tool_calls = outcome.tool_calls
                if not tool_calls or tools is None:
                    await finish_success()
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
                    # the model often narrates ("Let me pull up…") before a
                    # tool call; separate it from the next round's answer so
                    # they don't run together ("…detail.Here's…"). Emit the
                    # break too, so persisted content matches what streamed.
                    joined = "".join(content_parts)
                    if joined and not joined[-1].isspace():
                        content_parts.append("\n\n")
                        await emit(
                            ChatDeltaEvent(
                                chat_id=chat_id,
                                message_id=assistant_id,
                                delta="\n\n",
                            )
                        )
                round_index += 1
        except asyncio.CancelledError:
            if persisted_message is None:
                if success_finish_task is not None:
                    await asyncio.shield(success_finish_task)
                else:
                    await finish_done(cancelled=True)
            raise
        except Exception as exc:
            log_turn_error(exc)
            await finish_error(
                ChatErrorEvent(
                    chat_id=chat_id,
                    message_id=assistant_id,
                    code="llm",
                    detail=str(exc),
                )
            )
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
    item_by_id = {item.short_id: item for item in context_items}
    item_by_id.update(tool_sources)
    fact_by_id = {fact.fact_id: fact for fact in context_facts}
    note_registry, fact_registry = _registry_maps(retriever)
    cited_refs = [match.group(0)[1:-1] for match in _CITATION_RE.finditer(content)]

    for ref in cited_refs:
        if ref.startswith("N") and ref not in item_by_id and ref in note_registry:
            item = _materialize_registry_note(ref, retriever)
            if item is not None:
                item_by_id[ref] = item
        elif ref.startswith("F") and ref not in fact_by_id:
            key = fact_registry.get(ref)
            if key is None:
                continue
            fact = _materialize_registry_fact(ref, key, retriever)
            if fact is not None:
                fact_by_id[ref] = fact

    valid_ids = set(item_by_id) | set(fact_by_id)
    cited: list[str] = []
    stripped = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal stripped
        ref = match.group(0).lstrip()[1:-1]
        if ref in valid_ids:
            cited.append(ref)
            return match.group(0)
        stripped += 1
        return ""

    stripped_content = _CITATION_STRIP_RE.sub(replace, content)
    source_ids = [ref for ref in cited if ref.startswith("N") and ref in item_by_id]
    source_ids.extend(item.short_id for item in context_items if item.short_id not in source_ids)
    fact_ids = [ref for ref in cited if ref.startswith("F") and ref in fact_by_id]
    fact_ids.extend(fact.fact_id for fact in context_facts if fact.fact_id not in fact_ids)
    sources = [item_by_id[ref] for ref in _unique(source_ids)[: config.CHAT_CONTEXT_NOTES] if ref in item_by_id]
    facts = [fact_by_id[ref] for ref in _unique(fact_ids)[: config.CHAT_CONTEXT_FACTS] if ref in fact_by_id]
    return stripped_content, sources, facts, stripped


def _registry_maps(retriever: Any) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    grounding = getattr(retriever, "_grounding", None)
    note_by_short: dict[str, str] = {}
    fact_by_short: dict[str, tuple[str, str]] = {}
    note_ids_fn = getattr(grounding, "note_short_ids", None)
    note_ids = note_ids_fn() if callable(note_ids_fn) else {}
    if isinstance(note_ids, dict):
        note_by_short = {str(short_id): str(note_id) for note_id, short_id in note_ids.items()}
    fact_ids_fn = getattr(grounding, "fact_short_ids", None)
    fact_ids = fact_ids_fn() if callable(fact_ids_fn) else {}
    if isinstance(fact_ids, dict):
        for key, short_id in fact_ids.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            fact_by_short[str(short_id)] = (str(key[0]), str(key[1]))
    return note_by_short, fact_by_short


def _materialize_registry_note(ref: str, retriever: Any) -> GroundingItem | None:
    resolve = getattr(retriever, "resolve", None)
    if not callable(resolve):
        return None
    note_id = resolve(ref)
    if note_id is None:
        return None
    index = getattr(retriever, "index", None)
    note_fn = getattr(index, "note", None)
    if not callable(note_fn):
        return None
    note = note_fn(note_id)
    if note is None:
        return None
    age_days = (time.time() - float(getattr(note, "modified", time.time()))) / 86_400.0
    return GroundingItem(
        note_id=str(getattr(note, "note_id", note_id)),
        short_id=ref,
        title=str(getattr(note, "title", "")),
        gist=_display_line_with_entities(
            index,
            str(getattr(note, "note_id", note_id)),
            str(getattr(note, "gist", "")),
        ),
        source=str(getattr(note, "source", "apple_notes")),
        age_days=age_days,
        score=0.0,
    )


def _materialize_registry_fact(
    ref: str,
    key: tuple[str, str],
    retriever: Any,
) -> GroundedFact | None:
    index = getattr(retriever, "index", None)
    neighbor_edges = getattr(index, "neighbor_edges", None)
    if not callable(neighbor_edges):
        return None
    src, dst = key
    for other_id, _weight, fact in neighbor_edges(src):
        if tuple(sorted((src, str(other_id)))) != tuple(sorted(key)):
            continue
        fact_text = str(fact or "").strip()
        if not fact_text:
            return None
        return GroundedFact(fact_id=ref, src=src, dst=dst, text=fact_text)
    return None


def _display_line_with_entities(index: Any, note_id: str, display_line: str) -> str:
    entity_surfaces_for = getattr(index, "entity_surfaces_for", None)
    if callable(entity_surfaces_for):
        surfaces = [
            str(surface).strip()
            for surface in entity_surfaces_for(note_id, limit=3)
            if str(surface).strip()
        ]
        if surfaces:
            return f"{display_line} [{', '.join(surfaces)}]"
        return display_line
    enrichment_for = getattr(index, "enrichment_for", None)
    if not callable(enrichment_for):
        return display_line
    enrichment = enrichment_for(note_id)
    entities = getattr(enrichment, "entities", []) if enrichment is not None else []
    surfaces = [
        str(getattr(entity, "surface", "")).strip()
        for entity in entities
        if str(getattr(entity, "surface", "")).strip()
    ][:3]
    if not surfaces:
        return display_line
    return f"{display_line} [{', '.join(surfaces)}]"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
