"""Stage B — the streamed GLM-5.2 thinking turn loop (spec chat §1.3–1.4).

One ChatEngine per server; turns are per-chat single-flight with
CHAT_MAX_CONCURRENT_TURNS across chats. A turn:

1. persist the user message; create the empty assistant row (→ chat_ack)
2. Speculatively launch raw-message retrieval while the turn is prepared. At
   the first prompt boundary, inject it as PRE-RETRIEVED EVIDENCE if ready,
   wait only PRE_RETRIEVAL_MAX_WAIT_S otherwise, and fail open. The toggle-off
   path retains the legacy rewritten Stage-A flow (→ chat_retrieval)
3. stream_chat(CHAT_MODEL, reasoning=CHAT_REASONING, tools=TOOL_DEFINITIONS):
   reasoning deltas → chat_thinking, content deltas → chat_delta; on
   tool_call: emit chat_activity(running), execute, emit chat_activity(done),
   append tool message, re-open the stream — until no tool calls or
   CHAT_MAX_TOOL_CALLS (with an evidence-sufficiency nudge when nothing this
   turn cleared the retrieval floor, §14.4)
4. citation validation against this turn's read evidence (identifier
   validity, availability, fact provenance, temporal consistency): one
   constrained repair attempt, then unsupported-sentence stripping (stripped
   claims persist on the message), then an evidence-excerpt fallback; on
   acceptance the retriever commits its accepted-turn state and saves
   (→ chat_done)
5. auto-title after the first completed exchange (one flash complete_json;
   fallback: first 40 chars) (→ chat_meta)

Cancellation (chat_cancel) and mid-stream errors persist the partial content
exactly as streamed with interrupted=True — no validation, no excerpt, nothing
cited — and restore the retriever's pre-turn snapshot so failed turns never
advance the accumulator (→ chat_done with cancelled, or chat_error).
Fallback model only on stream-OPEN failure. Session-log record kinds:
chat_turn, chat_retrieval.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Coroutine, Mapping, Sequence
from uuid import uuid4

from .. import config
from ..dates import current_datetime_line, evidence_date, timestamp_date
from ..indexer.store import Index
from ..llm import OpenRouterClient
from ..ports import LLMError, SpanExtractorPort
from ..profile_context import user_profile_block
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
    ChatStrippedClaim,
    ChatThinkingEvent,
    ChatUsage,
    EvidencePackPayload,
)
from ..retrieval.types import EvidencePack, EvidencePassage
from ..retrieval.validate import EvidenceValidator
from .retrieval import ChatRetriever
from .tools import TOOL_DEFINITIONS, execute_tool, render_search_result

# async callback the server provides: emits one ws event dict/model to the client
EmitFn = Callable[[Any], Coroutine[Any, Any, None]]
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]

# GLM-5.2 leaks its native tool-call markup as plain text when it wants a
# tool on the forced no-tools round (live smoke 2026-07-08) — never render it
_TOOL_CALL_REMNANT_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL)


def _strip_tool_call_remnants(content: str) -> str:
    return _TOOL_CALL_REMNANT_RE.sub("", content).rstrip()


def _preserves_named_entities(original: str, rewritten: str) -> bool:
    ignored = {
        "and", "how", "i", "it", "my", "tell", "the", "that", "this",
        "what", "when", "where", "who", "why",
    }
    named = {
        token.casefold()
        for token in re.findall(r"\b[A-Z][\w'-]*\b", original)
        if token.casefold() not in ignored
    }
    lowered = rewritten.casefold()
    return all(re.search(rf"\b{re.escape(token)}\b", lowered) for token in named)


def make_tool_executor(
    retriever: ChatRetriever,
    *,
    now: Callable[[], float] = time.time,
) -> ToolExecutor:
    """Expose chat's bounded evidence-tool executor for other runtime agents."""

    async def run(name: str, arguments: dict[str, Any]) -> str:
        result_json, _step = await _execute_tool_async(
            name, arguments, retriever, now=now()
        )
        return result_json

    return run


async def _execute_tool_async(
    name: str,
    arguments: dict[str, Any],
    retriever: ChatRetriever,
    *,
    now: float,
) -> tuple[str, ChatActivityStep]:
    return await asyncio.to_thread(
        execute_tool,
        name,
        arguments,
        retriever,
        now=now,
    )


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


async def _speculative_pack(
    task: asyncio.Task[EvidencePack],
    *,
    max_wait_s: float | None,
) -> EvidencePack | None:
    """Read a speculative result without cancelling its worker on timeout."""
    try:
        if max_wait_s is None:
            return await asyncio.shield(task)
        return await asyncio.wait_for(
            asyncio.shield(task),
            timeout=max(0.0, float(max_wait_s)),
        )
    except TimeoutError:
        return None
    except Exception:
        return None


def _run_speculative_retrieval(
    retriever: Any,
    message: str,
    *,
    now: float,
) -> EvidencePack:
    """Use the production search_evidence path with a rollback snapshot."""
    search = getattr(retriever, "search_evidence", None)
    if callable(search):
        begin = getattr(retriever, "begin_turn", None)
        search_kwargs: dict[str, Any] = {}
        if callable(begin):
            search_kwargs["session_reset_override"] = begin()
        return search(
            message,
            k=config.CHAT_SEARCH_TOOL_K,
            temporal="any",
            now=now,
            **search_kwargs,
        )
    # Narrow compatibility seam for older tests/custom retrievers.
    return retriever.context_pack(
        message,
        now=now,
        conversation_context=(),
        rewritten_query=None,
    )


def _consume_speculative_failure(task: asyncio.Task[EvidencePack]) -> None:
    """Retrieve orphaned task exceptions after a prompt-boundary timeout."""
    if not task.cancelled():
        task.exception()


class ChatEngine:
    def __init__(
        self,
        index: Index,
        llm: OpenRouterClient,
        logger: Any,
        *,
        span_extractor: SpanExtractorPort | None = None,
        retriever_factory: Callable[[str], ChatRetriever] | None = None,
        validator_factory: Callable[[Any], EvidenceValidator] | None = None,
    ) -> None:
        self.index = index
        self.llm = llm
        self.logger = logger
        self._span_extractor = span_extractor
        self._retriever_factory = retriever_factory
        self._validator_factory = validator_factory
        self._lock = asyncio.Lock()
        self._active: dict[str, asyncio.Task[Any]] = {}
        self._active_count = 0

    async def send(
        self,
        chat_id: str,
        text: str,
        emit: EmitFn,
        *,
        seed_note_ids: Sequence[str] = (),
    ) -> None:
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
                seed_note_ids=tuple(seed_note_ids),
            )
        finally:
            await self._release(reservation.chat_id)

    def seed_chat(self, chat_id: str, note_ids: Sequence[str]) -> None:
        """Persist graph-handoff seeds (POST /chats seed_note_ids) so the
        chat's first turn is biased toward those notes; unknown ids are
        ignored inside the retriever."""
        if self._retriever_factory is None or not note_ids:
            return
        retriever = self._retriever_factory(chat_id)
        seed = getattr(retriever, "seed_notes", None)
        if not callable(seed):
            return
        seed(note_ids)
        retriever.save()

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
                restore_pre_turn=True,
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
        restore_pre_turn: bool = False,
        seed_note_ids: Sequence[str] = (),
    ) -> None:
        turn_started = time.monotonic()
        wall_now = time.time()

        def make_retriever() -> Any:
            if self._retriever_factory is not None:
                value = self._retriever_factory(chat_id)
            else:
                # Test seams may replace ChatRetriever. Production wiring
                # supplies a factory bound to the immutable knowledge snapshot.
                value = ChatRetriever(self.index, chat_id)  # type: ignore[call-arg]
            if restore_pre_turn:
                restore = getattr(value, "restore_pre_turn", None)
                if callable(restore):
                    restore()
            if seed_note_ids:
                seed = getattr(value, "seed_notes", None)
                if callable(seed):
                    seed(seed_note_ids)
            return value

        retriever: Any | None = None
        speculative_retriever: Any | None = None
        speculative_task: asyncio.Task[EvidencePack] | None = None
        if config.CHAT_PRE_RETRIEVAL:
            # Start before message persistence/ack so retrieval overlaps all
            # remaining turn preparation, not merely the prompt-boundary wait.
            retriever = make_retriever()
            speculative_retriever = retriever
            speculative_task = asyncio.create_task(
                asyncio.to_thread(
                    _run_speculative_retrieval,
                    retriever,
                    text,
                    now=wall_now,
                )
            )
            speculative_task.add_done_callback(_consume_speculative_failure)
            await asyncio.sleep(0)  # schedule the worker before synchronous turn setup
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

        if retriever is None:
            retriever = make_retriever()
        context_pack: EvidencePack | None = None
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        activity: list[ChatActivityStep] = []
        usage = {"prompt": 0, "completion": 0, "reasoning": 0}
        ttft_ms: float | None = None
        tool_summaries: list[dict[str, Any]] = []
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

        def persist_turn(*, cancelled: bool, interrupted: bool = False) -> ChatMessage:
            nonlocal persisted_message
            if persisted_message is not None:
                return persisted_message
            interrupted = interrupted or cancelled
            raw_content = _strip_tool_call_remnants("".join(content_parts))
            turn_pack_fn = getattr(retriever, "turn_pack", None)
            audit_pack = turn_pack_fn() if callable(turn_pack_fn) else context_pack
            if interrupted:
                # H7: persist the partial exactly as streamed — no validation,
                # no excerpt substitution, nothing cited.
                final_content = raw_content
                cited_ids: tuple[str, ...] = ()
                stripped_claims: tuple[ChatStrippedClaim, ...] = ()
            else:
                final_content, cited_ids, stripped_claims = _finalize_evidence(
                    raw_content,
                    retriever,
                    self._validator_factory,
                    audit_pack,
                )
            retrieved_ids = list(audit_pack.trace.selected_ids) if audit_pack is not None else []
            consulted_ids = list(audit_pack.trace.read_ids) if audit_pack is not None else []
            total_ms = (time.monotonic() - turn_started) * 1000.0
            message = ChatMessage(
                id=assistant_id,
                chat_id=chat_id,
                role="assistant",
                content=final_content,
                thinking="".join(thinking_parts),
                evidence=EvidencePackPayload.from_domain(audit_pack) if audit_pack is not None else None,
                retrieved_ids=retrieved_ids,
                consulted_ids=consulted_ids,
                cited_ids=list(cited_ids),
                original_query=audit_pack.trace.original_query if audit_pack is not None else text,
                rewritten_query=audit_pack.trace.rewritten_query if audit_pack is not None else text,
                retrieval_decision=audit_pack.decision.reason if audit_pack is not None else "missing",
                retrieval_confidence=audit_pack.decision.confidence if audit_pack is not None else 0.0,
                active_build_id=audit_pack.trace.active_build_id if audit_pack is not None else "",
                activity=activity,
                usage=ChatUsage(
                    prompt=usage["prompt"],
                    completion=usage["completion"],
                    reasoning=usage["reasoning"],
                    ttft_ms=ttft_ms or 0.0,
                    total_ms=total_ms,
                ),
                cancelled=cancelled,
                interrupted=interrupted,
                stripped_claims=list(stripped_claims),
                created=assistant_created,
            )
            self.index.finish_chat_message(message)
            if interrupted:
                # H7: failed turns never advance the accumulator — roll the
                # retriever back to its pre-turn snapshot before saving.
                discard = getattr(retriever, "discard_turn", None)
                if callable(discard):
                    discard()
            else:
                commit = getattr(retriever, "commit_turn", None)
                if callable(commit):
                    commit()
            retriever.save()
            self.logger.log(
                "chat_turn",
                chat_id=chat_id,
                message_id=assistant_id,
                model=model,
                retrieval={
                    "passages": len(context_pack.passages) if context_pack is not None else 0,
                    "facts": len(context_pack.facts) if context_pack is not None else 0,
                    "decision": context_pack.decision.reason if context_pack is not None else "missing",
                    "retrieved_ids": retrieved_ids,
                    "consulted_ids": consulted_ids,
                    "cited_ids": list(cited_ids),
                },
                tools=tool_summaries,
                usage=message.usage.model_dump(mode="json"),
                ttft_ms=message.usage.ttft_ms,
                total_ms=total_ms,
                stripped_citations=len(stripped_claims),
                stripped_claims=[claim.model_dump(mode="json") for claim in stripped_claims],
                cancelled=cancelled,
                interrupted=interrupted,
            )
            persisted_message = message
            return message

        async def finish_done(*, cancelled: bool) -> None:
            if not cancelled:
                await self._repair_once(content_parts, retriever, context_pack)
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
                # H7: errored turns persist the raw partial, uncited.
                persist_turn(cancelled=False, interrupted=True)
            except Exception as exc:
                log_turn_error(exc)
            await emit(event)

        try:
            history_for_rewrite = tuple(
                message.content
                for message in self.index.chat_messages(chat_id)
                if message.role == "user" and message.id != user_message_id
            )[-2:]
            pre_retrieved = speculative_task is not None
            if speculative_task is not None:
                context_pack = await _speculative_pack(
                    speculative_task,
                    max_wait_s=config.PRE_RETRIEVAL_MAX_WAIT_S,
                )
                if context_pack is None:
                    # Keep the slow/erroring worker isolated: it may finish in
                    # its thread, but an unused pack must not affect this turn's
                    # accepted-state commit or failed-turn rollback.
                    retriever = make_retriever()
            else:
                rewritten_query = await self._rewrite_query_for_retrieval(
                    retriever,
                    text,
                    history_for_rewrite,
                )
                # Toggle-off is the legacy Stage-A path, byte-for-byte at the
                # prompt boundary, including dependent-query rewriting.
                context_pack = await asyncio.to_thread(
                    retriever.context_pack,
                    text,
                    now=wall_now,
                    conversation_context=history_for_rewrite,
                    rewritten_query=rewritten_query,
                )
            if context_pack is not None:
                self.logger.log(
                    "chat_retrieval",
                    chat_id=chat_id,
                    original_query=context_pack.trace.original_query,
                    rewritten_query=context_pack.trace.rewritten_query,
                    decision=context_pack.decision.reason,
                    confidence=context_pack.decision.confidence,
                    selected_ids=list(context_pack.trace.selected_ids),
                    active_build_id=context_pack.trace.active_build_id,
                )
                await emit(
                    ChatRetrievalEvent(
                        chat_id=chat_id,
                        message_id=assistant_id,
                        evidence=EvidencePackPayload.from_domain(context_pack),
                        retrieved_ids=list(context_pack.trace.selected_ids),
                        consulted_ids=list(context_pack.trace.read_ids),
                    )
                )
            messages = self._build_messages(
                chat_id,
                text,
                context_pack,
                retriever=retriever,
                current_user_id=user_message_id,
                assistant_id=assistant_id,
                pre_retrieved=pre_retrieved,
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
                    notice = (
                        "(Tool budget exhausted — tools are no longer "
                        "available. Answer now using only the "
                        "information you already have.)"
                    )
                    # M11/§14.4: deterministic evidence-sufficiency check
                    # before the final no-tools round — if nothing read this
                    # turn cleared the retrieval floor, tell the model to
                    # state the gap instead of synthesizing.
                    sufficient_fn = getattr(retriever, "evidence_sufficient", None)
                    if callable(sufficient_fn) and not sufficient_fn():
                        notice += (
                            " Evidence check: nothing retrieved this turn "
                            "cleared the retrieval confidence floor for this "
                            "question. State plainly what information is "
                            "missing instead of synthesizing an answer, and "
                            "cite only evidence you actually have."
                        )
                    messages.append({"role": "user", "content": notice})
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
                        tool_name=str(call.get("name") or ""),
                        arguments=(
                            call.get("arguments")
                            if isinstance(call.get("arguments"), dict)
                            else {}
                        ),
                    )
                    await emit(
                        ChatActivityEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            step_index=step_index,
                            step=running,
                        )
                    )
                    arguments = (
                        call.get("arguments")
                        if isinstance(call.get("arguments"), dict)
                        else {}
                    )
                    started = time.monotonic()
                    cached_pack: EvidencePack | None = None
                    if _matches_speculative_query(call, text):
                        cached_pack = context_pack
                        if cached_pack is None and speculative_task is not None:
                            cached_pack = await _speculative_pack(
                                speculative_task,
                                max_wait_s=None,
                            )
                        if cached_pack is not None and speculative_retriever is not None:
                            # If the first prompt timed out, use the retriever
                            # whose registry owns the cached E#/F# identities.
                            if tool_count == 0:
                                retriever = speculative_retriever
                            context_pack = cached_pack
                    if cached_pack is not None:
                        result_json, done_step = render_search_result(
                            cached_pack,
                            str(arguments.get("query") or "").strip(),
                        )
                    else:
                        # Tool execution is retrieval work — keep it off the loop.
                        result_json, done_step = await _execute_tool_async(
                            str(call.get("name") or ""),
                            arguments,
                            retriever,
                            now=time.time(),
                        )
                    latency_ms = (time.monotonic() - started) * 1000.0
                    done_step.status = "done"
                    done_step.tool_name = str(call.get("name") or "")
                    done_step.arguments = (
                        call.get("arguments")
                        if isinstance(call.get("arguments"), dict)
                        else {}
                    )
                    done_step.evidence_ids = list(_tool_result_evidence_ids(result_json))
                    done_step.latency_ms = latency_ms
                    activity.append(done_step)
                    tool_summaries.append(
                        {
                            "name": call.get("name", ""),
                            "label": done_step.label,
                            "latency_ms": latency_ms,
                            "note_ids": done_step.note_ids,
                        }
                    )
                    await emit(
                        ChatActivityEvent(
                            chat_id=chat_id,
                            message_id=assistant_id,
                            step_index=step_index,
                            step=done_step,
                        )
                    )
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

    async def _repair_once(
        self,
        content_parts: list[str],
        retriever: Any,
        context_pack: EvidencePack | None,
    ) -> None:
        """Run at most one evidence-constrained repair before final persistence."""

        if self._validator_factory is None:
            return
        registry = getattr(retriever, "registry", None)
        if registry is None:
            return
        draft = _strip_tool_call_remnants("".join(content_parts))
        validator = self._validator_factory(registry)
        turn_pack_fn = getattr(retriever, "turn_pack", None)
        pack = turn_pack_fn() if callable(turn_pack_fn) else context_pack
        supplied_ids = pack.trace.read_ids if pack is not None else ()
        initial = validator.validate(
            draft,
            mode="chat",
            supplied_ids=supplied_ids,
            available_ids=supplied_ids,
        )
        if initial.accepted or not initial.repair_required:
            return
        try:
            result = await self.llm.complete_json(
                config.CHAT_MODEL,
                (
                    "Return JSON with one key, answer. Repair the draft using only the supplied "
                    "evidence. Remove unsupported claims. Every factual sentence must cite E# or F#."
                ),
                f"<evidence>\n{_evidence_block(pack) if pack is not None else 'none'}\n</evidence>\n\n<draft>\n{draft}\n</draft>",
                timeout_s=config.SUMMARY_TIMEOUT_S,
            )
        except Exception:
            return
        repaired = str(result.get("answer") or "").strip()
        if not repaired:
            return
        checked = validator.validate(
            repaired,
            mode="chat",
            supplied_ids=supplied_ids,
            available_ids=supplied_ids,
        )
        if checked.accepted:
            content_parts[:] = [checked.accepted_output]

    async def _rewrite_query_for_retrieval(
        self,
        retriever: Any,
        message: str,
        context: tuple[str, ...],
    ) -> str | None:
        dependent_fn = getattr(retriever, "is_dependent_query", None)
        if not callable(dependent_fn) or not dependent_fn(message, context):
            return None
        fallback_fn = getattr(retriever, "fallback_rewrite", None)
        fallback = (
            str(fallback_fn(message, context))
            if callable(fallback_fn)
            else "\n".join((*context, message))
        )
        try:
            result = await self.llm.complete_json(
                config.SUMMARY_MODEL,
                (
                    "Return JSON with one key, query. Rewrite the final user message into one "
                    "self-contained retrieval query using the preceding user turns. Preserve "
                    "intent and every named entity. Do not answer the question."
                ),
                json.dumps(
                    {"previous_user_turns": list(context), "original": message},
                    ensure_ascii=False,
                ),
                timeout_s=config.SUMMARY_TIMEOUT_S,
            )
        except Exception:
            return fallback
        rewritten = str(result.get("query") or "").strip()
        source_text = "\n".join((*context, message))
        if not rewritten or not _preserves_named_entities(source_text, rewritten):
            return fallback
        return rewritten

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
        context_pack: EvidencePack | None,
        *,
        retriever: Any = None,
        current_user_id: int,
        assistant_id: int,
        pre_retrieved: bool = False,
    ) -> list[dict[str, Any]]:
        history = [
            message
            for message in self.index.chat_messages(chat_id)
            if message.id not in {current_user_id, assistant_id}
        ]
        history = history[-config.CHAT_HISTORY_MAX_TURNS * 2 :]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(user_profile_block(self.index))}
        ]
        for message in history:
            if message.role == "assistant" and not message.content:
                continue
            messages.append({"role": message.role, "content": message.content})
        content = user_text
        if context_pack is not None:
            # M6: expose bounded non-selected fact-support passages — they
            # become read/citable and render as compact quotes under facts.
            expose = getattr(retriever, "expose_fact_support", None)
            support = expose(context_pack) if callable(expose) else {}
            evidence = _evidence_block(context_pack, support)
            if pre_retrieved:
                content = (
                    "<pre_retrieved_evidence>\n"
                    "Pre-retrieved evidence for the user's message:\n"
                    f"{evidence}\n"
                    "</pre_retrieved_evidence>\n\n"
                    f"{user_text}"
                )
            else:
                content = f"<evidence>\n{evidence}\n</evidence>\n\n{user_text}"
        messages.append({"role": "user", "content": content})
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


def _system_prompt(profile_block: str = "") -> str:
    prompt = current_datetime_line(time.time()) + "\n" + (
        "You answer questions over the user's personal knowledge base. Every factual "
        "sentence must cite an exact [E#] passage, a supported [F#] fact, or both. [N#] "
        "is navigation metadata and cannot ground a factual claim. Cite only identifiers "
        "in supplied evidence or tool results. If retrieval abstained or evidence is "
        "insufficient, use tools or state the gap. Evidence under a LOW-CONFIDENCE "
        "header (or marked low_confidence) did not clear retrieval's confidence floor: "
        "treat it as a hint, verify it before asserting, and say so explicitly if you "
        "rely on it. When only LOW-CONFIDENCE evidence exists, answer with it and label "
        "the uncertainty rather than refusing. Be direct and write markdown."
    )
    if not profile_block:
        return prompt
    return (
        f"{prompt}\n\n"
        "The user profile below is background context for interpreting names, relationships, "
        "interests, and likely referents. It is not citable evidence. Ground factual answers "
        "in the supplied [E#] passages and [F#] facts.\n\n"
        f"{profile_block}"
    )


def _evidence_block(
    pack: EvidencePack,
    support: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    fallback = bool(getattr(pack.decision, "fallback", False))
    if not pack.decision.accepted and not fallback:
        # §14.2 abstention message — only for truly empty packs (§0.1).
        return (
            "Knowledge-base retrieval abstained: "
            f"{pack.decision.reason} (confidence {pack.decision.confidence:.2f})."
        )
    lines: list[str] = []
    accepted_passages = (
        () if fallback else tuple(row for row in pack.passages if not row.low_confidence)
    )
    low_confidence_passages = tuple(
        row for row in pack.passages if fallback or row.low_confidence
    )

    notes = {row.note_id: row.note_ref for row in pack.notes}

    def append_passage(item: EvidencePassage) -> None:
        lines.append(
            f"[{item.evidence_id}] ({evidence_date(item)}) {item.text}\n"
            f'  [{notes.get(item.note_id, item.note_id)}] "{item.title}" '
            f"({item.source}, chars {item.char_start}-{item.char_end})"
        )

    for item in accepted_passages:
        append_passage(item)
    if low_confidence_passages:
        lines.append(
            "LOW-CONFIDENCE CONTEXT — retrieval did not clear its confidence "
            "floor; treat these passages as hints, verify before asserting, "
            "and say so if you rely on them."
        )
        for item in low_confidence_passages:
            append_passage(item)
    if fallback:
        # Fallback packs are passages only — facts/paths sections omitted.
        return "\n".join(lines) if lines else "none"
    rendered_refs = {row.evidence_id for row in pack.passages}
    for fact in pack.facts:
        valid = (
            f"; observed={timestamp_date(fact.observed_at)}; "
            f"valid={timestamp_date(fact.valid_from)}"
            f"..{timestamp_date(fact.valid_to, unknown='open')}"
        )
        # M6: advertise only support refs the model can actually cite —
        # rendered passages plus the exposed (read) support snippet.
        exposed = (support or {}).get(fact.fact_ref)
        citable = [
            ref
            for ref in fact.source_evidence_ids
            if ref in rendered_refs or (exposed is not None and ref == exposed[0])
        ]
        sources = " ".join(f"[{ref}]" for ref in citable)
        lines.append(
            f"[{fact.fact_ref}] {fact.subject} — {fact.predicate} → {fact.object}; "
            f"status={fact.status}{valid}; supported by {sources}"
        )
        if exposed is not None:
            ref, title, date, snippet = exposed
            lines.append(f'  [{ref}] ({date}) "{title}": "{snippet}"')
    for path in pack.paths:
        lines.append(f"[{path.path_id}] {path.rendered}")
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
    if name == "expand_evidence":
        return "expand"
    return "search"


def _matches_speculative_query(call: Mapping[str, Any], user_text: str) -> bool:
    if str(call.get("name") or "") != "search_evidence":
        return False
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    query = str(arguments.get("query") or "").strip()
    raw = str(user_text).strip()
    return bool(query and raw) and query.casefold() == raw.casefold()


def _running_label(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "tool")
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if name == "search_evidence":
        return f'Searching "{arguments.get("query", "")}"'
    if name == "read_note":
        return f'Reading {arguments.get("note", "")}'
    if name == "expand_evidence":
        return f'Expanding {arguments.get("evidence", "")}'
    return f"Running {name}"


def _tool_result_evidence_ids(result_json: str) -> tuple[str, ...]:
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return ()
    refs = re.findall(r'"(?:N|E|F|P)\d+"', json.dumps(payload, sort_keys=True))
    return tuple(dict.fromkeys(ref.strip('"') for ref in refs))


def _finalize_evidence(
    content: str,
    retriever: Any,
    validator_factory: Callable[[Any], EvidenceValidator] | None,
    pack: EvidencePack | None,
) -> tuple[str, tuple[str, ...], tuple[ChatStrippedClaim, ...]]:
    """Validate whole claims; unsupported citations never leave a bare claim.

    Returns (final content, cited refs, stripped claims). Runtime retrievers
    always own an EvidenceRegistry — there is no unvalidated path (H7)."""

    registry = retriever.registry
    validator = (
        validator_factory(registry)
        if validator_factory is not None
        else EvidenceValidator(registry)
    )
    supplied = pack.trace.read_ids if pack is not None else ()
    result = validator.validate(
        content,
        mode="chat",
        supplied_ids=supplied,
        available_ids=supplied,
    )
    if result.accepted:
        return result.accepted_output, result.cited_ids, ()

    claims = tuple(
        ChatStrippedClaim(sentence=sentence, reason=reason)
        for sentence, reason in result.unsupported_details
    )
    repaired = _remove_unsupported_sentences(
        content,
        (*result.unsupported_sentences, *result.invalid_citations),
    )
    if repaired.strip() and repaired != content:
        repaired_result = validator.validate(
            repaired,
            mode="chat",
            supplied_ids=supplied,
            available_ids=supplied,
        )
        if repaired_result.accepted:
            return repaired_result.accepted_output, repaired_result.cited_ids, claims

    excerpt, excerpt_ids = _evidence_excerpt(pack, registry)
    if not claims:
        claims = (
            ChatStrippedClaim(
                sentence="", reason=result.degraded_reason or "invalid_citation"
            ),
        )
    return excerpt, excerpt_ids, claims


def _remove_unsupported_sentences(content: str, unsupported: tuple[str, ...]) -> str:
    repaired = content
    for rendered in unsupported:
        if rendered in repaired:
            repaired = repaired.replace(rendered, "")
        elif rendered.startswith(("N", "E", "F", "P")):
            # Invalid bare references are removed only as part of their sentence
            # by the validator; never strip a marker and preserve its claim.
            continue
    return "\n".join(line.rstrip() for line in repaired.splitlines() if line.strip()).strip()


def _evidence_excerpt(pack: EvidencePack | None, registry: Any) -> tuple[str, tuple[str, ...]]:
    passages = pack.passages if pack is not None else ()
    if not passages:
        return "I couldn't find enough validated knowledge-base evidence to answer that.", ()
    lines = [
        "I couldn't validate a synthesized answer, but these source excerpts were retrieved:",
    ]
    for passage in passages[:3]:
        registry.mark_cited(passage.evidence_id)
        lines.append(f"> {passage.text}\n> [{passage.evidence_id}]")
    return "\n\n".join(lines), tuple(row.evidence_id for row in passages[:3])
