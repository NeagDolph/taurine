from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from taurine_core import config
from taurine_core.indexer import store
from taurine_core.ports import IndexPort
from taurine_core.schemas import (
    Actionable,
    ActionableEvent,
    AmbientEvent,
    AmbientItem,
    Chunk,
    Expansion,
    Hit,
    MeetingSummary,
    RetrievalEvent,
    RetrievalResult,
    SummaryEvent,
    TranscriptEvent,
    TriggerEvent,
    TriggerMatch,
)
from taurine_core.session_log import NullSessionLogger, log_grounding, log_summary

from .ambient import AmbientRetriever
from .exact import ExactCollector
from .grounding import GroundingUpdate


Event = TranscriptEvent | AmbientEvent | SummaryEvent | ActionableEvent | RetrievalEvent | TriggerEvent
EventSink = Callable[[Event], Awaitable[None] | None]


class Clock(Protocol):
    def now(self) -> float:
        ...


class TriggerLike(Protocol):
    def offer(
        self,
        update: GroundingUpdate,
        window_emb: np.ndarray | None,
        new_chars: int,
    ) -> tuple[bool, str]:
        ...

    def note_summary(self, update: GroundingUpdate, *, no_update: bool) -> None:
        ...

    def note_rejected(self) -> None:
        ...


class SummarizerLike(Protocol):
    async def summarize(
        self,
        state: MeetingSummary | None,
        transcript_delta: str,
        update: GroundingUpdate,
        open_actionables: list[Actionable],
    ) -> Any:
        ...


class LoggerLike(Protocol):
    def log(self, kind: str, **payload: Any) -> None:
        ...


class _SystemClock:
    def now(self) -> float:
        return time.monotonic()


class LiveSession:
    def __init__(
        self,
        *,
        index: IndexPort,
        exact: ExactCollector,
        ambient: AmbientRetriever,
        trigger: TriggerLike,
        summarizer: SummarizerLike,
        emit: EventSink,
        clock: Clock | None = None,
        logger: LoggerLike | None = None,
    ) -> None:
        self._index = index
        self._exact = exact
        self._ambient = ambient
        self._trigger = trigger
        self._summarizer = summarizer
        self._emit = emit
        self._clock = clock or _SystemClock()
        self._logger = logger or NullSessionLogger()
        self._transcript: list[Chunk] = []
        self._partials: dict[int, str] = {}
        self._c1_segment_seen: dict[int, set[str]] = {}
        self._next_seq = 1
        self.last_grounding: GroundingUpdate | None = None
        self._summary: MeetingSummary | None = None
        self._actionables: list[Actionable] = []
        self._transcript_delta = ""
        self._summary_task: asyncio.Task[None] | None = None

    async def on_chunk(self, chunk: Chunk) -> None:
        chunk = self._chunk_with_seq(chunk)
        self._logger.log(
            "chunk",
            seq=chunk.seq,
            channel=chunk.channel,
            text=chunk.text,
            ts=chunk.ts,
            final=chunk.final,
        )
        await self._emit_event(
            TranscriptEvent(
                channel=chunk.channel,
                text=chunk.text,
                final=chunk.final,
                ts=chunk.ts,
                seq=chunk.seq,
            )
        )
        if not chunk.final:
            self._partials[chunk.channel] = chunk.text
            await self._run_c1(chunk)
            await self._run_c2()
            return

        self._partials.pop(chunk.channel, None)
        self._transcript.append(chunk)
        self._append_transcript_delta(chunk)
        await self._run_c1(chunk)
        self._c1_segment_seen.pop(chunk.channel, None)
        await self._run_c2()

    def current_summary(self) -> MeetingSummary:
        return self._summary.model_copy(deep=True) if self._summary is not None else MeetingSummary()

    async def set_actionable_status(
        self,
        actionable_id: str,
        status: str,
    ) -> None:
        if status not in {"confirmed", "dismissed"}:
            return
        actionables = self._summary.actionables if self._summary is not None else self._actionables
        for index, item in enumerate(actionables):
            if item.id != actionable_id:
                continue
            updated = item.model_copy(update={"status": status})
            updated_actionables = list(actionables)
            updated_actionables[index] = updated
            self._actionables = updated_actionables
            if self._summary is not None:
                self._summary = self._summary.model_copy(
                    update={"actionables": updated_actionables}
                )
            await self._emit_event(
                ActionableEvent(
                    op="dismiss" if status == "dismissed" else "update",
                    item=updated,
                )
            )
            return

    async def _run_c1(self, chunk: Chunk) -> None:
        segment_seen = self._c1_segment_seen.setdefault(chunk.channel, set())
        seqs = [chunk.seq] if chunk.final else []
        for note_id, alias, is_title in self._exact.on_chunk(chunk):
            if note_id in segment_seen:
                continue
            segment_seen.add(note_id)
            note = self._index.note(note_id)
            if note is None:
                continue
            event = TriggerEvent(
                trigger_id=_trigger_id(),
                kind="exact",
                seqs=seqs,
                span=alias,
                matches=[
                    TriggerMatch(
                        note_id=note_id,
                        title=note.title,
                        score=1.0,
                    )
                ],
                action="grounded",
            )
            await self._emit_event(event)
            self._logger.log(
                "c1_hit",
                seq=chunk.seq,
                alias=alias,
                note_id=note_id,
                title=note.title,
                is_title=is_title,
                action="grounded",
            )

    async def _run_c2(self) -> None:
        window_chunks = self._window_chunks()
        window_text = self._window_text(window_chunks, self._active_partials())
        if not window_text:
            return
        context_chunks = self._transcript[
            -(config.C2_WINDOW_CHUNKS + config.C2_CONTEXT_CHUNKS) :
        ]
        context_text = self._window_text(context_chunks, self._active_partials())

        ambient_result = await self._ambient.on_window(window_text, context_text=context_text)
        if ambient_result is None:
            return

        grounding = (
            ambient_result
            if isinstance(ambient_result, GroundingUpdate)
            else getattr(self._ambient, "last_grounding", None)
        )
        self.last_grounding = grounding
        if grounding is None:
            return

        result = self._retrieval_from_grounding(window_text, grounding)
        trigger_id = _trigger_id()
        seqs = [chunk.seq for chunk in window_chunks]
        await self._emit_event(
            TriggerEvent(
                trigger_id=trigger_id,
                kind="search",
                seqs=seqs,
                span=result.heard,
                matches=self._search_trigger_matches(result),
                action="grounded" if grounding.items else "none",
            )
        )
        self._log_c2_pass(trigger_id, seqs, result, grounding)
        await self._emit_event(
            RetrievalEvent(
                heard=result.heard,
                hits=result.hits,
                expansions=result.expansions,
                retrieval_id=trigger_id,
                seqs=seqs,
                extracted=result.extracted,
            )
        )
        await self._emit_ambient(grounding)
        await self._offer_summary(grounding, window_text, ambient_result)

    async def _offer_summary(
        self,
        update: GroundingUpdate,
        window_text: str,
        ambient_result: Any,
    ) -> None:
        window_embedding = self._window_embedding(ambient_result, window_text)
        fire, reason = self._trigger.offer(
            update,
            window_embedding,
            len(self._transcript_delta),
        )
        if fire and self._summary_in_flight():
            log_grounding(self._logger, update, decision=False, reason="in_flight")
            return
        log_grounding(self._logger, update, decision=fire, reason=reason)
        if not fire:
            return

        delta = self._transcript_delta
        open_actionables = [
            item for item in self._actionables if item.status != "dismissed"
        ]
        self._summary_task = asyncio.create_task(
            self._run_summary(update, delta, open_actionables)
        )
        self._summary_task.add_done_callback(self._clear_summary_task)

    async def _run_summary(
        self,
        update: GroundingUpdate,
        transcript_delta: str,
        open_actionables: list[Actionable],
    ) -> None:
        try:
            result = await self._summarizer.summarize(
                self._summary,
                transcript_delta,
                update,
                open_actionables,
            )
        except Exception as exc:
            self._trigger.note_rejected()
            log_summary(
                self._logger,
                model="",
                latency_ms=0.0,
                no_update=False,
                rejected=True,
                reject_reason=str(exc),
                citation_stats={},
                accepted_actionable_ids=[],
            )
            return

        if result is None:
            return

        if result.no_update:
            self._trigger.note_summary(update, no_update=True)
            log_summary(
                self._logger,
                model=result.model,
                latency_ms=result.latency_ms,
                no_update=True,
                rejected=False,
                reject_reason="",
                citation_stats=result.citation_stats,
                accepted_actionable_ids=[],
            )
            return

        if result.summary is None:
            self._trigger.note_rejected()
            log_summary(
                self._logger,
                model=result.model,
                latency_ms=result.latency_ms,
                no_update=False,
                rejected=bool(result.rejected),
                reject_reason=result.reject_reason,
                citation_stats=result.citation_stats,
                accepted_actionable_ids=[],
            )
            return

        self._summary = result.summary
        self._actionables = list(result.summary.actionables)
        self._transcript_delta = ""
        self._trigger.note_summary(update, no_update=False)
        accepted_actionable_ids = [
            item.id for item in [*result.new_actionables, *result.updated_actionables]
        ]
        log_summary(
            self._logger,
            model=result.model,
            latency_ms=result.latency_ms,
            no_update=False,
            rejected=False,
            reject_reason="",
            citation_stats=result.citation_stats,
            accepted_actionable_ids=accepted_actionable_ids,
        )
        await self._emit_event(SummaryEvent.from_summary(result.summary))
        for item in result.new_actionables:
            await self._emit_event(ActionableEvent(op="new", item=item))
        for item in result.updated_actionables:
            await self._emit_event(ActionableEvent(op="update", item=item))

    def _clear_summary_task(self, task: asyncio.Task[None]) -> None:
        if self._summary_task is task:
            self._summary_task = None

    def _summary_in_flight(self) -> bool:
        return self._summary_task is not None and not self._summary_task.done()

    def _window_embedding(
        self,
        ambient_result: Any,
        window_text: str,
    ) -> np.ndarray | None:
        embedding = getattr(ambient_result, "window_embedding", None)
        if embedding is None:
            embedding = getattr(self._ambient, "last_window_embedding", None)
        if embedding is not None:
            return embedding
        try:
            return store.embed_queries([window_text])[0]
        except Exception:
            return None

    async def _emit_ambient(self, update: GroundingUpdate) -> None:
        await self._emit_event(
            AmbientEvent(
                items=[
                    AmbientItem(
                        note_id=item.note_id,
                        title=item.title,
                        gist=item.gist,
                        score=item.score,
                        short_id=item.short_id,
                        source=item.source,
                        age_days=item.age_days,
                        why=item.why,
                    )
                    for item in update.items
                ]
            )
        )

    def _append_transcript_delta(self, chunk: Chunk) -> None:
        text = chunk.text.strip()
        if not text:
            return
        line = f"[ch{chunk.channel}] {text}"
        self._transcript_delta = (
            line if not self._transcript_delta else f"{self._transcript_delta}\n{line}"
        )
        if len(self._transcript_delta) > config.SUMMARY_DELTA_MAX_CHARS:
            self._transcript_delta = self._transcript_delta[-config.SUMMARY_DELTA_MAX_CHARS :]

    def _window_chunks(self) -> list[Chunk]:
        return self._transcript[-config.C2_WINDOW_CHUNKS :]

    def _window_text(self, chunks: list[Chunk], partials: dict[int, str]) -> str:
        parts = [chunk.text for chunk in chunks]
        parts.extend(text for _channel, text in sorted(partials.items()))
        return " ".join(parts).strip()

    def _active_partials(self) -> dict[int, str]:
        return {
            channel: text.strip()
            for channel, text in self._partials.items()
            if text.strip()
        }

    async def _emit_event(self, event: Event) -> None:
        result = self._emit(event)
        if isawaitable(result):
            await result

    def _chunk_with_seq(self, chunk: Chunk) -> Chunk:
        if not chunk.final:
            return chunk.model_copy(update={"seq": 0})
        seq = self._next_seq
        self._next_seq += 1
        return chunk.model_copy(update={"seq": seq})

    def _search_trigger_matches(self, result: RetrievalResult) -> list[TriggerMatch]:
        matches: list[TriggerMatch] = []
        for hit in result.hits:
            note = self._index.note(hit.note_id)
            matches.append(
                TriggerMatch(
                    note_id=hit.note_id,
                    title=note.title if note is not None else hit.note_id,
                    rank=hit.rank,
                    score=hit.score,
                )
            )
        for expansion in result.expansions:
            note = self._index.note(expansion.note_id)
            matches.append(
                TriggerMatch(
                    note_id=expansion.note_id,
                    title=note.title if note is not None else expansion.note_id,
                    score=expansion.score,
                    via_note_id=expansion.from_note_id,
                    hop=expansion.hop,
                )
            )
        return matches

    def _retrieval_from_grounding(
        self,
        heard: str,
        update: GroundingUpdate,
    ) -> RetrievalResult:
        fused_ids = set(getattr(self._ambient, "last_fused_ids", set()))
        extracted = list(getattr(self._ambient, "last_extracted", []))
        hits = [
            Hit(note_id=item.note_id, rank=rank, score=item.score)
            for rank, item in enumerate(update.items, start=1)
        ]
        expansions = [
            Expansion(
                from_note_id="",
                note_id=note_id,
                hop=0,
                score=mass,
                via="ppr",
            )
            for note_id, mass in update.ppr_top
            if note_id not in fused_ids
        ]
        return RetrievalResult(
            heard=heard,
            hits=hits,
            expansions=expansions,
            extracted=extracted,
        )

    def _log_c2_pass(
        self,
        trigger_id: str,
        seqs: list[int],
        result: RetrievalResult,
        grounding: GroundingUpdate,
    ) -> None:
        debug = dict(getattr(self._ambient, "last_debug", None) or {})
        if not debug:
            debug = {
                "window": result.heard,
                "gliner_spans": [],
                "gliner_dropped": [],
                "gliner_context_queries": [],
                "dense": [],
                "dense_gliner": [],
                "bm25": [],
                "exact": [],
                "fused": [],
                "ppr_top": [],
                "blend": [],
                "grounding_revision": grounding.revision,
            }
        debug.setdefault("gliner_spans", [])
        debug.setdefault("gliner_dropped", [])
        debug.setdefault("gliner_context_queries", [])
        debug.setdefault("dense_gliner", [])
        debug["grounding_ids"] = [item.note_id for item in grounding.items]
        self._logger.log("c2_pass", trigger_id=trigger_id, seqs=seqs, **debug)


def _trigger_id() -> str:
    return uuid4().hex[:12]
