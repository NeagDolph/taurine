"""Sequence-safe live session orchestration over evidence-first retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any, Protocol

import numpy as np

from taurine_core import config
from taurine_core.live.ambient import AmbientRetriever
from taurine_core.live.grounding import GroundingEngine, GroundingUpdate
from taurine_core.live.ledger import SummaryLedgerSnapshot, TranscriptLedger
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.schemas import (
    Actionable,
    ActionableEvent,
    Chunk,
    LiveGroundingEvent,
    MeetingSummary,
    SummaryEvent,
    TranscriptEvent,
)


Event = TranscriptEvent | LiveGroundingEvent | SummaryEvent | ActionableEvent
EventSink = Callable[[Event], Awaitable[None] | None]


class TriggerLike(Protocol):
    def offer(
        self,
        update: GroundingUpdate,
        window_emb: np.ndarray | None,
        new_chars: int,
    ) -> tuple[bool, str]: ...

    def note_summary(self, update: GroundingUpdate, *, no_update: bool) -> None: ...

    def note_rejected(self) -> None: ...


class SummarizerLike(Protocol):
    async def summarize(
        self,
        state: MeetingSummary | None,
        transcript_evidence: str,
        update: GroundingUpdate,
        open_actionables: list[Actionable],
    ) -> Any: ...


class LoggerLike(Protocol):
    def log(self, kind: str, **payload: Any) -> None: ...


class _NullLogger:
    def log(self, kind: str, **payload: Any) -> None:
        del kind, payload


class LiveSession:
    def __init__(
        self,
        *,
        registry: EvidenceRegistry,
        ambient: AmbientRetriever,
        grounding: GroundingEngine,
        trigger: TriggerLike,
        summarizer: SummarizerLike,
        emit: EventSink,
        logger: LoggerLike | None = None,
    ) -> None:
        if grounding.registry is not registry:
            raise ValueError("live retrieval, grounding, and transcript must share a registry")
        self.registry = registry
        self.ledger = TranscriptLedger(registry)
        self._ambient = ambient
        self._grounding = grounding
        self._trigger = trigger
        self._summarizer = summarizer
        self._emit = emit
        self._logger = logger or _NullLogger()
        self._partials: dict[int, str] = {}
        self.last_grounding: GroundingUpdate | None = None
        self._summary: MeetingSummary | None = None
        self._actionables: list[Actionable] = []
        self._summary_task: asyncio.Task[None] | None = None

    async def on_chunk(self, chunk: Chunk) -> None:
        if chunk.final:
            text = chunk.text.strip()
            if not text:
                return
            row = self.ledger.append(chunk)
            emitted = chunk.model_copy(update={"text": row.text, "seq": row.seq})
            self._partials.pop(chunk.channel, None)
        else:
            emitted = chunk.model_copy(update={"seq": 0})
            self._partials[chunk.channel] = chunk.text
        self._logger.log(
            "chunk",
            seq=emitted.seq,
            channel=emitted.channel,
            text=emitted.text,
            ts=emitted.ts,
            final=emitted.final,
        )
        await self._emit_event(
            TranscriptEvent(
                channel=emitted.channel,
                text=emitted.text,
                final=emitted.final,
                ts=emitted.ts,
                seq=emitted.seq,
            )
        )
        await self._run_retrieval(chunk.ts)

    def current_summary(self) -> MeetingSummary:
        return (
            self._summary.model_copy(deep=True)
            if self._summary is not None
            else MeetingSummary()
        )

    async def set_actionable_status(self, actionable_id: str, status: str) -> None:
        if status not in {"confirmed", "dismissed"}:
            return
        rows = self._summary.actionables if self._summary else self._actionables
        for index, row in enumerate(rows):
            if row.id != actionable_id:
                continue
            updated = row.model_copy(update={"status": status})
            next_rows = list(rows)
            next_rows[index] = updated
            self._actionables = next_rows
            if self._summary is not None:
                self._summary = self._summary.model_copy(
                    update={"actionables": next_rows}
                )
            await self._emit_event(
                ActionableEvent(
                    op="dismiss" if status == "dismissed" else "update",
                    item=updated,
                )
            )
            return

    async def _run_retrieval(self, now: float) -> None:
        window = self.ledger.query_window(now=now, partials=self._partials)
        if not window.query_text.strip():
            return
        context_text = "\n".join(row.text for row in window.context)
        pack = await self._ambient.on_window(
            window.query_text,
            context_text=context_text or None,
            session_reset=self._grounding.session_reset(),
        )
        if pack is None:
            return
        update = self._grounding.on_pack(pack)
        self.last_grounding = update
        await self._emit_event(LiveGroundingEvent.from_domain(update))
        self._logger.log(
            "retrieval",
            accepted=pack.decision.accepted,
            reason=pack.decision.reason,
            confidence=pack.decision.confidence,
            revision=update.revision,
            new_support_ids=list(update.new_support_ids),
            active_build_id=pack.trace.active_build_id,
            calibration_hash=pack.trace.calibration_hash,
            latency_ms=dict(pack.trace.latency_ms),
        )
        await self._offer_summary(update)

    async def _offer_summary(self, update: GroundingUpdate) -> None:
        pending = self.ledger.pending()
        if not pending:
            return
        fire, reason = self._trigger.offer(
            update,
            None,
            sum(len(row.text) for row in pending),
        )
        if fire and self._summary_in_flight():
            self._logger.log("summary_trigger", decision=False, reason="in_flight")
            return
        self._logger.log("summary_trigger", decision=fire, reason=reason)
        if not fire:
            return
        snapshot = self.ledger.begin_summary(
            max_chars=config.SUMMARY_DELTA_MAX_CHARS
        )
        transcript_evidence = self.ledger.render(snapshot.chunks)
        open_actionables = [
            row for row in self._actionables if row.status != "dismissed"
        ]
        self._summary_task = asyncio.create_task(
            self._run_summary(update, snapshot, transcript_evidence, open_actionables)
        )
        self._summary_task.add_done_callback(self._clear_summary_task)

    async def _run_summary(
        self,
        update: GroundingUpdate,
        snapshot: SummaryLedgerSnapshot,
        transcript_evidence: str,
        open_actionables: list[Actionable],
    ) -> None:
        try:
            result = await self._summarizer.summarize(
                self._summary,
                transcript_evidence,
                update,
                open_actionables,
            )
        except Exception as exc:
            self.ledger.reject_summary(snapshot)
            self._trigger.note_rejected()
            self._log_summary_rejection(str(exc))
            return
        if result is None:
            self.ledger.reject_summary(snapshot)
            self._trigger.note_rejected()
            self._log_summary_rejection("llm_unavailable")
            return
        if result.no_update:
            self.ledger.reject_summary(snapshot)
            self._trigger.note_summary(update, no_update=True)
            self._logger.log(
                "summary",
                no_update=True,
                rejected=False,
                model=result.model,
                latency_ms=result.latency_ms,
                through_seq=snapshot.through_seq,
            )
            return
        if result.summary is None or result.rejected:
            self.ledger.reject_summary(snapshot)
            self._trigger.note_rejected()
            self._log_summary_rejection(result.reject_reason or "unsupported_claim")
            return

        self.ledger.accept_summary(snapshot)
        self.ledger.prune_accepted(retain=2)
        self._summary = result.summary
        self._actionables = list(result.summary.actionables)
        self._trigger.note_summary(update, no_update=False)
        self._logger.log(
            "summary",
            no_update=False,
            rejected=False,
            model=result.model,
            latency_ms=result.latency_ms,
            through_seq=snapshot.through_seq,
            cited_ids=(
                list(result.validation.cited_ids)
                if result.validation is not None
                else []
            ),
        )
        await self._emit_event(SummaryEvent.from_summary(result.summary))
        for row in result.new_actionables:
            await self._emit_event(ActionableEvent(op="new", item=row))
        for row in result.updated_actionables:
            await self._emit_event(ActionableEvent(op="update", item=row))

    def _log_summary_rejection(self, reason: str) -> None:
        self._logger.log(
            "summary",
            no_update=False,
            rejected=True,
            reject_reason=reason,
        )

    def _summary_in_flight(self) -> bool:
        return self._summary_task is not None and not self._summary_task.done()

    def _clear_summary_task(self, task: asyncio.Task[None]) -> None:
        if self._summary_task is task:
            self._summary_task = None

    async def _emit_event(self, event: Event) -> None:
        pending = self._emit(event)
        if isawaitable(pending):
            await pending
