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
        self._topic_start_seq: int | None = None
        # User-set actionable statuses always win over summarizer snapshots
        # (a dismissal during an in-flight summary must survive acceptance).
        self._user_status: dict[str, str] = {}
        self._closed = False

    async def aclose(self) -> None:
        """Idempotent shutdown: cancel and drain any in-flight summary task.

        Callers must await this before closing the session logger so the
        cancelled task cannot log into a closed sink.
        """

        self._closed = True
        task = self._summary_task
        self._summary_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

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
        self._user_status[actionable_id] = status
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
        window = self.ledger.query_window(
            now=now,
            partials=self._partials,
            topic_start_seq=self._topic_start_seq,
        )
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
        topic_shift = bool(getattr(self._ambient, "last_topic_shift", False))
        if topic_shift and window.citeable:
            # Later query windows must not span the detected pivot. The
            # newest chunk of the pivoting window is what confirmed the
            # shift, so it starts the new topic region.
            self._topic_start_seq = window.citeable[-1].seq
        update = self._grounding.on_pack(pack, topic_shift=topic_shift)
        self.last_grounding = update
        await self._emit_event(LiveGroundingEvent.from_domain(update))
        self._logger.log(
            "retrieval",
            accepted=pack.decision.accepted,
            reason=pack.decision.reason,
            confidence=pack.decision.confidence,
            revision=update.revision,
            new_support_ids=list(update.new_support_ids),
            topic_shift=topic_shift,
            active_build_id=pack.trace.active_build_id,
            calibration_hash=pack.trace.calibration_hash,
            latency_ms=dict(pack.trace.latency_ms),
        )
        await self._offer_summary(update)

    async def _offer_summary(self, update: GroundingUpdate) -> None:
        if self._closed:
            return
        pending = self.ledger.pending()
        if not pending:
            return
        fire, reason = self._trigger.offer(
            update,
            getattr(self._ambient, "last_window_embedding", None),
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
        # Any exit that neither accepts nor rejects the checkpoint —
        # including task cancellation — must release it, or the ledger stays
        # in-flight forever and summaries die for the session (M8).
        settled = False

        def reject(reason: str) -> None:
            nonlocal settled
            settled = True
            self.ledger.reject_summary(snapshot)
            self._trigger.note_rejected()
            self._log_summary_rejection(reason)

        try:
            try:
                result = await self._summarizer.summarize(
                    self._summary,
                    transcript_evidence,
                    update,
                    open_actionables,
                )
            except Exception as exc:
                reject(str(exc))
                return
            if result is None:
                reject("llm_unavailable")
                return
            if result.no_update:
                settled = True
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
                reject(result.reject_reason or "unsupported_claim")
                return

            settled = True
            self.ledger.accept_summary(snapshot)
            self.ledger.prune_accepted(retain=2)
            # User-set statuses (confirm/dismiss during the in-flight call)
            # take precedence over the summarizer's stale snapshot (M9).
            actionables = self._apply_user_status(result.summary.actionables)
            summary = result.summary.model_copy(update={"actionables": actionables})
            self._summary = summary
            self._actionables = list(actionables)
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
            await self._emit_event(SummaryEvent.from_summary(summary))
            for row in self._apply_user_status(result.new_actionables):
                await self._emit_event(ActionableEvent(op="new", item=row))
            for row in self._apply_user_status(result.updated_actionables):
                await self._emit_event(ActionableEvent(op="update", item=row))
        finally:
            if not settled:
                # Cancellation path: release the checkpoint, then let the
                # CancelledError propagate.
                self.ledger.reject_summary(snapshot)
                self._trigger.note_rejected()
                self._log_summary_rejection("cancelled")

    def _apply_user_status(self, rows: list[Actionable]) -> list[Actionable]:
        return [
            row.model_copy(update={"status": self._user_status[row.id]})
            if self._user_status.get(row.id, row.status) != row.status
            else row
            for row in rows
        ]

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
