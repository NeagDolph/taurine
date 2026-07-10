"""Throttled live policy adapter over the shared evidence retrieval engine."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from taurine_core import config
from taurine_core.retrieval.types import EvidencePack, RetrievalQuery


class Clock(Protocol):
    def now(self) -> float: ...


class LiveRetrievalPort(Protocol):
    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        entity_spans: Sequence[tuple[str, str | None]] = (),
        session_reset: Mapping[str, float] | None = None,
        deadline_at: float | None = None,
    ) -> EvidencePack: ...


class LiveSpanExtractorPort(Protocol):
    def extract(self, text: str) -> Sequence[tuple[str, float, str]]: ...


class AmbientRetriever:
    """Apply live cadence/window/deadline policy without duplicating retrieval."""

    def __init__(
        self,
        engine: LiveRetrievalPort,
        clock: Clock,
        on_debug: Callable[[dict[str, Any]], None] | None = None,
        span_extractor: LiveSpanExtractorPort | None = None,
        wall_clock: Callable[[], float] = time.time,
        deadline_clock: Callable[[], float] = time.monotonic,
        deadline_budget_s: float = 0.75,
    ) -> None:
        if deadline_budget_s <= 0.0:
            raise ValueError("live retrieval deadline budget must be positive")
        self.last_pack: EvidencePack | None = None
        self.last_debug: dict[str, Any] | None = None
        self.last_entity_spans: tuple[tuple[str, str | None], ...] = ()
        self._engine = engine
        self._clock = clock
        self._wall_clock = wall_clock
        self._deadline_clock = deadline_clock
        self._deadline_budget_s = float(deadline_budget_s)
        self._on_debug = on_debug
        self._span_extractor = span_extractor
        self._last_run_at: float | None = None

    async def on_window(
        self,
        window_text: str,
        context_text: str | None = None,
        *,
        session_reset: Mapping[str, float] | None = None,
    ) -> EvidencePack | None:
        """Retrieve one live evidence pack; ``None`` means cadence throttle only."""

        self.last_pack = None
        self.last_debug = None
        self.last_entity_spans = ()
        cadence_now = self._clock.now()
        if (
            self._last_run_at is not None
            and cadence_now - self._last_run_at < config.C2_THROTTLE_S
        ):
            return None
        self._last_run_at = cadence_now

        query_text = _newest(window_text, config.LIVE_WINDOW_MAX_CHARS)
        context = _newest(context_text or "", config.LIVE_WINDOW_MAX_CHARS)
        conversation_context = (
            (context,) if context and context != query_text else ()
        )
        entity_spans = self._entity_spans(query_text)
        self.last_entity_spans = entity_spans
        query = RetrievalQuery(
            text=query_text,
            mode="live",
            now=float(self._wall_clock()),
            conversation_context=conversation_context,
        )
        deadline_at = self._deadline_clock() + self._deadline_budget_s
        pack = await asyncio.to_thread(
            self._engine.retrieve,
            query,
            entity_spans=entity_spans,
            session_reset=dict(session_reset or {}),
            deadline_at=deadline_at,
        )
        if pack.query.mode != "live":
            raise ValueError("shared live retrieval returned a non-live evidence pack")
        self.last_pack = pack
        self._record_debug(pack, deadline_at)
        return pack

    def _entity_spans(self, text: str) -> tuple[tuple[str, str | None], ...]:
        if self._span_extractor is None:
            return ()
        try:
            extracted = self._span_extractor.extract(text)
        except Exception:
            return ()
        rows: list[tuple[str, str | None]] = []
        seen: set[tuple[str, str | None]] = set()
        for surface, _confidence, label in extracted:
            normalized_surface = str(surface).strip()
            normalized_label = str(label).strip() or None
            row = (normalized_surface, normalized_label)
            if not normalized_surface or row in seen:
                continue
            seen.add(row)
            rows.append(row)
        return tuple(rows)

    def _record_debug(self, pack: EvidencePack, deadline_at: float) -> None:
        debug = {
            "accepted": pack.decision.accepted,
            "reason": pack.decision.reason,
            "confidence": pack.decision.confidence,
            "degraded": bool(pack.trace.debug.get("rerank_degraded", False)),
            "passage_ids": [row.evidence_id for row in pack.passages],
            "fact_ids": [row.fact_ref for row in pack.facts],
            "active_build_id": pack.trace.active_build_id,
            "deadline_at": deadline_at,
        }
        self.last_debug = debug
        if self._on_debug is None:
            return
        try:
            self._on_debug(debug)
        except Exception:
            return


def _newest(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text if len(text) <= max_chars else text[-max_chars:]
