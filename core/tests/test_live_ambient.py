from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live.ambient import AmbientRetriever
from taurine_core.retrieval.types import (
    EvidencePack,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
)


@dataclass
class FakeClock:
    value: float = 100.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSpanExtractor:
    def __init__(self, rows: list[tuple[str, float, str]] | Exception) -> None:
        self.rows = rows
        self.calls: list[str] = []

    def extract(self, text: str) -> list[tuple[str, float, str]]:
        self.calls.append(text)
        if isinstance(self.rows, Exception):
            raise self.rows
        return self.rows


class FakeSharedEngine:
    def __init__(self, packs: list[EvidencePack]) -> None:
        self.packs = packs
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        entity_spans=(),
        session_reset=None,
        deadline_at=None,
    ) -> EvidencePack:
        self.calls.append(
            {
                "query": query,
                "entity_spans": tuple(entity_spans),
                "session_reset": dict(session_reset or {}),
                "deadline_at": deadline_at,
            }
        )
        return self.packs.pop(0)


@pytest.mark.asyncio
async def test_live_adapter_throttles_and_calls_shared_engine_with_live_policy() -> None:
    clock = FakeClock()
    pack = _pack("first", accepted=True)
    engine = FakeSharedEngine([pack, pack])
    spans = FakeSpanExtractor(
        [("Maya", 0.96, "person"), ("Launch", 0.91, "project")]
    )
    debug: list[dict[str, object]] = []
    retriever = AmbientRetriever(
        engine,
        clock,
        span_extractor=spans,
        wall_clock=lambda: 1_700_000_000.0,
        deadline_clock=lambda: 500.0,
        deadline_budget_s=0.75,
        on_debug=debug.append,
    )
    text = "old" * 3000 + " current Maya Launch"

    result = await retriever.on_window(
        text,
        session_reset={"p-old": 0.4},
    )

    assert result is pack
    [call] = engine.calls
    query = call["query"]
    assert isinstance(query, RetrievalQuery)
    assert query.mode == "live"
    assert query.now == 1_700_000_000.0
    assert query.text == text[-config.LIVE_WINDOW_MAX_CHARS :]
    assert call["entity_spans"] == (("Maya", "person"), ("Launch", "project"))
    assert call["session_reset"] == {"p-old": 0.4}
    assert call["deadline_at"] == pytest.approx(500.75)
    assert retriever.last_pack is pack
    assert debug[0]["accepted"] is True

    assert await retriever.on_window("throttled") is None
    assert retriever.last_pack is None
    assert len(engine.calls) == 1

    clock.advance(config.C2_THROTTLE_S)
    assert await retriever.on_window("next") is pack
    assert len(engine.calls) == 2


@pytest.mark.asyncio
async def test_adapter_exposes_abstained_and_degraded_packs_unchanged() -> None:
    abstained = EvidencePack.abstained(
        RetrievalQuery("none", "live", now=1.0), "below_floor"
    )
    degraded = _pack("degraded", accepted=True, degraded=True)
    engine = FakeSharedEngine([abstained, degraded])
    clock = FakeClock()
    retriever = AmbientRetriever(engine, clock)

    assert await retriever.on_window("none") is abstained
    assert not retriever.last_pack.decision.accepted  # type: ignore[union-attr]

    clock.advance(config.C2_THROTTLE_S)
    assert await retriever.on_window("degraded") is degraded
    assert retriever.last_pack is degraded
    assert retriever.last_pack.trace.debug["rerank_degraded"] is True


@pytest.mark.asyncio
async def test_span_extraction_failure_degrades_to_shared_heads_without_entities() -> None:
    pack = _pack("fallback", accepted=True)
    engine = FakeSharedEngine([pack])
    retriever = AmbientRetriever(
        engine,
        FakeClock(),
        span_extractor=FakeSpanExtractor(RuntimeError("extractor unavailable")),
    )

    assert await retriever.on_window("Maya") is pack
    assert engine.calls[0]["entity_spans"] == ()
    assert retriever.last_entity_spans == ()


def _pack(text: str, *, accepted: bool, degraded: bool = False) -> EvidencePack:
    query = RetrievalQuery(text, "live", now=1.0)
    return EvidencePack(
        decision=RetrievalDecision(accepted, 0.9 if accepted else 0.0, "accepted"),
        query=query,
        passages=(),
        facts=(),
        paths=(),
        notes=(),
        trace=RetrievalTrace(
            original_query=text,
            debug={"rerank_degraded": degraded},
        ),
    )
