from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
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


@pytest.mark.asyncio
async def test_gliner_span_timeout_fails_open_within_the_live_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "LIVE_SPANS_BUDGET_S", 0.05)

    class SlowExtractor:
        def extract(self, text: str) -> list[tuple[str, float, str]]:
            time.sleep(0.5)
            return [("Maya", 0.9, "person")]

    pack = _pack("slow", accepted=True)
    engine = FakeSharedEngine([pack])
    retriever = AmbientRetriever(engine, FakeClock(), span_extractor=SlowExtractor())

    assert await retriever.on_window("Maya") is pack
    assert engine.calls[0]["entity_spans"] == ()
    assert retriever.last_spans_timeout
    assert retriever.last_debug is not None
    assert retriever.last_debug["spans_timeout"] is True


@pytest.mark.asyncio
async def test_deadline_clock_starts_before_span_extraction() -> None:
    order: list[str] = []

    def deadline_clock() -> float:
        order.append("deadline")
        return 500.0

    class RecordingExtractor:
        def extract(self, text: str) -> list[tuple[str, float, str]]:
            order.append("spans")
            return []

    engine = FakeSharedEngine([_pack("x", accepted=True)])
    retriever = AmbientRetriever(
        engine,
        FakeClock(),
        span_extractor=RecordingExtractor(),
        deadline_clock=deadline_clock,
    )

    await retriever.on_window("query")

    assert order[:2] == ["deadline", "spans"]
    assert engine.calls[0]["deadline_at"] == pytest.approx(500.75)


@pytest.mark.asyncio
async def test_topic_shift_fires_on_accepted_pivot_and_freezes_on_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "LIVE_TOPIC_SHIFT_SIM", 0.35)
    east = np.array([1.0, 0.0], dtype=np.float32)
    north = np.array([0.0, 1.0], dtype=np.float32)

    def embedder(text: str) -> np.ndarray:
        return east if "east" in text else north

    clock = FakeClock()
    abstained = EvidencePack.abstained(
        RetrievalQuery("east filler", "live", now=3.0), "below_floor"
    )
    engine = FakeSharedEngine(
        [
            _pack("east one", accepted=True),
            _pack("north pivot", accepted=True),
            abstained,
            _pack("east return", accepted=True),
        ]
    )
    retriever = AmbientRetriever(engine, clock, query_embedder=embedder)

    await retriever.on_window("east one")
    assert not retriever.last_topic_shift  # first accepted window seeds the EMA

    clock.advance(config.C2_THROTTLE_S)
    await retriever.on_window("north pivot")
    assert retriever.last_topic_shift
    assert retriever.last_debug is not None
    assert retriever.last_debug["topic_shift"] is True

    clock.advance(config.C2_THROTTLE_S)
    await retriever.on_window("east filler")  # abstained: EMA must not move
    assert not retriever.last_topic_shift

    clock.advance(config.C2_THROTTLE_S)
    await retriever.on_window("east return")  # vs north EMA → pivot again
    assert retriever.last_topic_shift


@pytest.mark.asyncio
async def test_window_embedding_is_exposed_for_the_trigger() -> None:
    vector = np.array([0.6, 0.8], dtype=np.float32)
    engine = FakeSharedEngine([_pack("x", accepted=True)])
    retriever = AmbientRetriever(engine, FakeClock(), query_embedder=lambda text: vector)

    await retriever.on_window("anything")

    assert retriever.last_window_embedding is not None
    assert float(np.linalg.norm(retriever.last_window_embedding)) == pytest.approx(1.0)


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
