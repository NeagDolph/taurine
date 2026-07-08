from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live import ambient as ambient_module
from taurine_core.live.ambient import AmbientRetriever
from taurine_core.live.grounding import GroundingUpdate
from taurine_core.schemas import GroundingItem, GroundingWhy

from tests.fakes import FakeClock, FakeIndex, note


class UnusedReranker:
    async def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        raise AssertionError("ambient retrieval must not call rerank")


class QueryAwareIndex(FakeIndex):
    def __init__(
        self,
        *,
        dense_by_query: dict[str, list[tuple[str, float]]],
        bm25_by_query: dict[str, list[tuple[str, float]]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dense_by_query = dense_by_query
        self.bm25_by_query = bm25_by_query or {}
        self.dense_queries: list[str] = []
        self.bm25_queries: list[str] = []

    def dense_search(self, text: str, k: int) -> list[tuple[str, float]]:
        self.dense_queries.append(text)
        return self.dense_by_query.get(text, [])[:k]

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        self.bm25_queries.append(text)
        if text in self.bm25_by_query:
            return self.bm25_by_query[text][:k]
        return super().bm25_search(text, k)


class FakeSpanExtractor:
    def __init__(
        self,
        spans: list[tuple[str, float, str]],
        dropped: list[tuple[str, float, str]] | None = None,
    ) -> None:
        self.spans = spans
        self.last_dropped = dropped or []
        self.calls: list[str] = []

    def extract(self, text: str) -> list[tuple[str, float, str]]:
        self.calls.append(text)
        return self.spans


@dataclass
class RecordingGroundingEngine:
    update: GroundingUpdate
    calls: list[dict[str, Any]]

    def on_pass(
        self,
        window_text: str,
        fused: list[tuple[str, float]],
        exact_hits: list[tuple[str, str]],
        now: float,
        *,
        head_hits: dict[str, set[str]] | None = None,
    ) -> GroundingUpdate:
        self.calls.append(
            {
                "window_text": window_text,
                "fused": fused,
                "exact_hits": exact_hits,
                "now": now,
                "head_hits": head_hits or {},
            }
        )
        return self.update

    @property
    def last_blend(self) -> dict[str, dict[str, float]]:
        return {
            item.note_id: {
                "rel": 1.0,
                "recency": 1.0,
                "source_rank": 1.0,
                "quality": 1.0,
                "final": item.score,
            }
            for item in self.update.items
        }


def _patch_grounding(
    monkeypatch: pytest.MonkeyPatch,
    update: GroundingUpdate | None = None,
) -> RecordingGroundingEngine:
    update = update or GroundingUpdate(
        items=[
            GroundingItem(
                note_id="alpha",
                short_id="N1",
                title="Alpha",
                gist="Alpha gist",
                score=0.9,
                why=GroundingWhy(heads=["dense"]),
            )
        ],
        revision=1,
        ppr_top=[("ppr-only", 0.2), ("alpha", 0.1)],
    )
    engine = RecordingGroundingEngine(update=update, calls=[])
    monkeypatch.setattr(ambient_module, "GroundingEngine", lambda index: engine)
    return engine


@pytest.mark.asyncio
async def test_heads_rrf_exact_injection_and_debug_payload(monkeypatch) -> None:
    engine = _patch_grounding(monkeypatch)
    window = "ask armand about exact alpha"
    bm25_query = f"{window} armaan"
    span_extractor = FakeSpanExtractor(
        [("armand", 0.95, "person")],
        dropped=[("thing", 0.7, "topic")],
    )
    index = QueryAwareIndex(
        notes={
            "alpha": note("alpha", "Alpha"),
            "span": note("span", "Armaan Plan"),
            "bm25": note("bm25", "BM25"),
            "exact": note("exact", "Exact Alpha"),
        },
        dense_by_query={
            window: [("alpha", 0.9)],
            "armand": [("span", 0.7)],
            "armaan": [("span", 0.95)],
        },
        bm25_by_query={bm25_query: [("bm25", 5.0)]},
        alias_rows=[
            ("armaan plan", "span", True),
            ("exact alpha", "exact", True),
        ],
    )

    result = await AmbientRetriever(
        index,
        UnusedReranker(),
        FakeClock(start=123.0),
        on_debug=lambda _payload: None,
        span_extractor=span_extractor,
    ).on_window(window)

    assert result is engine.update
    [call] = engine.calls
    assert call["window_text"] == window
    assert call["exact_hits"] == [("exact", "exact alpha")]
    assert call["head_hits"] == {
        "alpha": {"dense", "gliner"},
        "span": {"gliner"},
        "bm25": {"bm25"},
        "exact": {"exact"},
    }
    assert call["fused"] == [
        (
            "alpha",
            pytest.approx((1 / (config.RRF_K + 1)) + (1 / (config.RRF_K + 2))),
        ),
        ("span", pytest.approx(1 / (config.RRF_K + 1))),
        ("bm25", pytest.approx(1 / (config.RRF_K + 1))),
        ("exact", pytest.approx(1 / (config.RRF_K + 1))),
    ]
    assert index.dense_queries == [window, "armand", "armaan", window]
    assert index.bm25_queries == [bm25_query]
    assert span_extractor.calls == [window]


@pytest.mark.asyncio
async def test_on_window_records_grounding_retrieval_payload(monkeypatch) -> None:
    update = GroundingUpdate(
        items=[
            GroundingItem(
                note_id="alpha",
                short_id="N1",
                title="Alpha",
                gist="Alpha gist",
                score=0.8,
            )
        ],
        revision=7,
        ppr_top=[("ppr-only", 0.2), ("alpha", 0.1)],
    )
    _patch_grounding(monkeypatch, update)
    debug_records: list[dict[str, Any]] = []
    index = QueryAwareIndex(
        notes={"alpha": note("alpha", "Alpha")},
        dense_by_query={"window": [("alpha", 0.9)]},
    )
    retriever = AmbientRetriever(
        index,
        UnusedReranker(),
        FakeClock(),
        on_debug=debug_records.append,
    )

    result = await retriever.on_window("window")

    assert result is update
    assert retriever.last_grounding is update
    assert retriever.last_extracted == []
    assert retriever.last_fused_ids == {"alpha"}
    assert retriever.last_ppr_expansions == [("ppr-only", 0.2)]
    debug = debug_records[0]
    assert debug["ppr_top"] == [["ppr-only", 0.2], ["alpha", 0.1]]
    assert debug["blend"] == [
        {
            "note_id": "alpha",
            "rel": 1.0,
            "recency": 1.0,
            "source_rank": 1.0,
            "quality": 1.0,
            "final": 0.8,
        }
    ]
    assert debug["grounding_revision"] == 7
    assert "rerank_used" not in debug
    assert "graph_vote" not in debug
    assert "salient" not in debug


@pytest.mark.asyncio
async def test_throttled_window_returns_none_and_clears_transient_state(monkeypatch) -> None:
    _patch_grounding(monkeypatch)
    clock = FakeClock()
    retriever = AmbientRetriever(
        QueryAwareIndex(
            notes={"alpha": note("alpha", "Alpha")},
            dense_by_query={"first": [("alpha", 0.9)], "second": [("alpha", 0.8)]},
        ),
        UnusedReranker(),
        clock,
    )

    assert await retriever.on_window("first") is not None
    assert retriever.last_grounding is not None
    assert await retriever.on_window("second") is None
    assert retriever.last_grounding is None

    clock.advance(config.C2_THROTTLE_S)
    assert await retriever.on_window("second") is not None


@pytest.mark.asyncio
async def test_span_context_query_uses_context_text_and_logs_drops(monkeypatch) -> None:
    _patch_grounding(monkeypatch)
    window = "Deepgram follow-up"
    context = "Shopify handles checkout payments. The team moved to Deepgram follow-up."
    span_extractor = FakeSpanExtractor(
        [("Shopify", 0.96, "organization")],
        dropped=[("dude", 0.7, "person")],
    )
    index = QueryAwareIndex(
        notes={"context": note("context", "Context Note")},
        dense_by_query={
            window: [],
            "Shopify": [],
            context: [("context", 0.93)],
        },
    )
    debug_records: list[dict[str, Any]] = []

    await AmbientRetriever(
        index,
        UnusedReranker(),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window, context_text=context)

    assert index.dense_queries == [window, "Shopify", context]
    assert debug_records[0]["gliner_context_queries"] == [context]
    assert debug_records[0]["gliner_dropped"] == [["dude", 0.7, "person"]]


@pytest.mark.asyncio
async def test_span_context_query_caps_around_span(monkeypatch) -> None:
    monkeypatch.setattr(config, "SPAN_CONTEXT_MAX_CHARS", 64)
    _patch_grounding(monkeypatch)
    window = "Deepgram follow-up"
    context = (
        "Earlier alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. "
        "Then the roadmap team mentioned Shopify while comparing checkout operations "
        "against a very long implementation timeline. "
        "After that they moved to Deepgram follow-up details."
    )
    span_extractor = FakeSpanExtractor([("Shopify", 0.96, "organization")])
    index = QueryAwareIndex(dense_by_query={window: [], "Shopify": []})
    debug_records: list[dict[str, Any]] = []

    await AmbientRetriever(
        index,
        UnusedReranker(),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window, context_text=context)

    [context_query] = debug_records[0]["gliner_context_queries"]
    assert len(context_query) <= 64
    assert "Shopify" in context_query
    assert context_query in index.dense_queries


@pytest.mark.asyncio
async def test_empty_span_extractor_keeps_dense_and_bm25_heads_only(monkeypatch) -> None:
    engine = _patch_grounding(monkeypatch)
    span_extractor = FakeSpanExtractor([])
    index = QueryAwareIndex(
        notes={"dense": note("dense"), "bm25": note("bm25")},
        dense_by_query={"window": [("dense", 0.9)]},
        bm25_by_query={"window": [("bm25", 5.0)]},
    )

    result = await AmbientRetriever(
        index,
        UnusedReranker(),
        FakeClock(),
        span_extractor=span_extractor,
    ).on_window("window")

    assert result is engine.update
    assert engine.calls[0]["fused"] == [
        ("dense", pytest.approx(1 / (config.RRF_K + 1))),
        ("bm25", pytest.approx(1 / (config.RRF_K + 1))),
    ]
    assert engine.calls[0]["head_hits"] == {
        "dense": {"dense"},
        "bm25": {"bm25"},
    }
