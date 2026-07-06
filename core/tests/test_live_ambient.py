from __future__ import annotations

import pytest

from taurine_core import config
from taurine_core.live.ambient import AmbientRetriever
from taurine_core.ports import RerankError

from tests.fakes import FakeClock, FakeIndex, note


class OkReranker:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}

    async def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        del query
        default = float(len(docs))
        ranked: list[tuple[str, float]] = []
        for index, (note_id, _doc) in enumerate(docs):
            ranked.append((note_id, self.scores.get(note_id, default - index)))
        return sorted(ranked, key=lambda row: row[1], reverse=True)


class FailingReranker:
    async def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        del query, docs
        raise RerankError("rerank unavailable")


class RecordingReranker(OkReranker):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        self.calls.append((query, docs))
        return await super().rerank(query, docs)


class QueryAwareIndex(FakeIndex):
    def __init__(
        self,
        *,
        dense_by_query: dict[str, list[tuple[str, float]]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dense_by_query = dense_by_query
        self.dense_queries: list[str] = []

    def dense_search(self, text: str, k: int) -> list[tuple[str, float]]:
        self.dense_queries.append(text)
        return self.dense_by_query.get(text, [])[:k]


@pytest.fixture
def ok_reranker() -> OkReranker:
    return OkReranker()


@pytest.fixture
def failing_reranker() -> FailingReranker:
    return FailingReranker()


@pytest.fixture
def fake_index() -> FakeIndex:
    return FakeIndex(
        notes={
            "seed": note("seed", "Seed Topic"),
            "neighbor": note("neighbor", "Neighbor Topic"),
        },
        dense=[("seed", 0.91)],
        links={"seed": [("neighbor", 0.8)]},
        chunks={
            "seed": "seed chunk",
            "neighbor": "neighbor chunk",
        },
    )


@pytest.fixture
def fake_index_dualpath() -> FakeIndex:
    return FakeIndex(
        notes={
            "seed": note("seed", "Seed Topic"),
            "dual": note("dual", "Dual Path"),
        },
        dense=[("seed", 0.91), ("dual", 0.72)],
        links={"seed": [("dual", 0.8)]},
        chunks={
            "seed": "seed chunk",
            "dual": "dual chunk",
        },
    )


async def test_expanded_neighbor_joins_pool_with_decayed_score(
    fake_index: FakeIndex,
    ok_reranker: OkReranker,
) -> None:
    # fake_index: "seed" ranks #1 by dense; "neighbor" linked w/ weight 0.8, absent from search
    r = AmbientRetriever(fake_index, ok_reranker, FakeClock())
    res = await r.on_window("talk about the seed topic")

    assert res is not None
    exp = [e for e in res.expansions if e.note_id == "neighbor"]
    assert exp and exp[0].hop == 1 and exp[0].from_note_id == "seed"
    assert exp[0].score == pytest.approx((2 / (config.RRF_K + 1)) * 0.8 * config.HOP_DECAY)
    assert [
        (item.note_id, item.via_note_id, item.hop)
        for item in r.last_slate_adds
        if item.note_id == "neighbor"
    ] == [("neighbor", "seed", 1)]


async def test_rerank_failure_falls_back_to_fused_order(
    fake_index: FakeIndex,
    failing_reranker: FailingReranker,
) -> None:
    res = await AmbientRetriever(fake_index, failing_reranker, FakeClock()).on_window("x")

    assert res is not None and res.hits
    assert res.hits[0].note_id == "seed"


async def test_direct_beats_expanded_for_dual_path_note(
    fake_index_dualpath: FakeIndex,
    ok_reranker: OkReranker,
) -> None:
    res = await AmbientRetriever(
        fake_index_dualpath,
        ok_reranker,
        FakeClock(),
    ).on_window("dual path topic")

    assert res is not None
    assert all(e.note_id != "dual" for e in res.expansions)


async def test_throttled_window_returns_none_and_no_slate_adds(
    fake_index: FakeIndex,
    ok_reranker: OkReranker,
) -> None:
    clock = FakeClock()
    retriever = AmbientRetriever(fake_index, ok_reranker, clock)

    assert await retriever.on_window("first") is not None
    assert retriever.last_slate_adds
    assert await retriever.on_window("second") is None
    assert retriever.last_slate_adds == []

    clock.advance(config.C2_THROTTLE_S)
    assert await retriever.on_window("third") is not None


async def test_rerank_docs_include_title_gist_best_chunk_and_limit() -> None:
    note_ids = [f"n{idx}" for idx in range(config.RERANK_MAX_CANDIDATES + 5)]
    index = FakeIndex(
        notes={note_id: note(note_id, f"Title {note_id}", f"Gist {note_id}") for note_id in note_ids},
        dense=[(note_id, 0.9 - idx * 0.01) for idx, note_id in enumerate(note_ids)],
        chunks={note_id: f"best chunk {note_id}" for note_id in note_ids},
    )
    reranker = RecordingReranker()

    await AmbientRetriever(index, reranker, FakeClock()).on_window("window text")

    [(query, docs)] = reranker.calls
    assert query == "window text"
    assert len(docs) == config.RERANK_MAX_CANDIDATES
    assert docs[0] == ("n0", "Title n0 — Gist n0\nbest chunk n0")


async def test_min_cosine_floor_only_filters_dense_only_low_scores() -> None:
    index = FakeIndex(
        notes={
            "seed": note("seed"),
            "low": note("low"),
            "bm25-low": note("bm25-low"),
            "expanded-low": note("expanded-low"),
        },
        dense=[("seed", 0.9), ("low", 0.1), ("bm25-low", 0.1)],
        bm25=[("bm25-low", 12.0)],
        links={"seed": [("expanded-low", 0.8)]},
    )
    reranker = OkReranker(
        {
            "low": 0.99,
            "bm25-low": 0.8,
            "expanded-low": 0.7,
            "seed": 0.6,
        }
    )

    retriever = AmbientRetriever(index, reranker, FakeClock())
    res = await retriever.on_window("floor query")

    assert res is not None
    hit_ids = [hit.note_id for hit in res.hits]
    assert "low" not in hit_ids
    assert "bm25-low" in hit_ids
    assert "expanded-low" in hit_ids
    assert "low" not in [item.note_id for item in retriever.last_slate_adds]


async def test_debug_callback_reports_one_c2_pass_with_internal_rows() -> None:
    debug_records: list[dict[str, object]] = []
    index = FakeIndex(
        notes={
            "seed": note("seed", "Seed Topic"),
            "low": note("low", "Low Dense"),
            "expanded": note("expanded", "Expanded Topic"),
        },
        dense=[("seed", 0.9), ("low", 0.1)],
        bm25=[("seed", 5.0)],
        links={"seed": [("expanded", 0.8)]},
        chunks={
            "seed": "seed chunk",
            "low": "low chunk",
            "expanded": "expanded chunk",
        },
    )
    reranker = OkReranker({"low": 0.99, "seed": 0.7, "expanded": 0.6})

    res = await AmbientRetriever(
        index,
        reranker,
        FakeClock(),
        on_debug=debug_records.append,
    ).on_window("seed topic")

    assert res is not None
    assert len(debug_records) == 1
    debug = debug_records[0]
    assert debug["window"] == "seed topic"
    assert debug["salient"] == "seed topic"
    assert debug["dense"] == [["seed", 0.9], ["low", 0.1]]
    assert debug["dense_salient"] == [["seed", 0.9], ["low", 0.1]]
    assert debug["bm25"] == [["seed", 5.0]]
    assert debug["fused"] == [
        ["seed", pytest.approx((1 / (config.RRF_K + 1)) * 3)],
        ["low", pytest.approx(2 / (config.RRF_K + 2))],
    ]
    assert debug["expansions"] == [
        {
            "from": "seed",
            "to": "expanded",
            "score": pytest.approx((1 / (config.RRF_K + 1)) * 3 * 0.8 * config.HOP_DECAY),
        }
    ]
    assert debug["rerank_used"] is True
    assert debug["rerank_error"] is None
    assert debug["ordered"][:3] == [["low", 0.99], ["seed", 0.7], ["expanded", 0.6]]
    assert debug["floor_dropped"] == ["low"]
    assert debug["slate_adds"] == ["seed", "expanded"]


async def test_salient_query_uses_rare_terms_and_dense_salient_joins_rrf() -> None:
    window = "alpha beta gamma delta epsilon zeta theta"
    salient = "beta gamma delta epsilon zeta theta"
    index = QueryAwareIndex(
        notes={
            "window": note("window", "Window Note"),
            "salient": note("salient", "Salient Note"),
            "bm": note("bm", "BM Note"),
        },
        dense_by_query={
            window: [("window", 0.9)],
            salient: [("salient", 0.88), ("window", 0.5)],
        },
        bm25=[("bm", 5.0)],
        chunks={
            "window": "window chunk",
            "salient": "salient chunk",
            "bm": "bm chunk",
        },
        doc_frequencies={
            "alpha": 10,
            "beta": 0,
            "gamma": 1,
            "delta": 0,
            "epsilon": 2,
            "zeta": 3,
            "theta": 0,
        },
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker({"salient": 0.99, "window": 0.8, "bm": 0.7}),
        FakeClock(),
        on_debug=debug_records.append,
    ).on_window(window)

    assert res is not None
    assert index.dense_queries == [window, salient]
    assert [hit.note_id for hit in res.hits] == ["salient", "window", "bm"]
    debug = debug_records[0]
    assert debug["salient"] == salient
    assert debug["dense"] == [["window", 0.9]]
    assert debug["dense_salient"] == [["salient", 0.88], ["window", 0.5]]
    assert debug["fused"] == [
        ["window", pytest.approx((1 / (config.RRF_K + 1)) + (1 / (config.RRF_K + 2)))],
        ["salient", pytest.approx(1 / (config.RRF_K + 1))],
        ["bm", pytest.approx(1 / (config.RRF_K + 1))],
    ]
