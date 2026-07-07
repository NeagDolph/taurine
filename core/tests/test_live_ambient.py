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
        doc_frequencies={"seed": 1, "topic": 1},
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


async def test_fused_protect_rescues_fused_top_note_demoted_by_rerank() -> None:
    ordered_ids = [f"ordered-{idx}" for idx in range(1, config.C2_TOP_K + 2)]
    note_ids = ["protected", *ordered_ids]
    index = FakeIndex(
        notes={note_id: note(note_id, note_id.title()) for note_id in note_ids},
        dense=[
            ("protected", 0.95),
            *[
                (note_id, 0.9 - idx * 0.01)
                for idx, note_id in enumerate(ordered_ids)
            ],
        ],
        chunks={note_id: f"chunk {note_id}" for note_id in note_ids},
    )
    rerank_scores = {
        note_id: float(config.C2_TOP_K + 2 - idx)
        for idx, note_id in enumerate(ordered_ids, start=1)
    }
    rerank_scores["protected"] = 0.1
    debug_records: list[dict[str, object]] = []

    retriever = AmbientRetriever(
        index,
        OkReranker(rerank_scores),
        FakeClock(),
        on_debug=debug_records.append,
    )
    res = await retriever.on_window("protect retrieval")

    assert res is not None
    slate_ids = [item.note_id for item in retriever.last_slate_adds]
    assert "protected" in slate_ids
    assert ordered_ids[config.C2_TOP_K - 1] not in slate_ids
    assert len(slate_ids) == config.C2_TOP_K
    protected_hit = next(hit for hit in res.hits if hit.note_id == "protected")
    assert protected_hit.rank == config.C2_TOP_K + 2
    assert protected_hit.score == pytest.approx(0.1)
    protected_score = next(
        item.score for item in retriever.last_slate_adds if item.note_id == "protected"
    )
    assert protected_score == pytest.approx(0.1)
    assert debug_records[0]["protected"] == ["protected"]
    assert debug_records[0]["slate_adds"][-1] == "protected"


async def test_dense_protect_rescues_dense_window_note_buried_by_fusion_and_rerank() -> None:
    ordered_ids = [f"ordered-{idx}" for idx in range(1, config.C2_TOP_K + 2)]
    dense_protected = "dense-protected"
    window = "protect dense candidate"
    index = QueryAwareIndex(
        notes={
            note_id: note(note_id, note_id.title())
            for note_id in [*ordered_ids, dense_protected]
        },
        dense_by_query={
            window: [
                (ordered_ids[0], 0.99),
                (dense_protected, 0.98),
                *[
                    (note_id, 0.9 - idx * 0.01)
                    for idx, note_id in enumerate(ordered_ids[1:], start=1)
                ],
            ],
        },
        bm25=[(note_id, 20.0 - idx) for idx, note_id in enumerate(ordered_ids)],
        chunks={note_id: f"chunk {note_id}" for note_id in [*ordered_ids, dense_protected]},
    )
    rerank_scores = {
        note_id: float(config.C2_TOP_K + 2 - idx)
        for idx, note_id in enumerate(ordered_ids, start=1)
    }
    rerank_scores[dense_protected] = 0.1
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker(rerank_scores),
        FakeClock(),
        on_debug=debug_records.append,
    ).on_window(window)

    assert res is not None
    slate_ids = [item.note_id for item in res.hits]
    assert dense_protected in slate_ids
    assert ordered_ids[config.C2_TOP_K - 1] not in slate_ids
    assert len(slate_ids) == config.C2_TOP_K
    assert debug_records[0]["protected"] == [dense_protected]
    assert debug_records[0]["slate_adds"][-1] == dense_protected


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
        doc_frequencies={"seed": 1, "topic": 1},
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
        ["expanded", pytest.approx(1 / (config.RRF_K + 1))],
    ]
    assert debug["graph_vote"] == [
        [
            "expanded",
            pytest.approx((1 / (config.RRF_K + 1)) * 3 * 0.8 * config.GRAPH_VOTE_WEIGHT),
        ]
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


async def test_salient_query_excludes_terms_below_df_floor_and_dense_salient_joins_rrf() -> None:
    window = "alpha beta gamma delta epsilon zeta theta"
    salient = "alpha gamma epsilon zeta"
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


async def test_salient_query_skips_dense_salient_when_df_floor_removes_all_terms() -> None:
    window = "mored armand shopif"
    index = QueryAwareIndex(
        notes={"window": note("window", "Window Note")},
        dense_by_query={window: [("window", 0.9)]},
        chunks={"window": "window chunk"},
        doc_frequencies={"mored": 0, "armand": 0, "shopif": 0},
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker({"window": 0.8}),
        FakeClock(),
        on_debug=debug_records.append,
    ).on_window(window)

    assert res is not None
    assert index.dense_queries == [window]
    assert debug_records[0]["salient"] is None
    assert debug_records[0]["dense_salient"] == []


async def test_graph_vote_adds_fifth_rrf_list_and_lifts_direct_neighbor() -> None:
    index = FakeIndex(
        notes={
            "seed-a": note("seed-a", "Seed A"),
            "seed-b": note("seed-b", "Seed B"),
            "filler-1": note("filler-1", "Filler 1"),
            "filler-2": note("filler-2", "Filler 2"),
            "voted": note("voted", "Voted"),
        },
        dense=[
            ("seed-a", 0.95),
            ("seed-b", 0.94),
            ("filler-1", 0.93),
            ("filler-2", 0.92),
            ("voted", 0.91),
        ],
        bm25=[
            ("seed-a", 12.0),
            ("seed-b", 11.0),
            ("filler-1", 10.0),
            ("filler-2", 9.0),
        ],
        links={
            "seed-a": [("voted", 1.0)],
            "seed-b": [("voted", 1.0)],
        },
        chunks={
            "seed-a": "seed a chunk",
            "seed-b": "seed b chunk",
            "filler-1": "filler 1 chunk",
            "filler-2": "filler 2 chunk",
            "voted": "voted chunk",
        },
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker(),
        FakeClock(),
        on_debug=debug_records.append,
    ).on_window("graph vote")

    assert res is not None
    debug = debug_records[0]
    assert debug["graph_vote"] == [
        [
            "voted",
            pytest.approx(
                (
                    (2 / (config.RRF_K + 1))
                    + (2 / (config.RRF_K + 2))
                )
                * config.GRAPH_VOTE_WEIGHT
            ),
        ]
    ]
    assert [note_id for note_id, _score in debug["fused"][:3]] == [
        "seed-a",
        "seed-b",
        "voted",
    ]


async def test_span_queries_form_dense_gliner_list_and_survive_floor() -> None:
    window = "alpha checkout"
    span_extractor = FakeSpanExtractor(
        [("shopify checkout", 0.95, "topic"), ("Ashish", 0.9, "person")]
    )
    index = QueryAwareIndex(
        notes={
            "window": note("window", "Window Note"),
            "span-only": note("span-only", "Span Note"),
        },
        dense_by_query={
            window: [("window", 0.9)],
            "shopify checkout": [("span-only", 0.94), ("window", 0.4)],
            "Ashish": [("span-only", 0.7)],
        },
        chunks={
            "window": "window chunk",
            "span-only": "span chunk",
        },
        doc_frequencies={"alpha": 1, "checkout": 1},
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker({"span-only": 0.99, "window": 0.8}),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window)

    assert res is not None
    assert span_extractor.calls == [window]
    assert index.dense_queries == [window, window, "shopify checkout", "Ashish"]
    assert res.extracted == ["shopify checkout", "Ashish"]
    assert [hit.note_id for hit in res.hits] == ["span-only", "window"]
    debug = debug_records[0]
    assert debug["gliner_spans"] == [["shopify checkout", 0.95, "topic"], ["Ashish", 0.9, "person"]]
    assert debug["dense_gliner"] == [["span-only", 0.94], ["window", 0.4]]
    assert debug["fused"] == [
        ["window", pytest.approx((1 / (config.RRF_K + 1)) * 2 + (1 / (config.RRF_K + 2)))],
        ["span-only", pytest.approx(1 / (config.RRF_K + 1))],
    ]


async def test_span_context_query_uses_context_text_before_window_and_logs_drops() -> None:
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
        chunks={"context": "context chunk"},
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker({"context": 0.91}),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window, context_text=context)

    assert res is not None
    assert span_extractor.calls == [window]
    assert index.dense_queries == [window, "Shopify", context]
    assert [hit.note_id for hit in res.hits] == ["context"]
    debug = debug_records[0]
    assert debug["gliner_context_queries"] == [context]
    assert debug["gliner_dropped"] == [["dude", 0.7, "person"]]


async def test_span_context_query_caps_around_span(monkeypatch) -> None:
    monkeypatch.setattr(config, "SPAN_CONTEXT_MAX_CHARS", 64)
    window = "Deepgram follow-up"
    context = (
        "Earlier alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. "
        "Then the roadmap team mentioned Shopify while comparing checkout operations "
        "against a very long implementation timeline. "
        "After that they moved to Deepgram follow-up details."
    )
    span_extractor = FakeSpanExtractor([("Shopify", 0.96, "organization")])
    index = QueryAwareIndex(dense_by_query={window: [], "Shopify": []})
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker(),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window, context_text=context)

    assert res is not None
    [context_query] = debug_records[0]["gliner_context_queries"]
    assert len(context_query) <= 64
    assert "Shopify" in context_query
    assert context_query in index.dense_queries


async def test_span_context_query_skips_absent_span() -> None:
    window = "Deepgram follow-up"
    span_extractor = FakeSpanExtractor([("Shopify", 0.96, "organization")])
    index = QueryAwareIndex(dense_by_query={window: [], "Shopify": []})
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker(),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window, context_text="No vendor appears in this context.")

    assert res is not None
    assert index.dense_queries == [window, "Shopify"]
    assert debug_records[0]["gliner_context_queries"] == []


async def test_fuzzy_snap_queries_original_and_snapped_span_and_logs_snap() -> None:
    window = "ask armand"
    span_extractor = FakeSpanExtractor([("armand", 0.95, "person")])
    index = QueryAwareIndex(
        notes={"armaan": note("armaan", "Armaan Talk")},
        dense_by_query={
            window: [],
            "armand": [],
            "armaan": [("armaan", 0.92)],
        },
        alias_rows=[("armaan talk", "armaan", True)],
        chunks={"armaan": "Armaan Talk chunk"},
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker({"armaan": 0.88}),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window)

    assert res is not None
    assert res.extracted == ["armand"]
    assert "armaan" in index.dense_queries
    assert [hit.note_id for hit in res.hits] == ["armaan"]
    debug = debug_records[0]
    assert debug["snapped"] == [["armand", "armaan"]]
    assert debug["dense_gliner"] == [["armaan", 0.92]]


@pytest.mark.parametrize(
    ("span", "label"),
    [
        ("armand", "topic"),
        ("xylophone", "person"),
    ],
)
async def test_fuzzy_snap_skips_ineligible_labels_and_distant_spans(
    span: str,
    label: str,
) -> None:
    window = f"ask {span}"
    span_extractor = FakeSpanExtractor([(span, 0.95, label)])
    index = QueryAwareIndex(
        notes={"armaan": note("armaan", "Armaan Talk")},
        dense_by_query={
            window: [],
            span: [],
            "armaan": [("armaan", 0.92)],
        },
        alias_rows=[("armaan talk", "armaan", True)],
        chunks={"armaan": "Armaan Talk chunk"},
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker({"armaan": 0.88}),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window)

    assert res is not None
    assert res.hits == []
    assert "armaan" not in index.dense_queries
    assert "snapped" not in debug_records[0]
    assert "bm25_query" not in debug_records[0]


async def test_snapped_terms_extend_bm25_query_and_log_query() -> None:
    window = "ask armand about timeline"
    span_extractor = FakeSpanExtractor([("armand", 0.95, "person")])
    bm25_query = f"{window} armaan"
    index = QueryAwareIndex(
        notes={"bm25": note("bm25", "Armaan Talk")},
        dense_by_query={
            window: [],
            "armand": [],
            "armaan": [],
        },
        bm25_by_query={bm25_query: [("bm25", 4.0)]},
        alias_rows=[("armaan talk", "bm25", True)],
        chunks={"bm25": "Armaan Talk chunk"},
    )
    debug_records: list[dict[str, object]] = []

    res = await AmbientRetriever(
        index,
        OkReranker({"bm25": 0.77}),
        FakeClock(),
        on_debug=debug_records.append,
        span_extractor=span_extractor,
    ).on_window(window)

    assert res is not None
    assert index.bm25_queries == [bm25_query]
    assert [hit.note_id for hit in res.hits] == ["bm25"]
    debug = debug_records[0]
    assert debug["bm25_query"] == bm25_query
    assert debug["bm25"] == [["bm25", 4.0]]


async def test_empty_span_extractor_degrades_to_three_list_rrf() -> None:
    window = "alpha beta gamma delta epsilon zeta theta"
    salient = "alpha gamma epsilon zeta"
    span_extractor = FakeSpanExtractor([])
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
        span_extractor=span_extractor,
    ).on_window(window)

    assert res is not None
    assert span_extractor.calls == [window]
    assert index.dense_queries == [window, salient]
    assert res.extracted == []
    debug = debug_records[0]
    assert debug["gliner_spans"] == []
    assert debug["dense_gliner"] == []
    assert debug["fused"] == [
        ["window", pytest.approx((1 / (config.RRF_K + 1)) + (1 / (config.RRF_K + 2)))],
        ["salient", pytest.approx(1 / (config.RRF_K + 1))],
        ["bm", pytest.approx(1 / (config.RRF_K + 1))],
    ]
