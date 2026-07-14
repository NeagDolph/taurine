from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.models_local import (
    LocalModelUnavailable,
    ModelDeadlineExceeded,
    _QwenYesNoPairScorer,
    local_pair_reranker,
)
from taurine_core.retrieval.rerank import (
    LocalPassageReranker,
    OpenRouterRerankClient,
    PassageRerankCandidate,
    RerankError,
)


_ARTIFACT = (
    Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
)


class FakeScorer:
    def __init__(self, scores: tuple[float, ...] | Exception) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...], str, float | None]] = []

    def score_pairs(
        self,
        query: str,
        documents: tuple[str, ...],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> tuple[float, ...]:
        self.calls.append((query, documents, instruction, deadline_at))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


def _candidate(index: int, *, text: str | None = None) -> PassageRerankCandidate:
    return PassageRerankCandidate(
        passage_id=f"p{index:02d}",
        note_id=f"n{index:02d}",
        title=f"Title {index}",
        source="apple_notes",
        text=text or f"Exact source passage {index}",
        recall_score=1.0 - index / 100.0,
        hint="via fact F1" if index == 0 else "",
    )


def test_openrouter_rerank_client_posts_one_indexed_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.25},
                    {"index": 0, "relevance_score": 0.75},
                ]
            },
        )

    client = OpenRouterRerankClient(
        "secret-key",
        model_id="cohere/rerank-v3.5",
        transport=httpx.MockTransport(handler),
    )
    scores = client.score_pairs(
        "what shipped?",
        ("first document", "second document"),
        instruction="local-only prompt framing",
    )

    assert scores == pytest.approx((0.75, 0.25))
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/rerank"
    assert request.headers["Authorization"] == "Bearer secret-key"
    assert json.loads(request.content) == {
        "model": "cohere/rerank-v3.5",
        "query": "what shipped?",
        "documents": ["first document", "second document"],
        "top_n": 2,
    }


@pytest.mark.parametrize("status", [429, 500, 503])
def test_openrouter_rerank_client_retries_retryable_http_statuses(status: int) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(status, request=request, text="provider detail")
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"index": 0, "relevance_score": 0.8}]},
        )

    client = OpenRouterRerankClient(
        "secret",
        model_id="test-model",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    assert client.score_pairs("query", ("document",), instruction="ignored") == (0.8,)
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_openrouter_rerank_client_retries_transport_errors() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadTimeout("temporary rerank timeout", request=request)
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"index": 0, "relevance_score": 0.7}]},
        )

    client = OpenRouterRerankClient(
        "secret",
        model_id="test-model",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    assert client.score_pairs("query", ("document",), instruction="ignored") == (0.7,)
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_openrouter_rerank_client_does_not_retry_non_retryable_status() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request, text="secret provider detail")

    client = OpenRouterRerankClient(
        "secret",
        model_id="test-model",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    with pytest.raises(RerankError, match="HTTP 400") as raised:
        client.score_pairs("query", ("document",), instruction="ignored")
    assert attempts == 1
    assert sleeps == []
    assert "provider detail" not in str(raised.value)


def test_openrouter_rerank_client_exhausts_three_attempts_with_sanitized_error() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, request=request, text="secret provider detail")

    client = OpenRouterRerankClient(
        "secret-key",
        model_id="test-model",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    with pytest.raises(RerankError, match="HTTP 500") as raised:
        client.score_pairs("query", ("document",), instruction="ignored")
    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"results": []}, "wrong row count"),
        (
            {
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.7},
                ]
            },
            "invalid row indexes",
        ),
        (
            {"results": [{"index": 0, "relevance_score": float("nan")}]},
            "non-finite",
        ),
    ],
)
def test_openrouter_rerank_client_rejects_invalid_success_payloads(
    payload: object,
    message: str,
) -> None:
    client = OpenRouterRerankClient(
        "secret",
        model_id="test-model",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
                content=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
        ),
    )
    document_count = 2 if "indexes" in message else 1

    with pytest.raises(RerankError, match=message):
        client.score_pairs(
            "query",
            tuple(f"document-{index}" for index in range(document_count)),
            instruction="ignored",
        )


def test_openrouter_rerank_client_requires_api_key() -> None:
    with pytest.raises(RerankError, match="OPEN_ROUTER_API_KEY"):
        OpenRouterRerankClient("", model_id="test-model")


def test_passage_reranker_scores_exact_passages_and_applies_mode_floor() -> None:
    # floor semantics use a pinned literal: the shipping artifact's rerank
    # floors are data-fitted per corpus and no longer suit boundary fixtures
    scorer = FakeScorer((0.9, 0.4))
    candidates = (
        _candidate(0, text="The exact first source text."),
        _candidate(1, text="The exact second source text."),
    )

    result = LocalPassageReranker(
        scorer,
        mode="live",
        acceptance_floor=0.76,
    ).rerank("What was decided?", candidates)

    assert [row.candidate.passage_id for row in result.accepted] == ["p00"]
    assert result.decisions[1].reason == "below_passage_rerank_floor"
    documents = scorer.calls[0][1]
    assert "The exact first source text." in documents[0]
    assert "Title: Title 0" in documents[0]
    assert "Source: apple_notes" in documents[0]
    assert "via fact F1" in documents[0]
    assert "exact source passage" in scorer.calls[0][2]
    assert "canonical fact triple" not in scorer.calls[0][2]


@pytest.mark.parametrize("mode", ["live", "chat", "tool"])
def test_passage_reranker_caps_scoring_at_24_and_preserves_unscored_tail(
    mode: str,
) -> None:
    artifact = load_calibration(_ARTIFACT)
    scorer = FakeScorer(tuple(0.9 for _ in range(24)))
    candidates = tuple(
        replace(
            _candidate(index),
            recall_score=(1.0 if index in {0, 1} else 1.0 - index / 100.0),
        )
        for index in reversed(range(25))
    )
    floor = (
        artifact.rerank.live_floor if mode == "live" else artifact.rerank.chat_floor
    )

    result = LocalPassageReranker(
        scorer, mode=mode, acceptance_floor=floor  # type: ignore[arg-type]
    ).rerank("query", candidates)

    assert len(scorer.calls[0][1]) == 24
    assert [
        document.split("Passage:\n", 1)[1].splitlines()[0]
        for document in scorer.calls[0][1]
    ] == [
        f"Exact source passage {index}" for index in range(24)
    ]
    assert len(result.decisions) == 25
    assert [row.candidate.passage_id for row in result.decisions] == [
        f"p{index:02d}" for index in range(25)
    ]
    assert len(result.accepted) == 24
    assert result.decisions[-1].score == 0.0
    assert result.decisions[-1].accepted is False
    assert result.decisions[-1].reason == "outside_rerank_candidate_cap"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (LocalModelUnavailable("missing"), "local_reranker_unavailable"),
        (ModelDeadlineExceeded("late"), "local_reranker_deadline"),
    ],
)
def test_passage_reranker_fails_closed_with_degraded_status(
    error: Exception, reason: str
) -> None:
    result = LocalPassageReranker(
        FakeScorer(error), mode="live", acceptance_floor=0.76
    ).rerank("query", (_candidate(0),))

    assert result.degraded
    assert result.reason == reason
    assert result.accepted == ()
    assert result.decisions[0].reason == reason


def test_passage_reranker_rejects_malformed_scorer_cardinality() -> None:
    result = LocalPassageReranker(
        FakeScorer(()), mode="chat", acceptance_floor=0.66
    ).rerank("query", (_candidate(0),))

    assert result.degraded
    assert result.reason == "local_reranker_invalid_output"
    assert result.accepted == ()


def test_local_pair_reranker_fails_offline_when_bundle_is_absent() -> None:
    with pytest.raises(LocalModelUnavailable, match="bundled local model"):
        local_pair_reranker("tests/definitely-not-a-downloaded-reranker")


def test_local_pair_scorer_microbatches_bounded_documents(monkeypatch) -> None:
    scorer = object.__new__(_QwenYesNoPairScorer)
    calls: list[tuple[str, ...]] = []

    def score_batch(instruction: str, query: str, documents: tuple[str, ...]):
        assert instruction == "passage-task"
        assert query == "query"
        calls.append(documents)
        return tuple(0.5 for _ in documents)

    monkeypatch.setattr(scorer, "_score_batch", score_batch)

    scores = scorer.score_pairs(
        "query",
        tuple(f"document-{index}" for index in range(17)),
        instruction="passage-task",
    )

    assert [len(batch) for batch in calls] == [8, 8, 1]
    assert scores == tuple(0.5 for _ in range(17))
