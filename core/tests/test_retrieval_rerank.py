from __future__ import annotations

from pathlib import Path

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
    PassageRerankCandidate,
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


@pytest.mark.parametrize(("mode", "cap"), [("live", 30), ("chat", 60), ("tool", 60)])
def test_passage_reranker_enforces_mode_candidate_cap(mode: str, cap: int) -> None:
    artifact = load_calibration(_ARTIFACT)
    scorer = FakeScorer(tuple(0.9 for _ in range(cap)))
    candidates = tuple(_candidate(index) for index in range(75))
    floor = (
        artifact.rerank.live_floor if mode == "live" else artifact.rerank.chat_floor
    )

    result = LocalPassageReranker(
        scorer, mode=mode, acceptance_floor=floor  # type: ignore[arg-type]
    ).rerank("query", candidates)

    assert len(scorer.calls[0][1]) == cap
    assert len(result.decisions) == cap
    assert [row.candidate.passage_id for row in result.decisions] == [
        f"p{index:02d}" for index in range(cap)
    ]


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
