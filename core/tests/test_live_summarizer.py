from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from taurine_core import config
from taurine_core.live.grounding import GroundingUpdate
from taurine_core.live.summarizer import Summarizer
from taurine_core.ports import LLMError
from taurine_core.schemas import (
    Actionable,
    GroundedFact,
    GroundingItem,
    MeetingSummary,
)

from tests.fakes import FakeIndex


@dataclass
class Call:
    model: str
    system: str
    user: str
    timeout_s: float | None


class FakeLLM:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[Call] = []

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append(Call(model, system, user, timeout_s))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


async def test_no_update_passthrough_records_model_and_latency() -> None:
    llm = FakeLLM([{"no_update": True}])
    summarizer = Summarizer(llm, FakeIndex())  # type: ignore[arg-type]

    result = await summarizer.summarize(None, "minor chatter", _grounding(), [])

    assert result is not None
    assert result.summary is None
    assert result.no_update is True
    assert result.model == config.SUMMARY_MODEL
    assert result.latency_ms >= 0.0
    assert [(call.model, call.timeout_s) for call in llm.calls] == [
        (config.SUMMARY_MODEL, config.SUMMARY_TIMEOUT_S)
    ]


async def test_unknown_citations_are_stripped_without_rejecting_when_under_limit() -> None:
    llm = FakeLLM(
        [
            {
                "summary_md": "Valid [N1] fact [F1] other [N2] bad [N99]",
                "highlights": [{"fact_id": "F1", "why": "relevant"}],
                "actionables": [],
            }
        ]
    )
    summarizer = Summarizer(llm, FakeIndex())  # type: ignore[arg-type]

    result = await summarizer.summarize(None, "delta", _grounding(), [])

    assert result is not None
    assert result.no_update is False
    assert result.summary is not None
    assert "[N99]" not in result.summary.summary_md
    assert result.summary.summary_md.count("[") == 3
    assert result.summary.rev == 1
    assert result.summary.highlights[0].fact_id == "F1"


async def test_invalid_citation_ratio_rejects_pass_distinct_from_no_update() -> None:
    llm = FakeLLM(
        [
            {
                "summary_md": "Bad [N999] worse [F999] good [N1]",
                "highlights": [{"fact_id": "F404", "why": "unknown"}],
                "actionables": [],
            }
        ]
    )
    summarizer = Summarizer(llm, FakeIndex())  # type: ignore[arg-type]

    result = await summarizer.summarize(None, "delta", _grounding(), [])

    assert result is not None
    assert result.summary is None
    assert result.no_update is False
    assert result.rejected is True
    assert result.reject_reason == "invalid_citations"


async def test_actionables_apply_quote_confidence_grounding_gates_and_cap() -> None:
    transcript = (
        "I'll send Ashish the integration timeline before Friday. "
        "Maya will draft the launch checklist today. "
        "Jordan will update the rollout tracker tomorrow. "
        "Taylor will write the QA plan by Monday."
    )
    llm = FakeLLM(
        [
            {
                "summary_md": "Timeline is now owned [N1][F1]",
                "highlights": [],
                "actionables": [
                    _actionable_payload(
                        title="Send Ashish the integration timeline",
                        quote="I'll send Ashish the integration timeline before Friday",
                        confidence=0.95,
                        grounding=["N1", "BAD", "F1"],
                    ),
                    _actionable_payload(
                        title="Draft launch checklist",
                        quote="Maya will draft the launch checklist today",
                        confidence=0.91,
                        grounding=["N2"],
                    ),
                    _actionable_payload(
                        title="Update rollout tracker",
                        quote="Jordan will update the rollout tracker tomorrow",
                        confidence=0.89,
                        grounding=["N1"],
                    ),
                    _actionable_payload(
                        title="Fabricated commitment",
                        quote="This quote was never said in the transcript",
                        confidence=0.99,
                        grounding=["N1"],
                    ),
                    _actionable_payload(
                        title="Low confidence",
                        quote="Taylor will write the QA plan by Monday",
                        confidence=config.ACTIONABLE_MIN_CONF - 0.01,
                        grounding=["N1"],
                    ),
                    _actionable_payload(
                        title="Unknown grounding",
                        quote="Taylor will write the QA plan by Monday",
                        confidence=0.92,
                        grounding=["NOPE"],
                    ),
                ],
            }
        ]
    )
    summarizer = Summarizer(llm, FakeIndex())  # type: ignore[arg-type]

    result = await summarizer.summarize(None, transcript, _grounding(), [])

    assert result is not None
    assert result.summary is not None
    assert [item.title for item in result.new_actionables] == [
        "Send Ashish the integration timeline",
        "Draft launch checklist",
    ]
    assert result.new_actionables[0].grounding == ["N1", "F1"]
    assert all(re.fullmatch(r"act-[0-9a-f]{8}", item.id) for item in result.new_actionables)
    assert [item.id for item in result.summary.actionables] == [
        item.id for item in result.new_actionables
    ]


async def test_dedupe_updates_existing_actionable_and_open_items_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        vectors = [
            [1.0, 0.0] if "timeline" in text.lower() else [0.0, 1.0]
            for text in texts
        ]
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr("taurine_core.indexer.store.embed_texts", fake_embed_texts)
    open_items = [
        Actionable(
            id="act-existing",
            title="Send Ashish the integration timeline",
            quote="old quote",
            status="open",
        ),
        Actionable(
            id="act-other",
            title="Confirm launch room",
            quote="old room quote",
            status="confirmed",
        ),
    ]
    llm = FakeLLM(
        [
            {
                "summary_md": "Timeline is updated [N1]",
                "highlights": [],
                "actionables": [
                    _actionable_payload(
                        title="Send Ashish the integration timeline",
                        quote="I'll send Ashish the integration timeline before Friday",
                        confidence=0.95,
                        grounding=["N1"],
                    )
                ],
            }
        ]
    )
    summarizer = Summarizer(llm, FakeIndex())  # type: ignore[arg-type]

    result = await summarizer.summarize(
        MeetingSummary(rev=4, summary_md="old"),
        "I'll send Ashish the integration timeline before Friday",
        _grounding(),
        open_items,
    )

    assert result is not None
    assert result.summary is not None
    assert result.summary.rev == 5
    assert result.new_actionables == []
    assert [item.id for item in result.updated_actionables] == ["act-existing"]
    assert [(item.id, item.status) for item in result.summary.actionables] == [
        ("act-existing", "open"),
        ("act-other", "confirmed"),
    ]
    assert result.summary.actionables[0].quote == (
        "I'll send Ashish the integration timeline before Friday"
    )


async def test_fallback_model_is_used_after_primary_llm_error() -> None:
    llm = FakeLLM(
        [
            LLMError("primary unavailable"),
            {"summary_md": "Fallback summary [N1]", "highlights": [], "actionables": []},
        ]
    )
    summarizer = Summarizer(llm, FakeIndex())  # type: ignore[arg-type]

    result = await summarizer.summarize(None, "delta", _grounding(), [])

    assert result is not None
    assert result.summary is not None
    assert result.model == config.SUMMARY_FALLBACK_MODEL
    assert [call.model for call in llm.calls] == [
        config.SUMMARY_MODEL,
        config.SUMMARY_FALLBACK_MODEL,
    ]


async def test_both_models_fail_returns_none() -> None:
    llm = FakeLLM([LLMError("primary"), LLMError("fallback")])
    summarizer = Summarizer(llm, FakeIndex())  # type: ignore[arg-type]

    assert await summarizer.summarize(None, "delta", _grounding(), []) is None


def _grounding() -> GroundingUpdate:
    return GroundingUpdate(
        revision=3,
        items=[
            GroundingItem(
                note_id="n1",
                short_id="N1",
                title="Integration timeline",
                gist="Timeline details",
                source="apple_notes",
                age_days=12.0,
                score=0.92,
            ),
            GroundingItem(
                note_id="n2",
                short_id="N2",
                title="Launch checklist",
                gist="Checklist details",
                source="obsidian",
                age_days=90.0,
                score=0.81,
            ),
        ],
        facts=[GroundedFact(fact_id="F1", src="n1", dst="n2", text="Timeline feeds launch")],
    )


def _actionable_payload(
    *,
    title: str,
    quote: str,
    confidence: float,
    grounding: list[str],
) -> dict[str, Any]:
    return {
        "title": title,
        "quote": quote,
        "assignee": "me",
        "due": "Friday",
        "grounding": grounding,
        "why_new": "explicit commitment",
        "confidence": confidence,
    }
