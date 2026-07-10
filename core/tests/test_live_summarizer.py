from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live.grounding import GroundingUpdate
from taurine_core.live.summarizer import Summarizer
from taurine_core.ports import LLMError
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePassage,
    EvidencePath,
    RetrievalDecision,
)
from taurine_core.retrieval.validate import EvidenceValidator

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


class Entails:
    def score(self, premise: str, hypothesis: str) -> float:
        del premise, hypothesis
        return 0.99


class Contradicts:
    def score(self, premise: str, hypothesis: str) -> float:
        del premise, hypothesis
        return 0.01


async def test_prompt_supplies_exact_t_e_f_p_n_and_uncitable_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "PROFILE_CONTEXT_MAX_CHARS", 24)
    index = FakeIndex()
    index.user_card = lambda: "Neil builds Taurine and studies knowledge graphs."  # type: ignore[attr-defined,method-assign]
    llm = FakeLLM([{"no_update": True}])
    registry, update = _evidence()

    result = await Summarizer(
        llm,
        index,  # type: ignore[arg-type]
        EvidenceValidator(registry, Entails(), entailment_floor=0.8),
    ).summarize(None, '[T1] [channel 0] "Maya owns launch."', update, [])

    assert result is not None and result.no_update
    prompt = llm.calls[0].user
    assert "<user_profile>\nNeil builds Taurine and\n</user_profile>" in prompt
    assert "[T1]" in prompt
    for ref in ("E1", "F1", "P1", "N1"):
        assert f"[{ref}]" in prompt
    assert "not evidence" in llm.calls[0].system


async def test_accepts_sentence_level_validated_summary_and_evidence_snapshot() -> None:
    registry, update = _evidence()
    llm = FakeLLM(
        [
            {
                "summary_md": "Maya owns the launch. [F1]",
                "highlights": [{"fact_id": "F1", "why": "ownership"}],
                "actionables": [],
            }
        ]
    )

    result = await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] [channel 0] "Maya owns the launch."', update, []
    )

    assert result is not None and not result.rejected
    assert result.summary is not None
    assert result.summary.summary_md == "Maya owns the launch. [F1]"
    assert result.summary.evidence is not None
    assert result.summary.evidence.passages[0].evidence_id == "E1"
    assert registry.refs("cited") == ("F1",)


async def test_abstained_pass_cannot_reuse_accumulated_session_evidence() -> None:
    registry, accepted = _evidence()
    abstained = replace(
        accepted,
        decision=RetrievalDecision(False, 0.1, "below_live_floor"),
        support_passages=(),
        support_facts=(),
        support_paths=(),
        support_notes=(),
    )
    llm = FakeLLM([{"summary_md": "Maya owns the launch. [F1]", "actionables": []}])

    result = await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] [channel 0] "A different topic."', abstained, []
    )

    assert result is not None and result.rejected
    assert result.reject_reason == "fact_provenance"
    assert "[F1]" not in llm.calls[0].user


@pytest.mark.parametrize(
    ("summary", "scorer", "reason"),
    [
        ("Maya owns the launch.", Entails(), "citation_coverage"),
        ("Maya owns the launch. [E99]", Entails(), "invalid_citation"),
        ("Maya owns the launch. [E1]", Contradicts(), "entailment_below_floor"),
        ("Maya owns the launch. [E1]", None, "nli_unavailable_live_fail_closed"),
    ],
)
async def test_rejects_unsupported_unknown_contradicted_and_unvalidated_claims(
    summary: str,
    scorer: object | None,
    reason: str,
) -> None:
    registry, update = _evidence()
    llm = FakeLLM([{"summary_md": summary, "actionables": []}])

    result = await _summarizer(llm, registry, scorer).summarize(
        None, '[T1] [channel 0] "Maya owns the launch."', update, []
    )

    assert result is not None and result.rejected
    assert result.summary is None
    assert result.reject_reason == reason
    assert summary not in (result.validation.accepted_output if result.validation else "")


async def test_fallback_model_is_single_retry_and_double_failure_returns_none() -> None:
    registry, update = _evidence()
    fallback = FakeLLM([LLMError("primary"), {"no_update": True}])
    result = await _summarizer(fallback, registry, Entails()).summarize(
        None, '[T1] [channel 0] "hello"', update, []
    )
    assert result is not None and result.model == config.SUMMARY_FALLBACK_MODEL

    failed = FakeLLM([LLMError("primary"), LLMError("fallback")])
    assert (
        await _summarizer(failed, registry, Entails()).summarize(
            None, '[T1] [channel 0] "hello"', update, []
        )
        is None
    )


def _summarizer(llm: FakeLLM, registry: EvidenceRegistry, scorer: object | None) -> Summarizer:
    return Summarizer(
        llm,  # type: ignore[arg-type]
        FakeIndex(),  # type: ignore[arg-type]
        EvidenceValidator(registry, scorer, entailment_floor=0.8),  # type: ignore[arg-type]
    )


def _evidence() -> tuple[EvidenceRegistry, GroundingUpdate]:
    registry = EvidenceRegistry()
    registry.register("transcript", "chunk-1", {"text": "Maya owns the launch."})
    registry.mark_read("T1")
    registry.register("note", "n1", {"title": "Launch"})
    registry.mark_read("N1")
    registry.register(
        "passage", "p1", {"text": "Maya owns the launch.", "note_id": "n1"}
    )
    registry.mark_read("E1")
    registry.register(
        "fact",
        "fact-1",
        {"source_evidence_ids": ["E1"], "status": "active"},
    )
    registry.mark_read("F1")
    registry.register("path", "path-1", {"rendered": "E1 -> F1"})
    registry.mark_read("P1")
    passage = EvidencePassage(
        "E1", "p1", "n1", "Launch", "obsidian", "Maya owns the launch.",
        0, 21, 1.0, 2.0, 0.9, 0.95,
    )
    return registry, GroundingUpdate(
        decision=RetrievalDecision(True, 0.95, "accepted", ("dense", "triple")),
        passages=(passage,),
        facts=(
            EvidenceFact(
                "F1", ("fact-1",), "Maya", "owns", "launch", "forward",
                "active", 2.0, None, None, 0.95, ("E1",),
            ),
        ),
        paths=(EvidencePath("P1", ("E1", "F1"), ("supports",), ("fact-1",), 0.9, "E1 -> F1"),),
        notes=(EvidenceNote("N1", "n1", "Launch", "obsidian", 2.0, 0.95, ("E1",)),),
        display_passages=(passage,),
        display_notes=(EvidenceNote("N1", "n1", "Launch", "obsidian", 2.0, 0.95, ("E1",)),),
        no_new_evidence=False,
        new_support_ids=("E1", "F1"),
        revision=1,
        support_passages=(passage,),
        support_facts=(
            EvidenceFact(
                "F1", ("fact-1",), "Maya", "owns", "launch", "forward",
                "active", 2.0, None, None, 0.95, ("E1",),
            ),
        ),
        support_paths=(EvidencePath("P1", ("E1", "F1"), ("supports",), ("fact-1",), 0.9, "E1 -> F1"),),
        support_notes=(EvidenceNote("N1", "n1", "Launch", "obsidian", 2.0, 0.95, ("E1",)),),
    )
