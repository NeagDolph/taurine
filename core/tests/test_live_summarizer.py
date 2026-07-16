from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from taurine_core import config
from taurine_core.dates import timestamp_date
from taurine_core.live.grounding import GroundingUpdate
from taurine_core.live import summarizer as summarizer_module
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

from taurine_core.schemas import Actionable

from tests.fakes import FakeIndex


@dataclass
class Call:
    model: str
    system: str
    user: str
    timeout_s: float | None
    schema: dict[str, Any] | None


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
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(Call(model, system, user, timeout_s, schema))
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
        EvidenceValidator(registry),
    ).summarize(None, '[T1] [channel 0] "Maya owns launch."', update, [])

    assert result is not None and result.no_update
    prompt = llm.calls[0].user
    assert "<user_profile>\nNeil builds Taurine and\n</user_profile>" in prompt
    assert "[T1]" in prompt
    for ref in ("E1", "F1", "P1", "N1"):
        assert f"[{ref}]" in prompt
    assert "[E1] (2026-07-14) Maya owns the launch." in prompt
    assert f"observed={timestamp_date(2.0)}" in prompt
    assert "not evidence" in llm.calls[0].system


async def test_user_notes_precede_transcript_and_use_live_model_schema() -> None:
    registry, update = _evidence()
    llm = FakeLLM([{"no_update": True}])

    result = await _summarizer(llm, registry, Entails()).summarize(
        None,
        '[T1] You: "Maya owns the launch."',
        update,
        [],
        notepad_md="# Launch\nMaya owns the launch.",
    )

    assert result is not None and result.no_update
    call = llm.calls[0]
    assert call.model == config.LIVE_SUMMARY_MODEL
    assert call.schema is not None
    assert call.schema["title"] == "live_summary"
    assert call.user.index("<user_notes>") < call.user.index("<transcript_evidence>")
    assert (
        "User notes signal what the user considers important — weight related "
        "transcript content higher; do not restate the notes back."
        in call.user
    )
    assert "# Launch\nMaya owns the launch." in call.user


async def test_user_notes_block_is_omitted_for_whitespace() -> None:
    registry, update = _evidence()
    llm = FakeLLM([{"no_update": True}])

    await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] You: "hello"', update, [], notepad_md=" \n\t "
    )

    assert "<user_notes>" not in llm.calls[0].user


async def test_user_notes_truncation_uses_exact_head_marker_tail_math() -> None:
    registry, update = _evidence()
    llm = FakeLLM([{"no_update": True}])
    notes = "H" * 5_800 + "discarded" * 100 + "T" * 2_000
    marker = "\n…[notes truncated]…\n"

    await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] You: "hello"', update, [], notepad_md=notes
    )

    block = llm.calls[0].user.split("<user_notes>\n", 1)[1].split(
        "\n</user_notes>", 1
    )[0]
    rendered_notes = block.split("\n", 1)[1]
    assert rendered_notes == notes[:5_800] + marker + notes[-2_000:]
    assert len(rendered_notes) <= config.MEETING_NOTES_PROMPT_MAX_CHARS


def test_summary_model_aliases_remain_on_deepseek() -> None:
    assert config.SUMMARY_MODEL == "deepseek/deepseek-v4-flash"
    assert config.IDEA_SEARCH_MODEL == config.SUMMARY_MODEL
    assert config.CONTEXT_ENRICH_MODEL == config.SUMMARY_MODEL


async def test_live_summary_prompt_datetime_is_request_scoped_and_reused_for_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    registry, update = _evidence()
    llm = FakeLLM([LLMError("primary"), {"no_update": True}])
    timestamp = 1_720_000_000.0
    monkeypatch.setattr(summarizer_module.time, "time", lambda: timestamp)

    result = await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] [channel 0] "Maya owns the launch."', update, []
    )

    expected = datetime.fromtimestamp(timestamp).astimezone().strftime(
        "Current date and time: %Y-%m-%d %H:%M %Z."
    )
    assert result is not None and result.no_update
    assert [call.system.splitlines()[0] for call in llm.calls] == [expected, expected]


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


async def test_labeled_transcript_round_trip_preserves_t_citation_validation() -> None:
    registry, update = _evidence()
    llm = FakeLLM(
        [{"summary_md": "Maya owns the launch. [T1]", "actionables": []}]
    )

    result = await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] You: "Maya owns the launch."', update, []
    )

    assert result is not None and not result.rejected
    assert result.summary is not None
    assert result.summary.summary_md == "Maya owns the launch. [T1]"
    assert (
        "Transcript lines are labeled You (the user's own mic) and Them (other call "
        "participants); attribute statements accordingly."
        in llm.calls[0].system
    )


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
        # coverage and NLI stages were removed 2026-07-10; only citation
        # validity/availability still rejects a summary
        ("Maya owns the launch. [E99]", Entails(), "invalid_citation"),
    ],
)
async def test_rejects_unknown_citations(
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


async def test_rolling_summary_retains_previously_accepted_citations_then_prunes() -> None:
    """C3a: refs cited by the last ACCEPTED summary stay citeable; refs the
    current accepted summary drops lose grandfathered status."""

    registry, update1 = _evidence()
    llm = FakeLLM(
        [
            {"summary_md": "Maya owns the launch. [E1]", "actionables": []},
            {
                "summary_md": "Maya owns the launch. [E1]\nBudget approved. [T2]",
                "actionables": [],
            },
            {"summary_md": "Maya owns the launch. [E1]", "actionables": []},
        ]
    )
    summarizer = _summarizer(llm, registry, Entails())

    one = await summarizer.summarize(
        None, '[T1] [channel 0] "Maya owns the launch."', update1, []
    )
    assert one is not None and not one.rejected

    # pass 2 supplies only E2/T2, yet may retain the pass-1 sentence citing E1
    update2 = _pass_update(registry, passage_id="p2", note_id="n2")
    registry.register("transcript", "chunk-2", {"text": "Budget approved."})
    registry.mark_read("T2")
    two = await summarizer.summarize(
        one.summary, '[T2] [channel 0] "Budget approved."', update2, []
    )
    assert two is not None and not two.rejected
    assert two.summary is not None and two.summary.rev == 2

    # pass 2's accepted summary cited E1 and T2 — but a pass-3 summary after
    # one that DROPPED E1 must fail. Simulate by pruning: accept a summary
    # citing only T2, then try E1 again.
    llm.responses.insert(0, {"summary_md": "Budget approved. [T2]", "actionables": []})
    update3 = _pass_update(registry, passage_id="p3", note_id="n3")
    three = await summarizer.summarize(
        two.summary, "(no new transcript)", update3, []
    )
    assert three is not None and not three.rejected

    update4 = _pass_update(registry, passage_id="p4", note_id="n4")
    four = await summarizer.summarize(
        three.summary, "(no new transcript)", update4, []
    )
    assert four is not None and four.rejected
    assert four.reject_reason == "evidence_unavailable"


async def test_never_validated_old_ref_still_fails() -> None:
    """A ref that was read in some earlier pass but never cited by an
    accepted summary is not citeable later (strictness preserved)."""

    registry, _update1 = _evidence()  # registers E1 as read
    update2 = _pass_update(registry, passage_id="p2", note_id="n2")
    llm = FakeLLM([{"summary_md": "Maya owns the launch. [E1]", "actionables": []}])

    result = await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] [channel 0] "unrelated speech."', update2, []
    )

    assert result is not None and result.rejected
    assert result.reject_reason == "evidence_unavailable"


async def test_invalidly_cited_highlight_is_dropped_without_rejecting_the_summary() -> None:
    registry, update = _evidence()
    llm = FakeLLM(
        [
            {
                "summary_md": "Maya owns the launch. [F1]",
                "highlights": [
                    {"fact_id": "F1", "why": "Maya owns it"},
                    {"fact_id": "F9", "why": "unknown fact reference"},
                ],
                "actionables": [],
            }
        ]
    )

    result = await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] [channel 0] "Maya owns the launch."', update, []
    )

    assert result is not None and not result.rejected
    assert result.summary is not None
    assert [row.why for row in result.summary.highlights] == ["Maya owns it"]


async def test_open_actionable_grounding_refs_are_included_in_prompt() -> None:
    registry, update = _evidence()
    llm = FakeLLM([{"no_update": True}])
    row = Actionable(
        id="a1", title="Do X", quote="do x", grounding=["T1", "F1"], status="open"
    )

    await _summarizer(llm, registry, Entails()).summarize(
        None, '[T1] [channel 0] "Maya owns the launch."', update, [row]
    )

    assert '"grounding": ["T1", "F1"]' in llm.calls[0].user


def _pass_update(
    registry: EvidenceRegistry, *, passage_id: str, note_id: str
) -> GroundingUpdate:
    """A minimal accepted pass whose only support is one fresh passage."""

    text = f"Evidence {passage_id}."
    note_ref = registry.register("note", note_id, {"title": note_id})
    registry.mark_read(note_ref)
    evidence_ref = registry.register(
        "passage", passage_id, {"text": text, "note_id": note_id}
    )
    registry.mark_read(evidence_ref)
    passage = EvidencePassage(
        evidence_ref, passage_id, note_id, note_id, "obsidian", text,
        0, len(text), 1.0, 2.0, 0.9, 0.95,
    )
    note = EvidenceNote(note_ref, note_id, note_id, "obsidian", 2.0, 0.95, (evidence_ref,))
    return GroundingUpdate(
        decision=RetrievalDecision(True, 0.9, "accepted", ("dense",)),
        passages=(passage,),
        facts=(),
        paths=(),
        notes=(note,),
        display_passages=(passage,),
        display_notes=(note,),
        no_new_evidence=False,
        new_support_ids=(evidence_ref,),
        revision=2,
        support_passages=(passage,),
        support_facts=(),
        support_paths=(),
        support_notes=(note,),
    )


def _summarizer(llm: FakeLLM, registry: EvidenceRegistry, scorer: object | None) -> Summarizer:
    return Summarizer(
        llm,  # type: ignore[arg-type]
        FakeIndex(),  # type: ignore[arg-type]
        EvidenceValidator(registry),  # type: ignore[arg-type]
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
        0, 21, 1.0, 2.0, 0.9, 0.95, passage_date="2026-07-14",
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
