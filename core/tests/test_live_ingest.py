from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live.grounding import GroundingEngine, GroundingUpdate
from taurine_core.live.ingest import LiveSession
from taurine_core.live.summarizer import SummaryResult
from taurine_core.live.trigger import SummaryTrigger
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.types import (
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
)
from taurine_core.schemas import Actionable, Chunk, MeetingSummary

from tests.fakes import FakeClock


@dataclass
class SequenceAmbient:
    results: list[EvidencePack | None]
    calls: list[tuple[str, str | None, dict[str, float]]] = field(default_factory=list)

    async def on_window(
        self,
        window_text: str,
        context_text: str | None = None,
        *,
        session_reset: dict[str, float] | None = None,
    ) -> EvidencePack | None:
        self.calls.append((window_text, context_text, dict(session_reset or {})))
        return self.results.pop(0) if self.results else None


@dataclass
class Trigger:
    decisions: list[tuple[bool, str]]
    noted: list[tuple[int, bool]] = field(default_factory=list)
    rejected: int = 0

    def offer(
        self,
        update: GroundingUpdate,
        window_emb: object,
        new_chars: int,
    ) -> tuple[bool, str]:
        del update, window_emb, new_chars
        return self.decisions.pop(0) if self.decisions else (False, "no_change")

    def note_summary(self, update: GroundingUpdate, *, no_update: bool) -> None:
        self.noted.append((update.revision, no_update))

    def note_rejected(self) -> None:
        self.rejected += 1


@dataclass
class SummaryStub:
    results: list[SummaryResult | None]
    wait_on: asyncio.Event | None = None
    calls: list[tuple[str, GroundingUpdate, str]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def summarize(
        self,
        state: MeetingSummary | None,
        transcript_evidence: str,
        update: GroundingUpdate,
        open_actionables: list[object],
        notepad_md: str = "",
    ) -> SummaryResult | None:
        del state, open_actionables
        self.calls.append((transcript_evidence, update, notepad_md))
        self.started.set()
        if self.wait_on is not None:
            await self.wait_on.wait()
        return self.results.pop(0) if self.results else None


async def test_partials_are_query_only_and_finals_get_stable_transcript_ids() -> None:
    registry = EvidenceRegistry()
    ambient = SequenceAmbient([None, None])
    session, events = _session(registry, ambient=ambient)

    await session.on_chunk(Chunk(channel=0, text="partial launch", ts=1.0, final=False))
    await session.on_chunk(Chunk(channel=0, text="final launch", ts=2.0, final=True))

    entries = registry.snapshot()["entries"]
    assert [row["ref"] for row in entries] == ["T1"]  # type: ignore[index]
    assert "partial launch" in ambient.calls[0][0]
    assert events[0].seq == 0
    assert events[1].seq == 1
    assert events[1].final


async def test_retrieval_emits_evidence_native_grounding_event() -> None:
    registry = EvidenceRegistry()
    pack = _pack(registry)
    session, events = _session(registry, ambient=SequenceAmbient([pack]))

    await session.on_chunk(Chunk(channel=0, text="What does Maya own?", ts=2.0))

    event = next(row for row in events if row.type == "grounding")
    assert event.decision.accepted
    assert event.passages[0].evidence_id == "E1"
    assert event.notes[0].note_ref == "N1"
    assert event.new_support_ids == ["E1"]
    assert not event.no_new_evidence


async def test_accepted_summary_advances_only_snapshotted_sequence() -> None:
    registry = EvidenceRegistry()
    release = asyncio.Event()
    summary = SummaryStub(
        [SummaryResult(summary=MeetingSummary(rev=1, summary_md="Accepted [T1]"), no_update=False)],
        wait_on=release,
    )
    session, events = _session(
        registry,
        ambient=SequenceAmbient([_pack(registry), None]),
        trigger=Trigger([(True, "jaccard")]),
        summarizer=summary,
    )

    await session.on_chunk(Chunk(channel=0, text="first final", ts=1.0))
    await summary.started.wait()
    await session.on_chunk(Chunk(channel=0, text="second final", ts=2.0))
    release.set()
    await _drain(session)

    assert summary.calls[0][0].startswith('[T1] You: "first final"')
    assert session.ledger.last_accepted_summary_seq == 1
    assert [row.seq for row in session.ledger.pending()] == [2]
    assert any(row.type == "summary" and row.rev == 1 for row in events)


async def test_notepad_is_snapshotted_into_summary_call() -> None:
    registry = EvidenceRegistry()
    summary = SummaryStub([SummaryResult(summary=None, no_update=True)])
    session, _ = _session(
        registry,
        ambient=SequenceAmbient([_pack(registry)]),
        trigger=Trigger([(True, "jaccard")]),
        summarizer=summary,
    )
    session.set_notepad("# Decisions\nShip Friday")

    await session.on_chunk(Chunk(channel=0, text="Ship Friday", ts=1.0))
    await _drain(session)

    assert summary.calls[0][2] == "# Decisions\nShip Friday"


async def test_no_update_and_rejection_leave_transcript_checkpoint_pending() -> None:
    registry = EvidenceRegistry()
    no_update = SummaryStub([SummaryResult(summary=None, no_update=True)])
    trigger = Trigger([(True, "jaccard"), (True, "jaccard")])
    session, _ = _session(
        registry,
        ambient=SequenceAmbient([_pack(registry), _pack(registry, passage_id="p2")]),
        trigger=trigger,
        summarizer=no_update,
    )
    await session.on_chunk(Chunk(channel=0, text="one", ts=1.0))
    await _drain(session)
    assert session.ledger.last_accepted_summary_seq == 0
    assert [row.seq for row in session.ledger.pending()] == [1]

    no_update.results.append(
        SummaryResult(
            summary=None,
            no_update=False,
            rejected=True,
            reject_reason="entailment_below_floor",
        )
    )
    await session.on_chunk(Chunk(channel=0, text="two", ts=2.0))
    await _drain(session)
    assert session.ledger.last_accepted_summary_seq == 0
    assert [row.seq for row in session.ledger.pending()] == [1, 2]
    assert trigger.rejected == 1


async def test_abstention_event_preserves_grounding_and_does_not_invent_support() -> None:
    registry = EvidenceRegistry()
    accepted = _pack(registry)
    abstained = EvidencePack.abstained(RetrievalQuery("filler", "live", 3.0), "below_live_floor")
    session, events = _session(
        registry,
        ambient=SequenceAmbient([accepted, abstained]),
    )
    await session.on_chunk(Chunk(channel=0, text="launch", ts=1.0))
    await session.on_chunk(Chunk(channel=0, text="um", ts=2.0))

    grounding = [row for row in events if row.type == "grounding"]
    assert grounding[1].no_new_evidence
    assert grounding[1].new_support_ids == []
    assert grounding[1].passages[0].evidence_id == "E1"
    assert grounding[1].revision == grounding[0].revision


async def test_no_kb_session_produces_transcript_only_first_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session whose retrieval always abstains still summarizes (C3b)."""

    monkeypatch.setattr(config, "SUMMARY_MIN_NEW_CHARS", 5)
    registry = EvidenceRegistry()
    abstained = EvidencePack.abstained(
        RetrievalQuery("we discussed the plan", "live", 1.0), "below_live_floor"
    )
    summary = SummaryStub(
        [
            SummaryResult(
                summary=MeetingSummary(rev=1, summary_md="Discussed the plan. [T1]"),
                no_update=False,
            )
        ]
    )
    session, events = _session(
        registry,
        ambient=SequenceAmbient([abstained]),
        trigger=SummaryTrigger(FakeClock().now),  # type: ignore[arg-type]
        summarizer=summary,
    )

    await session.on_chunk(Chunk(channel=0, text="we discussed the plan", ts=1.0))
    await _drain(session)

    assert session.ledger.last_accepted_summary_seq == 1
    assert any(row.type == "summary" and row.rev == 1 for row in events)


async def test_cancelled_in_flight_summary_releases_checkpoint_for_next_trigger() -> None:
    registry = EvidenceRegistry()
    release = asyncio.Event()
    summary = SummaryStub([], wait_on=release)
    trigger = Trigger([(True, "jaccard"), (True, "jaccard")])
    session, _ = _session(
        registry,
        ambient=SequenceAmbient([_pack(registry), _pack(registry, passage_id="p2")]),
        trigger=trigger,
        summarizer=summary,
    )

    await session.on_chunk(Chunk(channel=0, text="first final", ts=1.0))
    await summary.started.wait()
    task = session._summary_task
    assert task is not None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert session.ledger.in_flight_through_seq is None
    assert trigger.rejected == 1

    # the next trigger must be able to begin a fresh summary pass
    summary.wait_on = None
    await session.on_chunk(Chunk(channel=0, text="second final", ts=2.0))
    await _drain(session)
    assert session.ledger.in_flight_through_seq is None
    assert trigger.rejected == 2  # empty stub → llm_unavailable rejection, not a crash


async def test_aclose_drains_in_flight_summary_and_is_idempotent() -> None:
    registry = EvidenceRegistry()
    release = asyncio.Event()
    summary = SummaryStub([], wait_on=release)
    trigger = Trigger([(True, "jaccard"), (True, "jaccard")])
    session, _ = _session(
        registry,
        ambient=SequenceAmbient([_pack(registry), _pack(registry, passage_id="p2")]),
        trigger=trigger,
        summarizer=summary,
    )
    await session.on_chunk(Chunk(channel=0, text="first final", ts=1.0))
    await summary.started.wait()

    await session.aclose()
    assert session._summary_task is None
    assert session.ledger.in_flight_through_seq is None

    await session.aclose()  # idempotent

    # a closed session must not start new summary tasks
    await session.on_chunk(Chunk(channel=0, text="second final", ts=2.0))
    assert session._summary_task is None


async def test_user_dismissal_during_in_flight_summary_survives_acceptance() -> None:
    registry = EvidenceRegistry()
    open_row = Actionable(
        id="act-1", title="Ship it", quote="ship it", grounding=["T1"], status="open"
    )
    release = asyncio.Event()
    summary = SummaryStub(
        [
            SummaryResult(
                summary=MeetingSummary(
                    rev=1, summary_md="A [T1]", actionables=[open_row]
                ),
                no_update=False,
            ),
            SummaryResult(
                summary=MeetingSummary(
                    rev=2, summary_md="B [T2]", actionables=[open_row]
                ),
                no_update=False,
            ),
        ]
    )
    trigger = Trigger([(True, "jaccard"), (True, "jaccard")])
    session, events = _session(
        registry,
        ambient=SequenceAmbient([_pack(registry), _pack(registry, passage_id="p2")]),
        trigger=trigger,
        summarizer=summary,
    )

    await session.on_chunk(Chunk(channel=0, text="first final", ts=1.0))
    await _drain(session)
    assert session.current_summary().actionables[0].status == "open"

    summary.wait_on = release
    summary.started.clear()
    await session.on_chunk(Chunk(channel=0, text="second final", ts=2.0))
    await summary.started.wait()
    await session.set_actionable_status("act-1", "dismissed")
    release.set()
    await _drain(session)

    assert session.current_summary().rev == 2
    assert session.current_summary().actionables[0].status == "dismissed"


async def test_topic_shift_clears_grounding_and_bounds_later_windows() -> None:
    registry = EvidenceRegistry()
    first = _pack(registry)
    second = _pack(registry, passage_id="p2")
    third = _pack(registry, passage_id="p3")
    ambient = ShiftAmbient([first, second, third], topic_shifts=[False, True, False])
    session, events = _session(registry, ambient=ambient)

    await session.on_chunk(Chunk(channel=0, text="alpha topic speech", ts=1.0))
    await session.on_chunk(Chunk(channel=0, text="beta pivot speech", ts=2.0))
    await session.on_chunk(Chunk(channel=0, text="beta continues here", ts=3.0))

    # pivot pass cleared the accumulated grounding down to the new pack
    grounding = [row for row in events if row.type == "grounding"]
    assert {row.evidence_id for row in grounding[1].passages} == {"E2"}
    # the pass after the pivot must not span pre-pivot speech
    assert "alpha topic speech" not in ambient.calls[2][0]
    assert "beta pivot speech" in ambient.calls[2][0]


async def test_rolling_summary_two_pass_real_summarizer_advances_checkpoint() -> None:
    """Real seam (C3a): LiveSession + Summarizer + validator + ledger. The
    pass-2 rolling summary retains a pass-1 cited sentence, validates, and
    advances the transcript checkpoint."""

    from taurine_core.live.summarizer import Summarizer
    from taurine_core.retrieval.validate import EvidenceValidator

    from tests.fakes import FakeIndex

    class ScriptedLLM:
        def __init__(self, responses: list[dict[str, Any]]) -> None:
            self.responses = list(responses)

        async def complete_json(
            self,
            model: str,
            system: str,
            user: str,
            timeout_s: float | None = None,
            schema: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del model, system, user, timeout_s, schema
            return self.responses.pop(0)

    class Entails:
        def score(self, premise: str, hypothesis: str) -> float:
            del premise, hypothesis
            return 0.99

    registry = EvidenceRegistry()
    llm = ScriptedLLM(
        [
            {"summary_md": "Maya owns the launch. [T1]", "actionables": []},
            {
                "summary_md": "Maya owns the launch. [T1]\nBudget approved. [T2]",
                "actionables": [],
            },
        ]
    )
    summarizer = Summarizer(
        llm,  # type: ignore[arg-type]
        FakeIndex(),  # type: ignore[arg-type]
        EvidenceValidator(registry),
    )
    abstain = lambda text, ts: EvidencePack.abstained(  # noqa: E731
        RetrievalQuery(text, "live", ts), "below_live_floor"
    )
    events: list[Any] = []
    session = LiveSession(
        registry=registry,
        ambient=SequenceAmbient([abstain("one", 1.0), abstain("two", 2.0)]),  # type: ignore[arg-type]
        grounding=GroundingEngine(registry),
        trigger=Trigger([(True, "first_summary"), (True, "staleness")]),
        summarizer=summarizer,
        emit=events.append,
    )

    await session.on_chunk(Chunk(channel=0, text="Maya owns the launch.", ts=1.0))
    await _drain(session)
    assert session.ledger.last_accepted_summary_seq == 1

    await session.on_chunk(Chunk(channel=0, text="Budget approved.", ts=2.0))
    await _drain(session)

    assert session.ledger.last_accepted_summary_seq == 2
    revs = [row.rev for row in events if row.type == "summary"]
    assert revs == [1, 2]
    assert "Maya owns the launch. [T1]" in session.current_summary().summary_md


@dataclass
class ShiftAmbient(SequenceAmbient):
    topic_shifts: list[bool] = field(default_factory=list)
    last_topic_shift: bool = False
    last_window_embedding: Any = None

    async def on_window(
        self,
        window_text: str,
        context_text: str | None = None,
        *,
        session_reset: dict[str, float] | None = None,
    ) -> EvidencePack | None:
        result = await super().on_window(
            window_text, context_text, session_reset=session_reset
        )
        self.last_topic_shift = (
            self.topic_shifts.pop(0) if self.topic_shifts else False
        )
        return result


def _session(
    registry: EvidenceRegistry,
    *,
    ambient: SequenceAmbient,
    trigger: Trigger | None = None,
    summarizer: SummaryStub | None = None,
) -> tuple[LiveSession, list[Any]]:
    events: list[Any] = []
    grounding = GroundingEngine(registry)
    return (
        LiveSession(
            registry=registry,
            ambient=ambient,  # type: ignore[arg-type]
            grounding=grounding,
            trigger=trigger or Trigger([]),
            summarizer=summarizer or SummaryStub([]),
            emit=events.append,
        ),
        events,
    )


def _pack(registry: EvidenceRegistry, *, passage_id: str = "p1") -> EvidencePack:
    note_ref = registry.register("note", "n1", {"title": "Launch"})
    registry.mark_read(note_ref)
    evidence_ref = registry.register(
        "passage",
        passage_id,
        {"passage_id": passage_id, "note_id": "n1", "text": "Maya owns launch."},
    )
    registry.mark_read(evidence_ref)
    passage = EvidencePassage(
        evidence_ref, passage_id, "n1", "Launch", "obsidian", "Maya owns launch.",
        0, 17, 1.0, 2.0, 0.9, 0.95,
    )
    return EvidencePack(
        RetrievalDecision(True, 0.95, "accepted", ("dense",)),
        RetrievalQuery("launch", "live", 2.0),
        (passage,),
        (),
        (),
        (EvidenceNote(note_ref, "n1", "Launch", "obsidian", 2.0, 0.95, (evidence_ref,)),),
        RetrievalTrace(original_query="launch"),
    )


async def _drain(session: LiveSession) -> None:
    task = session._summary_task
    if task is not None:
        await task
    await asyncio.sleep(0)
