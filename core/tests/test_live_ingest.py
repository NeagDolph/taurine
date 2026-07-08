from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from taurine_core import config
from taurine_core.live.grounding import GroundingUpdate
from taurine_core.live.ingest import LiveSession
from taurine_core.live.summarizer import SummaryResult
from taurine_core.live.trigger import SummaryTrigger
from taurine_core.schemas import (
    Actionable,
    Chunk,
    GroundedFact,
    GroundingItem,
    GroundingWhy,
    MeetingSummary,
)

from tests.fakes import FakeClock, FakeIndex, note


class NoopAmbient:
    last_debug: dict[str, Any] | None = None
    last_grounding: GroundingUpdate | None = None
    last_fused_ids: set[str] = set()
    last_extracted: list[str] = []
    last_window_embedding: np.ndarray | None = None

    async def on_window(self, window_text: str, context_text: str | None = None) -> None:
        del window_text, context_text
        return None


@dataclass
class SequenceExact:
    hits: list[list[tuple[str, str, bool]]]

    def on_chunk(self, chunk: Chunk) -> list[tuple[str, str, bool]]:
        del chunk
        return self.hits.pop(0) if self.hits else []


@dataclass
class SequenceAmbient:
    results: list[Any | None]
    debug_sequences: list[dict[str, Any] | None] = field(default_factory=list)
    fused_id_sequences: list[set[str]] = field(default_factory=list)
    extracted_sequences: list[list[str]] = field(default_factory=list)
    embeddings: list[np.ndarray | None] = field(default_factory=list)
    windows: list[str] = field(default_factory=list)
    contexts: list[str | None] = field(default_factory=list)
    last_debug: dict[str, Any] | None = None
    last_grounding: GroundingUpdate | None = None
    last_fused_ids: set[str] = field(default_factory=set)
    last_extracted: list[str] = field(default_factory=list)
    last_window_embedding: np.ndarray | None = None

    async def on_window(
        self,
        window_text: str,
        context_text: str | None = None,
    ) -> Any | None:
        self.windows.append(window_text)
        self.contexts.append(context_text)
        self.last_debug = self.debug_sequences.pop(0) if self.debug_sequences else None
        self.last_fused_ids = self.fused_id_sequences.pop(0) if self.fused_id_sequences else set()
        self.last_extracted = self.extracted_sequences.pop(0) if self.extracted_sequences else []
        self.last_window_embedding = self.embeddings.pop(0) if self.embeddings else None
        result = self.results.pop(0) if self.results else None
        self.last_grounding = result if isinstance(result, GroundingUpdate) else None
        return result


@dataclass
class RecordingTrigger:
    decisions: list[tuple[bool, str]]
    offers: list[tuple[GroundingUpdate, np.ndarray | None, int]] = field(default_factory=list)
    noted: list[tuple[GroundingUpdate, bool]] = field(default_factory=list)
    rejected_count: int = 0

    def offer(
        self,
        update: GroundingUpdate,
        window_emb: np.ndarray | None,
        new_chars: int,
    ) -> tuple[bool, str]:
        self.offers.append((update, window_emb, new_chars))
        return self.decisions.pop(0) if self.decisions else (False, "no_change")

    def note_summary(self, update: GroundingUpdate, *, no_update: bool) -> None:
        self.noted.append((update, no_update))

    def note_rejected(self) -> None:
        self.rejected_count += 1


@dataclass
class RecordingSummarizer:
    results: list[SummaryResult | None]
    wait_on: asyncio.Event | None = None
    calls: list[tuple[MeetingSummary | None, str, GroundingUpdate, list[Actionable]]] = field(
        default_factory=list
    )
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def summarize(
        self,
        state: MeetingSummary | None,
        transcript_delta: str,
        update: GroundingUpdate,
        open_actionables: list[Actionable],
    ) -> SummaryResult | None:
        self.calls.append((state, transcript_delta, update, list(open_actionables)))
        self.started.set()
        if self.wait_on is not None:
            await self.wait_on.wait()
        return self.results.pop(0) if self.results else None


@dataclass
class RecordingLogger:
    records: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def log(self, kind: str, **payload: Any) -> None:
        self.records.append((kind, payload))

    def close(self) -> None:
        self.records.append(("closed", {}))


def chunk(text: str, ts: float = 1000.0, channel: int = 0) -> Chunk:
    return Chunk(channel=channel, text=text, ts=ts, final=True)


def interim(text: str, ts: float = 1000.0, channel: int = 0) -> Chunk:
    return Chunk(channel=channel, text=text, ts=ts, final=False)


def grounding_item(
    note_id: str,
    title: str,
    gist: str,
    score: float,
    *,
    short_id: str = "N1",
) -> GroundingItem:
    return GroundingItem(
        note_id=note_id,
        short_id=short_id,
        title=title,
        gist=gist,
        source="obsidian",
        age_days=12.5,
        score=score,
        why=GroundingWhy(heads=["dense", "ppr"], ppr_mass=0.25),
    )


def accepted_result(
    *,
    summary: MeetingSummary,
    new_actionables: list[Actionable] | None = None,
    updated_actionables: list[Actionable] | None = None,
) -> SummaryResult:
    return SummaryResult(
        summary=summary,
        no_update=False,
        new_actionables=new_actionables or [],
        updated_actionables=updated_actionables or [],
        latency_ms=12.5,
        model="model-a",
        citation_stats={"total": 1.0, "valid": 1.0, "invalid": 0.0},
    )


def session_with(
    *,
    index: FakeIndex,
    clock: FakeClock,
    exact: SequenceExact | None = None,
    ambient: Any | None = None,
    trigger: RecordingTrigger | None = None,
    summarizer: RecordingSummarizer | None = None,
    logger: RecordingLogger | None = None,
) -> tuple[LiveSession, list[Any], RecordingTrigger, RecordingSummarizer, RecordingLogger]:
    events: list[Any] = []
    trigger = trigger or RecordingTrigger([])
    summarizer = summarizer or RecordingSummarizer([])
    logger = logger or RecordingLogger()
    session = LiveSession(
        index=index,
        exact=exact or SequenceExact([]),
        ambient=ambient or NoopAmbient(),
        trigger=trigger,
        summarizer=summarizer,
        emit=events.append,
        clock=clock,
        logger=logger,
    )
    return session, events, trigger, summarizer, logger


async def drain_summary_task() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_c1_exact_hits_emit_grounded_trigger_without_pin_events() -> None:
    clock = FakeClock()
    index = FakeIndex(notes={"roadmap": note("roadmap", "Q3 Roadmap", "Roadmap gist")})
    session, events, _trigger, _summarizer, logger = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact([[("roadmap", "q3 roadmap", True)]]),
    )

    await session.on_chunk(chunk("let's open the q3 roadmap"))

    triggers = [event for event in events if event.type == "trigger"]
    assert [(event.kind, event.seqs, event.span, event.matches[0].note_id, event.action) for event in triggers] == [
        ("exact", [1], "q3 roadmap", "roadmap", "grounded")
    ]
    assert "pin" not in [event.type for event in events]
    assert "unpin" not in [event.type for event in events]
    assert [payload["action"] for kind, payload in logger.records if kind == "c1_hit"] == [
        "grounded"
    ]


@pytest.mark.asyncio
async def test_c2_pass_emits_grounded_ambient_event_and_logs_grounding() -> None:
    clock = FakeClock()
    logger = RecordingLogger()
    index = FakeIndex(
        notes={
            "direct": note("direct", "Direct Note", "Direct gist"),
            "expanded": note("expanded", "Expanded Note", "Expanded gist"),
        }
    )
    update = GroundingUpdate(
        items=[grounding_item("direct", "Direct Note", "Direct gist", 0.91)],
        facts=[GroundedFact(fact_id="F1", src="direct", dst="expanded", text="related")],
        revision=3,
        ppr_top=[("expanded", 0.31), ("direct", 0.12)],
    )
    debug = {
        "window": "window text",
        "dense": [["direct", 0.9]],
        "gliner_spans": [["shopify checkout", 0.95], ["Ashish", 0.9]],
        "gliner_dropped": [],
        "gliner_context_queries": [],
        "dense_gliner": [["direct", 0.87]],
        "bm25": [],
        "exact": [],
        "fused": [["direct", 0.016]],
        "ppr_top": [["expanded", 0.31], ["direct", 0.12]],
        "blend": [{"note_id": "direct", "final": 0.91}],
        "grounding_revision": 3,
    }
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    ambient = SequenceAmbient(
        results=[update],
        debug_sequences=[debug],
        fused_id_sequences=[{"direct"}],
        extracted_sequences=[["shopify checkout", "Ashish"]],
        embeddings=[embedding],
    )
    trigger = RecordingTrigger([(False, "no_change")])
    session, events, trigger, _summarizer, logger = session_with(
        index=index,
        clock=clock,
        ambient=ambient,
        trigger=trigger,
        logger=logger,
    )

    await session.on_chunk(chunk("window text"))

    [trigger_event] = [event for event in events if event.type == "trigger"]
    [retrieval] = [event for event in events if event.type == "retrieval"]
    [ambient_event] = [event for event in events if event.type == "ambient"]
    assert trigger_event.kind == "search"
    assert trigger_event.action == "grounded"
    assert trigger_event.seqs == [1]
    assert [(match.note_id, match.rank, match.score, match.via_note_id, match.hop) for match in trigger_event.matches] == [
        ("direct", 1, 0.91, None, None),
        ("expanded", None, 0.31, "", 0),
    ]
    assert retrieval.retrieval_id == trigger_event.trigger_id
    assert retrieval.extracted == ["shopify checkout", "Ashish"]
    assert ambient_event.items[0].model_dump() == {
        "note_id": "direct",
        "title": "Direct Note",
        "gist": "Direct gist",
        "score": 0.91,
        "short_id": "N1",
        "source": "obsidian",
        "age_days": 12.5,
        "why": {"heads": ["dense", "ppr"], "ppr_mass": 0.25, "exact_alias": ""},
    }
    assert trigger.offers == [(update, embedding, len("[ch0] window text"))]
    assert session.last_grounding is update
    assert [kind for kind, _payload in logger.records].count("c2_pass") == 1
    c2_payload = [payload for kind, payload in logger.records if kind == "c2_pass"][0]
    assert c2_payload["trigger_id"] == trigger_event.trigger_id
    assert c2_payload["seqs"] == [1]
    assert c2_payload["grounding_ids"] == ["direct"]
    assert c2_payload["gliner_spans"] == [["shopify checkout", 0.95], ["Ashish", 0.9]]
    assert c2_payload["dense_gliner"] == [["direct", 0.87]]
    assert c2_payload["ppr_top"] == [["expanded", 0.31], ["direct", 0.12]]
    [grounding_payload] = [payload for kind, payload in logger.records if kind == "grounding"]
    assert grounding_payload["trigger_decision"] is False
    assert grounding_payload["trigger_reason"] == "no_change"
    assert "slate_adds" not in c2_payload
    assert "rerank_used" not in c2_payload


@pytest.mark.asyncio
async def test_trigger_fire_summarizer_gets_accumulated_delta_and_accepted_result_emits_events() -> None:
    clock = FakeClock()
    action = Actionable(
        id="act-1",
        title="Send timeline",
        quote="I'll send the timeline Friday",
        grounding=["N1"],
        confidence=0.93,
    )
    update = GroundingUpdate(
        items=[grounding_item("direct", "Direct Note", "Direct gist", 0.91)],
        revision=3,
    )
    summary = MeetingSummary(
        rev=1,
        summary_md="Discussing the timeline [N1]",
        actionables=[action],
        grounding=update.items,
        updated_at=123.0,
    )
    ambient = SequenceAmbient(results=[update], embeddings=[np.array([1.0, 0.0])])
    trigger = RecordingTrigger([(True, "jaccard")])
    summarizer = RecordingSummarizer([accepted_result(summary=summary, new_actionables=[action])])
    session, events, trigger, summarizer, logger = session_with(
        index=FakeIndex(notes={"direct": note("direct")}),
        clock=clock,
        ambient=ambient,
        trigger=trigger,
        summarizer=summarizer,
    )

    await session.on_chunk(chunk("I'll send the timeline Friday"))
    await drain_summary_task()

    assert len(summarizer.calls) == 1
    state, transcript_delta, called_update, open_actionables = summarizer.calls[0]
    assert state is None
    assert transcript_delta == "[ch0] I'll send the timeline Friday"
    assert called_update is update
    assert open_actionables == []
    assert trigger.noted == [(update, False)]
    [summary_event] = [event for event in events if event.type == "summary"]
    assert summary_event.rev == 1
    assert summary_event.summary_md == "Discussing the timeline [N1]"
    assert summary_event.actionables == [action]
    [actionable_event] = [event for event in events if event.type == "actionable"]
    assert actionable_event.op == "new"
    assert actionable_event.item == action
    assert session.current_summary().rev == 1
    assert session._transcript_delta == ""
    [summary_payload] = [payload for kind, payload in logger.records if kind == "summary"]
    assert summary_payload["accepted_actionable_ids"] == ["act-1"]


@pytest.mark.asyncio
async def test_no_update_records_backoff_without_events_or_delta_reset() -> None:
    update = GroundingUpdate(items=[grounding_item("direct", "Direct Note", "Direct gist", 0.91)], revision=3)
    ambient = SequenceAmbient(results=[update])
    trigger = RecordingTrigger([(True, "jaccard")])
    summarizer = RecordingSummarizer(
        [SummaryResult(summary=None, no_update=True, latency_ms=5.0, model="model-a")]
    )
    session, events, trigger, _summarizer, logger = session_with(
        index=FakeIndex(notes={"direct": note("direct")}),
        clock=FakeClock(),
        ambient=ambient,
        trigger=trigger,
        summarizer=summarizer,
    )

    await session.on_chunk(chunk("minor chatter that changes nothing"))
    await drain_summary_task()

    assert trigger.noted == [(update, True)]
    assert [event.type for event in events].count("summary") == 0
    assert [event.type for event in events].count("actionable") == 0
    assert session._transcript_delta == "[ch0] minor chatter that changes nothing"
    [summary_payload] = [payload for kind, payload in logger.records if kind == "summary"]
    assert summary_payload["no_update"] is True
    assert summary_payload["rejected"] is False


@pytest.mark.asyncio
async def test_rejected_summary_records_pass_without_events_delta_reset_or_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "SUMMARY_MIN_INTERVAL_S", 10.0)
    monkeypatch.setattr(config, "SUMMARY_MIN_NEW_CHARS", 1)
    monkeypatch.setattr(config, "SUMMARY_CHANGE_TOP_K", 3)
    monkeypatch.setattr(config, "SUMMARY_CHANGE_JACCARD", 0.6)
    monkeypatch.setattr(config, "SUMMARY_EMPTY_BACKOFF_S", 40.0)

    clock = FakeClock()
    update = GroundingUpdate(
        items=[grounding_item("direct", "Direct Note", "Direct gist", 0.91)],
        revision=3,
    )
    ambient = SequenceAmbient(
        results=[update, update, update],
        embeddings=[
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ],
    )
    summarizer = RecordingSummarizer(
        [
            SummaryResult(
                summary=None,
                no_update=False,
                rejected=True,
                latency_ms=5.0,
                model="model-a",
                reject_reason="invalid_citations",
                citation_stats={"total": 1.0, "valid": 0.0, "invalid": 1.0},
            ),
            SummaryResult(
                summary=None,
                no_update=False,
                rejected=True,
                latency_ms=5.0,
                model="model-a",
                reject_reason="invalid_citations",
                citation_stats={"total": 1.0, "valid": 0.0, "invalid": 1.0},
            ),
        ]
    )
    previous_summary = MeetingSummary(
        rev=7,
        summary_md="Existing summary [N1]",
        grounding=update.items,
        updated_at=900.0,
    )
    session, events, _trigger, summarizer, logger = session_with(
        index=FakeIndex(notes={"direct": note("direct")}),
        clock=clock,
        ambient=ambient,
        trigger=SummaryTrigger(clock.now),
        summarizer=summarizer,
    )
    session._summary = previous_summary

    await session.on_chunk(chunk("first rejected update"))
    await drain_summary_task()

    assert session.current_summary() == previous_summary
    assert session._transcript_delta == "[ch0] first rejected update"
    assert [event.type for event in events].count("summary") == 0
    assert [event.type for event in events].count("actionable") == 0
    [summary_payload] = [payload for kind, payload in logger.records if kind == "summary"]
    assert summary_payload["no_update"] is False
    assert summary_payload["rejected"] is True

    await session.on_chunk(chunk("immediate rejected update"))
    await drain_summary_task()

    assert len(summarizer.calls) == 1
    grounding_reasons = [
        payload["trigger_reason"]
        for kind, payload in logger.records
        if kind == "grounding"
    ]
    assert grounding_reasons == ["jaccard", "min_interval"]

    clock.advance(10.1)
    await session.on_chunk(chunk("later rejected update"))
    await drain_summary_task()

    assert len(summarizer.calls) == 2
    assert session.current_summary() == previous_summary
    assert session._transcript_delta == (
        "[ch0] first rejected update\n"
        "[ch0] immediate rejected update\n"
        "[ch0] later rejected update"
    )
    assert [event.type for event in events].count("summary") == 0
    assert [event.type for event in events].count("actionable") == 0
    grounding_reasons = [
        payload["trigger_reason"]
        for kind, payload in logger.records
        if kind == "grounding"
    ]
    assert grounding_reasons == ["jaccard", "min_interval", "jaccard"]
    summary_payloads = [payload for kind, payload in logger.records if kind == "summary"]
    assert [payload["rejected"] for payload in summary_payloads] == [True, True]


@pytest.mark.asyncio
async def test_both_model_failure_throttles_like_rejection() -> None:
    clock = FakeClock()
    update = GroundingUpdate(
        items=[grounding_item("direct", "Direct Note", "Direct gist", 0.91)],
        revision=3,
    )
    ambient = SequenceAmbient(results=[update])
    trigger = RecordingTrigger([(True, "jaccard")])
    summarizer = RecordingSummarizer([None])
    previous_summary = MeetingSummary(rev=7, summary_md="Existing [N1]", grounding=update.items)
    session, events, trigger, summarizer, logger = session_with(
        index=FakeIndex(notes={"direct": note("direct")}),
        clock=clock,
        ambient=ambient,
        trigger=trigger,
        summarizer=summarizer,
    )
    session._summary = previous_summary

    await session.on_chunk(chunk("model outage window"))
    await drain_summary_task()

    assert trigger.rejected_count == 1
    assert trigger.noted == []
    assert session.current_summary() == previous_summary
    assert session._transcript_delta == "[ch0] model outage window"
    assert [event.type for event in events].count("summary") == 0
    assert [event.type for event in events].count("actionable") == 0
    [summary_payload] = [payload for kind, payload in logger.records if kind == "summary"]
    assert summary_payload["rejected"] is True
    assert summary_payload["reject_reason"] == "llm_unavailable"


@pytest.mark.asyncio
async def test_in_flight_summary_pass_skips_second_fire() -> None:
    release = asyncio.Event()
    first = GroundingUpdate(items=[grounding_item("one", "One", "One gist", 0.91)], revision=1)
    second = GroundingUpdate(items=[grounding_item("two", "Two", "Two gist", 0.92)], revision=2)
    ambient = SequenceAmbient(results=[first, second])
    trigger = RecordingTrigger([(True, "jaccard"), (True, "jaccard")])
    summary = MeetingSummary(rev=1, summary_md="One [N1]", grounding=first.items)
    summarizer = RecordingSummarizer([accepted_result(summary=summary)], wait_on=release)
    session, _events, _trigger, summarizer, logger = session_with(
        index=FakeIndex(notes={"one": note("one"), "two": note("two")}),
        clock=FakeClock(),
        ambient=ambient,
        trigger=trigger,
        summarizer=summarizer,
    )

    await session.on_chunk(chunk("first transcript"))
    await summarizer.started.wait()
    await session.on_chunk(chunk("second transcript"))

    assert len(summarizer.calls) == 1
    assert ("grounding", {"trigger_decision": False, "trigger_reason": "in_flight"}) in [
        (kind, {"trigger_decision": payload.get("trigger_decision"), "trigger_reason": payload.get("trigger_reason")})
        for kind, payload in logger.records
        if kind == "grounding"
    ]

    release.set()
    await drain_summary_task()


@pytest.mark.asyncio
async def test_in_flight_chunks_survive_accepted_pass_delta_reset() -> None:
    release = asyncio.Event()
    first = GroundingUpdate(items=[grounding_item("one", "One", "One gist", 0.91)], revision=1)
    second = GroundingUpdate(items=[grounding_item("one", "One", "One gist", 0.91)], revision=1)
    ambient = SequenceAmbient(results=[first, second])
    trigger = RecordingTrigger([(True, "jaccard"), (False, "min_interval")])
    summary = MeetingSummary(rev=1, summary_md="One [N1]", grounding=first.items)
    summarizer = RecordingSummarizer([accepted_result(summary=summary)], wait_on=release)
    session, _events, _trigger, summarizer, _logger = session_with(
        index=FakeIndex(notes={"one": note("one")}),
        clock=FakeClock(),
        ambient=ambient,
        trigger=trigger,
        summarizer=summarizer,
    )

    await session.on_chunk(chunk("first transcript"))
    await summarizer.started.wait()
    await session.on_chunk(chunk("arrived while llm in flight"))

    release.set()
    await drain_summary_task()

    # the accepted pass consumed only what it was dispatched with; the
    # in-flight chunk stays queued for the next pass
    assert summarizer.calls[0][1] == "[ch0] first transcript"
    assert session._transcript_delta == "[ch0] arrived while llm in flight"


@pytest.mark.asyncio
async def test_actionable_status_updates_session_state_and_emits_actionable_events() -> None:
    action = Actionable(
        id="act-1",
        title="Send timeline",
        quote="I'll send the timeline Friday",
        grounding=["N1"],
        confidence=0.93,
    )
    summary = MeetingSummary(rev=1, summary_md="Summary", actionables=[action])
    summarizer = RecordingSummarizer([accepted_result(summary=summary, new_actionables=[action])])
    session, events, _trigger, _summarizer, _logger = session_with(
        index=FakeIndex(notes={"direct": note("direct")}),
        clock=FakeClock(),
        ambient=SequenceAmbient(results=[GroundingUpdate(items=[grounding_item("direct", "Direct", "gist", 0.9)], revision=1)]),
        trigger=RecordingTrigger([(True, "jaccard")]),
        summarizer=summarizer,
    )
    await session.on_chunk(chunk("I'll send the timeline Friday"))
    await drain_summary_task()

    await session.set_actionable_status("act-1", "confirmed")
    await session.set_actionable_status("act-1", "dismissed")

    status_events = [event for event in events if event.type == "actionable"][1:]
    assert [(event.op, event.item.status) for event in status_events] == [
        ("update", "confirmed"),
        ("dismiss", "dismissed"),
    ]
    assert session.current_summary().actionables[0].status == "dismissed"


@pytest.mark.asyncio
async def test_c2_window_appends_partials_and_passes_context_text() -> None:
    ambient = SequenceAmbient(results=[None, None, None, None, GroundingUpdate(revision=1)])
    session, _events, _trigger, _summarizer, _logger = session_with(
        index=FakeIndex(),
        clock=FakeClock(),
        exact=SequenceExact([[], [], [], [], []]),
        ambient=ambient,
    )

    await session.on_chunk(chunk("chunk 1", ts=1.0))
    await session.on_chunk(chunk("chunk 2", ts=2.0))
    await session.on_chunk(chunk("chunk 3", ts=3.0))
    await session.on_chunk(chunk("chunk 4", ts=4.0))
    await session.on_chunk(interim("partial insight", ts=5.0))

    assert ambient.windows[-1] == "chunk 2 chunk 3 chunk 4 partial insight"
    assert ambient.contexts[-1] == "chunk 1 chunk 2 chunk 3 chunk 4 partial insight"
