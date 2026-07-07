from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live.ingest import LiveSession
from taurine_core.live.slate import Slate
from taurine_core.schemas import Chunk, Expansion, Hit, Pick, RetrievalResult, SlateItem

from tests.fakes import FakeClock, FakeIndex, note


class NoopAmbient:
    last_slate_adds: list[SlateItem] = []
    last_debug: dict[str, Any] | None = None

    async def on_window(self, window_text: str) -> None:
        del window_text
        return None


@dataclass
class SequenceExact:
    hits: list[list[tuple[str, str, bool]]]

    def on_chunk(self, chunk: Chunk) -> list[tuple[str, str, bool]]:
        del chunk
        return self.hits.pop(0) if self.hits else []


@dataclass
class SequenceAmbient:
    results: list[RetrievalResult | None]
    slate_add_sequences: list[list[SlateItem]]
    debug_sequences: list[dict[str, Any] | None] = field(default_factory=list)
    windows: list[str] = field(default_factory=list)
    last_slate_adds: list[SlateItem] = field(default_factory=list)
    last_debug: dict[str, Any] | None = None

    async def on_window(self, window_text: str) -> RetrievalResult | None:
        self.windows.append(window_text)
        self.last_slate_adds = (
            self.slate_add_sequences.pop(0) if self.slate_add_sequences else []
        )
        self.last_debug = self.debug_sequences.pop(0) if self.debug_sequences else None
        return self.results.pop(0) if self.results else None


@dataclass
class RecordingCadence:
    instant_allowed: bool = True
    may_present_allowed: bool = False
    attempts: list[float] = field(default_factory=list)
    promotions: list[float] = field(default_factory=list)
    empties: list[tuple[int, float]] = field(default_factory=list)

    def instant_pin_allowed(self, now: float | None = None) -> bool:
        del now
        return self.instant_allowed

    def may_present(self, revision: int, now: float | None = None) -> bool:
        del revision, now
        return self.may_present_allowed

    def explain(self, revision: int, now: float | None = None) -> str:
        del revision, now
        return "allowed" if self.may_present_allowed else "same_revision"

    def note_attempt(self, now: float | None = None) -> None:
        self.attempts.append(float(now or 0.0))

    def note_promotion(self, now: float | None = None) -> None:
        self.promotions.append(float(now or 0.0))

    def note_empty(self, revision: int, now: float | None = None) -> None:
        self.empties.append((revision, float(now or 0.0)))


@dataclass
class StaticPresenter:
    picks: list[Pick]
    calls: list[tuple[list[Chunk], list[SlateItem], dict[str, str], dict[int, str] | None]] = field(default_factory=list)

    async def present(
        self,
        transcript: list[Chunk],
        slate: list[SlateItem],
        via_titles: dict[str, str],
        *,
        revision: int | None = None,
        partials: dict[int, str] | None = None,
    ) -> list[Pick]:
        del revision
        self.calls.append((transcript, slate, via_titles, partials))
        return self.picks


@dataclass
class RecordingLogger:
    records: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def log(self, kind: str, **payload: Any) -> None:
        self.records.append((kind, payload))

    def close(self) -> None:
        self.records.append(("closed", {}))


def chunk(text: str, ts: float = 1000.0) -> Chunk:
    return Chunk(channel=0, text=text, ts=ts, final=True)


def interim(text: str, ts: float = 1000.0, channel: int = 0) -> Chunk:
    return Chunk(channel=channel, text=text, ts=ts, final=False)


def session_with(
    *,
    index: FakeIndex,
    clock: FakeClock,
    exact: SequenceExact,
    cadence: RecordingCadence,
    slate: Slate | None = None,
    ambient: Any | None = None,
    presenter: StaticPresenter | None = None,
    logger: RecordingLogger | None = None,
) -> tuple[LiveSession, list[Any]]:
    events: list[Any] = []
    session = LiveSession(
        index=index,
        slate=slate or Slate(clock),
        cadence=cadence,
        exact=exact,
        ambient=ambient or NoopAmbient(),
        presenter=presenter or StaticPresenter([]),
        emit=events.append,
        clock=clock,
        logger=logger or RecordingLogger(),
    )
    return session, events


@pytest.mark.asyncio
async def test_title_hit_produces_instant_pin_without_starting_cooldown() -> None:
    clock = FakeClock()
    index = FakeIndex(notes={"roadmap": note("roadmap", "Q3 Roadmap", "Roadmap gist")})
    cadence = RecordingCadence(instant_allowed=True, may_present_allowed=False)
    session, events = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact([[("roadmap", "q3 roadmap", True)]]),
        cadence=cadence,
    )

    await session.on_chunk(chunk("let's open the q3 roadmap"))

    pins = [event for event in events if event.type == "pin"]
    assert len(pins) == 1
    assert pins[0].card.note_id == "roadmap"
    assert pins[0].card.summary == "Roadmap gist"
    assert pins[0].card.provenance == "exact"
    assert cadence.promotions == []


@pytest.mark.asyncio
async def test_transcript_seq_and_c1_trigger_events_cover_pin_slate_and_none() -> None:
    clock = FakeClock()
    index = FakeIndex(
        notes={
            "pinned": note("pinned", "Pinned Title", "Pinned gist"),
            "slated": note("slated", "Slated Title", "Slated gist"),
            "suppressed": note("suppressed", "Suppressed Title", "Suppressed gist"),
        }
    )
    preloaded_slate = Slate(clock)
    preloaded_slate.add(
        SlateItem(
            note_id="suppressed",
            title="Suppressed Title",
            gist="Suppressed gist",
            score=1.0,
            source="exact",
            heard="old",
            decay=config.DECAY_EXACT,
            added_rev=0,
        )
    )
    preloaded_slate.dismiss("suppressed", now=clock.now())
    session, events = session_with(
        index=index,
        clock=clock,
        slate=preloaded_slate,
        exact=SequenceExact(
            [
                [],
                [
                    ("pinned", "pinned title", True),
                    ("slated", "slated alias", False),
                    ("suppressed", "suppressed alias", False),
                ],
            ]
        ),
        cadence=RecordingCadence(instant_allowed=True, may_present_allowed=False),
    )

    await session.on_chunk(Chunk(channel=0, text="interim words", ts=10.0, final=False))
    await session.on_chunk(chunk("pinned title plus slated alias and suppressed alias", ts=11.0))

    transcripts = [event for event in events if event.type == "transcript"]
    assert [event.seq for event in transcripts] == [0, 1]

    triggers = [event for event in events if event.type == "trigger"]
    assert [
        (event.kind, event.seqs, event.span, event.matches[0].note_id, event.action)
        for event in triggers
    ] == [
        ("exact", [1], "pinned title", "pinned", "pinned"),
        ("exact", [1], "slated alias", "slated", "slated"),
        ("exact", [1], "suppressed alias", "suppressed", "none"),
    ]
    assert all(trigger.matches[0].score == 1.0 for trigger in triggers)

    event_types = [event.type for event in events]
    assert event_types.index("trigger") < event_types.index("pin")
    assert "slated" in [item.note_id for item in session.slate_items()]
    assert "suppressed" not in [item.note_id for item in session.slate_items()]


@pytest.mark.asyncio
async def test_c2_trigger_event_correlates_with_retrieval_event_and_logger() -> None:
    clock = FakeClock()
    logger = RecordingLogger()
    index = FakeIndex(
        notes={
            "direct": note("direct", "Direct Note", "Direct gist"),
            "expanded": note("expanded", "Expanded Note", "Expanded gist"),
        }
    )
    result = RetrievalResult(
        heard="window text",
        hits=[Hit(note_id="direct", rank=1, score=0.91)],
        expansions=[
            Expansion(
                from_note_id="direct",
                note_id="expanded",
                hop=1,
                score=0.31,
            )
        ],
        extracted=["shopify checkout", "Ashish"],
    )
    debug = {
        "window": "window text",
        "salient": "window text",
        "dense": [["direct", 0.9]],
        "dense_salient": [["direct", 0.8]],
        "gliner_spans": [["shopify checkout", 0.95], ["Ashish", 0.9]],
        "dense_gliner": [["direct", 0.87]],
        "bm25": [],
        "fused": [["direct", 0.016]],
        "expansions": [{"from": "direct", "to": "expanded", "score": 0.31}],
        "rerank_used": True,
        "rerank_error": None,
        "ordered": [["direct", 0.91]],
        "floor_dropped": [],
        "slate_adds": ["direct"],
    }
    ambient = SequenceAmbient(
        results=[result],
        slate_add_sequences=[
            [
                SlateItem(
                    note_id="direct",
                    title="Direct Note",
                    gist="Direct gist",
                    score=0.91,
                    source="search",
                    heard="window text",
                    decay=config.DECAY_SEARCH,
                    added_rev=0,
                )
            ]
        ],
        debug_sequences=[debug],
    )
    session, events = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact([[]]),
        cadence=RecordingCadence(may_present_allowed=False),
        ambient=ambient,
        logger=logger,
    )

    await session.on_chunk(chunk("window text"))

    [trigger] = [event for event in events if event.type == "trigger"]
    [retrieval] = [event for event in events if event.type == "retrieval"]
    assert trigger.kind == "search"
    assert trigger.seqs == [1]
    assert trigger.span == "window text"
    assert trigger.action == "slated"
    assert [(match.note_id, match.rank, match.score, match.via_note_id, match.hop) for match in trigger.matches] == [
        ("direct", 1, 0.91, None, None),
        ("expanded", None, 0.31, "direct", 1),
    ]
    assert retrieval.retrieval_id == trigger.trigger_id
    assert retrieval.seqs == [1]
    assert retrieval.extracted == ["shopify checkout", "Ashish"]
    assert [kind for kind, _payload in logger.records].count("c2_pass") == 1
    c2_payload = [payload for kind, payload in logger.records if kind == "c2_pass"][0]
    assert c2_payload["trigger_id"] == trigger.trigger_id
    assert c2_payload["seqs"] == [1]
    assert c2_payload["slate_adds"] == ["direct"]
    assert c2_payload["gliner_spans"] == [["shopify checkout", 0.95], ["Ashish", 0.9]]
    assert c2_payload["dense_gliner"] == [["direct", 0.87]]


@pytest.mark.asyncio
async def test_c2_trigger_event_action_none_when_no_slate_add_survives() -> None:
    clock = FakeClock()
    index = FakeIndex(notes={"direct": note("direct", "Direct Note", "Direct gist")})
    result = RetrievalResult(
        heard="window text",
        hits=[Hit(note_id="direct", rank=1, score=0.91)],
        expansions=[],
    )
    ambient = SequenceAmbient(results=[result], slate_add_sequences=[[]])
    session, events = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact([[]]),
        cadence=RecordingCadence(may_present_allowed=False),
        ambient=ambient,
    )

    await session.on_chunk(chunk("window text"))

    [trigger] = [event for event in events if event.type == "trigger"]
    assert trigger.kind == "search"
    assert trigger.action == "none"
    assert trigger.seqs == [1]


@pytest.mark.asyncio
async def test_interim_c1_fires_once_per_segment_and_resets_after_final() -> None:
    clock = FakeClock()
    logger = RecordingLogger()
    index = FakeIndex(notes={"roadmap": note("roadmap", "Roadmap Note", "Roadmap gist")})
    session, events = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact(
            [
                [("roadmap", "roadmap", False)],
                [("roadmap", "roadmap", False)],
                [("roadmap", "roadmap", False)],
            ]
        ),
        cadence=RecordingCadence(may_present_allowed=False),
        logger=logger,
    )

    await session.on_chunk(interim("roadmap is coming together", ts=10.0))
    await session.on_chunk(chunk("roadmap is coming together", ts=11.0))
    await session.on_chunk(interim("roadmap follow-up", ts=12.0))

    transcripts = [event for event in events if event.type == "transcript"]
    assert [(event.final, event.seq) for event in transcripts] == [(False, 0), (True, 1), (False, 0)]

    triggers = [event for event in events if event.type == "trigger"]
    assert [(event.kind, event.seqs, event.span, event.matches[0].note_id) for event in triggers] == [
        ("exact", [], "roadmap", "roadmap"),
        ("exact", [], "roadmap", "roadmap"),
    ]
    assert [payload["seq"] for kind, payload in logger.records if kind == "c1_hit"] == [0, 0]


@pytest.mark.asyncio
async def test_c2_window_appends_partials_and_fires_on_interim_with_final_seqs_only() -> None:
    clock = FakeClock()
    index = FakeIndex(notes={"direct": note("direct", "Direct Note", "Direct gist")})
    result = RetrievalResult(
        heard="final context partial insight",
        hits=[Hit(note_id="direct", rank=1, score=0.91)],
        expansions=[],
    )
    ambient = SequenceAmbient(
        results=[None, result],
        slate_add_sequences=[
            [],
            [
                SlateItem(
                    note_id="direct",
                    title="Direct Note",
                    gist="Direct gist",
                    score=0.91,
                    source="search",
                    heard="final context partial insight",
                    decay=config.DECAY_SEARCH,
                    added_rev=0,
                )
            ],
        ],
    )
    session, events = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact([[], []]),
        cadence=RecordingCadence(may_present_allowed=False),
        ambient=ambient,
    )

    await session.on_chunk(chunk("final context", ts=10.0))
    await session.on_chunk(interim("partial insight", ts=11.0))

    assert ambient.windows == ["final context", "final context partial insight"]
    [trigger] = [event for event in events if event.type == "trigger"]
    [retrieval] = [event for event in events if event.type == "retrieval"]
    assert trigger.seqs == [1]
    assert retrieval.seqs == [1]
    assert trigger.span == "final context partial insight"


@pytest.mark.asyncio
async def test_presenter_receives_current_partials_on_interim_attempt() -> None:
    clock = FakeClock()
    index = FakeIndex(notes={"roadmap": note("roadmap", "Roadmap Note", "Roadmap gist")})
    slate = Slate(clock)
    slate.add(
        SlateItem(
            note_id="roadmap",
            title="Roadmap Note",
            gist="Roadmap gist",
            score=0.8,
            source="search",
            heard="previous",
            decay=config.DECAY_SEARCH,
            added_rev=0,
        )
    )
    presenter = StaticPresenter([])
    session, _events = session_with(
        index=index,
        clock=clock,
        slate=slate,
        exact=SequenceExact([[]]),
        cadence=RecordingCadence(may_present_allowed=True),
        presenter=presenter,
    )

    await session.on_chunk(interim("still describing the roadmap", ts=10.0))

    assert len(presenter.calls) == 1
    transcript, prompted, _via_titles, partials = presenter.calls[0]
    assert transcript == []
    assert [item.note_id for item in prompted] == ["roadmap"]
    assert partials == {0: "still describing the roadmap"}


@pytest.mark.asyncio
async def test_presenter_empty_emits_no_pin_and_notes_empty_backoff() -> None:
    clock = FakeClock()
    index = FakeIndex(notes={"roadmap": note("roadmap", "Q3 Roadmap")})
    cadence = RecordingCadence(instant_allowed=False, may_present_allowed=True)
    presenter = StaticPresenter([])
    session, events = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact([[("roadmap", "roadmap", False)]]),
        cadence=cadence,
        presenter=presenter,
    )

    await session.on_chunk(chunk("maybe the roadmap note is relevant"))

    assert [event.type for event in events].count("pin") == 0
    assert len(presenter.calls) == 1
    assert len(cadence.attempts) == 1
    assert cadence.promotions == []
    assert cadence.empties == [(1, clock.now())]


@pytest.mark.asyncio
async def test_presenter_non_slate_pick_pins_index_note_as_linked_card() -> None:
    clock = FakeClock()
    index = FakeIndex(
        notes={
            "seed": note("seed", "Seed Note", "Seed gist"),
            "neighbor": note("neighbor", "Neighbor Note", "Neighbor gist"),
        }
    )
    slate = Slate(clock)
    slate.add(
        SlateItem(
            note_id="seed",
            title="Seed Note",
            gist="Seed gist",
            score=0.8,
            source="search",
            heard="seed heard",
            decay=config.DECAY_SEARCH,
            added_rev=0,
        )
    )
    cadence = RecordingCadence(instant_allowed=False, may_present_allowed=True)
    presenter = StaticPresenter([Pick(note_id="neighbor", why="Armaan was named")])
    session, events = session_with(
        index=index,
        clock=clock,
        slate=slate,
        exact=SequenceExact([[]]),
        cadence=cadence,
        presenter=presenter,
    )

    await session.on_chunk(chunk("Armand came up again"))

    pins = [event for event in events if event.type == "pin"]
    assert len(pins) == 1
    assert pins[0].card.note_id == "neighbor"
    assert pins[0].card.title == "Neighbor Note"
    assert pins[0].card.summary == "Armaan was named"
    assert pins[0].card.provenance == "linked"
    assert cadence.promotions == [clock.now()]
    assert cadence.empties == []


@pytest.mark.asyncio
async def test_dismiss_unpins_and_shown_note_never_reenters_slate() -> None:
    clock = FakeClock()
    index = FakeIndex(notes={"roadmap": note("roadmap", "Q3 Roadmap")})
    cadence = RecordingCadence(instant_allowed=True, may_present_allowed=False)
    session, events = session_with(
        index=index,
        clock=clock,
        exact=SequenceExact(
            [
                [("roadmap", "q3 roadmap", True)],
                [("roadmap", "roadmap", False)],
                [("roadmap", "roadmap", False)],
            ]
        ),
        cadence=cadence,
    )

    await session.on_chunk(chunk("show q3 roadmap"))
    card_id = [event.card.card_id for event in events if event.type == "pin"][0]
    await session.dismiss(card_id)
    await session.on_chunk(chunk("roadmap should stay suppressed"))

    assert [event.card_id for event in events if event.type == "unpin"] == [card_id]
    assert session.slate_items() == []

    clock.advance(config.SUPPRESS_S + 0.1)
    await session.on_chunk(chunk("roadmap can enter the slate later"))

    assert session.slate_items() == []
