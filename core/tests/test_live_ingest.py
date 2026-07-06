from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live.ingest import LiveSession
from taurine_core.live.slate import Slate
from taurine_core.schemas import Chunk, Pick, SlateItem

from tests.fakes import FakeClock, FakeIndex, note


class NoopAmbient:
    last_slate_adds: list[SlateItem] = []

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

    def note_attempt(self, now: float | None = None) -> None:
        self.attempts.append(float(now or 0.0))

    def note_promotion(self, now: float | None = None) -> None:
        self.promotions.append(float(now or 0.0))

    def note_empty(self, revision: int, now: float | None = None) -> None:
        self.empties.append((revision, float(now or 0.0)))


@dataclass
class StaticPresenter:
    picks: list[Pick]
    calls: list[tuple[list[Chunk], list[SlateItem], dict[str, str]]] = field(default_factory=list)

    async def present(
        self,
        transcript: list[Chunk],
        slate: list[SlateItem],
        via_titles: dict[str, str],
    ) -> list[Pick]:
        self.calls.append((transcript, slate, via_titles))
        return self.picks


def chunk(text: str, ts: float = 1000.0) -> Chunk:
    return Chunk(channel=0, text=text, ts=ts, final=True)


def session_with(
    *,
    index: FakeIndex,
    clock: FakeClock,
    exact: SequenceExact,
    cadence: RecordingCadence,
    presenter: StaticPresenter | None = None,
) -> tuple[LiveSession, list[Any]]:
    events: list[Any] = []
    session = LiveSession(
        index=index,
        slate=Slate(clock),
        cadence=cadence,
        exact=exact,
        ambient=NoopAmbient(),
        presenter=presenter or StaticPresenter([]),
        emit=events.append,
        clock=clock,
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
async def test_dismiss_unpins_and_suppresses_slate_reentry() -> None:
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

    assert [item.note_id for item in session.slate_items()] == ["roadmap"]
