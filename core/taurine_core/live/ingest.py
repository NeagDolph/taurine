from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Protocol
from uuid import uuid4

from taurine_core import config
from taurine_core.ports import IndexPort
from taurine_core.schemas import (
    AmbientEvent,
    AmbientItem,
    Card,
    Chunk,
    PinEvent,
    Pick,
    RetrievalEvent,
    SlateItem,
    TranscriptEvent,
    UnpinEvent,
)

from .ambient import AmbientRetriever
from .cadence import Cadence
from .exact import ExactCollector
from .slate import Slate


Event = TranscriptEvent | AmbientEvent | PinEvent | UnpinEvent | RetrievalEvent
EventSink = Callable[[Event], Awaitable[None] | None]


class Clock(Protocol):
    def now(self) -> float:
        ...


class PresenterLike(Protocol):
    async def present(
        self,
        transcript: list[Chunk],
        slate: list[SlateItem],
        via_titles: dict[str, str],
    ) -> list[Pick]:
        ...


class _SystemClock:
    def now(self) -> float:
        return time.monotonic()


class LiveSession:
    def __init__(
        self,
        *,
        index: IndexPort,
        slate: Slate,
        cadence: Cadence,
        exact: ExactCollector,
        ambient: AmbientRetriever,
        presenter: PresenterLike,
        emit: EventSink,
        clock: Clock | None = None,
    ) -> None:
        self._index = index
        self._slate = slate
        self._cadence = cadence
        self._exact = exact
        self._ambient = ambient
        self._presenter = presenter
        self._emit = emit
        self._clock = clock or _SystemClock()
        self._transcript: list[Chunk] = []
        self._pinned: list[Card] = []

    async def on_chunk(self, chunk: Chunk) -> None:
        await self._emit_event(
            TranscriptEvent(
                channel=chunk.channel,
                text=chunk.text,
                final=chunk.final,
                ts=chunk.ts,
            )
        )
        if not chunk.final:
            return

        self._transcript.append(chunk)
        await self._run_c1(chunk)
        await self._run_c2()
        await self._run_presenter()

    async def dismiss(self, card_id: str) -> None:
        card = next((candidate for candidate in self._pinned if candidate.card_id == card_id), None)
        if card is None:
            return

        before_revision = self._slate.revision
        self._pinned = [candidate for candidate in self._pinned if candidate.card_id != card_id]
        self._slate.dismiss(card.note_id, now=self._clock.now())
        await self._emit_event(UnpinEvent(card_id=card_id))
        await self._emit_ambient_if_changed(before_revision)

    def slate_items(self) -> list[SlateItem]:
        return self._slate.items()

    async def _run_c1(self, chunk: Chunk) -> None:
        before_revision = self._slate.revision
        now = self._clock.now()
        for note_id, _alias, is_title in self._exact.on_chunk(chunk):
            note = self._index.note(note_id)
            if note is None:
                continue
            if is_title and self._cadence.instant_pin_allowed(now=now):
                await self._pin(
                    Card(
                        card_id=_card_id(note_id),
                        note_id=note_id,
                        title=note.title,
                        summary=note.gist,
                        provenance="exact",
                        heard=chunk.text,
                    )
                )
                continue
            # rate-capped title hits fall through to a slate add

            self._slate.add(
                SlateItem(
                    note_id=note_id,
                    title=note.title,
                    gist=note.gist,
                    score=1.0,
                    source="exact",
                    heard=chunk.text,
                    decay=config.DECAY_EXACT,
                    added_rev=0,
                )
            )
        await self._emit_ambient_if_changed(before_revision)

    async def _run_c2(self) -> None:
        window_text = self._window_text()
        if not window_text:
            return

        before_revision = self._slate.revision
        result = await self._ambient.on_window(window_text)
        if result is None:
            return

        for item in self._ambient.last_slate_adds:
            self._slate.add(item)

        await self._emit_event(
            RetrievalEvent(
                heard=result.heard,
                hits=result.hits,
                expansions=result.expansions,
            )
        )
        await self._emit_ambient_if_changed(before_revision)

    async def _run_presenter(self) -> None:
        prompted_items = self._slate.items()
        if not prompted_items:
            return

        now = self._clock.now()
        attempt_revision = self._slate.revision
        if not self._cadence.may_present(attempt_revision, now=now):
            return

        self._cadence.note_attempt(now=now)
        picks = await self._presenter.present(
            self._transcript,
            prompted_items,
            self._via_titles(prompted_items),
        )

        self._slate.mark_viewed([item.note_id for item in prompted_items])
        if not picks:
            self._cadence.note_empty(attempt_revision, now=now)
            await self._emit_ambient_if_changed(attempt_revision)
            return

        item_by_id = {item.note_id: item for item in prompted_items}
        before_revision = self._slate.revision
        promoted = 0
        for pick in picks:
            item = item_by_id.get(pick.note_id)
            if item is None:
                continue
            await self._pin(_card_from_pick(pick, item))
            self._slate.kick(item.note_id, "pinned")
            self._cadence.note_promotion(now=now)
            promoted += 1

        if promoted == 0:
            self._cadence.note_empty(attempt_revision, now=now)
        await self._emit_ambient_if_changed(before_revision)

    async def _pin(self, card: Card) -> None:
        duplicate_ids = [
            existing.card_id for existing in self._pinned if existing.note_id == card.note_id
        ]
        self._pinned = [existing for existing in self._pinned if existing.note_id != card.note_id]
        for card_id in duplicate_ids:
            await self._emit_event(UnpinEvent(card_id=card_id))

        self._pinned.insert(0, card)
        await self._emit_event(PinEvent(card=card))

        while len(self._pinned) > config.MAX_PINNED:
            displaced = self._pinned.pop()
            await self._emit_event(UnpinEvent(card_id=displaced.card_id))

    async def _emit_ambient_if_changed(self, before_revision: int) -> None:
        if self._slate.revision == before_revision:
            return
        await self._emit_event(
            AmbientEvent(
                items=[
                    AmbientItem(
                        note_id=item.note_id,
                        title=item.title,
                        gist=item.gist,
                        score=item.score,
                    )
                    for item in self._slate.items()
                ]
            )
        )

    def _window_text(self) -> str:
        return " ".join(
            chunk.text for chunk in self._transcript[-config.C2_WINDOW_CHUNKS :]
        ).strip()

    def _via_titles(self, items: list[SlateItem]) -> dict[str, str]:
        titles: dict[str, str] = {}
        for item in items:
            if not item.via_note_id or item.via_note_id in titles:
                continue
            note = self._index.note(item.via_note_id)
            titles[item.via_note_id] = note.title if note is not None else item.via_note_id
        return titles

    async def _emit_event(self, event: Event) -> None:
        result = self._emit(event)
        if isawaitable(result):
            await result


def _card_from_pick(pick: Pick, item: SlateItem) -> Card:
    provenance = "linked" if item.via_note_id else item.source
    return Card(
        card_id=_card_id(item.note_id),
        note_id=item.note_id,
        title=item.title,
        summary=pick.why,
        provenance=provenance,
        heard=item.heard,
    )


def _card_id(note_id: str) -> str:
    return f"{note_id}:{uuid4().hex[:12]}"
