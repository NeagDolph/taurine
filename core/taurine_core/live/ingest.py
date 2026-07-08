from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any, Protocol
from uuid import uuid4

from taurine_core import config
from taurine_core.ports import IndexPort
from taurine_core.schemas import (
    AmbientEvent,
    AmbientItem,
    Card,
    Chunk,
    Expansion,
    Hit,
    PinEvent,
    Pick,
    RetrievalResult,
    RetrievalEvent,
    SlateItem,
    TranscriptEvent,
    TriggerEvent,
    TriggerMatch,
    UnpinEvent,
)
from taurine_core.session_log import NullSessionLogger

from .ambient import AmbientRetriever
from .cadence import Cadence
from .exact import ExactCollector
from .grounding import GroundingUpdate
from .slate import Slate


Event = TranscriptEvent | AmbientEvent | PinEvent | UnpinEvent | RetrievalEvent | TriggerEvent
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
        *,
        revision: int | None = None,
        partials: dict[int, str] | None = None,
    ) -> list[Pick]:
        ...


class LoggerLike(Protocol):
    def log(self, kind: str, **payload: Any) -> None:
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
        logger: LoggerLike | None = None,
    ) -> None:
        self._index = index
        self._slate = slate
        self._cadence = cadence
        self._exact = exact
        self._ambient = ambient
        self._presenter = presenter
        self._emit = emit
        self._clock = clock or _SystemClock()
        self._logger = logger or NullSessionLogger()
        self._transcript: list[Chunk] = []
        self._partials: dict[int, str] = {}
        self._c1_segment_seen: dict[int, set[str]] = {}
        self._pinned: list[Card] = []
        self._next_seq = 1
        self.last_grounding: GroundingUpdate | None = None

    async def on_chunk(self, chunk: Chunk) -> None:
        chunk = self._chunk_with_seq(chunk)
        self._logger.log(
            "chunk",
            seq=chunk.seq,
            channel=chunk.channel,
            text=chunk.text,
            ts=chunk.ts,
            final=chunk.final,
        )
        await self._emit_event(
            TranscriptEvent(
                channel=chunk.channel,
                text=chunk.text,
                final=chunk.final,
                ts=chunk.ts,
                seq=chunk.seq,
            )
        )
        if not chunk.final:
            self._partials[chunk.channel] = chunk.text
            await self._run_c1(chunk)
            await self._run_c2()
            await self._run_presenter()
            return

        self._partials.pop(chunk.channel, None)
        self._transcript.append(chunk)
        await self._run_c1(chunk)
        self._c1_segment_seen.pop(chunk.channel, None)
        await self._run_c2()
        await self._run_presenter()

    async def dismiss(self, card_id: str) -> None:
        card = next((candidate for candidate in self._pinned if candidate.card_id == card_id), None)
        if card is None:
            return

        before_revision = self._slate.revision
        self._pinned = [candidate for candidate in self._pinned if candidate.card_id != card_id]
        self._slate.dismiss(card.note_id, now=self._clock.now())
        self._logger.log("unpin", card_id=card_id, reason="dismissed")
        await self._emit_event(UnpinEvent(card_id=card_id))
        await self._emit_ambient_if_changed(before_revision)

    def slate_items(self) -> list[SlateItem]:
        return self._slate.items()

    async def _run_c1(self, chunk: Chunk) -> None:
        before_revision = self._slate.revision
        now = self._clock.now()
        segment_seen = self._c1_segment_seen.setdefault(chunk.channel, set())
        seqs = [chunk.seq] if chunk.final else []
        for note_id, alias, is_title in self._exact.on_chunk(chunk):
            if note_id in segment_seen:
                continue
            segment_seen.add(note_id)
            note = self._index.note(note_id)
            if note is None:
                continue
            if (
                is_title
                and note_id not in self._slate.shown
                and self._cadence.instant_pin_allowed(now=now)
            ):
                await self._emit_trigger(
                    TriggerEvent(
                        trigger_id=_trigger_id(),
                        kind="exact",
                        seqs=seqs,
                        span=alias,
                        matches=[
                            TriggerMatch(
                                note_id=note_id,
                                title=note.title,
                                score=1.0,
                            )
                        ],
                        action="pinned",
                    )
                )
                self._logger.log(
                    "c1_hit",
                    seq=chunk.seq,
                    alias=alias,
                    note_id=note_id,
                    title=note.title,
                    is_title=is_title,
                    action="pinned",
                )
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

            add_revision = self._slate.revision
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
            action = "slated" if self._slate.revision != add_revision else "none"
            await self._emit_trigger(
                TriggerEvent(
                    trigger_id=_trigger_id(),
                    kind="exact",
                    seqs=seqs,
                    span=alias,
                    matches=[
                        TriggerMatch(
                            note_id=note_id,
                            title=note.title,
                            score=1.0,
                        )
                    ],
                    action=action,
                )
            )
            self._logger.log(
                "c1_hit",
                seq=chunk.seq,
                alias=alias,
                note_id=note_id,
                title=note.title,
                is_title=is_title,
                action=action,
            )
        await self._emit_ambient_if_changed(before_revision)

    async def _run_c2(self) -> None:
        window_chunks = self._window_chunks()
        window_text = self._window_text(window_chunks, self._active_partials())
        if not window_text:
            return
        context_chunks = self._transcript[
            -(config.C2_WINDOW_CHUNKS + config.C2_CONTEXT_CHUNKS) :
        ]
        context_text = self._window_text(context_chunks, self._active_partials())

        before_revision = self._slate.revision
        ambient_result = await self._ambient.on_window(window_text, context_text=context_text)
        if ambient_result is None:
            return

        grounding = (
            ambient_result
            if isinstance(ambient_result, GroundingUpdate)
            else getattr(self._ambient, "last_grounding", None)
        )
        self.last_grounding = grounding
        result = (
            self._retrieval_from_grounding(window_text, grounding)
            if grounding is not None
            else ambient_result
        )
        slate_adds = (
            self._slate_items_from_grounding(grounding, window_text)
            if grounding is not None
            else list(getattr(self._ambient, "last_slate_adds", []))
        )

        trigger_id = _trigger_id()
        for item in slate_adds:
            self._slate.add(item)
        slate_ids = {item.note_id for item in self._slate.items()}
        survived_adds = [item.note_id for item in slate_adds if item.note_id in slate_ids]
        seqs = [chunk.seq for chunk in window_chunks]
        await self._emit_trigger(
            TriggerEvent(
                trigger_id=trigger_id,
                kind="search",
                seqs=seqs,
                span=result.heard,
                matches=self._search_trigger_matches(result),
                action="slated" if survived_adds else "none",
            )
        )
        self._log_c2_pass(trigger_id, seqs, result, survived_adds)

        await self._emit_event(
            RetrievalEvent(
                heard=result.heard,
                hits=result.hits,
                expansions=result.expansions,
                retrieval_id=trigger_id,
                seqs=seqs,
                extracted=result.extracted,
            )
        )
        await self._emit_ambient_if_changed(before_revision)

    async def _run_presenter(self) -> None:
        prompted_items = self._slate.items()
        if not prompted_items:
            return

        now = self._clock.now()
        attempt_revision = self._slate.revision
        reason = self._cadence.explain(attempt_revision, now=now)
        decision = self._cadence.may_present(attempt_revision, now=now)
        self._logger.log(
            "cadence",
            decision=decision,
            revision=attempt_revision,
            reason=reason,
        )
        if not decision:
            return

        self._cadence.note_attempt(now=now)
        picks = await self._presenter.present(
            self._transcript,
            prompted_items,
            self._via_titles(prompted_items),
            revision=attempt_revision,
            partials=self._active_partials() or None,
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
            card = (
                _card_from_pick(pick, item)
                if item is not None
                else self._card_from_index_pick(pick)
            )
            if card is None:
                continue
            if not await self._pin(card):
                continue
            if item is not None:
                self._slate.kick(item.note_id, "pinned")
            self._cadence.note_promotion(now=now)
            promoted += 1

        if promoted == 0:
            self._cadence.note_empty(attempt_revision, now=now)
        await self._emit_ambient_if_changed(before_revision)

    def _card_from_index_pick(self, pick: Pick) -> Card | None:
        note = self._index.note(pick.note_id)
        if note is None:
            return None
        return Card(
            card_id=_card_id(note.note_id),
            note_id=note.note_id,
            title=note.title,
            summary=pick.why,
            provenance="linked",
            heard=self._window_text(self._window_chunks(), self._active_partials()),
        )

    async def _pin(self, card: Card) -> bool:
        if card.note_id in self._slate.shown:
            return False
        self._slate.mark_shown(card.note_id)
        self._slate.kick(card.note_id, "pinned")
        duplicate_ids = [
            existing.card_id for existing in self._pinned if existing.note_id == card.note_id
        ]
        self._pinned = [existing for existing in self._pinned if existing.note_id != card.note_id]
        for card_id in duplicate_ids:
            self._logger.log("unpin", card_id=card_id, reason="superseded")
            await self._emit_event(UnpinEvent(card_id=card_id))

        self._pinned.insert(0, card)
        self._logger.log(
            "pin",
            card_id=card.card_id,
            note_id=card.note_id,
            title=card.title,
            provenance=card.provenance,
            summary=card.summary,
        )
        await self._emit_event(PinEvent(card=card))

        while len(self._pinned) > config.MAX_PINNED:
            displaced = self._pinned.pop()
            self._logger.log("unpin", card_id=displaced.card_id, reason="displaced")
            await self._emit_event(UnpinEvent(card_id=displaced.card_id))
        return True

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

    def _window_chunks(self) -> list[Chunk]:
        return self._transcript[-config.C2_WINDOW_CHUNKS :]

    def _window_text(self, chunks: list[Chunk], partials: dict[int, str]) -> str:
        parts = [chunk.text for chunk in chunks]
        parts.extend(text for _channel, text in sorted(partials.items()))
        return " ".join(parts).strip()

    def _active_partials(self) -> dict[int, str]:
        return {
            channel: text.strip()
            for channel, text in self._partials.items()
            if text.strip()
        }

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

    async def _emit_trigger(self, event: TriggerEvent) -> None:
        await self._emit_event(event)

    def _chunk_with_seq(self, chunk: Chunk) -> Chunk:
        if not chunk.final:
            return chunk.model_copy(update={"seq": 0})
        seq = self._next_seq
        self._next_seq += 1
        return chunk.model_copy(update={"seq": seq})

    def _search_trigger_matches(self, result: Any) -> list[TriggerMatch]:
        matches: list[TriggerMatch] = []
        for hit in result.hits:
            note = self._index.note(hit.note_id)
            matches.append(
                TriggerMatch(
                    note_id=hit.note_id,
                    title=note.title if note is not None else hit.note_id,
                    rank=hit.rank,
                    score=hit.score,
                )
            )
        for expansion in result.expansions:
            note = self._index.note(expansion.note_id)
            matches.append(
                TriggerMatch(
                    note_id=expansion.note_id,
                    title=note.title if note is not None else expansion.note_id,
                    score=expansion.score,
                    via_note_id=expansion.from_note_id,
                    hop=expansion.hop,
                )
            )
        return matches

    def _retrieval_from_grounding(
        self,
        heard: str,
        update: GroundingUpdate,
    ) -> RetrievalResult:
        fused_ids = set(getattr(self._ambient, "last_fused_ids", set()))
        extracted = list(getattr(self._ambient, "last_extracted", []))
        hits = [
            Hit(note_id=item.note_id, rank=rank, score=item.score)
            for rank, item in enumerate(update.items, start=1)
        ]
        expansions = [
            Expansion(
                from_note_id="",
                note_id=note_id,
                hop=0,
                score=mass,
                via="ppr",
            )
            for note_id, mass in update.ppr_top
            if note_id not in fused_ids
        ]
        return RetrievalResult(
            heard=heard,
            hits=hits,
            expansions=expansions,
            extracted=extracted,
        )

    def _slate_items_from_grounding(
        self,
        update: GroundingUpdate,
        heard: str,
    ) -> list[SlateItem]:
        return [
            SlateItem(
                note_id=item.note_id,
                title=item.title,
                gist=item.gist,
                score=item.score,
                source="search",
                heard=heard,
                decay=config.DECAY_SEARCH,
                added_rev=0,
            )
            for item in update.items
        ]

    def _log_c2_pass(
        self,
        trigger_id: str,
        seqs: list[int],
        result: Any,
        survived_adds: list[str],
    ) -> None:
        debug = dict(getattr(self._ambient, "last_debug", None) or {})
        if not debug:
            debug = {
                "window": result.heard,
                "gliner_spans": [],
                "gliner_dropped": [],
                "gliner_context_queries": [],
                "dense": [],
                "dense_gliner": [],
                "bm25": [],
                "exact": [],
                "fused": [],
                "ppr_top": [],
                "blend": [],
                "grounding_revision": (
                    self.last_grounding.revision if self.last_grounding is not None else 0
                ),
                "slate_adds": [],
            }
        debug.setdefault("gliner_spans", [])
        debug.setdefault("gliner_dropped", [])
        debug.setdefault("gliner_context_queries", [])
        debug.setdefault("dense_gliner", [])
        debug["slate_adds"] = survived_adds
        self._logger.log("c2_pass", trigger_id=trigger_id, seqs=seqs, **debug)


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


def _trigger_id() -> str:
    return uuid4().hex[:12]
