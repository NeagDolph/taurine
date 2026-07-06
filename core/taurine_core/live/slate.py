from __future__ import annotations

import time
from typing import Protocol

from taurine_core import config
from taurine_core.schemas import SlateItem


class Clock(Protocol):
    def now(self) -> float:
        ...


class _SystemClock:
    def now(self) -> float:
        return time.monotonic()


class Slate:
    def __init__(self, clock: Clock | None = None) -> None:
        self.revision = 0
        self._clock = clock or _SystemClock()
        self._items: dict[str, SlateItem] = {}
        self._suppressed_until: dict[str, float] = {}

    def add(self, item: SlateItem) -> None:
        suppressed_until = self._suppressed_until.get(item.note_id)
        if suppressed_until is not None:
            if self._clock.now() < suppressed_until:
                return
            del self._suppressed_until[item.note_id]

        existing = self._items.get(item.note_id)
        if existing is not None:
            item.source = self._stronger_source(existing.source, item.source)

        self.revision += 1
        item.views = 0
        item.added_rev = self.revision
        self._items[item.note_id] = item
        self._evict_over_quota(item.source)

    def kick(self, note_id: str, reason: str) -> None:
        del reason
        if note_id not in self._items:
            return
        del self._items[note_id]
        self.revision += 1

    def dismiss(self, note_id: str, now: float) -> None:
        self._suppressed_until[note_id] = now + config.SUPPRESS_S
        self.kick(note_id, "dismissed")

    def mark_viewed(self, note_ids: list[str]) -> None:
        for note_id in note_ids:
            item = self._items.get(note_id)
            if item is None:
                continue
            item.views += 1

        decayed = [
            note_id
            for note_id, item in self._items.items()
            if item.views >= item.decay
        ]
        for note_id in decayed:
            self.kick(note_id, "decayed")

    def items(self) -> list[SlateItem]:
        return sorted(
            self._items.values(),
            key=lambda item: (
                self._provenance_strength(item.source),
                item.score,
                item.added_rev,
            ),
            reverse=True,
        )

    def _evict_over_quota(self, source: str) -> None:
        quota = config.QUOTA_EXACT if source == "exact" else config.QUOTA_SEARCH
        lane_items = [item for item in self._items.values() if item.source == source]
        if len(lane_items) <= quota:
            return

        evicted = max(lane_items, key=lambda item: (item.views, -item.added_rev))
        del self._items[evicted.note_id]
        self.revision += 1

    @staticmethod
    def _stronger_source(current: str, incoming: str) -> str:
        if current == "exact" or incoming == "exact":
            return "exact"
        return "search"

    @staticmethod
    def _provenance_strength(source: str) -> int:
        return 1 if source == "exact" else 0
