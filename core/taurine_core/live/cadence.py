from __future__ import annotations

from typing import Protocol

from taurine_core import config


class Clock(Protocol):
    def now(self) -> float:
        ...


class Cadence:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._last_attempt_at: float | None = None
        self._last_promotion_at: float | None = None
        self._last_attempted_revision: int | None = None
        self._last_allowed_revision: int | None = None
        self._promotion_times: list[float] = []
        self._empty_until_by_revision: dict[int, float] = {}
        self._empty_retry_used: set[int] = set()
        self._reoffered_revisions: set[int] = set()

    def may_present(
        self,
        revision: int | None = None,
        now: float | None = None,
        *,
        rev: int | None = None,
    ) -> bool:
        current_revision = self._resolve_revision(revision, rev)
        current_now = self._resolve_now(now)
        self._last_allowed_revision = None

        if not self._passes_timing_gates(current_now):
            return False

        empty_until = self._empty_until_by_revision.get(current_revision)
        if empty_until is not None and current_now < empty_until:
            return False

        if self._last_attempted_revision != current_revision:
            self._last_allowed_revision = current_revision
            return True

        if (
            empty_until is not None
            and current_revision not in self._empty_retry_used
        ):
            self._last_allowed_revision = current_revision
            return True

        if (
            current_revision not in self._reoffered_revisions
            and self._last_attempt_at is not None
            and current_now - self._last_attempt_at >= config.REOFFER_S
        ):
            self._last_allowed_revision = current_revision
            return True

        return False

    def explain(
        self,
        revision: int | None = None,
        now: float | None = None,
        *,
        rev: int | None = None,
    ) -> str:
        current_revision = self._resolve_revision(revision, rev)
        current_now = self._resolve_now(now)

        if (
            self._last_attempt_at is not None
            and current_now - self._last_attempt_at < config.DEBOUNCE_S
        ):
            return "debounce"
        if (
            self._last_promotion_at is not None
            and current_now - self._last_promotion_at < config.COOLDOWN_S
        ):
            return "cooldown"
        if not self._under_rate_cap_without_mutation(current_now):
            return "rate_cap"

        empty_until = self._empty_until_by_revision.get(current_revision)
        if empty_until is not None and current_now < empty_until:
            return "empty_backoff"

        if self._last_attempted_revision != current_revision:
            return "allowed"

        if (
            empty_until is not None
            and current_revision not in self._empty_retry_used
        ):
            return "allowed"

        if (
            current_revision not in self._reoffered_revisions
            and self._last_attempt_at is not None
            and current_now - self._last_attempt_at >= config.REOFFER_S
        ):
            return "reoffer"

        return "same_revision"

    def note_attempt(self, now: float | None = None) -> None:
        current_now = self._resolve_now(now)
        revision = self._last_allowed_revision

        if revision is not None:
            if self._last_attempted_revision == revision:
                empty_until = self._empty_until_by_revision.get(revision)
                if empty_until is not None and current_now >= empty_until:
                    self._empty_retry_used.add(revision)
                else:
                    self._reoffered_revisions.add(revision)
            self._last_attempted_revision = revision

        self._last_attempt_at = current_now

    def note_promotion(self, now: float | None = None) -> None:
        current_now = self._resolve_now(now)
        self._last_promotion_at = current_now
        self._record_promotion(current_now)

    def note_empty(
        self,
        revision: int | None = None,
        now: float | None = None,
        *,
        rev: int | None = None,
    ) -> None:
        current_revision = self._resolve_revision(revision, rev)
        current_now = self._resolve_now(now)
        self._empty_until_by_revision[current_revision] = (
            current_now + config.EMPTY_BACKOFF_S
        )
        self._empty_retry_used.discard(current_revision)
        self._last_attempted_revision = current_revision
        self._last_attempt_at = current_now

    def instant_pin_allowed(self, now: float | None = None) -> bool:
        current_now = self._resolve_now(now)
        if not self._under_rate_cap(current_now):
            return False
        self._record_promotion(current_now)
        return True

    def _passes_timing_gates(self, now: float) -> bool:
        if (
            self._last_attempt_at is not None
            and now - self._last_attempt_at < config.DEBOUNCE_S
        ):
            return False
        if (
            self._last_promotion_at is not None
            and now - self._last_promotion_at < config.COOLDOWN_S
        ):
            return False
        return self._under_rate_cap(now)

    def _under_rate_cap(self, now: float) -> bool:
        self._promotion_times = [
            ts for ts in self._promotion_times if now - ts < 60.0
        ]
        return len(self._promotion_times) < config.MAX_PROMOS_PER_MIN

    def _under_rate_cap_without_mutation(self, now: float) -> bool:
        recent = [ts for ts in self._promotion_times if now - ts < 60.0]
        return len(recent) < config.MAX_PROMOS_PER_MIN

    def _record_promotion(self, now: float) -> None:
        self._promotion_times = [
            ts for ts in self._promotion_times if now - ts < 60.0
        ]
        self._promotion_times.append(now)

    def _resolve_now(self, now: float | None) -> float:
        return self._clock.now() if now is None else now

    @staticmethod
    def _resolve_revision(revision: int | None, rev: int | None) -> int:
        resolved = revision if revision is not None else rev
        if resolved is None:
            raise TypeError("revision is required")
        return resolved
