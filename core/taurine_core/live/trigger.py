"""Summary trigger policy (spec §4.6).

Fires the summarizer when the *evidence* changes, never on keywords:

    fire iff (jaccard(top-8 vs last summarized) < SUMMARY_CHANGE_JACCARD
              OR drift: cos(window_emb, topic_centroid) < SUMMARY_DRIFT_COSINE
              OR forced: staleness > SUMMARY_MAX_STALENESS_S with new speech)
         AND now - last_pass > SUMMARY_MIN_INTERVAL_S
         AND new final chars since last pass ≥ SUMMARY_MIN_NEW_CHARS

Topic centroid = EMA (TOPIC_CENTROID_DECAY) of window embeddings; a drift
firing resets it. A no_update reply applies SUMMARY_EMPTY_BACKOFF_S before
the same grounding revision may re-fire. Every decision (+ first blocking
gate) is session-logged for debugging.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from .. import config
from .grounding import GroundingUpdate


class SummaryTrigger:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._last_pass_at: float | None = None
        self._last_summarized_ids: set[str] | None = None
        self._empty_revision: int | None = None
        self._empty_backoff_until: float = -math.inf
        self._topic_centroid: np.ndarray | None = None
        self._last_decision = "not_offered"

    def offer(
        self,
        update: GroundingUpdate,
        window_emb: np.ndarray | None,
        new_chars: int,
    ) -> tuple[bool, str]:
        """(fire?, reason) — reason is the trigger cause or first blocking gate."""
        now = self.clock()
        drift_reason = self._update_centroid(window_emb)

        if self._last_pass_at is not None:
            elapsed = now - self._last_pass_at
            if elapsed < config.SUMMARY_MIN_INTERVAL_S:
                return self._decision(False, "min_interval")

        if new_chars < config.SUMMARY_MIN_NEW_CHARS:
            return self._decision(False, "min_new_chars")

        top_ids = _top_ids(update)
        reason = self._change_reason(top_ids, drift_reason, now)
        if reason is None:
            return self._decision(False, "no_change")

        if (
            self._empty_revision == update.revision
            and now < self._empty_backoff_until
        ):
            return self._decision(False, "empty_backoff")

        return self._decision(True, reason)

    def note_summary(self, update: GroundingUpdate, *, no_update: bool) -> None:
        """Record an accepted summarizer pass (or a no_update reply → backoff)."""
        now = self.clock()
        self._last_pass_at = now
        if no_update:
            self._empty_revision = update.revision
            self._empty_backoff_until = now + config.SUMMARY_EMPTY_BACKOFF_S
            return

        self._last_summarized_ids = _top_ids(update)
        self._empty_revision = None
        self._empty_backoff_until = -math.inf

    def note_rejected(self) -> None:
        """Record a rejected summarizer pass without accepting or backing off."""
        self._last_pass_at = self.clock()

    def explain(self) -> str:
        return self._last_decision

    def _change_reason(
        self,
        top_ids: set[str],
        drift_reason: str | None,
        now: float,
    ) -> str | None:
        if top_ids and self._jaccard(top_ids) < config.SUMMARY_CHANGE_JACCARD:
            return "jaccard"
        if drift_reason is not None:
            return drift_reason
        if (
            self._last_pass_at is not None
            and now - self._last_pass_at > config.SUMMARY_MAX_STALENESS_S
        ):
            return "staleness"
        return None

    def _jaccard(self, top_ids: set[str]) -> float:
        if self._last_summarized_ids is None:
            return 0.0
        union = top_ids | self._last_summarized_ids
        if not union:
            return 1.0
        return len(top_ids & self._last_summarized_ids) / len(union)

    def _update_centroid(self, window_emb: np.ndarray | None) -> str | None:
        if window_emb is None:
            return None

        vector = _normalize(window_emb)
        if vector is None:
            return None

        if self._topic_centroid is None:
            self._topic_centroid = vector
            return None

        cosine = float(np.dot(vector, self._topic_centroid))
        if cosine < config.SUMMARY_DRIFT_COSINE:
            self._topic_centroid = vector
            return "drift"

        updated = (
            config.TOPIC_CENTROID_DECAY * self._topic_centroid
            + (1.0 - config.TOPIC_CENTROID_DECAY) * vector
        )
        self._topic_centroid = _normalize(updated)
        return None

    def _decision(self, fire: bool, reason: str) -> tuple[bool, str]:
        self._last_decision = reason
        return fire, reason


def _top_ids(update: GroundingUpdate) -> set[str]:
    support = [
        *(row.evidence_id for row in update.passages),
        *(row.fact_ref for row in update.facts),
    ]
    return set(support[: config.SUMMARY_CHANGE_TOP_K])


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    return arr / norm
