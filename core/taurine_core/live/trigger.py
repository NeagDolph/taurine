"""Summary trigger policy (spec §4.6) — replaces the presenter cadence.

Fires the summarizer when the *evidence* changes, never on keywords:

    fire iff (jaccard(top-8 vs last summarized) < SUMMARY_CHANGE_JACCARD
              OR drift: cos(window_emb, topic_centroid) < SUMMARY_DRIFT_COSINE
              OR forced: staleness > SUMMARY_MAX_STALENESS_S with new speech)
         AND now - last_pass > SUMMARY_MIN_INTERVAL_S
         AND new final chars since last pass ≥ SUMMARY_MIN_NEW_CHARS

Topic centroid = EMA (TOPIC_CENTROID_DECAY) of window embeddings; a drift
firing resets it. A no_update reply applies SUMMARY_EMPTY_BACKOFF_S before
the same grounding revision may re-fire. Every decision (+ first blocking
gate) is session-logged, mirroring Cadence.explain.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .. import config
from .grounding import GroundingUpdate


class SummaryTrigger:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        raise NotImplementedError  # Unit I

    def offer(
        self,
        update: GroundingUpdate,
        window_emb: np.ndarray | None,
        new_chars: int,
    ) -> tuple[bool, str]:
        """(fire?, reason) — reason is the trigger cause or first blocking gate."""
        raise NotImplementedError  # Unit I

    def note_summary(self, update: GroundingUpdate, *, no_update: bool) -> None:
        """Record an accepted summarizer pass (or a no_update reply → backoff)."""
        raise NotImplementedError  # Unit I
