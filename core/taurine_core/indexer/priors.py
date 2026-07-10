"""Static per-source priors: trust rank and recency decay (spec §4.4).

Consumed by the knowledge build's quality/value rank computation
(`build._compute_note_ranks`). Pure functions over config tables.
"""
from __future__ import annotations

from .. import config


def source_rank(source: str) -> float:
    return config.SOURCE_RANK.get(source, config.SOURCE_RANK_DEFAULT)


def recency_multiplier(source: str, age_days: float) -> float:
    """Exponential age decay with grace offset and floor (§4.4).

    ``RECENCY_FLOOR + (1−RECENCY_FLOOR) · 2^(−max(0, age−OFFSET)/half_life)``.
    """
    half_life = config.RECENCY_HALF_LIFE_DAYS.get(
        source, config.RECENCY_HALF_LIFE_DEFAULT_DAYS
    )
    effective = max(0.0, float(age_days) - config.RECENCY_OFFSET_DAYS)
    decay = float(2.0 ** (-effective / half_life))
    return config.RECENCY_FLOOR + (1.0 - config.RECENCY_FLOOR) * decay
