"""Deterministic build-time content-quality and note-salience scoring (§0.2).

No LLM, no embeddings, no wall clock: a note's ``content_quality`` is a pure
function of its text, and ``note_salience = ingest_salience × content_quality``
gates extraction (never searchability). Length is only ONE input — a short
dense note stays extractable while a long boilerplate note does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# tokens at which the length ramp saturates: a note this long earns full
# length credit; shorter notes keep a floor so short-but-dense text survives
TOKEN_SATURATION = 40
# the length component contributes at most this share of the final score, so
# raw length alone can never gate a note (§0.2 "never gate on len(text)")
_TOKEN_RAMP_FLOOR = 0.4
# alpha-character ratio at (or above) which text counts as fully prose-like
_ALPHA_HEALTHY = 0.65
# unique-token ratio at (or above) which text counts as non-repetitive
_UNIQUE_HEALTHY = 0.45


@dataclass(frozen=True)
class ContentQuality:
    token_ramp: float
    alpha_ratio: float
    unique_ratio: float
    line_penalty: float
    quality: float


def content_quality(text: str) -> float:
    """Content quality in [0, 1]; deterministic and cheap."""

    return content_quality_components(text).quality


def content_quality_components(text: str) -> ContentQuality:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return ContentQuality(0.0, 0.0, 0.0, 1.0, 0.0)
    token_ramp = min(1.0, len(tokens) / TOKEN_SATURATION)
    compact = "".join(text.split())
    alpha_ratio = sum(char.isalpha() for char in compact) / len(compact)
    unique_ratio = len({token.casefold() for token in tokens}) / len(tokens)
    lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        duplicate_fraction = (len(lines) - len(set(lines))) / len(lines)
        line_penalty = 1.0 - duplicate_fraction
    else:
        line_penalty = 1.0
    quality = (
        (_TOKEN_RAMP_FLOOR + (1.0 - _TOKEN_RAMP_FLOOR) * token_ramp)
        * _clip01(alpha_ratio / _ALPHA_HEALTHY)
        * _clip01(unique_ratio / _UNIQUE_HEALTHY)
        * line_penalty
    )
    return ContentQuality(
        token_ramp=token_ramp,
        alpha_ratio=alpha_ratio,
        unique_ratio=unique_ratio,
        line_penalty=line_penalty,
        quality=_clip01(quality),
    )


def note_salience(ingest_salience: float, text: str) -> float:
    """Combined salience stored on the note row: connector × content quality."""

    return max(0.0, float(ingest_salience)) * content_quality(text)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
