"""Rolling meeting summarizer — ONE deepseek-v4-flash pass per trigger (spec §5).

Input (ReSum-style bounded state, never the whole transcript):
previous MeetingSummary JSON (or "none") + transcript delta since the last
*accepted* pass (≤ SUMMARY_DELTA_MAX_CHARS) + grounding block
(``[N3] "title" (source, age) — gist`` / ``[F7] (N3↔N9) "fact"``) + open
actionables with status.

Output: MeetingSummary delta or no_update (a first-class, correct answer).

Validation (§5.4, constrained citation — engineering guarantees):
- unknown [N#]/[F#] citations stripped; > SUMMARY_INVALID_CITATION_MAX of
  citations invalid → whole pass rejected (previous summary kept, logged);
- actionables: quote must fuzzy-match the transcript delta
  (rapidfuzz ≥ ACTIONABLE_QUOTE_MIN_RATIO), confidence ≥ ACTIONABLE_MIN_CONF,
  ≤ ACTIONABLE_MAX_PER_PASS accepted, title-embedding cosine ≥
  ACTIONABLE_DEDUPE_SIM vs an open actionable → update not new;
- single fallback-model retry (SUMMARY_FALLBACK_MODEL); on double failure the
  pass is skipped silently — no summary change is acceptable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..indexer.store import Index
from ..llm import OpenRouterClient
from ..schemas import Actionable, MeetingSummary
from .grounding import GroundingUpdate


@dataclass
class SummaryResult:
    summary: MeetingSummary | None       # None ⇔ no_update
    no_update: bool
    new_actionables: list[Actionable] = field(default_factory=list)
    updated_actionables: list[Actionable] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""


class Summarizer:
    def __init__(self, llm: OpenRouterClient, index: Index) -> None:
        self.llm = llm
        self.index = index
        raise NotImplementedError  # Unit I

    async def summarize(
        self,
        state: MeetingSummary | None,
        transcript_delta: str,
        update: GroundingUpdate,
        open_actionables: list[Actionable],
    ) -> SummaryResult | None:
        """One LLM pass; returns None only when both models fail."""
        raise NotImplementedError  # Unit I
