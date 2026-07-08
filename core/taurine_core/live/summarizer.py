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
import json
import re
import time
from typing import Any
from uuid import uuid4

import numpy as np
from rapidfuzz import fuzz

from .. import config
from ..indexer import store
from ..indexer.store import Index
from ..llm import OpenRouterClient
from ..ports import LLMError
from ..schemas import Actionable, MeetingSummary, SummaryHighlight
from .grounding import GroundingUpdate


CITATION_RE = re.compile(r"\[(?:N|F)\d+\]")

SYSTEM_PROMPT = """You are the rolling-meeting-summarizer for a live meeting.

Update the markdown summary of the meeting so far using only the provided
previous summary, transcript delta, grounding notes, grounded facts, and open
actionables. Cite every factual claim with [N#] or [F#] ids from the grounding
block ONLY. Be conservative: create an actionable only when the transcript
contains an explicit commitment with an assignee, a verb, and a deliverable,
and include the verbatim quote. Prefer {"no_update": true} when nothing
material changed. Note ages are supplied; discount stale evidence.

Return only a JSON object in one of these shapes:
{"no_update": true}
or
{"no_update": false,
 "summary_md": "markdown with [N#]/[F#] citations",
 "highlights": [{"fact_id": "F1", "why": "why this fact matters now"}],
 "actionables": [{"title": "deliverable", "quote": "verbatim transcript quote",
                  "assignee": "owner", "due": "date or empty",
                  "grounding": ["N1", "F1"], "why_new": "why this is new",
                  "confidence": 0.9}]}
"""


@dataclass
class SummaryResult:
    summary: MeetingSummary | None       # None when no_update or rejected
    no_update: bool
    new_actionables: list[Actionable] = field(default_factory=list)
    updated_actionables: list[Actionable] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    rejected: bool = False
    reject_reason: str = ""
    citation_stats: dict[str, float] = field(default_factory=dict)


class Summarizer:
    def __init__(self, llm: OpenRouterClient, index: Index) -> None:
        self.llm = llm
        self.index = index

    async def summarize(
        self,
        state: MeetingSummary | None,
        transcript_delta: str,
        update: GroundingUpdate,
        open_actionables: list[Actionable],
    ) -> SummaryResult | None:
        """One LLM pass; returns None only when both models fail."""
        started = time.monotonic()
        prompt_delta = _truncate_newest(transcript_delta, config.SUMMARY_DELTA_MAX_CHARS)
        user_prompt = _build_user_prompt(state, prompt_delta, update, open_actionables)

        try:
            raw = await self.llm.complete_json(
                config.SUMMARY_MODEL,
                SYSTEM_PROMPT,
                user_prompt,
                timeout_s=config.SUMMARY_TIMEOUT_S,
            )
            model = config.SUMMARY_MODEL
        except LLMError:
            try:
                raw = await self.llm.complete_json(
                    config.SUMMARY_FALLBACK_MODEL,
                    SYSTEM_PROMPT,
                    user_prompt,
                    timeout_s=config.SUMMARY_TIMEOUT_S,
                )
                model = config.SUMMARY_FALLBACK_MODEL
            except LLMError:
                return None

        latency_ms = (time.monotonic() - started) * 1000.0
        if raw.get("no_update") is True:
            return SummaryResult(
                summary=None,
                no_update=True,
                latency_ms=latency_ms,
                model=model,
            )

        valid_ids = _valid_grounding_ids(update)
        valid_fact_ids = {fact.fact_id for fact in update.facts}
        summary_md, citation_stats = _validated_summary_md(
            str(raw.get("summary_md") or ""),
            valid_ids,
        )
        if _citation_rejects(citation_stats):
            return SummaryResult(
                summary=None,
                no_update=False,
                latency_ms=latency_ms,
                model=model,
                rejected=True,
                reject_reason="invalid_citations",
                citation_stats=citation_stats,
            )

        candidates = _validated_actionables(raw.get("actionables"), prompt_delta, valid_ids)
        new_actionables, updated_actionables, merged_actionables = _merge_actionables(
            candidates,
            open_actionables,
        )
        next_rev = (state.rev if state is not None else 0) + 1
        summary = MeetingSummary(
            rev=next_rev,
            summary_md=summary_md,
            highlights=_validated_highlights(raw.get("highlights"), valid_fact_ids),
            actionables=merged_actionables,
            grounding=update.items,
            facts=update.facts,
            updated_at=time.time(),
        )
        return SummaryResult(
            summary=summary,
            no_update=False,
            new_actionables=new_actionables,
            updated_actionables=updated_actionables,
            latency_ms=latency_ms,
            model=model,
            citation_stats=citation_stats,
        )


def _build_user_prompt(
    state: MeetingSummary | None,
    transcript_delta: str,
    update: GroundingUpdate,
    open_actionables: list[Actionable],
) -> str:
    previous = "none" if state is None else json.dumps(state.model_dump(mode="json"))
    grounding = _grounding_block(update)
    open_rows = _open_actionables_block(open_actionables)
    return (
        "<previous_summary>\n"
        f"{previous}\n"
        "</previous_summary>\n\n"
        "<transcript_delta>\n"
        f"{transcript_delta}\n"
        "</transcript_delta>\n\n"
        "<grounding>\n"
        f"{grounding}\n"
        "</grounding>\n\n"
        "<open_actionables>\n"
        f"{open_rows}\n"
        "</open_actionables>"
    )


def _grounding_block(update: GroundingUpdate) -> str:
    note_ids = {item.note_id: item.short_id for item in update.items}
    lines: list[str] = []
    for item in update.items:
        lines.append(
            f'[{item.short_id}] "{item.title}" '
            f"({item.source}, age {_age_label(item.age_days)}, score {item.score:.2f}) "
            f"— {item.gist}"
        )
    for fact in update.facts:
        src = note_ids.get(fact.src, fact.src)
        dst = note_ids.get(fact.dst, fact.dst)
        lines.append(f'[{fact.fact_id}] ({src}<->{dst}) "{fact.text}"')
    return "\n".join(lines) if lines else "none"


def _open_actionables_block(open_actionables: list[Actionable]) -> str:
    if not open_actionables:
        return "none"
    rows = [
        {
            "id": item.id,
            "title": item.title,
            "status": item.status,
            "assignee": item.assignee,
            "due": item.due,
        }
        for item in open_actionables
    ]
    return json.dumps(rows, ensure_ascii=False)


def _age_label(age_days: float) -> str:
    if age_days < 1.0:
        return "today"
    if age_days < 60.0:
        return f"{int(round(age_days))}d"
    return f"{int(round(age_days / 30.0))}mo"


def _truncate_newest(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _valid_grounding_ids(update: GroundingUpdate) -> set[str]:
    return {item.short_id for item in update.items} | {fact.fact_id for fact in update.facts}


def _validated_summary_md(
    summary_md: str,
    valid_ids: set[str],
) -> tuple[str, dict[str, float]]:
    total = 0
    valid = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal total, valid
        total += 1
        citation_id = match.group(0)[1:-1]
        if citation_id in valid_ids:
            valid += 1
            return match.group(0)
        return ""

    stripped = CITATION_RE.sub(replace, summary_md)
    invalid = total - valid
    ratio = 1.0 if total == 0 else valid / total
    return stripped, {
        "total": float(total),
        "valid": float(valid),
        "invalid": float(invalid),
        "valid_ratio": ratio,
    }


def _citation_rejects(stats: dict[str, float]) -> bool:
    total = stats.get("total", 0.0)
    if total <= 0.0:
        return False
    return stats.get("valid_ratio", 1.0) < (1.0 - config.SUMMARY_INVALID_CITATION_MAX)


def _validated_highlights(rows: Any, valid_fact_ids: set[str]) -> list[SummaryHighlight]:
    if not isinstance(rows, list):
        return []
    highlights: list[SummaryHighlight] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fact_id = str(row.get("fact_id") or "")
        if fact_id not in valid_fact_ids:
            continue
        highlights.append(SummaryHighlight(fact_id=fact_id, why=str(row.get("why") or "")))
    return highlights


def _validated_actionables(
    rows: Any,
    transcript_delta: str,
    valid_ids: set[str],
) -> list[Actionable]:
    if not isinstance(rows, list):
        return []
    transcript_norm = _normalize_text(transcript_delta)
    accepted: list[Actionable] = []
    for row in rows:
        actionable = _validated_actionable(row, transcript_norm, valid_ids)
        if actionable is not None:
            accepted.append(actionable)
    accepted.sort(key=lambda item: item.confidence, reverse=True)
    return accepted[: config.ACTIONABLE_MAX_PER_PASS]


def _validated_actionable(
    row: Any,
    transcript_norm: str,
    valid_ids: set[str],
) -> Actionable | None:
    if not isinstance(row, dict):
        return None

    title = str(row.get("title") or "").strip()
    quote = str(row.get("quote") or "").strip()
    confidence = _float(row.get("confidence"))
    if not title or not quote:
        return None
    if confidence < config.ACTIONABLE_MIN_CONF:
        return None
    if _quote_ratio(quote, transcript_norm) < config.ACTIONABLE_QUOTE_MIN_RATIO:
        return None

    grounding = [
        str(item)
        for item in row.get("grounding") or []
        if isinstance(item, str) and item in valid_ids
    ]
    if not grounding:
        return None

    return Actionable(
        id="",
        title=title,
        quote=quote,
        assignee=str(row.get("assignee") or ""),
        due=str(row.get("due") or ""),
        grounding=grounding,
        why_new=str(row.get("why_new") or ""),
        confidence=confidence,
    )


def _quote_ratio(quote: str, transcript_norm: str) -> float:
    quote_norm = _normalize_text(quote)
    if not quote_norm or not transcript_norm:
        return 0.0
    return fuzz.partial_ratio(quote_norm, transcript_norm) / 100.0


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _merge_actionables(
    candidates: list[Actionable],
    open_actionables: list[Actionable],
) -> tuple[list[Actionable], list[Actionable], list[Actionable]]:
    new_actionables: list[Actionable] = []
    updated_by_id: dict[str, Actionable] = {}

    for candidate in candidates:
        existing = _matching_open_actionable(candidate.title, open_actionables)
        if existing is not None:
            if existing.id in updated_by_id:
                continue
            updated_by_id[existing.id] = candidate.model_copy(
                update={"id": existing.id, "status": existing.status}
            )
            continue

        new_actionables.append(candidate.model_copy(update={"id": _new_actionable_id()}))

    merged: list[Actionable] = []
    for item in open_actionables:
        merged.append(updated_by_id.get(item.id, item))
    merged.extend(new_actionables)
    return new_actionables, list(updated_by_id.values()), merged


def _matching_open_actionable(
    title: str,
    open_actionables: list[Actionable],
) -> Actionable | None:
    if not open_actionables:
        return None

    embeddings = store.embed_texts([title] + [item.title for item in open_actionables])
    if len(embeddings) != len(open_actionables) + 1:
        return None
    query = embeddings[0]
    best: tuple[float, Actionable] | None = None
    for vector, item in zip(embeddings[1:], open_actionables, strict=True):
        cosine = _cosine(query, vector)
        if best is None or cosine > best[0]:
            best = (cosine, item)

    if best is None or best[0] < config.ACTIONABLE_DEDUPE_SIM:
        return None
    return best[1]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _new_actionable_id() -> str:
    return f"act-{uuid4().hex[:8]}"
