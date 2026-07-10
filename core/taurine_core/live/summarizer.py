"""Evidence-first rolling live summarization with fail-closed validation."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
from rapidfuzz import fuzz

from taurine_core import config
from taurine_core.indexer import store
from taurine_core.indexer.store import Index
from taurine_core.llm import OpenRouterClient
from taurine_core.ports import LLMError
from taurine_core.profile_context import user_profile_block
from taurine_core.retrieval.validate import EvidenceValidator, ValidationResult
from taurine_core.schemas import (
    Actionable,
    EvidenceFactPayload,
    EvidenceNotePayload,
    EvidencePackPayload,
    EvidencePassagePayload,
    EvidencePathPayload,
    MeetingSummary,
    RetrievalDecisionPayload,
    SummaryHighlight,
)

from .grounding import GroundingUpdate


SYSTEM_PROMPT = """You are Taurine's rolling live-meeting summarizer.

Update the prior summary using only the supplied transcript and knowledge-base
evidence. Every factual sentence must cite its support. Use [T#] for transcript
claims and [E#] or [F#] for knowledge-base claims. [N#] is navigation metadata
and [P#] explains a path; neither supports a factual sentence by itself.

The optional <user_profile> is uncitable background for resolving names only.
It is not evidence. Never cite it. Preserve temporal qualifiers and conflicts. If there
is no material supported update, return {"no_update": true}.

Return JSON only:
{"no_update": true}
or
{"no_update": false,
 "summary_md": "markdown with sentence-level citations",
 "highlights": [{"fact_id": "F1", "why": "why it matters"}],
 "actionables": [{"title": "deliverable", "quote": "verbatim transcript",
   "assignee": "owner", "due": "date or empty",
   "grounding": ["T1", "E1", "F1"], "why_new": "reason",
   "confidence": 0.9}]}
"""


@dataclass
class SummaryResult:
    summary: MeetingSummary | None
    no_update: bool
    new_actionables: list[Actionable] = field(default_factory=list)
    updated_actionables: list[Actionable] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    rejected: bool = False
    reject_reason: str = ""
    citation_stats: dict[str, float] = field(default_factory=dict)
    validation: ValidationResult | None = None


class Summarizer:
    def __init__(
        self,
        llm: OpenRouterClient,
        index: Index,
        validator: EvidenceValidator,
    ) -> None:
        self.llm = llm
        self.index = index
        self.validator = validator
        # Accepted-evidence ledger (C3a): refs cited by the last ACCEPTED
        # summary stay citeable on later passes (they were validated at
        # acceptance and their payload text lets NLI re-check them). Pruned
        # wholesale to the current accepted summary's citations on each
        # acceptance; never contains unvalidated refs.
        self._accepted_citation_refs: set[str] = set()

    async def summarize(
        self,
        state: MeetingSummary | None,
        transcript_evidence: str,
        update: GroundingUpdate,
        open_actionables: list[Actionable],
    ) -> SummaryResult | None:
        """Generate once (plus model fallback) and accept only validated output."""

        started = time.monotonic()
        transcript = transcript_evidence
        prompt = _build_user_prompt(
            state,
            transcript,
            update,
            open_actionables,
            profile_block=user_profile_block(self.index),
        )
        try:
            raw = await self.llm.complete_json(
                config.SUMMARY_MODEL,
                SYSTEM_PROMPT,
                prompt,
                timeout_s=config.SUMMARY_TIMEOUT_S,
            )
            model = config.SUMMARY_MODEL
        except LLMError:
            try:
                raw = await self.llm.complete_json(
                    config.SUMMARY_FALLBACK_MODEL,
                    SYSTEM_PROMPT,
                    prompt,
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

        valid_ids = _valid_evidence_ids(transcript, update)
        # C3a: refs cited by the previous ACCEPTED summary stay available so
        # the rolling update may retain its sentences. supplied_ids remains
        # the current pass only — nothing unvalidated becomes citeable.
        available_ids = valid_ids | self._resolvable_accepted_refs()
        actionables = _validated_actionables(
            raw.get("actionables"), transcript, available_ids
        )
        summary_md = str(raw.get("summary_md") or "").strip()
        validation_text = _validation_text(summary_md, actionables)
        validation = self.validator.validate(
            validation_text,
            mode="live",
            supplied_ids=valid_ids,
            available_ids=available_ids,
        )
        citation_stats = {
            "total": float(len(validation.cited_ids)),
            "valid": float(len(validation.cited_ids) - len(validation.invalid_citations)),
            "invalid": float(len(validation.invalid_citations)),
        }
        if not validation.accepted:
            return SummaryResult(
                summary=None,
                no_update=False,
                latency_ms=latency_ms,
                model=model,
                rejected=True,
                reject_reason=validation.degraded_reason or "unsupported_claim",
                citation_stats=citation_stats,
                validation=validation,
            )

        new_actionables, updated_actionables, merged_actionables = _merge_actionables(
            actionables, open_actionables
        )
        highlights = self._supported_highlights(
            raw.get("highlights"),
            supplied_ids=valid_ids,
            available_ids=available_ids,
        )
        self._note_accepted(
            validation.cited_ids, (row.fact_id for row in highlights)
        )
        next_rev = (state.rev if state is not None else 0) + 1
        evidence = _evidence_payload(update)
        summary = MeetingSummary(
            rev=next_rev,
            summary_md=summary_md,
            highlights=highlights,
            actionables=merged_actionables,
            evidence=evidence,
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
            validation=validation,
        )

    def _resolvable_accepted_refs(self) -> set[str]:
        """Previously-accepted citation refs that still resolve with payload
        text, so NLI can re-check any sentence that keeps citing them."""

        registry = self.validator.registry
        kept: set[str] = set()
        for ref in self._accepted_citation_refs:
            record = registry.resolve(ref)
            if record is None:
                continue
            if record.kind in {"passage", "transcript"} and not _record_text(record):
                continue
            kept.add(ref)
        return kept

    def _note_accepted(
        self,
        cited_ids: Iterable[str],
        highlight_fact_ids: Iterable[str] = (),
    ) -> None:
        """Prune the accepted-evidence ledger to the current accepted summary.

        Fact sources ride along so grandfathered fact citations keep passing
        provenance checks (they were validated as available at acceptance).
        """

        registry = self.validator.registry
        refs = set(cited_ids) | set(highlight_fact_ids)
        for ref in tuple(refs):
            record = registry.resolve(ref)
            if record is None:
                refs.discard(ref)
                continue
            if record.kind == "fact":
                sources = record.payload.get("source_evidence_ids")
                if isinstance(sources, (list, tuple)):
                    refs.update(str(source) for source in sources)
        self._accepted_citation_refs = refs

    def _supported_highlights(
        self,
        rows: Any,
        *,
        supplied_ids: set[str],
        available_ids: set[str],
    ) -> list[SummaryHighlight]:
        """NLI-gate each highlight's ``why`` text individually (L17): a
        highlight failing validation drops the highlight, not the summary."""

        kept: list[SummaryHighlight] = []
        for row in _validated_highlights(rows, available_ids):
            why = row.why.strip()
            if not why:
                kept.append(row)
                continue
            result = self.validator.validate(
                f"{why.rstrip('.')}. [{row.fact_id}]",
                mode="live",
                supplied_ids=supplied_ids,
                available_ids=available_ids,
            )
            if result.accepted:
                kept.append(row)
        return kept


def _record_text(record: Any) -> str:
    for key in ("text", "body", "transcript", "content", "quote"):
        value = record.payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_user_prompt(
    state: MeetingSummary | None,
    transcript: str,
    update: GroundingUpdate,
    open_actionables: list[Actionable],
    *,
    profile_block: str = "",
) -> str:
    previous = "none" if state is None else json.dumps(state.model_dump(mode="json"))
    sections = (
        f"<previous_summary>\n{previous}\n</previous_summary>",
        f"<transcript_evidence>\n{transcript or 'none'}\n</transcript_evidence>",
        f"<knowledge_evidence>\n{_grounding_block(update)}\n</knowledge_evidence>",
        "<open_actionables>\n"
        f"{_open_actionables_block(open_actionables)}\n"
        "</open_actionables>",
    )
    body = "\n\n".join(sections)
    return f"{profile_block}\n\n{body}" if profile_block else body


def _grounding_block(update: GroundingUpdate) -> str:
    lines: list[str] = []
    for row in update.support_passages:
        lines.append(
            f'[{row.evidence_id}] [{_note_ref(row.note_id, update)}] '
            f'"{row.title}" ({row.source}) — {row.text}'
        )
    for row in update.support_facts:
        validity = f"valid {row.valid_from or 'unknown'}..{row.valid_to or 'open'}"
        lines.append(
            f"[{row.fact_ref}] {row.subject} --{row.predicate}--> {row.object} "
            f"({row.status}; {validity}; sources {','.join(row.source_evidence_ids)})"
        )
    for row in update.support_paths:
        lines.append(f"[{row.path_id}] {row.rendered}")
    for row in update.support_notes:
        lines.append(f'[{row.note_ref}] note "{row.title}" ({row.source})')
    return "\n".join(lines) if lines else "none"


def _note_ref(note_id: str, update: GroundingUpdate) -> str:
    return next(
        (row.note_ref for row in update.support_notes if row.note_id == note_id),
        "N?",
    )


def _open_actionables_block(rows: list[Actionable]) -> str:
    if not rows:
        return "none"
    return json.dumps(
        [
            {
                "id": row.id,
                "title": row.title,
                "status": row.status,
                "assignee": row.assignee,
                "due": row.due,
                "grounding": list(row.grounding),
            }
            for row in rows
        ],
        ensure_ascii=False,
    )


def _valid_evidence_ids(transcript: str, update: GroundingUpdate) -> set[str]:
    refs = {
        *(row.evidence_id for row in update.support_passages),
        *(row.fact_ref for row in update.support_facts),
        *(row.path_id for row in update.support_paths),
        *(row.note_ref for row in update.support_notes),
    }
    refs.update(
        token[1:-1]
        for token in transcript.split()
        if len(token) > 3 and token.startswith("[T") and token.endswith("]")
    )
    return refs


def _validation_text(summary_md: str, actionables: list[Actionable]) -> str:
    rows = [summary_md]
    rows.extend(
        f"{row.title.rstrip('.')} . " + "".join(f"[{ref}]" for ref in row.grounding)
        for row in actionables
    )
    return "\n".join(row for row in rows if row.strip())


def _evidence_payload(update: GroundingUpdate) -> EvidencePackPayload:
    event_decision = RetrievalDecisionPayload(
        accepted=update.decision.accepted,
        confidence=update.decision.confidence,
        reason=update.decision.reason,
        accepted_heads=list(update.decision.accepted_heads),
        rejected_heads=list(update.decision.rejected_heads),
    )
    return EvidencePackPayload(
        decision=event_decision,
        passages=[
            EvidencePassagePayload(**_listified(vars(row)))
            for row in update.support_passages
        ],
        facts=[
            EvidenceFactPayload(**_listified(vars(row)))
            for row in update.support_facts
        ],
        paths=[
            EvidencePathPayload(**_listified(vars(row)))
            for row in update.support_paths
        ],
        notes=[
            EvidenceNotePayload(**_listified(vars(row)))
            for row in update.support_notes
        ],
    )


def _listified(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in values.items()
    }


def _validated_highlights(rows: Any, valid_ids: set[str]) -> list[SummaryHighlight]:
    if not isinstance(rows, list):
        return []
    return [
        SummaryHighlight(
            fact_id=str(row.get("fact_id")), why=str(row.get("why") or "")
        )
        for row in rows
        if isinstance(row, dict) and str(row.get("fact_id")) in valid_ids
        and str(row.get("fact_id")).startswith("F")
    ]


def _validated_actionables(
    rows: Any,
    transcript: str,
    valid_ids: set[str],
) -> list[Actionable]:
    if not isinstance(rows, list):
        return []
    transcript_norm = _normalize_text(transcript)
    accepted: list[Actionable] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        quote = str(row.get("quote") or "").strip()
        confidence = _float(row.get("confidence"))
        grounding = [
            str(ref)
            for ref in row.get("grounding") or []
            if str(ref) in valid_ids and str(ref)[:1] in {"T", "E", "F"}
        ]
        if (
            not title
            or not quote
            or not grounding
            or confidence < config.ACTIONABLE_MIN_CONF
            or _quote_ratio(quote, transcript_norm) < config.ACTIONABLE_QUOTE_MIN_RATIO
        ):
            continue
        accepted.append(
            Actionable(
                id="",
                title=title,
                quote=quote,
                assignee=str(row.get("assignee") or ""),
                due=str(row.get("due") or ""),
                grounding=grounding,
                why_new=str(row.get("why_new") or ""),
                confidence=confidence,
            )
        )
    accepted.sort(key=lambda row: row.confidence, reverse=True)
    return accepted[: config.ACTIONABLE_MAX_PER_PASS]


def _quote_ratio(quote: str, transcript: str) -> float:
    quote_norm = _normalize_text(quote)
    return (
        fuzz.partial_ratio(quote_norm, transcript) / 100.0
        if quote_norm and transcript
        else 0.0
    )


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
    new_rows: list[Actionable] = []
    updated: dict[str, Actionable] = {}
    for candidate in candidates:
        existing = _matching_open_actionable(candidate.title, open_actionables)
        if existing is None:
            new_rows.append(candidate.model_copy(update={"id": _new_actionable_id()}))
        elif existing.id not in updated:
            updated[existing.id] = candidate.model_copy(
                update={"id": existing.id, "status": existing.status}
            )
    merged = [updated.get(row.id, row) for row in open_actionables]
    merged.extend(new_rows)
    return new_rows, list(updated.values()), merged


def _matching_open_actionable(
    title: str, rows: list[Actionable]
) -> Actionable | None:
    if not rows:
        return None
    embeddings = store.embed_texts([title, *(row.title for row in rows)])
    if len(embeddings) != len(rows) + 1:
        return None
    best = max(
        ((_cosine(embeddings[0], vector), row) for vector, row in zip(embeddings[1:], rows, strict=True)),
        key=lambda item: item[0],
        default=None,
    )
    return best[1] if best is not None and best[0] >= config.ACTIONABLE_DEDUPE_SIM else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else 0.0


def _new_actionable_id() -> str:
    return f"act-{uuid4().hex[:8]}"
