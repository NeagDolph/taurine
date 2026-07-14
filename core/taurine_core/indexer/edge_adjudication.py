"""Grounded LLM edge adjudication (DEG-RAG "triple reflection", edge-quality
spec items 1, 5b, 6).

Three lanes share one judge callable (``edge_adjudicator_for``) and one
BuildCache verdict family:

- fact adjudication: every candidate/active fact is judged against its own
  source passage text — support, significance, direction, predicate fit —
  with a mandatory verbatim quote, verified in code;
- semantic-edge reasons: each surviving cross-note semantic edge gets a
  ≤14-word concrete reason or is dropped;
- merge adjudication: gray-zone entity pairs the deterministic resolver
  declined are judged from surfaces plus mention contexts.

Every lane is fail-open and cache-hard: no LLM or a failed batch means
today's behavior (facts kept unadjudicated, edges kept with empty reasons,
no merges) and a build can NEVER fail from adjudication. The judge only ever
sees supplied source text; verdicts that violate the grounding contract
(quote not in the text, unknown keys, non-registry relabels) are treated as
malformed and discarded.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from taurine_core import config
from taurine_core.indexer.entities import GrayZonePair, ResolutionResult, _digits
from taurine_core.indexer.facts import (
    FactRecord,
    PREDICATES,
    reconcile_temporal_facts,
    relabel_fact,
)

_LOG = logging.getLogger("taurine.build")

_FACT_PROMPT_VERSION = "fact-adjudication-v1"
_REASON_PROMPT_VERSION = "semantic-reason-v1"
_MERGE_PROMPT_VERSION = "merge-adjudication-v1"

_FACT_SYSTEM_PROMPT = (
    "You are a strict adjudicator of facts extracted into a personal knowledge "
    "graph. You receive one or more source notes, each with its own text and the "
    "candidate facts (subject — predicate — object) extracted from it. Judge each "
    "fact ONLY against its own note's supplied text, never against another note "
    "or world knowledge. For each fact return an "
    "object with exactly these fields:\n"
    '- "fact_key": copied verbatim from the input fact.\n'
    '- "support": "supported" if the text plainly asserts the relation, '
    '"partial" if it only suggests or ambiguously states it, "unsupported" '
    "otherwise.\n"
    '- "quote": a verbatim substring of the supplied text that best supports '
    "or refutes the fact. Copy the characters exactly; do not paraphrase.\n"
    '- "significant": false for trivial or incidental connections that do not '
    "describe a real, useful relationship — for example, a fact like "
    '"3g supersedes 2g" extracted from casual slang is insignificant.\n'
    '- "direction_ok": whether subject and object are in the correct order '
    "for the predicate.\n"
    '- "relabel": a predicate from the provided predicate list that fits the '
    "quoted text better than the current one, else null.\n"
    '- "score": 0.0-1.0 overall reliability combining accuracy against the '
    "quoted text, significance, entity-type/predicate compatibility, and "
    "direction. Unsupported, insignificant, wrong-direction, or "
    "type-incompatible facts score below 0.3; facts the text only partially "
    "supports score 0.3-0.7; plainly asserted, meaningful facts score 0.7 or "
    'above.\nReturn JSON {"facts": [...]} with one entry per input fact.'
)

_REASON_SYSTEM_PROMPT = (
    "You judge whether two text passages from different notes in a personal "
    "knowledge base are genuinely related. For each pair: if a concrete, "
    "specific relationship connects the CONTENTS of the two passages (the "
    "same event, project, decision, or people doing connected things), state "
    "it in at most 14 words, concretely naming what connects them. If the "
    "passages merely share writing style, register, format, or vague topical "
    'similarity, answer exactly "NONE". Use only the supplied text; never '
    'world knowledge. Return JSON {"pairs": {"<key>": "<reason or NONE>"}} '
    "with one entry per input pair, reusing each key verbatim."
)

_MERGE_SYSTEM_PROMPT = (
    "You decide whether two entity surface forms from a personal knowledge "
    "base refer to the same real-world entity. Judge ONLY from the supplied "
    "surfaces and their usage contexts, never from world knowledge. Different "
    "people can share a first name; an acronym can name several things; "
    "merge only when the contexts clearly point at one entity. Return JSON "
    '{"pairs": {"<key>": {"same_entity": true|false, "confidence": 0.0-1.0, '
    '"why": "<at most 10 words>"}}} with one entry per input pair, reusing '
    "each key verbatim."
)


def edge_adjudicator_for(llm: Any) -> Callable[[str, dict], Any]:
    """Bounded judge callable shared by the server and CLI (all lanes)."""

    def adjudicate(system_prompt: str, payload: dict) -> Any:
        return asyncio.run(
            llm.complete_json(
                config.ADJUDICATION_MODEL,
                system_prompt,
                json.dumps(payload, ensure_ascii=False),
                timeout_s=config.SUMMARY_TIMEOUT_S,
                temperature=0.0,
                max_tokens=config.ADJUDICATION_MAX_TOKENS,
            )
        )

    return adjudicate


# ── item 1: fact adjudication ──────────────────────────────────────────────


def _needs_adjudication(fact: FactRecord, note_sources: Mapping[str, str]) -> bool:
    """Gate: only risky facts are worth an LLM grounding check.

    Always judge episodic sources, low-confidence facts, exclusive predicates
    (they drive 'current' answers), and LLM-completed relations. Trust the rest.
    """
    from taurine_core.indexer.facts import PREDICATES

    if fact.extraction_method == "llm_relation_completion":
        return True
    if fact.confidence < config.ADJUDICATION_TRUST_CONF:
        return True
    spec = PREDICATES.get(fact.predicate)
    if spec is not None and spec.exclusive_object:
        return True
    source = note_sources.get(fact.source_note_id, "")
    if source not in config.ADJUDICATION_EVERGREEN_SOURCES:
        return True
    return False


@dataclass(frozen=True)
class FactJudgment:
    fact_id: str
    support: str
    quote: str
    significant: bool
    direction_ok: bool
    relabel: str | None
    score: float


@dataclass(frozen=True)
class _NoteSection:
    note_id: str
    title: str
    text: str


@dataclass(frozen=True)
class _FactBatch:
    """One judge call: one or more notes' relevant passage text plus their facts."""

    sections: tuple[_NoteSection, ...]
    facts: tuple[FactRecord, ...]
    assertions: Mapping[str, str]


@dataclass(frozen=True)
class FactAdjudicationOutcome:
    facts: tuple[FactRecord, ...]
    adjudicated_by_predicate: dict[str, int]
    kept_by_predicate: dict[str, int]
    downweighted_by_predicate: dict[str, int]
    rejected_by_predicate: dict[str, int]
    relabeled_by_predicate: dict[str, int]
    unadjudicated: int
    failed_batches: int
    malformed_rows: int

    @property
    def adjudicated(self) -> int:
        return sum(self.adjudicated_by_predicate.values())


def adjudicate_facts(
    facts: Sequence[FactRecord],
    *,
    passages_by_note: Mapping[str, Sequence[Any]],
    note_titles: Mapping[str, str],
    judge: Callable[[str, dict], Any] | None,
    cache: Any | None = None,
    model_id: str = config.ADJUDICATION_MODEL,
    note_sources: Mapping[str, str] | None = None,
) -> FactAdjudicationOutcome:
    """Judge candidate/active facts against their source passages; fail-open.

    Outcomes: score < ADJUDICATION_REJECT_BELOW ⇒ status 'rejected' (audit
    row); mid-band scores are stored and multiply the fact's graph edge
    weight downstream; valid relabels re-derive fact identity and temporal
    reconciliation re-runs over the adjusted set.

    ``note_sources`` enables intelligent gating: a high-confidence fact from an
    evergreen written source on a non-exclusive predicate is trusted and never
    sent to the judge (GLiNER already gated it). Episodic (chat) sources, low
    confidence, exclusive predicates, and LLM-completed relations are always
    judged — that is where the noise lives.
    """

    ordered = tuple(sorted(facts, key=lambda fact: fact.fact_id))
    sources = dict(note_sources or {})
    eligible = [
        fact
        for fact in ordered
        if fact.status in {"active", "candidate"} and _needs_adjudication(fact, sources)
    ]
    if not eligible:
        return _fact_outcome(ordered, {}, failed_batches=0, malformed_rows=0)

    draft_by_passage = {
        draft.passage_id: draft
        for drafts in passages_by_note.values()
        for draft in drafts
    }
    keys = {
        fact.fact_id: _fact_key(model_id, fact, draft_by_passage)
        for fact in eligible
        if fact.source_passage_id in draft_by_passage
    }
    cached: dict[str, FactJudgment] = {}
    if cache is not None and keys:
        by_key = cache.cached_judgments(sorted(set(keys.values())))
        for fact_id, key in keys.items():
            row = by_key.get(key)
            if row is None:
                continue
            judgment = _judgment_from_payload(fact_id, row)
            if judgment is not None:
                cached[fact_id] = judgment

    pending = [
        fact for fact in eligible if fact.fact_id in keys and fact.fact_id not in cached
    ]
    judgments: dict[str, FactJudgment] = dict(cached)
    if bool(config.NLI_SCREEN_ENABLED) and pending:
        try:
            from taurine_core import models_local
            from taurine_core.indexer.nli_screen import screen

            scorer = models_local.local_nli_scorer(config.EVIDENCE_NLI_MODEL)
        except Exception as exc:
            _LOG.warning("NLI fact screen unavailable; using LLM lane: %s", exc)
        else:
            screened = screen(pending, draft_by_passage, scorer)
            judgments.update(screened.kept)
            pending = screened.pending

    if judge is None:
        return _fact_outcome(
            ordered, judgments, failed_batches=0, malformed_rows=0
        )

    batches = _fact_batches(pending, draft_by_passage, note_titles)
    failed = 0
    malformed = 0
    if batches:
        workers = max(1, min(int(config.ADJUDICATION_CONCURRENCY), len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            responses = list(
                executor.map(_call_fail_open(judge, _FACT_SYSTEM_PROMPT, _fact_payload), batches)
            )
        for batch, response in zip(batches, responses):
            if response is None:
                failed += 1
                continue
            parsed, batch_malformed = _parse_fact_response(response, batch)
            if parsed is None:
                _LOG.warning(
                    "fact adjudication batch for note %s returned a malformed response",
                    ", ".join(section.note_id for section in batch.sections),
                )
                failed += 1
                continue
            malformed += batch_malformed
            judgments.update(parsed)
            if cache is not None:
                cache.store_judgments(
                    [
                        (keys[fact_id], "fact", _judgment_payload(judgment))
                        for fact_id, judgment in sorted(parsed.items())
                    ]
                )
    return _fact_outcome(ordered, judgments, failed_batches=failed, malformed_rows=malformed)


def _fact_outcome(
    facts: tuple[FactRecord, ...],
    judgments: Mapping[str, FactJudgment],
    *,
    failed_batches: int,
    malformed_rows: int,
) -> FactAdjudicationOutcome:
    adjudicated: dict[str, int] = {}
    kept: dict[str, int] = {}
    downweighted: dict[str, int] = {}
    rejected: dict[str, int] = {}
    relabeled: dict[str, int] = {}
    final: dict[str, FactRecord] = {}
    for fact in facts:
        judgment = judgments.get(fact.fact_id)
        if judgment is None:
            final.setdefault(fact.fact_id, fact)
            continue
        row = fact
        relabel = judgment.relabel
        if relabel and relabel in PREDICATES and relabel != row.predicate:
            row = relabel_fact(row, relabel)
            _bump(relabeled, fact.predicate)
        score = judgment.score
        _bump(adjudicated, row.predicate)
        if score < config.ADJUDICATION_REJECT_BELOW:
            row = replace(row, status="rejected")
            _bump(rejected, row.predicate)
        elif score < config.ADJUDICATION_DOWNWEIGHT_BELOW:
            _bump(downweighted, row.predicate)
        else:
            _bump(kept, row.predicate)
        row = replace(
            row,
            adjudication_score=score,
            adjudication_quote=judgment.quote,
            adjudicated=True,
        )
        # a relabel can collide with an already-extracted identity: dedupe,
        # preferring the earlier (already-judged or original) row
        final.setdefault(row.fact_id, row)
    survivors = [row for row in final.values() if row.status != "rejected"]
    rejected_rows = [row for row in final.values() if row.status == "rejected"]
    # relabels shift predicate groups and rejections remove exclusivity
    # participants; run temporal reconciliation once more over the survivors
    reconciled = reconcile_temporal_facts(tuple(survivors))
    adjusted = tuple(
        sorted((*reconciled, *rejected_rows), key=lambda row: row.fact_id)
    )
    return FactAdjudicationOutcome(
        facts=adjusted,
        adjudicated_by_predicate=dict(sorted(adjudicated.items())),
        kept_by_predicate=dict(sorted(kept.items())),
        downweighted_by_predicate=dict(sorted(downweighted.items())),
        rejected_by_predicate=dict(sorted(rejected.items())),
        relabeled_by_predicate=dict(sorted(relabeled.items())),
        unadjudicated=len(adjusted) - sum(adjudicated.values()),
        failed_batches=failed_batches,
        malformed_rows=malformed_rows,
    )


def _fact_batches(
    facts: Sequence[FactRecord],
    draft_by_passage: Mapping[str, Any],
    note_titles: Mapping[str, str],
) -> tuple[_FactBatch, ...]:
    """Deterministic batches packing multiple notes per call (measured builds
    averaged ~3 facts/call against a 25-fact cap when batches were per-note).

    A note's facts stay contiguous and ordered; the supplied text stays
    bounded at ADJUDICATION_TEXT_MAX_CHARS per call."""

    by_note: dict[str, list[FactRecord]] = {}
    for fact in facts:
        by_note.setdefault(fact.source_note_id, []).append(fact)
    max_chars = int(config.ADJUDICATION_TEXT_MAX_CHARS)
    max_facts = max(1, int(config.ADJUDICATION_FACTS_PER_CALL))
    batches: list[_FactBatch] = []

    current_facts: list[FactRecord] = []
    # note_id → {passage_id: text}, insertion-ordered
    current_notes: dict[str, dict[str, str]] = {}
    current_chars = 0

    def flush() -> None:
        nonlocal current_facts, current_notes, current_chars
        if current_facts:
            sections = tuple(
                _NoteSection(
                    note_id=note_id,
                    title=note_titles.get(note_id, ""),
                    text="\n\n".join(passages.values()),
                )
                for note_id, passages in current_notes.items()
            )
            batches.append(
                _FactBatch(
                    sections=sections,
                    facts=tuple(current_facts),
                    assertions={
                        fact.fact_id: _assertion_text(
                            fact, draft_by_passage[fact.source_passage_id]
                        )
                        for fact in current_facts
                    },
                )
            )
        current_facts, current_notes, current_chars = [], {}, 0

    for note_id in sorted(by_note):
        rows = sorted(
            by_note[note_id],
            key=lambda fact: (
                draft_by_passage[fact.source_passage_id].ordinal,
                fact.assertion_start,
                fact.fact_id,
            ),
        )
        for fact in rows:
            draft = draft_by_passage[fact.source_passage_id]
            passage_text = draft.text[:max_chars]
            seen = draft.passage_id in current_notes.get(note_id, {})
            added = 0 if seen else len(passage_text) + 2
            if current_facts and (
                len(current_facts) >= max_facts
                or current_chars + added > max_chars
            ):
                flush()
                added = len(passage_text) + 2
            current_notes.setdefault(note_id, {})
            if draft.passage_id not in current_notes[note_id]:
                current_notes[note_id][draft.passage_id] = passage_text
                current_chars += added
            current_facts.append(fact)
    flush()
    return tuple(batches)


def _fact_payload(batch: _FactBatch) -> dict:
    facts_by_note: dict[str, list[FactRecord]] = {}
    for fact in batch.facts:
        facts_by_note.setdefault(fact.source_note_id, []).append(fact)
    return {
        "predicates": {
            name: spec.description for name, spec in sorted(PREDICATES.items())
        },
        "notes": [
            {
                "note_title": section.title,
                "text": section.text,
                "facts": [
                    {
                        "fact_key": fact.fact_id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "predicate_description": PREDICATES[
                            fact.predicate
                        ].description,
                        "object": fact.object,
                        "assertion": batch.assertions.get(fact.fact_id, ""),
                    }
                    for fact in facts_by_note.get(section.note_id, ())
                ],
            }
            for section in batch.sections
        ],
    }


def _parse_fact_response(
    response: Any, batch: _FactBatch
) -> tuple[dict[str, FactJudgment] | None, int]:
    """(judgments by fact_id, malformed row count); None ⇒ whole batch failed.

    The grounding contract is enforced here, not in the prompt: a judgment
    whose quote does not appear verbatim (modulo whitespace) in the supplied
    text is malformed and the fact stays unadjudicated (fail-open keep)."""

    rows = response.get("facts") if isinstance(response, Mapping) else None
    if not isinstance(rows, list):
        return None, 0
    known = {fact.fact_id: fact.source_note_id for fact in batch.facts}
    normalized_by_note = {
        section.note_id: _normalize_ws(section.text) for section in batch.sections
    }
    parsed: dict[str, FactJudgment] = {}
    malformed = 0
    for raw in rows:
        judgment = _judgment_from_payload(
            str(raw.get("fact_key", "")) if isinstance(raw, Mapping) else "", raw
        )
        if (
            judgment is None
            or judgment.fact_id not in known
            or judgment.fact_id in parsed
            or not judgment.quote
            or _normalize_ws(judgment.quote)
            not in normalized_by_note.get(known[judgment.fact_id], "")
        ):
            malformed += 1
            continue
        parsed[judgment.fact_id] = judgment
    return parsed, malformed


def _judgment_from_payload(fact_id: str, raw: Any) -> FactJudgment | None:
    if not isinstance(raw, Mapping) or not fact_id:
        return None
    support = str(raw.get("support", "")).strip().casefold()
    if support not in {"supported", "partial", "unsupported"}:
        return None
    quote = raw.get("quote")
    score = raw.get("score")
    if not isinstance(quote, str) or not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    relabel_raw = raw.get("relabel")
    relabel = None
    if isinstance(relabel_raw, str) and relabel_raw.strip():
        relabel = relabel_raw.strip().casefold()
        if relabel not in PREDICATES:
            return None
    return FactJudgment(
        fact_id=fact_id,
        support=support,
        quote=quote,
        significant=bool(raw.get("significant", True)),
        direction_ok=bool(raw.get("direction_ok", True)),
        relabel=relabel,
        score=min(1.0, max(0.0, float(score))),
    )


def _judgment_payload(judgment: FactJudgment) -> dict:
    return {
        "support": judgment.support,
        "quote": judgment.quote,
        "significant": judgment.significant,
        "direction_ok": judgment.direction_ok,
        "relabel": judgment.relabel,
        "score": judgment.score,
    }


def _assertion_text(fact: FactRecord, draft: Any) -> str:
    start = max(0, fact.assertion_start - draft.char_start)
    end = max(start, fact.assertion_end - draft.char_start)
    return draft.text[start:end][:500]


def _fact_key(model_id: str, fact: FactRecord, draft_by_passage: Mapping[str, Any]) -> str:
    """(model, prompt version, fact logical key, source passage content hash)."""
    draft = draft_by_passage[fact.source_passage_id]
    identity = "\x1f".join(
        (
            _FACT_PROMPT_VERSION,
            model_id,
            fact.subject,
            fact.predicate,
            fact.object,
            str(fact.assertion_start),
            str(fact.assertion_end),
            draft.content_hash,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# ── item 5b: semantic-edge reasons ─────────────────────────────────────────


@dataclass(frozen=True)
class SemanticPair:
    """One cross-note semantic edge candidate (src < dst by passage id)."""

    src_passage_id: str
    dst_passage_id: str
    src_title: str
    dst_title: str
    src_text: str
    dst_text: str
    src_content_hash: str
    dst_content_hash: str

    @property
    def pair(self) -> tuple[str, str]:
        return (self.src_passage_id, self.dst_passage_id)


@dataclass(frozen=True)
class SemanticReasonOutcome:
    reasons: dict[tuple[str, str], str]
    dropped: frozenset[tuple[str, str]]
    reasoned_count: int
    dropped_count: int
    failed_batches: int


_NO_REASON = "NONE"


def adjudicate_semantic_edges(
    pairs: Sequence[SemanticPair],
    *,
    judge: Callable[[str, dict], Any] | None,
    cache: Any | None = None,
    model_id: str = config.ADJUDICATION_MODEL,
) -> SemanticReasonOutcome:
    """Reason (or drop) cross-note semantic edges; fail-open keeps edges with
    empty reasons."""

    ordered = tuple(sorted(pairs, key=lambda row: row.pair))
    if judge is None or not ordered:
        return SemanticReasonOutcome({}, frozenset(), 0, 0, 0)
    keys = {row.pair: _reason_key(model_id, row) for row in ordered}
    verdicts: dict[tuple[str, str], str] = {}
    if cache is not None:
        by_key = cache.cached_judgments(sorted(set(keys.values())))
        for pair, key in keys.items():
            row = by_key.get(key)
            if isinstance(row, Mapping) and isinstance(row.get("reason"), str):
                verdicts[pair] = str(row["reason"])
    pending = [row for row in ordered if row.pair not in verdicts]
    batch_size = max(1, int(config.SEMANTIC_REASON_BATCH))
    batches = [
        tuple(pending[start : start + batch_size])
        for start in range(0, len(pending), batch_size)
    ]
    failed = 0
    if batches:
        workers = max(1, min(int(config.ADJUDICATION_CONCURRENCY), len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            responses = list(
                executor.map(
                    _call_fail_open(judge, _REASON_SYSTEM_PROMPT, _reason_payload),
                    batches,
                )
            )
        for batch, response in zip(batches, responses):
            parsed = _parse_reason_response(response, batch)
            if parsed is None:
                failed += 1
                continue
            verdicts.update(parsed)
            if cache is not None:
                cache.store_judgments(
                    [
                        (keys[pair], "semantic_reason", {"reason": reason})
                        for pair, reason in sorted(parsed.items())
                    ]
                )
    reasons: dict[tuple[str, str], str] = {}
    dropped: set[tuple[str, str]] = set()
    for row in ordered:
        verdict = verdicts.get(row.pair)
        if verdict is None:
            continue  # fail-open: edge kept, reason ''
        if verdict.strip().upper() == _NO_REASON or not verdict.strip():
            dropped.add(row.pair)
        else:
            reasons[row.pair] = _clamp_words(verdict, 14)
    return SemanticReasonOutcome(
        reasons=reasons,
        dropped=frozenset(dropped),
        reasoned_count=len(reasons),
        dropped_count=len(dropped),
        failed_batches=failed,
    )


def _reason_payload(batch: Sequence[SemanticPair]) -> dict:
    snippet = int(config.SEMANTIC_REASON_SNIPPET_CHARS)
    return {
        "pairs": [
            {
                "key": str(index),
                "a": {"title": row.src_title, "snippet": row.src_text[:snippet]},
                "b": {"title": row.dst_title, "snippet": row.dst_text[:snippet]},
            }
            for index, row in enumerate(batch)
        ]
    }


def _parse_reason_response(
    response: Any, batch: Sequence[SemanticPair]
) -> dict[tuple[str, str], str] | None:
    if response is None:
        return None
    rows = response.get("pairs") if isinstance(response, Mapping) else None
    if not isinstance(rows, Mapping):
        return None
    parsed: dict[tuple[str, str], str] = {}
    for raw_key, raw_reason in rows.items():
        try:
            index = int(raw_key)
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(batch) or not isinstance(raw_reason, str):
            continue
        parsed[batch[index].pair] = raw_reason.strip()
    return parsed


def _reason_key(model_id: str, row: SemanticPair) -> str:
    """(model, prompt version, src/dst passage content hashes, orderless)."""
    left, right = sorted((row.src_content_hash, row.dst_content_hash))
    identity = "\x1f".join((_REASON_PROMPT_VERSION, model_id, left, right))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# ── item 6: merge adjudication ─────────────────────────────────────────────


@dataclass(frozen=True)
class MergeOutcome:
    approved: tuple[tuple[str, str, str], ...]  # (etype, left_norm, right_norm)
    judged_count: int
    approved_count: int
    failed_batches: int


def merge_mention_contexts(
    result: ResolutionResult,
    note_texts: Mapping[str, str],
    *,
    per_entity: int = 2,
    pad: int = 80,
) -> dict[str, tuple[str, ...]]:
    """Up to ``per_entity`` ±``pad``-char mention snippets per gray-zone side."""

    wanted = {
        entity_id
        for pair in result.gray_zone_pairs
        for entity_id in (pair.left_entity_id, pair.right_entity_id)
    }
    by_entity: dict[str, list[Any]] = {}
    for mention in result.mentions:
        if mention.entity_id in wanted:
            by_entity.setdefault(mention.entity_id, []).append(mention)
    contexts: dict[str, tuple[str, ...]] = {}
    for entity_id in sorted(wanted):
        rows = sorted(
            by_entity.get(entity_id, ()),
            key=lambda row: (row.note_id, row.mention.char_start, row.mention_id),
        )[: max(0, int(per_entity))]
        snippets = []
        for row in rows:
            text = note_texts.get(row.note_id, "")
            start = max(0, row.mention.char_start - pad)
            end = min(len(text), row.mention.char_end + pad)
            snippet = _normalize_ws(text[start:end])
            if snippet:
                snippets.append(snippet)
        contexts[entity_id] = tuple(snippets)
    return contexts


def adjudicate_merges(
    pairs: Sequence[GrayZonePair],
    *,
    contexts: Mapping[str, Sequence[str]],
    judge: Callable[[str, dict], Any] | None,
    cache: Any | None = None,
    model_id: str = config.ADJUDICATION_MODEL,
) -> MergeOutcome:
    """Judge gray-zone entity pairs; approve merges at same_entity ∧
    confidence ≥ MERGE_ADJUDICATION_MIN_CONF. Digit guards are re-enforced in
    code — a yes verdict never merges across them."""

    ordered = tuple(
        sorted(pairs, key=lambda row: (row.etype, row.left_normalized, row.right_normalized))
    )
    if judge is None or not ordered:
        return MergeOutcome((), 0, 0, 0)
    keys = {_pair_id(row): _merge_key(model_id, row) for row in ordered}
    verdicts: dict[tuple[str, str, str], dict] = {}
    if cache is not None:
        by_key = cache.cached_judgments(sorted(set(keys.values())))
        for pair_id, key in keys.items():
            row = by_key.get(key)
            if isinstance(row, Mapping) and isinstance(row.get("same_entity"), bool):
                verdicts[pair_id] = dict(row)
    pending = [row for row in ordered if _pair_id(row) not in verdicts]
    batch_size = max(1, int(config.MERGE_ADJUDICATION_BATCH))
    batches = [
        tuple(pending[start : start + batch_size])
        for start in range(0, len(pending), batch_size)
    ]
    failed = 0
    if batches:
        workers = max(1, min(int(config.ADJUDICATION_CONCURRENCY), len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            responses = list(
                executor.map(
                    _call_fail_open(
                        judge,
                        _MERGE_SYSTEM_PROMPT,
                        lambda batch: _merge_payload(batch, contexts),
                    ),
                    batches,
                )
            )
        for batch, response in zip(batches, responses):
            parsed = _parse_merge_response(response, batch)
            if parsed is None:
                failed += 1
                continue
            verdicts.update(parsed)
            if cache is not None:
                cache.store_judgments(
                    [
                        (keys[pair_id], "merge", verdict)
                        for pair_id, verdict in sorted(parsed.items())
                    ]
                )
    approved: list[tuple[str, str, str]] = []
    for row in ordered:
        verdict = verdicts.get(_pair_id(row))
        if verdict is None:
            continue
        confidence = verdict.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            continue
        if not (
            bool(verdict.get("same_entity"))
            and float(confidence) >= float(config.MERGE_ADJUDICATION_MIN_CONF)
        ):
            continue
        # absolute guards, judge-proof: same etype is structural (pairs carry
        # one etype); mismatched digits never merge
        if _digits(row.left_surface) != _digits(row.right_surface):
            _LOG.info(
                "merge adjudication: digit guard vetoes %r ↔ %r despite yes verdict",
                row.left_surface,
                row.right_surface,
            )
            continue
        approved.append(_pair_id(row))
    return MergeOutcome(
        approved=tuple(approved),
        judged_count=len(verdicts),
        approved_count=len(approved),
        failed_batches=failed,
    )


def _merge_payload(
    batch: Sequence[GrayZonePair], contexts: Mapping[str, Sequence[str]]
) -> dict:
    return {
        "pairs": [
            {
                "key": str(index),
                "type": row.etype,
                "a": {
                    "surface": row.left_surface,
                    "contexts": list(contexts.get(row.left_entity_id, ()))[:2],
                },
                "b": {
                    "surface": row.right_surface,
                    "contexts": list(contexts.get(row.right_entity_id, ()))[:2],
                },
            }
            for index, row in enumerate(batch)
        ]
    }


def _parse_merge_response(
    response: Any, batch: Sequence[GrayZonePair]
) -> dict[tuple[str, str, str], dict] | None:
    if response is None:
        return None
    rows = response.get("pairs") if isinstance(response, Mapping) else None
    if not isinstance(rows, Mapping):
        return None
    parsed: dict[tuple[str, str, str], dict] = {}
    for raw_key, raw in rows.items():
        try:
            index = int(raw_key)
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(batch) or not isinstance(raw, Mapping):
            continue
        confidence = raw.get("confidence")
        if not isinstance(raw.get("same_entity"), bool):
            continue
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            continue
        parsed[_pair_id(batch[index])] = {
            "same_entity": bool(raw["same_entity"]),
            "confidence": min(1.0, max(0.0, float(confidence))),
            "why": _clamp_words(str(raw.get("why", "")), 10),
        }
    return parsed


def _pair_id(row: GrayZonePair) -> tuple[str, str, str]:
    return (row.etype, row.left_normalized, row.right_normalized)


def _merge_key(model_id: str, row: GrayZonePair) -> str:
    """(etype, surfaceA, surfaceB, prompt version) — surfaces sorted."""
    left, right = sorted((row.left_surface, row.right_surface))
    identity = "\x1f".join((_MERGE_PROMPT_VERSION, model_id, row.etype, left, right))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# ── shared plumbing ────────────────────────────────────────────────────────


def _call_fail_open(
    judge: Callable[[str, dict], Any],
    system_prompt: str,
    payload_for: Callable[[Any], dict],
) -> Callable[[Any], Any]:
    def call(batch: Any) -> Any:
        try:
            return judge(system_prompt, payload_for(batch))
        except Exception as exc:  # noqa: BLE001 - adjudication must never fail a build
            _LOG.warning(
                "edge adjudication batch failed (%s): %s", type(exc).__name__, exc
            )
            return None

    return call


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _clamp_words(text: str, limit: int) -> str:
    words = _normalize_ws(text).split()
    return " ".join(words[:limit])
