"""Bounded offline relation completion and open-predicate quarantine."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from taurine_core.indexer.facts import FactRecord, PREDICATES, promote_fact
from taurine_core.schemas import ExtractedRelation

# open predicates that are plain synonyms of registered ones become normal
# facts instead of quarantined proposals (L21c); keys are casefolded and
# underscore-normalized before lookup
_PREDICATE_SYNONYMS: dict[str, str] = {
    "based_in": "located_at",
    "belongs_to": "member_of",
    "collaborates_with": "works_with",
    "employed_at": "works_at",
    "employed_by": "works_at",
    "finished": "completed",
    "has": "owns_or_has",
    "in_charge_of": "responsible_for",
    "leads": "responsible_for",
    "located_in": "located_at",
    "meets_with": "met_with",
    "owes": "paid_or_owes",
    "owns": "owns_or_has",
    "paid": "paid_or_owes",
    "replaces": "supersedes",
    "works_for": "works_at",
}


@dataclass(frozen=True)
class CompletionEntity:
    entity_id: str
    surface: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class CompletionPassage:
    passage_id: str
    text: str
    char_start: int

    @property
    def char_end(self) -> int:
        return self.char_start + len(self.text)


@dataclass(frozen=True)
class CompletionNote:
    note_id: str
    value_score: float
    observed_at: float
    entities: tuple[CompletionEntity, ...]
    passages: tuple[CompletionPassage, ...]
    active_relation_count: int
    schema_gap: bool


@dataclass(frozen=True)
class PredicateProposal:
    proposal_id: str
    proposed_predicate: str
    description: str
    subject_entity_id: str
    object_entity_id: str
    source_passage_id: str
    assertion_start: int
    assertion_end: int
    confidence: float
    extraction_method: str
    build_id: str


@dataclass(frozen=True)
class CompletionAlias:
    entity_id: str
    alias: str
    origin: str = "completion"


@dataclass(frozen=True)
class CompletionResult:
    facts: tuple[FactRecord, ...]
    proposals: tuple[PredicateProposal, ...]
    aliases: tuple[CompletionAlias, ...] = ()
    rejected: int = 0


def select_completion_notes(notes: Sequence[CompletionNote]) -> tuple[CompletionNote, ...]:
    if not notes:
        return ()
    ordered = sorted(notes, key=lambda row: (-row.value_score, row.note_id))
    value_count = min(len(ordered), max(5, math.ceil(len(ordered) * 0.15)))
    selected = {row.note_id: row for row in ordered[:value_count]}
    for note in ordered:
        if (len(note.entities) >= 2 and note.active_relation_count == 0) or note.schema_gap:
            selected[note.note_id] = note
    return tuple(sorted(selected.values(), key=lambda row: (-row.value_score, row.note_id)))


def parse_completion_response(
    payload: object,
    notes: Sequence[CompletionNote],
    *,
    build_id: str,
) -> CompletionResult:
    if not isinstance(payload, dict):
        return CompletionResult((), ())
    note_index = {note.note_id: note for note in notes}
    facts: list[FactRecord] = []
    proposals: list[PredicateProposal] = []
    rejected = 0
    rows = payload.get("facts")
    for raw in rows if isinstance(rows, list) else ():
        parsed = _parse_row(raw, note_index, build_id)
        if parsed is None:
            rejected += 1
        elif isinstance(parsed, FactRecord):
            facts.append(parsed)
        else:
            proposals.append(parsed)
    aliases: set[CompletionAlias] = set()
    alias_rows = payload.get("aliases")
    for raw in alias_rows if isinstance(alias_rows, list) else ():
        alias = _parse_alias(raw, note_index)
        if alias is None:
            rejected += 1
        else:
            aliases.add(alias)
    return CompletionResult(
        facts=tuple(sorted(facts, key=lambda row: row.fact_id)),
        proposals=tuple(sorted(proposals, key=lambda row: row.proposal_id)),
        aliases=tuple(sorted(aliases, key=lambda row: (row.entity_id, row.alias))),
        rejected=rejected,
    )


def _parse_row(
    raw: object,
    notes: dict[str, CompletionNote],
    build_id: str,
) -> FactRecord | PredicateProposal | None:
    if not isinstance(raw, dict):
        return None
    note = notes.get(str(raw.get("note_id", "")))
    if note is None:
        return None
    passages = {row.passage_id: row for row in note.passages}
    passage = passages.get(str(raw.get("passage_id", "")))
    # every mention span is kept per entity (H8): the LLM may ground the fact
    # against any occurrence, not just an arbitrary last-seen one
    mentions_by_entity: dict[str, list[CompletionEntity]] = {}
    for row in note.entities:
        mentions_by_entity.setdefault(row.entity_id, []).append(row)
    subject_id = str(raw.get("subject_entity_id", ""))
    object_id = str(raw.get("object_entity_id", ""))
    subject_mentions = mentions_by_entity.get(subject_id, [])
    object_mentions = mentions_by_entity.get(object_id, [])
    if passage is None or not subject_mentions or not object_mentions or subject_id == object_id:
        return None
    assertion_start = _integer(raw.get("assertion_start"))
    assertion_end = _integer(raw.get("assertion_end"))
    confidence = _number(raw.get("confidence"))
    if assertion_start is None or assertion_end is None or confidence is None:
        return None
    if not passage.char_start <= assertion_start < assertion_end <= passage.char_end:
        return None

    def grounded(row: CompletionEntity) -> bool:
        return (
            assertion_start <= row.char_start < row.char_end <= assertion_end
            and passage.text.startswith(row.surface, row.char_start - passage.char_start)
        )

    pairs = [
        (subject, object_)
        for subject in subject_mentions
        for object_ in object_mentions
        if grounded(subject) and grounded(object_)
    ]
    if not pairs:
        return None
    # record the tightest containing pair: minimal joint extent, then earliest
    subject, object_ = min(
        pairs,
        key=lambda pair: (
            max(pair[0].char_end, pair[1].char_end)
            - min(pair[0].char_start, pair[1].char_start),
            pair[0].char_start,
            pair[1].char_start,
            pair[0].char_end,
            pair[1].char_end,
        ),
    )
    predicate = _normalize_predicate(str(raw.get("predicate", "")))
    if not predicate:
        return None
    if predicate not in PREDICATES:
        description = str(raw.get("predicate_description", "")).strip()
        if not description:
            return None
        identity = "\x1f".join(
            (
                predicate,
                subject.entity_id,
                object_.entity_id,
                passage.passage_id,
                str(assertion_start),
                str(assertion_end),
            )
        )
        return PredicateProposal(
            proposal_id="proposal-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            proposed_predicate=predicate,
            description=description,
            subject_entity_id=subject.entity_id,
            object_entity_id=object_.entity_id,
            source_passage_id=passage.passage_id,
            assertion_start=assertion_start,
            assertion_end=assertion_end,
            confidence=confidence,
            extraction_method="llm_relation_completion",
            build_id=build_id,
        )
    relation = ExtractedRelation(
        head=subject.surface,
        predicate=predicate,
        tail=object_.surface,
        confidence=confidence,
        subject_start=subject.char_start,
        subject_end=subject.char_end,
        object_start=object_.char_start,
        object_end=object_.char_end,
        assertion_start=assertion_start,
        assertion_end=assertion_end,
        extraction_method="llm_relation_completion",
        schema_id="relation-completion-v1",
        status="active" if confidence >= 0.7 else "candidate",
    )
    return promote_fact(
        relation,
        subject_entity_id=subject.entity_id,
        object_entity_id=object_.entity_id,
        source_passage_id=passage.passage_id,
        source_note_id=note.note_id,
        observed_at=note.observed_at,
        build_id=build_id,
    )


def _normalize_predicate(value: str) -> str:
    """Casefold/underscore-normalize, then map registered synonyms (L21c)."""
    normalized = re.sub(r"[\s\-]+", "_", value.strip().casefold())
    if normalized in PREDICATES:
        return normalized
    return _PREDICATE_SYNONYMS.get(normalized, normalized)


def _parse_alias(raw: object, notes: dict[str, CompletionNote]) -> CompletionAlias | None:
    if not isinstance(raw, dict):
        return None
    note = notes.get(str(raw.get("note_id", "")))
    if note is None:
        return None
    entity_id = str(raw.get("entity_id", ""))
    alias = str(raw.get("alias", "")).strip()
    if not alias or all(row.entity_id != entity_id for row in note.entities):
        return None
    # the alias must appear verbatim in the note's source text (L21b)
    if not any(alias in passage.text for passage in note.passages):
        return None
    return CompletionAlias(entity_id=entity_id, alias=alias)


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
