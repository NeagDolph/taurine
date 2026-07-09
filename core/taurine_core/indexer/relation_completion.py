"""Bounded offline relation completion and open-predicate quarantine."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

from taurine_core.indexer.facts import FactRecord, PREDICATES, promote_fact
from taurine_core.schemas import ExtractedRelation


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
class CompletionResult:
    facts: tuple[FactRecord, ...]
    proposals: tuple[PredicateProposal, ...]
    rejected: int


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
    note_index = {note.note_id: note for note in notes}
    facts: list[FactRecord] = []
    proposals: list[PredicateProposal] = []
    rejected = 0
    rows = payload.get("facts") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return CompletionResult((), (), 0)
    for raw in rows:
        parsed = _parse_row(raw, note_index, build_id)
        if parsed is None:
            rejected += 1
        elif isinstance(parsed, FactRecord):
            facts.append(parsed)
        else:
            proposals.append(parsed)
    return CompletionResult(
        facts=tuple(sorted(facts, key=lambda row: row.fact_id)),
        proposals=tuple(sorted(proposals, key=lambda row: row.proposal_id)),
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
    entities = {row.entity_id: row for row in note.entities}
    subject = entities.get(str(raw.get("subject_entity_id", "")))
    object_ = entities.get(str(raw.get("object_entity_id", "")))
    if passage is None or subject is None or object_ is None or subject.entity_id == object_.entity_id:
        return None
    assertion_start = _integer(raw.get("assertion_start"))
    assertion_end = _integer(raw.get("assertion_end"))
    confidence = _number(raw.get("confidence"))
    if assertion_start is None or assertion_end is None or confidence is None:
        return None
    if not (
        passage.char_start <= assertion_start < assertion_end <= passage.char_end
        and assertion_start <= subject.char_start < subject.char_end <= assertion_end
        and assertion_start <= object_.char_start < object_.char_end <= assertion_end
    ):
        return None
    subject_local = subject.char_start - passage.char_start
    object_local = object_.char_start - passage.char_start
    if not passage.text.startswith(subject.surface, subject_local) or not passage.text.startswith(
        object_.surface, object_local
    ):
        return None
    predicate = str(raw.get("predicate", "")).strip()
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


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
