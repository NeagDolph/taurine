"""Bounded offline relation completion and open-predicate quarantine."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from taurine_core import config
from taurine_core.indexer.facts import FactRecord, PREDICATES, promote_fact
from taurine_core.schemas import ExtractedRelation

_COMPLETION_PROMPT_VERSION = "relation-completion-v1"

COMPLETION_SYSTEM_PROMPT = (
    "Return JSON {facts:[...]}. Each fact row MUST contain exactly these "
    "fields: note_id, passage_id, subject_entity_id, object_entity_id, "
    "predicate, assertion_start, assertion_end, confidence (0-1). Offsets "
    "are note-global character positions inside the cited passage, and the "
    "assertion span must contain a listed mention of both entities. Find "
    "only relations directly supported by the supplied passage text; reuse "
    "entity_id and passage_id values exactly; select a registered predicate "
    "when possible. For an unregistered predicate, also provide "
    "predicate_description. Do not infer unsupported relations. Notes "
    "carrying a `focus` field likely assert that a project or organization "
    "uses a tool/technology — extract those `uses` relations when the text "
    "supports them (subject = the project/system, object = the tool)."
)

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
    # tool-hop gate: the note mentions an eligible tool/product plus a
    # project/organization anchor but asserts no active `uses` fact — the
    # exact shape "which projects use X?" queries need and GLiNER misses in
    # technical prose. These notes join the completion lane (capped) with a
    # usage-relation focus nudge.
    tool_hop: bool = False


def completion_note_payload(note: CompletionNote) -> dict:
    # shared by the LLM call and the cache key so they can never drift apart
    return {
        "note_id": note.note_id,
        "observed_at": note.observed_at,
        "entities": [asdict(entity) for entity in note.entities],
        "passages": [asdict(passage) for passage in note.passages],
        **(
            {"focus": "tool/technology usage — check for `uses` relations"}
            if note.tool_hop
            else {}
        ),
    }


def completion_note_key(model_id: str, note: CompletionNote) -> str:
    """Content key: any change to the note's payload, the prompt, the model,
    or the predicate registry busts the cache naturally."""
    identity = json.dumps(
        {
            "version": _COMPLETION_PROMPT_VERSION,
            "model": model_id,
            "prompt": hashlib.sha256(
                COMPLETION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "predicates": [
                [name, spec.description] for name, spec in sorted(PREDICATES.items())
            ],
            "note": completion_note_payload(note),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "completion:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


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
    # the no-relation arm self-scales: barren bottom-half chatter (≥2 entities,
    # 0 facts) stays out of the LLM call instead of being re-asked every build
    values = sorted(row.value_score for row in ordered)
    pct = min(1.0, max(0.0, float(config.COMPLETION_NO_RELATION_MIN_PERCENTILE)))
    threshold = values[
        min(len(values) - 1, int(math.floor(pct * (len(values) - 1))))
    ]
    for note in ordered:
        if note.schema_gap or (
            len(note.entities) >= 2
            and note.active_relation_count == 0
            and note.value_score >= threshold
        ):
            selected[note.note_id] = note
    # tool-hop lane: strongest unselected tool/anchor notes join, capped, so
    # "which projects use X" facts get a chance without unbounding the call
    tool_hop_budget = max(0, int(config.COMPLETION_TOOL_HOP_CAP))
    for note in ordered:
        if tool_hop_budget <= 0:
            break
        if note.tool_hop and note.note_id not in selected:
            selected[note.note_id] = note
            tool_hop_budget -= 1
    return tuple(sorted(selected.values(), key=lambda row: (-row.value_score, row.note_id)))


def group_completion_rows(
    payload: object, notes: Sequence[CompletionNote]
) -> dict[str, dict]:
    """Raw model rows grouped by resolved note, for per-note cache storage.

    Fact rows resolve via note_id with the same passage_id fallback the parser
    uses; alias rows attribute to the first supplied note listing the entity.
    Every supplied note gets an entry (empty results are cached too)."""

    grouped: dict[str, dict] = {
        note.note_id: {"facts": [], "aliases": []} for note in notes
    }
    if not isinstance(payload, dict):
        return grouped
    note_ids = set(grouped)
    passage_to_note = {
        passage.passage_id: note.note_id
        for note in notes
        for passage in note.passages
    }
    rows = payload.get("facts")
    for raw in rows if isinstance(rows, list) else ():
        if not isinstance(raw, dict):
            continue
        note_id = str(raw.get("note_id", ""))
        if note_id not in note_ids:
            note_id = passage_to_note.get(str(raw.get("passage_id", "")), "")
        if note_id in note_ids:
            grouped[note_id]["facts"].append(raw)
    alias_rows = payload.get("aliases")
    for raw in alias_rows if isinstance(alias_rows, list) else ():
        if not isinstance(raw, dict):
            continue
        entity_id = str(raw.get("entity_id", ""))
        for note in notes:
            if any(row.entity_id == entity_id for row in note.entities):
                grouped[note.note_id]["aliases"].append(raw)
                break
    return grouped


def parse_completion_response(
    payload: object,
    notes: Sequence[CompletionNote],
    *,
    build_id: str,
) -> CompletionResult:
    if not isinstance(payload, dict):
        return CompletionResult((), ())
    note_index = {note.note_id: note for note in notes}
    note_by_passage = {
        passage.passage_id: note for note in notes for passage in note.passages
    }
    facts: list[FactRecord] = []
    proposals: list[PredicateProposal] = []
    rejected = 0
    rows = payload.get("facts")
    for raw in rows if isinstance(rows, list) else ():
        parsed = _parse_row(raw, note_index, build_id, note_by_passage)
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
    note_by_passage: dict[str, CompletionNote] | None = None,
) -> FactRecord | PredicateProposal | None:
    if not isinstance(raw, dict):
        return None
    note = notes.get(str(raw.get("note_id", "")))
    if note is None and note_by_passage is not None:
        # models frequently omit note_id even when instructed; passage ids are
        # globally unique, so the passage resolves the note unambiguously —
        # without this fallback the lane silently rejected every proposal
        note = note_by_passage.get(str(raw.get("passage_id", "")))
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
