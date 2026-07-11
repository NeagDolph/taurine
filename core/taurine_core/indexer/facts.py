"""Directed predicate registry and temporal fact reconciliation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from taurine_core import config
from taurine_core.retrieval.types import TemporalIntent
from taurine_core.schemas import ExtractedRelation


FactStatus = Literal[
    "active", "historical", "superseded", "conflicted", "candidate", "rejected"
]


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    description: str
    directional: bool = True
    inverse: str | None = None
    reverse_traversal_weight: float = config.FACT_REVERSE_WEIGHT_DEFAULT
    cardinality: str = "many_to_many"
    temporal: bool = True
    exclusive_object: bool = False
    base_weight: float = 0.25


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    subject_entity_id: str
    subject: str
    predicate: str
    object_entity_id: str
    object: str
    source_passage_id: str
    source_note_id: str
    assertion_start: int
    assertion_end: int
    subject_start: int
    subject_end: int
    object_start: int
    object_end: int
    confidence: float
    extraction_method: str
    schema_id: str
    observed_at: float
    valid_from: float | None
    valid_to: float | None
    invalidated_at: float | None
    superseded_by: str | None
    status: FactStatus
    build_id: str
    # grounded adjudication verdict (edge-quality item 1); adjudicated stays
    # False when the judge lane was unavailable or the row was skipped
    adjudication_score: float | None = None
    adjudication_quote: str = ""
    adjudicated: bool = False


def _predicate(
    name: str,
    description: str,
    *,
    inverse: str | None = None,
    directional: bool = True,
    reverse: float = config.FACT_REVERSE_WEIGHT_DEFAULT,
    exclusive: bool = False,
    temporal: bool = True,
    base_weight: float = 0.25,
) -> PredicateSpec:
    return PredicateSpec(
        name=name,
        description=description,
        directional=directional,
        inverse=inverse,
        reverse_traversal_weight=reverse if directional else 1.0,
        temporal=temporal,
        exclusive_object=exclusive,
        base_weight=base_weight,
    )


PREDICATES: dict[str, PredicateSpec] = {
    "works_with": _predicate("works_with", "works with", inverse="works_with"),
    "works_at": _predicate("works_at", "works at", inverse="employs", exclusive=True),
    "member_of": _predicate("member_of", "is a member of", inverse="has_member"),
    "part_of": _predicate("part_of", "is part of", inverse="contains"),
    "located_at": _predicate("located_at", "is located at", inverse="location_of", exclusive=True),
    "scheduled_for": _predicate("scheduled_for", "is scheduled for", inverse="scheduled_item", exclusive=True),
    "paid_or_owes": _predicate("paid_or_owes", "paid or owes", inverse="paid_by_or_owed_by"),
    "owns_or_has": _predicate("owns_or_has", "owns or has", inverse="owned_or_held_by"),
    "uses": _predicate("uses", "uses", inverse="used_by"),
    "about": _predicate("about", "is about", inverse="topic_of", temporal=False),
    "met_with": _predicate("met_with", "met with", inverse="met_with"),
    "related_to": _predicate(
        "related_to", "is related to", directional=False, reverse=1.0, temporal=False, base_weight=0.15
    ),
    "decided": _predicate("decided", "decided", inverse="decided_by"),
    "prefers": _predicate("prefers", "prefers", inverse="preferred_by"),
    "blocked_by": _predicate("blocked_by", "is blocked by", inverse="blocks"),
    "responsible_for": _predicate(
        "responsible_for", "is responsible for", inverse="responsibility_of"
    ),
    "completed": _predicate("completed", "completed", inverse="completed_by"),
    "supersedes": _predicate("supersedes", "supersedes", inverse="superseded_by"),
}


def promote_fact(
    relation: ExtractedRelation,
    *,
    subject_entity_id: str,
    object_entity_id: str,
    source_passage_id: str,
    source_note_id: str,
    observed_at: float,
    build_id: str,
    valid_from: float | None = None,
    valid_to: float | None = None,
) -> FactRecord:
    if relation.predicate not in PREDICATES:
        raise ValueError(f"unregistered predicate: {relation.predicate}")
    if subject_entity_id == object_entity_id:
        raise ValueError("fact endpoints resolve to the same entity")
    fact_id = derive_fact_id(
        subject_entity_id,
        relation.predicate,
        object_entity_id,
        source_passage_id,
        relation.assertion_start,
        relation.assertion_end,
    )
    return FactRecord(
        fact_id=fact_id,
        subject_entity_id=subject_entity_id,
        subject=relation.head,
        predicate=relation.predicate,
        object_entity_id=object_entity_id,
        object=relation.tail,
        source_passage_id=source_passage_id,
        source_note_id=source_note_id,
        assertion_start=relation.assertion_start,
        assertion_end=relation.assertion_end,
        subject_start=relation.subject_start,
        subject_end=relation.subject_end,
        object_start=relation.object_start,
        object_end=relation.object_end,
        confidence=relation.confidence,
        extraction_method=relation.extraction_method,
        schema_id=relation.schema_id,
        observed_at=float(observed_at),
        valid_from=valid_from,
        valid_to=valid_to,
        invalidated_at=None,
        superseded_by=None,
        status=relation.status,
        build_id=build_id,
    )


def derive_fact_id(
    subject_entity_id: str,
    predicate: str,
    object_entity_id: str,
    source_passage_id: str,
    assertion_start: int,
    assertion_end: int,
) -> str:
    """The one fact-identity derivation, shared by promotion and relabeling."""
    identity = "\x1f".join(
        (
            subject_entity_id,
            predicate,
            object_entity_id,
            source_passage_id,
            str(assertion_start),
            str(assertion_end),
        )
    )
    return "fact-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:28]


def relabel_fact(fact: FactRecord, predicate: str) -> FactRecord:
    """Re-derive a fact under an adjudicator-chosen registry predicate.

    Endpoint validation re-runs the same invariants promotion enforces; the
    caller re-runs temporal reconciliation over the adjusted set.
    """
    if predicate not in PREDICATES:
        raise ValueError(f"unregistered predicate: {predicate}")
    if fact.subject_entity_id == fact.object_entity_id:
        raise ValueError("fact endpoints resolve to the same entity")
    return replace(
        fact,
        predicate=predicate,
        fact_id=derive_fact_id(
            fact.subject_entity_id,
            predicate,
            fact.object_entity_id,
            fact.source_passage_id,
            fact.assertion_start,
            fact.assertion_end,
        ),
        # supersession links were derived under the old predicate grouping
        superseded_by=None,
    )


def adjudication_multiplier(adjudicated: bool | int, score: float | None) -> float:
    """Edge-weight multiplier for a judged fact (edge-quality item 1).

    Unadjudicated rows and scores at/above the downweight band pass through
    at 1.0; mid-band scores multiply the fact's graph edge weight directly.
    """
    if not adjudicated or score is None:
        return 1.0
    value = min(1.0, max(0.0, float(score)))
    if value >= config.ADJUDICATION_DOWNWEIGHT_BELOW:
        return 1.0
    return value


def canonical_fact_text(fact: FactRecord) -> str:
    predicate = PREDICATES[fact.predicate]
    return f"{fact.subject} — {fact.predicate} ({predicate.description}) — {fact.object}"


def dedupe_note_facts(facts: Sequence[FactRecord]) -> tuple[FactRecord, ...]:
    """Collapse identical logical triples repeated by the SAME source note.

    The earliest assertion span survives and carries the group's max
    confidence. Cross-note duplicates stay separate provenance rows (§4.5);
    statuses never mix (an active and a candidate row of the same triple
    make different temporal claims).
    """
    groups: dict[tuple[str, str, str, str, str], list[FactRecord]] = {}
    for fact in facts:
        key = (
            fact.source_note_id,
            fact.subject_entity_id,
            fact.predicate,
            fact.object_entity_id,
            fact.status,
        )
        groups.setdefault(key, []).append(fact)
    collapsed: list[FactRecord] = []
    for group in groups.values():
        winner = min(
            group,
            key=lambda fact: (fact.assertion_start, fact.assertion_end, fact.fact_id),
        )
        confidence = max(fact.confidence for fact in group)
        if confidence != winner.confidence:
            winner = replace(winner, confidence=confidence)
        collapsed.append(winner)
    return tuple(sorted(collapsed, key=lambda fact: fact.fact_id))


def reconcile_temporal_facts(facts: Sequence[FactRecord]) -> tuple[FactRecord, ...]:
    facts = dedupe_note_facts(facts)
    rows = {fact.fact_id: fact for fact in facts}
    groups: dict[tuple[str, str], list[FactRecord]] = {}
    for fact in facts:
        groups.setdefault((fact.subject_entity_id, fact.predicate), []).append(fact)

    for (_, predicate), group in groups.items():
        spec = PREDICATES.get(predicate)
        active = [fact for fact in group if fact.status == "active"]
        if spec is None or not spec.exclusive_object or len(active) <= 1:
            continue
        objects = {fact.object_entity_id for fact in active}
        if len(objects) <= 1:
            continue
        latest_time = max(fact.observed_at for fact in active)
        latest = [fact for fact in active if fact.observed_at == latest_time]
        if len({fact.object_entity_id for fact in latest}) != 1:
            # a true observed_at tie between different objects is ambiguous
            for fact in active:
                rows[fact.fact_id] = replace(fact, status="conflicted")
            continue
        winner = min(latest, key=lambda fact: fact.fact_id)
        winner_start = winner.valid_from if winner.valid_from is not None else winner.observed_at
        # rows asserting the winning object from other sources are a support
        # group (§8.2), never superseded by their own object
        prior = [
            fact for fact in active if fact.object_entity_id != winner.object_entity_id
        ]
        if any(
            (fact.valid_from if fact.valid_from is not None else fact.observed_at)
            >= winner_start
            for fact in prior
        ):
            for fact in active:
                rows[fact.fact_id] = replace(fact, status="conflicted")
            continue
        for fact in prior:
            rows[fact.fact_id] = replace(
                fact,
                status="superseded",
                valid_to=(
                    winner_start
                    if fact.valid_to is None
                    else min(fact.valid_to, winner_start)
                ),
                invalidated_at=winner.observed_at,
                superseded_by=winner.fact_id,
            )
    return tuple(rows[fact_id] for fact_id in sorted(rows))


def facts_for_intent(
    facts: Sequence[FactRecord],
    intent: TemporalIntent | None,
    *,
    now: float,
) -> tuple[FactRecord, ...]:
    kind = intent.kind if intent is not None else "current"
    at = intent.at if intent is not None and intent.at is not None else now
    selected: list[FactRecord] = []
    for fact in facts:
        # candidates never surface; rejected rows exist for audit only
        if fact.status in {"candidate", "rejected"}:
            continue
        if kind == "historical":
            if fact.status in {"historical", "superseded"}:
                selected.append(fact)
            continue
        if kind == "any":
            selected.append(fact)
            continue
        if fact.status != "active":
            continue
        if fact.valid_from is not None and at < fact.valid_from:
            continue
        if fact.valid_to is not None and at >= fact.valid_to:
            continue
        selected.append(fact)
    return tuple(sorted(selected, key=lambda row: (row.predicate, row.fact_id)))
