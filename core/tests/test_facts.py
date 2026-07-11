from __future__ import annotations

from dataclasses import replace

from taurine_core.indexer.facts import (
    PREDICATES,
    FactRecord,
    canonical_fact_text,
    facts_for_intent,
    promote_fact,
    reconcile_temporal_facts,
)
from taurine_core.retrieval.types import TemporalIntent
from taurine_core.schemas import ExtractedRelation


def test_predicate_registry_is_complete_and_directional() -> None:
    assert len(PREDICATES) == 18
    assert PREDICATES["works_at"].directional is True
    assert PREDICATES["works_at"].inverse == "employs"
    assert PREDICATES["works_at"].reverse_traversal_weight == 0.35
    assert PREDICATES["related_to"].directional is False
    assert PREDICATES["scheduled_for"].exclusive_object is True
    # a person has one current employer (edge-quality spec item 2) but may
    # legitimately hold many memberships/possessions
    assert PREDICATES["works_at"].exclusive_object is True
    assert PREDICATES["member_of"].exclusive_object is False
    assert PREDICATES["owns_or_has"].exclusive_object is False


def test_promoted_fact_preserves_direction_offsets_time_and_deterministic_text() -> None:
    candidate = ExtractedRelation(
        head="Maya",
        predicate="responsible_for",
        tail="launch checklist",
        confidence=0.91,
        subject_start=0,
        subject_end=4,
        object_start=14,
        object_end=30,
        assertion_start=0,
        assertion_end=31,
        extraction_method="gliner2_batch",
        schema_id="schema-1",
        status="active",
    )

    first = promote_fact(
        candidate,
        subject_entity_id="entity-maya",
        object_entity_id="entity-launch",
        source_passage_id="passage-1",
        source_note_id="note-1",
        observed_at=10.0,
        build_id="build-1",
    )
    second = promote_fact(
        candidate,
        subject_entity_id="entity-maya",
        object_entity_id="entity-launch",
        source_passage_id="passage-1",
        source_note_id="note-1",
        observed_at=10.0,
        build_id="build-1",
    )

    assert first == second
    assert first.status == "active"
    assert first.subject_start == 0
    assert first.object_start == 14
    assert first.observed_at == 10.0
    assert canonical_fact_text(first) == (
        "Maya — responsible_for (is responsible for) — launch checklist"
    )


def test_exclusive_temporal_fact_is_superseded_without_deleting_history() -> None:
    earlier = _fact("f1", "scheduled_for", "date-1", 10.0, valid_from=10.0)
    later = _fact("f2", "scheduled_for", "date-2", 20.0, valid_from=20.0)

    rows = {row.fact_id: row for row in reconcile_temporal_facts((earlier, later))}

    assert rows["f1"].status == "superseded"
    assert rows["f1"].superseded_by == "f2"
    assert rows["f1"].valid_to == 20.0
    assert rows["f2"].status == "active"
    assert rows["f2"].superseded_by is None


def test_same_time_exclusive_contradiction_remains_explicit_conflict() -> None:
    left = _fact("f1", "scheduled_for", "date-1", 20.0)
    right = _fact("f2", "scheduled_for", "date-2", 20.0)

    rows = reconcile_temporal_facts((left, right))

    assert {row.status for row in rows} == {"conflicted"}
    assert all(row.superseded_by is None for row in rows)


def test_recent_source_describing_older_validity_does_not_supersede_current_fact() -> None:
    current = _fact("f1", "scheduled_for", "date-current", 10.0, valid_from=10.0)
    recent_history = _fact("f2", "scheduled_for", "date-old", 20.0, valid_from=5.0)

    rows = reconcile_temporal_facts((current, recent_history))

    assert {row.status for row in rows} == {"conflicted"}


def test_nonexclusive_facts_coexist_and_temporal_intent_filters_status() -> None:
    one = _fact("f1", "works_with", "person-1", 10.0)
    two = _fact("f2", "works_with", "person-2", 20.0)
    old = replace(_fact("f3", "scheduled_for", "date-old", 5.0), status="superseded")
    reconciled = reconcile_temporal_facts((one, two, old))

    assert {row.status for row in reconciled if row.predicate == "works_with"} == {"active"}
    current = facts_for_intent(reconciled, TemporalIntent(kind="current"), now=25.0)
    historical = facts_for_intent(reconciled, TemporalIntent(kind="historical"), now=25.0)
    assert {row.fact_id for row in current} == {"f1", "f2"}
    assert {row.fact_id for row in historical} == {"f3"}


def test_eight_employers_regression_exactly_one_active() -> None:
    # modeled on the real failure: resume snapshots spanning 2022-2026 plus a
    # 2026 intro note left 8 simultaneously-active employers for one subject
    employers = tuple(
        _fact(f"f{index}", "works_at", f"org-{index}", float(index) * 100.0)
        for index in range(1, 9)
    )

    rows = {row.fact_id: row for row in reconcile_temporal_facts(employers)}

    active = sorted(row.fact_id for row in rows.values() if row.status == "active")
    assert active == ["f8"]
    for index in range(1, 8):
        row = rows[f"f{index}"]
        assert row.status == "superseded"
        assert row.superseded_by == "f8"
        assert row.valid_to == 800.0
        assert row.invalidated_at == 800.0


def test_same_object_latest_tie_is_support_not_conflict() -> None:
    # two notes asserting the SAME current employer at the same observed time
    # are a support group (§4.5), not a contradiction
    old = _fact("f1", "works_at", "org-old", 10.0)
    dup_a = _fact("f2", "works_at", "org-new", 20.0)
    dup_b = _fact("f3", "works_at", "org-new", 20.0)

    rows = {row.fact_id: row for row in reconcile_temporal_facts((old, dup_a, dup_b))}

    assert rows["f1"].status == "superseded"
    assert rows["f1"].superseded_by == "f2"  # deterministic min-fact-id winner
    assert rows["f2"].status == "active"
    assert rows["f3"].status == "active"


def test_different_object_observed_at_tie_is_conflicted() -> None:
    old = _fact("f1", "works_at", "org-old", 10.0)
    left = _fact("f2", "works_at", "org-a", 20.0)
    right = _fact("f3", "works_at", "org-b", 20.0)

    rows = {row.fact_id: row for row in reconcile_temporal_facts((old, left, right))}

    assert {row.status for row in rows.values()} == {"conflicted"}


def test_same_note_duplicate_triples_collapse_to_one_row() -> None:
    later_span = _fact(
        "f1", "works_at", "org-1", 10.0,
        source_note_id="note-x", assertion=(40, 60), confidence=0.9,
    )
    earlier_span = _fact(
        "f2", "works_at", "org-1", 10.0,
        source_note_id="note-x", assertion=(0, 20), confidence=0.6,
    )
    cross_note = _fact(
        "f3", "works_at", "org-1", 10.0,
        source_note_id="note-y", assertion=(0, 20), confidence=0.5,
    )

    rows = {row.fact_id: row for row in reconcile_temporal_facts((later_span, earlier_span, cross_note))}

    # earliest assertion span survives, carrying the group's max confidence;
    # the cross-note duplicate stays a separate provenance row
    assert set(rows) == {"f2", "f3"}
    assert rows["f2"].confidence == 0.9
    assert rows["f2"].assertion_start == 0
    assert rows["f3"].confidence == 0.5


def test_same_note_dedupe_never_mixes_statuses() -> None:
    active = _fact(
        "f1", "works_at", "org-1", 10.0,
        source_note_id="note-x", assertion=(0, 20), confidence=0.9,
    )
    candidate = replace(
        _fact(
            "f2", "works_at", "org-1", 10.0,
            source_note_id="note-x", assertion=(30, 50), confidence=0.4,
        ),
        status="candidate",
    )

    rows = {row.fact_id: row for row in reconcile_temporal_facts((active, candidate))}

    assert set(rows) == {"f1", "f2"}
    assert rows["f1"].status == "active"
    assert rows["f2"].status == "candidate"


def _fact(
    fact_id: str,
    predicate: str,
    object_entity_id: str,
    observed_at: float,
    *,
    valid_from: float | None = None,
    source_note_id: str | None = None,
    assertion: tuple[int, int] = (0, 10),
    confidence: float = 0.9,
) -> FactRecord:
    return FactRecord(
        fact_id=fact_id,
        subject_entity_id="subject-1",
        subject="Launch",
        predicate=predicate,
        object_entity_id=object_entity_id,
        object=object_entity_id,
        source_passage_id=f"passage-{fact_id}",
        source_note_id=source_note_id if source_note_id is not None else f"note-{fact_id}",
        assertion_start=assertion[0],
        assertion_end=assertion[1],
        subject_start=0,
        subject_end=3,
        object_start=5,
        object_end=10,
        confidence=confidence,
        extraction_method="fake",
        schema_id="schema-1",
        observed_at=observed_at,
        valid_from=valid_from,
        valid_to=None,
        invalidated_at=None,
        superseded_by=None,
        status="active",
        build_id="build-1",
    )
