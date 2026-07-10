from __future__ import annotations

from taurine_core.indexer.relation_completion import (
    CompletionAlias,
    CompletionEntity,
    CompletionNote,
    CompletionPassage,
    parse_completion_response,
    select_completion_notes,
)


def test_completion_selector_includes_value_top_floor_sparse_and_schema_gap() -> None:
    notes = tuple(
        CompletionNote(
            note_id=f"n{i}",
            value_score=float(10 - i),
            observed_at=float(i),
            entities=(
                (
                    CompletionEntity("a", "Alpha", 0, 5),
                    CompletionEntity("b", "Beta", 6, 10),
                )
                if i == 9
                else ()
            ),
            passages=(),
            active_relation_count=0 if i == 9 else 1,
            schema_gap=i == 8,
        )
        for i in range(10)
    )

    selected = select_completion_notes(notes)

    assert {row.note_id for row in selected} == {"n0", "n1", "n2", "n3", "n4", "n8", "n9"}
    assert [row.note_id for row in selected[:5]] == ["n0", "n1", "n2", "n3", "n4"]


def test_completion_parser_accepts_registered_fact_and_quarantines_open_predicate() -> None:
    text = "Maya reviewed Launch because vendor access was blocked."
    note = CompletionNote(
        note_id="n1",
        value_score=1.0,
        observed_at=10.0,
        entities=(
            CompletionEntity("maya", "Maya", 0, 4),
            CompletionEntity("launch", "Launch", 14, 20),
            CompletionEntity("vendor", "vendor access", 29, 42),
        ),
        passages=(CompletionPassage("p1", text, 0),),
        active_relation_count=0,
        schema_gap=True,
    )
    payload = {
        "facts": [
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "launch",
                "predicate": "blocked_by",
                "object_entity_id": "vendor",
                "assertion_start": 0,
                "assertion_end": len(text),
                "confidence": 0.88,
            },
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "maya",
                "predicate": "reviewed_for",
                "predicate_description": "a person reviewed a project",
                "object_entity_id": "launch",
                "assertion_start": 0,
                "assertion_end": 20,
                "confidence": 0.8,
            },
        ]
    }

    result = parse_completion_response(payload, (note,), build_id="build-1")

    assert len(result.facts) == 1
    assert result.facts[0].predicate == "blocked_by"
    assert result.facts[0].source_passage_id == "p1"
    assert result.facts[0].subject_start == 14
    assert result.facts[0].object_start == 29
    assert len(result.proposals) == 1
    assert result.proposals[0].proposed_predicate == "reviewed_for"
    assert result.proposals[0].source_passage_id == "p1"


def test_completion_accepts_any_grounded_mention_of_multi_mention_entity() -> None:
    """H8: an entity mentioned at offsets 10 and 500 must ground at either."""
    text = "well, ok. Maya" + " pad" * 120 + ". Maya runs Launch."
    first = text.index("Maya")
    second = text.rindex("Maya")
    launch = text.rindex("Launch")
    note = CompletionNote(
        note_id="n1",
        value_score=1.0,
        observed_at=10.0,
        entities=(
            # deliberately listed with the grounded mention first: a last-wins
            # entity map would keep only the ungrounded offset-10 mention
            CompletionEntity("maya", "Maya", second, second + 4),
            CompletionEntity("maya", "Maya", first, first + 4),
            CompletionEntity("launch", "Launch", launch, launch + 6),
        ),
        passages=(CompletionPassage("p1", text, 0),),
        active_relation_count=0,
        schema_gap=False,
    )
    payload = {
        "facts": [
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "maya",
                "predicate": "responsible_for",
                "object_entity_id": "launch",
                "assertion_start": second,
                "assertion_end": len(text),
                "confidence": 0.9,
            }
        ]
    }

    result = parse_completion_response(payload, (note,), build_id="build-1")

    assert result.rejected == 0
    assert len(result.facts) == 1
    assert result.facts[0].subject_start == second
    assert result.facts[0].object_start == launch


def test_completion_picks_tightest_mention_pair_inside_wide_assertion() -> None:
    text = "Maya intro" + " pad" * 60 + ". Maya runs Launch now."
    first = text.index("Maya")
    second = text.rindex("Maya")
    launch = text.rindex("Launch")
    note = CompletionNote(
        note_id="n1",
        value_score=1.0,
        observed_at=10.0,
        entities=(
            CompletionEntity("maya", "Maya", first, first + 4),
            CompletionEntity("maya", "Maya", second, second + 4),
            CompletionEntity("launch", "Launch", launch, launch + 6),
        ),
        passages=(CompletionPassage("p1", text, 0),),
        active_relation_count=0,
        schema_gap=False,
    )
    payload = {
        "facts": [
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "maya",
                "predicate": "responsible_for",
                "object_entity_id": "launch",
                "assertion_start": 0,
                "assertion_end": len(text),
                "confidence": 0.9,
            }
        ]
    }

    result = parse_completion_response(payload, (note,), build_id="build-1")

    assert len(result.facts) == 1
    assert result.facts[0].subject_start == second


def test_completion_aliases_are_validated_verbatim() -> None:
    text = "Maya (M. Chen) runs Launch."
    note = CompletionNote(
        note_id="n1",
        value_score=1.0,
        observed_at=10.0,
        entities=(
            CompletionEntity("maya", "Maya", 0, 4),
            CompletionEntity("launch", "Launch", 20, 26),
        ),
        passages=(CompletionPassage("p1", text, 0),),
        active_relation_count=0,
        schema_gap=False,
    )
    payload = {
        "facts": [],
        "aliases": [
            {"note_id": "n1", "entity_id": "maya", "alias": "M. Chen"},
            {"note_id": "n1", "entity_id": "maya", "alias": "Maya C."},
            {"note_id": "n1", "entity_id": "ghost", "alias": "Maya"},
        ],
    }

    result = parse_completion_response(payload, (note,), build_id="build-1")

    assert result.aliases == (
        CompletionAlias(entity_id="maya", alias="M. Chen", origin="completion"),
    )
    assert result.rejected == 2


def test_open_predicates_normalize_to_registered_before_quarantine() -> None:
    """L21c: synonyms of registered predicates become normal facts."""
    text = "Maya works for Acme and owns Launch."
    note = CompletionNote(
        note_id="n1",
        value_score=1.0,
        observed_at=10.0,
        entities=(
            CompletionEntity("maya", "Maya", 0, 4),
            CompletionEntity("acme", "Acme", 15, 19),
            CompletionEntity("launch", "Launch", 29, 35),
        ),
        passages=(CompletionPassage("p1", text, 0),),
        active_relation_count=0,
        schema_gap=False,
    )
    payload = {
        "facts": [
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "maya",
                "predicate": "Works For",
                "object_entity_id": "acme",
                "assertion_start": 0,
                "assertion_end": len(text),
                "confidence": 0.9,
            },
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "maya",
                "predicate": "owns",
                "object_entity_id": "launch",
                "assertion_start": 0,
                "assertion_end": len(text),
                "confidence": 0.85,
            },
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "maya",
                "predicate": "mentored",
                "predicate_description": "a person mentored another party",
                "object_entity_id": "launch",
                "assertion_start": 0,
                "assertion_end": len(text),
                "confidence": 0.8,
            },
        ]
    }

    result = parse_completion_response(payload, (note,), build_id="build-1")

    assert sorted(fact.predicate for fact in result.facts) == ["owns_or_has", "works_at"]
    assert [row.proposed_predicate for row in result.proposals] == ["mentored"]
    assert result.rejected == 0


def test_completion_parser_rejects_offsets_outside_source_passage() -> None:
    note = CompletionNote(
        note_id="n1",
        value_score=1.0,
        observed_at=10.0,
        entities=(CompletionEntity("maya", "Maya", 0, 4), CompletionEntity("launch", "Launch", 10, 16)),
        passages=(CompletionPassage("p1", "Maya owns Launch.", 0),),
        active_relation_count=0,
        schema_gap=False,
    )
    payload = {
        "facts": [
            {
                "note_id": "n1",
                "passage_id": "p1",
                "subject_entity_id": "maya",
                "predicate": "responsible_for",
                "object_entity_id": "launch",
                "assertion_start": 0,
                "assertion_end": 999,
                "confidence": 0.9,
            }
        ]
    }

    result = parse_completion_response(payload, (note,), build_id="build-1")

    assert result.facts == ()
    assert result.proposals == ()
    assert result.rejected == 1
