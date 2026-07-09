from __future__ import annotations

from taurine_core.indexer.relation_completion import (
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
