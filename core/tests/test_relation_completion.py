from __future__ import annotations

from taurine_core.indexer.relation_completion import (
    CompletionAlias,
    CompletionEntity,
    CompletionNote,
    CompletionPassage,
    parse_completion_response,
    select_completion_notes,
)


def _note(
    note_id: str,
    *,
    value: float,
    entities: int,
    relations: int,
    schema_gap: bool = False,
) -> CompletionNote:
    return CompletionNote(
        note_id=note_id,
        value_score=value,
        observed_at=0.0,
        entities=tuple(
            CompletionEntity(f"entity-{index}", f"Entity {index}", index, index + 1)
            for index in range(entities)
        ),
        passages=(),
        active_relation_count=relations,
        schema_gap=schema_gap,
    )


def test_no_relation_gate_requires_median_value() -> None:
    notes = [
        _note(f"note-{i:02d}", value=1.0 - i * 0.1, entities=2, relations=0)
        for i in range(10)
    ]

    selected = {note.note_id for note in select_completion_notes(notes)}

    assert "note-09" not in selected
    assert "note-08" not in selected
    assert "note-03" in selected


def test_schema_gap_ignores_value_threshold() -> None:
    notes = [
        _note(f"note-{i:02d}", value=1.0 - i * 0.1, entities=2, relations=0)
        for i in range(10)
    ]
    notes.append(
        _note("note-gap", value=0.0, entities=0, relations=5, schema_gap=True)
    )

    selected = {note.note_id for note in select_completion_notes(notes)}

    assert "note-gap" in selected


def test_completion_selector_includes_value_top_floor_sparse_and_schema_gap() -> None:
    notes = tuple(
        CompletionNote(
            note_id=f"n{i}",
            value_score=5.5 if i == 9 else float(10 - i),
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


def test_tool_hop_notes_join_selection_capped(monkeypatch) -> None:
    from taurine_core import config
    from taurine_core.indexer.relation_completion import (
        CompletionNote,
        select_completion_notes,
    )

    monkeypatch.setattr(config, "COMPLETION_TOOL_HOP_CAP", 2)

    def note(note_id: str, value: float, *, tool_hop: bool, relations: int = 1):
        return CompletionNote(
            note_id=note_id,
            value_score=value,
            observed_at=0.0,
            entities=(),
            passages=(),
            active_relation_count=relations,
            schema_gap=False,
            tool_hop=tool_hop,
        )

    filler = [note(f"filler-{i:02d}", 0.9 - i * 0.001, tool_hop=False) for i in range(40)]
    notes = filler + [
        note("hop-a", 0.5, tool_hop=True),
        note("hop-b", 0.4, tool_hop=True),
        note("hop-c", 0.3, tool_hop=True),  # over the cap
    ]
    selected = {row.note_id for row in select_completion_notes(notes)}
    # value gate takes the top 15% of 43 (~7 filler notes); the hop lane adds
    # exactly COMPLETION_TOOL_HOP_CAP=2 more by value order
    assert {"hop-a", "hop-b"} <= selected
    assert "hop-c" not in selected
