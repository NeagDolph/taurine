from __future__ import annotations

from taurine_core.indexer.entities import EntityResolver, MentionInput
from taurine_core.schemas import ExtractedMention


def test_resolution_merges_case_corporate_suffix_and_acronym_without_rewriting_mentions() -> None:
    resolver = EntityResolver(total_notes=4)
    mentions = (
        _mention("m1", "n1", "Wink, Inc.", "organization", 0),
        _mention("m2", "n2", "wink", "organization", 5),
        _mention("m3", "n3", "Recommendation Engine", "project", 10),
        _mention("m4", "n4", "RE", "project", 15),
    )

    result = resolver.resolve_mentions(mentions)

    assert len(result.entities) == 2
    assert result.mention_to_entity["m1"] == result.mention_to_entity["m2"]
    assert result.mention_to_entity["m3"] == result.mention_to_entity["m4"]
    assert [row.mention.surface for row in result.mentions] == [
        "Wink, Inc.",
        "wink",
        "Recommendation Engine",
        "RE",
    ]
    project = next(row for row in result.entities if row.etype == "project")
    assert "RE" in project.aliases


def test_resolution_never_merges_different_digits_or_types() -> None:
    resolver = EntityResolver(total_notes=3, similarity=lambda _left, _right: 0.999)
    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "Project 12", "project", 0),
            _mention("m2", "n2", "Project 13", "project", 0),
            _mention("m3", "n3", "Project 12", "organization", 0),
        )
    )

    assert len(result.entities) == 3
    assert len(set(result.mention_to_entity.values())) == 3


def test_query_lookup_is_casefolded_and_reports_ambiguous_acronyms() -> None:
    resolver = EntityResolver(total_notes=3)
    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "Alpha Company", "organization", 0),
            _mention("m2", "n2", "Atlas Consulting", "organization", 0),
            _mention("m3", "n3", "Maya", "person", 0),
        )
    )

    maya = resolver.lookup_query_surface("maya", "person", "talk to maya")
    ambiguous = resolver.lookup_query_surface("AC", "organization", "AC update")
    missing = resolver.lookup_query_surface("unknown", None, "")

    assert maya.status == "resolved"
    assert len(maya.candidates) == 1
    assert maya.candidates[0].reason == "casefold_alias"
    assert ambiguous.status == "ambiguous"
    assert len(ambiguous.candidates) == 2
    assert missing.status == "missing"
    assert result.entities


def test_hub_and_date_entities_keep_provenance_but_default_non_traversable() -> None:
    resolver = EntityResolver(total_notes=10, owner_aliases={"Neil"})
    mentions = tuple(
        [_mention(f"owner-{i}", f"n{i}", "Neil", "person", i) for i in range(7)]
        + [_mention("date", "n8", "July 18", "date_or_deadline", 0)]
        + [_mention("specific", "n9", "Maya", "person", 0)]
    )

    result = resolver.resolve_mentions(mentions)

    owner = next(row for row in result.entities if row.canonical == "Neil")
    date = next(row for row in result.entities if row.etype == "date_or_deadline")
    maya = next(row for row in result.entities if row.canonical == "Maya")
    assert owner.traversal_eligible is False
    assert owner.traversal_reason == "owner_hub"
    assert date.traversal_eligible is False
    assert date.traversal_reason == "date_default_off"
    assert maya.traversal_eligible is True
    assert resolver.query_traversal_ids("Neil", "person", "Neil's notes") == (
        owner.entity_id,
    )


def test_entity_ids_and_resolution_are_deterministic_across_input_order() -> None:
    mentions = (
        _mention("m1", "n1", "Wink", "organization", 0),
        _mention("m2", "n2", "wink inc", "organization", 0),
        _mention("m3", "n3", "Maya", "person", 0),
    )
    first = EntityResolver(total_notes=3).resolve_mentions(mentions)
    second = EntityResolver(total_notes=3).resolve_mentions(tuple(reversed(mentions)))

    assert [(row.entity_id, row.canonical) for row in first.entities] == [
        (row.entity_id, row.canonical) for row in second.entities
    ]
    assert first.mention_to_entity == second.mention_to_entity


def _mention(
    mention_id: str,
    note_id: str,
    surface: str,
    etype: str,
    start: int,
) -> MentionInput:
    return MentionInput(
        mention_id=mention_id,
        note_id=note_id,
        passage_id=f"passage-{note_id}",
        mention=ExtractedMention(
            surface=surface,
            etype=etype,
            confidence=0.9,
            char_start=start,
            char_end=start + len(surface),
            extractor="fake",
            schema_id="schema-1",
        ),
    )
