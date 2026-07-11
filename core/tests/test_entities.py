from __future__ import annotations

import time

import pytest

from taurine_core import config
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


def test_acronym_never_bridges_two_expansions() -> None:
    """H1: 'JS' matches both John Smith and Jane Stevens, so it merges with
    neither, and union-find transitivity cannot fuse the two people."""
    resolver = EntityResolver(total_notes=3)

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "John Smith", "person", 0),
            _mention("m2", "n2", "Jane Stevens", "person", 0),
            _mention("m3", "n3", "JS", "person", 0),
        )
    )

    assert len(result.entities) == 3
    assert len(set(result.mention_to_entity.values())) == 3
    canonicals = {row.canonical for row in result.entities}
    assert canonicals == {"John Smith", "Jane Stevens", "JS"}
    # alias linking still records the ambiguity for query-time disambiguation
    assert resolver.lookup_query_surface("JS", "person", "").status == "ambiguous"


def test_unique_acronym_merges_and_canonical_prefers_expansion() -> None:
    resolver = EntityResolver(total_notes=3)

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "John Smith", "person", 0),
            _mention("m2", "n2", "JS", "person", 0),
            _mention("m3", "n3", "JS", "person", 5),
        )
    )

    assert len(result.entities) == 1
    # the expansion outranks the acronym even when the acronym is more frequent
    assert result.entities[0].canonical == "John Smith"
    assert "JS" in result.entities[0].aliases


def test_acronym_resolution_is_order_independent() -> None:
    mentions = (
        _mention("m1", "n1", "John Smith", "person", 0),
        _mention("m2", "n2", "Jane Stevens", "person", 0),
        _mention("m3", "n3", "JS", "person", 0),
        _mention("m4", "n4", "Recommendation Engine", "project", 0),
        _mention("m5", "n5", "RE", "project", 0),
    )

    first = EntityResolver(total_notes=5).resolve_mentions(mentions)
    second = EntityResolver(total_notes=5).resolve_mentions(tuple(reversed(mentions)))

    assert [(row.entity_id, row.canonical) for row in first.entities] == [
        (row.entity_id, row.canonical) for row in second.entities
    ]
    assert first.mention_to_entity == second.mention_to_entity
    assert first.mention_to_entity["m4"] == first.mention_to_entity["m5"]
    assert len({first.mention_to_entity[key] for key in ("m1", "m2", "m3")}) == 3


def test_embedding_similarity_is_candidate_generation_never_sufficient() -> None:
    resolver = EntityResolver(total_notes=4, similarity=lambda _left, _right: 0.99)

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "WinkLabs", "organization", 0),
            _mention("m2", "n2", "Wink Labs", "organization", 0),
            _mention("m3", "n3", "Contoso", "organization", 0),
        )
    )

    # lexical compatibility gates the embedding candidate (§7.1 stage 6)
    assert result.mention_to_entity["m1"] == result.mention_to_entity["m2"]
    assert result.mention_to_entity["m3"] != result.mention_to_entity["m1"]


def test_embedding_similarity_below_threshold_never_merges() -> None:
    resolver = EntityResolver(total_notes=4, similarity=lambda _left, _right: 0.90)

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "WinkLabs", "organization", 0),
            _mention("m2", "n2", "Wink Labs", "organization", 0),
        )
    )

    assert result.mention_to_entity["m1"] != result.mention_to_entity["m2"]


def test_person_embedding_merge_requires_last_name_and_containment() -> None:
    resolver = EntityResolver(total_notes=4, similarity=lambda _left, _right: 0.99)

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "Dr. Maya Chen", "person", 0),
            _mention("m2", "n2", "Maya Chen", "person", 0),
            _mention("m3", "n3", "John Smith", "person", 0),
            _mention("m4", "n4", "Jane Smith", "person", 0),
        )
    )

    assert result.mention_to_entity["m1"] == result.mention_to_entity["m2"]
    # §7.2: two distinct full names never merge on a shared last name alone
    assert result.mention_to_entity["m3"] != result.mention_to_entity["m4"]


def test_concept_stoplist_entities_default_non_traversable() -> None:
    resolver = EntityResolver(total_notes=5)

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "Ideas", "concept", 0),
            _mention("m2", "n2", "Kubernetes", "concept", 0),
        )
    )

    ideas = next(row for row in result.entities if row.canonical == "Ideas")
    kubernetes = next(row for row in result.entities if row.canonical == "Kubernetes")
    assert ideas.traversal_eligible is False
    assert ideas.traversal_reason == "concept_stoplist"
    assert kubernetes.traversal_eligible is True


def test_document_frequency_hub_reason_is_reported() -> None:
    resolver = EntityResolver(total_notes=10)
    mentions = tuple(
        _mention(f"m{i}", f"n{i}", "Acme", "organization", 0) for i in range(6)
    ) + (_mention("rare", "n9", "Contoso", "organization", 0),)

    result = resolver.resolve_mentions(mentions)

    acme = next(row for row in result.entities if row.canonical == "Acme")
    contoso = next(row for row in result.entities if row.canonical == "Contoso")
    assert acme.traversal_eligible is False
    assert acme.traversal_reason == "document_frequency_hub"
    assert contoso.traversal_eligible is True


@pytest.mark.slow
def test_resolution_is_near_linear_for_disjoint_surfaces() -> None:
    """M3: 5000 mentions resolve well under the 60 s finalization budget."""
    mentions = tuple(
        _mention(f"m{i:05d}", f"n{i % 500}", f"Person {i} Surname{i}", "person", 0)
        for i in range(5000)
    )

    started = time.perf_counter()
    result = EntityResolver(total_notes=500).resolve_mentions(mentions)
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert len(result.entities) == 5000


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
    assert owner.traversal_reason == "owner_alias_hub"
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


# ── owner auto-detection (edge-quality item 3) ─────────────────────────────

_SOURCES = ("imessage", "apple_notes", "gmail", "obsidian")


def _owner_corpus() -> tuple[tuple[MentionInput, ...], dict[str, str]]:
    """Modeled on the real corpus: 'Neil' df=82 across 4 sources,
    'Neil Agrawal' df=28, runner-up 'Ernie' df=23, 100 notes total."""
    mentions: list[MentionInput] = []
    for index in range(82):
        mentions.append(_mention(f"neil-{index:03d}", f"n{index:03d}", "Neil", "person", 0))
    for index in range(28):
        mentions.append(
            _mention(f"agrawal-{index:03d}", f"n{index:03d}", "Neil Agrawal", "person", 10)
        )
    for index in range(23):
        mentions.append(_mention(f"ernie-{index:03d}", f"n{index:03d}", "Ernie", "person", 30))
    note_sources = {
        f"n{index:03d}": _SOURCES[index % len(_SOURCES)] for index in range(100)
    }
    return tuple(mentions), note_sources


def test_owner_is_autodetected_from_frequency_dominance_and_source_spread() -> None:
    mentions, note_sources = _owner_corpus()
    resolver = EntityResolver(total_notes=100, note_sources=note_sources)

    result = resolver.resolve_mentions(mentions)

    by_canonical = {row.canonical: row for row in result.entities}
    neil = by_canonical["Neil"]
    agrawal = by_canonical["Neil Agrawal"]
    ernie = by_canonical["Ernie"]
    assert neil.df == 82 and agrawal.df == 28 and ernie.df == 23
    assert neil.traversal_eligible is False
    assert neil.traversal_reason == "owner_alias_hub"
    # token-superset of the detected owner alias is flagged with the same reason
    assert agrawal.traversal_eligible is False
    assert agrawal.traversal_reason == "owner_alias_hub"
    assert ernie.traversal_eligible is True
    assert ernie.traversal_reason == "eligible"


def test_owner_autodetection_requires_three_sources() -> None:
    mentions, note_sources = _owner_corpus()
    single_source = {note_id: "imessage" for note_id in note_sources}
    resolver = EntityResolver(total_notes=100, note_sources=single_source)

    result = resolver.resolve_mentions(mentions)

    by_canonical = {row.canonical: row for row in result.entities}
    assert by_canonical["Neil"].traversal_reason == "document_frequency_hub"
    assert by_canonical["Neil Agrawal"].traversal_eligible is True


def test_owner_autodetection_requires_double_the_runner_up() -> None:
    mentions: list[MentionInput] = []
    for index in range(30):
        mentions.append(_mention(f"neil-{index:03d}", f"n{index:03d}", "Neil", "person", 0))
    for index in range(20):
        mentions.append(_mention(f"ernie-{index:03d}", f"n{index:03d}", "Ernie", "person", 10))
    note_sources = {f"n{index:03d}": _SOURCES[index % 4] for index in range(100)}
    resolver = EntityResolver(total_notes=100, note_sources=note_sources)

    result = resolver.resolve_mentions(tuple(mentions))

    by_canonical = {row.canonical: row for row in result.entities}
    # 30 < 2 × 20: no detection, both stay eligible (df below the hub cap)
    assert by_canonical["Neil"].traversal_eligible is True
    assert by_canonical["Ernie"].traversal_eligible is True


def test_owner_name_variants_cannot_block_autodetection() -> None:
    """The runner-up must be a genuinely different person: 'Neil Agrawal'
    (token superset of the leader) is skipped when computing dominance."""
    mentions: list[MentionInput] = []
    for index in range(30):
        mentions.append(_mention(f"neil-{index:03d}", f"n{index:03d}", "Neil", "person", 0))
    for index in range(20):
        mentions.append(
            _mention(f"agrawal-{index:03d}", f"n{index:03d}", "Neil Agrawal", "person", 10)
        )
    for index in range(10):
        mentions.append(_mention(f"ernie-{index:03d}", f"n{index:03d}", "Ernie", "person", 30))
    note_sources = {f"n{index:03d}": _SOURCES[index % 4] for index in range(100)}
    resolver = EntityResolver(total_notes=100, note_sources=note_sources)

    result = resolver.resolve_mentions(tuple(mentions))

    by_canonical = {row.canonical: row for row in result.entities}
    # 30 < 2 × 28 would fail, but the variant is excluded → runner-up is Ernie
    assert by_canonical["Neil"].traversal_reason == "owner_alias_hub"
    assert by_canonical["Neil Agrawal"].traversal_reason == "owner_alias_hub"
    assert by_canonical["Ernie"].traversal_eligible is True


def test_owner_autodetection_disabled_without_note_sources() -> None:
    mentions, _ = _owner_corpus()
    resolver = EntityResolver(total_notes=100)

    result = resolver.resolve_mentions(mentions)

    by_canonical = {row.canonical: row for row in result.entities}
    assert by_canonical["Neil"].traversal_reason == "document_frequency_hub"
    assert by_canonical["Neil Agrawal"].traversal_eligible is True


def test_explicit_owner_aliases_always_win_over_autodetection() -> None:
    mentions, note_sources = _owner_corpus()
    extra = tuple(
        _mention(f"maya-{index}", f"n{index:03d}", "Maya Chen", "person", 50)
        for index in range(2)
    )
    resolver = EntityResolver(
        total_notes=100,
        owner_aliases={"maya chen"},
        note_sources=note_sources,
    )

    result = resolver.resolve_mentions(mentions + extra)

    by_canonical = {row.canonical: row for row in result.entities}
    # the configured owner is flagged even at low df, and its token-subset too
    assert by_canonical["Maya Chen"].traversal_reason == "owner_alias_hub"
    # auto-detection is skipped entirely: 'Neil' keeps the frequency reason
    assert by_canonical["Neil"].traversal_reason == "document_frequency_hub"
    assert by_canonical["Neil Agrawal"].traversal_eligible is True


def test_configured_owner_alias_flags_token_subset_and_superset_surfaces() -> None:
    resolver = EntityResolver(total_notes=10, owner_aliases={"neil"})

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "Neil", "person", 0),
            _mention("m2", "n2", "Neil Agrawal", "person", 0),
            _mention("m3", "n3", "Ernie", "person", 0),
        )
    )

    by_canonical = {row.canonical: row for row in result.entities}
    assert by_canonical["Neil"].traversal_reason == "owner_alias_hub"
    assert by_canonical["Neil Agrawal"].traversal_reason == "owner_alias_hub"
    assert by_canonical["Ernie"].traversal_eligible is True


# ── gray-zone emission + forced merges (edge-quality item 6) ────────────────


def test_embedding_gray_zone_pairs_are_emitted_not_merged() -> None:
    """Similarity just under the merge floor is surfaced for the judge."""
    resolver = EntityResolver(
        total_notes=4,
        similarity=lambda left, right: 0.94,
        similarity_threshold=0.95,
    )

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "Launch Plan", "project", 0),
            _mention("m2", "n2", "Launch Planning", "project", 0),
        )
    )

    assert len(result.entities) == 2
    (pair,) = [
        row for row in result.gray_zone_pairs if row.reason == "embedding_gray_zone"
    ]
    assert {pair.left_surface, pair.right_surface} == {"Launch Plan", "Launch Planning"}
    assert pair.etype == "project"


def test_person_guard_blocked_merge_lands_in_the_gray_zone() -> None:
    """'Neil'/'Neil Agrawal' with embeddings above the floor: the person guard
    blocks the deterministic merge, so the pair goes to the judge instead."""
    resolver = EntityResolver(
        total_notes=4,
        similarity=lambda left, right: 0.97,
        similarity_threshold=0.95,
    )

    result = resolver.resolve_mentions(
        (
            _mention("m1", "n1", "Neil", "person", 0),
            _mention("m2", "n2", "Neil Agrawal", "person", 0),
        )
    )

    assert len(result.entities) == 2
    assert any(
        {row.left_surface, row.right_surface} == {"Neil", "Neil Agrawal"}
        for row in result.gray_zone_pairs
    )


def test_token_subset_pairs_are_emitted_without_embeddings() -> None:
    result = EntityResolver(total_notes=4).resolve_mentions(
        (
            _mention("m1", "n1", "Neil", "person", 0),
            _mention("m2", "n2", "Neil Agrawal", "person", 0),
            _mention("m3", "n3", "Ernie", "person", 0),
        )
    )

    (pair,) = [row for row in result.gray_zone_pairs if row.reason == "token_overlap"]
    assert {pair.left_surface, pair.right_surface} == {"Neil", "Neil Agrawal"}
    # same token counts never emit ("Neil" vs "Ernie" share nothing anyway)
    assert len(result.gray_zone_pairs) == 1


def test_ambiguous_acronym_pairs_are_emitted_for_the_judge() -> None:
    result = EntityResolver(total_notes=3).resolve_mentions(
        (
            _mention("m1", "n1", "John Smith", "person", 0),
            _mention("m2", "n2", "Jane Stevens", "person", 0),
            _mention("m3", "n3", "JS", "person", 0),
        )
    )

    ambiguous = [row for row in result.gray_zone_pairs if row.reason == "acronym_ambiguous"]
    surfaces = {
        frozenset({row.left_surface, row.right_surface}) for row in ambiguous
    }
    assert surfaces == {
        frozenset({"JS", "John Smith"}),
        frozenset({"JS", "Jane Stevens"}),
    }


def test_gray_zone_pairs_are_capped_by_combined_df(monkeypatch) -> None:
    monkeypatch.setattr(config, "MERGE_GRAY_ZONE_CAP", 1)
    result = EntityResolver(total_notes=8).resolve_mentions(
        (
            _mention("m1", "n1", "Neil", "person", 0),
            _mention("m2", "n2", "Neil", "person", 0),
            _mention("m3", "n3", "Neil Agrawal", "person", 0),
            _mention("m4", "n4", "Maya", "person", 0),
            _mention("m5", "n5", "Maya Chen", "person", 0),
        )
    )

    (pair,) = result.gray_zone_pairs
    # Neil (df=2) + Neil Agrawal (df=1) outranks Maya (1) + Maya Chen (1)
    assert {pair.left_surface, pair.right_surface} == {"Neil", "Neil Agrawal"}
    assert pair.combined_df == 3


def test_forced_merges_union_the_pair_before_insertion() -> None:
    resolver = EntityResolver(total_notes=4)
    mentions = (
        _mention("m1", "n1", "Neil", "person", 0),
        _mention("m2", "n2", "Neil Agrawal", "person", 0),
    )

    baseline = resolver.resolve_mentions(mentions)
    merged = EntityResolver(total_notes=4).resolve_mentions(
        mentions, forced_merges=(("person", "neil", "neil agrawal"),)
    )

    assert len(baseline.entities) == 2
    assert len(merged.entities) == 1
    assert merged.mention_to_entity["m1"] == merged.mention_to_entity["m2"]
    entity = merged.entities[0]
    assert {"Neil", "Neil Agrawal"} <= set(entity.aliases)


def test_forced_merges_never_cross_the_digit_guard() -> None:
    result = EntityResolver(total_notes=4).resolve_mentions(
        (
            _mention("m1", "n1", "Launch 2", "project", 0),
            _mention("m2", "n2", "Launch", "project", 0),
        ),
        forced_merges=(("project", "launch 2", "launch"),),
    )

    assert len(result.entities) == 2


def test_gray_zone_emission_is_deterministic_across_input_order() -> None:
    mentions = (
        _mention("m1", "n1", "Neil", "person", 0),
        _mention("m2", "n2", "Neil Agrawal", "person", 0),
        _mention("m3", "n3", "Maya", "person", 0),
        _mention("m4", "n4", "Maya Chen", "person", 0),
    )

    forward = EntityResolver(total_notes=4).resolve_mentions(mentions)
    backward = EntityResolver(total_notes=4).resolve_mentions(tuple(reversed(mentions)))

    assert forward.gray_zone_pairs == backward.gray_zone_pairs
