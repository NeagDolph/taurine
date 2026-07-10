from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies, BuildError, main
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.retrieval.snapshot import load_index_snapshot
from taurine_core.schemas import ExtractedMention, ExtractedRelation, Extraction


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def embedder(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        row = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) + 1
        row /= np.linalg.norm(row)
        rows.append(row)
    return np.vstack(rows)


def test_clean_build_is_atomic_deterministic_and_order_independent(tmp_path: Path) -> None:
    notes = [
        make_note("b", "Beta", "Maya owns Launch."),
        make_note("a", "Alpha", "Launch is scheduled for July 18."),
    ]
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"

    first = BuildCoordinator(deps).build(notes, first_path, replace_derived=True)
    second = BuildCoordinator(deps).build(list(reversed(notes)), second_path, replace_derived=True)

    assert first.graph_hash == second.graph_hash
    assert first.source_snapshot == second.source_snapshot
    assert first.invariant_counts["notes"] == 2
    assert first.invariant_counts["passages"] == 2
    assert first.entity_count == 0
    assert first.fact_count == 0
    assert first.extraction_failure_count == 2
    assert first.failed_notes == (
        ("a", "extractor unavailable"),
        ("b", "extractor unavailable"),
    )
    assert first.component_count >= 1
    assert first.dangling_node_count >= 0
    assert set(first.stage_timings_ms) == set(first.stages)
    assert all(value >= 0.0 for value in first.stage_timings_ms.values())
    assert first.calibration_hash == "sha256:initial-conservative-2026-07-09"
    assert first.release_ready is False
    assert "uncalibrated_retrieval_head:live:dense" in first.release_blockers
    assert first.stages == (
        "source",
        "passages",
        "embeddings",
        "extraction",
        "entities",
        "relation_completion",
        "facts",
        "base_graph",
        "projection",
        "ranks",
        "fact_embeddings",
        "validation",
        "activation",
    )
    store = KnowledgeStore(first_path, read_only=True)
    try:
        assert store.counts()["notes"] == 2
        assert store.counts()["passages"] == 2
        assert store.build(first.build_id).status == "complete"  # type: ignore[union-attr]
        assert store.validate_invariants().ok
    finally:
        store.close()
    snapshot = load_index_snapshot(first_path)
    assert snapshot.passage_matrix.shape == (2, 16)
    assert snapshot.fact_matrix.shape == (0, 16)


def test_empty_build_bootstraps_a_loadable_snapshot(tmp_path: Path) -> None:
    """Startup creates an empty index when none exists — no sources are read."""
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    target = tmp_path / "knowledge.db"

    report = BuildCoordinator(deps).build((), target, replace_derived=True)

    assert report.note_count == 0
    assert report.passage_count == 0
    assert report.extraction_failure_count == 0
    assert target.exists()
    snapshot = load_index_snapshot(target)
    assert snapshot.passage_matrix.shape[0] == 0
    assert snapshot.fact_matrix.shape[0] == 0
    assert len(snapshot.projection) == 0


def test_failed_build_never_replaces_active_target(tmp_path: Path) -> None:
    target = tmp_path / "active.db"
    target.write_bytes(b"known-active-snapshot")
    deps = BuildDependencies(
        tokenizer=Tokenizer(),
        embedder=lambda _texts: (_ for _ in ()).throw(RuntimeError("embedding failed")),
        embedding_dim=16,
    )

    with pytest.raises(BuildError, match="embedding failed"):
        BuildCoordinator(deps).build(
            [make_note("a", "Alpha", "body")], target, replace_derived=True
        )

    assert target.read_bytes() == b"known-active-snapshot"
    assert not list(tmp_path.glob("active.build-*.db"))


def test_cli_from_enabled_connectors_uses_clean_replace_contract(tmp_path: Path) -> None:
    target = tmp_path / "knowledge.db"
    notes = [make_note("a", "Alpha", "body")]
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)

    code = main(
        ["--db", str(target), "--from-enabled-connectors", "--replace-derived", "--yes"],
        dependencies=deps,
        connector_loader=lambda: notes,
    )

    assert code == 0
    assert target.exists()


def test_replace_derived_flag_is_required(tmp_path: Path) -> None:
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    with pytest.raises(BuildError, match="replace-derived"):
        BuildCoordinator(deps).build([], tmp_path / "knowledge.db", replace_derived=False)


def test_build_finalizes_extracted_entities_mentions_and_directed_facts(tmp_path: Path) -> None:
    note = make_note("n1", "Launch", "Maya owns Launch.")

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(surface="Maya", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="Launch", etype="project", confidence=0.9, char_start=10, char_end=16, extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[
                    ExtractedRelation(head="Maya", predicate="owns_or_has", tail="Launch", confidence=0.9, subject_start=0, subject_end=4, object_start=10, object_end=16, assertion_start=0, assertion_end=17, extraction_method="fake", schema_id="s1", status="active")
                ],
                schema_id="s1",
            )
        }

    deps = BuildDependencies(
        tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
    )
    target = tmp_path / "knowledge.db"

    report = BuildCoordinator(deps).build([note], target, replace_derived=True)

    store = KnowledgeStore(target, read_only=True)
    try:
        counts = store.counts()
        assert counts["entities"] == 2
        assert counts["entity_mentions"] == 2
        assert counts["facts"] == 1
        assert counts["fact_embeddings"] == 1
        assert counts["passage_entity_edges"] == 2
        assert store.fact_contexts()[0].canonical_text == (
            "Maya — owns_or_has (owns or has) — Launch"
        )
        assert report.health == "degraded"
    finally:
        store.close()


def test_degenerate_facts_are_skipped_not_fatal(tmp_path: Path) -> None:
    """A relation whose endpoints merge into one entity costs one fact, not the build."""
    note = make_note("n1", "Meet", "Maya Chen met Maya Chen.")

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(surface="Maya Chen", etype="person", confidence=0.95, char_start=0, char_end=9, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="Maya Chen", etype="person", confidence=0.9, char_start=14, char_end=23, extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[
                    ExtractedRelation(head="Maya Chen", predicate="met_with", tail="Maya Chen", confidence=0.9, subject_start=0, subject_end=9, object_start=14, object_end=23, assertion_start=0, assertion_end=24, extraction_method="fake", schema_id="s1", status="active")
                ],
                schema_id="s1",
            )
        }

    deps = BuildDependencies(
        tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
    )
    target = tmp_path / "knowledge.db"

    report = BuildCoordinator(deps).build([note], target, replace_derived=True)

    assert report.degenerate_fact_count == 1
    assert report.degenerate_facts[0][0] == "n1"
    assert report.fact_count == 0
    assert target.exists()
    store = KnowledgeStore(target, read_only=True)
    try:
        assert store.counts()["facts"] == 0
        assert store.counts()["entities"] == 1
    finally:
        store.close()


def test_supersession_across_notes_survives_insert_order(tmp_path: Path) -> None:
    """Exclusive predicates produce superseded_by links; the build must insert them safely."""
    note_old = make_note("n1", "Old", "Maya lives-in Paris today.", modified=10.0)
    note_new = make_note("n2", "New", "Maya lives-in London today.", modified=20.0)

    def extractor(notes, *, tokenizer):
        del tokenizer
        results = {}
        for note in notes:
            place = "Paris" if note.note_id == "n1" else "London"
            results[note.note_id] = Extraction(
                mentions=[
                    ExtractedMention(surface="Maya", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface=place, etype="place", confidence=0.9, char_start=14, char_end=14 + len(place), extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[
                    ExtractedRelation(head="Maya", predicate="located_at", tail=place, confidence=0.9, subject_start=0, subject_end=4, object_start=14, object_end=14 + len(place), assertion_start=0, assertion_end=14 + len(place), extraction_method="fake", schema_id="s1", status="active")
                ],
                schema_id="s1",
            )
        return results

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note_old, note_new], target, replace_derived=True)

    assert report.fact_count == 2
    store = KnowledgeStore(target, read_only=True)
    try:
        rows = list(
            store._conn.execute("SELECT status, superseded_by FROM facts ORDER BY observed_at")
        )
        assert rows[0][0] == "superseded"
        assert rows[0][1] is not None
        assert rows[1][0] == "active"
        assert store.validate_invariants().ok
    finally:
        store.close()


def test_duplicate_relations_and_mentions_are_deduped(tmp_path: Path) -> None:
    note = make_note("n1", "Launch", "Maya owns Launch.")

    def extractor(_notes, *, tokenizer):
        del tokenizer
        mention_a = ExtractedMention(surface="Maya", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1")
        mention_dup = ExtractedMention(surface="MAYA", etype="person", confidence=0.5, char_start=0, char_end=4, extractor="fake", schema_id="s2")
        mention_b = ExtractedMention(surface="Launch", etype="project", confidence=0.9, char_start=10, char_end=16, extractor="fake", schema_id="s1")
        relation = ExtractedRelation(head="Maya", predicate="owns_or_has", tail="Launch", confidence=0.9, subject_start=0, subject_end=4, object_start=10, object_end=16, assertion_start=0, assertion_end=17, extraction_method="fake", schema_id="s1", status="active")
        return {
            "n1": Extraction(
                mentions=[mention_a, mention_dup, mention_b],
                fact_candidates=[relation, relation.model_copy()],
                schema_id="s1",
            )
        }

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note], target, replace_derived=True)

    assert report.fact_count == 1
    store = KnowledgeStore(target, read_only=True)
    try:
        counts = store.counts()
        assert counts["facts"] == 1
        assert counts["entity_mentions"] == 2  # the duplicate span kept once
        # the higher-confidence duplicate wins
        surface = store._conn.execute(
            "SELECT surface FROM entity_mentions WHERE char_start = 0"
        ).fetchone()[0]
        assert surface == "Maya"
    finally:
        store.close()


def test_failed_build_is_kept_when_env_flag_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAURINE_KEEP_FAILED_BUILD", "1")
    target = tmp_path / "knowledge.db"
    deps = BuildDependencies(
        tokenizer=Tokenizer(),
        embedder=lambda _texts: (_ for _ in ()).throw(RuntimeError("embedding failed")),
        embedding_dim=16,
    )

    with pytest.raises(BuildError, match="embedding failed"):
        BuildCoordinator(deps).build(
            [make_note("a", "Alpha", "body")], target, replace_derived=True
        )

    kept = list(tmp_path.glob("knowledge.build-*.failed"))
    assert len(kept) == 1
    assert not target.exists()


def test_remove_source_surgically_prunes_notes_entities_and_facts(tmp_path: Path) -> None:
    note_a = make_note("a1", "Alpha", "Maya owns Launch.").model_copy(update={"source": "src_a"})
    note_b = make_note("b1", "Beta", "Rex uses Tools.").model_copy(update={"source": "src_b"})

    def extractor(notes, *, tokenizer):
        del tokenizer
        results = {}
        for note in notes:
            if note.note_id == "a1":
                results["a1"] = Extraction(
                    mentions=[
                        ExtractedMention(surface="Maya", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                        ExtractedMention(surface="Launch", etype="project", confidence=0.9, char_start=10, char_end=16, extractor="fake", schema_id="s1"),
                    ],
                    fact_candidates=[
                        ExtractedRelation(head="Maya", predicate="owns_or_has", tail="Launch", confidence=0.9, subject_start=0, subject_end=4, object_start=10, object_end=16, assertion_start=0, assertion_end=17, extraction_method="fake", schema_id="s1", status="active")
                    ],
                    schema_id="s1",
                )
            else:
                results["b1"] = Extraction(
                    mentions=[
                        ExtractedMention(surface="Rex", etype="person", confidence=0.95, char_start=0, char_end=3, extractor="fake", schema_id="s1"),
                        ExtractedMention(surface="Tools", etype="project", confidence=0.9, char_start=9, char_end=14, extractor="fake", schema_id="s1"),
                    ],
                    fact_candidates=[
                        ExtractedRelation(head="Rex", predicate="uses", tail="Tools", confidence=0.9, subject_start=0, subject_end=3, object_start=9, object_end=14, assertion_start=0, assertion_end=15, extraction_method="fake", schema_id="s1", status="active")
                    ],
                    schema_id="s1",
                )
        return results

    target = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note_a, note_b], target, replace_derived=True)

    store = KnowledgeStore(target)
    try:
        assert store.counts()["notes"] == 2
        assert store.counts()["entities"] == 4
        removed = store.remove_source("src_a")
        assert removed == 1
        counts = store.counts()
        assert counts["notes"] == 1
        assert counts["passages"] == 1
        assert counts["entities"] == 2
        assert counts["facts"] == 1
        assert counts["fact_embeddings"] == 1
        assert store.validate_invariants().ok
    finally:
        store.close()

    snapshot = load_index_snapshot(target)
    assert snapshot.passage_matrix.shape[0] == 1
    assert all("a1" not in edge.src_note_id for edge in snapshot.projection)


def test_build_preserves_extracted_observed_and_valid_time(tmp_path: Path) -> None:
    note = make_note("n1", "Launch", "Maya owns Launch.")

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(surface="Maya", etype="person", confidence=.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="Launch", etype="project", confidence=.9, char_start=10, char_end=16, extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[
                    ExtractedRelation(
                        head="Maya", predicate="owns_or_has", tail="Launch",
                        confidence=.9, subject_start=0, subject_end=4,
                        object_start=10, object_end=16, assertion_start=0,
                        assertion_end=17, extraction_method="fake", schema_id="s1",
                        status="historical", observed_at=100.0,
                        valid_from=50.0, valid_to=90.0,
                    )
                ],
                schema_id="s1",
            )
        }

    target = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note], target, replace_derived=True)

    store = KnowledgeStore(target, read_only=True)
    try:
        fact = store.fact_contexts()[0]
        assert (fact.status, fact.observed_at, fact.valid_from, fact.valid_to) == (
            "historical", 100.0, 50.0, 90.0
        )
    finally:
        store.close()
