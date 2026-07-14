from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies, BuildError, main
from taurine_core.indexer.build_cache import BuildCache
from taurine_core.indexer.embed import EmbeddingError
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.retrieval.snapshot import load_index_snapshot
from taurine_core.schemas import ExtractedMention, ExtractedRelation, Extraction, NoteRecord


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


def test_build_dependencies_resolve_embedding_backend_at_construction(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMBED_BACKEND", "local")
    local = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder)
    assert local.embedding_model_id == config.EMBED_MODEL
    assert local.embedding_dim == 1024

    monkeypatch.setattr(config, "EMBED_BACKEND", "openrouter")
    remote = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder)
    assert remote.embedding_model_id == config.EMBED_API_MODEL
    assert remote.embedding_dim == 4096


def _completion_extractor(notes, *, tokenizer):
    del tokenizer
    results = {}
    for note in notes:
        maya = note.text.index("Maya")
        launch = note.text.index("Launch")
        results[note.note_id] = Extraction(
            mentions=[
                ExtractedMention(
                    surface="Maya",
                    etype="person",
                    confidence=0.95,
                    char_start=maya,
                    char_end=maya + 4,
                    extractor="fake",
                    schema_id="s1",
                ),
                ExtractedMention(
                    surface="Launch",
                    etype="project",
                    confidence=0.95,
                    char_start=launch,
                    char_end=launch + 6,
                    extractor="fake",
                    schema_id="s1",
                ),
            ],
            fact_candidates=[],
            schema_id="s1",
        )
    return results


def test_completion_cache_skips_repeat_calls_and_caches_empty(tmp_path: Path) -> None:
    calls: list[int] = []

    def completer(notes):
        calls.append(len(notes))
        return {
            "facts": [],
            "aliases": [],
            "completed_note_ids": [note.note_id for note in notes],
        }

    note = make_note(
        "note-a",
        "Launch planning",
        "Maya coordinates the Launch project and reviews its detailed weekly roadmap.",
    )
    cache = BuildCache(tmp_path / "cache.db")
    try:
        coordinator = BuildCoordinator(
            BuildDependencies(
                tokenizer=Tokenizer(),
                embedder=embedder,
                embedding_dim=16,
                extractor=_completion_extractor,
                relation_completer=completer,
                completion_cache=cache,
            )
        )
        coordinator.build([note], tmp_path / "first.db", replace_derived=True)
        assert calls == [1]
        assert calls[0] > 0

        coordinator.build([note], tmp_path / "second.db", replace_derived=True)
        assert calls == [1]
    finally:
        cache.close()


def test_completion_failed_batch_notes_retry_next_build(tmp_path: Path) -> None:
    calls: list[int] = []

    def completer(notes):
        calls.append(len(notes))
        return {"facts": [], "aliases": [], "completed_note_ids": []}

    note = make_note(
        "note-a",
        "Launch planning",
        "Maya coordinates the Launch project and reviews its detailed weekly roadmap.",
    )
    cache = BuildCache(tmp_path / "cache.db")
    try:
        coordinator = BuildCoordinator(
            BuildDependencies(
                tokenizer=Tokenizer(),
                embedder=embedder,
                embedding_dim=16,
                extractor=_completion_extractor,
                relation_completer=completer,
                completion_cache=cache,
            )
        )
        coordinator.build([note], tmp_path / "first.db", replace_derived=True)
        coordinator.build([note], tmp_path / "second.db", replace_derived=True)
        assert calls == [1, 1]
    finally:
        cache.close()


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
    artifact = load_calibration(
        Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
    )
    assert first.calibration_hash == artifact.hash
    assert first.release_ready is False
    # dense is self-calibrated (sample_count > 0) and must NOT be a blocker;
    # heads without fitted curves still block release
    assert "uncalibrated_retrieval_head:live:dense" not in first.release_blockers
    assert "uncalibrated_retrieval_head:live:bm25" in first.release_blockers
    assert first.stages == (
        "source",
        "passages",
        "context_enrichment",
        "embeddings",
        "extraction",
        "entities",
        "relation_completion",
        "facts",
        "adjudication",
        "base_graph",
        "projection",
        "ranks",
        "clusters",
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
    assert snapshot.passage_matrix.shape == (0, config.embed_dim())
    assert snapshot.fact_matrix.shape == (0, config.embed_dim())
    assert len(snapshot.projection) == 0


def test_failed_build_never_replaces_active_target(tmp_path: Path) -> None:
    target = tmp_path / "active.db"
    target.write_bytes(b"known-active-snapshot")
    deps = BuildDependencies(
        tokenizer=Tokenizer(),
        embedder=lambda _texts: (_ for _ in ()).throw(
            EmbeddingError("OpenRouter embeddings returned HTTP 503 after 3 attempts")
        ),
        embedding_dim=16,
    )

    with pytest.raises(BuildError, match="OpenRouter embeddings returned HTTP 503"):
        BuildCoordinator(deps).build(
            [make_note("a", "Alpha", "body")], target, replace_derived=True
        )

    assert target.read_bytes() == b"known-active-snapshot"
    assert not list(tmp_path.glob("active.build-*.db"))


def test_cli_installs_but_exits_3_when_release_blockers_remain(tmp_path: Path) -> None:
    """M10/§22: the CLI must not report false green — an uncalibrated build
    still installs (availability) but exits nonzero with blockers printed."""
    target = tmp_path / "knowledge.db"
    notes = [make_note("a", "Alpha", "body")]
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)

    code = main(
        ["--db", str(target), "--from-enabled-connectors", "--replace-derived", "--yes"],
        dependencies=deps,
        connector_loader=lambda: notes,
    )

    assert code == 3
    assert target.exists()


def test_cli_install_degraded_flag_accepts_blockers_with_exit_0(tmp_path: Path) -> None:
    target = tmp_path / "knowledge.db"
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)

    code = main(
        [
            "--db",
            str(target),
            "--from-enabled-connectors",
            "--replace-derived",
            "--yes",
            "--install-degraded",
        ],
        dependencies=deps,
        connector_loader=lambda: [make_note("a", "Alpha", "body")],
    )

    assert code == 0
    assert target.exists()


def test_cli_exits_2_and_installs_nothing_on_build_failure(tmp_path: Path) -> None:
    target = tmp_path / "knowledge.db"
    deps = BuildDependencies(
        tokenizer=Tokenizer(),
        embedder=lambda _texts: (_ for _ in ()).throw(RuntimeError("embedding failed")),
        embedding_dim=16,
    )

    code = main(
        ["--db", str(target), "--from-enabled-connectors", "--replace-derived", "--yes"],
        dependencies=deps,
        connector_loader=lambda: [make_note("a", "Alpha", "body")],
    )

    assert code == 2
    assert not target.exists()


def test_cli_no_relation_completion_downgrades_the_blocker(tmp_path: Path) -> None:
    target = tmp_path / "knowledge.db"

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {"a": Extraction(mentions=[], fact_candidates=[], schema_id="s1")}

    deps = BuildDependencies(
        tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
    )
    report_holder: list = []

    class SpyCoordinator(BuildCoordinator):
        def build(self, *args, **kwargs):
            report = super().build(*args, **kwargs)
            report_holder.append(report)
            return report

    import taurine_core.indexer.build as build_module

    original = build_module.BuildCoordinator
    build_module.BuildCoordinator = SpyCoordinator  # type: ignore[misc]
    try:
        main(
            [
                "--db",
                str(target),
                "--from-enabled-connectors",
                "--replace-derived",
                "--yes",
                "--no-relation-completion",
            ],
            dependencies=deps,
            connector_loader=lambda: [make_note("a", "Alpha", "body")],
        )
    finally:
        build_module.BuildCoordinator = original  # type: ignore[misc]

    assert report_holder
    assert "relation_completion_unavailable" not in report_holder[0].release_blockers


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


# ── §0.2 salience + extraction gating ─────────────────────────────────────


def _named_extractor(surfaces_by_note):
    def extractor(notes, *, tokenizer):
        del tokenizer
        results = {}
        for note in notes:
            mentions = []
            for surface in surfaces_by_note.get(note.note_id, ()):
                start = note.text.find(surface)
                if start < 0:
                    continue
                mentions.append(
                    ExtractedMention(
                        surface=surface,
                        etype="person",
                        confidence=0.95,
                        char_start=start,
                        char_end=start + len(surface),
                        extractor="fake",
                        schema_id="s1",
                    )
                )
            results[note.note_id] = Extraction(
                mentions=mentions, fact_candidates=[], schema_id="s1"
            )
        return results

    return extractor


def test_low_salience_notes_skip_extraction_but_stay_searchable(tmp_path: Path) -> None:
    good = make_note("good", "Plan", "Maya owns the Zephyr launch checklist.")
    junk = make_note("junk", "Filler", "yeah " * 12)
    seen_by_extractor: list[list[str]] = []

    def extractor(notes, *, tokenizer):
        del tokenizer
        seen_by_extractor.append([note.note_id for note in notes])
        return {
            note.note_id: Extraction(
                mentions=[
                    ExtractedMention(
                        surface="Maya",
                        etype="person",
                        confidence=0.95,
                        char_start=0,
                        char_end=4,
                        extractor="fake",
                        schema_id="s1",
                    )
                ],
                fact_candidates=[],
                schema_id="s1",
            )
            for note in notes
        }

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([good, junk], target, replace_derived=True)

    assert seen_by_extractor == [["good"]]  # GLiNER never sees the junk note
    assert report.salience_gated_count == 1
    assert report.salience_gated_by_source == {"apple_notes": 1}
    assert report.salience_stats is not None and report.salience_stats["count"] == 2.0
    # gating is never a failure and never degrades health by itself
    assert report.failed_notes == ()
    assert "junk" not in dict(report.failed_notes)
    store = KnowledgeStore(target, read_only=True)
    try:
        notes = {row.note_id: row for row in store.source_notes()}
        assert notes["junk"].extraction_status == "skipped_low_salience"
        assert notes["good"].extraction_status == "complete"
        assert notes["junk"].salience < notes["good"].salience
        # passages, FTS, and embeddings are ALWAYS built for gated notes
        assert store.counts()["passages"] == 2
        assert store.counts()["passage_embeddings"] == 2
        assert store.search_passages_bm25("yeah", 5) != ()
        # but no entity/fact participation
        assert all(
            mention_row[0] != "junk"
            for mention_row in store._conn.execute(
                "SELECT note_id FROM entity_mentions"
            )
        )
    finally:
        store.close()


def test_connector_ingest_salience_reaches_the_note_row_and_the_gate(
    tmp_path: Path,
) -> None:
    """§0.2 plumbing (M4): a low connector salience gates a note whose content
    alone would extract, and both factors stay inspectable on the row."""
    text = (
        "Follow-up on the vendor contract: Maya wants the revised terms "
        "reviewed by legal before the Thursday sync so we can countersign."
    )
    normal = make_note("kept", "Contract", text)
    demoted = make_note("demoted", "Contract copy", text + " Extra.").model_copy(
        update={"ingest_salience": 0.1}
    )

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=embedder,
            embedding_dim=16,
            extractor=_named_extractor({"kept": ("Maya",), "demoted": ("Maya",)}),
        )
    ).build([normal, demoted], target, replace_derived=True)

    assert report.salience_gated_count == 1
    store = KnowledgeStore(target, read_only=True)
    try:
        notes = {row.note_id: row for row in store.source_notes()}
        assert notes["demoted"].ingest_salience == pytest.approx(0.1)
        assert notes["demoted"].extraction_status == "skipped_low_salience"
        assert notes["kept"].extraction_status == "complete"
    finally:
        store.close()


def test_value_scores_multiply_in_salience(tmp_path: Path) -> None:
    text = (
        "Planning notes for the quarterly review: gather the metrics deck, "
        "confirm attendees, and align on the launch narrative beforehand."
    )
    high = make_note("high", "Review", text, modified=1.0)
    low = make_note("low", "Review B", text + " More.", modified=1.0).model_copy(
        update={"ingest_salience": 0.5}
    )

    target = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    ).build([high, low], target, replace_derived=True)

    store = KnowledgeStore(target, read_only=True)
    try:
        notes = {row.note_id: row for row in store.source_notes()}
        assert notes["low"].value_score < notes["high"].value_score
        assert notes["low"].value_score == pytest.approx(
            notes["high"].value_score * notes["low"].salience / notes["high"].salience,
            rel=0.05,
        )
    finally:
        store.close()


# ── M13 (build half) — hub provenance ─────────────────────────────────────


def test_hub_exclusions_report_real_traversal_reasons_and_persist(
    tmp_path: Path,
) -> None:
    note = make_note(
        "n1", "Plan", "Maya moved the deadline to tomorrow for the launch review."
    )

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(surface="Maya", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="tomorrow", etype="date_or_deadline", confidence=0.9, char_start=26, char_end=34, extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[],
                schema_id="s1",
            )
        }

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note], target, replace_derived=True)

    reasons = dict(report.hub_exclusions)
    assert "date_default_off" in reasons.values()
    store = KnowledgeStore(target, read_only=True)
    try:
        rows = {
            row["canonical"]: (bool(row["traversal_eligible"]), str(row["traversal_reason"]))
            for row in store._conn.execute(
                "SELECT canonical, traversal_eligible, traversal_reason FROM entities"
            )
        }
        assert rows["tomorrow"] == (False, "date_default_off")
        assert rows["Maya"] == (True, "eligible")
        context = store.entity_contexts(
            [entity_id for entity_id, reason in report.hub_exclusions]
        )[0]
        assert context.traversal_reason == "date_default_off"
    finally:
        store.close()


def test_owner_aliases_gate_the_owner_entity_as_a_hub(tmp_path: Path) -> None:
    """L18 (build half): aliases passed into the build reach the resolver."""
    notes = [
        make_note(f"n{index}", f"Note {index}", f"Neil reviewed item {index} today.")
        for index in range(4)
    ]

    def extractor(records, *, tokenizer):
        del tokenizer
        return {
            note.note_id: Extraction(
                mentions=[
                    ExtractedMention(surface="Neil", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1")
                ],
                fact_candidates=[],
                schema_id="s1",
            )
            for note in records
        }

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build(notes, target, replace_derived=True, owner_aliases=("Neil",))

    assert dict(report.hub_exclusions)
    assert set(dict(report.hub_exclusions).values()) == {"owner_alias_hub"}


# ── H2 degeneracy + L24 invariants ────────────────────────────────────────


def test_degenerate_contains_weights_marks_health_degraded() -> None:
    from taurine_core.indexer.build import _degenerate_contains_weights

    flat = [0.7] * 10
    assert _degenerate_contains_weights(flat) is True
    varied = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert _degenerate_contains_weights(varied) is False
    # tiny graphs are exempt: coincidental equality is expected
    assert _degenerate_contains_weights([0.5, 0.5, 0.5]) is False


def test_contains_weight_stats_summarize_the_distribution(tmp_path: Path) -> None:
    note = make_note("n1", "Launch", "Maya owns Launch.")

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(surface="Maya", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="Launch", etype="project", confidence=0.9, char_start=10, char_end=16, extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[],
                schema_id="s1",
            )
        }

    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note], tmp_path / "knowledge.db", replace_derived=True)

    stats = report.contains_weight_stats
    assert stats is not None
    assert stats["count"] == 2.0
    assert 0.0 < stats["min"] <= stats["median"] <= stats["max"] < 1.0
    assert report.health in {"healthy", "degraded"}
    assert "degenerate_contains_weights" not in report.degraded_reasons


def test_column_stochastic_check_rejects_a_corrupt_matrix() -> None:
    from scipy.sparse import csr_matrix as _csr

    from taurine_core.indexer.build import _assert_column_stochastic

    class FakeGraph:
        matrix = _csr(np.asarray([[0.5, 0.0], [0.4, 0.0]]))  # column 0 sums to 0.9
        dangling_rows = (1,)

    with pytest.raises(BuildError, match="non-stochastic"):
        _assert_column_stochastic(FakeGraph())


def test_stored_count_check_rejects_divergence() -> None:
    from taurine_core.indexer.build import _assert_stored_counts

    with pytest.raises(BuildError, match="diverge"):
        _assert_stored_counts({"notes": 2}, {"notes": 3})
    _assert_stored_counts({"notes": 2, "facts": 0}, {"notes": 2, "facts": 0})


def test_graph_hash_round_trips_through_the_store(tmp_path: Path) -> None:
    """L24: stored rows reproduce the recorded graph hash exactly."""
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

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note], target, replace_derived=True)

    store = KnowledgeStore(target, read_only=True)
    try:
        assert store.recompute_graph_hash(report.build_id) == report.graph_hash
        assert store.verify_graph_hash(report.build_id) is True
    finally:
        store.close()


# ── M14 — same-source near-duplicate guard ────────────────────────────────


def test_note_ranks_exclude_same_source_near_duplicate_edges() -> None:
    from taurine_core.indexer.build import _compute_note_ranks
    from taurine_core.indexer.projection import ProjectionEdge

    notes = [
        make_note("a", "A", "alpha body"),
        make_note("b", "B", "alpha body copy"),
        make_note("c", "C", "unrelated"),
    ]

    def edge(src: str, dst: str, semantic: float) -> ProjectionEdge:
        return ProjectionEdge(
            src_note_id=src,
            dst_note_id=dst,
            edge_type="semantic",
            weight=semantic,
            shared=(),
            fact_ids=(),
            mechanisms={"semantic": semantic},
        )

    duplicate = edge("a", "b", 0.97)  # same source, above DUP_EDGE_SIM
    normal = edge("a", "c", 0.5)  # same source, below the near-duplicate floor

    cross_source_notes = [
        notes[0],
        notes[1].model_copy(update={"source": "obsidian"}),
        notes[2],
    ]

    _, _, flagged = _compute_note_ranks(notes, [duplicate, normal], now=100.0)
    assert flagged == (("a", "b"),)

    _, _, cross_flagged = _compute_note_ranks(
        cross_source_notes, [duplicate, normal], now=100.0
    )
    assert cross_flagged == ()  # different sources are never duplicate-guarded

    quality_with_dup, _, _ = _compute_note_ranks(notes, [duplicate], now=100.0)
    quality_without, _, _ = _compute_note_ranks(notes, [], now=100.0)
    assert quality_with_dup == quality_without  # the edge left the matrix


def test_build_report_flags_near_duplicate_pairs(tmp_path: Path) -> None:
    shared = (
        "Weekly planning notes for the checkout redesign project covering "
        "the payment vendor migration and the rollout schedule for Thursday."
    )
    # identical title+body → identical passage embeddings under the fake
    # hash embedder → a semantic mechanism of 1.0 between the two notes
    first = make_note("dup1", "Plan", shared)
    second = make_note("dup2", "Plan", shared)

    report = BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    ).build([first, second], tmp_path / "knowledge.db", replace_derived=True)

    assert report.near_duplicate_note_pairs == (("dup1", "dup2"),)


# ── L25 — startup temp sweep ──────────────────────────────────────────────


def test_sweep_removes_only_stale_orphaned_build_files(tmp_path: Path) -> None:
    import os as _os

    from taurine_core.indexer.build import sweep_stale_build_files

    target = tmp_path / "taurine-knowledge.db"
    target.write_bytes(b"active")
    stale = tmp_path / "taurine-knowledge.build-old.db"
    stale.write_bytes(b"orphan")
    fresh = tmp_path / "taurine-knowledge.build-new.db"
    fresh.write_bytes(b"in-progress")
    failed = tmp_path / "taurine-knowledge.build-x.db.failed"
    failed.write_bytes(b"post-mortem")
    old = 1_000_000.0
    _os.utime(stale, (old, old))

    removed = sweep_stale_build_files(target, now=old + 100_000.0)

    assert removed == (stale,)
    assert not stale.exists()
    assert target.exists() and fresh.exists() and failed.exists()


# ── L26 — real model manifest fingerprints ────────────────────────────────


def test_model_fingerprints_ride_config_manifest_not_model_manifest(
    tmp_path: Path,
) -> None:
    report = BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    ).build([], tmp_path / "knowledge.db", replace_derived=True)

    fingerprints = json.loads(report.config_manifest["model_fingerprints"])
    assert set(fingerprints) == {"embedding", "gliner"}
    assert fingerprints["embedding"] == f"api:{config.EMBED_API_MODEL}"
    # the model_manifest keeps ONLY the artifact identity strings the runtime
    # compares for equality — fingerprints must never leak into it
    artifact = load_calibration(
        Path(__file__).parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
    )
    expected_manifest = dict(artifact.model_manifest)
    expected_manifest["embedding"] = config.embed_model_id()
    assert report.model_manifest == expected_manifest


def test_model_fingerprint_tracks_local_artifact_changes(tmp_path: Path, monkeypatch) -> None:
    import os as _os

    import taurine_core.indexer.build as build_module

    model_dir = tmp_path / "some-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setattr(build_module, "local_model_path", lambda repo_id: str(model_dir))

    first = build_module._model_fingerprint("org/some-model")
    assert first.startswith("sha256:")
    assert first == build_module._model_fingerprint("org/some-model")

    (model_dir / "config.json").write_text('{"changed": true}')
    _os.utime(model_dir / "config.json", (2_000_000_000, 2_000_000_000))
    assert build_module._model_fingerprint("org/some-model") != first

    monkeypatch.setattr(build_module, "local_model_path", lambda repo_id: None)
    assert build_module._model_fingerprint("org/none") == "unavailable"


# ── L28 — sync resilience and single-read orchestration ───────────────────


class _MemoryState:
    def __init__(self) -> None:
        self.states: dict[str, dict] = {}

    def connector_state(self, connector_id: str) -> dict:
        return dict(self.states.get(connector_id, {"enabled": True}))

    def set_connector_state(self, connector_id: str, state: dict) -> None:
        self.states[connector_id] = dict(state)

    def replace_source_records(self, connector_id, records):
        return len(records), 0, 0


class _StubConnector:
    kind = "local"

    def __init__(self, connector_id: str, records, *, up: bool = True) -> None:
        self.connector_id = connector_id
        self.label = connector_id.title()
        self.records = list(records)
        self.up = up
        self.extract_calls = 0
        self.ingest_salience_scores = {row.note_id: 0.6 for row in self.records}

    def available(self):
        return (True, "") if self.up else (False, "token expired")

    def default_config(self):
        return {}

    def extract(self, config):
        self.extract_calls += 1
        return list(self.records)


def test_extract_all_enabled_skips_unavailable_connectors_instead_of_aborting() -> None:
    from taurine_core.connectors.registry import ConnectorRegistry

    good = _StubConnector("good", [make_note("g1", "G", "body")])
    down = _StubConnector("down", [make_note("d1", "D", "body")], up=False)
    registry = ConnectorRegistry([good, down])

    extraction = registry.extract_all_enabled(_MemoryState())

    assert [row.note_id for row in extraction.records] == ["g1"]
    assert extraction.unavailable == ("down",)
    # §0.2 plumbing: connector salience is stamped onto the records
    assert extraction.records[0].ingest_salience == pytest.approx(0.6)


def test_sync_all_enabled_collecting_reads_each_source_exactly_once() -> None:
    from taurine_core.connectors.registry import ConnectorRegistry

    good = _StubConnector("good", [make_note("g1", "G", "body")])
    down = _StubConnector("down", [], up=False)
    registry = ConnectorRegistry([good, down])

    extraction = registry.sync_all_enabled_collecting(_MemoryState(), None)

    assert good.extract_calls == 1
    assert [row.note_id for row in extraction.records] == ["g1"]
    assert extraction.records[0].ingest_salience == pytest.approx(0.6)
    assert extraction.unavailable == ("down",)
    assert [report.connector_id for report in extraction.reports] == ["good", "down"]
    assert extraction.reports[1].error == "token expired"


def test_carry_forward_records_reads_a_source_from_the_active_db(tmp_path: Path) -> None:
    from taurine_core.indexer.build import carry_forward_records

    kept = make_note("gmail:t1", "Thread", "Maya confirmed the Thursday review.")
    kept = kept.model_copy(update={"source": "gmail", "ingest_salience": 0.8})
    other = make_note("a1", "Alpha", "body")
    target = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    ).build([kept, other], target, replace_derived=True)

    carried = carry_forward_records(target, ["gmail"])

    assert [row.note_id for row in carried] == ["gmail:t1"]
    assert carried[0].source == "gmail"
    assert carried[0].text == kept.text
    assert carried[0].ingest_salience == pytest.approx(0.8)
    assert carry_forward_records(target, []) == ()
    assert carry_forward_records(tmp_path / "missing.db", ["gmail"]) == ()


def test_degraded_reasons_mark_build_health_degraded(tmp_path: Path) -> None:
    report = BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    ).build(
        [],
        tmp_path / "knowledge.db",
        replace_derived=True,
        degraded_reasons=("connector_unavailable:gmail",),
    )

    assert report.health == "degraded"
    assert "connector_unavailable:gmail" in report.degraded_reasons


def test_profile_owner_aliases_parse_the_user_card_heading() -> None:
    from taurine_core.profile_context import owner_aliases

    class FakeIndex:
        def __init__(self, card: str | None) -> None:
            self._card = card

        def user_card(self) -> str | None:
            return self._card

    card = "# Neil Pathangay\n\nNeil is a software engineer working on Taurine."
    aliases = owner_aliases(FakeIndex(card), connector_identity=lambda: ())
    assert "neil pathangay" in aliases
    assert "neil" in aliases
    assert "pathangay" in aliases

    # generic headings never become owner aliases
    assert (
        owner_aliases(
            FakeIndex("# User Profile\n\nSome background."),
            connector_identity=lambda: (),
        )
        == ()
    )
    assert owner_aliases(FakeIndex(None), connector_identity=lambda: ()) == ()


def test_profile_owner_aliases_merge_imessage_self_identity() -> None:
    from taurine_core.profile_context import owner_aliases

    class FakeIndex:
        def user_card(self) -> str | None:
            return None

    aliases = owner_aliases(
        FakeIndex(), connector_identity=lambda: ("Neil Pathangay",)
    )
    assert "neil pathangay" in aliases
    assert "neil" in aliases
    assert "pathangay" in aliases


def _works_at_extractor(org_by_note):
    def extractor(notes, *, tokenizer):
        del tokenizer
        results = {}
        for note in notes:
            org = org_by_note[note.note_id]
            results[note.note_id] = Extraction(
                mentions=[
                    ExtractedMention(surface="Neil", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface=org, etype="org", confidence=0.9, char_start=14, char_end=14 + len(org), extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[
                    ExtractedRelation(head="Neil", predicate="works_at", tail=org, confidence=0.9, subject_start=0, subject_end=4, object_start=14, object_end=14 + len(org), assertion_start=0, assertion_end=14 + len(org), extraction_method="fake", schema_id="s1", status="active")
                ],
                schema_id="s1",
            )
        return results

    return extractor


def _snapshot_note(note_id: str, title: str, text: str, *, created: float, modified: float) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=title,
        folder="Fixture",
        created=created,
        modified=modified,
        content_hash=hashlib.sha256((title + text).encode("utf-8")).hexdigest(),
        text=text,
    )


def test_snapshot_document_anchors_observed_at_so_old_resume_loses(tmp_path: Path) -> None:
    """A 2024 resume synced AFTER a 2026 intro note must not outrank it."""
    resume_date = datetime(2024, 1, 18, tzinfo=timezone.utc).timestamp()
    intro_time = datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()
    resume = _snapshot_note(
        "n1", "neil_agrawal_resume_1_18_24", "Neil works-at Wolters.",
        created=100.0, modified=intro_time + 5000.0,
    )
    intro = _snapshot_note(
        "n2", "Intro", "Neil works-at Klaviyo.",
        created=intro_time, modified=intro_time,
    )

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16,
            extractor=_works_at_extractor({"n1": "Wolters", "n2": "Klaviyo"}),
        )
    ).build([resume, intro], target, replace_derived=True)

    assert report.fact_count == 2
    store = KnowledgeStore(target, read_only=True)
    try:
        rows = list(
            store._conn.execute(
                "SELECT source_note_id, observed_at, status FROM facts ORDER BY source_note_id"
            )
        )
        assert rows[0][1] == pytest.approx(resume_date)  # doc date, not modified
        assert rows[0][2] == "superseded"
        assert rows[1][1] == pytest.approx(intro_time)
        assert rows[1][2] == "active"
        flags = dict(
            store._conn.execute("SELECT note_id, snapshot_document FROM notes")
        )
        assert flags == {"n1": 1, "n2": 0}
        assert store.source_notes()[0].snapshot_document is True
        assert store.validate_invariants().ok
    finally:
        store.close()


def test_snapshot_without_parseable_date_uses_created_never_modified(tmp_path: Path) -> None:
    resume = _snapshot_note(
        "n1", "Neil Agrawal Resume", "Neil works-at Klaviyo.",
        created=500.0, modified=999999.0,
    )

    target = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16,
            extractor=_works_at_extractor({"n1": "Klaviyo"}),
        )
    ).build([resume], target, replace_derived=True)

    store = KnowledgeStore(target, read_only=True)
    try:
        observed = store._conn.execute("SELECT observed_at FROM facts").fetchone()[0]
        assert observed == pytest.approx(500.0)
    finally:
        store.close()


def test_same_note_duplicate_triples_collapse_in_the_real_build(tmp_path: Path) -> None:
    text = "Neil works-at Klaviyo. Neil works-at Klaviyo."
    note = _snapshot_note("n1", "Intro", text, created=50.0, modified=50.0)

    def extractor(_notes, *, tokenizer):
        del tokenizer
        mentions = []
        candidates = []
        for offset in (0, 23):
            mentions.append(ExtractedMention(surface="Neil", etype="person", confidence=0.95, char_start=offset, char_end=offset + 4, extractor="fake", schema_id="s1"))
            mentions.append(ExtractedMention(surface="Klaviyo", etype="org", confidence=0.9, char_start=offset + 14, char_end=offset + 21, extractor="fake", schema_id="s1"))
            candidates.append(ExtractedRelation(head="Neil", predicate="works_at", tail="Klaviyo", confidence=0.6 if offset == 0 else 0.9, subject_start=offset, subject_end=offset + 4, object_start=offset + 14, object_end=offset + 21, assertion_start=offset, assertion_end=offset + 21, extraction_method="fake", schema_id="s1", status="active"))
        return {"n1": Extraction(mentions=mentions, fact_candidates=candidates, schema_id="s1")}

    target = tmp_path / "knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build([note], target, replace_derived=True)

    assert report.fact_count == 1
    store = KnowledgeStore(target, read_only=True)
    try:
        row = store._conn.execute(
            "SELECT assertion_start, confidence, status FROM facts"
        ).fetchone()
        assert row[0] == 0  # earliest assertion span kept
        assert row[1] == pytest.approx(0.9)  # max confidence kept
        assert row[2] == "active"
    finally:
        store.close()


def test_build_memory_watchdog_aborts_cleanly_over_ceiling(
    tmp_path: Path, monkeypatch
) -> None:
    # A near-zero ceiling trips the watchdog on the first stage boundary; the
    # build must fail with BuildError and never install the target.
    monkeypatch.setattr(config, "BUILD_MEMORY_CEILING_GB", 0.0001)
    target = tmp_path / "guarded.db"
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    with pytest.raises(BuildError, match="resident set exceeded"):
        BuildCoordinator(deps).build(
            [make_note("a", "Alpha", "Launch is scheduled for July 18.")],
            target,
            replace_derived=True,
        )
    assert not target.exists()  # nothing installed


def test_build_records_peak_memory(tmp_path: Path) -> None:
    deps = BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    report = BuildCoordinator(deps).build(
        [make_note("a", "Alpha", "Launch is scheduled for July 18.")],
        tmp_path / "peak.db",
        replace_derived=True,
    )
    assert report.peak_memory_gb > 0.0
    assert report.peak_memory_stage != ""
