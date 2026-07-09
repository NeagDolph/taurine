from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from taurine_core.indexer.embed import FactEmbedding, embed_facts
from taurine_core.indexer.facts import FactRecord
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.passages import PassageDraft
from taurine_core.retrieval.snapshot import load_index_snapshot
from taurine_core.schemas import NoteRecord


def test_embed_facts_uses_canonical_triple_only_and_normalizes() -> None:
    fact = _fact(subject="Maya Tung", object_="Wink Labs")
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.array([[3.0, 4.0]], dtype=np.float32)

    rows = embed_facts(
        (fact,),
        embedder=fake_embed,
        model_id="embed-v1",
        build_id="build-1",
        dim=2,
    )

    assert calls == [["Maya Tung — works_at (works at) — Wink Labs"]]
    assert rows[0].text == calls[0][0]
    assert np.linalg.norm(rows[0].vector) == pytest.approx(1.0)


def test_store_fact_context_uses_canonical_entities_and_global_assertion_slice(
    tmp_path: Path,
) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _write_fact_fixture(store)

        contexts = store.fact_contexts()

        assert len(contexts) == 1
        assert contexts[0].subject == "Maya Tung"
        assert contexts[0].object == "Wink Labs"
        assert contexts[0].assertion == "Maya works at Wink."
        assert contexts[0].canonical_text == (
            "Maya Tung — works_at (works at) — Wink Labs"
        )
        assert contexts[0].logical_key == (
            "entity-maya",
            "works_at",
            "entity-wink",
        )
        assert contexts[0].recognition_document.startswith(
            "Triple: Maya Tung — works_at (works at) — Wink Labs\n"
            "Assertion: Maya works at Wink.\n"
        )
        assert store.fact_contexts(("missing",)) == ()
        assert [row.fact_id for row in store.fact_contexts(("fact-1",))] == ["fact-1"]
    finally:
        store.close()


def test_store_round_trips_fact_embeddings_in_fact_id_order(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _write_fact_fixture(store)
        rows = (
            FactEmbedding("fact-z", "embed-v1", "z", 2, np.array([0.0, 1.0]), "build-1"),
            FactEmbedding("fact-1", "embed-v1", "a", 2, np.array([1.0, 0.0]), "build-1"),
        )
        # Only the fixture fact may satisfy the foreign key.
        store.write_fact_embeddings((rows[1],))

        stored = store.fact_embeddings()

        assert [row.fact_id for row in stored] == ["fact-1"]
        assert np.array_equal(stored[0].vector, np.array([1.0, 0.0], dtype=np.float32))
    finally:
        store.close()


def test_store_rejects_fact_embedding_from_a_different_build(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _write_fact_fixture(store)

        with pytest.raises(ValueError, match="build-2.*build-1"):
            store.write_fact_embeddings(
                (
                    FactEmbedding(
                        "fact-1",
                        "embed-v1",
                        "canonical",
                        2,
                        np.array([1.0, 0.0]),
                        "build-2",
                    ),
                )
            )
    finally:
        store.close()


def test_snapshot_loader_normalizes_sorted_matrices_and_loads_full_graph(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.db"
    store = KnowledgeStore(path)
    try:
        _write_fact_fixture(store)
        store.begin_build(
            "build-1",
            source_snapshot="source-hash",
            model_manifest={"embedding": "embed-v1"},
            config_manifest={"graph": "v1"},
            calibration_hash="calibration-hash",
            started_at=1.0,
        )
        from taurine_core.indexer.embed import PassageEmbedding

        store.write_passage_embeddings(
            (
                PassageEmbedding(
                    "passage-1", "embed-v1", 2, np.array([3.0, 4.0]), "build-1"
                ),
            )
        )
        store.write_fact_embeddings(
            (
                FactEmbedding(
                    "fact-1",
                    "embed-v1",
                    "Maya Tung — works_at (works at) — Wink Labs",
                    2,
                    np.array([0.0, 8.0]),
                    "build-1",
                ),
            )
        )
        store.write_passage_entity_edges(
            (("passage-1", "entity-maya", 0.95, 1, 1.0, 0.95),),
            build_id="build-1",
        )
        store.complete_build(
            "build-1",
            completed_at=2.0,
            health="healthy",
            graph_hash="graph-hash",
            base_edge_count=4,
            projection_count=0,
        )
    finally:
        store.close()

    snapshot = load_index_snapshot(path)

    assert snapshot.passage_ids == ("passage-1",)
    assert snapshot.fact_ids == ("fact-1",)
    assert snapshot.passage_matrix[0].tolist() == pytest.approx([0.6, 0.8])
    assert snapshot.fact_matrix[0].tolist() == pytest.approx([0.0, 1.0])
    assert snapshot.base_graph.build_id == "build-1"
    assert snapshot.base_graph.node_ids == (
        "passage-1",
        "entity-maya",
        "entity-wink",
    )
    forward = next(
        edge
        for edge in snapshot.base_graph.edges
        if edge.kind == "fact" and edge.direction == "forward"
    )
    assert forward.fact_ids == ("fact-1",)
    assert snapshot.alias_index["maya"] == ("entity-maya",)


def test_store_base_graph_groups_logical_facts_by_independent_source(
    tmp_path: Path,
) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        _write_fact_fixture(store)
        original = _fact()
        store.insert_fact(replace(original, fact_id="fact-same-note", confidence=0.7))

        note = NoteRecord(
            note_id="note-2",
            title="Employment duplicate",
            source="apple_notes",
            text="Maya works at Wink.",
            content_hash="note-2-hash",
            created=1.0,
            modified=3.0,
        )
        store.write_note(note, build_id="build-1", extraction_status="complete")
        store.write_passages(
            note.note_id,
            (
                PassageDraft(
                    passage_id="passage-2",
                    ordinal=0,
                    char_start=0,
                    char_end=19,
                    token_count=4,
                    text=note.text,
                    content_hash="passage-2-hash",
                ),
            ),
            created=1.0,
            modified=3.0,
            build_id="build-1",
        )
        store.insert_fact(
            replace(
                original,
                fact_id="fact-independent-note",
                source_passage_id="passage-2",
                source_note_id="note-2",
                assertion_start=0,
                assertion_end=19,
                subject_start=0,
                subject_end=4,
                object_start=14,
                object_end=18,
                confidence=0.5,
            )
        )

        graph = store.base_graph("build-1")

        fact_edges = [edge for edge in graph.edges if edge.kind == "fact"]
        assert len(fact_edges) == 2
        forward = next(edge for edge in fact_edges if edge.direction == "forward")
        reverse = next(edge for edge in fact_edges if edge.direction == "reverse")
        assert forward.fact_ids == (
            "fact-1",
            "fact-independent-note",
            "fact-same-note",
        )
        assert forward.weight == pytest.approx(
            0.25 * (1.0 - (1.0 - 0.9) * (1.0 - 0.5))
        )
        assert reverse.weight == pytest.approx(forward.weight * 0.35)
    finally:
        store.close()


def _write_fact_fixture(store: KnowledgeStore) -> None:
    note = NoteRecord(
        note_id="note-1",
        title="Employment",
        source="apple_notes",
        text="Header. Maya works at Wink. Footer.",
        content_hash="note-hash",
        created=1.0,
        modified=2.0,
    )
    store.write_note(note, build_id="build-1", extraction_status="complete")
    store.write_passages(
        note.note_id,
        (
            PassageDraft(
                passage_id="passage-1",
                ordinal=0,
                char_start=8,
                char_end=27,
                token_count=4,
                text="Maya works at Wink.",
                content_hash="passage-hash",
            ),
        ),
        created=1.0,
        modified=2.0,
        build_id="build-1",
    )
    for entity_id, canonical, normalized in (
        ("entity-maya", "Maya Tung", "maya tung"),
        ("entity-wink", "Wink Labs", "wink labs"),
    ):
        store.insert_entity(
            entity_id=entity_id,
            canonical=canonical,
            normalized=normalized,
            etype="person" if entity_id == "entity-maya" else "organization",
            df=1,
            traversal_eligible=True,
            ambiguity_key=normalized,
            build_id="build-1",
        )
    store.insert_alias("entity-maya", "Maya", "mention", 0.95)
    store.insert_mention(
        mention_id="mention-maya",
        entity_id="entity-maya",
        passage_id="passage-1",
        note_id="note-1",
        surface="Maya",
        etype="person",
        char_start=8,
        char_end=12,
        confidence=0.95,
        extractor="fake",
        schema_id="schema-1",
        build_id="build-1",
    )
    store.insert_mention(
        mention_id="mention-wink",
        entity_id="entity-wink",
        passage_id="passage-1",
        note_id="note-1",
        surface="Wink",
        etype="organization",
        char_start=22,
        char_end=26,
        confidence=0.95,
        extractor="fake",
        schema_id="schema-1",
        build_id="build-1",
    )
    store.insert_fact(_fact())


def _fact(*, subject: str = "Maya", object_: str = "Wink") -> FactRecord:
    return FactRecord(
        fact_id="fact-1",
        subject_entity_id="entity-maya",
        subject=subject,
        predicate="works_at",
        object_entity_id="entity-wink",
        object=object_,
        source_passage_id="passage-1",
        source_note_id="note-1",
        assertion_start=8,
        assertion_end=27,
        subject_start=8,
        subject_end=12,
        object_start=22,
        object_end=26,
        confidence=0.9,
        extraction_method="fake",
        schema_id="schema-1",
        observed_at=2.0,
        valid_from=None,
        valid_to=None,
        invalidated_at=None,
        superseded_by=None,
        status="active",
        build_id="build-1",
    )
