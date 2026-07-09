from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies, BuildError, main
from taurine_core.indexer.knowledge_store import KnowledgeStore
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
    assert first.stages == (
        "source",
        "passages",
        "embeddings",
        "extraction",
        "entities",
        "facts",
        "base_graph",
        "projection",
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
        assert report.health == "healthy"
    finally:
        store.close()
