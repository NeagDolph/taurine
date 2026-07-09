from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from taurine_core.indexer.embed import embed_passages
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.passages import PassageDraft
from taurine_core.schemas import NoteRecord


def test_embed_passages_is_ordered_normalized_and_uses_metadata() -> None:
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.array([[0.0, 5.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)

    rows = embed_passages(
        (_passage("p2", 1, "Second exact span."), _passage("p1", 0, "First exact span.")),
        title="Roadmap",
        source="apple_notes",
        embedder=fake_embed,
        model_id="fake-embedding-v1",
        build_id="build-1",
        dim=3,
    )

    assert [row.passage_id for row in rows] == ["p1", "p2"]
    assert np.linalg.norm(rows[0].vector) == pytest.approx(1.0)
    assert np.linalg.norm(rows[1].vector) == pytest.approx(1.0)
    assert calls == [
        [
            "Title: Roadmap\nSource: apple_notes\nPassage:\nFirst exact span.",
            "Title: Roadmap\nSource: apple_notes\nPassage:\nSecond exact span.",
        ]
    ]


def test_embed_passages_rejects_wrong_shape_and_zero_vectors() -> None:
    passage = (_passage("p1", 0, "Exact span."),)
    with pytest.raises(ValueError, match=r"returned \(1, 2\).*expected \(1, 3\)"):
        embed_passages(
            passage,
            title="T",
            source="local_folders",
            embedder=lambda _: np.ones((1, 2), dtype=np.float32),
            model_id="fake",
            build_id="build-1",
            dim=3,
        )
    with pytest.raises(ValueError, match="zero-norm"):
        embed_passages(
            passage,
            title="T",
            source="local_folders",
            embedder=lambda _: np.zeros((1, 3), dtype=np.float32),
            model_id="fake",
            build_id="build-1",
            dim=3,
        )


def test_store_round_trips_embeddings_and_passage_level_bm25(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        note = _note()
        passages = (
            _passage("p1", 0, "Maya owns the launch checklist."),
            _passage("p2", 1, "The review moved to July 18."),
        )
        store.write_note(note, build_id="build-1", extraction_status="complete")
        store.write_passages(
            note.note_id,
            passages,
            created=note.created,
            modified=note.modified,
            build_id="build-1",
        )
        embedded = embed_passages(
            passages,
            title=note.title,
            source=note.source,
            embedder=lambda _: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            model_id="fake-v1",
            build_id="build-1",
            dim=2,
        )
        store.write_passage_embeddings(embedded)

        stored = store.passage_embeddings()
        hits = store.search_passages_bm25("July review", 5)

        assert [row.passage_id for row in stored] == ["p1", "p2"]
        assert np.array_equal(stored[1].vector, np.array([0.0, 1.0], dtype=np.float32))
        assert hits[0].passage_id == "p2"
        assert hits[0].raw_score > 0.0
        assert hits[0].token_coverage == pytest.approx(1.0)

        partial_hits = store.search_passages_bm25("July missingterm", 5)
        assert [row.passage_id for row in partial_hits] == ["p2"]
        assert partial_hits[0].token_coverage == pytest.approx(0.5)

        store.delete_note(note.note_id)
        assert store.passage_embeddings() == ()
    finally:
        store.close()


def _note() -> NoteRecord:
    return NoteRecord(
        note_id="note-1",
        title="Launch roadmap",
        source="apple_notes",
        created=1.0,
        modified=2.0,
        content_hash="note-hash",
        text="Maya owns the launch checklist. The review moved to July 18.",
    )


def _passage(passage_id: str, ordinal: int, text: str) -> PassageDraft:
    return PassageDraft(
        passage_id=passage_id,
        ordinal=ordinal,
        char_start=ordinal * 40,
        char_end=ordinal * 40 + len(text),
        token_count=len(text.split()),
        text=text,
        content_hash=f"hash-{passage_id}",
    )
