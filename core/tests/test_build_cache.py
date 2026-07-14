"""Content-keyed build caches: repeat syncs only pay for changed notes."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.indexer import build as build_module
from taurine_core.indexer import build_cache as build_cache_module
from taurine_core.indexer import extract as extract_module
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.indexer.build_cache import BuildCache
from taurine_core.indexer.passages import PassageDraft
from taurine_core.schemas import Extraction, ExtractedMention, ExtractedRelation


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        import re

        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def _embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        row = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) + 1
        row /= np.linalg.norm(row)
        rows.append(row)
    return np.vstack(rows)


def test_caching_embedder_embeds_each_text_once(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def inner(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return _embed(texts)

    cache = BuildCache(tmp_path / "cache.db")
    try:
        embed = cache.caching_embedder("model-x", inner)
        first = embed(["alpha", "beta", "gamma"])
        assert calls == [["alpha", "beta", "gamma"]]

        second = embed(["alpha", "beta", "gamma"])
        assert calls == [["alpha", "beta", "gamma"]]  # all hits, inner untouched
        np.testing.assert_allclose(first, second)

        mixed = embed(["beta", "delta"])
        assert calls[-1] == ["delta"]  # only the miss is embedded
        np.testing.assert_allclose(mixed[0], first[1])
    finally:
        cache.close()


def test_caching_embedder_is_model_scoped(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def inner(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return _embed(texts)

    cache = BuildCache(tmp_path / "cache.db")
    try:
        cache.caching_embedder("model-a", inner)(["alpha"])
        cache.caching_embedder("model-b", inner)(["alpha"])
        assert calls == [["alpha"], ["alpha"]]
    finally:
        cache.close()


def test_caching_extractor_reuses_unchanged_notes(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def inner(notes, *, tokenizer=None, progress=None):
        del tokenizer
        calls.append([note.note_id for note in notes])
        if progress is not None:
            progress(len(notes), len(notes))
        return {
            note.note_id: Extraction(
                mentions=[
                    ExtractedMention(
                        surface="Maya",
                        etype="person",
                        confidence=0.9,
                        char_start=0,
                        char_end=4,
                        extractor="fake",
                        schema_id="s1",
                    )
                ],
                schema_id="s1",
            )
            for note in notes
        }

    note_a = make_note("a", "Alpha", "Maya was here.")
    note_b = make_note("b", "Beta", "Maya was there.")
    cache = BuildCache(tmp_path / "cache.db")
    try:
        extract = cache.caching_extractor(inner)
        ticks: list[tuple[int, int]] = []
        first = extract([note_a, note_b], progress=lambda done, total: ticks.append((done, total)))
        assert calls == [["a", "b"]]
        assert set(first) == {"a", "b"}
        assert ticks[-1] == (2, 2)

        second = extract([note_a, note_b])
        assert calls == [["a", "b"]]  # both cached
        assert second["a"].mentions[0].surface == "Maya"

        changed = make_note("a", "Alpha", "Maya left.")  # new content hash
        extract([changed, note_b])
        assert calls[-1] == ["a"]
    finally:
        cache.close()


def test_caching_extractor_never_caches_failures(tmp_path: Path) -> None:
    def inner(notes, *, tokenizer=None, progress=None):
        del tokenizer, progress
        return {note.note_id: Extraction(failed=True, error="boom") for note in notes}

    note = make_note("a", "Alpha", "text")
    cache = BuildCache(tmp_path / "cache.db")
    try:
        extract = cache.caching_extractor(inner)
        assert extract([note])["a"].failed is True
        rows = cache._extraction_rows(["ignored"])
        assert rows == {}
        count = cache._conn.execute("SELECT count(*) FROM extraction_cache").fetchone()[0]
        assert count == 0
    finally:
        cache.close()


def test_extraction_cache_key_covers_schema_thresholds_and_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H5: every real extraction input must be part of the cache key."""
    note = make_note("a", "Alpha", "Maya was here.")

    def key() -> str:
        return build_cache_module._extraction_key(
            note, build_cache_module._schema_fingerprint()
        )

    baseline = key()
    assert key() == baseline  # deterministic

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            config,
            "INGEST_RELATION_TYPES",
            {**config.INGEST_RELATION_TYPES, "works_at": "{head} is employed by {tail}"},
        )
        assert key() != baseline

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            config,
            "INGEST_ENTITY_TYPES",
            {**config.INGEST_ENTITY_TYPES, "person": "changed description"},
        )
        assert key() != baseline

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "GLINER_CHUNK_SIZE", 123)
        assert key() != baseline

    artifact = extract_module._calibration()
    updated = artifact.model_copy(
        update={
            "extraction": artifact.extraction.model_copy(
                update={
                    "promotion_thresholds": {
                        **artifact.extraction.promotion_thresholds,
                        "works_at": 0.99,
                    }
                }
            )
        }
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(extract_module, "_calibration", lambda: updated)
        assert key() != baseline


def test_extraction_cache_fingerprint_scopes_relex_backend_and_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "EXTRACTOR_BACKEND", "gliner2")
    gliner2 = build_cache_module._schema_fingerprint()

    monkeypatch.setattr(config, "RELEX_RELATION_THRESHOLD", 0.91)
    assert build_cache_module._schema_fingerprint() == gliner2

    monkeypatch.setattr(config, "EXTRACTOR_BACKEND", "relex")
    relex = build_cache_module._schema_fingerprint()
    assert relex != gliner2

    monkeypatch.setattr(config, "RELEX_ENTITY_THRESHOLD", 0.35)
    assert build_cache_module._schema_fingerprint() != relex

    monkeypatch.setattr(config, "EXTRACTOR_BACKEND", "hybrid")
    assert build_cache_module._schema_fingerprint() not in {gliner2, relex}


def test_corrupt_extraction_cache_row_degrades_to_miss(tmp_path: Path) -> None:
    """L8: one bad row costs one recompute, never the build."""
    calls: list[list[str]] = []

    def inner(notes, *, tokenizer=None, progress=None):
        del tokenizer, progress
        calls.append([note.note_id for note in notes])
        return {note.note_id: Extraction(schema_id="s1") for note in notes}

    note_a = make_note("a", "Alpha", "Maya was here.")
    note_b = make_note("b", "Beta", "Maya was there.")
    cache = BuildCache(tmp_path / "cache.db")
    try:
        extract = cache.caching_extractor(inner)
        extract([note_a, note_b])
        assert calls == [["a", "b"]]

        cache._conn.execute(
            "UPDATE extraction_cache SET payload = '{not json' WHERE note_id = 'a'"
        )
        cache._conn.commit()

        result = extract([note_a, note_b])
        assert calls[-1] == ["a"]  # only the corrupt row recomputes
        assert set(result) == {"a", "b"}
        assert cache.stats["cache_corrupt"] == 1

        extract([note_a, note_b])
        assert calls[-1] == ["a"]  # repaired row is a hit again
        assert cache.stats["cache_corrupt"] == 1
    finally:
        cache.close()


def test_corrupt_embedding_cache_row_degrades_to_miss(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def inner(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return _embed(texts)

    cache = BuildCache(tmp_path / "cache.db")
    try:
        embed = cache.caching_embedder("model-x", inner)
        first = embed(["alpha", "beta"])
        cache._conn.execute(
            "UPDATE embedding_cache SET vec = X'00' WHERE rowid = 1"
        )
        cache._conn.commit()

        second = embed(["alpha", "beta"])
        assert cache.stats["cache_corrupt"] == 1
        assert len(calls) == 2 and len(calls[-1]) == 1  # only the bad row re-embeds
        np.testing.assert_allclose(first, second)
    finally:
        cache.close()


def test_passage_for_span_accepts_adjacent_overlapping_pair() -> None:
    """L5: a span straddling two adjacent overlapping passages keeps its
    evidence, recorded against the earlier passage."""

    def draft(passage_id: str, ordinal: int, start: int, end: int) -> PassageDraft:
        return PassageDraft(
            passage_id=passage_id,
            ordinal=ordinal,
            char_start=start,
            char_end=end,
            token_count=10,
            text="x" * (end - start),
            content_hash="h",
        )

    drafts = [draft("p0", 0, 0, 100), draft("p1", 1, 80, 200), draft("p3", 3, 300, 400)]

    assert build_module._passage_for_span(drafts, 10, 50) == (drafts[0], False)
    # p1 alone covers 80-200; a span crossing 80 while ending past 100 needs the pair
    assert build_module._passage_for_span(drafts, 70, 150) == (drafts[0], True)
    assert build_module._passage_for_span(drafts, 150, 350) == (None, False)
    assert build_module._passage_for_span(drafts, 70, 250) == (None, False)


def test_same_span_dual_etype_endpoint_binds_deterministically(tmp_path: Path) -> None:
    """L6: named types outrank concept; remaining ties break alphabetically."""
    note = make_note("n1", "Work", "Jane works at Acme today.")

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(surface="Jane", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="Acme", etype="concept", confidence=0.95, char_start=14, char_end=18, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="Acme", etype="product", confidence=0.9, char_start=14, char_end=18, extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[
                    ExtractedRelation(head="Jane", predicate="works_at", tail="Acme", confidence=0.9, subject_start=0, subject_end=4, object_start=14, object_end=18, assertion_start=0, assertion_end=25, extraction_method="fake", schema_id="s1", status="active")
                ],
                schema_id="s1",
            )
        }

    target = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16, extractor=extractor
        )
    ).build([note], target, replace_derived=True)

    conn = sqlite3.connect(target)
    try:
        rows = conn.execute(
            "SELECT e.etype FROM facts f JOIN entities e"
            " ON e.entity_id = f.object_entity_id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("product",)]


def test_second_clean_build_reuses_cached_embeddings(tmp_path: Path) -> None:
    calls: list[int] = []

    def inner(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return _embed(texts)

    notes = [
        make_note("a", "Alpha", "Launch is scheduled for July."),
        make_note("b", "Beta", "Maya owns Launch."),
    ]
    cache = BuildCache(tmp_path / "cache.db")
    try:
        deps = BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=cache.caching_embedder("model-x", inner),
            embedding_dim=16,
        )
        BuildCoordinator(deps).build(notes, tmp_path / "first.db", replace_derived=True)
        embedded_first = sum(calls)
        assert embedded_first > 0

        BuildCoordinator(deps).build(notes, tmp_path / "second.db", replace_derived=True)
        assert sum(calls) == embedded_first  # nothing re-embedded
    finally:
        cache.close()
