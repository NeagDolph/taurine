"""Content-keyed build caches: repeat syncs only pay for changed notes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tests.conftest import make_note
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.indexer.build_cache import BuildCache
from taurine_core.schemas import Extraction, ExtractedMention


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
