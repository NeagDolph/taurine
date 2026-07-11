"""Item 4 — contextual enrichment of episodic passages.

Contexts are retrieval inputs only: they reach the FTS context column and
the embedding frame, never the stored passage text, offsets, or evidence
rendering. Everything fails open to empty context.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from taurine_core import config
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.indexer.build_cache import BuildCache
from taurine_core.indexer.context_enrich import (
    EnrichmentBatch,
    build_enrichment_batches,
    context_enricher_for,
    enrich_passages,
)
from taurine_core.indexer.embed import embed_passages
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.passages import PassageDraft
from taurine_core.schemas import NoteRecord


def _note(
    note_id: str = "note-1",
    *,
    source: str = "imessage",
    title: str = "Group chat with Adi and Ernest",
    text: str = "",
) -> NoteRecord:
    body = text or "Adi: potluck saturday?\nErnest: sounds good, Tuesday works.\n" * 8
    return NoteRecord(
        note_id=note_id,
        title=title,
        source=source,
        created=1_750_000_000.0,
        modified=1_750_100_000.0,
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        text=body,
    )


def _draft(passage_id: str, ordinal: int, text: str, *, char_start: int = 0) -> PassageDraft:
    return PassageDraft(
        passage_id=passage_id,
        ordinal=ordinal,
        char_start=char_start,
        char_end=char_start + len(text),
        token_count=len(text.split()),
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def _drafts(count: int) -> tuple[PassageDraft, ...]:
    rows = []
    start = 0
    for index in range(count):
        text = f"passage body {index} sounds good tuesday works"
        rows.append(_draft(f"p{index}", index, text, char_start=start))
        start += len(text) + 1
    return tuple(rows)


def _response(batch: EnrichmentBatch, prefix: str = "ctx") -> dict:
    return {
        "contexts": {
            str(row.ordinal): f"{prefix} for {row.passage_id}" for row in batch.passages
        }
    }


# -- batching ----------------------------------------------------------------


def test_batches_group_by_note_cap_size_and_share_one_window() -> None:
    note = _note()
    drafts = _drafts(12)

    batches = build_enrichment_batches(note, drafts)

    assert [len(batch.passages) for batch in batches] == [10, 2]
    assert all(batch.note_id == note.note_id for batch in batches)
    # the surrounding window is supplied once per call, not per passage
    assert batches[0].window in note.text.replace("…", "")
    assert batches[0].title == note.title
    assert batches[0].date == "2025-06-15"
    assert [row.ordinal for row in batches[0].passages] == list(range(10))


def test_non_episodic_sources_and_empty_notes_produce_no_batches() -> None:
    assert build_enrichment_batches(_note(source="apple_notes"), _drafts(2)) == ()
    assert build_enrichment_batches(_note(source="obsidian"), _drafts(2)) == ()
    assert build_enrichment_batches(_note(), ()) == ()
    for source in ("imessage", "gmail", "outlook"):
        assert build_enrichment_batches(_note(source=source), _drafts(1))


def test_overlong_window_keeps_head_and_tail() -> None:
    body = "start marker " + ("filler word " * 2000) + "end marker"
    note = _note(text=body)
    # a batch spanning the whole long note gets a head+tail window
    draft = _draft("p0", 0, body)

    (batch,) = build_enrichment_batches(note, (draft,))

    assert len(batch.window) <= 6100
    assert batch.window.startswith("start marker")
    assert batch.window.endswith("end marker")
    assert "…" in batch.window


# -- enrichment + parsing ----------------------------------------------------


def test_enrich_parses_contexts_scoped_to_episodic_sources() -> None:
    calls: list[EnrichmentBatch] = []

    def enricher(batch: EnrichmentBatch) -> dict:
        calls.append(batch)
        return _response(batch)

    result = enrich_passages(
        (
            (_note("a", source="imessage"), _drafts(2)),
            (_note("b", source="apple_notes"), _drafts(2)),
        ),
        enricher,
    )

    assert len(calls) == 1  # apple_notes never reaches the LLM
    assert result.contexts == {"p0": "ctx for p0", "p1": "ctx for p1"}
    assert result.enriched_count == 2
    assert result.failed_batches == 0


def test_contexts_are_whitespace_normalized_and_clamped_to_240_chars() -> None:
    def enricher(batch: EnrichmentBatch) -> dict:
        return {"contexts": {"0": "  a\nlong   context " + "x" * 400}}

    result = enrich_passages(((_note(), _drafts(1)),), enricher)

    context = result.contexts["p0"]
    assert context.startswith("a long context")
    assert len(context) == 240


def test_failed_and_malformed_batches_fail_open_and_are_counted() -> None:
    note = _note()
    drafts = _drafts(12)  # two batches

    def enricher(batch: EnrichmentBatch) -> dict:
        if batch.passages[0].ordinal == 0:
            raise RuntimeError("provider down")
        return {"not-contexts": []}

    result = enrich_passages(((note, drafts),), enricher)

    assert result.contexts == {}
    assert result.enriched_count == 0
    assert result.failed_batches == 2


def test_absent_enricher_yields_empty_contexts() -> None:
    result = enrich_passages(((_note(), _drafts(3)),), None)

    assert result.contexts == {}
    assert result.enriched_count == 0
    assert result.failed_batches == 0


def test_unknown_ordinals_and_non_string_contexts_are_ignored() -> None:
    def enricher(batch: EnrichmentBatch) -> dict:
        return {"contexts": {"0": "good context", "9": "phantom", "1": 42, "x": "bad"}}

    result = enrich_passages(((_note(), _drafts(2)),), enricher)

    assert result.contexts == {"p0": "good context"}
    assert result.failed_batches == 0


# -- caching -----------------------------------------------------------------


def test_cache_skips_llm_for_unchanged_passages(tmp_path: Path) -> None:
    cache = BuildCache(tmp_path / "cache.db")
    note = _note()
    drafts = _drafts(3)
    calls: list[EnrichmentBatch] = []

    def enricher(batch: EnrichmentBatch) -> dict:
        calls.append(batch)
        return _response(batch)

    try:
        first = enrich_passages(((note, drafts),), enricher, cache=cache)
        second = enrich_passages(((note, drafts),), enricher, cache=cache)
    finally:
        cache.close()

    assert len(calls) == 1
    assert first.contexts == second.contexts
    assert second.enriched_count == 3


def test_cache_key_covers_model_content_and_window(tmp_path: Path) -> None:
    cache = BuildCache(tmp_path / "cache.db")
    note = _note()
    drafts = _drafts(2)
    calls: list[EnrichmentBatch] = []

    def enricher(batch: EnrichmentBatch) -> dict:
        calls.append(batch)
        return _response(batch)

    try:
        enrich_passages(((note, drafts),), enricher, cache=cache)
        # different model: full re-enrichment
        enrich_passages(((note, drafts),), enricher, cache=cache, model_id="other-model")
        assert len(calls) == 2
        # changed surrounding text (window) with identical passage content
        moved = _note(text="totally different surrounding thread text " * 40)
        enrich_passages(((moved, drafts),), enricher, cache=cache)
        assert len(calls) == 3
        # one changed passage re-enriches only that passage
        changed = (drafts[0], _draft("p1-new", 1, "brand new text", char_start=drafts[1].char_start))
        enrich_passages(((note, changed),), enricher, cache=cache)
    finally:
        cache.close()

    assert len(calls) == 4
    assert [row.passage_id for row in calls[3].passages] == ["p1-new"]


def test_failed_batches_are_never_cached(tmp_path: Path) -> None:
    cache = BuildCache(tmp_path / "cache.db")
    note = _note()
    drafts = _drafts(1)
    attempts: list[int] = []

    def flaky(batch: EnrichmentBatch) -> dict:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("timeout")
        return _response(batch)

    try:
        first = enrich_passages(((note, drafts),), flaky, cache=cache)
        second = enrich_passages(((note, drafts),), flaky, cache=cache)
    finally:
        cache.close()

    assert first.contexts == {}
    assert first.failed_batches == 1
    assert second.contexts == {"p0": "ctx for p0"}
    assert len(attempts) == 2


# -- the production enricher callable -----------------------------------------


def test_context_enricher_for_calls_complete_json_with_enrich_knobs() -> None:
    recorded: dict = {}

    class FakeLLM:
        async def complete_json(self, model, system, user, timeout_s=None, temperature=None):
            recorded.update(
                model=model, system=system, user=user, timeout_s=timeout_s, temperature=temperature
            )
            return {"contexts": {"0": "june 2026 potluck chat"}}

    (batch,) = build_enrichment_batches(_note(), _drafts(1))
    response = context_enricher_for(FakeLLM())(batch)

    assert response == {"contexts": {"0": "june 2026 potluck chat"}}
    assert recorded["model"] == config.CONTEXT_ENRICH_MODEL
    assert recorded["temperature"] == 0.0
    assert recorded["timeout_s"] == config.SUMMARY_TIMEOUT_S
    assert "never invent" in recorded["system"]
    import json

    payload = json.loads(recorded["user"])
    assert payload["window"] == batch.window
    assert payload["title"] == batch.title
    assert payload["passages"] == [{"ordinal": 0, "text": batch.passages[0].text}]


# -- embedding frame ----------------------------------------------------------


def test_embed_frame_includes_context_only_when_nonempty() -> None:
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.eye(len(texts), 3, dtype=np.float32) + 0.1

    embed_passages(
        (_draft("p1", 0, "First exact span."), _draft("p2", 1, "Second exact span.")),
        title="Roadmap",
        source="imessage",
        contexts={"p1": "From a June 2026 potluck chat.", "p2": ""},
        embedder=fake_embed,
        model_id="fake-v1",
        build_id="build-1",
        dim=3,
    )

    assert calls == [
        [
            "Title: Roadmap\nSource: imessage\nContext: From a June 2026 potluck chat.\n"
            "Passage:\nFirst exact span.",
            "Title: Roadmap\nSource: imessage\nPassage:\nSecond exact span.",
        ]
    ]


def test_embedding_cache_key_incorporates_context(tmp_path: Path) -> None:
    """The BuildCache embedder keys on the framed text, so a new context must
    re-embed while an unchanged (or absent) context stays a cache hit."""

    cache = BuildCache(tmp_path / "cache.db")
    inner_calls: list[list[str]] = []

    def inner(texts: list[str]) -> np.ndarray:
        inner_calls.append(list(texts))
        return np.full((len(texts), 3), 0.5, dtype=np.float32)

    embedder = cache.caching_embedder("fake-v1", inner)
    draft = (_draft("p1", 0, "Exact span."),)
    kwargs = dict(
        title="T", source="imessage", embedder=embedder,
        model_id="fake-v1", build_id="build-1", dim=3,
    )
    try:
        embed_passages(draft, **kwargs)
        embed_passages(draft, **kwargs)  # unchanged: cache hit
        embed_passages(draft, contexts={"p1": "situating prefix"}, **kwargs)
        embed_passages(draft, contexts={"p1": "situating prefix"}, **kwargs)  # hit again
        embed_passages(draft, contexts={"p1": "different prefix"}, **kwargs)
    finally:
        cache.close()

    assert len(inner_calls) == 3
    assert all(len(texts) == 1 for texts in inner_calls)
    assert {text.count("Context:") for texts in inner_calls[1:] for text in texts} == {1}


# -- knowledge store ----------------------------------------------------------


def test_store_context_feeds_fts_and_coverage_but_not_passage_text(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        note = _note()
        drafts = _drafts(2)
        store.write_note(note, build_id="build-1", extraction_status="complete")
        store.write_passages(
            note.note_id, drafts, created=note.created, modified=note.modified, build_id="build-1"
        )
        assert store.search_passages_bm25("zanzibar", 5) == ()

        store.set_passage_contexts({"p0": "Planning the Zanzibar potluck with Adi."})

        hits = store.search_passages_bm25("zanzibar", 5)
        assert [hit.passage_id for hit in hits] == ["p0"]
        assert hits[0].token_coverage == pytest.approx(1.0)
        # stored passage text, offsets, and evidence rendering are unchanged
        (context_row,) = store.passage_contexts(["p0"])
        assert context_row.text == drafts[0].text
        assert (context_row.char_start, context_row.char_end) == (
            drafts[0].char_start,
            drafts[0].char_end,
        )
        row = store._conn.execute(
            "SELECT text, context FROM passages WHERE passage_id = 'p0'"
        ).fetchone()
        assert row["text"] == drafts[0].text
        assert row["context"] == "Planning the Zanzibar potluck with Adi."
        # body-term search still works and un-enriched passages stay empty
        assert store.search_passages_bm25("tuesday", 5)
        other = store._conn.execute(
            "SELECT context FROM passages WHERE passage_id = 'p1'"
        ).fetchone()
        assert other["context"] == ""
    finally:
        store.close()


def test_note_delete_removes_context_fts_rows(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        note = _note()
        store.write_note(note, build_id="build-1", extraction_status="complete")
        store.write_passages(
            note.note_id, _drafts(1), created=note.created, modified=note.modified,
            build_id="build-1",
        )
        store.set_passage_contexts({"p0": "zanzibar context"})
        store.delete_note(note.note_id)
        assert store.search_passages_bm25("zanzibar", 5) == ()
    finally:
        store.close()


# -- real wiring through BuildCoordinator -------------------------------------


class _Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        import re

        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def _hash_embedder(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        row = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) + 1
        row /= np.linalg.norm(row)
        rows.append(row)
    return np.vstack(rows)


class _FakeContextLLM:
    """Deterministic fake for the real context_enricher_for wiring."""

    async def complete_json(self, model, system, user, timeout_s=None, temperature=None):
        import json

        payload = json.loads(user)
        return {
            "contexts": {
                str(row["ordinal"]): f"June 2026 xyzzygram thread about ordinal {row['ordinal']}"
                for row in payload["passages"]
            }
        }


def test_build_enriches_episodic_notes_and_bm25_finds_context_terms(tmp_path: Path) -> None:
    notes = [
        NoteRecord(
            note_id="chat-1",
            title="Potluck chat",
            source="imessage",
            created=1_750_000_000.0,
            modified=1_750_100_000.0,
            content_hash="chat-hash",
            text="sounds good, Tuesday works for the potluck plan.",
        ),
        NoteRecord(
            note_id="doc-1",
            title="Quarterly doc",
            source="apple_notes",
            created=1_750_000_000.0,
            modified=1_750_100_000.0,
            content_hash="doc-hash",
            text="The quarterly roadmap review moved to July.",
        ),
    ]
    cache = BuildCache(tmp_path / "cache.db")
    deps = BuildDependencies(
        tokenizer=_Tokenizer(),
        embedder=cache.caching_embedder("fake", _hash_embedder),
        embedding_dim=16,
        context_enricher=context_enricher_for(_FakeContextLLM()),
        context_cache=cache,
    )
    target = tmp_path / "knowledge.db"

    report = BuildCoordinator(deps).build(notes, target, replace_derived=True)
    cache.close()

    assert "context_enrichment" in report.stages
    assert report.context_enriched_count == 1  # the imessage passage only
    assert report.context_enrich_failed_batches == 0

    store = KnowledgeStore(target, read_only=True)
    try:
        # BM25 on a context-only term finds the enriched passage
        hits = store.search_passages_bm25("xyzzygram", 5)
        assert len(hits) == 1
        (passage,) = store.passage_contexts([hits[0].passage_id])
        assert passage.note_id == "chat-1"
        # stored passage text is source-exact — the context never leaks in
        assert passage.text == notes[0].text
        assert "xyzzygram" not in passage.text
        rows = store._conn.execute(
            "SELECT note_id, context FROM passages ORDER BY note_id"
        ).fetchall()
        by_note = {row["note_id"]: row["context"] for row in rows}
        assert "xyzzygram" in by_note["chat-1"]
        assert by_note["doc-1"] == ""
    finally:
        store.close()


def test_build_without_enricher_matches_todays_behavior(tmp_path: Path) -> None:
    notes = [
        NoteRecord(
            note_id="chat-1",
            title="Potluck chat",
            source="imessage",
            created=1_750_000_000.0,
            modified=1_750_100_000.0,
            content_hash="chat-hash",
            text="sounds good, Tuesday works for the potluck plan.",
        )
    ]
    deps = BuildDependencies(tokenizer=_Tokenizer(), embedder=_hash_embedder, embedding_dim=16)

    report = BuildCoordinator(deps).build(notes, tmp_path / "k.db", replace_derived=True)

    assert report.context_enriched_count == 0
    assert report.context_enrich_failed_batches == 0
    store = KnowledgeStore(tmp_path / "k.db", read_only=True)
    try:
        assert store.search_passages_bm25("tuesday", 5)
    finally:
        store.close()


def test_failing_enricher_degrades_to_empty_context_and_counts(tmp_path: Path) -> None:
    notes = [
        NoteRecord(
            note_id="chat-1",
            title="Potluck chat",
            source="imessage",
            created=1_750_000_000.0,
            modified=1_750_100_000.0,
            content_hash="chat-hash",
            text="sounds good, Tuesday works for the potluck plan.",
        )
    ]

    def broken(batch: EnrichmentBatch) -> dict:
        raise RuntimeError("no provider")

    deps = BuildDependencies(
        tokenizer=_Tokenizer(),
        embedder=_hash_embedder,
        embedding_dim=16,
        context_enricher=broken,
    )

    report = BuildCoordinator(deps).build(notes, tmp_path / "k.db", replace_derived=True)

    assert report.context_enriched_count == 0
    assert report.context_enrich_failed_batches == 1
    assert report.build_id  # build completed normally


def test_context_changes_embeddings_via_cache_key(tmp_path: Path) -> None:
    """Same passage, enriched vs not: the framed text differs, so the vector
    differs — proof the embedding input (and cache key) carries the context."""

    note = NoteRecord(
        note_id="chat-1",
        title="Potluck chat",
        source="imessage",
        created=1_750_000_000.0,
        modified=1_750_100_000.0,
        content_hash="chat-hash",
        text="sounds good, Tuesday works for the potluck plan.",
    )
    base = BuildDependencies(tokenizer=_Tokenizer(), embedder=_hash_embedder, embedding_dim=16)
    enriched = BuildDependencies(
        tokenizer=_Tokenizer(),
        embedder=_hash_embedder,
        embedding_dim=16,
        context_enricher=context_enricher_for(_FakeContextLLM()),
    )

    BuildCoordinator(base).build([note], tmp_path / "plain.db", replace_derived=True)
    BuildCoordinator(enriched).build([note], tmp_path / "rich.db", replace_derived=True)

    plain = KnowledgeStore(tmp_path / "plain.db", read_only=True)
    rich = KnowledgeStore(tmp_path / "rich.db", read_only=True)
    try:
        (plain_vec,) = plain.passage_embeddings()
        (rich_vec,) = rich.passage_embeddings()
        assert not np.array_equal(plain_vec.vector, rich_vec.vector)
    finally:
        plain.close()
        rich.close()
