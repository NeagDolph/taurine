from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.ports import LLMError
from taurine_core.schemas import Extraction

NOTE_A = make_note(
    "note-a",
    "Alpha Plan",
    "Alpha rollout first line.\nDetails about the billing launch.",
)


def source_with(*notes):
    return lambda: list(notes)


class _Schema:
    def entities(self, _types: dict[str, str]) -> "_Schema":
        return self

    def relations(self, _types: list[str]) -> "_Schema":
        return self


class FakeExtractorModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create_schema(self) -> _Schema:
        return _Schema()

    def batch_extract(
        self,
        texts: list[str],
        schema: object,
        *,
        batch_size: int,
        threshold: float,
        include_confidence: bool,
        max_len: int | None,
    ) -> list[dict]:
        del schema, batch_size, threshold, include_confidence, max_len
        self.calls.extend(texts)
        return [_raw_extraction_for(text) for text in texts]


class SlimLLM:
    def __init__(self) -> None:
        self.slim_calls: list[str] = []
        self.adjudication_calls = 0

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        del system, timeout_s
        if model == config.ADJUDICATE_MODEL:
            self.adjudication_calls += 1
            return {"pairs": {}}
        self.slim_calls.append(user)
        return {
            "notes": {
                "note-a": {
                    "aliases": ["billing launch"],
                    "utterances": ["Can we talk through the billing launch?"],
                }
            }
        }


class FailingLLM:
    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        del model, system, user, timeout_s
        raise LLMError("unavailable")


@pytest.fixture
def fake_extractor(monkeypatch) -> FakeExtractorModel:
    from taurine_core.indexer import extract

    model = FakeExtractorModel()
    monkeypatch.setattr(extract, "_extractor_model", lambda: model)
    return model


@pytest.fixture(autouse=True)
def deterministic_sync_embeddings(monkeypatch):
    from taurine_core.indexer import sync

    monkeypatch.setattr(sync, "embed_texts", _deterministic_vectors)


def _deterministic_vectors(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        row = rng.normal(size=config.EMBED_DIM).astype(np.float32)
        row /= np.linalg.norm(row)
        rows.append(row)
    return np.vstack(rows) if rows else np.empty((0, config.EMBED_DIM), dtype=np.float32)


def _raw_extraction_for(text: str) -> dict:
    title = text.splitlines()[0]
    if "Alpha" in title:
        return {
            "entities": {
                "project": [{"text": "Alpha Plan", "confidence": 0.96}],
                "concept": [{"text": "billing launch", "confidence": 0.91}],
                "organization": [{"text": "Stripe", "confidence": 0.88}],
            },
            "relation_extraction": {
                "about": [
                    {
                        "head": {"text": "Alpha Plan", "confidence": 0.96},
                        "tail": {"text": "billing launch", "confidence": 0.91},
                        "confidence": 0.89,
                    }
                ]
            },
        }
    return {
        "entities": {"project": [{"text": title, "confidence": 0.9}]},
        "relation_extraction": {},
    }


def test_unchanged_hash_skips_extraction_and_slim(store, fake_extractor):
    from taurine_core.indexer.sync import sync_once

    llm = SlimLLM()
    sync_once(store, source_with(NOTE_A), llm)
    fake_extractor.calls.clear()
    llm.slim_calls.clear()

    sync_once(store, source_with(NOTE_A), llm)

    assert fake_extractor.calls == []
    assert llm.slim_calls == []


def test_extraction_failure_still_indexes_and_retries_next_sync(store, monkeypatch):
    from taurine_core.indexer import extract
    from taurine_core.indexer.sync import sync_once

    monkeypatch.setattr(extract, "_extractor_model", lambda: None)
    report = sync_once(store, source_with(NOTE_A), None)  # type: ignore[arg-type]

    assert report.enrich_failures == 1
    assert store.note(NOTE_A.note_id) is not None
    assert store.bm25_search(NOTE_A.title, 5)
    assert store.cached_extraction(NOTE_A.note_id, NOTE_A.content_hash) is None

    model = FakeExtractorModel()
    monkeypatch.setattr(extract, "_extractor_model", lambda: model)
    retry = sync_once(store, source_with(NOTE_A), None)  # type: ignore[arg-type]

    assert retry.changed == 1
    assert model.calls
    assert store.cached_extraction(NOTE_A.note_id, NOTE_A.content_hash) == Extraction(
        entities=store.enrichment_for(NOTE_A.note_id).entities,
        relations=store.enrichment_for(NOTE_A.note_id).relations,
        failed=False,
    )


def test_failed_extraction_with_llm_is_not_slim_cached_and_retries_same_hash(
    store, monkeypatch
):
    from taurine_core.indexer import extract
    from taurine_core.indexer.sync import sync_once

    llm = SlimLLM()
    monkeypatch.setattr(extract, "_extractor_model", lambda: None)

    report = sync_once(store, source_with(NOTE_A), llm)  # type: ignore[arg-type]

    assert report.enrich_failures == 1
    assert llm.slim_calls == []
    assert store.cached_extraction(NOTE_A.note_id, NOTE_A.content_hash) is None
    assert (
        store._conn.execute(
            """
            SELECT COUNT(*) AS count FROM enrich_cache
            WHERE note_id = ? AND content_hash = ?
            """,
            (NOTE_A.note_id, NOTE_A.content_hash),
        ).fetchone()["count"]
        == 0
    )
    assert (
        "billing launch",
        NOTE_A.note_id,
        False,
    ) not in store.aliases()

    model = FakeExtractorModel()
    monkeypatch.setattr(extract, "_extractor_model", lambda: model)
    llm.slim_calls.clear()

    retry = sync_once(store, source_with(NOTE_A), llm)  # type: ignore[arg-type]

    assert retry.changed == 1
    assert model.calls
    assert llm.slim_calls
    assert store.cached_extraction(NOTE_A.note_id, NOTE_A.content_hash) is not None


def test_changed_hash_reextracts_and_reports_changed(store, fake_extractor):
    from taurine_core.indexer.sync import sync_once

    changed = make_note(
        NOTE_A.note_id,
        NOTE_A.title,
        NOTE_A.text + "\nA new launch detail.",
    )

    first = sync_once(store, source_with(NOTE_A), None)  # type: ignore[arg-type]
    fake_extractor.calls.clear()
    second = sync_once(store, source_with(changed), None)  # type: ignore[arg-type]

    assert first.added == 1
    assert second.changed == 1
    assert fake_extractor.calls
    assert store.note(NOTE_A.note_id).content_hash == changed.content_hash


def test_sync_passes_stored_user_card_to_slim_enrichment(store, fake_extractor):
    from taurine_core.indexer.sync import sync_once

    store.set_user_card("## User\nWorks with Stripe.")
    llm = SlimLLM()

    sync_once(store, source_with(NOTE_A), llm)

    assert len(llm.slim_calls) == 1
    assert "## User\nWorks with Stripe." in llm.slim_calls[0]


def test_sync_adds_display_line_heuristic_aliases_and_slim_utterances(
    store, fake_extractor
):
    from taurine_core.indexer.sync import sync_once

    sync_once(store, source_with(NOTE_A), SlimLLM())

    note = store.note(NOTE_A.note_id)
    assert note.gist == "Alpha rollout first line."
    aliases = set(store.aliases())
    assert ("alpha plan", NOTE_A.note_id, True) in aliases
    assert ("billing launch", NOTE_A.note_id, False) in aliases
    assert ("stripe", NOTE_A.note_id, False) not in aliases
    kinds = [kind for _row_id, _note_id, kind, _text in store._embedding_rows]
    assert "gist" not in kinds
    assert "chunk" in kinds
    assert "utterance" in kinds


def test_slim_enrichment_embeds_gated_note_chunks_once_per_sync(
    store, fake_extractor, monkeypatch
):
    from taurine_core.indexer import sync
    from taurine_core.indexer.sync import sync_once

    embed_calls: list[list[str]] = []

    def tracking_embed_texts(texts: list[str]) -> np.ndarray:
        embed_calls.append(list(texts))
        return _deterministic_vectors(texts)

    monkeypatch.setattr(sync, "embed_texts", tracking_embed_texts)

    sync_once(store, source_with(NOTE_A), SlimLLM())

    chunk_texts = sync.chunk_text(NOTE_A.text) or [NOTE_A.text or NOTE_A.title]
    chunk_counts = {
        chunk: sum(batch.count(chunk) for batch in embed_calls) for chunk in chunk_texts
    }
    assert chunk_counts == {chunk: 1 for chunk in chunk_texts}
    assert any("Can we talk through the billing launch?" in batch for batch in embed_calls)


def test_removed_note_is_deleted(store, fake_extractor):
    from taurine_core.indexer.sync import sync_once

    sync_once(store, source_with(NOTE_A), None)  # type: ignore[arg-type]
    report = sync_once(store, source_with(), None)  # type: ignore[arg-type]

    assert report.removed == 1
    assert store.note(NOTE_A.note_id) is None


def test_sync_source_removes_only_notes_from_that_source(store, fake_extractor):
    from taurine_core.indexer.sync import sync_source

    source_a = make_note("source-a:alpha", "Alpha", "Alpha source text").model_copy(
        update={"source": "source-a"}
    )
    source_b = make_note("source-b:beta", "Beta", "Beta source text").model_copy(
        update={"source": "source-b"}
    )

    sync_source(store, "source-a", [source_a], None)  # type: ignore[arg-type]
    sync_source(store, "source-b", [source_b], None)  # type: ignore[arg-type]
    report = sync_source(store, "source-a", [], None)  # type: ignore[arg-type]

    assert report.connector_id == "source-a"
    assert report.removed == 1
    assert store.note(source_a.note_id) is None
    assert store.note(source_b.note_id) is not None


def test_sync_source_tags_records_with_requested_source(store, fake_extractor):
    from taurine_core.indexer.sync import sync_source

    raw = make_note("obsidian:vault/plan.md", "Plan", "Plan body").model_copy(
        update={"source": "wrong-source"}
    )

    report = sync_source(store, "obsidian", [raw], None)  # type: ignore[arg-type]

    assert report.connector_id == "obsidian"
    assert report.added == 1
    assert store.note(raw.note_id).source == "obsidian"
