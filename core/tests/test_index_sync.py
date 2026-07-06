from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.ports import LLMError

NOTE_A = make_note(
    "note-a",
    "Alpha Plan",
    "Alpha rollout first line.\nDetails about the billing launch.",
)


def source_with(*notes):
    return lambda: list(notes)


class StubLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete_json(self, model: str, system: str, user: str) -> dict:
        self.calls.append(user)
        return {
            "gist": "Alpha rollout and billing launch.",
            "utterances": [
                "Can we talk about the alpha rollout?",
                "What is happening with billing launch?",
                "Do we have the alpha plan?",
                "Where did the rollout land?",
            ],
            "aliases": ["alpha plan", "billing launch"],
            "topics": ["planning", "billing"],
            "entities": ["Alpha"],
        }


class FailingLLM:
    async def complete_json(self, model: str, system: str, user: str) -> dict:
        raise LLMError("unavailable")


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def failing_llm() -> FailingLLM:
    return FailingLLM()


@pytest.fixture(autouse=True)
def deterministic_sync_embeddings(monkeypatch):
    from taurine_core.indexer import sync

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            row = rng.normal(size=config.EMBED_DIM).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row)
        return np.vstack(rows) if rows else np.empty((0, config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr(sync, "embed_texts", fake_embed_texts)


def test_unchanged_hash_skips_enrichment(store, stub_llm):
    from taurine_core.indexer.sync import sync_once

    sync_once(store, source_with(NOTE_A), stub_llm)
    stub_llm.calls.clear()
    sync_once(store, source_with(NOTE_A), stub_llm)   # same content
    assert stub_llm.calls == []


def test_enrich_failure_still_indexes(store, failing_llm):
    from taurine_core.indexer.sync import sync_once

    report = sync_once(store, source_with(NOTE_A), failing_llm)
    assert report.enrich_failures == 1
    assert store.note(NOTE_A.note_id) is not None
    assert store.bm25_search(NOTE_A.title, 5)          # searchable on raw text


def test_changed_hash_reenriches_and_reports_changed(store, stub_llm):
    from taurine_core.indexer.sync import sync_once

    changed = make_note(
        NOTE_A.note_id,
        NOTE_A.title,
        NOTE_A.text + "\nA new launch detail.",
    )

    first = sync_once(store, source_with(NOTE_A), stub_llm)
    second = sync_once(store, source_with(changed), stub_llm)

    assert first.added == 1
    assert second.changed == 1
    assert len(stub_llm.calls) == 2
    assert store.note(NOTE_A.note_id).content_hash == changed.content_hash


def test_removed_note_is_deleted(store, stub_llm):
    from taurine_core.indexer.sync import sync_once

    sync_once(store, source_with(NOTE_A), stub_llm)
    report = sync_once(store, source_with(), stub_llm)

    assert report.removed == 1
    assert store.note(NOTE_A.note_id) is None
