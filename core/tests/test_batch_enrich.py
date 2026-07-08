from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.ports import LLMError

_PAIR_ID_RE = re.compile(r"\bpair-\d+\b")


def _payload(gist: str) -> dict:
    return {
        "gist": gist,
        "utterances": [f"What about {gist}?"],
        "aliases": [gist],
        "topics": ["Batch Topic"],
        "entities": ["Batch Entity"],
    }


def _notes(count: int):
    return [
        make_note(
            f"note-{index}",
            f"Title {index}",
            f"Body text for title {index}. This note has batchable content.",
        )
        for index in range(count)
    ]


def _hash_embeddings(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        row = rng.normal(size=config.EMBED_DIM).astype(np.float32)
        row /= np.linalg.norm(row)
        rows.append(row)
    return np.vstack(rows) if rows else np.empty((0, config.EMBED_DIM), dtype=np.float32)


class BatchAwareLLM:
    def __init__(
        self,
        note_ids: list[str],
        *,
        missing_ids: set[str] | None = None,
        batch_errors: int = 0,
        fail_per_note_titles: set[str] | None = None,
    ) -> None:
        self.note_ids = note_ids
        self.missing_ids = missing_ids or set()
        self.batch_errors = batch_errors
        self.fail_per_note_titles = fail_per_note_titles or set()
        self.calls: list[tuple[str, str, str, float | None]] = []
        self.batch_calls = 0
        self.adjudication_calls = 0
        self.batch_users: list[str] = []
        self.per_note_users: list[str] = []

    @property
    def per_note_prompt_count(self) -> int:
        return len(self.per_note_users)

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        self.calls.append((model, system, user, timeout_s))
        if model == config.ADJUDICATE_MODEL:
            self.adjudication_calls += 1
            return {
                "pairs": {
                    pair_id: {
                        "related": True,
                        "fact": "Batch notes share the same generated enrichment.",
                    }
                    for pair_id in dict.fromkeys(_PAIR_ID_RE.findall(user))
                }
            }
        if all(note_id in user for note_id in self.note_ids):
            self.batch_calls += 1
            self.batch_users.append(user)
            if self.batch_errors > 0:
                self.batch_errors -= 1
                raise LLMError("batch unavailable")
            return {
                "notes": {
                    note_id: _payload(f"Batch gist {note_id}")
                    for note_id in self.note_ids
                    if note_id not in self.missing_ids
                }
            }

        self.per_note_users.append(user)
        for title in self.fail_per_note_titles:
            if f"Title: {title}\n" in user:
                raise LLMError("per-note unavailable")
        title_line = next(line for line in user.splitlines() if line.startswith("Title: "))
        return _payload(f"Per-note gist {title_line.removeprefix('Title: ')}")


def test_cluster_notes_groups_by_similarity_and_respects_caps(monkeypatch) -> None:
    from taurine_core.indexer.batch_enrich import cluster_notes

    monkeypatch.setattr(config, "ENRICH_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "ENRICH_BATCH_CHAR_BUDGET", 10_000)
    notes = [
        make_note("alpha-1", "Alpha 1", "Alpha rollout plan."),
        make_note("alpha-2", "Alpha 2", "Alpha launch notes."),
        make_note("alpha-3", "Alpha 3", "Alpha follow-up."),
        make_note("beta-1", "Beta 1", "Different subject."),
    ]
    vectors = {
        "Alpha 1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "Alpha 2": np.array([0.99, 0.01, 0.0], dtype=np.float32),
        "Alpha 3": np.array([0.98, 0.02, 0.0], dtype=np.float32),
        "Beta 1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    def embed(texts: list[str]) -> np.ndarray:
        return np.vstack([vectors[text.split("\n", 1)[0]] for text in texts])

    clusters = cluster_notes(notes, embed)

    assert [note.note_id for note in clusters[0]] == ["alpha-1", "alpha-2"]
    assert all(len(cluster) <= 2 for cluster in clusters)
    assert sorted(note.note_id for cluster in clusters for note in cluster) == sorted(
        note.note_id for note in notes
    )

    huge = make_note("huge", "Huge", "x" * 60)
    small = make_note("small", "Small", "y" * 10)
    monkeypatch.setattr(config, "ENRICH_BATCH_SIZE", 10)
    monkeypatch.setattr(config, "ENRICH_BATCH_CHAR_BUDGET", 50)
    vectors = {
        "Huge": np.array([1.0, 0.0], dtype=np.float32),
        "Small": np.array([0.99, 0.01], dtype=np.float32),
    }

    clusters = cluster_notes([huge, small], embed)

    assert [note.note_id for note in clusters[0]] == ["huge"]
    assert sorted(note.note_id for cluster in clusters for note in cluster) == ["huge", "small"]


@pytest.mark.asyncio
async def test_enrich_batch_validates_keyed_response_without_fallback() -> None:
    from taurine_core.indexer.batch_enrich import enrich_batch

    notes = _notes(3)
    llm = BatchAwareLLM([note.note_id for note in notes])

    result = await enrich_batch(notes, llm, user_card="## User\nWorks on batch tests.")

    assert set(result) == {note.note_id for note in notes}
    assert all(failed is False for _enrichment, failed in result.values())
    assert result["note-0"][0].topics == ["batch topic"]
    assert llm.batch_calls == 1
    assert llm.per_note_prompt_count == 0
    assert all(note.note_id in llm.batch_users[0] for note in notes)


@pytest.mark.asyncio
async def test_enrich_batch_falls_back_only_for_missing_note_id() -> None:
    from taurine_core.indexer.batch_enrich import enrich_batch

    notes = _notes(3)
    missing = notes[1]
    llm = BatchAwareLLM([note.note_id for note in notes], missing_ids={missing.note_id})

    result = await enrich_batch(notes, llm)

    assert set(result) == {note.note_id for note in notes}
    assert llm.batch_calls == 1
    assert llm.per_note_prompt_count == 1
    assert f"Title: {missing.title}\n" in llm.per_note_users[0]
    assert result[missing.note_id][0].gist == f"Per-note gist {missing.title}"
    assert all(failed is False for _enrichment, failed in result.values())


@pytest.mark.asyncio
async def test_enrich_batch_falls_back_all_notes_after_batch_retry_fails(monkeypatch) -> None:
    from taurine_core.indexer import batch_enrich

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(batch_enrich.asyncio, "sleep", no_sleep)
    notes = _notes(3)
    llm = BatchAwareLLM([note.note_id for note in notes], batch_errors=2)

    result = await batch_enrich.enrich_batch(notes, llm)

    assert set(result) == {note.note_id for note in notes}
    assert llm.batch_calls == 2
    assert llm.per_note_prompt_count == len(notes)
    assert all(enrichment.gist.startswith("Per-note gist") for enrichment, _failed in result.values())
    assert all(failed is False for _enrichment, failed in result.values())


def test_sync_source_batches_and_caches_successes_only(monkeypatch, store) -> None:
    from taurine_core.indexer import batch_enrich, enrich, sync
    from taurine_core.indexer.sync import sync_source

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(sync, "embed_texts", _hash_embeddings)
    monkeypatch.setattr(batch_enrich.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(enrich.asyncio, "sleep", no_sleep)
    notes = _notes(config.ENRICH_BATCH_MIN_NOTES)
    failed_note = notes[-1]
    llm = BatchAwareLLM(
        [note.note_id for note in notes],
        missing_ids={failed_note.note_id},
        fail_per_note_titles={failed_note.title},
    )
    progress: list[tuple[str, int, int]] = []

    report = sync_source(store, "batch-test", notes, llm, lambda *args: progress.append(args))

    assert report.enrich_failures == 1
    assert ("enriching", len(notes), len(notes)) in progress
    assert ("finalizing", len(notes), len(notes)) in progress
    assert progress[-1][0] == "adjudicating"
    assert llm.batch_calls == 1
    assert llm.per_note_prompt_count == 2
    assert llm.adjudication_calls == 1
    for note in notes[:-1]:
        assert store.cached_enrichment(note.note_id, note.content_hash) is not None
    assert store.cached_enrichment(failed_note.note_id, failed_note.content_hash) is None
    assert store.note(failed_note.note_id) is not None
