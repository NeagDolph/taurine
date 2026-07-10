"""Threaded stress of the store + connector sync surface.

Regression for the 2026-07-07 field failure: the server runs sync/connector/
reenrich work in asyncio.to_thread, and a bare sqlite3 connection raised
sqlite3.ProgrammingError ("SQLite objects created in a thread can only be
used in that same thread"), silently killing every sync. These tests hammer
the real Index from many threads at once — any thread-affinity or locking
regression fails loudly here.
"""
from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from taurine_core import config
from taurine_core.connectors.registry import ConnectorRegistry
from taurine_core.indexer.store import Index
from taurine_core.schemas import NoteRecord


def _note(source: str, n: int, body: str | None = None) -> NoteRecord:
    text = body or f"note body {source} {n} about topic-{n % 7}"
    return NoteRecord(
        note_id=f"{source}:doc-{n}",
        title=f"{source} doc {n}",
        modified=1000.0 + n,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
        source=source,
    )


class InstantLLM:
    async def complete_json(self, model, system, user, timeout_s=None):
        return {
            "gist": "a gist",
            "utterances": ["someone mentions this"],
            "aliases": ["alias"],
            "topics": ["topic a", "topic b"],
            "entities": ["Entity"],
        }


class _Schema:
    def entities(self, _types):
        return self

    def relations(self, _types):
        return self


class FakeExtractorModel:
    def create_schema(self):
        return _Schema()

    def batch_extract(self, texts, schema, *, batch_size, threshold, include_confidence, max_len):
        del schema, batch_size, threshold, include_confidence, max_len
        return [self._raw(text) for text in texts]

    def extract(self, text, schema, *, threshold, include_confidence, max_len):
        del schema, threshold, include_confidence, max_len
        return self._raw(text)

    def _raw(self, text):
        title = text.splitlines()[0]
        return {
            "entities": {
                "concept": [{"text": "topic", "confidence": 0.9}],
                "project": [{"text": title, "confidence": 0.9}],
            },
            "relation_extraction": {},
        }


class FlakyExtractorModel(FakeExtractorModel):
    """Deterministically fails one note after batch fallback; sync must survive and report."""

    def batch_extract(self, texts, schema, *, batch_size, threshold, include_confidence, max_len):
        del schema, batch_size, threshold, include_confidence, max_len
        if any("doc 3" in text for text in texts):
            raise RuntimeError("synthetic batch failure")
        return [self._raw(text) for text in texts]

    def extract(self, text, schema, *, threshold, include_confidence, max_len):
        del schema, threshold, include_confidence, max_len
        if "doc 3" in text:
            raise RuntimeError("synthetic extraction failure")
        return self._raw(text)


class FakeConnector:
    kind = "local"

    def __init__(self, connector_id: str, count: int) -> None:
        self.connector_id = connector_id
        self.label = connector_id
        self.count = count

    def available(self):
        return True, ""

    def default_config(self):
        return {}

    def extract(self, cfg):
        return [_note(self.connector_id, i) for i in range(self.count)]


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    from taurine_core.indexer import extract as extract_mod
    from taurine_core.indexer import store as store_mod

    def fake_embed(texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), config.EMBED_DIM)).astype(np.float32)

    monkeypatch.setattr(store_mod, "embed_texts", fake_embed, raising=False)
    monkeypatch.setattr(extract_mod, "_extractor_model", lambda: FakeExtractorModel())
    yield


def test_index_survives_concurrent_readers_and_writers(tmp_path) -> None:
    index = Index(tmp_path / "stress.db")
    registry = ConnectorRegistry([FakeConnector("src_a", 25), FakeConnector("src_b", 25)])
    registry.set_enabled(index, "src_a", True)
    registry.set_enabled(index, "src_b", True)
    llm = InstantLLM()
    errors: list[BaseException] = []

    def guard(fn):
        def run():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - collecting for assert
                errors.append(exc)

        return run

    def reader():
        deadline = time.time() + 6
        while time.time() < deadline:
            index.graph()
            index.status()
            index.note_count_by_source()
            index.aliases()
            for source in ("src_a", "src_b"):
                index.preview(f"{source}:doc-1")

    def syncer(source: str):
        for _round in range(3):
            report = registry.sync_connector(source, index, llm)
            assert report.error == "", report.error

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(guard(reader)) for _ in range(4)]
        futures += [pool.submit(guard(lambda: syncer("src_a")))]
        futures += [pool.submit(guard(lambda: syncer("src_b")))]
        for future in futures:
            future.result()

    assert errors == [], f"threaded operations raised: {errors[:3]}"
    counts = index.note_count_by_source()
    assert counts.get("src_a") == 25 and counts.get("src_b") == 25


def test_sync_from_worker_thread_matches_server_pattern(tmp_path) -> None:
    # exact shape of the field failure: Index created on the main thread,
    # sync executed on a worker thread (asyncio.to_thread equivalent)
    index = Index(tmp_path / "threaded.db")
    registry = ConnectorRegistry([FakeConnector("src_a", 5)])
    registry.set_enabled(index, "src_a", True)

    result: list = []
    thread = threading.Thread(
        target=lambda: result.append(registry.sync_connector("src_a", index, InstantLLM()))
    )
    thread.start()
    thread.join(timeout=30)
    assert result and result[0].error == "", result and result[0].error
    assert result[0].added == 5


def test_unexpected_extract_crash_surfaces_in_report(tmp_path) -> None:
    class ExplodingConnector(FakeConnector):
        def extract(self, cfg):
            raise RuntimeError("boom at depth")

    index = Index(tmp_path / "crash.db")
    registry = ConnectorRegistry([ExplodingConnector("src_x", 0)])
    registry.set_enabled(index, "src_x", True)

    report = registry.sync_connector("src_x", index, InstantLLM())
    assert "RuntimeError" in report.error and "boom" in report.error
    assert "src_x" not in registry.syncing
    info = {i.connector_id: i for i in registry.list_info(index)}["src_x"]
    assert "last sync failed" in info.detail


def test_flaky_extraction_reports_failures_without_crashing(tmp_path, monkeypatch) -> None:
    from taurine_core.indexer import extract as extract_mod

    monkeypatch.setattr(extract_mod, "_extractor_model", lambda: FlakyExtractorModel())
    index = Index(tmp_path / "flaky.db")
    registry = ConnectorRegistry([FakeConnector("src_a", 12)])
    registry.set_enabled(index, "src_a", True)

    report = registry.sync_connector("src_a", index, None)  # type: ignore[arg-type]
    assert report.error == ""
    assert report.added == 12                      # failed extraction still indexes raw text
    assert report.enrich_failures == 0  # clean build owns extraction; source sync is raw only
