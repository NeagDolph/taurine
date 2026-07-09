from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord


@dataclass
class FakeConnector:
    connector_id: str
    label: str
    records: list[NoteRecord] = field(default_factory=list)
    kind: str = "local"
    default: dict[str, Any] = field(default_factory=dict)
    available_result: tuple[bool, str] = (True, "")
    configs_seen: list[dict[str, Any]] = field(default_factory=list)

    def available(self) -> tuple[bool, str]:
        return self.available_result

    def default_config(self) -> dict:
        return dict(self.default)

    def extract(self, config: dict) -> list[NoteRecord]:
        self.configs_seen.append(dict(config))
        return list(self.records)


class StubLLM:
    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        del model, system, user, timeout_s
        return {
            "gist": "Connector note gist.",
            "utterances": [
                "Can we discuss the connector note?",
                "What changed in this connector note?",
                "Where is the connector source?",
                "Do we need this connector note?",
            ],
            "aliases": ["connector note"],
            "topics": ["connectors", "work"],
            "entities": ["Taurine"],
        }


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


def test_markdown_file_to_record_caps_body_namespaces_id_and_uses_first_heading(tmp_path):
    from taurine_core.connectors.base import markdown_file_to_note

    root = tmp_path / "Vault"
    path = root / "Folder" / "Doc.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Project Plan\n\n" + "x" * (config.CONNECTOR_DOC_MAX_CHARS + 100), encoding="utf-8")

    note = markdown_file_to_note(path, connector_id="obsidian", root=root)

    assert note.note_id == "obsidian:Folder/Doc.md"
    assert note.title == "Project Plan"
    assert note.folder == "Folder"
    assert note.source == "obsidian"
    assert len(note.text) == config.CONNECTOR_DOC_MAX_CHARS
    assert note.content_hash == hashlib.sha256((note.title + note.text).encode("utf-8")).hexdigest()
    assert note.created == pytest.approx(
        getattr(path.stat(), "st_birthtime", path.stat().st_mtime)
    )


def test_markdown_file_to_record_uses_filename_when_heading_missing(tmp_path):
    from taurine_core.connectors.base import markdown_file_to_note

    path = tmp_path / "Daily Note.md"
    path.write_text("plain body", encoding="utf-8")

    note = markdown_file_to_note(path, connector_id="codex_mem")

    assert note.note_id == "codex_mem:Daily Note.md"
    assert note.title == "Daily Note"
    assert note.folder == ""


def test_apple_notes_connector_tags_records_and_wraps_access_errors(monkeypatch):
    from taurine_core.connectors.apple_notes import AppleNotesConnector
    from taurine_core.indexer.notes import NotesAccessError

    calls: list[str] = []
    raw = make_note("x-coredata://note", "Apple", "Apple body").model_copy(
        update={"source": "wrong"}
    )
    connector = AppleNotesConnector(
        app_factory=lambda: object(),
        extractor=lambda app=None: calls.append("extract") or [raw],
    )

    assert connector.extract({}) == [raw.model_copy(update={"source": "apple_notes"})]
    assert calls == ["extract"]

    failing = AppleNotesConnector(
        app_factory=lambda: object(),
        extractor=lambda app=None: (_ for _ in ()).throw(NotesAccessError("denied")),
    )
    with pytest.raises(ConnectorError, match="denied"):
        failing.extract({})


def test_extract_notes_converts_html_and_hashes_content():
    from taurine_core.indexer.notes import _record_from_raw

    class RawNote:
        name = "HTML Note"
        id = "x-coredata://html"
        body = "<div>Hello <b>world</b><br>Next line&nbsp;here</div>"
        folder = "Notes"
        modification_date = 200.0
        creation_date = 100.0

    note = _record_from_raw(RawNote())

    assert note.text == "Hello world\nNext line here"
    assert note.content_hash == hashlib.sha256(
        ("HTML Note" + "Hello world\nNext line here").encode("utf-8")
    ).hexdigest()
    assert len(note.content_hash) == 64


def test_apple_notes_raw_record_uses_creation_date_with_modified_fallback():
    from taurine_core.indexer.notes import _record_from_raw

    class RawNote:
        name = "Apple"
        id = "x-coredata://note"
        body = "<div>Apple body</div>"
        folder = "Notes"
        modification_date = 200.0
        creation_date = 100.0

    note = _record_from_raw(RawNote())

    assert note.modified == 200.0
    assert note.created == 100.0

    del RawNote.creation_date
    fallback = _record_from_raw(RawNote())
    assert fallback.created == 200.0


def test_apple_notes_available_uses_cheap_account_probe_without_extraction():
    from taurine_core.connectors.apple_notes import AppleNotesConnector

    class FakeApp:
        default_account = "iCloud"

        def account(self, name: str) -> object:
            assert name == "iCloud"
            return object()

    def fail_extract(app: object | None = None) -> list[NoteRecord]:
        raise AssertionError("available() should not extract notes")

    connector = AppleNotesConnector(app_factory=FakeApp, extractor=fail_extract)

    assert connector.available() == (True, "")


def test_registry_enable_disable_purge_and_sync_updates_state(store):
    from taurine_core.connectors.registry import ConnectorRegistry

    note = make_note("fake:alpha", "Alpha", "Alpha body").model_copy(update={"source": "fake"})
    connector = FakeConnector(
        connector_id="fake",
        label="Fake",
        records=[note],
        default={"path": "/tmp/default"},
    )
    registry = ConnectorRegistry([connector])

    initial = registry.list_info(store)[0]
    assert initial.enabled is False
    assert initial.docs == 0
    assert initial.config == {"path": "/tmp/default"}

    registry.set_config(store, "fake", {"path": "/tmp/custom"})
    registry.set_enabled(store, "fake", True)
    report = registry.sync_connector("fake", store, StubLLM())

    assert report.connector_id == "fake"
    assert report.added == 1
    assert store.note(note.note_id).source == "fake"
    assert connector.configs_seen == [{"path": "/tmp/custom"}]
    synced = registry.list_info(store)[0]
    assert synced.enabled is True
    assert synced.docs == 1
    assert synced.last_sync is not None
    assert synced.config == {"path": "/tmp/custom"}

    registry.set_enabled(store, "fake", False)

    kept = registry.list_info(store)[0]
    assert kept.enabled is False
    assert kept.docs == 1                      # disable keeps data stored
    assert store.note(note.note_id) is not None

    registry.purge_data(store, "fake")

    purged = registry.list_info(store)[0]
    assert purged.enabled is False
    assert purged.docs == 0
    assert store.note(note.note_id) is None


def test_registry_sync_all_enabled_skips_disabled_connectors(store):
    from taurine_core.connectors.registry import ConnectorRegistry

    enabled = FakeConnector(
        connector_id="enabled",
        label="Enabled",
        records=[
            make_note("enabled:one", "Enabled", "Enabled body").model_copy(
                update={"source": "enabled"}
            )
        ],
    )
    disabled = FakeConnector(
        connector_id="disabled",
        label="Disabled",
        records=[
            make_note("disabled:one", "Disabled", "Disabled body").model_copy(
                update={"source": "disabled"}
            )
        ],
    )
    registry = ConnectorRegistry([enabled, disabled])
    registry.set_enabled(store, "enabled", True)

    reports = registry.sync_all_enabled(store, StubLLM())

    assert [report.connector_id for report in reports] == ["enabled"]
    assert store.note("enabled:one") is not None
    assert store.note("disabled:one") is None


def test_registry_build_connectors_falls_back_to_apple_notes_when_catalog_missing(monkeypatch):
    from taurine_core.connectors import registry

    real_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None) -> Any:
        if name == "taurine_core.connectors.catalog":
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    connectors = registry.build_connectors()

    assert [connector.connector_id for connector in connectors] == ["apple_notes"]
