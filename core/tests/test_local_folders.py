from __future__ import annotations

from pathlib import Path

import pytest

from taurine_core import config
from taurine_core.connectors.local_folders import LocalFoldersConnector
from taurine_core.ports import ConnectorError


def test_extracts_supported_documents_with_stable_ids(tmp_path: Path) -> None:
    docs = tmp_path / "Docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "plan.md").write_text("# Launch Plan\nbody", encoding="utf-8")
    (docs / "sub" / "notes.txt").write_text("plain text notes", encoding="utf-8")
    (docs / "photo.jpg").write_bytes(b"\xff\xd8")
    (docs / ".hidden.md").write_text("# skip me", encoding="utf-8")

    connector = LocalFoldersConnector()
    available, detail = connector.available()
    records = sorted(
        connector.extract({"folders": [str(docs)]}), key=lambda record: record.note_id
    )

    assert available is True
    assert detail == ""
    assert [record.note_id for record in records] == [
        f"local_folders:{(docs / 'plan.md').as_posix()}",
        f"local_folders:{(docs / 'sub' / 'notes.txt').as_posix()}",
    ]
    plan, notes = records
    assert plan.title == "Launch Plan"
    assert plan.folder == "Docs"
    assert plan.source == "local_folders"
    assert len(plan.content_hash) == 64
    assert notes.title == "notes"
    assert notes.folder == "Docs/sub"
    assert notes.text == "plain text notes"
    assert notes.modified == pytest.approx((docs / "sub" / "notes.txt").stat().st_mtime)


def test_caps_body_and_skips_empty_excluded_and_oversized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    (docs / "node_modules").mkdir(parents=True)
    (docs / "big.md").write_text("# Big\n" + "x" * (config.CONNECTOR_DOC_MAX_CHARS + 50))
    (docs / "empty.txt").write_text("   \n")
    (docs / "node_modules" / "dep.md").write_text("# dep")
    monkeypatch.setattr(
        "taurine_core.connectors.local_folders._MAX_FILE_BYTES", 1_000_000
    )

    records = LocalFoldersConnector().extract({"folders": [str(docs)]})

    assert [record.title for record in records] == ["Big"]
    assert len(records[0].text) == config.CONNECTOR_DOC_MAX_CHARS


def test_overlapping_folders_dedupe_and_missing_folder_skipped(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha")

    records = LocalFoldersConnector().extract(
        {"folders": [str(docs), str(docs), str(tmp_path / "gone")]}
    )

    assert len(records) == 1


def test_all_folders_missing_raises() -> None:
    with pytest.raises(ConnectorError):
        LocalFoldersConnector().extract({"folders": ["/nonexistent/one", "/nonexistent/two"]})


def test_no_folders_returns_empty() -> None:
    assert LocalFoldersConnector().extract({"folders": []}) == []
    assert LocalFoldersConnector().extract({}) == []
    assert LocalFoldersConnector().default_config() == {"folders": []}


def test_pdf_extraction_via_pypdf(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    docs = tmp_path / "docs"
    docs.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(docs / "blank.pdf", "wb") as handle:
        writer.write(handle)
    (docs / "real.txt").write_text("anchor")

    # blank pdf extracts no text and is skipped rather than erroring the sync
    records = LocalFoldersConnector().extract({"folders": [str(docs)]})

    assert [record.title for record in records] == ["real"]
