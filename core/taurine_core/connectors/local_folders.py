"""Local folders connector: index documents from user-attached directories.

Config: {"folders": ["/path/a", ...]} — managed by the connectors UI folder
picker. Each folder is walked recursively; supported documents become
NoteRecords (.md/.txt read directly, .pdf via pypdf, .docx/.rtf/.doc via
macOS `textutil`). Hidden entries and common dependency/build directories
are skipped, as are files over _MAX_FILE_BYTES and documents whose extracted
text is empty (e.g. scanned PDFs). A folder that no longer exists is skipped
so one stale path can't block the others; if every configured folder is
missing the sync errors so the UI surfaces it.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from taurine_core import config as taurine_config
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord

_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)

_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
_PDF_SUFFIXES = {".pdf"}
_TEXTUTIL_SUFFIXES = {".docx", ".doc", ".rtf"}
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | _PDF_SUFFIXES | _TEXTUTIL_SUFFIXES

_EXCLUDED_DIR_NAMES = {"node_modules", "__pycache__", "venv", "templates", "template"}
_MAX_FILE_BYTES = 32 * 1024 * 1024
_PDF_MAX_PAGES = 80
_TEXTUTIL_TIMEOUT_S = 30.0


class LocalFoldersConnector:
    connector_id = "local_folders"
    label = "Folders"
    kind = "local"

    def available(self) -> tuple[bool, str]:
        # always usable; "no folders attached" is a config state the UI shows
        return True, ""

    def default_config(self) -> dict:
        return {"folders": []}

    def extract(self, config: dict) -> list[NoteRecord]:
        folders = _folders(config)
        roots = [Path(folder).expanduser() for folder in folders]
        existing = [root for root in roots if root.is_dir()]
        if roots and not existing:
            missing = ", ".join(str(root) for root in roots)
            raise ConnectorError(f"No attached folder exists: {missing}")

        records: dict[str, NoteRecord] = {}
        for root in existing:
            try:
                paths = _document_paths(root)
            except OSError as exc:
                raise ConnectorError(f"Folder walk failed: {exc}") from exc
            for path in paths:
                record = _record(root, path)
                if record is not None:
                    records.setdefault(record.note_id, record)
        return list(records.values())


def _folders(config: dict) -> list[str]:
    raw = config.get("folders")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _document_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [
            dirname
            for dirname in dirs
            if not dirname.startswith(".") and dirname.lower() not in _EXCLUDED_DIR_NAMES
        ]
        for filename in files:
            if filename.startswith("."):
                continue
            if Path(filename).suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            paths.append(Path(dirpath) / filename)
    return sorted(paths, key=lambda path: path.as_posix())


def _record(root: Path, path: Path) -> NoteRecord | None:
    """Extract one document; None when unreadable, oversized, or empty."""
    try:
        stat = path.stat()
        if stat.st_size > _MAX_FILE_BYTES:
            return None
        text = _extract_text(path)
    except Exception:  # noqa: BLE001 — one bad file must not block the folder
        return None
    if not text.strip():
        return None

    capped = text[: taurine_config.CONNECTOR_DOC_MAX_CHARS]
    title = _title(path, text)
    parent = path.parent
    folder = (
        root.name
        if parent == root
        else f"{root.name}/{parent.relative_to(root).as_posix()}"
    )
    content_hash = hashlib.sha256((title + capped).encode("utf-8")).hexdigest()
    return NoteRecord(
        note_id=f"{LocalFoldersConnector.connector_id}:{path.as_posix()}",
        title=title,
        folder=folder,
        modified=stat.st_mtime,
        created=getattr(stat, "st_birthtime", stat.st_mtime),
        content_hash=content_hash,
        text=capped,
        source=LocalFoldersConnector.connector_id,
    )


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in _PDF_SUFFIXES:
        return _pdf_text(path)
    return _textutil_text(path)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = reader.pages[:_PDF_MAX_PAGES]
    return "\n".join(page.extract_text() or "" for page in pages)


def _textutil_text(path: Path) -> str:
    """macOS textutil converts docx/doc/rtf to plain text."""
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        capture_output=True,
        timeout=_TEXTUTIL_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise ConnectorError(
            f"textutil failed for {path.name}: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode("utf-8", errors="replace")


def _title(path: Path, text: str) -> str:
    if path.suffix.lower() in {".md", ".markdown"}:
        match = _HEADING_RE.search(text)
        if match:
            return match.group(1).strip()
    return path.stem
