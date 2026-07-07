from __future__ import annotations

import hashlib
import re
from pathlib import Path

from taurine_core import config
from taurine_core.schemas import NoteRecord


def markdown_file_to_note(
    path: str | Path,
    *,
    connector_id: str,
    root: str | Path | None = None,
) -> NoteRecord:
    file_path = Path(path).expanduser()
    root_path = Path(root).expanduser() if root is not None else file_path.parent
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    text = raw_text[: config.CONNECTOR_DOC_MAX_CHARS]
    title = first_heading_or_filename(raw_text, file_path)
    relative = _relative_path(file_path, root_path)
    folder = "" if relative.parent == Path(".") else relative.parent.as_posix()
    return NoteRecord(
        note_id=namespaced_file_id(connector_id, relative),
        title=title,
        folder=folder,
        modified=file_path.stat().st_mtime,
        content_hash=content_hash(title, text),
        text=text,
        source=connector_id,
    )


def first_heading_or_filename(markdown: str, path: str | Path) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            title = match.group(1).strip()
            if title:
                return title
    return Path(path).stem


def namespaced_file_id(connector_id: str, relative_path: str | Path) -> str:
    return f"{connector_id}:{Path(relative_path).as_posix()}"


def content_hash(title: str, text: str) -> str:
    return hashlib.sha256((title + text).encode("utf-8")).hexdigest()


def _relative_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)
