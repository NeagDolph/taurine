from __future__ import annotations

import glob
import hashlib
import re
from pathlib import Path

from taurine_core import config as taurine_config
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord


_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


class ClaudeMemoriesConnector:
    connector_id = "claude_mem"
    label = "Claude memories"
    kind = "local"

    def available(self) -> tuple[bool, str]:
        try:
            return (_matching_paths(taurine_config.CLAUDE_MEMORY_GLOB) != [], "")
        except OSError as exc:
            return False, str(exc)

    def default_config(self) -> dict:
        return {"glob": taurine_config.CLAUDE_MEMORY_GLOB}

    def extract(self, config: dict) -> list[NoteRecord]:
        pattern = str(config.get("glob") or taurine_config.CLAUDE_MEMORY_GLOB)
        try:
            return [_record(path) for path in _matching_paths(pattern)]
        except OSError as exc:
            raise ConnectorError(f"Claude memories extraction failed: {exc}") from exc


def _matching_paths(pattern: str) -> list[Path]:
    expanded = str(Path(pattern).expanduser())
    return sorted(Path(path) for path in glob.glob(expanded, recursive=True) if Path(path).is_file())


def _record(path: Path) -> NoteRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    capped = text[: taurine_config.CONNECTOR_DOC_MAX_CHARS]
    title = _title(text, path)
    content_hash = hashlib.sha256((title + capped).encode("utf-8")).hexdigest()
    stat = path.stat()
    return NoteRecord(
        note_id=f"claude-mem:{_home_relative(path)}",
        title=title,
        folder=_project_folder(path),
        modified=stat.st_mtime,
        created=getattr(stat, "st_birthtime", stat.st_mtime),
        content_hash=content_hash,
        text=capped,
        source=ClaudeMemoriesConnector.connector_id,
    )


def _title(text: str, path: Path) -> str:
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else path.stem


def _project_folder(path: Path) -> str:
    parts = path.parts
    if "projects" in parts:
        idx = parts.index("projects")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def _home_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.home().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
