from __future__ import annotations

import glob
import hashlib
import re
from pathlib import Path

from taurine_core import config as taurine_config
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord


_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


class CodexMemoriesConnector:
    connector_id = "codex_mem"
    label = "Codex memories"
    kind = "local"

    def available(self) -> tuple[bool, str]:
        try:
            if _matching_paths(taurine_config.CODEX_MEMORY_GLOBS):
                return True, ""
            return False, "No Codex memory files found"
        except OSError as exc:
            return False, str(exc)

    def default_config(self) -> dict:
        return {"globs": list(taurine_config.CODEX_MEMORY_GLOBS)}

    def extract(self, config: dict) -> list[NoteRecord]:
        patterns = config.get("globs") or taurine_config.CODEX_MEMORY_GLOBS
        try:
            return [_record(path) for path in _matching_paths(patterns)]
        except OSError as exc:
            raise ConnectorError(f"Codex memories extraction failed: {exc}") from exc


def _matching_paths(patterns: object) -> list[Path]:
    if isinstance(patterns, str):
        raw_patterns = [patterns]
    else:
        raw_patterns = [str(pattern) for pattern in patterns]  # type: ignore[union-attr]
    matches: dict[str, Path] = {}
    for pattern in raw_patterns:
        expanded = str(Path(pattern).expanduser())
        for path in glob.glob(expanded, recursive=True):
            candidate = Path(path)
            if candidate.is_file() and candidate.suffix.lower() == ".md":
                matches[candidate.expanduser().resolve().as_posix()] = candidate
    return [matches[key] for key in sorted(matches)]


def _record(path: Path) -> NoteRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    capped = text[: taurine_config.CONNECTOR_DOC_MAX_CHARS]
    title = _title(text, path)
    content_hash = hashlib.sha256((title + capped).encode("utf-8")).hexdigest()
    stat = path.stat()
    return NoteRecord(
        note_id=f"codex-mem:{_home_relative(path)}",
        title=title,
        folder=path.parent.name,
        modified=stat.st_mtime,
        created=getattr(stat, "st_birthtime", stat.st_mtime),
        content_hash=content_hash,
        text=capped,
        source=CodexMemoriesConnector.connector_id,
    )


def _title(text: str, path: Path) -> str:
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else path.stem


def _home_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.home().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
