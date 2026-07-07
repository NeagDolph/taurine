from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from taurine_core import config as taurine_config
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord


_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_EXCLUDED_DIR_NAMES = {"templates", "template"}


class ObsidianConnector:
    connector_id = "obsidian"
    label = "Obsidian"
    kind = "local"

    def available(self) -> tuple[bool, str]:
        try:
            vault = _discover_vault()
            if vault is None:
                return False, "No Obsidian vault found"
            if not vault.is_dir():
                return False, f"Obsidian vault does not exist: {vault}"
            return True, ""
        except (OSError, json.JSONDecodeError) as exc:
            return False, str(exc)

    def default_config(self) -> dict:
        return {"vault": ""}

    def extract(self, config: dict) -> list[NoteRecord]:
        vault = Path(str(config.get("vault"))).expanduser() if config.get("vault") else _discover_vault()
        if vault is None:
            raise ConnectorError("Obsidian vault not configured and no vault found")
        if not vault.is_dir():
            raise ConnectorError(f"Obsidian vault does not exist: {vault}")
        try:
            return [_record(vault, path) for path in _markdown_paths(vault)]
        except OSError as exc:
            raise ConnectorError(f"Obsidian extraction failed: {exc}") from exc


def _discover_vault() -> Path | None:
    app_json = Path(taurine_config.OBSIDIAN_APP_JSON).expanduser()
    if not app_json.is_file():
        return None
    data = json.loads(app_json.read_text(encoding="utf-8"))
    vaults = data.get("vaults")
    if not isinstance(vaults, dict) or not vaults:
        return None
    entries: list[tuple[float, int, Path]] = []
    for index, raw in enumerate(vaults.values()):
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        entries.append((_latest_timestamp(raw), -index, Path(str(raw["path"])).expanduser()))
    if not entries:
        return None
    entries.sort(reverse=True)
    return entries[0][2]


def _latest_timestamp(raw: dict[str, Any]) -> float:
    for key in ("ts", "mtime", "lastOpened", "last_opened", "opened", "openTime"):
        value = raw.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _markdown_paths(vault: Path) -> list[Path]:
    paths: list[Path] = []
    for root, dirs, files in os.walk(vault):
        dirs[:] = [
            dirname
            for dirname in dirs
            if not dirname.startswith(".") and dirname.lower() not in _EXCLUDED_DIR_NAMES
        ]
        for filename in files:
            if filename.startswith(".") or Path(filename).suffix.lower() != ".md":
                continue
            paths.append(Path(root) / filename)
    return sorted(paths, key=lambda path: path.relative_to(vault).as_posix())


def _record(vault: Path, path: Path) -> NoteRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    capped = text[: taurine_config.CONNECTOR_DOC_MAX_CHARS]
    title = _title(text, path)
    relpath = path.relative_to(vault).as_posix()
    parent = path.parent.relative_to(vault)
    folder = "" if parent == Path(".") else parent.as_posix()
    content_hash = hashlib.sha256((title + capped).encode("utf-8")).hexdigest()
    return NoteRecord(
        note_id=f"{ObsidianConnector.connector_id}:{relpath}",
        title=title,
        folder=folder,
        modified=path.stat().st_mtime,
        content_hash=content_hash,
        text=capped,
        source=ObsidianConnector.connector_id,
    )


def _title(text: str, path: Path) -> str:
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else path.stem
