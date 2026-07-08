from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any

from taurine_core.schemas import NoteRecord


class NotesAccessError(Exception):
    pass


def extract_notes(app: object | None = None) -> list[NoteRecord]:
    try:
        if app is None:
            from macnotesapp import NotesApp

            app = NotesApp()
        account = _value(getattr(app, "default_account"))
        if isinstance(account, str):
            account = app.account(account)  # macnotesapp: default_account is a name
        raw_notes = _value(getattr(account, "notes"))
        return [_record_from_raw(raw_note) for raw_note in raw_notes]
    except Exception as exc:
        if isinstance(exc, NotesAccessError):
            raise
        raise NotesAccessError(str(exc)) from exc


def open_note(note_id: str) -> None:
    escaped = note_id.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "Notes"\nshow note id "{escaped}"\nactivate\nend tell'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NotesAccessError(str(exc)) from exc


def html_to_text(html: str) -> str:
    parser = _PlainTextParser()
    parser.feed(html)
    parser.close()
    raw = unescape("".join(parser.parts)).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    return "\n".join(line for line in lines if line)


def _record_from_raw(raw_note: object) -> NoteRecord:
    title = str(_value(getattr(raw_note, "name", "")))
    note_id = str(_value(getattr(raw_note, "id", "")))
    body = str(_value(getattr(raw_note, "body", "")))
    text = html_to_text(body)
    folder = _folder_name(getattr(raw_note, "folder", ""))
    modified = _timestamp(_value(getattr(raw_note, "modification_date", 0.0)))
    created = _timestamp(_value(getattr(raw_note, "creation_date", modified))) or modified
    content_hash = hashlib.sha256((title + text).encode("utf-8")).hexdigest()
    return NoteRecord(
        note_id=note_id,
        title=title,
        folder=folder,
        modified=modified,
        created=created,
        content_hash=content_hash,
        text=text,
    )


def _value(value: Any) -> Any:
    return value() if callable(value) else value


def _folder_name(folder: object) -> str:
    if isinstance(folder, str):
        return folder
    return str(_value(getattr(folder, "name", "")))


def _timestamp(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return float(str(value))
    except ValueError:
        return 0.0


class _PlainTextParser(HTMLParser):
    _BLOCK_TAGS = {"br", "div", "p", "li", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS and self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
