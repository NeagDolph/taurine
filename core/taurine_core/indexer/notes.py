from __future__ import annotations

import hashlib
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any

from taurine_core.schemas import NoteRecord

_LOG = logging.getLogger("taurine.connectors.apple_notes")

# One AppleEvent to Notes.app can block forever client-side even while the
# app answers everyone else (observed with the noteGetContainer fallback on
# shared-folder notes). Each conversion runs under a watchdog; a timed-out
# worker thread is abandoned, which leaks one thread per poison note — an
# accepted price for the sync always finishing.
NOTE_CONVERT_TIMEOUT_S = 20.0

# When conversions fail wholesale, Notes.app itself is broken (quit mid-sync,
# revoked automation permission). Failing the connector keeps the previously
# synced notes instead of silently shrinking the store to the few survivors.
MAX_SKIPPED_FRACTION = 0.25


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
        raw_notes = list(_value(getattr(account, "notes")))
    except Exception as exc:
        if isinstance(exc, NotesAccessError):
            raise
        raise NotesAccessError(str(exc)) from exc

    # Per-note isolation: macnotesapp resolves attributes through AppleScript,
    # which fails on individual notes (shared folders, odd containers, e.g.
    # "Can't make container into type text"). One bad note must not take the
    # whole account down.
    records: list[NoteRecord] = []
    skipped: list[str] = []
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apple-note-convert")
    try:
        for position, raw_note in enumerate(raw_notes):
            # error rendering runs inside the watchdog too: _safe_note_id is
            # itself an AppleScript lookup and can block like any other
            future = executor.submit(_convert_raw_note, raw_note)
            try:
                record, failure = future.result(timeout=NOTE_CONVERT_TIMEOUT_S)
            except FutureTimeoutError:
                skipped.append(f"note #{position + 1}: conversion timed out")
                # The worker is stuck in a native AppleEvent and cannot be
                # interrupted; abandon that executor and continue on a new one.
                executor.shutdown(wait=False)
                executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="apple-note-convert"
                )
                continue
            if failure is not None:
                skipped.append(failure)
                continue
            if record is not None and record.note_id:
                records.append(record)
    finally:
        executor.shutdown(wait=False)
    if skipped:
        _LOG.warning(
            "apple notes: skipped %d unreadable notes: %s",
            len(skipped),
            "; ".join(skipped[:5]),
        )
    if raw_notes and len(skipped) > max(3.0, MAX_SKIPPED_FRACTION * len(raw_notes)):
        raise NotesAccessError(
            f"apple notes conversion failed for {len(skipped)} of {len(raw_notes)} "
            "notes — Notes.app looks unavailable, keeping previously synced data"
        )
    return records


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
    try:
        folder = _folder_name(getattr(raw_note, "folder", ""))
    except Exception:  # AppleScript container lookups fail on some notes
        folder = ""
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


def _convert_raw_note(raw_note: object) -> tuple[NoteRecord | None, str | None]:
    try:
        return _record_from_raw(raw_note), None
    except Exception as exc:
        return None, f"{_safe_note_id(raw_note)}: {exc}"


def _value(value: Any) -> Any:
    return value() if callable(value) else value


def _safe_note_id(raw_note: object) -> str:
    try:
        return str(_value(getattr(raw_note, "id", ""))) or "<unknown>"
    except Exception:
        return "<unknown>"


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
