from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from taurine_core import config as taurine_config
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord


_APPLE_EPOCH_UNIX = 978_307_200.0
_FULL_DISK_ACCESS_DETAIL = "grant Full Disk Access to Taurine/terminal"
_METADATA_STRINGS = {
    "NSString",
    "NSDictionary",
    "NSNumber",
    "NSObject",
    "NSAttributedString",
    "NSMutableAttributedString",
    "streamtyped",
}


@dataclass(frozen=True)
class _Message:
    text: str
    is_from_me: bool
    timestamp: float
    handle_id: str


class IMessageConnector:
    connector_id = "imessage"
    label = "iMessage"
    kind = "local"

    def available(self) -> tuple[bool, str]:
        db_path = _db_path({})
        if not db_path.is_file():
            return False, f"iMessage database not found: {db_path}"
        try:
            with _copied_db(db_path) as copied:
                with _open_readonly(copied) as conn:
                    conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            return True, ""
        except PermissionError:
            return False, _FULL_DISK_ACCESS_DETAIL
        except (OSError, sqlite3.Error) as exc:
            return False, str(exc)

    def default_config(self) -> dict:
        return {"db": taurine_config.IMESSAGE_DB}

    def extract(self, config: dict) -> list[NoteRecord]:
        db_path = _db_path(config)
        if not db_path.is_file():
            raise ConnectorError(f"iMessage database not found: {db_path}")
        try:
            with _copied_db(db_path) as copied:
                with _open_readonly(copied) as conn:
                    return _extract_records(conn)
        except PermissionError as exc:
            raise ConnectorError(f"iMessage database unreadable: {_FULL_DISK_ACCESS_DETAIL}") from exc
        except sqlite3.OperationalError as exc:
            raise ConnectorError(f"iMessage schema unexpected: {exc}") from exc
        except sqlite3.Error as exc:
            raise ConnectorError(f"iMessage database read failed: {exc}") from exc
        except OSError as exc:
            raise ConnectorError(f"iMessage database copy failed: {exc}") from exc


def _db_path(config: dict) -> Path:
    return Path(str(config.get("db") or taurine_config.IMESSAGE_DB)).expanduser()


@contextmanager
def _copied_db(source: Path) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="taurine-imessage-") as tempdir:
        destination = Path(tempdir) / source.name
        shutil.copy2(source, destination)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{source}{suffix}")
            if sidecar.exists():
                shutil.copy2(sidecar, Path(f"{destination}{suffix}"))
        yield destination


@contextmanager
def _open_readonly(path: Path) -> Iterator[sqlite3.Connection]:
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _extract_records(conn: sqlite3.Connection) -> list[NoteRecord]:
    chats = conn.execute(
        "SELECT ROWID AS chat_rowid, guid, display_name FROM chat ORDER BY ROWID"
    ).fetchall()
    records: list[NoteRecord] = []
    for chat in chats:
        record = _record_for_chat(conn, chat)
        if record is not None:
            records.append(record)
    return records


def _record_for_chat(conn: sqlite3.Connection, chat: sqlite3.Row) -> NoteRecord | None:
    rows = conn.execute(
        """
        SELECT
            m.ROWID AS message_rowid,
            m.text,
            m.attributedBody,
            m.date,
            m.is_from_me,
            h.id AS handle_id
        FROM chat_message_join cmj
        JOIN message m ON m.ROWID = cmj.message_id
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        WHERE cmj.chat_id = ?
        ORDER BY m.date DESC, m.ROWID DESC
        """,
        (chat["chat_rowid"],),
    ).fetchall()
    messages = [_message_from_row(row) for row in rows]
    messages = [message for message in messages if message is not None]
    if not messages:
        return None

    cutoff = time.time() - taurine_config.IMESSAGE_WINDOW_DAYS * 24 * 60 * 60
    lines_desc: list[str] = []
    accumulated = 0
    newest_ts: float | None = None
    handle_ids: list[str] = []
    seen_handles: set[str] = set()
    for message in messages:
        if message.handle_id and message.handle_id not in seen_handles:
            handle_ids.append(message.handle_id)
            seen_handles.add(message.handle_id)
        if message.timestamp < cutoff:
            break
        if newest_ts is None:
            newest_ts = message.timestamp
        line = f"[{'me' if message.is_from_me else 'them'}] {message.text}"
        line_length = len(line) + (1 if lines_desc else 0)
        if accumulated + line_length > taurine_config.IMESSAGE_MAX_CHARS:
            if not lines_desc:
                line = line[: taurine_config.IMESSAGE_MAX_CHARS]
                line_length = len(line)
            else:
                break
        lines_desc.append(line)
        accumulated += line_length

    if not lines_desc or newest_ts is None:
        return None

    transcript = "\n".join(reversed(lines_desc))
    content_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    title_name = str(chat["display_name"] or "").strip() or ", ".join(handle_ids) or str(chat["guid"])
    return NoteRecord(
        note_id=f"{IMessageConnector.connector_id}:{chat['guid']}",
        title=f"iMessage: {title_name}",
        folder="iMessage",
        modified=newest_ts,
        content_hash=content_hash,
        text=transcript,
        source=IMessageConnector.connector_id,
    )


def _message_from_row(row: sqlite3.Row) -> _Message | None:
    text = row["text"]
    if text is None:
        text = _decode_attributed_body(row["attributedBody"])
    if text is None:
        return None
    normalized = " ".join(str(text).split())
    if not normalized:
        return None
    return _Message(
        text=normalized,
        is_from_me=bool(row["is_from_me"]),
        timestamp=_apple_date_to_unix(float(row["date"])),
        handle_id=str(row["handle_id"] or ""),
    )


def _apple_date_to_unix(raw: float) -> float:
    if abs(raw) > 10_000_000_000:
        raw = raw / 1_000_000_000
    return _APPLE_EPOCH_UNIX + raw


def _decode_attributed_body(blob: object) -> str | None:
    if blob is None:
        return None
    data = bytes(blob)
    marker_index = data.find(b"NSString")
    payload = data[marker_index + len(b"NSString") :] if marker_index >= 0 else data
    candidates: list[str] = []
    for raw in re.findall(rb"[\x09\x0a\x0d\x20-\x7e]{2,}", payload):
        decoded = raw.decode("utf-8", errors="ignore").strip().strip("\x00")
        if not decoded or decoded in _METADATA_STRINGS:
            continue
        if decoded.startswith("__") or decoded.startswith("NS."):
            continue
        candidates.append(decoded)
    if not candidates:
        return None
    return max(candidates, key=len)
