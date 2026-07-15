from __future__ import annotations

import glob as glob_module
import hashlib
import re
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
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
        return {
            "db": taurine_config.IMESSAGE_DB,
            "contacts_glob": taurine_config.CONTACTS_DB_GLOB,
            "require_contact": True,
        }

    def extract(self, config: dict) -> list[NoteRecord]:
        db_path = _db_path(config)
        if not db_path.is_file():
            raise ConnectorError(f"iMessage database not found: {db_path}")
        contacts = _contact_names(config)
        try:
            with _copied_db(db_path) as copied:
                with _open_readonly(copied) as conn:
                    return _extract_records(
                        conn,
                        contacts,
                        require_contact=bool(config.get("require_contact", True)),
                    )
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


def self_identity_names(config: dict | None = None) -> tuple[str, ...]:
    """Best-effort owner display names for owner-hub detection (item 3).

    The ``account`` column on sent messages holds the owner's own handles
    ("E:me@icloud.com" / "P:+1617…"); resolving them through Contacts yields
    the owner's display name when they keep their own card. Cheap: one small
    query against the copied db. Returns () on any failure — never raises.
    """

    try:
        cfg = dict(config or {})
        db_path = _db_path(cfg)
        if not db_path.is_file():
            return ()
        contacts = _contact_names(cfg)
        if not contacts:
            return ()
        with _copied_db(db_path) as copied:
            with _open_readonly(copied) as conn:
                columns = {
                    str(info["name"])
                    for info in conn.execute("PRAGMA table_info(message)")
                }
                if "account" not in columns:
                    return ()
                rows = conn.execute(
                    "SELECT DISTINCT account FROM message"
                    " WHERE is_from_me = 1 AND account IS NOT NULL LIMIT 32"
                ).fetchall()
        names: set[str] = set()
        for row in rows:
            handle = re.sub(r"^[EPep]:", "", str(row["account"] or "")).strip()
            name = _resolve_handle(handle, contacts)
            if name:
                names.add(name)
        return tuple(sorted(names))
    except Exception:
        return ()


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


def _extract_records(
    conn: sqlite3.Connection,
    contacts: dict[str, str] | None = None,
    *,
    require_contact: bool = True,
) -> list[NoteRecord]:
    chats = conn.execute(
        "SELECT ROWID AS chat_rowid, guid, display_name FROM chat ORDER BY ROWID"
    ).fetchall()
    records: list[NoteRecord] = []
    for chat in chats:
        record = _record_for_chat(
            conn, chat, contacts or {}, require_contact=require_contact
        )
        if record is not None:
            records.append(record)
    return records


def _record_for_chat(
    conn: sqlite3.Connection,
    chat: sqlite3.Row,
    contacts: dict[str, str],
    *,
    require_contact: bool = True,
) -> NoteRecord | None:
    message_columns = {
        str(info["name"]) for info in conn.execute("PRAGMA table_info(message)")
    }
    optional = "".join(
        f"            m.{column},\n"
        for column in ("associated_message_type", "item_type")
        if column in message_columns
    )
    rows = conn.execute(
        f"""
        SELECT
            m.ROWID AS message_rowid,
            m.text,
            m.attributedBody,
            m.date,
            m.is_from_me,
{optional}            h.id AS handle_id
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

    # rows arrive newest-first; drop the whole conversation if it has gone
    # quiet for longer than the window
    newest_ts = messages[0].timestamp
    cutoff = time.time() - taurine_config.IMESSAGE_WINDOW_DAYS * 24 * 60 * 60
    if newest_ts < cutoff:
        return None

    # keep the most recent messages within the window, capped at the message
    # limit — the smaller of "last N days" and "last N messages" wins
    selected: list[_Message] = []
    for message in messages:
        if message.timestamp < cutoff:
            break
        selected.append(message)
        if len(selected) >= taurine_config.IMESSAGE_MAX_MESSAGES:
            break

    handle_ids: list[str] = []
    participant_handles: list[str] = []
    seen_handles: set[str] = set()
    seen_participants: set[str] = set()
    for message in selected:
        if message.handle_id and message.handle_id not in seen_handles:
            handle_ids.append(message.handle_id)
            seen_handles.add(message.handle_id)
        if (
            not message.is_from_me
            and message.handle_id
            and message.handle_id not in seen_participants
        ):
            participant_handles.append(message.handle_id)
            seen_participants.add(message.handle_id)

    if require_contact and not any(
        _resolve_handle(handle, contacts) for handle in participant_handles
    ):
        return None

    transcript_lines: list[str] = []
    current_day = ""
    for message in reversed(selected):
        day = datetime.fromtimestamp(message.timestamp).strftime("%Y-%m-%d")
        if day != current_day:
            transcript_lines.append(f"— {day} —")
            current_day = day
        if message.is_from_me:
            speaker = "me"
        else:
            speaker = _resolve_handle(message.handle_id, contacts) or "them"
        transcript_lines.append(f"[{speaker}] {message.text}")

    # Intentionally no character cap: CONNECTOR_DOC_MAX_CHARS belongs to file
    # connectors (and email applies its own transcript policy). iMessage is
    # bounded only by its per-chat time window and runaway-safety message cap.
    transcript = "\n".join(transcript_lines)
    content_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    participant_names = [
        _resolve_handle(handle, contacts) or handle for handle in handle_ids
    ]
    title_name = (
        str(chat["display_name"] or "").strip()
        or ", ".join(participant_names)
        or str(chat["guid"])
    )
    return NoteRecord(
        note_id=f"{IMessageConnector.connector_id}:{chat['guid']}",
        title=f"iMessage: {title_name}",
        folder="iMessage",
        modified=newest_ts,
        created=selected[-1].timestamp,
        content_hash=content_hash,
        text=transcript,
        source=IMessageConnector.connector_id,
    )


_JUNK_TEXT = re.compile(
    r"^("
    r"NS[A-Z][A-Za-z]*"                      # NSValue, NSNumber, ... archiver class names
    r"|.*\$(archiver|version|top|objects|class).*"  # X$versionY$archiverT$topX$objects
    r"|Mat_[0-9A-Fa-f_-]{16,}.*"             # sticker/attachment resource ids
    r"|bplist00.*"
    r"|__kIM[A-Za-z]+"
    r")$"
)


def _message_from_row(row: sqlite3.Row) -> _Message | None:
    keys = row.keys()
    associated = row["associated_message_type"] if "associated_message_type" in keys else 0
    if associated not in (0, None):
        return None  # tapbacks/reactions ("Loved ..."), edits, stickers
    item_type = row["item_type"] if "item_type" in keys else 0
    if item_type not in (0, None):
        return None  # group-membership/system events
    text = row["text"]
    if text is None:
        text = _decode_attributed_body(row["attributedBody"])
    if text is None:
        return None
    normalized = " ".join(str(text).split())
    if not normalized or _JUNK_TEXT.match(normalized):
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
    """Extract the plain text of a typedstream-encoded NSMutableAttributedString.

    Primary path: the pytypedstream parser (a real implementation of Apple's
    NSArchiver typedstream format). The serialization order guarantees the
    text (NSMutableString) precedes the attribute-run NSDictionaries, so the
    first string value in a depth-first walk IS the message text. Fallback:
    a manual parse of the length-prefixed NSString payload.
    """
    if blob is None:
        return None
    data = bytes(blob)
    text = _typedstream_text(data)
    if text is None:
        text = _fallback_nsstring_payload(data)
    if text is None:
        return None
    # strip attachment placeholders (U+FFFC) and replacement chars
    text = text.replace("\ufffc", " ").replace("\ufffd", " ").strip()
    return text or None


def _typedstream_text(data: bytes) -> str | None:
    try:
        import typedstream

        unarchived = typedstream.unarchive_from_data(data)
    except Exception:
        return None
    return _first_string_value(unarchived, depth=0)


def _first_string_value(obj: object, depth: int) -> str | None:
    """Depth-first search for the first str leaf in a parsed typedstream graph.

    pytypedstream wraps values in objects exposing `.value`, and represents
    unknown classes generically with `.contents` / `.elements` sequences —
    field order is preserved, so the first string found is the text field.
    """
    if depth > 8 or obj is None:
        return None
    if isinstance(obj, str):
        return obj
    value = getattr(obj, "value", None)
    if value is not None and value is not obj:
        found = _first_string_value(value, depth + 1)
        if found is not None:
            return found
    for attr in ("contents", "elements", "values"):
        seq = getattr(obj, attr, None)
        if isinstance(seq, (list, tuple)):
            for item in seq:
                found = _first_string_value(item, depth + 1)
                if found is not None:
                    return found
    return None


def _fallback_nsstring_payload(data: bytes) -> str | None:
    """Manual parse per the typedstream layout (kept for parser failures).

    Layout after the b"NSString" class name: class-info bytes, a 0x2B ('+')
    type tag, then the length (1 byte, or 0x81 + uint16 LE, or 0x82 +
    uint32 LE), then the UTF-8 payload.
    """
    marker_index = data.find(b"NSString")
    if marker_index == -1:
        return None
    cursor = data.find(b"+", marker_index + len(b"NSString"))
    if cursor == -1 or cursor - marker_index > 16:
        return None
    cursor += 1
    if cursor >= len(data):
        return None
    length = data[cursor]
    cursor += 1
    if length == 0x81:
        if cursor + 2 > len(data):
            return None
        length = int.from_bytes(data[cursor : cursor + 2], "little")
        cursor += 2
    elif length == 0x82:
        if cursor + 4 > len(data):
            return None
        length = int.from_bytes(data[cursor : cursor + 4], "little")
        cursor += 4
    raw = data[cursor : cursor + length]
    if not raw:
        return None
    text = raw.decode("utf-8", errors="ignore").strip().strip("\x00")
    return text or None


# ── Contacts.app name resolution (AddressBook sqlite, read via temp copy) ──

def _contact_names(config: dict) -> dict[str, str]:
    """handle (normalized phone digits / lowercase email) → display name.

    Best-effort: any failure yields {} and chats fall back to raw handles.
    Reads every AddressBook source db (iCloud accounts live under Sources/).
    """
    pattern = str(config.get("contacts_glob") or taurine_config.CONTACTS_DB_GLOB)
    names: dict[str, str] = {}
    for db in sorted(glob_module.glob(str(Path(pattern).expanduser()), recursive=True)):
        try:
            with _copied_db(Path(db)) as copied:
                with _open_readonly(copied) as conn:
                    _collect_contacts(conn, names)
        except (OSError, sqlite3.Error):
            continue
    return names


def _collect_contacts(conn: sqlite3.Connection, names: dict[str, str]) -> None:
    phone_rows = conn.execute(
        """
        SELECT r.ZFIRSTNAME AS first, r.ZLASTNAME AS last,
               r.ZORGANIZATION AS org, p.ZFULLNUMBER AS value
        FROM ZABCDPHONENUMBER p JOIN ZABCDRECORD r ON r.Z_PK = p.ZOWNER
        """
    ).fetchall()
    email_rows = conn.execute(
        """
        SELECT r.ZFIRSTNAME AS first, r.ZLASTNAME AS last,
               r.ZORGANIZATION AS org, e.ZADDRESS AS value
        FROM ZABCDEMAILADDRESS e JOIN ZABCDRECORD r ON r.Z_PK = e.ZOWNER
        """
    ).fetchall()
    for row in phone_rows:
        name = _contact_display_name(row)
        digits = _phone_digits(str(row["value"] or ""))
        if not name or not digits:
            continue
        names.setdefault(digits, name)
        if len(digits) > 10:
            names.setdefault(digits[-10:], name)
    for row in email_rows:
        name = _contact_display_name(row)
        email = str(row["value"] or "").strip().lower()
        if name and email:
            names.setdefault(email, name)


def _contact_display_name(row: sqlite3.Row) -> str:
    parts = [str(row["first"] or "").strip(), str(row["last"] or "").strip()]
    full = " ".join(part for part in parts if part)
    return full or str(row["org"] or "").strip()


def _phone_digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _resolve_handle(handle: str, contacts: dict[str, str]) -> str | None:
    if not handle or not contacts:
        return None
    cleaned = handle.strip()
    if "@" in cleaned:
        return contacts.get(cleaned.lower())
    digits = _phone_digits(cleaned)
    if not digits:
        return None
    return (
        contacts.get(digits)
        or (contacts.get(digits[-10:]) if len(digits) >= 10 else None)
    )
