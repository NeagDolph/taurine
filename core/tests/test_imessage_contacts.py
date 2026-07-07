"""Contacts.app name resolution for the iMessage connector."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from taurine_core.connectors.imessage import (
    IMessageConnector,
    _contact_names,
    _resolve_handle,
)

APPLE_EPOCH_UNIX = 978_307_200.0


def _make_contacts_db(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    """rows: (first, last, org, phone, email) — empty strings skipped."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE ZABCDRECORD (Z_PK INTEGER PRIMARY KEY, ZFIRSTNAME TEXT,
            ZLASTNAME TEXT, ZORGANIZATION TEXT);
        CREATE TABLE ZABCDPHONENUMBER (Z_PK INTEGER PRIMARY KEY, ZOWNER INTEGER,
            ZFULLNUMBER TEXT);
        CREATE TABLE ZABCDEMAILADDRESS (Z_PK INTEGER PRIMARY KEY, ZOWNER INTEGER,
            ZADDRESS TEXT);
        """
    )
    for pk, (first, last, org, phone, email) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO ZABCDRECORD (Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION)"
            " VALUES (?, ?, ?, ?)",
            (pk, first or None, last or None, org or None),
        )
        if phone:
            conn.execute(
                "INSERT INTO ZABCDPHONENUMBER (ZOWNER, ZFULLNUMBER) VALUES (?, ?)",
                (pk, phone),
            )
        if email:
            conn.execute(
                "INSERT INTO ZABCDEMAILADDRESS (ZOWNER, ZADDRESS) VALUES (?, ?)",
                (pk, email),
            )
    conn.commit()
    conn.close()


def _make_chat_db(path: Path, handle: str, texts: list[tuple[str, bool]]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, display_name TEXT);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT,
            attributedBody BLOB, date INTEGER, is_from_me INTEGER,
            handle_id INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        """
    )
    conn.execute("INSERT INTO chat (ROWID, guid, display_name) VALUES (1, 'guid-1', NULL)")
    conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, ?)", (handle,))
    now_apple_ns = int((time.time() - APPLE_EPOCH_UNIX) * 1_000_000_000)
    for rowid, (text, from_me) in enumerate(texts, start=1):
        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id)"
            " VALUES (?, ?, ?, ?, 1)",
            (rowid, text, now_apple_ns - rowid, int(from_me)),
        )
        conn.execute(
            "INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, ?)",
            (rowid,),
        )
    conn.commit()
    conn.close()


def test_contact_names_phone_and_email_and_last10(tmp_path: Path) -> None:
    db = tmp_path / "AddressBook-v22.abcddb"
    _make_contacts_db(
        db,
        [
            ("Jonah", "Chang", "", "+1 (512) 850-9385", "jonah@example.com"),
            ("", "", "Acme Corp", "512-555-0199", ""),
        ],
    )
    names = _contact_names({"contacts_glob": str(db)})

    assert _resolve_handle("+15128509385", names) == "Jonah Chang"
    assert _resolve_handle("5128509385", names) == "Jonah Chang"      # last-10 alias
    assert _resolve_handle("JONAH@example.com".lower(), names) == "Jonah Chang"
    assert _resolve_handle("Jonah@Example.com", names) == "Jonah Chang"  # case-insensitive
    assert _resolve_handle("+15125550199", names) == "Acme Corp"      # org fallback
    assert _resolve_handle("+19999999999", names) is None


def test_extract_uses_contact_names_in_title_and_speakers(tmp_path: Path) -> None:
    contacts_db = tmp_path / "AddressBook-v22.abcddb"
    _make_contacts_db(contacts_db, [("Jonah", "Chang", "", "+15128509385", "")])
    chat_db = tmp_path / "chat.db"
    _make_chat_db(chat_db, "+15128509385", [("hey!", False), ("yo", True)])

    records = IMessageConnector().extract(
        {"db": str(chat_db), "contacts_glob": str(contacts_db)}
    )

    assert len(records) == 1
    record = records[0]
    assert record.title == "iMessage: Jonah Chang"
    assert "[Jonah Chang] hey!" in record.text
    assert "[me] yo" in record.text


def test_extract_falls_back_to_raw_handle_without_contacts(tmp_path: Path) -> None:
    chat_db = tmp_path / "chat.db"
    _make_chat_db(chat_db, "+15128509385", [("hey!", False)])

    records = IMessageConnector().extract(
        {"db": str(chat_db), "contacts_glob": str(tmp_path / "missing-*.abcddb")}
    )

    assert records[0].title == "iMessage: +15128509385"
    assert "[them] hey!" in records[0].text


def test_contact_names_tolerates_broken_db(tmp_path: Path) -> None:
    broken = tmp_path / "AddressBook-v22.abcddb"
    broken.write_text("not a sqlite database")
    assert _contact_names({"contacts_glob": str(broken)}) == {}
