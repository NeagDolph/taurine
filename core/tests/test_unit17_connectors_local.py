from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from taurine_core import config
from taurine_core.connectors.catalog import CONNECTORS
from taurine_core.connectors.claude_memories import ClaudeMemoriesConnector
from taurine_core.connectors.codex_memories import CodexMemoriesConnector
from taurine_core.connectors.imessage import IMessageConnector
from taurine_core.connectors.obsidian import ObsidianConnector
from taurine_core.ports import ConnectorError


APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)


def test_claude_memories_extracts_markdown_files_with_caps_and_stable_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    memory_dir = home / ".claude" / "projects" / "taurine" / "memory"
    memory_dir.mkdir(parents=True)
    note_path = memory_dir / "launch-plan.md"
    body = "# Launch Plan\n" + ("x" * (config.CONNECTOR_DOC_MAX_CHARS + 25))
    note_path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    connector = ClaudeMemoriesConnector()
    available, detail = connector.available()
    records = connector.extract({"glob": str(memory_dir / "*.md")})

    assert available is True
    assert detail == ""
    assert len(records) == 1
    record = records[0]
    assert record.note_id == "claude-mem:.claude/projects/taurine/memory/launch-plan.md"
    assert record.title == "Launch Plan"
    assert record.folder == "taurine"
    assert record.source == "claude_mem"
    assert len(record.text) == config.CONNECTOR_DOC_MAX_CHARS
    assert len(record.content_hash) == 64
    assert record.modified == pytest.approx(note_path.stat().st_mtime)
    assert record.created == pytest.approx(
        getattr(note_path.stat(), "st_birthtime", note_path.stat().st_mtime)
    )


def test_claude_memories_falls_back_to_filename_when_no_heading(tmp_path: Path) -> None:
    note_path = tmp_path / "notes" / "scratch-memory.md"
    note_path.parent.mkdir()
    note_path.write_text("plain text without a heading", encoding="utf-8")

    records = ClaudeMemoriesConnector().extract({"glob": str(note_path)})

    assert records[0].title == "scratch-memory"
    assert records[0].folder == "notes"


def test_codex_memories_extracts_multiple_globs_and_reports_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    agents = home / ".codex" / "AGENTS.md"
    project_memory = home / ".codex" / "memories" / "taurine" / "handoff.md"
    agents.parent.mkdir(parents=True)
    project_memory.parent.mkdir(parents=True)
    agents.write_text("# Agent Rules\nKeep changes scoped.", encoding="utf-8")
    project_memory.write_text("# Handoff\nUse connector ports.", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    connector = CodexMemoriesConnector()
    records = connector.extract({"globs": [str(agents), str(project_memory)]})
    available, detail = connector.available()

    assert available is True
    assert detail == ""
    assert [record.note_id for record in records] == [
        "codex-mem:.codex/AGENTS.md",
        "codex-mem:.codex/memories/taurine/handoff.md",
    ]
    assert [record.title for record in records] == ["Agent Rules", "Handoff"]
    assert [record.folder for record in records] == [".codex", "taurine"]
    assert {record.source for record in records} == {"codex_mem"}
    assert records[0].created == pytest.approx(
        getattr(agents.stat(), "st_birthtime", agents.stat().st_mtime)
    )
    assert records[1].created == pytest.approx(
        getattr(project_memory.stat(), "st_birthtime", project_memory.stat().st_mtime)
    )

    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    available, detail = connector.available()
    assert available is False
    assert "No Codex memory files found" in detail


def test_obsidian_extracts_vault_markdown_and_skips_hidden_dirs(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "Projects").mkdir(parents=True)
    (vault / ".obsidian").mkdir()
    (vault / ".hidden").mkdir()
    (vault / "Daily.md").write_text("# Daily Note\nToday", encoding="utf-8")
    (vault / "Projects" / "Plan.md").write_text("No heading here", encoding="utf-8")
    (vault / ".obsidian" / "workspace.md").write_text("# Ignore", encoding="utf-8")
    (vault / ".hidden" / "secret.md").write_text("# Ignore", encoding="utf-8")

    records = ObsidianConnector().extract({"vault": str(vault)})

    assert [record.note_id for record in records] == [
        "obsidian:Daily.md",
        "obsidian:Projects/Plan.md",
    ]
    assert [record.title for record in records] == ["Daily Note", "Plan"]
    assert [record.folder for record in records] == ["", "Projects"]
    assert {record.source for record in records} == {"obsidian"}
    daily_stat = (vault / "Daily.md").stat()
    assert records[0].created == pytest.approx(
        getattr(daily_stat, "st_birthtime", daily_stat.st_mtime)
    )


def test_obsidian_discovers_most_recent_vault_from_app_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    app_dir = home / "Library" / "Application Support" / "obsidian"
    app_dir.mkdir(parents=True)
    older.mkdir()
    newer.mkdir()
    (older / "Old.md").write_text("# Old", encoding="utf-8")
    (newer / "New.md").write_text("# New", encoding="utf-8")
    (app_dir / "obsidian.json").write_text(
        json.dumps(
            {
                "vaults": {
                    "old": {"path": str(older), "ts": 100},
                    "new": {"path": str(newer), "ts": 200},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    records = ObsidianConnector().extract({})

    assert [record.note_id for record in records] == ["obsidian:New.md"]


def test_imessage_extracts_bounded_chat_transcripts_with_attribution_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    db_path = home / "Library" / "Messages" / "chat.db"
    _create_imessage_db(db_path)
    now = datetime.now(tz=UTC)
    newest = now - timedelta(days=1)
    inside_window = now - timedelta(days=100)
    outside_window = now - timedelta(days=config.IMESSAGE_WINDOW_DAYS + 20)
    _insert_handle(db_path, 1, "+15551234567")
    _insert_chat(db_path, 1, "iMessage;-;+15551234567", "Alice")
    _insert_message(db_path, 1, 1, 1, "outside window", outside_window, from_me=False)
    _insert_message(db_path, 1, 2, 1, "inside window", inside_window, from_me=False)
    _insert_message(db_path, 1, 3, 1, None, inside_window + timedelta(hours=1), from_me=True)
    _insert_message(db_path, 1, 4, 1, "newest reply", newest, from_me=True)
    monkeypatch.setenv("HOME", str(home))

    connector = IMessageConnector()
    available, detail = connector.available()
    first = connector.extract({"require_contact": False})
    second = connector.extract({"require_contact": False})

    assert available is True
    assert detail == ""
    assert len(first) == 1
    record = first[0]
    assert record.note_id == "imessage:iMessage;-;+15551234567"
    assert record.title == "iMessage: Alice"
    assert record.folder == "iMessage"
    assert record.source == "imessage"
    assert record.modified == pytest.approx(newest.timestamp(), abs=1.0)
    assert record.created == pytest.approx(inside_window.timestamp(), abs=1.0)
    assert record.text.splitlines() == [
        f"— {inside_window.astimezone().strftime('%Y-%m-%d')} —",
        "[them] inside window",
        f"— {newest.astimezone().strftime('%Y-%m-%d')} —",
        "[me] newest reply",
    ]
    assert "outside window" not in record.text
    assert first[0].content_hash == second[0].content_hash


def test_imessage_defaults_keep_a_year_with_a_runaway_only_message_cap() -> None:
    assert config.IMESSAGE_WINDOW_DAYS == 365
    assert config.IMESSAGE_MAX_MESSAGES == 2000


def test_imessage_keeps_messages_older_than_six_months_within_the_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    db_path = home / "Library" / "Messages" / "chat.db"
    _create_imessage_db(db_path)
    now = datetime.now(tz=UTC)
    _insert_handle(db_path, 1, "+15550000000")
    _insert_chat(db_path, 1, "year-window-guid", "Year Window")
    _insert_message(
        db_path,
        1,
        1,
        1,
        "eight months ago",
        now - timedelta(days=240),
        from_me=False,
    )
    _insert_message(
        db_path,
        1,
        2,
        1,
        "recent",
        now - timedelta(days=1),
        from_me=True,
    )
    monkeypatch.setenv("HOME", str(home))

    record = IMessageConnector().extract({"require_contact": False})[0]

    assert record.text.splitlines() == [
        f"— {(now - timedelta(days=240)).astimezone().strftime('%Y-%m-%d')} —",
        "[them] eight months ago",
        f"— {(now - timedelta(days=1)).astimezone().strftime('%Y-%m-%d')} —",
        "[me] recent",
    ]


def test_imessage_caps_at_max_messages_keeping_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    db_path = home / "Library" / "Messages" / "chat.db"
    _create_imessage_db(db_path)
    now = datetime.now(tz=UTC)
    _insert_handle(db_path, 1, "+15550000000")
    _insert_chat(db_path, 1, "chat-guid", None)
    total = config.IMESSAGE_MAX_MESSAGES + 40  # all recent (within window)
    for idx in range(total):
        _insert_message(
            db_path,
            1,
            idx + 1,
            1,
            f"message-{idx}",
            now - timedelta(minutes=total - idx),  # idx grows => newer
            from_me=False,
        )
    monkeypatch.setenv("HOME", str(home))

    record = IMessageConnector().extract({"require_contact": False})[0]
    lines = record.text.splitlines()
    message_lines = [line for line in lines if not line.startswith("— ")]

    # only the most-recent N kept, in chronological order (oldest kept first)
    assert len(message_lines) == config.IMESSAGE_MAX_MESSAGES
    assert message_lines[0] == f"[them] message-{total - config.IMESSAGE_MAX_MESSAGES}"
    assert message_lines[-1] == f"[them] message-{total - 1}"
    assert (
        f"[them] message-{total - config.IMESSAGE_MAX_MESSAGES - 1}"
        not in message_lines
    )
    assert all(line.endswith(" —") for line in lines if line.startswith("— "))
    assert record.title == "iMessage: +15550000000"


def test_imessage_excludes_conversations_older_than_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    db_path = home / "Library" / "Messages" / "chat.db"
    _create_imessage_db(db_path)
    now = datetime.now(tz=UTC)
    stale = now - timedelta(days=config.IMESSAGE_WINDOW_DAYS + 10)
    _insert_handle(db_path, 1, "+15559999999")
    _insert_chat(db_path, 1, "stale-guid", "Old Friend")
    _insert_message(db_path, 1, 1, 1, "haven't talked in ages", stale, from_me=False)
    # an active chat proves only the stale conversation is dropped
    _insert_handle(db_path, 2, "+15551112222")
    _insert_chat(db_path, 2, "active-guid", "Recent Friend")
    _insert_message(db_path, 2, 2, 2, "still chatting", now - timedelta(days=1), from_me=False)
    monkeypatch.setenv("HOME", str(home))

    records = IMessageConnector().extract({"require_contact": False})

    assert {record.title for record in records} == {"iMessage: Recent Friend"}


def test_imessage_handles_old_second_precision_apple_dates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    db_path = home / "Library" / "Messages" / "chat.db"
    _create_imessage_db(db_path)
    when = datetime.now(tz=UTC) - timedelta(days=1)
    seconds_since_apple_epoch = int((when - APPLE_EPOCH).total_seconds())
    _insert_handle(db_path, 1, "codex@example.com")
    _insert_chat(db_path, 1, "second-precision-chat", None)
    _insert_raw_message(db_path, 1, 1, 1, "legacy date", seconds_since_apple_epoch, from_me=True)
    monkeypatch.setenv("HOME", str(home))

    record = IMessageConnector().extract({"require_contact": False})[0]

    assert record.modified == pytest.approx(when.timestamp(), abs=1.0)
    assert record.text == (
        f"— {when.astimezone().strftime('%Y-%m-%d')} —\n[me] legacy date"
    )


def test_imessage_raises_connector_error_for_unexpected_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "broken.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT)")

    with pytest.raises(ConnectorError, match="iMessage schema"):
        IMessageConnector().extract({"db": str(db_path)})


def test_connector_catalog_exposes_local_connectors() -> None:
    connector_ids = [connector.connector_id for connector in CONNECTORS]

    assert "claude_mem" in connector_ids
    assert "codex_mem" in connector_ids
    assert "obsidian" in connector_ids
    assert "imessage" in connector_ids


def _create_imessage_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT NOT NULL,
                display_name TEXT
            );
            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY,
                id TEXT NOT NULL
            );
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                attributedBody BLOB,
                date INTEGER NOT NULL,
                is_from_me INTEGER NOT NULL DEFAULT 0,
                handle_id INTEGER
            );
            CREATE TABLE chat_message_join (
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL
            );
            """
        )


def _insert_chat(path: Path, rowid: int, guid: str, display_name: str | None) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO chat (ROWID, guid, display_name) VALUES (?, ?, ?)",
            (rowid, guid, display_name),
        )


def _insert_handle(path: Path, rowid: int, handle_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO handle (ROWID, id) VALUES (?, ?)", (rowid, handle_id))


def _insert_message(
    path: Path,
    chat_id: int,
    message_id: int,
    handle_id: int,
    text: str | None,
    when: datetime,
    *,
    from_me: bool,
) -> None:
    date_ns = int((when - APPLE_EPOCH).total_seconds() * 1_000_000_000)
    _insert_raw_message(path, chat_id, message_id, handle_id, text, date_ns, from_me=from_me)


def _insert_raw_message(
    path: Path,
    chat_id: int,
    message_id: int,
    handle_id: int,
    text: str | None,
    raw_date: int,
    *,
    from_me: bool,
) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO message (ROWID, text, attributedBody, date, is_from_me, handle_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, text, None, raw_date, 1 if from_me else 0, handle_id),
        )
        conn.execute(
            "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
            (chat_id, message_id),
        )
