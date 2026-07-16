from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from taurine_core import config
from taurine_core.schemas import (
    ChatActivityStep,
    ChatMessage,
    ChatMeta,
    ChatStrippedClaim,
    ChatUsage,
    EvidencePackPayload,
    GraphEdge,
    GraphNode,
    GraphResponse,
    NeighborRef,
    NotePreview,
    NoteRecord,
)

if TYPE_CHECKING:
    from taurine_core.retrieval.snapshot import IndexSnapshot, RetrievalSnapshotPort


class _Rows:
    """Fully-materialized query result, safe to consume from any thread."""

    def __init__(self, rows: list[sqlite3.Row]) -> None:
        self._rows = rows

    def fetchall(self) -> list[sqlite3.Row]:
        return self._rows

    def fetchone(self) -> sqlite3.Row | None:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _ThreadSafeConnection:
    """Serializes sqlite access across threads.

    The server runs sync/connector/reenrich work in asyncio.to_thread pools;
    a bare sqlite3 connection is bound to its creating thread. This wrapper
    opens with check_same_thread=False and guards every operation with an
    RLock, materializing read results inside the lock so cursors never
    cross threads.
    """

    _READ_PREFIXES = ("SELECT", "PRAGMA", "WITH", "EXPLAIN")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, sql: str, params: Any = ()) -> Any:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            if sql.lstrip()[:8].upper().startswith(self._READ_PREFIXES):
                return _Rows(cursor.fetchall())
            return cursor

    def executemany(self, sql: str, seq: Any) -> Any:
        with self._lock:
            return self._conn.executemany(sql, seq)

    def executescript(self, script: str) -> Any:
        with self._lock:
            return self._conn.executescript(script)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> _ThreadSafeConnection:
        self._lock.acquire()
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return self._conn.__exit__(exc_type, exc, tb)
        finally:
            self._lock.release()

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._conn.row_factory = value


def embed_texts(texts: list[str]) -> np.ndarray:
    from .embed import embed_texts as real_embed_texts

    return real_embed_texts(texts)


class Index:
    def __init__(
        self,
        db_path: str | Path,
        *,
        read_only: bool = False,
        knowledge_snapshots: "RetrievalSnapshotPort | None" = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._read_only = bool(read_only)
        self._knowledge_snapshots = knowledge_snapshots
        if self._read_only:
            if not self.db_path.exists():
                raise FileNotFoundError(self.db_path)
            uri = f"file:{quote(str(self.db_path.resolve()), safe='/')}?mode=ro"
            raw = sqlite3.connect(uri, check_same_thread=False, uri=True)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            raw = sqlite3.connect(self.db_path, check_same_thread=False)
        raw.row_factory = sqlite3.Row
        self._conn = _ThreadSafeConnection(raw)
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not self._read_only:
            self._init_schema()
        self._disabled_sources = self._load_disabled_sources()

    def close(self) -> None:
        self._conn.close()

    def retrieval_snapshot(self) -> "IndexSnapshot":
        if self._knowledge_snapshots is None:
            raise RuntimeError("no active derived knowledge snapshot")
        return self._knowledge_snapshots.get()

    def active_snapshot_id(self) -> str:
        if self._knowledge_snapshots is None:
            return ""
        return self._knowledge_snapshots.get().build_id

    def replace_source_records(
        self,
        source_id: str,
        records: Sequence[NoteRecord],
    ) -> tuple[int, int, int]:
        """Refresh the product/source cache without building a legacy graph."""

        incoming = {
            note.note_id: note.model_copy(update={"source": source_id})
            for note in records
        }
        existing = {note.note_id: note for note in self.notes(source_id)}
        removed = sorted(set(existing) - set(incoming))
        added = sorted(set(incoming) - set(existing))
        changed = sorted(
            note_id
            for note_id in set(incoming) & set(existing)
            if incoming[note_id].content_hash != existing[note_id].content_hash
        )
        replaced = (*removed, *changed)
        with self._conn:
            for note_id in replaced:
                self._conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM aliases WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM note_entities WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM entity_relations WHERE note_id = ?", (note_id,))
                self._conn.execute(
                    "DELETE FROM edges WHERE src = ? OR dst = ?", (note_id, note_id)
                )
            for note_id in (*added, *changed):
                note = incoming[note_id]
                self._conn.execute(
                    """
                    INSERT INTO notes(
                        note_id, title, folder, modified, created, content_hash,
                        text, gist, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note.note_id,
                        note.title,
                        note.folder,
                        note.modified,
                        note.created or note.modified,
                        note.content_hash,
                        note.text,
                        _display_line(note),
                        source_id,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO notes_fts(note_id, title, body) VALUES (?, ?, ?)",
                    (note.note_id, note.title, note.text),
                )
            self._set_last_sync(time.time())
        return len(added), len(changed), len(removed)

    def delete_note(self, note_id: str) -> None:
        with self._conn:
            touched_entity_ids = self._entity_ids_touched_by_note(note_id)
            self._conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM aliases WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM note_entities WHERE note_id = ?", (note_id,))
            self._conn.execute("DELETE FROM entity_relations WHERE note_id = ?", (note_id,))
            self._conn.execute(
                "DELETE FROM edges WHERE src = ? OR dst = ?",
                (note_id, note_id),
            )
            self._recompute_entity_df(touched_entity_ids)
            self._set_last_sync(time.time())

    def set_user_card(self, text: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO meta(key, value, updated)
                VALUES ('user_card', ?, ?)
                """,
                (text, time.time()),
            )

    def user_card(self) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'user_card'"
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_source_enabled(self, source: str, enabled: bool) -> None:
        """Ledger switch: disabled sources keep their rows but drop out of the
        active index (graph, search, aliases, counts) until re-enabled."""
        disabled = set(self._disabled_sources)
        if enabled:
            disabled.discard(source)
        else:
            disabled.add(source)
        if disabled == self._disabled_sources:
            return
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO meta(key, value, updated)
                VALUES ('disabled_sources', ?, ?)
                """,
                (json.dumps(sorted(disabled)), time.time()),
            )
        self._disabled_sources = disabled

    def disabled_sources(self) -> set[str]:
        return set(self._disabled_sources)

    def _load_disabled_sources(self) -> set[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'disabled_sources'"
        ).fetchone()
        if row is None:
            return set()
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError:
            return set()
        return {str(item) for item in value} if isinstance(value, list) else set()

    def _active_sources_clause(self, column: str) -> tuple[str, tuple]:
        if not self._disabled_sources:
            return "", ()
        placeholders = ",".join("?" for _ in self._disabled_sources)
        return f" AND {column} NOT IN ({placeholders})", tuple(sorted(self._disabled_sources))

    def connector_state(self, connector_id: str) -> dict:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (_connector_meta_key(connector_id),),
        ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def set_connector_state(self, connector_id: str, state: dict) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO meta(key, value, updated)
                VALUES (?, ?, ?)
                """,
                (_connector_meta_key(connector_id), json.dumps(state, sort_keys=True), time.time()),
            )

    def neighbor_edges(self, note_id: str) -> list[tuple[str, float, str]]:
        """(other_id, weight, fact) for active neighbors — the fact-aware
        variant of neighbors(), used where edge facts are surfaced."""
        src_clause, src_params = self._active_sources_clause("n1.source")
        dst_clause, dst_params = self._active_sources_clause("n2.source")
        rows = self._conn.execute(
            f"""
            SELECT src, dst, weight, fact FROM edges
            JOIN notes n1 ON n1.note_id = edges.src
            JOIN notes n2 ON n2.note_id = edges.dst
            WHERE (src = ? OR dst = ?){src_clause}{dst_clause}
            ORDER BY weight DESC
            """,
            (note_id, note_id, *src_params, *dst_params),
        ).fetchall()
        return [
            (
                row["dst"] if row["src"] == note_id else row["src"],
                float(row["weight"]),
                row["fact"],
            )
            for row in rows
        ]

    def set_ingest_salience(self, scores: dict[str, float]) -> None:
        with self._conn:
            self._conn.execute("UPDATE notes SET ingest_salience = 1.0")
            self._conn.executemany(
                "UPDATE notes SET ingest_salience = ? WHERE note_id = ?",
                [(float(score), note_id) for note_id, score in scores.items()],
            )

    def ingest_salience(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT note_id, ingest_salience FROM notes"
        ).fetchall()
        return {row["note_id"]: float(row["ingest_salience"]) for row in rows}

    # ── meetings persistence (spec 2026-07-16-meetings-overhaul §2.1) ─
    @staticmethod
    def _meeting_from_row(row: sqlite3.Row) -> dict[str, Any]:
        suggestions_value = json.loads(row["suggestions_json"])
        if isinstance(suggestions_value, dict):
            suggestions = suggestions_value.get("edits", [])
            suggestions_base_rev = suggestions_value.get("base_rev", 0)
        else:  # defensive compatibility with pre-contract development rows
            suggestions = suggestions_value if isinstance(suggestions_value, list) else []
            suggestions_base_rev = 0
        transcript = json.loads(row["transcript_json"])
        return {
            "meeting_id": row["meeting_id"],
            "title": row["title"],
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "started_at": (
                float(row["started_at"]) if row["started_at"] is not None else None
            ),
            "ended_at": float(row["ended_at"]) if row["ended_at"] is not None else None,
            "notes_md": row["notes_md"],
            "notes_rev": int(row["notes_rev"]),
            "final_summary_md": row["final_summary_md"],
            "suggestions": suggestions if isinstance(suggestions, list) else [],
            "suggestions_base_rev": int(suggestions_base_rev),
            "transcript": transcript if isinstance(transcript, list) else [],
            "chat_id": row["chat_id"],
            "status": row["status"],
        }

    def meeting_create(self, title: str = "") -> dict[str, Any]:
        meeting_id = uuid4().hex
        now = time.time()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO meetings(meeting_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (meeting_id, title, now, now),
            )
        meeting = self.meeting_get(meeting_id)
        if meeting is None:  # pragma: no cover - insertion invariant
            raise RuntimeError("meeting insert did not persist")
        return meeting

    def meeting_get(self, meeting_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)
        ).fetchone()
        return self._meeting_from_row(row) if row is not None else None

    def meetings_list(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT meeting_id, title, created_at, updated_at, started_at, ended_at, status
            FROM meetings
            ORDER BY created_at DESC, rowid DESC
            """
        ).fetchall()
        return [
            {
                "meeting_id": row["meeting_id"],
                "title": row["title"],
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
                "started_at": (
                    float(row["started_at"]) if row["started_at"] is not None else None
                ),
                "ended_at": (
                    float(row["ended_at"]) if row["ended_at"] is not None else None
                ),
                "status": row["status"],
            }
            for row in rows
        ]

    def meeting_update(
        self,
        meeting_id: str,
        *,
        title: str | None = None,
        notes_md: str | None = None,
        suggestions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self._conn:
            row = self._conn.execute(
                "SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
            if row is None:
                raise KeyError(meeting_id)
            updates: list[str] = []
            values: list[Any] = []
            content_changed = False
            if title is not None and title != row["title"]:
                updates.append("title = ?")
                values.append(title)
                content_changed = True
            if notes_md is not None and notes_md != row["notes_md"]:
                updates.extend(("notes_md = ?", "notes_rev = notes_rev + 1"))
                values.append(notes_md)
                content_changed = True
            if suggestions is not None:
                stored = json.loads(row["suggestions_json"])
                base_rev = stored.get("base_rev", 0) if isinstance(stored, dict) else 0
                next_value = {"base_rev": int(base_rev), "edits": suggestions}
                updates.append("suggestions_json = ?")
                values.append(json.dumps(next_value, separators=(",", ":")))
            if content_changed:
                updates.append("updated_at = ?")
                values.append(time.time())
            if updates:
                values.append(meeting_id)
                self._conn.execute(
                    f"UPDATE meetings SET {', '.join(updates)} WHERE meeting_id = ?",
                    values,
                )
        meeting = self.meeting_get(meeting_id)
        if meeting is None:  # pragma: no cover - row held by transaction
            raise KeyError(meeting_id)
        return meeting

    def meeting_delete(self, meeting_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM meetings WHERE meeting_id = ?", (meeting_id,))

    def meeting_append_transcript(
        self, meeting_id: str, chunks: list[dict[str, Any]]
    ) -> None:
        if not chunks:
            return
        with self._conn:
            row = self._conn.execute(
                "SELECT transcript_json FROM meetings WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
            if row is None:
                raise KeyError(meeting_id)
            transcript = json.loads(row["transcript_json"])
            if not isinstance(transcript, list):
                transcript = []
            transcript.extend(chunks)
            self._conn.execute(
                "UPDATE meetings SET transcript_json = ? WHERE meeting_id = ?",
                (json.dumps(transcript, separators=(",", ":")), meeting_id),
            )

    def meeting_set_status(
        self,
        meeting_id: str,
        status: str,
        *,
        started_at: float | None = None,
        ended_at: float | None = None,
    ) -> None:
        if status not in {"draft", "live", "summarizing", "done"}:
            raise ValueError(f"invalid meeting status: {status}")
        assignments = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, time.time()]
        if started_at is not None:
            assignments.append("started_at = ?")
            values.append(started_at)
        if ended_at is not None:
            assignments.append("ended_at = ?")
            values.append(ended_at)
        values.append(meeting_id)
        with self._conn:
            cursor = self._conn.execute(
                f"UPDATE meetings SET {', '.join(assignments)} WHERE meeting_id = ?",
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(meeting_id)

    def meeting_set_chat(self, meeting_id: str, chat_id: str) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE meetings SET chat_id = ? WHERE meeting_id = ?",
                (chat_id, meeting_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(meeting_id)

    def meeting_for_chat(self, chat_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM meetings WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return self._meeting_from_row(row) if row is not None else None

    def meeting_set_final(
        self,
        meeting_id: str,
        *,
        final_summary_md: str,
        suggestions: list[dict[str, Any]],
        base_rev: int,
        title: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        suggestions_json = json.dumps(
            {"base_rev": int(base_rev), "edits": suggestions}, separators=(",", ":")
        )
        with self._conn:
            row = self._conn.execute(
                "SELECT title FROM meetings WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
            if row is None:
                raise KeyError(meeting_id)
            next_title = title if title is not None and not row["title"] else row["title"]
            self._conn.execute(
                """
                UPDATE meetings
                SET final_summary_md = ?, suggestions_json = ?, title = ?, updated_at = ?
                WHERE meeting_id = ?
                """,
                (final_summary_md, suggestions_json, next_title, now, meeting_id),
            )
        meeting = self.meeting_get(meeting_id)
        if meeting is None:  # pragma: no cover - row held by transaction
            raise KeyError(meeting_id)
        return meeting

    def meetings_done(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM meetings WHERE status = 'done' ORDER BY updated_at DESC"
        ).fetchall()
        return [self._meeting_from_row(row) for row in rows]

    # ── chat persistence (spec 2026-07-08-chat-knowledge-base §1.5) ────
    def create_chat(self, chat_id: str, *, now: float) -> ChatMeta:
        with self._conn:
            self._conn.execute(
                "INSERT INTO chats(chat_id, title, created, updated) VALUES (?, '', ?, ?)",
                (chat_id, now, now),
            )
        return ChatMeta(chat_id=chat_id, created=now, updated=now)

    def chat(self, chat_id: str) -> ChatMeta | None:
        row = self._conn.execute(
            """
            SELECT c.chat_id, c.title, c.created, c.updated,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = c.chat_id) AS message_count
            FROM chats c WHERE c.chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        return ChatMeta(**dict(row)) if row is not None else None

    def chats(self) -> list[ChatMeta]:
        rows = self._conn.execute(
            """
            SELECT c.chat_id, c.title, c.created, c.updated,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = c.chat_id) AS message_count
            FROM chats c ORDER BY c.updated DESC
            """
        ).fetchall()
        return [ChatMeta(**dict(row)) for row in rows]

    def rename_chat(self, chat_id: str, title: str, *, now: float) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE chats SET title = ?, updated = ? WHERE chat_id = ?",
                (title, now, chat_id),
            )

    def delete_chat(self, chat_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
            self._conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))

    def chat_registry(self, chat_id: str) -> dict:
        """Per-chat retrieval state: N#/F# registries + seed accumulator."""
        row = self._conn.execute(
            "SELECT registry_json FROM chats WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return json.loads(row["registry_json"]) if row is not None else {}

    def set_chat_registry(self, chat_id: str, registry: dict) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE chats SET registry_json = ? WHERE chat_id = ?",
                (json.dumps(registry), chat_id),
            )

    def add_chat_message(self, message: ChatMessage) -> int:
        """Persist a message; returns its rowid. Assistant rows may start
        empty (created at ack, finished via finish_chat_message)."""
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO chat_messages(chat_id, role, content, thinking, evidence_json, retrieved_ids_json,
                                          consulted_ids_json, cited_ids_json, original_query,
                                          rewritten_query, retrieval_decision, retrieval_confidence,
                                          active_build_id, activity_json,
                                          usage_json, cancelled, interrupted,
                                          stripped_claims_json, created)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.chat_id,
                    message.role,
                    message.content,
                    message.thinking,
                    message.evidence.model_dump_json() if message.evidence is not None else "null",
                    json.dumps(message.retrieved_ids),
                    json.dumps(message.consulted_ids),
                    json.dumps(message.cited_ids),
                    message.original_query,
                    message.rewritten_query,
                    message.retrieval_decision,
                    message.retrieval_confidence,
                    message.active_build_id,
                    json.dumps([item.model_dump(mode="json") for item in message.activity]),
                    message.usage.model_dump_json(),
                    int(message.cancelled),
                    int(message.interrupted),
                    json.dumps([claim.model_dump(mode="json") for claim in message.stripped_claims]),
                    message.created,
                ),
            )
            self._conn.execute(
                "UPDATE chats SET updated = ? WHERE chat_id = ?",
                (message.created, message.chat_id),
            )
            return int(cursor.lastrowid)

    def finish_chat_message(self, message: ChatMessage) -> None:
        """Overwrite the streamed-into assistant row (matched by message.id)."""
        with self._conn:
            self._conn.execute(
                """
                UPDATE chat_messages SET content = ?, thinking = ?, evidence_json = ?, retrieved_ids_json = ?,
                    consulted_ids_json = ?, cited_ids_json = ?, original_query = ?,
                    rewritten_query = ?, retrieval_decision = ?, retrieval_confidence = ?,
                    active_build_id = ?, activity_json = ?,
                    usage_json = ?, cancelled = ?, interrupted = ?,
                    stripped_claims_json = ?
                WHERE id = ?
                """,
                (
                    message.content,
                    message.thinking,
                    message.evidence.model_dump_json() if message.evidence is not None else "null",
                    json.dumps(message.retrieved_ids),
                    json.dumps(message.consulted_ids),
                    json.dumps(message.cited_ids),
                    message.original_query,
                    message.rewritten_query,
                    message.retrieval_decision,
                    message.retrieval_confidence,
                    message.active_build_id,
                    json.dumps([item.model_dump(mode="json") for item in message.activity]),
                    message.usage.model_dump_json(),
                    int(message.cancelled),
                    int(message.interrupted),
                    json.dumps([claim.model_dump(mode="json") for claim in message.stripped_claims]),
                    message.id,
                ),
            )

    def chat_messages(self, chat_id: str) -> list[ChatMessage]:
        rows = self._conn.execute(
            """
            SELECT id, chat_id, role, content, thinking, evidence_json,
                   retrieved_ids_json, consulted_ids_json, cited_ids_json,
                   original_query, rewritten_query, retrieval_decision,
                   retrieval_confidence, active_build_id,
                   activity_json, usage_json, cancelled, interrupted,
                   stripped_claims_json, created
            FROM chat_messages WHERE chat_id = ? ORDER BY id
            """,
            (chat_id,),
        ).fetchall()
        return [
            ChatMessage(
                id=int(row["id"]),
                chat_id=row["chat_id"],
                role=row["role"],
                content=row["content"],
                thinking=row["thinking"],
                evidence=(
                    EvidencePackPayload(**json.loads(row["evidence_json"]))
                    if row["evidence_json"] and row["evidence_json"] != "null"
                    else None
                ),
                retrieved_ids=list(json.loads(row["retrieved_ids_json"])),
                consulted_ids=list(json.loads(row["consulted_ids_json"])),
                cited_ids=list(json.loads(row["cited_ids_json"])),
                original_query=str(row["original_query"]),
                rewritten_query=str(row["rewritten_query"]),
                retrieval_decision=str(row["retrieval_decision"]),
                retrieval_confidence=float(row["retrieval_confidence"]),
                active_build_id=str(row["active_build_id"]),
                activity=[ChatActivityStep(**item) for item in json.loads(row["activity_json"])],
                usage=ChatUsage(**json.loads(row["usage_json"] or "{}")),
                cancelled=bool(row["cancelled"]),
                interrupted=bool(row["interrupted"]),
                stripped_claims=[
                    ChatStrippedClaim(**item)
                    for item in json.loads(row["stripped_claims_json"] or "[]")
                ],
                created=float(row["created"]),
            )
            for row in rows
        ]

    def delete_chat_message(self, message_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chat_messages WHERE id = ?", (message_id,))

    def note(self, note_id: str) -> NoteRecord | None:
        row = self._conn.execute(
            """
            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
                   source, quality_rank
            FROM notes WHERE note_id = ?
            """,
            (note_id,),
        ).fetchone()
        if row is None:
            return None
        return NoteRecord(**dict(row))

    def notes(self, source: str | None = None) -> list[NoteRecord]:
        if source is not None:
            rows = self._conn.execute(
                """
                SELECT note_id, title, folder, modified, created, content_hash, text, gist,
                       source, quality_rank
                FROM notes WHERE source = ? ORDER BY note_id
                """,
                (source,),
            ).fetchall()
            return [NoteRecord(**dict(row)) for row in rows]
        rows = self._conn.execute(
            """
            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
                   source, quality_rank
            FROM notes ORDER BY note_id
            """
        ).fetchall()
        return [NoteRecord(**dict(row)) for row in rows]

    def note_ids(self, source: str | None = None) -> set[str]:
        if source is not None:
            rows = self._conn.execute(
                "SELECT note_id FROM notes WHERE source = ?",
                (source,),
            ).fetchall()
            return {row["note_id"] for row in rows}
        rows = self._conn.execute("SELECT note_id FROM notes").fetchall()
        return {row["note_id"] for row in rows}

    def note_count_by_source(self) -> dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT source, COUNT(*) AS count
            FROM notes GROUP BY source ORDER BY source
            """
        ).fetchall()
        return {row["source"]: int(row["count"]) for row in rows}

    def preview(self, note_id: str) -> NotePreview | None:
        note = self.note(note_id)
        if note is None:
            return None
        neighbors: list[NeighborRef] = []
        for neighbor_id, weight, fact in self.neighbor_edges(note_id):
            neighbor = self.note(neighbor_id)
            if neighbor is None:
                continue
            neighbors.append(
                NeighborRef(
                    note_id=neighbor.note_id,
                    title=neighbor.title,
                    source=neighbor.source,
                    weight=weight,
                    fact=fact,
                )
            )
        neighbors.sort(key=lambda item: item.weight, reverse=True)
        return NotePreview(
            note_id=note.note_id,
            title=note.title,
            source=note.source,
            gist=note.gist,
            text=note.text,
            neighbors=neighbors,
        )

    def aliases(self) -> list[tuple[str, str, bool]]:
        clause, params = self._active_sources_clause("n.source")
        rows = self._conn.execute(
            f"""
            SELECT a.alias_lowercase AS alias_lowercase, a.note_id AS note_id, a.is_title AS is_title
            FROM aliases a JOIN notes n ON n.note_id = a.note_id
            WHERE 1=1{clause}
            ORDER BY a.note_id, a.is_title DESC, a.alias_lowercase
            """,
            params,
        ).fetchall()
        return [
            (row["alias_lowercase"], row["note_id"], bool(row["is_title"]))
            for row in rows
        ]

    def graph(self) -> GraphResponse:
        active_notes = [
            note for note in self.notes() if note.source not in self._disabled_sources
        ]
        active_ids = {note.note_id for note in active_notes}
        degree: dict[str, int] = defaultdict(int)
        edge_rows = self._conn.execute(
            """
            SELECT src, dst, type, weight, shared_json, fact, fact_provenance
            FROM edges ORDER BY weight DESC, src, dst
            """
        ).fetchall()
        edges: list[GraphEdge] = []
        for row in edge_rows:
            if row["src"] not in active_ids or row["dst"] not in active_ids:
                continue
            degree[row["src"]] += 1
            degree[row["dst"]] += 1
            edges.append(
                GraphEdge(
                    src=row["src"],
                    dst=row["dst"],
                    type=row["type"],
                    weight=float(row["weight"]),
                    shared=json.loads(row["shared_json"]),
                    fact=row["fact"],
                    fact_provenance=row["fact_provenance"],
                )
            )
        nodes = [
            GraphNode(
                note_id=note.note_id,
                title=note.title,
                degree=degree[note.note_id],
                source=note.source,
                gist=note.gist,
            )
            for note in active_notes
        ]
        return GraphResponse(nodes=nodes, edges=edges)

    def status(self) -> tuple[int, float | None]:
        clause, params = self._active_sources_clause("source")
        row = self._conn.execute(
            f"SELECT COUNT(*) AS count FROM notes WHERE 1=1{clause}", params
        ).fetchone()
        metadata = self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_sync_ts'"
        ).fetchone()
        last_sync = float(metadata["value"]) if metadata is not None else None
        return int(row["count"]), last_sync

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    folder TEXT NOT NULL,
                    modified REAL NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    gist TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'apple_notes'
                );

                CREATE TABLE IF NOT EXISTS aliases (
                    alias_lowercase TEXT NOT NULL,
                    note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
                    is_title INTEGER NOT NULL,
                    PRIMARY KEY(alias_lowercase, note_id)
                );

                CREATE TABLE IF NOT EXISTS enrich_cache (
                    note_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    enrichment_json TEXT NOT NULL,
                    PRIMARY KEY(note_id, content_hash)
                );

                CREATE TABLE IF NOT EXISTS entities(
                    entity_id INTEGER PRIMARY KEY,
                    canonical TEXT NOT NULL,
                    etype TEXT NOT NULL,
                    df INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(canonical, etype)
                );

                CREATE TABLE IF NOT EXISTS entity_surfaces(
                    entity_id INTEGER NOT NULL,
                    surface TEXT NOT NULL,
                    PRIMARY KEY(entity_id, surface)
                );

                CREATE TABLE IF NOT EXISTS note_entities(
                    note_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    conf REAL NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(note_id, entity_id)
                );

                CREATE TABLE IF NOT EXISTS entity_relations(
                    note_id TEXT NOT NULL,
                    src_entity INTEGER NOT NULL,
                    rel_type TEXT NOT NULL,
                    dst_entity INTEGER NOT NULL,
                    conf REAL NOT NULL,
                    PRIMARY KEY(note_id, src_entity, rel_type, dst_entity)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('chunk', 'gist', 'utterance')),
                    text TEXT NOT NULL,
                    vec BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    shared_json TEXT NOT NULL,
                    fact TEXT NOT NULL DEFAULT '',
                    fact_provenance TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(src, dst)
                );

                CREATE TABLE IF NOT EXISTS term_vectors (
                    term TEXT PRIMARY KEY,
                    vec BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fact_embeddings (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    fact_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    vec BLOB NOT NULL,
                    PRIMARY KEY(src, dst)
                );

                CREATE TABLE IF NOT EXISTS edge_judgments (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    sig TEXT NOT NULL,
                    related INTEGER NOT NULL,
                    fact TEXT NOT NULL DEFAULT '',
                    judged_at REAL NOT NULL,
                    PRIMARY KEY(src, dst)
                );

                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    registry_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS meetings (
                    meeting_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    ended_at REAL,
                    notes_md TEXT NOT NULL DEFAULT '',
                    notes_rev INTEGER NOT NULL DEFAULT 0,
                    final_summary_md TEXT NOT NULL DEFAULT '',
                    suggestions_json TEXT NOT NULL DEFAULT '{"base_rev":0,"edits":[]}',
                    transcript_json TEXT NOT NULL DEFAULT '[]',
                    chat_id TEXT,
                    status TEXT NOT NULL DEFAULT 'draft'
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    thinking TEXT NOT NULL DEFAULT '',
                    evidence_json TEXT NOT NULL DEFAULT 'null',
                    retrieved_ids_json TEXT NOT NULL DEFAULT '[]',
                    consulted_ids_json TEXT NOT NULL DEFAULT '[]',
                    cited_ids_json TEXT NOT NULL DEFAULT '[]',
                    original_query TEXT NOT NULL DEFAULT '',
                    rewritten_query TEXT NOT NULL DEFAULT '',
                    retrieval_decision TEXT NOT NULL DEFAULT '',
                    retrieval_confidence REAL NOT NULL DEFAULT 0,
                    active_build_id TEXT NOT NULL DEFAULT '',
                    activity_json TEXT NOT NULL DEFAULT '[]',
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    cancelled INTEGER NOT NULL DEFAULT 0,
                    interrupted INTEGER NOT NULL DEFAULT 0,
                    stripped_claims_json TEXT NOT NULL DEFAULT '[]',
                    created REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated REAL NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
                USING fts5(note_id UNINDEXED, title, body);
                """
            )
            self._ensure_notes_source_column()
            self._ensure_edges_fact_column()
            self._ensure_edges_fact_provenance_column()
            self._ensure_notes_created_column()
            self._ensure_notes_quality_rank_column()
            self._ensure_notes_value_columns()
            self._ensure_chat_messages_columns()

    def _ensure_chat_messages_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(chat_messages)").fetchall()
        }
        required: tuple[tuple[str, str], ...] = (
            ("evidence_json", "TEXT NOT NULL DEFAULT 'null'"),
            ("retrieved_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("consulted_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("cited_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("original_query", "TEXT NOT NULL DEFAULT ''"),
            ("rewritten_query", "TEXT NOT NULL DEFAULT ''"),
            ("retrieval_decision", "TEXT NOT NULL DEFAULT ''"),
            ("retrieval_confidence", "REAL NOT NULL DEFAULT 0"),
            ("active_build_id", "TEXT NOT NULL DEFAULT ''"),
            ("interrupted", "INTEGER NOT NULL DEFAULT 0"),
            ("stripped_claims_json", "TEXT NOT NULL DEFAULT '[]'"),
        )
        for name, definition in required:
            if name not in columns:
                self._conn.execute(
                    f"ALTER TABLE chat_messages ADD COLUMN {name} {definition}"
                )

    def _ensure_notes_created_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "created" in columns:
            return
        self._conn.execute("ALTER TABLE notes ADD COLUMN created REAL NOT NULL DEFAULT 0")
        self._conn.execute("UPDATE notes SET created = modified WHERE created = 0")

    def _ensure_notes_quality_rank_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "quality_rank" in columns:
            return
        self._conn.execute(
            "ALTER TABLE notes ADD COLUMN quality_rank REAL NOT NULL DEFAULT 0.0"
        )

    def _ensure_notes_value_columns(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "ingest_salience" not in columns:
            self._conn.execute("ALTER TABLE notes ADD COLUMN ingest_salience REAL DEFAULT 1.0")
        if "value_score" not in columns:
            self._conn.execute("ALTER TABLE notes ADD COLUMN value_score REAL DEFAULT 0.0")

    def _ensure_notes_source_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        if "source" in columns:
            return
        self._conn.execute(
            "ALTER TABLE notes ADD COLUMN source TEXT NOT NULL DEFAULT 'apple_notes'"
        )

    def _ensure_edges_fact_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "fact" in columns:
            return
        self._conn.execute("ALTER TABLE edges ADD COLUMN fact TEXT NOT NULL DEFAULT ''")

    def _ensure_edges_fact_provenance_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "fact_provenance" in columns:
            return
        self._conn.execute(
            "ALTER TABLE edges ADD COLUMN fact_provenance TEXT NOT NULL DEFAULT ''"
        )

    def _entity_ids_touched_by_note(self, note_id: str) -> set[int]:
        rows = self._conn.execute(
            """
            SELECT entity_id FROM note_entities WHERE note_id = ?
            UNION
            SELECT src_entity AS entity_id FROM entity_relations WHERE note_id = ?
            UNION
            SELECT dst_entity AS entity_id FROM entity_relations WHERE note_id = ?
            """,
            (note_id, note_id, note_id),
        ).fetchall()
        return {int(row["entity_id"]) for row in rows}

    def _recompute_entity_df(self, entity_ids: set[int]) -> None:
        for entity_id in sorted(entity_ids):
            self._conn.execute(
                """
                UPDATE entities
                SET df = (
                    SELECT COUNT(DISTINCT note_id)
                    FROM note_entities
                    WHERE entity_id = ?
                )
                WHERE entity_id = ?
                """,
                (entity_id, entity_id),
            )

    def _set_last_sync(self, timestamp: float) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO metadata(key, value)
            VALUES ('last_sync_ts', ?)
            """,
            (str(timestamp),),
        )

def _display_line(note: NoteRecord) -> str:
    for line in note.text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return _clamp_display_line(cleaned)
    return _clamp_display_line(note.title)


def _clamp_display_line(value: str) -> str:
    max_chars = config.GIST_MAX_CHARS
    if len(value) <= max_chars:
        return value
    clipped = value[:max_chars]
    if clipped[-1].isspace() or value[max_chars : max_chars + 1].isspace():
        return clipped.rstrip()
    head = clipped.rsplit(" ", 1)[0].rstrip()
    return head or clipped.rstrip()


def _connector_meta_key(connector_id: str) -> str:
    return f"connector:{connector_id}"
