from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from taurine_core import config, server as server_module
from taurine_core.indexer.store import Index
from taurine_core.live.grounding import GroundingEngine
from taurine_core.live.ingest import LiveSession
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.schemas import Chunk, MeetingSummary, SyncReport
from taurine_core.server import Runtime, UpdateBus, create_app, _parse_control


@pytest.fixture
def index(tmp_path: Any) -> Index:
    store = Index(tmp_path / "meetings.db")
    try:
        yield store
    finally:
        store.close()


def test_meeting_store_crud_and_revision_semantics(
    index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = {"value": 100.0}
    monkeypatch.setattr("taurine_core.indexer.store.time.time", lambda: now["value"])

    created = index.meeting_create()
    assert len(created["meeting_id"]) == 32
    assert created == {
        "meeting_id": created["meeting_id"],
        "title": "",
        "created_at": 100.0,
        "updated_at": 100.0,
        "started_at": None,
        "ended_at": None,
        "notes_md": "",
        "notes_rev": 0,
        "final_summary_md": "",
        "suggestions": [],
        "suggestions_base_rev": 0,
        "transcript": [],
        "chat_id": None,
        "status": "draft",
    }

    now["value"] = 101.0
    updated = index.meeting_update(
        created["meeting_id"], title="Planning", notes_md="First note"
    )
    assert updated["title"] == "Planning"
    assert updated["notes_md"] == "First note"
    assert updated["notes_rev"] == 1
    assert updated["updated_at"] == 101.0

    now["value"] = 102.0
    unchanged = index.meeting_update(created["meeting_id"], notes_md="First note")
    assert unchanged["notes_rev"] == 1
    assert unchanged["updated_at"] == 101.0

    index.meeting_delete(created["meeting_id"])
    assert index.meeting_get(created["meeting_id"]) is None


def test_meeting_list_excludes_bodies_and_is_newest_first(index: Index) -> None:
    first = index.meeting_create("First")
    second = index.meeting_create("Second")
    index.meeting_update(first["meeting_id"], notes_md="private notes")
    index.meeting_append_transcript(
        first["meeting_id"],
        [{"channel": 0, "text": "private transcript", "ts": 1.0, "seq": 1}],
    )

    rows = index.meetings_list()
    assert [row["meeting_id"] for row in rows] == [second["meeting_id"], first["meeting_id"]]
    assert set(rows[0]) == {
        "meeting_id",
        "title",
        "created_at",
        "updated_at",
        "started_at",
        "ended_at",
        "status",
    }
    assert "notes_md" not in rows[1]
    assert "transcript" not in rows[1]
    assert "suggestions" not in rows[1]


def test_meeting_transcript_append_preserves_batch_order(index: Index) -> None:
    meeting = index.meeting_create()
    first = [
        {"channel": 0, "text": "one", "ts": 1.0, "seq": 1},
        {"channel": 1, "text": "two", "ts": 2.0, "seq": 2},
    ]
    second = [{"channel": 0, "text": "three", "ts": 3.0, "seq": 3}]

    index.meeting_append_transcript(meeting["meeting_id"], first)
    index.meeting_append_transcript(meeting["meeting_id"], second)

    assert index.meeting_get(meeting["meeting_id"])["transcript"] == first + second


async def test_session_transcript_buffer_flushes_at_count_or_interval(
    index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_count = index.meeting_create("Count")
    count_buffer = server_module._MeetingTranscriptBuffer(index, by_count["meeting_id"])
    for seq in range(1, 20):
        await count_buffer.append({"channel": 0, "text": str(seq), "ts": seq, "seq": seq})
    assert index.meeting_get(by_count["meeting_id"])["transcript"] == []
    await count_buffer.append({"channel": 0, "text": "20", "ts": 20, "seq": 20})
    assert len(index.meeting_get(by_count["meeting_id"])["transcript"]) == 20

    monkeypatch.setattr(server_module._MeetingTranscriptBuffer, "FLUSH_INTERVAL_S", 0)
    by_time = index.meeting_create("Time")
    time_buffer = server_module._MeetingTranscriptBuffer(index, by_time["meeting_id"])
    await time_buffer.append({"channel": 1, "text": "timer", "ts": 1, "seq": 1})
    await asyncio.sleep(0.01)
    assert index.meeting_get(by_time["meeting_id"])["transcript"] == [
        {"channel": 1, "text": "timer", "ts": 1, "seq": 1}
    ]


def test_meeting_status_chat_and_final_result_contract(index: Index) -> None:
    meeting = index.meeting_create()
    meeting_id = meeting["meeting_id"]
    index.meeting_set_status(meeting_id, "live", started_at=10.0)
    live = index.meeting_get(meeting_id)
    assert live["status"] == "live"
    assert live["started_at"] == 10.0

    index.meeting_set_chat(meeting_id, "chat-1234")
    assert index.meeting_for_chat("chat-1234")["meeting_id"] == meeting_id

    final = index.meeting_set_final(
        meeting_id,
        final_summary_md="Summary",
        suggestions=[
            {
                "id": "edit-1",
                "anchor": "",
                "occurrence": 0,
                "insert_md": "Add this",
                "rationale": "Useful",
                "status": "pending",
            }
        ],
        base_rev=4,
        title="Generated title",
    )
    assert final["title"] == "Generated title"
    assert final["suggestions_base_rev"] == 4
    assert final["suggestions"][0]["id"] == "edit-1"

    index.meeting_set_status(meeting_id, "done", ended_at=20.0)
    done = index.meetings_done()
    assert done[0]["meeting_id"] == meeting_id
    assert done[0]["ended_at"] == 20.0
    with pytest.raises(ValueError, match="invalid meeting status"):
        index.meeting_set_status(meeting_id, "archived")


def test_meetings_schema_uses_object_suggestions_default(index: Index) -> None:
    row = next(
        row
        for row in index._conn.execute("PRAGMA table_info(meetings)").fetchall()
        if row["name"] == "suggestions_json"
    )
    assert json.loads(row["dflt_value"].strip("'")) == {"base_rev": 0, "edits": []}


def _runtime(index: Index, live_sessions: list[Any] | None = None) -> Runtime:
    sessions = live_sessions if live_sessions is not None else []

    def make_live_session(
        emit: Any,
        logger: Any,
        on_final_chunk: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> FakeLiveSession:
        del emit, logger
        live = FakeLiveSession(on_final_chunk=on_final_chunk)
        sessions.append(live)
        return live

    return Runtime(
        index=index,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=make_live_session,
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(on_chunk),
    )


def test_meetings_rest_partial_patch_delete_and_updates(index: Index) -> None:
    runtime = _runtime(index)
    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/updates") as updates:
            # This runtime has no connector registry, so that conditional
            # snapshot kind is omitted.
            snapshot = {
                updates.receive_json()["type"] for _ in range(len(UpdateBus.KINDS) - 1)
            }
            assert "meetings" in snapshot

            created = client.post("/meetings", json={"title": "Kickoff"})
            assert created.status_code == 200
            meeting = created.json()
            assert meeting["title"] == "Kickoff"
            assert meeting["notes_rev"] == 0
            assert updates.receive_json()["type"] == "meetings"

            notes = client.patch(
                f"/meetings/{meeting['meeting_id']}", json={"notes_md": "Notes"}
            )
            assert notes.status_code == 200
            assert notes.json()["notes_rev"] == 1

            renamed = client.patch(
                f"/meetings/{meeting['meeting_id']}", json={"title": "Renamed"}
            )
            assert renamed.json()["notes_md"] == "Notes"
            assert updates.receive_json()["type"] == "meetings"

            listed = client.get("/meetings").json()
            assert listed == {"meetings": [index.meetings_list()[0]]}
            assert client.get("/meetings/missing").status_code == 404
            assert client.patch("/meetings/missing", json={"title": "Nope"}).status_code == 404
            assert client.delete("/meetings/missing").status_code == 404
            assert client.delete(f"/meetings/{meeting['meeting_id']}").json() == {"ok": True}
            assert updates.receive_json()["type"] == "meetings"


def test_parse_start_control_accepts_optional_meeting_id() -> None:
    legacy = _parse_control(
        json.dumps({"type": "start", "channels": [{"id": 0, "label": "mic"}]})
    )
    bound = _parse_control(
        json.dumps(
            {
                "type": "start",
                "channels": [{"id": 0, "label": "mic"}],
                "meeting_id": "abc123",
            }
        )
    )
    assert legacy.meeting_id is None
    assert bound.meeting_id == "abc123"


async def test_live_session_final_chunk_hook_uses_assigned_sequence() -> None:
    class NoopAmbient:
        async def on_window(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class NoopTrigger:
        def offer(self, *args: Any, **kwargs: Any) -> tuple[bool, str]:
            del args, kwargs
            return False, "no_change"

        def note_summary(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def note_rejected(self) -> None:
            return None

    class NoopSummarizer:
        async def summarize(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    registry = EvidenceRegistry()
    finals: list[dict[str, Any]] = []
    session = LiveSession(
        registry=registry,
        ambient=NoopAmbient(),  # type: ignore[arg-type]
        grounding=GroundingEngine(registry),
        trigger=NoopTrigger(),  # type: ignore[arg-type]
        summarizer=NoopSummarizer(),  # type: ignore[arg-type]
        emit=lambda event: None,
        on_final_chunk=finals.append,
    )

    await session.on_chunk(Chunk(channel=1, text="working", ts=1.0, final=False))
    await session.on_chunk(Chunk(channel=1, text="final words", ts=2.0, final=True))

    assert finals == [{"channel": 1, "text": "final words", "ts": 2.0, "seq": 1}]


def test_rest_patch_suggestions_only_changes_statuses(index: Index) -> None:
    meeting = index.meeting_create()
    suggestion = {
        "id": "edit-1",
        "anchor": "topic",
        "occurrence": 1,
        "insert_md": "More detail",
        "rationale": "Useful context",
        "status": "pending",
    }
    index.meeting_set_final(
        meeting["meeting_id"],
        final_summary_md="Summary",
        suggestions=[suggestion],
        base_rev=3,
    )

    with TestClient(create_app(_runtime(index))) as client:
        changed_content = dict(suggestion, insert_md="Injected content", status="accepted")
        response = client.patch(
            f"/meetings/{meeting['meeting_id']}",
            json={"suggestions": [changed_content]},
        )
        assert response.status_code == 422

        accepted = dict(suggestion, status="accepted")
        response = client.patch(
            f"/meetings/{meeting['meeting_id']}", json={"suggestions": [accepted]}
        )
        assert response.status_code == 200
        assert response.json()["suggestions"] == [accepted]
        assert response.json()["suggestions_base_rev"] == 3

    other = dict(suggestion, id="edit-2")
    with pytest.raises(HTTPException) as error:
        server_module._suggestion_status_patch(
            [suggestion, other], [other, suggestion]
        )
    assert error.value.status_code == 422


@dataclass
class FakeLiveSession:
    on_final_chunk: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None
    chunks: list[Chunk] = field(default_factory=list)
    closed: bool = False

    async def on_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        if chunk.final and self.on_final_chunk is not None:
            result = self.on_final_chunk(
                {
                    "channel": chunk.channel,
                    "text": chunk.text,
                    "ts": chunk.ts,
                    "seq": len(self.chunks),
                }
            )
            if asyncio.iscoroutine(result):
                await result

    async def set_actionable_status(self, actionable_id: str, status: str) -> None:
        del actionable_id, status

    def current_summary(self) -> MeetingSummary:
        return MeetingSummary(summary_md=" | ".join(chunk.text for chunk in self.chunks))

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class FakeASRClient:
    on_chunk: Callable[[Chunk], Awaitable[None]]
    starts: list[list[int]] = field(default_factory=list)
    stopped: int = 0
    sent: int = 0

    async def start(self, channels: list[int]) -> None:
        self.starts.append(channels)

    async def send_pcm_s16(self, channel: int, pcm: bytes) -> bool:
        del pcm
        self.sent += 1
        await self.on_chunk(
            Chunk(channel=channel, text=f"session-{self.sent}", ts=float(self.sent), final=True)
        )
        return True

    async def stop(self) -> None:
        self.stopped += 1


def test_each_start_builds_fresh_session_and_keeps_transcripts_separate(index: Index) -> None:
    first = index.meeting_create("First")
    second = index.meeting_create("Second")
    live_sessions: list[FakeLiveSession] = []
    runtime = _runtime(index, live_sessions)

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/session") as ws:
            assert ws.receive_json()["type"] == "status"
            ws.send_json(
                {
                    "type": "start",
                    "channels": [{"id": 0, "label": "mic"}],
                    "meeting_id": first["meeting_id"],
                }
            )
            ws.send_bytes(b"\x00\x01\x00")
            ws.send_json(
                {
                    "type": "start",
                    "channels": [{"id": 1, "label": "system"}],
                    "meeting_id": second["meeting_id"],
                }
            )
            ws.send_bytes(b"\x01\x02\x00")
            ws.send_json({"type": "stop"})

    assert len(live_sessions) == 2
    assert [chunk.channel for chunk in live_sessions[0].chunks] == [0]
    assert [chunk.channel for chunk in live_sessions[1].chunks] == [1]
    assert [row["channel"] for row in index.meeting_get(first["meeting_id"])["transcript"]] == [0]
    assert [row["channel"] for row in index.meeting_get(second["meeting_id"])["transcript"]] == [1]
    assert index.meeting_get(second["meeting_id"])["status"] == "done"


def test_disconnect_grace_finalizes_only_when_timer_fires(
    index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MEETING_FINALIZE_GRACE_S", 3600)
    meeting = index.meeting_create("Grace")
    runtime = _runtime(index)

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/session") as ws:
            assert ws.receive_json()["type"] == "status"
            ws.send_json(
                {
                    "type": "start",
                    "channels": [{"id": 0, "label": "mic"}],
                    "meeting_id": meeting["meeting_id"],
                }
            )
            ws.send_bytes(b"\x00\x01\x00")

        assert index.meeting_get(meeting["meeting_id"])["status"] == "live"
        handle = runtime.meeting_finalizers[meeting["meeting_id"]]
        handle._run()
        assert index.meeting_get(meeting["meeting_id"])["status"] == "done"


def test_disconnect_flushes_final_chunk_emitted_by_asr_stop(index: Index) -> None:
    meeting = index.meeting_create("Stop final")

    @dataclass
    class StopFinalASR(FakeASRClient):
        async def stop(self) -> None:
            self.stopped += 1
            if self.stopped == 1:
                await self.on_chunk(
                    Chunk(channel=0, text="last words", ts=9.0, final=True)
                )

    runtime = Runtime(
        index=index,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger, on_final_chunk=None: FakeLiveSession(
            on_final_chunk=on_final_chunk
        ),
        make_asr_client=lambda on_chunk, on_status, index_status: StopFinalASR(on_chunk),
    )

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/session") as ws:
            assert ws.receive_json()["type"] == "status"
            ws.send_json(
                {
                    "type": "start",
                    "channels": [{"id": 0, "label": "mic"}],
                    "meeting_id": meeting["meeting_id"],
                }
            )

    stored = index.meeting_get(meeting["meeting_id"])
    assert stored["transcript"] == [
        {"channel": 0, "text": "last words", "ts": 9.0, "seq": 1}
    ]


def test_stop_finalization_is_idempotent(index: Index) -> None:
    meeting = index.meeting_create("Stop")
    runtime = _runtime(index)

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/session") as ws:
            assert ws.receive_json()["type"] == "status"
            ws.send_json(
                {
                    "type": "start",
                    "channels": [{"id": 0, "label": "mic"}],
                    "meeting_id": meeting["meeting_id"],
                }
            )
            ws.send_json({"type": "stop"})
            ws.send_json({"type": "stop"})

    finalized = index.meeting_get(meeting["meeting_id"])
    assert finalized["status"] == "done"
    assert finalized["ended_at"] is not None
    assert meeting["meeting_id"] not in runtime.meeting_finalizers


async def test_stale_disconnect_cannot_replace_newer_epoch_finalizer(
    index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MEETING_FINALIZE_GRACE_S", 3600)
    meeting = index.meeting_create("Epoch")
    meeting_id = meeting["meeting_id"]
    index.meeting_set_status(meeting_id, "live", started_at=1.0)
    runtime = _runtime(index)
    runtime.meeting_epochs[meeting_id] = 2
    newer = server_module._MeetingFinalizationSnapshot(meeting_id, 2, "", [])
    stale = server_module._MeetingFinalizationSnapshot(meeting_id, 1, "", [])

    server_module._schedule_meeting_finalization(runtime, newer)
    newer_handle = runtime.meeting_finalizers[meeting_id]
    server_module._schedule_meeting_finalization(runtime, stale)

    assert runtime.meeting_finalizers[meeting_id] is newer_handle
    newer_handle._run()
    assert index.meeting_get(meeting_id)["status"] == "done"
