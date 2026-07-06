from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from taurine_core.indexer.sync import SyncReport
from taurine_core.schemas import (
    Chunk,
    GraphEdge,
    GraphNode,
    GraphResponse,
    IndexStatus,
    TranscriptEvent,
)
from taurine_core.server import Runtime, create_app, required_secrets, run_sync_once_safely


@dataclass
class GraphIndex:
    graph_response: GraphResponse = field(
        default_factory=lambda: GraphResponse(
            nodes=[GraphNode(note_id="n1", title="Roadmap", degree=1)],
            edges=[
                GraphEdge(
                    src="n1",
                    dst="n2",
                    type="topic",
                    weight=0.7,
                    shared=["roadmap"],
                )
            ],
        )
    )

    def graph(self) -> GraphResponse:
        return self.graph_response

    def status(self) -> tuple[int, float | None]:
        return (1, 123.0)


@dataclass
class EchoLiveSession:
    emit: Any
    dismisses: list[str] = field(default_factory=list)

    async def on_chunk(self, chunk: Chunk) -> None:
        await self.emit(
            TranscriptEvent(
                channel=chunk.channel,
                text=chunk.text,
                final=chunk.final,
                ts=chunk.ts,
            )
        )

    async def dismiss(self, card_id: str) -> None:
        self.dismisses.append(card_id)


@dataclass
class FakeASRClient:
    on_chunk: Any
    on_status: Any
    index_status: Any
    starts: list[list[int]] = field(default_factory=list)
    sent: list[tuple[int, bytes]] = field(default_factory=list)
    stopped: bool = False

    async def start(self, channels: list[int]) -> None:
        self.starts.append(channels)

    async def send_pcm_s16(self, channel: int, pcm: bytes) -> bool:
        self.sent.append((channel, pcm))
        await self.on_chunk(Chunk(channel=channel, text="heard words", ts=55.0, final=True))
        return True

    async def stop(self) -> None:
        self.stopped = True


def test_graph_and_sync_endpoints_use_runtime_services() -> None:
    sync_calls: list[bool] = []
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: sync_calls.append(True)
        or SyncReport(added=1, changed=2, removed=3, enrich_failures=4),
        open_note=lambda note_id: None,
        make_live_session=lambda emit: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.get("/graph").json() == {
            "nodes": [{"note_id": "n1", "title": "Roadmap", "degree": 1}],
            "edges": [
                {
                    "src": "n1",
                    "dst": "n2",
                    "type": "topic",
                    "weight": 0.7,
                    "shared": ["roadmap"],
                }
            ],
        }
        assert client.post("/index/sync").json() == {
            "added": 1,
            "changed": 2,
            "removed": 3,
            "enrich_failures": 4,
        }
    assert sync_calls == [True]


def test_session_routes_binary_frames_and_serializes_schema_events() -> None:
    live_sessions: list[EchoLiveSession] = []
    asr_clients: list[FakeASRClient] = []
    opened: list[str] = []

    def make_live_session(emit: Any) -> EchoLiveSession:
        live = EchoLiveSession(emit)
        live_sessions.append(live)
        return live

    def make_asr_client(on_chunk: Any, on_status: Any, index_status: Any) -> FakeASRClient:
        asr = FakeASRClient(on_chunk, on_status, index_status)
        asr_clients.append(asr)
        return asr

    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=opened.append,
        make_live_session=make_live_session,
        make_asr_client=make_asr_client,
    )

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/session") as ws:
            ws.send_json({"type": "start", "channels": [{"id": 2, "label": "system"}]})
            ws.send_bytes(bytes([2]) + b"\x01\x00\x02\x00")

            greeting = ws.receive_json()
            assert greeting["type"] == "status"  # initial status on connect

            assert ws.receive_json() == {
                "type": "transcript",
                "channel": 2,
                "text": "heard words",
                "final": True,
                "ts": 55.0,
            }

            ws.send_json({"type": "dismiss", "card_id": "card-1"})
            ws.send_json({"type": "open_note", "note_id": "n1"})
            ws.send_json({"type": "stop"})

    assert asr_clients[0].starts == [[2]]
    assert asr_clients[0].sent == [(2, b"\x01\x00\x02\x00")]
    assert asr_clients[0].stopped
    assert live_sessions[0].dismisses == ["card-1"]
    assert opened == ["n1"]
    assert asr_clients[0].index_status() == IndexStatus(notes=1, last_sync=123.0)


def test_required_secrets_error_names_missing_key() -> None:
    with pytest.raises(RuntimeError, match="TAURINE_ASR_TOKEN"):
        required_secrets({})


@pytest.mark.asyncio
async def test_periodic_sync_failure_is_nonfatal() -> None:
    calls = 0

    def failing_sync() -> SyncReport:
        nonlocal calls
        calls += 1
        raise RuntimeError("notes unavailable")

    runtime = Runtime(
        index=GraphIndex(),
        sync_once=failing_sync,
        open_note=lambda note_id: None,
        make_live_session=lambda emit: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    await run_sync_once_safely(runtime)

    assert calls == 1
