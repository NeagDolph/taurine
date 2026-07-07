from __future__ import annotations

import json
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from taurine_core import session_log
from taurine_core.indexer.sync import SyncReport
from taurine_core.schemas import (
    Chunk,
    ConnectorInfo,
    ConnectorSyncReport,
    GraphEdge,
    GraphNode,
    GraphResponse,
    IndexStatus,
    NeighborRef,
    NotePreview,
    ReenrichStatus,
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

    def preview(self, note_id: str) -> NotePreview | None:
        if note_id != "obsidian:vault/alpha.md":
            return None
        return NotePreview(
            note_id=note_id,
            title="Alpha",
            source="obsidian",
            gist="Alpha gist",
            text="Alpha body",
            neighbors=[
                NeighborRef(
                    note_id="apple:beta",
                    title="Beta",
                    source="apple_notes",
                    weight=0.7,
                )
            ],
        )


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


@dataclass
class FakeConnectorRegistry:
    syncing: set[str] = field(default_factory=set)
    enabled_calls: list[tuple[str, bool, bool]] = field(default_factory=list)
    config_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    sync_calls: list[str] = field(default_factory=list)

    def list_info(self, index: object) -> list[ConnectorInfo]:
        del index
        return [
            ConnectorInfo(
                connector_id="fake",
                label="Fake",
                kind="local",
                enabled=True,
                available=True,
                docs=3,
                syncing="fake" in self.syncing,
                config={"path": "/tmp/fake"},
            )
        ]

    def set_enabled(
        self,
        index: object,
        connector_id: str,
        enabled: bool,
        *,
        purge: bool = False,
    ) -> None:
        del index
        if connector_id != "fake":
            raise KeyError(connector_id)
        self.enabled_calls.append((connector_id, enabled, purge))

    def set_config(self, index: object, connector_id: str, config: dict) -> None:
        del index
        if connector_id != "fake":
            raise KeyError(connector_id)
        self.config_calls.append((connector_id, dict(config)))

    def sync_connector(
        self,
        connector_id: str,
        index: object,
        llm: object,
    ) -> ConnectorSyncReport:
        del index, llm
        if connector_id != "fake":
            raise KeyError(connector_id)
        self.sync_calls.append(connector_id)
        time.sleep(0.2)
        return ConnectorSyncReport(connector_id=connector_id, added=1)


def test_graph_and_sync_endpoints_use_runtime_services() -> None:
    sync_calls: list[bool] = []
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: sync_calls.append(True)
        or SyncReport(added=1, changed=2, removed=3, enrich_failures=4),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.get("/graph").json() == {
            "nodes": [
                {"note_id": "n1", "title": "Roadmap", "degree": 1, "source": "apple_notes"}
            ],
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
        assert client.get("/index/reenrich").json() == {
            "running": False,
            "phase": "idle",
            "done": 0,
            "total": 0,
        }
    assert sync_calls == [True]


def test_index_sync_uses_enabled_connector_sync_callable() -> None:
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
        sync_all_enabled=lambda: [
            ConnectorSyncReport(connector_id="fake", added=1, changed=0, removed=0)
        ],
    )

    with TestClient(create_app(runtime)) as client:
        assert client.post("/index/sync").json() == [
            {
                "connector_id": "fake",
                "added": 1,
                "changed": 0,
                "removed": 0,
                "enrich_failures": 0,
                "error": "",
            }
        ]


def test_connector_rest_endpoints_and_single_flight_sync() -> None:
    registry = FakeConnectorRegistry()
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
        connector_registry=registry,
        connector_llm=object(),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.get("/connectors").json() == [
            {
                "connector_id": "fake",
                "label": "Fake",
                "kind": "local",
                "enabled": True,
                "available": True,
                "detail": "",
                "docs": 3,
                "last_sync": None,
                "syncing": False,
                "config": {"path": "/tmp/fake"},
            }
        ]
        assert client.post("/connectors/fake/enable").status_code == 200
        assert client.post("/connectors/fake/disable?purge=true").status_code == 200
        assert registry.enabled_calls == [("fake", True, False), ("fake", False, True)]

        assert client.put("/connectors/fake/config", json={"path": "/tmp/new"}).status_code == 200
        assert registry.config_calls == [("fake", {"path": "/tmp/new"})]
        assert client.put("/connectors/fake/config", json={"nested": {"path": "/tmp"}}).status_code == 422

        assert client.post("/connectors/fake/sync").status_code == 202
        assert client.post("/connectors/fake/sync").status_code == 409
        assert client.get("/connectors").json()[0]["syncing"] is True


def test_connector_and_note_routes_return_404s() -> None:
    registry = FakeConnectorRegistry()
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
        connector_registry=registry,
        connector_llm=object(),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.post("/connectors/missing/enable").status_code == 404
        assert client.post("/connectors/missing/sync").status_code == 404
        assert client.get("/notes/missing").status_code == 404


def test_note_preview_route_handles_url_encoded_note_ids() -> None:
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    with TestClient(create_app(runtime)) as client:
        response = client.get("/notes/obsidian%3Avault%2Falpha.md")

    assert response.status_code == 200
    assert response.json()["note_id"] == "obsidian:vault/alpha.md"
    assert response.json()["neighbors"] == [
        {
            "note_id": "apple:beta",
            "title": "Beta",
            "source": "apple_notes",
            "weight": 0.7,
        }
    ]


def test_reenrich_endpoint_is_single_flight_and_exposes_status() -> None:
    async def slow_reenrich(progress: Any) -> dict[str, int]:
        await asyncio.sleep(0.2)
        progress(ReenrichStatus(running=True, phase="notes", done=0, total=1))
        progress(ReenrichStatus(running=False, phase="idle", done=1, total=1))
        return {"total": 1}

    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
        reenrich_all=slow_reenrich,
    )

    with TestClient(create_app(runtime)) as client:
        assert client.post("/index/reenrich").status_code == 202
        assert client.post("/index/reenrich").status_code == 409
        status = client.get("/index/reenrich").json()

    assert status == {
        "running": True,
        "phase": "profile",
        "done": 0,
        "total": 0,
    }


def test_session_routes_binary_frames_and_serializes_schema_events() -> None:
    live_sessions: list[EchoLiveSession] = []
    asr_clients: list[FakeASRClient] = []
    opened: list[str] = []

    def make_live_session(emit: Any, logger: Any) -> EchoLiveSession:
        del logger
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
                "seq": 0,
            }

            ws.send_json({"type": "dismiss", "card_id": "card-1"})
            ws.send_json({"type": "open_note", "note_id": "n1"})
            ws.send_json({"type": "stop"})

    assert asr_clients[0].starts == [[2]]
    assert asr_clients[0].sent == [(2, b"\x01\x00\x02\x00")]
    assert asr_clients[0].stopped
    assert live_sessions[0].dismisses == ["card-1"]
    assert opened == ["n1"]
    assert asr_clients[0].index_status() == IndexStatus(
        notes=1,
        last_sync=123.0,
        reenrich=ReenrichStatus(running=False),
    )


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
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    await run_sync_once_safely(runtime)

    assert calls == 1


def test_sessions_endpoints_list_and_parse_jsonl_fixture(tmp_path, monkeypatch) -> None:
    session_file = tmp_path / "20260706-120000-abcdef12.jsonl"
    records = [
        {"ts": 111.0, "kind": "session_start", "channels": [{"id": 0, "label": "mic"}]},
        {"ts": 112.0, "kind": "chunk", "seq": 1, "channel": 0, "text": "hello", "final": True},
    ]
    session_file.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    monkeypatch.setattr(session_log, "SESSIONS_DIR", tmp_path)
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.get("/sessions").json() == [
            {
                "id": "20260706-120000-abcdef12",
                "path": str(session_file),
                "started": 111.0,
                "size_bytes": session_file.stat().st_size,
            }
        ]
        assert client.get("/sessions/20260706-120000-abcdef12").json() == records
        assert client.get("/sessions/missing").status_code == 404
