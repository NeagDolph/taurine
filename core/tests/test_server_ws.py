from __future__ import annotations

import json
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from taurine_core import session_log
from taurine_core.indexer.profile import ProfileProgress
from taurine_core.indexer.sync import SyncReport
from taurine_core.ports import LLMError
from taurine_core.schemas import (
    Actionable,
    ActionableEvent,
    Chunk,
    ConnectorInfo,
    ConnectorSyncReport,
    GroundingItem,
    GraphEdge,
    GraphNode,
    GraphResponse,
    IndexStatus,
    MeetingSummary,
    NeighborRef,
    NotePreview,
    ReenrichStatus,
    SummaryEvent,
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
class FakeProfileGenerator:
    result: str | Exception
    delay_s: float = 0.05

    async def generate(self, on_progress) -> str:
        on_progress(ProfileProgress("surveying", "Surveying the graph", 0, 0))
        await asyncio.sleep(self.delay_s)
        if isinstance(self.result, Exception):
            raise self.result
        on_progress(ProfileProgress("synthesizing", "Writing the profile", 7, 3))
        return self.result


@dataclass
class EchoLiveSession:
    emit: Any
    summary: MeetingSummary = field(default_factory=MeetingSummary)
    statuses: list[tuple[str, str]] = field(default_factory=list)

    async def on_chunk(self, chunk: Chunk) -> None:
        await self.emit(
            TranscriptEvent(
                channel=chunk.channel,
                text=chunk.text,
                final=chunk.final,
                ts=chunk.ts,
            )
        )

    async def set_actionable_status(self, actionable_id: str, status: str) -> None:
        self.statuses.append((actionable_id, status))

    def current_summary(self) -> MeetingSummary:
        return self.summary


@dataclass
class SummaryEchoLiveSession(EchoLiveSession):
    async def on_chunk(self, chunk: Chunk) -> None:
        await super().on_chunk(chunk)
        action = Actionable(
            id="act-1",
            title="Send timeline",
            quote="I'll send the timeline Friday",
            grounding=["N1"],
            confidence=0.9,
        )
        summary = MeetingSummary(
            rev=2,
            summary_md="Discussing timeline [N1]",
            actionables=[action],
            grounding=[
                GroundingItem(
                    note_id="n1",
                    short_id="N1",
                    title="Timeline",
                    gist="Timeline gist",
                    score=0.91,
                )
            ],
            updated_at=123.0,
        )
        self.summary = summary
        await self.emit(SummaryEvent.from_summary(summary))
        await self.emit(ActionableEvent(op="new", item=action))


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
    enabled_calls: list[tuple[str, bool]] = field(default_factory=list)
    purge_calls: list[str] = field(default_factory=list)
    config_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    sync_calls: list[str] = field(default_factory=list)
    on_change: Any = None

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

    def set_enabled(self, index: object, connector_id: str, enabled: bool) -> None:
        del index
        if connector_id != "fake":
            raise KeyError(connector_id)
        self.enabled_calls.append((connector_id, enabled))

    def purge_data(self, index: object, connector_id: str) -> int:
        del index
        if connector_id != "fake":
            raise KeyError(connector_id)
        self.purge_calls.append(connector_id)
        return 3

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
                {
                    "note_id": "n1",
                    "title": "Roadmap",
                    "degree": 1,
                    "source": "apple_notes",
                    "gist": "",
                }
            ],
            "edges": [
                {
                    "src": "n1",
                    "dst": "n2",
                    "type": "topic",
                    "weight": 0.7,
                    "shared": ["roadmap"],
                    "fact": "",
                    "fact_provenance": "",
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
                "sync_phase": "",
                "sync_done": 0,
                "sync_total": 0,
                "config": {"path": "/tmp/fake"},
            }
        ]
        assert client.post("/connectors/fake/enable").status_code == 200
        assert client.post("/connectors/fake/disable").status_code == 200
        assert registry.enabled_calls == [("fake", True), ("fake", False)]
        assert client.post("/connectors/fake/data/delete").status_code == 200
        assert registry.purge_calls == ["fake"]

        assert client.put("/connectors/fake/config", json={"path": "/tmp/new"}).status_code == 200
        assert registry.config_calls == [("fake", {"path": "/tmp/new"})]
        assert (
            client.put("/connectors/fake/config", json={"folders": ["/a", "/b"]}).status_code
            == 200
        )
        assert registry.config_calls[-1] == ("fake", {"folders": ["/a", "/b"]})
        assert client.put("/connectors/fake/config", json={"nested": {"path": "/tmp"}}).status_code == 422
        assert (
            client.put("/connectors/fake/config", json={"folders": [["/a"]]}).status_code == 422
        )

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
            "fact": "",
        }
    ]


def test_reenrich_endpoint_is_single_flight_and_exposes_status() -> None:
    async def slow_reenrich(progress: Any, source: Any = None) -> dict[str, int]:
        del source
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
        "phase": "notes",
        "done": 0,
        "total": 0,
    }


def test_session_routes_binary_frames_and_serializes_schema_events() -> None:
    live_sessions: list[EchoLiveSession] = []
    asr_clients: list[FakeASRClient] = []
    opened: list[str] = []

    def make_live_session(emit: Any, logger: Any) -> EchoLiveSession:
        del logger
        live = SummaryEchoLiveSession(emit)
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

            assert ws.receive_json() == {
                "type": "summary",
                "rev": 2,
                "summary_md": "Discussing timeline [N1]",
                "highlights": [],
                "actionables": [
                    {
                        "id": "act-1",
                        "title": "Send timeline",
                        "quote": "I'll send the timeline Friday",
                        "assignee": "",
                        "due": "",
                        "grounding": ["N1"],
                        "why_new": "",
                        "confidence": 0.9,
                        "status": "open",
                    }
                ],
                "grounding": [
                    {
                        "note_id": "n1",
                        "short_id": "N1",
                        "title": "Timeline",
                        "gist": "Timeline gist",
                        "source": "apple_notes",
                        "age_days": 0.0,
                        "score": 0.91,
                        "why": {"heads": [], "ppr_mass": 0.0, "exact_alias": ""},
                    }
                ],
                "facts": [],
                "updated_at": 123.0,
            }
            assert ws.receive_json() == {
                "type": "actionable",
                "op": "new",
                "item": {
                    "id": "act-1",
                    "title": "Send timeline",
                    "quote": "I'll send the timeline Friday",
                    "assignee": "",
                    "due": "",
                    "grounding": ["N1"],
                    "why_new": "",
                    "confidence": 0.9,
                    "status": "open",
                },
            }

            ws.send_json(
                {
                    "type": "actionable_status",
                    "id": "act-1",
                    "status": "confirmed",
                }
            )
            ws.send_json({"type": "open_note", "note_id": "n1"})
            ws.send_json({"type": "stop"})

    assert asr_clients[0].starts == [[2]]
    assert asr_clients[0].sent == [(2, b"\x01\x00\x02\x00")]
    assert asr_clients[0].stopped
    assert live_sessions[0].statuses == [("act-1", "confirmed")]
    assert opened == ["n1"]
    assert asr_clients[0].index_status() == IndexStatus(
        notes=1,
        last_sync=123.0,
        reenrich=ReenrichStatus(running=False),
    )


def test_summary_route_returns_empty_or_active_live_summary() -> None:
    active_live = EchoLiveSession(
        emit=lambda event: None,
        summary=MeetingSummary(rev=7, summary_md="Active summary"),
    )
    live_sessions: list[EchoLiveSession] = []

    def make_live_session(emit: Any, logger: Any) -> EchoLiveSession:
        del emit, logger
        live_sessions.append(active_live)
        return active_live

    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=make_live_session,
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.get("/summary").json() == {
            "rev": 0,
            "summary_md": "",
            "highlights": [],
            "actionables": [],
            "grounding": [],
            "facts": [],
            "updated_at": 0.0,
        }
        with client.websocket_connect("/session") as ws:
            assert ws.receive_json()["type"] == "status"
            assert client.get("/summary").json()["summary_md"] == "Active summary"


def test_summary_route_404s_without_live_session_capability() -> None:
    runtime = Runtime(
        index=GraphIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.get("/summary").status_code == 404


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


def test_profile_endpoints_roundtrip() -> None:
    class ProfileIndex(GraphIndex):
        def __init__(self) -> None:
            super().__init__()
            self.card: str | None = None

        def user_card(self) -> str | None:
            return self.card

        def set_user_card(self, text: str) -> None:
            self.card = text

    runtime = Runtime(
        index=ProfileIndex(),
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
        response = client.get("/profile").json()
        assert response["card"] is None
        assert response["generation"] == {
            "running": False,
            "phase": "idle",
            "activity": "",
            "nodes_read": 0,
            "tool_calls": 0,
            "started_at": None,
            "elapsed_s": 0.0,
            "error": "",
        }
        saved = client.put("/profile", json={"card": "  Neil builds Taurine.  "}).json()
        assert saved["card"] == "Neil builds Taurine."
        assert saved["generation"]["phase"] == "idle"
        assert client.get("/profile").json()["card"] == "Neil builds Taurine."
        assert client.put("/profile", json={"card": "   "}).status_code == 422


def test_profile_generation_is_single_flight_and_overwrites_only_on_success() -> None:
    class ProfileIndex(GraphIndex):
        card = "Existing manual profile"

        def user_card(self) -> str | None:
            return self.card

        def set_user_card(self, text: str) -> None:
            self.card = text

    index = ProfileIndex()
    runtime = Runtime(
        index=index,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
        profile_generator=FakeProfileGenerator("# Generated profile"),
    )

    with TestClient(create_app(runtime)) as client:
        accepted = client.post("/profile/generate")
        assert accepted.status_code == 202
        assert accepted.json()["generation"]["running"] is True
        assert client.post("/profile/generate").status_code == 409
        assert index.card == "Existing manual profile"

        deadline = time.time() + 1.0
        payload = client.get("/profile").json()
        while payload["generation"]["running"] and time.time() < deadline:
            time.sleep(0.01)
            payload = client.get("/profile").json()

        assert payload["generation"]["phase"] == "idle"
        assert payload["card"] == "# Generated profile"


def test_profile_generation_failure_preserves_existing_card() -> None:
    class ProfileIndex(GraphIndex):
        card = "Keep this profile"

        def user_card(self) -> str | None:
            return self.card

        def set_user_card(self, text: str) -> None:
            self.card = text

    runtime = Runtime(
        index=ProfileIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=lambda emit, logger: EchoLiveSession(emit),
        make_asr_client=lambda on_chunk, on_status, index_status: FakeASRClient(
            on_chunk,
            on_status,
            index_status,
        ),
        profile_generator=FakeProfileGenerator(LLMError("provider unavailable"), delay_s=0),
    )

    with TestClient(create_app(runtime)) as client:
        assert client.post("/profile/generate").status_code == 202
        deadline = time.time() + 1.0
        payload = client.get("/profile").json()
        while payload["generation"]["phase"] != "failed" and time.time() < deadline:
            time.sleep(0.01)
            payload = client.get("/profile").json()

        assert payload["card"] == "Keep this profile"
        assert payload["generation"]["running"] is False
        assert payload["generation"]["phase"] == "failed"
        assert payload["generation"]["error"] == "provider unavailable"


def test_updates_websocket_sends_snapshot_and_change_events() -> None:
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
        with client.websocket_connect("/updates") as websocket:
            snapshot = {}
            for _ in range(5):
                message = websocket.receive_json()
                snapshot[message["type"]] = message
            assert set(snapshot) == {"connectors", "reenrich", "graph", "profile", "chats"}
            assert snapshot["connectors"]["connectors"][0]["connector_id"] == "fake"
            assert snapshot["reenrich"]["status"]["running"] is False
            assert snapshot["graph"]["version"] == 0
            assert snapshot["profile"]["card"] is None
            assert snapshot["profile"]["generation"]["phase"] == "idle"
            assert snapshot["chats"]["chats"] == []

            # a state change publishes fresh events over the socket
            assert client.post("/connectors/fake/disable").status_code == 200
            kinds = set()
            while "graph" not in kinds:
                kinds.add(websocket.receive_json()["type"])
            assert runtime.graph_version == 1
