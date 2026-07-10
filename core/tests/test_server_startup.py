"""Deferred startup: port serves immediately, /startup reports build progress."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import taurine_core.server as server
from taurine_core.schemas import GraphResponse


class StartupIndex:
    def graph(self) -> GraphResponse:
        return GraphResponse(nodes=[], edges=[])

    def status(self) -> tuple[int, float | None]:
        return 0, None

    def chats(self) -> list[Any]:
        return []


def _minimal_runtime() -> server.Runtime:
    return server.Runtime(
        index=StartupIndex(),
        sync_once=lambda: None,
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
    )


def _wait_until_ready(client: TestClient, timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get("/startup").json()
        if status["ready"]:
            return status
        time.sleep(0.02)
    pytest.fail("server never became ready")


def test_deferred_startup_serves_status_then_activates(monkeypatch) -> None:
    release = threading.Event()

    def fake_build_runtime(progress=None) -> server.Runtime:
        if progress is not None:
            progress("building_index", "embeddings 3/9")
        if not release.wait(timeout=5):
            raise AssertionError("test never released the build")
        return _minimal_runtime()

    monkeypatch.setattr(server, "build_runtime", fake_build_runtime)

    with TestClient(server.create_app()) as client:
        status = client.get("/startup").json()
        assert status["ready"] is False
        # the build thread may not have reported yet; poll for the phase
        deadline = time.monotonic() + 5
        while status["phase"] != "building_index" and time.monotonic() < deadline:
            time.sleep(0.02)
            status = client.get("/startup").json()
        assert status["phase"] == "building_index"
        assert status["detail"] == "embeddings 3/9"
        assert status["elapsed_s"] >= 0.0

        response = client.get("/graph")
        assert response.status_code == 503
        assert "server is starting" in response.json()["detail"]

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/updates") as websocket:
                websocket.receive_text()
        assert excinfo.value.code == 1013

        release.set()
        ready = _wait_until_ready(client)
        assert ready["phase"] == "ready"
        assert client.get("/graph").status_code == 200


def test_deferred_startup_failure_reports_error(monkeypatch) -> None:
    def fake_build_runtime(progress=None) -> server.Runtime:
        raise RuntimeError("Missing required secret: OPEN_ROUTER_API_KEY")

    monkeypatch.setattr(server, "build_runtime", fake_build_runtime)

    with TestClient(server.create_app()) as client:
        deadline = time.monotonic() + 5
        status = client.get("/startup").json()
        while status["phase"] != "failed" and time.monotonic() < deadline:
            time.sleep(0.02)
            status = client.get("/startup").json()
        assert status["phase"] == "failed"
        assert "OPEN_ROUTER_API_KEY" in status["error"]

        response = client.get("/graph")
        assert response.status_code == 503
        assert "startup failed" in response.json()["detail"]


def test_injected_runtime_is_ready_immediately() -> None:
    with TestClient(server.create_app(_minimal_runtime())) as client:
        status = client.get("/startup").json()
        assert status["ready"] is True
        assert status["phase"] == "ready"
        assert client.get("/graph").status_code == 200
