from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from taurine_core.indexer.sync import SyncReport
from taurine_core.schemas import ChatAckEvent, ChatDoneEvent, ChatMessage, ChatUsage
from taurine_core.server import Runtime, create_app


@dataclass
class MinimalIndex:
    created: dict[str, Any] = field(default_factory=dict)

    def status(self) -> tuple[int, float | None]:
        return 0, None

    def graph(self) -> Any:
        return {"nodes": [], "edges": []}


class StubChatEngine:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.regenerated: list[str] = []

    async def send(self, chat_id: str, text: str, emit: Any) -> None:
        self.sent.append((chat_id, text))
        await emit(ChatAckEvent(chat_id="chat-stub", message_id=2, user_message_id=1))
        await emit(
            ChatDoneEvent(
                chat_id="chat-stub",
                message_id=2,
                message=ChatMessage(
                    id=2,
                    chat_id="chat-stub",
                    role="assistant",
                    content="answer",
                    usage=ChatUsage(prompt=1),
                ),
            )
        )

    async def cancel(self, chat_id: str) -> None:
        self.cancelled.append(chat_id)

    async def regenerate(self, chat_id: str, emit: Any) -> None:
        self.regenerated.append(chat_id)
        await emit(ChatAckEvent(chat_id=chat_id, message_id=4, user_message_id=3))


class FailingChatEngine:
    async def send(self, chat_id: str, text: str, emit: Any) -> None:
        del chat_id, text, emit
        raise RuntimeError("chat task exploded")

    async def cancel(self, chat_id: str) -> None:
        del chat_id

    async def regenerate(self, chat_id: str, emit: Any) -> None:
        del chat_id, emit


def test_chat_websocket_relays_stubbed_engine_events_and_ignores_bad_controls() -> None:
    engine = StubChatEngine()
    runtime = Runtime(
        index=MinimalIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        make_chat_engine=lambda: engine,
    )

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/chat") as ws:
            ws.send_text("not json")
            ws.send_json({"type": "unknown"})
            ws.send_json({"type": "chat_send", "chat_id": "", "text": "hello"})
            assert ws.receive_json()["type"] == "chat_ack"
            assert ws.receive_json()["type"] == "chat_done"
            ws.send_json({"type": "chat_cancel", "chat_id": "chat-stub"})
            ws.send_json({"type": "chat_regenerate", "chat_id": "chat-stub"})
            assert ws.receive_json()["type"] == "chat_ack"

    assert engine.sent == [("", "hello")]
    assert engine.cancelled == ["chat-stub"]
    assert engine.regenerated == ["chat-stub"]


def test_chat_websocket_logs_unexpected_task_exception(caplog: pytest.LogCaptureFixture) -> None:
    runtime = Runtime(
        index=MinimalIndex(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        make_chat_engine=lambda: FailingChatEngine(),
    )
    caplog.set_level(logging.ERROR, logger="taurine.chat")

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/chat") as ws:
            ws.send_json({"type": "chat_send", "chat_id": "", "text": "hello"})

    assert any(
        "chat websocket task failed" in record.getMessage()
        and record.exc_info is not None
        and isinstance(record.exc_info[1], RuntimeError)
        for record in caplog.records
    )


def test_chat_rest_crud_and_update_bus_publish(store: Any) -> None:
    runtime = Runtime(
        index=store,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        make_chat_engine=lambda: StubChatEngine(),
    )

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/updates") as updates:
            snapshot_types = {updates.receive_json()["type"] for _ in range(4)}
            assert "chats" in snapshot_types
            created = client.post("/chats").json()
            assert created["chat_id"].startswith("chat-")
            assert updates.receive_json()["type"] == "chats"

            store.add_chat_message(
                ChatMessage(
                    chat_id=created["chat_id"],
                    role="user",
                    content="hello",
                    created=created["created"] + 1,
                )
            )
            assert client.get("/chats").json()[0]["message_count"] == 1
            detail = client.get(f"/chats/{created['chat_id']}").json()
            assert detail["chat"]["chat_id"] == created["chat_id"]
            assert detail["messages"][0]["content"] == "hello"

            long_title = "  " + "x" * 150 + "  "
            renamed = client.patch(f"/chats/{created['chat_id']}", json={"title": long_title}).json()
            assert renamed["title"] == "x" * 120
            assert updates.receive_json()["type"] == "chats"

            assert client.get("/chats/missing").status_code == 404
            assert client.patch("/chats/missing", json={"title": "Nope"}).status_code == 404
            assert client.delete(f"/chats/{created['chat_id']}").status_code == 200
            assert updates.receive_json()["type"] == "chats"
            assert client.get(f"/chats/{created['chat_id']}").status_code == 404
            assert store.chat_messages(created["chat_id"]) == []
