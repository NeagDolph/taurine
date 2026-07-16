from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from taurine_core.schemas import SyncReport
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
        self.seeded: list[tuple[str, tuple[str, ...]]] = []
        self.cancelled: list[str] = []
        self.regenerated: list[str] = []

    async def send(
        self,
        chat_id: str,
        text: str,
        emit: Any,
        *,
        seed_note_ids: tuple[str, ...] = (),
    ) -> None:
        self.sent.append((chat_id, text))
        if seed_note_ids:
            self.seeded.append((chat_id, tuple(seed_note_ids)))
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


def test_chat_send_seed_note_ids_reach_the_engine() -> None:
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
            ws.send_json(
                {
                    "type": "chat_send",
                    "chat_id": "",
                    "text": "tell me about this note",
                    "seed_note_ids": ["n1", "n2"],
                }
            )
            assert ws.receive_json()["type"] == "chat_ack"
            assert ws.receive_json()["type"] == "chat_done"

    assert engine.sent == [("", "tell me about this note")]
    assert engine.seeded == [("", ("n1", "n2"))]


def test_post_chats_accepts_optional_seed_note_ids(store: Any) -> None:
    class SeedingChatEngine(StubChatEngine):
        def __init__(self) -> None:
            super().__init__()
            self.seed_calls: list[tuple[str, tuple[str, ...]]] = []

        def seed_chat(self, chat_id: str, note_ids: Any) -> None:
            self.seed_calls.append((chat_id, tuple(note_ids)))

    engine = SeedingChatEngine()
    runtime = Runtime(
        index=store,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        make_chat_engine=lambda: engine,
    )

    with TestClient(create_app(runtime)) as client:
        created = client.post(
            "/chats", json={"seed_note_ids": ["n1", "n1", "  ", "n2"]}
        ).json()
        assert created["chat_id"].startswith("chat-")
        assert engine.seed_calls == [(created["chat_id"], ("n1", "n2"))]

        # no body and non-list payloads leave seeding untouched
        assert client.post("/chats").status_code == 200
        assert client.post("/chats", json={"seed_note_ids": "n1"}).status_code == 200
        assert len(engine.seed_calls) == 1


def test_ws_seeded_chat_biases_first_stage_a_teleport_one_turn_only(store: Any) -> None:
    """seed_note_ids verification: a seeded chat's first Stage A teleport
    includes the seed passages; the bias is gone on the second turn."""

    from types import SimpleNamespace

    from taurine_core.chat.engine import ChatEngine
    from taurine_core.chat.retrieval import ChatRetriever
    from taurine_core.llm import StreamEvent
    from taurine_core.retrieval.registry import EvidenceRegistry
    from taurine_core.retrieval.types import (
        EvidenceNote,
        EvidencePack,
        EvidencePassage,
        RetrievalDecision,
        RetrievalQuery,
        RetrievalTrace,
    )

    engines: list[Any] = []

    class FakeEngine:
        def __init__(self, registry: EvidenceRegistry) -> None:
            self.registry = registry
            self.snapshot = SimpleNamespace(passage_ids=("p1",), base_graph=SimpleNamespace(edges=()))
            self.store = SimpleNamespace(
                passage_contexts=lambda passage_ids: tuple(
                    SimpleNamespace(
                        passage_id="p1", note_id="n1", title="Roadmap", source="obsidian",
                        text="The roadmap launches on July 18.", char_start=0, char_end=32,
                        created=1.0, modified=2.0,
                    )
                    for passage_id in passage_ids
                    if passage_id == "p1"
                ),
                note_contexts=lambda note_ids: (),
            )
            self.resets: list[dict[str, float] | None] = []
            engines.append(self)

        def retrieve(self, query: RetrievalQuery, *, session_reset=None, entity_spans=(), passage_limit=None, **_kwargs) -> EvidencePack:
            del entity_spans, passage_limit
            self.resets.append(dict(session_reset) if session_reset else None)
            note_ref = self.registry.register("note", "n1", {"note_id": "n1"})
            evidence_ref = self.registry.register(
                "passage", "p1",
                {"passage_id": "p1", "note_id": "n1", "text": "The roadmap launches on July 18.", "char_start": 0, "char_end": 32},
            )
            self.registry.mark_read(note_ref)
            self.registry.mark_read(evidence_ref)
            return EvidencePack(
                decision=RetrievalDecision(True, .9, "accepted", ("dense",)),
                query=query,
                passages=(EvidencePassage(evidence_ref, "p1", "n1", "Roadmap", "obsidian", "The roadmap launches on July 18.", 0, 32, 1.0, 2.0, .8, .9),),
                facts=(), paths=(),
                notes=(EvidenceNote(note_ref, "n1", "Roadmap", "obsidian", 2.0, .9, (evidence_ref,)),),
                trace=RetrievalTrace(
                    original_query=query.text, rewritten_query=query.text,
                    selected_ids=(note_ref, evidence_ref),
                    read_ids=(note_ref, evidence_ref), active_build_id="build-ws",
                ),
            )

    class SeedLLM:
        async def stream_chat(self, model, messages, *, tools=None, reasoning=None, timeout_s):
            del model, messages, tools, reasoning, timeout_s
            yield StreamEvent(kind="content", text="ok")
            yield StreamEvent(kind="done")

        async def complete_json(self, model, system, user, timeout_s=None):
            del model, system, user, timeout_s
            return {"title": "Seeded"}

    class NullLogger:
        def log(self, kind: str, **payload: Any) -> None:
            del kind, payload

    class AcceptAll:
        def validate(self, output: str, *, mode: str, supplied_ids=(), available_ids=()) -> Any:
            del mode, supplied_ids, available_ids
            from taurine_core.retrieval.validate import ValidationResult

            return ValidationResult(True, output, (), (), False, (), "")

    chat_engine = ChatEngine(
        store,
        SeedLLM(),
        NullLogger(),
        retriever_factory=lambda chat_id: ChatRetriever(
            store, chat_id,
            engine_factory=FakeEngine,  # type: ignore[arg-type]
            tool_engine_factory=FakeEngine,  # type: ignore[arg-type]
        ),
        validator_factory=lambda registry: AcceptAll(),
    )
    runtime = Runtime(
        index=store,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        make_chat_engine=lambda: chat_engine,
    )

    def drain_until_done(ws: Any) -> None:
        while True:
            if ws.receive_json()["type"] == "chat_done":
                return

    with TestClient(create_app(runtime)) as client:
        created = client.post("/chats", json={"seed_note_ids": ["n1", "unknown"]}).json()
        chat_id = created["chat_id"]
        with client.websocket_connect("/chat") as ws:
            ws.send_json({"type": "chat_send", "chat_id": chat_id, "text": "first question"})
            drain_until_done(ws)
            ws.send_json({"type": "chat_send", "chat_id": chat_id, "text": "unrelated second question"})
            drain_until_done(ws)

    stage_a_resets = [reset for engine in engines for reset in engine.resets]
    assert stage_a_resets[0] == {"p1": 1.0}  # seeded teleport on turn one
    assert all(reset is None for reset in stage_a_resets[1:])  # one turn only


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
            snapshot_types = {updates.receive_json()["type"] for _ in range(5)}
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
