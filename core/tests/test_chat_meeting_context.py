from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from taurine_core import config
from taurine_core.chat import engine as engine_module
from taurine_core.chat.engine import (
    ChatEngine,
    _MeetingEvidence,
    _finalize_evidence,
    _system_prompt,
)
from taurine_core.profile_context import user_profile_block
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.validate import EvidenceValidator
from taurine_core.schemas import (
    ChatAckEvent,
    ChatDoneEvent,
    ChatMessage,
    ChatMetaEvent,
    ChatSendMsg,
    ChatUsage,
    MeetingSummary,
    NoteRecord,
    SyncReport,
)
from taurine_core.server import Runtime, create_app


def _meeting_evidence(
    index: Any,
    registry: EvidenceRegistry,
    meeting_id: str,
    *,
    live_summary: MeetingSummary | None = None,
) -> Any:
    builder = getattr(engine_module, "_build_meeting_evidence", None)
    assert callable(builder), "meeting evidence builder is not implemented"
    return builder(index, registry, meeting_id, live_summary=live_summary)


def test_registry_and_validator_accept_available_meeting_evidence() -> None:
    registry = EvidenceRegistry()
    meeting_ref = registry.register(
        "meeting", "meeting-1:notes:2", {"text": "Launch is Friday."}
    )
    registry.mark_read(meeting_ref)

    result = EvidenceValidator(registry).validate(
        "The launch is Friday [M1].",
        mode="chat",
        supplied_ids=(meeting_ref,),
        available_ids=(meeting_ref,),
    )

    assert meeting_ref == "M1"
    assert result.accepted is True
    assert result.cited_ids == ("M1",)


async def test_finalize_and_repair_paths_add_meeting_ids_to_turn_availability(
    store: Any,
) -> None:
    registry = EvidenceRegistry()
    meeting_ref = registry.register(
        "meeting", "meeting-1:notes:1", {"text": "Owner is Maya."}
    )
    registry.mark_read(meeting_ref)
    retriever = SimpleNamespace(registry=registry)

    content, cited, stripped = _finalize_evidence(
        "The owner is Maya [M1].",
        retriever,
        None,
        None,
        meeting_ids=(meeting_ref,),
    )

    assert content == "The owner is Maya [M1]."
    assert cited == (meeting_ref,)
    assert stripped == ()

    available_calls: list[tuple[str, ...]] = []

    class RecordingValidator:
        def validate(
            self,
            output: str,
            *,
            mode: str,
            supplied_ids: tuple[str, ...] = (),
            available_ids: tuple[str, ...] = (),
        ) -> Any:
            available_calls.append(tuple(available_ids))
            return EvidenceValidator(registry).validate(
                output,
                mode=mode,
                supplied_ids=supplied_ids,
                available_ids=available_ids,
            )

    engine = ChatEngine(
        store,
        None,
        None,
        validator_factory=lambda _registry: RecordingValidator(),
    )
    parts = ["The owner is Maya [M1]."]
    await engine._repair_once(
        parts,
        retriever,
        None,
        meeting_evidence=_MeetingEvidence(
            block="[M1] Owner is Maya.", ids=(meeting_ref,)
        ),
    )

    assert parts == ["The owner is Maya [M1]."]
    assert available_calls == [(meeting_ref,)]


def test_meeting_evidence_is_versioned_capped_and_drops_oldest_windows(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = store.meeting_create("Launch review")
    meeting_id = meeting["meeting_id"]
    store.meeting_update(meeting_id, notes_md="Owner: Maya")
    store.meeting_append_transcript(
        meeting_id,
        [
            {
                "channel": channel,
                "text": f"turn-{seq} " + ("detail " * 10),
                "ts": 1_720_000_000.0 + seq,
                "seq": seq,
            }
            for seq, channel in enumerate(([0] * 4) + ([1] * 4) + ([0] * 4), start=1)
        ],
    )
    monkeypatch.setattr(config, "MEETING_CONTEXT_MAX_CHARS", 360, raising=False)
    registry = EvidenceRegistry()

    first = _meeting_evidence(store, registry, meeting_id)

    assert len(first.block) <= config.MEETING_CONTEXT_MAX_CHARS
    assert first.ids
    assert "Owner: Maya" in first.block
    assert "turn-1 detail" not in first.block
    assert "turn-12" in first.block
    assert all(ref.startswith("M") for ref in first.ids)

    store.meeting_update(meeting_id, notes_md="Owner: Priya")
    second = _meeting_evidence(store, registry, meeting_id)

    assert "Owner: Priya" in second.block
    assert "Owner: Maya" not in second.block
    assert second.ids[0] != first.ids[0]


def test_meeting_evidence_uses_live_summary_then_final_summary(store: Any) -> None:
    meeting = store.meeting_create("Status")
    meeting_id = meeting["meeting_id"]
    registry = EvidenceRegistry()

    live = _meeting_evidence(
        store,
        registry,
        meeting_id,
        live_summary=MeetingSummary(rev=7, summary_md="Live decision: ship Friday."),
    )
    assert "Live decision: ship Friday." in live.block

    store.meeting_set_final(
        meeting_id,
        final_summary_md="Final decision: ship Monday.",
        suggestions=[],
        base_rev=0,
    )
    final = _meeting_evidence(store, registry, meeting_id)
    assert "Final decision: ship Monday." in final.block
    assert "Live decision: ship Friday." not in final.block


def test_non_meeting_build_messages_remains_byte_identical(store: Any) -> None:
    store.create_chat("chat-plain", now=1.0)
    history_id = store.add_chat_message(
        ChatMessage(
            chat_id="chat-plain", role="user", content="Earlier", created=2.0
        )
    )
    assistant_id = store.add_chat_message(
        ChatMessage(chat_id="chat-plain", role="assistant", content="", created=3.0)
    )
    engine = ChatEngine(store, None, None)

    actual = engine._build_messages(
        "chat-plain",
        "Question",
        None,
        current_user_id=history_id,
        assistant_id=assistant_id,
    )
    expected = [
        {"role": "system", "content": _system_prompt(user_profile_block(store))},
        {"role": "user", "content": "Question"},
    ]

    assert json.dumps(actual, separators=(",", ":")) == json.dumps(
        expected, separators=(",", ":")
    )


def test_chat_send_schema_exposes_optional_meeting_id() -> None:
    plain = ChatSendMsg(type="chat_send", text="hello")
    scoped = ChatSendMsg(type="chat_send", text="hello", meeting_id="meeting-1")

    assert plain.meeting_id is None
    assert scoped.meeting_id == "meeting-1"


@dataclass
class _MeetingChatEngine:
    index: Any
    sent: list[tuple[str, str, str | None]] = field(default_factory=list)
    regenerated: list[tuple[str, str | None]] = field(default_factory=list)

    async def send(
        self,
        chat_id: str,
        text: str,
        emit: Any,
        *,
        meeting_id: str | None = None,
        seed_note_ids: tuple[str, ...] = (),
    ) -> None:
        del seed_note_ids
        if not chat_id:
            chat_id = "chat-meeting"
            created = self.index.create_chat(chat_id, now=10.0)
            await emit(ChatMetaEvent(chat=created))
        self.sent.append((chat_id, text, meeting_id))
        await emit(ChatAckEvent(chat_id=chat_id, message_id=2, user_message_id=1))
        await emit(
            ChatDoneEvent(
                chat_id=chat_id,
                message_id=2,
                message=ChatMessage(
                    id=2,
                    chat_id=chat_id,
                    role="assistant",
                    content="answer",
                    usage=ChatUsage(),
                ),
            )
        )

    async def cancel(self, chat_id: str) -> None:
        del chat_id

    async def regenerate(
        self, chat_id: str, emit: Any, *, meeting_id: str | None = None
    ) -> None:
        self.regenerated.append((chat_id, meeting_id))
        await emit(ChatAckEvent(chat_id=chat_id, message_id=4, user_message_id=1))


def _runtime(store: Any, engine: Any) -> Runtime:
    return Runtime(
        index=store,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        make_chat_engine=lambda: engine,
    )


def _drain_done(websocket: Any) -> None:
    while websocket.receive_json()["type"] != "chat_done":
        pass


def test_first_meeting_send_links_and_titles_chat_then_regenerate_restores_context(
    store: Any,
) -> None:
    meeting = store.meeting_create("Launch review")
    meeting_id = meeting["meeting_id"]
    engine = _MeetingChatEngine(store)

    with TestClient(create_app(_runtime(store, engine))) as client:
        with client.websocket_connect("/chat") as websocket:
            websocket.send_json(
                {
                    "type": "chat_send",
                    "chat_id": "",
                    "text": "What did we decide?",
                    "meeting_id": meeting_id,
                }
            )
            _drain_done(websocket)
            websocket.send_json(
                {"type": "chat_regenerate", "chat_id": "chat-meeting"}
            )
            assert websocket.receive_json()["type"] == "chat_ack"

    assert engine.sent == [("chat-meeting", "What did we decide?", meeting_id)]
    assert store.meeting_get(meeting_id)["chat_id"] == "chat-meeting"
    assert store.chat("chat-meeting").title == "Meeting: Launch review"
    assert engine.regenerated == [("chat-meeting", meeting_id)]


def test_unknown_meeting_id_is_ignored_gracefully(store: Any) -> None:
    engine = _MeetingChatEngine(store)

    with TestClient(create_app(_runtime(store, engine))) as client:
        with client.websocket_connect("/chat") as websocket:
            websocket.send_json(
                {
                    "type": "chat_send",
                    "chat_id": "",
                    "text": "hello",
                    "meeting_id": "missing",
                }
            )
            _drain_done(websocket)

    assert engine.sent == [("chat-meeting", "hello", None)]
    assert store.chat("chat-meeting").title == ""


def test_meeting_engine_seeds_indexed_taurine_note(store: Any) -> None:
    meeting = store.meeting_create("Seeded")
    note_id = f"taurine:{meeting['meeting_id']}"
    store.replace_source_records(
        "taurine",
        [
            NoteRecord(
                note_id=note_id,
                title="Seeded",
                text="Meeting record",
                folder="Meetings",
                created=1.0,
                modified=1.0,
                content_hash="hash",
            )
        ],
    )
    seed_helper = getattr(engine_module, "_meeting_seed_note_ids", None)
    assert callable(seed_helper), "meeting retrieval seed helper is not implemented"

    assert seed_helper(store, meeting["meeting_id"]) == (note_id,)
