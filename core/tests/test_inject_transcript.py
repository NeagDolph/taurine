from __future__ import annotations

import importlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import pytest

from taurine_core import config, server as server_module
from taurine_core.indexer.store import Index
from taurine_core.schemas import Chunk, MeetingSummary, SyncReport
from taurine_core.server import Runtime, SessionContext


def test_test_controls_flag_is_enabled_only_by_exact_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = os.environ.get("TAURINE_TEST_CONTROLS")
    try:
        monkeypatch.delenv("TAURINE_TEST_CONTROLS", raising=False)
        importlib.reload(config)
        assert config.TEST_CONTROLS_ENABLED is False

        monkeypatch.setenv("TAURINE_TEST_CONTROLS", "true")
        importlib.reload(config)
        assert config.TEST_CONTROLS_ENABLED is False

        monkeypatch.setenv("TAURINE_TEST_CONTROLS", "1")
        importlib.reload(config)
        assert config.TEST_CONTROLS_ENABLED is True
    finally:
        if original is None:
            monkeypatch.delenv("TAURINE_TEST_CONTROLS", raising=False)
        else:
            monkeypatch.setenv("TAURINE_TEST_CONTROLS", original)
        importlib.reload(config)


@dataclass
class RecordingLogger:
    records: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def log(self, kind: str, **payload: Any) -> None:
        self.records.append((kind, payload))


async def test_inject_transcript_is_rejected_and_logged_when_gate_is_off(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "TEST_CONTROLS_ENABLED", False, raising=False)
    index = Index(tmp_path / "inject-disabled.db")
    logger = RecordingLogger()
    emitted: list[Any] = []
    runtime = _runtime(index)

    async def emit(event: Any) -> None:
        emitted.append(event)

    started = await server_module._handle_control(
        json.dumps(
            {
                "type": "inject_transcript",
                "chunks": [{"channel": 0, "text": "should not enter"}],
            }
        ),
        ctx_holder=[SessionContext(generation=1, meeting_id=None)],
        transcript_buffer=[None],
        on_final_chunk=lambda chunk: None,
        runtime=runtime,
        emit=emit,
        logger=logger,
        session_started=True,
    )

    assert started is True
    assert emitted[-1].detail == "invalid control message: test controls are disabled"
    assert logger.records == [
        ("control_rejected", {"detail": "test controls are disabled"})
    ]


@dataclass
class SequencingLive:
    on_final_chunk: Callable[[dict[str, Any]], Awaitable[None] | None]
    chunks: list[Chunk] = field(default_factory=list)

    async def on_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        result = self.on_final_chunk(
            {
                "channel": chunk.channel,
                "text": chunk.text,
                "ts": chunk.ts,
                "seq": len(self.chunks),
            }
        )
        if result is not None:
            await result

    async def set_actionable_status(self, actionable_id: str, status: str) -> None:
        del actionable_id, status

    def set_notepad(self, text_md: str) -> None:
        del text_md

    def current_summary(self) -> MeetingSummary:
        return MeetingSummary()

    async def aclose(self) -> None:
        return None


async def test_enabled_inject_transcript_uses_live_on_chunk_and_persists_sequence(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "TEST_CONTROLS_ENABLED", True, raising=False)
    monkeypatch.setattr(server_module.time, "time", lambda: 1234.5)
    index = Index(tmp_path / "inject-enabled.db")
    meeting = index.meeting_create("Injected")
    meeting_id = meeting["meeting_id"]
    buffer = server_module._MeetingTranscriptBuffer(index, meeting_id)

    async def persist(chunk: dict[str, Any]) -> None:
        await buffer.append(chunk)

    live = SequencingLive(persist)
    context = SessionContext(generation=1, meeting_id=meeting_id, live=live)
    logger = RecordingLogger()

    async def emit(_event: Any) -> None:
        return None

    started = await server_module._handle_control(
        json.dumps(
            {
                "type": "inject_transcript",
                "chunks": [
                    {"channel": 0, "text": "We should ship Friday."},
                    {"channel": 1, "text": "Maya owns the rollout."},
                ],
            }
        ),
        ctx_holder=[context],
        transcript_buffer=[buffer],
        on_final_chunk=persist,
        runtime=_runtime(index),
        emit=emit,
        logger=logger,
        session_started=True,
    )
    await buffer.flush()

    assert started is True
    assert [
        (chunk.channel, chunk.text, chunk.final, chunk.ts, chunk.seq)
        for chunk in live.chunks
    ] == [
        (0, "We should ship Friday.", True, 1234.5, 0),
        (1, "Maya owns the rollout.", True, 1234.5, 0),
    ]
    assert index.meeting_get(meeting_id)["transcript"] == [
        {
            "channel": 0,
            "text": "We should ship Friday.",
            "ts": 1234.5,
            "seq": 1,
        },
        {
            "channel": 1,
            "text": "Maya owns the rollout.",
            "ts": 1234.5,
            "seq": 2,
        },
    ]


def _runtime(index: Index) -> Runtime:
    return Runtime(
        index=index,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
    )


def test_e2e_harness_validates_every_suggestion_anchor() -> None:
    module = _load_e2e_module()

    notes = "Anchor once\nAnchor twice\nAnchor twice"
    assert module.suggestions_valid(
        notes,
        [
            {"anchor": "", "occurrence": 1},
            {"anchor": "Anchor once", "occurrence": 1},
            {"anchor": "Anchor twice", "occurrence": 2},
        ],
    )
    assert not module.suggestions_valid(
        notes, [{"anchor": "Anchor once", "occurrence": 2}]
    )


def test_e2e_harness_check_source_populates_taurine_cache(tmp_path: Any) -> None:
    database = tmp_path / "source-check.db"
    index = Index(database)
    meeting = index.meeting_create("Source check")
    meeting_id = meeting["meeting_id"]
    index.meeting_update(meeting_id, notes_md="Persist this meeting")
    index.meeting_set_final(
        meeting_id,
        final_summary_md="A final summary",
        suggestions=[],
        base_rev=1,
    )
    index.meeting_set_status(meeting_id, "done")
    index.close()

    module = _load_e2e_module()
    assert module.check_source_cache(meeting_id, database)

    stored = Index(database)
    try:
        assert stored.note(f"taurine:{meeting_id}") is not None
    finally:
        stored.close()


def _load_e2e_module() -> Any:
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scripts", "e2e_meeting_flow.py"
    )
    assert os.path.isfile(script_path), "deterministic meeting E2E harness is missing"
    spec = importlib.util.spec_from_file_location("e2e_meeting_flow", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
