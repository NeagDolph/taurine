from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

import pytest

from taurine_core.devfeed import run_devfeed
from taurine_core.indexer.sync import SyncReport
from taurine_core.schemas import Chunk, TranscriptEvent
from taurine_core.server import Runtime


@dataclass
class CapturingSession:
    emit: Any
    chunks: list[Chunk] = field(default_factory=list)

    async def on_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        await self.emit(
            TranscriptEvent(
                channel=chunk.channel,
                text=chunk.text,
                final=chunk.final,
                ts=chunk.ts,
            )
        )


@pytest.mark.asyncio
async def test_devfeed_turns_stdin_lines_into_final_chunks_and_json_events() -> None:
    sessions: list[CapturingSession] = []

    def make_session(emit: Any) -> CapturingSession:
        session = CapturingSession(emit)
        sessions.append(session)
        return session

    runtime = Runtime(
        index=object(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=make_session,
        make_asr_client=lambda on_chunk, on_status, index_status: None,  # type: ignore[arg-type]
    )
    out = StringIO()

    await run_devfeed(["hello world\n", "\n"], runtime=runtime, output=out, now=lambda: 99.0)

    assert sessions[0].chunks == [
        Chunk(channel=0, text="hello world", ts=99.0, final=True)
    ]
    assert json.loads(out.getvalue()) == {
        "type": "transcript",
        "channel": 0,
        "text": "hello world",
        "final": True,
        "ts": 99.0,
    }
