from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from websockets.asyncio.server import serve

from taurine_core.asr import ASRClient
from taurine_core.schemas import Chunk, IndexStatus, StatusEvent


async def test_asr_client_sends_float32_audio_and_emits_one_final_chunk() -> None:
    received_frames: list[bytes] = []
    requested_paths: list[str] = []
    got_binary = asyncio.Event()
    got_done = asyncio.Event()

    async def handler(ws) -> None:
        requested_paths.append(ws.request.path)
        async for message in ws:
            if isinstance(message, bytes):
                received_frames.append(message)
                got_binary.set()
                await ws.send(json.dumps({"text": "remind", "segment": 7}))
                await ws.send(json.dumps({"text": "remind me", "segment": 7}))
                await ws.send(
                    json.dumps(
                        {"text": "remind me about the deck", "segment": 7, "is_final": 1}
                    )
                )
            elif message == "Done":
                got_done.set()
                await ws.close()

    chunks: list[Chunk] = []
    statuses: list[StatusEvent] = []
    final_seen = asyncio.Event()

    async def on_chunk(chunk: Chunk) -> None:
        chunks.append(chunk)
        if chunk.final:
            final_seen.set()

    def on_status(event: StatusEvent) -> None:
        statuses.append(event)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = ASRClient(
            token="secret-token",
            on_chunk=on_chunk,
            on_status=on_status,
            index_status=lambda: IndexStatus(notes=3, last_sync=42.0),
            url=f"ws://127.0.0.1:{port}/taurine",
            reconnect_min_s=0.01,
            reconnect_max_s=0.02,
        )

        await client.start([0])
        await client.wait_connected(0, timeout=1.0)

        pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        assert await client.send_pcm_s16(channel=0, pcm=pcm)

        await asyncio.wait_for(got_binary.wait(), timeout=1.0)
        await asyncio.wait_for(final_seen.wait(), timeout=1.0)
        await client.stop()
        await asyncio.wait_for(got_done.wait(), timeout=1.0)

    assert requested_paths == ["/taurine?token=secret-token"]
    assert np.frombuffer(received_frames[0], dtype="<f4").tolist() == pytest.approx(
        [-1.0, 0.0, 32767 / 32768.0]
    )
    assert [chunk.final for chunk in chunks] == [False, False, True]
    assert chunks[-1].channel == 0
    assert chunks[-1].text == "remind me about the deck"
    assert statuses[-1].asr == "disconnected"
