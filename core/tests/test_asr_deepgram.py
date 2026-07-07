from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from websockets.asyncio.server import serve

from taurine_core import config
from taurine_core.asr import ASRClient
from taurine_core.schemas import Chunk, StatusEvent


async def test_deepgram_channel_uses_deepgram_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_paths: list[str] = []
    auth_headers: list[str | None] = []
    received_audio: list[bytes] = []
    control_messages: list[dict[str, str]] = []
    got_audio = asyncio.Event()
    got_close = asyncio.Event()

    async def handler(ws) -> None:
        requested_paths.append(ws.request.path)
        auth_headers.append(ws.request.headers.get("Authorization"))
        async for message in ws:
            if isinstance(message, bytes):
                received_audio.append(message)
                got_audio.set()
                await ws.send(
                    json.dumps(
                        {
                            "channel": {"alternatives": [{"transcript": ""}]},
                            "is_final": False,
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "channel": {"alternatives": [{"transcript": "checking"}]},
                            "is_final": False,
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "channel": {"alternatives": [{"transcript": "   "}]},
                            "is_final": True,
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "channel": {
                                "alternatives": [{"transcript": "checking the roadmap"}]
                            },
                            "is_final": True,
                        }
                    )
                )
            else:
                payload = json.loads(message)
                control_messages.append(payload)
                if payload == {"type": "CloseStream"}:
                    got_close.set()
                    await ws.close()

    chunks: list[Chunk] = []
    final_seen = asyncio.Event()

    async def on_chunk(chunk: Chunk) -> None:
        chunks.append(chunk)
        if chunk.final:
            final_seen.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(config, "ASR_PROVIDER", "deepgram")
        monkeypatch.setattr(config, "DEEPGRAM_URL", f"ws://127.0.0.1:{port}/v1/listen")
        monkeypatch.setattr(config, "DEEPGRAM_MODEL", "nova-3")
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")

        client = ASRClient(
            token="taurine-token",
            on_chunk=on_chunk,
            reconnect_min_s=0.01,
            reconnect_max_s=0.02,
        )

        await client.start([1])
        await client.wait_connected(1, timeout=1.0)

        pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        assert await client.send_pcm_s16(channel=1, pcm=pcm)

        await asyncio.wait_for(got_audio.wait(), timeout=1.0)
        await asyncio.wait_for(final_seen.wait(), timeout=1.0)
        await client.stop()
        await asyncio.wait_for(got_close.wait(), timeout=1.0)

    parsed = urlparse(requested_paths[0])
    assert parsed.path == "/v1/listen"
    assert parse_qs(parsed.query) == {
        "model": ["nova-3"],
        "encoding": ["linear16"],
        "sample_rate": ["16000"],
        "channels": ["1"],
        "interim_results": ["true"],
        "smart_format": ["true"],
    }
    assert auth_headers == ["Token dg-secret"]
    assert received_audio == [pcm]
    assert [chunk.text for chunk in chunks] == ["checking", "checking the roadmap"]
    assert [chunk.final for chunk in chunks] == [False, True]
    assert chunks[-1].channel == 1
    assert {"type": "CloseStream"} in control_messages


async def test_deepgram_missing_api_key_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ASR_PROVIDER", "deepgram")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    statuses: list[StatusEvent] = []
    got_error = asyncio.Event()

    def on_status(event: StatusEvent) -> None:
        statuses.append(event)
        if event.asr == "error":
            got_error.set()

    client = ASRClient(
        token="taurine-token",
        on_chunk=lambda chunk: None,
        on_status=on_status,
        reconnect_min_s=0.01,
        reconnect_max_s=0.02,
    )

    await client.start([0])
    await asyncio.wait_for(got_error.wait(), timeout=1.0)

    error_statuses = [event for event in statuses if event.asr == "error"]
    assert await client.send_pcm_s16(channel=0, pcm=b"\x00\x00") is False
    assert "DEEPGRAM_API_KEY" in error_statuses[-1].detail

    await client.stop()


async def test_taurine_provider_still_uses_vadgate_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ASR_PROVIDER", "taurine")
    requested_paths: list[str] = []
    received_frames: list[bytes] = []
    got_binary = asyncio.Event()
    got_done = asyncio.Event()

    async def handler(ws) -> None:
        requested_paths.append(ws.request.path)
        async for message in ws:
            if isinstance(message, bytes):
                received_frames.append(message)
                got_binary.set()
            elif message == "Done":
                got_done.set()
                await ws.close()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = ASRClient(
            token="secret-token",
            on_chunk=lambda chunk: None,
            url=f"ws://127.0.0.1:{port}/taurine",
            reconnect_min_s=0.01,
            reconnect_max_s=0.02,
        )

        await client.start([0])
        await client.wait_connected(0, timeout=1.0)

        pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        assert await client.send_pcm_s16(channel=0, pcm=pcm)

        await asyncio.wait_for(got_binary.wait(), timeout=1.0)
        await client.stop()
        await asyncio.wait_for(got_done.wait(), timeout=1.0)

    assert requested_paths == ["/taurine?token=secret-token"]
    assert np.frombuffer(received_frames[0], dtype="<f4").tolist() == pytest.approx(
        [-1.0, 0.0, 32767 / 32768.0]
    )
