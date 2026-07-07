from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from inspect import isawaitable
from typing import Any, Literal
from urllib.parse import urlencode

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from taurine_core import config
from taurine_core.schemas import Chunk, IndexStatus, StatusEvent


ChunkCallback = Callable[[Chunk], Awaitable[None] | None]
StatusCallback = Callable[[StatusEvent], Awaitable[None] | None]
IndexStatusProvider = Callable[[], IndexStatus]
AsrState = Literal["disconnected", "connecting", "connected", "error"]


class ASRClient:
    def __init__(
        self,
        token: str,
        on_chunk: ChunkCallback,
        *,
        on_status: StatusCallback | None = None,
        index_status: IndexStatusProvider | None = None,
        url: str = config.ASR_URL,
        reconnect_min_s: float = config.ASR_RECONNECT_MIN_S,
        reconnect_max_s: float = config.ASR_RECONNECT_MAX_S,
    ) -> None:
        self._token = token
        self._on_chunk = on_chunk
        self._on_status = on_status
        self._index_status = index_status or (lambda: IndexStatus(notes=0, last_sync=None))
        self._provider = _select_provider(url)
        self._url = _url_with_token(url, token)
        self._deepgram_url = _deepgram_url(config.DEEPGRAM_URL, config.DEEPGRAM_MODEL)
        self._deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
        self._reconnect_min_s = reconnect_min_s
        self._reconnect_max_s = reconnect_max_s
        self._channels: dict[int, _ASRChannel | _DeepgramChannel] = {}

    async def start(self, channels: Iterable[int]) -> None:
        for channel in channels:
            if channel in self._channels:
                continue
            if self._provider == "taurine":
                client = _ASRChannel(
                    channel=channel,
                    url=self._url,
                    on_chunk=self._on_chunk,
                    emit_status=self._emit_status,
                    reconnect_min_s=self._reconnect_min_s,
                    reconnect_max_s=self._reconnect_max_s,
                )
            elif self._provider == "deepgram":
                client = _DeepgramChannel(
                    channel=channel,
                    url=self._deepgram_url,
                    api_key=self._deepgram_api_key,
                    on_chunk=self._on_chunk,
                    emit_status=self._emit_status,
                    reconnect_min_s=self._reconnect_min_s,
                    reconnect_max_s=self._reconnect_max_s,
                )
            else:
                await self._emit_status("error", f"unsupported ASR_PROVIDER {self._provider!r}")
                continue
            self._channels[channel] = client
            client.start()

    async def wait_connected(self, channel: int, timeout: float) -> None:
        await self._channels[channel].wait_connected(timeout)

    async def send_pcm_s16(self, channel: int, pcm: bytes) -> bool:
        client = self._channels.get(channel)
        if client is None:
            return False
        return await client.send_pcm_s16(pcm)

    async def stop(self) -> None:
        channels = list(self._channels.values())
        self._channels.clear()
        await asyncio.gather(*(channel.stop() for channel in channels))

    async def _emit_status(self, state: AsrState, detail: str = "") -> None:
        if self._on_status is None:
            return
        event = StatusEvent(asr=state, index=self._index_status(), detail=detail)
        await _maybe_await(self._on_status(event))


class _ASRChannel:
    def __init__(
        self,
        *,
        channel: int,
        url: str,
        on_chunk: ChunkCallback,
        emit_status: Callable[[AsrState, str], Awaitable[None]],
        reconnect_min_s: float,
        reconnect_max_s: float,
    ) -> None:
        self._channel = channel
        self._url = url
        self._on_chunk = on_chunk
        self._emit_status = emit_status
        self._reconnect_min_s = reconnect_min_s
        self._reconnect_max_s = reconnect_max_s
        self._connected = asyncio.Event()
        self._stopping = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._ws: Any | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def wait_connected(self, timeout: float) -> None:
        await asyncio.wait_for(self._connected.wait(), timeout=timeout)

    async def send_pcm_s16(self, pcm: bytes) -> bool:
        if not pcm or len(pcm) % 2:
            return False
        ws = self._ws
        if ws is None:
            return False
        floats = (np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0).astype(
            "<f4",
            copy=False,
        )
        try:
            async with self._send_lock:
                await ws.send(floats.tobytes())
        except ConnectionClosed:
            return False
        return True

    async def stop(self) -> None:
        self._stopping.set()
        ws = self._ws
        if ws is not None:
            try:
                async with self._send_lock:
                    await ws.send("Done")
            except ConnectionClosed:
                pass
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=8.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        backoff = self._reconnect_min_s
        while not self._stopping.is_set():
            self._connected.clear()
            await self._emit_status("connecting", f"channel {self._channel}")
            try:
                async with connect(self._url, max_size=None, open_timeout=15) as ws:
                    self._ws = ws
                    self._connected.set()
                    backoff = self._reconnect_min_s
                    await self._emit_status("connected", f"channel {self._channel}")
                    async for message in ws:
                        await self._handle_message(message)
            except ConnectionClosed:
                pass
            except OSError as exc:
                await self._emit_status("error", str(exc))
            except Exception as exc:  # pragma: no cover - defensive websocket loop guard
                await self._emit_status("error", str(exc))
            finally:
                self._ws = None
                self._connected.clear()

            if self._stopping.is_set():
                break
            await self._emit_status("disconnected", f"retrying channel {self._channel}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._reconnect_max_s)

        await self._emit_status("disconnected", f"channel {self._channel}")

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            return
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        chunk = Chunk(
            channel=self._channel,
            text=text,
            ts=time.time(),
            final=bool(payload.get("is_final")),
        )
        await _maybe_await(self._on_chunk(chunk))


class _DeepgramChannel:
    def __init__(
        self,
        *,
        channel: int,
        url: str,
        api_key: str | None,
        on_chunk: ChunkCallback,
        emit_status: Callable[[AsrState, str], Awaitable[None]],
        reconnect_min_s: float,
        reconnect_max_s: float,
    ) -> None:
        self._channel = channel
        self._url = url
        self._api_key = api_key
        self._on_chunk = on_chunk
        self._emit_status = emit_status
        self._reconnect_min_s = reconnect_min_s
        self._reconnect_max_s = reconnect_max_s
        self._connected = asyncio.Event()
        self._stopping = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._ws: Any | None = None
        self._last_audio_sent = 0.0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def wait_connected(self, timeout: float) -> None:
        await asyncio.wait_for(self._connected.wait(), timeout=timeout)

    async def send_pcm_s16(self, pcm: bytes) -> bool:
        if not pcm or len(pcm) % 2:
            return False
        ws = self._ws
        if ws is None:
            return False
        try:
            async with self._send_lock:
                await ws.send(pcm)
                self._last_audio_sent = time.monotonic()
        except ConnectionClosed:
            return False
        return True

    async def stop(self) -> None:
        self._stopping.set()
        ws = self._ws
        if ws is not None:
            try:
                async with self._send_lock:
                    await ws.send(json.dumps({"type": "CloseStream"}))
            except ConnectionClosed:
                pass
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=8.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        if not self._api_key:
            await self._emit_status(
                "error",
                "DEEPGRAM_API_KEY is required when ASR_PROVIDER is 'deepgram'",
            )
            await self._emit_status("disconnected", f"channel {self._channel}")
            return

        backoff = self._reconnect_min_s
        headers = {"Authorization": f"Token {self._api_key}"}
        while not self._stopping.is_set():
            keepalive_task: asyncio.Task[None] | None = None
            self._connected.clear()
            await self._emit_status("connecting", f"channel {self._channel}")
            try:
                async with connect(
                    self._url,
                    max_size=None,
                    open_timeout=15,
                    additional_headers=headers,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    self._last_audio_sent = time.monotonic()
                    backoff = self._reconnect_min_s
                    keepalive_task = asyncio.create_task(self._keepalive(ws))
                    await self._emit_status("connected", f"channel {self._channel}")
                    async for message in ws:
                        await self._handle_message(message)
            except ConnectionClosed:
                pass
            except OSError as exc:
                await self._emit_status("error", str(exc))
            except Exception as exc:  # pragma: no cover - defensive websocket loop guard
                await self._emit_status("error", str(exc))
            finally:
                if keepalive_task is not None:
                    keepalive_task.cancel()
                    await asyncio.gather(keepalive_task, return_exceptions=True)
                self._ws = None
                self._connected.clear()

            if self._stopping.is_set():
                break
            await self._emit_status("disconnected", f"retrying channel {self._channel}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._reconnect_max_s)

        await self._emit_status("disconnected", f"channel {self._channel}")

    async def _keepalive(self, ws: Any) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(5.0)
            if self._stopping.is_set() or self._ws is not ws:
                return
            if time.monotonic() - self._last_audio_sent < 5.0:
                continue
            try:
                async with self._send_lock:
                    await ws.send(json.dumps({"type": "KeepAlive"}))
            except ConnectionClosed:
                return

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            return
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        channel = payload.get("channel")
        if not isinstance(channel, dict):
            return
        alternatives = channel.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            return
        alternative = alternatives[0]
        if not isinstance(alternative, dict):
            return

        text = str(alternative.get("transcript") or "").strip()
        if not text:
            return
        chunk = Chunk(
            channel=self._channel,
            text=text,
            ts=time.time(),
            final=bool(payload.get("is_final")),
        )
        await _maybe_await(self._on_chunk(chunk))


async def _maybe_await(value: Awaitable[None] | None) -> None:
    if isawaitable(value):
        await value


def _select_provider(url: str) -> str:
    provider = str(config.ASR_PROVIDER).strip().lower()
    if provider == "deepgram" and url != config.ASR_URL:
        return "taurine"
    return provider


def _url_with_token(url: str, token: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'token': token})}"


def _deepgram_url(url: str, model: str) -> str:
    separator = "&" if "?" in url else "?"
    query = urlencode(
        {
            "model": model,
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "interim_results": "true",
            "smart_format": "true",
        }
    )
    return f"{url}{separator}{query}"
