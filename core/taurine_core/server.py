from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from taurine_core import config
from taurine_core.asr import ASRClient, ChunkCallback, IndexStatusProvider, StatusCallback
from taurine_core.indexer.notes import extract_notes, open_note
from taurine_core.indexer.store import Index
from taurine_core.indexer.sync import SyncReport, sync_once
from taurine_core.live.ambient import AmbientRetriever
from taurine_core.live.cadence import Cadence
from taurine_core.live.exact import ExactCollector
from taurine_core.live.ingest import Event, EventSink, LiveSession
from taurine_core.live.presenter import Presenter
from taurine_core.live.rerank import CohereReranker
from taurine_core.live.slate import Slate
from taurine_core.llm import OpenRouterClient
from taurine_core.schemas import (
    DismissMsg,
    GraphResponse,
    IndexStatus,
    OpenNoteMsg,
    StartMsg,
    StatusEvent,
    StopMsg,
)


REQUIRED_SECRET_KEYS = ("TAURINE_ASR_TOKEN", "OPEN_ROUTER_API_KEY", "COHERE_API_KEY")


class ASRClientLike(Protocol):
    async def start(self, channels: Iterable[int]) -> None:
        ...

    async def send_pcm_s16(self, channel: int, pcm: bytes) -> bool:
        ...

    async def stop(self) -> None:
        ...


class LiveSessionLike(Protocol):
    async def on_chunk(self, chunk: Any) -> None:
        ...

    async def dismiss(self, card_id: str) -> None:
        ...


@dataclass
class Runtime:
    index: Any
    sync_once: Callable[[], SyncReport]
    open_note: Callable[[str], None]
    make_live_session: Callable[[EventSink], LiveSessionLike]
    make_asr_client: Callable[
        [ChunkCallback, StatusCallback, IndexStatusProvider],
        ASRClientLike,
    ]
    enable_periodic_sync: bool = False


class _MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_runtime = runtime or build_runtime()
        app.state.runtime = active_runtime
        periodic_task: asyncio.Task[None] | None = None
        if active_runtime.enable_periodic_sync:
            periodic_task = asyncio.create_task(_periodic_sync(active_runtime))
        try:
            yield
        finally:
            if periodic_task is not None:
                periodic_task.cancel()
                await asyncio.gather(periodic_task, return_exceptions=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/graph")
    async def graph() -> GraphResponse:
        active_runtime: Runtime = app.state.runtime
        return active_runtime.index.graph()

    @app.post("/index/sync")
    async def index_sync() -> dict[str, Any]:
        active_runtime: Runtime = app.state.runtime
        report = await asyncio.to_thread(active_runtime.sync_once)
        return _report_dict(report)

    @app.websocket("/session")
    async def session(websocket: WebSocket) -> None:
        await websocket.accept()
        active_runtime: Runtime = app.state.runtime
        events: asyncio.Queue[Event | StatusEvent] = asyncio.Queue()

        async def emit(event: Event | StatusEvent) -> None:
            await events.put(event)

        live = active_runtime.make_live_session(emit)
        asr = active_runtime.make_asr_client(
            live.on_chunk,
            emit,
            lambda: _index_status(active_runtime),
        )
        sender = asyncio.create_task(_send_events(websocket, events))
        try:
            while True:
                message = await websocket.receive()
                kind = message.get("type")
                if kind == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    await _handle_control(
                        message["text"],
                        asr=asr,
                        live=live,
                        runtime=active_runtime,
                        emit=emit,
                    )
                elif message.get("bytes") is not None:
                    try:
                        channel, pcm = split_audio_frame(message["bytes"])
                    except ValueError as exc:
                        await emit(_status_error(active_runtime, str(exc)))
                        continue
                    await asr.send_pcm_s16(channel, pcm)
        except WebSocketDisconnect:
            pass
        finally:
            await asr.stop()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    return app


def build_runtime() -> Runtime:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    secrets = required_secrets()
    index = Index(Path("taurine.db"))
    llm = OpenRouterClient(secrets["OPEN_ROUTER_API_KEY"])
    reranker = CohereReranker(secrets["COHERE_API_KEY"])

    def make_live_session(emit: EventSink) -> LiveSession:
        clock = _MonotonicClock()
        return LiveSession(
            index=index,
            slate=Slate(clock),
            cadence=Cadence(clock),
            exact=ExactCollector(index),
            ambient=AmbientRetriever(index, reranker, clock),
            presenter=Presenter(llm),
            emit=emit,
            clock=clock,
        )

    def make_asr_client(
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
        index_status: IndexStatusProvider,
    ) -> ASRClient:
        return ASRClient(
            token=secrets["TAURINE_ASR_TOKEN"],
            on_chunk=on_chunk,
            on_status=on_status,
            index_status=index_status,
        )

    return Runtime(
        index=index,
        sync_once=lambda: sync_once(index, extract_notes, llm),
        open_note=open_note,
        make_live_session=make_live_session,
        make_asr_client=make_asr_client,
        enable_periodic_sync=True,
    )


def required_secrets(env: Mapping[str, str | None] | None = None) -> dict[str, str]:
    values = os.environ if env is None else env
    secrets: dict[str, str] = {}
    for key in REQUIRED_SECRET_KEYS:
        value = values.get(key)
        if not value:
            raise RuntimeError(f"Missing required secret: {key}")
        secrets[key] = value
    return secrets


def split_audio_frame(frame: bytes) -> tuple[int, bytes]:
    if not frame:
        raise ValueError("audio frame missing channel byte")
    return frame[0], frame[1:]


async def _handle_control(
    raw: str,
    *,
    asr: ASRClientLike,
    live: LiveSessionLike,
    runtime: Runtime,
    emit: EventSink,
) -> None:
    try:
        message = _parse_control(raw)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        await emit(_status_error(runtime, f"invalid control message: {exc}"))
        return

    if isinstance(message, StartMsg):
        channel_ids = [int(channel["id"]) for channel in message.channels if "id" in channel]
        await asr.start(channel_ids)
    elif isinstance(message, StopMsg):
        await asr.stop()
    elif isinstance(message, DismissMsg):
        await live.dismiss(message.card_id)
    elif isinstance(message, OpenNoteMsg):
        try:
            await asyncio.to_thread(runtime.open_note, message.note_id)
        except Exception as exc:  # pragma: no cover - platform integration guard
            await emit(_status_error(runtime, str(exc)))


def _parse_control(raw: str) -> StartMsg | StopMsg | DismissMsg | OpenNoteMsg:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("control message must be a JSON object")
    message_type = payload.get("type")
    if message_type == "start":
        return StartMsg.model_validate(payload)
    if message_type == "stop":
        return StopMsg.model_validate(payload)
    if message_type == "dismiss":
        return DismissMsg.model_validate(payload)
    if message_type == "open_note":
        return OpenNoteMsg.model_validate(payload)
    raise ValueError(f"unsupported control type: {message_type}")


async def _send_events(
    websocket: WebSocket,
    events: asyncio.Queue[Event | StatusEvent],
) -> None:
    while True:
        event = await events.get()
        await websocket.send_text(event.model_dump_json())


async def _periodic_sync(runtime: Runtime) -> None:
    while True:
        await asyncio.sleep(config.SYNC_INTERVAL_S)
        await run_sync_once_safely(runtime)


async def run_sync_once_safely(runtime: Runtime) -> None:
    try:
        await asyncio.to_thread(runtime.sync_once)
    except Exception:
        pass


def _index_status(runtime: Runtime) -> IndexStatus:
    notes, last_sync = runtime.index.status()
    return IndexStatus(notes=notes, last_sync=last_sync)


def _status_error(runtime: Runtime, detail: str) -> StatusEvent:
    return StatusEvent(asr="error", index=_index_status(runtime), detail=detail)


def _report_dict(report: Any) -> dict[str, Any]:
    if is_dataclass(report):
        return asdict(report)
    if isinstance(report, dict):
        return report
    return dict(report)


app = create_app()
