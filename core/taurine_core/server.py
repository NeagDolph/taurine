from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import ValidationError

from taurine_core import config
from taurine_core import session_log
from taurine_core.asr import ASRClient, ChunkCallback, IndexStatusProvider, StatusCallback
from taurine_core.connectors.registry import ConnectorRegistry
from taurine_core.indexer.notes import open_note
from taurine_core.indexer.reenrich import reenrich_all
from taurine_core.indexer.store import Index
from taurine_core.indexer.sync import SyncReport
from taurine_core.live.ambient import AmbientRetriever
from taurine_core.live.cadence import Cadence
from taurine_core.live.exact import ExactCollector
from taurine_core.live.ingest import Event, EventSink, LiveSession
from taurine_core.live.presenter import Presenter
from taurine_core.live.rerank import CohereReranker
from taurine_core.live.slate import Slate
from taurine_core.live.spans import Gliner2SpanExtractor
from taurine_core.llm import OpenRouterClient
from taurine_core.session_log import NullSessionLogger, SessionLogger
from taurine_core.schemas import (
    ConnectorInfo,
    DismissMsg,
    GraphResponse,
    IndexStatus,
    NotePreview,
    OpenNoteMsg,
    ReenrichStatus,
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


class UpdateBus:
    """Fan-out of state-change notifications to /updates websockets.

    publish() is thread-safe — registry callbacks fire from sync worker
    threads — and only enqueues a kind marker; each websocket builds fresh
    payloads when it drains its queue, so bursts coalesce naturally.
    """

    KINDS = ("connectors", "reenrich", "graph", "profile")

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._subscribers.discard(queue)

    def publish(self, kind: str) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def fanout() -> None:
            for queue in tuple(self._subscribers):
                queue.put_nowait(kind)

        try:
            loop.call_soon_threadsafe(fanout)
        except RuntimeError:  # loop shut down mid-publish
            pass


@dataclass
class Runtime:
    index: Any
    sync_once: Callable[[], SyncReport]
    open_note: Callable[[str], None]
    make_live_session: Callable[[EventSink, Any], LiveSessionLike]
    make_asr_client: Callable[
        [ChunkCallback, StatusCallback, IndexStatusProvider],
        ASRClientLike,
    ]
    make_session_logger: Callable[[], Any] = NullSessionLogger
    enable_periodic_sync: bool = False
    sync_all_enabled: Callable[[], Any] | None = None
    connector_registry: Any = None
    connector_llm: Any = None
    connector_sync_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    reenrich_all: Callable[..., Awaitable[Any]] | None = None
    reenrich_status: ReenrichStatus = field(
        default_factory=lambda: ReenrichStatus(running=False)
    )
    reenrich_task: asyncio.Task[Any] | None = None
    reenrich_last_emit_s: float = 0.0
    status_sinks: set[Callable[[], Awaitable[None]]] = field(default_factory=set)
    updates: UpdateBus = field(default_factory=UpdateBus)
    graph_version: int = 0


class _MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_runtime = runtime or build_runtime()
        app.state.runtime = active_runtime
        active_runtime.updates.bind_loop(asyncio.get_running_loop())
        if active_runtime.connector_registry is not None:
            active_runtime.connector_registry.on_change = lambda: active_runtime.updates.publish(
                "connectors"
            )
        periodic_task: asyncio.Task[None] | None = None
        if active_runtime.enable_periodic_sync:
            periodic_task = asyncio.create_task(_periodic_sync(active_runtime))
        try:
            yield
        finally:
            if periodic_task is not None:
                periodic_task.cancel()
                await asyncio.gather(periodic_task, return_exceptions=True)
            if active_runtime.reenrich_task is not None:
                active_runtime.reenrich_task.cancel()
                await asyncio.gather(active_runtime.reenrich_task, return_exceptions=True)
            for task in tuple(active_runtime.connector_sync_tasks.values()):
                task.cancel()
            if active_runtime.connector_sync_tasks:
                await asyncio.gather(
                    *active_runtime.connector_sync_tasks.values(),
                    return_exceptions=True,
                )

    app = FastAPI(lifespan=lifespan)
    # localhost-bound service consumed by the Electron renderer (vite origin
    # in dev, file:// in prod) — REST needs CORS; the ws is exempt anyway
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["*"],
    )

    @app.get("/graph")
    async def graph() -> GraphResponse:
        active_runtime: Runtime = app.state.runtime
        return active_runtime.index.graph()

    @app.post("/index/sync")
    async def index_sync() -> Any:
        active_runtime: Runtime = app.state.runtime
        sync = active_runtime.sync_all_enabled or active_runtime.sync_once
        report = await asyncio.to_thread(sync)
        _bump_graph(active_runtime)
        return _report_payload(report)

    @app.get("/connectors")
    async def connectors() -> list[ConnectorInfo]:
        active_runtime: Runtime = app.state.runtime
        registry = _connector_registry(active_runtime)
        return registry.list_info(active_runtime.index)

    @app.post("/connectors/{connector_id}/enable")
    async def connector_enable(connector_id: str) -> ConnectorInfo:
        active_runtime: Runtime = app.state.runtime
        registry = _connector_registry(active_runtime)
        try:
            registry.set_enabled(active_runtime.index, connector_id, True)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        _bump_graph(active_runtime)
        return _connector_info(active_runtime, connector_id)

    @app.post("/connectors/{connector_id}/disable")
    async def connector_disable(connector_id: str) -> ConnectorInfo:
        active_runtime: Runtime = app.state.runtime
        registry = _connector_registry(active_runtime)
        try:
            registry.set_enabled(active_runtime.index, connector_id, False)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        _bump_graph(active_runtime)
        return _connector_info(active_runtime, connector_id)

    @app.post("/connectors/{connector_id}/data/delete")
    async def connector_delete_data(connector_id: str) -> ConnectorInfo:
        active_runtime: Runtime = app.state.runtime
        registry = _connector_registry(active_runtime)
        if connector_id in registry.syncing:
            raise HTTPException(status_code=409, detail="connector sync running — wait for it to finish")
        try:
            await asyncio.to_thread(registry.purge_data, active_runtime.index, connector_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        _bump_graph(active_runtime)
        return _connector_info(active_runtime, connector_id)

    @app.post("/connectors/{connector_id}/sync", status_code=202)
    async def connector_sync(connector_id: str) -> dict[str, str]:
        active_runtime: Runtime = app.state.runtime
        registry = _connector_registry(active_runtime)
        _connector_info(active_runtime, connector_id)
        task = active_runtime.connector_sync_tasks.get(connector_id)
        if (
            connector_id in registry.syncing
            or task is not None
            and not task.done()
        ):
            raise HTTPException(status_code=409, detail="connector sync already running")
        registry.syncing.add(connector_id)
        active_runtime.connector_sync_tasks[connector_id] = asyncio.create_task(
            _run_connector_sync(active_runtime, connector_id)
        )
        return {"connector_id": connector_id, "status": "syncing"}

    @app.put("/connectors/{connector_id}/config")
    async def connector_config(
        connector_id: str,
        payload: Any = Body(...),
    ) -> ConnectorInfo:
        active_runtime: Runtime = app.state.runtime
        registry = _connector_registry(active_runtime)
        config_payload = _flat_config(payload)
        try:
            registry.set_config(active_runtime.index, connector_id, config_payload)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        return _connector_info(active_runtime, connector_id)

    @app.get("/notes/{note_id:path}")
    async def note_preview(note_id: str) -> NotePreview:
        active_runtime: Runtime = app.state.runtime
        preview = active_runtime.index.preview(note_id)
        if preview is None:
            raise HTTPException(status_code=404, detail="note not found")
        return preview

    @app.post("/index/reenrich", status_code=202)
    async def index_reenrich(source: str | None = None) -> ReenrichStatus:
        active_runtime: Runtime = app.state.runtime
        if (
            active_runtime.reenrich_status.running
            or active_runtime.reenrich_task is not None
            and not active_runtime.reenrich_task.done()
        ):
            raise HTTPException(status_code=409, detail="re-enrich already running")
        if active_runtime.reenrich_all is None:
            raise HTTPException(status_code=503, detail="re-enrich unavailable")

        _set_reenrich_status(
            active_runtime,
            ReenrichStatus(
                running=True,
                phase="notes" if source else "profile",
                done=0,
                total=0,
            ),
            force_emit=True,
        )
        active_runtime.reenrich_task = asyncio.create_task(
            _run_reenrich(active_runtime, source)
        )
        return active_runtime.reenrich_status

    @app.get("/index/reenrich")
    async def index_reenrich_status() -> ReenrichStatus:
        active_runtime: Runtime = app.state.runtime
        return active_runtime.reenrich_status

    @app.get("/profile")
    async def get_profile() -> dict[str, Any]:
        active_runtime: Runtime = app.state.runtime
        card = getattr(active_runtime.index, "user_card", lambda: None)()
        return {"card": card}

    @app.put("/profile")
    async def put_profile(payload: Any = Body(...)) -> dict[str, Any]:
        active_runtime: Runtime = app.state.runtime
        card = payload.get("card") if isinstance(payload, dict) else None
        if not isinstance(card, str) or not card.strip():
            raise HTTPException(status_code=422, detail="card must be a non-empty string")
        active_runtime.index.set_user_card(card.strip())  # type: ignore[attr-defined]
        active_runtime.updates.publish("profile")
        return {"card": card.strip()}

    @app.get("/sessions")
    async def sessions() -> list[dict[str, Any]]:
        return [_session_summary(path) for path in _session_files()]

    @app.get("/sessions/{session_id}")
    async def session_records(session_id: str) -> list[dict[str, Any]]:
        path = _session_path(session_id)
        if path is None:
            raise HTTPException(status_code=404, detail="session not found")
        return _read_session_records(path)

    @app.websocket("/updates")
    async def updates(websocket: WebSocket) -> None:
        await websocket.accept()
        active_runtime: Runtime = app.state.runtime
        queue = active_runtime.updates.subscribe()

        async def pump() -> None:
            while True:
                kinds = {await queue.get()}
                while not queue.empty():
                    kinds.add(queue.get_nowait())
                await _send_updates(websocket, active_runtime, kinds)

        pump_task: asyncio.Task[None] | None = None
        try:
            await _send_updates(websocket, active_runtime, set(UpdateBus.KINDS))
            pump_task = asyncio.create_task(pump())
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            active_runtime.updates.unsubscribe(queue)
            if pump_task is not None:
                pump_task.cancel()
                await asyncio.gather(pump_task, return_exceptions=True)

    @app.websocket("/session")
    async def session(websocket: WebSocket) -> None:
        await websocket.accept()
        active_runtime: Runtime = app.state.runtime
        logger = active_runtime.make_session_logger()
        session_started = False
        events: asyncio.Queue[Event | StatusEvent] = asyncio.Queue()
        asr_state: dict[str, str] = {"value": "disconnected"}
        status_detail: dict[str, str] = {"value": ""}

        async def emit(event: Event | StatusEvent) -> None:
            if isinstance(event, StatusEvent):
                asr_state["value"] = event.asr
                status_detail["value"] = event.detail
                logger.log("asr_status", state=event.asr, detail=event.detail)
            await events.put(event)

        async def emit_index_status() -> None:
            await emit(
                StatusEvent(
                    asr=asr_state["value"],  # type: ignore[arg-type]
                    index=_index_status(active_runtime),
                    detail=status_detail["value"],
                )
            )

        active_runtime.status_sinks.add(emit_index_status)
        live = active_runtime.make_live_session(emit, logger)
        asr = active_runtime.make_asr_client(
            live.on_chunk,
            emit,
            lambda: _index_status(active_runtime),
        )
        sender = asyncio.create_task(_send_events(websocket, events))
        await emit(
            StatusEvent(asr="disconnected", index=_index_status(active_runtime))
        )
        try:
            while True:
                message = await websocket.receive()
                kind = message.get("type")
                if kind == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    session_started = await _handle_control(
                        message["text"],
                        asr=asr,
                        live=live,
                        runtime=active_runtime,
                        emit=emit,
                        logger=logger,
                        session_started=session_started,
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
            active_runtime.status_sinks.discard(emit_index_status)
            if not session_started:
                logger.log("session_start", channels=[])
            await asr.stop()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            logger.close()

    return app


def build_runtime() -> Runtime:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    secrets = required_secrets()
    index = Index(Path("taurine.db"))
    llm = OpenRouterClient(secrets["OPEN_ROUTER_API_KEY"])
    reranker = CohereReranker(secrets["COHERE_API_KEY"])
    span_extractor = Gliner2SpanExtractor()
    connector_registry = ConnectorRegistry()
    connector_registry.reconcile_ledger(index)

    def make_live_session(emit: EventSink, logger: SessionLogger) -> LiveSession:
        clock = _MonotonicClock()
        return LiveSession(
            index=index,
            slate=Slate(
                clock,
                on_event=lambda op, payload: logger.log("slate", op=op, **payload),
            ),
            cadence=Cadence(clock),
            exact=ExactCollector(index),
            ambient=AmbientRetriever(
                index,
                reranker,
                clock,
                span_extractor=span_extractor,
            ),
            presenter=Presenter(llm, index, on_log=lambda payload: logger.log("presenter", **payload)),
            emit=emit,
            clock=clock,
            logger=logger,
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
        sync_once=lambda: connector_registry.sync_all_enabled(index, llm),
        sync_all_enabled=lambda: connector_registry.sync_all_enabled(index, llm),
        open_note=open_note,
        make_live_session=make_live_session,
        make_asr_client=make_asr_client,
        make_session_logger=lambda: SessionLogger.create(session_log.SESSIONS_DIR),
        enable_periodic_sync=True,
        connector_registry=connector_registry,
        connector_llm=llm,
        reenrich_all=lambda progress, source=None: reenrich_all(index, llm, progress, source=source),
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
    logger: Any,
    session_started: bool,
) -> bool:
    try:
        message = _parse_control(raw)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        await emit(_status_error(runtime, f"invalid control message: {exc}"))
        return session_started

    logger.log("control", message=message.model_dump())

    if isinstance(message, StartMsg):
        if not session_started:
            logger.log("session_start", channels=message.channels)
            session_started = True
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
    return session_started


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


async def _run_connector_sync(runtime: Runtime, connector_id: str) -> None:
    registry = _connector_registry(runtime)
    try:
        await asyncio.to_thread(
            registry.sync_connector,
            connector_id,
            runtime.index,
            runtime.connector_llm,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.getLogger("taurine.session").exception(
            "connector sync crashed: %s", connector_id
        )
    finally:
        registry.syncing.discard(connector_id)
        runtime.connector_sync_tasks.pop(connector_id, None)
        _bump_graph(runtime)
        await _emit_index_status(runtime)


async def _run_reenrich(runtime: Runtime, source: str | None = None) -> None:
    assert runtime.reenrich_all is not None
    try:
        await runtime.reenrich_all(
            lambda status: _set_reenrich_status(runtime, status),
            source,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        current = runtime.reenrich_status
        _set_reenrich_status(
            runtime,
            ReenrichStatus(
                running=False,
                phase="idle",
                done=current.done,
                total=current.total,
            ),
            force_emit=True,
        )
    finally:
        if runtime.reenrich_status.running:
            current = runtime.reenrich_status
            _set_reenrich_status(
                runtime,
                ReenrichStatus(
                    running=False,
                    phase="idle",
                    done=current.done,
                    total=current.total,
                ),
                force_emit=True,
            )
        else:
            await _emit_index_status(runtime)
        # re-enrichment rewrites gists/edges and (on full runs) the user card
        _bump_graph(runtime)
        runtime.updates.publish("profile")
        runtime.updates.publish("connectors")


def _set_reenrich_status(
    runtime: Runtime,
    status: ReenrichStatus,
    *,
    force_emit: bool = False,
) -> None:
    runtime.reenrich_status = status
    now = time.monotonic()
    should_emit = (
        force_emit
        or not status.running
        or now - runtime.reenrich_last_emit_s >= 2.0
    )
    if not should_emit:
        return
    runtime.reenrich_last_emit_s = now
    runtime.updates.publish("reenrich")
    asyncio.create_task(_emit_index_status(runtime))


def _bump_graph(runtime: Runtime) -> None:
    runtime.graph_version += 1
    runtime.updates.publish("graph")


async def _send_updates(
    websocket: WebSocket,
    runtime: Runtime,
    kinds: set[str],
) -> None:
    """One JSON message per changed kind; payloads are built fresh at send
    time so coalesced bursts reflect the latest state."""
    if "connectors" in kinds and runtime.connector_registry is not None:
        infos = await asyncio.to_thread(
            runtime.connector_registry.list_info, runtime.index
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connectors",
                    "connectors": [info.model_dump() for info in infos],
                }
            )
        )
    if "reenrich" in kinds:
        await websocket.send_text(
            json.dumps(
                {"type": "reenrich", "status": runtime.reenrich_status.model_dump()}
            )
        )
    if "graph" in kinds:
        await websocket.send_text(
            json.dumps({"type": "graph", "version": runtime.graph_version})
        )
    if "profile" in kinds:
        card = getattr(runtime.index, "user_card", lambda: None)()
        await websocket.send_text(json.dumps({"type": "profile", "card": card}))


async def _emit_index_status(runtime: Runtime) -> None:
    if not runtime.status_sinks:
        return
    await asyncio.gather(
        *(sink() for sink in tuple(runtime.status_sinks)),
        return_exceptions=True,
    )


def _index_status(runtime: Runtime) -> IndexStatus:
    notes, last_sync = runtime.index.status()
    return IndexStatus(
        notes=notes,
        last_sync=last_sync,
        reenrich=runtime.reenrich_status,
    )


def _status_error(runtime: Runtime, detail: str) -> StatusEvent:
    return StatusEvent(asr="error", index=_index_status(runtime), detail=detail)


def _connector_registry(runtime: Runtime) -> Any:
    if runtime.connector_registry is None:
        raise HTTPException(status_code=503, detail="connectors unavailable")
    return runtime.connector_registry


def _connector_info(runtime: Runtime, connector_id: str) -> ConnectorInfo:
    registry = _connector_registry(runtime)
    for info in registry.list_info(runtime.index):
        if info.connector_id == connector_id:
            return info
    raise HTTPException(status_code=404, detail="connector not found")


def _flat_config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="config must be a JSON object")
    config_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise HTTPException(status_code=422, detail="config keys must be strings")
        if isinstance(value, dict | list | tuple | set):
            raise HTTPException(status_code=422, detail="config values must be flat")
        config_payload[key] = value
    return config_payload


def _report_payload(report: Any) -> Any:
    if isinstance(report, list):
        return [_report_payload(item) for item in report]
    if isinstance(report, BaseModel):
        return report.model_dump()
    if is_dataclass(report):
        return asdict(report)
    if isinstance(report, dict):
        return report
    return dict(report)


def _session_files() -> list[Path]:
    base_dir = session_log.SESSIONS_DIR
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    return sorted(base_dir.glob("*.jsonl"), key=lambda path: path.name, reverse=True)


def _session_path(session_id: str) -> Path | None:
    for path in _session_files():
        if path.stem == session_id:
            return path
    return None


def _session_summary(path: Path) -> dict[str, Any]:
    records = _read_session_records(path)
    started = records[0].get("ts") if records else path.stat().st_mtime
    return {
        "id": path.stem,
        "path": str(path),
        "started": started,
        "size_bytes": path.stat().st_size,
    }


def _read_session_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    except (OSError, json.JSONDecodeError):
        return []
    return records


app = create_app()
