from __future__ import annotations

import asyncio
import faulthandler
import json
import logging
import os
import signal
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import ValidationError

from taurine_core import config
from taurine_core import session_log
from taurine_core.asr import ASRClient, ChunkCallback, IndexStatusProvider, StatusCallback
from taurine_core.chat.engine import ChatEngine
from taurine_core.chat.retrieval import ChatRetriever
from taurine_core.connectors.registry import ConnectorRegistry
from taurine_core.indexer.notes import open_note
from taurine_core.indexer.build import BuildCancelled, BuildCoordinator, BuildDependencies
from taurine_core.indexer.build_cache import BuildCache
from taurine_core.indexer.embed import embed_texts
from taurine_core.indexer.communities import projection_communities
from taurine_core.indexer.extract import extract_notes
from taurine_core.indexer.facts import PREDICATES
from taurine_core.indexer.knowledge_store import KnowledgeSchemaError
from taurine_core.models_local import passage_tokenizer
from taurine_core.indexer.profile import ProfileGenerator, ProfileProgress
from taurine_core.indexer.store import Index
from taurine_core.indexer.sync import SyncReport
from taurine_core.live.ambient import AmbientRetriever
from taurine_core.live.grounding import GroundingEngine
from taurine_core.live.ingest import Event, EventSink, LiveSession
from taurine_core.live.spans import Gliner2SpanExtractor
from taurine_core.live.summarizer import Summarizer
from taurine_core.live.trigger import SummaryTrigger
from taurine_core.llm import OpenRouterClient
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.runtime import RetrievalRuntime
from taurine_core.session_log import NullSessionLogger, SessionLogger
from taurine_core.schemas import (
    ActionableStatusMsg,
    ChatCancelMsg,
    ChatMetaEvent,
    ChatRegenerateMsg,
    ChatSendMsg,
    ConnectorInfo,
    FactPayload,
    FactsResponse,
    GraphEdge,
    GraphNode,
    GraphResponse,
    IndexStatus,
    LocalGraphResponse,
    MeetingSummary,
    NeighborRef,
    NotePreview,
    OpenNoteMsg,
    ProfileGenerationStatus,
    StartMsg,
    StartupStatus,
    SyncStatus,
    StatusEvent,
    StopMsg,
)


REQUIRED_SECRET_KEYS = ("TAURINE_ASR_TOKEN", "OPEN_ROUTER_API_KEY")


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

    async def set_actionable_status(self, actionable_id: str, status: str) -> None:
        ...

    def current_summary(self) -> MeetingSummary:
        ...


class UpdateBus:
    """Fan-out of state-change notifications to /updates websockets.

    publish() is thread-safe — registry callbacks fire from sync worker
    threads — and only enqueues a kind marker; each websocket builds fresh
    payloads when it drains its queue, so bursts coalesce naturally.
    """

    KINDS = ("connectors", "sync", "graph", "profile", "chats")

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

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
    make_live_session: Callable[[EventSink, Any], LiveSessionLike] | None
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
    sync_status: SyncStatus = field(default_factory=SyncStatus)
    sync_task: asyncio.Task[Any] | None = None
    sync_cancel: threading.Event | None = None
    sync_last_emit_s: float = 0.0
    # True when the graph no longer matches the enabled sources (a source was
    # enabled, disabled, or deleted); cleared by a successful sync.
    index_stale: bool = False
    build_cache: Any = None
    status_sinks: set[Callable[[], Awaitable[None]]] = field(default_factory=set)
    updates: UpdateBus = field(default_factory=UpdateBus)
    graph_version: int = 0
    active_live_session: LiveSessionLike | None = None
    make_chat_engine: Callable[[], Any] | None = None
    chat_engine: Any = None
    make_profile_generator: Callable[[], Any] | None = None
    profile_generator: Any = None
    profile_generation: ProfileGenerationStatus = field(
        default_factory=ProfileGenerationStatus
    )
    profile_generation_task: asyncio.Task[Any] | None = None
    knowledge_runtime: Any = None
    regenerate_knowledge: Callable[..., Any] | None = None


class _MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The port must bind immediately: building the runtime can involve a
        # full source re-extraction and knowledge build (minutes to hours), so
        # it runs on a daemon thread while /startup reports progress and every
        # other endpoint answers 503.
        app.state.runtime = None
        app.state.startup = StartupStatus(started_at=time.time())
        try:  # `kill -USR1 <pid>` dumps all thread stacks for hang diagnosis
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except (AttributeError, ValueError, OSError):  # unsupported platform/thread
            pass
        # Surface taurine.* INFO logs (sync/build stage progress) on the
        # console; the root logger only prints warnings by default.
        taurine_logger = logging.getLogger("taurine")
        if not taurine_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
            )
            taurine_logger.addHandler(handler)
            taurine_logger.setLevel(logging.INFO)
        loop = asyncio.get_running_loop()
        state: dict[str, Any] = {"runtime": None, "periodic_task": None}

        def activate(active_runtime: Runtime, *, deferred: bool = False) -> None:
            state["runtime"] = active_runtime
            active_runtime.updates.bind_loop(loop)
            if active_runtime.chat_engine is None and active_runtime.make_chat_engine is not None:
                active_runtime.chat_engine = active_runtime.make_chat_engine()
            if (
                active_runtime.profile_generator is None
                and active_runtime.make_profile_generator is not None
            ):
                active_runtime.profile_generator = active_runtime.make_profile_generator()
            if active_runtime.connector_registry is not None:
                active_runtime.connector_registry.on_change = lambda: active_runtime.updates.publish(
                    "connectors"
                )
            if active_runtime.enable_periodic_sync:
                state["periodic_task"] = asyncio.create_task(_periodic_sync(active_runtime))
            app.state.runtime = active_runtime
            app.state.startup = StartupStatus(
                ready=True,
                phase="ready",
                started_at=app.state.startup.started_at,
            )
            if deferred:
                # Clients that connected mid-startup hold stale (empty) state.
                _bump_graph(active_runtime)
                active_runtime.updates.publish("connectors")
                active_runtime.updates.publish("chats")

        if runtime is not None:
            activate(runtime)
        else:
            started_at = app.state.startup.started_at

            def report(phase: str, detail: str = "") -> None:
                app.state.startup = StartupStatus(
                    phase=phase, detail=detail, started_at=started_at
                )

            def build_in_thread() -> None:
                try:
                    built = build_runtime(progress=report)
                except Exception as exc:  # surfaced via /startup, not a crash
                    logging.getLogger("taurine.server").exception("runtime startup failed")
                    app.state.startup = StartupStatus(
                        phase="failed",
                        error=str(exc)[:300] or type(exc).__name__,
                        started_at=started_at,
                    )
                    return
                try:
                    loop.call_soon_threadsafe(
                        lambda: activate(built, deferred=True)
                    )
                except RuntimeError:  # server shut down before startup finished
                    pass

            threading.Thread(
                target=build_in_thread, name="taurine-startup", daemon=True
            ).start()
        try:
            yield
        finally:
            active_runtime = state["runtime"]
            periodic_task = state["periodic_task"]
            if periodic_task is not None:
                periodic_task.cancel()
                await asyncio.gather(periodic_task, return_exceptions=True)
            # When active_runtime is None the startup thread is still
            # building; it is a daemon thread, so the process exits without
            # waiting for it. A partial build only leaves a temporary
            # *.build-*.db file that the next build reuses or replaces.
            if active_runtime is not None:
                if active_runtime.sync_task is not None:
                    active_runtime.sync_task.cancel()
                    await asyncio.gather(active_runtime.sync_task, return_exceptions=True)
                if active_runtime.profile_generation_task is not None:
                    active_runtime.profile_generation_task.cancel()
                    await asyncio.gather(
                        active_runtime.profile_generation_task,
                        return_exceptions=True,
                    )
                for task in tuple(active_runtime.connector_sync_tasks.values()):
                    task.cancel()
                if active_runtime.connector_sync_tasks:
                    await asyncio.gather(
                        *active_runtime.connector_sync_tasks.values(),
                        return_exceptions=True,
                    )
                chat_logger = getattr(
                    getattr(active_runtime, "chat_engine", None), "logger", None
                )
                close = getattr(chat_logger, "close", None)
                if callable(close):
                    close()
                close_knowledge = getattr(active_runtime.knowledge_runtime, "close", None)
                if callable(close_knowledge):
                    close_knowledge()
                close_cache = getattr(active_runtime.build_cache, "close", None)
                if callable(close_cache):
                    close_cache()

    app = FastAPI(lifespan=lifespan)
    # localhost-bound service consumed by the Electron renderer (vite origin
    # in dev, file:// in prod) — REST needs CORS; the ws is exempt anyway
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    @app.get("/startup")
    async def startup_status() -> StartupStatus:
        status: StartupStatus = app.state.startup
        if status.ready or status.started_at is None:
            return status
        return status.model_copy(
            update={"elapsed_s": max(0.0, time.time() - status.started_at)}
        )

    @app.get("/graph")
    async def graph() -> GraphResponse:
        active_runtime = _ready_runtime(app)
        return _full_graph(active_runtime)

    @app.get("/graph/local")
    async def graph_local(center: str, depth: int = 2) -> LocalGraphResponse:
        active_runtime = _ready_runtime(app)
        return _local_graph(
            _full_graph(active_runtime),
            center=center,
            depth=min(3, max(1, int(depth))),
        )

    @app.get("/facts")
    async def facts(ids: str = "") -> FactsResponse:
        active_runtime = _ready_runtime(app)
        requested = tuple(
            dict.fromkeys(part.strip() for part in ids.split(",") if part.strip())
        )[:50]
        if not requested or active_runtime.knowledge_runtime is None:
            return FactsResponse(facts=[])
        snapshot = active_runtime.knowledge_runtime.snapshot
        store = snapshot.open_knowledge_store()
        try:
            rows = store.fact_contexts(requested)
        finally:
            store.close()
        return FactsResponse(
            facts=[
                FactPayload(
                    fact_id=row.fact_id,
                    subject=row.subject,
                    predicate=row.predicate,
                    object=row.object,
                    source_note_id=row.source_note_id,
                    confidence=row.confidence,
                    observed_at=row.observed_at,
                )
                for row in rows
            ]
        )

    @app.post("/index/sync", status_code=202)
    async def index_sync() -> SyncStatus:
        active_runtime = _ready_runtime(app)
        if (
            active_runtime.sync_status.running
            or active_runtime.sync_task is not None
            and not active_runtime.sync_task.done()
        ):
            raise HTTPException(status_code=409, detail="sync already running")
        active_runtime.sync_cancel = threading.Event()
        _set_sync_status(
            active_runtime,
            SyncStatus(running=True, phase="syncing"),
            force_emit=True,
        )
        active_runtime.sync_task = asyncio.create_task(_run_global_sync(active_runtime))
        return _sync_status_payload(active_runtime)

    @app.get("/index/sync")
    async def index_sync_status() -> SyncStatus:
        active_runtime = _ready_runtime(app)
        return _sync_status_payload(active_runtime)

    @app.post("/index/sync/cancel")
    async def index_sync_cancel() -> SyncStatus:
        active_runtime = _ready_runtime(app)
        if active_runtime.sync_status.running and active_runtime.sync_cancel is not None:
            active_runtime.sync_cancel.set()
            _set_sync_status(
                active_runtime,
                active_runtime.sync_status.model_copy(update={"detail": "cancelling"}),
                force_emit=True,
            )
        return _sync_status_payload(active_runtime)

    @app.get("/connectors")
    async def connectors() -> list[ConnectorInfo]:
        active_runtime = _ready_runtime(app)
        registry = _connector_registry(active_runtime)
        return registry.list_info(active_runtime.index)

    @app.post("/connectors/{connector_id}/enable")
    async def connector_enable(connector_id: str) -> ConnectorInfo:
        # Enabling only marks the source and flags the graph stale; nothing is
        # read or indexed until the user presses sync.
        active_runtime = _ready_runtime(app)
        registry = _connector_registry(active_runtime)
        try:
            registry.set_enabled(active_runtime.index, connector_id, True)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        _mark_index_stale(active_runtime)
        active_runtime.updates.publish("connectors")
        return _connector_info(active_runtime, connector_id)

    @app.post("/connectors/{connector_id}/disable")
    async def connector_disable(connector_id: str) -> ConnectorInfo:
        # Disabling surgically removes the source's rows from the derived
        # index (instant, no re-extraction); weights of the surviving graph
        # stay as-built until the next sync rebuilds cleanly.
        active_runtime = _ready_runtime(app)
        registry = _connector_registry(active_runtime)
        try:
            registry.set_enabled(active_runtime.index, connector_id, False)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        await _remove_source_from_knowledge(active_runtime, connector_id)
        _mark_index_stale(active_runtime)
        active_runtime.updates.publish("connectors")
        return _connector_info(active_runtime, connector_id)

    @app.post("/connectors/{connector_id}/data/delete")
    async def connector_delete_data(connector_id: str) -> ConnectorInfo:
        active_runtime = _ready_runtime(app)
        registry = _connector_registry(active_runtime)
        if connector_id in registry.syncing:
            raise HTTPException(status_code=409, detail="connector sync running — wait for it to finish")
        if (
            active_runtime.sync_status.running
            or active_runtime.sync_task is not None
            and not active_runtime.sync_task.done()
        ):
            raise HTTPException(status_code=409, detail="sync running — wait for it to finish")
        try:
            # Deletion is a privacy boundary: disable the source and cut its
            # rows out of both the source store and the derived index right
            # now — no re-extraction, no background indexing.
            registry.set_enabled(active_runtime.index, connector_id, False)
            await asyncio.to_thread(registry.purge_data, active_runtime.index, connector_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        await _remove_source_from_knowledge(active_runtime, connector_id)
        _mark_index_stale(active_runtime)
        active_runtime.updates.publish("connectors")
        return _connector_info(active_runtime, connector_id)

    @app.post("/connectors/{connector_id}/sync", status_code=202)
    async def connector_sync(connector_id: str) -> dict[str, str]:
        active_runtime = _ready_runtime(app)
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
        active_runtime = _ready_runtime(app)
        registry = _connector_registry(active_runtime)
        config_payload = _flat_config(payload)
        try:
            registry.set_config(active_runtime.index, connector_id, config_payload)
        except KeyError:
            raise HTTPException(status_code=404, detail="connector not found") from None
        return _connector_info(active_runtime, connector_id)

    @app.get("/notes/{note_id:path}")
    async def note_preview(note_id: str) -> NotePreview:
        active_runtime = _ready_runtime(app)
        preview = (
            _knowledge_note_preview(active_runtime.knowledge_runtime, note_id)
            if active_runtime.knowledge_runtime is not None
            else active_runtime.index.preview(note_id)
        )
        if preview is None:
            raise HTTPException(status_code=404, detail="note not found")
        return preview

    @app.get("/profile")
    async def get_profile() -> dict[str, Any]:
        active_runtime = _ready_runtime(app)
        return _profile_payload(active_runtime)

    @app.post("/profile/generate", status_code=202)
    async def generate_profile() -> dict[str, Any]:
        active_runtime = _ready_runtime(app)
        if (
            active_runtime.profile_generation.running
            or active_runtime.profile_generation_task is not None
            and not active_runtime.profile_generation_task.done()
        ):
            raise HTTPException(status_code=409, detail="profile generation already running")
        if active_runtime.profile_generator is None:
            raise HTTPException(status_code=503, detail="profile generation unavailable")
        active_runtime.profile_generation = ProfileGenerationStatus(
            running=True,
            phase="surveying",
            activity="Surveying the knowledge graph",
            started_at=time.time(),
        )
        active_runtime.profile_generation_task = asyncio.create_task(
            _run_profile_generation(active_runtime)
        )
        active_runtime.updates.publish("profile")
        return _profile_payload(active_runtime)

    @app.get("/summary")
    async def get_summary() -> MeetingSummary:
        active_runtime = _ready_runtime(app)
        if active_runtime.make_live_session is None:
            raise HTTPException(status_code=404, detail="live session unavailable")
        live = active_runtime.active_live_session
        if live is None:
            return MeetingSummary()
        return live.current_summary()

    @app.get("/chats")
    async def chats() -> list[Any]:
        active_runtime = _ready_runtime(app)
        return active_runtime.index.chats()

    @app.post("/chats")
    async def create_chat(payload: Any = Body(None)) -> Any:
        active_runtime = _ready_runtime(app)
        chat = active_runtime.index.create_chat(_new_chat_id(active_runtime.index), now=time.time())
        seed_note_ids = _seed_note_ids(payload)
        if seed_note_ids and (
            active_runtime.chat_engine is not None
            or active_runtime.make_chat_engine is not None
        ):
            engine = _chat_engine(active_runtime)
            seed = getattr(engine, "seed_chat", None)
            if callable(seed):
                await asyncio.to_thread(seed, chat.chat_id, seed_note_ids)
        active_runtime.updates.publish("chats")
        return chat

    @app.get("/chats/{chat_id}")
    async def chat_detail(chat_id: str) -> dict[str, Any]:
        active_runtime = _ready_runtime(app)
        chat = active_runtime.index.chat(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="chat not found")
        return {"chat": chat, "messages": active_runtime.index.chat_messages(chat_id)}

    @app.patch("/chats/{chat_id}")
    async def rename_chat(chat_id: str, payload: Any = Body(...)) -> Any:
        active_runtime = _ready_runtime(app)
        if active_runtime.index.chat(chat_id) is None:
            raise HTTPException(status_code=404, detail="chat not found")
        if not isinstance(payload, dict) or not isinstance(payload.get("title"), str):
            raise HTTPException(status_code=422, detail="title must be a string")
        active_runtime.index.rename_chat(chat_id, payload["title"].strip()[:120], now=time.time())
        active_runtime.updates.publish("chats")
        return active_runtime.index.chat(chat_id)

    @app.delete("/chats/{chat_id}")
    async def delete_chat(chat_id: str) -> dict[str, bool]:
        active_runtime = _ready_runtime(app)
        if active_runtime.index.chat(chat_id) is None:
            raise HTTPException(status_code=404, detail="chat not found")
        active_runtime.index.delete_chat(chat_id)
        active_runtime.updates.publish("chats")
        return {"ok": True}

    @app.put("/profile")
    async def put_profile(payload: Any = Body(...)) -> dict[str, Any]:
        active_runtime = _ready_runtime(app)
        card = payload.get("card") if isinstance(payload, dict) else None
        if not isinstance(card, str) or not card.strip():
            raise HTTPException(status_code=422, detail="card must be a non-empty string")
        active_runtime.index.set_user_card(card.strip())  # type: ignore[attr-defined]
        if not active_runtime.profile_generation.running:
            active_runtime.profile_generation = ProfileGenerationStatus()
        active_runtime.updates.publish("profile")
        return _profile_payload(active_runtime)

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
        active_runtime = await _ready_runtime_ws(app, websocket)
        if active_runtime is None:
            return
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
        active_runtime = await _ready_runtime_ws(app, websocket)
        if active_runtime is None:
            return
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
        if active_runtime.make_live_session is None:
            await websocket.close(code=1011)
            logger.close()
            return
        live = active_runtime.make_live_session(emit, logger)
        active_runtime.active_live_session = live
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
            if active_runtime.active_live_session is live:
                active_runtime.active_live_session = None
            active_runtime.status_sinks.discard(emit_index_status)
            if not session_started:
                logger.log("session_start", channels=[])
            await asr.stop()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            logger.close()

    @app.websocket("/chat")
    async def chat(websocket: WebSocket) -> None:
        await websocket.accept()
        active_runtime = await _ready_runtime_ws(app, websocket)
        if active_runtime is None:
            return
        engine = _chat_engine(active_runtime)
        send_lock = asyncio.Lock()
        tasks: set[asyncio.Task[Any]] = set()

        async def emit(event: Any) -> None:
            if isinstance(event, ChatMetaEvent):
                active_runtime.updates.publish("chats")
            text = event.model_dump_json() if isinstance(event, BaseModel) else json.dumps(event)
            async with send_lock:
                await websocket.send_text(text)

        def track(task: asyncio.Task[Any]) -> None:
            tasks.add(task)

            def done(finished: asyncio.Task[Any]) -> None:
                tasks.discard(finished)
                try:
                    exc = finished.exception()
                except asyncio.CancelledError:
                    return
                if exc is None:
                    return
                task_logger = getattr(engine, "logger", None)
                log = getattr(task_logger, "log", None)
                if callable(log):
                    try:
                        log("chat_task_error", detail=str(exc))
                        return
                    except Exception:
                        pass
                logging.getLogger("taurine.chat").exception(
                    "chat websocket task failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

            task.add_done_callback(done)

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw = message.get("text")
                if raw is None:
                    continue
                try:
                    control = _parse_chat_control(raw)
                except (json.JSONDecodeError, ValidationError, ValueError):
                    continue
                if isinstance(control, ChatSendMsg):
                    if control.seed_note_ids:
                        track(
                            asyncio.create_task(
                                engine.send(
                                    control.chat_id,
                                    control.text,
                                    emit,
                                    seed_note_ids=control.seed_note_ids,
                                )
                            )
                        )
                    else:
                        track(asyncio.create_task(engine.send(control.chat_id, control.text, emit)))
                elif isinstance(control, ChatCancelMsg):
                    await engine.cancel(control.chat_id)
                elif isinstance(control, ChatRegenerateMsg):
                    track(asyncio.create_task(engine.regenerate(control.chat_id, emit)))
        except WebSocketDisconnect:
            pass
        finally:
            for task in tuple(tasks):
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    return app


def build_runtime(progress: Callable[[str, str], None] | None = None) -> Runtime:
    def report(phase: str, detail: str = "") -> None:
        if progress is not None:
            progress(phase, detail)

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    secrets = required_secrets()
    index = Index(Path("taurine.db"))
    llm = OpenRouterClient(secrets["OPEN_ROUTER_API_KEY"])
    report("loading_models", "span extraction model")
    span_extractor = Gliner2SpanExtractor()
    connector_registry = ConnectorRegistry()
    connector_registry.reconcile_ledger(index)

    def complete_relations(notes: Any) -> object:
        payload = {
            "predicates": {
                name: spec.description for name, spec in sorted(PREDICATES.items())
            },
            "notes": [
                {
                    "note_id": note.note_id,
                    "observed_at": note.observed_at,
                    "entities": [asdict(entity) for entity in note.entities],
                    "passages": [asdict(passage) for passage in note.passages],
                }
                for note in notes
            ],
        }
        return asyncio.run(
            llm.complete_json(
                config.CHAT_MODEL,
                (
                    "Return JSON {facts:[...]}. Find only relations directly supported by the "
                    "supplied passage text. Reuse entity_id and passage_id exactly, select a "
                    "registered predicate when possible, and provide source-exact assertion_start "
                    "and assertion_end offsets plus confidence. For an unregistered predicate, "
                    "also provide predicate_description. Do not infer unsupported relations."
                ),
                json.dumps(payload, ensure_ascii=False),
                timeout_s=config.SUMMARY_TIMEOUT_S,
            )
        )

    report("loading_models", "passage tokenizer")
    # Embeddings and extractions are content-keyed, so repeat syncs only pay
    # for notes that actually changed.
    build_cache = BuildCache(Path("taurine-cache.db"))
    knowledge_builder = BuildCoordinator(
        BuildDependencies(
            tokenizer=passage_tokenizer(),
            embedder=build_cache.caching_embedder(config.EMBED_MODEL, embed_texts),
            extractor=build_cache.caching_extractor(extract_notes),
            relation_completer=complete_relations,
        )
    )
    knowledge_database = Path("taurine-knowledge.db")
    try:
        report("opening_index", "loading knowledge index")
        knowledge_runtime = RetrievalRuntime(knowledge_database)
    except (FileNotFoundError, ValueError, KnowledgeSchemaError):
        # Startup never reads sources or indexes user data: bootstrap an
        # empty (but valid) knowledge index so retrieval components load.
        # Real data enters only through the user-triggered sync.
        report("opening_index", "creating empty knowledge index")
        knowledge_builder.build((), knowledge_database, replace_derived=True)
        knowledge_runtime = RetrievalRuntime(knowledge_database)
    report("finalizing", "")

    def regenerate_knowledge(progress: Callable[[str], None] | None = None) -> Any:
        def note(detail: str) -> None:
            if progress is not None:
                progress(detail)

        note("reading enabled sources")
        records = connector_registry.extract_all_enabled(
            index,
            progress=lambda connector_id: note(f"reading {connector_id}"),
        )
        build_report = knowledge_builder.build(
            records,
            knowledge_runtime.knowledge_database,
            replace_derived=True,
            on_stage=note,
        )
        knowledge_runtime.activate_from_disk()
        return build_report

    def make_live_session(emit: EventSink, logger: SessionLogger) -> LiveSession:
        clock = _MonotonicClock()
        registry = EvidenceRegistry()
        grounding = GroundingEngine(registry)
        return LiveSession(
            registry=registry,
            ambient=AmbientRetriever(
                knowledge_runtime.make_engine("live", registry),
                clock,
                span_extractor=span_extractor,
            ),
            grounding=grounding,
            trigger=SummaryTrigger(clock.now),
            summarizer=Summarizer(
                llm,
                index,
                knowledge_runtime.make_validator(registry),
            ),
            emit=emit,
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
        # Autosync is intentionally off: indexing only runs on user intent
        # (the sync button or a data delete).
        enable_periodic_sync=False,
        connector_registry=connector_registry,
        connector_llm=llm,
        make_chat_engine=lambda: ChatEngine(
            index,
            llm,
            SessionLogger.create(session_log.SESSIONS_DIR),
            span_extractor=span_extractor,
            retriever_factory=lambda chat_id: ChatRetriever(
                index,
                chat_id,
                engine_factory=lambda registry: knowledge_runtime.make_engine(
                    "chat", registry
                ),
                tool_engine_factory=lambda registry: knowledge_runtime.make_engine(
                    "tool", registry
                ),
                span_extractor=span_extractor,
            ),
            validator_factory=knowledge_runtime.make_validator,
        ),
        make_profile_generator=lambda: ProfileGenerator(
            index,
            llm,
            retriever=ChatRetriever(
                index,
                f"__profile_generation__-{uuid4().hex}",
                engine_factory=lambda registry: knowledge_runtime.make_engine(
                    "tool", registry
                ),
                tool_engine_factory=lambda registry: knowledge_runtime.make_engine(
                    "tool", registry
                ),
                span_extractor=span_extractor,
            ),
        ),
        knowledge_runtime=knowledge_runtime,
        regenerate_knowledge=regenerate_knowledge,
        build_cache=build_cache,
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


def _full_graph(runtime: Runtime) -> GraphResponse:
    if runtime.knowledge_runtime is None:
        legacy = runtime.index.graph()
        return (
            legacy
            if isinstance(legacy, GraphResponse)
            else GraphResponse.model_validate(legacy)
        )
    return _knowledge_graph(runtime.knowledge_runtime)


def _local_graph(
    graph: GraphResponse,
    *,
    center: str,
    depth: int,
) -> LocalGraphResponse:
    """BFS over the (undirected) projection adjacency from center."""

    node_ids = {node.note_id for node in graph.nodes}
    if center not in node_ids:
        raise HTTPException(status_code=404, detail="note not found")
    adjacency: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.src in node_ids and edge.dst in node_ids:
            adjacency.setdefault(edge.src, set()).add(edge.dst)
            adjacency.setdefault(edge.dst, set()).add(edge.src)
    visited = {center}
    frontier = [center]
    for _hop in range(depth):
        frontier = sorted(
            {
                neighbor
                for node in frontier
                for neighbor in adjacency.get(node, ())
                if neighbor not in visited
            }
        )
        if not frontier:
            break
        visited.update(frontier)
    return LocalGraphResponse(
        center=center,
        depth=depth,
        nodes=[node for node in graph.nodes if node.note_id in visited],
        edges=[
            edge
            for edge in graph.edges
            if edge.src in visited and edge.dst in visited
        ],
    )


def _seed_note_ids(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    values = payload.get("seed_note_ids")
    if not isinstance(values, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip() for value in values if str(value).strip()
        )
    )


def _knowledge_graph(knowledge_runtime: RetrievalRuntime) -> GraphResponse:
    snapshot = knowledge_runtime.snapshot
    store = snapshot.open_knowledge_store()
    try:
        notes = {row.note_id: row for row in store.source_notes()}
        note_entities = store.note_entities()
    finally:
        store.close()
    labels_fn = getattr(knowledge_runtime, "community_labels", None)
    communities: Mapping[str, int] = (
        labels_fn(snapshot)
        if callable(labels_fn)
        else projection_communities(snapshot.projection)
    )
    degree = {note_id: 0 for note_id in notes}
    edges: list[GraphEdge] = []
    for row in snapshot.projection:
        if row.src_note_id not in notes or row.dst_note_id not in notes:
            continue
        degree[row.src_note_id] += 1
        degree[row.dst_note_id] += 1
        edges.append(
            GraphEdge(
                src=row.src_note_id,
                dst=row.dst_note_id,
                type=row.edge_type,
                weight=row.weight,
                shared=list(row.shared),
                fact_ids=list(row.fact_ids),
                fact_provenance=(
                    "directed_facts" if row.fact_ids else ""
                ),
                mechanisms=dict(row.mechanisms),
            )
        )
    return GraphResponse(
        nodes=[
            GraphNode(
                note_id=note.note_id,
                title=note.title,
                degree=degree[note.note_id],
                source=note.source,
                gist=note.display_line,
                created=note.created,
                modified=note.modified,
                folder=note.folder,
                community=communities.get(note.note_id, -1),
                value_score=note.value_score,
                entities=list(note_entities.get(note.note_id, ())),
            )
            for note in notes.values()
        ],
        edges=edges,
    )


def _knowledge_note_preview(
    knowledge_runtime: RetrievalRuntime,
    note_id: str,
) -> NotePreview | None:
    snapshot = knowledge_runtime.snapshot
    store = snapshot.open_knowledge_store()
    try:
        notes = {row.note_id: row for row in store.source_notes()}
        note = notes.get(note_id)
        if note is None:
            return None
        fact_rows = {row.fact_id: row for row in store.fact_contexts()}
        neighbors: list[NeighborRef] = []
        for edge in snapshot.projection:
            if note_id not in {edge.src_note_id, edge.dst_note_id}:
                continue
            neighbor_id = (
                edge.dst_note_id if edge.src_note_id == note_id else edge.src_note_id
            )
            neighbor = notes.get(neighbor_id)
            if neighbor is None:
                continue
            facts = [
                fact_rows[fact_id].canonical_text
                for fact_id in edge.fact_ids
                if fact_id in fact_rows
            ]
            neighbors.append(
                NeighborRef(
                    note_id=neighbor.note_id,
                    title=neighbor.title,
                    source=neighbor.source,
                    weight=edge.weight,
                    fact="; ".join(facts),
                )
            )
        neighbors.sort(key=lambda row: (-row.weight, row.note_id))
        return NotePreview(
            note_id=note.note_id,
            title=note.title,
            source=note.source,
            gist=note.display_line,
            text=note.text,
            neighbors=neighbors,
        )
    finally:
        store.close()


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
    elif isinstance(message, ActionableStatusMsg):
        await live.set_actionable_status(message.id, message.status)
    elif isinstance(message, OpenNoteMsg):
        try:
            await asyncio.to_thread(runtime.open_note, message.note_id)
        except Exception as exc:  # pragma: no cover - platform integration guard
            await emit(_status_error(runtime, str(exc)))
    return session_started


def _parse_control(raw: str) -> StartMsg | StopMsg | ActionableStatusMsg | OpenNoteMsg:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("control message must be a JSON object")
    message_type = payload.get("type")
    if message_type == "start":
        return StartMsg.model_validate(payload)
    if message_type == "stop":
        return StopMsg.model_validate(payload)
    if message_type == "actionable_status":
        return ActionableStatusMsg.model_validate(payload)
    if message_type == "open_note":
        return OpenNoteMsg.model_validate(payload)
    raise ValueError(f"unsupported control type: {message_type}")


def _parse_chat_control(raw: str) -> ChatSendMsg | ChatCancelMsg | ChatRegenerateMsg:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("chat control message must be a JSON object")
    message_type = payload.get("type")
    if message_type == "chat_send":
        return ChatSendMsg.model_validate(payload)
    if message_type == "chat_cancel":
        return ChatCancelMsg.model_validate(payload)
    if message_type == "chat_regenerate":
        return ChatRegenerateMsg.model_validate(payload)
    raise ValueError(f"unsupported chat control type: {message_type}")


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
        if runtime.regenerate_knowledge is not None:
            await asyncio.to_thread(runtime.regenerate_knowledge)
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
        if runtime.regenerate_knowledge is not None:
            await asyncio.to_thread(runtime.regenerate_knowledge)
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


async def _run_global_sync(runtime: Runtime) -> None:
    """The single user-triggered job: sync enabled sources, then reindex."""
    log = logging.getLogger("taurine.sync")
    cancel = runtime.sync_cancel
    error = ""
    cancelled = False
    started = time.monotonic()

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise BuildCancelled("sync cancelled")

    def indexing_progress(detail: str) -> None:
        check_cancel()
        _set_sync_status(
            runtime,
            SyncStatus(running=True, phase="indexing", detail=detail),
        )

    try:
        log.info("sync started")
        _set_sync_status(
            runtime,
            SyncStatus(running=True, phase="syncing", detail="reading enabled sources"),
        )
        sync = runtime.sync_all_enabled or runtime.sync_once
        await asyncio.to_thread(sync)
        check_cancel()
        if runtime.regenerate_knowledge is not None:
            _set_sync_status(runtime, SyncStatus(running=True, phase="indexing"))
            await asyncio.to_thread(runtime.regenerate_knowledge, indexing_progress)
        runtime.index_stale = False
        log.info("sync complete in %.1f s", time.monotonic() - started)
    except asyncio.CancelledError:
        raise
    except BuildCancelled:
        cancelled = True
        runtime.index_stale = True  # sources may have advanced past the index
        log.info("sync cancelled after %.1f s", time.monotonic() - started)
    except Exception as exc:
        log.exception("sync failed after %.1f s", time.monotonic() - started)
        error = str(exc)[:300] or type(exc).__name__
        runtime.index_stale = True
    finally:
        runtime.sync_cancel = None
        _set_sync_status(
            runtime,
            SyncStatus(
                running=False,
                phase="idle",
                detail="cancelled" if cancelled else "",
                error=error,
            ),
            force_emit=True,
        )
        _bump_graph(runtime)
        runtime.updates.publish("connectors")
        await _emit_index_status(runtime)


async def _run_profile_generation(runtime: Runtime) -> None:
    generator = runtime.profile_generator
    assert generator is not None
    try:
        card = await generator.generate(
            lambda progress: _set_profile_progress(runtime, progress)
        )
        runtime.index.set_user_card(card)
        runtime.profile_generation = ProfileGenerationStatus()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        current = _profile_status(runtime)
        logging.getLogger("taurine.profile").exception("profile generation failed")
        runtime.profile_generation = ProfileGenerationStatus(
            running=False,
            phase="failed",
            activity=current.activity,
            nodes_read=current.nodes_read,
            tool_calls=current.tool_calls,
            started_at=current.started_at,
            elapsed_s=current.elapsed_s,
            error=str(exc)[:300] or "Profile generation failed",
        )
    finally:
        runtime.profile_generation_task = None
        runtime.updates.publish("profile")


def _set_profile_progress(runtime: Runtime, progress: ProfileProgress) -> None:
    started_at = runtime.profile_generation.started_at or time.time()
    runtime.profile_generation = ProfileGenerationStatus(
        running=True,
        phase=progress.phase,
        activity=progress.activity,
        nodes_read=progress.nodes_read,
        tool_calls=progress.tool_calls,
        started_at=started_at,
        elapsed_s=max(0.0, time.time() - started_at),
    )
    runtime.updates.publish("profile")


def _profile_status(runtime: Runtime) -> ProfileGenerationStatus:
    status = runtime.profile_generation
    if not status.running or status.started_at is None:
        return status
    return status.model_copy(
        update={"elapsed_s": max(0.0, time.time() - status.started_at)}
    )


def _profile_payload(runtime: Runtime) -> dict[str, Any]:
    card = getattr(runtime.index, "user_card", lambda: None)()
    return {
        "card": card,
        "generation": _profile_status(runtime).model_dump(mode="json"),
    }


def _stage_token(detail: str) -> str:
    return detail.split(" ", 1)[0] if detail else ""


def _set_sync_status(
    runtime: Runtime,
    status: SyncStatus,
    *,
    force_emit: bool = False,
) -> None:
    """Status fan-out; safe from worker threads (indexing progress callbacks
    run inside asyncio.to_thread).

    Same-stage ticks (embeddings 12/463 → 13/463) are throttled to one emit
    per 2s, but phase and stage transitions always emit — otherwise the last
    tick before a long silent stage lingers on screen and reads as a hang.
    """
    previous = runtime.sync_status
    runtime.sync_status = status
    now = time.monotonic()
    should_emit = (
        force_emit
        or not status.running
        or status.phase != previous.phase
        or _stage_token(status.detail) != _stage_token(previous.detail)
        or now - runtime.sync_last_emit_s >= 2.0
    )
    if not should_emit:
        return
    runtime.sync_last_emit_s = now
    runtime.updates.publish("sync")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = runtime.updates.loop
        if loop is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(_emit_index_status(runtime), loop)
        return
    loop.create_task(_emit_index_status(runtime))


def _sync_status_payload(runtime: Runtime) -> SyncStatus:
    return runtime.sync_status.model_copy(update={"stale": runtime.index_stale})


def _mark_index_stale(runtime: Runtime) -> None:
    runtime.index_stale = True
    runtime.updates.publish("sync")


async def _remove_source_from_knowledge(runtime: Runtime, connector_id: str) -> None:
    remove = getattr(runtime.knowledge_runtime, "remove_source", None)
    if not callable(remove):
        return
    removed = await asyncio.to_thread(remove, connector_id)
    if removed:
        logging.getLogger("taurine.sync").info(
            "removed %d %s notes from the knowledge index", removed, connector_id
        )
        _bump_graph(runtime)


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
    if "sync" in kinds:
        await websocket.send_text(
            json.dumps(
                {"type": "sync", "status": _sync_status_payload(runtime).model_dump()}
            )
        )
    if "graph" in kinds:
        await websocket.send_text(
            json.dumps({"type": "graph", "version": runtime.graph_version})
        )
    if "profile" in kinds:
        await websocket.send_text(
            json.dumps({"type": "profile", **_profile_payload(runtime)})
        )
    if "chats" in kinds:
        chats = getattr(runtime.index, "chats", lambda: [])()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "chats",
                    "chats": [
                        chat.model_dump(mode="json") if isinstance(chat, BaseModel) else chat
                        for chat in chats
                    ],
                }
            )
        )


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
        sync=_sync_status_payload(runtime),
    )


def _status_error(runtime: Runtime, detail: str) -> StatusEvent:
    return StatusEvent(asr="error", index=_index_status(runtime), detail=detail)


def _ready_runtime(app: FastAPI) -> Runtime:
    runtime: Runtime | None = app.state.runtime
    if runtime is None:
        status: StartupStatus = app.state.startup
        if status.error:
            detail = f"server startup failed: {status.error}"
        else:
            detail = f"server is starting ({status.phase})"
        raise HTTPException(status_code=503, detail=detail)
    return runtime


async def _ready_runtime_ws(app: FastAPI, websocket: WebSocket) -> Runtime | None:
    """Close pre-ready websockets with 1013 so clients retry after startup."""
    runtime: Runtime | None = app.state.runtime
    if runtime is None:
        await websocket.close(code=1013)
    return runtime


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


def _chat_engine(runtime: Runtime) -> Any:
    if runtime.chat_engine is None:
        if runtime.make_chat_engine is None:
            raise HTTPException(status_code=503, detail="chat unavailable")
        runtime.chat_engine = runtime.make_chat_engine()
    return runtime.chat_engine


def _new_chat_id(index: Any) -> str:
    while True:
        chat_id = f"chat-{uuid4().hex[:8]}"
        if index.chat(chat_id) is None:
            return chat_id


def _flat_config(payload: Any) -> dict[str, Any]:
    """Scalars and lists of scalars only — matches connector configs like
    codex_mem globs / local_folders folders; nesting stays rejected."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="config must be a JSON object")
    config_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise HTTPException(status_code=422, detail="config keys must be strings")
        if isinstance(value, list):
            if any(isinstance(item, dict | list | tuple | set) for item in value):
                raise HTTPException(status_code=422, detail="config lists must hold scalars")
            config_payload[key] = list(value)
            continue
        if isinstance(value, dict | tuple | set):
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
