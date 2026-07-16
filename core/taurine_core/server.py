from __future__ import annotations

import asyncio
import faulthandler
import inspect
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
from taurine_core.connectors.registry import ConnectorRegistry, SourceExtraction
from taurine_core.indexer.notes import open_note
from taurine_core.indexer.build import (
    BuildCancelled,
    BuildCoordinator,
    BuildDependencies,
    carry_forward_records,
    relation_completer_for,
    sweep_stale_build_files,
)
from taurine_core.indexer.context_enrich import context_enricher_for
from taurine_core.indexer.edge_adjudication import (
    SemanticPair,
    adjudicate_semantic_edges,
    edge_adjudicator_for,
)
from taurine_core.profile_context import owner_aliases as profile_owner_aliases
from taurine_core.indexer.build_cache import BuildCache
from taurine_core.indexer.embed import embed_texts
from taurine_core import idea_search
from taurine_core.indexer.communities import fallback_clusters
from taurine_core.indexer.extract import extract_notes_dispatch as extract_notes
from taurine_core.indexer.knowledge_store import KnowledgeSchemaError, KnowledgeStore
from taurine_core.models_local import passage_tokenizer
from taurine_core.indexer.profile import ProfileGenerator, ProfileProgress
from taurine_core.indexer.store import Index
from taurine_core.live import ambient as live_ambient
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
    ClusterInfo,
    ConnectorInfo,
    FactPayload,
    FactsResponse,
    GraphEdge,
    GraphNode,
    GraphResponse,
    IdeaSearchRequest,
    IdeaSearchResponse,
    IndexStatus,
    LocalGraphResponse,
    MeetingSummary,
    MeetingCreateRequest,
    MeetingUpdateRequest,
    NeighborRef,
    NotePreview,
    NotepadMsg,
    OpenNoteMsg,
    ProfileGenerationStatus,
    StartMsg,
    StartupStatus,
    SyncStatus,
    StatusEvent,
    StopMsg,
    SyncReport,
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

    def set_notepad(self, text_md: str) -> None:
        ...

    def current_summary(self) -> MeetingSummary:
        ...

    async def aclose(self) -> None:
        ...


class UpdateBus:
    """Fan-out of state-change notifications to /updates websockets.

    publish() is thread-safe — registry callbacks fire from sync worker
    threads — and only enqueues a kind marker; each websocket builds fresh
    payloads when it drains its queue, so bursts coalesce naturally.
    """

    KINDS = ("connectors", "sync", "graph", "profile", "chats", "meetings")

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
class SessionContext:
    generation: int
    meeting_id: str | None
    live: LiveSession | None = None
    asr: ASRClient | None = None


@dataclass
class Runtime:
    index: Any
    sync_once: Callable[[], SyncReport]
    open_note: Callable[[str], None]
    make_live_session: Callable[..., LiveSessionLike] | None
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
    active_sessions: dict[str, SessionContext] = field(default_factory=dict)
    meeting_epochs: dict[str, int] = field(default_factory=dict)
    meeting_finalizers: dict[
        str, asyncio.TimerHandle | asyncio.Task[Any]
    ] = field(default_factory=dict)
    active_unbound_sessions: dict[int, SessionContext] = field(default_factory=dict)
    context_epochs: dict[int, int] = field(default_factory=dict)
    context_transcript_buffers: dict[int, Any] = field(default_factory=dict)
    pending_live_sessions: dict[
        int, Callable[[], LiveSessionLike | None]
    ] = field(default_factory=dict)
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

    def current_live_session(self) -> LiveSessionLike | None:
        contexts = [
            *self.active_sessions.values(),
            *self.active_unbound_sessions.values(),
        ]
        for context in reversed(contexts):
            if context.live is not None:
                return context.live
        for initialize in reversed(self.pending_live_sessions.values()):
            live = initialize()
            if live is not None:
                return live
        return None


class EdgeReasonRequest(BaseModel):
    src_passage_id: str
    dst_passage_id: str


class EdgeReasonResponse(BaseModel):
    reason: str
    generated: bool


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
                finalizer_tasks = [
                    finalizer
                    for finalizer in active_runtime.meeting_finalizers.values()
                    if isinstance(finalizer, asyncio.Task)
                ]
                for finalizer in active_runtime.meeting_finalizers.values():
                    finalizer.cancel()
                active_runtime.meeting_finalizers.clear()
                if finalizer_tasks:
                    await asyncio.gather(*finalizer_tasks, return_exceptions=True)
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

    @app.post("/graph/idea")
    async def graph_idea(payload: IdeaSearchRequest) -> IdeaSearchResponse:
        active_runtime = _ready_runtime(app)
        if not payload.idea.strip():
            raise HTTPException(status_code=400, detail="idea must not be empty")
        if (
            active_runtime.connector_llm is None
            or active_runtime.knowledge_runtime is None
        ):
            raise HTTPException(
                status_code=503,
                detail="idea search requires a configured LLM and knowledge index",
            )
        return await idea_search.search_idea(
            payload.idea,
            snapshot=active_runtime.knowledge_runtime.snapshot,
            llm=active_runtime.connector_llm,
            limit=payload.limit,
        )

    @app.post("/graph/edge-reason")
    async def graph_edge_reason(payload: EdgeReasonRequest) -> EdgeReasonResponse:
        from taurine_core.indexer.edge_prune import is_template_reason

        active_runtime = _ready_runtime(app)
        if active_runtime.knowledge_runtime is None:
            raise HTTPException(status_code=503, detail="knowledge index unavailable")
        snapshot = active_runtime.knowledge_runtime.snapshot
        database_path = getattr(snapshot, "database_path", None)
        if database_path is None:
            raise HTTPException(status_code=503, detail="knowledge index unavailable")
        store = KnowledgeStore(database_path, read_only=True)
        try:
            context = store.semantic_edge_context(
                payload.src_passage_id, payload.dst_passage_id
            )
        finally:
            store.close()
        if context is None:
            raise HTTPException(status_code=404, detail="semantic edge not found")
        if not is_template_reason(context.reason):
            return EdgeReasonResponse(reason=context.reason, generated=False)
        if active_runtime.connector_llm is None:
            return EdgeReasonResponse(reason=context.reason, generated=False)

        pair = SemanticPair(
            src_passage_id=context.src_passage_id,
            dst_passage_id=context.dst_passage_id,
            src_title=context.src_title,
            dst_title=context.dst_title,
            src_text=context.src_text,
            dst_text=context.dst_text,
            src_content_hash=context.src_content_hash,
            dst_content_hash=context.dst_content_hash,
        )
        outcome = await asyncio.to_thread(
            adjudicate_semantic_edges,
            [pair],
            judge=edge_adjudicator_for(active_runtime.connector_llm),
        )
        reason = outcome.reasons.get(pair.pair)
        if not reason:
            return EdgeReasonResponse(reason=context.reason, generated=False)
        try:
            writable = KnowledgeStore(database_path)
            try:
                persisted = writable.update_semantic_edge_reason(
                    pair.src_passage_id, pair.dst_passage_id, reason
                )
            finally:
                writable.close()
        except Exception as exc:
            logging.getLogger("taurine.server").warning(
                "could not persist lazy semantic-edge reason: %s", exc
            )
            persisted = False
        return EdgeReasonResponse(reason=reason, generated=persisted)

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
        live = active_runtime.current_live_session()
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

    @app.get("/meetings")
    async def meetings() -> dict[str, list[dict[str, Any]]]:
        active_runtime = _ready_runtime(app)
        return {"meetings": active_runtime.index.meetings_list()}

    @app.post("/meetings")
    async def create_meeting(
        payload: MeetingCreateRequest | None = Body(None),
    ) -> dict[str, Any]:
        active_runtime = _ready_runtime(app)
        meeting = active_runtime.index.meeting_create(payload.title if payload else "")
        active_runtime.updates.publish("meetings")
        return meeting

    @app.get("/meetings/{meeting_id}")
    async def meeting_detail(meeting_id: str) -> dict[str, Any]:
        active_runtime = _ready_runtime(app)
        meeting = active_runtime.index.meeting_get(meeting_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="meeting not found")
        return meeting

    @app.patch("/meetings/{meeting_id}")
    async def update_meeting(
        meeting_id: str, payload: MeetingUpdateRequest = Body(...)
    ) -> dict[str, Any]:
        active_runtime = _ready_runtime(app)
        previous = active_runtime.index.meeting_get(meeting_id)
        if previous is None:
            raise HTTPException(status_code=404, detail="meeting not found")
        suggestions = payload.suggestions
        if suggestions is not None:
            suggestions = _suggestion_status_patch(
                previous["suggestions"], suggestions
            )
        meeting = active_runtime.index.meeting_update(
            meeting_id,
            title=payload.title,
            notes_md=payload.notes_md,
            suggestions=suggestions,
        )
        if payload.title is not None and meeting["title"] != previous["title"]:
            active_runtime.updates.publish("meetings")
        return meeting

    @app.delete("/meetings/{meeting_id}")
    async def delete_meeting(meeting_id: str) -> dict[str, bool]:
        active_runtime = _ready_runtime(app)
        if active_runtime.index.meeting_get(meeting_id) is None:
            raise HTTPException(status_code=404, detail="meeting not found")
        finalizer = active_runtime.meeting_finalizers.pop(meeting_id, None)
        if finalizer is not None:
            finalizer.cancel()
        active_runtime.index.meeting_delete(meeting_id)
        active_runtime.updates.publish("meetings")
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
        ctx_holder = [SessionContext(generation=0, meeting_id=None)]
        transcript_buffer: list[_MeetingTranscriptBuffer | None] = [None]

        async def on_final_chunk(chunk: dict[str, Any]) -> None:
            buffer = transcript_buffer[0]
            if buffer is not None:
                await buffer.append(chunk)

        pending_key = id(ctx_holder)

        def initialize_pending_live() -> LiveSessionLike | None:
            context = ctx_holder[0]
            if context.generation != 0:
                return context.live
            if context.live is None:
                context.live = _make_live_session(
                    active_runtime, emit, logger, on_final_chunk
                )  # type: ignore[assignment]
            return context.live

        active_runtime.pending_live_sessions[pending_key] = initialize_pending_live

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
                        ctx_holder=ctx_holder,
                        transcript_buffer=transcript_buffer,
                        on_final_chunk=on_final_chunk,
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
                    asr = ctx_holder[0].asr
                    if asr is None:
                        await emit(_status_error(active_runtime, "session has not started"))
                        continue
                    await asr.send_pcm_s16(channel, pcm)
        except WebSocketDisconnect:
            pass
        finally:
            active_runtime.pending_live_sessions.pop(pending_key, None)
            active_runtime.status_sinks.discard(emit_index_status)
            if not session_started:
                logger.log("session_start", channels=[])
            context = ctx_holder[0]
            if context.meeting_id is not None:
                snapshot = await _capture_finalization_snapshot(
                    active_runtime, context, transcript_buffer[0]
                )
                if snapshot is not None:
                    _schedule_meeting_finalization(active_runtime, snapshot)
            await _teardown_session_context(context)
            if context.meeting_id is not None and snapshot is not None:
                await _refresh_snapshot_transcript(
                    active_runtime, snapshot, transcript_buffer[0]
                )
            _remove_active_context(active_runtime, context)
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

    complete_relations = relation_completer_for(llm)

    report("loading_models", "passage tokenizer")
    # Embeddings and extractions are content-keyed, so repeat syncs only pay
    # for notes that actually changed.
    build_cache = BuildCache(Path("taurine-cache.db"))
    knowledge_builder = BuildCoordinator(
        BuildDependencies(
            tokenizer=passage_tokenizer(),
            embedder=build_cache.caching_embedder(config.embed_model_id(), embed_texts),
            extractor=build_cache.caching_extractor(extract_notes),
            relation_completer=complete_relations,
            context_enricher=context_enricher_for(llm),
            context_cache=build_cache,
            edge_adjudicator=edge_adjudicator_for(llm),
            adjudication_cache=build_cache,
            completion_cache=build_cache,
        )
    )
    knowledge_database = Path("taurine-knowledge.db")
    # L25: crash-orphaned temp builds older than a day are swept at startup
    sweep_stale_build_files(knowledge_database)
    try:
        report("opening_index", "loading knowledge index")
        knowledge_runtime = RetrievalRuntime(knowledge_database)
    except (FileNotFoundError, ValueError, KnowledgeSchemaError) as exc:
        # Startup never reads sources or indexes user data: bootstrap an
        # empty (but valid) knowledge index so retrieval components load.
        # Real data enters only through the user-triggered sync.
        if knowledge_database.exists():
            # A present-but-incompatible index (manifest/schema drift after an
            # upgrade) must never be silently destroyed: keep a timestamped
            # copy beside it and say so loudly.
            backup = knowledge_database.with_name(
                f"{knowledge_database.name}.incompatible-{int(time.time())}"
            )
            os.replace(knowledge_database, backup)
            logging.getLogger("taurine.server").warning(
                "knowledge index incompatible with current code (%s); moved to %s "
                "and starting empty — run a sync to rebuild",
                exc,
                backup.name,
            )
        report("opening_index", "creating empty knowledge index")
        knowledge_builder.build((), knowledge_database, replace_derived=True)
        knowledge_runtime = RetrievalRuntime(knowledge_database)
    report("finalizing", "")

    def regenerate_knowledge(
        progress: Callable[[str], None] | None = None,
        extraction: SourceExtraction | None = None,
    ) -> Any:
        def note(detail: str) -> None:
            if progress is not None:
                progress(detail)

        if extraction is None:
            # standalone reindex: read the sources once here. The global sync
            # path instead passes the records it already extracted (L28).
            note("reading enabled sources")
            extraction = connector_registry.extract_all_enabled(
                index,
                progress=lambda connector_id: note(f"reading {connector_id}"),
            )
        records = list(extraction.records)
        degraded_reasons: list[str] = []
        if extraction.unavailable:
            # enabled-but-unavailable connector: carry its notes forward from
            # the active knowledge DB instead of wiping the source (L28)
            present = {record.note_id for record in records}
            carried = carry_forward_records(
                knowledge_runtime.knowledge_database, extraction.unavailable
            )
            records.extend(
                record for record in carried if record.note_id not in present
            )
            degraded_reasons = [
                f"connector_unavailable:{name}" for name in extraction.unavailable
            ]
        build_report = knowledge_builder.build(
            records,
            knowledge_runtime.knowledge_database,
            replace_derived=True,
            on_stage=note,
            owner_aliases=profile_owner_aliases(index),
            degraded_reasons=degraded_reasons,
        )
        knowledge_runtime.activate_from_disk()
        return build_report

    def make_live_session(
        emit: EventSink,
        logger: SessionLogger,
        on_final_chunk: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> LiveSession:
        clock = _MonotonicClock()
        registry = EvidenceRegistry()
        grounding = GroundingEngine(registry)
        return LiveSession(
            registry=registry,
            ambient=AmbientRetriever(
                knowledge_runtime.make_engine("live", registry),
                clock,
                span_extractor=span_extractor,
                query_embedder=live_ambient.default_query_embedder,
                deadline_budget_s=config.LIVE_RETRIEVAL_DEADLINE_S,
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
            on_final_chunk=on_final_chunk,
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
        # collecting variant: the global sync hands its extracted records to
        # regenerate_knowledge so each source is read exactly once (L28)
        sync_all_enabled=lambda: connector_registry.sync_all_enabled_collecting(
            index, llm
        ),
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
    local_nodes = [node for node in graph.nodes if node.note_id in visited]
    present_clusters = {node.community for node in local_nodes if node.community >= 0}
    return LocalGraphResponse(
        center=center,
        depth=depth,
        nodes=local_nodes,
        edges=[
            edge
            for edge in graph.edges
            if edge.src in visited and edge.dst in visited
        ],
        clusters=[
            cluster for cluster in graph.clusters if cluster.id in present_clusters
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
    communities: Mapping[str, int] | None = getattr(snapshot, "note_clusters", None)
    cluster_rows = getattr(snapshot, "clusters", None)
    if communities is None or cluster_rows is None:
        # Snapshot without cluster data (legacy fake/pre-cluster runtime):
        # remap label propagation to the contiguous-cluster contract.
        communities, cluster_rows = fallback_clusters(snapshot.projection)
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
        clusters=[
            ClusterInfo(
                id=int(row.cluster_id),
                name=row.name,
                description=row.description,
                size=int(row.size),
            )
            for row in cluster_rows
        ],
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


@dataclass(frozen=True)
class _MeetingFinalizationSnapshot:
    meeting_id: str
    epoch: int
    notes_md: str
    transcript: list[dict[str, Any]]


class _MeetingTranscriptBuffer:
    FLUSH_CHUNKS = 20
    FLUSH_INTERVAL_S = 10.0

    def __init__(self, index: Any, meeting_id: str) -> None:
        self._index = index
        self._meeting_id = meeting_id
        self._chunks: list[dict[str, Any]] = []
        self._timer: asyncio.TimerHandle | None = None
        self._lock = asyncio.Lock()

    async def append(self, chunk: dict[str, Any]) -> None:
        async with self._lock:
            self._chunks.append(dict(chunk))
            if self._timer is None:
                self._timer = asyncio.get_running_loop().call_later(
                    self.FLUSH_INTERVAL_S, self._flush_from_timer
                )
            should_flush = len(self._chunks) >= self.FLUSH_CHUNKS
        if should_flush:
            await self.flush()

    def _flush_from_timer(self) -> None:
        self._timer = None
        asyncio.create_task(self.flush())

    async def flush(self) -> None:
        async with self._lock:
            timer = self._timer
            self._timer = None
            chunks = self._chunks
            self._chunks = []
        if timer is not None:
            timer.cancel()
        if not chunks:
            return
        try:
            self._index.meeting_append_transcript(self._meeting_id, chunks)
        except KeyError:
            return


def _make_live_session(
    runtime: Runtime,
    emit: EventSink,
    logger: Any,
    on_final_chunk: Callable[[dict[str, Any]], Awaitable[None] | None],
) -> LiveSessionLike:
    factory = runtime.make_live_session
    if factory is None:  # pragma: no cover - guarded by websocket setup
        raise RuntimeError("live session unavailable")
    parameters = inspect.signature(factory).parameters.values()
    supports_keyword = any(
        parameter.name == "on_final_chunk"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_keyword:
        return factory(emit, logger, on_final_chunk=on_final_chunk)  # type: ignore[call-arg]
    positional = sum(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )
    if positional >= 3:
        return factory(emit, logger, on_final_chunk)  # type: ignore[call-arg]
    return factory(emit, logger)


async def _teardown_session_context(context: SessionContext) -> None:
    live_close = getattr(context.live, "aclose", None)
    if live_close is not None:
        await live_close()
    if context.asr is not None:
        await context.asr.stop()


def _remove_active_context(runtime: Runtime, context: SessionContext) -> None:
    if context.meeting_id is not None:
        if runtime.active_sessions.get(context.meeting_id) is context:
            runtime.active_sessions.pop(context.meeting_id, None)
    else:
        runtime.active_unbound_sessions.pop(id(context), None)
    runtime.context_epochs.pop(id(context), None)
    runtime.context_transcript_buffers.pop(id(context), None)


async def _capture_finalization_snapshot(
    runtime: Runtime,
    context: SessionContext,
    transcript_buffer: _MeetingTranscriptBuffer | None,
) -> _MeetingFinalizationSnapshot | None:
    meeting_id = context.meeting_id
    if meeting_id is None:
        return None
    if transcript_buffer is not None:
        await transcript_buffer.flush()
    meeting = runtime.index.meeting_get(meeting_id)
    if meeting is None or meeting["status"] != "live":
        return None
    return _MeetingFinalizationSnapshot(
        meeting_id=meeting_id,
        epoch=runtime.context_epochs.get(id(context), -1),
        notes_md=meeting["notes_md"],
        transcript=list(meeting["transcript"]),
    )


async def _refresh_snapshot_transcript(
    runtime: Runtime,
    snapshot: _MeetingFinalizationSnapshot,
    transcript_buffer: _MeetingTranscriptBuffer | None,
) -> None:
    if transcript_buffer is not None:
        await transcript_buffer.flush()
    meeting = runtime.index.meeting_get(snapshot.meeting_id)
    if meeting is not None:
        snapshot.transcript[:] = meeting["transcript"]


def _finalize_meeting_snapshot(
    runtime: Runtime, snapshot: _MeetingFinalizationSnapshot
) -> bool:
    if runtime.meeting_epochs.get(snapshot.meeting_id) != snapshot.epoch:
        return False
    if snapshot.meeting_id in runtime.active_sessions:
        return False
    meeting = runtime.index.meeting_get(snapshot.meeting_id)
    if meeting is None or meeting["status"] == "done":
        return False
    runtime.index.meeting_set_status(
        snapshot.meeting_id, "done", ended_at=time.time()
    )
    runtime.updates.publish("meetings")
    return True


def _cancel_meeting_finalizer(runtime: Runtime, meeting_id: str) -> None:
    finalizer = runtime.meeting_finalizers.pop(meeting_id, None)
    if finalizer is not None:
        finalizer.cancel()


def _schedule_meeting_finalization(
    runtime: Runtime, snapshot: _MeetingFinalizationSnapshot
) -> None:
    if runtime.meeting_epochs.get(snapshot.meeting_id) != snapshot.epoch:
        return
    _cancel_meeting_finalizer(runtime, snapshot.meeting_id)
    handle: asyncio.TimerHandle | None = None

    def expire() -> None:
        current = runtime.meeting_finalizers.get(snapshot.meeting_id)
        if current is not handle:
            return
        runtime.meeting_finalizers.pop(snapshot.meeting_id, None)
        _finalize_meeting_snapshot(runtime, snapshot)

    handle = asyncio.get_running_loop().call_later(
        config.MEETING_FINALIZE_GRACE_S, expire
    )
    runtime.meeting_finalizers[snapshot.meeting_id] = handle


def split_audio_frame(frame: bytes) -> tuple[int, bytes]:
    if not frame:
        raise ValueError("audio frame missing channel byte")
    return frame[0], frame[1:]


async def _handle_control(
    raw: str,
    *,
    ctx_holder: list[SessionContext],
    transcript_buffer: list[_MeetingTranscriptBuffer | None],
    on_final_chunk: Callable[[dict[str, Any]], Awaitable[None] | None],
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
        runtime.pending_live_sessions.pop(id(ctx_holder), None)
        if message.meeting_id is not None:
            meeting = runtime.index.meeting_get(message.meeting_id)
            if meeting is None:
                await emit(_status_error(runtime, "meeting not found"))
                return session_started

        previous = ctx_holder[0]
        if previous.generation > 0:
            snapshot = None
            if previous.meeting_id is not None:
                snapshot = await _capture_finalization_snapshot(
                    runtime, previous, transcript_buffer[0]
                )
                if snapshot is not None:
                    _schedule_meeting_finalization(runtime, snapshot)
            await _teardown_session_context(previous)
            if snapshot is not None:
                await _refresh_snapshot_transcript(
                    runtime, snapshot, transcript_buffer[0]
                )
            _remove_active_context(runtime, previous)
        elif previous.live is not None or previous.asr is not None:
            await _teardown_session_context(previous)

        context = SessionContext(
            generation=previous.generation + 1,
            meeting_id=message.meeting_id,
        )
        ctx_holder[0] = context
        transcript_buffer[0] = (
            _MeetingTranscriptBuffer(runtime.index, message.meeting_id)
            if message.meeting_id is not None
            else None
        )
        context.live = _make_live_session(runtime, emit, logger, on_final_chunk)
        context.asr = runtime.make_asr_client(
            context.live.on_chunk,
            emit,
            lambda: _index_status(runtime),
        )

        if message.meeting_id is not None:
            meeting_id = message.meeting_id
            _cancel_meeting_finalizer(runtime, meeting_id)
            replaced = runtime.active_sessions.get(meeting_id)
            if replaced is not None and replaced is not context:
                replaced_buffer = runtime.context_transcript_buffers.get(id(replaced))
                await _teardown_session_context(replaced)
                if replaced_buffer is not None:
                    await replaced_buffer.flush()
                _remove_active_context(runtime, replaced)
            epoch = runtime.meeting_epochs.get(meeting_id, 0) + 1
            runtime.meeting_epochs[meeting_id] = epoch
            runtime.context_epochs[id(context)] = epoch
            if transcript_buffer[0] is not None:
                runtime.context_transcript_buffers[id(context)] = transcript_buffer[0]
            runtime.active_sessions.pop(meeting_id, None)
            runtime.active_sessions[meeting_id] = context
            meeting = runtime.index.meeting_get(meeting_id)
            if meeting is not None and meeting["status"] != "live":
                runtime.index.meeting_set_status(
                    meeting_id,
                    "live",
                    started_at=time.time() if meeting["started_at"] is None else None,
                )
                runtime.updates.publish("meetings")
        else:
            runtime.active_unbound_sessions[id(context)] = context

        if not session_started:
            logger.log("session_start", channels=message.channels)
            session_started = True
        channel_ids = [int(channel["id"]) for channel in message.channels if "id" in channel]
        await context.asr.start(channel_ids)
    elif isinstance(message, StopMsg):
        context = ctx_holder[0]
        if context.asr is not None:
            await context.asr.stop()
        if context.meeting_id is not None:
            context_epoch = runtime.context_epochs.get(id(context), -1)
            live_close = getattr(context.live, "aclose", None)
            if live_close is not None:
                await live_close()
            snapshot = await _capture_finalization_snapshot(
                runtime, context, transcript_buffer[0]
            )
            _remove_active_context(runtime, context)
            if runtime.meeting_epochs.get(context.meeting_id) == context_epoch:
                _cancel_meeting_finalizer(runtime, context.meeting_id)
                if snapshot is not None:
                    _finalize_meeting_snapshot(runtime, snapshot)
    elif isinstance(message, ActionableStatusMsg):
        live = ctx_holder[0].live
        if live is not None:
            await live.set_actionable_status(message.id, message.status)
    elif isinstance(message, NotepadMsg):
        context = ctx_holder[0]
        if message.text_md is None:
            return session_started
        if message.meeting_id != context.meeting_id:
            logging.getLogger("taurine.session").debug(
                "ignoring notepad update for unbound meeting %s",
                message.meeting_id,
            )
        elif context.live is not None:
            context.live.set_notepad(message.text_md)
    elif isinstance(message, OpenNoteMsg):
        try:
            await asyncio.to_thread(runtime.open_note, message.note_id)
        except Exception as exc:  # pragma: no cover - platform integration guard
            await emit(_status_error(runtime, str(exc)))
    return session_started


def _parse_control(
    raw: str,
) -> StartMsg | StopMsg | NotepadMsg | ActionableStatusMsg | OpenNoteMsg:
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
    if message_type == "notepad":
        return NotepadMsg.model_validate(payload)
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


def _suggestion_status_patch(
    stored: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(stored) != len(incoming):
        raise HTTPException(
            status_code=422, detail="suggestions patch may only change statuses"
        )
    for previous, row in zip(stored, incoming, strict=True):
        if (
            not isinstance(previous, dict)
            or not isinstance(row, dict)
            or not isinstance(previous.get("id"), str)
            or row.get("id") != previous["id"]
            or row.get("status") not in {
                "pending",
                "accepted",
                "rejected",
            }
        ):
            raise HTTPException(
                status_code=422, detail="suggestions patch may only change statuses"
            )
        if {key: value for key, value in row.items() if key != "status"} != {
            key: value for key, value in previous.items() if key != "status"
        }:
            raise HTTPException(
                status_code=422, detail="suggestions patch may only change statuses"
            )
    return incoming


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


def _accepts_parameter(func: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


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
        sync_result = await asyncio.to_thread(sync)
        check_cancel()
        if runtime.regenerate_knowledge is not None:
            _set_sync_status(runtime, SyncStatus(running=True, phase="indexing"))
            # single read per sync (L28): hand the records extracted above to
            # the rebuild when both sides support it (fakes may not)
            regenerate_args: tuple[Any, ...] = (indexing_progress,)
            if isinstance(sync_result, SourceExtraction) and _accepts_parameter(
                runtime.regenerate_knowledge, "extraction"
            ):
                regenerate_args = (indexing_progress, sync_result)
            await asyncio.to_thread(runtime.regenerate_knowledge, *regenerate_args)
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
    if "meetings" in kinds:
        meetings = getattr(runtime.index, "meetings_list", lambda: [])()
        await websocket.send_text(
            json.dumps({"type": "meetings", "meetings": meetings})
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
