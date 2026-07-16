from __future__ import annotations

import asyncio
import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from taurine_core import config, server as server_module
from taurine_core.connectors.base import content_hash
from taurine_core.connectors.registry import ConnectorRegistry, build_connectors
from taurine_core.indexer.store import Index
from taurine_core.schemas import FinalSummaryEvent, SyncReport
from taurine_core.server import Runtime, create_app


def _connector_class() -> type[Any]:
    try:
        module = importlib.import_module("taurine_core.connectors.taurine_meetings")
    except ModuleNotFoundError:
        pytest.fail("Taurine meetings connector is missing")
    return module.TaurineMeetingsConnector


def _done_meeting(
    index: Index,
    *,
    title: str = "",
    notes_md: str = "",
    final_summary_md: str = "",
) -> dict[str, Any]:
    meeting = index.meeting_create(title)
    if notes_md:
        index.meeting_update(meeting["meeting_id"], notes_md=notes_md)
    if final_summary_md:
        current = index.meeting_get(meeting["meeting_id"])
        assert current is not None
        index.meeting_set_final(
            meeting["meeting_id"],
            final_summary_md=final_summary_md,
            suggestions=[],
            base_rev=current["notes_rev"],
        )
    index.meeting_set_status(meeting["meeting_id"], "done", ended_at=1.0)
    stored = index.meeting_get(meeting["meeting_id"])
    assert stored is not None
    return stored


def _runtime(index: Index, registry: ConnectorRegistry) -> Runtime:
    return Runtime(
        index=index,
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        connector_registry=registry,
        connector_llm=object(),
    )


def test_connector_extracts_exact_done_meeting_records(store: Index) -> None:
    TaurineMeetingsConnector = _connector_class()
    titled = _done_meeting(
        store,
        title="Roadmap Review",
        notes_md="Decide launch scope.",
        final_summary_md="Launch stays in July.",
    )
    untitled = _done_meeting(store, final_summary_md="A summary-only meeting.")
    _done_meeting(store)
    draft = store.meeting_create("Draft")
    store.meeting_update(draft["meeting_id"], notes_md="Not finished")

    records = TaurineMeetingsConnector(store).extract({})
    by_id = {record.note_id: record for record in records}

    titled_text = (
        "Decide launch scope.\n\n## Meeting summary\nLaunch stays in July."
    )
    titled_record = by_id[f"taurine:{titled['meeting_id']}"]
    assert titled_record.source == "taurine"
    assert titled_record.title == "Roadmap Review"
    assert titled_record.text == titled_text
    assert titled_record.modified == titled["updated_at"]
    assert titled_record.created == titled["created_at"]
    assert titled_record.content_hash == content_hash("Roadmap Review", titled_text)

    fallback_title = (
        f"Meeting {datetime.fromtimestamp(untitled['created_at']).date():%Y-%m-%d}"
    )
    summary_only_text = "\n\n## Meeting summary\nA summary-only meeting."
    fallback_record = by_id[f"taurine:{untitled['meeting_id']}"]
    assert fallback_record.title == fallback_title
    assert fallback_record.text == summary_only_text
    assert fallback_record.content_hash == content_hash(
        fallback_title, summary_only_text
    )
    assert len(records) == 2


def test_hidden_connector_is_force_enabled_in_every_registry_path(store: Index) -> None:
    TaurineMeetingsConnector = _connector_class()
    first = _done_meeting(store, notes_md="First meeting notes")
    second = _done_meeting(store, final_summary_md="Second meeting summary")
    connector = TaurineMeetingsConnector(store)
    registry = ConnectorRegistry([connector])

    store.set_connector_state("taurine", {"enabled": False})
    store.set_source_enabled("taurine", False)
    registry.reconcile_ledger(store)

    assert registry.is_effectively_enabled(connector, store.connector_state("taurine"))
    assert registry.list_info(store)[0].enabled is True
    assert "taurine" not in store.disabled_sources()

    collected = registry.sync_all_enabled_collecting(store, None)  # type: ignore[arg-type]
    assert [record.note_id for record in collected.records] == sorted(
        [f"taurine:{first['meeting_id']}", f"taurine:{second['meeting_id']}"]
    )
    assert [report.connector_id for report in collected.reports] == ["taurine"]

    store.meeting_delete(first["meeting_id"])
    registry.sync_all_enabled_collecting(store, None)  # type: ignore[arg-type]
    assert store.note(f"taurine:{first['meeting_id']}") is None
    assert store.note(f"taurine:{second['meeting_id']}") is not None

    store.set_connector_state("taurine", {"enabled": False})
    extracted = registry.extract_all_enabled(store)
    assert [record.note_id for record in extracted.records] == sorted(
        [f"taurine:{second['meeting_id']}"]
    )

    registry.set_enabled(store, "taurine", False)
    assert registry.list_info(store)[0].enabled is True
    assert "taurine" not in store.disabled_sources()


def test_runtime_catalog_factory_constructs_taurine_connector_with_index(
    store: Index,
) -> None:
    connectors = build_connectors(index=store)
    taurine = next(connector for connector in connectors if connector.connector_id == "taurine")

    assert taurine.hidden is True
    assert taurine._index is store
    assert config.TAURINE_SOURCE_ID == "taurine"
    assert config.SOURCE_RANK["taurine"] == 0.95
    assert config.RECENCY_HALF_LIFE_DAYS["taurine"] == 60
    assert config.MEETING_SYNC_DELAY_S == 60


@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        ("post", "/connectors/taurine/enable", None),
        ("post", "/connectors/taurine/disable", None),
        ("post", "/connectors/taurine/data/delete", None),
        ("post", "/connectors/taurine/sync", None),
        ("put", "/connectors/taurine/config", {}),
    ],
)
def test_hidden_connector_is_absent_and_all_mutations_404(
    store: Index, method: str, path: str, json: dict[str, Any] | None
) -> None:
    TaurineMeetingsConnector = _connector_class()
    registry = ConnectorRegistry([TaurineMeetingsConnector(store)])

    with TestClient(create_app(_runtime(store, registry))) as client:
        assert client.get("/connectors").json() == []
        response = getattr(client, method)(path, json=json)

    assert response.status_code == 404


def test_hidden_connector_is_absent_from_initial_and_refreshed_updates(
    store: Index,
) -> None:
    TaurineMeetingsConnector = _connector_class()
    registry = ConnectorRegistry([TaurineMeetingsConnector(store)])
    runtime = _runtime(store, registry)

    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect("/updates") as updates:
            initial = updates.receive_json()
            assert initial["type"] == "connectors"
            assert initial["connectors"] == []
            for _ in range(len(server_module.UpdateBus.KINDS) - 1):
                updates.receive_json()

            runtime.updates.publish("connectors")
            refreshed = updates.receive_json()
            assert refreshed["type"] == "connectors"
            assert refreshed["connectors"] == []


def test_enabled_connector_build_path_collects_done_meetings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from taurine_core.connectors import catalog
    from taurine_core.indexer import build as build_module
    from taurine_core.indexer.build import BuildDependencies

    monkeypatch.chdir(tmp_path)
    source_index = Index(tmp_path / "taurine.db")
    meeting = _done_meeting(source_index, notes_md="Captured by full build")
    for connector in catalog.CONNECTORS:
        source_index.set_connector_state(connector.connector_id, {"enabled": False})
    source_index.close()
    captured: list[Any] = []

    class RecordingCoordinator:
        def __init__(self, dependencies: Any) -> None:
            del dependencies

        def build(
            self,
            records: Any,
            target: str | Path,
            **kwargs: Any,
        ) -> Any:
            del kwargs
            captured.extend(records)
            return SimpleNamespace(
                target=Path(target), release_ready=True, release_blockers=()
            )

    monkeypatch.setattr(build_module, "BuildCoordinator", RecordingCoordinator)

    result = build_module.main(
        [
            "--db",
            str(tmp_path / "knowledge.db"),
            "--from-enabled-connectors",
            "--replace-derived",
            "--yes",
        ],
        dependencies=BuildDependencies(tokenizer=object(), embedder=lambda texts: []),
    )

    assert result == 0
    assert [record.note_id for record in captured] == [
        f"taurine:{meeting['meeting_id']}"
    ]


async def test_finalizations_schedule_one_coalesced_delayed_global_sync(
    store: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    meetings = [store.meeting_create("One"), store.meeting_create("Two")]
    for meeting in meetings:
        store.meeting_set_status(meeting["meeting_id"], "summarizing", ended_at=1.0)

    async def fake_run_final_summary(
        snapshot: dict[str, Any],
        llm: Any,
        execute_tool: Any,
        profile_block: str,
        emit: Any,
        index: Index,
    ) -> None:
        del llm, execute_tool, profile_block
        persisted = index.meeting_set_final(
            snapshot["meeting_id"],
            final_summary_md="Summary",
            suggestions=[],
            base_rev=snapshot["notes_rev"],
            title=snapshot["title"],
        )
        index.meeting_set_status(snapshot["meeting_id"], "done")
        await emit(
            FinalSummaryEvent(
                meeting_id=snapshot["meeting_id"],
                title=persisted["title"],
                final_summary_md="Summary",
                base_rev=snapshot["notes_rev"],
            )
        )

    monkeypatch.setattr(server_module, "run_final_summary", fake_run_final_summary)
    monkeypatch.setattr(config, "MEETING_SYNC_DELAY_S", 3600)
    runtime = _runtime(store, ConnectorRegistry([]))
    runtime.final_summary_llm = object()
    runtime.make_final_summary_tool_executor = lambda meeting_id: object()

    for meeting in meetings:
        snapshot = server_module._MeetingFinalizationSnapshot(
            meeting_id=meeting["meeting_id"],
            epoch=1,
            title=meeting["title"],
            notes_md=meeting["notes_md"],
            notes_rev=meeting["notes_rev"],
            transcript=[],
            last_live_summary={},
        )
        await server_module._run_meeting_finalization(runtime, snapshot, lambda event: None)

    pending = runtime.meeting_sync_task
    assert pending is not None
    assert not pending.done()
    await asyncio.sleep(0)
    assert runtime.meeting_sync_task is pending
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)


async def test_finalization_does_not_queue_sync_while_global_sync_is_running(
    store: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = store.meeting_create("Already syncing")
    store.meeting_set_status(meeting["meeting_id"], "summarizing", ended_at=1.0)

    async def fake_run_final_summary(
        snapshot: dict[str, Any],
        llm: Any,
        execute_tool: Any,
        profile_block: str,
        emit: Any,
        index: Index,
    ) -> None:
        del llm, execute_tool, profile_block, emit
        index.meeting_set_final(
            snapshot["meeting_id"],
            final_summary_md="Summary",
            suggestions=[],
            base_rev=0,
        )
        index.meeting_set_status(snapshot["meeting_id"], "done")

    monkeypatch.setattr(server_module, "run_final_summary", fake_run_final_summary)
    runtime = _runtime(store, ConnectorRegistry([]))
    runtime.final_summary_llm = object()
    runtime.make_final_summary_tool_executor = lambda meeting_id: object()
    runtime.sync_status = runtime.sync_status.model_copy(update={"running": True})
    snapshot = server_module._MeetingFinalizationSnapshot(
        meeting_id=meeting["meeting_id"],
        epoch=1,
        title=meeting["title"],
        notes_md="",
        notes_rev=0,
        transcript=[],
        last_live_summary={},
    )

    await server_module._run_meeting_finalization(runtime, snapshot, lambda event: None)

    assert runtime.meeting_sync_task is None


async def test_delayed_meeting_sync_runs_existing_global_sync_once(
    store: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    runtime = _runtime(store, ConnectorRegistry([]))
    runtime.sync_once = lambda: calls.append("sync") or SyncReport(0, 0, 0, 0)
    monkeypatch.setattr(config, "MEETING_SYNC_DELAY_S", 0)

    assert server_module._schedule_meeting_sync(runtime) is True
    assert server_module._schedule_meeting_sync(runtime) is False
    delayed = runtime.meeting_sync_task
    assert delayed is not None
    await delayed
    global_sync = runtime.sync_task
    assert global_sync is not None
    await global_sync

    assert calls == ["sync"]


async def test_delayed_meeting_sync_skips_if_sync_starts_during_wait(
    store: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    original_sleep = asyncio.sleep

    async def blocked_sleep(delay: float) -> None:
        del delay
        await release.wait()

    runtime = _runtime(store, ConnectorRegistry([]))
    monkeypatch.setattr(server_module.asyncio, "sleep", blocked_sleep)
    assert server_module._schedule_meeting_sync(runtime) is True
    delayed = runtime.meeting_sync_task
    assert delayed is not None
    await original_sleep(0)

    runtime.sync_status = runtime.sync_status.model_copy(update={"running": True})
    release.set()
    await delayed

    assert runtime.sync_task is None


async def test_empty_finalization_result_does_not_schedule_sync(
    store: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = store.meeting_create("Failed summary")
    store.meeting_set_status(meeting["meeting_id"], "summarizing", ended_at=1.0)

    async def fake_run_final_summary(
        snapshot: dict[str, Any],
        llm: Any,
        execute_tool: Any,
        profile_block: str,
        emit: Any,
        index: Index,
    ) -> None:
        del llm, execute_tool, profile_block, emit
        index.meeting_set_status(snapshot["meeting_id"], "done")

    monkeypatch.setattr(server_module, "run_final_summary", fake_run_final_summary)
    runtime = _runtime(store, ConnectorRegistry([]))
    runtime.final_summary_llm = object()
    runtime.make_final_summary_tool_executor = lambda meeting_id: object()
    snapshot = server_module._MeetingFinalizationSnapshot(
        meeting_id=meeting["meeting_id"],
        epoch=1,
        title=meeting["title"],
        notes_md="",
        notes_rev=0,
        transcript=[],
        last_live_summary={},
    )

    await server_module._run_meeting_finalization(runtime, snapshot, lambda event: None)

    assert runtime.meeting_sync_task is None


def test_patch_to_done_meeting_does_not_schedule_sync(store: Index) -> None:
    meeting = _done_meeting(store, notes_md="Before", final_summary_md="Summary")
    runtime = _runtime(store, ConnectorRegistry([]))

    with TestClient(create_app(runtime)) as client:
        response = client.patch(
            f"/meetings/{meeting['meeting_id']}", json={"notes_md": "After"}
        )

    assert response.status_code == 200
    assert runtime.meeting_sync_task is None
