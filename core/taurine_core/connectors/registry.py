from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Iterable
from typing import Any

from taurine_core.ports import ConnectorError, ConnectorPort, LLMPort
from taurine_core.schemas import ConnectorInfo, ConnectorSyncReport
from taurine_core.schemas import NoteRecord

from .apple_notes import AppleNotesConnector


def build_connectors() -> list[ConnectorPort]:
    try:
        module = importlib.import_module("taurine_core.connectors.catalog")
    except ModuleNotFoundError as exc:
        if exc.name not in (None, "taurine_core.connectors.catalog") and str(exc) != (
            "taurine_core.connectors.catalog"
        ):
            raise
        return [AppleNotesConnector()]
    connectors = getattr(module, "CONNECTORS")
    return [_materialize(connector) for connector in connectors]


class ConnectorRegistry:
    def __init__(self, connectors: Iterable[ConnectorPort] | None = None) -> None:
        self._connectors = list(connectors) if connectors is not None else build_connectors()
        self._by_id = {connector.connector_id: connector for connector in self._connectors}
        self.syncing: set[str] = set()
        self.sync_progress: dict[str, tuple[str, int, int]] = {}
        # server hook, called (from any thread) whenever connector state or
        # sync progress changes so the /updates websocket can broadcast
        self.on_change: Callable[[], None] | None = None

    def list_info(self, index: object) -> list[ConnectorInfo]:
        counts = index.note_count_by_source()  # type: ignore[attr-defined]
        return [
            self._info_for(index, connector, int(counts.get(connector.connector_id, 0)))
            for connector in self._connectors
        ]

    def set_enabled(self, index: object, connector_id: str, enabled: bool) -> None:
        """Enable/disable the connector. Data is always kept: the index ledger
        just deactivates the source. purge_data() is the explicit delete."""
        self._connector(connector_id)
        state = self._state(index, connector_id)
        state["enabled"] = bool(enabled)
        self._save_state(index, connector_id, state)
        set_source_enabled = getattr(index, "set_source_enabled", None)
        if callable(set_source_enabled):
            set_source_enabled(connector_id, bool(enabled))
        self._notify()

    def purge_data(self, index: object, connector_id: str) -> int:
        self._connector(connector_id)
        note_ids = sorted(index.note_ids(connector_id))  # type: ignore[attr-defined]
        for note_id in note_ids:
            index.delete_note(note_id)  # type: ignore[attr-defined]
        self._notify()
        return len(note_ids)

    def reconcile_ledger(self, index: object) -> None:
        """Align the index's source ledger with persisted enabled flags (covers
        DBs written before the ledger existed)."""
        set_source_enabled = getattr(index, "set_source_enabled", None)
        if not callable(set_source_enabled):
            return
        for connector in self._connectors:
            state = self._state(index, connector.connector_id)
            set_source_enabled(
                connector.connector_id,
                self._enabled(connector.connector_id, state),
            )

    def set_config(self, index: object, connector_id: str, config: dict) -> None:
        self._connector(connector_id)
        state = self._state(index, connector_id)
        state["config"] = dict(config)
        self._save_state(index, connector_id, state)
        self._notify()

    def sync_connector(
        self,
        connector_id: str,
        index: object,
        llm: LLMPort,
    ) -> ConnectorSyncReport:
        connector = self._connector(connector_id)
        del llm
        state = self._state(index, connector_id)
        self.syncing.add(connector_id)
        self.sync_progress[connector_id] = ("extracting", 0, 0)
        self._notify()
        try:
            available, detail = connector.available()
            if not available:
                report = ConnectorSyncReport(
                    connector_id=connector_id,
                    error=detail or "connector unavailable",
                )
            else:
                try:
                    records = list(connector.extract(self._config(connector, state)))
                    self._set_progress(connector_id, "source_cache", 0, len(records))
                    added, changed, removed = index.replace_source_records(  # type: ignore[attr-defined]
                        connector_id,
                        records,
                    )
                    report = ConnectorSyncReport(
                        connector_id=connector_id,
                        added=added,
                        changed=changed,
                        removed=removed,
                        enrich_failures=0,
                    )
                    self._set_progress(
                        connector_id, "source_cache", len(records), len(records)
                    )
                    salience_scores = getattr(connector, "ingest_salience_scores", None)
                    set_ingest_salience = getattr(index, "set_ingest_salience", None)
                    ingest_salience = getattr(index, "ingest_salience", None)
                    if isinstance(salience_scores, dict) and salience_scores and callable(
                        set_ingest_salience
                    ):
                        merged_scores = (
                            dict(ingest_salience()) if callable(ingest_salience) else {}
                        )
                        merged_scores.update(salience_scores)
                        set_ingest_salience(merged_scores)
                except ConnectorError as exc:
                    report = ConnectorSyncReport(connector_id=connector_id, error=str(exc))
                except Exception as exc:  # any layer failing must surface, not vanish
                    report = ConnectorSyncReport(
                        connector_id=connector_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        finally:
            self.syncing.discard(connector_id)
            self.sync_progress.pop(connector_id, None)

        state["last_sync"] = time.time()
        state["last_report"] = report.model_dump()
        self._save_state(index, connector_id, state)
        self._notify()
        return report

    def _set_progress(self, connector_id: str, phase: str, done: int, total: int) -> None:
        self.sync_progress[connector_id] = (phase, done, total)
        self._notify()

    def _notify(self) -> None:
        callback = self.on_change
        if callback is not None:
            try:
                callback()
            except Exception:  # notification must never break connector work
                pass

    def sync_all_enabled(self, index: object, llm: LLMPort) -> list[ConnectorSyncReport]:
        reports: list[ConnectorSyncReport] = []
        for connector in self._connectors:
            if self._enabled(connector.connector_id, self._state(index, connector.connector_id)):
                reports.append(self.sync_connector(connector.connector_id, index, llm))
        return reports

    def extract_all_enabled(self, index: object) -> list[NoteRecord]:
        """Read one clean source snapshot without writing derived index state."""

        records: list[NoteRecord] = []
        for connector in self._connectors:
            state = self._state(index, connector.connector_id)
            if not self._enabled(connector.connector_id, state):
                continue
            available, detail = connector.available()
            if not available:
                raise ConnectorError(
                    detail or f"connector {connector.connector_id} is unavailable"
                )
            records.extend(connector.extract(self._config(connector, state)))
        return sorted(records, key=lambda row: (row.source, row.note_id))

    def _info_for(self, index: object, connector: ConnectorPort, docs: int) -> ConnectorInfo:
        state = self._state(index, connector.connector_id)
        try:
            available, detail = connector.available()
        except Exception as exc:
            available, detail = False, str(exc)
        last_error = str((state.get("last_report") or {}).get("error") or "")
        if available and not detail and last_error:
            detail = f"last sync failed: {last_error}"
        phase, done, total = self.sync_progress.get(connector.connector_id, ("", 0, 0))
        return ConnectorInfo(
            connector_id=connector.connector_id,
            label=connector.label,
            kind=connector.kind,  # type: ignore[arg-type]
            enabled=self._enabled(connector.connector_id, state),
            available=available,
            detail=detail,
            docs=docs,
            last_sync=_float_or_none(state.get("last_sync")),
            syncing=connector.connector_id in self.syncing,
            sync_phase=phase,
            sync_done=done,
            sync_total=total,
            config=self._config(connector, state),
        )

    def _connector(self, connector_id: str) -> ConnectorPort:
        try:
            return self._by_id[connector_id]
        except KeyError as exc:
            raise KeyError(f"unknown connector: {connector_id}") from exc

    @staticmethod
    def _state(index: object, connector_id: str) -> dict:
        return dict(index.connector_state(connector_id))  # type: ignore[attr-defined]

    @staticmethod
    def _save_state(index: object, connector_id: str, state: dict) -> None:
        index.set_connector_state(connector_id, state)  # type: ignore[attr-defined]

    @staticmethod
    def _enabled(connector_id: str, state: dict) -> bool:
        default = connector_id == "apple_notes"
        return bool(state.get("enabled", default))

    @staticmethod
    def _config(connector: ConnectorPort, state: dict) -> dict:
        config = connector.default_config()
        stored = state.get("config")
        if isinstance(stored, dict):
            config.update(stored)
        return config


def _materialize(connector: Any) -> ConnectorPort:
    if isinstance(connector, type):
        return connector()
    return connector


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
