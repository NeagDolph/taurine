from __future__ import annotations

import importlib
import time
from collections.abc import Iterable
from typing import Any

from taurine_core.ports import ConnectorError, ConnectorPort, LLMPort
from taurine_core.schemas import ConnectorInfo, ConnectorSyncReport

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

    def list_info(self, index: object) -> list[ConnectorInfo]:
        counts = index.note_count_by_source()  # type: ignore[attr-defined]
        return [
            self._info_for(index, connector, int(counts.get(connector.connector_id, 0)))
            for connector in self._connectors
        ]

    def set_enabled(
        self,
        index: object,
        connector_id: str,
        enabled: bool,
        *,
        purge: bool = False,
    ) -> None:
        self._connector(connector_id)
        state = self._state(index, connector_id)
        state["enabled"] = bool(enabled)
        self._save_state(index, connector_id, state)
        if purge:
            for note_id in sorted(index.note_ids(connector_id)):  # type: ignore[attr-defined]
                index.delete_note(note_id)  # type: ignore[attr-defined]

    def set_config(self, index: object, connector_id: str, config: dict) -> None:
        self._connector(connector_id)
        state = self._state(index, connector_id)
        state["config"] = dict(config)
        self._save_state(index, connector_id, state)

    def sync_connector(
        self,
        connector_id: str,
        index: object,
        llm: LLMPort,
    ) -> ConnectorSyncReport:
        connector = self._connector(connector_id)
        state = self._state(index, connector_id)
        self.syncing.add(connector_id)
        try:
            available, detail = connector.available()
            if not available:
                report = ConnectorSyncReport(
                    connector_id=connector_id,
                    error=detail or "connector unavailable",
                )
            else:
                try:
                    records = connector.extract(self._config(connector, state))
                    from taurine_core.indexer.sync import sync_source

                    report = sync_source(index, connector_id, records, llm)  # type: ignore[arg-type]
                except ConnectorError as exc:
                    report = ConnectorSyncReport(connector_id=connector_id, error=str(exc))
                except Exception as exc:  # any layer failing must surface, not vanish
                    report = ConnectorSyncReport(
                        connector_id=connector_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        finally:
            self.syncing.discard(connector_id)

        state["last_sync"] = time.time()
        state["last_report"] = report.model_dump()
        self._save_state(index, connector_id, state)
        return report

    def sync_all_enabled(self, index: object, llm: LLMPort) -> list[ConnectorSyncReport]:
        reports: list[ConnectorSyncReport] = []
        for connector in self._connectors:
            if self._enabled(connector.connector_id, self._state(index, connector.connector_id)):
                reports.append(self.sync_connector(connector.connector_id, index, llm))
        return reports

    def _info_for(self, index: object, connector: ConnectorPort, docs: int) -> ConnectorInfo:
        state = self._state(index, connector.connector_id)
        try:
            available, detail = connector.available()
        except Exception as exc:
            available, detail = False, str(exc)
        last_error = str((state.get("last_report") or {}).get("error") or "")
        if available and not detail and last_error:
            detail = f"last sync failed: {last_error}"
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
