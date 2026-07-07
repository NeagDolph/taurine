from __future__ import annotations

from collections.abc import Callable
from typing import Any

from taurine_core.indexer.notes import NotesAccessError, extract_notes
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord


class AppleNotesConnector:
    connector_id = "apple_notes"
    label = "Apple Notes"
    kind = "local"

    def __init__(
        self,
        *,
        app_factory: Callable[[], object] | None = None,
        extractor: Callable[..., list[NoteRecord]] = extract_notes,
    ) -> None:
        self._app_factory = app_factory
        self._extractor = extractor

    def available(self) -> tuple[bool, str]:
        try:
            app = self._make_app()
            account = _value(getattr(app, "default_account", None))
            if isinstance(account, str) and hasattr(app, "account"):
                account = app.account(account)
            if account is None:
                return False, "Apple Notes default account is unavailable"
            return True, ""
        except ModuleNotFoundError:
            return False, "Install macnotesapp to read Apple Notes"
        except Exception as exc:
            return False, str(exc)

    def default_config(self) -> dict:
        return {}

    def extract(self, config: dict) -> list[NoteRecord]:
        del config
        try:
            app = self._make_app() if self._app_factory is not None else None
            records = self._extractor(app) if app is not None else self._extractor()
        except NotesAccessError as exc:
            raise ConnectorError(str(exc)) from exc
        return [
            record.model_copy(update={"source": self.connector_id})
            for record in records
        ]

    def _make_app(self) -> object:
        if self._app_factory is not None:
            return self._app_factory()
        from macnotesapp import NotesApp

        return NotesApp()


def _value(value: Any) -> Any:
    return value() if callable(value) else value
