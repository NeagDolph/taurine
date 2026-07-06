from __future__ import annotations

import json
import time
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO
from uuid import uuid4


SESSIONS_DIR = Path("sessions")


class NullSessionLogger:
    path: Path | None = None

    def log(self, kind: str, **payload: Any) -> None:
        del kind, payload

    def close(self) -> None:
        return


class SessionLogger:
    def __init__(self, path: Path | None, handle: TextIO | None) -> None:
        self.path = path
        self._handle = handle
        self._closed = False

    @classmethod
    def create(cls, base_dir: Path = SESSIONS_DIR) -> SessionLogger:
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            path = base_dir / f"{session_id}.jsonl"
            return cls(path, path.open("a", encoding="utf-8"))
        except Exception:
            return cls(None, None)

    def log(self, kind: str, **payload: Any) -> None:
        if self._closed:
            return
        self._write(kind, payload)

    def close(self) -> None:
        if self._closed:
            return
        self._write("session_end", {})
        self._closed = True
        try:
            if self._handle is not None:
                self._handle.close()
        except Exception:
            return

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            if self._handle is None:
                return
            record = {"ts": time.time(), "kind": kind, **payload}
            self._handle.write(json.dumps(record, default=str) + "\n")
            self._handle.flush()
        except Exception:
            return
