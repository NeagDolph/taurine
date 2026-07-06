from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO
from uuid import uuid4


SESSIONS_DIR = Path("sessions")

_console = logging.getLogger("taurine.session")
if not _console.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s │ %(message)s", datefmt="%H:%M:%S"))
    _console.addHandler(_handler)
    _console.setLevel(logging.INFO)
    _console.propagate = False


def _short(text: Any, limit: int = 56) -> str:
    value = str(text).replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _note_ref(value: Any) -> str:
    """Compact a note reference for console display (x-coredata ids → tail)."""
    text = str(value)
    if "://" in text:
        return text.rsplit("/", 1)[-1]
    return text


def _seq_range(seqs: list[Any]) -> str:
    if not seqs:
        return "#?"
    if len(seqs) == 1:
        return f"#{seqs[0]}"
    return f"#{seqs[0]}–{seqs[-1]}"


def _span_summary(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    spans: list[str] = []
    for row in rows[:2]:
        if isinstance(row, (list, tuple)) and row:
            spans.append(str(row[0]))
    if not spans:
        return ""
    return f" spans=[{_short('; '.join(spans), 40)}]"


def console_line(kind: str, payload: dict[str, Any]) -> str | None:
    """Lean one-line console rendering of a session-log record.

    Returns None for records that are noise at console granularity (interim
    chunks, routine slate adds, debounce/same-revision cadence blocks) — the
    JSONL file always has everything.
    """
    if kind == "session_start":
        return f"session started (channels={payload.get('channels')})"
    if kind == "session_end":
        return "session ended"
    if kind == "chunk":
        if not payload.get("final"):
            return None
        return f"#{payload.get('seq')} ch{payload.get('channel')} “{_short(payload.get('text'))}”"
    if kind == "asr_status":
        state = payload.get("state")
        detail = payload.get("detail") or ""
        return f"asr: {state}{f' ({_short(detail, 48)})' if state == 'error' and detail else ''}"
    if kind == "control":
        return f"control: {_short(payload.get('message'), 72)}"
    if kind == "c1_hit":
        target = payload.get("title") or _note_ref(payload.get("note_id"))
        return (
            f"c1 #{payload.get('seq')} ‘{payload.get('alias')}’ → "
            f"{payload.get('action')} “{_short(target, 40)}”"
        )
    if kind == "c2_pass":
        ordered = payload.get("ordered") or []
        expansions = payload.get("expansions") or []
        adds = payload.get("slate_adds") or []
        rerank = "rerank ok" if payload.get("rerank_used") else (
            f"rerank FAILED: {_short(payload.get('rerank_error'), 40)}"
            if payload.get("rerank_error")
            else "rerank off"
        )
        top = f" top={_note_ref(ordered[0][0])}" if ordered else ""
        spans = _span_summary(payload.get("gliner_spans"))
        return (
            f"c2 {_seq_range(payload.get('seqs') or [])} “{_short(payload.get('window'), 40)}” → "
            f"{len(ordered)} ranked, {len(expansions)} via graph, {rerank}, "
            f"+{len(adds)} slated{top}{spans}"
        )
    if kind == "slate":
        op = payload.get("op")
        if op in ("add", "renew"):
            return None  # summarized by the c1/c2 lines
        return (
            f"slate {op} {_note_ref(payload.get('note_id'))}"
            f" ({payload.get('reason') or payload.get('source') or ''}) rev={payload.get('revision')}"
        )
    if kind == "cadence":
        if payload.get("decision"):
            return f"cadence ✓ rev={payload.get('revision')}"
        reason = payload.get("reason")
        if reason in ("debounce", "same_revision"):
            return None  # constant chatter at chunk rate
        return f"cadence ✗ {reason} rev={payload.get('revision')}"
    if kind == "presenter":
        if payload.get("error"):
            return f"presenter ERROR: {_short(payload.get('error'), 64)}"
        picks = payload.get("picks") or []
        picked = ", ".join(_note_ref(p.get("note_id", "?")) for p in picks) if picks else "empty"
        return (
            f"presenter {payload.get('model_used')} {round(payload.get('latency_ms') or 0)}ms → "
            f"{picked}"
        )
    if kind == "pin":
        target = payload.get("title") or _note_ref(payload.get("note_id"))
        return (
            f"pin [{payload.get('provenance')}] “{_short(target, 40)}” — "
            f"{_short(payload.get('summary'), 48)}"
        )
    if kind == "unpin":
        return f"unpin ({payload.get('reason')})"
    return f"{kind}: {_short(payload, 80)}"


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
        self._label = path.stem[-8:] if path is not None else "nofile"

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
            if self._handle is not None:
                record = {"ts": time.time(), "kind": kind, **payload}
                self._handle.write(json.dumps(record, default=str) + "\n")
                self._handle.flush()
        except Exception:
            pass
        try:
            line = console_line(kind, payload)
            if line is not None:
                _console.info("%s │ %s", self._label, line)
        except Exception:
            return
