from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO
from uuid import uuid4


SESSIONS_DIR = Path("sessions")
KIND_GROUNDING = "grounding"
KIND_SUMMARY = "summary"

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
    chunks) — the JSONL file always has everything.
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
        grounding_ids = payload.get("grounding_ids") or []
        ppr_top = payload.get("ppr_top") or []
        top = f" top={_note_ref(grounding_ids[0])}" if grounding_ids else ""
        spans = _span_summary(payload.get("gliner_spans"))
        return (
            f"c2 {_seq_range(payload.get('seqs') or [])} “{_short(payload.get('window'), 40)}” → "
            f"{len(grounding_ids)} grounded, {len(ppr_top)} ppr{top}{spans}"
        )
    if kind == KIND_GROUNDING:
        mark = "✓" if payload.get("trigger_decision") else "✗"
        items = payload.get("items") or []
        first = ""
        if items:
            row = items[0]
            if isinstance(row, dict):
                first = f" top={row.get('short_id') or _note_ref(row.get('note_id'))}"
        return (
            f"grounding {mark} {payload.get('trigger_reason')} "
            f"rev={payload.get('revision')}{first} facts={len(payload.get('fact_ids') or [])}"
        )
    if kind == KIND_SUMMARY:
        if payload.get("rejected"):
            return (
                f"summary rejected {payload.get('reject_reason')} "
                f"{round(payload.get('latency_ms') or 0)}ms"
            )
        if payload.get("no_update"):
            return f"summary no_update {round(payload.get('latency_ms') or 0)}ms"
        accepted = payload.get("accepted_actionable_ids") or []
        return (
            f"summary {payload.get('model')} {round(payload.get('latency_ms') or 0)}ms "
            f"actions={len(accepted)}"
        )
    if kind == "chat_retrieval":
        top = payload.get("top") or []
        top_ref = ""
        if top and isinstance(top[0], dict):
            top_ref = f" top={top[0].get('id')}"
        facts = payload.get("facts") or []
        return (
            f"chat {payload.get('chat_id')} retrieval “{_short(payload.get('query'), 40)}” → "
            f"{len(top)} notes, {len(facts)} facts{top_ref}"
        )
    if kind == "chat_turn":
        usage = payload.get("usage") or {}
        tools = payload.get("tools") or []
        cancelled = " cancelled" if payload.get("cancelled") else ""
        stripped = payload.get("stripped_citations") or 0
        return (
            f"chat {payload.get('chat_id')} msg={payload.get('message_id')} {payload.get('model')} "
            f"{round(payload.get('total_ms') or 0)}ms tools={len(tools)} "
            f"usage={usage.get('prompt', 0)}/{usage.get('completion', 0)}+{usage.get('reasoning', 0)} "
            f"stripped={stripped}{cancelled}"
        )
    return f"{kind}: {_short(payload, 80)}"


class NullSessionLogger:
    path: Path | None = None

    def log(self, kind: str, **payload: Any) -> None:
        del kind, payload

    def close(self) -> None:
        return


def log_grounding(
    logger: Any,
    update: Any,
    *,
    decision: bool,
    reason: str,
) -> None:
    logger.log(
        KIND_GROUNDING,
        revision=getattr(update, "revision", 0),
        items=[
            {
                "note_id": getattr(item, "note_id", ""),
                "short_id": getattr(item, "short_id", ""),
                "score": getattr(item, "score", 0.0),
            }
            for item in getattr(update, "items", [])
        ],
        fact_ids=[
            getattr(fact, "fact_id", "")
            for fact in getattr(update, "facts", [])
        ],
        trigger_decision=decision,
        trigger_reason=reason,
    )


def log_summary(
    logger: Any,
    *,
    model: str,
    latency_ms: float,
    no_update: bool,
    rejected: bool,
    reject_reason: str,
    citation_stats: dict[str, Any],
    accepted_actionable_ids: list[str],
) -> None:
    logger.log(
        KIND_SUMMARY,
        model=model,
        latency_ms=latency_ms,
        no_update=no_update,
        rejected=rejected,
        reject_reason=reject_reason,
        citation_stats=citation_stats,
        accepted_actionable_ids=accepted_actionable_ids,
    )


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
