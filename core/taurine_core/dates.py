"""Local, request-time date formatting shared by chat and live prompts."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def current_datetime_line(timestamp: float) -> str:
    local = datetime.fromtimestamp(float(timestamp)).astimezone()
    return local.strftime("Current date and time: %Y-%m-%d %H:%M %Z.")


def timestamp_date(value: object, *, unknown: str = "?") -> str:
    try:
        timestamp = float(value)  # type: ignore[arg-type]
        if timestamp > 0.0:
            return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d")
    except (OverflowError, OSError, TypeError, ValueError):
        pass
    return unknown


def evidence_date(passage: Any) -> str:
    passage_date = str(getattr(passage, "passage_date", "") or "").strip()
    if passage_date:
        return passage_date
    for value in (
        getattr(passage, "modified", None),
        getattr(passage, "created", None),
    ):
        rendered = timestamp_date(value, unknown="")
        if rendered:
            return rendered
    return "unknown-date"
