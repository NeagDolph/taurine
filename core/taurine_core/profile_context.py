"""Bounded persisted user-profile context shared by runtime agents."""

from __future__ import annotations

from typing import Any

from . import config


def user_profile_block(index: Any) -> str:
    card = getattr(index, "user_card", lambda: None)()
    if not isinstance(card, str) or not card.strip():
        return ""
    bounded = card.strip()[: config.PROFILE_CONTEXT_MAX_CHARS].rstrip()
    return f"<user_profile>\n{bounded}\n</user_profile>"
