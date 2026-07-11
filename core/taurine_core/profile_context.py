"""Bounded persisted user-profile context shared by runtime agents."""

from __future__ import annotations

import re
from typing import Any

from . import config


def user_profile_block(index: Any) -> str:
    card = getattr(index, "user_card", lambda: None)()
    if not isinstance(card, str) or not card.strip():
        return ""
    bounded = card.strip()[: config.PROFILE_CONTEXT_MAX_CHARS].rstrip()
    return f"<user_profile>\n{bounded}\n</user_profile>"


# generic heading words that are never a personal name
_NAME_STOPWORDS = frozenset(
    """user profile overview background summary about knowledge graph notes
    identity work projects interests education career people""".split()
)
_NAME_RE = re.compile(r"^([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})(?=$|[\s,:—–(-])")


_imessage_identity_cache: tuple[str, ...] | None = None


def _imessage_self_identity() -> tuple[str, ...]:
    """Owner names from the iMessage connector, cached per process (the
    lookup copies chat.db, so it must run at most once per server run)."""

    global _imessage_identity_cache
    if _imessage_identity_cache is None:
        try:
            from taurine_core.connectors.imessage import self_identity_names

            _imessage_identity_cache = tuple(self_identity_names())
        except Exception:
            _imessage_identity_cache = ()
    return _imessage_identity_cache


def owner_aliases(
    index: Any,
    *,
    connector_identity: Any = None,
) -> tuple[str, ...]:
    """Casefolded owner aliases for the entity resolver's hub guard (L18).

    Configured aliases (``config.OWNER_ALIASES``), a name parsed from the
    persisted user card's opening heading, and the iMessage connector's
    self-identity (item 3a) — each expanded with individual name tokens so
    first-name-only mentions are covered. ``connector_identity`` overrides
    the iMessage lookup (tests inject ``lambda: ()``). Empty when unknown.
    """

    aliases = {str(value) for value in config.OWNER_ALIASES}
    card = getattr(index, "user_card", lambda: None)()
    names = [_card_owner_name(card)] if isinstance(card, str) else []
    identity = connector_identity if connector_identity is not None else _imessage_self_identity
    names.extend(identity())
    for name in names:
        if not name:
            continue
        aliases.add(name)
        aliases.update(token for token in name.split() if len(token) >= 3)
    return tuple(sorted(alias.casefold() for alias in aliases))


def _card_owner_name(card: str) -> str | None:
    lines = [line.strip() for line in card.splitlines() if line.strip()][:8]
    for position, line in enumerate(lines):
        # only the opening line and markdown headings/emphasis can name the
        # owner — arbitrary prose sentences must not become aliases
        if position > 0 and not line.startswith(("#", "*", "_")):
            continue
        stripped = line.lstrip("#*_-• ").rstrip("*_ ").strip()
        match = _NAME_RE.match(stripped)
        if match is None:
            continue
        candidate = match.group(1)
        tokens = candidate.split()
        if any(token.casefold() in _NAME_STOPWORDS for token in tokens):
            continue
        if any(any(char.isdigit() for char in token) for token in tokens):
            continue
        return candidate
    return None
