"""Evidence-native chat tool definitions and local executor."""

from __future__ import annotations

import json
from typing import Any

from taurine_core import config
from taurine_core.retrieval.types import EvidencePack
from taurine_core.schemas import ChatActivityStep, EvidencePackPayload

from .retrieval import ChatRetriever


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "Search source passages, directed facts, and explanatory paths. Use a self-contained query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": config.CHAT_SEARCH_TOOL_K},
                    "temporal": {
                        "type": "string",
                        "enum": ["current", "historical", "any"],
                        "default": "any",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Read query-relevant registered passages from a note. Follow next_cursor when truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Registered N# identifier"},
                    "query": {"type": "string"},
                    "cursor": {"type": "string"},
                    "max_chars": {"type": "integer", "default": config.CHAT_NOTE_READ_MAX_CHARS},
                },
                "required": ["note", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_evidence",
            "description": "Traverse directed graph connections from registered E#, F#, or N# evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["outgoing", "incoming", "both"],
                        "default": "both",
                    },
                    "max_hops": {"type": "integer", "default": 2},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["evidence"],
            },
        },
    },
]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    retriever: ChatRetriever,
    *,
    now: float,
) -> tuple[str, ChatActivityStep]:
    """Execute a bounded tool call; model-visible failures never escape the turn."""

    try:
        if name == "search_evidence":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return _failure("query is required", kind="search")
            k = _int_arg(arguments.get("k"), config.CHAT_SEARCH_TOOL_K, maximum=24)
            temporal = str(arguments.get("temporal") or "any")
            if temporal not in {"current", "historical", "any"}:
                return _failure("invalid temporal filter", kind="search")
            pack = retriever.search_evidence(
                query,
                k=k,
                temporal=temporal,  # type: ignore[arg-type]
                now=now,
            )
            return render_search_result(pack, query)

        if name == "read_note":
            note_ref = str(arguments.get("note") or "").strip()
            query = str(arguments.get("query") or "").strip()
            if not note_ref:
                return _failure("note is required", kind="read")
            if not query:
                return _failure("query is required", kind="read")
            cursor_value = arguments.get("cursor")
            cursor = str(cursor_value) if cursor_value else None
            max_chars = _int_arg(
                arguments.get("max_chars"),
                config.CHAT_NOTE_READ_MAX_CHARS,
                maximum=config.CHAT_NOTE_READ_MAX_CHARS,
            )
            payload = retriever.read_note(
                note_ref,
                query=query,
                cursor=cursor,
                max_chars=max_chars,
            )
            return _json(payload), ChatActivityStep(
                kind="read",
                label=f'Read {note_ref}: {len(payload.get("passages", []))} passages',
                note_ids=_resolved_note_ids(retriever, (note_ref,)),
            )

        if name == "expand_evidence":
            evidence = str(arguments.get("evidence") or "").strip()
            if not evidence:
                return _failure("evidence is required", kind="expand")
            direction = str(arguments.get("direction") or "both")
            if direction not in {"outgoing", "incoming", "both"}:
                return _failure("invalid expansion direction", kind="expand")
            max_hops = _int_arg(arguments.get("max_hops"), 2, maximum=4)
            limit = _int_arg(arguments.get("limit"), 8, maximum=24)
            payload = retriever.expand_evidence(
                evidence,
                direction=direction,  # type: ignore[arg-type]
                max_hops=max_hops,
                limit=limit,
            )
            note_refs = tuple(
                str(row.get("note") or "")
                for row in payload.get("results", [])
                if isinstance(row, dict)
            )
            return _json(payload), ChatActivityStep(
                kind="expand",
                label=f'Expanded {evidence} → {len(payload.get("results", []))} passages',
                note_ids=_resolved_note_ids(retriever, note_refs),
            )
    except (KeyError, ValueError) as exc:
        return _failure(str(exc), kind=_tool_kind(name))

    return _failure(f"unknown tool: {name}", kind="search")


def render_search_result(
    pack: EvidencePack,
    query: str,
) -> tuple[str, ChatActivityStep]:
    """Render search output for both executed and speculative tool results."""
    # §0.1: fallback packs carry decision.fallback=True and every passage row
    # is marked low_confidence in the payload.
    payload = EvidencePackPayload.from_domain(pack).model_dump(mode="json")
    if pack.decision.accepted:
        label = f'Searched "{query}" → {len(pack.passages)} passages'
    elif getattr(pack.decision, "fallback", False):
        label = f'Searched "{query}" → {len(pack.passages)} low-confidence passages'
    else:
        label = f'Searched "{query}" → no supported evidence'
    return _json(payload), ChatActivityStep(
        kind="search",
        label=label,
        note_ids=[row.note_id for row in pack.notes],
    )


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _failure(message: str, *, kind: str) -> tuple[str, ChatActivityStep]:
    return _json({"error": message}), ChatActivityStep(
        kind=kind,  # type: ignore[arg-type]
        label=f"Failed: {message}",
        note_ids=[],
    )


def _int_arg(value: Any, default: int, *, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        parsed = default
    return min(parsed, maximum)


def _tool_kind(name: str) -> str:
    if name == "read_note":
        return "read"
    if name == "expand_evidence":
        return "expand"
    return "search"


def _resolved_note_ids(retriever: ChatRetriever, refs: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for ref in refs:
        note_id = retriever.resolve_note(ref)
        if note_id is not None and note_id not in values:
            values.append(note_id)
    return values
