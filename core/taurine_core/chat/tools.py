"""Stage B tool surface (spec chat §1.3) — definitions + executor.

Three tools, executed locally against the ChatRetriever/Index; every call
becomes a ChatActivityStep streamed to the UI. Tool responses are compact
JSON strings for the model; full payloads go to the session log.
"""
from __future__ import annotations

import json
from typing import Any

from .. import config
from ..indexer.store import Index
from ..schemas import ChatActivityStep
from .retrieval import ChatRetriever

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Search the user's notes with the full retrieval pipeline "
                "(semantic + keyword + graph traversal). Use a concrete, "
                "self-contained query — resolve pronouns yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": config.CHAT_SEARCH_TOOL_K},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Read a note's full text plus its graph facts and neighbors. Pass an [N#] id or note_id.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_note",
            "description": "List a note's graph neighbors with connection facts — what else relates to it. Pass an [N#] id or note_id.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    retriever: ChatRetriever,
    index: Index,
    *,
    now: float,
) -> tuple[str, ChatActivityStep]:
    """Run one tool; returns (JSON result for the model, UI activity step).

    Unknown tool / unresolvable note → an error JSON the model can read
    (never raises into the turn loop)."""
    if name == "search_notes":
        query = str(arguments.get("query") or "").strip()
        k = _int_arg(arguments.get("k"), config.CHAT_SEARCH_TOOL_K)
        rows = retriever.search(query, k, now=now)
        payload = {
            "rows": [
                {
                    "short_id": item.short_id,
                    "title": item.title,
                    "source": item.source,
                    "age_days": round(float(item.age_days), 1),
                    "score": round(float(item.score), 3),
                    "gist": item.gist,
                }
                for item in rows
            ]
        }
        return _json(payload), ChatActivityStep(
            kind="search",
            label=f'Searched "{query}" → {len(rows)} notes',
            note_ids=[item.note_id for item in rows],
        )

    if name == "read_note":
        ref = str(arguments.get("note") or "").strip()
        note_id = retriever.resolve(ref)
        if note_id is None:
            return _failure(f"could not resolve note: {ref}", kind="read")
        note = index.note(note_id)
        if note is None:
            return _failure(f"could not resolve note: {ref}", kind="read")
        neighbors = []
        for other_id, weight, fact in index.neighbor_edges(note_id):
            other = index.note(other_id)
            if other is None:
                continue
            neighbors.append(
                {
                    "note_id": other.note_id,
                    "title": other.title,
                    "source": other.source,
                    "weight": round(float(weight), 3),
                    "fact": fact,
                }
            )
        return _json(
            {
                "note_id": note.note_id,
                "title": note.title,
                "source": note.source,
                "age_days": _age_days(note.modified, now),
                "text": note.text[: config.CHAT_NOTE_READ_MAX_CHARS],
                "neighbors": neighbors,
            }
        ), ChatActivityStep(kind="read", label=f"Read: {note.title}", note_ids=[note.note_id])

    if name == "expand_note":
        ref = str(arguments.get("note") or "").strip()
        note_id = retriever.resolve(ref)
        if note_id is None:
            return _failure(f"could not resolve note: {ref}", kind="expand")
        note = index.note(note_id)
        if note is None:
            return _failure(f"could not resolve note: {ref}", kind="expand")
        edge_types = _edge_types(index)
        neighbors = []
        for other_id, weight, fact in index.neighbor_edges(note_id)[:12]:
            other = index.note(other_id)
            if other is None:
                continue
            key = tuple(sorted((note_id, other_id)))
            neighbors.append(
                {
                    "note_id": other.note_id,
                    "title": other.title,
                    "source": other.source,
                    "weight": round(float(weight), 3),
                    "type": edge_types.get(key, "semantic"),
                    "fact": fact,
                }
            )
        return _json({"neighbors": neighbors}), ChatActivityStep(
            kind="expand",
            label=f"Expanded: {note.title}",
            note_ids=[note.note_id],
        )

    return _failure(f"unknown tool: {name}", kind="search")


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _failure(message: str, *, kind: str) -> tuple[str, ChatActivityStep]:
    return _json({"error": message}), ChatActivityStep(
        kind=kind,  # type: ignore[arg-type]
        label=f"Failed: {message}",
        note_ids=[],
    )


def _int_arg(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _age_days(modified: float, now: float) -> float:
    return round(max(0.0, (float(now) - float(modified)) / 86_400.0), 1)


def _edge_types(index: Index) -> dict[tuple[str, str], str]:
    try:
        graph = index.graph()
    except Exception:
        return {}
    return {
        tuple(sorted((edge.src, edge.dst))): edge.type
        for edge in getattr(graph, "edges", [])
    }
