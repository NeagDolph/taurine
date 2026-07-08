"""Stage B tool surface (spec chat §1.3) — definitions + executor.

Three tools, executed locally against the ChatRetriever/Index; every call
becomes a ChatActivityStep streamed to the UI. Tool responses are compact
JSON strings for the model; full payloads go to the session log.
"""
from __future__ import annotations

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
    raise NotImplementedError  # Stage B
