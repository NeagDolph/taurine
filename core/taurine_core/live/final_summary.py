"""End-of-meeting tool-agent summary and additive note suggestions."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

from taurine_core import config
from taurine_core.chat.tools import TOOL_DEFINITIONS
from taurine_core.llm_agent import run_tool_agent
from taurine_core.schemas import (
    FinalSummaryEvent,
    FinalSummaryStatusEvent,
    SuggestedEdit,
)


FINAL_SUMMARY_SCHEMA: dict[str, Any] = {
    "title": "final_summary",
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": 120},
        "final_summary_md": {"type": "string", "minLength": 1, "maxLength": 40000},
        "suggested_edits": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "anchor": {"type": "string", "maxLength": 8000},
                    "occurrence": {"type": "integer", "minimum": 1},
                    "insert_md": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "rationale": {"type": "string", "maxLength": 2000},
                },
                "required": ["anchor", "occurrence", "insert_md", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "final_summary_md", "suggested_edits"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You produce the polished final record of a meeting.
Use the knowledge-base tools when they can resolve names, projects, history, or related facts.
The transcript labels channel 0 as You (the user) and channel 1 as Them (the other side).
Return a concise title, a polished markdown final summary, and at most 10 suggested note edits.
Suggested edits are additive insertions or expansions of the user's own notes, never rewrites.
Each non-empty anchor must be an exact substring from the supplied notes and occurrence is 1-based.
Use anchor "" to append at the end. Do not invent commitments or overwrite the user's wording.
The final response must match the supplied JSON schema exactly."""


async def run_final_summary(
    meeting: dict[str, Any],
    llm: Any,
    execute_tool: Callable[[str, dict[str, Any]], Awaitable[str]],
    profile_block: str,
    emit: Callable[[Any], Awaitable[None]],
    index: Any,
) -> None:
    """Generate, validate, persist, and best-effort emit one final summary."""

    meeting_id = str(meeting.get("meeting_id") or "")
    await _safe_emit(
        emit,
        FinalSummaryStatusEvent(meeting_id=meeting_id, state="running"),
    )
    try:
        result = await run_tool_agent(
            llm,
            model=config.FINAL_SUMMARY_MODEL,
            system=_SYSTEM_PROMPT,
            user=_user_prompt(meeting, profile_block),
            tools=TOOL_DEFINITIONS,
            execute_tool=execute_tool,
            max_tool_calls=config.FINAL_SUMMARY_MAX_TOOL_CALLS,
            timeout_s=config.FINAL_SUMMARY_TIMEOUT_S,
            schema=FINAL_SUMMARY_SCHEMA,
            max_tokens=config.FINAL_SUMMARY_MAX_TOKENS,
        )
        title, final_summary_md, edits = _validated_result(meeting, result)
        base_rev = int(meeting.get("notes_rev") or 0)
        persisted = index.meeting_set_final(
            meeting_id,
            final_summary_md=final_summary_md,
            suggestions=[row.model_dump(mode="json") for row in edits],
            base_rev=base_rev,
            title=title,
        )
        index.meeting_set_status(meeting_id, "done")
        await _safe_emit(
            emit,
            FinalSummaryEvent(
                meeting_id=meeting_id,
                title=str(persisted["title"]),
                final_summary_md=final_summary_md,
                base_rev=base_rev,
                suggested_edits=edits,
            ),
        )
        await _safe_emit(
            emit,
            FinalSummaryStatusEvent(meeting_id=meeting_id, state="done"),
        )
    except Exception as exc:
        index.meeting_set_status(meeting_id, "done")
        await _safe_emit(
            emit,
            FinalSummaryStatusEvent(
                meeting_id=meeting_id,
                state="error",
                detail=(str(exc).strip() or type(exc).__name__)[:300],
            ),
        )


def _user_prompt(meeting: Mapping[str, Any], profile_block: str) -> str:
    notes_md = str(meeting.get("notes_md") or "")
    notes_rev = int(meeting.get("notes_rev") or 0)
    transcript = _render_transcript(meeting.get("transcript"))
    live_summary = meeting.get("last_live_summary")
    if not isinstance(live_summary, Mapping):
        live_summary = {}
    summary_md = str(live_summary.get("summary_md") or "")
    actionables = live_summary.get("actionables")
    if not isinstance(actionables, list):
        actionables = []
    bounded_profile = profile_block.strip()
    return (
        f"<meeting_title>\n{str(meeting.get('title') or '')}\n</meeting_title>\n\n"
        f"<user_notes notes_rev=\"{notes_rev}\">\n{notes_md}\n</user_notes>\n\n"
        f"<labeled_transcript>\n{transcript}\n</labeled_transcript>\n\n"
        f"<last_live_summary>\n{summary_md}\n</last_live_summary>\n\n"
        "<actionables_with_status>\n"
        f"{json.dumps(actionables, separators=(',', ':'), ensure_ascii=False)}\n"
        "</actionables_with_status>\n\n"
        f"{bounded_profile}"
    ).strip()


def _render_transcript(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    lines: list[str] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        channel = row.get("channel")
        if channel == 0:
            speaker = "You"
        elif channel == 1:
            speaker = "Them"
        else:
            speaker = f"Speaker {channel}"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _validated_result(
    meeting: Mapping[str, Any], result: Any
) -> tuple[str, str, list[SuggestedEdit]]:
    if not isinstance(result, Mapping):
        raise ValueError("final summary result was not an object")
    if len(json.dumps(result, ensure_ascii=False)) > config.FINAL_SUMMARY_MAX_CHARS:
        raise ValueError("final summary result was oversized")
    generated_title = result.get("title")
    final_summary = result.get("final_summary_md")
    raw_edits = result.get("suggested_edits")
    if not isinstance(generated_title, str) or not generated_title.strip():
        raise ValueError("final summary title was empty")
    if len(generated_title) > 120:
        raise ValueError("final summary title was oversized")
    if not isinstance(final_summary, str) or not final_summary.strip():
        raise ValueError("final summary markdown was empty")
    if len(final_summary) > 40_000:
        raise ValueError("final summary markdown was oversized")
    if not isinstance(raw_edits, list):
        raise ValueError("suggested_edits was not an array")

    existing_title = str(meeting.get("title") or "").strip()
    title = existing_title or generated_title.strip()[:120]
    notes_md = str(meeting.get("notes_md") or "")
    edits: list[SuggestedEdit] = []
    for row in raw_edits:
        validated = _validated_edit(row, notes_md)
        if validated is not None:
            edits.append(validated)
        if len(edits) == 10:
            break
    return title, final_summary.strip(), edits


def _validated_edit(value: Any, notes_md: str) -> SuggestedEdit | None:
    if not isinstance(value, Mapping):
        return None
    anchor = value.get("anchor")
    occurrence = value.get("occurrence")
    insert_md = value.get("insert_md")
    rationale = value.get("rationale")
    if (
        not isinstance(anchor, str)
        or not isinstance(occurrence, int)
        or isinstance(occurrence, bool)
        or occurrence < 1
        or not isinstance(insert_md, str)
        or not insert_md.strip()
        or not isinstance(rationale, str)
    ):
        return None
    if len(anchor) > 8_000 or len(insert_md) > 8_000 or len(rationale) > 2_000:
        raise ValueError("suggested edit was oversized")
    if anchor and _occurrences(notes_md, anchor) < occurrence:
        return None
    return SuggestedEdit(
        id=uuid4().hex[:8],
        anchor=anchor,
        occurrence=occurrence,
        insert_md=insert_md,
        rationale=rationale,
    )


def _occurrences(text: str, anchor: str) -> int:
    count = 0
    start = 0
    while True:
        position = text.find(anchor, start)
        if position < 0:
            return count
        count += 1
        start = position + len(anchor)


async def _safe_emit(emit: Callable[[Any], Awaitable[None]], event: Any) -> None:
    try:
        await emit(event)
    except Exception:
        return
