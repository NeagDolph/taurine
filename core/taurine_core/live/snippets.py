"""LLM-picked "most important snippet" for notes surfaced by live retrieval."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any, Protocol

from taurine_core import config
from taurine_core.live.grounding import GroundingUpdate
from taurine_core.retrieval.types import EvidenceNote
from taurine_core.schemas import NoteSuggestionEvent

SnippetSink = Callable[[NoteSuggestionEvent], Awaitable[None] | None]


class SnippetLLM(Protocol):
    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        timeout_s: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


SYSTEM_PROMPT = """You surface the single most important snippet of a saved \
note for a live conversation.

For each note you receive its retrieved passages and the current stretch of \
conversation. Pick the one detail from the note's passages that matters most \
right now — a concrete fact, number, decision, or commitment the speakers \
would want in front of them. Quote or tightly compress the note's own words \
(under 25 words); never invent content that is not in the passages, and never \
summarize the conversation itself.

Return JSON only:
{"snippets": [{"note_ref": "N1", "snippet": "the most important detail"}]}
"""

SNIPPET_SCHEMA: dict[str, Any] = {
    "title": "note_snippets",
    "type": "object",
    "properties": {
        "snippets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_ref": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["note_ref", "snippet"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["snippets"],
    "additionalProperties": False,
}


class SnippetAgent:
    """Emit one NoteSuggestionEvent per newly displayed note per session.

    Snippet generation is best-effort: an LLM failure falls back to the
    note's top passage text, and a metadata-only note ships with an empty
    snippet — the suggestion itself must never be lost to a model error.
    """

    def __init__(
        self,
        llm: SnippetLLM | None,
        *,
        max_per_session: int = config.NOTE_SNIPPET_MAX_PER_MEETING,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._llm = llm
        self._max = max_per_session
        self._clock = clock
        self._seen: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def offer(
        self,
        update: GroundingUpdate,
        transcript_tail: str,
        emit: SnippetSink,
    ) -> None:
        """Schedule snippets for notes newly entering the display set."""

        if self._closed or len(self._seen) >= self._max:
            return
        fresh = [
            note
            for note in update.display_notes
            if note.note_ref not in self._seen
        ][: max(0, self._max - len(self._seen))]
        if not fresh:
            return
        self._seen.update(note.note_ref for note in fresh)
        passages = {
            note.note_ref: [
                row.text
                for row in update.display_passages
                if row.note_id == note.note_id and row.text.strip()
            ]
            for note in fresh
        }
        task = asyncio.create_task(
            self._run(tuple(fresh), passages, transcript_tail, emit)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        self._closed = True
        tasks = tuple(self._tasks)
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(
        self,
        notes: tuple[EvidenceNote, ...],
        passages: dict[str, list[str]],
        transcript_tail: str,
        emit: SnippetSink,
    ) -> None:
        with_text = [note for note in notes if passages.get(note.note_ref)]
        picked = await self._pick_snippets(with_text, passages, transcript_tail)
        for note in notes:
            fallback = next(iter(passages.get(note.note_ref, ())), "")
            snippet = _clip(picked.get(note.note_ref) or fallback)
            await self._emit(
                emit,
                NoteSuggestionEvent(
                    note_id=note.note_id,
                    note_ref=note.note_ref,
                    title=note.title,
                    source=note.source,
                    score=float(note.score),
                    snippet=snippet,
                    ts=self._clock(),
                ),
            )

    async def _pick_snippets(
        self,
        notes: list[EvidenceNote],
        passages: dict[str, list[str]],
        transcript_tail: str,
    ) -> dict[str, str]:
        if self._llm is None or not notes:
            return {}
        blocks = [
            f"[{note.note_ref}] {note.title}\n"
            + "\n".join(f"- {text}" for text in passages[note.note_ref][:4])
            for note in notes
        ]
        tail = transcript_tail[-config.NOTE_SNIPPET_TRANSCRIPT_TAIL_CHARS :]
        prompt = (
            "<conversation>\n" + tail + "\n</conversation>\n\n"
            "<notes>\n" + "\n\n".join(blocks) + "\n</notes>"
        )
        try:
            raw = await self._llm.complete_json(
                config.NOTE_SNIPPET_MODEL,
                SYSTEM_PROMPT,
                prompt,
                timeout_s=config.NOTE_SNIPPET_TIMEOUT_S,
                schema=SNIPPET_SCHEMA,
            )
        except Exception:
            return {}
        valid = {note.note_ref for note in notes}
        picked: dict[str, str] = {}
        for row in raw.get("snippets") or []:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("note_ref") or "")
            snippet = str(row.get("snippet") or "").strip()
            if ref in valid and snippet:
                picked[ref] = snippet
        return picked

    async def _emit(self, emit: SnippetSink, event: NoteSuggestionEvent) -> None:
        pending = emit(event)
        if isawaitable(pending):
            await pending


def _clip(text: str) -> str:
    cleaned = " ".join(text.split())
    limit = config.NOTE_SNIPPET_MAX_CHARS
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"
