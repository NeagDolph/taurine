from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from taurine_core import config
from taurine_core.ports import IndexPort, LLMError
from taurine_core.schemas import Chunk, Pick, SlateItem


SYSTEM_PROMPT = (
    "You decide which notes, if any, get pinned on screen during a live meeting. "
    "A pin interrupts the user — it is a strong claim that this note is about what is "
    "being discussed RIGHT NOW. Pin only when you KNOW the note is directly relevant: "
    "the conversation explicitly involves the note's subject (a person, project, company, "
    "document, or event named or unmistakably referenced in the transcript), or the note "
    "answers a question just raised. Topical adjacency, shared themes, or 'might be "
    "useful' are NOT enough — when in doubt, pick nothing; empty picks are the normal "
    "outcome, not a failure. "
    "You receive the FULL conversation so far: use it to tell a passing mention from an "
    "actual subject of discussion, and favor what the conversation is about now over "
    "what it was about earlier. "
    "Weigh named entities (people, companies, projects) above generic similarity, and be "
    'robust to ASR spelling drift, for example "Armand" may mean "Armaan". '
    "Certainty requires evidence: if a gist alone does not PROVE direct relevance, call "
    "read_note and verify the content before picking — never pin on a hunch. Neighbors "
    "of slated notes are legitimate picks only after you have read and verified them. "
    "Each pick's 'why' must cite the specific thing said in the conversation AND the "
    "specific note content it connects to; if you cannot write that sentence, do not pick. "
    "Return only JSON shaped as {\"picks\":[{\"note_id\":\"...\",\"why\":\"...\"}]}."
)

READ_NOTE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Read title, gist, and the beginning of a note by short or full note_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Short id from note_directory or the full note_id.",
                    }
                },
                "required": ["note_id"],
                "additionalProperties": False,
            },
        },
    }
]

FINAL_ANSWER_MESSAGE = "Answer now with the JSON contract."


class Presenter:
    def __init__(
        self,
        llm: Any,
        index: IndexPort,
        on_log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._llm = llm
        self._index = index
        self._on_log = on_log

    async def present(
        self,
        transcript: list[Chunk],
        slate: list[SlateItem],
        via_titles: dict[str, str] | None = None,
        *,
        revision: int | None = None,
        partials: dict[int, str] | None = None,
    ) -> list[Pick]:
        started = time.monotonic()
        raw_response: dict[str, Any] | None = None
        model_used: str | None = None
        error: str | None = None
        if not slate:
            self._log(
                revision=revision,
                model_used=None,
                transcript_chunks=len(transcript),
                slate_note_ids=[],
                raw_response=None,
                picks=[],
                tool_calls=[],
                latency_ms=_elapsed_ms(started),
                error=None,
            )
            return []

        prompt_context = _build_prompt_context(self._index, slate)
        user_prompt = _build_user_prompt(
            transcript,
            slate,
            via_titles or {},
            prompt_context,
            partials,
        )
        tool_calls: list[dict[str, Any]] = []
        read_note_ids: set[str] = set()
        try:
            raw_response, tool_calls, read_note_ids = await self._complete_with_tools(
                config.PRESENTER_MODEL,
                user_prompt,
                prompt_context,
            )
            model_used = config.PRESENTER_MODEL
        except LLMError as primary_error:
            try:
                raw_response, tool_calls, read_note_ids = await self._complete_with_tools(
                    config.PRESENTER_FALLBACK_MODEL,
                    user_prompt,
                    prompt_context,
                )
                model_used = config.PRESENTER_FALLBACK_MODEL
            except LLMError as fallback_error:
                error = f"{primary_error}; {fallback_error}"
                self._log(
                    revision=revision,
                    model_used=None,
                    transcript_chunks=len(transcript),
                    slate_note_ids=[item.note_id for item in slate],
                    raw_response=None,
                    picks=[],
                    tool_calls=[],
                    latency_ms=_elapsed_ms(started),
                    error=error,
                )
                return []

        picks = _validated_picks(raw_response, slate, read_note_ids, prompt_context)
        self._log(
            revision=revision,
            model_used=model_used,
            transcript_chunks=len(transcript),
            slate_note_ids=[item.note_id for item in slate],
            raw_response=raw_response,
            picks=[pick.model_dump() for pick in picks],
            tool_calls=tool_calls,
            latency_ms=_elapsed_ms(started),
            error=None,
        )
        return picks

    async def _complete_with_tools(
        self,
        model: str,
        user_prompt: str,
        prompt_context: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        tool_logs: list[dict[str, Any]] = []
        read_results: list[dict[str, Any]] = []
        read_note_ids: set[str] = set()

        while len(tool_logs) < config.PRESENTER_MAX_TOOL_CALLS:
            assistant = await self._llm.complete_tools(
                model,
                SYSTEM_PROMPT,
                messages,
                READ_NOTE_TOOLS,
            )
            tool_calls = _assistant_tool_calls(assistant)
            if not tool_calls:
                parsed = _parse_assistant_json(assistant)
                if parsed is not None:
                    return parsed, tool_logs, read_note_ids
                break

            messages.append(_assistant_history_message(assistant))
            for tool_call in tool_calls:
                if len(tool_logs) >= config.PRESENTER_MAX_TOOL_CALLS:
                    break
                requested_note_id = _tool_call_note_id(tool_call)
                content, ok, resolved_note_id = self._read_note_result(
                    requested_note_id,
                    prompt_context,
                )
                if resolved_note_id is not None:
                    read_note_ids.add(resolved_note_id)
                tool_logs.append({"note_id": requested_note_id, "ok": ok})
                read_results.append(
                    {
                        "note_id": requested_note_id,
                        "ok": ok,
                        "content": content,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id", "")),
                        "name": "read_note",
                        "content": content,
                    }
                )

        final_prompt = _build_final_prompt(user_prompt, read_results)
        return (
            await self._llm.complete_json(model, SYSTEM_PROMPT, final_prompt),
            tool_logs,
            read_note_ids,
        )

    def _read_note_result(
        self,
        note_id: str,
        prompt_context: dict[str, Any],
    ) -> tuple[str, bool, str | None]:
        resolved_note_id = _resolve_note_id(note_id, prompt_context)
        if resolved_note_id is None:
            return f"ERROR: unknown note_id {note_id}", False, None

        note = self._index.note(resolved_note_id)
        if note is None:
            return f"ERROR: unknown note_id {note_id}", False, None

        return (
            json.dumps(
                {
                    "note_id": note.note_id,
                    "title": note.title,
                    "gist": note.gist,
                    "text": note.text[:1200],
                }
            ),
            True,
            note.note_id,
        )

    def _log(self, **payload: Any) -> None:
        if self._on_log is None:
            return
        try:
            self._on_log(payload)
        except Exception:
            return


def _build_user_prompt(
    transcript: list[Chunk],
    slate: list[SlateItem],
    via_titles: dict[str, str],
    prompt_context: dict[str, Any],
    partials: dict[int, str] | None = None,
) -> str:
    recent = transcript[-config.PRESENTER_TRANSCRIPT_CHUNKS :]
    transcript_lines = [f"channel {chunk.channel}: {chunk.text}" for chunk in recent]
    transcript_lines.extend(
        f"channel {channel} (in progress): {text}"
        for channel, text in sorted((partials or {}).items())
        if text.strip()
    )
    slate_lines = [_slate_line(item, via_titles, prompt_context) for item in slate]
    directory_lines = [
        f'{short_id} -> "{title}"'
        for short_id, title in prompt_context["directory_titles"].items()
    ]
    return "\n".join(
        [
            "Contract: pick 0-2 notes, copy note_id exactly. Pin only notes DIRECTLY",
            "relevant to what is being discussed right now — verified, not guessed.",
            "Empty picks are the normal outcome. Provenance: exact > search > linked.",
            "The transcript below is the entire conversation so far, oldest first;",
            "the last lines are the current moment.",
            "<transcript>",
            *transcript_lines,
            "</transcript>",
            "<slate>",
            *slate_lines,
            "</slate>",
            "<note_directory>",
            *directory_lines,
            "</note_directory>",
        ]
    )


def _build_prompt_context(index: IndexPort, slate: list[SlateItem]) -> dict[str, Any]:
    short_to_full: dict[str, str] = {}
    full_ids: set[str] = set()
    directory_titles: dict[str, str] = {}
    neighbors_by_note_id: dict[str, list[str]] = {}

    def add_directory_entry(note_id: str, title: str) -> None:
        short_id = _short_id(note_id)
        full_ids.add(note_id)
        short_to_full.setdefault(short_id, note_id)
        directory_titles.setdefault(short_id, title)

    for item in slate:
        add_directory_entry(item.note_id, item.title)

    for item in slate:
        neighbor_ids = [
            note_id
            for note_id, _weight in index.neighbors(item.note_id)[
                : config.PRESENTER_NEIGHBORS_SHOWN
            ]
        ]
        neighbors_by_note_id[item.note_id] = neighbor_ids
        for neighbor_id in neighbor_ids:
            note = index.note(neighbor_id)
            add_directory_entry(neighbor_id, note.title if note is not None else neighbor_id)

    return {
        "short_to_full": short_to_full,
        "full_ids": full_ids,
        "directory_titles": directory_titles,
        "neighbors_by_note_id": neighbors_by_note_id,
    }


def _slate_line(
    item: SlateItem,
    via_titles: dict[str, str],
    prompt_context: dict[str, Any],
) -> str:
    if item.source == "exact":
        provenance = "[exact: title spoken]"
    elif item.via_note_id:
        via_title = via_titles.get(item.via_note_id, item.via_note_id)
        provenance = f'[linked: connected to "{via_title}", matched "{item.heard}"]'
    else:
        provenance = f'[search: matched "{item.heard}"]'
    neighbors = ", ".join(
        _short_id(note_id)
        for note_id in prompt_context["neighbors_by_note_id"].get(item.note_id, [])
    )
    return (
        f'{provenance} note_id: {item.note_id} short_id: {_short_id(item.note_id)} '
        f'"{item.title}" - {item.gist} neighbors: [{neighbors}]'
    )


def _validated_picks(
    response: dict[str, Any],
    slate: list[SlateItem],
    read_note_ids: set[str],
    prompt_context: dict[str, Any],
) -> list[Pick]:
    raw_picks = response.get("picks")
    if not isinstance(raw_picks, list):
        return []

    allowed_ids = {item.note_id for item in slate} | read_note_ids
    seen_ids: set[str] = set()
    picks: list[Pick] = []
    for raw_pick in raw_picks:
        if not isinstance(raw_pick, dict):
            continue
        note_id = raw_pick.get("note_id")
        if not isinstance(note_id, str):
            continue
        resolved_note_id = _resolve_note_id(note_id, prompt_context)
        if resolved_note_id is None:
            resolved_note_id = note_id
        if resolved_note_id not in allowed_ids or resolved_note_id in seen_ids:
            continue
        why = raw_pick.get("why", "")
        if not isinstance(why, str):
            why = str(why)
        picks.append(Pick(note_id=resolved_note_id, why=why[: config.WHY_MAX_CHARS]))
        seen_ids.add(resolved_note_id)
        if len(picks) >= config.PRESENTER_MAX_PICKS:
            break
    return picks


def _assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


def _assistant_history_message(message: dict[str, Any]) -> dict[str, Any]:
    history = {
        "role": "assistant",
        "content": message.get("content"),
    }
    tool_calls = _assistant_tool_calls(message)
    if tool_calls:
        history["tool_calls"] = tool_calls
    return history


def _tool_call_note_id(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict) or function.get("name") != "read_note":
        return ""
    arguments = function.get("arguments")
    try:
        if isinstance(arguments, str):
            payload = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            payload = arguments
        else:
            payload = {}
    except json.JSONDecodeError:
        payload = {}
    note_id = payload.get("note_id")
    return note_id if isinstance(note_id, str) else ""


def _parse_assistant_json(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        parsed = _parse_json_content(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_final_prompt(user_prompt: str, read_results: list[dict[str, Any]]) -> str:
    lines = [user_prompt]
    if read_results:
        lines.append("<read_note_results>")
        for result in read_results:
            lines.append(
                f'note_id: {result["note_id"]} ok: {result["ok"]} result: {result["content"]}'
            )
        lines.append("</read_note_results>")
    lines.append(FINAL_ANSWER_MESSAGE)
    return "\n".join(lines)


def _resolve_note_id(note_id: str, prompt_context: dict[str, Any]) -> str | None:
    if note_id in prompt_context["full_ids"]:
        return note_id
    return prompt_context["short_to_full"].get(note_id)


def _short_id(note_id: str) -> str:
    return note_id.rsplit("/", 1)[-1]


def _parse_json_content(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0
