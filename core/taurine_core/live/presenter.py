from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import Chunk, Pick, SlateItem


SYSTEM_PROMPT = (
    "You choose which notes, if any, are worth showing in a live meeting. "
    "Return only JSON shaped as {\"picks\":[{\"note_id\":\"...\",\"why\":\"...\"}]}."
)


class Presenter:
    def __init__(
        self,
        llm: LLMPort,
        on_log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._llm = llm
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
                latency_ms=_elapsed_ms(started),
                error=None,
            )
            return []

        user_prompt = _build_user_prompt(transcript, slate, via_titles or {}, partials)
        try:
            raw_response = await self._llm.complete_json(
                config.PRESENTER_MODEL,
                SYSTEM_PROMPT,
                user_prompt,
            )
            model_used = config.PRESENTER_MODEL
        except LLMError as primary_error:
            try:
                raw_response = await self._llm.complete_json(
                    config.PRESENTER_FALLBACK_MODEL,
                    SYSTEM_PROMPT,
                    user_prompt,
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
                    latency_ms=_elapsed_ms(started),
                    error=error,
                )
                return []

        picks = _validated_picks(raw_response, slate)
        self._log(
            revision=revision,
            model_used=model_used,
            transcript_chunks=len(transcript),
            slate_note_ids=[item.note_id for item in slate],
            raw_response=raw_response,
            picks=[pick.model_dump() for pick in picks],
            latency_ms=_elapsed_ms(started),
            error=None,
        )
        return picks

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
    partials: dict[int, str] | None = None,
) -> str:
    recent = transcript[-config.PRESENTER_TRANSCRIPT_CHUNKS :]
    transcript_lines = [f"channel {chunk.channel}: {chunk.text}" for chunk in recent]
    transcript_lines.extend(
        f"channel {channel} (in progress): {text}"
        for channel, text in sorted((partials or {}).items())
        if text.strip()
    )
    slate_lines = [_slate_line(item, via_titles) for item in slate]
    return "\n".join(
        [
            "Contract: pick 0-2 notes, copy note_id exactly, prefer the current moment over score.",
            "Provenance strength: exact > search > linked. Empty picks are valid.",
            "<recent_transcript>",
            *transcript_lines,
            "</recent_transcript>",
            "<slate>",
            *slate_lines,
            "</slate>",
        ]
    )


def _slate_line(item: SlateItem, via_titles: dict[str, str]) -> str:
    if item.source == "exact":
        provenance = "[exact: title spoken]"
    elif item.via_note_id:
        via_title = via_titles.get(item.via_note_id, item.via_note_id)
        provenance = f'[linked: connected to "{via_title}", matched "{item.heard}"]'
    else:
        provenance = f'[search: matched "{item.heard}"]'
    return f'{provenance} {item.note_id} "{item.title}" - {item.gist}'


def _validated_picks(response: dict[str, Any], slate: list[SlateItem]) -> list[Pick]:
    raw_picks = response.get("picks")
    if not isinstance(raw_picks, list):
        return []

    slate_ids = {item.note_id for item in slate}
    picks: list[Pick] = []
    for raw_pick in raw_picks:
        if not isinstance(raw_pick, dict):
            continue
        note_id = raw_pick.get("note_id")
        if not isinstance(note_id, str) or note_id not in slate_ids:
            continue
        why = raw_pick.get("why", "")
        if not isinstance(why, str):
            why = str(why)
        picks.append(Pick(note_id=note_id, why=why[: config.WHY_MAX_CHARS]))
        if len(picks) >= config.PRESENTER_MAX_PICKS:
            break
    return picks


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0
