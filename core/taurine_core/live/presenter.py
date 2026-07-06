from __future__ import annotations

from typing import Any

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import Chunk, Pick, SlateItem


SYSTEM_PROMPT = (
    "You choose which notes, if any, are worth showing in a live meeting. "
    "Return only JSON shaped as {\"picks\":[{\"note_id\":\"...\",\"why\":\"...\"}]}."
)


class Presenter:
    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def present(
        self,
        transcript: list[Chunk],
        slate: list[SlateItem],
        via_titles: dict[str, str] | None = None,
    ) -> list[Pick]:
        if not slate:
            return []

        user_prompt = _build_user_prompt(transcript, slate, via_titles or {})
        try:
            response = await self._llm.complete_json(config.PRESENTER_MODEL, SYSTEM_PROMPT, user_prompt)
        except LLMError:
            try:
                response = await self._llm.complete_json(
                    config.PRESENTER_FALLBACK_MODEL,
                    SYSTEM_PROMPT,
                    user_prompt,
                )
            except LLMError:
                return []

        return _validated_picks(response, slate)


def _build_user_prompt(
    transcript: list[Chunk],
    slate: list[SlateItem],
    via_titles: dict[str, str],
) -> str:
    recent = transcript[-config.PRESENTER_TRANSCRIPT_CHUNKS :]
    transcript_lines = [f"channel {chunk.channel}: {chunk.text}" for chunk in recent]
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
