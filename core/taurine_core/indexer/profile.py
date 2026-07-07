from __future__ import annotations

from taurine_core import config
from taurine_core.ports import LLMPort


async def build_user_card(index: object, llm: LLMPort) -> str:
    disabled = getattr(index, "disabled_sources", set)()
    notes = [
        note
        for note in index.notes()  # type: ignore[attr-defined]
        if note.source not in disabled
    ]
    response = await llm.complete_json(
        config.PROFILE_MODEL,
        _system_prompt(),
        _user_prompt(notes),
        timeout_s=config.PROFILE_TIMEOUT_S,
    )
    return _clamp_card(str(response.get("card", "")).strip())


def _system_prompt() -> str:
    return (
        "Return only JSON with a single key named card. The card must be dense "
        "markdown background information about the note owner: who they are, "
        "what they work on, key people and roles or relationships, companies "
        "or clients, active projects, and recurring topics. Prefer concise "
        "facts useful for disambiguating future note enrichments."
    )


def _user_prompt(notes: object) -> str:
    blocks: list[str] = []
    for note in notes:  # type: ignore[union-attr]
        body = str(note.text)[: config.PROFILE_NOTE_CHAR_BUDGET]
        blocks.append(
            "\n".join(
                [
                    f"===== BEGIN NOTE {note.note_id} =====",
                    f"Title: {note.title}",
                    "Body:",
                    body,
                    f"===== END NOTE {note.note_id} =====",
                ]
            )
        )
    return (
        "Build the user information card from this complete note corpus. "
        "Use only the data below.\n\n"
        + "\n\n".join(blocks)
    )


def _clamp_card(card: str) -> str:
    if len(card) <= config.PROFILE_MAX_CHARS:
        return card
    return card[: config.PROFILE_MAX_CHARS].rstrip()
