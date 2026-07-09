from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable

from pydantic import BaseModel, Field

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import NoteRecord


class SlimEnrichment(BaseModel):
    aliases: list[str] = Field(default_factory=list)
    utterances: list[str] = Field(default_factory=list)


def select_slim_notes(
    changed: list[NoteRecord],
    value_scores: dict[str, float],
) -> list[NoteRecord]:
    if not changed:
        return []

    # Ceil is deliberate: favor covering more high-value notes.
    fraction_count = math.ceil(
        len(changed) * max(0.0, float(config.SLIM_LLM_VALUE_FRACTION))
    )
    floor_count = min(max(0, int(config.SLIM_LLM_MIN_NOTES)), len(changed))
    count = min(len(changed), max(fraction_count, floor_count))

    ranked = sorted(
        enumerate(changed),
        key=lambda item: (-float(value_scores.get(item[1].note_id, 0.0)), item[0]),
    )
    return [note for _index, note in ranked[:count]]


async def slim_enrich_batch(
    notes: list[NoteRecord],
    llm: LLMPort,
    user_card: str | None,
) -> dict[str, SlimEnrichment]:
    if not notes:
        return {}

    results: dict[str, SlimEnrichment] = {}
    batch_size = max(1, int(config.ENRICH_BATCH_SIZE))
    for offset in range(0, len(notes), batch_size):
        batch = notes[offset : offset + batch_size]
        results.update(await _slim_enrich_one_batch(batch, llm, user_card))
    return results


async def _slim_enrich_one_batch(
    notes: list[NoteRecord],
    llm: LLMPort,
    user_card: str | None,
) -> dict[str, SlimEnrichment]:
    for attempt in range(2):
        try:
            response = await llm.complete_json(
                config.ENRICH_MODEL,
                _system_prompt(),
                _user_prompt(notes, user_card),
                timeout_s=config.ENRICH_TIMEOUT_S,
            )
        except LLMError:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            return _empty_results(notes)
        break

    note_data = response.get("notes") if isinstance(response, dict) else None
    if not isinstance(note_data, dict):
        note_data = {}

    results: dict[str, SlimEnrichment] = {}
    for note in notes:
        entry = note_data.get(note.note_id)
        results[note.note_id] = (
            _validate_slim_enrichment(entry) if isinstance(entry, dict) else SlimEnrichment()
        )
    return results


def _empty_results(notes: list[NoteRecord]) -> dict[str, SlimEnrichment]:
    return {note.note_id: SlimEnrichment() for note in notes}


def _validate_slim_enrichment(data: dict) -> SlimEnrichment:
    # Drop empty values before clamping so later valid entries can fill the limit.
    return SlimEnrichment(
        aliases=_clean_list(data.get("aliases"))[:6],
        utterances=_clean_list(data.get("utterances"))[:4]
        if config.SLIM_UTTERANCES
        else [],
    )


def _system_prompt() -> str:
    if not config.SLIM_UTTERANCES:
        return (
            'Return only JSON: {"notes": {"<note_id>": {"aliases": [..]}}}. '
            "Aliases: up to 6 short spoken names for the note or its subject."
        )
    return (
        'Return only JSON: {"notes": {"<note_id>": {"aliases": [..], '
        '"utterances": [..]}}}. Aliases: up to 6 short spoken names for the '
        "note or its subject. Utterances: 4 realistic spoken sentences someone "
        "would say in a meeting when this note is relevant."
    )


def _user_prompt(notes: list[NoteRecord], user_card: str | None = None) -> str:
    return _user_card_block(user_card) + "\n\n".join(_note_block(note) for note in notes)


def _user_card_block(user_card: str | None) -> str:
    card = user_card.strip() if user_card else ""
    if not card:
        return ""
    return (
        "DATA: USER INFORMATION CARD\n"
        "Use this background only to disambiguate people, companies, "
        "projects, topics, entities, aliases, and realistic spoken "
        "utterances for the note below.\n"
        "----- BEGIN USER CARD -----\n"
        f"{card}\n"
        "----- END USER CARD -----\n\n"
    )


def _note_block(note: NoteRecord) -> str:
    return (
        "----- BEGIN NOTE -----\n"
        f"note_id: {note.note_id}\n"
        f"Title: {note.title}\n"
        f"Folder: {note.folder}\n"
        f"Body:\n{note.text}\n"
        "----- END NOTE -----"
    )


def _clean_list(value: object) -> list[str]:
    if isinstance(value, str):
        values: Iterable[object] = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        return []
    cleaned: list[str] = []
    for item in values:
        text = _clean_string(item)
        if text:
            cleaned.append(text)
    return cleaned


def _clean_string(value: object) -> str:
    return str(value).strip() if value is not None else ""
