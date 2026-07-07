from __future__ import annotations

import asyncio
from collections.abc import Iterable

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import Enrichment, NoteRecord


async def enrich_note(
    note: NoteRecord,
    llm: LLMPort,
    user_card: str | None = None,
) -> Enrichment:
    enrichment, _ = await enrich_note_with_status(note, llm, user_card=user_card)
    return enrichment


async def enrich_note_with_status(
    note: NoteRecord,
    llm: LLMPort,
    user_card: str | None = None,
) -> tuple[Enrichment, bool]:
    for attempt in range(2):  # one retry absorbs transient provider errors
        try:
            response = await llm.complete_json(
                config.ENRICH_MODEL,
                _system_prompt(),
                _user_prompt(note, user_card),
                timeout_s=config.ENRICH_TIMEOUT_S,
            )
        except LLMError:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            return fallback_enrichment(note), True
        return validate_enrichment(response, note), False
    return fallback_enrichment(note), True  # unreachable; satisfies type checkers


def validate_enrichment(data: dict, note: NoteRecord) -> Enrichment:
    gist = _clean_string(data.get("gist")) or _fallback_gist(note)
    aliases = _clean_list(data.get("aliases"))[:6]
    topics = [_normalize_topic(topic) for topic in _clean_list(data.get("topics"))]
    topics = _dedupe([topic for topic in topics if topic])[:4]
    entities = _dedupe(_clean_list(data.get("entities")))
    utterances = _clean_list(data.get("utterances"))
    return Enrichment(
        gist=_clamp_gist(gist),
        utterances=utterances,
        aliases=aliases,
        topics=topics,
        entities=entities,
    )


def fallback_enrichment(note: NoteRecord) -> Enrichment:
    return Enrichment(
        gist=_fallback_gist(note),
        utterances=[],
        aliases=[note.title],
        topics=[],
        entities=[],
    )


def _system_prompt() -> str:
    keys = ", ".join(Enrichment.model_fields.keys())
    return (
        "Return only a JSON object for a note enrichment. "
        f"The object must contain exactly these keys: {keys}. "
        f"Use a one-line gist of at most {config.GIST_MAX_CHARS} characters."
    )


def _user_prompt(note: NoteRecord, user_card: str | None = None) -> str:
    card = user_card.strip() if user_card else ""
    card_block = ""
    if card:
        card_block = (
            "DATA: USER INFORMATION CARD\n"
            "Use this background only to disambiguate people, companies, "
            "projects, topics, entities, aliases, and realistic spoken "
            "utterances for the note below.\n"
            "----- BEGIN USER CARD -----\n"
            f"{card}\n"
            "----- END USER CARD -----\n\n"
        )
    return (
        card_block
        + f"Title: {note.title}\n"
        f"Folder: {note.folder}\n"
        f"Body:\n{note.text}\n"
    )


def _fallback_gist(note: NoteRecord) -> str:
    for line in note.text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return _clamp_gist(cleaned)
    return _clamp_gist(note.title)


def _clamp_gist(value: str) -> str:
    max_chars = config.GIST_MAX_CHARS
    if len(value) <= max_chars:
        return value
    clipped = value[:max_chars]
    if clipped[-1].isspace() or value[max_chars : max_chars + 1].isspace():
        return clipped.rstrip()
    head = clipped.rsplit(" ", 1)[0].rstrip()
    return head or clipped.rstrip()


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
    return _dedupe(cleaned)


def _clean_string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_topic(value: str) -> str:
    return " ".join(value.lower().split())


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
