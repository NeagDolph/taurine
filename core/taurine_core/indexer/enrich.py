from __future__ import annotations

from collections.abc import Iterable

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import Enrichment, NoteRecord


async def enrich_note(note: NoteRecord, llm: LLMPort) -> Enrichment:
    enrichment, _ = await enrich_note_with_status(note, llm)
    return enrichment


async def enrich_note_with_status(note: NoteRecord, llm: LLMPort) -> tuple[Enrichment, bool]:
    try:
        response = await llm.complete_json(
            config.ENRICH_MODEL,
            _system_prompt(),
            _user_prompt(note),
        )
    except LLMError:
        return fallback_enrichment(note), True
    return validate_enrichment(response, note), False


def validate_enrichment(data: dict, note: NoteRecord) -> Enrichment:
    gist = _clean_string(data.get("gist")) or _fallback_gist(note)
    aliases = _clean_list(data.get("aliases"))[:6]
    topics = [_normalize_topic(topic) for topic in _clean_list(data.get("topics"))]
    topics = _dedupe([topic for topic in topics if topic])[:4]
    entities = _dedupe(_clean_list(data.get("entities")))
    utterances = _clean_list(data.get("utterances"))
    return Enrichment(
        gist=gist[:120],
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
        f"The object must contain exactly these keys: {keys}."
    )


def _user_prompt(note: NoteRecord) -> str:
    return (
        f"Title: {note.title}\n"
        f"Folder: {note.folder}\n"
        f"Body:\n{note.text}\n"
    )


def _fallback_gist(note: NoteRecord) -> str:
    for line in note.text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:120]
    return note.title[:120]


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
