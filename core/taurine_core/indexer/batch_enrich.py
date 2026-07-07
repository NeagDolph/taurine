from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import Enrichment, NoteRecord

from .enrich import enrich_note_with_status, validate_enrichment

EmbedFn = Callable[[list[str]], np.ndarray]
NoteDone = Callable[[str, Enrichment, bool], None]


def cluster_notes(notes: list[NoteRecord], embed: EmbedFn) -> list[list[NoteRecord]]:
    if not notes:
        return []

    texts = [f"{note.title}\n{note.text[:600]}" for note in notes]
    vectors = np.asarray(embed(texts), dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape[0] != len(notes):
        raise ValueError(f"embed returned {vectors.shape[0]} rows for {len(notes)} notes")
    vectors = _normalize_rows(vectors)

    batch_size = max(1, int(config.ENRICH_BATCH_SIZE))
    char_budget = max(0, int(config.ENRICH_BATCH_CHAR_BUDGET))
    unassigned = set(range(len(notes)))
    clusters: list[list[NoteRecord]] = []

    while unassigned:
        anchor = next(index for index in range(len(notes)) if index in unassigned)
        unassigned.remove(anchor)
        batch_indices = [anchor]
        batch_chars = len(notes[anchor].text)

        if batch_chars <= char_budget and len(batch_indices) < batch_size:
            candidates = [index for index in range(len(notes)) if index in unassigned]
            if candidates:
                scores = vectors[candidates] @ vectors[anchor]
                for order_index in np.argsort(-scores, kind="stable"):
                    if len(batch_indices) >= batch_size:
                        break
                    candidate = candidates[int(order_index)]
                    next_chars = batch_chars + len(notes[candidate].text)
                    if next_chars > char_budget:
                        break
                    batch_indices.append(candidate)
                    unassigned.remove(candidate)
                    batch_chars = next_chars

        clusters.append([notes[index] for index in batch_indices])

    return clusters


async def enrich_batch(
    notes: list[NoteRecord],
    llm: LLMPort,
    user_card: str | None = None,
) -> dict[str, tuple[Enrichment, bool]]:
    if not notes:
        return {}

    for attempt in range(2):
        try:
            response = await llm.complete_json(
                config.ENRICH_MODEL,
                _system_prompt(len(notes)),
                _user_prompt(notes, user_card),
                timeout_s=config.ENRICH_TIMEOUT_S * 2,
            )
        except LLMError:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            return await _fallback_many(notes, llm, user_card)
        break

    note_data = response.get("notes") if isinstance(response, dict) else None
    if not isinstance(note_data, dict):
        note_data = {}

    results: dict[str, tuple[Enrichment, bool]] = {}
    fallback_notes: list[NoteRecord] = []
    for note in notes:
        entry = note_data.get(note.note_id)
        if isinstance(entry, dict):
            results[note.note_id] = (validate_enrichment(entry, note), False)
        else:
            fallback_notes.append(note)

    if fallback_notes:
        results.update(await _fallback_many(fallback_notes, llm, user_card))
    return results


async def enrich_all_batched(
    notes: list[NoteRecord],
    llm: LLMPort,
    user_card: str | None,
    embed: EmbedFn,
    on_note_done: NoteDone,
) -> dict[str, tuple[Enrichment, bool]]:
    if not notes:
        return {}

    if len(notes) < config.ENRICH_BATCH_MIN_NOTES:
        return await _enrich_individually(notes, llm, user_card, on_note_done)

    results: dict[str, tuple[Enrichment, bool]] = {}
    semaphore = asyncio.Semaphore(config.REENRICH_CONCURRENCY)
    batches = cluster_notes(notes, embed)

    async def one_batch(batch: list[NoteRecord]) -> None:
        async with semaphore:
            batch_results = await enrich_batch(batch, llm, user_card=user_card)
        for note in batch:
            enrichment, failed = batch_results[note.note_id]
            results[note.note_id] = (enrichment, failed)
            on_note_done(note.note_id, enrichment, failed)

    await asyncio.gather(*(one_batch(batch) for batch in batches))
    return results


async def _enrich_individually(
    notes: list[NoteRecord],
    llm: LLMPort,
    user_card: str | None,
    on_note_done: NoteDone,
) -> dict[str, tuple[Enrichment, bool]]:
    semaphore = asyncio.Semaphore(config.REENRICH_CONCURRENCY)
    results: dict[str, tuple[Enrichment, bool]] = {}

    async def one(note: NoteRecord) -> None:
        async with semaphore:
            enrichment, failed = await enrich_note_with_status(note, llm, user_card=user_card)
        results[note.note_id] = (enrichment, failed)
        on_note_done(note.note_id, enrichment, failed)

    await asyncio.gather(*(one(note) for note in notes))
    return results


async def _fallback_many(
    notes: list[NoteRecord],
    llm: LLMPort,
    user_card: str | None,
) -> dict[str, tuple[Enrichment, bool]]:
    semaphore = asyncio.Semaphore(config.REENRICH_CONCURRENCY)
    results: dict[str, tuple[Enrichment, bool]] = {}

    async def one(note: NoteRecord) -> None:
        async with semaphore:
            results[note.note_id] = await enrich_note_with_status(note, llm, user_card=user_card)

    await asyncio.gather(*(one(note) for note in notes))
    return results


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=normalized, where=norms > 0)
    return normalized


def _system_prompt(note_count: int) -> str:
    keys = ", ".join(Enrichment.model_fields.keys())
    return (
        f"You will receive {note_count} independent notes for enrichment. "
        "Return only JSON with this shape: "
        '{"notes": {"<note_id>": {gist, utterances, aliases, topics, entities}, ...}}. '
        "Every note_id from the input must be present. "
        "Treat notes independently; never blend facts, names, or commitments from one "
        "note into another note's enrichment. "
        f"Each enrichment object must contain exactly these keys: {keys}. "
        f"Use a one-line gist of at most {config.GIST_MAX_CHARS} characters."
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
