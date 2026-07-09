"""GLiNER2 ingestion extraction adapter.

This module is intentionally not wired into sync yet. It turns local GLiNER2
entity/relation output into the typed extraction shape the ingest graph will
store in later tasks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from taurine_core import config
from taurine_core.live import spans
from taurine_core.schemas import EntityMention, Extraction, NoteRecord, RelationTriple

_LOGGER = logging.getLogger("taurine.indexer.extract")


@dataclass
class _EntityAccumulator:
    surface: str
    conf: float


@dataclass
class _RelationAccumulator:
    head: str
    rel_type: str
    tail: str
    conf: float


def extract_notes(notes: list[NoteRecord]) -> dict[str, Extraction]:
    """Extract typed entities and relation triples for notes.

    Inference failures are isolated to the note where possible and never
    propagate to callers.
    """
    if not notes:
        return {}

    model = _extractor_model()
    if model is None:
        _LOGGER.warning("GLiNER2 unavailable: ingestion extraction skipped")
        return {note.note_id: Extraction(failed=True) for note in notes}

    try:
        schema = _build_schema(model)
    except Exception as exc:
        _LOGGER.warning("GLiNER2 schema build failed: %s", exc)
        return {note.note_id: Extraction(failed=True) for note in notes}

    results: dict[str, Extraction] = {}
    for batch in _batches(notes, _ingest_batch_size()):
        short_items: list[tuple[NoteRecord, str]] = []
        for note in batch:
            text = _note_text(note)
            if _is_long_text(text):
                _flush_short_batch(model, schema, short_items, results)
                short_items = []
                results[note.note_id] = _extract_one(model, schema, note, text)
            else:
                short_items.append((note, text))
        _flush_short_batch(model, schema, short_items, results)
    return results


def _extractor_model() -> Any | None:
    return spans.shared_model()


def _build_schema(model: Any) -> Any:
    schema = model.create_schema()
    schema = schema.entities(dict(config.INGEST_ENTITY_TYPES))
    return schema.relations(list(config.INGEST_RELATION_TYPES))


def _flush_short_batch(
    model: Any,
    schema: Any,
    items: list[tuple[NoteRecord, str]],
    results: dict[str, Extraction],
) -> None:
    if not items:
        return

    if not hasattr(model, "batch_extract"):
        for note, text in items:
            results[note.note_id] = _extract_one(model, schema, note, text)
        return

    texts = [text for _, text in items]
    try:
        raw_results = model.batch_extract(
            texts,
            schema,
            batch_size=len(texts),
            threshold=_global_threshold(),
            include_confidence=True,
            max_len=_chunk_max_len(),
        )
        if not isinstance(raw_results, list) or len(raw_results) != len(items):
            raise ValueError("GLiNER2 batch result length mismatch")
    except Exception as exc:
        _LOGGER.warning("GLiNER2 batch extraction failed; retrying per note: %s", exc)
        for note, text in items:
            results[note.note_id] = _extract_one(model, schema, note, text)
        return

    for (note, text), raw in zip(items, raw_results):
        try:
            results[note.note_id] = _normalize_extraction([raw], text)
        except Exception as exc:
            _LOGGER.warning("GLiNER2 result parsing failed for note %s: %s", note.note_id, exc)
            results[note.note_id] = Extraction(failed=True)


def _extract_one(model: Any, schema: Any, note: NoteRecord, text: str) -> Extraction:
    try:
        raw_results = [
            model.extract(
                chunk,
                schema,
                threshold=_global_threshold(),
                include_confidence=True,
                max_len=_chunk_max_len(),
            )
            for chunk in _text_chunks(text)
        ]
    except Exception as exc:
        _LOGGER.warning("GLiNER2 extraction failed for note %s: %s", note.note_id, exc)
        return Extraction(failed=True)

    try:
        return _normalize_extraction(raw_results, text)
    except Exception as exc:
        _LOGGER.warning("GLiNER2 result parsing failed for note %s: %s", note.note_id, exc)
        return Extraction(failed=True)


def _normalize_extraction(raw_results: list[object], note_text: str) -> Extraction:
    entity_acc: dict[tuple[str, str], _EntityAccumulator] = {}
    entity_order: list[tuple[str, str]] = []

    for raw in raw_results:
        for surface, etype, conf in _iter_entities(raw):
            if etype not in config.INGEST_ENTITY_TYPES:
                continue
            surface = surface.strip()
            if not surface or conf < config.GLINER_ENTITY_MIN_CONF:
                continue
            key = (_lower_normalized(surface), etype)
            current = entity_acc.get(key)
            if current is None:
                entity_acc[key] = _EntityAccumulator(surface=surface, conf=conf)
                entity_order.append(key)
            else:
                if conf > current.conf:
                    current.surface = surface
                    current.conf = conf

    entities = [
        EntityMention(
            surface=entity_acc[key].surface,
            etype=key[1],
            conf=entity_acc[key].conf,
            mentions=_surface_mentions(entity_acc[key].surface, note_text),
        )
        for key in entity_order
    ]

    entity_surfaces: dict[str, str] = {}
    for entity in entities:
        entity_surfaces.setdefault(_lower_normalized(entity.surface), entity.surface)

    relation_acc: dict[tuple[str, str, str], _RelationAccumulator] = {}
    relation_order: list[tuple[str, str, str]] = []
    for raw in raw_results:
        for head, rel_type, tail, conf in _iter_relations(raw):
            if rel_type not in config.INGEST_RELATION_TYPES:
                continue
            head_key = _lower_normalized(head)
            tail_key = _lower_normalized(tail)
            if (
                not head_key
                or not tail_key
                or conf < config.GLINER_RELATION_MIN_CONF
                or head_key not in entity_surfaces
                or tail_key not in entity_surfaces
            ):
                continue
            surface_head = entity_surfaces[head_key]
            surface_tail = entity_surfaces[tail_key]
            key = (head_key, rel_type, tail_key)
            current = relation_acc.get(key)
            if current is None:
                relation_acc[key] = _RelationAccumulator(
                    head=surface_head,
                    rel_type=rel_type,
                    tail=surface_tail,
                    conf=conf,
                )
                relation_order.append(key)
            elif conf > current.conf:
                current.head = surface_head
                current.tail = surface_tail
                current.conf = conf

    relations = [
        RelationTriple(
            head=relation_acc[key].head,
            rel_type=relation_acc[key].rel_type,
            tail=relation_acc[key].tail,
            conf=relation_acc[key].conf,
        )
        for key in relation_order
    ]
    return Extraction(entities=entities, relations=relations)


def _iter_entities(raw: object) -> list[tuple[str, str, float]]:
    if not isinstance(raw, dict):
        return []
    entities = raw.get("entities")
    if not isinstance(entities, dict):
        return []

    found: list[tuple[str, str, float]] = []
    for etype, values in entities.items():
        if isinstance(values, (str, dict)):
            values = [values]
        if not isinstance(values, list):
            continue
        for item in values:
            surface, conf = _span_text_conf(item)
            if surface:
                found.append((surface, str(etype), conf))
    return found


def _iter_relations(raw: object) -> list[tuple[str, str, str, float]]:
    if not isinstance(raw, dict):
        return []

    rel_map = raw.get("relation_extraction")
    if not isinstance(rel_map, dict):
        rel_map = {
            rel_type: raw.get(rel_type)
            for rel_type in config.INGEST_RELATION_TYPES
            if rel_type in raw
        }
    if not isinstance(rel_map, dict):
        return []

    found: list[tuple[str, str, str, float]] = []
    for rel_type, values in rel_map.items():
        if isinstance(values, (tuple, dict)):
            values = [values]
        if not isinstance(values, list):
            continue
        for item in values:
            parsed = _relation_item(item)
            if parsed is not None:
                head, tail, conf = parsed
                found.append((head, str(rel_type), tail, conf))
    return found


def _relation_item(item: object) -> tuple[str, str, float] | None:
    if isinstance(item, tuple) and len(item) == 2:
        return str(item[0]), str(item[1]), 1.0
    if isinstance(item, list) and len(item) == 2:
        return str(item[0]), str(item[1]), 1.0
    if not isinstance(item, dict):
        return None

    head, head_conf = _span_text_conf(item.get("head"))
    tail, tail_conf = _span_text_conf(item.get("tail"))
    if not head or not tail:
        return None
    item_conf = _confidence(item, default=min(head_conf, tail_conf))
    return head, tail, item_conf


def _span_text_conf(item: object) -> tuple[str, float]:
    if isinstance(item, str):
        return item.strip(), 1.0
    if isinstance(item, dict):
        text = item.get("text", item.get("surface", item.get("span", "")))
        if not isinstance(text, str):
            return "", 0.0
        return text.strip(), _confidence(item)
    return "", 0.0


def _confidence(item: dict[str, Any], *, default: float = 1.0) -> float:
    value = item.get("confidence", item.get("conf", item.get("score", default)))
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _batches(notes: list[NoteRecord], size: int) -> list[list[NoteRecord]]:
    return [notes[index : index + size] for index in range(0, len(notes), size)]


def _note_text(note: NoteRecord) -> str:
    return f"{note.title}\n{note.text}"


def _surface_mentions(surface: str, text: str) -> int:
    pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)", re.IGNORECASE)
    return max(1, len(pattern.findall(text)))


def _is_long_text(text: str) -> bool:
    size = _chunk_size()
    return size > 0 and len(_tokens(text)) > size


def _text_chunks(text: str) -> list[str]:
    tokens = _tokens(text)
    size = _chunk_size()
    if size <= 0 or len(tokens) <= size:
        return [text]

    overlap = max(0, min(config.GLINER_CHUNK_OVERLAP, size - 1))
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _global_threshold() -> float:
    return min(config.GLINER_ENTITY_MIN_CONF, config.GLINER_RELATION_MIN_CONF)


def _ingest_batch_size() -> int:
    return max(1, int(config.GLINER_INGEST_BATCH))


def _chunk_size() -> int:
    return max(0, int(config.GLINER_CHUNK_SIZE))


def _chunk_max_len() -> int | None:
    size = _chunk_size()
    return size if size > 0 else None


def _lower_normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
