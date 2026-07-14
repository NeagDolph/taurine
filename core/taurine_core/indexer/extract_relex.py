"""GLiNER-Relex note extraction and GLiNER2/Relex hybrid composition."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from taurine_core import config
from taurine_core.indexer.facts import PREDICATES
from taurine_core.models_local import _to_inference_device, model_ref
from taurine_core.schemas import (
    ExtractedMention,
    ExtractedRelation,
    Extraction,
    NoteRecord,
    RelationTriple,
)


_LOG = logging.getLogger("taurine.indexer.extract_relex")
_MODEL: Any | None = None
_MODEL_LOAD_FAILED = False


def extract_notes(
    notes: list[NoteRecord],
    *,
    tokenizer: Any = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Extraction]:
    """Extract whole-note entities and relations with isolated failures."""
    if not notes:
        return {}
    model = _relex_model()
    if model is None:
        return {
            note.note_id: Extraction(failed=True, error="relex unavailable")
            for note in notes
        }

    results: dict[str, Extraction] = {}
    for note in notes:
        try:
            text, entities, relations = _predict_with_oom_fallback(
                model, note.text, tokenizer
            )
            results[note.note_id] = _normalize_result(text, entities, relations)
        except Exception as exc:  # one bad note/model response never fails the build
            _LOG.warning("Relex extraction failed for note %s: %s", note.note_id, exc)
            results[note.note_id] = Extraction(
                failed=True, error="relex inference failed"
            )
        finally:
            _release_device_cache()
        if progress is not None:
            progress(len(results), len(notes))
    return {note.note_id: results[note.note_id] for note in notes}


def _relex_model() -> Any | None:
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None
    try:
        _MODEL = _load_model()
    except Exception as exc:
        _MODEL_LOAD_FAILED = True
        _LOG.warning("Relex model load failed: %s", exc)
        return None
    return _MODEL


def _load_model() -> Any:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_ref(config.RELEX_MODEL))
    return _to_inference_device(model)


def _release_device_cache() -> None:
    """Bound per-note MPS intermediates while retaining the cached model."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _predict_with_oom_fallback(
    model: Any,
    note_text: str,
    tokenizer: Any,
) -> tuple[str, object, object]:
    text = _bounded_text(note_text, tokenizer)
    try:
        entities, relations = _predict(model, text)
    except Exception as exc:
        if "mps backend out of memory" not in str(exc).casefold():
            raise
        _LOG.warning("Relex MPS OOM; moving model to CPU and retrying full context")
        model.to("cpu")
        _release_device_cache()
        entities, relations = _predict(model, text)
    return text, entities, relations


def _predict(model: Any, text: str) -> tuple[object, object]:
    return model.predict_relations(
        text,
        labels=list(config.INGEST_ENTITY_TYPES),
        relations=list(PREDICATES),
        flat_ner=True,
        threshold=float(config.RELEX_ENTITY_THRESHOLD),
        adjacency_threshold=float(config.RELEX_ADJACENCY_THRESHOLD),
        relation_threshold=float(config.RELEX_RELATION_THRESHOLD),
        multi_label=False,
    )


def _bounded_text(text: str, tokenizer: Any, *, maximum: int | None = None) -> str:
    maximum = max(
        1,
        int(config.RELEX_MAX_TOKENS) if maximum is None else int(maximum),
    )
    if tokenizer is not None:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = [
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
            if int(end) > int(start)
        ]
    else:
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    if len(offsets) <= maximum:
        return text
    return text[: offsets[maximum - 1][1]]


def _schema_id() -> str:
    payload = {
        "model": config.RELEX_MODEL,
        "entities": config.INGEST_ENTITY_TYPES,
        "predicates": {
            name: {
                "description": spec.description,
                "subject_etypes": sorted(spec.subject_etypes),
                "object_etypes": sorted(spec.object_etypes),
            }
            for name, spec in sorted(PREDICATES.items())
        },
        "thresholds": {
            "entity": config.RELEX_ENTITY_THRESHOLD,
            "relation": config.RELEX_RELATION_THRESHOLD,
            "adjacency": config.RELEX_ADJACENCY_THRESHOLD,
        },
        "max_tokens": config.RELEX_MAX_TOKENS,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"relex-{digest}"


def _normalize_result(
    text: str,
    raw_entities: object,
    raw_relations: object,
) -> Extraction:
    schema_id = _schema_id()
    mentions_by_key: dict[tuple[str, int, int, str], ExtractedMention] = {}
    entity_rows = raw_entities if isinstance(raw_entities, list) else []
    for raw in entity_rows:
        mention = _mention(raw, text, schema_id)
        if mention is not None and mention.confidence >= float(
            config.RELEX_ENTITY_THRESHOLD
        ):
            _keep_mention(mentions_by_key, mention)

    facts_by_key: dict[tuple[str, int, int, int, int], ExtractedRelation] = {}
    relation_rows = raw_relations if isinstance(raw_relations, list) else []
    for raw in relation_rows:
        parsed = _relation(raw, entity_rows, text, schema_id)
        if parsed is None:
            continue
        fact, endpoints = parsed
        _keep_mention(mentions_by_key, endpoints[0])
        _keep_mention(mentions_by_key, endpoints[1])
        key = (
            fact.predicate,
            fact.subject_start,
            fact.subject_end,
            fact.object_start,
            fact.object_end,
        )
        current = facts_by_key.get(key)
        if current is None or fact.confidence > current.confidence:
            facts_by_key[key] = fact

    mentions = sorted(
        mentions_by_key.values(),
        key=lambda row: (
            row.char_start,
            row.char_end,
            row.etype,
            row.surface.casefold(),
        ),
    )
    facts = sorted(
        facts_by_key.values(),
        key=lambda row: (
            row.assertion_start,
            row.predicate,
            row.subject_start,
            row.object_start,
        ),
    )
    return _extraction_from_rows(mentions, facts, schema_id)


def _mention(raw: object, text: str, schema_id: str) -> ExtractedMention | None:
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label", raw.get("type", ""))).strip().casefold()
    span = _valid_span(raw, text)
    if label not in config.INGEST_ENTITY_TYPES or span is None:
        return None
    score = _score(raw, 0.0)
    start, end, surface = span
    return ExtractedMention(
        surface=surface,
        etype=label,
        confidence=score,
        char_start=start,
        char_end=end,
        extractor=config.RELEX_MODEL,
        schema_id=schema_id,
    )


def _relation(
    raw: object,
    entities: list[object],
    text: str,
    schema_id: str,
) -> tuple[ExtractedRelation, tuple[ExtractedMention, ExtractedMention]] | None:
    if not isinstance(raw, dict):
        return None
    predicate = str(raw.get("relation", "")).strip().casefold().replace("-", "_")
    spec = PREDICATES.get(predicate)
    score = _score(raw, 0.0)
    if spec is None or score < float(config.RELEX_RELATION_THRESHOLD):
        return None
    head_raw = raw.get("head")
    tail_raw = raw.get("tail")
    head = _endpoint_mention(head_raw, entities, text, schema_id, score)
    tail = _endpoint_mention(tail_raw, entities, text, schema_id, score)
    if head is None or tail is None:
        return None
    # predict_relations has no pairs_filter in gliner 0.2.27. Enforce the
    # registry's subject/object etype constraints from endpoint `type` fields.
    if head.etype not in spec.subject_etypes or tail.etype not in spec.object_etypes:
        return None
    if head.char_start == tail.char_start and head.char_end == tail.char_end:
        return None
    assertion_start = min(head.char_start, tail.char_start)
    assertion_end = max(head.char_end, tail.char_end)
    if assertion_end <= assertion_start:
        return None
    # Relex intentionally does not infer historical relations. Keep historical
    # false/active here and let the existing temporal reconciliation own history.
    fact = ExtractedRelation(
        head=head.surface,
        predicate=predicate,
        tail=tail.surface,
        confidence=score,
        subject_start=head.char_start,
        subject_end=head.char_end,
        object_start=tail.char_start,
        object_end=tail.char_end,
        assertion_start=assertion_start,
        assertion_end=assertion_end,
        extraction_method="relex",
        schema_id=schema_id,
        status="active" if score >= float(config.REL_FACT_MIN_CONF) else "candidate",
    )
    return fact, (head, tail)


def _endpoint_mention(
    raw: object,
    entities: list[object],
    text: str,
    schema_id: str,
    relation_score: float,
) -> ExtractedMention | None:
    if not isinstance(raw, dict):
        return None
    endpoint = dict(raw)
    endpoint["label"] = raw.get("type", raw.get("label", ""))
    entity_index = raw.get("entity_idx")
    if isinstance(entity_index, int) and 0 <= entity_index < len(entities):
        entity = entities[entity_index]
        if isinstance(entity, dict):
            endpoint["score"] = entity.get(
                "score", entity.get("confidence", relation_score)
            )
    endpoint.setdefault("score", relation_score)
    return _mention(endpoint, text, schema_id)


def _valid_span(raw: dict[str, Any], text: str) -> tuple[int, int, str] | None:
    start = raw.get("start")
    end = raw.get("end")
    surface = raw.get("text")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(surface, str)
        or start < 0
        or end <= start
        or end > len(text)
        or text[start:end] != surface
    ):
        return None
    return start, end, surface


def _score(raw: dict[str, Any], default: float) -> float:
    value = raw.get("score", raw.get("confidence", default))
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else default
    )


def _keep_mention(
    mentions: dict[tuple[str, int, int, str], ExtractedMention],
    mention: ExtractedMention,
) -> None:
    key = (
        mention.etype,
        mention.char_start,
        mention.char_end,
        mention.surface.casefold(),
    )
    current = mentions.get(key)
    if current is None or mention.confidence > current.confidence:
        mentions[key] = mention


def _extraction_from_rows(
    mentions: list[ExtractedMention],
    facts: list[ExtractedRelation],
    schema_id: str,
) -> Extraction:
    from taurine_core.indexer import extract as gliner_extract

    return Extraction(
        entities=gliner_extract._legacy_entities(mentions),
        relations=[
            RelationTriple(
                head=row.head,
                rel_type=row.predicate,
                tail=row.tail,
                conf=row.confidence,
            )
            for row in facts
            if gliner_extract._valid_relation_endpoints(row.head, row.tail)
        ],
        mentions=mentions,
        fact_candidates=facts,
        schema_id=schema_id,
    )


def merge_hybrid(gliner: Extraction, relex: Extraction) -> Extraction:
    """Use GLiNER2 entities and Relex relations, adding missed endpoints only."""
    if relex.failed:
        return gliner
    if gliner.failed:
        return relex
    endpoint_keys = {
        (row.subject_start, row.subject_end, row.head.casefold())
        for row in relex.fact_candidates
    } | {
        (row.object_start, row.object_end, row.tail.casefold())
        for row in relex.fact_candidates
    }
    mentions = list(gliner.mentions)
    existing = {
        (row.etype, row.char_start, row.char_end, row.surface.casefold())
        for row in mentions
    }
    for row in relex.mentions:
        key = (row.etype, row.char_start, row.char_end, row.surface.casefold())
        endpoint = (row.char_start, row.char_end, row.surface.casefold())
        if endpoint in endpoint_keys and key not in existing:
            mentions.append(row)
            existing.add(key)
    mentions.sort(
        key=lambda row: (
            row.char_start,
            row.char_end,
            row.etype,
            row.surface.casefold(),
        )
    )
    schema_id = (
        "hybrid-"
        + hashlib.sha256(
            f"{gliner.schema_id}\x1f{relex.schema_id}".encode()
        ).hexdigest()[:24]
    )
    facts = [
        row.model_copy(
            update={"extraction_method": "relex_hybrid", "schema_id": schema_id}
        )
        for row in relex.fact_candidates
    ]
    return _extraction_from_rows(mentions, facts, schema_id)


def merge_union(gliner: Extraction, relex: Extraction) -> Extraction:
    """Both extractors' relations, GLiNER2-first mentions (judge filters later).

    Judge-sampled precision measured the two novel-fact pools at equal quality
    (2026-07-14), so the union dominates either alone: recall is at least
    GLiNER2's, yield is both pools, and adjudication stays the quality gate.
    Duplicate logical triples keep the higher-confidence row."""
    if relex.failed:
        return gliner
    if gliner.failed:
        return relex
    hybrid = merge_hybrid(gliner, relex)
    by_triple: dict[tuple[str, str, str], ExtractedRelation] = {}
    for row in (*gliner.fact_candidates, *hybrid.fact_candidates):
        key = (row.head.casefold(), row.predicate.casefold(), row.tail.casefold())
        current = by_triple.get(key)
        if current is None or row.confidence > current.confidence:
            by_triple[key] = row
    schema_id = (
        "union-"
        + hashlib.sha256(
            f"{gliner.schema_id}\x1f{relex.schema_id}".encode()
        ).hexdigest()[:24]
    )
    facts = [
        row.model_copy(update={"schema_id": schema_id})
        for _, row in sorted(
            by_triple.items(), key=lambda item: (item[1].assertion_start, item[0])
        )
    ]
    return _extraction_from_rows(list(hybrid.mentions), facts, schema_id)
