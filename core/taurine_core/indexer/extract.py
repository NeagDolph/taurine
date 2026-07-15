"""GLiNER2 batch extraction with source-global mention and fact provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from taurine_core import config
from taurine_core.evaluation.graph_retrieval import CalibrationArtifact, load_calibration
from taurine_core.live import spans
from taurine_core.schemas import (
    EntityMention,
    ExtractedMention,
    ExtractedRelation,
    Extraction,
    NoteRecord,
    RelationTriple,
)

_LOGGER = logging.getLogger("taurine.indexer.extract")
_EXTRACTOR_ID = "fastino/gliner2-large-v1"
_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "evals" / "graph_retrieval" / "calibration-v1.json"
)


class OffsetTokenizerPort(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, Sequence[Any]]: ...


@dataclass(frozen=True)
class _SchemaBundle:
    family: str
    schema_id: str
    schema: Any


@dataclass(frozen=True)
class _Pending:
    note: NoteRecord
    text: str
    offsets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Span:
    text: str
    confidence: float
    start: int | None
    end: int | None
    explicit: bool


@dataclass(frozen=True)
class _RelationRaw:
    head: _Span
    predicate: str
    tail: _Span
    confidence: float
    observed_at: float | None = None
    valid_from: float | None = None
    valid_to: float | None = None
    historical: bool = False


def extract_notes(
    notes: list[NoteRecord],
    *,
    tokenizer: OffsetTokenizerPort | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Extraction]:
    """Extract notes in bounded source-family batches with isolated failures."""
    if not notes:
        return {}
    model = _extractor_model()
    if model is None:
        return {
            note.note_id: Extraction(failed=True, error="GLiNER2 unavailable")
            for note in notes
        }

    grouped: dict[str, list[NoteRecord]] = {}
    for note in notes:
        grouped.setdefault(_source_family(note.source), []).append(note)

    results: dict[str, Extraction] = {}

    def report_progress() -> None:
        if progress is not None:
            progress(len(results), len(notes))

    for family, family_notes in grouped.items():
        try:
            bundle = _build_schema(model, family)
        except Exception as exc:
            _LOGGER.warning("GLiNER2 schema build failed for %s: %s", family, exc)
            for note in family_notes:
                results[note.note_id] = Extraction(
                    failed=True,
                    error="schema build failed",
                )
            report_progress()
            continue
        _extract_family(model, bundle, family_notes, tokenizer, results, report_progress)

    return {note.note_id: results[note.note_id] for note in notes}


def extract_notes_dispatch(
    notes: list[NoteRecord],
    *,
    tokenizer: OffsetTokenizerPort | None = None,
    progress: Callable[[int, int], None] | None = None,
    backend: str | None = None,
) -> dict[str, Extraction]:
    """Select the configured extractor while preserving GLiNER2 by default."""
    selected = backend or config.EXTRACTOR_BACKEND
    if selected == "gliner2":
        return extract_notes(notes, tokenizer=tokenizer, progress=progress)

    from taurine_core.indexer import extract_relex

    if selected == "relex":
        return extract_relex.extract_notes(notes, tokenizer=tokenizer, progress=progress)
    if selected in {"hybrid", "union"}:
        # Both passes report through the caller's progress callback, mapping
        # pass 1 to [0, N/2] and pass 2 to [N/2, N]. This is not cosmetic:
        # the sync status UI feeds off these ticks, and the build's cancel
        # check runs inside the callback — with progress=None a full-corpus
        # two-model extraction shows "0/N" for its entire runtime and cannot
        # be cancelled until it finishes.
        total = len(notes)
        pass1 = (
            None
            if progress is None
            else lambda done, _t: progress(done // 2, total)
        )
        gliner_results = extract_notes(notes, tokenizer=tokenizer, progress=pass1)
        # Hybrid needs GLiNER2 outputs, not both multi-GB models resident at
        # once. Release its process cache before loading Relex; active callers
        # retain their own references and later span queries reload lazily.
        spans.release_shared_model()
        pass2 = (
            None
            if progress is None
            else lambda done, _t: progress(total // 2 + done // 2, total)
        )
        relex_results = extract_relex.extract_notes(notes, tokenizer=tokenizer, progress=pass2)
        combine = (
            extract_relex.merge_union
            if selected == "union"
            else extract_relex.merge_hybrid
        )
        merged = {
            note.note_id: combine(
                gliner_results.get(
                    note.note_id,
                    Extraction(failed=True, error="GLiNER2 result missing"),
                ),
                relex_results.get(
                    note.note_id,
                    Extraction(failed=True, error="relex result missing"),
                ),
            )
            for note in notes
        }
        if progress is not None:
            progress(len(merged), len(notes))
        return merged
    _LOGGER.warning("unknown extractor backend %r; using GLiNER2", selected)
    return extract_notes(notes, tokenizer=tokenizer, progress=progress)


def _extractor_model() -> Any | None:
    return spans.shared_model()


def _build_schema(model: Any, family: str = "written") -> _SchemaBundle:
    entity_descriptions = dict(config.INGEST_ENTITY_TYPES)
    relation_descriptions = _relation_descriptions(family)
    schema_data = {
        "family": family,
        "entities": entity_descriptions,
        "relations": relation_descriptions,
    }
    schema_id = "schema-" + hashlib.sha256(
        json.dumps(schema_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    schema = model.create_schema().entities(entity_descriptions).relations(relation_descriptions)
    return _SchemaBundle(family=family, schema_id=schema_id, schema=schema)


def _extract_family(
    model: Any,
    bundle: _SchemaBundle,
    notes: list[NoteRecord],
    tokenizer: OffsetTokenizerPort | None,
    results: dict[str, Extraction],
    report_progress: Callable[[], None] | None = None,
) -> None:
    def progressed() -> None:
        if report_progress is not None:
            report_progress()

    short_batch: list[_Pending] = []
    batch_tokens = 0
    for note in notes:
        text = note.text
        try:
            offsets = _token_offsets(model, text, tokenizer)
        except Exception as exc:
            _LOGGER.warning("GLiNER2 tokenization failed for note %s: %s", note.note_id, exc)
            results[note.note_id] = Extraction(
                schema_id=bundle.schema_id,
                failed=True,
                error="tokenization failed",
            )
            progressed()
            continue

        if _is_long(offsets):
            _flush_short_batch(model, bundle, short_batch, results)
            short_batch = []
            batch_tokens = 0
            progressed()
            results[note.note_id] = _extract_long(
                model,
                bundle,
                note,
                text,
                offsets,
            )
            progressed()
            continue

        pending = _Pending(note=note, text=text, offsets=offsets)
        token_count = len(offsets)
        would_overflow = bool(short_batch) and (
            len(short_batch) >= _ingest_batch_size()
            or batch_tokens + token_count > _batch_token_budget()
        )
        if would_overflow:
            _flush_short_batch(model, bundle, short_batch, results)
            short_batch = []
            batch_tokens = 0
            progressed()
        short_batch.append(pending)
        batch_tokens += token_count
    _flush_short_batch(model, bundle, short_batch, results)
    progressed()


def _flush_short_batch(
    model: Any,
    bundle: _SchemaBundle,
    items: list[_Pending],
    results: dict[str, Extraction],
) -> None:
    if not items:
        return
    texts = [item.text for item in items]
    try:
        raw_results = model.batch_extract(
            texts,
            bundle.schema,
            batch_size=len(texts),
            threshold=_global_threshold(),
            include_confidence=True,
            include_spans=True,
            max_len=_chunk_size(),
        )
        if not isinstance(raw_results, list) or len(raw_results) != len(items):
            raise ValueError("GLiNER2 batch result length mismatch")
    except Exception as exc:
        _LOGGER.warning("GLiNER2 batch extraction failed; retrying per note: %s", exc)
        for item in items:
            results[item.note.note_id] = _extract_short_one(model, bundle, item)
        return

    for item, raw in zip(items, raw_results, strict=True):
        results[item.note.note_id] = _safe_normalize(
            [raw],
            item.text,
            bundle.schema_id,
            "gliner2_batch",
        )


def _extract_short_one(model: Any, bundle: _SchemaBundle, item: _Pending) -> Extraction:
    try:
        raw = model.extract(
            item.text,
            bundle.schema,
            threshold=_global_threshold(),
            include_confidence=True,
            include_spans=True,
            max_len=_chunk_size(),
        )
    except Exception as exc:
        _LOGGER.warning("GLiNER2 extraction failed for note %s: %s", item.note.note_id, exc)
        return Extraction(
            schema_id=bundle.schema_id,
            failed=True,
            error="inference failed",
        )
    return _safe_normalize(
        [raw],
        item.text,
        bundle.schema_id,
        "gliner2_single",
    )


def _extract_long(
    model: Any,
    bundle: _SchemaBundle,
    note: NoteRecord,
    text: str,
    offsets: tuple[tuple[int, int], ...],
) -> Extraction:
    try:
        if hasattr(model, "batch_extract_long"):
            raw_results = model.batch_extract_long(
                [text],
                bundle.schema,
                chunk_size=_chunk_size(),
                chunk_overlap=_chunk_overlap(),
                threshold=_global_threshold(),
                include_confidence=True,
                include_spans=True,
            )
            if not isinstance(raw_results, list) or len(raw_results) != 1:
                raise ValueError("GLiNER2 long result length mismatch")
            return _safe_normalize(
                [raw_results[0]],
                text,
                bundle.schema_id,
                "gliner2_batch_long",
            )

        windows = _long_windows(text, offsets)
        raw_windows = model.batch_extract(
            [window_text for window_text, _ in windows],
            bundle.schema,
            batch_size=min(_ingest_batch_size(), len(windows)),
            threshold=_global_threshold(),
            include_confidence=True,
            include_spans=True,
            max_len=_chunk_size(),
        )
        if not isinstance(raw_windows, list) or len(raw_windows) != len(windows):
            raise ValueError("GLiNER2 compatibility long result length mismatch")
        remapped = [
            _shift_spans(raw, char_start)
            for raw, (_, char_start) in zip(raw_windows, windows, strict=True)
        ]
        return _safe_normalize(
            remapped,
            text,
            bundle.schema_id,
            "gliner2_token_window_batch",
        )
    except Exception as exc:
        _LOGGER.warning("GLiNER2 long extraction failed for note %s: %s", note.note_id, exc)
        return Extraction(
            schema_id=bundle.schema_id,
            failed=True,
            error="long inference failed",
        )


def _safe_normalize(
    raw_results: list[object],
    text: str,
    schema_id: str,
    method: str,
) -> Extraction:
    try:
        return _normalize_extraction(raw_results, text, schema_id, method)
    except Exception as exc:
        _LOGGER.warning("GLiNER2 result parsing failed: %s", exc)
        return Extraction(schema_id=schema_id, failed=True, error="result parsing failed")


def _normalize_extraction(
    raw_results: list[object],
    note_text: str,
    schema_id: str = "",
    method: str = "gliner2",
) -> Extraction:
    calibration = _calibration()
    mentions_by_key: dict[tuple[str, int, int, str], ExtractedMention] = {}
    for raw in raw_results:
        for span, etype in _iter_entities(raw):
            if etype not in config.INGEST_ENTITY_TYPES:
                continue
            floor = calibration.extraction.entity_thresholds.get(
                etype, config.GLINER_ENTITY_MIN_CONF
            )
            if span.confidence < floor:
                continue
            for resolved in _resolve_span(span, note_text):
                key = (etype, resolved.start or 0, resolved.end or 0, resolved.text.casefold())
                row = ExtractedMention(
                    surface=resolved.text,
                    etype=etype,
                    confidence=resolved.confidence,
                    char_start=int(resolved.start or 0),
                    char_end=int(resolved.end or 0),
                    extractor=_EXTRACTOR_ID,
                    schema_id=schema_id,
                )
                current = mentions_by_key.get(key)
                if current is None or row.confidence > current.confidence:
                    mentions_by_key[key] = row

    mentions = sorted(
        mentions_by_key.values(),
        key=lambda row: (row.char_start, row.char_end, row.etype, row.surface.casefold()),
    )
    entities = _legacy_entities(mentions)

    facts_by_key: dict[tuple[str, int, int, int, int], ExtractedRelation] = {}
    for raw in raw_results:
        for relation in _iter_relations(raw):
            if relation.predicate not in config.INGEST_RELATION_TYPES:
                continue
            floor = calibration.extraction.relation_thresholds.get(
                relation.predicate, config.GLINER_RELATION_MIN_CONF
            )
            if relation.confidence < floor:
                continue
            fact = _resolve_relation(
                relation,
                mentions,
                note_text,
                schema_id,
                method,
                calibration,
            )
            if fact is None:
                continue
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

    fact_candidates = sorted(
        facts_by_key.values(),
        key=lambda row: (
            row.assertion_start,
            row.predicate,
            row.subject_start,
            row.object_start,
        ),
    )
    relations = [
        RelationTriple(
            head=row.head,
            rel_type=row.predicate,
            tail=row.tail,
            conf=row.confidence,
        )
        for row in fact_candidates
        if _valid_relation_endpoints(row.head, row.tail)
    ]
    return Extraction(
        entities=entities,
        relations=relations,
        mentions=mentions,
        fact_candidates=fact_candidates,
        schema_id=schema_id,
    )


def _iter_entities(raw: object) -> list[tuple[_Span, str]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("entities"), dict):
        return []
    found: list[tuple[_Span, str]] = []
    for etype, values in raw["entities"].items():
        if isinstance(values, (str, dict)):
            values = [values]
        if not isinstance(values, list):
            continue
        for item in values:
            span = _span(item)
            if span.text:
                found.append((span, str(etype)))
    return found


def _iter_relations(raw: object) -> list[_RelationRaw]:
    if not isinstance(raw, dict):
        return []
    values_by_type = raw.get("relation_extraction")
    if not isinstance(values_by_type, dict):
        values_by_type = {
            predicate: raw[predicate]
            for predicate in config.INGEST_RELATION_TYPES
            if predicate in raw
        }
    found: list[_RelationRaw] = []
    for predicate, values in values_by_type.items():
        if isinstance(values, (tuple, dict)):
            values = [values]
        if not isinstance(values, list):
            continue
        for item in values:
            parsed = _relation(item, str(predicate))
            if parsed is not None:
                found.append(parsed)
    return found


def _relation(item: object, predicate: str) -> _RelationRaw | None:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        # legacy tuple shape carries no confidence: fail closed (L7)
        return _RelationRaw(_span(item[0]), predicate, _span(item[1]), 0.0)
    if not isinstance(item, dict):
        return None
    head = _span(item.get("head"))
    tail = _span(item.get("tail"))
    if not head.text or not tail.text:
        return None
    confidence = _confidence(item, default=min(head.confidence, tail.confidence))
    return _RelationRaw(
        head,
        predicate,
        tail,
        confidence,
        observed_at=_temporal_value(item.get("observed_at")),
        valid_from=_temporal_value(item.get("valid_from")),
        valid_to=_temporal_value(item.get("valid_to")),
        historical=str(item.get("status") or "").casefold() == "historical",
    )


def _temporal_value(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _span(item: object) -> _Span:
    if isinstance(item, str):
        # bare-string shape carries no confidence: fail closed (L7)
        return _Span(item.strip(), 0.0, None, None, False)
    if not isinstance(item, dict):
        return _Span("", 0.0, None, None, False)
    value = item.get("text", item.get("surface", item.get("span", "")))
    text = value.strip() if isinstance(value, str) else ""
    start = item.get("start")
    end = item.get("end")
    explicit = isinstance(start, int) and isinstance(end, int)
    return _Span(
        text=text,
        confidence=_confidence(item),
        start=int(start) if explicit else None,
        end=int(end) if explicit else None,
        explicit=explicit,
    )


def _resolve_span(span: _Span, text: str) -> tuple[_Span, ...]:
    if span.explicit:
        if (
            span.start is None
            or span.end is None
            or span.start < 0
            or span.end <= span.start
            or span.end > len(text)
            or text[span.start : span.end] != span.text
        ):
            return ()
        return (span,)
    pattern = re.compile(rf"(?<!\w){re.escape(span.text)}(?!\w)", re.IGNORECASE)
    return tuple(
        _Span(
            text=text[match.start() : match.end()],
            confidence=span.confidence,
            start=match.start(),
            end=match.end(),
            explicit=False,
        )
        for match in pattern.finditer(text)
    )


def _legacy_entities(mentions: list[ExtractedMention]) -> list[EntityMention]:
    grouped: dict[tuple[str, str], list[ExtractedMention]] = {}
    order: list[tuple[str, str]] = []
    for mention in mentions:
        key = (mention.surface.casefold(), mention.etype)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(mention)
    return [
        EntityMention(
            surface=max(grouped[key], key=lambda row: row.confidence).surface,
            etype=key[1],
            conf=max(row.confidence for row in grouped[key]),
            mentions=len(grouped[key]),
        )
        for key in order
    ]


def _resolve_relation(
    relation: _RelationRaw,
    mentions: list[ExtractedMention],
    text: str,
    schema_id: str,
    method: str,
    calibration: CalibrationArtifact,
) -> ExtractedRelation | None:
    heads = _endpoint_candidates(relation.head, mentions, text)
    tails = _endpoint_candidates(relation.tail, mentions, text)
    pairs: list[tuple[ExtractedMention, ExtractedMention, tuple[int, int]]] = []
    for head in heads:
        for tail in tails:
            if head.etype != tail.etype or head.surface.casefold() != tail.surface.casefold():
                assertion = _assertion_span(text, head.char_start, head.char_end, tail.char_start, tail.char_end)
                if assertion is not None:
                    pairs.append((head, tail, assertion))
    if not pairs:
        return None
    pairs.sort(
        key=lambda pair: (
            abs(pair[0].char_start - pair[1].char_start),
            pair[0].char_start,
            pair[1].char_start,
        )
    )
    head, tail, assertion = pairs[0]
    ambiguous = len(pairs) > 1
    promotion_floor = calibration.extraction.promotion_thresholds.get(
        relation.predicate, config.REL_FACT_MIN_CONF
    )
    return ExtractedRelation(
        head=head.surface,
        predicate=relation.predicate,
        tail=tail.surface,
        confidence=relation.confidence,
        subject_start=head.char_start,
        subject_end=head.char_end,
        object_start=tail.char_start,
        object_end=tail.char_end,
        assertion_start=assertion[0],
        assertion_end=assertion[1],
        extraction_method=method,
        schema_id=schema_id,
        status=(
            "historical"
            if relation.historical and relation.confidence >= promotion_floor and not ambiguous
            else "active"
            if relation.confidence >= promotion_floor and not ambiguous
            else "candidate"
        ),
        observed_at=relation.observed_at,
        valid_from=relation.valid_from,
        valid_to=relation.valid_to,
    )


def _endpoint_candidates(
    span: _Span,
    mentions: list[ExtractedMention],
    text: str,
) -> list[ExtractedMention]:
    candidates = [row for row in mentions if row.surface.casefold() == span.text.casefold()]
    if not span.explicit:
        return candidates
    if (
        span.start is None
        or span.end is None
        or span.start < 0
        or span.end > len(text)
        or text[span.start : span.end] != span.text
    ):
        return []
    return [
        row
        for row in candidates
        if row.char_start == span.start and row.char_end == span.end
    ]


def _assertion_span(
    text: str,
    subject_start: int,
    subject_end: int,
    object_start: int,
    object_end: int,
) -> tuple[int, int] | None:
    left = min(subject_start, object_start)
    right = max(subject_end, object_end)
    for start, end in _sentence_spans(text):
        if start <= left and right <= end:
            return start, end
    separator = text.rfind("\n\n", 0, left)
    paragraph_start = separator + 2 if separator >= 0 else 0
    paragraph_end = text.find("\n\n", right)
    if paragraph_end < 0:
        paragraph_end = len(text)
    if paragraph_start <= left and right <= paragraph_end:
        return paragraph_start, paragraph_end
    return None


def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans_out: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?](?:[\"')\]]*)(?:\s+|$)", text):
        end = match.start() + len(match.group(0).rstrip())
        if end > start:
            spans_out.append((start, end))
        start = match.end()
    if start < len(text):
        spans_out.append((start, len(text)))
    return tuple(spans_out)


def _token_offsets(
    model: Any,
    text: str,
    tokenizer: OffsetTokenizerPort | None,
) -> tuple[tuple[int, int], ...]:
    if tokenizer is not None:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return tuple(
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
            if int(end) > int(start)
        )
    processor = getattr(model, "processor", None)
    splitter = getattr(processor, "word_splitter", None)
    if callable(splitter):
        return tuple((int(start), int(end)) for _, start, end in splitter(text, lower=True))
    # Deterministic fake-model seam. Production GLiNER2 always supplies word_splitter.
    return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


def _long_windows(
    text: str,
    offsets: tuple[tuple[int, int], ...],
) -> tuple[tuple[str, int], ...]:
    size = _chunk_size()
    overlap = _chunk_overlap()
    windows: list[tuple[str, int]] = []
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        windows.append((text[char_start:char_end], char_start))
        if end >= len(offsets):
            break
        start = max(start + 1, end - overlap)
    return tuple(windows)


def _shift_spans(value: object, offset: int) -> object:
    shifted = copy.deepcopy(value)

    def visit(item: object) -> None:
        if isinstance(item, dict):
            if isinstance(item.get("start"), int) and isinstance(item.get("end"), int):
                item["start"] += offset
                item["end"] += offset
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(shifted)
    return shifted


# §6.2: the core predicate registry is always on; only these optional extras
# vary by family. Written notes are the durable default and enable them all.
_FAMILY_OPTIONAL_PREDICATES: dict[str, frozenset[str]] = {
    "email": frozenset({"responsible_for", "scheduled_for", "blocked_by", "completed"}),
    "conversation": frozenset({"met_with", "decided", "prefers"}),
}
_OPTIONAL_PREDICATES: frozenset[str] = frozenset().union(
    *_FAMILY_OPTIONAL_PREDICATES.values()
)


def _relation_descriptions(family: str) -> dict[str, str]:
    emphasis = {
        "email": "email ownership, deadlines, blockers, and completion",
        "conversation": "conversation decisions, preferences, meetings, and commitments",
        "written": "durable written facts, plans, relationships, and status",
    }[family]
    enabled_optional = _FAMILY_OPTIONAL_PREDICATES.get(family, _OPTIONAL_PREDICATES)
    return {
        name: f"{template.format(head='subject', tail='object')}; focus on {emphasis}"
        for name, template in config.INGEST_RELATION_TYPES.items()
        if name not in _OPTIONAL_PREDICATES or name in enabled_optional
    }


def _source_family(source: str) -> str:
    if source in {"gmail", "outlook"}:
        return "email"
    if source in {"imessage", "live_session"}:
        return "conversation"
    return "written"


def _calibration() -> CalibrationArtifact:
    return load_calibration(_CALIBRATION_PATH)


def _global_threshold() -> float:
    calibration = _calibration().extraction
    return min(
        min(calibration.entity_thresholds.values()),
        min(calibration.relation_thresholds.values()),
    )


def _confidence(item: dict[str, Any], *, default: float = 1.0) -> float:
    value = item.get("confidence", item.get("conf", item.get("score", default)))
    return float(value) if isinstance(value, (int, float)) else default


def _is_long(offsets: tuple[tuple[int, int], ...]) -> bool:
    return _chunk_size() > 0 and len(offsets) > _chunk_size()


def _ingest_batch_size() -> int:
    return max(1, int(config.GLINER_INGEST_BATCH))


def _batch_token_budget() -> int:
    return max(1, int(config.GLINER_BATCH_TOKEN_BUDGET))


def _chunk_size() -> int:
    return max(1, int(config.GLINER_CHUNK_SIZE))


def _chunk_overlap() -> int:
    return max(0, min(int(config.GLINER_CHUNK_OVERLAP), _chunk_size() - 1))


def _valid_relation_endpoints(head: str, tail: str) -> bool:
    if re.sub(r"\s+", " ", head).strip().casefold() == re.sub(
        r"\s+", " ", tail
    ).strip().casefold():
        return False
    return len(re.sub(r"\s+", "", head.strip())) >= 3 and len(
        re.sub(r"\s+", "", tail.strip())
    ) >= 3
