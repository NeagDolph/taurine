from __future__ import annotations

import logging
import re
from typing import Any

from taurine_core import config
from taurine_core.models_local import model_ref


_MODEL: Any | None = None
_MODEL_LOAD_FAILED = False
_LOGGER = logging.getLogger("taurine.session")


class Gliner2SpanExtractor:
    """GLiNER2 schema-based extraction (label → description) as a C2 query head.

    Spans are prioritized by schema order, then in-text order. GLiNER2 can return
    either plain span strings or richer dicts containing confidence.
    """

    def __init__(self) -> None:
        self.last_dropped: list[tuple[str, float, str]] = []

    def extract(self, text: str) -> list[tuple[str, float, str]]:
        self.last_dropped = []
        query = text.strip()
        if not query:
            return []

        model = _get_model()
        if model is None:
            return []

        try:
            result = model.extract_entities(
                query,
                dict(config.GLINER2_SCHEMA),
                threshold=config.GLINER2_THRESHOLD,
                include_confidence=True,
            )
        except Exception:
            return []

        kept, dropped = _normalize_result(result)
        self.last_dropped = dropped
        return kept


def _get_model() -> Any | None:
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None

    try:
        _MODEL = _load_model()
    except Exception as exc:
        _MODEL_LOAD_FAILED = True
        _LOGGER.warning("GLiNER2 model load failed: %s", exc)
        return None
    return _MODEL


def _load_model() -> Any:
    from gliner2 import GLiNER2

    return GLiNER2.from_pretrained(model_ref(config.GLINER2_MODEL))


def _normalize_result(
    result: object,
) -> tuple[list[tuple[str, float, str]], list[tuple[str, float, str]]]:
    entities = result.get("entities") if isinstance(result, dict) else None
    if not isinstance(entities, dict):
        return [], []

    deduped: dict[str, tuple[str, float, str]] = {}
    dropped: list[tuple[str, float, str]] = []
    stoplist = {_lower_normalized(item) for item in config.GLINER_SPAN_STOPLIST}
    # schema order defines priority at the cap
    for label in config.GLINER2_SCHEMA:
        for raw, score in _spans_for_label(entities.get(label)):
            span = raw.strip()
            if _drop_span(span, label, stoplist):
                dropped.append((span, score, label))
                continue
            key = _lower_normalized(span)
            if key in deduped:
                continue
            deduped[key] = (span, score, label)
    return list(deduped.values())[: config.GLINER_MAX_SPANS], dropped


def _spans_for_label(value: object) -> list[tuple[str, float]]:
    if not isinstance(value, list):
        return []
    spans: list[tuple[str, float]] = []
    for item in value:
        if isinstance(item, str):
            spans.append((item, 1.0))
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            confidence = item.get("confidence")
            score = float(confidence) if isinstance(confidence, (int, float)) else 1.0
            spans.append((item["text"], score))
    return spans


def _drop_span(span: str, label: str, stoplist: set[str]) -> bool:
    if len(span) < 3:
        return True

    normalized = _lower_normalized(span)
    if normalized in stoplist:
        return True

    without_article = re.sub(r"^(?:the|a|an)\s+", "", normalized)
    if without_article in stoplist:
        return True

    if label in config.GLINER_PROPER_LABELS:
        has_uppercase = any(char.isupper() for char in span)
        has_multiple_words = len(normalized.split()) >= 2
        if not has_uppercase and not has_multiple_words:
            return True

    return False


def _lower_normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
