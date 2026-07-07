from __future__ import annotations

import logging
from typing import Any

from taurine_core import config
from taurine_core.models_local import model_ref


_MODEL: Any | None = None
_MODEL_LOAD_FAILED = False
_LOGGER = logging.getLogger("taurine.session")


class Gliner2SpanExtractor:
    """GLiNER2 schema-based extraction (label → description) as a C2 query head.

    The model returns {"entities": {label: [span, ...]}} with no scores;
    spans are prioritized by schema order, then in-text order, and scored
    uniformly at 1.0 (fusion only needs the span texts).
    """

    def extract(self, text: str) -> list[tuple[str, float, str]]:
        query = text.strip()
        if not query:
            return []

        model = _get_model()
        if model is None:
            return []

        try:
            result = model.extract_entities(query, dict(config.GLINER2_SCHEMA))
        except Exception:
            return []

        return _normalize_result(result)


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


def _normalize_result(result: object) -> list[tuple[str, float, str]]:
    entities = result.get("entities") if isinstance(result, dict) else None
    if not isinstance(entities, dict):
        return []

    deduped: dict[str, tuple[str, float, str]] = {}
    # schema order defines priority at the cap
    for label in config.GLINER2_SCHEMA:
        for raw in _spans_for_label(entities.get(label)):
            span = raw.strip()
            key = span.lower()
            if not span or key in deduped:
                continue
            deduped[key] = (span, 1.0, label)
    return list(deduped.values())[: config.GLINER_MAX_SPANS]


def _spans_for_label(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    spans: list[str] = []
    for item in value:
        if isinstance(item, str):
            spans.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            spans.append(item["text"])  # tolerate richer per-span dicts
    return spans
