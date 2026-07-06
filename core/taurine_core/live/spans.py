from __future__ import annotations

import logging
from typing import Any

from taurine_core import config


_MODEL: Any | None = None
_MODEL_LOAD_FAILED = False
_LOGGER = logging.getLogger("taurine.session")


class GlinerSpanExtractor:
    def extract(self, text: str) -> list[tuple[str, float]]:
        query = text.strip()
        if not query:
            return []

        model = _get_model()
        if model is None:
            return []

        try:
            predictions = model.predict_entities(
                query,
                config.GLINER_LABELS,
                threshold=config.GLINER_THRESHOLD,
            )
        except Exception:
            return []

        return _normalize_predictions(predictions)


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
        _LOGGER.warning("GLiNER model load failed: %s", exc)
        return None
    return _MODEL


def _load_model() -> Any:
    from gliner import GLiNER

    return GLiNER.from_pretrained(config.GLINER_MODEL)


def _normalize_predictions(predictions: object) -> list[tuple[str, float]]:
    deduped: dict[str, tuple[str, float]] = {}
    for prediction in _iter_predictions(predictions):
        span = _prediction_text(prediction).strip()
        if not span:
            continue
        score = _prediction_score(prediction)
        if score is None or score < config.GLINER_THRESHOLD:
            continue
        key = span.lower()
        existing = deduped.get(key)
        if existing is None or score > existing[1]:
            deduped[key] = (span, score)

    return sorted(deduped.values(), key=lambda item: item[1], reverse=True)[
        : config.GLINER_MAX_SPANS
    ]


def _iter_predictions(predictions: object) -> list[object]:
    if isinstance(predictions, list):
        return predictions
    return []


def _prediction_text(prediction: object) -> str:
    if isinstance(prediction, dict):
        value = prediction.get("text", "")
    else:
        value = getattr(prediction, "text", "")
    return str(value)


def _prediction_score(prediction: object) -> float | None:
    if isinstance(prediction, dict):
        value = prediction.get("score")
    else:
        value = getattr(prediction, "score", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
