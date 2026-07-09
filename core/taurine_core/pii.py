"""PII redaction for connector text (regex layer + GLiNER2 schema layer).

Regex owns fixed-format identifiers — cards validated by Luhn, SSNs,
keyword-context account numbers, one-time codes — where it beats any model.
GLiNER2 (the already-cached live model) catches contextual PII through
GLINER2_PII_SCHEMA. Names, organizations, and addresses are deliberately
never redacted: entity linking needs them.

redact_pii never raises and degrades to regex-only when the model is
unavailable — redaction must not block a sync.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from taurine_core import config

_LOGGER = logging.getLogger("taurine.pii")
_WARNED_MODEL_MISSING = False

# 13–19 digits with optional single space/dash separators (card lengths);
# candidates are confirmed by Luhn before redacting
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")
_SSN_DASHED = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSN_CONTEXT = re.compile(
    r"(?i)\b(ssn|social security(?: number)?)\s*[:#]?\s*(\d{3}[ \-]?\d{2}[ \-]?\d{4})\b"
)
_ACCOUNT_CONTEXT = re.compile(
    r"(?i)\b(account|acct|routing|iban)\s*(?:number|no\.?|#)?\s*[:#]?\s*"
    r"([A-Z]{0,2}\d[\d \-]{4,28}\d)\b"
)
_CODE_CONTEXT = re.compile(
    r"(?i)\b((?:verification |security |one.time )?(?:code|otp|passcode|pin))"
    r"\s*(?:is)?\s*[:#]?\s*(\d{4,8})\b"
)


def redact_pii(text: str) -> str:
    if not text:
        return text
    return _redact_gliner(_redact_structured(text))


# ── regex layer ────────────────────────────────────────────────────────


def _redact_structured(text: str) -> str:
    text = _CARD_CANDIDATE.sub(_replace_card, text)
    text = _SSN_CONTEXT.sub(lambda m: f"{m.group(1)} [redacted-ssn]", text)
    text = _SSN_DASHED.sub("[redacted-ssn]", text)
    text = _ACCOUNT_CONTEXT.sub(lambda m: f"{m.group(1)} [redacted-account]", text)
    text = _CODE_CONTEXT.sub(lambda m: f"{m.group(1)} [redacted-code]", text)
    return text


def _replace_card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return "[redacted-card]"
    return match.group(0)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


# ── GLiNER2 layer ──────────────────────────────────────────────────────


def _redact_gliner(text: str) -> str:
    model = _pii_model()
    if model is None:
        return text
    try:
        result = model.extract_entities(
            text,
            dict(config.GLINER2_PII_SCHEMA),
            threshold=config.EMAIL_PII_GLINER_THRESHOLD,
            include_confidence=True,
        )
    except Exception:
        return text
    entities = result.get("entities") if isinstance(result, dict) else None
    if not isinstance(entities, dict):
        return text
    for label, values in entities.items():
        placeholder = f"[redacted-{str(label).replace('_', '-')}]"
        for item in values if isinstance(values, list) else []:
            span = item.get("text") if isinstance(item, dict) else item
            if not isinstance(span, str):
                continue
            span = span.strip()
            # tiny spans would shred unrelated text via substring replacement
            if len(span) >= 4:
                text = text.replace(span, placeholder)
    return text


def _pii_model() -> Any | None:
    global _WARNED_MODEL_MISSING
    from taurine_core.live import spans

    model = spans.shared_model()
    if model is None and not _WARNED_MODEL_MISSING:
        _WARNED_MODEL_MISSING = True
        _LOGGER.warning("GLiNER2 unavailable: PII redaction running regex-only")
    return model
