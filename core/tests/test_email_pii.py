"""PII redaction: regex layer (structured identifiers) + GLiNER2 layer."""
from __future__ import annotations

import pytest

from taurine_core import pii


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """Default: no GLiNER2 model — regex-only. Tests that need the model
    layer monkeypatch _pii_model themselves."""
    monkeypatch.setattr(pii, "_pii_model", lambda: None)


def test_luhn_valid_card_redacted() -> None:
    text = "my card is 4111 1111 1111 1111 thanks"
    result = pii.redact_pii(text)
    assert "[redacted-card]" in result
    assert "4111" not in result


def test_luhn_valid_card_with_dashes_redacted() -> None:
    result = pii.redact_pii("card: 5500-0000-0000-0004")
    assert "[redacted-card]" in result
    assert "5500" not in result


def test_luhn_invalid_number_preserved() -> None:
    # fails the Luhn check → not a card, keep it (could be an order id)
    text = "reference number 4111 1111 1111 1112"
    assert pii.redact_pii(text) == text


def test_ssn_dashed_form_redacted() -> None:
    result = pii.redact_pii("her SSN is 123-45-6789 on file")
    assert "[redacted-ssn]" in result
    assert "123-45-6789" not in result


def test_ssn_context_form_redacted() -> None:
    result = pii.redact_pii("SSN: 123 45 6789")
    assert "[redacted-ssn]" in result
    assert "6789" not in result


def test_account_number_with_context_redacted() -> None:
    result = pii.redact_pii("routing number: 021000021 for the wire")
    assert "[redacted-account]" in result
    assert "021000021" not in result


def test_bare_long_number_without_context_preserved() -> None:
    # 9 digits, no card length, no keyword context → left alone
    text = "tracking id 021000021"
    assert pii.redact_pii(text) == text


def test_verification_code_redacted() -> None:
    result = pii.redact_pii("Your verification code is 482913 — do not share it")
    assert "[redacted-code]" in result
    assert "482913" not in result


def test_names_and_addresses_preserved() -> None:
    text = "Neil met Sarah Chen at 123 Main Street, Springfield to sign the lease"
    assert pii.redact_pii(text) == text


def test_empty_text_passthrough() -> None:
    assert pii.redact_pii("") == ""


class _FakeModel:
    def __init__(self, entities: dict) -> None:
        self._entities = entities
        self.calls: list[str] = []

    def extract_entities(self, text, schema, threshold, include_confidence):
        self.calls.append(text)
        return {"entities": self._entities}


def test_gliner_spans_replaced_with_labeled_placeholder(monkeypatch) -> None:
    model = _FakeModel({"password_or_credential": [{"text": "hunter2secret"}]})
    monkeypatch.setattr(pii, "_pii_model", lambda: model)
    result = pii.redact_pii("the wifi password is hunter2secret ok")
    assert "[redacted-password-or-credential]" in result
    assert "hunter2secret" not in result
    assert model.calls  # model layer actually ran


def test_gliner_tiny_spans_ignored(monkeypatch) -> None:
    # spans under 4 chars would shred unrelated text via str.replace
    model = _FakeModel({"government_id": [{"text": "ab"}]})
    monkeypatch.setattr(pii, "_pii_model", lambda: model)
    text = "absolutely fine sentence about abs"
    assert pii.redact_pii(text) == text


def test_gliner_failure_degrades_to_regex_only(monkeypatch) -> None:
    class _Boom:
        def extract_entities(self, *args, **kwargs):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(pii, "_pii_model", lambda: _Boom())
    result = pii.redact_pii("card 4111 1111 1111 1111 and text")
    assert "[redacted-card]" in result  # regex layer already ran
