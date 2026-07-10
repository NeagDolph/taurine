from __future__ import annotations

import pytest

from taurine_core import config
from taurine_core.indexer.salience import (
    TOKEN_SATURATION,
    content_quality,
    content_quality_components,
    note_salience,
)


DENSE_SHORT = "Maya owns the Zephyr launch checklist."
CLEAN_PROSE = (
    "The launch review moved to Thursday because the payments vendor slipped "
    "their certification window. Maya will confirm the revised timeline with "
    "the platform team, and Rahul owns the rollback plan in case the new "
    "checkout flow regresses under load during the first week."
)


def test_empty_and_whitespace_text_score_zero() -> None:
    assert content_quality("") == 0.0
    assert content_quality("   \n\t  ") == 0.0


def test_short_but_dense_note_scores_above_the_extraction_floor() -> None:
    quality = content_quality(DENSE_SHORT)
    assert quality >= config.EXTRACTION_SALIENCE_FLOOR
    assert quality > 0.4


def test_long_clean_prose_scores_near_one() -> None:
    assert content_quality(CLEAN_PROSE) > 0.85


def test_repeated_token_filler_scores_low() -> None:
    filler = "yeah " * 12
    assert content_quality(filler) < config.EXTRACTION_SALIENCE_FLOOR


def test_long_boilerplate_with_repeated_lines_scores_low() -> None:
    boilerplate = "\n".join(["You have a new notification waiting in your inbox."] * 10)
    assert content_quality(boilerplate) < config.EXTRACTION_SALIENCE_FLOOR
    # length alone must never rescue junk: this is longer than the dense note
    assert len(boilerplate) > len(DENSE_SHORT)


def test_symbol_soup_scores_low_via_alpha_ratio() -> None:
    soup = "#### 1234 5678 ---- $$$$ 00:12 9999 ****"
    assert content_quality(soup) < config.EXTRACTION_SALIENCE_FLOOR


def test_token_ramp_component_saturates_at_forty_tokens() -> None:
    at_saturation = content_quality_components("word " * TOKEN_SATURATION)
    beyond = content_quality_components("word " * (TOKEN_SATURATION * 3))
    short = content_quality_components("word " * 10)
    assert at_saturation.token_ramp == 1.0
    assert beyond.token_ramp == 1.0
    assert short.token_ramp == pytest.approx(10 / TOKEN_SATURATION)


def test_alpha_ratio_component_measures_alpha_characters() -> None:
    prose = content_quality_components("alpha beta gamma")
    digits = content_quality_components("111 222 333")
    assert prose.alpha_ratio == 1.0
    assert digits.alpha_ratio == 0.0
    assert content_quality("alpha beta gamma") > content_quality("111 222 333")


def test_unique_ratio_component_penalizes_repetition() -> None:
    unique = content_quality_components("alpha beta gamma delta")
    repeated = content_quality_components("alpha alpha alpha alpha")
    assert unique.unique_ratio == 1.0
    assert repeated.unique_ratio == pytest.approx(0.25)
    assert unique.quality > repeated.quality


def test_repeated_line_penalty_component() -> None:
    varied = "first line of notes\nsecond line of notes\nthird line of notes"
    duplicated = "first line of notes\nfirst line of notes\nfirst line of notes"
    varied_parts = content_quality_components(varied)
    duplicated_parts = content_quality_components(duplicated)
    assert varied_parts.line_penalty == 1.0
    assert duplicated_parts.line_penalty == pytest.approx(1 / 3)
    assert varied_parts.quality > duplicated_parts.quality


def test_note_salience_multiplies_connector_and_content_scores() -> None:
    quality = content_quality(CLEAN_PROSE)
    assert note_salience(1.0, CLEAN_PROSE) == pytest.approx(quality)
    assert note_salience(0.2, CLEAN_PROSE) == pytest.approx(0.2 * quality)
    assert note_salience(-1.0, CLEAN_PROSE) == 0.0
    # a junk note keeps a low combined score no matter the connector boost
    assert note_salience(1.0, "yeah " * 12) < config.EXTRACTION_SALIENCE_FLOOR


def test_content_quality_is_deterministic_and_bounded() -> None:
    for text in ("", DENSE_SHORT, CLEAN_PROSE, "yeah " * 12):
        first = content_quality(text)
        assert first == content_quality(text)
        assert 0.0 <= first <= 1.0
