from __future__ import annotations

from datetime import datetime, timezone

from tests.conftest import make_note
from taurine_core.indexer.doc_kind import snapshot_document


def _ts(year: int, month: int, day: int) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def test_resume_filename_with_m_d_yy_date() -> None:
    assert snapshot_document(make_note("n", "resume_1_18_24", "body")) == (
        True,
        _ts(2024, 1, 18),
    )


def test_resume_filename_with_name_prefix() -> None:
    assert snapshot_document(make_note("n", "neil_agrawal_resume_11_5_24", "x")) == (
        True,
        _ts(2024, 11, 5),
    )


def test_resume_filename_with_four_digit_year_date() -> None:
    assert snapshot_document(make_note("n", "resume_1_18_2024", "x")) == (
        True,
        _ts(2024, 1, 18),
    )


def test_letter_prefixed_year_pattern() -> None:
    assert snapshot_document(make_note("n", "GC2026_grade_report", "x")) == (
        True,
        _ts(2026, 1, 1),
    )


def test_cv_token_with_bare_year() -> None:
    assert snapshot_document(make_note("n", "CV 2024", "x")) == (True, _ts(2024, 1, 1))


def test_day_month_swap_when_first_group_is_not_a_month() -> None:
    assert snapshot_document(make_note("n", "resume_18_1_24", "x")) == (
        True,
        _ts(2024, 1, 18),
    )


def test_snapshot_titles_without_parseable_dates() -> None:
    assert snapshot_document(make_note("n", "Neil Agrawal Resume", "x")) == (True, None)
    assert snapshot_document(make_note("n", "Klaviyo Offer Letter", "x")) == (True, None)
    assert snapshot_document(make_note("n", "Fall Transcript", "x")) == (True, None)
    assert snapshot_document(make_note("n", "Curriculum Vitae", "x")) == (True, None)


def test_invalid_numeric_date_falls_back_to_none() -> None:
    assert snapshot_document(make_note("n", "resume_13_32_99", "x")) == (True, None)


def test_resume_shaped_content_without_snapshot_title() -> None:
    text = (
        "Neil Agrawal\n\nEDUCATION\nNortheastern University\n\n"
        "EXPERIENCE\nKlaviyo\n\nSKILLS\nPython, SQL"
    )
    assert snapshot_document(make_note("n", "Untitled", text)) == (True, None)


def test_offer_letter_content_without_snapshot_title() -> None:
    text = "Hi Neil,\n\nWe are pleased to offer you the position of Software Engineer."
    assert snapshot_document(make_note("n", "Untitled", text)) == (True, None)


def test_dated_form_letter_content() -> None:
    text = "January 18, 2024\n\nDear Neil,\n\nThank you for your application."
    assert snapshot_document(make_note("n", "Untitled", text)) == (True, None)


def test_ordinary_notes_are_not_snapshots() -> None:
    assert snapshot_document(make_note("n", "Groceries", "milk\neggs")) == (False, None)
    # a date token alone is not a snapshot marker
    assert snapshot_document(make_note("n", "Meeting notes 3_4_24", "roadmap")) == (
        False,
        None,
    )
    # 'cv' requires its own token boundary
    assert snapshot_document(make_note("n", "PCV valve notes", "car stuff")) == (
        False,
        None,
    )
    # one section-like line is not resume-shaped
    assert snapshot_document(make_note("n", "Ideas", "Projects\n- app idea")) == (
        False,
        None,
    )
