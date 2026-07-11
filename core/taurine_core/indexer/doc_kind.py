"""Deterministic snapshot-document classification for temporal fact anchoring.

A snapshot document (resume/CV, offer letter, transcript or grade report,
dated form letter) describes the world as of the document's own date, not as
of the note's sync or modification timestamps. Facts promoted from snapshot
documents anchor ``observed_at`` to the date parsed from the title/filename
when available, else ``note.created`` — never ``note.modified`` — so an old
resume synced yesterday cannot outrank a recent first-person note.

No LLM, no embeddings: pure regex heuristics so builds stay deterministic.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Protocol

__all__ = ["snapshot_document"]


class _NoteLike(Protocol):
    title: str
    text: str


# token boundaries that treat '_' and '-' as separators (filenames like
# "neil_agrawal_resume_11_5_24" defeat \b because '_' is a word character)
_B = r"(?<![a-z0-9])"
_E = r"(?![a-z0-9])"

_TITLE_MARKERS = tuple(
    re.compile(_B + pattern + _E, re.IGNORECASE)
    for pattern in (
        r"r[eé]sum[eé]s?",
        r"cv",
        r"curriculum[\s_-]+vitae",
        r"offer[\s_-]+letters?",
        r"transcripts?",
        r"grade[\s_-]+reports?",
        r"report[\s_-]+cards?",
    )
)

# resume-shaped body: at least two distinct section headers on their own lines
_SECTION_HEADER = re.compile(
    r"^\s*(education|(?:work[\s_-]|professional[\s_-])?experience|skills"
    r"|projects|certifications?|awards|publications)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_OFFER_CONTENT = re.compile(
    r"pleased to offer you|offer of employment", re.IGNORECASE
)
_TRANSCRIPT_CONTENT = re.compile(
    r"academic transcript|grade report|cumulative gpa", re.IGNORECASE
)
# dated form letter: a formal date line plus a salutation near the top
_LETTER_DATE = re.compile(
    r"^\s*(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s+\d{1,2},\s+\d{4}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SALUTATION = re.compile(r"^\s*dear\b", re.IGNORECASE | re.MULTILINE)

# filename dates: "resume_1_18_24" (M_D_YY, US order), "resume_1_18_2024"
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})[._-](\d{1,2})[._-](\d{4}|\d{2})(?!\d)")
# embedded or bare year: "GC2026_...", "CV 2024"
_YEAR = re.compile(r"(?<!\d)(19[7-9]\d|20\d{2})(?!\d)")


def _parse_title_date(title: str) -> float | None:
    match = _NUMERIC_DATE.search(title)
    if match:
        month, day, year = (int(group) for group in match.groups())
        if year < 100:
            year += 2000
        if month > 12 and day <= 12:
            month, day = day, month
        if 1970 <= year <= 2100:
            try:
                return datetime(year, month, day, tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass  # not a real calendar date; fall through to bare year
    match = _YEAR.search(title)
    if match:
        return datetime(int(match.group(1)), 1, 1, tzinfo=timezone.utc).timestamp()
    return None


def snapshot_document(note: _NoteLike) -> tuple[bool, float | None]:
    """Classify a note as a snapshot document and parse its own date.

    Returns ``(is_snapshot, doc_date)`` where ``doc_date`` is a UTC epoch
    timestamp parsed from the title/filename, or ``None`` when the title
    carries no parseable date. Content dates are never used — only the
    title/filename is trusted for the document's own date.
    """
    title = note.title or ""
    text = note.text or ""
    is_snapshot = any(marker.search(title) for marker in _TITLE_MARKERS)
    if not is_snapshot:
        headers = {
            match.group(1).casefold()
            for match in _SECTION_HEADER.finditer(text[:4000])
        }
        head = text[:600]
        is_snapshot = (
            len(headers) >= 2
            or bool(_OFFER_CONTENT.search(text[:2000]))
            or bool(_TRANSCRIPT_CONTENT.search(text[:2000]))
            or bool(_LETTER_DATE.search(head) and _SALUTATION.search(head))
        )
    if not is_snapshot:
        return False, None
    return True, _parse_title_date(title)
