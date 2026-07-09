"""Email funnel, rendering, and NoteRecord conversion (provider-agnostic)."""
from __future__ import annotations

import pytest

from taurine_core import config, pii
from taurine_core.connectors.email_common import (
    EmailMessage,
    EmailThread,
    clean_body,
    funnel_threads,
    html_to_text,
    thread_to_record,
)

_NOW = 1_760_000_000.0
_DAY = 86_400.0


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(pii, "_pii_model", lambda: None)


def _msg(
    text: str = "hello there",
    *,
    sender: str = "alice@example.com",
    name: str = "Alice",
    age_days: float = 1.0,
    from_me: bool = False,
    unsubscribe: bool = False,
    recipients: tuple[str, ...] = (),
) -> EmailMessage:
    return EmailMessage(
        text=text,
        sender_name=name,
        sender_email=sender,
        timestamp=_NOW - age_days * _DAY,
        is_from_me=from_me,
        has_unsubscribe=unsubscribe,
        recipients=recipients,
    )


def _thread(*messages: EmailMessage, subject: str = "Lunch plans", **kwargs) -> EmailThread:
    return EmailThread(thread_id="t1", subject=subject, messages=list(messages), **kwargs)


def _run(*threads: EmailThread, contacts: dict[str, str] | None = None):
    return funnel_threads(list(threads), contacts=contacts or {}, now=_NOW)


# ── stage 1: deterministic drops ───────────────────────────────────────


def test_thread_older_than_window_dropped() -> None:
    kept, report = _run(_thread(_msg(age_days=config.EMAIL_WINDOW_DAYS + 10, from_me=True)))
    assert kept == []
    assert report.dropped["window"] == 1


def test_unsubscribe_header_dropped() -> None:
    kept, report = _run(_thread(_msg(unsubscribe=True), _msg()))
    assert kept == []
    assert report.dropped["unsubscribe"] == 1


def test_unsubscribe_rescued_by_reply() -> None:
    kept, _ = _run(_thread(_msg(unsubscribe=True), _msg(from_me=True)))
    assert len(kept) == 1


def test_bulk_classified_dropped() -> None:
    kept, report = _run(_thread(_msg(), bulk_classified=True))
    assert kept == []
    assert report.dropped["unsubscribe"] == 1


def test_noreply_only_thread_dropped() -> None:
    kept, report = _run(_thread(_msg(sender="noreply@bank.com"), _msg(sender="billing@bank.com")))
    assert kept == []
    assert report.dropped["noreply"] == 1


def test_noreply_rescued_by_reply() -> None:
    kept, _ = _run(_thread(_msg(sender="noreply@bank.com"), _msg(from_me=True)))
    assert len(kept) == 1


def test_transactional_subject_dropped() -> None:
    kept, report = _run(
        _thread(_msg(), _msg(), subject="Your receipt from Acme Corp")
    )
    assert kept == []
    assert report.dropped["transactional"] == 1


def test_transactional_rescued_by_reply() -> None:
    kept, _ = _run(
        _thread(_msg(), _msg(from_me=True), subject="Invoice #42 — payment due")
    )
    assert len(kept) == 1


def test_unknown_single_message_sender_kept() -> None:
    # no unknown-sender drop (2026-07-09 calibration): calendar invites and
    # accepts are single-message threads from unfamiliar addresses
    kept, _ = _run(_thread(_msg(sender="stranger@cold-outreach.io")))
    assert len(kept) == 1


def test_unknown_sender_ranks_below_known_sender() -> None:
    known = EmailThread(
        thread_id="known", subject="Sync", messages=[_msg(sender="alice@example.com")]
    )
    unknown = EmailThread(
        thread_id="unknown",
        subject="Intro",
        messages=[_msg(sender="stranger@cold-outreach.io")],
    )
    kept, _ = _run(unknown, known, contacts={"alice@example.com": "Alice"})
    assert [t.thread_id for t in kept] == ["known", "unknown"]


def test_correspondent_counts_as_known_for_ranking() -> None:
    # bob is a recipient of a message YOU sent in another thread → his
    # single-message thread outranks an equally-recent unknown sender's
    replied = EmailThread(
        thread_id="t-replied",
        subject="Q3 numbers",
        messages=[_msg(from_me=True, recipients=("bob@corp.com",)), _msg(sender="bob@corp.com")],
    )
    from_bob = EmailThread(
        thread_id="t-bob", subject="One more thing", messages=[_msg(sender="bob@corp.com")]
    )
    from_stranger = EmailThread(
        thread_id="t-stranger", subject="Hello", messages=[_msg(sender="who@nowhere.io")]
    )
    kept, _ = _run(replied, from_stranger, from_bob)
    assert kept[-1].thread_id == "t-stranger"
    assert {t.thread_id for t in kept} == {"t-replied", "t-bob", "t-stranger"}


def test_empty_thread_dropped() -> None:
    kept, report = _run(EmailThread(thread_id="t0", subject="", messages=[]))
    assert kept == []
    assert report.dropped["empty"] == 1


# ── stage 2: scoring + cap ─────────────────────────────────────────────


def test_replied_recent_thread_outranks_unreplied_old() -> None:
    old = EmailThread(
        thread_id="old",
        subject="Old chatter",
        messages=[_msg(age_days=90.0), _msg(age_days=89.0)],
    )
    fresh = EmailThread(
        thread_id="fresh",
        subject="Timeline",
        messages=[_msg(age_days=1.0), _msg(age_days=0.5, from_me=True)],
    )
    kept, _ = _run(old, fresh)
    assert [t.thread_id for t in kept] == ["fresh", "old"]


def test_cap_applies_after_scoring(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMAIL_MAX_THREADS", 1)
    low = EmailThread(thread_id="low", subject="a", messages=[_msg(age_days=60.0), _msg(age_days=59.0)])
    high = EmailThread(
        thread_id="high", subject="b", messages=[_msg(age_days=1.0), _msg(from_me=True)]
    )
    kept, report = _run(low, high)
    assert [t.thread_id for t in kept] == ["high"]
    assert report.dropped["cap"] == 1


# ── stage 3: rendering ─────────────────────────────────────────────────


def test_record_labels_me_and_sender_names() -> None:
    thread = _thread(
        _msg("can you send the deck?", name="Alice"),
        _msg("sure, tomorrow", from_me=True, age_days=0.5),
    )
    record = thread_to_record(thread, "gmail")
    assert record is not None
    assert "[Alice] can you send the deck?" in record.text
    assert "[me] sure, tomorrow" in record.text
    assert record.note_id == "gmail:t1"
    assert record.source == "gmail"
    assert record.folder == "Email"


def test_title_strips_reply_prefixes() -> None:
    thread = _thread(_msg(), subject="Re: RE: Fwd: Budget review")
    record = thread_to_record(thread, "gmail")
    assert record.title == "Email: Budget review"


def test_title_falls_back_to_participants() -> None:
    thread = _thread(_msg(name="Alice"), subject="")
    record = thread_to_record(thread, "outlook")
    assert record.title == "Email: Alice"


def test_record_timestamps_and_hash_stable() -> None:
    thread = _thread(_msg(age_days=3.0), _msg(from_me=True, age_days=1.0))
    first = thread_to_record(thread, "gmail")
    second = thread_to_record(thread, "gmail")
    assert first.content_hash == second.content_hash
    assert first.modified == _NOW - 1.0 * _DAY
    assert first.created == _NOW - 3.0 * _DAY


def test_message_cap_keeps_newest(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMAIL_MAX_MESSAGES", 2)
    thread = _thread(
        _msg("oldest", age_days=3.0),
        _msg("middle", age_days=2.0),
        _msg("newest", age_days=1.0),
    )
    thread.messages.sort(key=lambda m: m.timestamp)
    record = thread_to_record(thread, "gmail")
    assert "oldest" not in record.text
    assert "middle" in record.text and "newest" in record.text


def test_doc_cap_drops_oldest_lines(monkeypatch) -> None:
    monkeypatch.setattr(config, "CONNECTOR_DOC_MAX_CHARS", 60)
    thread = _thread(
        _msg("A" * 40, age_days=2.0),
        _msg("B" * 40, age_days=1.0),
    )
    thread.messages.sort(key=lambda m: m.timestamp)
    record = thread_to_record(thread, "gmail")
    assert "BBBB" in record.text and "AAAA" not in record.text


def test_pii_redacted_before_hash() -> None:
    thread = _thread(_msg("card 4111 1111 1111 1111 ok"))
    record = thread_to_record(thread, "gmail")
    assert "[redacted-card]" in record.text
    assert "4111" not in record.text


def test_thread_with_only_empty_bodies_returns_none() -> None:
    thread = _thread(_msg("  "), _msg(">"))
    assert thread_to_record(thread, "gmail") is None


# ── body cleaning ──────────────────────────────────────────────────────


def test_clean_body_cuts_on_wrote_marker() -> None:
    body = "See you at 3.\n\nOn Mon, Jul 6, 2026 at 9:14 AM Alice <a@x.com> wrote:\n> earlier stuff"
    assert clean_body(body) == "See you at 3."


def test_clean_body_cuts_original_message_block() -> None:
    body = "Works for me.\n-----Original Message-----\nFrom: Bob\nSent: Monday"
    assert clean_body(body) == "Works for me."


def test_clean_body_cuts_outlook_header_block() -> None:
    body = "Approved.\nFrom: Alice <a@x.com>\nSent: Monday 9am\nTo: me\nSubject: Re: thing\nold text"
    assert clean_body(body) == "Approved."


def test_clean_body_drops_quoted_lines_and_signature() -> None:
    body = "Sounds good\n> quoted reply\n>> deeper quote\n-- \nNeil\nCEO, Example Corp"
    assert clean_body(body) == "Sounds good"


def test_html_to_text_strips_tags_and_style() -> None:
    markup = "<html><style>p{color:red}</style><body><p>Hello <b>world</b></p><div>bye</div></body></html>"
    text = html_to_text(markup)
    assert "Hello world" in text
    assert "bye" in text
    assert "color" not in text


def test_html_entities_unescaped() -> None:
    assert "a & b" in html_to_text("<p>a &amp; b</p>")
