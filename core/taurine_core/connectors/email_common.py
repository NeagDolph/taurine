"""Provider-agnostic email machinery: the relevance funnel, thread rendering,
and thread → NoteRecord conversion. Deterministic — no LLM anywhere.

Funnel stages (spec 2026-07-08-email-connectors):
  1. deterministic drops   — window, unsubscribe/bulk, noreply-only,
                             transactional subjects (a reply from you rescues
                             any of these)
  2. importance score+cap  — recency × replied × known-participant × flagged
                             × length, top EMAIL_MAX_THREADS survive
  3. rendering             — [me]/[Name] transcript, quote/signature stripping
  4. PII redaction         — per message, before hashing, so raw PII never
                             reaches the DB, embeddings, or FTS
"""
from __future__ import annotations

import hashlib
import html as html_module
import logging
import re
import time
from dataclasses import dataclass, field

from taurine_core import config
from taurine_core.pii import redact_pii
from taurine_core.schemas import NoteRecord

_LOGGER = logging.getLogger("taurine.sync")


@dataclass(frozen=True)
class EmailMessage:
    text: str                      # plain-text body (HTML already converted)
    sender_name: str
    sender_email: str
    timestamp: float
    is_from_me: bool
    has_unsubscribe: bool = False  # List-Unsubscribe or Precedence: bulk/list
    recipients: tuple[str, ...] = ()


@dataclass
class EmailThread:
    thread_id: str                 # provider id; namespaced into note_id later
    subject: str
    messages: list[EmailMessage] = field(default_factory=list)
    flagged: bool = False          # starred / flagged / high importance
    bulk_classified: bool = False  # provider's own bulk signal (Outlook "other")


@dataclass
class FunnelReport:
    dropped: dict[str, int]
    kept: int


def funnel_threads(
    threads: list[EmailThread],
    *,
    contacts: dict[str, str],
    now: float | None = None,
) -> tuple[list[EmailThread], FunnelReport]:
    """Filter, score, and cap threads. Sorts each thread's messages
    oldest-first in place. contacts maps lowercase email → display name."""
    now = time.time() if now is None else now
    cutoff = now - config.EMAIL_WINDOW_DAYS * 86_400
    correspondents = _correspondent_set(threads)
    transactional = re.compile(config.EMAIL_TRANSACTIONAL_SUBJECT_RE)
    dropped = {
        "empty": 0,
        "window": 0,
        "unsubscribe": 0,
        "noreply": 0,
        "transactional": 0,
        "cap": 0,
    }
    survivors: list[EmailThread] = []
    for thread in threads:
        thread.messages.sort(key=lambda message: message.timestamp)
        if not thread.messages:
            dropped["empty"] += 1
            continue
        if thread.messages[-1].timestamp < cutoff:
            dropped["window"] += 1
            continue
        replied = any(message.is_from_me for message in thread.messages)
        if not replied:
            if thread.bulk_classified or any(m.has_unsubscribe for m in thread.messages):
                dropped["unsubscribe"] += 1
                continue
            senders = [m for m in thread.messages if not m.is_from_me]
            if senders and all(_is_noreply(m.sender_email) for m in senders):
                dropped["noreply"] += 1
                continue
            if transactional.search(thread.subject or ""):
                dropped["transactional"] += 1
                continue
        # no unknown-sender drop: calendar invites/accepts are single-message
        # threads from addresses you may never have written to, and real-inbox
        # calibration (2026-07-09) showed the rule eating exactly those.
        # Unknown senders simply score no known_participant weight, so they
        # sink in the cap ordering instead of vanishing.
        survivors.append(thread)

    survivors.sort(
        key=lambda thread: _score(thread, contacts, correspondents, now), reverse=True
    )
    if len(survivors) > config.EMAIL_MAX_THREADS:
        dropped["cap"] = len(survivors) - config.EMAIL_MAX_THREADS
        survivors = survivors[: config.EMAIL_MAX_THREADS]
    return survivors, FunnelReport(dropped=dropped, kept=len(survivors))


def log_funnel(connector_id: str, report: FunnelReport) -> None:
    drops = ", ".join(f"{key}={count}" for key, count in report.dropped.items() if count)
    _LOGGER.info(
        "%s funnel: kept %d threads (dropped: %s)",
        connector_id,
        report.kept,
        drops or "none",
    )


def _correspondent_set(threads: list[EmailThread]) -> set[str]:
    """Addresses you demonstrably correspond with: recipients of your sent
    messages plus senders in any thread you replied to."""
    known: set[str] = set()
    for thread in threads:
        if not any(message.is_from_me for message in thread.messages):
            continue
        for message in thread.messages:
            if message.is_from_me:
                known.update(address.lower() for address in message.recipients if address)
            elif message.sender_email:
                known.add(message.sender_email.lower())
    return known


_NOREPLY_PATTERN = re.compile(r"no[\-_.]?reply|do[\-_.]?not[\-_.]?reply")


def _is_noreply(address: str) -> bool:
    local = address.lower().split("@", 1)[0].split("+", 1)[0]
    return local in config.EMAIL_NOREPLY_LOCALPARTS or bool(_NOREPLY_PATTERN.search(local))


def _known_sender(
    message: EmailMessage, contacts: dict[str, str], correspondents: set[str]
) -> bool:
    address = message.sender_email.lower()
    return bool(address) and (address in contacts or address in correspondents)


def _score(
    thread: EmailThread,
    contacts: dict[str, str],
    correspondents: set[str],
    now: float,
) -> float:
    weights = config.EMAIL_SCORE_WEIGHTS
    age_days = max(0.0, (now - thread.messages[-1].timestamp) / 86_400)
    recency = 0.5 ** (age_days / config.EMAIL_RECENCY_SCORE_HALF_LIFE_DAYS)
    replied = any(message.is_from_me for message in thread.messages)
    known = any(
        _known_sender(message, contacts, correspondents)
        for message in thread.messages
        if not message.is_from_me
    )
    length = min(len(thread.messages), 8) / 8
    return (
        weights["recency"] * recency
        + weights["replied"] * replied
        + weights["known_participant"] * known
        + weights["flagged"] * thread.flagged
        + weights["length"] * length
    )


# ── rendering ──────────────────────────────────────────────────────────


def thread_to_record(thread: EmailThread, connector_id: str) -> NoteRecord | None:
    """Render a funnel survivor to a NoteRecord. Messages must already be
    oldest-first (funnel_threads guarantees it). Redaction happens per
    message BEFORE the transcript hash."""
    messages = thread.messages[-config.EMAIL_MAX_MESSAGES :]
    lines: list[str] = []
    timestamps: list[float] = []
    for message in messages:
        body = clean_body(message.text)
        if not body:
            continue
        body = redact_pii(body)[: config.EMAIL_MESSAGE_MAX_CHARS]
        speaker = (
            "me"
            if message.is_from_me
            else (message.sender_name or message.sender_email or "them")
        )
        lines.append(f"[{speaker}] {body}")
        timestamps.append(message.timestamp)
    if not lines:
        return None

    transcript = "\n".join(lines)
    while len(transcript) > config.CONNECTOR_DOC_MAX_CHARS and len(lines) > 1:
        lines.pop(0)
        timestamps.pop(0)
        transcript = "\n".join(lines)
    transcript = transcript[: config.CONNECTOR_DOC_MAX_CHARS]

    subject = re.sub(r"(?i)^\s*(?:(?:re|fwd?|fw)\s*:\s*)+", "", thread.subject or "").strip()
    title = f"Email: {subject or _participants(messages) or thread.thread_id}"
    return NoteRecord(
        note_id=f"{connector_id}:{thread.thread_id}",
        title=title,
        folder="Email",
        modified=timestamps[-1],
        created=timestamps[0],
        content_hash=hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        text=transcript,
        source=connector_id,
    )


def _participants(messages: list[EmailMessage]) -> str:
    names: list[str] = []
    for message in messages:
        if message.is_from_me:
            continue
        name = message.sender_name or message.sender_email
        if name and name not in names:
            names.append(name)
    return ", ".join(names[:4])


# quoted-tail markers: everything from the first match onward is a quote of
# earlier messages already present as their own transcript lines
_QUOTE_MARKERS = (
    re.compile(r"^\s*On .{0,160}wrote:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*-{3,}\s*Original Message\s*-{3,}", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"^\s*From:\s.{0,200}?^\s*Sent:\s", re.IGNORECASE | re.MULTILINE | re.DOTALL
    ),
    re.compile(r"^\s*_{10,}\s*$", re.MULTILINE),
)
_SIGNATURE_MARKER = re.compile(r"^--\s*$", re.MULTILINE)


def clean_body(text: str) -> str:
    """Strip quoted reply tails, `>`-quoted lines, and signatures; collapse
    whitespace to a single line."""
    body = text
    for marker in _QUOTE_MARKERS:
        match = marker.search(body)
        if match:
            body = body[: match.start()]
    match = _SIGNATURE_MARKER.search(body)
    if match:
        body = body[: match.start()]
    lines = [line for line in body.splitlines() if not line.lstrip().startswith(">")]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


# ── HTML → text (deterministic, good enough for email bodies) ──────────

_HTML_INVISIBLE = re.compile(r"<(style|script|head|title)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_BREAK = re.compile(r"<(?:br|/p|/div|/tr|/li|/h[1-6]|/table)[^>]*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")


def html_to_text(markup: str) -> str:
    text = _HTML_INVISIBLE.sub(" ", markup)
    text = _HTML_COMMENT.sub(" ", text)
    text = _HTML_BREAK.sub("\n", text)
    text = _HTML_TAG.sub(" ", text)
    text = html_module.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()
