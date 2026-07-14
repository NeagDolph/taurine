"""Contextual enrichment of episodic passages (Anthropic contextual retrieval).

iMessage/email fragments embed meaninglessly on their own ("sounds good,
Tuesday works"), so a cheap LLM writes a 1-2 sentence situating prefix per
passage. The prefix is a retrieval input ONLY: it feeds the embedding frame
and the FTS context column, never the stored passage text, offsets, evidence
rendering, or validation text. Fail-open: no LLM or a failed batch degrades
to empty context (today's behavior), counted in the build report. Results
are cached in BuildCache by (prompt version, model, passage content hash,
batch-window hash) so incremental syncs cost ~nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from taurine_core import config
from taurine_core.indexer.passages import PassageDraft
from taurine_core.schemas import NoteRecord

_LOG = logging.getLogger("taurine.build")

# only context-free episodic fragments benefit; documents are self-situating
ENRICHABLE_SOURCES = frozenset({"imessage", "gmail", "outlook"})

_PROMPT_VERSION = "context-enrich-v1"
_CONTEXT_MAX_CHARS = 240
_WINDOW_PAD_CHARS = 1500
# a too-long window keeps its head and tail (thread start + recent turns)
_WINDOW_MAX_CHARS = 6000

_SYSTEM_PROMPT = (
    "You situate fragments of a personal knowledge base for retrieval. Given "
    "one source document (an iMessage thread or an email) and numbered "
    "passages from it, write 1-2 short sentences of situating context per "
    "passage: WHO is talking or writing, WHAT thread or email this is, "
    "roughly WHEN, and WHAT the passage concerns — information the fragment "
    "alone lacks. Use ONLY what is visible in the supplied title, date, "
    "window and passage text; never invent names, dates or facts. Return "
    'JSON {"contexts": {"<ordinal>": "<context>"}} with one entry per input '
    "passage, reusing each ordinal verbatim. Each context is at most 240 "
    "characters and describes the surrounding situation, not a summary of "
    "the passage itself."
)


@dataclass(frozen=True)
class EnrichmentPassage:
    ordinal: int
    passage_id: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class EnrichmentBatch:
    """One LLM call: a note's shared window plus up to ~10 of its passages."""

    note_id: str
    title: str
    source: str
    date: str
    window: str
    passages: tuple[EnrichmentPassage, ...]

    @property
    def window_hash(self) -> str:
        """Identity of everything batch-shared that shapes the contexts."""
        identity = "\x1f".join((self.title, self.source, self.date, self.window))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EnrichmentResult:
    contexts: dict[str, str]  # passage_id -> non-empty situating context
    enriched_count: int
    failed_batches: int


def template_context(
    note: NoteRecord,
    passage: EnrichmentPassage,
    position: str,
) -> str:
    """Build a deterministic situating prefix from note metadata only."""

    del passage  # the signature makes the passage association explicit
    source = str(getattr(note, "source", "") or "").casefold()
    title = " ".join(str(getattr(note, "title", "") or "").split())
    if source == "imessage":
        head = "iMessage thread"
        participants = _title_value(title, "iMessage:")
        if participants:
            head += f" between {participants}"
    elif source in {"gmail", "outlook"}:
        head = "Email"
        subject = _title_value(title, "Email:")
        if subject:
            head += f" about {subject}"
    else:
        return ""

    clauses = [head]
    display_line = str(
        getattr(note, "display_line", "") or getattr(note, "gist", "") or ""
    )
    sender_match = re.match(r"\s*\[([^\]\r\n]+)\]", display_line)
    if sender_match and (sender := " ".join(sender_match.group(1).split())):
        clauses.append(f"from {sender}")

    try:
        created = float(getattr(note, "created", 0.0) or 0.0)
        if created > 0:
            clauses.append(
                datetime.fromtimestamp(created, tz=timezone.utc).strftime("%B %Y")
            )
    except (OSError, OverflowError, TypeError, ValueError):
        pass

    if position in {"start of", "middle of", "end of"}:
        clauses.append(f"{position} the thread")
    return ("; ".join(clauses) + ".")[:_CONTEXT_MAX_CHARS]


def _title_value(title: str, prefix: str) -> str:
    if not title.casefold().startswith(prefix.casefold()):
        return ""
    return title[len(prefix) :].strip()


def build_enrichment_batches(
    note: NoteRecord,
    drafts: Sequence[PassageDraft],
) -> tuple[EnrichmentBatch, ...]:
    """Deterministic per-note batches; empty for non-episodic sources."""

    if note.source not in ENRICHABLE_SOURCES or not drafts:
        return ()
    ordered = sorted(drafts, key=lambda draft: draft.ordinal)
    date = datetime.fromtimestamp(
        float(note.created or note.modified), tz=timezone.utc
    ).strftime("%Y-%m-%d")
    size = max(1, int(config.CONTEXT_ENRICH_BATCH))
    batches: list[EnrichmentBatch] = []
    for start in range(0, len(ordered), size):
        group = ordered[start : start + size]
        batches.append(
            EnrichmentBatch(
                note_id=note.note_id,
                title=note.title,
                source=note.source,
                date=date,
                window=_window(note.text, group[0].char_start, group[-1].char_end),
                passages=tuple(
                    EnrichmentPassage(
                        ordinal=draft.ordinal,
                        passage_id=draft.passage_id,
                        text=draft.text,
                        content_hash=draft.content_hash,
                    )
                    for draft in group
                ),
            )
        )
    return tuple(batches)


def context_enricher_for(llm: Any) -> Callable[[EnrichmentBatch], Any]:
    """Bounded LLM enrichment callable shared by the server and CLI."""

    def enrich_batch(batch: EnrichmentBatch) -> Any:
        payload = {
            "title": batch.title,
            "source": batch.source,
            "date": batch.date,
            "window": batch.window,
            "passages": [
                {"ordinal": row.ordinal, "text": row.text} for row in batch.passages
            ],
        }
        return asyncio.run(
            llm.complete_json(
                config.CONTEXT_ENRICH_MODEL,
                _SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                timeout_s=config.SUMMARY_TIMEOUT_S,
                temperature=0.0,
            )
        )

    return enrich_batch


def enrich_passages(
    notes: Sequence[tuple[NoteRecord, Sequence[PassageDraft]]],
    enricher: Callable[[EnrichmentBatch], Any] | None,
    *,
    cache: Any | None = None,
    model_id: str = config.CONTEXT_ENRICH_MODEL,
) -> EnrichmentResult:
    """Situate episodic passages using the configured enrichment mode."""

    mode = str(getattr(config, "CONTEXT_ENRICH_MODE", "llm") or "llm").casefold()
    if mode not in {"llm", "template", "hybrid"}:
        _LOG.warning("unknown context enrichment mode %r; using llm", mode)
        mode = "llm"
    if mode == "llm":
        return _enrich_passages_llm(
            notes,
            enricher,
            cache=cache,
            model_id=model_id,
        )

    template_contexts = _template_contexts(notes)
    if mode == "template":
        return EnrichmentResult(template_contexts, len(template_contexts), 0)

    gated_notes = tuple(
        (note, drafts)
        for note, drafts in notes
        if note.source in ENRICHABLE_SOURCES
        and drafts
        and (
            len(note.text) > int(config.CONTEXT_LLM_MIN_CHARS)
            or len(drafts) >= int(config.CONTEXT_LLM_MIN_PASSAGES)
        )
    )
    llm_result = _enrich_passages_llm(
        gated_notes,
        enricher,
        cache=cache,
        model_id=model_id,
    )
    contexts = dict(template_contexts)
    contexts.update(llm_result.contexts)
    return EnrichmentResult(
        contexts=contexts,
        enriched_count=len(contexts),
        failed_batches=llm_result.failed_batches,
    )


def _template_contexts(
    notes: Sequence[tuple[NoteRecord, Sequence[PassageDraft]]],
) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for note, drafts in sorted(notes, key=lambda pair: pair[0].note_id):
        if note.source not in ENRICHABLE_SOURCES or not drafts:
            continue
        ordered = sorted(drafts, key=lambda draft: draft.ordinal)
        denominator = max(1, len(ordered) - 1)
        for index, draft in enumerate(ordered):
            fraction = index / denominator
            if fraction < 1 / 3:
                position = "start of"
            elif fraction > 2 / 3:
                position = "end of"
            else:
                position = "middle of"
            context = template_context(
                note,
                EnrichmentPassage(
                    ordinal=draft.ordinal,
                    passage_id=draft.passage_id,
                    text=draft.text,
                    content_hash=draft.content_hash,
                ),
                position,
            )
            if context:
                contexts[draft.passage_id] = context
    return contexts


def _enrich_passages_llm(
    notes: Sequence[tuple[NoteRecord, Sequence[PassageDraft]]],
    enricher: Callable[[EnrichmentBatch], Any] | None,
    *,
    cache: Any | None,
    model_id: str,
) -> EnrichmentResult:
    """Existing bounded LLM/cache path, kept isolated from templates.

    ``cache`` is a BuildCache (or anything with cached_contexts /
    store_contexts). No enricher ⇒ empty contexts, exactly today's behavior.
    """

    if enricher is None:
        return EnrichmentResult({}, 0, 0)
    batches = [
        batch
        for note, drafts in sorted(notes, key=lambda pair: pair[0].note_id)
        for batch in build_enrichment_batches(note, drafts)
    ]
    if not batches:
        return EnrichmentResult({}, 0, 0)

    keys = {
        row.passage_id: _context_key(model_id, row.content_hash, batch.window_hash)
        for batch in batches
        for row in batch.passages
    }
    cached: dict[str, str] = {}
    if cache is not None:
        by_key = cache.cached_contexts(sorted(set(keys.values())))
        cached = {
            passage_id: by_key[key]
            for passage_id, key in keys.items()
            if key in by_key
        }
    pending: list[EnrichmentBatch] = []
    for batch in batches:
        missing = tuple(row for row in batch.passages if row.passage_id not in cached)
        if missing:
            # cache hits shrink the call but the shared window (and therefore
            # the cache key) stays that of the original batch
            pending.append(
                EnrichmentBatch(
                    note_id=batch.note_id,
                    title=batch.title,
                    source=batch.source,
                    date=batch.date,
                    window=batch.window,
                    passages=missing,
                )
            )

    contexts: dict[str, str] = dict(cached)
    failed = 0
    if pending:
        workers = max(1, min(int(config.CONTEXT_ENRICH_CONCURRENCY), len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            responses = list(executor.map(_call_fail_open(enricher), pending))
        for batch, response in zip(pending, responses):
            if response is None:
                failed += 1
                continue
            parsed = _parse_response(response, batch)
            if parsed is None:
                _LOG.warning(
                    "context enrichment batch for note %s returned a malformed response",
                    batch.note_id,
                )
                failed += 1
                continue
            contexts.update(parsed)
            if cache is not None:
                cache.store_contexts(
                    [
                        (keys[passage_id], passage_id, context)
                        for passage_id, context in sorted(parsed.items())
                    ]
                )
    nonempty = {passage_id: value for passage_id, value in contexts.items() if value}
    return EnrichmentResult(
        contexts=nonempty,
        enriched_count=len(nonempty),
        failed_batches=failed,
    )


def _call_fail_open(
    enricher: Callable[[EnrichmentBatch], Any],
) -> Callable[[EnrichmentBatch], Any]:
    def call(batch: EnrichmentBatch) -> Any:
        try:
            return enricher(batch)
        except Exception as exc:  # noqa: BLE001 - enrichment must never fail a build
            _LOG.warning(
                "context enrichment batch for note %s failed (%s): %s",
                batch.note_id,
                type(exc).__name__,
                exc,
            )
            return None

    return call


def _parse_response(
    response: Any, batch: EnrichmentBatch
) -> dict[str, str] | None:
    """{passage_id: clamped context} for the batch's own ordinals, else None.

    Entries the model omitted stay unparsed (uncached, empty this build);
    a non-mapping response fails the whole batch.
    """

    rows = response.get("contexts") if isinstance(response, Mapping) else None
    if not isinstance(rows, Mapping):
        return None
    by_ordinal = {row.ordinal: row.passage_id for row in batch.passages}
    parsed: dict[str, str] = {}
    for raw_ordinal, raw_context in rows.items():
        try:
            ordinal = int(raw_ordinal)
        except (TypeError, ValueError):
            continue
        passage_id = by_ordinal.get(ordinal)
        if passage_id is None or not isinstance(raw_context, str):
            continue
        parsed[passage_id] = " ".join(raw_context.split())[:_CONTEXT_MAX_CHARS]
    return parsed


def _window(text: str, char_start: int, char_end: int) -> str:
    start = max(0, int(char_start) - _WINDOW_PAD_CHARS)
    end = min(len(text), int(char_end) + _WINDOW_PAD_CHARS)
    window = text[start:end]
    if len(window) <= _WINDOW_MAX_CHARS:
        return window
    half = _WINDOW_MAX_CHARS // 2
    return f"{window[:half]}\n…\n{window[-half:]}"


def _context_key(model_id: str, content_hash: str, window_hash: str) -> str:
    identity = "\x1f".join((_PROMPT_VERSION, model_id, content_hash, window_hash))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
