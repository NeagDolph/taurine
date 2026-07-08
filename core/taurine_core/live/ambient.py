from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rapidfuzz.distance import Levenshtein

from taurine_core import config
from taurine_core.ports import IndexPort, RerankerPort, SpanExtractorPort
from taurine_core.schemas import Chunk

from .exact import ExactCollector
from .grounding import GroundingEngine, GroundingUpdate


class Clock(Protocol):
    def now(self) -> float:
        ...


@dataclass(frozen=True)
class _Candidate:
    note_id: str
    score: float


class AmbientRetriever:
    def __init__(
        self,
        index: IndexPort,
        reranker: RerankerPort,
        clock: Clock,
        on_debug: Callable[[dict[str, Any]], None] | None = None,
        span_extractor: SpanExtractorPort | None = None,
    ) -> None:
        del reranker
        self.last_slate_adds: list[Any] = []
        self.last_debug: dict[str, Any] | None = None
        self.last_grounding: GroundingUpdate | None = None
        self.last_extracted: list[str] = []
        self.last_fused_ids: set[str] = set()
        self.last_ppr_expansions: list[tuple[str, float]] = []
        self._index = index
        self._clock = clock
        self._on_debug = on_debug
        self._span_extractor = span_extractor
        self._last_run_at: float | None = None
        self._snap_vocabulary = _snap_vocabulary(index.aliases())
        self._exact = ExactCollector(index)
        self._grounding = GroundingEngine(index)  # type: ignore[arg-type]

    async def on_window(
        self,
        window_text: str,
        context_text: str | None = None,
    ) -> GroundingUpdate | None:
        self.last_slate_adds = []
        self.last_debug = None
        self.last_grounding = None
        self.last_extracted = []
        self.last_fused_ids = set()
        self.last_ppr_expansions = []

        span_context_text = context_text if context_text is not None else window_text
        now = self._clock.now()
        if (
            self._last_run_at is not None
            and now - self._last_run_at < config.C2_THROTTLE_S
        ):
            return None
        self._last_run_at = now

        gliner_spans = self._extract_spans(window_text)
        gliner_dropped = list(getattr(self._span_extractor, "last_dropped", []))
        gliner_queries, snapped, gliner_context_queries = self._gliner_queries(
            gliner_spans,
            span_context_text,
        )
        extracted = [span for span, _score, _label in gliner_spans]
        dense = self._index.dense_search(window_text, config.RERANK_MAX_CANDIDATES)
        dense_gliner = self._dense_gliner(gliner_queries)
        bm25_query = _bm25_query(window_text, snapped)
        bm25 = self._index.bm25_search(bm25_query, config.RERANK_MAX_CANDIDATES)
        exact_hits = self._exact_hits(window_text, now)
        exact_ranked = [(note_id, 1.0) for note_id, _alias in exact_hits]

        exact_lists = [[row] for row in exact_ranked]
        fused_scores = self._rrf_scores(dense, dense_gliner, bm25, *exact_lists)
        fused = [
            (candidate.note_id, candidate.score)
            for candidate in self._ordered_candidates(fused_scores)
        ]
        head_hits = _head_hits(
            dense=dense,
            dense_gliner=dense_gliner,
            bm25=bm25,
            exact=exact_ranked,
        )

        update = self._grounding.on_pass(
            window_text,
            fused,
            exact_hits,
            now,
            head_hits=head_hits,
        )
        self.last_grounding = update
        self.last_extracted = extracted
        self.last_fused_ids = {note_id for note_id, _score in fused}
        self.last_ppr_expansions = [
            (note_id, mass)
            for note_id, mass in update.ppr_top
            if note_id not in self.last_fused_ids
        ]

        self._record_debug(
            window=window_text,
            gliner_spans=gliner_spans,
            gliner_dropped=gliner_dropped,
            gliner_context_queries=gliner_context_queries,
            dense=dense,
            dense_gliner=dense_gliner,
            bm25=bm25,
            exact=exact_ranked,
            bm25_query=bm25_query,
            fused=fused,
            update=update,
            snapped=snapped,
        )
        return update

    def _rrf_scores(
        self,
        *ranked_lists: list[tuple[str, float]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for rows in ranked_lists:
            for rank, (note_id, _score) in enumerate(rows, start=1):
                scores[note_id] = scores.get(note_id, 0.0) + (
                    1.0 / (config.RRF_K + rank)
                )
        return scores

    def _extract_spans(self, window_text: str) -> list[tuple[str, float, str]]:
        if self._span_extractor is None:
            return []
        try:
            return self._span_extractor.extract(window_text)
        except Exception:
            return []

    def _gliner_queries(
        self,
        gliner_spans: list[tuple[str, float, str]],
        context_text: str,
    ) -> tuple[list[str], list[tuple[str, str]], list[str]]:
        queries: list[str] = []
        seen_queries: set[str] = set()
        snapped: list[tuple[str, str]] = []
        context_queries: list[str] = []
        for span, _span_score, label in gliner_spans:
            _append_query(queries, seen_queries, span)
            snapped_span = self._snap_span(span, label)
            if snapped_span is not None:
                _append_query(queries, seen_queries, snapped_span)
                snapped.append((span, snapped_span))
            context_query = _span_context_query(span, context_text)
            if context_query is not None and _append_query(
                queries,
                seen_queries,
                context_query,
            ):
                context_queries.append(context_query)
        return queries, snapped, context_queries

    def _snap_span(self, span: str, label: str) -> str | None:
        normalized = span.strip().lower()
        if (
            not normalized
            or label.lower() not in config.FUZZY_SNAP_LABELS
            or len(normalized) < config.FUZZY_SNAP_MIN_LEN
            or not self._snap_vocabulary
        ):
            return None

        best: tuple[int, str] | None = None
        for candidate in self._snap_vocabulary:
            if candidate == normalized:
                continue
            distance = Levenshtein.distance(normalized, candidate)
            if best is None or (distance, candidate) < best:
                best = (distance, candidate)
        if best is None or best[0] > config.FUZZY_SNAP_MAX_DIST:
            return None
        return best[1]

    def _dense_gliner(
        self,
        gliner_queries: list[str],
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for query in gliner_queries:
            for note_id, score in self._index.dense_search(
                query,
                config.RERANK_MAX_CANDIDATES,
            ):
                scores[note_id] = max(scores.get(note_id, float("-inf")), score)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : config.RERANK_MAX_CANDIDATES
        ]

    def _exact_hits(self, window_text: str, now: float) -> list[tuple[str, str]]:
        chunk = Chunk(channel=0, text=window_text, ts=now, final=True)
        hits: list[tuple[str, str]] = []
        seen: set[str] = set()
        for note_id, alias, _is_title in self._exact.on_chunk(chunk):
            if note_id in seen:
                continue
            seen.add(note_id)
            hits.append((note_id, alias))
        return hits

    def _ordered_candidates(self, scores: dict[str, float]) -> list[_Candidate]:
        first_seen = {note_id: index for index, note_id in enumerate(scores)}
        return [
            _Candidate(note_id, score)
            for note_id, score in sorted(
                scores.items(),
                key=lambda item: (-item[1], first_seen[item[0]], item[0]),
            )
        ][: config.RERANK_MAX_CANDIDATES]

    def _record_debug(
        self,
        *,
        window: str,
        gliner_spans: list[tuple[str, float, str]],
        gliner_dropped: list[tuple[str, float, str]],
        gliner_context_queries: list[str],
        dense: list[tuple[str, float]],
        dense_gliner: list[tuple[str, float]],
        bm25: list[tuple[str, float]],
        exact: list[tuple[str, float]],
        bm25_query: str,
        fused: list[tuple[str, float]],
        update: GroundingUpdate,
        snapped: list[tuple[str, str]],
    ) -> None:
        blend = [
            {"note_id": item.note_id, **self._grounding.last_blend.get(item.note_id, {})}
            for item in update.items
        ]
        debug = {
            "window": window,
            "gliner_spans": _rows(gliner_spans),
            "gliner_dropped": _rows(gliner_dropped),
            "gliner_context_queries": gliner_context_queries,
            "dense": _rows(dense),
            "dense_gliner": _rows(dense_gliner),
            "bm25": _rows(bm25),
            "exact": _rows(exact),
            "fused": _rows(fused),
            "ppr_top": _rows(update.ppr_top),
            "blend": blend,
            "grounding_revision": update.revision,
            "grounding_items": [
                [item.note_id, item.short_id, item.score] for item in update.items
            ],
            "facts": [[fact.fact_id, fact.src, fact.dst, fact.text] for fact in update.facts],
        }
        if bm25_query != window:
            debug["bm25_query"] = bm25_query
        if snapped:
            debug["snapped"] = _rows(snapped)
        self.last_debug = debug
        if self._on_debug is None:
            return
        try:
            self._on_debug(debug)
        except Exception:
            return


def _head_hits(
    *,
    dense: list[tuple[str, float]],
    dense_gliner: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    exact: list[tuple[str, float]],
) -> dict[str, set[str]]:
    heads: dict[str, set[str]] = {}
    for label, rows in (
        ("dense", dense),
        ("gliner", dense_gliner),
        ("bm25", bm25),
        ("exact", exact),
    ):
        for note_id, _score in rows:
            heads.setdefault(note_id, set()).add(label)
    return heads


def _rows(rows: list[tuple[Any, ...]]) -> list[list[Any]]:
    return [list(row) for row in rows]


def _append_query(queries: list[str], seen: set[str], query: str) -> bool:
    query = query.strip()
    if not query or query in seen:
        return False
    seen.add(query)
    queries.append(query)
    return True


def _span_context_query(span: str, context_text: str) -> str | None:
    span = span.strip()
    if not span or not context_text:
        return None

    matches = list(re.finditer(re.escape(span), context_text, flags=re.IGNORECASE))
    if not matches:
        return None

    match = matches[-1]
    sentence_spans = _sentence_spans(context_text)
    if not sentence_spans:
        return None

    sentence_index = next(
        (
            index
            for index, (start, end) in enumerate(sentence_spans)
            if start <= match.start() < end
        ),
        None,
    )
    if sentence_index is None:
        return None

    start_index = max(0, sentence_index - config.SPAN_CONTEXT_SENTENCES)
    end_index = min(
        len(sentence_spans) - 1,
        sentence_index + config.SPAN_CONTEXT_SENTENCES,
    )
    query_start = sentence_spans[start_index][0]
    query_end = sentence_spans[end_index][1]
    query = context_text[query_start:query_end].strip()
    if not query:
        return None

    span_start = match.start() - query_start
    return _cap_context_query(query, span_start, len(match.group(0)))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.start()
        if start < end:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _cap_context_query(query: str, span_start: int, span_len: int) -> str:
    max_chars = config.SPAN_CONTEXT_MAX_CHARS
    if len(query) <= max_chars:
        return query

    span_end = span_start + span_len
    spare_chars = max(max_chars - span_len, 0)
    start = max(0, span_start - (spare_chars // 2))
    end = min(len(query), start + max_chars)
    start = max(0, end - max_chars)

    start = _right_word_boundary(query, start)
    end = _left_word_boundary(query, end)

    if start > span_start:
        start = _left_word_boundary(query, span_start)
    if end < span_end:
        end = _right_word_boundary(query, span_end)

    while end - start > max_chars and start < span_start:
        next_start = _right_word_boundary(query, start + 1)
        if next_start <= start or next_start > span_start:
            break
        start = next_start
    while end - start > max_chars and end > span_end:
        next_end = _left_word_boundary(query, end - 1)
        if next_end >= end or next_end < span_end:
            break
        end = next_end

    capped = query[start:end].strip()
    if len(capped) <= max_chars:
        return capped
    return capped[:max_chars].rsplit(" ", 1)[0].strip() or capped[:max_chars].strip()


def _right_word_boundary(text: str, index: int) -> int:
    if (
        index <= 0
        or index >= len(text)
        or text[index - 1].isspace()
        or text[index].isspace()
    ):
        return index
    boundary = text.find(" ", index)
    return len(text) if boundary == -1 else boundary + 1


def _left_word_boundary(text: str, index: int) -> int:
    if (
        index <= 0
        or index >= len(text)
        or text[index - 1].isspace()
        or text[index].isspace()
    ):
        return index
    boundary = text.rfind(" ", 0, index)
    return 0 if boundary == -1 else boundary


def _snap_vocabulary(alias_rows: list[tuple[str, str, bool]]) -> tuple[str, ...]:
    vocabulary: set[str] = set()
    for alias, _note_id, is_title in alias_rows:
        normalized = alias.strip().lower()
        if not normalized:
            continue
        vocabulary.add(normalized)
        if is_title:
            vocabulary.update(token for token in normalized.split() if token)
    return tuple(sorted(vocabulary))


def _bm25_query(window_text: str, snapped: list[tuple[str, str]]) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for _original, snapped_term in snapped:
        if snapped_term in seen:
            continue
        seen.add(snapped_term)
        terms.append(snapped_term)
    if not terms:
        return window_text
    return " ".join([window_text, *terms])
