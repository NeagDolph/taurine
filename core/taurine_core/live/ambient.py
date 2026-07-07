from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
from typing import Protocol

from rapidfuzz.distance import Levenshtein

from taurine_core import config
from taurine_core.ports import IndexPort, RerankerPort, RerankError, SpanExtractorPort
from taurine_core.schemas import Expansion, Hit, RetrievalResult, SlateItem


class Clock(Protocol):
    def now(self) -> float:
        ...


@dataclass(frozen=True)
class _Candidate:
    note_id: str
    score: float


@dataclass(frozen=True)
class _ExpansionArrival:
    from_note_id: str
    note_id: str
    hop: int
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
        self.last_slate_adds: list[SlateItem] = []
        self.last_debug: dict[str, Any] | None = None
        self._index = index
        self._reranker = reranker
        self._clock = clock
        self._on_debug = on_debug
        self._span_extractor = span_extractor
        self._last_run_at: float | None = None
        self._snap_vocabulary = _snap_vocabulary(index.aliases())

    async def on_window(self, window_text: str) -> RetrievalResult | None:
        self.last_slate_adds = []
        self.last_debug = None
        now = self._clock.now()
        if (
            self._last_run_at is not None
            and now - self._last_run_at < config.C2_THROTTLE_S
        ):
            return None
        self._last_run_at = now

        salient = self._salient_query(window_text)
        gliner_spans = self._extract_spans(window_text)
        gliner_queries, snapped = self._gliner_queries(gliner_spans)
        extracted = [span for span, _score, _label in gliner_spans]
        dense = self._index.dense_search(window_text, config.RERANK_MAX_CANDIDATES)
        dense_salient = (
            self._index.dense_search(salient, config.RERANK_MAX_CANDIDATES)
            if salient
            else []
        )
        dense_gliner = self._dense_gliner(gliner_queries)
        bm25_query = _bm25_query(window_text, snapped)
        bm25 = self._index.bm25_search(bm25_query, config.RERANK_MAX_CANDIDATES)
        dense_scores = self._dense_scores(dense, dense_salient, dense_gliner)
        bm25_note_ids = {note_id for note_id, _score in bm25}

        base_direct_scores = self._rrf_scores(dense, dense_salient, dense_gliner, bm25)
        graph_vote = self._graph_vote(base_direct_scores)
        direct_scores = self._rrf_scores(
            dense,
            dense_salient,
            dense_gliner,
            bm25,
            graph_vote,
        )
        candidate_scores = dict(base_direct_scores)
        expansion_arrivals = self._expand(candidate_scores, base_direct_scores)
        for candidate in self._ordered_candidates(direct_scores):
            candidate_scores[candidate.note_id] = max(
                candidate_scores.get(candidate.note_id, float("-inf")),
                candidate.score,
            )

        pooled = self._ordered_candidates(candidate_scores)
        docs = self._docs_for_candidates(window_text, pooled)
        if not docs:
            self._record_debug(
                window=window_text,
                salient=salient,
                gliner_spans=gliner_spans,
                dense=dense,
                dense_salient=dense_salient,
                dense_gliner=dense_gliner,
                bm25=bm25,
                graph_vote=graph_vote,
                bm25_query=bm25_query,
                direct_scores=direct_scores,
                expansion_arrivals=expansion_arrivals,
                rerank_used=False,
                rerank_error=None,
                ordered=[],
                floor_dropped=[],
                slate_adds=[],
                protected=[],
                snapped=snapped,
            )
            return RetrievalResult(heard=window_text, hits=[], expansions=[], extracted=extracted)

        ordered, rerank_used, rerank_error = await self._reranked_or_fallback(
            window_text,
            docs,
            candidate_scores,
        )
        filtered = self._apply_floor(
            ordered,
            dense_scores,
            bm25_note_ids,
            set(expansion_arrivals),
        )
        filtered_ids = {note_id for note_id, _score in filtered}
        floor_dropped = [note_id for note_id, _score in ordered if note_id not in filtered_ids]
        ordered_ranks = {
            note_id: rank
            for rank, (note_id, _score) in enumerate(ordered, start=1)
        }
        protected_ids = self._floor_passing_protected_ids(
            direct_scores,
            dense,
            dense_scores,
            bm25_note_ids,
            set(expansion_arrivals),
            ordered_ranks,
        )
        top, protected = self._protect_fused(filtered, protected_ids, ordered_ranks)

        expansions = [
            Expansion(
                from_note_id=arrival.from_note_id,
                note_id=arrival.note_id,
                hop=arrival.hop,
                score=arrival.score,
            )
            for arrival in sorted(
                expansion_arrivals.values(),
                key=lambda item: item.score,
                reverse=True,
            )
            if arrival.note_id in {note_id for note_id, _score in docs}
        ]

        hits = [
            Hit(note_id=note_id, rank=ordered_ranks.get(note_id, rank), score=score)
            for rank, (note_id, score) in enumerate(top, start=1)
        ]
        self.last_slate_adds = [
            self._slate_item(note_id, score, window_text, expansion_arrivals)
            for note_id, score in top
        ]
        self._record_debug(
            window=window_text,
            salient=salient,
            gliner_spans=gliner_spans,
            dense=dense,
            dense_salient=dense_salient,
            dense_gliner=dense_gliner,
            bm25=bm25,
            graph_vote=graph_vote,
            bm25_query=bm25_query,
            direct_scores=direct_scores,
            expansion_arrivals=expansion_arrivals,
            rerank_used=rerank_used,
            rerank_error=rerank_error,
            ordered=ordered,
            floor_dropped=floor_dropped,
            slate_adds=[note_id for note_id, _score in top],
            protected=protected,
            snapped=snapped,
        )
        return RetrievalResult(
            heard=window_text,
            hits=hits,
            expansions=expansions,
            extracted=extracted,
        )

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

    def _salient_query(self, window_text: str) -> str | None:
        seen: set[str] = set()
        terms: list[tuple[int, str, int]] = []
        for index, term in enumerate(re.findall(r"[a-z0-9]+", window_text.lower())):
            if (
                len(term) < config.SALIENT_MIN_TERM_LEN
                or term in config.ALIAS_STOPWORDS
                or term in seen
            ):
                continue
            doc_frequency = self._index.doc_frequency(term)
            if doc_frequency < config.SALIENT_MIN_DF:
                continue
            seen.add(term)
            terms.append((index, term, doc_frequency))

        if not terms:
            return None

        scored = [
            (1.0 / (1 + doc_frequency), index, term)
            for index, term, doc_frequency in terms
        ]
        selected = {
            term
            for _score, _index, term in sorted(
                scored,
                key=lambda item: (-item[0], item[1]),
            )[: config.SALIENT_TERMS]
        }
        query = " ".join(
            term for _index, term, _doc_frequency in terms if term in selected
        )
        return query or None

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
    ) -> tuple[list[str], list[tuple[str, str]]]:
        queries: list[str] = []
        snapped: list[tuple[str, str]] = []
        for span, _span_score, label in gliner_spans:
            queries.append(span)
            snapped_span = self._snap_span(span, label)
            if snapped_span is None:
                continue
            queries.append(snapped_span)
            snapped.append((span, snapped_span))
        return queries, snapped

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

    def _graph_vote(
        self,
        direct_scores: dict[str, float],
    ) -> list[tuple[str, float]]:
        vote_scores: dict[str, float] = {}
        for candidate in self._ordered_candidates(direct_scores)[: config.GRAPH_VOTE_TOP]:
            for neighbor_id, edge_weight in self._index.neighbors(candidate.note_id):
                vote_scores[neighbor_id] = vote_scores.get(neighbor_id, 0.0) + (
                    candidate.score * edge_weight * config.GRAPH_VOTE_WEIGHT
                )
        return sorted(
            vote_scores.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[: config.RERANK_MAX_CANDIDATES]

    @staticmethod
    def _dense_scores(
        *ranked_lists: list[tuple[str, float]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for rows in ranked_lists:
            for note_id, score in rows:
                scores[note_id] = max(scores.get(note_id, float("-inf")), score)
        return scores

    def _expand(
        self,
        candidate_scores: dict[str, float],
        direct_scores: dict[str, float],
    ) -> dict[str, _ExpansionArrival]:
        arrivals: dict[str, _ExpansionArrival] = {}
        seeds = self._ordered_candidates(direct_scores)[: config.EXPAND_SEEDS]
        for seed in seeds:
            for neighbor_id, edge_weight in self._index.neighbors(seed.note_id):
                if neighbor_id in direct_scores:
                    continue
                score = seed.score * edge_weight * config.HOP_DECAY
                if score <= candidate_scores.get(neighbor_id, float("-inf")):
                    continue
                candidate_scores[neighbor_id] = score
                arrivals[neighbor_id] = _ExpansionArrival(
                    from_note_id=seed.note_id,
                    note_id=neighbor_id,
                    hop=1,
                    score=score,
                )
        return arrivals

    def _ordered_candidates(self, scores: dict[str, float]) -> list[_Candidate]:
        return [
            _Candidate(note_id, score)
            for note_id, score in sorted(
                scores.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        ][: config.RERANK_MAX_CANDIDATES]

    def _docs_for_candidates(
        self,
        window_text: str,
        candidates: list[_Candidate],
    ) -> list[tuple[str, str]]:
        docs: list[tuple[str, str]] = []
        for candidate in candidates:
            note = self._index.note(candidate.note_id)
            if note is None:
                continue
            best_chunk = self._index.best_chunk(candidate.note_id, window_text)
            docs.append(
                (
                    candidate.note_id,
                    f"{note.title} — {note.gist}\n{best_chunk}",
                )
            )
        return docs

    async def _reranked_or_fallback(
        self,
        window_text: str,
        docs: list[tuple[str, str]],
        candidate_scores: dict[str, float],
    ) -> tuple[list[tuple[str, float]], bool, str | None]:
        doc_ids = {note_id for note_id, _doc in docs}
        fallback = [
            (candidate.note_id, candidate.score)
            for candidate in self._ordered_candidates(candidate_scores)
            if candidate.note_id in doc_ids
        ]
        try:
            reranked = await self._reranker.rerank(window_text, docs)
        except (RerankError, TimeoutError) as exc:
            return fallback, False, str(exc)

        seen: set[str] = set()
        ordered: list[tuple[str, float]] = []
        for note_id, score in reranked:
            if note_id not in doc_ids or note_id in seen:
                continue
            ordered.append((note_id, score))
            seen.add(note_id)

        ordered.extend((note_id, score) for note_id, score in fallback if note_id not in seen)
        return ordered, True, None

    def _apply_floor(
        self,
        ordered: list[tuple[str, float]],
        dense_scores: dict[str, float],
        bm25_note_ids: set[str],
        graph_note_ids: set[str],
    ) -> list[tuple[str, float]]:
        filtered: list[tuple[str, float]] = []
        for note_id, score in ordered:
            has_bm25_hit = note_id in bm25_note_ids
            has_graph_path = note_id in graph_note_ids
            dense_score = dense_scores.get(note_id)
            if (
                not has_bm25_hit
                and not has_graph_path
                and (dense_score is None or dense_score < config.MIN_COSINE)
            ):
                continue
            filtered.append((note_id, score))
        return filtered

    def _floor_passing_protected_ids(
        self,
        direct_scores: dict[str, float],
        dense: list[tuple[str, float]],
        dense_scores: dict[str, float],
        bm25_note_ids: set[str],
        graph_note_ids: set[str],
        ordered_ranks: dict[str, int],
    ) -> set[str]:
        fused_protected = [
            (candidate.note_id, candidate.score)
            for candidate in self._ordered_candidates(direct_scores)[: config.FUSED_PROTECT]
            if candidate.note_id in ordered_ranks
        ]
        dense_protected = [
            (note_id, score)
            for note_id, score in dense[: config.DENSE_PROTECT]
            if note_id in ordered_ranks
        ]
        return {
            note_id
            for note_id, _score in self._apply_floor(
                [*fused_protected, *dense_protected],
                dense_scores,
                bm25_note_ids,
                graph_note_ids,
            )
        }

    def _protect_fused(
        self,
        filtered: list[tuple[str, float]],
        protected_ids: set[str],
        ordered_ranks: dict[str, int],
    ) -> tuple[list[tuple[str, float]], list[str]]:
        top = list(filtered[: config.C2_TOP_K])
        top_ids = {note_id for note_id, _score in top}
        appended: list[str] = []
        for note_id, score in filtered[config.C2_TOP_K :]:
            if note_id not in protected_ids or note_id in top_ids:
                continue
            top.append((note_id, score))
            top_ids.add(note_id)
            appended.append(note_id)

        while len(top) > config.C2_TOP_K:
            drop_candidates = [
                index
                for index, (note_id, _score) in enumerate(top)
                if note_id not in protected_ids
            ] or list(range(len(top)))
            drop_index = max(
                drop_candidates,
                key=lambda index: ordered_ranks.get(top[index][0], index + 1),
            )
            dropped_id = top[drop_index][0]
            del top[drop_index]
            top_ids.discard(dropped_id)
        return top, [note_id for note_id in appended if note_id in top_ids]

    def _slate_item(
        self,
        note_id: str,
        score: float,
        window_text: str,
        expansion_arrivals: dict[str, _ExpansionArrival],
    ) -> SlateItem:
        note = self._index.note(note_id)
        if note is None:
            raise ValueError(f"missing note for candidate {note_id}")
        arrival = expansion_arrivals.get(note_id)
        return SlateItem(
            note_id=note_id,
            title=note.title,
            gist=note.gist,
            score=score,
            source="search",
            heard=window_text,
            decay=config.DECAY_SEARCH,
            added_rev=0,
            via_note_id=arrival.from_note_id if arrival is not None else None,
            hop=arrival.hop if arrival is not None else None,
        )

    def _record_debug(
        self,
        *,
        window: str,
        salient: str | None,
        gliner_spans: list[tuple[str, float, str]],
        dense: list[tuple[str, float]],
        dense_salient: list[tuple[str, float]],
        dense_gliner: list[tuple[str, float]],
        bm25: list[tuple[str, float]],
        graph_vote: list[tuple[str, float]],
        bm25_query: str,
        direct_scores: dict[str, float],
        expansion_arrivals: dict[str, _ExpansionArrival],
        rerank_used: bool,
        rerank_error: str | None,
        ordered: list[tuple[str, float]],
        floor_dropped: list[str],
        slate_adds: list[str],
        protected: list[str],
        snapped: list[tuple[str, str]],
    ) -> None:
        debug = {
            "window": window,
            "salient": salient,
            "gliner_spans": _rows(gliner_spans),
            "dense": _rows(dense),
            "dense_salient": _rows(dense_salient),
            "dense_gliner": _rows(dense_gliner),
            "bm25": _rows(bm25),
            "graph_vote": _rows(graph_vote[:10]),
            "fused": _rows(
                [
                    (candidate.note_id, candidate.score)
                    for candidate in self._ordered_candidates(direct_scores)
                ]
            ),
            "expansions": [
                {"from": arrival.from_note_id, "to": arrival.note_id, "score": arrival.score}
                for arrival in sorted(
                    expansion_arrivals.values(),
                    key=lambda item: item.score,
                    reverse=True,
                )
            ],
            "rerank_used": rerank_used,
            "rerank_error": rerank_error,
            "ordered": _rows(ordered),
            "floor_dropped": floor_dropped,
            "slate_adds": slate_adds,
            "protected": protected,
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


def _rows(rows: list[tuple[Any, ...]]) -> list[list[Any]]:
    return [list(row) for row in rows]


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
