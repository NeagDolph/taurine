"""Grounding engine: seeds → PPR → prior blend → grounding set (spec §4.3–4.5).

Owned state (per session):
- meeting seed accumulator (decayed sum of fused top-K across passes — the
  "entire transcription so far" as a PPR personalization vector),
- previous PPR vector (warm start),
- cached NoteGraphMatrix, rebuilt when ``index.graph_version()`` changes,
- stable short-id registries: note_id → "N#" and (src, dst) → "F#"
  (assigned once per session, never reused — citations must stay valid
  across summary revisions).

No LLM calls; target < 50 ms per pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .. import config
from ..indexer.rank import NoteGraphMatrix, recency_multiplier, source_rank
from ..indexer.store import Index
from ..schemas import GroundedFact, GroundingItem, GroundingWhy, NoteRecord


SECONDS_PER_DAY = 86_400.0
SEED_PRUNE_EPSILON = 1e-6
WHY_HEAD_ORDER = ("dense", "gliner", "bm25", "exact", "fact", "ppr")
_DEFAULT_HYSTERESIS = object()


@dataclass
class GroundingUpdate:
    """One pass's output, consumed by the trigger and the summarizer."""

    items: list[GroundingItem] = field(default_factory=list)   # top GROUNDING_TOP_K by blended score
    facts: list[GroundedFact] = field(default_factory=list)    # ≤ GROUNDING_FACTS_MAX, stable fact_ids
    revision: int = 0                                          # bumps when membership/order changes
    ppr_top: list[tuple[str, float]] = field(default_factory=list)  # for the map's retrieval event


class GroundingEngine:
    def __init__(
        self,
        index: Index,
        *,
        top_k: int | None = None,
        facts_max: int | None = None,
        seed_decay: float | None = None,
        hysteresis: float | None | object = _DEFAULT_HYSTERESIS,
        query_prompt: str | None = None,
    ) -> None:
        self.index = index
        self._top_k = config.GROUNDING_TOP_K if top_k is None else int(top_k)
        self._facts_max = (
            config.GROUNDING_FACTS_MAX if facts_max is None else int(facts_max)
        )
        self._seed_decay = (
            config.SEED_ACCUM_DECAY if seed_decay is None else float(seed_decay)
        )
        self._hysteresis = (
            config.GROUNDING_HYSTERESIS
            if hysteresis is _DEFAULT_HYSTERESIS
            else hysteresis
        )
        self._query_prompt = query_prompt
        self._seeds: dict[str, float] = {}
        self._prev_ppr_raw: np.ndarray | None = None
        self._prev_ppr_matrix_identity: int | None = None
        self._matrix: NoteGraphMatrix | None = None
        self._matrix_version: int | None = None
        self._note_short_ids: dict[str, str] = {}
        self._fact_short_ids: dict[tuple[str, str], str] = {}
        self._next_note_short_id = 1
        self._next_fact_short_id = 1
        self._previous_order: list[str] = []
        self._revision = 0
        self.last_blend: dict[str, dict[str, float]] = {}

    def export_state(self) -> dict[str, Any]:
        return {
            "seeds": {
                note_id: float(score)
                for note_id, score in sorted(self._seeds.items())
            },
            "note_short_ids": dict(sorted(self._note_short_ids.items())),
            "fact_short_ids": [
                {"src": src, "dst": dst, "short_id": short_id}
                for (src, dst), short_id in sorted(self._fact_short_ids.items())
            ],
            "next_note_short_id": self._next_note_short_id,
            "next_fact_short_id": self._next_fact_short_id,
            "previous_order": list(self._previous_order),
            "revision": self._revision,
        }

    def restore_state(self, state: dict) -> None:
        raw_seeds = state.get("seeds", {})
        self._seeds = (
            {
                str(note_id): float(score)
                for note_id, score in raw_seeds.items()
                if float(score) >= SEED_PRUNE_EPSILON
            }
            if isinstance(raw_seeds, dict)
            else {}
        )

        raw_note_ids = state.get("note_short_ids", {})
        self._note_short_ids = (
            {str(note_id): str(short_id) for note_id, short_id in raw_note_ids.items()}
            if isinstance(raw_note_ids, dict)
            else {}
        )

        self._fact_short_ids = {}
        raw_fact_ids = state.get("fact_short_ids", [])
        if isinstance(raw_fact_ids, list):
            for row in raw_fact_ids:
                if not isinstance(row, dict):
                    continue
                src = row.get("src")
                dst = row.get("dst")
                short_id = row.get("short_id")
                if src is None or dst is None or short_id is None:
                    continue
                self._fact_short_ids[_edge_key(str(src), str(dst))] = str(short_id)

        self._next_note_short_id = max(
            _state_counter(state.get("next_note_short_id"), "N"),
            _next_counter_from_short_ids(self._note_short_ids.values(), "N"),
        )
        self._next_fact_short_id = max(
            _state_counter(state.get("next_fact_short_id"), "F"),
            _next_counter_from_short_ids(self._fact_short_ids.values(), "F"),
        )

        raw_previous_order = state.get("previous_order", [])
        self._previous_order = (
            [str(note_id) for note_id in raw_previous_order]
            if isinstance(raw_previous_order, list)
            else []
        )
        self._revision = _state_counter(state.get("revision"), "")
        self._prev_ppr_raw = None
        self._prev_ppr_matrix_identity = None

    def on_pass(
        self,
        window_text: str,
        fused: Sequence[tuple[str, float]],
        exact_hits: Sequence[tuple[str, str]],
        now: float,
        *,
        head_hits: dict[str, set[str]] | None = None,
        decay_seeds: bool = True,
    ) -> GroundingUpdate:
        """Advance the accumulator and produce the current grounding set.

        ``fused``: RRF-fused (note_id, score), best first, from the heads
        (dense window, GLiNER spans, BM25, exact-as-list). ``exact_hits``:
        (note_id, alias) C1 matches this window (max-strength seeds; recorded
        in GroundingWhy.exact_alias).

        Steps (spec §4.3–4.5): decay+add seeds → PPR (warm start; matrix
        rebuilt on graph_version change) → blend
        ``rel = (1−β)·norm(fused) + β·norm(ppr)`` then
        ``final = rel × recency × source_rank × exp(QUALITY_LAMBDA·quality_rank)``
        → hysteresis vs previous set (challenger > incumbent × GROUNDING_HYSTERESIS)
        → top-K items + fact selection (edge facts within the set or set→top-5
        seeds, plus this pass's fact_dense_search hits).
        """
        fused_rows = _unique_ranked(fused)
        exact_aliases = _exact_aliases(exact_hits)
        self._advance_seeds(fused_rows, exact_aliases, decay=decay_seeds)

        matrix = self._graph_matrix()
        matrix_identity = id(matrix.matrix)
        warm_start = (
            self._prev_ppr_raw
            if self._prev_ppr_matrix_identity == matrix_identity
            else None
        )
        ppr_mass, raw = matrix.ppr(self._seeds, warm_start=warm_start)
        self._prev_ppr_raw = raw
        self._prev_ppr_matrix_identity = matrix_identity

        ppr_top = _top_items(ppr_mass, config.SEED_TOP_K)
        ppr_pool = _top_items(ppr_mass, config.GROUNDING_PPR_POOL_K)
        fact_hits = _fact_dense_search(
            self.index,
            window_text,
            self._facts_max,
            self._query_prompt,
        )
        notes_by_id = {note.note_id: note for note in self.index.notes()}
        active_ids = set(notes_by_id)

        fact_bonus: dict[str, float] = {}
        for src, dst, score, _fact in fact_hits:
            if src in active_ids:
                fact_bonus[src] = fact_bonus.get(src, 0.0) + max(0.0, float(score) / 2.0)
            if dst in active_ids:
                fact_bonus[dst] = fact_bonus.get(dst, 0.0) + max(0.0, float(score) / 2.0)

        fused_scores = {
            note_id: float(score)
            for note_id, score in fused_rows
            if note_id in active_ids
        }
        ppr_scores = {
            note_id: float(score)
            for note_id, score in ppr_pool
            if note_id in active_ids
        }
        candidate_order: dict[str, int] = {}
        for note_id, _score in fused_rows:
            if note_id in active_ids:
                candidate_order.setdefault(note_id, len(candidate_order))
        for note_id, _score in ppr_pool:
            if note_id in active_ids:
                candidate_order.setdefault(note_id, len(candidate_order))
        for note_id in sorted(fact_bonus):
            if note_id in active_ids:
                candidate_order.setdefault(note_id, len(candidate_order))
        candidate_ids = (
            set(fused_scores)
            | set(ppr_scores)
            | {note_id for note_id in fact_bonus if note_id in active_ids}
        )
        fused_norm = _minmax_norm(fused_scores, candidate_ids)
        ppr_norm = _minmax_norm(ppr_scores, candidate_ids)

        blend: dict[str, dict[str, float]] = {}
        for note_id in candidate_ids:
            note = notes_by_id.get(note_id)
            if note is None:
                continue
            rel = (
                (1.0 - config.PPR_BLEND) * fused_norm.get(note_id, 0.0)
                + config.PPR_BLEND * ppr_norm.get(note_id, 0.0)
                + fact_bonus.get(note_id, 0.0)
            )
            rel = min(1.0, max(0.0, rel))
            age_days = (float(now) - float(note.modified)) / SECONDS_PER_DAY
            recency = recency_multiplier(note.source, age_days)
            source = source_rank(note.source)
            quality = math.exp(config.QUALITY_LAMBDA * float(note.quality_rank))
            final = rel * recency * source * quality
            blend[note_id] = {
                "rel": rel,
                "recency": recency,
                "source_rank": source,
                "quality": quality,
                "final": final,
                "age_days": age_days,
            }

        ordered_ids = sorted(
            blend,
            key=lambda note_id: (
                -blend[note_id]["final"],
                candidate_order.get(note_id, len(candidate_order)),
                note_id,
            ),
        )
        selected_ids = self._apply_hysteresis(ordered_ids, blend, active_ids)
        selected_set = set(selected_ids)
        selected_blend = {note_id: blend[note_id] for note_id in selected_ids}
        self.last_blend = {
            note_id: {
                "rel": values["rel"],
                "recency": values["recency"],
                "source_rank": values["source_rank"],
                "quality": values["quality"],
                "final": values["final"],
            }
            for note_id, values in selected_blend.items()
        }

        facts = self._grounded_facts(selected_set, fact_hits, active_ids)
        items = [
            self._grounding_item(
                note_id,
                notes_by_id[note_id],
                selected_blend[note_id],
                ppr_mass.get(note_id, 0.0),
                exact_aliases,
                fact_bonus,
                head_hits or {},
            )
            for note_id in selected_ids
            if note_id in notes_by_id
        ]

        order = [item.note_id for item in items]
        if order != self._previous_order:
            self._revision += 1
            self._previous_order = order

        return GroundingUpdate(
            items=items,
            facts=facts,
            revision=self._revision,
            ppr_top=ppr_top,
        )

    def _advance_seeds(
        self,
        fused: list[tuple[str, float]],
        exact_aliases: dict[str, str],
        *,
        decay: bool,
    ) -> None:
        if decay:
            self._seeds = {
                note_id: score * self._seed_decay
                for note_id, score in self._seeds.items()
            }
        else:
            self._seeds = dict(self._seeds)
        for note_id, score in fused[: config.SEED_TOP_K]:
            self._seeds[note_id] = self._seeds.get(note_id, 0.0) + float(score)

        exact_score = max((score for _note_id, score in fused), default=1.0)
        for note_id in exact_aliases:
            self._seeds[note_id] = self._seeds.get(note_id, 0.0) + exact_score

        self._seeds = {
            note_id: score
            for note_id, score in self._seeds.items()
            if score >= SEED_PRUNE_EPSILON
        }

    def _graph_matrix(self) -> NoteGraphMatrix:
        version = int(self.index.graph_version())
        if self._matrix is None or self._matrix_version != version:
            self._matrix = NoteGraphMatrix.build(self.index)
            self._matrix_version = version
            self._prev_ppr_raw = None
            self._prev_ppr_matrix_identity = None
        return self._matrix

    def _apply_hysteresis(
        self,
        ordered_ids: list[str],
        blend: dict[str, dict[str, float]],
        active_ids: set[str],
    ) -> list[str]:
        selected: list[str] = []
        used: set[str] = set()
        limit = self._top_k

        for slot in range(limit):
            incumbent = (
                self._previous_order[slot]
                if slot < len(self._previous_order)
                else None
            )
            incumbent_valid = (
                incumbent is not None
                and incumbent in active_ids
                and incumbent in blend
                and incumbent not in used
            )
            challenger = next(
                (note_id for note_id in ordered_ids if note_id not in used),
                None,
            )

            if incumbent_valid and incumbent is not None:
                if challenger is None or challenger == incumbent:
                    chosen = incumbent
                elif self._hysteresis is None:
                    chosen = challenger
                elif blend[challenger]["final"] > (
                    blend[incumbent]["final"] * float(self._hysteresis)
                ):
                    chosen = challenger
                else:
                    chosen = incumbent
            else:
                chosen = challenger

            if chosen is None:
                break
            selected.append(chosen)
            used.add(chosen)

        for note_id in ordered_ids:
            if len(selected) >= limit:
                break
            if note_id in used:
                continue
            selected.append(note_id)
            used.add(note_id)

        return selected

    def _grounding_item(
        self,
        note_id: str,
        note: NoteRecord,
        blend: dict[str, float],
        ppr_mass: float,
        exact_aliases: dict[str, str],
        fact_bonus: dict[str, float],
        head_hits: dict[str, set[str]],
    ) -> GroundingItem:
        heads = set(head_hits.get(note_id, set()))
        if note_id in exact_aliases:
            heads.add("exact")
        if note_id in fact_bonus:
            heads.add("fact")
        if ppr_mass > 0.0:
            heads.add("ppr")
        ordered_heads = [head for head in WHY_HEAD_ORDER if head in heads]
        return GroundingItem(
            note_id=note_id,
            short_id=self._note_short_id(note_id),
            title=note.title,
            gist=note.gist,
            source=note.source,
            age_days=blend["age_days"],
            score=blend["final"],
            why=GroundingWhy(
                heads=ordered_heads,
                ppr_mass=float(ppr_mass),
                exact_alias=exact_aliases.get(note_id, ""),
            ),
        )

    def _grounded_facts(
        self,
        grounding_ids: set[str],
        fact_hits: list[tuple[str, str, float, str]],
        active_ids: set[str],
    ) -> list[GroundedFact]:
        selected: list[GroundedFact] = []
        seen: set[tuple[str, str]] = set()

        def append_fact(src: str, dst: str, text: str) -> None:
            if len(selected) >= self._facts_max:
                return
            key = _edge_key(src, dst)
            if key in seen or key[0] not in active_ids or key[1] not in active_ids:
                return
            fact = text.strip()
            if not fact:
                return
            seen.add(key)
            selected.append(
                GroundedFact(
                    fact_id=self._fact_short_id(key),
                    src=key[0],
                    dst=key[1],
                    text=fact,
                )
            )

        for src, dst, _score, fact in fact_hits:
            append_fact(src, dst, fact)

        top_seeds = {
            note_id
            for note_id, _score in sorted(
                self._seeds.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )[: config.GROUNDING_FACT_SEED_TOP_K]
        }
        within_edges: list[tuple[float, str, str, str]] = []
        seed_edges: list[tuple[float, str, str, str]] = []
        for edge in self.index.graph().edges:
            fact = edge.fact.strip()
            if not fact:
                continue
            if edge.src not in active_ids or edge.dst not in active_ids:
                continue
            endpoints = {edge.src, edge.dst}
            row = (float(edge.weight), edge.src, edge.dst, fact)
            if endpoints <= grounding_ids:
                within_edges.append(row)
            elif endpoints & grounding_ids and endpoints & top_seeds:
                seed_edges.append(row)

        for _weight, src, dst, fact in sorted(
            within_edges,
            key=lambda row: (row[0], row[1], row[2]),
            reverse=True,
        ):
            append_fact(src, dst, fact)
        for _weight, src, dst, fact in sorted(
            seed_edges,
            key=lambda row: (row[0], row[1], row[2]),
            reverse=True,
        ):
            append_fact(src, dst, fact)

        return selected

    def _note_short_id(self, note_id: str) -> str:
        short_id = self._note_short_ids.get(note_id)
        if short_id is None:
            short_id = f"N{self._next_note_short_id}"
            self._next_note_short_id += 1
            self._note_short_ids[note_id] = short_id
        return short_id

    def _fact_short_id(self, key: tuple[str, str]) -> str:
        fact_id = self._fact_short_ids.get(key)
        if fact_id is None:
            fact_id = f"F{self._next_fact_short_id}"
            self._next_fact_short_id += 1
            self._fact_short_ids[key] = fact_id
        return fact_id


def _unique_ranked(rows: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for index, (note_id, score) in enumerate(rows):
        first_seen.setdefault(note_id, index)
        scores[note_id] = max(scores.get(note_id, float("-inf")), float(score))
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], first_seen[item[0]], item[0]),
    )


def _exact_aliases(exact_hits: Sequence[tuple[str, str]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for note_id, alias in exact_hits:
        aliases.setdefault(note_id, alias)
    return aliases


def _top_items(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[
        :limit
    ]


def _minmax_norm(
    scores: dict[str, float],
    candidate_ids: set[str],
) -> dict[str, float]:
    if not candidate_ids:
        return {}
    values = {note_id: float(scores.get(note_id, 0.0)) for note_id in candidate_ids}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {note_id: 1.0 for note_id in candidate_ids}
    return {note_id: (value - low) / (high - low) for note_id, value in values.items()}


def _edge_key(src: str, dst: str) -> tuple[str, str]:
    left, right = sorted((src, dst))
    return left, right


def _fact_dense_search(
    index: Index,
    text: str,
    k: int,
    prompt: str | None,
) -> list[tuple[str, str, float, str]]:
    if prompt is None:
        return index.fact_dense_search(text, k)
    try:
        return index.fact_dense_search(text, k, prompt=prompt)  # type: ignore[call-arg]
    except TypeError as exc:
        if "prompt" not in str(exc):
            raise
        return index.fact_dense_search(text, k)


def _state_counter(value: Any, prefix: str) -> int:
    try:
        counter = int(value)
    except (TypeError, ValueError):
        counter = 0 if prefix == "" else 1
    return max(counter, 0 if prefix == "" else 1)


def _next_counter_from_short_ids(short_ids: Any, prefix: str) -> int:
    highest = 0
    for short_id in short_ids:
        text = str(short_id)
        if not text.startswith(prefix):
            continue
        try:
            highest = max(highest, int(text[len(prefix) :]))
        except ValueError:
            continue
    return highest + 1
