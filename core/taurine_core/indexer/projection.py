"""Deterministic degree-capped note projection derived from the base graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from taurine_core import config
from taurine_core.indexer.base_graph import BaseGraph


@dataclass(frozen=True)
class ProjectionPassage:
    passage_id: str
    note_id: str


@dataclass(frozen=True)
class ProjectionEdge:
    src_note_id: str
    dst_note_id: str
    edge_type: str
    weight: float
    shared: tuple[str, ...]
    fact_ids: tuple[str, ...]
    mechanisms: Mapping[str, float]


@dataclass
class _Candidate:
    scores: dict[str, list[float]] = field(default_factory=dict)
    shared: set[str] = field(default_factory=set)
    shared_entity_ids: set[str] = field(default_factory=set)
    fact_ids: set[str] = field(default_factory=set)

    def add(self, mechanism: str, score: float) -> None:
        self.scores.setdefault(mechanism, []).append(min(1.0, max(0.0, float(score))))


@dataclass(frozen=True)
class _RenderedCandidate:
    edge: ProjectionEdge
    shared_entity_count: int


def build_projection(
    graph: BaseGraph,
    passages: Sequence[ProjectionPassage],
    entity_labels: Mapping[str, str],
    *,
    entity_types: Mapping[str, str] | None = None,
    max_degree: int,
) -> tuple[ProjectionEdge, ...]:
    passage_note = {row.passage_id: row.note_id for row in passages}
    candidates: dict[tuple[str, str], _Candidate] = {}

    seen_semantic: set[tuple[str, str]] = set()
    entity_note_strength: dict[str, dict[str, float]] = {}
    for edge in graph.edges:
        if edge.kind == "semantic":
            passage_pair = tuple(sorted((edge.src_id, edge.dst_id)))
            if passage_pair in seen_semantic:
                continue
            seen_semantic.add(passage_pair)
            left_note = passage_note.get(edge.src_id)
            right_note = passage_note.get(edge.dst_id)
            if left_note is None or right_note is None or left_note == right_note:
                continue
            _candidate(candidates, left_note, right_note).add("semantic", edge.weight)
        elif edge.kind == "contains" and edge.src_id in passage_note:
            note_id = passage_note[edge.src_id]
            strengths = entity_note_strength.setdefault(edge.dst_id, {})
            strengths[note_id] = max(strengths.get(note_id, 0.0), edge.weight)

    for entity_id, strengths in sorted(entity_note_strength.items()):
        etype = entity_types.get(entity_id, "") if entity_types is not None else ""
        etype_weight = max(
            0.0,
            float(
                config.PROJECTION_ENTITY_ETYPE_WEIGHTS.get(
                    etype,
                    config.PROJECTION_ENTITY_ETYPE_DEFAULT,
                )
            ),
        )
        if etype_weight <= 0.0:
            continue
        note_ids = sorted(strengths)
        for index, left in enumerate(note_ids):
            for right in note_ids[index + 1 :]:
                contribution = strengths[left] * strengths[right] * etype_weight
                if contribution <= 0.0:
                    continue
                candidate = _candidate(candidates, left, right)
                candidate.add("entity", contribution)
                candidate.shared_entity_ids.add(entity_id)
                label = entity_labels.get(entity_id)
                if label:
                    candidate.shared.add(label)

    for edge in graph.edges:
        if edge.kind != "fact" or edge.direction != "forward":
            continue
        left_notes = sorted(entity_note_strength.get(edge.src_id, {}))
        right_notes = sorted(entity_note_strength.get(edge.dst_id, {}))
        # L1: a traversal-ineligible endpoint (date/pronoun/hub) has no
        # contains-derived notes; fall back to the fact's own provenance so
        # the fact still connects its source notes to notes holding the OTHER
        # (eligible) endpoint. Dates-only facts (both sides empty) stay out.
        provenance = sorted(edge.metadata.get("source_note_ids", ()))
        if left_notes and not right_notes:
            right_notes = provenance
        elif right_notes and not left_notes:
            left_notes = provenance
        for left in left_notes:
            for right in right_notes:
                if left == right:
                    continue
                candidate = _candidate(candidates, left, right)
                candidate.add("fact", edge.weight)
                candidate.fact_ids.update(edge.fact_ids)

    rendered = [_render(pair, candidate) for pair, candidate in candidates.items()]
    rendered.sort(
        key=lambda row: (
            -row.edge.weight,
            row.edge.src_note_id,
            row.edge.dst_note_id,
        )
    )
    selected = _select_edges(rendered, max(0, int(max_degree)))
    return tuple(sorted(selected, key=lambda row: (row.src_note_id, row.dst_note_id)))


def _candidate(
    candidates: dict[tuple[str, str], _Candidate],
    left: str,
    right: str,
) -> _Candidate:
    pair = (left, right) if left < right else (right, left)
    return candidates.setdefault(pair, _Candidate())


def _render(pair: tuple[str, str], candidate: _Candidate) -> _RenderedCandidate:
    # Within a mechanism: strongest signal + a bounded support bonus (§10.7's
    # "maximum plus bounded supporting-edge bonus"). Noisy-or here saturated —
    # six shared friend-names or overlapping passage pairs pinned the pair at
    # ~1.0 regardless of how weak each link was, flattening the whole map.
    mechanisms = {
        mechanism: _strongest_plus_support(scores)
        for mechanism, scores in sorted(candidate.scores.items())
    }
    # Across mechanisms (≤3 terms) noisy-or stays: independent evidence kinds
    # legitimately reinforce.
    weight = _noisy_or(list(mechanisms.values()))
    if set(mechanisms) == {"semantic"}:
        edge_type = "semantic"
    elif set(mechanisms) == {"entity"}:
        edge_type = "entity"
    else:
        edge_type = "mixed"
    return _RenderedCandidate(
        edge=ProjectionEdge(
            src_note_id=pair[0],
            dst_note_id=pair[1],
            edge_type=edge_type,
            weight=weight,
            shared=tuple(
                sorted(candidate.shared, key=lambda value: (value.casefold(), value))
            ),
            fact_ids=tuple(sorted(candidate.fact_ids)),
            mechanisms=mechanisms,
        ),
        shared_entity_count=len(candidate.shared_entity_ids),
    )


def _select_edges(
    candidates: list[_RenderedCandidate],
    max_degree: int,
) -> list[ProjectionEdge]:
    if max_degree <= 0:
        return []
    notes = sorted(
        {
            note_id
            for candidate in candidates
            for note_id in (candidate.edge.src_note_id, candidate.edge.dst_note_id)
        }
    )
    parent = {note_id: note_id for note_id in notes}
    degree = {note_id: 0 for note_id in notes}
    selected: list[ProjectionEdge] = []
    selected_pairs: set[tuple[str, str]] = set()

    def find(note_id: str) -> str:
        while parent[note_id] != note_id:
            parent[note_id] = parent[parent[note_id]]
            note_id = parent[note_id]
        return note_id

    def add(edge: ProjectionEdge) -> None:
        selected.append(edge)
        selected_pairs.add((edge.src_note_id, edge.dst_note_id))
        degree[edge.src_note_id] += 1
        degree[edge.dst_note_id] += 1

    # L2: the spanning-forest phase is pure — no degree cap and no weight
    # floor — so every note keeps its single best route into the graph even
    # when that route is weak, and a hub note bridging more components than
    # max_degree still connects them all. The cap and the low-value floor
    # apply only to the fill phase below.
    for candidate in candidates:
        edge = candidate.edge
        left_root, right_root = find(edge.src_note_id), find(edge.dst_note_id)
        if left_root == right_root:
            continue
        add(edge)
        parent[max(left_root, right_root)] = min(left_root, right_root)

    fill_floor = float(config.PROJECTION_FILL_MIN_WEIGHT)
    min_shared = max(0, int(config.PROJECTION_ENTITY_MIN_SHARED))
    for candidate in candidates:
        edge = candidate.edge
        if (edge.src_note_id, edge.dst_note_id) in selected_pairs:
            continue
        if edge.weight < fill_floor:
            continue
        has_non_entity_support = any(
            edge.mechanisms.get(mechanism, 0.0) > 0.0
            for mechanism in ("semantic", "fact")
        )
        if (
            edge.mechanisms.get("entity", 0.0) > 0.0
            and not has_non_entity_support
            and candidate.shared_entity_count < min_shared
        ):
            continue
        if degree[edge.src_note_id] >= max_degree or degree[edge.dst_note_id] >= max_degree:
            continue
        add(edge)
    return selected


def _noisy_or(scores: list[float]) -> float:
    remaining = 1.0
    for score in scores:
        remaining *= 1.0 - min(1.0, max(0.0, score))
    return 1.0 - remaining


def _strongest_plus_support(scores: list[float]) -> float:
    """Max score plus a small bounded bonus per additional supporting link.

    Mirrors the retrieval engine's note aggregation: one strong, specific
    connection outranks any pile of weak ones, and extra shared entities or
    passage pairs add at most SUPPORT_BONUS_CAP on top.
    """
    if not scores:
        return 0.0
    strongest = max(scores)
    bonus = min(_SUPPORT_BONUS_CAP, _SUPPORT_BONUS * (len(scores) - 1))
    return min(1.0, strongest + bonus)


_SUPPORT_BONUS = 0.05
_SUPPORT_BONUS_CAP = 0.15
