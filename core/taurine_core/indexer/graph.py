from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from taurine_core import config
from taurine_core.schemas import EdgeType, Enrichment, GraphEdge, NoteRecord


@dataclass
class _Candidate:
    src: str
    dst: str
    mechanisms: dict[str, float] = field(default_factory=dict)
    shared: set[str] = field(default_factory=set)

    def add(self, mechanism: str, weight: float, terms: set[str] | None = None) -> None:
        if weight <= 0:
            return
        self.mechanisms[mechanism] = max(self.mechanisms.get(mechanism, 0.0), _bounded(weight))
        if terms:
            self.shared.update(terms)

    def edge(self) -> GraphEdge:
        mechanism_names = set(self.mechanisms)
        edge_type: EdgeType
        if len(mechanism_names) == 1:
            edge_type = next(iter(mechanism_names))  # type: ignore[assignment]
        else:
            edge_type = "mixed"
        remaining = 1.0
        for weight in self.mechanisms.values():
            remaining *= 1.0 - weight
        return GraphEdge(
            src=self.src,
            dst=self.dst,
            type=edge_type,
            weight=1.0 - remaining,
            shared=sorted(self.shared),
        )


def rebuild_all(index: object) -> None:
    edges = _cap_edges(_candidate_edges(index))
    index.replace_edges(edges)  # type: ignore[attr-defined]


def rebuild_for(index: object, note_id: str) -> None:
    existing = [
        edge
        for edge in index.graph().edges  # type: ignore[attr-defined]
        if note_id not in (edge.src, edge.dst)
    ]
    existing_degree: dict[str, int] = defaultdict(int)
    for edge in existing:
        existing_degree[edge.src] += 1
        existing_degree[edge.dst] += 1

    incident = [
        edge
        for edge in _candidate_edges(index)
        if note_id in (edge.src, edge.dst)
    ]
    incident.sort(key=lambda edge: edge.weight, reverse=True)

    selected: list[GraphEdge] = []
    target_degree = 0
    other_degree = dict(existing_degree)
    for edge in incident:
        other = edge.dst if edge.src == note_id else edge.src
        if target_degree >= config.GRAPH_MAX_DEGREE:
            break
        if other_degree.get(other, 0) >= config.GRAPH_MAX_DEGREE:
            continue
        selected.append(edge)
        target_degree += 1
        other_degree[other] = other_degree.get(other, 0) + 1

    index.replace_edges(selected, incident_note_id=note_id)  # type: ignore[attr-defined]


def _candidate_edges(index: object) -> list[GraphEdge]:
    notes: list[NoteRecord] = index.notes()  # type: ignore[attr-defined]
    candidates: dict[tuple[str, str], _Candidate] = {}
    _add_semantic_edges(index, notes, candidates)
    _add_shared_edges(index, notes, candidates, "topic")
    _add_shared_edges(index, notes, candidates, "entity")
    return [candidate.edge() for candidate in candidates.values()]


def _add_semantic_edges(
    index: object,
    notes: list[NoteRecord],
    candidates: dict[tuple[str, str], _Candidate],
) -> None:
    vectors = {
        note.note_id: vector
        for note in notes
        if (vector := _note_vector(index, note.note_id)) is not None
    }
    top_by_note: dict[str, set[str]] = {}
    sims: dict[tuple[str, str], float] = {}

    for note_id, vector in vectors.items():
        neighbors: list[tuple[str, float]] = []
        for other_id, other_vector in vectors.items():
            if other_id == note_id:
                continue
            score = float(vector @ other_vector)
            if score >= config.GRAPH_MIN_SIM:
                neighbors.append((other_id, score))
                sims[_key(note_id, other_id)] = score
        neighbors.sort(key=lambda item: item[1], reverse=True)
        top_by_note[note_id] = {other_id for other_id, _ in neighbors[: config.GRAPH_KNN]}

    for (left, right), score in sims.items():
        if right not in top_by_note.get(left, set()) and left not in top_by_note.get(right, set()):
            continue
        _candidate(candidates, left, right).add("semantic", score)


def _add_shared_edges(
    index: object,
    notes: list[NoteRecord],
    candidates: dict[tuple[str, str], _Candidate],
    mechanism: str,
) -> None:
    terms_by_note: dict[str, set[str]] = {}
    frequency: dict[str, int] = defaultdict(int)
    for note in notes:
        enrichment = index.enrichment_for(note.note_id)  # type: ignore[attr-defined]
        terms = _terms(enrichment, mechanism)
        terms_by_note[note.note_id] = terms
        for term in terms:
            frequency[term] += 1

    for left_index, left in enumerate(notes):
        left_terms = terms_by_note[left.note_id]
        if not left_terms:
            continue
        for right in notes[left_index + 1 :]:
            shared = left_terms & terms_by_note[right.note_id]
            if not shared:
                continue
            weight = sum(_term_weight(frequency[term]) for term in shared)
            if weight < config.GRAPH_MIN_SHARED_W:
                continue
            _candidate(candidates, left.note_id, right.note_id).add(mechanism, weight, shared)


def _note_vector(index: object, note_id: str) -> np.ndarray | None:
    gist_vectors = index.embedding_vectors(note_id, "gist")  # type: ignore[attr-defined]
    if gist_vectors:
        return _normalize(gist_vectors[0])
    chunk_vectors = index.embedding_vectors(note_id, "chunk")  # type: ignore[attr-defined]
    if not chunk_vectors:
        return None
    return _normalize(np.mean(np.vstack(chunk_vectors), axis=0).astype(np.float32))


def _terms(enrichment: Enrichment | None, mechanism: str) -> set[str]:
    if enrichment is None:
        return set()
    if mechanism == "topic":
        return {" ".join(topic.lower().split()) for topic in enrichment.topics if topic.strip()}
    return {entity.strip() for entity in enrichment.entities if entity.strip()}


def _candidate(
    candidates: dict[tuple[str, str], _Candidate],
    left: str,
    right: str,
) -> _Candidate:
    src, dst = _key(left, right)
    key = (src, dst)
    if key not in candidates:
        candidates[key] = _Candidate(src=src, dst=dst)
    return candidates[key]


def _key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _term_weight(note_count: int) -> float:
    return 1.0 / (1.0 + math.log(1.0 + note_count))


def _bounded(weight: float) -> float:
    return min(max(float(weight), 0.0), 0.999999)


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm == 0:
        return None
    return values / norm


def _cap_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    degree: dict[str, int] = defaultdict(int)
    selected: list[GraphEdge] = []
    for edge in sorted(edges, key=lambda item: item.weight, reverse=True):
        if degree[edge.src] >= config.GRAPH_MAX_DEGREE:
            continue
        if degree[edge.dst] >= config.GRAPH_MAX_DEGREE:
            continue
        selected.append(edge)
        degree[edge.src] += 1
        degree[edge.dst] += 1
    return selected
