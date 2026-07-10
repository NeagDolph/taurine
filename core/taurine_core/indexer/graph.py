from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz.distance import Levenshtein

from taurine_core import config
from taurine_core.schemas import EdgeType, Enrichment, GraphEdge, NoteRecord

_SEPARATOR_RE = re.compile(r"[-_]+")
_WHITESPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


@dataclass
class _Candidate:
    src: str
    dst: str
    mechanisms: dict[str, float] = field(default_factory=dict)
    shared: set[str] = field(default_factory=set)
    semantic_score: float | None = None

    def add(self, mechanism: str, weight: float, terms: set[str] | None = None) -> None:
        if weight <= 0:
            return
        bounded = _bounded(weight)
        self.mechanisms[mechanism] = max(self.mechanisms.get(mechanism, 0.0), bounded)
        if mechanism == "semantic":
            self.semantic_score = max(self.semantic_score or 0.0, float(weight))
        if terms:
            self.shared.update(terms)

    def edge(self, *, fact: str = "", fact_provenance: str = "") -> GraphEdge:
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
            fact=fact,
            fact_provenance=fact_provenance,
        )


@dataclass(frozen=True)
class _EntitySpecificity:
    eligible_ids: frozenset[int]
    eligible_terms: frozenset[tuple[str, str]]
    labels: Mapping[int, str]


def embed_texts(texts: list[str]) -> np.ndarray:
    from .embed import embed_texts as real_embed_texts

    return real_embed_texts(texts)


def rebuild_all(index: object) -> None:
    kept, margin = _candidates(index)
    edges = _cap_edges(_apply_judgments(index, kept, margin, _valid_judgments(index)))
    index.replace_edges(edges)  # type: ignore[attr-defined]


def _candidate_edges(index: object) -> list[GraphEdge]:
    kept, _margin = _candidates(index)
    return [candidate.edge() for candidate in kept]


def _candidates(index: object) -> tuple[list[_Candidate], list[_Candidate]]:
    notes: list[NoteRecord] = index.notes()  # type: ignore[attr-defined]
    kept: dict[tuple[str, str], _Candidate] = {}
    relaxed: dict[tuple[str, str], _Candidate] = {}
    _canonicalize_extractions(index, notes)
    enrichments, resolved_terms = _term_resolution(index, notes)
    specificity = _entity_specificity(index, notes)
    _add_semantic_edges(index, notes, kept, relaxed)
    _add_shared_edges(
        notes, kept, relaxed, "topic", enrichments, resolved_terms, specificity
    )
    _add_shared_edges(
        notes, kept, relaxed, "entity", enrichments, resolved_terms, specificity
    )
    _add_relation_edges(index, notes, kept, specificity)
    margin = [
        candidate for key, candidate in relaxed.items() if key not in kept
    ]
    return list(kept.values()), _cap_margin_candidates(margin)


def _canonicalize_extractions(index: object, notes: list[NoteRecord]) -> None:
    if not hasattr(index, "replace_note_extraction"):
        return

    enrichments: dict[str, Enrichment] = {}
    term_counts_by_etype: dict[str, Counter[str]] = defaultdict(Counter)
    all_terms: set[str] = set()
    for note in notes:
        enrichment = index.enrichment_for(note.note_id)  # type: ignore[attr-defined]
        if enrichment is None:
            continue
        enrichments[note.note_id] = enrichment
        for entity in enrichment.entities:
            term = _entity_term(entity.surface, entity.etype)
            if not term:
                continue
            term_counts_by_etype[entity.etype][term] += 1
            all_terms.add(term)

    if not term_counts_by_etype:
        _replace_entity_surfaces(index, {})
        return

    _ensure_term_vectors(index, all_terms)
    resolved_by_etype = {
        etype: resolve_terms(index, counts)
        for etype, counts in term_counts_by_etype.items()
    }

    for note_id, enrichment in enrichments.items():
        surface_map: dict[str, str] = {}
        entity_accumulator: dict[tuple[str, str], tuple[float, int]] = {}
        changed = False
        for entity in enrichment.entities:
            normalized = _entity_term(entity.surface, entity.etype)
            if not normalized:
                continue
            canonical = resolved_by_etype.get(entity.etype, {}).get(normalized, normalized)
            surface_map[entity.surface] = canonical
            surface_map[normalized] = canonical
            if canonical != entity.surface:
                changed = True
            key = (canonical, entity.etype)
            prior_conf, prior_count = entity_accumulator.get(key, (0.0, 0))
            entity_accumulator[key] = (
                max(prior_conf, float(entity.conf)),
                prior_count + int(entity.mentions),
            )

        relations: list[tuple[str, str, str, float]] = []
        for relation in enrichment.relations:
            head = _mapped_relation_surface(relation.head, surface_map)
            tail = _mapped_relation_surface(relation.tail, surface_map)
            if head != relation.head or tail != relation.tail:
                changed = True
            relations.append((head, relation.rel_type, tail, float(relation.conf)))

        if not changed:
            continue

        entities = [
            (canonical, etype, conf, count)
            for (canonical, etype), (conf, count) in entity_accumulator.items()
        ]
        index.replace_note_extraction(note_id, entities, relations)  # type: ignore[attr-defined]

    _replace_entity_surfaces(index, resolved_by_etype)


def _entity_term(surface: str, etype: str) -> str:
    return _canonicalize_term(surface, lowercase=etype == "concept")


def _mapped_relation_surface(surface: str, surface_map: Mapping[str, str]) -> str:
    stripped = surface.strip()
    if stripped in surface_map:
        return surface_map[stripped]
    normalized = _canonicalize_term(stripped, lowercase=False)
    return surface_map.get(normalized, stripped)


def _replace_entity_surfaces(
    index: object,
    resolved_by_etype: Mapping[str, Mapping[str, str]],
) -> None:
    replace_entity_surfaces = getattr(index, "replace_entity_surfaces", None)
    if not callable(replace_entity_surfaces):
        return
    entity_ids = {
        (canonical, etype): int(entity_id)
        for entity_id, canonical, etype, _df in index.entity_rows()  # type: ignore[attr-defined]
    }
    rows: list[tuple[int, str]] = []
    for etype, resolved in resolved_by_etype.items():
        for surface, canonical in resolved.items():
            entity_id = entity_ids.get((canonical, etype))
            if entity_id is not None:
                rows.append((entity_id, surface))
    replace_entity_surfaces(rows)


def _add_semantic_edges(
    index: object,
    notes: list[NoteRecord],
    kept: dict[tuple[str, str], _Candidate],
    relaxed: dict[tuple[str, str], _Candidate],
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
            if score >= config.MARGIN_SEMANTIC_SIM:
                _candidate(relaxed, note_id, other_id).add("semantic", score)
            if score >= config.GRAPH_MIN_SIM:
                neighbors.append((other_id, score))
                key = _key(note_id, other_id)
                sims[key] = max(sims.get(key, 0.0), score)
        neighbors.sort(key=lambda item: item[1], reverse=True)
        top_by_note[note_id] = {other_id for other_id, _ in neighbors[: config.GRAPH_KNN]}

    for (left, right), score in sims.items():
        if right not in top_by_note.get(left, set()) and left not in top_by_note.get(right, set()):
            continue
        _candidate(kept, left, right).add("semantic", score)


def _add_shared_edges(
    notes: list[NoteRecord],
    kept: dict[tuple[str, str], _Candidate],
    relaxed: dict[tuple[str, str], _Candidate],
    mechanism: str,
    enrichments: dict[str, Enrichment | None],
    resolved_terms: dict[str, str],
    specificity: _EntitySpecificity | None,
) -> None:
    terms_by_note: dict[str, set[str]] = {}
    frequency: dict[str, int] = defaultdict(int)
    for note in notes:
        terms = {
            resolved_terms.get(term, term)
            for term, etype in _term_keys(enrichments.get(note.note_id), mechanism)
            if _term_is_specific(term, etype, resolved_terms, specificity)
        }
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
            if weight >= config.MARGIN_SHARED_W:
                _candidate(relaxed, left.note_id, right.note_id).add(
                    mechanism, weight, shared
                )
            if weight >= config.GRAPH_MIN_SHARED_W:
                _candidate(kept, left.note_id, right.note_id).add(
                    mechanism, weight, shared
                )


def _add_relation_edges(
    index: object,
    notes: list[NoteRecord],
    kept: dict[tuple[str, str], _Candidate],
    specificity: _EntitySpecificity | None,
) -> None:
    if not (
        hasattr(index, "entity_rows")
        and hasattr(index, "note_entity_rows")
        and hasattr(index, "relation_rows")
    ):
        return

    note_ids = {note.note_id for note in notes}
    entity_labels = (
        dict(specificity.labels)
        if specificity is not None
        else {
            entity_id: canonical
            for entity_id, canonical, _etype, _df in index.entity_rows()  # type: ignore[attr-defined]
        }
    )
    eligible_entity_ids = specificity.eligible_ids if specificity is not None else None
    notes_by_entity: dict[int, set[str]] = defaultdict(set)
    for note_id, entity_id, _conf, _count in index.note_entity_rows():  # type: ignore[attr-defined]
        if note_id in note_ids:
            notes_by_entity[int(entity_id)].add(note_id)

    for note_id, src_entity, rel_type, dst_entity, conf in index.relation_rows():  # type: ignore[attr-defined]
        if note_id not in note_ids:
            continue
        src_entity = int(src_entity)
        dst_entity = int(dst_entity)
        if eligible_entity_ids is not None and (
            src_entity not in eligible_entity_ids or dst_entity not in eligible_entity_ids
        ):
            continue
        weight = config.REL_EDGE_W * conf
        if rel_type == "related_to":
            weight *= config.REL_CATCHALL_FACTOR
        if weight <= 0.0:
            continue
        shared = {
            label
            for label in (entity_labels.get(src_entity), entity_labels.get(dst_entity))
            if label
        }
        for src_note in notes_by_entity.get(src_entity, set()):
            for dst_note in notes_by_entity.get(dst_entity, set()):
                if src_note == dst_note:
                    continue
                if note_id not in (src_note, dst_note):
                    continue
                _candidate(kept, src_note, dst_note).add("entity", weight, shared)


def pair_sig(
    index: object,
    src: str,
    dst: str,
    note_hashes: Mapping[str, str] | None = None,
) -> str:
    src, dst = _key(src, dst)
    if note_hashes is None:
        left = index.note(src)  # type: ignore[attr-defined]
        right = index.note(dst)  # type: ignore[attr-defined]
        if left is None or right is None:
            raise KeyError(f"cannot sign missing note pair {src!r}, {dst!r}")
        left_hash = left.content_hash
        right_hash = right.content_hash
    else:
        left_hash = note_hashes[src]
        right_hash = note_hashes[dst]
    return hashlib.sha256(f"{left_hash}:{right_hash}".encode("utf-8")).hexdigest()


def _valid_judgments(index: object) -> dict[tuple[str, str], tuple[bool, str]]:
    note_hashes = {
        note.note_id: note.content_hash
        for note in index.notes()  # type: ignore[attr-defined]
    }
    valid: dict[tuple[str, str], tuple[bool, str]] = {}
    for (src, dst), (sig, related, fact) in index.edge_judgments().items():  # type: ignore[attr-defined]
        if src not in note_hashes or dst not in note_hashes:
            continue
        if sig != pair_sig(index, src, dst, note_hashes):
            continue
        valid[_key(src, dst)] = (related, fact)
    return valid


def promoted_relation_fact(
    index: object,
    src: str,
    dst: str,
    *,
    active_note_count: int | None = None,
    eligible_entity_labels: frozenset[str] | None = None,
) -> str | None:
    relations_between = getattr(index, "relations_between_scoped", None)
    if not callable(relations_between):
        return None
    if eligible_entity_labels is None:
        active_count = active_note_count or _index_active_note_count(index)
        eligible_entity_labels = _eligible_entity_labels(index, active_count)
    relations = []
    for head, rel_type, tail, conf in relations_between(src, dst):
        if eligible_entity_labels is not None and (
            str(head) not in eligible_entity_labels or str(tail) not in eligible_entity_labels
        ):
            continue
        template = config.INGEST_RELATION_TYPES.get(str(rel_type))
        if template is None:
            continue
        if float(conf) < config.REL_FACT_MIN_CONF:
            continue
        relations.append((float(conf), str(head), str(rel_type), str(tail), template))
    if not relations:
        return None
    _conf, head, _rel_type, tail, template = sorted(
        relations,
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )[0]
    return _WHITESPACE_RE.sub(" ", template.format(head=head, tail=tail).strip())


def _apply_judgments(
    index: object,
    kept: list[_Candidate],
    margin: list[_Candidate],
    judgments: Mapping[tuple[str, str], tuple[bool, str]],
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    active_note_count = _index_active_note_count(index)
    eligible_entity_labels = _eligible_entity_labels(index, active_note_count)
    for candidate in kept:
        promoted_fact = promoted_relation_fact(
            index,
            candidate.src,
            candidate.dst,
            active_note_count=active_note_count,
            eligible_entity_labels=eligible_entity_labels,
        )
        if promoted_fact is not None:
            edges.append(candidate.edge(fact=promoted_fact, fact_provenance="gliner"))
            continue
        edge = candidate.edge()
        judgment = judgments.get((candidate.src, candidate.dst))
        if judgment is not None:
            related, fact = judgment
            if not related and edge.weight < config.VETO_MAX_W:
                continue
            if related:
                edge = candidate.edge(
                    fact=fact,
                    fact_provenance="llm" if fact else "",
                )
        edges.append(edge)

    for candidate in margin:
        promoted_fact = promoted_relation_fact(
            index,
            candidate.src,
            candidate.dst,
            active_note_count=active_note_count,
            eligible_entity_labels=eligible_entity_labels,
        )
        if promoted_fact is not None:
            edges.append(candidate.edge(fact=promoted_fact, fact_provenance="gliner"))
            continue
        judgment = judgments.get((candidate.src, candidate.dst))
        if judgment is None:
            continue
        related, fact = judgment
        if related:
            edges.append(
                candidate.edge(
                    fact=fact,
                    fact_provenance="llm" if fact else "",
                )
            )
    return edges


def _note_vector(index: object, note_id: str) -> np.ndarray | None:
    chunk_vectors = index.embedding_vectors(note_id, "chunk")  # type: ignore[attr-defined]
    if not chunk_vectors:
        return None
    return _normalize(np.mean(np.vstack(chunk_vectors), axis=0).astype(np.float32))


def _terms(enrichment: Enrichment | None, mechanism: str) -> set[str]:
    return {term for term, _etype in _term_keys(enrichment, mechanism)}


def _term_keys(enrichment: Enrichment | None, mechanism: str) -> set[tuple[str, str]]:
    if enrichment is None:
        return set()
    if mechanism == "topic":
        return {
            (term, entity.etype)
            for entity in enrichment.entities
            if entity.etype == "concept"
            if (term := _canonicalize_term(entity.surface, lowercase=True))
        }
    return {
        (term, entity.etype)
        for entity in enrichment.entities
        if entity.etype != "concept"
        if (term := _canonicalize_term(entity.surface, lowercase=False))
    }


def _term_is_specific(
    term: str,
    etype: str,
    resolved_terms: Mapping[str, str],
    specificity: _EntitySpecificity | None,
) -> bool:
    if specificity is None:
        return True
    return (resolved_terms.get(term, term), etype) in specificity.eligible_terms


def _entity_specificity(
    index: object,
    notes: list[NoteRecord],
) -> _EntitySpecificity | None:
    entity_rows = getattr(index, "entity_rows", None)
    if not callable(entity_rows):
        return None
    active_note_count = len({note.note_id for note in notes})
    labels: dict[int, str] = {}
    eligible_ids: set[int] = set()
    eligible_terms: set[tuple[str, str]] = set()
    for entity_id, canonical, etype, df in entity_rows():
        entity_id = int(entity_id)
        canonical = str(canonical)
        etype = str(etype)
        labels[entity_id] = canonical
        if not _entity_is_specific(etype, int(df), active_note_count):
            continue
        eligible_ids.add(entity_id)
        term = _entity_term(canonical, etype)
        if term:
            eligible_terms.add((term, etype))
    return _EntitySpecificity(
        eligible_ids=frozenset(eligible_ids),
        eligible_terms=frozenset(eligible_terms),
        labels=labels,
    )


def _entity_is_specific(etype: str, df: int, active_note_count: int) -> bool:
    if etype in config.SHARED_EDGE_ETYPE_STOPLIST:
        return False
    cap = math.ceil(config.ENTITY_SHARED_DF_FRACTION_CAP * max(active_note_count, 0))
    return int(df) <= cap


def _index_active_note_count(index: object) -> int:
    notes = getattr(index, "notes", None)
    if not callable(notes):
        return 0
    return len({note.note_id for note in notes()})


def _eligible_entity_labels(
    index: object,
    active_note_count: int,
) -> frozenset[str] | None:
    entity_rows = getattr(index, "entity_rows", None)
    if not callable(entity_rows):
        return None
    return frozenset(
        str(canonical)
        for _entity_id, canonical, etype, df in entity_rows()
        if _entity_is_specific(str(etype), int(df), active_note_count)
    )


def _term_resolution(
    index: object,
    notes: list[NoteRecord],
) -> tuple[dict[str, Enrichment | None], dict[str, str]]:
    enrichments: dict[str, Enrichment | None] = {}
    term_counts: Counter[str] = Counter()
    for note in notes:
        enrichment = index.enrichment_for(note.note_id)  # type: ignore[attr-defined]
        enrichments[note.note_id] = enrichment
        for mechanism in ("topic", "entity"):
            term_counts.update(_terms(enrichment, mechanism))
    return enrichments, resolve_terms(index, term_counts)


def resolve_terms(index: object, terms: Mapping[str, int] | Iterable[str]) -> dict[str, str]:
    """Resolve normalized term strings to corpus-level canonical forms."""
    term_counts = (
        Counter({term: int(count) for term, count in terms.items() if count > 0})
        if isinstance(terms, Mapping)
        else Counter(terms)
    )
    if not term_counts:
        return {}

    ordered_terms = sorted(term_counts)
    vectors = _ensure_term_vectors(index, ordered_terms)

    vector_terms: list[str] = []
    normalized_vectors: list[np.ndarray] = []
    for term in ordered_terms:
        normalized = _normalize(vectors[term])
        if normalized is None:
            continue
        vector_terms.append(term)
        normalized_vectors.append(normalized)

    parent = {term: term for term in ordered_terms}

    def find(term: str) -> str:
        root = term
        while parent[root] != root:
            root = parent[root]
        while parent[term] != term:
            term, parent[term] = parent[term], root
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    if len(normalized_vectors) >= 2:
        matrix = np.vstack(normalized_vectors).astype(np.float32)
        similarities = matrix @ matrix.T
        left_indexes, right_indexes = np.where(
            np.triu(similarities >= config.TERM_SYNONYM_MIN_SIM, k=1)
        )
        for left_index, right_index in zip(left_indexes, right_indexes, strict=True):
            left = vector_terms[int(left_index)]
            right = vector_terms[int(right_index)]
            if _digit_sequence(left) != _digit_sequence(right):
                continue
            if not _lexically_compatible(left, right):
                continue
            union(left, right)

    groups: dict[str, list[str]] = defaultdict(list)
    for term in ordered_terms:
        groups[find(term)].append(term)

    resolved: dict[str, str] = {}
    for members in groups.values():
        canonical = min(members, key=lambda term: (-term_counts[term], len(term), term))
        for member in members:
            resolved[member] = canonical
    return resolved


def _ensure_term_vectors(index: object, terms: Iterable[str]) -> dict[str, np.ndarray]:
    ordered_terms = sorted({term for term in terms if term})
    if not ordered_terms:
        return {}
    vectors = index.term_vectors(ordered_terms)  # type: ignore[attr-defined]
    missing = [term for term in ordered_terms if term not in vectors]
    if missing:
        embedded = np.asarray(embed_texts(missing), dtype=np.float32)
        if embedded.shape != (len(missing), config.EMBED_DIM):
            raise ValueError(
                f"term embedding model returned {embedded.shape}, "
                f"expected {(len(missing), config.EMBED_DIM)}"
            )
        new_vectors: dict[str, np.ndarray] = {}
        for term, vector in zip(missing, embedded, strict=True):
            normalized = _normalize(vector)
            new_vectors[term] = (
                normalized.astype(np.float32)
                if normalized is not None
                else np.asarray(vector, dtype=np.float32)
            )
        index.set_term_vectors(new_vectors)  # type: ignore[attr-defined]
        vectors.update(new_vectors)
    return vectors


def _canonicalize_term(term: str, *, lowercase: bool) -> str:
    normalized = _SEPARATOR_RE.sub(" ", term.strip())
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.lower() if lowercase else normalized


def _digit_sequence(term: str) -> tuple[str, ...]:
    return tuple(_DIGIT_RE.findall(term))


def _lexically_compatible(left: str, right: str) -> bool:
    """True when two terms are plausibly string variants of each other.

    Embedding similarity alone merges semantic neighbors that are distinct
    referents ('Calumet Street'↔'Tremont Street' 0.956, 'Cologne'↔
    'Düsseldorf' 0.957 — measured 2026-07-08); genuine variants are always
    lexically close too (containment after stripping non-alphanumerics, or
    normalized Levenshtein ≥ TERM_LEXICAL_MIN_RATIO)."""
    left_compact = _compact(left)
    right_compact = _compact(right)
    if not left_compact or not right_compact:
        return False
    if left_compact in right_compact or right_compact in left_compact:
        return True
    return (
        Levenshtein.normalized_similarity(left_compact, right_compact)
        >= config.TERM_LEXICAL_MIN_RATIO
    )


def _compact(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", term.lower())


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


def _cap_margin_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    degree: dict[str, int] = defaultdict(int)
    selected: list[_Candidate] = []
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate.edge().weight, candidate.src, candidate.dst),
    )
    for candidate in ordered:
        if degree[candidate.src] >= config.MARGIN_MAX_PER_NOTE:
            continue
        if degree[candidate.dst] >= config.MARGIN_MAX_PER_NOTE:
            continue
        selected.append(candidate)
        degree[candidate.src] += 1
        degree[candidate.dst] += 1
    return selected
