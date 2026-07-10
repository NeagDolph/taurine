"""Quality rank + PPR traversal machinery (spec 2026-07-08 §2.2, §4.2–4.3).

Two consumers share the boosted-weight CSR graph built here:

- ``compute_quality_ranks`` — rebuild-time weighted PageRank with a
  trust+recency teleport (source_rank × recency per note), log-median
  transformed and clipped; persisted via ``Index.set_quality_ranks``.
  Same-source near-duplicate edges (mean chunk cosine ≥ DUP_EDGE_SIM) are
  excluded from the transition matrix (collusion-clique guard).
- ``ppr`` — query-time Personalized PageRank over the same matrix,
  damping PPR_DAMPING, warm-startable from the previous pass's vector.

Also owns ``refresh_fact_embeddings`` (content-addressed by fact_hash:
embeds new/changed edge facts, prunes stale rows, reloads the fact
matrix) — called at the end of every graph rebuild, mirroring the
term_vectors pattern.

Pure numpy/scipy; no LLM calls anywhere in this module.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy.sparse import csr_matrix, diags

from .. import config
from .store import Index


class NoteGraphMatrix:
    """Column-stochastic CSR transition matrix over active notes.

    ``weights`` are the traversal-boosted edge weights (§4.2):
    ``w' = min(1, w × FACT_EDGE_BOOST^[fact] × SHARED_EDGE_BOOST^[shared-type])``.
    With ``for_quality=True``, same-source near-duplicate edges are dropped
    before normalization (§2.2 guard) — traversal keeps them, quality doesn't.
    """

    def __init__(self, note_ids: list[str], matrix: object) -> None:
        self.note_ids = note_ids
        self.row_of = {note_id: i for i, note_id in enumerate(note_ids)}
        self.matrix = matrix  # scipy.sparse.csr_matrix, column-normalized

    @classmethod
    def build(cls, index: Index, *, for_quality: bool = False) -> "NoteGraphMatrix":
        graph = index.graph()
        note_ids = [node.note_id for node in graph.nodes]
        if not note_ids:
            return cls([], csr_matrix((0, 0), dtype=np.float32))

        row_of = {note_id: offset for offset, note_id in enumerate(note_ids)}
        source_by_id = {node.note_id: node.source for node in graph.nodes}
        vector_cache: dict[str, np.ndarray | None] = {}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for edge in graph.edges:
            if edge.src not in row_of or edge.dst not in row_of:
                continue
            if for_quality and _same_source_duplicate_edge(
                index, edge.src, edge.dst, source_by_id, vector_cache
            ):
                continue
            weight = _boosted_edge_weight(edge.weight, edge.type, edge.fact)
            if weight <= 0.0:
                continue
            src = row_of[edge.src]
            dst = row_of[edge.dst]
            rows.extend((dst, src))
            cols.extend((src, dst))
            data.extend((weight, weight))

        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(note_ids), len(note_ids)),
            dtype=np.float32,
        )
        strengths = np.asarray(matrix.sum(axis=0)).ravel()
        inv_strengths = np.zeros_like(strengths, dtype=np.float32)
        nonzero = strengths > 0.0
        inv_strengths[nonzero] = 1.0 / strengths[nonzero]
        normalized = matrix @ diags(inv_strengths, dtype=np.float32, format="csr")
        return cls(note_ids, normalized.tocsr())

    def ppr(
        self,
        seeds: dict[str, float],
        *,
        entity_seeds: dict[int, float] | None = None,
        damping: float = config.PPR_DAMPING,
        tol: float = config.PPR_TOL,
        max_iters: int = config.PPR_MAX_ITERS,
        warm_start: np.ndarray | None = None,
    ) -> tuple[dict[str, float], np.ndarray]:
        """Personalized PageRank from a seed distribution.

        Returns (note_id → mass for nonzero notes, raw vector for warm-start).
        Empty/unknown-only seeds → ({}, zero vector). Power iteration:
        ``r ← damping·P·r + (1−damping)·t`` with t = normalized seeds.
        """
        size = len(self.note_ids)
        zero = np.zeros(size, dtype=np.float64)
        if size == 0:
            return {}, zero

        teleport = np.zeros(size, dtype=np.float64)
        for note_id, weight in seeds.items():
            row = self.row_of.get(note_id)
            if row is None or weight <= 0.0:
                continue
            teleport[row] += float(weight) * config.PPR_PASSAGE_RESET_W
        del entity_seeds
        total = float(teleport.sum())
        if total <= 0.0:
            return {}, zero
        teleport /= total

        if warm_start is not None and warm_start.shape == (size,):
            rank = np.asarray(warm_start, dtype=np.float64).copy()
        else:
            rank = teleport.copy()

        for _ in range(max(0, int(max_iters))):
            updated = damping * (self.matrix @ rank) + (1.0 - damping) * teleport
            if float(np.abs(updated - rank).sum()) < tol:
                rank = np.asarray(updated, dtype=np.float64)
                break
            rank = np.asarray(updated, dtype=np.float64)

        masses = {
            note_id: float(rank[offset])
            for offset, note_id in enumerate(self.note_ids)
            if rank[offset] > 0.0
        }
        return masses, rank


def compute_quality_ranks(index: Index, *, now: float | None = None) -> dict[str, float]:
    """Weighted PageRank with teleport ∝ source_rank × recency (§2.2).

    QUALITY_DAMPING, QUALITY_TOL, QUALITY_MAX_ITERS; teleport from
    ``source_rank(source) × recency_multiplier(source, age_days)``; isolated
    notes get pure teleport mass. Returns and persists
    ``clip(log(r/median(r)), ±QUALITY_CLIP)`` via ``index.set_quality_ranks``.
    """
    graph_matrix = NoteGraphMatrix.build(index, for_quality=True)
    if not graph_matrix.note_ids:
        index.set_quality_ranks({})
        return {}

    timestamp = time.time() if now is None else float(now)
    notes = {note.note_id: note for note in index.notes()}
    seeds: dict[str, float] = {}
    for note_id in graph_matrix.note_ids:
        note = notes.get(note_id)
        if note is None:
            continue
        age_days = (timestamp - note.modified) / 86400.0
        seeds[note_id] = source_rank(note.source) * recency_multiplier(note.source, age_days)

    _masses, raw = graph_matrix.ppr(
        seeds,
        damping=config.QUALITY_DAMPING,
        tol=config.QUALITY_TOL,
        max_iters=config.QUALITY_MAX_ITERS,
    )
    median = float(np.median(raw)) if raw.size else 0.0
    ranks: dict[str, float] = {}
    for offset, note_id in enumerate(graph_matrix.note_ids):
        value = float(raw[offset])
        if value <= 0.0 or median <= 0.0:
            rank = -config.QUALITY_CLIP
        else:
            rank = float(
                np.clip(np.log(value / median), -config.QUALITY_CLIP, config.QUALITY_CLIP)
            )
        ranks[note_id] = rank
    index.set_quality_ranks(ranks)
    return ranks


def compute_value_scores(index: Index, *, now: float | None = None) -> dict[str, float]:
    """Persist per-note value scores for slim enrichment prioritization.

    value = minmax(quality_rank) × value-recency × source_rank × ingest_salience.
    The quality min-max maps active notes to [0.1, 1.0]; if all ranks are
    equal, quality contributes 1.0 for every note.
    """
    disabled_sources = (
        index.disabled_sources() if hasattr(index, "disabled_sources") else set()
    )
    notes = [note for note in index.notes() if note.source not in disabled_sources]
    if not notes:
        index.set_value_scores({})
        return {}

    timestamp = time.time() if now is None else float(now)
    salience = index.ingest_salience()
    ranks = [float(note.quality_rank) for note in notes]
    min_rank = min(ranks)
    max_rank = max(ranks)
    rank_span = max_rank - min_rank

    scores: dict[str, float] = {}
    for note in notes:
        if rank_span == 0.0:
            quality = 1.0
        else:
            quality = 0.1 + ((float(note.quality_rank) - min_rank) / rank_span) * 0.9
        age_days = (timestamp - note.modified) / 86400.0
        recency = float(0.5 ** (age_days / config.VALUE_RECENCY_HALF_LIFE_DAYS))
        scores[note.note_id] = (
            quality
            * recency
            * source_rank(note.source)
            * float(salience.get(note.note_id, 1.0))
        )

    index.set_value_scores(scores)
    return scores


def refresh_fact_embeddings(index: Index) -> None:
    """Embed new/changed edge facts; prune stale rows; ``index.load_facts()``.

    fact_hash = sha256 of the fact text; unchanged hashes are never re-embedded.
    Uses document encoding (``store.embed_texts``), like note units.
    """
    from . import store

    facts: dict[tuple[str, str], tuple[str, str]] = {}
    for edge in index.graph().edges:
        text = str(edge.fact or "").strip()
        if not text:
            continue
        src, dst = sorted((edge.src, edge.dst))
        fact_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        facts[(src, dst)] = (fact_hash, text)

    stored = index.fact_embedding_hashes()
    pending = [
        (key, fact_hash, text)
        for key, (fact_hash, text) in facts.items()
        if stored.get(key) != fact_hash
    ]
    if pending:
        vectors = np.asarray(
            store.embed_texts([text for _key, _fact_hash, text in pending]),
            dtype=np.float32,
        )
        if vectors.shape != (len(pending), config.EMBED_DIM):
            raise ValueError(
                f"fact embedding model returned {vectors.shape}, "
                f"expected {(len(pending), config.EMBED_DIM)}"
            )
        for item, vector in zip(pending, vectors, strict=True):
            (src, dst), fact_hash, text = item
            index.upsert_fact_embedding(src, dst, fact_hash, text, vector)

    index.prune_fact_embeddings(set(facts))
    index.load_facts()


def source_rank(source: str) -> float:
    return config.SOURCE_RANK.get(source, config.SOURCE_RANK_DEFAULT)


def recency_multiplier(source: str, age_days: float) -> float:
    """Exponential age decay with grace offset and floor (§4.4).

    ``RECENCY_FLOOR + (1−RECENCY_FLOOR) · 2^(−max(0, age−OFFSET)/half_life)``.
    """
    half_life = config.RECENCY_HALF_LIFE_DAYS.get(
        source, config.RECENCY_HALF_LIFE_DEFAULT_DAYS
    )
    effective = max(0.0, float(age_days) - config.RECENCY_OFFSET_DAYS)
    decay = float(2.0 ** (-effective / half_life))
    return config.RECENCY_FLOOR + (1.0 - config.RECENCY_FLOOR) * decay


def _boosted_edge_weight(weight: float, edge_type: str, fact: str) -> float:
    boosted = float(weight)
    if fact:
        boosted *= config.FACT_EDGE_BOOST
    if edge_type in ("topic", "entity", "mixed"):
        boosted *= config.SHARED_EDGE_BOOST
    return min(1.0, boosted)


def _column_normalize(matrix: csr_matrix) -> csr_matrix:
    strengths = np.asarray(matrix.sum(axis=0)).ravel()
    inv_strengths = np.zeros_like(strengths, dtype=np.float32)
    nonzero = strengths > 0.0
    inv_strengths[nonzero] = 1.0 / strengths[nonzero]
    return matrix @ diags(inv_strengths, dtype=np.float32, format="csr")


def _same_source_duplicate_edge(
    index: Index,
    left: str,
    right: str,
    source_by_id: dict[str, str],
    vector_cache: dict[str, np.ndarray | None],
) -> bool:
    if source_by_id.get(left) != source_by_id.get(right):
        return False
    left_vector = _note_mean_vector(index, left, vector_cache)
    right_vector = _note_mean_vector(index, right, vector_cache)
    if left_vector is None or right_vector is None:
        return False
    return float(left_vector @ right_vector) >= config.DUP_EDGE_SIM


def _note_mean_vector(
    index: Index,
    note_id: str,
    cache: dict[str, np.ndarray | None],
) -> np.ndarray | None:
    if note_id in cache:
        return cache[note_id]
    vectors = index.embedding_vectors(note_id, kind="chunk")
    if not vectors:
        cache[note_id] = None
        return None
    vector = np.mean(np.vstack(vectors), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        cache[note_id] = None
        return None
    cache[note_id] = vector / norm
    return cache[note_id]
