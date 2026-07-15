"""Confidence-preserving fusion and shared evidence-pack assembly."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from taurine_core import config
from taurine_core.dates import evidence_date
from taurine_core.indexer.base_graph import (
    BaseEdge,
    FactGraphInput,
    grouped_fact_edges,
    materialize_base_graph,
)
from taurine_core.indexer.facts import adjudication_multiplier
from taurine_core.indexer.knowledge_store import BM25PassageHit
from taurine_core.retrieval.calibration import AbstentionPolicy, HeadCalibrator
from taurine_core.retrieval.heads import (
    BM25PassageHead,
    DensePassageHead,
    ExactAliasHead,
    HeadResult,
    QueryEntityHead,
    RawHeadCandidate,
)
from taurine_core.retrieval.paths import EvidencePathFinder, PathCandidate
from taurine_core.retrieval.ppr import PPRResult, TypedPPR
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.rerank import (
    LocalPassageReranker,
    PassageRerankCandidate,
    PassageRerankDecision,
)
from taurine_core.retrieval.evidence_payloads import (
    FACT_APPEND_FIELDS,
    fact_group_payload,
    fact_source_identity,
    passage_payload,
)
from taurine_core.retrieval.snapshot import IndexSnapshot
from taurine_core.retrieval.triples import (
    TripleRecognitionResult,
    fact_temporal_admission,
)
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    EvidencePath,
    HeadHit,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
    TargetKind,
)


class PassageContext(Protocol):
    passage_id: str
    note_id: str
    title: str
    source: str
    text: str
    char_start: int
    char_end: int
    created: float
    modified: float
    passage_date: str | None


class NoteContext(Protocol):
    note_id: str
    title: str
    source: str
    modified: float


class EvidenceStore(Protocol):
    def search_passages_bm25(
        self, query: str, k: int
    ) -> tuple[BM25PassageHit, ...]: ...

    def passage_contexts(
        self, passage_ids: Sequence[str]
    ) -> tuple[PassageContext, ...]: ...

    def note_contexts(self, note_ids: Sequence[str]) -> tuple[NoteContext, ...]: ...

    def fact_contexts(
        self,
        fact_ids: tuple[str, ...] | None = None,
        *,
        include_candidates: bool = False,
        build_id: str | None = None,
        entity_ids: tuple[str, ...] | None = None,
        active_only: bool = False,
    ) -> tuple[Any, ...]: ...

    def passage_ids_for_entities(
        self,
        entity_ids: Sequence[str],
        *,
        build_id: str | None = None,
    ) -> tuple[str, ...]: ...


class TripleRecognizerPort(Protocol):
    def recognize(
        self,
        query: RetrievalQuery,
        query_vector: np.ndarray,
        resolved_entity_ids: set[str],
        required_entity_ids: set[str],
        deadline_at: float | None = None,
    ) -> TripleRecognitionResult: ...


QueryEmbedder = Callable[[str], np.ndarray]
EntityResolver = Callable[[str, str | None, str], Sequence[RawHeadCandidate] | object]


@dataclass(frozen=True)
class FusedHit:
    target_kind: TargetKind
    target_id: str
    score: float
    heads: tuple[str, ...]
    hits: tuple[HeadHit, ...]


def weighted_score_aware_rrf(
    accepted_hits: Sequence[HeadHit],
    *,
    head_weights: Mapping[str, float],
    rrf_k: int,
) -> dict[tuple[TargetKind, str], FusedHit]:
    """Fuse accepted hits without converting facts into endpoint bonuses.

    Duplicate ``(head, target)`` rows are credited once. The strongest calibrated
    row wins, with rank and target ID providing stable tie breaks.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    deduplicated: dict[tuple[str, TargetKind, str], HeadHit] = {}
    for hit in accepted_hits:
        if hit.rank < 1:
            raise ValueError("head hit ranks must start at one")
        if not math.isfinite(hit.calibrated_score) or hit.calibrated_score < 0.0:
            continue
        key = (hit.head, hit.target_kind, hit.target_id)
        current = deduplicated.get(key)
        if current is None or (-hit.calibrated_score, hit.rank, hit.target_id) < (
            -current.calibrated_score,
            current.rank,
            current.target_id,
        ):
            deduplicated[key] = hit

    grouped: dict[tuple[TargetKind, str], list[HeadHit]] = {}
    for hit in deduplicated.values():
        grouped.setdefault((hit.target_kind, hit.target_id), []).append(hit)

    fused: dict[tuple[TargetKind, str], FusedHit] = {}
    for key, rows in grouped.items():
        ordered = tuple(
            sorted(rows, key=lambda hit: (hit.head, hit.rank, hit.target_id))
        )
        score = sum(
            max(0.0, float(head_weights.get(hit.head, 0.0)))
            * min(1.0, float(hit.calibrated_score))
            / (rrf_k + hit.rank)
            for hit in ordered
        )
        if score <= 0.0:
            continue
        fused[key] = FusedHit(
            target_kind=key[0],
            target_id=key[1],
            score=score,
            heads=tuple(hit.head for hit in ordered),
            hits=ordered,
        )
    return dict(
        sorted(
            fused.items(),
            key=lambda item: (-item[1].score, item[0][0], item[0][1]),
        )
    )


def empty_pack(query: RetrievalQuery, decision: RetrievalDecision) -> EvidencePack:
    if decision.accepted:
        raise ValueError("an accepted decision cannot produce an empty abstention pack")
    return EvidencePack(
        decision=decision,
        query=query,
        passages=(),
        facts=(),
        paths=(),
        notes=(),
        trace=RetrievalTrace(
            original_query=query.text,
            rewritten_query=query.text,
        ),
    )


# Hard wall-clock cap on the evidence-path phase of one retrieval; path
# discovery is supplementary and must never dominate query latency.
_PATH_PHASE_BUDGET_S = 3.0


class SharedRetrievalEngine:
    """One immutable-snapshot retrieval pipeline shared by live, chat, and tools."""

    def __init__(
        self,
        snapshot: IndexSnapshot,
        store: EvidenceStore,
        calibrator: HeadCalibrator,
        abstention: AbstentionPolicy,
        *,
        query_embedder: QueryEmbedder,
        registry: EvidenceRegistry,
        reranker: LocalPassageReranker,
        entity_resolver: EntityResolver | None = None,
        triple_recognizer: TripleRecognizerPort | None = None,
        head_weights: Mapping[str, float] | None = None,
        fallback_on_abstain: bool = False,
        calibration_status: str = "verified",
        chat_retrieval_mode: str = "graph",
    ) -> None:
        self.snapshot = snapshot
        self.store = store
        self.calibrator = calibrator
        self.abstention = abstention
        self.query_embedder = query_embedder
        self.registry = registry
        self.reranker = reranker
        self.entity_resolver = entity_resolver
        self.triple_recognizer = triple_recognizer
        self.head_weights = dict(head_weights or calibrator.head_weights)
        # §0.1 chat/tool fallback lane; live must always keep this False.
        self.fallback_on_abstain = bool(fallback_on_abstain)
        self.calibration_status = str(calibration_status)
        self.chat_retrieval_mode = str(chat_retrieval_mode)
        self._hub_ids: frozenset[str] | None = None
        # item 5b: grounded per-edge reasons ride semantic-edge metadata from
        # the store; surfaced as rerank hints and in evidence path rendering
        self._semantic_reasons = _semantic_reason_index(snapshot.base_graph)
        # L29: materialized temporal graphs keyed by (snapshot, intent, time,
        # activated hubs). The engine is bound to one immutable snapshot, so a
        # snapshot swap always means a fresh engine and a fresh cache.
        self._graph_cache: OrderedDict[tuple[Any, ...], tuple[Any, set[str]]] = (
            OrderedDict()
        )

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        entity_spans: Sequence[tuple[str, str | None]] = (),
        session_reset: Mapping[str, float] | None = None,
        deadline_at: float | None = None,
        passage_limit: int | None = None,
    ) -> EvidencePack:
        started = monotonic()
        latency: dict[str, float] = {}
        head_errors: dict[str, str] = {}
        configured_limit = (
            config.LIVE_EVIDENCE_PASSAGES
            if query.mode == "live"
            else config.CHAT_EVIDENCE_PASSAGES
        )
        selection_limit = (
            configured_limit
            if passage_limit is None
            else min(configured_limit, max(0, int(passage_limit)))
        )
        dense_facts_mode = (
            query.mode in {"chat", "tool"}
            and self.chat_retrieval_mode == "dense_facts"
        )

        try:
            query_vector: np.ndarray | None = np.asarray(
                self.query_embedder(query.text), dtype=np.float32
            )
        except Exception:  # noqa: BLE001 - remote embedding is fail-open at query time
            query_vector = None
            unavailable_reason = "query_embedding_unavailable"
            head_errors["dense"] = unavailable_reason
            if self.triple_recognizer is not None:
                head_errors["triple"] = unavailable_reason
            logging.getLogger("taurine.retrieval").warning(
                "query embedding unavailable; dense and triple heads unavailable"
            )
        latency["query_embedding"] = _elapsed_ms(started)

        head_started = monotonic()

        def isolated(head: str, search: Callable[[], HeadResult]) -> HeadResult:
            # §17.3: one broken head degrades retrieval instead of killing it.
            try:
                return search()
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                head_errors[head] = _sanitized_error(exc)
                return HeadResult((), (), status="unavailable", reason=head_errors[head])

        dense = (
            isolated(
                "dense",
                lambda: DensePassageHead(self.snapshot, self.calibrator).search(
                    query_vector,
                    k=(
                        int(config.DENSE_SWEEP_K)
                        if dense_facts_mode
                        else len(self.snapshot.passage_ids)
                    ),
                ),
            )
            if query_vector is not None
            else HeadResult(
                (), (), status="unavailable", reason="query_embedding_unavailable"
            )
        )
        candidate_cap = (
            config.LIVE_RERANK_CANDIDATES
            if query.mode == "live"
            else config.CHAT_RERANK_CANDIDATES
        )
        conversational_mode = query.mode in {"chat", "tool"}
        bm25 = isolated(
            "bm25",
            lambda: BM25PassageHead(self.store, self.calibrator).search(
                query.text, k=candidate_cap
            ),
        )
        exact = isolated(
            "exact",
            lambda: ExactAliasHead(self.snapshot, self.calibrator).search(query.text),
        )
        entity_head_enabled = self.entity_resolver is not None and bool(entity_spans)
        entity = (
            isolated(
                "entity",
                lambda: QueryEntityHead(self.entity_resolver, self.calibrator).search(
                    query.text, entity_spans
                ),
            )
            if entity_head_enabled
            else HeadResult((), ())
        )
        resolved_entity_ids = {
            hit.target_id
            for hit in (*exact.accepted, *entity.accepted)
            if hit.target_kind == "entity"
        }
        required_entity_ids = {
            hit.target_id for hit in entity.accepted if hit.target_kind == "entity"
        }
        if dense_facts_mode:
            latency["heads"] = _elapsed_ms(head_started)
            return self._retrieve_dense_facts(
                query,
                dense=dense,
                bm25=bm25,
                exact=exact,
                entity=entity,
                resolved_entity_ids=resolved_entity_ids,
                latency=latency,
                head_errors=head_errors,
                deadline_at=deadline_at,
                selection_limit=selection_limit,
            )
        if self.triple_recognizer is not None and query_vector is not None:
            try:
                triple = self.triple_recognizer.recognize(
                    query,
                    query_vector,
                    resolved_entity_ids,
                    required_entity_ids,
                    deadline_at,
                )
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                head_errors["triple"] = _sanitized_error(exc)
                triple = TripleRecognitionResult(
                    (), (), {}, True, "triple_recognizer_unavailable"
                )
        elif self.triple_recognizer is not None:
            triple = TripleRecognitionResult(
                (), (), {}, True, "query_embedding_unavailable"
            )
        else:
            triple = TripleRecognitionResult(
                (), (), {}, False, "triple_recognizer_disabled"
            )
        triple_hits = _triple_head_hits(triple)
        accepted_hits = (
            *dense.accepted,
            *bm25.accepted,
            *exact.accepted,
            *entity.accepted,
            *triple_hits.accepted,
        )
        rejected_hits = (
            *dense.rejected,
            *bm25.rejected,
            *exact.rejected,
            *entity.rejected,
            *triple_hits.rejected,
        )
        ungated_dense_hits: tuple[HeadHit, ...] = ()
        if conversational_mode and bool(config.DENSE_UNGATED_CANDIDACY):
            # Candidacy is intentionally separate from confidence: round 3
            # measured true conversational matches below the dense curve's 8B
            # null ceiling. Preserve the calibrated rejection for abstention
            # accounting while giving Cohere the raw cosine for admission.
            dense_top_k = sorted(dense.all_hits, key=lambda hit: hit.rank)[
                : max(0, int(candidate_cap))
            ]
            dense_rejected = set(dense.rejected)
            ungated_dense_hits = tuple(
                replace(
                    hit,
                    calibrated_score=float(hit.raw_score),
                    reason="ungated_dense_candidacy",
                )
                for hit in dense_top_k
                if hit in dense_rejected
                if hit.target_kind == "passage" and math.isfinite(hit.raw_score)
            )
        candidacy_hits = (*accepted_hits, *ungated_dense_hits)
        trace_rejected_hits = _bounded_rejected_hits(
            rejected_hits, dense_limit=candidate_cap
        )
        latency["heads"] = _elapsed_ms(head_started)
        ran_heads = {"dense", "bm25", "exact"}
        if entity_head_enabled:
            ran_heads.add("entity")
        if self.triple_recognizer is not None:
            ran_heads.add("triple")
        unavailable_heads = tuple(sorted(head_errors))
        base_debug: dict[str, object] = {
            "dense_reset_considered": len(self.snapshot.passage_ids),
            "calibration_status": self.calibration_status,
        }
        if conversational_mode:
            base_debug["dense_ungated_candidates"] = len(ungated_dense_hits)
        if head_errors:
            base_debug["unavailable_heads"] = dict(sorted(head_errors.items()))
        if set(head_errors) == ran_heads:
            return _abstained_pack(
                query,
                RetrievalDecision(
                    accepted=False,
                    confidence=0.0,
                    reason="all_heads_unavailable",
                ),
                accepted_hits,
                trace_rejected_hits,
                self.snapshot,
                latency,
                triple,
                debug=base_debug,
            )
        pre_decision = self.abstention.pre_ppr(
            accepted_hits, rejected_hits, unavailable_heads=unavailable_heads
        )
        if not pre_decision.accepted:
            fallback_pack = self._fallback_pack(
                query,
                pre_decision,
                dense,
                bm25,
                accepted_hits,
                trace_rejected_hits,
                latency,
                triple,
                debug=base_debug,
                limit=min(selection_limit, int(config.RETRIEVAL_FALLBACK_PASSAGES)),
            )
            if fallback_pack is not None:
                return fallback_pack
            return _abstained_pack(
                query,
                pre_decision,
                accepted_hits,
                trace_rejected_hits,
                self.snapshot,
                latency,
                triple,
                debug=base_debug,
            )

        fused = weighted_score_aware_rrf(
            candidacy_hits,
            head_weights=self.head_weights,
            rrf_k=self.calibrator.rrf_k,
        )
        passage_reset = {
            hit.target_id: hit.calibrated_score
            for hit in dense.accepted
            if hit.target_kind == "passage"
        }
        query_entity_reset = {
            entity_id: max(
                hit.calibrated_score
                for hit in (*exact.accepted, *entity.accepted)
                if hit.target_id == entity_id
            )
            for entity_id in resolved_entity_ids
        }
        entity_passages = getattr(self.store, "passage_ids_for_entities", None)
        entity_passage_ids: set[str] = set()
        # neighborhood assembly: remember WHICH resolved entity anchors each
        # passage so the reranker sees "mentions <entity> (query entity)"
        # instead of an unexplained expansion candidate
        entity_anchor: dict[str, list[str]] = {}
        if callable(entity_passages) and resolved_entity_ids:
            contexts_fn = getattr(self.store, "entity_contexts", None)
            canonical_by_id = {
                row.entity_id: row.canonical
                for row in (
                    contexts_fn(
                        tuple(sorted(resolved_entity_ids)),
                        build_id=self.snapshot.build_id,
                    )
                    if callable(contexts_fn)
                    else ()
                )
            }
            for entity_id in sorted(resolved_entity_ids):
                ids = set(
                    entity_passages((entity_id,), build_id=self.snapshot.build_id)
                )
                entity_passage_ids.update(ids)
                name = canonical_by_id.get(entity_id, "")
                if name:
                    for passage_id in ids:
                        entity_anchor.setdefault(passage_id, []).append(name)
        # M7: resolved-entity mass enters the teleport only through
        # query_entity_reset; mirroring it into passage_reset double-counted
        # entity evidence and let weak entities inherit the strongest weight.

        passage_rows = self.store.passage_contexts(self.snapshot.passage_ids)
        passage_by_id = {row.passage_id: row for row in passage_rows}
        fact_rows = self.store.fact_contexts()
        hub_ids = self._hub_entity_ids()
        activated_hubs = tuple(sorted(hub_ids & resolved_entity_ids))
        hub_activation = self._hub_activation(activated_hubs, passage_by_id)
        retrieval_graph, traversable_fact_ids = self._retrieval_graph(
            query, fact_rows, hub_activation
        )
        ppr_started = monotonic()
        ppr = TypedPPR(retrieval_graph).run(
            passage_reset=passage_reset,
            recognized_entity_reset=triple.entity_seed_weights,
            query_entity_reset=query_entity_reset,
            session_reset=session_reset,
        )
        latency["ppr"] = _elapsed_ms(ppr_started)

        fact_status = {
            str(row.fact_id): (
                "active" if str(row.fact_id) in traversable_fact_ids else str(row.status)
            )
            for row in fact_rows
        }
        # neighborhood facts: a resolved query entity surfaces its strongest
        # asserted facts even when triple recognition (query↔triple cosine)
        # missed them — "which projects use Docker?" gets Docker's `uses`
        # facts as first-class evidence. Hub entities only when explicitly
        # activated (they are the user's stated target).
        neighborhood_fact_ids = _neighborhood_fact_ids(
            fact_rows,
            resolved_entity_ids=resolved_entity_ids,
            traversable_fact_ids=traversable_fact_ids,
            hub_ids=hub_ids,
            activated_hubs=frozenset(activated_hubs),
            cap=int(config.ENTITY_NEIGHBORHOOD_FACTS),
        )
        note_by_passage = {
            row.passage_id: row.note_id for row in passage_rows
        }
        calibrated_direct_passages = {
            hit.target_id
            for hit in accepted_hits
            if hit.target_kind == "passage"
        }
        candidacy_direct_passages = {
            hit.target_id
            for hit in candidacy_hits
            if hit.target_kind == "passage"
        }
        calibrated_direct_passages.update(entity_passage_ids)
        candidacy_direct_passages.update(entity_passage_ids)
        triple_support: dict[str, float] = {}
        for triple_row in triple.triples:
            for passage_id in triple_row.source_passage_ids:
                triple_support[passage_id] = max(
                    triple_support.get(passage_id, 0.0), float(triple_row.score)
                )
        ppr_passages = tuple(
            node_id
            for node_id, _score in sorted(
                (
                    (node_id, score)
                    for node_id, score in ppr.ranks.items()
                    if node_id in passage_by_id and score > 0.0
                ),
                key=lambda row: (-row[1], row[0]),
            )[:candidate_cap]
        )
        candidate_ids = (
            set(candidacy_direct_passages)
            | set(ppr_passages)
            | set(triple_support)
        )

        path_started = monotonic()
        # M15/§11.9: hub entities are penalized on paths. Hubs explicitly
        # activated for this query (§10.5 temporary enablement) are exempt —
        # they are the user's stated target, not a traversal shortcut.
        path_finder = EvidencePathFinder(
            retrieval_graph,
            fact_status=fact_status,
            hub_entity_ids=frozenset(hub_ids - set(activated_hubs)),
            note_by_passage=note_by_passage,
        )
        recognized_endpoint_ids = {
            entity_id
            for row in triple.triples
            for entity_id in (row.subject_entity_id, row.object_entity_id)
        }
        seed_ids = tuple(
            sorted(
                calibrated_direct_passages
                | resolved_entity_ids
                | recognized_endpoint_ids
            )
        )
        paths_by_target: dict[str, tuple[PathCandidate, ...]] = {}
        path_support: dict[str, float] = {}
        fact_by_id = {str(row.fact_id): row for row in fact_rows}
        for passage_id in ppr_passages:
            if passage_id in calibrated_direct_passages:
                continue
            # Paths are supplementary evidence: the whole phase gets a hard
            # wall-clock budget so one dense query cannot stall retrieval.
            if monotonic() - path_started > _PATH_PHASE_BUDGET_S:
                break
            found = path_finder.find(seed_ids=seed_ids, target_id=passage_id)
            if found:
                paths_by_target[passage_id] = found
                for path in found:
                    candidate_ids.update(
                        node_id for node_id in path.node_ids if node_id in passage_by_id
                    )
                    for fact_id in path.fact_ids:
                        context = fact_by_id.get(fact_id)
                        if context is None:
                            continue
                        source_passage_id = str(context.source_passage_id)
                        candidate_ids.add(source_passage_id)
                        path_support[source_passage_id] = max(
                            path_support.get(source_passage_id, 0.0), path.weight
                        )
        latency["paths"] = _elapsed_ms(path_started)

        # neighborhood facts' source passages join the pool so the facts stay
        # citable (M6 requires a supplied/selected support passage)
        fact_by_id_all = {str(row.fact_id): row for row in fact_rows}
        neighborhood_fact_passage_ids: set[str] = set()
        for fact_id in neighborhood_fact_ids:
            context = fact_by_id_all.get(fact_id)
            if context is not None and str(context.source_passage_id) in passage_by_id:
                source_passage_id = str(context.source_passage_id)
                candidate_ids.add(source_passage_id)
                neighborhood_fact_passage_ids.add(source_passage_id)
        reserved_candidate_ids = (
            set(entity_passage_ids)
            | set(triple_support)
            | neighborhood_fact_passage_ids
        )
        rerank_candidates = _order_rerank_candidates(
            candidate_ids,
            passage_by_id,
            candidacy_hits,
            fused,
            ppr,
            triple_support,
            path_support,
            paths_by_target,
            cap=(
                int(config.RERANK_MAX_CANDIDATES)
                if conversational_mode
                else candidate_cap
            ),
            reserved_candidate_ids=reserved_candidate_ids,
            reserved_slots=(
                int(config.CHAT_NON_DENSE_RESERVED_CANDIDATES)
                if conversational_mode
                else 0
            ),
            semantic_reasons=self._semantic_reasons,
            entity_anchor={
                passage_id: tuple(sorted(set(names)))
                for passage_id, names in entity_anchor.items()
            },
        )
        rerank_started = monotonic()
        reranked = self.reranker.rerank(
            query.text, rerank_candidates, deadline_at=deadline_at
        )
        latency["rerank"] = _elapsed_ms(rerank_started)
        base_debug["reranker_model_id"] = reranked.model_id
        base_debug["rerank_floor_source"] = reranked.floor_source
        fallback = ""
        if reranked.degraded:
            fallback = "calibrated_direct_only"
            ranked = _direct_fallback(
                calibrated_direct_passages, accepted_hits, passage_by_id
            )
        else:
            ranked = reranked.accepted

        selection_started = monotonic()
        selected, exclusions = _mmr_select(
            ranked,
            self.snapshot,
            mmr_lambda=self.calibrator.mmr_lambda,
            source_diversity_penalty=self.calibrator.source_diversity_penalty,
            limit=selection_limit,
        )
        if (
            conversational_mode
            and not reranked.degraded
            and len(selected) < int(config.CHAT_FILL_TRIGGER)
        ):
            evidence_floor = (
                self.abstention.rules.evidence_floor
                if reranked.evidence_floor_override is None
                else float(reranked.evidence_floor_override)
            )
            below_floor = tuple(
                row
                for row in reranked.decisions
                if row.reason == "below_passage_rerank_floor"
                and float(row.score) >= evidence_floor
            )
            selected, fill_exclusions = _mmr_select(
                below_floor,
                self.snapshot,
                mmr_lambda=self.calibrator.mmr_lambda,
                source_diversity_penalty=self.calibrator.source_diversity_penalty,
                limit=min(selection_limit, int(config.CHAT_FILL_MAX_PASSAGES)),
                initial=selected,
            )
            exclusions = (*exclusions, *fill_exclusions)
        selected, near_duplicate_suppressed = _suppress_near_duplicate_passages(
            selected, self.snapshot
        )
        latency["mmr"] = _elapsed_ms(selection_started)
        evidence_scores = tuple(row.score for row in selected)
        post_decision = self.abstention.post_rerank(
            accepted_hits,
            evidence_scores,
            rejected_hits,
            recognized_triple_confidence=max(
                (row.score for row in triple.triples), default=0.0
            ),
            evidence_target_ids=tuple(
                row.candidate.passage_id for row in selected
            ),
            evidence_floor_override=reranked.evidence_floor_override,
        )
        rerank_trace = _rerank_trace(reranked.decisions)
        if not post_decision.accepted:
            fallback_pack = self._fallback_pack(
                query,
                post_decision,
                dense,
                bm25,
                accepted_hits,
                trace_rejected_hits,
                latency,
                triple,
                debug=base_debug,
                limit=selection_limit,
                reranked_rows=(
                    selected if query.mode in {"chat", "tool"} else None
                ),
            )
            if fallback_pack is not None:
                return fallback_pack
            return _abstained_pack(
                query,
                post_decision,
                accepted_hits,
                trace_rejected_hits,
                self.snapshot,
                latency,
                triple,
                ppr=ppr,
                rerank=rerank_trace,
                mmr_exclusions=exclusions,
                debug={
                    **base_debug,
                    "rerank_degraded": reranked.degraded,
                    "rerank_fallback": fallback,
                },
            )

        # §8.3: conflicted facts never traverse, but they may be PRESENTED
        # (flagged with their conflicted status) when the query's temporal
        # intent is historical or the query independently seeds both
        # endpoints of the conflicted pair.
        intent_kind = (
            query.temporal_intent.kind if query.temporal_intent is not None else "current"
        )
        presentable_conflicted = {
            str(row.fact_id)
            for row in fact_rows
            if str(row.status) == "conflicted"
            and (
                intent_kind == "historical"
                or (
                    str(row.subject_entity_id) != str(row.object_entity_id)
                    and {str(row.subject_entity_id), str(row.object_entity_id)}
                    <= resolved_entity_ids
                )
            )
        }

        pack_started = monotonic()
        pack = self._assemble_pack(
            query,
            post_decision,
            selected,
            passage_by_id,
            triple,
            fact_rows,
            paths_by_target,
            accepted_hits,
            trace_rejected_hits,
            ppr,
            rerank_trace,
            exclusions,
            latency,
            debug={
                **base_debug,
                "rerank_degraded": reranked.degraded,
                "rerank_fallback": fallback,
                "pre_abstention": pre_decision.reason,
                "neighborhood_fact_ids": sorted(neighborhood_fact_ids),
                "entity_anchor_passages": len(entity_anchor),
                **(
                    {
                        "near_duplicate_passages_suppressed": (
                            near_duplicate_suppressed
                        )
                    }
                    if near_duplicate_suppressed
                    else {}
                ),
            },
            extra_fact_ids=set(presentable_conflicted) | neighborhood_fact_ids,
        )
        latency["pack"] = _elapsed_ms(pack_started)
        return replace(pack, trace=replace(pack.trace, latency_ms=dict(latency)))

    def _retrieve_dense_facts(
        self,
        query: RetrievalQuery,
        *,
        dense: HeadResult,
        bm25: HeadResult,
        exact: HeadResult,
        entity: HeadResult,
        resolved_entity_ids: set[str],
        latency: dict[str, float],
        head_errors: Mapping[str, str],
        deadline_at: float | None,
        selection_limit: int,
    ) -> EvidencePack:
        """Dense/BM25 chat sweep with a bounded active-fact SQL overlay."""

        sweep_k = max(0, int(config.DENSE_SWEEP_K))
        accepted_hits = (
            *dense.accepted,
            *bm25.accepted,
            *exact.accepted,
            *entity.accepted,
        )
        rejected_hits = _bounded_rejected_hits(
            (*dense.rejected, *bm25.rejected, *exact.rejected, *entity.rejected),
            dense_limit=sweep_k,
        )
        ranked_ids, recall_scores, candidate_heads = _dense_sweep_union(
            dense,
            bm25,
            cap=sweep_k,
        )
        passage_by_id = {
            str(row.passage_id): row
            for row in self.store.passage_contexts(ranked_ids)
        }
        candidates = tuple(
            PassageRerankCandidate(
                passage_id=passage_id,
                note_id=str(passage_by_id[passage_id].note_id),
                title=str(passage_by_id[passage_id].title),
                source=str(passage_by_id[passage_id].source),
                text=str(passage_by_id[passage_id].text),
                recall_score=recall_scores[passage_id],
                hint=" + ".join(candidate_heads[passage_id]),
            )
            for passage_id in ranked_ids
            if passage_id in passage_by_id
        )
        debug: dict[str, object] = {
            "retrieval_mode": "dense_facts",
            "dense_sweep_candidates": len(candidates),
            "dense_sweep_k": sweep_k,
            "calibration_status": self.calibration_status,
        }
        if head_errors:
            debug["unavailable_heads"] = dict(sorted(head_errors.items()))
        empty_triple = TripleRecognitionResult(
            (), (), {}, False, "dense_facts_no_triple"
        )
        if not candidates:
            return _abstained_pack(
                query,
                RetrievalDecision(False, 0.0, "empty_dense_sweep"),
                accepted_hits,
                rejected_hits,
                self.snapshot,
                latency,
                empty_triple,
                debug=debug,
            )

        rerank_started = monotonic()
        reranked = self.reranker.rerank(
            query.text,
            candidates,
            deadline_at=deadline_at,
            candidate_cap=sweep_k,
        )
        latency["rerank"] = _elapsed_ms(rerank_started)
        debug.update(
            {
                "reranker_model_id": reranked.model_id,
                "rerank_floor_source": reranked.floor_source,
                "rerank_degraded": reranked.degraded,
            }
        )
        selected = tuple(
            row
            for row in sorted(reranked.decisions, key=lambda item: item.rank)
            if row.reason != "outside_rerank_candidate_cap"
        )[: max(0, int(selection_limit))]

        pack_started = monotonic()
        selected_passage_ids = {
            row.candidate.passage_id for row in selected
        }
        note_ref_by_id: dict[str, str] = {}
        evidence_ref_by_passage: dict[str, str] = {}
        evidence_passages: list[EvidencePassage] = []
        for decision in selected:
            passage = passage_by_id[decision.candidate.passage_id]
            note_id = str(passage.note_id)
            note_ref = note_ref_by_id.get(note_id)
            if note_ref is None:
                note_ref = self.registry.register("note", note_id, {"note_id": note_id})
                self.registry.mark_read(note_ref)
                note_ref_by_id[note_id] = note_ref
            evidence_ref = self.registry.register(
                "passage", str(passage.passage_id), passage_payload(passage)
            )
            self.registry.mark_read(evidence_ref)
            evidence_ref_by_passage[str(passage.passage_id)] = evidence_ref
            evidence_passages.append(
                EvidencePassage(
                    evidence_id=evidence_ref,
                    passage_id=str(passage.passage_id),
                    note_id=note_id,
                    title=str(passage.title),
                    source=str(passage.source),
                    text=str(passage.text),
                    char_start=int(passage.char_start),
                    char_end=int(passage.char_end),
                    created=float(passage.created),
                    modified=float(passage.modified),
                    retrieval_score=float(decision.candidate.recall_score),
                    rerank_score=float(decision.score),
                    low_confidence=not decision.accepted,
                    passage_date=evidence_date(passage),
                )
            )

        fact_rows = (
            self.store.fact_contexts(
                build_id=self.snapshot.build_id,
                entity_ids=tuple(sorted(resolved_entity_ids)),
                active_only=True,
            )
            if resolved_entity_ids
            else ()
        )
        overlay_groups = _dense_fact_overlay_groups(
            fact_rows,
            resolved_entity_ids,
            cap=int(config.CHAT_FACT_OVERLAY_MAX),
        )
        overlay_rows = tuple(row for group in overlay_groups for row in group)
        source_ids = tuple(
            dict.fromkeys(str(row.source_passage_id) for row in overlay_rows)
        )
        source_by_id = {
            str(row.passage_id): row
            for row in self.store.passage_contexts(source_ids)
        }
        for passage_id in source_ids:
            passage = source_by_id.get(passage_id)
            if passage is None or passage_id in evidence_ref_by_passage:
                continue
            evidence_ref = self.registry.register(
                "passage", passage_id, passage_payload(passage)
            )
            self.registry.mark_read(evidence_ref)
            evidence_ref_by_passage[passage_id] = evidence_ref

        variant_order = {
            _fact_variant(group[0]): index
            for index, group in enumerate(overlay_groups)
        }
        evidence_facts = _assemble_facts(
            self.registry,
            empty_triple,
            overlay_rows,
            evidence_ref_by_passage,
            (),
            build_id=self.snapshot.build_id,
            supplied_passage_ids=selected_passage_ids,
            extra_fact_ids={str(row.fact_id) for row in overlay_rows},
            require_supplied_source=False,
            variant_order=variant_order,
        )
        debug["fact_overlay_count"] = len(evidence_facts)

        note_rows = {
            str(row.note_id): row
            for row in self.store.note_contexts(tuple(sorted(note_ref_by_id)))
        }
        evidence_notes = _assemble_notes(
            note_rows,
            note_ref_by_id,
            evidence_passages,
            support_bonus=self.calibrator.note_support_bonus,
            support_bonus_cap=self.calibrator.note_support_bonus_cap,
        )
        current_refs = {
            *note_ref_by_id.values(),
            *evidence_ref_by_passage.values(),
            *(row.fact_ref for row in evidence_facts),
        }
        read_ids = tuple(
            ref for ref in self.registry.refs("read") if ref in current_refs
        )
        confidence = max((float(row.score) for row in selected), default=0.0)
        selected_heads = tuple(
            head
            for head in ("dense", "bm25")
            if any(head in heads for heads in candidate_heads.values())
        )
        decision = RetrievalDecision(
            True,
            confidence,
            "accepted_dense_facts",
            accepted_heads=selected_heads,
            rejected_heads=tuple(sorted(head_errors)),
        )
        latency["pack"] = _elapsed_ms(pack_started)
        trace = RetrievalTrace(
            original_query=query.text,
            rewritten_query=query.text,
            head_hits=tuple(accepted_hits),
            rejected_hits=tuple(rejected_hits),
            entity_resolutions=_entity_trace(
                (*exact.accepted, *exact.rejected, *entity.accepted, *entity.rejected)
            ),
            rerank=_rerank_trace(reranked.decisions),
            selected_ids=read_ids,
            read_ids=read_ids,
            cited_ids=(),
            active_build_id=self.snapshot.build_id,
            calibration_hash=self.snapshot.calibration_hash,
            latency_ms=dict(latency),
            debug=debug,
        )
        return EvidencePack(
            decision=decision,
            query=query,
            passages=tuple(evidence_passages),
            facts=evidence_facts,
            paths=(),
            notes=evidence_notes,
            trace=trace,
        )

    def _assemble_pack(
        self,
        query: RetrievalQuery,
        decision: RetrievalDecision,
        selected: Sequence[PassageRerankDecision],
        passage_by_id: Mapping[str, PassageContext],
        triple: TripleRecognitionResult,
        fact_rows: Sequence[Any],
        paths_by_target: Mapping[str, tuple[PathCandidate, ...]],
        accepted_hits: Sequence[HeadHit],
        rejected_hits: Sequence[HeadHit],
        ppr: PPRResult,
        rerank_trace: tuple[Mapping[str, object], ...],
        exclusions: tuple[Mapping[str, object], ...],
        latency: Mapping[str, float],
        debug: Mapping[str, object],
        extra_fact_ids: set[str] = frozenset(),  # type: ignore[assignment]
    ) -> EvidencePack:
        selected_ids = {row.candidate.passage_id for row in selected}
        selected_paths = tuple(
            path
            for target_id in sorted(paths_by_target)
            if target_id in selected_ids
            for path in paths_by_target[target_id]
            if _path_has_selected_support(path, fact_rows, selected_ids)
        )
        evidence_paths: list[EvidencePath] = []
        path_refs_by_target: dict[str, list[str]] = {}
        fact_by_id = {str(row.fact_id): row for row in fact_rows}
        for target_id in sorted(paths_by_target):
            if target_id not in selected_ids:
                continue
            for path in paths_by_target[target_id]:
                if path not in selected_paths:
                    continue
                rendered = _render_evidence_path(
                    path, passage_by_id, fact_by_id, self._semantic_reasons
                )
                ref = self.registry.register(
                    "path",
                    path.path_key,
                    {
                        "node_ids": list(path.node_ids),
                        "edge_kinds": list(path.edge_kinds),
                        "fact_ids": list(path.fact_ids),
                        "rendered": rendered,
                    },
                )
                self.registry.mark_read(ref)
                path_refs_by_target.setdefault(target_id, []).append(ref)
                evidence_paths.append(
                    EvidencePath(
                        path_id=ref,
                        node_ids=path.node_ids,
                        edge_kinds=path.edge_kinds,
                        fact_ids=path.fact_ids,
                        weight=path.weight,
                        rendered=rendered,
                    )
                )

        evidence_passages: list[EvidencePassage] = []
        evidence_ref_by_passage: dict[str, str] = {}
        note_ref_by_id: dict[str, str] = {}
        note_ids = tuple(sorted({row.candidate.note_id for row in selected}))
        note_rows = {row.note_id: row for row in self.store.note_contexts(note_ids)}
        for row in selected:
            passage = passage_by_id[row.candidate.passage_id]
            note_ref = note_ref_by_id.get(passage.note_id)
            if note_ref is None:
                note_ref = self.registry.register("note", passage.note_id, {"note_id": passage.note_id})
                self.registry.mark_read(note_ref)
                note_ref_by_id[passage.note_id] = note_ref
            evidence_ref = self.registry.register(
                "passage",
                passage.passage_id,
                passage_payload(passage),
            )
            self.registry.mark_read(evidence_ref)
            evidence_ref_by_passage[passage.passage_id] = evidence_ref
            evidence_passages.append(
                EvidencePassage(
                    evidence_id=evidence_ref,
                    passage_id=passage.passage_id,
                    note_id=passage.note_id,
                    title=passage.title,
                    source=passage.source,
                    text=passage.text,
                    char_start=passage.char_start,
                    char_end=passage.char_end,
                    created=passage.created,
                    modified=passage.modified,
                    retrieval_score=row.candidate.recall_score,
                    rerank_score=row.score,
                    path_ids=tuple(path_refs_by_target.get(passage.passage_id, ())),
                    low_confidence=not row.accepted,
                    passage_date=evidence_date(passage),
                )
            )

        relevant_fact_ids = (
            {fact_id for row in triple.triples for fact_id in row.fact_ids}
            | {fact_id for path in selected_paths for fact_id in path.fact_ids}
            | set(extra_fact_ids)
        )
        for context in fact_rows:
            if str(context.fact_id) not in relevant_fact_ids:
                continue
            passage = passage_by_id.get(str(context.source_passage_id))
            if passage is None or passage.passage_id in evidence_ref_by_passage:
                continue
            evidence_ref_by_passage[passage.passage_id] = self.registry.register(
                "passage",
                passage.passage_id,
                passage_payload(passage),
            )

        evidence_facts = _assemble_facts(
            self.registry,
            triple,
            fact_rows,
            evidence_ref_by_passage,
            selected_paths,
            build_id=self.snapshot.build_id,
            supplied_passage_ids=selected_ids,
            extra_fact_ids=set(extra_fact_ids),
        )
        evidence_notes = _assemble_notes(
            note_rows,
            note_ref_by_id,
            evidence_passages,
            support_bonus=self.calibrator.note_support_bonus,
            support_bonus_cap=self.calibrator.note_support_bonus_cap,
        )
        current_refs = {
            *[row.note_ref for row in evidence_notes],
            *[row.evidence_id for row in evidence_passages],
            *[row.fact_ref for row in evidence_facts],
            *[row.path_id for row in evidence_paths],
        }
        current_read_ids = tuple(
            ref for ref in self.registry.refs("read") if ref in current_refs
        )
        trace = RetrievalTrace(
            original_query=query.text,
            rewritten_query=query.text,
            head_hits=tuple(accepted_hits),
            rejected_hits=tuple(rejected_hits),
            entity_resolutions=_entity_trace((*accepted_hits, *rejected_hits)),
            triple_decisions=_triple_trace(triple),
            teleport=dict(ppr.teleport),
            ppr=_ppr_trace(ppr, set(self.snapshot.passage_ids)),
            paths=_path_trace(paths_by_target),
            rerank=rerank_trace,
            mmr_exclusions=exclusions,
            selected_ids=current_read_ids,
            read_ids=current_read_ids,
            cited_ids=(),
            active_build_id=self.snapshot.build_id,
            calibration_hash=self.snapshot.calibration_hash,
            latency_ms=dict(latency),
            debug=dict(debug),
        )
        return EvidencePack(
            decision=decision,
            query=query,
            passages=tuple(evidence_passages),
            facts=evidence_facts,
            paths=tuple(evidence_paths),
            notes=evidence_notes,
            trace=trace,
        )


    def _hub_entity_ids(self) -> frozenset[str]:
        """Traversal-ineligible entities in this snapshot, computed once."""

        if self._hub_ids is None:
            entity_nodes = tuple(
                node_id
                for node_id, kind in zip(
                    self.snapshot.base_graph.node_ids,
                    self.snapshot.base_graph.node_kinds,
                    strict=True,
                )
                if kind == "entity"
            )
            contexts = getattr(self.store, "entity_contexts", None)
            rows = (
                contexts(entity_nodes, build_id=self.snapshot.build_id)
                if callable(contexts) and entity_nodes
                else ()
            )
            self._hub_ids = frozenset(
                str(row.entity_id) for row in rows if not row.traversal_eligible
            )
        return self._hub_ids

    def _hub_activation(
        self,
        activated_hubs: Sequence[str],
        passage_by_id: Mapping[str, PassageContext],
    ) -> dict[str, tuple[str, ...]]:
        """§10.5: an exact query entity may be temporarily enabled.

        For each explicitly resolved hub, synthesize a bounded set of contains
        links for this query's matrix only — strongest notes (by mention
        passage count) first, deterministic order.
        """

        if not activated_hubs:
            return {}
        lookup = getattr(self.store, "passage_ids_for_entities", None)
        if not callable(lookup):
            return {}
        activation: dict[str, tuple[str, ...]] = {}
        cap = max(0, int(config.HUB_ACTIVATION_MAX_EDGES))
        for entity_id in activated_hubs:
            passage_ids = tuple(
                passage_id
                for passage_id in lookup((entity_id,), build_id=self.snapshot.build_id)
                if passage_id in passage_by_id
            )
            if not passage_ids or cap == 0:
                continue
            per_note: dict[str, list[str]] = {}
            for passage_id in passage_ids:
                per_note.setdefault(passage_by_id[passage_id].note_id, []).append(
                    passage_id
                )
            ordered_notes = sorted(
                per_note, key=lambda note_id: (-len(per_note[note_id]), note_id)
            )
            bounded: list[str] = []
            for note_id in ordered_notes:
                for passage_id in sorted(per_note[note_id]):
                    bounded.append(passage_id)
                    if len(bounded) >= cap:
                        break
                if len(bounded) >= cap:
                    break
            activation[entity_id] = tuple(bounded)
        return activation

    def _retrieval_graph(
        self,
        query: RetrievalQuery,
        fact_rows: Sequence[Any],
        hub_activation: Mapping[str, tuple[str, ...]],
    ) -> tuple[Any, set[str]]:
        """L29: LRU-cache the materialized temporal graph per intent/activation.

        Fact rows are immutable within a snapshot generation (the accepted L4
        remove_source staleness aside), so the cache key needs only the
        temporal-selection inputs and the activated hub set.
        """

        intent = query.temporal_intent
        kind = intent.kind if intent is not None else "current"
        at = intent.at if intent is not None else None
        if kind == "any":
            time_key: float | None = None
        elif at is not None:
            time_key = at
        else:
            time_key = query.now
        key = (
            id(self.snapshot),
            self.snapshot.build_id,
            kind,
            time_key,
            tuple(sorted(hub_activation.items())),
        )
        cached = self._graph_cache.get(key)
        if cached is not None:
            self._graph_cache.move_to_end(key)
            return cached
        result = _temporal_retrieval_graph(
            self.snapshot, fact_rows, query, hub_activation=hub_activation
        )
        self._graph_cache[key] = result
        while len(self._graph_cache) > _GRAPH_CACHE_SIZE:
            self._graph_cache.popitem(last=False)
        return result

    def _fallback_floor(self, head: str) -> float:
        # §0.1: the fallback lane floor is 0.75 × the head's acceptance floor,
        # read through the calibrator so the source of floors stays swappable.
        return float(config.RETRIEVAL_FALLBACK_FLOOR_RATIO) * float(
            self.calibrator.acceptance_floor(head)
        )

    def _fallback_pack(
        self,
        query: RetrievalQuery,
        decision: RetrievalDecision,
        dense: HeadResult,
        bm25: HeadResult,
        accepted_hits: Sequence[HeadHit],
        rejected_hits: Sequence[HeadHit],
        latency: Mapping[str, float],
        triple: TripleRecognitionResult,
        *,
        debug: Mapping[str, object],
        limit: int,
        reranked_rows: Sequence[PassageRerankDecision] | None = None,
    ) -> EvidencePack | None:
        """§0.1 chat/tool fallback lane: a bounded best-effort pack after an
        abstention. Pre-PPR fallback remains capped at four passages and one
        per note; post-rerank chat/tool fallback keeps the requested budget
        and MMR selection. Both are passages only, registered and marked read
        like normal evidence, with an unaccepted ``fallback=True`` decision.
        """

        cap = max(0, int(limit))
        if not self.fallback_on_abstain or cap == 0:
            return None
        if reranked_rows is not None:
            ordered = [
                (
                    row.candidate.passage_id,
                    float(row.candidate.recall_score),
                    float(row.score),
                )
                for row in reranked_rows
            ]
        else:
            best: dict[str, float] = {}
            for head, result in (("dense", dense), ("bm25", bm25)):
                if result.status != "ok":
                    continue
                floor = self._fallback_floor(head)
                for hit in (*result.accepted, *result.rejected):
                    if hit.target_kind != "passage":
                        continue
                    score = float(hit.calibrated_score)
                    if math.isfinite(score) and score >= floor:
                        best[hit.target_id] = max(best.get(hit.target_id, 0.0), score)
            if not best:
                return None
            ordered = [
                (passage_id, score, score)
                for passage_id, score in sorted(
                    best.items(), key=lambda row: (-row[1], row[0])
                )
            ]
        contexts = {
            row.passage_id: row
            for row in self.store.passage_contexts(
                tuple(passage_id for passage_id, _recall, _rerank in ordered)
            )
        }
        chosen: list[tuple[PassageContext, float, float]] = []
        seen_notes: set[str] = set()
        for passage_id, retrieval_score, rerank_score in ordered:
            row = contexts.get(passage_id)
            if row is None or len(chosen) >= cap:
                continue
            if reranked_rows is None and row.note_id in seen_notes:
                continue
            seen_notes.add(row.note_id)
            chosen.append((row, retrieval_score, rerank_score))
        if not chosen:
            return None
        note_rows = {
            row.note_id: row
            for row in self.store.note_contexts(tuple(sorted(seen_notes)))
        }
        note_ref_by_id: dict[str, str] = {}
        passages: list[EvidencePassage] = []
        for row, retrieval_score, rerank_score in chosen:
            note_ref = note_ref_by_id.get(row.note_id)
            if note_ref is None:
                note_ref = self.registry.register(
                    "note", row.note_id, {"note_id": row.note_id}
                )
                self.registry.mark_read(note_ref)
                note_ref_by_id[row.note_id] = note_ref
            evidence_ref = self.registry.register(
                "passage", row.passage_id, passage_payload(row)
            )
            self.registry.mark_read(evidence_ref)
            passages.append(
                EvidencePassage(
                    evidence_id=evidence_ref,
                    passage_id=row.passage_id,
                    note_id=row.note_id,
                    title=row.title,
                    source=row.source,
                    text=row.text,
                    char_start=row.char_start,
                    char_end=row.char_end,
                    created=row.created,
                    modified=row.modified,
                    retrieval_score=retrieval_score,
                    rerank_score=rerank_score,
                    low_confidence=True,
                    passage_date=evidence_date(row),
                )
            )
        notes = _assemble_notes(
            note_rows,
            note_ref_by_id,
            passages,
            support_bonus=self.calibrator.note_support_bonus,
            support_bonus_cap=self.calibrator.note_support_bonus_cap,
        )
        current_refs = {
            *note_ref_by_id.values(),
            *(row.evidence_id for row in passages),
        }
        read_ids = tuple(
            ref for ref in self.registry.refs("read") if ref in current_refs
        )
        fallback_decision = RetrievalDecision(
            accepted=False,
            confidence=max(rerank_score for _row, _recall, rerank_score in chosen),
            reason="fallback_low_confidence",
            accepted_heads=decision.accepted_heads,
            rejected_heads=decision.rejected_heads,
            fallback=True,
        )
        trace = RetrievalTrace(
            original_query=query.text,
            rewritten_query=query.text,
            head_hits=tuple(accepted_hits),
            rejected_hits=tuple(rejected_hits),
            entity_resolutions=_entity_trace((*accepted_hits, *rejected_hits)),
            triple_decisions=_triple_trace(triple),
            selected_ids=read_ids,
            read_ids=read_ids,
            cited_ids=(),
            active_build_id=self.snapshot.build_id,
            calibration_hash=self.snapshot.calibration_hash,
            latency_ms=dict(latency),
            debug={**debug, "fallback_abstention_reason": decision.reason},
        )
        return EvidencePack(
            decision=fallback_decision,
            query=query,
            passages=tuple(passages),
            facts=(),
            paths=(),
            notes=notes,
            trace=trace,
        )


_GRAPH_CACHE_SIZE = 8


def _sanitized_error(exc: BaseException) -> str:
    detail = " ".join(str(exc).split())[:120]
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _triple_head_hits(result: TripleRecognitionResult) -> HeadResult:
    """Convert recognition output into head hits without malformed rows (L11).

    Only retained (seeded) triples become accepted hits, ranked 1..n among
    themselves so preliminary rejections never offset ranks. Each hit carries
    the recognition score as ``calibrated_score`` and the calibrated embedding
    probability as ``raw_score``. Non-retained scored decisions and
    preliminary rejections appear only as rejected hits after the retained
    ranks.
    """

    decision_by_key = {row.logical_key: row for row in result.decisions}
    retained_keys = {row.logical_key for row in result.triples}
    accepted: list[HeadHit] = []
    for rank, row in enumerate(result.triples, 1):
        decision = decision_by_key.get(row.logical_key)
        accepted.append(
            HeadHit(
                target_kind="fact",
                target_id="|".join(row.fact_ids),
                raw_score=float(
                    decision.embedding_score if decision is not None else row.score
                ),
                calibrated_score=float(row.score),
                rank=rank,
                head="triple",
                reason=(
                    decision.reason
                    if decision is not None
                    else "accepted_recognized_triple"
                ),
            )
        )
    remaining = tuple(
        row for row in result.decisions if row.logical_key not in retained_keys
    )
    ordered_remaining = sorted(
        remaining,
        key=lambda row: (-row.recognition_score, -row.embedding_score, row.logical_key),
    )
    rejected: list[HeadHit] = []
    for offset, row in enumerate(ordered_remaining, len(accepted) + 1):
        rejected.append(
            HeadHit(
                target_kind="fact",
                target_id="|".join(row.fact_ids),
                raw_score=float(row.embedding_score),
                calibrated_score=float(
                    row.recognition_score if row.recognition_score > 0.0 else 0.0
                ),
                rank=offset,
                head="triple",
                reason=row.reason,
            )
        )
    return HeadResult(tuple(accepted), tuple(rejected))


def _neighborhood_fact_ids(
    fact_rows: Sequence[Any],
    *,
    resolved_entity_ids: set[str],
    traversable_fact_ids: set[str] | frozenset[str],
    hub_ids: frozenset[str] | set[str],
    activated_hubs: frozenset[str],
    cap: int,
) -> set[str]:
    """Top asserted facts touching resolved query entities (confidence desc).

    Hub entities contribute only when explicitly activated by the query —
    otherwise every query naming the owner would drag in their whole fact
    neighborhood.
    """
    anchors = {
        entity_id
        for entity_id in resolved_entity_ids
        if entity_id not in hub_ids or entity_id in activated_hubs
    }
    if not anchors or cap <= 0:
        return set()
    scored = sorted(
        (
            (-float(getattr(row, "confidence", 0.5)), str(row.fact_id))
            for row in fact_rows
            if str(row.fact_id) in traversable_fact_ids
            and (
                str(getattr(row, "subject_entity_id", "")) in anchors
                or str(getattr(row, "object_entity_id", "")) in anchors
            )
        ),
    )
    return {fact_id for _, fact_id in scored[:cap]}


def _order_rerank_candidates(
    candidate_ids: set[str],
    passage_by_id: Mapping[str, PassageContext],
    accepted_hits: Sequence[HeadHit],
    fused: Mapping[tuple[TargetKind, str], FusedHit],
    ppr: PPRResult,
    triple_support: Mapping[str, float],
    path_support: Mapping[str, float],
    paths_by_target: Mapping[str, tuple[PathCandidate, ...]],
    *,
    cap: int,
    reserved_candidate_ids: set[str] | frozenset[str] = frozenset(),
    reserved_slots: int = 0,
    semantic_reasons: Mapping[tuple[str, str], str] = {},
    entity_anchor: Mapping[str, tuple[str, ...]] = {},
) -> tuple[PassageRerankCandidate, ...]:
    """H6: fill the bounded rerank pool by tier so incomparable score families
    never evict each other: (1) direct head hits by calibrated score desc,
    (2) recognized-triple support passages by recognition score desc, (3)
    PPR/path expansions by PPR mass desc (path weight as tie-break). Reserve a
    bounded tail of the pool for entity/exact/fact anchored passages."""

    known = {
        passage_id for passage_id in candidate_ids if passage_id in passage_by_id
    }
    supplementary_ids = (
        set(triple_support) | set(path_support) | set(paths_by_target)
    )
    direct_scores: dict[str, float] = {}
    for hit in accepted_hits:
        if (
            hit.target_kind == "passage"
            and hit.target_id in known
            and not (
                hit.reason == "ungated_dense_candidacy"
                and hit.target_id in supplementary_ids
            )
        ):
            score = float(hit.calibrated_score)
            if math.isfinite(score):
                direct_scores[hit.target_id] = max(
                    direct_scores.get(hit.target_id, 0.0), score
                )
    path_weights: dict[str, float] = dict(path_support)
    for target_id, paths in paths_by_target.items():
        for path in paths:
            path_weights[target_id] = max(
                path_weights.get(target_id, 0.0), float(path.weight)
            )
    ordered: list[tuple[str, float]] = []
    ordered.extend(sorted(direct_scores.items(), key=lambda row: (-row[1], row[0])))
    ordered.extend(
        sorted(
            (
                (passage_id, float(score))
                for passage_id, score in triple_support.items()
                if passage_id in known and passage_id not in direct_scores
            ),
            key=lambda row: (-row[1], row[0]),
        )
    )
    remaining = known - set(direct_scores) - set(triple_support)
    for passage_id in sorted(
        remaining,
        key=lambda row: (
            -float(ppr.ranks.get(row, 0.0)),
            -path_weights.get(row, 0.0),
            row,
        ),
    ):
        mass = float(ppr.ranks.get(passage_id, 0.0))
        ordered.append(
            (passage_id, mass if mass > 0.0 else path_weights.get(passage_id, 0.0))
        )
    seen: set[str] = set()
    unique_ordered: list[tuple[str, float]] = []
    for passage_id, tier_score in ordered:
        if passage_id in seen:
            continue
        seen.add(passage_id)
        unique_ordered.append((passage_id, tier_score))

    bounded_cap = max(0, int(cap))
    reservation = min(bounded_cap, max(0, int(reserved_slots)))
    reserved = [
        row for row in unique_ordered if row[0] in reserved_candidate_ids
    ][:reservation]
    reserved_ids = {passage_id for passage_id, _score in reserved}
    general = [
        row for row in unique_ordered if row[0] not in reserved_ids
    ][: bounded_cap - len(reserved)]
    selected_ids = {
        passage_id for passage_id, _score in (*reserved, *general)
    }

    candidates: list[PassageRerankCandidate] = []
    for passage_id, tier_score in unique_ordered:
        if passage_id not in selected_ids:
            continue
        row = passage_by_id[passage_id]
        fused_hit = fused.get(("passage", passage_id))
        candidates.append(
            PassageRerankCandidate(
                passage_id=passage_id,
                note_id=row.note_id,
                title=row.title,
                source=row.source,
                text=row.text,
                recall_score=float(tier_score),
                hint=_candidate_hint(
                    passage_id,
                    triple_support,
                    path_support,
                    paths_by_target,
                    semantic_reasons,
                    entity_anchor,
                ),
                fused_score=float(fused_hit.score) if fused_hit is not None else 0.0,
                ppr_mass=float(ppr.ranks.get(passage_id, 0.0)),
                triple_support=float(triple_support.get(passage_id, 0.0)),
                path_weight=float(path_weights.get(passage_id, 0.0)),
            )
        )
    return tuple(candidates)


def _candidate_hint(
    passage_id: str,
    triple_support: Mapping[str, float],
    path_support: Mapping[str, float],
    paths_by_target: Mapping[str, tuple[PathCandidate, ...]],
    semantic_reasons: Mapping[tuple[str, str], str] = {},
    entity_anchor: Mapping[str, tuple[str, ...]] = {},
) -> str:
    hints: list[str] = []
    if passage_id in triple_support or passage_id in path_support:
        hints.append("supports a recognized fact")
    anchors = entity_anchor.get(passage_id, ())
    if anchors:
        hints.append(f"mentions the query entity: {', '.join(anchors[:3])}")
    if passage_id in paths_by_target:
        hints.append("reached by an explanatory graph path")
        reason = _path_semantic_reason(
            paths_by_target.get(passage_id, ()), semantic_reasons
        )
        if reason:
            hints.append(f"linked because: {reason}")
    return "; ".join(hints)


def _semantic_reason_index(graph: Any) -> dict[tuple[str, str], str]:
    """Nonempty grounded reasons per semantic edge, keyed by sorted pair."""

    reasons: dict[tuple[str, str], str] = {}
    for edge in graph.edges:
        if edge.kind != "semantic" or not edge.metadata:
            continue
        reason = str(edge.metadata.get("reason", ""))
        if reason:
            left, right = sorted((edge.src_id, edge.dst_id))
            reasons[(left, right)] = reason
    return reasons


def _path_semantic_reason(
    paths: Sequence[PathCandidate],
    semantic_reasons: Mapping[tuple[str, str], str],
) -> str:
    """First grounded reason along a target's explanatory paths, if any."""

    for path in paths:
        for index, kind in enumerate(path.edge_kinds):
            if not kind.startswith("semantic"):
                continue
            pair = tuple(sorted((path.node_ids[index], path.node_ids[index + 1])))
            reason = semantic_reasons.get(pair, "")
            if reason:
                return reason
    return ""


def _direct_fallback(
    direct_passages: set[str],
    accepted_hits: Sequence[HeadHit],
    passage_by_id: Mapping[str, PassageContext],
) -> tuple[PassageRerankDecision, ...]:
    strongest: dict[str, float] = {}
    for hit in accepted_hits:
        if hit.target_kind == "passage" and hit.target_id in direct_passages:
            strongest[hit.target_id] = max(
                strongest.get(hit.target_id, 0.0), float(hit.calibrated_score)
            )
    ordered = sorted(strongest.items(), key=lambda row: (-row[1], row[0]))
    return tuple(
        PassageRerankDecision(
            candidate=PassageRerankCandidate(
                passage_id=passage_id,
                note_id=passage_by_id[passage_id].note_id,
                title=passage_by_id[passage_id].title,
                source=passage_by_id[passage_id].source,
                text=passage_by_id[passage_id].text,
                recall_score=score,
            ),
            score=score,
            accepted=True,
            rank=rank,
            reason="accepted_calibrated_direct_fallback",
        )
        for rank, (passage_id, score) in enumerate(ordered, 1)
        if passage_id in passage_by_id
    )


def _mmr_select(
    candidates: Sequence[PassageRerankDecision],
    snapshot: IndexSnapshot,
    *,
    mmr_lambda: float = 0.2,
    source_diversity_penalty: float = 0.0,
    limit: int,
    initial: Sequence[PassageRerankDecision] = (),
) -> tuple[
    tuple[PassageRerankDecision, ...],
    tuple[Mapping[str, object], ...],
]:
    by_id = {row.candidate.passage_id: row for row in candidates}
    vector_by_id = {
        passage_id: snapshot.passage_matrix[index]
        for index, passage_id in enumerate(snapshot.passage_ids)
    }
    selected = list(initial[: max(0, int(limit))])
    selected_ids = {row.candidate.passage_id for row in selected}
    remaining = set(by_id) - selected_ids
    per_note: dict[str, int] = {}
    for row in selected:
        note_id = row.candidate.note_id
        per_note[note_id] = per_note.get(note_id, 0) + 1
    exclusions: list[Mapping[str, object]] = []
    while remaining and len(selected) < max(0, int(limit)):
        scored: list[tuple[float, float, str]] = []
        capped: list[str] = []
        for passage_id in sorted(remaining):
            row = by_id[passage_id]
            if per_note.get(row.candidate.note_id, 0) >= 2:
                capped.append(passage_id)
                continue
            similarity = max(
                (
                    _cosine(vector_by_id.get(passage_id), vector_by_id.get(chosen.candidate.passage_id))
                    for chosen in selected
                ),
                default=0.0,
            )
            source_penalty = (
                float(source_diversity_penalty)
                if any(
                    chosen.candidate.source == row.candidate.source
                    for chosen in selected
                )
                else 0.0
            )
            mmr_score = (
                float(row.score)
                - float(mmr_lambda) * max(0.0, similarity)
                - source_penalty
            )
            scored.append((mmr_score, float(row.score), passage_id))
        for passage_id in capped:
            remaining.remove(passage_id)
            exclusions.append({"passage_id": passage_id, "reason": "same_note_cap"})
        if not scored:
            break
        _mmr, _relevance, chosen_id = max(
            scored, key=lambda row: (row[0], row[1], _reverse_key(row[2]))
        )
        chosen = by_id[chosen_id]
        selected.append(chosen)
        remaining.remove(chosen_id)
        per_note[chosen.candidate.note_id] = per_note.get(chosen.candidate.note_id, 0) + 1
    exclusions.extend(
        {"passage_id": passage_id, "reason": "mode_budget_or_mmr"}
        for passage_id in sorted(remaining)
    )
    return tuple(selected), tuple(exclusions)


def _suppress_near_duplicate_passages(
    selected: Sequence[PassageRerankDecision],
    snapshot: IndexSnapshot,
) -> tuple[tuple[PassageRerankDecision, ...], int]:
    """Drop lower-scored near-duplicates from the final, already-bounded set."""

    vector_by_id = {
        passage_id: snapshot.passage_matrix[index]
        for index, passage_id in enumerate(snapshot.passage_ids)
    }
    ranked = sorted(
        enumerate(selected),
        key=lambda item: (-float(item[1].score), item[1].candidate.passage_id),
    )
    kept_indices: list[int] = []
    for index, row in ranked:
        vector = vector_by_id.get(row.candidate.passage_id)
        if vector is not None and any(
            _cosine(
                vector,
                vector_by_id.get(selected[kept].candidate.passage_id),
            )
            >= float(config.EVIDENCE_NEAR_DUP_SIM)
            for kept in kept_indices
        ):
            continue
        kept_indices.append(index)
    kept = set(kept_indices)
    return (
        tuple(row for index, row in enumerate(selected) if index in kept),
        len(selected) - len(kept),
    )


def _dense_sweep_union(
    dense: HeadResult,
    bm25: HeadResult,
    *,
    cap: int,
) -> tuple[tuple[str, ...], dict[str, float], dict[str, tuple[str, ...]]]:
    """Fuse ungated dense hits with accepted BM25 hits into a bounded sweep."""

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    heads: dict[str, set[str]] = {}
    for head_name, hits in (
        ("dense", dense.all_hits),
        ("bm25", bm25.accepted),
    ):
        for hit in hits:
            if hit.target_kind != "passage":
                continue
            passage_id = str(hit.target_id)
            rank = max(1, int(hit.rank))
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (60.0 + rank)
            best_rank[passage_id] = min(best_rank.get(passage_id, rank), rank)
            heads.setdefault(passage_id, set()).add(head_name)
    ranked = tuple(
        sorted(
            scores,
            key=lambda passage_id: (
                -scores[passage_id],
                best_rank[passage_id],
                passage_id,
            ),
        )[: max(0, int(cap))]
    )
    return (
        ranked,
        {passage_id: scores[passage_id] for passage_id in ranked},
        {
            passage_id: tuple(
                head for head in ("dense", "bm25") if head in heads[passage_id]
            )
            for passage_id in ranked
        },
    )


def _fact_variant(
    row: Any,
) -> tuple[tuple[str, str, str], str, object, object]:
    return (
        tuple(row.logical_key),
        str(row.status),
        getattr(row, "valid_from", None),
        getattr(row, "valid_to", None),
    )


def _dense_fact_overlay_groups(
    fact_rows: Sequence[Any],
    resolved_entity_ids: set[str],
    *,
    cap: int,
) -> tuple[tuple[Any, ...], ...]:
    """Select active fact variants touching aliases resolved from the query."""

    grouped: dict[
        tuple[tuple[str, str, str], str, object, object], list[Any]
    ] = {}
    for row in fact_rows:
        if str(row.status) != "active":
            continue
        if not (
            str(row.subject_entity_id) in resolved_entity_ids
            or str(row.object_entity_id) in resolved_entity_ids
        ):
            continue
        grouped.setdefault(_fact_variant(row), []).append(row)

    def group_sort_key(
        item: tuple[
            tuple[tuple[str, str, str], str, object, object], list[Any]
        ],
    ) -> tuple[float, float, str]:
        variant, rows = item
        adjudication = max(
            (
                float(score)
                for row in rows
                if (score := getattr(row, "adjudication_score", None)) is not None
            ),
            default=float("-inf"),
        )
        observed = max(float(getattr(row, "observed_at", 0.0)) for row in rows)
        return (-adjudication, -observed, repr(variant))

    return tuple(
        tuple(sorted(rows, key=lambda row: str(row.fact_id)))
        for _variant, rows in sorted(grouped.items(), key=group_sort_key)[
            : max(0, int(cap))
        ]
    )


def _assemble_facts(
    registry: EvidenceRegistry,
    triple: TripleRecognitionResult,
    fact_rows: Sequence[Any],
    evidence_ref_by_passage: Mapping[str, str],
    paths: Sequence[PathCandidate],
    *,
    build_id: str,
    supplied_passage_ids: set[str],
    extra_fact_ids: set[str] = frozenset(),  # type: ignore[assignment]
    require_supplied_source: bool = True,
    variant_order: Mapping[
        tuple[tuple[str, str, str], str, object, object], int
    ] | None = None,
) -> tuple[EvidenceFact, ...]:
    path_fact_ids = {fact_id for path in paths for fact_id in path.fact_ids}
    recognized_by_fact = {
        fact_id: row for row in triple.triples for fact_id in row.fact_ids
    }
    contexts = {
        str(row.fact_id): row
        for row in fact_rows
        if str(row.fact_id) in path_fact_ids
        or str(row.fact_id) in recognized_by_fact
        or str(row.fact_id) in extra_fact_ids
    }
    grouped: dict[tuple[tuple[str, str, str], str, object, object], list[Any]] = {}
    for context in contexts.values():
        variant = (
            tuple(context.logical_key),
            str(context.status),
            getattr(context, "valid_from", None),
            getattr(context, "valid_to", None),
        )
        grouped.setdefault(variant, []).append(context)
    facts: list[EvidenceFact] = []
    def group_sort_key(
        item: tuple[
            tuple[tuple[str, str, str], str, object, object], list[Any]
        ],
    ) -> tuple[int, str]:
        variant = item[0]
        if variant_order is None:
            return (0, repr(variant))
        return (variant_order.get(variant, len(variant_order)), repr(variant))

    for variant, rows in sorted(grouped.items(), key=group_sort_key):
        logical_key, status, valid_from, valid_to = variant
        ordered = tuple(sorted(rows, key=lambda row: str(row.fact_id)))
        source_ids = tuple(
            evidence_ref_by_passage[row.source_passage_id]
            for row in ordered
            if row.source_passage_id in evidence_ref_by_passage
        )
        has_supplied_source = any(
            str(row.source_passage_id) in supplied_passage_ids for row in ordered
        )
        if not source_ids or (require_supplied_source and not has_supplied_source):
            continue
        recognized = next(
            (recognized_by_fact[str(row.fact_id)] for row in ordered if str(row.fact_id) in recognized_by_fact),
            None,
        )
        first = ordered[0]
        source_identity = fact_source_identity(
            build_id, logical_key, status, valid_from, valid_to
        )
        source_evidence_ids = tuple(dict.fromkeys(source_ids))
        core_payload = fact_group_payload(
            logical_key, status, valid_from, valid_to, ordered
        )
        ref = registry.ref_for("fact", source_identity)
        if ref is None:
            ref = registry.register("fact", source_identity, core_payload)
        registry.merge_payload(
            ref,
            {
                "fact_ids": [str(row.fact_id) for row in ordered],
                "source_evidence_ids": list(source_evidence_ids),
            },
            append_list_fields=FACT_APPEND_FIELDS,
        )
        registry.mark_read(ref)
        facts.append(
            EvidenceFact(
                fact_ref=ref,
                fact_ids=tuple(str(row.fact_id) for row in ordered),
                subject=str(first.subject),
                predicate=str(first.predicate),
                object=str(first.object),
                direction="forward",
                status=status,
                observed_at=max(float(row.observed_at) for row in ordered),
                valid_from=valid_from,  # type: ignore[arg-type]
                valid_to=valid_to,  # type: ignore[arg-type]
                confidence=float(
                    recognized.score
                    if recognized is not None
                    else max(float(getattr(row, "confidence", 0.0)) for row in ordered)
                ),
                source_evidence_ids=source_evidence_ids,
            )
        )
    return tuple(facts)


def _assemble_notes(
    note_rows: Mapping[str, NoteContext],
    note_ref_by_id: Mapping[str, str],
    passages: Sequence[EvidencePassage],
    *,
    support_bonus: float,
    support_bonus_cap: float,
) -> tuple[EvidenceNote, ...]:
    by_note: dict[str, list[EvidencePassage]] = {}
    for passage in passages:
        by_note.setdefault(passage.note_id, []).append(passage)
    notes: list[EvidenceNote] = []
    for note_id, rows in by_note.items():
        ordered = sorted(rows, key=lambda row: (-row.rerank_score, row.passage_id))
        strongest = ordered[0].rerank_score
        bonus = min(
            float(support_bonus_cap),
            float(support_bonus) * max(0, len(ordered) - 1),
        )
        note = note_rows.get(note_id)
        first = ordered[0]
        notes.append(
            EvidenceNote(
                note_ref=note_ref_by_id[note_id],
                note_id=note_id,
                title=str(note.title if note is not None else first.title),
                source=str(note.source if note is not None else first.source),
                modified=float(note.modified if note is not None else first.modified),
                score=min(1.0, strongest + bonus),
                evidence_ids=tuple(row.evidence_id for row in ordered),
            )
        )
    return tuple(sorted(notes, key=lambda row: (-row.score, row.note_id)))


def _abstained_pack(
    query: RetrievalQuery,
    decision: RetrievalDecision,
    accepted_hits: Sequence[HeadHit],
    rejected_hits: Sequence[HeadHit],
    snapshot: IndexSnapshot,
    latency: Mapping[str, float],
    triple: TripleRecognitionResult,
    *,
    ppr: PPRResult | None = None,
    rerank: tuple[Mapping[str, object], ...] = (),
    mmr_exclusions: tuple[Mapping[str, object], ...] = (),
    debug: Mapping[str, object] | None = None,
) -> EvidencePack:
    return EvidencePack(
        decision=decision,
        query=query,
        passages=(),
        facts=(),
        paths=(),
        notes=(),
        trace=RetrievalTrace(
            original_query=query.text,
            rewritten_query=query.text,
            head_hits=tuple(accepted_hits),
            rejected_hits=tuple(rejected_hits),
            entity_resolutions=_entity_trace((*accepted_hits, *rejected_hits)),
            triple_decisions=_triple_trace(triple),
            teleport={} if ppr is None else dict(ppr.teleport),
            ppr=(
                {}
                if ppr is None
                else _ppr_trace(ppr, set(snapshot.passage_ids))
            ),
            rerank=rerank,
            mmr_exclusions=mmr_exclusions,
            selected_ids=(),
            read_ids=(),
            cited_ids=(),
            active_build_id=snapshot.build_id,
            calibration_hash=snapshot.calibration_hash,
            latency_ms=dict(latency),
            debug=dict(debug or {}),
        ),
    )


def _path_has_selected_support(
    path: PathCandidate,
    fact_rows: Sequence[Any],
    selected_passage_ids: set[str],
) -> bool:
    if not path.fact_ids:
        return True
    source_by_fact = {
        str(row.fact_id): str(row.source_passage_id) for row in fact_rows
    }
    return all(
        fact_id in source_by_fact
        and source_by_fact[fact_id] in selected_passage_ids
        for fact_id in path.fact_ids
    )


def _temporal_retrieval_graph(
    snapshot: IndexSnapshot,
    fact_rows: Sequence[Any],
    query: RetrievalQuery,
    *,
    hub_activation: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[Any, set[str]]:
    """Materialize the query's fact overlay using the one shared temporal
    predicate (M2), plus synthesized contains links for explicitly
    query-activated hub entities (M13, §10.5)."""

    intent = query.temporal_intent
    kind = intent.kind if intent is not None else "current"
    at = intent.at if intent is not None else None
    selected: list[tuple[Any, float]] = []
    for fact in fact_rows:
        admitted, weight, _reason = fact_temporal_admission(
            status=str(fact.status),
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            kind=kind,
            at=at,
            now=query.now,
            for_traversal=True,
        )
        if admitted:
            selected.append((fact, weight))
    fact_inputs = tuple(
        FactGraphInput(
            fact_id=str(fact.fact_id),
            subject_entity_id=str(fact.subject_entity_id),
            predicate=str(fact.predicate),
            object_entity_id=str(fact.object_entity_id),
            source_note_id=str(fact.source_note_id),
            confidence=float(getattr(fact, "confidence", 1.0)) * weight,
            status="active",
            # the query-time overlay rebuilds fact edges from rows, so the
            # build-time adjudication downweight must be re-threaded here
            adjudication_multiplier=adjudication_multiplier(
                getattr(fact, "adjudicated", False),
                getattr(fact, "adjudication_score", None),
            ),
        )
        for fact, weight in selected
    )
    non_fact_edges = tuple(
        edge for edge in snapshot.base_graph.edges if edge.kind != "fact"
    )
    overlay_edges = grouped_fact_edges(fact_inputs)
    known_nodes = set(snapshot.base_graph.node_ids)
    hub_edges: list[BaseEdge] = []
    hub_weight = float(config.HUB_ACTIVATION_EDGE_WEIGHT)
    for entity_id in sorted(hub_activation or {}):
        if entity_id not in known_nodes:
            continue
        for passage_id in (hub_activation or {})[entity_id]:
            if passage_id not in known_nodes:
                continue
            hub_edges.append(BaseEdge(passage_id, entity_id, "contains", hub_weight))
            hub_edges.append(BaseEdge(entity_id, passage_id, "contains", hub_weight))
    graph = materialize_base_graph(
        build_id=snapshot.build_id,
        node_ids=snapshot.base_graph.node_ids,
        node_kinds=snapshot.base_graph.node_kinds,
        edges=(*non_fact_edges, *overlay_edges, *hub_edges),
    )
    return graph, {str(fact.fact_id) for fact, _weight in selected}


def _render_evidence_path(
    path: PathCandidate,
    passages: Mapping[str, PassageContext],
    facts: Mapping[str, Any],
    semantic_reasons: Mapping[tuple[str, str], str] = {},
) -> str:
    entity_labels: dict[str, str] = {}
    for fact in facts.values():
        entity_labels[str(fact.subject_entity_id)] = str(fact.subject)
        entity_labels[str(fact.object_entity_id)] = str(fact.object)

    def label(node_id: str) -> str:
        passage = passages.get(node_id)
        if passage is not None:
            return f'"{passage.title}" passage'
        return entity_labels.get(node_id, node_id)

    rendered = label(path.node_ids[0]) if path.node_ids else ""
    for index, kind in enumerate(path.edge_kinds):
        left = path.node_ids[index]
        right = path.node_ids[index + 1]
        relation = kind
        if kind.startswith("fact:"):
            matching = next(
                (
                    fact
                    for fact_id in path.fact_ids
                    if (fact := facts.get(fact_id)) is not None
                    and {str(fact.subject_entity_id), str(fact.object_entity_id)}
                    == {left, right}
                ),
                None,
            )
            relation = str(matching.predicate) if matching is not None else "related_to"
        elif kind.startswith("contains:"):
            relation = "mentions"
        elif kind.startswith("semantic:"):
            relation = "semantically_related"
            # item 5b: the grounded judge's concrete connection, when present
            reason = semantic_reasons.get(tuple(sorted((left, right))), "")
            if reason:
                relation = f"semantically_related ({reason})"
        rendered += f" --{relation}--> {label(right)}"
    return rendered


def _triple_trace(result: TripleRecognitionResult) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "logical_key": list(row.logical_key),
            "fact_ids": list(row.fact_ids),
            "embedding_score": row.embedding_score,
            "recognition_score": row.recognition_score,
            "accepted": row.accepted,
            "reason": row.reason,
            "used_fallback": row.used_fallback,
        }
        for row in result.decisions
    )


def _bounded_rejected_hits(
    hits: Sequence[HeadHit], *, dense_limit: int
) -> tuple[HeadHit, ...]:
    dense_seen = 0
    bounded: list[HeadHit] = []
    for hit in hits:
        if hit.head == "dense":
            if dense_seen >= max(0, int(dense_limit)):
                continue
            dense_seen += 1
        bounded.append(hit)
    return tuple(bounded)


def _entity_trace(hits: Sequence[HeadHit]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "entity_id": hit.target_id,
            "head": hit.head,
            "score": hit.calibrated_score,
            "reason": hit.reason,
        }
        for hit in hits
        if hit.target_kind == "entity"
    )


def _ppr_trace(
    result: PPRResult, passage_ids: set[str]
) -> Mapping[str, object]:
    return {
        "accepted": result.accepted,
        "reason": result.reason,
        "iterations": result.iterations,
        "residual": result.residual,
        "passage_mass": result.passage_mass,
        "entity_mass": result.entity_mass,
        "contributions": {
            node_id: [[source, score] for source, score in rows]
            for node_id, rows in result.contributions.items()
        },
        "top_passages": [
            [node_id, score]
            for node_id, score in sorted(
                (
                    row
                    for row in result.ranks.items()
                    if row[0] in passage_ids
                ),
                key=lambda row: (-row[1], row[0]),
            )[:12]
        ],
    }


def _path_trace(
    paths_by_target: Mapping[str, tuple[PathCandidate, ...]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "target_id": target_id,
            "path_key": path.path_key,
            "node_ids": list(path.node_ids),
            "edge_kinds": list(path.edge_kinds),
            "fact_ids": list(path.fact_ids),
            "weight": path.weight,
        }
        for target_id in sorted(paths_by_target)
        for path in paths_by_target[target_id]
    )


def _rerank_trace(
    rows: Sequence[PassageRerankDecision],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "passage_id": row.candidate.passage_id,
            "recall_score": row.candidate.recall_score,
            "score": row.score,
            "accepted": row.accepted,
            "rank": row.rank,
            "reason": row.reason,
        }
        for row in rows
    )


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 0:
        return 0.0
    return float(np.clip((left_array @ right_array) / denominator, -1.0, 1.0))


def _reverse_key(value: str) -> tuple[int, ...]:
    # Order-reversing key for max(): smallest ID wins ties. The trailing
    # sentinel (greater than any -ord) makes a prefix beat its extensions
    # (L12: "a" must win over "ab").
    return tuple(-ord(character) for character in value) + (1,)


def _elapsed_ms(started: float) -> float:
    return (monotonic() - started) * 1000.0
