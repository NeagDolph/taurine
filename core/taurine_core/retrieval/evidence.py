"""Confidence-preserving fusion and shared evidence-pack assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from taurine_core import config
from taurine_core.indexer.base_graph import (
    FactGraphInput,
    grouped_fact_edges,
    materialize_base_graph,
)
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
from taurine_core.retrieval.snapshot import IndexSnapshot
from taurine_core.retrieval.triples import TripleRecognitionResult
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

    def fact_contexts(self, fact_ids: tuple[str, ...] | None = None) -> tuple[Any, ...]: ...

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

        query_vector = np.asarray(self.query_embedder(query.text), dtype=np.float32)
        latency["query_embedding"] = _elapsed_ms(started)

        head_started = monotonic()
        dense = DensePassageHead(self.snapshot, self.calibrator).search(
            query_vector, k=len(self.snapshot.passage_ids)
        )
        candidate_cap = (
            config.LIVE_RERANK_CANDIDATES
            if query.mode == "live"
            else config.CHAT_RERANK_CANDIDATES
        )
        bm25 = BM25PassageHead(self.store, self.calibrator).search(
            query.text, k=candidate_cap
        )
        exact = ExactAliasHead(self.snapshot, self.calibrator).search(query.text)
        entity = (
            QueryEntityHead(self.entity_resolver, self.calibrator).search(
                query.text, entity_spans
            )
            if self.entity_resolver is not None and entity_spans
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
        triple = (
            self.triple_recognizer.recognize(
                query,
                query_vector,
                resolved_entity_ids,
                required_entity_ids,
                deadline_at,
            )
            if self.triple_recognizer is not None
            else TripleRecognitionResult((), (), {}, False, "triple_recognizer_disabled")
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
        trace_rejected_hits = _bounded_rejected_hits(
            rejected_hits, dense_limit=candidate_cap
        )
        latency["heads"] = _elapsed_ms(head_started)
        pre_decision = self.abstention.pre_ppr(accepted_hits, rejected_hits)
        if not pre_decision.accepted:
            return _abstained_pack(
                query,
                pre_decision,
                accepted_hits,
                trace_rejected_hits,
                self.snapshot,
                latency,
                triple,
                debug={"dense_reset_considered": len(self.snapshot.passage_ids)},
            )

        fused = weighted_score_aware_rrf(
            accepted_hits,
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
        entity_passage_ids = set(
            entity_passages(
                tuple(sorted(resolved_entity_ids)),
                build_id=self.snapshot.build_id,
            )
            if callable(entity_passages)
            else ()
        )
        for passage_id in entity_passage_ids:
            passage_reset[passage_id] = max(
                passage_reset.get(passage_id, 0.0),
                max(query_entity_reset.values(), default=0.0)
                * config.QUERY_ENTITY_RESET_W,
            )

        passage_rows = self.store.passage_contexts(self.snapshot.passage_ids)
        fact_rows = self.store.fact_contexts()
        retrieval_graph, traversable_fact_ids = _temporal_retrieval_graph(
            self.snapshot,
            fact_rows,
            query,
        )
        ppr_started = monotonic()
        ppr = TypedPPR(retrieval_graph).run(
            passage_reset=passage_reset,
            recognized_entity_reset=triple.entity_seed_weights,
            query_entity_reset=query_entity_reset,
            session_reset=session_reset,
        )
        latency["ppr"] = _elapsed_ms(ppr_started)

        passage_by_id = {row.passage_id: row for row in passage_rows}
        fact_status = {
            str(row.fact_id): (
                "active" if str(row.fact_id) in traversable_fact_ids else str(row.status)
            )
            for row in fact_rows
        }
        note_by_passage = {
            row.passage_id: row.note_id for row in passage_rows
        }
        direct_passages = {
            hit.target_id
            for hit in accepted_hits
            if hit.target_kind == "passage"
        }
        direct_passages.update(entity_passage_ids)
        support_scores = {
            passage_id: triple_row.score
            for triple_row in triple.triples
            for passage_id in triple_row.source_passage_ids
        }
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
        candidate_ids = set(direct_passages) | set(ppr_passages) | set(support_scores)

        path_started = monotonic()
        path_finder = EvidencePathFinder(
            retrieval_graph,
            fact_status=fact_status,
            note_by_passage=note_by_passage,
        )
        recognized_endpoint_ids = {
            entity_id
            for row in triple.triples
            for entity_id in (row.subject_entity_id, row.object_entity_id)
        }
        seed_ids = tuple(
            sorted(direct_passages | resolved_entity_ids | recognized_endpoint_ids)
        )
        paths_by_target: dict[str, tuple[PathCandidate, ...]] = {}
        fact_by_id = {str(row.fact_id): row for row in fact_rows}
        for passage_id in ppr_passages:
            if passage_id in direct_passages:
                continue
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
                        support_scores[source_passage_id] = max(
                            support_scores.get(source_passage_id, 0.0), path.weight
                        )
        latency["paths"] = _elapsed_ms(path_started)

        recall_scores = _candidate_scores(
            candidate_ids,
            fused,
            ppr,
            support_scores,
            paths_by_target,
        )
        rerank_candidates = tuple(
            PassageRerankCandidate(
                passage_id=passage_id,
                note_id=passage_by_id[passage_id].note_id,
                title=passage_by_id[passage_id].title,
                source=passage_by_id[passage_id].source,
                text=passage_by_id[passage_id].text,
                recall_score=recall_scores[passage_id],
                hint=_candidate_hint(passage_id, support_scores, paths_by_target),
            )
            for passage_id in sorted(candidate_ids)
            if passage_id in passage_by_id
        )
        rerank_started = monotonic()
        reranked = self.reranker.rerank(
            query.text, rerank_candidates, deadline_at=deadline_at
        )
        latency["rerank"] = _elapsed_ms(rerank_started)
        fallback = ""
        if reranked.degraded:
            fallback = "calibrated_direct_only"
            ranked = _direct_fallback(
                direct_passages, accepted_hits, passage_by_id
            )
        else:
            ranked = reranked.accepted

        selection_started = monotonic()
        configured_limit = (
            config.LIVE_EVIDENCE_PASSAGES
            if query.mode == "live"
            else config.CHAT_EVIDENCE_PASSAGES
        )
        selected, exclusions = _mmr_select(
            ranked,
            self.snapshot,
            mmr_lambda=self.calibrator.mmr_lambda,
            source_diversity_penalty=self.calibrator.source_diversity_penalty,
            limit=(
                configured_limit
                if passage_limit is None
                else min(configured_limit, max(0, int(passage_limit)))
            ),
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
        )
        rerank_trace = _rerank_trace(reranked.decisions)
        if not post_decision.accepted:
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
                    "dense_reset_considered": len(self.snapshot.passage_ids),
                    "rerank_degraded": reranked.degraded,
                    "rerank_fallback": fallback,
                },
            )

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
                "dense_reset_considered": len(self.snapshot.passage_ids),
                "rerank_degraded": reranked.degraded,
                "rerank_fallback": fallback,
                "pre_abstention": pre_decision.reason,
            },
        )
        latency["pack"] = _elapsed_ms(pack_started)
        return replace(pack, trace=replace(pack.trace, latency_ms=dict(latency)))

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
                rendered = _render_evidence_path(path, passage_by_id, fact_by_id)
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
                {
                    "passage_id": passage.passage_id,
                    "note_id": passage.note_id,
                    "text": passage.text,
                    "char_start": passage.char_start,
                    "char_end": passage.char_end,
                },
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
                )
            )

        relevant_fact_ids = {
            fact_id for row in triple.triples for fact_id in row.fact_ids
        } | {fact_id for path in selected_paths for fact_id in path.fact_ids}
        for context in fact_rows:
            if str(context.fact_id) not in relevant_fact_ids:
                continue
            passage = passage_by_id.get(str(context.source_passage_id))
            if passage is None or passage.passage_id in evidence_ref_by_passage:
                continue
            evidence_ref_by_passage[passage.passage_id] = self.registry.register(
                "passage",
                passage.passage_id,
                {
                    "passage_id": passage.passage_id,
                    "note_id": passage.note_id,
                    "text": passage.text,
                    "char_start": passage.char_start,
                    "char_end": passage.char_end,
                },
            )

        evidence_facts = _assemble_facts(
            self.registry,
            triple,
            fact_rows,
            evidence_ref_by_passage,
            selected_paths,
            build_id=self.snapshot.build_id,
            supplied_passage_ids=selected_ids,
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


def _triple_head_hits(result: TripleRecognitionResult) -> HeadResult:
    accepted: list[HeadHit] = []
    rejected: list[HeadHit] = []
    for rank, row in enumerate(result.decisions, 1):
        hit = HeadHit(
            target_kind="fact",
            target_id="|".join(row.fact_ids),
            raw_score=float(row.embedding_score),
            calibrated_score=float(row.embedding_score),
            rank=rank,
            head="triple",
            reason=row.reason,
        )
        (accepted if row.accepted else rejected).append(hit)
    return HeadResult(tuple(accepted), tuple(rejected))


def _candidate_scores(
    candidate_ids: set[str],
    fused: Mapping[tuple[TargetKind, str], FusedHit],
    ppr: PPRResult,
    support_scores: Mapping[str, float],
    paths_by_target: Mapping[str, tuple[PathCandidate, ...]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for passage_id in candidate_ids:
        values = [
            fused.get(("passage", passage_id), FusedHit("passage", passage_id, 0.0, (), ())).score,
            float(ppr.ranks.get(passage_id, 0.0)),
            float(support_scores.get(passage_id, 0.0)),
        ]
        values.extend(path.weight for path in paths_by_target.get(passage_id, ()))
        scores[passage_id] = max(values, default=0.0)
    return scores


def _candidate_hint(
    passage_id: str,
    support_scores: Mapping[str, float],
    paths_by_target: Mapping[str, tuple[PathCandidate, ...]],
) -> str:
    hints: list[str] = []
    if passage_id in support_scores:
        hints.append("supports a recognized fact")
    if passage_id in paths_by_target:
        hints.append("reached by an explanatory graph path")
    return "; ".join(hints)


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
) -> tuple[
    tuple[PassageRerankDecision, ...],
    tuple[Mapping[str, object], ...],
]:
    by_id = {row.candidate.passage_id: row for row in candidates}
    vector_by_id = {
        passage_id: snapshot.passage_matrix[index]
        for index, passage_id in enumerate(snapshot.passage_ids)
    }
    remaining = set(by_id)
    selected: list[PassageRerankDecision] = []
    per_note: dict[str, int] = {}
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


def _assemble_facts(
    registry: EvidenceRegistry,
    triple: TripleRecognitionResult,
    fact_rows: Sequence[Any],
    evidence_ref_by_passage: Mapping[str, str],
    paths: Sequence[PathCandidate],
    *,
    build_id: str,
    supplied_passage_ids: set[str],
) -> tuple[EvidenceFact, ...]:
    path_fact_ids = {fact_id for path in paths for fact_id in path.fact_ids}
    recognized_by_fact = {
        fact_id: row for row in triple.triples for fact_id in row.fact_ids
    }
    contexts = {
        str(row.fact_id): row
        for row in fact_rows
        if str(row.fact_id) in path_fact_ids or str(row.fact_id) in recognized_by_fact
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
    for variant, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        logical_key, status, valid_from, valid_to = variant
        ordered = tuple(sorted(rows, key=lambda row: str(row.fact_id)))
        source_ids = tuple(
            evidence_ref_by_passage[row.source_passage_id]
            for row in ordered
            if row.source_passage_id in evidence_ref_by_passage
        )
        if not source_ids or not any(
            str(row.source_passage_id) in supplied_passage_ids for row in ordered
        ):
            continue
        recognized = next(
            (recognized_by_fact[str(row.fact_id)] for row in ordered if str(row.fact_id) in recognized_by_fact),
            None,
        )
        first = ordered[0]
        source_identity = "\x1f".join(
            (build_id, *logical_key, status, repr(valid_from), repr(valid_to))
        )
        source_evidence_ids = tuple(dict.fromkeys(source_ids))
        core_payload = {
            "logical_key": list(logical_key),
            "subject_entity_id": str(first.subject_entity_id),
            "object_entity_id": str(first.object_entity_id),
            "subject": str(first.subject),
            "predicate": str(first.predicate),
            "object": str(first.object),
            "direction": "forward",
            "status": status,
            "observed_at": max(float(row.observed_at) for row in ordered),
            "valid_from": valid_from,
            "valid_to": valid_to,
        }
        ref = registry.ref_for("fact", source_identity)
        if ref is None:
            ref = registry.register("fact", source_identity, core_payload)
        registry.merge_payload(
            ref,
            {
                "fact_ids": [str(row.fact_id) for row in ordered],
                "source_evidence_ids": list(source_evidence_ids),
            },
            append_list_fields=frozenset({"fact_ids", "source_evidence_ids"}),
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
) -> tuple[Any, set[str]]:
    intent = query.temporal_intent
    kind = intent.kind if intent is not None else "current"
    at = intent.at if intent is not None and intent.at is not None else query.now
    selected: list[Any] = []
    for fact in fact_rows:
        status = str(fact.status)
        if status in {"candidate", "conflicted"}:
            continue
        valid_at = (
            float(fact.observed_at) <= at
            and (fact.valid_from is None or at >= float(fact.valid_from))
            and (fact.valid_to is None or at < float(fact.valid_to))
        )
        if kind == "historical":
            if intent is not None and intent.at is not None:
                if valid_at:
                    selected.append(fact)
            elif status in {"historical", "superseded"}:
                selected.append(fact)
        elif status == "active" and valid_at:
            selected.append(fact)
    fact_inputs = tuple(
        FactGraphInput(
            fact_id=str(fact.fact_id),
            subject_entity_id=str(fact.subject_entity_id),
            predicate=str(fact.predicate),
            object_entity_id=str(fact.object_entity_id),
            source_note_id=str(fact.source_note_id),
            confidence=float(getattr(fact, "confidence", 1.0)),
            status="active",
        )
        for fact in selected
    )
    non_fact_edges = tuple(
        edge for edge in snapshot.base_graph.edges if edge.kind != "fact"
    )
    overlay_edges = grouped_fact_edges(fact_inputs)
    graph = materialize_base_graph(
        build_id=snapshot.build_id,
        node_ids=snapshot.base_graph.node_ids,
        node_kinds=snapshot.base_graph.node_kinds,
        edges=(*non_fact_edges, *overlay_edges),
    )
    return graph, {str(fact.fact_id) for fact in selected}


def _render_evidence_path(
    path: PathCandidate,
    passages: Mapping[str, PassageContext],
    facts: Mapping[str, Any],
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
    return tuple(-ord(character) for character in value)


def _elapsed_ms(started: float) -> float:
    return (monotonic() - started) * 1000.0
