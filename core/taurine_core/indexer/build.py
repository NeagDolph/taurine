"""Deterministic clean knowledge-index build and atomic activation command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.sparse.csgraph import connected_components

from taurine_core import config
from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.base_graph import BaseGraphBuilder, EntityGraphInput, PassageEntityInput, PassageGraphInput
from taurine_core.indexer.embed import embed_facts, embed_passages, embed_texts
from taurine_core.indexer.entities import EntityResolver, MentionInput
from taurine_core.indexer.extract import extract_notes
from taurine_core.indexer.facts import promote_fact, reconcile_temporal_facts
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.passages import TokenizerPort, build_passages
from taurine_core.indexer.projection import ProjectionPassage, build_projection
from taurine_core.indexer.relation_completion import (
    CompletionEntity,
    CompletionNote,
    CompletionPassage,
    parse_completion_response,
    select_completion_notes,
)
from taurine_core.indexer.rank import recency_multiplier, source_rank
from taurine_core.models_local import passage_tokenizer
from taurine_core.schemas import NoteRecord


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildDependencies:
    tokenizer: TokenizerPort
    embedder: Callable[[list[str]], np.ndarray]
    embedding_dim: int = config.EMBED_DIM
    embedding_model_id: str = config.EMBED_MODEL
    now: Callable[[], float] = time.time
    extractor: Callable[..., dict[str, Any]] | None = None
    relation_completer: Callable[[Sequence[CompletionNote]], object] | None = None


@dataclass(frozen=True)
class BuildReport:
    build_id: str
    target: Path
    source_snapshot: str
    graph_hash: str
    health: str
    stages: tuple[str, ...]
    note_count: int
    passage_count: int
    entity_count: int
    fact_count: int
    base_edge_count: int
    projection_edge_count: int
    extraction_failure_count: int
    failed_notes: tuple[tuple[str, str], ...]
    component_count: int
    dangling_node_count: int
    hub_exclusions: tuple[tuple[str, str], ...]
    invariant_counts: dict[str, int]
    stage_timings_ms: dict[str, float]
    model_manifest: dict[str, str]
    config_manifest: dict[str, str]
    calibration_hash: str
    release_ready: bool
    release_blockers: tuple[str, ...]
    predicate_proposal_count: int
    completion_rejected_count: int
    completion_error: str


class BuildCoordinator:
    def __init__(self, dependencies: BuildDependencies) -> None:
        self.dependencies = dependencies

    def build(
        self,
        records: Sequence[NoteRecord],
        target: str | Path,
        *,
        replace_derived: bool,
    ) -> BuildReport:
        if not replace_derived:
            raise BuildError("--replace-derived is required for a clean knowledge build")
        target_path = Path(target)
        calibration = load_calibration(
            Path(__file__).resolve().parents[2]
            / "evals"
            / "graph_retrieval"
            / "calibration-v1.json"
        )
        if self.dependencies.embedding_model_id != calibration.model_manifest["embedding"]:
            raise BuildError("embedding model does not match the calibration artifact")
        model_manifest = dict(calibration.model_manifest)
        config_manifest = {
            "builder": "knowledge-build-v1",
            "config_hash": calibration.config_hash,
        }
        ordered = sorted(records, key=lambda row: row.note_id)
        source_snapshot = _source_snapshot(ordered)
        build_inputs = json.dumps(
            {
                "builder": "knowledge-build-v1",
                "source_snapshot": source_snapshot,
                "calibration_hash": calibration.hash,
                "model_manifest": model_manifest,
                "config_manifest": config_manifest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        build_id = "build-" + hashlib.sha256(
            build_inputs.encode("utf-8")
        ).hexdigest()[:20]
        temporary = target_path.with_name(
            f"{target_path.stem}.build-{build_id}{target_path.suffix or '.db'}"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        store: KnowledgeStore | None = None
        stages: list[str] = []
        stage_timings_ms: dict[str, float] = {}
        stage_started = time.monotonic()

        def stage_done(name: str) -> None:
            nonlocal stage_started
            now = time.monotonic()
            stages.append(name)
            stage_timings_ms[name] = (now - stage_started) * 1000.0
            stage_started = now

        try:
            store = KnowledgeStore(temporary)
            store.begin_build(
                build_id,
                source_snapshot=source_snapshot,
                model_manifest=model_manifest,
                config_manifest=config_manifest,
                calibration_hash=calibration.hash,
                started_at=self.dependencies.now(),
            )
            stage_done("source")
            passage_rows = []
            for note in ordered:
                store.write_note(note, build_id=build_id, extraction_status="pending")
                drafts = build_passages(
                    note,
                    self.dependencies.tokenizer,
                    version="passages-v1",
                )
                store.write_passages(
                    note.note_id,
                    drafts,
                    created=note.created or note.modified,
                    modified=note.modified,
                    build_id=build_id,
                )
                passage_rows.extend((note, draft) for draft in drafts)
            stage_done("passages")

            embedded_by_id = {}
            for note in ordered:
                drafts = tuple(draft for source, draft in passage_rows if source.note_id == note.note_id)
                embedded = embed_passages(
                    drafts,
                    title=note.title,
                    source=note.source,
                    embedder=self.dependencies.embedder,
                    model_id=self.dependencies.embedding_model_id,
                    build_id=build_id,
                    dim=self.dependencies.embedding_dim,
                )
                store.write_passage_embeddings(embedded)
                embedded_by_id.update({row.passage_id: row for row in embedded})
            stage_done("embeddings")
            extractions = (
                self.dependencies.extractor(ordered, tokenizer=self.dependencies.tokenizer)
                if self.dependencies.extractor is not None
                else {}
            )
            stage_done("extraction")
            mention_inputs = []
            passage_by_note = {
                note.note_id: [draft for source, draft in passage_rows if source.note_id == note.note_id]
                for note in ordered
            }
            for note in ordered:
                extraction = extractions.get(note.note_id)
                store.write_note(
                    note,
                    build_id=build_id,
                    extraction_status=("failed" if extraction is None or extraction.failed else "complete"),
                    extraction_error=(extraction.error if extraction is not None else "extractor unavailable"),
                )
                if extraction is None or extraction.failed:
                    continue
                for mention in extraction.mentions:
                    passage = next((p for p in passage_by_note[note.note_id] if p.char_start <= mention.char_start and mention.char_end <= p.char_end), None)
                    if passage is None:
                        continue
                    identity = f"{note.note_id}\x1f{mention.etype}\x1f{mention.char_start}\x1f{mention.char_end}"
                    mention_inputs.append(MentionInput("mention-" + hashlib.sha256(identity.encode()).hexdigest()[:24], note.note_id, passage.passage_id, mention))
            resolution = EntityResolver(total_notes=max(1, len(ordered))).resolve_mentions(mention_inputs)
            for entity in resolution.entities:
                store.insert_entity(entity_id=entity.entity_id, canonical=entity.canonical, normalized=entity.normalized, etype=entity.etype, df=entity.df, traversal_eligible=entity.traversal_eligible, ambiguity_key=entity.ambiguity_key, build_id=build_id)
                for alias in entity.aliases:
                    store.insert_alias(entity.entity_id, alias, "resolved", 1.0)
            for row in resolution.mentions:
                m = row.mention
                store.insert_mention(mention_id=row.mention_id, entity_id=row.entity_id, passage_id=row.passage_id, note_id=row.note_id, surface=m.surface, etype=m.etype, char_start=m.char_start, char_end=m.char_end, confidence=m.confidence, extractor=m.extractor, schema_id=m.schema_id, build_id=build_id)
            stage_done("entities")
            resolved_by_note_span = {(row.note_id, row.mention.char_start, row.mention.char_end, row.mention.surface.casefold()): row for row in resolution.mentions}
            fact_rows = []
            for note in ordered:
                extraction = extractions.get(note.note_id)
                if extraction is None:
                    continue
                for candidate in extraction.fact_candidates:
                    subject = resolved_by_note_span.get((note.note_id, candidate.subject_start, candidate.subject_end, candidate.head.casefold()))
                    object_ = resolved_by_note_span.get((note.note_id, candidate.object_start, candidate.object_end, candidate.tail.casefold()))
                    passage = next((p for p in passage_by_note[note.note_id] if p.char_start <= candidate.assertion_start and candidate.assertion_end <= p.char_end), None)
                    if subject and object_ and passage:
                        fact_rows.append(promote_fact(candidate, subject_entity_id=subject.entity_id, object_entity_id=object_.entity_id, source_passage_id=passage.passage_id, source_note_id=note.note_id, observed_at=(candidate.observed_at if candidate.observed_at is not None else note.modified), valid_from=candidate.valid_from, valid_to=candidate.valid_to, build_id=build_id))
            proposals: tuple[Any, ...] = ()
            completion_rejected = 0
            completion_error = ""
            if self.dependencies.relation_completer is not None and ordered:
                _provisional_quality, provisional_values = _compute_note_ranks(
                    ordered,
                    (),
                    now=self.dependencies.now(),
                )
                facts_by_note: dict[str, int] = {}
                for fact in fact_rows:
                    if fact.status == "active":
                        facts_by_note[fact.source_note_id] = facts_by_note.get(fact.source_note_id, 0) + 1
                completion_notes = tuple(
                    CompletionNote(
                        note_id=note.note_id,
                        value_score=provisional_values.get(note.note_id, 0.0),
                        observed_at=note.modified,
                        entities=tuple(
                            CompletionEntity(
                                row.entity_id,
                                row.mention.surface,
                                row.mention.char_start,
                                row.mention.char_end,
                            )
                            for row in resolution.mentions
                            if row.note_id == note.note_id
                        ),
                        passages=tuple(
                            CompletionPassage(draft.passage_id, draft.text, draft.char_start)
                            for draft in passage_by_note[note.note_id]
                        ),
                        active_relation_count=facts_by_note.get(note.note_id, 0),
                        schema_gap=False,
                    )
                    for note in ordered
                )
                selected_completion = select_completion_notes(completion_notes)
                if selected_completion:
                    try:
                        completed = parse_completion_response(
                            self.dependencies.relation_completer(selected_completion),
                            selected_completion,
                            build_id=build_id,
                        )
                    except Exception as exc:
                        completion_error = f"{type(exc).__name__}: relation completion failed"
                    else:
                        fact_rows.extend(completed.facts)
                        proposals = completed.proposals
                        completion_rejected = completed.rejected
            for proposal in proposals:
                store.insert_predicate_proposal(proposal)
            stage_done("relation_completion")
            fact_rows = list(reconcile_temporal_facts(fact_rows))
            for fact in fact_rows:
                store.insert_fact(fact)
            stage_done("facts")

            graph_passages = tuple(
                PassageGraphInput(
                    passage_id=draft.passage_id,
                    note_id=note.note_id,
                    ordinal=draft.ordinal,
                    vector=embedded_by_id[draft.passage_id].vector,
                )
                for note, draft in passage_rows
            )
            entity_graph = tuple(EntityGraphInput(e.entity_id, e.canonical, e.etype, e.df, e.traversal_eligible) for e in resolution.entities)
            contains_groups: dict[tuple[str, str], list[float]] = {}
            for mention in resolution.mentions:
                contains_groups.setdefault(
                    (mention.passage_id, mention.entity_id), []
                ).append(mention.mention.confidence)
            entity_df = {entity.entity_id: entity.df for entity in resolution.entities}
            contains_graph = tuple(
                PassageEntityInput(
                    passage_id=passage_id,
                    entity_id=entity_id,
                    confidence=max(confidences),
                    mentions=len(confidences),
                    df=entity_df[entity_id],
                )
                for (passage_id, entity_id), confidences in sorted(contains_groups.items())
            )
            base_graph = BaseGraphBuilder(
                build_id=build_id,
                semantic_k=config.BASE_GRAPH_PASSAGE_K,
                semantic_floor=config.GRAPH_MIN_SIM,
            ).build(graph_passages, entity_graph, contains_graph, fact_rows)
            store.write_semantic_edges(
                tuple(
                    (edge.src_id, edge.dst_id, edge.weight, edge.weight)
                    for edge in base_graph.edges
                    if edge.kind == "semantic" and edge.src_id < edge.dst_id
                ),
                build_id=build_id,
            )
            contains_by_pair = {
                (row.passage_id, row.entity_id): row for row in contains_graph
            }
            store.write_passage_entity_edges(
                tuple(
                    (
                        edge.src_id,
                        edge.dst_id,
                        contains_by_pair[(edge.src_id, edge.dst_id)].confidence,
                        int(edge.metadata["mentions"]),
                        float(edge.metadata["idf"]),
                        edge.weight,
                    )
                    for edge in base_graph.edges
                    if edge.kind == "contains"
                    and (edge.src_id, edge.dst_id) in contains_by_pair
                ),
                build_id=build_id,
            )
            stage_done("base_graph")
            projection = build_projection(
                base_graph,
                tuple(
                    ProjectionPassage(draft.passage_id, note.note_id)
                    for note, draft in passage_rows
                ),
                {entity.entity_id: entity.canonical for entity in resolution.entities},
                max_degree=config.PROJECTION_MAX_DEGREE,
            )
            store.write_projection(projection, build_id=build_id)
            stage_done("projection")
            quality_ranks, value_scores = _compute_note_ranks(
                ordered,
                projection,
                now=self.dependencies.now(),
            )
            store.set_note_ranks(
                quality_ranks,
                value_scores,
                build_id=build_id,
            )
            stage_done("ranks")
            fact_embeddings = embed_facts(
                store.fact_contexts(),
                embedder=self.dependencies.embedder,
                model_id=self.dependencies.embedding_model_id,
                build_id=build_id,
                dim=self.dependencies.embedding_dim,
            )
            store.write_fact_embeddings(fact_embeddings)
            stage_done("fact_embeddings")
            integrity = store.validate_invariants()
            if not integrity.ok:
                raise BuildError(f"knowledge invariant failure: {integrity}")
            graph_hash = _graph_hash(base_graph, projection)
            stage_done("validation")
            counts = store.counts()
            component_count = _component_count(base_graph)
            hub_exclusions = tuple(
                (row.entity_id, "document_frequency_hub")
                for row in sorted(resolution.entities, key=lambda item: item.entity_id)
                if not row.traversal_eligible
            )
            failed_notes = tuple(
                (
                    note.note_id,
                    (
                        row.error
                        if (row := extractions.get(note.note_id)) is not None
                        else "extractor unavailable"
                    ),
                )
                for note in ordered
                if (row := extractions.get(note.note_id)) is None or row.failed
            )
            release_blocker_rows = _calibration_release_blockers(calibration)
            if ordered and self.dependencies.relation_completer is None:
                release_blocker_rows.append("relation_completion_unavailable")
            if completion_error:
                release_blocker_rows.append("relation_completion_failed")
            release_blockers = tuple(sorted(release_blocker_rows))
            health = (
                "healthy"
                if self.dependencies.extractor is not None
                and not failed_notes
                and not release_blockers
                else "degraded"
            )
            store.complete_build(
                build_id,
                completed_at=self.dependencies.now(),
                health=health,
                graph_hash=graph_hash,
                base_edge_count=len(base_graph.edges),
                projection_count=len(projection),
            )
            store.close()
            store = None
            os.replace(temporary, target_path)
            stage_done("activation")
            return BuildReport(
                build_id=build_id,
                target=target_path,
                source_snapshot=source_snapshot,
                graph_hash=graph_hash,
                health=health,
                stages=tuple(stages),
                note_count=len(ordered),
                passage_count=len(passage_rows),
                entity_count=len(resolution.entities),
                fact_count=len(fact_rows),
                base_edge_count=len(base_graph.edges),
                projection_edge_count=len(projection),
                extraction_failure_count=len(failed_notes),
                failed_notes=failed_notes,
                component_count=component_count,
                dangling_node_count=len(base_graph.dangling_rows),
                hub_exclusions=hub_exclusions,
                invariant_counts={
                    key: int(value) for key, value in sorted(counts.items())
                },
                stage_timings_ms=stage_timings_ms,
                model_manifest=model_manifest,
                config_manifest=config_manifest,
                calibration_hash=calibration.hash,
                release_ready=health == "healthy",
                release_blockers=release_blockers,
                predicate_proposal_count=len(proposals),
                completion_rejected_count=completion_rejected,
                completion_error=completion_error,
            )
        except Exception as exc:
            if store is not None:
                store.close()
            if temporary.exists():
                temporary.unlink()
            if isinstance(exc, BuildError):
                raise
            raise BuildError(str(exc)) from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: BuildDependencies | None = None,
    connector_loader: Callable[[], Sequence[NoteRecord]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Build a clean Taurine knowledge index")
    parser.add_argument("--db", required=True)
    parser.add_argument("--from-enabled-connectors", action="store_true")
    parser.add_argument("--replace-derived", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if not args.from_enabled_connectors:
        raise BuildError("--from-enabled-connectors is required")
    if not args.yes:
        response = input(f"Replace derived knowledge index at {args.db}? [y/N] ")
        if response.strip().casefold() not in {"y", "yes"}:
            return 1
    if connector_loader is None:
        from taurine_core.connectors.registry import ConnectorRegistry
        from taurine_core.indexer.store import Index

        source_index = Index(Path("taurine.db"))
        connector_registry = ConnectorRegistry()

        def connector_loader() -> Sequence[NoteRecord]:
            return connector_registry.extract_all_enabled(source_index)
    deps = dependencies or BuildDependencies(
        tokenizer=passage_tokenizer(),
        embedder=embed_texts,
        extractor=extract_notes,
    )
    report = BuildCoordinator(deps).build(
        connector_loader(),
        args.db,
        replace_derived=args.replace_derived,
    )
    print(json.dumps({**report.__dict__, "target": str(report.target)}, sort_keys=True))
    return 0


def _source_snapshot(records: Sequence[NoteRecord]) -> str:
    payload = [(row.note_id, row.content_hash, row.source) for row in records]
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _calibration_release_blockers(calibration: Any) -> list[str]:
    blockers: list[str] = []
    for mode, settings in calibration.modes.items():
        for head, curve in settings.heads.items():
            if curve.sample_count <= 0 or not curve.held_out:
                blockers.append(f"uncalibrated_retrieval_head:{mode}:{head}")
    return sorted(blockers)


def _compute_note_ranks(
    notes: Sequence[NoteRecord],
    projection: Sequence[Any],
    *,
    now: float,
) -> tuple[dict[str, float], dict[str, float]]:
    note_ids = tuple(note.note_id for note in notes)
    if not note_ids:
        return {}, {}
    row_by_id = {note_id: index for index, note_id in enumerate(note_ids)}
    matrix = np.zeros((len(note_ids), len(note_ids)), dtype=np.float64)
    for edge in projection:
        left = row_by_id.get(edge.src_note_id)
        right = row_by_id.get(edge.dst_note_id)
        if left is None or right is None or left == right:
            continue
        weight = max(0.0, float(edge.weight))
        matrix[right, left] += weight
        matrix[left, right] += weight
    note_by_id = {note.note_id: note for note in notes}
    teleport = np.asarray(
        [
            source_rank(note_by_id[note_id].source)
            * recency_multiplier(
                note_by_id[note_id].source,
                (float(now) - note_by_id[note_id].modified) / 86400.0,
            )
            for note_id in note_ids
        ],
        dtype=np.float64,
    )
    teleport /= float(teleport.sum())
    strengths = matrix.sum(axis=0)
    transition = np.zeros_like(matrix)
    nonzero = strengths > 0.0
    transition[:, nonzero] = matrix[:, nonzero] / strengths[nonzero]
    rank = teleport.copy()
    damping = float(config.QUALITY_DAMPING)
    for _ in range(int(config.QUALITY_MAX_ITERS)):
        dangling = float(rank[~nonzero].sum())
        updated = damping * (transition @ rank + dangling * teleport) + (1.0 - damping) * teleport
        if float(np.abs(updated - rank).sum()) < float(config.QUALITY_TOL):
            rank = updated
            break
        rank = updated
    median = float(np.median(rank))
    quality = {
        note_id: float(
            np.clip(
                np.log(float(rank[index]) / median) if median > 0.0 and rank[index] > 0.0 else -config.QUALITY_CLIP,
                -config.QUALITY_CLIP,
                config.QUALITY_CLIP,
            )
        )
        for index, note_id in enumerate(note_ids)
    }
    low, high = min(quality.values()), max(quality.values())
    span = high - low
    values: dict[str, float] = {}
    for note_id in note_ids:
        note = note_by_id[note_id]
        quality_weight = 1.0 if span == 0.0 else 0.1 + ((quality[note_id] - low) / span) * 0.9
        age_days = max(0.0, (float(now) - note.modified) / 86400.0)
        values[note_id] = (
            quality_weight
            * float(0.5 ** (age_days / config.VALUE_RECENCY_HALF_LIFE_DAYS))
            * source_rank(note.source)
        )
    return quality, values


def _graph_hash(base_graph: object, projection: object) -> str:
    graph = base_graph
    payload = {
        "nodes": list(graph.node_ids),
        "edges": [
            (edge.src_id, edge.dst_id, edge.kind, edge.direction, round(edge.weight, 12), edge.fact_ids)
            for edge in graph.edges
        ],
        "projection": [
            (
                edge.src_note_id,
                edge.dst_note_id,
                edge.edge_type,
                round(edge.weight, 12),
                edge.shared,
                edge.fact_ids,
            )
            for edge in projection
        ],
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _component_count(graph: Any) -> int:
    if not graph.node_ids:
        return 0
    undirected = graph.matrix.maximum(graph.matrix.transpose())
    count, _labels = connected_components(undirected, directed=False)
    return int(count)


if __name__ == "__main__":
    raise SystemExit(main())
