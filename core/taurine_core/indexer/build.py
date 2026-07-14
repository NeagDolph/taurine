"""Deterministic clean knowledge-index build and atomic activation command."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import logging
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.sparse.csgraph import connected_components

from taurine_core import config
from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.base_graph import (
    BaseGraphBuilder,
    EntityGraphInput,
    PassageEntityInput,
    PassageGraphInput,
    materialize_base_graph,
)
from taurine_core.indexer.cluster_labels import (
    ClusterSummary,
    build_cluster_summaries,
    fallback_label,
)
from taurine_core.indexer.communities import ClusterRecord, cluster_projection
from taurine_core.indexer.context_enrich import (
    EnrichmentBatch,
    EnrichmentResult,
    context_enricher_for,
    enrich_passages,
)
from taurine_core.indexer.edge_adjudication import (
    MergeOutcome,
    SemanticPair,
    adjudicate_facts,
    adjudicate_merges,
    adjudicate_semantic_edges,
    edge_adjudicator_for,
    merge_mention_contexts,
)
from taurine_core.indexer.embed import embed_facts, embed_passages, embed_texts
from taurine_core.indexer.entities import EntityResolver, MentionInput
from taurine_core.indexer.extract import extract_notes_dispatch as extract_notes
from taurine_core.indexer.doc_kind import snapshot_document
from taurine_core.indexer.facts import PREDICATES, promote_fact, reconcile_temporal_facts
from taurine_core.indexer.knowledge_store import KnowledgeStore, compute_graph_hash
from taurine_core.indexer.passages import TokenizerPort, build_passages
from taurine_core.indexer.projection import ProjectionPassage, build_projection
from taurine_core.indexer.relation_completion import (
    COMPLETION_SYSTEM_PROMPT,
    CompletionEntity,
    CompletionNote,
    CompletionPassage,
    completion_note_key,
    completion_note_payload,
    group_completion_rows,
    parse_completion_response,
    select_completion_notes,
)
from taurine_core.indexer.priors import recency_multiplier, source_rank
from taurine_core.indexer.salience import content_quality
from taurine_core.models_local import local_model_path, passage_tokenizer
from taurine_core.schemas import NoteRecord


_LOG = logging.getLogger("taurine.build")


class BuildError(RuntimeError):
    pass


class BuildCancelled(BuildError):
    """Raised from a progress callback to abort a build cooperatively."""


@dataclass(frozen=True)
class BuildDependencies:
    tokenizer: TokenizerPort
    embedder: Callable[[list[str]], np.ndarray]
    embedding_dim: int = config.EMBED_DIM
    embedding_model_id: str = config.EMBED_MODEL
    now: Callable[[], float] = time.time
    extractor: Callable[..., dict[str, Any]] | None = None
    relation_completer: Callable[[Sequence[CompletionNote]], object] | None = None
    # optional situating-prefix lane (contextual retrieval): absence or any
    # batch failure means empty contexts, never a failed build. context_cache
    # is a BuildCache in production so unchanged passages re-enrich for free.
    context_enricher: Callable[[EnrichmentBatch], object] | None = None
    context_cache: Any | None = None
    # optional grounded edge-adjudication judge (edge-quality items 1/5b/6):
    # absence means facts stay unadjudicated, semantic edges keep empty
    # reasons, and no gray-zone merges happen — never a failed build.
    edge_adjudicator: Callable[[str, dict], Any] | None = None
    adjudication_cache: Any | None = None
    # per-note completion cache (BuildCache in production): unchanged notes —
    # including notes that yielded nothing — are never re-asked
    completion_cache: Any | None = None


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
    degenerate_fact_count: int
    degenerate_facts: tuple[tuple[str, str], ...]
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
    degraded_reasons: tuple[str, ...] = ()
    straddling_span_count: int = 0
    contains_weight_stats: dict[str, float] | None = None
    salience_stats: dict[str, float] | None = None
    salience_gated_count: int = 0
    salience_gated_by_source: dict[str, int] | None = None
    near_duplicate_note_pairs: tuple[tuple[str, str], ...] = ()
    cluster_count: int = 0
    cluster_size_histogram: dict[str, int] | None = None
    context_enriched_count: int = 0
    context_enrich_failed_batches: int = 0
    # grounded edge adjudication (audit contract, edge-quality items 1/5b/6):
    # per-outcome counts keyed by predicate, plus lane health counters
    adjudicated_fact_count: int = 0
    unadjudicated_fact_count: int = 0
    adjudication_failed_batches: int = 0
    adjudication_malformed_rows: int = 0
    adjudicated_by_predicate: dict[str, int] | None = None
    adjudication_kept_by_predicate: dict[str, int] | None = None
    adjudication_downweighted_by_predicate: dict[str, int] | None = None
    adjudication_rejected_by_predicate: dict[str, int] | None = None
    adjudication_relabeled_by_predicate: dict[str, int] | None = None
    semantic_reasoned_edge_count: int = 0
    semantic_reason_dropped_count: int = 0
    semantic_reason_failed_batches: int = 0
    merge_gray_zone_pair_count: int = 0
    merge_adjudicated_count: int = 0
    merge_approved_count: int = 0
    merge_failed_batches: int = 0
    peak_memory_gb: float = 0.0
    peak_memory_stage: str = ""


class BuildMemoryExceeded(Exception):
    """Raised when a build's resident set crosses the configured ceiling."""


class _BuildMemoryWatchdog:
    """Samples process RSS during a build.

    Records the peak (for observability — the build report surfaces it) and
    trips a cooperative abort flag when RSS crosses ``ceiling_gb``. The flag is
    checked at every stage boundary so a runaway build fails with a clean
    ``BuildError`` (temp DB discarded, prior snapshot untouched) instead of the
    OS OOM-killing the sidecar. Device (MPS/CUDA) runaway is caught separately
    by the allocator watermark cap set in ``taurine_core.__init__`` which turns
    an over-allocation into a catchable RuntimeError.
    """

    def __init__(self, ceiling_gb: float) -> None:
        self.ceiling_gb = float(ceiling_gb)
        self.peak_gb = 0.0
        self.peak_stage = ""
        self._current_stage = "startup"
        self._tripped = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = None
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception:  # noqa: BLE001 — psutil optional; fall back below
            self._proc = None

    def _rss_gb(self) -> float:
        if self._proc is not None:
            try:
                return self._proc.memory_info().rss / 1e9
            except Exception:  # noqa: BLE001
                return 0.0
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes, Linux kilobytes.
            return usage / (1e9 if sys.platform == "darwin" else 1e6)
        except Exception:  # noqa: BLE001
            return 0.0

    def set_stage(self, stage: str) -> None:
        self._current_stage = stage

    def check(self) -> None:
        # Sample synchronously so a stage boundary evaluates the ceiling
        # deterministically, independent of the poll thread's timing.
        self._sample()
        if self._tripped:
            raise BuildMemoryExceeded(
                f"build resident set exceeded {self.ceiling_gb:.4g} GB "
                f"(peak {self.peak_gb:.3g} GB at stage '{self.peak_stage}')"
            )

    def _sample(self) -> None:
        rss = self._rss_gb()
        if rss > self.peak_gb:
            self.peak_gb = rss
            self.peak_stage = self._current_stage
        if self.ceiling_gb > 0 and rss > self.ceiling_gb:
            self._tripped = True
            _LOG.error(
                "build memory watchdog: RSS %.1f GB > ceiling %.1f GB at stage %s — "
                "requesting abort",
                rss,
                self.ceiling_gb,
                self._current_stage,
            )

    def _run(self) -> None:
        # Sample immediately so a fast build still records a peak and a breached
        # ceiling trips before the first poll interval elapses.
        self._sample()
        while not self._stop.wait(0.5):
            self._sample()
            if self._tripped:
                return

    def __enter__(self) -> "_BuildMemoryWatchdog":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class BuildCoordinator:
    def __init__(self, dependencies: BuildDependencies) -> None:
        self.dependencies = dependencies

    def build(
        self,
        records: Sequence[NoteRecord],
        target: str | Path,
        *,
        replace_derived: bool,
        on_stage: Callable[[str], None] | None = None,
        owner_aliases: Sequence[str] = (),
        degraded_reasons: Sequence[str] = (),
        relation_completion_opt_out: bool = False,
    ) -> BuildReport:
        if not replace_derived:
            raise BuildError("--replace-derived is required for a clean knowledge build")
        watchdog = _BuildMemoryWatchdog(config.BUILD_MEMORY_CEILING_GB)

        def announce(stage: str) -> None:
            watchdog.set_stage(stage)
            watchdog.check()
            if on_stage is not None:
                on_stage(stage)
        target_path = Path(target)
        health_reasons = sorted(dict.fromkeys(degraded_reasons))
        calibration = load_calibration(
            Path(__file__).resolve().parents[2]
            / "evals"
            / "graph_retrieval"
            / "calibration-v1.json"
        )
        if self.dependencies.embedding_model_id != calibration.model_manifest["embedding"]:
            raise BuildError("embedding model does not match the calibration artifact")
        model_manifest = dict(calibration.model_manifest)
        # L26: fingerprints live in config_manifest, NOT model_manifest — the
        # runtime compares model_manifest for equality against the artifact's
        # identity strings, so fingerprints ride a separate key it tolerates.
        config_manifest = {
            "builder": "knowledge-build-v1",
            "config_hash": calibration.config_hash,
            "model_fingerprints": json.dumps(
                {
                    "embedding": _model_fingerprint(self.dependencies.embedding_model_id),
                    "gliner": _model_fingerprint(config.GLINER2_MODEL),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
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
            watchdog.check()
            _LOG.info(
                "build %s: stage %s done in %.0f ms", build_id, name, stage_timings_ms[name]
            )

        watchdog.__enter__()
        try:
            announce("source")
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
            announce("passages")
            # §0.2: note_salience = connector ingest salience × deterministic
            # content quality. Salience gates extraction only — passages, FTS,
            # and embeddings are ALWAYS built, and the gate re-evaluates on
            # every build (never a sticky failure).
            note_salience = {
                note.note_id: note.ingest_salience * content_quality(note.text)
                for note in ordered
            }
            gated_note_ids = {
                note_id
                for note_id, value in note_salience.items()
                if value < config.EXTRACTION_SALIENCE_FLOOR
            }
            passage_rows = []
            for note in ordered:
                store.write_note(
                    note,
                    build_id=build_id,
                    extraction_status="pending",
                    ingest_salience=note.ingest_salience,
                    salience=note_salience[note.note_id],
                )
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
            announce("context_enrichment")
            # Item 4 (contextual retrieval): episodic passages get a cached,
            # fail-open situating prefix that reaches ONLY the FTS context
            # column and the embedding frame below — passage text, offsets,
            # and evidence rendering are untouched.
            drafts_by_note: dict[str, list[Any]] = {}
            for note, draft in passage_rows:
                drafts_by_note.setdefault(note.note_id, []).append(draft)
            enrichment: EnrichmentResult = enrich_passages(
                tuple(
                    (note, tuple(drafts_by_note.get(note.note_id, ())))
                    for note in ordered
                ),
                self.dependencies.context_enricher,
                cache=self.dependencies.context_cache,
            )
            if enrichment.contexts:
                store.set_passage_contexts(enrichment.contexts)
            if enrichment.enriched_count or enrichment.failed_batches:
                _LOG.info(
                    "build %s: %d passages context-enriched, %d enrichment batches failed",
                    build_id,
                    enrichment.enriched_count,
                    enrichment.failed_batches,
                )
            stage_done("context_enrichment")

            embedded_by_id = {}
            for position, note in enumerate(ordered):
                announce(f"embeddings {position + 1}/{len(ordered)}")
                drafts = tuple(draft for source, draft in passage_rows if source.note_id == note.note_id)
                embedded = embed_passages(
                    drafts,
                    title=note.title,
                    source=note.source,
                    contexts=enrichment.contexts,
                    embedder=self.dependencies.embedder,
                    model_id=self.dependencies.embedding_model_id,
                    build_id=build_id,
                    dim=self.dependencies.embedding_dim,
                )
                store.write_passage_embeddings(embedded)
                embedded_by_id.update({row.passage_id: row for row in embedded})
            stage_done("embeddings")
            # extraction runs only for notes clearing the salience floor;
            # gated notes stay searchable but grow no entities/facts
            extractable = [note for note in ordered if note.note_id not in gated_note_ids]
            announce(f"extraction 0/{len(extractable)}")
            extractor_kwargs: dict[str, Any] = {"tokenizer": self.dependencies.tokenizer}
            if self.dependencies.extractor is not None and _accepts_kwarg(
                self.dependencies.extractor, "progress"
            ):
                extractor_kwargs["progress"] = lambda done, total: announce(
                    f"extraction {done}/{total}"
                )
            extractions = (
                self.dependencies.extractor(extractable, **extractor_kwargs)
                if self.dependencies.extractor is not None
                else {}
            )
            stage_done("extraction")
            announce("entities")
            mention_inputs = []
            straddling_spans = 0
            passage_by_note = {
                note.note_id: [draft for source, draft in passage_rows if source.note_id == note.note_id]
                for note in ordered
            }
            for note in ordered:
                if note.note_id in gated_note_ids:
                    store.write_note(
                        note,
                        build_id=build_id,
                        extraction_status="skipped_low_salience",
                        ingest_salience=note.ingest_salience,
                        salience=note_salience[note.note_id],
                    )
                    continue
                extraction = extractions.get(note.note_id)
                store.write_note(
                    note,
                    build_id=build_id,
                    extraction_status=("failed" if extraction is None or extraction.failed else "complete"),
                    extraction_error=(extraction.error if extraction is not None else "extractor unavailable"),
                    ingest_salience=note.ingest_salience,
                    salience=note_salience[note.note_id],
                )
                if extraction is None or extraction.failed:
                    continue
                for mention in extraction.mentions:
                    if mention.char_start < 0 or mention.char_end <= mention.char_start:
                        continue  # would violate the mention span CHECK constraint
                    passage, straddled = _passage_for_span(
                        passage_by_note[note.note_id], mention.char_start, mention.char_end
                    )
                    if passage is None:
                        continue
                    if straddled:
                        straddling_spans += 1
                    identity = f"{note.note_id}\x1f{mention.etype}\x1f{mention.char_start}\x1f{mention.char_end}"
                    mention_inputs.append(MentionInput("mention-" + hashlib.sha256(identity.encode()).hexdigest()[:24], note.note_id, passage.passage_id, mention))
            # mention identity is (note, etype, span) — surface excluded — so an
            # extractor repeating a span must not become a PK violation
            unique_mentions: dict[str, MentionInput] = {}
            for row in mention_inputs:
                current = unique_mentions.get(row.mention_id)
                if current is None or row.mention.confidence > current.mention.confidence:
                    unique_mentions[row.mention_id] = row
            mention_inputs = list(unique_mentions.values())
            # resolution stages 5-6 (§7.1): surface embeddings generate merge
            # candidates. The dependency embedder is BuildCache-backed in
            # production, so unique surfaces are content-cached across builds.
            similarity: Callable[[str, str], float] | None = None
            unique_surfaces = sorted({row.mention.surface.strip() for row in mention_inputs})
            if unique_surfaces:
                surface_matrix = np.asarray(
                    self.dependencies.embedder(list(unique_surfaces)), dtype=np.float32
                )
                norms = np.linalg.norm(surface_matrix, axis=1, keepdims=True)
                surface_matrix /= np.maximum(norms, 1e-12)
                surface_rows = {
                    surface: index for index, surface in enumerate(unique_surfaces)
                }

                def similarity(left: str, right: str) -> float:
                    left_row = surface_rows.get(left.strip())
                    right_row = surface_rows.get(right.strip())
                    if left_row is None or right_row is None:
                        return 0.0
                    return float(surface_matrix[left_row] @ surface_matrix[right_row])

            def resolve_entities(
                forced_merges: Sequence[tuple[str, str, str]] = (),
            ) -> Any:
                return EntityResolver(
                    total_notes=max(1, len(ordered)),
                    owner_aliases={value.casefold() for value in owner_aliases},
                    similarity=similarity,
                    similarity_threshold=config.ENTITY_EMBED_SIM,
                    note_sources={note.note_id: note.source for note in ordered},
                ).resolve_mentions(mention_inputs, forced_merges=forced_merges)

            resolution = resolve_entities()
            # item 6: gray-zone pairs the deterministic resolver declined go
            # to the grounded merge judge; approved pairs force a second
            # union pass BEFORE entities/mentions/facts are inserted.
            merge_outcome = MergeOutcome((), 0, 0, 0)
            gray_zone_pair_count = len(resolution.gray_zone_pairs)
            if self.dependencies.edge_adjudicator is not None and resolution.gray_zone_pairs:
                merge_outcome = adjudicate_merges(
                    resolution.gray_zone_pairs,
                    contexts=merge_mention_contexts(
                        resolution, {note.note_id: note.text for note in ordered}
                    ),
                    judge=self.dependencies.edge_adjudicator,
                    cache=self.dependencies.adjudication_cache,
                )
                if merge_outcome.approved:
                    _LOG.info(
                        "build %s: merge judge approved %d of %d gray-zone pairs",
                        build_id,
                        merge_outcome.approved_count,
                        gray_zone_pair_count,
                    )
                    resolution = resolve_entities(merge_outcome.approved)
            for entity in resolution.entities:
                store.insert_entity(entity_id=entity.entity_id, canonical=entity.canonical, normalized=entity.normalized, etype=entity.etype, df=entity.df, traversal_eligible=entity.traversal_eligible, traversal_reason=entity.traversal_reason, ambiguity_key=entity.ambiguity_key, build_id=build_id)
                for alias in entity.aliases:
                    store.insert_alias(entity.entity_id, alias, "resolved", 1.0)
            for row in resolution.mentions:
                m = row.mention
                store.insert_mention(mention_id=row.mention_id, entity_id=row.entity_id, passage_id=row.passage_id, note_id=row.note_id, surface=m.surface, etype=m.etype, char_start=m.char_start, char_end=m.char_end, confidence=m.confidence, extractor=m.extractor, schema_id=m.schema_id, build_id=build_id)
            stage_done("entities")
            announce("relation_completion")
            # keyed per (note, span, surface, etype): a span tagged with two
            # types keeps both resolutions instead of a hash-arbitrary one (L6)
            resolved_by_note_span: dict[tuple[str, int, int, str], list[Any]] = {}
            for row in resolution.mentions:
                resolved_by_note_span.setdefault(
                    (row.note_id, row.mention.char_start, row.mention.char_end, row.mention.surface.casefold()),
                    [],
                ).append(row)
            # snapshot documents (resumes, offer letters, transcripts) describe
            # the world as of their own date: their facts anchor observed_at to
            # the date parsed from the title, else note.created — never
            # note.modified — so an old resume synced yesterday cannot outrank
            # a recent first-person note (edge-quality spec item 2)
            snapshot_notes = {note.note_id: snapshot_document(note) for note in ordered}
            fact_observed_default: dict[str, float] = {}
            for note in ordered:
                is_snapshot, doc_date = snapshot_notes[note.note_id]
                if is_snapshot:
                    fact_observed_default[note.note_id] = float(
                        doc_date if doc_date is not None else note.created
                    )
                else:
                    fact_observed_default[note.note_id] = float(note.modified)
            store.set_note_snapshot_flags(
                {note_id: flag for note_id, (flag, _) in snapshot_notes.items()}
            )
            fact_rows = []
            degenerate_facts: list[tuple[str, str]] = []
            invalid_candidates = 0
            for note in ordered:
                extraction = extractions.get(note.note_id)
                if extraction is None:
                    continue
                for candidate in extraction.fact_candidates:
                    # A stale extraction cache can hold retired predicates or
                    # malformed spans; either would violate DB constraints.
                    if (
                        candidate.predicate not in PREDICATES
                        or candidate.assertion_start < 0
                        or candidate.assertion_end <= candidate.assertion_start
                    ):
                        invalid_candidates += 1
                        continue
                    subject = _resolve_endpoint(resolved_by_note_span, note.note_id, candidate.subject_start, candidate.subject_end, candidate.head)
                    object_ = _resolve_endpoint(resolved_by_note_span, note.note_id, candidate.object_start, candidate.object_end, candidate.tail)
                    passage, straddled = _passage_for_span(
                        passage_by_note[note.note_id],
                        candidate.assertion_start,
                        candidate.assertion_end,
                    )
                    if subject and object_ and passage:
                        if straddled:
                            straddling_spans += 1
                        # Entity resolution can merge distinct surfaces ("Neil",
                        # "Neil Pathangay") into one entity; a relation whose
                        # endpoints collapse together is dropped, never fatal.
                        if subject.entity_id == object_.entity_id:
                            degenerate_facts.append(
                                (note.note_id, f"{candidate.head} {candidate.predicate} {candidate.tail}")
                            )
                            continue
                        fact_rows.append(promote_fact(candidate, subject_entity_id=subject.entity_id, object_entity_id=object_.entity_id, source_passage_id=passage.passage_id, source_note_id=note.note_id, observed_at=(candidate.observed_at if candidate.observed_at is not None else fact_observed_default[note.note_id]), valid_from=candidate.valid_from, valid_to=candidate.valid_to, build_id=build_id))
            if invalid_candidates:
                _LOG.warning(
                    "build %s: skipped %d fact candidates with unregistered predicates or invalid spans",
                    build_id,
                    invalid_candidates,
                )
            if straddling_spans:
                _LOG.info(
                    "build %s: %d spans straddle adjacent overlapping passages (kept on the earlier passage)",
                    build_id,
                    straddling_spans,
                )
            if degenerate_facts:
                _LOG.warning(
                    "build %s: skipped %d degenerate facts (endpoints merged to one entity): %s",
                    build_id,
                    len(degenerate_facts),
                    "; ".join(f"{note_id}: {text}" for note_id, text in degenerate_facts[:10]),
                )
            proposals: tuple[Any, ...] = ()
            completion_aliases: tuple[Any, ...] = ()
            completion_rejected = 0
            completion_error = ""
            if self.dependencies.relation_completer is not None and ordered:
                # L23/§6.6: the provisional gate is real salience × recency ×
                # source_rank (empty projection → uniform quality weight)
                _, provisional_values, _ = _compute_note_ranks(
                    ordered,
                    (),
                    now=self.dependencies.now(),
                    salience=note_salience,
                )
                facts_by_note: dict[str, int] = {}
                uses_by_note: set[str] = set()
                for fact in fact_rows:
                    if fact.status == "active":
                        facts_by_note[fact.source_note_id] = facts_by_note.get(fact.source_note_id, 0) + 1
                        if fact.predicate == "uses":
                            uses_by_note.add(fact.source_note_id)
                mentions_by_note: dict[str, list[Any]] = {}
                for row in resolution.mentions:
                    mentions_by_note.setdefault(row.note_id, []).append(row)
                entity_kind = {
                    entity.entity_id: (entity.etype, bool(entity.traversal_eligible))
                    for entity in resolution.entities
                }

                def _tool_hop(note_id: str) -> bool:
                    # eligible tool/product mention + a project/org anchor, no
                    # active `uses` fact from this note yet
                    if note_id in uses_by_note:
                        return False
                    kinds = {
                        entity_kind[row.entity_id][0]
                        for row in mentions_by_note.get(note_id, ())
                        if entity_kind.get(row.entity_id, ("", False))[1]
                    }
                    return "product" in kinds and bool(
                        kinds & {"project", "organization"}
                    )

                completion_notes = tuple(
                    CompletionNote(
                        note_id=note.note_id,
                        value_score=provisional_values.get(note.note_id, 0.0),
                        # snapshot-aware: completed facts anchor to the same
                        # corrected observation time as extracted facts
                        observed_at=fact_observed_default[note.note_id],
                        # one CompletionEntity per mention: every span stays
                        # available for endpoint grounding (H8)
                        entities=tuple(
                            CompletionEntity(
                                row.entity_id,
                                row.mention.surface,
                                row.mention.char_start,
                                row.mention.char_end,
                            )
                            for row in mentions_by_note.get(note.note_id, ())
                        ),
                        passages=tuple(
                            CompletionPassage(draft.passage_id, draft.text, draft.char_start)
                            for draft in passage_by_note[note.note_id]
                        ),
                        active_relation_count=facts_by_note.get(note.note_id, 0),
                        schema_gap=False,
                        tool_hop=_tool_hop(note.note_id),
                    )
                    for note in ordered
                    if note.note_id not in gated_note_ids
                )
                selected_completion = select_completion_notes(completion_notes)
                if selected_completion:
                    cache = self.dependencies.completion_cache
                    keys = {
                        note.note_id: completion_note_key(
                            config.COMPLETION_MODEL, note
                        )
                        for note in selected_completion
                    }
                    cached_rows: dict[str, dict] = {}
                    if cache is not None:
                        by_key = cache.cached_judgments(sorted(set(keys.values())))
                        cached_rows = {
                            note_id: by_key[key]
                            for note_id, key in keys.items()
                            if key in by_key
                        }
                    pending = tuple(
                        note
                        for note in selected_completion
                        if note.note_id not in cached_rows
                    )
                    fresh: dict[str, Any] = {
                        "facts": [],
                        "aliases": [],
                        "completed_note_ids": [],
                    }
                    try:
                        if pending:
                            raw = self.dependencies.relation_completer(pending)
                            if isinstance(raw, dict):
                                fresh = {
                                    "facts": list(raw.get("facts") or []),
                                    "aliases": list(raw.get("aliases") or []),
                                    "completed_note_ids": list(
                                        raw.get("completed_note_ids") or []
                                    ),
                                }
                        merged_payload = {
                            "facts": [
                                row
                                for rows in cached_rows.values()
                                for row in rows.get("facts", [])
                            ]
                            + fresh["facts"],
                            "aliases": [
                                row
                                for rows in cached_rows.values()
                                for row in rows.get("aliases", [])
                            ]
                            + fresh["aliases"],
                        }
                        completed = parse_completion_response(
                            merged_payload,
                            selected_completion,
                            build_id=build_id,
                        )
                    except Exception as exc:
                        completion_error = f"{type(exc).__name__}: relation completion failed"
                    else:
                        fact_rows.extend(completed.facts)
                        proposals = completed.proposals
                        completion_rejected = completed.rejected
                        completion_aliases = completed.aliases
                        # only notes vouched for by a successful call may store;
                        # failed-batch notes retry next build
                        if cache is not None and fresh["completed_note_ids"]:
                            grouped = group_completion_rows(fresh, pending)
                            completed_ids = {
                                str(note_id)
                                for note_id in fresh["completed_note_ids"]
                            } & set(grouped)
                            cache.store_judgments(
                                [
                                    (keys[note_id], "completion", grouped[note_id])
                                    for note_id in sorted(completed_ids)
                                ]
                            )
            for proposal in {row.proposal_id: row for row in proposals}.values():
                store.insert_predicate_proposal(proposal)
            for alias in completion_aliases:
                store.insert_alias(alias.entity_id, alias.alias, alias.origin, 1.0)
            stage_done("relation_completion")
            announce("facts")
            # The same relation can arrive twice (extractor repeats, or the
            # LLM completion re-derives an extracted fact — extraction_method
            # is not part of the identity), so dedupe before reconciling.
            unique_facts = {fact.fact_id: fact for fact in fact_rows}
            if len(unique_facts) != len(fact_rows):
                _LOG.info(
                    "build %s: dropped %d duplicate facts",
                    build_id,
                    len(fact_rows) - len(unique_facts),
                )
            fact_rows = list(reconcile_temporal_facts(tuple(unique_facts.values())))
            stage_done("facts")
            announce("adjudication")
            # item 1: every candidate/active fact is judged against its own
            # source passage text (quote-verified in code). Rejected rows stay
            # for audit but are excluded from edges, embeddings, recall, and
            # serving; mid-band scores multiply the fact's edge weight.
            fact_adjudication = adjudicate_facts(
                fact_rows,
                passages_by_note=passage_by_note,
                note_titles={note.note_id: note.title for note in ordered},
                judge=self.dependencies.edge_adjudicator,
                cache=self.dependencies.adjudication_cache,
                note_sources={note.note_id: note.source for note in ordered},
            )
            fact_rows = list(fact_adjudication.facts)
            if fact_adjudication.adjudicated or fact_adjudication.failed_batches:
                _LOG.info(
                    "build %s: %d facts adjudicated (%d rejected, %d relabeled), "
                    "%d batches failed",
                    build_id,
                    fact_adjudication.adjudicated,
                    sum(fact_adjudication.rejected_by_predicate.values()),
                    sum(fact_adjudication.relabeled_by_predicate.values()),
                    fact_adjudication.failed_batches,
                )
            store.insert_facts(fact_rows)
            stage_done("adjudication")

            announce("base_graph")
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
                # L19: the semantic floor lives in the calibration artifact
                semantic_floor=calibration.graph.semantic_floor,
            ).build(graph_passages, entity_graph, contains_graph, fact_rows)
            # item 5b: cross-note semantic edges must carry a concrete
            # grounded reason or be dropped; same-note edges and fail-open
            # batches keep an empty reason. Dropped edges are filtered from
            # the in-memory graph BEFORE persistence so the stored rows keep
            # reproducing the graph hash.
            note_of_passage = {
                draft.passage_id: note.note_id for note, draft in passage_rows
            }
            draft_of_passage = {
                draft.passage_id: draft for _note, draft in passage_rows
            }
            title_of_note = {note.note_id: note.title for note in ordered}
            cross_note_semantic = [
                edge
                for edge in base_graph.edges
                if edge.kind == "semantic"
                and edge.src_id < edge.dst_id
                and note_of_passage[edge.src_id] != note_of_passage[edge.dst_id]
            ]
            # Only judge the strongest cross-note links (deterministic: weight
            # desc, then passage ids). The long tail keeps an empty reason and
            # is never dropped — this caps judge calls without losing edges.
            reason_cap = int(config.SEMANTIC_REASON_MAX_EDGES)
            if reason_cap > 0 and len(cross_note_semantic) > reason_cap:
                cross_note_semantic.sort(
                    key=lambda edge: (-edge.weight, edge.src_id, edge.dst_id)
                )
                cross_note_semantic = cross_note_semantic[:reason_cap]
            semantic_pairs = tuple(
                SemanticPair(
                    src_passage_id=edge.src_id,
                    dst_passage_id=edge.dst_id,
                    src_title=title_of_note[note_of_passage[edge.src_id]],
                    dst_title=title_of_note[note_of_passage[edge.dst_id]],
                    src_text=draft_of_passage[edge.src_id].text,
                    dst_text=draft_of_passage[edge.dst_id].text,
                    src_content_hash=draft_of_passage[edge.src_id].content_hash,
                    dst_content_hash=draft_of_passage[edge.dst_id].content_hash,
                )
                for edge in cross_note_semantic
            )
            semantic_reasons = adjudicate_semantic_edges(
                semantic_pairs,
                judge=self.dependencies.edge_adjudicator,
                cache=self.dependencies.adjudication_cache,
            )
            if semantic_reasons.dropped:
                _LOG.info(
                    "build %s: %d cross-note semantic edges dropped (no concrete "
                    "reason), %d reasoned",
                    build_id,
                    semantic_reasons.dropped_count,
                    semantic_reasons.reasoned_count,
                )
                base_graph = materialize_base_graph(
                    build_id=build_id,
                    node_ids=base_graph.node_ids,
                    node_kinds=base_graph.node_kinds,
                    edges=tuple(
                        edge
                        for edge in base_graph.edges
                        if edge.kind != "semantic"
                        or tuple(sorted((edge.src_id, edge.dst_id)))
                        not in semantic_reasons.dropped
                    ),
                )
            semantic_edge_rows = tuple(
                (
                    edge.src_id,
                    edge.dst_id,
                    edge.weight,
                    edge.weight,
                    semantic_reasons.reasons.get((edge.src_id, edge.dst_id), ""),
                )
                for edge in base_graph.edges
                if edge.kind == "semantic" and edge.src_id < edge.dst_id
            )
            store.write_semantic_edges(semantic_edge_rows, build_id=build_id)
            contains_by_pair = {
                (row.passage_id, row.entity_id): row for row in contains_graph
            }
            contains_edge_rows = tuple(
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
            )
            store.write_passage_entity_edges(contains_edge_rows, build_id=build_id)
            contains_weight_stats = _contains_weight_stats(
                [float(row[5]) for row in contains_edge_rows]
            )
            if _degenerate_contains_weights([float(row[5]) for row in contains_edge_rows]):
                health_reasons.append("degenerate_contains_weights")
            stage_done("base_graph")
            announce("projection")
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
            announce("ranks")
            quality_ranks, value_scores, near_duplicates = _compute_note_ranks(
                ordered,
                projection,
                now=self.dependencies.now(),
                salience=note_salience,
            )
            store.set_note_ranks(
                quality_ranks,
                value_scores,
                build_id=build_id,
            )
            stage_done("ranks")
            announce("clusters")
            # Membership and names are deterministic presentation metadata;
            # cluster tables stay OUT of the graph hash. Graph-hash inputs are
            # unchanged by this stage.
            cluster_assignments = cluster_projection(projection)
            cluster_membership = {
                note.note_id: int(cluster_assignments.get(note.note_id, -1))
                for note in ordered
            }
            eligible_entity_labels = {
                entity.entity_id: entity.canonical
                for entity in resolution.entities
                if entity.traversal_eligible
            }
            note_entity_labels: dict[str, list[str]] = {}
            for row in resolution.mentions:
                label = eligible_entity_labels.get(row.entity_id)
                if label is not None:
                    note_entity_labels.setdefault(row.note_id, []).append(label)
            stored_notes = store.source_notes()
            cluster_summaries = build_cluster_summaries(
                cluster_membership,
                titles={note.note_id: note.title for note in stored_notes},
                display_lines={
                    note.note_id: note.display_line for note in stored_notes
                },
                sources={note.note_id: note.source for note in stored_notes},
                value_scores=value_scores,
                note_entities=note_entity_labels,
            )
            cluster_records = []
            for summary in cluster_summaries:
                name, description = fallback_label(summary)
                cluster_records.append(
                    ClusterRecord(
                        cluster_id=summary.cluster_id,
                        size=summary.size,
                        name=name,
                        description=description,
                        label_source="fallback",
                        top_entities=summary.top_entities,
                    )
                )
            store.write_clusters(cluster_records, cluster_membership, build_id=build_id)
            stage_done("clusters")
            announce("fact_embeddings")
            fact_embeddings = embed_facts(
                store.fact_contexts(),
                embedder=self.dependencies.embedder,
                model_id=self.dependencies.embedding_model_id,
                build_id=build_id,
                dim=self.dependencies.embedding_dim,
            )
            store.write_fact_embeddings(fact_embeddings)
            stage_done("fact_embeddings")
            announce("validation")
            integrity = store.validate_invariants()
            if not integrity.ok:
                raise BuildError(f"knowledge invariant failure: {integrity}")
            # L24: every non-dangling column of the materialized matrix must
            # be stochastic before the build is declared complete.
            _assert_column_stochastic(base_graph)
            counts = store.counts()
            _assert_stored_counts(
                counts,
                {
                    "notes": len(ordered),
                    "passages": len(passage_rows),
                    "passage_embeddings": len(embedded_by_id),
                    "entities": len(resolution.entities),
                    "entity_mentions": len(resolution.mentions),
                    "facts": len(fact_rows),
                    "fact_embeddings": len(fact_embeddings),
                    "passage_semantic_edges": len(semantic_edge_rows),
                    "passage_entity_edges": len(contains_edge_rows),
                    "note_projection_edges": len(projection),
                },
            )
            graph_hash = compute_graph_hash(base_graph, projection)
            # L24: the stored rows must reproduce the in-memory graph exactly.
            recomputed_hash = store.recompute_graph_hash(build_id)
            if recomputed_hash != graph_hash:
                raise BuildError(
                    "stored graph rows do not reproduce the in-memory graph hash: "
                    f"{recomputed_hash} != {graph_hash}"
                )
            stage_done("validation")
            component_count = _component_count(base_graph)
            hub_exclusions = tuple(
                (row.entity_id, row.traversal_reason)
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
                for note in extractable
                if (row := extractions.get(note.note_id)) is None or row.failed
            )
            release_blocker_rows = _calibration_release_blockers(calibration)
            if (
                extractable
                and self.dependencies.relation_completer is None
                and not relation_completion_opt_out
            ):
                release_blocker_rows.append("relation_completion_unavailable")
            if completion_error:
                release_blocker_rows.append("relation_completion_failed")
            release_blockers = tuple(sorted(release_blocker_rows))
            health_reasons = sorted(dict.fromkeys(health_reasons))
            health = (
                "healthy"
                if self.dependencies.extractor is not None
                and not failed_notes
                and not release_blockers
                and not health_reasons
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
            _LOG.info(
                "build %s complete: %d notes, %d passages, %d entities, %d facts, "
                "%d extraction failures, %d degenerate facts skipped, health=%s",
                build_id,
                len(ordered),
                len(passage_rows),
                len(resolution.entities),
                len(fact_rows),
                len(failed_notes),
                len(degenerate_facts),
                health,
            )
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
                degenerate_fact_count=len(degenerate_facts),
                degenerate_facts=tuple(degenerate_facts),
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
                degraded_reasons=tuple(health_reasons),
                straddling_span_count=straddling_spans,
                contains_weight_stats=contains_weight_stats,
                salience_stats=_salience_stats(note_salience),
                salience_gated_count=len(gated_note_ids),
                salience_gated_by_source=_gated_by_source(ordered, gated_note_ids),
                near_duplicate_note_pairs=near_duplicates,
                cluster_count=len(cluster_records),
                cluster_size_histogram=_cluster_size_histogram(
                    [record.size for record in cluster_records]
                ),
                context_enriched_count=enrichment.enriched_count,
                context_enrich_failed_batches=enrichment.failed_batches,
                adjudicated_fact_count=fact_adjudication.adjudicated,
                unadjudicated_fact_count=fact_adjudication.unadjudicated,
                adjudication_failed_batches=fact_adjudication.failed_batches,
                adjudication_malformed_rows=fact_adjudication.malformed_rows,
                adjudicated_by_predicate=fact_adjudication.adjudicated_by_predicate,
                adjudication_kept_by_predicate=fact_adjudication.kept_by_predicate,
                adjudication_downweighted_by_predicate=(
                    fact_adjudication.downweighted_by_predicate
                ),
                adjudication_rejected_by_predicate=(
                    fact_adjudication.rejected_by_predicate
                ),
                adjudication_relabeled_by_predicate=(
                    fact_adjudication.relabeled_by_predicate
                ),
                semantic_reasoned_edge_count=semantic_reasons.reasoned_count,
                semantic_reason_dropped_count=semantic_reasons.dropped_count,
                semantic_reason_failed_batches=semantic_reasons.failed_batches,
                merge_gray_zone_pair_count=gray_zone_pair_count,
                merge_adjudicated_count=merge_outcome.judged_count,
                merge_approved_count=merge_outcome.approved_count,
                merge_failed_batches=merge_outcome.failed_batches,
                peak_memory_gb=round(watchdog.peak_gb, 3),
                peak_memory_stage=watchdog.peak_stage,
            )
        except Exception as exc:
            if store is not None:
                store.close()
            if temporary.exists():
                # TAURINE_KEEP_FAILED_BUILD=1 keeps the partial build on disk
                # for post-mortems; cancellations always clean up.
                if _keep_failed_builds() and not isinstance(exc, BuildCancelled):
                    failed_path = temporary.with_name(temporary.name + ".failed")
                    if failed_path.exists():
                        failed_path.unlink()
                    os.replace(temporary, failed_path)
                    _LOG.warning("build failed; partial build kept at %s", failed_path)
                else:
                    temporary.unlink()
            if isinstance(exc, BuildCancelled):
                _LOG.info("build cancelled")
                raise
            _LOG.exception("build failed")
            if isinstance(exc, BuildError):
                raise
            raise BuildError(str(exc)) from exc
        finally:
            watchdog.__exit__(None, None, None)


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: BuildDependencies | None = None,
    connector_loader: Callable[[], Sequence[NoteRecord]] | None = None,
) -> int:
    """§22 operator contract. Exit codes: 0 healthy install, 1 user abort,
    2 invariant/build failure (nothing installed), 3 installed but release
    blockers remain (suppressed by --install-degraded)."""

    parser = argparse.ArgumentParser(description="Build a clean Taurine knowledge index")
    parser.add_argument("--db", required=True)
    parser.add_argument("--from-enabled-connectors", action="store_true")
    parser.add_argument("--replace-derived", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--install-degraded",
        action="store_true",
        help="exit 0 even when release blockers remain (build still installs by default)",
    )
    parser.add_argument(
        "--no-relation-completion",
        action="store_true",
        help="skip the LLM relation-completion lane intentionally (warning, not a blocker)",
    )
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
            return connector_registry.extract_all_enabled(source_index).records
    cli_llm = None if dependencies is not None else _cli_llm_client()
    deps = dependencies or BuildDependencies(
        tokenizer=passage_tokenizer(),
        embedder=embed_texts,
        extractor=extract_notes,
        relation_completer=(
            None
            if args.no_relation_completion or cli_llm is None
            else relation_completer_for(cli_llm)
        ),
        context_enricher=None if cli_llm is None else context_enricher_for(cli_llm),
        edge_adjudicator=None if cli_llm is None else edge_adjudicator_for(cli_llm),
    )
    sweep_stale_build_files(args.db)
    if args.no_relation_completion:
        print("relation completion skipped by --no-relation-completion", file=sys.stderr)
    try:
        report = BuildCoordinator(deps).build(
            connector_loader(),
            args.db,
            replace_derived=args.replace_derived,
            relation_completion_opt_out=args.no_relation_completion,
        )
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({**report.__dict__, "target": str(report.target)}, sort_keys=True))
    if not report.release_ready:
        blockers = "\n".join(f"- {blocker}" for blocker in report.release_blockers)
        print(
            "installed, but the build is not release-ready:\n"
            + (blockers or "- degraded health"),
            file=sys.stderr,
        )
        return 0 if args.install_degraded else 3
    return 0


def _cli_llm_client() -> Any | None:
    """Shared OpenRouter client for the CLI's optional LLM lanes."""

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass
    key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        return None
    from taurine_core.llm import OpenRouterClient

    return OpenRouterClient(key)


def _cluster_size_histogram(sizes: Sequence[int]) -> dict[str, int]:
    """Bucketed size distribution for the build report."""

    buckets = {"1-2": 0, "3-9": 0, "10-19": 0, "20-35": 0, "36+": 0}
    for size in sizes:
        if size <= 2:
            buckets["1-2"] += 1
        elif size <= 9:
            buckets["3-9"] += 1
        elif size <= 19:
            buckets["10-19"] += 1
        elif size <= 35:
            buckets["20-35"] += 1
        else:
            buckets["36+"] += 1
    return buckets


def _keep_failed_builds() -> bool:
    return os.environ.get("TAURINE_KEEP_FAILED_BUILD", "").strip().lower() in {"1", "true", "yes"}


def _accepts_kwarg(func: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _passage_for_span(
    drafts: Sequence[Any],
    char_start: int,
    char_end: int,
) -> tuple[Any, bool]:
    """Locate the passage holding a span. Spec §6.4: a span may also straddle
    two ADJACENT overlapping passages whose union contains it — recorded
    against the earlier passage (L5). Returns (passage | None, straddled)."""
    for passage in drafts:
        if passage.char_start <= char_start and char_end <= passage.char_end:
            return passage, False
    ordered = sorted(drafts, key=lambda passage: passage.ordinal)
    for earlier, later in zip(ordered, ordered[1:]):
        if (
            later.ordinal == earlier.ordinal + 1
            and later.char_start <= earlier.char_end  # contiguous union
            and earlier.char_start <= char_start
            and char_end <= later.char_end
        ):
            return earlier, True
    return None, False


def _resolve_endpoint(
    resolved: dict[tuple[str, int, int, str], list[Any]],
    note_id: str,
    char_start: int,
    char_end: int,
    surface: str,
) -> Any | None:
    """Pick the resolved mention for a relation endpoint. GLiNER relations do
    not declare endpoint types, so a same-span etype collision binds
    deterministically: named types outrank the catch-all concept, then
    alphabetical order (L6)."""
    candidates = resolved.get((note_id, char_start, char_end, surface.casefold()))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (row.mention.etype == "concept", row.mention.etype),
    )


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
    salience: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], tuple[tuple[str, str], ...]]:
    note_ids = tuple(note.note_id for note in notes)
    if not note_ids:
        return {}, {}, ()
    note_by_id = {note.note_id: note for note in notes}
    row_by_id = {note_id: index for index, note_id in enumerate(note_ids)}
    matrix = np.zeros((len(note_ids), len(note_ids)), dtype=np.float64)
    near_duplicates: list[tuple[str, str]] = []
    for edge in projection:
        left = row_by_id.get(edge.src_note_id)
        right = row_by_id.get(edge.dst_note_id)
        if left is None or right is None or left == right:
            continue
        # M14/§10.4: same-source near-duplicates form mutually-reinforcing
        # cliques; their edges stay in the projection for display but are
        # excluded from the PageRank matrix and flagged in the build report.
        if (
            note_by_id[edge.src_note_id].source == note_by_id[edge.dst_note_id].source
            and float(getattr(edge, "mechanisms", {}).get("semantic", 0.0))
            >= config.DUP_EDGE_SIM
        ):
            near_duplicates.append((edge.src_note_id, edge.dst_note_id))
            continue
        weight = max(0.0, float(edge.weight))
        matrix[right, left] += weight
        matrix[left, right] += weight
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
        # §0.2: value multiplies in salience (restores the legacy
        # ingest_salience behavior) so the §6.6 completion gate is meaningful
        note_salience = 1.0 if salience is None else float(salience.get(note_id, 1.0))
        values[note_id] = (
            quality_weight
            * float(0.5 ** (age_days / config.VALUE_RECENCY_HALF_LIFE_DAYS))
            * source_rank(note.source)
            * note_salience
        )
    return quality, values, tuple(sorted(near_duplicates))


def _contains_weight_stats(weights: Sequence[float]) -> dict[str, float]:
    """Distribution summary of contains weights for the build report (H2)."""

    if not weights:
        return {"count": 0.0, "min": 0.0, "median": 0.0, "max": 0.0, "stddev": 0.0}
    ordered = sorted(weights)
    return {
        "count": float(len(ordered)),
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "max": float(ordered[-1]),
        "stddev": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def _degenerate_contains_weights(
    weights: Sequence[float],
    *,
    fraction: float = 0.8,
    tolerance: float = 1e-6,
    minimum_edges: int = 8,
) -> bool:
    """H2 health gate: >80% of contains weights collapsed onto one value means
    the idf/scale formula stopped discriminating (the pre-fix saturation bug).
    Tiny graphs are exempt — coincidental equality is expected there."""

    if len(weights) < minimum_edges:
        return False
    anchor = statistics.median(weights)
    close = sum(1 for weight in weights if abs(weight - anchor) <= tolerance)
    return close / len(weights) > fraction


def _assert_column_stochastic(graph: Any, *, tolerance: float = 1e-8) -> None:
    """L24 invariant: every non-dangling column of the materialized matrix
    sums to exactly one (within tolerance) before the build completes."""

    strengths = np.asarray(graph.matrix.sum(axis=0)).ravel()
    dangling = set(graph.dangling_rows)
    bad = [
        index
        for index, value in enumerate(strengths)
        if index not in dangling and abs(float(value) - 1.0) > tolerance
    ]
    if bad:
        raise BuildError(
            f"base graph has {len(bad)} non-stochastic columns (first: {bad[:5]})"
        )


def _assert_stored_counts(stored: Mapping[str, int], expected: Mapping[str, int]) -> None:
    """L24 invariant: stored row counts equal the in-memory build quantities."""

    mismatched = {
        name: (int(stored.get(name, -1)), int(count))
        for name, count in sorted(expected.items())
        if int(stored.get(name, -1)) != int(count)
    }
    if mismatched:
        raise BuildError(f"stored counts diverge from in-memory build: {mismatched}")


def _salience_stats(note_salience: Mapping[str, float]) -> dict[str, float]:
    values = sorted(note_salience.values())
    if not values:
        return {"count": 0.0, "min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "min": float(values[0]),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(values[-1]),
    }


def _gated_by_source(
    notes: Sequence[NoteRecord], gated_note_ids: set[str]
) -> dict[str, int]:
    gated: dict[str, int] = {}
    for note in notes:
        if note.note_id in gated_note_ids:
            gated[note.source] = gated.get(note.source, 0) + 1
    return dict(sorted(gated.items()))


def _model_fingerprint(repo_id: str) -> str:
    """Local artifact fingerprint (L26): size+mtime digest over the bundled
    model directory. 'unavailable' when the model is not downloaded."""

    root = local_model_path(repo_id)
    if root is None:
        return "unavailable"
    entries = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        try:
            info = path.stat()
        except OSError:
            continue
        entries.append((path.relative_to(root).as_posix(), info.st_size, int(info.st_mtime)))
    digest = hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def sweep_stale_build_files(
    target: str | Path,
    *,
    max_age_s: float = 86_400.0,
    now: float | None = None,
) -> tuple[Path, ...]:
    """Delete crash-orphaned ``<stem>.build-*.db`` temp files older than a day
    that are not the active database (L25). ``.failed`` post-mortem copies are
    deliberately kept."""

    target_path = Path(target)
    stamp = time.time() if now is None else float(now)
    suffix = target_path.suffix or ".db"
    removed: list[Path] = []
    for candidate in sorted(target_path.parent.glob(f"{target_path.stem}.build-*{suffix}")):
        if candidate == target_path:
            continue
        try:
            if stamp - candidate.stat().st_mtime > max_age_s:
                candidate.unlink()
                removed.append(candidate)
        except OSError:
            continue
    if removed:
        _LOG.info("swept %d stale build temp files", len(removed))
    return tuple(removed)


def carry_forward_records(
    knowledge_database: str | Path,
    sources: Sequence[str],
) -> tuple[NoteRecord, ...]:
    """Read one source's notes back out of the ACTIVE knowledge DB (L28).

    Used when an enabled connector is unavailable during a sync so a lost
    permission degrades the build instead of silently wiping the source.
    """

    wanted = set(sources)
    if not wanted:
        return ()
    path = Path(knowledge_database)
    if not path.exists():
        return ()
    try:
        store = KnowledgeStore(path, read_only=True)
    except Exception:
        return ()
    try:
        return tuple(
            NoteRecord(
                note_id=row.note_id,
                title=row.title,
                folder=row.folder,
                modified=row.modified,
                created=row.created,
                content_hash=row.content_hash,
                text=row.text,
                source=row.source,
                ingest_salience=row.ingest_salience,
            )
            for row in store.source_notes()
            if row.source in wanted
        )
    finally:
        store.close()


def relation_completer_for(llm: Any) -> Callable[[Sequence[CompletionNote]], object]:
    """Bounded LLM relation-completion callable shared by the server and CLI."""

    def complete_relations(notes: Sequence[CompletionNote]) -> object:
        # One call per bounded batch: the previous single-call design sent the
        # whole selection (100+ notes, ~0.5 MB) and overflowed the model's
        # context, so the lane silently produced nothing on real corpora.
        predicates = {
            name: spec.description for name, spec in sorted(PREDICATES.items())
        }
        batch_size = max(1, int(config.COMPLETION_NOTES_PER_CALL))
        batches = [
            notes[start : start + batch_size]
            for start in range(0, len(notes), batch_size)
        ]

        async def run_batches() -> dict[str, Any]:
            semaphore = asyncio.Semaphore(max(1, int(config.COMPLETION_CONCURRENCY)))

            async def one(batch: Sequence[CompletionNote]) -> dict[str, Any] | None:
                async with semaphore:
                    try:
                        return await llm.complete_json(
                            config.COMPLETION_MODEL,
                            COMPLETION_SYSTEM_PROMPT,
                            json.dumps(
                                {
                                    "predicates": predicates,
                                    "notes": [
                                        completion_note_payload(note) for note in batch
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            timeout_s=config.SUMMARY_TIMEOUT_S,
                            # deterministic completions keep fact identities
                            # stable across rebuilds so adjudication cache hits
                            temperature=0.0,
                            max_tokens=config.COMPLETION_MAX_TOKENS,
                        )
                    except Exception as exc:  # noqa: BLE001 — per-batch fail-open
                        _LOG.warning(
                            "relation completion batch failed (%s): %s",
                            type(exc).__name__,
                            exc,
                        )
                        return None

            results = await asyncio.gather(*(one(batch) for batch in batches))
            merged: dict[str, Any] = {
                "facts": [],
                "aliases": [],
                "completed_note_ids": [],
            }
            for batch, result in zip(batches, results):
                if not isinstance(result, dict):
                    continue
                # a successful call vouches for every note in its batch — including
                # notes the model returned nothing for (cached as empty, asked once)
                merged["completed_note_ids"].extend(note.note_id for note in batch)
                for key in ("facts", "aliases"):
                    rows = result.get(key)
                    if isinstance(rows, list):
                        merged[key].extend(rows)
            return merged

        return asyncio.run(run_batches())

    return complete_relations


def _component_count(graph: Any) -> int:
    if not graph.node_ids:
        return 0
    undirected = graph.matrix.maximum(graph.matrix.transpose())
    count, _labels = connected_components(undirected, directed=False)
    return int(count)


if __name__ == "__main__":
    raise SystemExit(main())
