# Knowledge Graph and Retrieval Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Taurine's overlapping note projection/retrieval graph with the approved evidence-first HippoRAG 2 architecture, including passage evidence, directed temporal facts, calibrated abstention, shared live/chat retrieval, and evidence-valid citations.

**Architecture:** Build an immutable disposable SQLite knowledge snapshot from connector source records, with passage/entity nodes and directed facts in one uncapped retrieval base graph plus a separate degree-capped note projection. A shared retrieval engine calibrates dense/BM25/exact/entity/triple heads, creates passage/entity PPR resets, reranks and diversifies evidence, and returns citeable evidence packs consumed by live and chat policy adapters.

**Tech Stack:** Python 3.12, SQLite/FTS5, NumPy/SciPy, GLiNER2 1.3.2, Qwen3 Embedding/Reranker, DeBERTa-v3 NLI, FastAPI/WebSockets, Pydantic 2, pytest, React 19/TypeScript/Electron.

**Specs:** `docs/superpowers/specs/2026-07-09-knowledge-graph-retrieval-overhaul-design.md`; rationale in `docs/audits/2026-07-09-knowledge-graph-retrieval-review.md`.

## Global Constraints

- No migration, backfill, dual-read, schema compatibility, or legacy graph fallback. The derived knowledge DB is deleted and regenerated.
- Preserve connector source data, chat history, authentication/configuration, and session metadata outside `taurine-knowledge.db`.
- Build `taurine-knowledge.build-<build_id>.db`, validate it, then atomically install and activate one immutable `IndexSnapshot`.
- Retrieval never imports or queries `note_projection_edges`.
- Production code follows red-green-refactor: add one behavior test, observe the expected failure, implement, and rerun focused plus full suites.
- Tests use deterministic model fakes unless marked integration; no unit test downloads or loads a large model.
- Every stored mention/fact retains exact note-global source offsets and build/schema/model identity.
- Every ranking head can reject all candidates; an abstained `EvidencePack` is empty.
- Factual output requires supplied `E#`, `F#`, or `T#` support; `N#` alone is navigation.
- New thresholds live in `core/evals/graph_retrieval/calibration-v1.json`, never duplicated as ranking constants.
- Commit after each task only after the focused tests and `cd core && uv run pytest -q` pass.

---

### Task 1: Evaluation contracts and initial calibration artifact

**Files:**

- Create: `core/taurine_core/evaluation/__init__.py`
- Create: `core/taurine_core/evaluation/graph_retrieval.py`
- Create: `core/evals/graph_retrieval/{extraction,retrieval_live,retrieval_chat,citations,entity_resolution}.jsonl`
- Create: `core/evals/graph_retrieval/calibration-v1.json`
- Create: `core/tests/test_graph_retrieval_eval.py`

**Interfaces:**

- Produces `EvaluationExample`, `RetrievalMetrics`, `load_jsonl(path)`, `recall_at`, `ndcg_at`, `mrr`, `support_set_recall`, `citation_coverage`, and `validate_split_isolation`.
- Calibration JSON contains `artifact_version`, model/config hashes, per-mode/per-head curves and floors, triple/rerank/NLI floors, abstention rules, sample counts, dates, and held-out metrics.

- [ ] Write failing tests proving metric values on hand-calculated rankings, invalid JSONL rejection, source-note split leakage detection, and calibration schema validation.
- [ ] Run `cd core && uv run pytest tests/test_graph_retrieval_eval.py -q`; expect import/file failures.
- [ ] Implement typed loaders, metric functions, privacy-safe deterministic seed datasets covering positive, multi-hop, temporal, null, citation, and entity-resolution cases, and the conservative initial artifact.
- [ ] Run the focused test and full Python suite.
- [ ] Commit `test: establish graph retrieval evaluation and calibration contracts`.

### Task 2: Retrieval/evidence types and configuration

**Files:**

- Create: `core/taurine_core/retrieval/__init__.py`
- Create: `core/taurine_core/retrieval/types.py`
- Modify: `core/taurine_core/config.py`
- Modify: `core/taurine_core/schemas.py`
- Create: `core/tests/test_retrieval_types.py`

**Interfaces:**

- Frozen dataclasses: `TemporalIntent`, `RetrievalQuery`, `HeadHit`, `RetrievalDecision`, `EvidencePassage`, `EvidenceFact`, `EvidencePath`, `EvidenceNote`, `EvidencePack`, `RetrievalTrace`.
- Public Pydantic event models mirror evidence payloads without private full-text debug fields.
- Add the exact §16 constants, rename `GRAPH_MAX_DEGREE` to `PROJECTION_MAX_DEGREE`, and remove active use of additive fact/PPR blend knobs only as callers migrate.

- [ ] Write failing serialization, immutability, abstained-pack, and event round-trip tests.
- [ ] Run focused tests and confirm missing types/constants failures.
- [ ] Implement dataclasses/config/public schemas with explicit defaults and tuple factories.
- [ ] Run focused and full suites.
- [ ] Commit `feat: define shared retrieval and evidence contracts`.

### Task 3: Token-aware passage construction

**Files:**

- Create: `core/taurine_core/indexer/passages.py`
- Modify: `core/taurine_core/models_local.py`
- Create: `core/tests/test_index_passages.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PassageDraft:
    passage_id: str
    ordinal: int
    char_start: int
    char_end: int
    token_count: int
    text: str
    content_hash: str

def build_passages(note: NoteRecord, tokenizer: TokenizerPort, *, version: str) -> tuple[PassageDraft, ...]: ...
```

- Use target/max/overlap 220/300/40, paragraph→sentence→clause boundary preference, exact source slicing, deterministic IDs, and title only as embedding metadata.

- [ ] Write failing tests for boundaries, overlong sentences, Unicode offsets, overlap, empty text, source reconstruction, and deterministic IDs.
- [ ] Observe focused failures.
- [ ] Implement segmentation against an injected tokenizer and expose a local Qwen tokenizer loader seam.
- [ ] Run focused and full suites.
- [ ] Commit `feat: add deterministic token-aware source passages`.

### Task 4: Disposable knowledge store and immutable snapshot

**Files:**

- Create: `core/taurine_core/indexer/knowledge_store.py`
- Create: `core/taurine_core/retrieval/snapshot.py`
- Modify: `core/taurine_core/indexer/store.py`
- Create: `core/tests/test_knowledge_store.py`
- Create: `core/tests/test_retrieval_snapshot.py`

**Interfaces:**

- Implement §4 tables exactly: notes, passages/FTS, passage embeddings, entities, aliases, mentions, facts, predicate proposals, fact embeddings, semantic/contains edges, projection edges, and index builds.
- `KnowledgeStore` exposes typed writes/reads and invariant validation; `IndexSnapshot` atomically contains passage/fact matrices, FTS connection, aliases, base graph, projection, ranks, build/calibration hashes.
- Existing `Index` continues to own non-derived chat/connector/session state and delegates active knowledge reads through a snapshot port.

- [ ] Write failing DDL/foreign-key/cascade/read-only/invariant/atomic-reference tests.
- [ ] Observe expected missing-schema failures.
- [ ] Implement fresh-schema creation only; do not add `ALTER TABLE` or old-row adapters.
- [ ] Run focused and full suites.
- [ ] Commit `feat: add disposable knowledge store and immutable snapshots`.

### Task 5: Passage dense and lexical indexes

**Files:**

- Modify: `core/taurine_core/indexer/embed.py`
- Modify: `core/taurine_core/indexer/knowledge_store.py`
- Create: `core/tests/test_passage_index.py`

**Interfaces:**

- `embed_passages(drafts, embedder) -> tuple[PassageEmbedding, ...]` stores normalized 1024-d vectors by passage ID/model hash.
- `KnowledgeStore.search_passages_bm25(query, k)` returns passage IDs with raw FTS5 BM25 and token coverage.
- Passage embedding text is deterministic `title + source metadata + exact passage`, while stored source span stays unchanged.

- [ ] Write failing tests for normalization, deterministic ordering, invalid dimensions, title metadata, passage-level BM25, and deletion cascades.
- [ ] Observe focused failures.
- [ ] Implement index writes/load/search with injected embedding seams.
- [ ] Run focused and full suites.
- [ ] Commit `feat: index source passages for dense and lexical retrieval`.

### Task 6: GLiNER2 batch/long extraction with provenance

**Files:**

- Rewrite: `core/taurine_core/indexer/extract.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/live/spans.py`
- Rewrite: `core/tests/test_index_extract.py`

**Interfaces:**

- `extract_notes(notes, tokenizer) -> dict[note_id, ExtractionResult]` selects source-family schema, short `batch_extract`, or generic `batch_extract_long(chunk_size=384, chunk_overlap=64, include_confidence=True, include_spans=True)`.
- `ExtractedMention` and `ExtractedRelation` include global source offsets, assertion offsets, confidence, schema ID, and method.
- Per-type extraction/promotion thresholds load from the active calibration artifact.

- [ ] Replace manual-chunk tests with failing short/long batch API, global span, token-budget, source-schema, threshold, ambiguous-endpoint, and per-note failure tests.
- [ ] Observe failures specifically because manual `_text_chunks` cannot satisfy offsets/API calls.
- [ ] Implement one cached shared model adapter and strict span/endpoint validation; delete manual whitespace chunk helpers.
- [ ] Run focused and full suites.
- [ ] Commit `feat: use GLiNER2 batch long extraction with exact provenance`.

### Task 7: Entity resolution and hub eligibility

**Files:**

- Create: `core/taurine_core/indexer/entities.py`
- Modify: `core/taurine_core/indexer/knowledge_store.py`
- Create: `core/tests/test_entities.py`

**Interfaces:**

```python
class EntityResolver:
    def resolve_mentions(self, mentions: Sequence[ExtractedMention]) -> ResolutionResult: ...
    def lookup_query_surface(self, surface: str, etype: str | None, context: str) -> EntityResolution: ...
```

- Ordered stages: exact, unambiguous casefold alias, normalization, acronym, embedding candidate, lexical/digit/type/context guards, bounded ambiguity lane.
- Canonicalization never rewrites source mention surfaces; df and traversal eligibility are deterministically recomputed.

- [ ] Write failing tests for exact/casefold/acronym resolution, ambiguity, digit/type/non-merge guards, ASR lowercase names, aliases under three chars, and hub/date activation.
- [ ] Observe missing resolver failures.
- [ ] Implement stable IDs, alias origins/confidence, candidate scoring, and query-specific hub overlay decisions.
- [ ] Run focused and full suites.
- [ ] Commit `feat: add provenance-preserving ambiguity-aware entity resolution`.

### Task 8: Predicate registry, temporal facts, and completion lane

**Files:**

- Create: `core/taurine_core/indexer/facts.py`
- Create: `core/taurine_core/indexer/relation_completion.py`
- Modify: `core/taurine_core/indexer/knowledge_store.py`
- Create: `core/tests/test_facts.py`
- Create: `core/tests/test_relation_completion.py`

**Interfaces:**

- Code-owned `PredicateSpec` registry with all §4.5 predicates, direction, inverse, reverse weight, cardinality, temporal/exclusive flags, and base weight.
- `promote_facts`, `reconcile_temporal_facts`, `facts_for_intent`, `canonical_fact_text` preserve source passage/assertion/endpoint spans and observed/valid/invalidated/superseded status.
- Completion selects top 15% by value with floor five plus sparse/schema-gap notes; registered facts require offsets, open predicates become proposals only.

- [ ] Write failing direction/provenance/promotion/candidate/supersession/conflict/current-history/open-proposal/completion-gate tests.
- [ ] Observe focused failures.
- [ ] Implement deterministic fact IDs, temporal reconciliation, and one batched injected-LLM completion call.
- [ ] Run focused and full suites.
- [ ] Commit `feat: add directed temporal facts and bounded relation completion`.

### Task 9: Base graph and dangling-correct PPR

**Files:**

- Create: `core/taurine_core/indexer/base_graph.py`
- Create: `core/taurine_core/retrieval/ppr.py`
- Modify: `core/taurine_core/indexer/rank.py`
- Create: `core/tests/test_base_graph.py`
- Create: `core/tests/test_retrieval_ppr.py`

**Interfaces:**

- Base nodes are passages plus referenced canonical entities only.
- Contains weights follow §10.2; facts produce directed forward/reverse weights; semantic passage edges use exact cosine top-12 union and exclude same-note adjacent overlap.
- `TypedPPR.run(teleport, temporal_overlay=None, warm_start=None) -> PPRResult` redistributes dangling mass and reports passage/entity mass, iterations, residual, and contributions.

- [ ] Write failing contains/hub/directed-edge/semantic-union/no-projection/dangling-mass/empty-seed/warm-snapshot tests.
- [ ] Observe failures against current note/entity `HeteroGraphMatrix` and leaking PPR mass.
- [ ] Implement deterministic sparse graph materialization and typed teleport PPR; keep projection quality rank separate.
- [ ] Run focused and full suites.
- [ ] Commit `feat: build uncapped passage entity graph with correct PPR`.

### Task 10: Deterministic note projection

**Files:**

- Create: `core/taurine_core/indexer/projection.py`
- Modify: `core/taurine_core/indexer/graph.py`
- Modify: `core/taurine_core/indexer/rank.py`
- Rewrite: `core/tests/test_index_graph.py`
- Rewrite: `core/tests/test_graph_hetero.py`

**Interfaces:**

- `build_projection(store, base_graph) -> tuple[ProjectionEdge, ...]` aggregates semantic/entity/fact mechanisms, builds a maximum-spanning forest, then fills strongest edges to `PROJECTION_MAX_DEGREE` with stable tie breaks.
- `/graph`, previews, and quality PageRank read only projection rows; retrieval has no dependency on this module/table.

- [ ] Write failing deterministic-order, spanning-forest, degree-cap, mechanism, projection-isolation, and full-rebuild-equivalence tests.
- [ ] Confirm current `rebuild_for` drift test fails under required contract.
- [ ] Implement projection and switch UI/quality callers; delete `rebuild_for` and double-counting matrix construction.
- [ ] Run focused and full suites.
- [ ] Commit `feat: isolate deterministic note projection from retrieval`.

### Task 11: Global build coordinator and atomic CLI

**Files:**

- Create: `core/taurine_core/indexer/build.py`
- Modify: `core/taurine_core/indexer/sync.py`
- Modify: `core/taurine_core/server.py`
- Create: `core/tests/test_index_build.py`
- Rewrite: `core/tests/test_index_sync.py`

**Interfaces:**

- `BuildCoordinator.build(records, target, *, replace_derived) -> BuildReport` implements all 17 ordered §9.1 stages, one full finalization, invariant report, graph hash, and atomic `os.replace` activation.
- CLI exactly supports `python -m taurine_core.indexer.build --db PATH --from-enabled-connectors --replace-derived` and confirms target unless `--yes` is provided for automation.
- Failed builds never activate; extraction failure yields searchable passages and no stale semantics.

- [ ] Write failing stage-order, connector-order determinism, extraction-failure, invariant-abort, atomic-swap, active-snapshot, CLI-report, and no-source-mutation tests.
- [ ] Observe missing coordinator/CLI failures.
- [ ] Implement coordinator and make multi-connector/server sync finalize once; remove per-source graph/rank/fact refresh.
- [ ] Run focused and full suites.
- [ ] Commit `feat: add deterministic atomic knowledge build coordinator`.

### Task 12: Calibration, retrieval heads, fusion, and abstention

**Files:**

- Create: `core/taurine_core/retrieval/calibration.py`
- Create: `core/taurine_core/retrieval/heads.py`
- Create: `core/taurine_core/retrieval/evidence.py`
- Create: `core/tests/test_retrieval_calibration.py`
- Create: `core/tests/test_retrieval_heads.py`

**Interfaces:**

- Version/hash-checked monotonic curves produce calibrated probabilities and acceptance floors.
- Heads: dense passage, BM25 passage, exact alias once, query entity, triple hook, optional utterance.
- Weighted score-aware RRF uses accepted calibrated hits only.
- `AbstentionPolicy.pre_ppr` and `.post_rerank` return explicit reasons and empty packs on rejection.

- [ ] Write failing curve interpolation/hash-mismatch/floor/token-coverage/exact-once/fusion/null/pre-post-abstention tests.
- [ ] Confirm filler/gibberish fixtures currently surface arbitrary notes.
- [ ] Implement calibrated heads/fusion/policies without min-max normalization or raw endpoint fact bonuses.
- [ ] Run focused and full suites.
- [ ] Commit `feat: add calibrated shared retrieval heads and abstention`.

### Task 13: Query-to-triple seeding and local reranking

**Files:**

- Create: `core/taurine_core/retrieval/triples.py`
- Create: `core/taurine_core/retrieval/rerank.py`
- Modify: `core/taurine_core/models_local.py`
- Create: `core/tests/test_retrieval_triples.py`
- Create: `core/tests/test_retrieval_rerank.py`

**Interfaces:**

- `TripleRecognizer` retrieves top 20 canonical fact embeddings, locally scores query plus triple/assertion, accepts at most five, and emits subject/object entity reset mass plus accepted fact IDs.
- Live timeout fallback requires stricter embedding floor plus entity overlap; chat ambiguity band may use one batched injected LLM acceptance call.
- `QwenReranker` batches query/passage or query/triple pairs, enforces live/chat candidate/time budgets, and exposes degraded status.

- [ ] Write failing recognition acceptance/rejection/temporal compatibility/no-triple fallback/timeout/batched-rerank tests.
- [ ] Observe missing recognizer failures.
- [ ] Implement lazy local model seams and deterministic fake-friendly scoring contracts.
- [ ] Run focused and full suites.
- [ ] Commit `feat: implement HippoRAG query to triple entity seeding`.

### Task 14: Paths, MMR, evidence packs, and registries

**Files:**

- Create: `core/taurine_core/retrieval/paths.py`
- Create: `core/taurine_core/retrieval/registry.py`
- Modify: `core/taurine_core/retrieval/evidence.py`
- Create: `core/tests/test_retrieval_paths.py`
- Create: `core/tests/test_evidence_registry.py`
- Create: `core/tests/test_evidence_pack.py`

**Interfaces:**

- Paths use negative-log weighted search, maximum four edges/two paths, preserve direction/fact IDs, reject inactive facts, and penalize hubs/repeated notes.
- `EvidenceRegistry` assigns persistent non-reused `N/E/F/P/T` IDs and records retrieved/read/cited states.
- Candidate union → PPR → rerank → MMR with max two passages/note and mode budgets; note scores use strongest passage plus bounded distinct-support bonus.

- [ ] Write failing path direction/provenance/hop/status, registry persistence/non-reuse, MMR/source-cap, support-passage inclusion, and empty-pack tests.
- [ ] Observe failures.
- [ ] Implement path extraction, one registry service, evidence rendering, and pack assembly.
- [ ] Run focused and full suites.
- [ ] Commit `feat: assemble citeable passage fact and path evidence packs`.

### Task 15: Citation, NLI, and temporal validation

**Files:**

- Create: `core/taurine_core/retrieval/validate.py`
- Modify: `core/taurine_core/models_local.py`
- Create: `core/tests/test_evidence_validation.py`

**Interfaces:**

- Stages: ID validity, supplied availability, factual-sentence coverage, fact-to-source provenance, local NLI entailment, temporal consistency.
- `ValidationResult` lists accepted output, unsupported sentences, invalid citations, repair requirement, cited IDs, and degraded reason.
- Missing NLI fails live factual updates closed; chat returns evidence excerpts or one repair round using the same pack.

- [ ] Write failing unknown/unsupplied/zero-citation/N-only/entailed/contradicted/historical-current/nonfactual/code-block tests.
- [ ] Observe missing validator failures.
- [ ] Implement deterministic sentence classification and injected NLI model seam; never silently strip a citation while retaining its claim.
- [ ] Run focused and full suites.
- [ ] Commit `feat: validate citation coverage entailment and temporal support`.

### Task 16: Live retrieval and sequence-safe summaries

**Files:**

- Rewrite: `core/taurine_core/live/ambient.py`
- Rewrite: `core/taurine_core/live/grounding.py`
- Modify: `core/taurine_core/live/ingest.py`
- Modify: `core/taurine_core/live/summarizer.py`
- Modify: `core/taurine_core/server.py`
- Rewrite: `core/tests/test_live_ambient.py`
- Rewrite: `core/tests/test_live_grounding.py`
- Extend: `core/tests/test_live_ingest.py`
- Rewrite: `core/tests/test_live_summarizer.py`

**Interfaces:**

- `AmbientRetriever` is a live policy adapter over the shared engine using 12–45 seconds, 6000 chars, finals for citations, and partials only for recall.
- `GroundingEngine` accumulates accepted passage/note evidence, resets on confirmed topic shift, and preserves hysteresis as display state only.
- `TranscriptLedger` tracks monotonic chunks, accepted checkpoint, and in-flight-through sequence; rollover cannot delete unsummarized speech.
- Summarizer receives exact `T/E/F/P/N` evidence and advances checkpoint only after validated accepted output.

- [ ] Write failing time-window, null-no-update, exact-once, topic-shift, evidence-event, rollover/in-flight checkpoint, cited-summary, unsupported-summary, and latency-timeout fallback tests.
- [ ] Confirm the legacy mutable-prefix race test reproduces the old defect.
- [ ] Switch live callers to shared retrieval/evidence/validation and remove raw fact bonuses/min-max/legacy registries.
- [ ] Run focused and full suites.
- [ ] Commit `feat: integrate evidence first retrieval into live transcription`.

### Task 17: Chat evidence retrieval, tools, regeneration, and UI contracts

**Files:**

- Rewrite: `core/taurine_core/chat/retrieval.py`
- Rewrite: `core/taurine_core/chat/tools.py`
- Modify: `core/taurine_core/chat/engine.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `app/src/renderer/src/lib/chat.ts`
- Modify: `app/src/renderer/src/components/chat/AssistantTurn.tsx`
- Rewrite: `core/tests/test_chat_retrieval.py`
- Rewrite: `core/tests/test_chat_tools.py`
- Extend: `core/tests/test_chat_engine.py`
- Extend: `core/tests/test_server_chat_ws.py`

**Interfaces:**

- Dependent queries rewrite using original plus last two user turns; original and rewritten forms persist in trace.
- Stage A supplies up to 12 exact passages/facts/paths; abstention explicitly reports no KB evidence.
- Tools become `search_evidence`, query-aware paginated `read_note`, and directed `expand_evidence`, all through one registry.
- Regeneration restores the pre-turn retrieval/registry snapshot; only cited sources appear in the primary rail and consulted sources are separate.

- [ ] Write failing rewrite/fallback/Stage-A/tool-registration/pagination/directed-expansion/sufficiency/topic-pivot/regeneration/repair/source-rail tests.
- [ ] Observe legacy metadata-card/raw-note-ID failures.
- [ ] Implement shared-engine adapter, evidence-native tools, snapshot restore, final validation, persistence, and renderer labels.
- [ ] Run focused Python tests, full Python suite, `cd app && npm run typecheck`, and renderer component tests if present.
- [ ] Commit `feat: integrate evidence native retrieval and citations into chat`.

### Task 18: Release regeneration, observability, cleanup, and documentation

**Files:**

- Modify: `core/taurine_core/server.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `app/src/renderer/src/components/TriggerInspector.tsx`
- Modify: `README.md`
- Modify: `docs/diagrams/live-voice-retrieval-pipeline.svg`
- Modify: `docs/diagrams/enrichment-graph-connection-pipeline.svg`
- Create: `core/tests/test_graph_retrieval_integration.py`
- Delete: retired graph/retrieval code and tests identified in Tasks 10, 12, 16, and 17.

**Interfaces:**

- Build reports expose counts, distributions, failures, hub reasons, components/dangling nodes, timings, hashes, and invariant counts.
- Retrieval traces expose raw/calibrated/rejected heads, entity/triple decisions, teleport/PPR/path/rerank/MMR/prior/abstention details, evidence states, hashes, and stage latency.
- Clean offline disposable regeneration is the only release path; no old graph rows are translated or preserved.

- [ ] Write integration tests for clean records→build→evidence, cross-boundary long relation, failed changed extraction, randomized connector order, triple/PPR support, dense fallback, null abstention, temporal conflict, live checkpoint, chat tools/citations, unsupported-claim rejection, and atomic activation.
- [ ] Run the focused integration suite and correct every failing contract.
- [ ] Remove legacy `HeteroGraphMatrix`, additive fact bonuses, query-local min-max, old tools, `rebuild_for`, manual GLiNER chunks, retired config, and superseded tests; verify with targeted `rg` scans.
- [ ] Run a disposable clean build with model fakes, then the packaged offline smoke path when local models are available; record held-out quality/latency gates without fabricating unlabeled metrics.
- [ ] Update README, diagrams, operator command, degraded-mode behavior, and evidence/citation UI semantics.
- [ ] Run `cd core && uv run ruff check .`, `uv run pytest -q`, `cd ../app && npm run lint && npm run build`, deterministic two-pass graph-hash check, and `git diff --check`.
- [ ] Commit `feat!: complete evidence first knowledge graph retrieval overhaul`.

## Final verification checklist

- [ ] Re-read every §23 definition-of-done item and map it to a passing test or verified command.
- [ ] Re-run G1–G9, R1–R7, and C1–C6 coverage scan against tests and production callers.
- [ ] Confirm no unresolved placeholders or deferred implementation markers and no migration/compatibility code entered the implementation.
- [ ] Confirm only source connector records are read during regeneration and the active DB swaps only after validation.
- [ ] Confirm Python tests, lint, Electron typechecks/build, integration replay, deterministic hashes, and clean git diff all pass from fresh commands.
