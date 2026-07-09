# Enrichment Streamlining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LLM-free ingestion (GLiNER2 entities + relations, no gists), a heterogeneous entity+note retrieval graph behind a note-note projection, and corners-gated adjudication (≤2 judgments/note) per `docs/superpowers/specs/2026-07-09-enrichment-streamlining-design.md`.

**Architecture:** New `indexer/extract.py` (GLiNER2 adapter) replaces the LLM enrichment call; `Enrichment` becomes a local-extraction artifact plus a value-gated slim LLM call for spoken aliases/utterances; graph.py grows entity nodes + relation edges internally while `index.graph()` keeps returning a note-note projection; adjudicate.py's bands are replaced by a budgeted uncertainty ∩ (bridge ∪ value) gate with GLiNER-relation skip.

**Tech Stack:** Python 3.12, gliner2>=1.3.2 (existing dep), numpy/scipy (existing), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-09-enrichment-streamlining-design.md`

## Global Constraints

- No new dependencies. Pure numpy/scipy for graph/community code.
- No LLM calls in extraction, graph build, projection, value scoring, or community detection. LLM surfaces are exactly two: the slim aliases/utterances call and adjudication.
- Tests never load the real GLiNER2 or embedding models — monkeypatch the module-level seams (`extract._extractor_model`, `graph.embed_texts`, `store.embed_texts`), mirroring existing tests (see `tests/test_email_pii.py`, `tests/test_graph_terms.py`).
- `notes.gist` column keeps its name; its content becomes the heuristic display line (first non-empty line clamped to `GIST_MAX_CHARS` 220). No schema rename, no app/renderer changes.
- The `edge_judgments` cache and `pair_sig` content-hash scheme are preserved verbatim — existing judgments stay valid.
- Every task ends with `cd core && uv run pytest -q` green and a commit.
- Codex dispatch: each task goes to a fresh Codex subagent (`codex exec`, GPT-5.5, reasoning effort xhigh), always backgrounded with stdin closed (`< /dev/null`). The orchestrator reviews the diff after every task before dispatching the next.

---

### Task 1: Config knobs

**Files:**
- Modify: `core/taurine_core/config.py` (new "GLiNER ingestion" section after the existing GLiNER2 query section; "value score / slim call" section near SOURCE_RANK; adjudication knobs replacing the margin/band constants' comments — old constants stay until Task 9)

**Interfaces (produces, all module constants):**
- `GLINER_INGEST_BATCH = 8`, `GLINER_CHUNK_SIZE = 384`, `GLINER_CHUNK_OVERLAP = 64`
- `GLINER_ENTITY_MIN_CONF = 0.5`, `GLINER_RELATION_MIN_CONF = 0.45`
- `INGEST_ENTITY_TYPES: dict[str, str]` — 8 keys `person, organization, project, product, place, event, date_or_deadline, concept`, values = one-line descriptions (concept: "abstract topic, technology, or practice the note is about")
- `INGEST_RELATION_TYPES: dict[str, str]` — 12 keys `works_with, works_at, member_of, part_of, located_at, scheduled_for, paid_or_owes, owns_or_has, uses, about, met_with, related_to`, values = human-readable fact templates with `{head}`/`{tail}` slots (e.g. `works_at`: `"{head} works at {tail}"`; `related_to`: `"{head} is related to {tail}"`)
- `REL_FACT_MIN_CONF = 0.7`, `REL_EDGE_W = 0.25`, `REL_CATCHALL_FACTOR = 0.5` (related_to down-weight)
- `PPR_PASSAGE_RESET_W = 1.0`, `PPR_ENTITY_RESET_W = 1.0`, `ENTITY_SEED_W = 0.8`
- `VALUE_RECENCY_HALF_LIFE_DAYS = 30.0`, `SLIM_LLM_VALUE_FRACTION = 0.15`, `SLIM_LLM_MIN_NOTES = 5`, `SLIM_UTTERANCES = True` (flipped by Task 12 if the ablation eliminates them)
- `ADJ_VALUE_FRACTION = 0.20`, `ADJUDICATE_MAX_PAIRS_PER_SYNC = 100`, `BRIDGE_RANK_BOOST = 1.5`, `COMMUNITY_LP_ITERS = 20`, `ADJ_TEXT_CAP = 700`

- [ ] **Step 1: Add all constants with the spec's comment style** (short trailing comments; cite calibration-pending on the two GLiNER conf thresholds)
- [ ] **Step 2: `cd core && uv run pytest -q` — passes (constants inert)**
- [ ] **Step 3: Commit** `feat: config knobs for enrichment streamlining (GLiNER ingest, value score, corners adjudication)`

### Task 2: Store — entity tables, salience/value columns, helpers

**Files:**
- Modify: `core/taurine_core/indexer/store.py` (`_ensure_schema` DDL + new helper methods)
- Test: `core/tests/test_store_entities.py`

**Interfaces (produces, methods on `Index`):**
- New DDL (in `_ensure_schema`, `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE` guarded by `PRAGMA table_info` like existing migrations):
```sql
CREATE TABLE IF NOT EXISTS entities(
    entity_id INTEGER PRIMARY KEY,
    canonical TEXT NOT NULL,
    etype TEXT NOT NULL,
    df INTEGER NOT NULL DEFAULT 0,
    UNIQUE(canonical, etype));
CREATE TABLE IF NOT EXISTS note_entities(
    note_id TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    conf REAL NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(note_id, entity_id));
CREATE TABLE IF NOT EXISTS entity_relations(
    note_id TEXT NOT NULL,
    src_entity INTEGER NOT NULL,
    rel_type TEXT NOT NULL,
    dst_entity INTEGER NOT NULL,
    conf REAL NOT NULL,
    PRIMARY KEY(note_id, src_entity, rel_type, dst_entity));
-- notes: ADD COLUMN ingest_salience REAL DEFAULT 1.0; ADD COLUMN value_score REAL DEFAULT 0.0
```
- `replace_note_extraction(note_id: str, entities: list[tuple[str, str, float, int]], relations: list[tuple[str, str, str, float]]) -> None` — entities as (canonical, etype, conf, count), relations as (src_canonical, rel_type, dst_canonical, conf) resolved to entity_ids (upserting into `entities`); deletes the note's previous rows first; recomputes `df` for touched entities.
- `entity_rows() -> list[tuple[int, str, str, int]]` (entity_id, canonical, etype, df); `note_entity_rows() -> list[tuple[str, int, float, int]]`; `relation_rows() -> list[tuple[str, int, str, int, float]]`.
- `relations_between(src_note: str, dst_note: str) -> list[tuple[str, str, str, float]]` — (head_canonical, rel_type, tail_canonical, max conf) for triples whose head-entity and tail-entity are contained by the two notes (either orientation).
- `set_ingest_salience(scores: dict[str, float]) -> None`; `set_value_scores(scores: dict[str, float]) -> None`; `value_scores() -> dict[str, float]`; both notes-table UPDATE loops in one transaction, mirroring `set_quality_ranks`.
- Purge/disable hooks: `remove_note` and `purge_data` also delete the note's `note_entities`/`entity_relations` rows and refresh `df`.

- [ ] **Step 1: Write failing tests** — round-trip replace/read for entities+relations; df recount on replace and on remove_note; relations_between finds the triple in both orientations and returns max conf; salience/value setters round-trip; schema migration is idempotent on an existing DB file (open twice).
- [ ] **Step 2: Run — fails (methods missing)**
- [ ] **Step 3: Implement DDL + helpers**
- [ ] **Step 4: `uv run pytest tests/test_store_entities.py -q` — PASS; full suite passes**
- [ ] **Step 5: Commit** `feat: store tables and helpers for entity/relation extraction and value scores`

### Task 3: GLiNER2 ingestion adapter (`indexer/extract.py`)

**Files:**
- Create: `core/taurine_core/indexer/extract.py`
- Modify: `core/taurine_core/schemas.py` (add `EntityMention`, `RelationTriple`, `Extraction` models — do NOT touch `Enrichment` yet)
- Test: `core/tests/test_index_extract.py`

**Interfaces (produces):**
```python
class EntityMention(BaseModel):
    surface: str
    etype: str        # key of config.INGEST_ENTITY_TYPES
    conf: float

class RelationTriple(BaseModel):
    head: str         # entity surface
    rel_type: str     # key of config.INGEST_RELATION_TYPES
    tail: str
    conf: float

class Extraction(BaseModel):
    entities: list[EntityMention] = []
    relations: list[RelationTriple] = []
    failed: bool = False   # GLiNER inference error → retry next sync
```
- `extract.extract_notes(notes: list[NoteRecord]) -> dict[str, Extraction]` — batches `GLINER_INGEST_BATCH` note texts (`f"{title}\n{text}"`) per forward pass; texts longer than the model window go through the library's long-document path (`GLINER_CHUNK_SIZE`/`GLINER_CHUNK_OVERLAP`); one combined schema per batch: entities from `INGEST_ENTITY_TYPES` (with descriptions) + relations from `INGEST_RELATION_TYPES` keys; thresholds `GLINER_ENTITY_MIN_CONF` / `GLINER_RELATION_MIN_CONF`; relation kept only if both endpoint surfaces also survived entity thresholding (case-insensitive surface match); dedupe entities by (lower(surface), etype) keeping max conf and a mention count; per-note inference exceptions → `Extraction(failed=True)`, logged, never raises.
- `extract._extractor_model()` — module-level seam returning the shared GLiNER2 instance via `live.spans.shared_model()`; tests monkeypatch this.
- **API note for the implementer:** verify the exact gliner2 1.3.x calls against the installed package (`uv run python -c "import gliner2, inspect; ..."`) — `create_schema().entities({...}).relations([...])` + `extract(text, schema)` per the README; `pii.py` and `live/spans.py` show the house style for wrapping it. Tests use a fake model object, so tests are API-version-proof; the live-verification task (12) exercises the real model.

- [ ] **Step 1: Write failing tests** (fake model returning canned schema output): thresholds drop low-conf entities/relations; relation dropped when an endpoint entity was dropped; dedupe keeps max conf + counts mentions; failure → `failed=True` and empty lists; batching splits at `GLINER_INGEST_BATCH` (fake model records call sizes)
- [ ] **Step 2: Run — fails (module missing)**
- [ ] **Step 3: Implement `extract.py` + schema models**
- [ ] **Step 4: `uv run pytest tests/test_index_extract.py -q` — PASS; full suite passes**
- [ ] **Step 5: Commit** `feat: GLiNER2 ingestion extraction adapter (typed entities + relation triples)`

### Task 4: Value scores + email salience persistence

**Files:**
- Modify: `core/taurine_core/indexer/rank.py` (add `compute_value_scores`), `core/taurine_core/connectors/email_common.py` (surface funnel scores), `core/taurine_core/connectors/gmail.py` + `outlook.py` (persist salience after upsert — follow how they currently call the funnel)
- Test: `core/tests/test_value_scores.py`

**Interfaces (produces):**
- `rank.compute_value_scores(index: Index, *, now: float | None = None) -> dict[str, float]` — for each active note: `minmax(quality_rank) × 0.5**(age_days / VALUE_RECENCY_HALF_LIFE_DAYS) × SOURCE_RANK.get(source, SOURCE_RANK_DEFAULT) × ingest_salience`; min-max over active notes maps quality_rank to [0.1, 1.0] (floor 0.1 so zero-rank notes aren't annihilated); persists via `index.set_value_scores` and returns the dict.
- `email_common.funnel_threads` additionally returns per-thread normalized scores: extend `FunnelReport` with `scores: dict[str, float]` (thread_id → min-max normalized `_score`, [0.1, 1.0]); connectors write them via `index.set_ingest_salience({note_id: score})` after upserting.

- [ ] **Step 1: Write failing tests** — formula (hand-computed fixture: two notes, known ranks/ages/sources/salience); floor behavior; persistence round-trip; funnel report carries normalized scores; unknown source falls back to `SOURCE_RANK_DEFAULT`
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Full suite passes (email tests updated for the extended FunnelReport)**
- [ ] **Step 5: Commit** `feat: persisted note value scores and email ingest salience`

### Task 5: Slim LLM enrichment (`indexer/slim_enrich.py`)

**Files:**
- Create: `core/taurine_core/indexer/slim_enrich.py`
- Test: `core/tests/test_slim_enrich.py`

**Interfaces (produces):**
- `slim_enrich.select_slim_notes(changed: list[NoteRecord], value_scores: dict[str, float]) -> list[NoteRecord]` — top `SLIM_LLM_VALUE_FRACTION` of `changed` by value score, minimum `min(SLIM_LLM_MIN_NOTES, len(changed))`.
- `async slim_enrich.slim_enrich_batch(notes: list[NoteRecord], llm: LLMPort, user_card: str | None) -> dict[str, SlimEnrichment]` where `SlimEnrichment(aliases: list[str], utterances: list[str])` (pydantic, in slim_enrich.py). One keyed-JSON call per batch of `ENRICH_BATCH_SIZE` notes (reuse the clustering-free simple chunking — value-selected sets are small), system prompt: `'Return only JSON: {"notes": {"<note_id>": {"aliases": [..], "utterances": [..]}}}. Aliases: up to 6 short spoken names for the note or its subject. Utterances: 4 realistic spoken sentences someone would say in a meeting when this note is relevant.'` — utterances key requested only when `config.SLIM_UTTERANCES`; user prompt carries the user card block + per-note `Title/Folder/Body` exactly like the old `batch_enrich._user_prompt`; per-note validation clamps aliases to 6 / utterances to 4, drops empties; LLM error → one retry → `{}` for missing notes (heuristic aliases only, no per-note fallback call).
- Consumes: `config.ENRICH_MODEL`, `ENRICH_TIMEOUT_S`, `LLMPort.complete_json`.

- [ ] **Step 1: Write failing tests** — selection math (fraction, floor, fewer-changed-than-floor); batch prompt contains user card and all note ids; validation clamps; utterances omitted from schema when `SLIM_UTTERANCES=False` (monkeypatch); LLM failure → empty dict entries
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: `uv run pytest tests/test_slim_enrich.py -q` — PASS**
- [ ] **Step 5: Commit** `feat: value-gated slim LLM enrichment (spoken aliases + utterances)`

### Task 6: The swap — LLM-free sync pipeline, gist elimination

The one large task; everything below changes together so the suite is green at the commit. This retires the old enrichment path.

**Files:**
- Modify: `core/taurine_core/schemas.py` — `Enrichment` becomes:
```python
class Enrichment(BaseModel):
    gist: str = ""                    # heuristic display line (first non-empty line, <=GIST_MAX_CHARS); NOT an LLM artifact
    aliases: list[str] = []           # title + heuristic + slim-call spoken variants
    utterances: list[str] = []        # slim-call only, while SLIM_UTTERANCES
    entities: list[EntityMention] = []
    relations: list[RelationTriple] = []
    extraction_failed: bool = False
```
  (`topics` deleted; downstream `_terms` maps `concept`-type entities to the `topic` mechanism.)
- Delete: `core/taurine_core/indexer/enrich.py`, `core/taurine_core/indexer/batch_enrich.py`, `core/tests/test_index_enrich_notes.py`, `core/tests/test_batch_enrich.py`
- Modify: `core/taurine_core/indexer/sync.py` — new pipeline (see below); `_embedding_units` drops the gist unit, keeps chunks always and utterances when present
- Modify: `core/taurine_core/indexer/graph.py` — `_note_vector` = mean chunk vectors only (delete gist branch); `_terms` reads typed entities (`concept` → topic mechanism lowercased, all other types → entity mechanism case-preserved)
- Modify: `core/taurine_core/indexer/rank.py` — `_first_gist_vector` → `_note_mean_vector` (chunk mean, cached), `DUP_EDGE_SIM` guard unchanged otherwise
- Modify: `core/taurine_core/indexer/store.py` — `upsert_note` computes `stored.gist` from the display-line heuristic (port `_fallback_gist`/`_clamp_gist` from the deleted enrich.py into store.py as `_display_line(note) -> str`); `_aliases_for` unchanged; `enrichment_for` returns the new model (entities/relations loaded from the new tables)
- Modify: `core/taurine_core/indexer/reenrich.py` + `core/taurine_core/indexer/adjudicate.py` prompts — `_pair_section` uses `Left text: {left.text[:ADJ_TEXT_CAP]}` instead of gists
- Modify: `core/taurine_core/live/summarizer.py` `_grounding_block` + `core/taurine_core/chat/engine.py` grounding line — append top-3 entity surfaces after the display line: `— {gist} [{e1}, {e2}, {e3}]` when entities exist
- Modify: `core/taurine_core/server.py` reenrich comment, `core/taurine_core/indexer/__init__.py` exports
- Test: rewrite `core/tests/test_index_sync.py`, `core/tests/test_index_reenrich.py` fixtures (fake extractor + fake slim LLM instead of fake enrichment LLM); update `test_graph_terms.py`, `test_index_store.py`, `test_index_reembed.py`, `test_graph_adjudicate.py` fixtures to the new `Enrichment` shape

**New sync order (inside `sync_source`, replacing the enrich phase):** extract (connector) → hash-diff → `extract.extract_notes` (local) → `_embedding_units` + `upsert_note` per note → graph rebuild → `compute_quality_ranks` → `compute_value_scores` → `slim_enrich` for `select_slim_notes` (merge aliases/utterances: re-upsert alias rows + append utterance embedding units for those notes) → adjudication (async, unchanged call site until Task 9) → `refresh_fact_embeddings`. Enrichment cache (`enrich_cache`) now stores the `Extraction` JSON keyed by content_hash, same table.

**Heuristic aliases (in sync, for every note):** normalized title + GLiNER entity surfaces that appear word-boundary in the title, filtered by existing `ALIAS_MIN_COMPACT_LEN`/stoplist guards; slim-call aliases are added on top for gated notes.

- [ ] **Step 1: Rewrite the fixtures/tests listed above to the new contracts (failing)**
- [ ] **Step 2: Implement the swap across all listed files**
- [ ] **Step 3: `cd core && uv run pytest -q` — full suite PASS**
- [ ] **Step 4: `grep -rn "gist" core/taurine_core --include="*.py" | grep -v display` — remaining hits are display-line/schema/column only (no LLM-gist semantics)**
- [ ] **Step 5: Commit** `feat!: LLM-free ingestion — GLiNER extraction replaces LLM enrichment, gists become display lines`

### Task 7: Heterogeneous retrieval graph + note projection

**Files:**
- Modify: `core/taurine_core/indexer/graph.py` (entity canonicalization into store tables during rebuild; projection), `core/taurine_core/indexer/rank.py` (heterogeneous matrix + PPR reset weights)
- Test: `core/tests/test_graph_hetero.py`

**Interfaces (produces):**
- `graph.rebuild_all` / `rebuild_for` additionally: canonicalize extraction entity surfaces with the existing `resolve_terms` machinery (per etype, lowercase only for `concept`), write via `index.replace_note_extraction`, then build note-note edges as today where shared-term edges now come from canonical entities (concept → `topic` mechanism, others → `entity`), **plus** relation-derived candidate pairs: two notes sharing a relation triple (via `relations_between`) get a candidate of mechanism `entity` with weight `REL_EDGE_W × conf` (× `REL_CATCHALL_FACTOR` for `related_to`), merged by the existing noisy-or.
- `rank.HeteroGraphMatrix.build(index) -> HeteroGraphMatrix` — nodes = active notes + entities (row ids `note:{id}` / `ent:{entity_id}`); edges: projected note-note boosted weights (as today) + `contains` (note↔entity, weight `idf × conf` with the existing `_term_weight` df curve) + `relation` (entity↔entity, `REL_EDGE_W × conf`); column-stochastic CSR; exposes `note_rows` slice map.
- `rank.ppr(...)` gains `entity_seeds: dict[int, float] | None = None`; reset vector = note seeds × `PPR_PASSAGE_RESET_W` + entity seeds × `PPR_ENTITY_RESET_W`, renormalized; returns note-row mass only (renormalized over notes) so grounding consumers see the same shape as before. `NoteGraphMatrix` (quality rank + existing PPR callers) stays until callers migrate in this task — `compute_quality_ranks` keeps using the note projection.
- `index.graph()` payload unchanged (projection): shared-entity edges appear exactly like today's shared-term edges (`shared` labels = canonical surfaces).

- [ ] **Step 1: Write failing tests** — canonicalization writes entities tables with df; relation candidate becomes an edge with the down-weighted catch-all; HeteroGraphMatrix shapes (n_notes+n_entities), contains/relation weights, column normalization; ppr with an entity seed ranks the containing note above an unconnected note; note-only mass renormalization; `index.graph()` returns no entity nodes
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Full suite passes (grounding/chat tests untouched — same PPR output shape)**
- [ ] **Step 5: Commit** `feat: heterogeneous entity+note retrieval graph with note-note projection`

### Task 8: Query-time entity seeding

**Files:**
- Modify: `core/taurine_core/live/grounding.py` (seed accumulator → entity seeds), `core/taurine_core/chat/retrieval.py` (same head), `core/taurine_core/indexer/store.py` (lookup helper)
- Test: extend `core/tests/test_chat_retrieval.py` + `core/tests/test_server_ws.py` fixtures if needed; new `core/tests/test_entity_seeding.py`

**Interfaces (produces):**
- `store.Index.entity_by_surface(surface: str) -> int | None` — canonical/synonym surface (case-folded per etype rules) → entity_id; backed by an in-memory dict rebuilt on graph rebuild (surfaces = every member of each union-find cluster, loaded from a new `entity_surfaces(entity_id, surface)` table written during canonicalization in Task 7 — add the DDL + write path here if Task 7 didn't).
- In grounding + chat retrieval: GLiNER spans (already extracted per window/query) that resolve via `entity_by_surface` contribute `entity_seeds[entity_id] = ENTITY_SEED_W × head_score` alongside existing note seeds; passed through to `rank.ppr(..., entity_seeds=...)`. Decay handling: entity seeds live only for the current pass (no accumulator changes — notes keep the accumulator; entity seeds are per-window transients).

- [ ] **Step 1: Write failing tests** — surface lookup (canonical + synonym member + case rules); a GLiNER span naming an entity raises the PPR mass of its containing notes in a 3-note fixture graph; absent surface → no seed, no error
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Full suite passes**
- [ ] **Step 5: Commit** `feat: GLiNER spans seed entity nodes directly in PPR`

### Task 9: Corners adjudication gate

**Files:**
- Modify: `core/taurine_core/indexer/adjudicate.py` (gate replaces `_pending_pairs` band logic), `core/taurine_core/indexer/graph.py` (`_apply_judgments` also promotes GLiNER-relation facts), `core/taurine_core/indexer/store.py` (edges table gains `fact_provenance TEXT DEFAULT ''` column + migration), `core/taurine_core/schemas.py` (`GraphEdge.fact_provenance: str = ""`)
- Create: `core/taurine_core/indexer/communities.py`
- Test: `core/tests/test_communities.py`, rewrite `core/tests/test_graph_adjudicate.py` gate cases

**Interfaces (produces):**
- `communities.label_propagation(matrix: NoteGraphMatrix) -> dict[str, int]` — synchronous LP over the note projection CSR, `COMMUNITY_LP_ITERS` max iters, deterministic (ties broken by lowest label; iteration order = sorted note_ids; seed labels = row index). Pure scipy/numpy.
- **GLiNER-relation promotion (in `graph._apply_judgments`, before judgment application):** for each kept/margin candidate pair, `index.relations_between(src, dst)`; if best conf ≥ `REL_FACT_MIN_CONF` → edge admitted/kept with `fact = INGEST_RELATION_TYPES[rel_type].format(head=head, tail=tail)`, `fact_provenance="gliner"`; such pairs are excluded from the adjudication queue entirely. Adjudicated facts write `fact_provenance="llm"`.
- **Gate (in `adjudicate._pending_pairs`):** candidates = weak-kept (weight < `VETO_MAX_W`) ∪ margin near-misses (existing machinery), minus GLiNER-promoted, minus valid-judgment holders; qualify if `communities[src] != communities[dst]` **or** `max(value[src], value[dst]) ≥ value_threshold` where `value_threshold` = the `ADJ_VALUE_FRACTION` quantile of active-note value scores; rank by `(value[src]+value[dst]) × (BRIDGE_RANK_BOOST if bridge else 1.0)` desc; truncate to `ADJUDICATE_MAX_PAIRS_PER_SYNC`. `_fact_eligible_strong_keys` keeps only strong edges whose either endpoint clears `value_threshold`.
- Old band constants (`MARGIN_MAX_PER_NOTE` cap etc.) remain as candidate-generation guards; `FACT_TOP_EDGES_PER_NOTE` unchanged in meaning.

- [ ] **Step 1: Write failing tests** — LP: two dumbbell cliques joined by one edge → 2 communities, bridge detected, deterministic across runs; gate: GLiNER-promoted pair skipped + fact templated + provenance set; non-bridge low-value ambiguous pair excluded; bridge low-value pair included; budget truncation keeps highest-ranked; value quantile math
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Full suite passes**
- [ ] **Step 5: Commit** `feat: corners adjudication — GLiNER fact promotion, bridge/value gate, hard budget`

### Task 10: Migration script

**Files:**
- Create: `core/migrate_enrichment.py` (repo-root-of-core script, `repair_enrich.py` style: argparse `--db`, `--dry-run`, prints a report)
- Test: `core/tests/test_migrate_enrichment.py` (fixture DB built with old-shape rows)

**Interfaces (produces):** `migrate(db_path: Path, *, llm: LLMPort | None, dry_run: bool) -> MigrationReport` executing, in order:
1. Delete `embeddings` rows kind `gist` (and kind `utterance` when `not config.SLIM_UTTERANCES`); reload matrix.
2. Recompute `notes.gist` display lines from `_display_line`.
3. GLiNER backfill: `extract.extract_notes` over all active notes (batched, progress-printed), `replace_note_extraction` + enrich_cache update.
4. Rebuild graph, quality ranks, value scores.
5. If `llm` provided and not dry_run: one budgeted adjudication pass + `refresh_fact_embeddings`.
`MigrationReport(gist_rows_deleted, utterance_rows_deleted, notes_backfilled, extraction_failures, judgments_written)`.

- [ ] **Step 1: Write failing tests** — old-shape fixture DB: gist rows deleted, display lines recomputed, backfill writes entity tables (fake extractor), dry-run mutates nothing, report counts correct
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Full suite passes**
- [ ] **Step 5: Commit** `feat: one-time enrichment-streamlining migration script`

### Task 11: Utterance ablation harness

**Files:**
- Create: `core/scripts/ablate_utterances.py` (+ `core/scripts/__init__.py` if absent)
- Test: `core/tests/test_ablate_utterances.py` (fixture session log + fake-model DB)

**Interfaces (produces):** `python -m scripts.ablate_utterances --db taurine.db --sessions sessions/ [--limit N]`:
- Parses `chunk` events (final only) from each `sessions/*.jsonl`, replays 8s windows through the C2 retrieval stack (reuse the grounding machinery the way `test_chat_retrieval.py` fixtures drive it) in two in-process conditions: baseline vs utterance-masked (mask = in-memory filter dropping `kind='utterance'` rows from the loaded embedding matrix — no DB mutation).
- Per window records: fused candidates with head attribution, final grounding set. Aggregates per session and corpus-wide: (a) utterance-only grounding fraction (items whose *only* introducing head row was an utterance embedding), (b) mean top-5 grounding Jaccard between conditions, (c) per-session diff lines.
- Prints a markdown report and exits 0; `--json out.json` for machine reading. **Decision rule printed with the verdict:** eliminate if (a) < 5% AND (b) ≥ 0.9.

- [ ] **Step 1: Write failing tests** — fixture: 2-note DB where note A is reachable only via an utterance row for the query text; harness reports utterance-only fraction > 0 and Jaccard < 1.0; masking leaves DB rows intact; verdict logic both directions
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Full suite passes**
- [ ] **Step 5: Commit** `feat: utterance ablation replay harness over stored session logs`

### Task 12: Live verification against the corpus (orchestrator-led, not Codex)

**Files:** none created except `docs/superpowers/specs/2026-07-09-enrichment-streamlining-design.md` (follow-up notes section) and possible config calibration edits.

- [ ] **Step 1: Snapshot** `cp core/taurine.db core/taurine.pre-streamline.db`
- [ ] **Step 2: GLiNER calibration** — run `extract.extract_notes` over ~50 real notes sampled across sources; eyeball entity/relation precision; adjust `GLINER_ENTITY_MIN_CONF`/`GLINER_RELATION_MIN_CONF`/`REL_FACT_MIN_CONF` (gate: if relation precision is unusable, set `REL_FACT_MIN_CONF = 1.01` per spec and note it)
- [ ] **Step 3: Migration dry-run then real run** on a DB copy (`--db` flag), with LLM: record MigrationReport, judgment count (target ≤ ~1,100 total), wall time, token cost
- [ ] **Step 4: Utterance ablation** — `python -m scripts.ablate_utterances` over all 219 session logs against the migrated DB copy; apply the pre-registered rule; if eliminated: flip `SLIM_UTTERANCES = False`, re-run migration step 1 for utterance rows, note verdict + numbers in the spec
- [ ] **Step 5: Replay recalibration** — grounding hysteresis spot-check on 3–5 replayed sessions (hetero-matrix PPR mass); adjust `PPR_*_RESET_W` only if grounding sets visibly churn
- [ ] **Step 6: Full suite + repo-wide grep audit** (`uv run pytest -q`; no orphaned imports of enrich/batch_enrich)
- [ ] **Step 7: Swap the migrated DB into place, run one real `sync_all_enabled` end-to-end, verify map view + note previews + live summary render via the app**
- [ ] **Step 8: Commit calibration + spec follow-up notes**

## Task dependency order

1 → 2 → 3 → 4 → 5 → 6 (needs all prior) → 7 → 8 (needs 7) → 9 (needs 7) → 10 (needs 6–9) → 11 (needs 6) → 12 (needs all). 8 and 9 are parallelizable; everything else is sequential.
