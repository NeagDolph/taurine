# Enrichment streamlining — LLM-free ingestion, heterogeneous graph, LLM at the corners

2026-07-09. Follows the enrichment/retrieval literature review (HippoRAG 2
extraction-over-summarization ablations, Mem0 fact consolidation, Graphiti
fact edges, GraphRAG/RAPTOR summaries earning their keep only at
cluster/community level). Supersedes the per-note LLM enrichment design in
`2026-07-06-taurine-mvp-design.md` §enrichment and extends
`2026-07-07-graph-connections-v2.md`.

## Motivation (measured on the real corpus, 768 notes)

- 47% of notes are <300 chars; 79% <1500. 220 notes (29%) have a gist
  *longer than the note itself* — the LLM summary is a paraphrase tax, and
  its embedding replaces the note's own text as the graph kNN vector
  (paraphrase noise, per HippoRAG 2's "LLM-generated noise" finding).
- Utterances are 3,376 of 5,340 embedding rows (63%) and ~2× the gist's
  output tokens, with zero ablation evidence.
- Adjudication holds 6,468 judgments ≈ 8.4/note on deepseek-v4-pro — the
  dominant LLM cost, far above enrichment.
- GLiNER runs only at query time today; all ingestion entities come from
  the LLM call.

## Goals

1. **Eliminate LLM gists entirely.** No per-note summary artifact; every
   role the gist plays gets a raw-text or embedding replacement.
2. **LLM-free ingestion by default.** GLiNER2 (already a dependency,
   `gliner2>=1.3.2`, CPU-first) produces typed entities and typed relations
   locally. One slim LLM call survives, gated to top-value notes only.
3. **HippoRAG-2-style heterogeneous retrieval graph** — entity nodes +
   passage (note) nodes, contains/relation edges, synonymy via canonical
   entity merging, PPR with per-node-type reset weights — behind a
   note-note projection so map/UI/quality-rank consumers are untouched.
4. **LLM at the corners:** adjudication drops from ~8.4 to ≤2 judgments per
   note via a hybrid gate (uncertainty band ∩ (cluster bridge ∪ high-value
   endpoint)) under a hard per-sync budget, with high-confidence GLiNER
   relations skipping adjudication entirely.
5. **Utterance ablation harness** over the 219 stored session logs, with a
   pre-registered decision rule; utterances survive (if at all) only inside
   the value-gated slim call.

## Design

### A. GLiNER2 ingestion extraction (no LLM)

One GLiNER2 `create_schema()` call per note (batched via
`batch_extract`, `GLINER_INGEST_BATCH` notes per forward pass; notes longer
than the model context go through `extract_entities_long` with
`GLINER_CHUNK_SIZE`/`GLINER_CHUNK_OVERLAP` and span remapping):

- **Entities** with types and descriptions:
  `person`, `organization`, `project`, `product`, `place`, `event`,
  `date_or_deadline`, `concept` (abstract topic/technology/practice — this
  replaces the old LLM `topics` field; there is no separate topics artifact
  anymore). Keep entities with confidence ≥ `GLINER_ENTITY_MIN_CONF`.
- **Relations** from a fixed vocabulary (GLiNER2 requires enumerated
  relation types — deliberate deviation from HippoRAG's schema-free OpenIE,
  which needs an LLM). Initial vocabulary, iterated during calibration:
  `works_with`, `works_at`, `member_of`, `part_of`, `located_at`,
  `scheduled_for`, `paid_or_owes`, `owns_or_has`, `uses`, `about`,
  `met_with`, `related_to` (catch-all, down-weighted). Keep relations with
  confidence ≥ `GLINER_RELATION_MIN_CONF`; both endpoints must also survive
  entity thresholding.
- Output stored per note: `entities: [(surface, type, conf)]`,
  `relations: [(head_surface, rel_type, tail_surface, conf)]`.
- **Failure fallback:** GLiNER inference error → empty entities/relations
  (note still gets chunk embeddings, kNN edges, title alias); logged, note
  flagged for retry on next sync. No LLM fallback.
- The `Enrichment` schema drops `gist`, `topics`, `utterances` as LLM
  fields; `entities` becomes typed; new `relations` field. Enrichment cache
  keying by content_hash unchanged.

### B. Gist elimination — replacements per role

| old gist role | replacement |
|---|---|
| note graph vector (semantic kNN, `_note_vector`) | mean of chunk vectors — the existing fallback becomes the only path; `embeddings` rows of kind `gist` deleted |
| `DUP_EDGE_SIM` same-source dup guard (rank.py) | chunk-mean cosine instead of gist cosine, same threshold, recalibrate on real corpus |
| adjudication pair prompt | raw note text, head-truncated at `ADJ_TEXT_CAP` (700) chars per side |
| live/chat grounding-block line, `tools.py`, UI previews | `display_line`: first non-empty line clamped to 220 chars (the current `_fallback_gist` heuristic), computed at upsert, stored in the existing `notes.gist` column (renamed usage, not renamed column — migration cheap) |
| `GIST_MAX_CHARS` | retired from prompts; kept only as the display-line clamp |

Known regression risk: for long notes whose first line is uninformative
("Meeting notes"), grounding-block lines get weaker. Mitigation: grounding
lines append the note's top-3 entity surfaces
(`"Meeting notes" — Braxton, ShopLC, recommendation engine`); fact edges
carry the semantic load, per HippoRAG 2 (facts + raw text beat summaries).

### C. Note value score (persisted, shared by slim call + adjudication)

`value = quality_rank_norm × recency_decay × SOURCE_RANK[source] ×
ingest_salience`

- `quality_rank_norm`: existing persisted `notes.quality_rank`, min-max
  normalized over active notes (0 default on first run).
- `recency_decay`: exp half-life over `modified`,
  `VALUE_RECENCY_HALF_LIFE_DAYS` (30).
- `ingest_salience`: **new persisted column** `notes.ingest_salience REAL
  DEFAULT 1.0`; the email funnel's existing `_score` (email_common.py) is
  min-max normalized per sync batch and written for gmail/outlook notes;
  1.0 for all other sources.
- Computed at rebuild time (after quality ranks), stored in
  `notes.value_score` for inspection; consumers read it, never recompute.

### D. Value-gated slim LLM call (the only per-note LLM)

- Gate: top `SLIM_LLM_VALUE_FRACTION` (0.15) of changed notes by value
  score, minimum floor `SLIM_LLM_MIN_NOTES` (5) per sync so small syncs
  still cover their best notes.
- Content: spoken **aliases** (≤6) only — plus **utterances** if and only
  if the §F ablation keeps them. Model `ENRICH_MODEL` (glm-5.2), batched
  exactly like today's batch_enrich (clustered, keyed JSON, per-note
  validation), just a much smaller schema and note population.
- Non-gated notes get heuristic aliases: normalized title + GLiNER entity
  surfaces that appear in the title (word-boundary, existing
  `ALIAS_MIN_COMPACT_LEN`/stoplist guards apply). C1 exact matching keeps
  working everywhere; only *spoken-variant* aliases become a
  top-value-notes feature.

### E. Two-layer graph

**Layer 1 — heterogeneous retrieval graph (LLM-free).**

- **Entity canonicalization** reuses the existing term_vectors machinery
  unchanged (embed unique surfaces, union-find at
  `TERM_SYNONYM_MIN_SIM` 0.95, digit guard, canonical = most frequent).
  Canonical entities become nodes; synonym membership is the node's
  surface-form set.
- New tables:
  - `entities(entity_id, canonical, etype, df)` — df = containing-note
    count;
  - `note_entities(note_id, entity_id, conf, count)` — the `contains`
    edges;
  - `entity_relations(src_entity, rel_type, dst_entity, conf, note_id)` —
    provenance-keyed so a note's re-extraction replaces its own triples.
- **Edge types and weights** in the retrieval matrix:
  - `contains` (note↔entity): rarity-weighted `idf(entity) × conf`, same
    df/rarity curve as today's shared-term weights;
  - `relation` (entity↔entity): `REL_EDGE_W × conf`, `related_to` halved;
  - `synonym`: implicit (merged into one canonical node), no explicit edge;
  - `semantic-kNN` (note↔note): unchanged mechanism, chunk-mean vectors;
  - `fact` (note↔note): adjudicated/GLiNER-promoted edges, boosted as
    today (`FACT_EDGE_BOOST`).
- **PPR** (rank.py) runs over the heterogeneous matrix. Reset
  (personalization) vector gets per-node-type weights:
  `PPR_PASSAGE_RESET_W` (1.0) for note seeds, `PPR_ENTITY_RESET_W` (1.0)
  for entity seeds, and — following HippoRAG 2's passage-node finding —
  entity→note flow is what surfaces associated notes, so retrieval output
  reads only the note rows of the stationary vector. Query-time seeding
  gains a new head: GLiNER spans from live/spans.py and chat retrieval that
  resolve (exact or synonym-snap) to a canonical entity seed that entity
  node directly, weight `ENTITY_SEED_W` (0.8), alongside the existing dense
  /BM25/alias note seeds.
- Warm-start, damping, hysteresis knobs unchanged; grounding thresholds
  recalibrated once on session-log replays (PPR mass redistributes when
  entity nodes join the matrix).

**Layer 2 — note projection (UI/rank compatibility).**

- `index.graph()` (map, /graph payload, NotePreview neighbors, quality
  rank) returns a projected note-note graph: kNN and fact edges pass
  through; each note-entity-note path collapses to a `shared` edge with
  weight `min(contains_w_left, contains_w_right)`, per-pair noisy-or merge
  with kNN exactly as today, existing keep-rules/thresholds apply.
- Quality rank (`compute_quality_ranks`) runs on the projection — teleport,
  dup-guard (now chunk-mean cosine), and log-median transform unchanged.
- Net effect: map/UI/rank see the same shape of graph they see today;
  entity nodes exist only inside retrieval.

### F. Utterance ablation harness (decides utterances' fate)

- New `core/scripts/ablate_utterances.py` (permanent eval tool): replays
  transcript `chunk` events from `core/sessions/*.jsonl` through the C2
  retrieval stack against the current DB in two conditions — utterance
  embedding rows present vs masked (in-memory row filter, no DB mutation).
- Recorded per 8s window: fused candidate lists with head attribution
  (which head(s) introduced each note), final grounding sets, ranks.
- Report: (a) fraction of grounding-set items introduced *only* by the
  utterance head; (b) top-5 grounding Jaccard between conditions per
  window; (c) per-session diff summaries.
- **Pre-registered decision rule:** if utterance-only items are <5% of
  grounding appearances AND mean top-5 Jaccard ≥ 0.9, utterances are
  eliminated (drop from slim call, delete `utterance` embedding rows in
  migration). Otherwise they are kept for slim-call (top-value) notes only,
  and the harness result is recorded in the spec follow-up notes.
- Caveat recorded up front: session logs are unlabeled; this measures
  *contribution and disruption*, not ground-truth relevance. Older logs
  lacking grounding events still contribute transcripts (both conditions
  are computed fresh; logged grounding sets are reference-only).

### G. Adjudication — hybrid corners gate (option 3c)

Candidate pairs come from the same cheap machinery as today (kept edges +
margin near-misses). The gate replaces the current band table:

1. **GLiNER-relation skip:** pair endpoints connected by ≥1 relation triple
   with conf ≥ `REL_FACT_MIN_CONF` (calibrated, start 0.7) → edge admitted
   without LLM; fact text rendered from the triple template
   (`"{head} — {rel_type} — {tail}"`, human-readable phrasing table),
   `edges.fact_provenance = 'gliner'`. Never enters the adjudication queue.
2. **Uncertainty band:** remaining pairs qualify only if cheap signals are
   ambiguous — kept edges with weight < `VETO_MAX_W` (0.45) or margin
   near-misses (semantic ≥ `MARGIN_SEMANTIC_SIM` or shared-w ≥
   `MARGIN_SHARED_W`), i.e. today's weak-kept ∪ margin bands. Strong kept
   edges are trusted without facts unless rule 3 selects them.
3. **Corners filter (∩ with band):** qualify if the pair is a **cluster
   bridge** — endpoints in different communities under label propagation
   over the projected note graph (pure scipy, deterministic iteration
   order, `COMMUNITY_LP_ITERS` 20) — **or** either endpoint is top
   `ADJ_VALUE_FRACTION` (0.20) by value score.
4. **Budget:** survivors ranked by `value(src) + value(dst)` (bridges get
   `BRIDGE_RANK_BOOST` ×1.5), cut at `ADJUDICATE_MAX_PAIRS_PER_SYNC` (100).
   Overflow pairs stay unjudged and compete again next sync — no starvation
   tracking needed since content-hash signatures keep them eligible.
5. `FACT_TOP_EDGES_PER_NOTE` fact-writing for strong edges is retained but
   only for notes in the top `ADJ_VALUE_FRACTION` — facts concentrate on
   the notes retrieval actually surfaces.

Judgment cache (`edge_judgments`, content-hash sigs) unchanged and remains
valid across the migration. Prompt change: gists replaced by raw text
truncated to `ADJ_TEXT_CAP` chars per note (§B). Batch size, model,
progress plumbing unchanged. Expected volume on the current corpus:
≈700–1,100 total judgments (≤1.5/note) vs 6,468 today, bounded by budget.

### H. Sync / re-enrich flow and migration

New sync order: extract → hash-diff → **GLiNER2 batch (local)** → chunk
embeddings → graph rebuild (heterogeneous + projection) → quality ranks →
value scores → **slim LLM call (gated)** → **budgeted adjudication** →
fact embeddings → final rebuild-apply. (Slim call after value scores so
gating uses fresh ranks; its aliases land before the next C1 reload.)

Full re-enrich: user-card build (unchanged, LLM) → GLiNER re-extraction
(local, cheap — full-corpus re-enrich no longer costs a per-note LLM pass)
→ same tail as sync.

Migration (one-time, in `repair_enrich.py` style):

1. Delete `embeddings` rows of kind `gist` (and kind `utterance` if the §F
   rule eliminates them); reload embedding matrix.
2. Recompute `notes.gist` as display lines (first-line clamp).
3. GLiNER backfill pass over all active notes → entities/relations tables.
4. Rebuild graph + quality ranks + value scores; run one budgeted
   adjudication pass (cache carries over — most strong pairs already
   judged).
5. Drop the old margin/band adjudication queue state (none is persisted
   beyond `edge_judgments`, so this is a no-op checkpoint).

### Config knobs (new / changed)

| knob | default | note |
|---|---|---|
| `GLINER_INGEST_BATCH` | 8 | notes per GLiNER forward batch |
| `GLINER_CHUNK_SIZE` / `GLINER_CHUNK_OVERLAP` | 384 / 64 | long-note windowing (tokens) |
| `GLINER_ENTITY_MIN_CONF` | 0.5 | calibrate on corpus |
| `GLINER_RELATION_MIN_CONF` | 0.45 | calibrate on corpus |
| `REL_FACT_MIN_CONF` | 0.7 | relation → fact without LLM |
| `REL_EDGE_W` | 0.25 | relation edge base weight |
| `PPR_PASSAGE_RESET_W` / `PPR_ENTITY_RESET_W` | 1.0 / 1.0 | tune on replays |
| `ENTITY_SEED_W` | 0.8 | query-time GLiNER-span entity seeds |
| `VALUE_RECENCY_HALF_LIFE_DAYS` | 30 | value-score decay |
| `SLIM_LLM_VALUE_FRACTION` / `SLIM_LLM_MIN_NOTES` | 0.15 / 5 | slim-call gate |
| `ADJ_VALUE_FRACTION` | 0.20 | corners value gate |
| `ADJUDICATE_MAX_PAIRS_PER_SYNC` | 100 | hard budget |
| `BRIDGE_RANK_BOOST` | 1.5 | bridge priority in budget ranking |
| `COMMUNITY_LP_ITERS` | 20 | label-propagation iterations |
| `ADJ_TEXT_CAP` | 700 | chars of raw text per note in adjudication prompt |
| retired | — | `GIST_MAX_CHARS` (prompt role), utterance knobs if eliminated |

### Testing

- Unit: GLiNER adapter (mocked model) — schema build, thresholding,
  long-note remapping, failure fallback; entity canonicalization into
  nodes; heterogeneous matrix construction (edge types/weights); note
  projection equivalence (kNN/fact passthrough, entity-path collapse);
  corners gate (band ∩ bridge/value, budget cut, GLiNER skip); display-line
  computation; value-score formula; slim-call gating.
- Replay: ablation harness itself gets fixture-log tests; grounding
  hysteresis recalibration measured on real session replays before/after
  the heterogeneous matrix lands.
- Live verification (pre-merge, on a DB copy): GLiNER extraction quality
  review on ~50 real notes (entity/relation precision eyeball +
  threshold calibration — same procedure as the 2026-07-07
  `TERM_SYNONYM_MIN_SIM` calibration); one full migration + sync + budgeted
  adjudication run with cost/latency measurement; map view + note previews
  visually unchanged.

### Risks / calibration gates

- **GLiNER relation quality on personal notes is unproven.** The
  calibration review is a *gate*: if relation precision is poor at any
  usable threshold, the relation vocabulary shrinks (or `REL_FACT_MIN_CONF`
  → 1.01, disabling the skip rule) and adjudication carries more weight —
  the budget still bounds cost.
- First-line display lines are weaker than gists for long, badly-titled
  notes; mitigated by entity-suffix rendering (§B), revisit only if live
  summaries degrade.
- PPR mass redistribution over the heterogeneous matrix invalidates current
  grounding hysteresis calibration — recalibrate on replays before merge.
- Heuristic aliases may reduce C1 recall on non-top-value notes; acceptable
  by design (spoken variants are a high-value-note feature), monitored via
  session logs.

## Non-goals

- Temporal fact validity / invalidation (still deferred, per v2 spec).
- Hierarchical/community summaries (RAPTOR/GraphRAG style) — nothing in
  the corpus is queried holistically yet.
- Entity nodes in the map UI or /graph payload (projection keeps them
  internal).
- Open-schema OpenIE triple extraction (requires LLM; GLiNER2's fixed
  vocabulary is the chosen tradeoff).
- Changing the user-card profile build.

## Execution split (proposed units)

1. **Unit A** — GLiNER2 ingestion adapter + schema/store changes
   (entities/relations tables, typed Enrichment, display lines, value
   score + `ingest_salience` persistence).
2. **Unit B** — gist elimination mechanics (embedding kind removal,
   `_note_vector` simplification, dup-guard switch, prompt/UI display-line
   swaps) + migration script.
3. **Unit C** — heterogeneous graph build + PPR reset weights + entity
   seeding + note projection.
4. **Unit D** — corners adjudication gate (bands, label propagation,
   budget, GLiNER-relation skip/templating) + slim LLM call.
5. **Unit E** — utterance ablation harness + replay recalibration; its
   verdict finalizes Unit D's slim-call schema.
6. **Verification** — calibration reviews, migration dry-run on DB copy,
   cost measurement, full suite.
