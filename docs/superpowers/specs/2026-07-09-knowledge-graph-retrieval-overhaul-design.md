# Knowledge graph and retrieval overhaul — evidence-first HippoRAG 2 architecture

**Date:** 2026-07-09

**Status:** approved design direction; implementation plan follows after spec review

**Basis:** `docs/audits/2026-07-09-knowledge-graph-retrieval-review.md`

**Supersedes where conflicting:**

- `2026-07-09-enrichment-streamlining-design.md`
- graph-generation and PPR portions of `2026-07-08-live-summary-graph-retrieval.md`
- retrieval, evidence, tool, and citation portions of `2026-07-08-chat-knowledge-base.md`
- `2026-07-07-graph-connections-v2.md`

The existing product behavior outside graph generation, retrieval, evidence assembly, citations, and their directly affected UI contracts remains in force.

## 1. Decision and scope

Taurine will replace its current overlapping projected-note/heterogeneous graph with an **evidence-first, HippoRAG-2-style retrieval architecture** optimized for a local small-to-medium personal knowledge base.

The system will retain the parts of the current implementation that are a good fit for Taurine:

- local GLiNER2 extraction with its DeBERTa-v3-large encoder;
- Qwen3 instruction-aware embeddings;
- FTS5/BM25, dense retrieval, and exact aliases;
- local query-time retrieval for the live transcription path;
- bounded LLM use for offline high-value ambiguity and final synthesis;
- source, recency, and quality priors, after they are measured by ablation;
- a compact note projection for the map and note-preview UI.

It will adopt the parts of HippoRAG 2 and later retrieval research that the current implementation is missing:

- passages as first-class graph and evidence nodes;
- source-span provenance for every entity and relation;
- query-to-triple retrieval with recognition filtering;
- phrase/entity and passage reset probabilities in PPR;
- original passages, facts, and graph paths in downstream context;
- calibrated head scores and a real abstention decision;
- separate retrieval graph and product projection;
- directed and temporally valid facts;
- evidence-level citations and claim support validation;
- deterministic full graph finalization with no incremental projection drift.

### 1.1 Mandatory findings addressed

All findings from the 2026-07-09 audit are in scope:

- **G1-G9:** GLiNER long extraction, extraction evaluation, relation recall, double counting, rebuild drift, entity resolution, stale graph state, directed provenance, and scale behavior.
- **R1-R7:** abstention, true query-to-triple seeding, passage serving, transcript checkpointing, dangling PPR mass, hub handling, and accumulator/exact-hit calibration.
- **C1-C6:** evidence-first chat context, tool registry correctness, query-aware note reading, semantic citation validation, multi-turn topic/query handling, and explanatory graph paths.

### 1.2 Explicit non-goals

This spec contains **no data migration work**. The current knowledge graph, derived embeddings, extraction cache, fact embeddings, and graph tables will be deleted and regenerated from source connector content after the new implementation is complete. There will be no dual-read layer, compatibility adapter, legacy row transformer, backfill migration, or rollback to the old graph schema.

Also excluded from the mandatory implementation:

- trained GNN retrieval;
- graph foundation models;
- hierarchical community reports and global map/reduce GraphRAG;
- a new graph database or external vector database;
- entity nodes in the product map;
- changes to connector authentication or source-specific PII policy;
- changes to the ASR provider or chat/summarizer generation models except where prompt and evidence contracts must change.

Community/global retrieval may be proposed later if real query logs show demand. Learned GNN retrieval may be proposed only after a labeled multi-hop benchmark shows the deterministic retrieval stack has plateaued.

## 2. Design principles

1. **Evidence is the retrieval product.** A note title or first-line display card is discovery metadata, not sufficient generation context. Every retrieved claim must resolve to a source passage or transcript span.
2. **One fact, one provenance chain.** Entities and relations never lose the original surface, passage, offsets, extraction method, confidence, or source time.
3. **The retrieval graph and UI graph are different products.** The base graph optimizes retrieval; the projection optimizes display and quality ranking. The projection is never fed back into base retrieval.
4. **No signal is accepted merely because it is top-ranked.** Dense, lexical, triple, entity, and rerank heads have calibrated floors. Empty retrieval is valid and expected.
5. **Graph traversal starts from contextual evidence.** Query-to-triple and dense passage scores create reset mass; NER-to-node is an auxiliary signal, not the sole graph entry point.
6. **Live and chat share infrastructure, not policy.** Live retrieval is local, bounded, stable, and precision-first. Chat can use broader recall, query rewriting, reranking, and agentic exploration.
7. **Full builds are deterministic.** Identical source content, model versions, configuration, and calibration artifacts produce identical passages, entities, facts, graph edges, projection edges, and build hashes.
8. **Failures do not preserve stale semantics.** When extraction fails for changed content, the new text remains searchable through passages, but old entities/facts do not remain attached to it.
9. **Optimize only after measurement.** Exact all-pairs work is acceptable at the current scale. ANN, sharding, and learned graph retrieval require measured scale or quality gates.

## 3. Target architecture

```text
connector records
      │
      ▼
source notes ──► token-aware passages + stable source offsets
      │                         │
      │                         ├──► Qwen3 passage embeddings
      │                         └──► FTS5 passage index
      ▼
GLiNER2 batch/long extraction
      │
      ├──► entity mentions ──► ambiguity-aware entity resolution
      └──► directed relation candidates ──► temporal/provenance facts
                                           │
                                           ▼
        ┌──────────────── RETRIEVAL BASE GRAPH ────────────────┐
        │ passage nodes ◄──contains──► entity nodes            │
        │ passage nodes ◄──semantic──► passage nodes           │
        │ entity nodes ──directed fact relation──► entity nodes│
        └───────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
     derived note projection       shared retrieval engine
     map / previews / quality      calibrated heads
                                   query-to-triple recognition
                                   passage + entity PPR resets
                                   PPR + path extraction
                                   rerank / MMR / abstain
                                             │
                             ┌───────────────┴──────────────┐
                             ▼                              ▼
                      live evidence pack              chat evidence pack
                      rolling grounding               agent evidence tools
                      transcript citations            source-span citations
```

### 3.1 Architectural layers

#### Layer A — source evidence

Notes remain the connector and UI document unit. Passages are immutable, content-addressed slices of note text with exact character offsets and their own lexical/dense indexes. Passage text is never synthesized.

#### Layer B — knowledge

Entity mentions retain their original spans. Canonical entity nodes group compatible mentions without rewriting or deleting those mentions. Directed facts link canonical entities and point back to the passage and assertion span that supports them. Temporal fields distinguish when a fact was observed from when it is valid.

#### Layer C — retrieval graph

The base graph contains passage nodes and entity nodes only. Query-to-triple search retrieves relation edges, not a separate disconnected “fact bonus.” Accepted triples seed their subject/object entity nodes. Dense similarity creates passage reset weights. PPR ranks passages through one coherent graph.

#### Layer D — projection

The product note graph is derived after the base graph is complete. It may collapse passage-semantic, note-entity-note, and fact paths into note edges and cap degree for usability. It has its own generation hash and is not an input to PPR.

#### Layer E — evidence

Retrieval returns an `EvidencePack`, not a list of note cards. The pack includes source passages, accepted facts, explanatory paths, document metadata, calibrated confidence, and stable citation IDs. Live and chat format different subsets of the same contract.

## 4. Storage model

The index remains SQLite plus in-memory NumPy/SciPy matrices. The disposable knowledge index moves into its own `taurine-knowledge.db`; chat history, connector configuration/authentication, session metadata, and other non-derived product state remain outside it. Each clean/full build is created in `taurine-knowledge.build-<build_id>.db`, validated, then atomically installed as the active knowledge database and loaded as one immutable in-memory snapshot. This separation makes deletion/regeneration safe and prevents a partial build from sharing tables with active queries.

Existing knowledge-index tables may be replaced because the knowledge index will be regenerated from scratch.

### 4.1 Source notes

`notes` remains the source-document table and product identity:

```sql
CREATE TABLE notes (
    note_id          TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    title            TEXT NOT NULL,
    folder           TEXT NOT NULL DEFAULT '',
    text              TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    created           REAL NOT NULL,
    modified          REAL NOT NULL,
    display_line      TEXT NOT NULL,
    ingest_salience   REAL NOT NULL DEFAULT 1.0,
    quality_rank      REAL NOT NULL DEFAULT 0.0,
    value_score       REAL NOT NULL DEFAULT 0.0,
    extraction_status TEXT NOT NULL CHECK (
        extraction_status IN ('pending', 'complete', 'failed')
    ),
    extraction_error  TEXT NOT NULL DEFAULT '',
    build_id           TEXT NOT NULL
);
```

`display_line` replaces the misleading `gist` name. It is for UI display only and is never embedded or passed as the sole evidence for generation.

### 4.2 Passages

```sql
CREATE TABLE passages (
    passage_id    TEXT PRIMARY KEY,
    note_id       TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    token_count   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    created       REAL NOT NULL,
    modified      REAL NOT NULL,
    build_id      TEXT NOT NULL,
    UNIQUE(note_id, ordinal),
    CHECK(char_start >= 0 AND char_end > char_start),
    CHECK(length(text) > 0)
);

CREATE INDEX passages_note_idx ON passages(note_id, ordinal);
CREATE VIRTUAL TABLE passages_fts USING fts5(
    passage_id UNINDEXED,
    note_id UNINDEXED,
    title,
    body,
    tokenize='unicode61'
);
```

`passage_id` is a deterministic hash of `note_id`, note `content_hash`, `char_start`, `char_end`, and the passage-builder version. Rebuilding unchanged input produces the same ID.

### 4.3 Passage embeddings

```sql
CREATE TABLE passage_embeddings (
    passage_id TEXT PRIMARY KEY REFERENCES passages(passage_id) ON DELETE CASCADE,
    model_id   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    build_id   TEXT NOT NULL
);
```

The in-memory embedding matrix preserves deterministic passage ordering by `passage_id`. The index records the exact model repository, local model revision/hash, prompt version, dimension, and normalization policy in build metadata.

Utterance embeddings, if retained by the existing ablation decision, become a separate query-expansion index keyed to a note. They are never graph nodes and never substitute for source passages.

### 4.4 Entity nodes, aliases, and mentions

```sql
CREATE TABLE entities (
    entity_id          TEXT PRIMARY KEY,
    canonical          TEXT NOT NULL,
    normalized         TEXT NOT NULL,
    etype              TEXT NOT NULL,
    df                 INTEGER NOT NULL,
    traversal_eligible INTEGER NOT NULL,
    ambiguity_key      TEXT NOT NULL,
    build_id           TEXT NOT NULL,
    UNIQUE(etype, normalized, ambiguity_key)
);

CREATE TABLE entity_aliases (
    entity_id   TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    surface     TEXT NOT NULL,
    surface_cf  TEXT NOT NULL,
    origin      TEXT NOT NULL,
    confidence  REAL NOT NULL,
    PRIMARY KEY(entity_id, surface, origin)
);

CREATE INDEX entity_alias_cf_idx ON entity_aliases(surface_cf);

CREATE TABLE entity_mentions (
    mention_id   TEXT PRIMARY KEY,
    entity_id    TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    passage_id   TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    note_id      TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    surface      TEXT NOT NULL,
    etype        TEXT NOT NULL,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    confidence   REAL NOT NULL,
    extractor    TEXT NOT NULL,
    schema_id    TEXT NOT NULL,
    build_id     TEXT NOT NULL,
    CHECK(char_start >= 0 AND char_end > char_start)
);

CREATE INDEX entity_mentions_entity_idx ON entity_mentions(entity_id);
CREATE INDEX entity_mentions_passage_idx ON entity_mentions(passage_id);
CREATE INDEX entity_mentions_note_idx ON entity_mentions(note_id);
```

Mention offsets are note-global character offsets. `mention_id` hashes note ID, content hash, type, surface, and offsets. Canonicalization never overwrites mentions.

### 4.5 Predicate registry and directed temporal facts

The predicate registry is code-owned configuration, validated at startup:

```python
PredicateSpec(
    name="works_at",
    description="A person is employed by or works for an organization",
    directional=True,
    inverse="employs",
    reverse_traversal_weight=0.35,
    cardinality="many_to_many",
    temporal=True,
    exclusive_object=False,
    base_weight=0.25,
)
```

Every predicate declares direction, optional inverse label, reverse traversal weight, temporal behavior, exclusivity, and base traversal weight. The initial registry includes the current predicates plus the audit-recommended personal-memory relations:

`works_with`, `works_at`, `member_of`, `part_of`, `located_at`, `scheduled_for`, `paid_or_owes`, `owns_or_has`, `uses`, `about`, `met_with`, `related_to`, `decided`, `prefers`, `blocked_by`, `responsible_for`, `completed`, and `supersedes`.

```sql
CREATE TABLE facts (
    fact_id             TEXT PRIMARY KEY,
    subject_entity_id   TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    predicate           TEXT NOT NULL,
    object_entity_id    TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    source_passage_id   TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    source_note_id      TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    assertion_start     INTEGER NOT NULL,
    assertion_end       INTEGER NOT NULL,
    subject_start       INTEGER NOT NULL,
    subject_end         INTEGER NOT NULL,
    object_start        INTEGER NOT NULL,
    object_end          INTEGER NOT NULL,
    confidence          REAL NOT NULL,
    extraction_method   TEXT NOT NULL,
    schema_id           TEXT NOT NULL,
    observed_at         REAL NOT NULL,
    valid_from          REAL,
    valid_to            REAL,
    invalidated_at      REAL,
    superseded_by       TEXT REFERENCES facts(fact_id),
    status              TEXT NOT NULL CHECK (
        status IN ('active', 'historical', 'superseded', 'conflicted', 'candidate')
    ),
    build_id            TEXT NOT NULL,
    CHECK(subject_entity_id <> object_entity_id),
    CHECK(assertion_start >= 0 AND assertion_end > assertion_start)
);

CREATE INDEX facts_subject_idx ON facts(subject_entity_id, predicate, status);
CREATE INDEX facts_object_idx ON facts(object_entity_id, predicate, status);
CREATE INDEX facts_passage_idx ON facts(source_passage_id);
CREATE INDEX facts_temporal_idx ON facts(observed_at, valid_from, valid_to, status);
```

`fact_id` hashes the canonical subject, predicate, object, source passage, and assertion offsets. Two sources may assert the same logical triple; they remain separate fact rows so provenance and time are not collapsed. Query-time grouping may combine their support.

Open predicate suggestions that are not in the code-owned registry never enter `facts`:

```sql
CREATE TABLE predicate_proposals (
    proposal_id        TEXT PRIMARY KEY,
    proposed_predicate TEXT NOT NULL,
    description        TEXT NOT NULL,
    subject_entity_id  TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    object_entity_id   TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    source_passage_id  TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    assertion_start    INTEGER NOT NULL,
    assertion_end      INTEGER NOT NULL,
    confidence         REAL NOT NULL,
    extraction_method  TEXT NOT NULL,
    build_id           TEXT NOT NULL
);
```

Proposals are evaluation data only. They become graph facts only after the predicate is deliberately added to the registry and a later clean build extracts/normalizes them under that registered schema.

### 4.6 Triple embeddings

```sql
CREATE TABLE fact_embeddings (
    fact_id    TEXT PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
    model_id   TEXT NOT NULL,
    text       TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    build_id   TEXT NOT NULL
);
```

The embedded text is deterministic and contains the canonical triple plus predicate description:

```text
Neil — works_at (is employed by or works for) — Wink
```

It does not use an LLM-written relationship between two notes. The original source sentence is available to the recognizer and evidence pack but is not concatenated into the canonical triple embedding.

### 4.7 Base graph edges

The base graph is stored in normalized edge tables and materialized into a sparse matrix per build:

```sql
CREATE TABLE passage_semantic_edges (
    src_passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    dst_passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    cosine         REAL NOT NULL,
    weight         REAL NOT NULL,
    build_id       TEXT NOT NULL,
    PRIMARY KEY(src_passage_id, dst_passage_id),
    CHECK(src_passage_id < dst_passage_id)
);

CREATE TABLE passage_entity_edges (
    passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    entity_id  TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    confidence REAL NOT NULL,
    mentions   INTEGER NOT NULL,
    idf        REAL NOT NULL,
    weight     REAL NOT NULL,
    build_id   TEXT NOT NULL,
    PRIMARY KEY(passage_id, entity_id)
);
```

Fact rows themselves define entity-to-entity graph edges. There is no separate projected entity/topic note edge in the base graph.

### 4.8 Product note projection

```sql
CREATE TABLE note_projection_edges (
    src_note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    dst_note_id TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL,
    shared      TEXT NOT NULL DEFAULT '[]',
    fact_ids    TEXT NOT NULL DEFAULT '[]',
    mechanisms  TEXT NOT NULL DEFAULT '{}',
    build_id    TEXT NOT NULL,
    PRIMARY KEY(src_note_id, dst_note_id),
    CHECK(src_note_id < dst_note_id)
);
```

This table is the only source for `/graph`, the map, note previews, and quality PageRank. Retrieval code must not import or query it.

### 4.9 Build metadata

```sql
CREATE TABLE index_builds (
    build_id            TEXT PRIMARY KEY,
    started_at          REAL NOT NULL,
    completed_at        REAL,
    status              TEXT NOT NULL CHECK (status IN ('building', 'complete', 'failed')),
    health              TEXT NOT NULL CHECK (health IN ('healthy', 'degraded')),
    source_snapshot     TEXT NOT NULL,
    model_manifest      TEXT NOT NULL,
    config_manifest     TEXT NOT NULL,
    calibration_hash    TEXT NOT NULL,
    passage_count       INTEGER NOT NULL DEFAULT 0,
    entity_count        INTEGER NOT NULL DEFAULT 0,
    fact_count          INTEGER NOT NULL DEFAULT 0,
    base_edge_count     INTEGER NOT NULL DEFAULT 0,
    projection_count    INTEGER NOT NULL DEFAULT 0,
    graph_hash          TEXT NOT NULL DEFAULT '',
    error               TEXT NOT NULL DEFAULT ''
);
```

Only a `complete` build can become the active in-memory index. A complete build may be marked degraded for availability, but it cannot pass release acceptance. Normal rebuilds prepare a complete new database file and swap the file/snapshot atomically; queries never observe a half-built matrix.

## 5. Passage construction

### 5.1 Token-aware segmentation

Replace `str.split()` token approximations with the bundled Qwen tokenizer for embedding passages and the bundled GLiNER2 tokenizer for extraction-window limits.

Passage construction requirements:

- target 220 Qwen tokens;
- hard maximum 300 Qwen tokens;
- 40-token overlap;
- prefer paragraph boundaries, then sentence boundaries, then clause boundaries;
- split an overlong sentence only as a last resort;
- preserve exact note-global `char_start` and `char_end`;
- never normalize or rewrite passage text;
- include the note title as embedding metadata, not inside the stored passage span;
- do not create empty or whitespace-only passages;
- deterministic output for the same text and passage-builder version.

The 220/300/40 defaults balance answer locality with enough context for personal notes and messages. They must be benchmarked against 150/220/32 and 300/420/64 variants before final calibration.

Email and iMessage connector rendering remains responsible for producing coherent source note text. Passage construction does not understand provider-specific quote or message boundaries beyond the text it receives, but it should prefer explicit newline/heading boundaries when present.

### 5.2 Display and evidence snippets

`display_line` remains the first non-empty line, clamped for UI use. Evidence snippets are generated from passages at retrieval time:

- start with the selected passage's exact text;
- optionally trim unrelated leading/trailing sentences only when source offsets are adjusted with it;
- never summarize or paraphrase;
- include note title, source, modified time, and passage offsets outside the quoted text;
- use the full passage for validation even if the UI renders a shorter preview.

## 6. GLiNER2 extraction overhaul

### 6.1 One shared model, correct APIs

Continue to use one process-cached `fastino/gliner2-large-v1` instance through `live/spans.py`. DeBERTa-v3-large is GLiNER2's encoder and requires no separate application integration.

The ingestion adapter must use:

- `batch_extract` for inputs that fit the model context;
- generic `batch_extract_long` for long notes with the combined entity/relation schema;
- `chunk_size=384` and `chunk_overlap=64` as initial GLiNER token values;
- `include_confidence=True` and `include_spans=True` in both paths;
- global note offsets returned by the long API;
- bounded batches by both note count and total tokenizer tokens.

Manual whitespace chunking in `indexer/extract.py` is removed. The obsolete module docstring claiming extraction is not wired into sync is corrected.

### 6.2 Source-aware schemas

All sources share a core entity schema:

`person`, `organization`, `project`, `product`, `place`, `event`, `date_or_deadline`, and `concept`.

The extraction adapter selects a relation schema by source family:

- **written/evergreen:** Apple Notes, Obsidian, local folders, Claude/Codex memories;
- **episodic conversation:** iMessage and live-session documents if stored;
- **email:** Gmail and Outlook.

Every family includes the core predicate registry. Descriptions change to match source language, and optional predicates are enabled where relevant. For example, email emphasizes `responsible_for`, `scheduled_for`, `blocked_by`, and `completed`; conversation emphasizes `met_with`, `decided`, `prefers`, and commitments.

Each schema has a stable `schema_id` hash of entity descriptions, predicates, and predicate descriptions. Extraction rows record it.

### 6.3 Per-type confidence thresholds

One global inference threshold may be used to avoid discarding candidates inside GLiNER2, but final acceptance is per type:

```python
ENTITY_THRESHOLDS: dict[str, float]
RELATION_THRESHOLDS: dict[str, float]
RELATION_PROMOTION_THRESHOLDS: dict[str, float]
```

The checked-in calibration artifact supplies the values. The initial implementation may seed it with the current 0.50 entity, 0.45 relation, and 0.70 promotion floors, but those numbers are not considered production-approved until the evaluation gate in §18 passes.

Three outcomes are distinct:

- below extraction floor: discard;
- above extraction floor but below promotion floor: store as `candidate`, searchable for analysis but not traversable as an asserted fact;
- above promotion floor: active fact, subject to endpoint, provenance, conflict, and temporal validation.

### 6.4 Endpoint and span validation

A relation is accepted only if:

- both endpoints resolve to retained entity mentions in the same extraction result;
- each endpoint's returned span exactly indexes its surface in source text;
- subject and object do not resolve to the same entity;
- predicate exists in the registry;
- the assertion sentence/paragraph contains both endpoint spans;
- offsets map to exactly one source passage or to two adjacent overlapping passages that share the assertion span;
- confidence clears the predicate's extraction threshold.

When GLiNER2 returns endpoint strings without offsets, the adapter performs bounded exact occurrence matching inside the extraction window. Ambiguous multiple matches are stored only as candidates unless one occurrence uniquely shares the relation's assertion sentence.

### 6.5 Extraction failure behavior

Extraction failure is isolated by note:

- passages, passage embeddings, and FTS rows remain valid;
- `notes.extraction_status='failed'` and the sanitized error are recorded;
- the failed note contributes no entity mentions or facts to the new build;
- no rows from a prior content hash are retained;
- the next sync retries extraction even when source content is unchanged;
- a build may complete with failed notes, but its health report and UI status expose the count;
- full release acceptance requires the failure rate below the gate in §18.

### 6.6 Bounded relation-completion lane

Fixed-schema extraction is the default. To address the relation-recall ceiling without turning every note into an LLM OpenIE call, a mandatory offline completion lane runs only for:

- the top 15% of changed notes by value score, with a floor of five notes per non-empty sync;
- notes containing at least two accepted entities but no active relation;
- notes whose high-value passage is selected by the labeled error-driven schema gap list.

One batched LLM call receives source passages, entity mention IDs, and the predicate registry. It may:

- select a registered predicate missed by GLiNER2;
- propose an open predicate with a short description;
- add aliases/acronyms/nicknames for existing entity mentions.

Open predicates are never immediately added to the graph. They are normalized to an existing predicate when semantically equivalent; otherwise they are stored in `predicate_proposals`. Repeated independently supported proposals are surfaced in the build report as evidence for a future deliberate registry change. Every completed fact still requires source offsets and a supporting passage.

This replaces broad note-pair adjudication as the primary accuracy backstop. Pair adjudication may remain only for the UI projection, never for constructing source facts.

## 7. Entity resolution

### 7.1 Resolution stages

Resolve mentions within each entity type through ordered stages:

1. exact canonical or exact known surface;
2. casefolded alias when the alias maps unambiguously to one entity;
3. deterministic normalization of whitespace, punctuation, separators, possessives, and common corporate suffix variants;
4. acronym/initialism match derived from canonical tokens;
5. embedding candidate generation above a calibrated threshold;
6. lexical, digit, type, and context compatibility verification;
7. high-value ambiguity resolution through the bounded completion lane.

Embedding similarity is candidate generation, never sufficient by itself.

### 7.2 Guards

- Different digit sequences never merge.
- Different entity types never merge automatically.
- Short aliases under three compact characters require explicit origin and unique mapping.
- Casefold matching is allowed for named entities only when unambiguous.
- Two distinct full names are not merged merely because they share a first or last name.
- Place, organization, and product neighbors require lexical or alias evidence.
- Pronouns and owner-name hubs are retained as provenance mentions only when useful, but default to `traversal_eligible=0` above their df cap.
- Date/deadline entities default to non-traversable unless the query expresses a temporal constraint or explicitly seeds the date.

### 7.3 Non-destructive canonicalization

Canonicalization writes entity and alias membership but never replaces `entity_mentions.surface`. Relation evidence points to mention offsets as well as canonical entity IDs. Re-resolving a mention in a later build changes only the build's canonical mapping; source provenance remains inspectable.

### 7.4 Query-time lookup

`EntityResolver.lookup_query_surface(surface, optional_type, query_context)` returns:

```python
EntityResolution(
    status="resolved" | "ambiguous" | "missing",
    candidates=[EntityCandidate(entity_id, score, reason)],
)
```

Ambiguous surfaces do not seed every candidate. They are passed to triple recognition and passage retrieval for disambiguation. Live retrieval drops unresolved ambiguity unless one candidate becomes dominant from context; chat may ask the agent or use a recognition call.

## 8. Temporal fact semantics

### 8.1 Time fields

- `observed_at`: when the supporting source was created/modified or the message occurred.
- `valid_from`: explicit or inferred beginning of validity; nullable.
- `valid_to`: explicit or inferred end of validity; nullable.
- `invalidated_at`: when Taurine determined the assertion was no longer active.
- `superseded_by`: later supporting fact that replaced it.

Relative dates are resolved only when the source has a reliable timestamp. The raw date mention and resolution confidence remain available. A low-confidence date resolution does not set `valid_from`/`valid_to`.

### 8.2 Predicate cardinality and conflict rules

The predicate registry declares whether multiple active objects are permitted. For exclusive-object predicates, a later contradictory fact does not delete history:

1. compare source times and explicit validity intervals;
2. mark the earlier row `superseded` and close `valid_to` when the newer row clearly succeeds it;
3. mark both `conflicted` when ordering or validity is ambiguous;
4. keep all provenance rows;
5. exclude conflicted facts from unqualified “current fact” answers unless the evidence pack presents the conflict explicitly.

For non-exclusive predicates, compatible objects coexist. Duplicate logical triples from different sources form a support group at retrieval time; their confidences are combined with noisy-or only after source independence checks.

### 8.3 Temporal query policy

The query analyzer identifies current, historical, and explicit-time intent:

- current intent prefers active facts valid now;
- historical intent filters/intersects `observed_at` and validity intervals;
- unspecified intent returns active facts first but may include a superseded fact when it explains a selected passage or path;
- explicit contradictions are returned with both source passages and time labels.

Recency is never a substitute for temporal validity. A recent note can describe an old fact; an old note can remain currently valid.

## 9. Deterministic build and sync orchestration

### 9.1 Build stages

One coordinator owns the complete finalization pipeline:

1. extract all enabled connector records and persist source notes;
2. write connector-derived salience before ranking or value gating;
3. detect new/changed/removed source notes by content hash;
4. construct passages and FTS rows;
5. embed passages and retained utterance aliases;
6. run GLiNER2 batch/long extraction;
7. resolve entities and aliases;
8. construct and temporally reconcile initial facts;
9. build provisional passage semantic/contains/fact edges and a provisional projection;
10. compute provisional quality/value scores used only for the relation-completion gate;
11. run bounded relation completion and re-resolve accepted registered-predicate facts;
12. rebuild final base graph edge tables and materialize the base graph matrix;
13. derive the final note projection;
14. compute final quality and value ranks;
15. build active fact embeddings and predicate-proposal reporting;
16. validate invariants, calculate graph hash, and mark the build complete;
17. atomically install the completed knowledge database and reload query matrices once.

`sync_all_enabled` may run connectors sequentially if provider constraints require it, but it invokes global finalization exactly once. `sync_source` no longer computes quality, values, adjudication, fact embeddings, or graph edges independently per connector.

### 9.2 Full versus incremental work

Source extraction, passage generation, embeddings, and GLiNER results remain content-addressed and incremental. **Graph finalization is full and deterministic** at Taurine's current scale:

- re-resolve all active entity mentions;
- recompute entity df and traversal eligibility;
- rebuild all base graph edge tables;
- rebuild the entire projection;
- recompute rank vectors;
- prune all zero-reference and stale derived rows.

There is no `rebuild_for(note_id)` in the active pipeline. It is removed after callers and tests switch to the build coordinator.

If full finalization exceeds the performance gate for three consecutive representative builds, the next design may introduce affected-neighborhood rebuilding. It must preserve a scheduled full reconciliation and deterministic equivalence tests.

### 9.3 Determinism

- sort all source IDs, passage IDs, entity candidates, fact IDs, and edge candidates before reduction;
- give every tie-break an explicit stable key;
- do not rely on set/dict iteration order for IDs or edge selection;
- pin model revisions and record local artifact hashes;
- version passage, normalization, schema, calibration, and graph algorithms;
- run the same build twice in CI fixtures and compare the complete graph hash.

### 9.4 Atomicity and cleanup

A build uses a new `build_id` and a new temporary knowledge database file. All rows are written with that ID, validated, then installed through an atomic file/snapshot swap. Failed build files are never queryable and may be removed on next startup. Activation reloads passage, fact, alias, graph, projection, and rank matrices as one `IndexSnapshot` object so a query cannot mix generations.

Successful finalization asserts:

- no passage references a missing note;
- no mention references a missing passage/entity;
- no fact references missing entities/passages;
- no self-facts;
- no zero-reference entities;
- no fact embedding without a fact;
- no projection edge without two active notes;
- all sparse-matrix columns are normalized or marked dangling;
- stored counts equal in-memory counts;
- active build hash matches recomputation.

## 10. Base graph construction

### 10.1 Nodes

- one node for every active passage;
- one node for every canonical entity with at least one active mention;
- no note nodes;
- no projected note edges;
- no orphan entities;
- no separate fact nodes.

Notes are aggregated after passage ranking. Facts are directed entity edges and independently retrievable triple units.

### 10.2 Passage-to-entity edges

For passage `p` containing entity `e`:

```text
idf(e) = log((N + 1) / (df(e) + 1)) + 1
mention_strength = max mention confidence in p
count_bonus = 1 + log(1 + min(mention_count, 3))
contains_weight = calibrated_clip(idf × mention_strength × count_bonus)
```

Weights are calibrated to a bounded graph range after evaluation, not normalized globally before the sparse matrix. Non-traversable entities retain provenance rows but produce no base-graph contains edge unless explicitly enabled by a query-specific matrix overlay.

### 10.3 Directed fact edges

For each active fact:

- forward edge: `predicate.base_weight × fact.confidence × temporal_weight`;
- reverse edge: forward weight multiplied by `predicate.reverse_traversal_weight`;
- `related_to` remains symmetric and downweighted;
- superseded/historical facts are excluded from the default current matrix but available in a temporal query overlay;
- conflicted facts are excluded unless the query asks about history/conflict or both alternatives are explicitly seeded;
- multiple provenance rows for the same logical triple combine only at matrix construction, retaining fact IDs for evidence.

The sparse matrix may contain both directions, but they are not equal unless the predicate is symmetric. Evidence paths always render the original predicate direction.

### 10.4 Passage semantic edges

At current scale, compute exact cosine similarities in a batched matrix multiplication. For each passage:

- collect candidates above the calibrated semantic floor;
- keep its top `BASE_GRAPH_PASSAGE_K` candidates;
- retain the undirected union of directed top-k selections;
- optionally require mutual top-k only if evaluation improves precision without harming recall;
- exclude same-note adjacent-overlap edges, which add no cross-document retrieval value;
- flag same-source near-duplicates for quality-rank exclusion, not retrieval deletion.

This is a retrieval candidate cap, not a product degree cap. Entity and fact paths remain independent, and there is no global greedy degree-12 cut.

### 10.5 Hub policy

Entity traversal eligibility is computed per type using labeled and df-based rules:

- dates/deadlines: off by default;
- pronouns/generic people: off;
- concepts: higher df tolerance than named entities, with a stoplist;
- owner/self aliases: off when above their cap unless explicitly query-seeded;
- exact query entity: may be temporarily enabled even when normally a hub;
- every hub decision is logged with type, df, cap, and reason.

Rarity weights do not alone solve hub traversal because column normalization can cancel scalar differences. Eligibility and query-specific activation are required.

### 10.6 PPR correctness

The base matrix is column-stochastic. PPR implements:

```text
r_next = damping × (P @ r + dangling_mass × teleport)
         + (1 - damping) × teleport
```

Requirements:

- total rank mass remains 1 within tolerance on every iteration;
- dangling mass is redistributed to the current teleport vector;
- warm starts are renormalized and discarded when the active snapshot changes;
- empty/unknown seeds produce an explicit empty result, not a fabricated ranking;
- passage/entity reset weights are separately configurable and calibrated;
- output includes passage mass, entity mass for diagnostics, convergence iterations, residual, and top incoming contributions;
- temporal overlays use the same invariants.

### 10.7 Note projection

After the base graph is complete, derive note-pair mechanisms:

- aggregate cross-note passage semantic edges using the maximum plus bounded supporting-edge bonus;
- collapse note-entity-note paths only for traversal-eligible specific entities;
- add directed-fact support between notes that contain the fact endpoints/provenance;
- combine mechanisms with noisy-or while retaining the mechanism map and fact IDs;
- construct a maximum-spanning forest first to avoid gratuitous isolated components;
- add per-note strongest remaining edges until `PROJECTION_MAX_DEGREE`;
- use deterministic tie-breaks;
- never feed the result back into base retrieval.

Quality PageRank runs on this projection with dangling redistribution and the same-source duplicate guard. Its source/recency teleport and final impact remain behind an ablation flag until §18 shows positive retrieval value.

## 11. Shared retrieval engine

Create `taurine_core/retrieval/` as the one shared retrieval subsystem. `live/` and `chat/` become policy adapters and consumers rather than independent ranking implementations.

### 11.1 Core contracts

```python
@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    mode: Literal["live", "chat", "tool"]
    now: float
    conversation_context: tuple[str, ...] = ()
    temporal_intent: TemporalIntent | None = None

@dataclass(frozen=True)
class HeadHit:
    target_kind: Literal["passage", "fact", "entity", "note"]
    target_id: str
    raw_score: float
    calibrated_score: float
    rank: int
    head: str
    reason: str

@dataclass(frozen=True)
class RetrievalDecision:
    accepted: bool
    confidence: float
    reason: str
    accepted_heads: tuple[str, ...]
    rejected_heads: tuple[str, ...]

@dataclass(frozen=True)
class EvidencePassage:
    evidence_id: str
    passage_id: str
    note_id: str
    title: str
    source: str
    text: str
    char_start: int
    char_end: int
    created: float
    modified: float
    retrieval_score: float
    rerank_score: float
    path_ids: tuple[str, ...]

@dataclass(frozen=True)
class EvidenceFact:
    fact_ref: str
    fact_ids: tuple[str, ...]
    subject: str
    predicate: str
    object: str
    direction: str
    status: str
    observed_at: float
    valid_from: float | None
    valid_to: float | None
    confidence: float
    source_evidence_ids: tuple[str, ...]

@dataclass(frozen=True)
class EvidencePack:
    decision: RetrievalDecision
    query: RetrievalQuery
    passages: tuple[EvidencePassage, ...]
    facts: tuple[EvidenceFact, ...]
    paths: tuple[EvidencePath, ...]
    notes: tuple[EvidenceNote, ...]
    trace: RetrievalTrace
```

An abstained pack has an empty passages/facts/paths payload. Note cards are derived from selected passages; they are not independent evidence.

### 11.2 Retrieval heads

Every query runs the heads allowed by its mode:

1. **Dense passage head** — Qwen3 query embedding against passage embeddings.
2. **BM25 passage head** — FTS5 over passage body/title.
3. **Exact alias head** — exact word-boundary note/entity aliases, credited once.
4. **Query entity head** — GLiNER2 query spans plus ambiguity-aware entity resolution.
5. **Triple head** — query embedding against canonical fact embeddings.
6. **Utterance head** — retained only if the prior ablation decision and new benchmark show positive lift; it contributes note/passsage candidates but never evidence text.

Live may additionally formulate span-context dense queries for ASR recovery. Fuzzy snapping produces an entity-resolution candidate and reason; it does not silently rewrite the query.

### 11.3 Score calibration

Remove query-local min-max normalization as a relevance estimator. It guarantees a best score even when every candidate is bad.

Each head owns a checked-in calibration curve fit on the training portion of the labeled evaluation set. Use monotonic isotonic regression when sample size is sufficient; otherwise use a threshold plus piecewise-linear mapping. The artifact contains:

- model/index version;
- source/mode applicability;
- raw-score knots and calibrated relevance probabilities;
- acceptance floor;
- training sample count and date;
- held-out metrics;
- artifact hash.

Unknown model/config hashes fail closed to conservative default floors and emit a calibration mismatch status. They do not reuse an incompatible curve.

### 11.4 Candidate fusion

RRF may remain as a tie-stable recall combiner, but it no longer erases confidence:

```text
fusion_score(target) =
    Σ accepted_head_weight[h]
      × calibrated_score(h, target)
      × 1 / (RRF_K + rank_h(target))
```

Only accepted head hits participate. Exact aliases contribute once. Triple hits do not add raw cosine to endpoint notes. Source/recency/quality priors are applied after reranking as bounded features, not multiplicative overrides, and only if their ablations pass.

### 11.5 Abstention

`AbstentionPolicy` makes an explicit decision before PPR and again after reranking.

Pre-PPR acceptance requires at least one of:

- a dense passage above its mode-specific floor;
- a BM25 hit above its floor with sufficient query-token coverage;
- an exact alias/entity resolution that is unambiguous;
- a recognized triple above its floor.

Post-rerank acceptance requires:

- at least one evidence passage above the evidence floor; and
- either a second independent head agrees, a high-precision exact/entity match exists, or the top rerank score exceeds the single-head override threshold.

The calibrated confidence combines top evidence probability, head agreement, score margin, and recognized-triple confidence through a small fitted logistic model. If the calibration set is too small for that model, use the explicit conservative rule table checked into the artifact.

Required outcomes:

- filler, silence-like text, and unrelated gibberish usually abstain;
- an abstention returns no arbitrary notes/facts;
- live keeps the prior grounding display but marks `no_new_evidence` and does not treat it as support for new summary claims;
- chat may answer from conversation history when appropriate, but must state that the knowledge base supplied no evidence;
- debug traces include the exact abstention reason and closest rejected scores.

### 11.6 Query-to-triple recognition

This replaces additive fact cosine bonuses and implements the HippoRAG 2 entry pattern.

1. Embed the full query with the mode-specific Qwen instruction.
2. Retrieve the top `TRIPLE_RECALL_K` active fact embeddings by cosine.
3. Assemble each candidate's canonical triple, predicate description, source assertion sentence, source time, and current/historical status.
4. Run recognition filtering.
5. Retain at most `TRIPLE_SEED_K` recognized triples.
6. Aggregate each recognized triple's score into its subject and object entity reset weights.
7. Preserve the accepted fact IDs for evidence assembly and path rendering.

Recognition has two profiles behind one interface:

- **live:** bundled `Qwen/Qwen3-Reranker-0.6B` scores query/triple+assertion pairs locally. A triple must clear the calibrated live recognition floor and any temporal/entity compatibility checks. If the reranker exceeds its time budget or is unavailable, use the stricter embedding floor plus query-entity overlap; never accept all top-k by default.
- **chat/tool:** the same reranker processes a wider candidate set. Candidates in a calibrated ambiguity band may be included in one batched recognition request to the current chat-capable LLM, which returns accepted fact IDs only. Failure falls back to the local decision.

The recognizer may reject a semantically similar but contextually irrelevant triple. It may not create a new triple.

If no triple survives, retrieval falls back to dense/BM25/exact passage retrieval and does not run entity-seeded PPR unless query entities independently resolve with high confidence.

### 11.7 Passage reset probabilities

For corpora up to 10,000 passages, score every passage embedding against the query so the graph follows HippoRAG 2's broad passage reset principle. Convert raw cosines through the dense calibration curve:

```text
passage_reset[p] = calibrated_dense_relevance(query, p)
```

Values below the dense reset floor become zero; accepted values are not truncated to only the top ten. For larger corpora, an ANN candidate set may approximate this only after a recall-equivalence gate.

The teleport vector combines:

```text
teleport = normalize(
    PASSAGE_RESET_W × passage_reset
  + ENTITY_RESET_W × recognized_triple_entity_reset
  + QUERY_ENTITY_RESET_W × independently_resolved_query_entities
  + SESSION_RESET_W × mode-specific session accumulator
)
```

Session history never replaces current-query passage reset mass.

### 11.8 PPR and candidate aggregation

Run PPR once on the active base graph snapshot. Candidate passages are the union of:

- passages accepted by dense/BM25/exact heads;
- top PPR passages;
- source passages supporting recognized triples;
- passages on selected explanatory paths.

Aggregate passage candidates to notes only after passage scoring. A note's score uses its strongest passage plus a bounded bonus for distinct supporting passages. This prevents a long note with many mediocre passages from dominating by volume.

### 11.9 Explanatory paths

For each graph-expanded passage not directly retrieved, calculate up to two paths from recognized entity or high-confidence passage seeds:

- maximum 4 graph edges;
- highest product of normalized edge weights, implemented as shortest path over negative log weights;
- preserve edge direction and fact IDs;
- penalize hub entities and repeated passages from the same note;
- reject paths containing inactive/conflicted facts unless temporal intent allows them;
- deduplicate paths with the same semantic explanation.

Paths are evidence organization, not independent truth. Every fact edge on a rendered path must have source passage evidence included or citeable in the pack.

### 11.10 Reranking and evidence selection

Use the bundled Qwen3 reranker for both triple recognition and passage evidence reranking, with separate prompts/calibration. Rerank at most:

- 30 passage candidates live;
- 60 passage candidates chat/tool.

Rerank input contains query, exact source passage, title, source, and a compact path/fact hint when graph-expanded. It never uses a generated gist.

After reranking, apply MMR/source diversity:

```text
selection = relevance
          - MMR_LAMBDA × max_similarity_to_selected
          - same_note_redundancy_penalty
```

Defaults:

- live: up to 8 evidence passages, 8 note cards, 12 facts, 8 paths;
- chat Stage A: up to 12 evidence passages, 12 note cards, 16 facts, 12 paths;
- at most 2 passages per note initially;
- a third passage is allowed only when required for a multi-part answer and the token budget permits it.

Final cutoffs are set by the evaluation artifact, not by min-max rank.

### 11.11 Priors

Recency, source rank, and graph quality become bounded rerank features:

- recency applies to episodic source relevance but does not invalidate facts;
- source rank is a weak tie-break, not a hard trust claim;
- quality rank may add at most the calibrated equivalent of a small rerank margin;
- each prior can be disabled independently;
- production enablement requires positive held-out nDCG/recall and no temporal regression.

Remove the current raw multiplicative chain that can push a less relevant note above stronger evidence without a calibrated interpretation.

## 12. Evidence identities, registries, and citation validation

### 12.1 Stable IDs

Each live session/chat owns registries:

- `N#` — user-facing note/document;
- `E#` — exact source passage/evidence span;
- `F#` — logical fact support group with provenance rows;
- `P#` — explanatory graph path;
- `T#` — transcript chunk/span in live summaries.

IDs are assigned once, persisted where the current chat/session contract persists them, and never reused. Every tool and retrieval response uses the same registry service. Raw note/passage/fact IDs may appear in debug logs but not as the model's primary citation syntax.

### 12.2 Evidence pack formatting

The model context renders exact evidence:

```text
[E3] [N2] "Project roadmap" (apple_notes, modified 2026-07-08, chars 842-1134)
"The recommendation engine launch moved to July 18 because ..."

[F4] Armaan — responsible_for — recommendation engine review
status=current; observed=2026-07-08; supported by [E3]

[P2] [E1] -> Armaan -> responsible_for -> recommendation engine -> [E3]
```

The full evidence text sent to the model is retained in the turn/session record by evidence ID or reproducible snapshot reference so citations can be audited later.

### 12.3 Validation stages

All generated factual outputs pass:

1. **Identifier validity:** every citation exists in the registry.
2. **Availability:** cited `E#`/`T#` text was actually supplied or read during the turn.
3. **Coverage:** every factual sentence has at least one `E#`, `F#`, or `T#`; `N#` alone is navigation, not factual evidence.
4. **Fact grounding:** every cited `F#` resolves to one or more supplied/citeable source passages.
5. **Entailment:** run a local `cross-encoder/nli-deberta-v3-base` claim/evidence check. A claim must clear the calibrated entailment floor against at least one cited evidence set.
6. **Temporal consistency:** current-tense claims cannot rely only on superseded/historical/conflicted facts without qualifying language.

Code blocks, headings, questions, explicit uncertainty, and non-factual conversational text are excluded from the coverage check through a deterministic sentence classifier. The classifier is tested on a labeled fixture set.

Unknown citations are never silently stripped while leaving the claim. Unsupported factual sentences are removed or cause the output to be rejected, depending on mode:

- live summary: retain the previous accepted summary and log rejection;
- chat: perform one constrained repair round using only the same evidence, then remove unsupported claims if repair fails;
- actionable: retain the existing quote/confidence/grounding gates and require a `T#` quote span.

A summary/answer with factual content and zero evidence citations is invalid.

### 12.4 Source UI semantics

Persist and render three separate sets:

- **retrieved:** considered by the retrieval engine;
- **read/supplied:** exposed to the answer model;
- **cited:** actually referenced in the accepted output.

The primary source rail shows cited sources first and labels uncited read sources as “consulted,” not “sources for this claim.” Retrieval-debug UI may show the broader candidate set.

## 13. Live transcription integration

### 13.1 Window construction

The live query window becomes time-aware and sequence-aware:

- include final chunks newest-first until both at least 3 chunks and at least `LIVE_WINDOW_MIN_S=12` seconds are covered;
- stop at `LIVE_WINDOW_MAX_S=45`, `LIVE_WINDOW_MAX_CHARS`, or the current topic boundary;
- include active partials only for dense/BM25 query formation, never as citeable transcript evidence;
- include up to two preceding final chunks for GLiNER span context;
- assign every final transcript chunk a monotonic sequence and stable `T#` registry ID.

This removes the unused `C2_WINDOW_MIN_S` state and makes its intended behavior real.

### 13.2 Live retrieval policy

Every throttle pass:

1. build the time-aware window;
2. run whole-window dense/BM25 heads;
3. run GLiNER query spans, ambiguity-aware resolution, and bounded ASR fuzzy candidates;
4. run local query-to-triple recognition;
5. make the pre-PPR abstention decision;
6. when accepted, create passage/entity/session reset mass and run PPR;
7. rerank and make the post-rerank abstention decision;
8. select an evidence pack;
9. update the live accumulator and grounding state only from calibrated accepted evidence.

Exact aliases enter fusion/teleport once. The current second max-score injection is removed.

### 13.3 Session accumulator and topic shifts

The accumulator stores accepted passage/note evidence scores, not arbitrary top-ranked results. On each accepted pass:

- decay prior mass by the calibrated live decay;
- add current accepted passage scores;
- cap per-note accumulated mass;
- prune below epsilon;
- sharply reduce old mass when the query-window embedding crosses the topic-shift threshold and the new evidence set confirms the pivot.

On abstention, do not add or decay evidence merely because a timer fired. A strong explicit topic boundary may still decay the old display state, but that is presentation lifecycle, not evidence for a summary.

### 13.4 Grounding state

The live event carries:

```python
LiveGroundingUpdate(
    decision: RetrievalDecision,
    notes: list[EvidenceNote],
    passages: list[EvidencePassage],
    facts: list[EvidenceFact],
    paths: list[EvidencePath],
    revision: int,
    no_new_evidence: bool,
)
```

Hysteresis compares calibrated note/evidence scores. An incumbent may remain visible during an abstained pass, but it is marked prior context and is excluded from new-summary support unless reaccepted.

### 13.5 Transcript delta checkpointing

Replace `_transcript_delta: str` consumption with a sequence ledger:

```python
TranscriptLedger(
    chunks: deque[SequencedChunk],
    last_accepted_summary_seq: int,
    in_flight_through_seq: int | None,
)
```

A summary request snapshots all unconsumed final chunks through sequence `S`. New chunks may arrive and old render buffers may trim while the call runs. On accepted summary, advance `last_accepted_summary_seq` to `S`; chunks with sequence greater than `S` remain pending. On no-update or rejection, follow the explicit trigger policy without deleting unconsumed chunks. No string prefix comparison or buffer-clear fallback remains.

### 13.6 Summarizer input and output

Input:

- previous accepted summary;
- citeable transcript chunks `[T#]` since the accepted checkpoint;
- selected external passages `[E#]` with note IDs `[N#]`;
- accepted facts `[F#]` with temporal status and source evidence;
- explanatory paths `[P#]` where useful;
- open actionables and their transcript/evidence grounding.

Output remains one structured summarizer call but uses sentence-level citations. Transcript-derived claims cite `T#`; personal-KB claims cite `E#`/`F#`. `N#` may accompany a passage for navigation but cannot satisfy coverage by itself.

After generation, run §12 validation. A rejected summary leaves both previous summary and transcript checkpoint unchanged. An accepted `no_update` follows trigger backoff but does not falsely mark unsummarized chunks consumed unless the model contract explicitly declares `consumed_through` and validation confirms it. The default is to keep them pending until a material accepted update.

### 13.7 Live latency budget

Warm local p95 targets on the reference development machine:

- query GLiNER2: ≤250 ms;
- dense/BM25/entity/triple recall combined: ≤150 ms;
- local triple recognition + passage rerank: ≤350 ms;
- PPR/path/evidence assembly: ≤75 ms;
- total retrieval excluding summary LLM: ≤750 ms.

The eight-second cadence makes this practical, but timeouts are still explicit. A timed-out recognizer/reranker falls back to conservative accepted heads and records degraded mode; it does not fill the pack with unfiltered results.

## 14. Chat integration

### 14.1 Conversation-aware retrieval query

Before Stage A, build a standalone retrieval query:

- use the current message unchanged when it is independently meaningful;
- include the last two user turns and resolved entity names when the message contains pronouns, ellipsis, or follow-up markers;
- run a single cheap structured rewrite with the configured flash model only for detected dependent queries;
- require the rewrite to preserve the user's intent and named entities;
- fall back to deterministic concatenation when rewrite fails;
- retain both original and rewritten query in the retrieval trace.

The answer model always receives the original user message. The rewrite is retrieval-only.

### 14.2 Stage A evidence pack

Stage A runs the shared engine in `chat` mode and supplies answer-bearing evidence immediately:

- up to 12 selected passages;
- source note metadata;
- accepted current/historical facts;
- compact explanatory paths;
- explicit retrieval confidence or abstention;
- no first-line-only note cards presented as evidence.

When Stage A abstains, the model is told that no KB evidence cleared the threshold and should use tools or state insufficiency. It must not infer personal facts from note titles.

### 14.3 Tool redesign

Replace the current three-tool surface with evidence-native tools:

#### `search_evidence`

```json
{
  "query": "self-contained query",
  "k": 8,
  "temporal": "current|historical|any"
}
```

Returns registered `E#`, `F#`, `P#`, and `N#` results with exact snippets, scores, and time/provenance. It runs a new shared retrieval query without decaying the turn accumulator.

#### `read_note`

```json
{
  "note": "N3",
  "query": "what to find in this note",
  "cursor": "optional opaque cursor",
  "max_chars": 6000
}
```

Returns the most query-relevant registered passages from anywhere in the note plus pagination. It never claims to return full text when truncated. Raw sequential pages are available through the cursor for explicit reading requests.

#### `expand_evidence`

```json
{
  "evidence": "E4|F2|N3",
  "direction": "outgoing|incoming|both",
  "max_hops": 2,
  "limit": 8
}
```

Returns registered neighboring passages, directed facts, and explanatory paths. It does not expose a flat uncited raw note-ID list.

All tools use one `EvidenceRegistry`. Every result that the model can use has a stable citation ID and is captured as read/supplied evidence.

### 14.4 Tool budget and evidence sufficiency

Keep a hard tool-call budget, but make it evidence-aware:

- Stage A should answer common single-note and multi-note questions without tools;
- tool calls may batch multiple evidence reads where the provider supports parallel calls;
- before the final no-tools round, the engine computes whether at least one supplied evidence set supports the requested answer;
- if evidence remains insufficient, instruct the model to state the gap rather than fabricate an answer.

The initial default remains six calls, subject to evaluation. The number is not raised to compensate for tools that return poor context.

### 14.5 Chat accumulator and regeneration

- store accumulator state by accepted user turn, not tool subquery;
- condition retention on topic similarity;
- entity/fact seeds are query-local unless reintroduced;
- snapshot retrieval state before each turn;
- regeneration restores the pre-turn snapshot so it cannot double-advance seeds or citation registries;
- search tools may extend the registry but do not permanently bias later turns unless their evidence is cited/accepted in the completed answer.

### 14.6 Finalization

Run §12 validation, perform at most one constrained repair round, and persist:

- original and rewritten query;
- retrieval decision and confidence;
- candidate/read/cited evidence IDs;
- accepted output and stripped/repaired claims;
- tool traces and latency;
- snapshot/build ID used for the turn.

Only cited evidence appears in the primary answer source list. Consulted evidence is stored separately.

## 15. Repository implementation boundaries

### 15.1 New modules

```text
core/taurine_core/
  indexer/
    passages.py          token-aware passage construction + offsets
    extract.py           GLiNER2 batch/long adapter only
    entities.py          mention normalization and entity resolution
    facts.py             predicate registry, provenance, temporal reconciliation
    build.py             global deterministic build coordinator
    base_graph.py        contains/semantic/fact matrix construction
    projection.py        derived note graph only
    rank.py              quality rank for projection; no query retrieval
    store.py             persistence facade, split internally where useful
  retrieval/
    __init__.py
    types.py             query/hit/decision/evidence contracts
    snapshot.py          immutable active index snapshot
    heads.py             dense, BM25, exact, entity recall
    triples.py           fact index + query-to-triple recognition
    calibration.py       score curves, floors, abstention policy
    ppr.py               typed resets, dangling-correct PPR
    paths.py             bounded provenance-preserving path extraction
    rerank.py            local Qwen reranker batching
    evidence.py          fusion, MMR, pack assembly
    registry.py          N/E/F/P/T stable IDs
    validate.py          coverage, provenance, NLI, temporal checks
  live/
    ambient.py           live policy adapter around retrieval engine
    grounding.py         live accumulator/hysteresis only
    ingest.py            sequenced transcript ledger/checkpoints
    summarizer.py        evidence-pack prompt + validation integration
  chat/
    retrieval.py         chat policy adapter/query rewrite
    tools.py             evidence-native tool definitions/executor
    engine.py            streaming agent loop + evidence finalization
```

Avoid creating large parallel “v2” modules that leave both architectures active. Implement behind interfaces and delete replaced code after each caller switches.

### 15.2 Existing responsibilities removed

- `indexer/graph.py` no longer owns entity canonicalization, extraction mutation, retrieval graph creation, and projection in one file; its remaining projection behavior moves to `projection.py`.
- `rank.HeteroGraphMatrix` moves to `retrieval/ppr.py` and builds only from base graph tables.
- `live/ambient.run_head_fusion` moves to `retrieval/heads.py`/`evidence.py`.
- `live/grounding.py` loses fact dense search, raw fact bonuses, registry implementation, and matrix construction.
- `chat/retrieval.py` loses duplicated head orchestration.
- note `gist` fields are removed from retrieval prompts.
- `rebuild_for` and per-source finalization are removed.
- all manual GLiNER long chunk helpers are removed.

### 15.3 Public/store interfaces

The storage facade exposes typed methods, not raw cross-layer SQL:

```python
Index.source_notes()
Index.passages(note_id=None)
Index.passage_embeddings()
Index.entity_mentions(passage_id=None, note_id=None, entity_id=None)
Index.resolve_entity(surface, etype=None, context="")
Index.active_facts(temporal_intent)
Index.fact_embeddings()
Index.base_graph_rows(temporal_intent)
Index.note_projection()
Index.active_snapshot_id()
Index.activate_snapshot(database_path, build_id)
```

Retrieval consumers depend on `RetrievalSnapshotPort`, not on the mutable build connection.

## 16. Configuration

Retire or replace current knobs that encode the flawed architecture:

- retire `GRAPH_MAX_DEGREE` from retrieval; rename it `PROJECTION_MAX_DEGREE`;
- retire additive `PPR_BLEND` as the final ranking formula;
- retire endpoint fact bonus logic and any fact bonus weight;
- retire `GROUNDING_FACTS_MAX` semantics based on unconditional fact hits;
- retire query-local min-max normalization;
- retire `C2_WINDOW_CHUNKS` as the only window bound;
- retire `rebuild_for` controls;
- retain `PPR_DAMPING` but recalibrate;
- retain source/recency/quality knobs behind ablation flags.

New configuration families:

```python
# passage construction
PASSAGE_TARGET_TOKENS = 220
PASSAGE_MAX_TOKENS = 300
PASSAGE_OVERLAP_TOKENS = 40

# GLiNER long extraction
GLINER_LONG_CHUNK_TOKENS = 384
GLINER_LONG_OVERLAP_TOKENS = 64
GLINER_BATCH_TOKEN_BUDGET = 3072

# base graph
BASE_GRAPH_PASSAGE_K = 12
FACT_REVERSE_WEIGHT_DEFAULT = 0.35

# HippoRAG-style query seeding
TRIPLE_RECALL_K = 20
TRIPLE_SEED_K = 5
PASSAGE_RESET_W = 1.0
ENTITY_RESET_W = 1.0
QUERY_ENTITY_RESET_W = 0.5
SESSION_RESET_W = 0.35

# rerank / evidence
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
LIVE_RERANK_CANDIDATES = 30
CHAT_RERANK_CANDIDATES = 60
LIVE_EVIDENCE_PASSAGES = 8
CHAT_EVIDENCE_PASSAGES = 12

# validation
EVIDENCE_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# live window
LIVE_WINDOW_MIN_S = 12.0
LIVE_WINDOW_MAX_S = 45.0
LIVE_WINDOW_MAX_CHARS = 6000
```

Semantic, head-acceptance, triple-recognition, rerank, and entailment floors live only in the versioned calibration artifact; they are not duplicated as in-code constants. The implementation plan must include creation of an initial checked-in artifact before enabling the associated feature.

## 17. Error handling and degraded operation

### 17.1 Model availability

- missing passage embedding model: build fails; no searchable index can be activated;
- missing GLiNER2: passages/dense/FTS may build with `status='complete', health='degraded'`, but the build cannot satisfy final release acceptance; entity/fact graph remains empty rather than stale;
- missing reranker live: use strict calibrated recall floors and mark degraded; do not accept unfiltered triples in the ambiguity band;
- missing reranker chat: allow local recall plus agent tools, with explicit low-confidence status;
- missing NLI validator: live factual summary updates fail closed; chat may return a clearly marked evidence excerpt response without synthesized factual claims.

### 17.2 Build failures

Every stage records started/completed counts, latency, and sanitized errors. A failed build never replaces the active snapshot. Individual note extraction failures are allowed as §6.5 defines, but invariant failures abort activation.

### 17.3 Query failures

Head failures are isolated and appear in `RetrievalTrace`. The decision policy knows which heads were unavailable and uses a stricter fallback. A graph/PPR failure falls back to calibrated direct evidence retrieval; it never returns the old projection as a hidden substitute.

### 17.4 Privacy

Session and retrieval logs store IDs, scores, reasons, offsets, model/config hashes, and optionally redacted previews. Full private passage text is not duplicated into general logs. The accepted chat/session record may reference an immutable evidence snapshot in the local DB for auditing.

## 18. Evaluation and release gates

### 18.1 Labeled datasets

Create versioned JSONL datasets under `core/evals/graph_retrieval/`:

1. `extraction.jsonl` — 150-250 stratified notes with entity spans/types, relation endpoints/predicates/direction, and temporal fields.
2. `retrieval_live.jsonl` — real or privacy-safe transcript windows with relevant passage/note/fact IDs plus null/filler windows.
3. `retrieval_chat.jsonl` — single-hop, multi-hop, temporal, conflict, global-like, follow-up, and unanswerable questions.
4. `citations.jsonl` — claims, supplied evidence, correct citations, entailment labels, and factual/non-factual sentence labels.
5. `entity_resolution.jsonl` — aliases, case variants, ASR errors, acronyms, ambiguous names, non-merges, and digit guards.

Split by source note, not individual query, so near-duplicate evidence cannot leak across train/calibration and held-out evaluation.

### 18.2 Extraction gates

- entity micro precision ≥0.90 and no entity type below 0.82 precision;
- entity macro recall must improve over the current fixed 0.50 threshold baseline;
- promoted relation precision ≥0.88 overall and ≥0.82 for every enabled predicate;
- direction accuracy ≥0.97 among correct promoted relations;
- long-document global span mapping = 100% exact on fixtures;
- entity canonicalization merge precision ≥0.98;
- no regression in digit/non-merge guard fixtures;
- representative-note extraction failure rate <1% excluding deliberately corrupted input.

Candidate facts below promotion thresholds are not counted as asserted graph facts.

### 18.3 Retrieval gates

Report at minimum Recall@5, Recall@12, nDCG@10, MRR, support-set recall, and latency.

Mandatory comparisons:

- dense passage only;
- dense + BM25;
- - exact/entity;
- - query-to-triple recognition;
- - PPR;
- - rerank/MMR;
- - priors;
- current production retriever frozen before overhaul.

Release requires:

- passage Recall@12 ≥0.90 on answerable held-out chat queries or at least +10 absolute points over the strongest non-graph baseline when the corpus cannot support 0.90;
- multi-hop complete support-set recall improves by at least 15% relative over dense+BM25;
- nDCG@10 does not regress on single-hop questions versus dense+BM25;
- null false-grounding rate ≤5%;
- unanswerable chat false-answer rate ≤5% when the model follows the supplied decision contract;
- graph features show positive held-out lift; features without lift remain disabled;
- the full evidence pack stays within mode token budgets.

### 18.4 Citation gates

- factual sentence citation coverage ≥0.98;
- citation entailment precision ≥0.95 on held-out labels;
- invalid/unknown citation acceptance = 0;
- factual output with zero evidence citations acceptance = 0;
- primary source rail contains no uncited item labeled as supporting evidence;
- current-tense claims supported only by superseded facts acceptance = 0.

### 18.5 Determinism and integrity gates

- identical two-pass rebuilds produce identical graph hashes;
- randomized connector iteration order produces the same hash;
- deleting a note removes every dependent passage, mention, fact, embedding, base edge, and projection reference;
- zero orphan entities and zero stale fact embeddings after finalization;
- PPR mass remains 1 within `1e-8` on graphs with dangling and isolated nodes;
- projection rebuild from the same base graph is identical;
- failed extraction never exposes prior-content semantic rows.

### 18.6 Performance gates

On the representative current corpus and development machine:

- full graph finalization after cached extraction/embeddings ≤60 seconds;
- warm live retrieval p95 ≤750 ms;
- warm chat Stage A p95 ≤1.5 seconds excluding optional LLM query rewrite;
- PPR + path extraction p95 ≤75 ms;
- memory use remains compatible with the packaged desktop sidecar;
- cold model load is measured separately and surfaced in startup status.

If a quality gate conflicts with a performance gate, accuracy wins for chat; live must use the conservative degraded profile rather than exceed its cadence budget.

## 19. Observability and debugging

### 19.1 Build report

Persist and expose:

- source note and passage counts by connector;
- extraction failures by source/schema;
- entity/fact counts and confidence distributions;
- candidate versus promoted fact counts by predicate;
- ambiguous entity counts;
- hub exclusions by type/reason;
- base graph nodes/edges/components/dangling nodes;
- projection edges/components/degrees;
- orphan/stale invariant counts;
- per-stage latency and model versions;
- final graph hash.

### 19.2 Retrieval trace

Every debug trace includes:

- original and rewritten query;
- each head's raw and calibrated hits;
- rejected hits and thresholds;
- entity-resolution candidates/reasons;
- triple candidates and recognition decisions;
- complete teleport composition by node type;
- PPR convergence/residual and top passage/entity mass;
- explanatory paths;
- rerank scores and MMR exclusions;
- prior contributions;
- abstention decision and reason;
- selected/read/cited evidence IDs;
- active build/calibration hashes;
- latency by stage.

The Trigger Inspector should present this without exposing full private text unless the user opens an evidence item.

## 20. Testing requirements

### 20.1 Unit tests

- tokenizer-aware passage boundaries, overlap, offsets, deterministic IDs;
- GLiNER short batch and generic long batch calls, global spans, confidence thresholds;
- endpoint ambiguity, exact span validation, source schema selection;
- non-destructive entity resolution, casefold ambiguity, acronyms, digit/type guards;
- temporal fact creation, supersession, conflict, historical/current selection;
- directed forward/reverse graph weights;
- hub eligibility and explicit query activation;
- exact passage semantic top-k union and same-note overlap exclusion;
- projection isolation: no projection edge appears in base matrix;
- deterministic projection forest/cap behavior;
- dangling-correct PPR mass and typed reset weights;
- query-to-triple recognition acceptance/rejection/fallback;
- calibrated head floors and abstention;
- exact alias credited once;
- MMR/source caps and note aggregation;
- path direction, provenance, hop cap, and inactive-fact exclusion;
- registry IDs across retrieval and every tool;
- citation coverage, NLI, temporal checks, and unsupported-claim rejection;
- transcript sequence checkpoint behavior during buffer rollover;
- chat regeneration state restore.

Tests never load real large models unless marked integration; module seams use deterministic fakes.

### 20.2 Integration tests

- clean source records → complete build → retrieval evidence pack;
- long note with cross-boundary relation → correct global source span;
- changed note with failed extraction → new passages, no stale entities/facts;
- same source content in different connector order → identical graph;
- query-to-triple → entity seeds → PPR-expanded supporting passage;
- no recognized triple → dense/BM25 fallback;
- unrelated query → abstained empty pack;
- temporal contradiction → current answer plus historical evidence when requested;
- live summary receives E/F/T evidence and safely checkpoints accepted chunks;
- chat Stage A answers from source passages without mandatory read tools;
- search/read/expand tools register and cite all discovered evidence;
- unsupported generated claim fails validation/repair;
- build activation swaps every matrix atomically.

### 20.3 Real-model and replay tests

- GLiNER2 extraction benchmark over the labeled stratified set;
- Qwen embedding/reranker calibration;
- session replay over stored transcripts with labeled relevant passages and null windows;
- chat query benchmark including multi-turn follow-ups;
- full clean regeneration against a disposable database;
- packaged sidecar smoke test with all local models offline.

## 21. Implementation sequence

This is one umbrella design but must be implemented in independently reviewable units. The later implementation plan will create exact tasks/tests/commits; the required dependency order is:

### Unit 0 — freeze baselines and evaluation contracts

- export frozen current-retriever results;
- add JSONL schemas, metric runner, deterministic fixtures, and calibration artifact format;
- label the minimum extraction/retrieval/null/citation sets needed for initial gates.

No new ranking behavior is enabled without this unit.

### Unit 1 — passage and build foundations

- new passage schema, tokenizer-aware splitter, passage embeddings/FTS;
- content-addressed IDs and immutable snapshot/build contracts;
- clean-build command and integrity validation.

### Unit 2 — GLiNER long extraction and mention provenance

- source-aware combined schemas;
- `batch_extract_long` integration;
- global entity/relation spans;
- extraction status/failure semantics;
- first extraction calibration artifact.

### Unit 3 — entity resolution and temporal facts

- aliases/mentions/canonical nodes;
- ambiguity-aware query lookup;
- predicate registry and directed provenance;
- temporal validity, supersession, conflicts;
- bounded relation-completion lane.

### Unit 4 — base graph and isolated projection

- passage/entity/fact base edges;
- semantic passage edges;
- hub policy;
- dangling-correct PPR;
- independent note projection and quality rank;
- remove `rebuild_for` and double counting.

### Unit 5 — calibrated shared retrieval

- shared retrieval types/snapshot/heads;
- head calibration and abstention;
- query-to-triple recognition;
- passage/entity reset vector;
- PPR candidate expansion;
- path extraction, rerank, MMR, priors ablation.

### Unit 6 — evidence and citation infrastructure

- N/E/F/P/T registry;
- evidence packs and prompt rendering;
- source-span availability tracking;
- coverage/NLI/temporal validation;
- retrieved/read/cited persistence/UI semantics.

### Unit 7 — live integration

- time-aware windows;
- live policy/accumulator/hysteresis;
- sequence transcript ledger;
- evidence-based summarizer/actionables;
- live replay calibration and latency gates.

### Unit 8 — chat integration

- dependent-query rewrite;
- evidence-first Stage A;
- evidence-native tools;
- accumulator/topic/regeneration behavior;
- final validation, repair, and cited-source UI.

### Unit 9 — clean regeneration and release validation

- delete the old derived knowledge index;
- regenerate from connector sources using the new build;
- run every §18 gate and packaged offline smoke test;
- remove dead old graph/retrieval code, configuration, and tests;
- update architecture diagrams and README.

Unit 9 is regeneration, not migration. It does not translate or preserve old graph-derived rows.

## 22. Clean-build operator contract

Provide one explicit command used in development and release verification:

```bash
python -m taurine_core.indexer.build --db <path> --from-enabled-connectors --replace-derived
```

It:

1. confirms the target path;
2. creates a new temporary disposable knowledge database and never edits the active file in place;
3. reads source content through enabled connectors;
4. runs the full build pipeline;
5. prints the build report, health status, calibration/model hashes, graph hash, and failed-note list;
6. exits nonzero on invariant or release-gate failure;
7. atomically installs the completed file as the target only after validation succeeds;
8. never mutates connector source data.

The temporary-file and atomic-install behavior is mandatory. It must not introduce legacy migration logic.

## 23. Definition of done

The overhaul is complete only when:

- all G1-G9, R1-R7, and C1-C6 findings have a corresponding passing test and implementation;
- long notes use GLiNER2's supported long API with exact source spans;
- the base graph contains passages/entities/directed facts and no projected shared edges;
- retrieval no longer reads the note projection;
- identical full builds are deterministic and contain no orphan/stale derived rows;
- query-to-triple recognition seeds entity nodes and additive endpoint fact bonuses are gone;
- passage reset mass and dangling-correct PPR are implemented;
- filler/gibberish can abstain with an empty evidence pack;
- live and chat receive answer-bearing source passages;
- every tool result is registered and citeable;
- transcript checkpoints cannot discard in-flight speech;
- facts preserve direction, source span, observation time, validity, and supersession/conflict state;
- factual output with zero or unsupported citations is rejected or repaired;
- cited and merely retrieved sources are distinguished;
- the held-out quality, citation, determinism, and latency gates in §18 pass;
- a clean regeneration succeeds offline against a disposable database;
- old overlapping graph/retrieval code paths and retired configuration are deleted;
- documentation and diagrams match the new architecture.

## Appendix A — audit finding coverage

| finding                             | primary design sections |
| ----------------------------------- | ----------------------- |
| G1 long-document API                | §6.1, §20               |
| G2 extraction evaluation            | §6.3, §18.1-18.2        |
| G3 fixed relation recall            | §4.5, §6.2, §6.6        |
| G4 double counting                  | §3, §10, §15.2          |
| G5 rebuild drift                    | §9                      |
| G6 case-sensitive entity resolution | §7                      |
| G7 stale/orphan state               | §6.5, §9.4              |
| G8 direction loss                   | §4.5, §8, §10.3         |
| G9 quadratic scale                  | §10.4, Appendix C       |
| R1 no abstention                    | §11.3-11.5              |
| R2 fake query-to-fact pattern       | §11.6-11.7              |
| R3 no source passages               | §4.2, §11.10, §12       |
| R4 transcript delta loss            | §13.5                   |
| R5 dangling PPR                     | §10.6                   |
| R6 hub traversal                    | §7.2, §10.5             |
| R7 exact/session double count       | §11.4, §13.2-13.3       |
| C1 metadata-only Stage A            | §14.2                   |
| C2 unregistered tool evidence       | §12.1, §14.3            |
| C3 truncated non-query-aware reads  | §14.3                   |
| C4 syntactic-only citations         | §12.3-12.4              |
| C5 multi-turn contamination         | §14.1, §14.5            |
| C6 flat graph expansions            | §11.9, §14.3            |

## Appendix B — research alignment

- **HippoRAG 2:** passage and phrase/entity nodes, query-to-triple recognition, passage/entity reset weights, PPR passage ranking, dense fallback, original passages for QA.
- **GLiNER2:** combined schema extraction, official batch/long APIs, confidence and global spans, local CPU-first inference.
- **MS-RAG:** chunk/relation/entity recall channels plus multi-semantic reranking.
- **KG2RAG:** graph expansion followed by coherent evidence organization.
- **GeAR/DRIFT:** agentic/multi-step exploration retained for chat, not imposed on live speech.
- **Zep/Graphiti and temporal GraphRAG work:** observed/valid time, provenance, incremental fact history, conflict/supersession rather than destructive replacement.
- **LeanRAG/GNN-RAG:** path organization is adopted; hierarchy and learned GNN retrieval are deferred because Taurine lacks their scale/query/label justification.

Primary references:

- [HippoRAG 2 paper](https://arxiv.org/html/2502.14802)
- [HippoRAG official implementation](https://github.com/OSU-NLP-Group/HippoRAG)
- [GLiNER2 official implementation and API](https://github.com/fastino-ai/GLiNER2)
- [GLiNER2 paper](https://arxiv.org/abs/2507.18546)
- [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [DeBERTa-v3-base NLI cross-encoder](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
- [KG2RAG](https://aclanthology.org/2025.naacl-long.449/)
- [MS-RAG](https://aclanthology.org/2025.emnlp-main.1151/)
- [GeAR](https://aclanthology.org/2025.findings-acl.624/)
- [GNN-RAG](https://aclanthology.org/2025.findings-acl.856/)
- [LeanRAG](https://ojs.aaai.org/index.php/AAAI/article/view/40789)
- [Zep temporal knowledge graph architecture](https://arxiv.org/abs/2501.13956)

## Appendix C — scale transition rule

The initial implementation deliberately uses exact local computation where it improves auditability:

- exact passage similarity matrix;
- full entity df/eligibility recomputation;
- full base graph/projection finalization;
- full passage query scoring up to 10,000 passages;
- SciPy PPR.

Introduce ANN/posting-list/incremental graph infrastructure only when representative measurements cross a gate:

- full cached finalization >60 seconds for three consecutive builds;
- passage matrix memory exceeds the packaged sidecar budget;
- warm retrieval violates §18.6 after algorithmic batching;
- exact-to-ANN held-out Recall@12 loss is ≤1 absolute point and null precision does not regress.

The first scale changes are Qwen vector top-k batching/ANN and entity posting-list candidate generation. They do not change evidence, provenance, graph, or citation contracts.
