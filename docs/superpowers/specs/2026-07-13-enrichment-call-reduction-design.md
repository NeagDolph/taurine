# Enrichment LLM Call Reduction — Design

**Date:** 2026-07-13
**Status:** Approved scope (Neil, in-session); implementation pending
**Motivation:** A cold full enrichment on the current corpus (577 notes) makes ~484 LLM
calls (~$0.71), and a warm sync still makes ~45 because two lanes re-run from scratch
every build. Relation completion alone is ~92% of spend because it runs on glm-5.2.

## Current state (measured 2026-07-13)

| Lane | Model | Cold calls | Cached? | Re-runs warm? |
|---|---|---:|---|---|
| Context enrichment | deepseek-v4-flash | 136 | yes (per-passage) | no |
| Relation completion | z-ai/glm-5.2 | 40 | **no** | **yes (all 40)** |
| Fact adjudication | gpt-oss-120b | 260 | yes (per-fact) | no |
| Semantic reasons | gpt-oss-120b | 30 | yes (per-pair) | no |
| Merge judge | gpt-oss-120b | 13 | yes (per-pair) | no |
| Cluster labels | deepseek-v4-flash | 5 | no | yes (all 5) |

Target after this work: cold ≈ 300 calls / ≈ $0.05; warm sync ≈ 0 calls when nothing
changed. No lane is replaced by heuristics — the over-reliance was re-asking identical
questions every sync, not asking them at all.

## Changes

### 1. Relation completion moves to gpt-oss-120b

`relation_completer_for` (build.py) switches `config.CHAT_MODEL` →
`config.ADJUDICATION_MODEL` for its `complete_json` calls. A dedicated
`COMPLETION_MODEL = ADJUDICATION_MODEL` config name keeps the lane independently
tunable. gpt-oss-120b already runs at reasoning effort=low via
`LLM_REASONING_BY_MODEL`; completion is structured extraction against supplied
passage text — the same shape as the judge work this model already does, and the
adjudicator (which validates completion output) stays the quality backstop.
Temperature stays 0.0. Cost: ~$0.66 → ~$0.03 per cold build.

**Risk/rollback:** if the post-swap build's completion acceptance (facts surviving
adjudication) drops well below the glm baseline (~156 active `uses` facts), revert the
model constant — one line.

### 2. Completion lane gets a content-keyed cache (the big win)

New `completion_cache` on `BuildDependencies` (same pattern as `context_cache` /
`adjudication_cache`, backed by the same BuildCache `judgments` table via
`cached_judgments`/`store_judgments` with a `completion:` key prefix).

- **Key:** `completion:` + sha256 over (model id, system-prompt hash, predicates
  registry hash, canonical JSON of the note's `_note_payload`). Any change to the
  note's text, entities, mentions, tool-hop focus, prompt, or predicate registry
  busts the key naturally.
- **Value:** the model's raw fact/alias rows for that note (pre-parse), stored as a
  dict. Parsing stays live each build so parser fixes apply without cache busts.
- **Flow:** before batching, resolve cached notes and drop them from the call set;
  after calls, group returned rows by resolved note and store per note. **Empty
  results are cached too** — a note that yields nothing is asked once, ever.
- **Fail-open guard:** only notes that were part of a *successful* batch may store
  (including empty stores). `complete_relations` must therefore report which notes
  completed, not just merged rows — it returns
  `{"facts": [...], "aliases": [...], "completed_note_ids": [...]}`; a failed batch
  contributes no completed ids, so its notes retry next build. `parse_completion_response`
  ignores the extra key (rows-by-note grouping for storage reuses its passage→note
  resolution helper).

Warm-sync effect: 40 calls → 0 for unchanged notes. This also subsumes any
"stop re-asking barren notes" heuristic.

### 3. Adjudication batches pack across notes

`_fact_batches` (edge_adjudication.py) currently flushes per note — measured average
~3 facts/call against a 25-fact cap, hence 260 calls. Change: keep per-note fact
grouping and ordering, but let one `_FactBatch` carry facts from **multiple notes**
under the existing `ADJUDICATION_FACTS_PER_CALL` and `ADJUDICATION_TEXT_MAX_CHARS`
limits. `_FactBatch` gains a per-note sections list (note_id, title, text) and
`_fact_payload` emits `notes: [{note_id, title, text, facts: [...]}]`; the system
prompt notes that facts arrive grouped by source note. `_parse_fact_response` is
already keyed by `fact_key` — unchanged. Per-fact cache keys (`_fact_key`) don't
include batch composition or the system prompt, so **existing cached judgments stay
valid**.

Effect: cold 260 → ~80–90 calls; adjudication wall time ~3× shorter at the same
concurrency. Judgment quality consideration: each fact still ships with its own
note's passage text and assertion window, so grounding is unchanged; only payload
framing changes.

### 4. No-relation completion gate tightens

`select_completion_notes` currently admits every note with ≥2 entities and 0 active
relations — 343 of 577 notes, mostly low-value chatter the adjudicator rejects.
Change: that gate additionally requires `value_score` at or above the corpus median
of the candidate set (new `COMPLETION_NO_RELATION_MIN_PERCENTILE = 0.5`, applied as
a percentile over the candidates so it self-scales with corpus quality). The
`schema_gap` arm stays unconditional (rare and targeted), the top-15% value lane and
the tool-hop lane are unchanged.

Effect: cold selection ~470 → ~300 notes (≈40 → ~25 calls). With the cache (change
2) this matters mainly for first-build latency and for keeping barren-note churn out
of the cache.

### 5. LLM cluster naming is removed; deterministic names stay

Kill the flash naming lane: delete `cluster_labeler_for` and `label_clusters` (and
their tests), drop `cluster_labeler` from `BuildDependencies`, remove the
`label_clusters` stage call in build.py so `fallback_label` (top entities/terms)
names every cluster, with `label_source="fallback"`. Delete `CLUSTER_LABEL_MODEL` /
`CLUSTER_LABEL_BATCH` config and the server-side wiring that constructs the labeler.
UI is untouched — sidebar and map legend keep showing cluster names; they're just
the deterministic ones now. `ClusterRecord.label_source` stays in the schema (older
DBs contain `"llm"` rows). The idea-lens agentic search (`IDEA_SEARCH_MODEL`) is a
separate feature and is untouched.

### 6. Dead config removed

`ENRICH_MODEL`, `ENRICH_BATCH_SIZE`, `ENRICH_BATCH_CHAR_BUDGET`,
`ENRICH_BATCH_MIN_NOTES` have no consumers (legacy of the deleted pre-overhaul
enrichment flow) — delete them.

## Testing

- **Completion cache:** unit tests for key stability (same payload → same key;
  entity/text/prompt change → new key), empty-result caching, failed-batch
  non-caching (note retried next call), and cached+fresh row merge through
  `parse_completion_response`. Integration: two consecutive builds over the same
  fixture corpus — second build makes zero completion LLM calls (count via fake llm).
- **Batch packing:** `_fact_batches` tests — multiple small notes pack into one
  batch; caps still split; ordering deterministic; payload groups facts under the
  right note sections; existing per-fact cache keys unchanged (assert `_fact_key`
  output identical pre/post change).
- **Gate:** `select_completion_notes` tests for the percentile arm (median
  candidate excluded/included at boundary; schema_gap still unconditional; tool-hop
  cap unchanged).
- **Cluster labels:** existing label tests replaced by fallback-only coverage; build
  integration asserts `label_source == "fallback"` and no labeler dependency.
- **End-to-end:** full rebuild on the production DB; compare active `uses` facts and
  adjudication accept/reject mix against the 2026-07-12 baseline (156 active uses
  facts); verify warm re-sync makes ~0 enrichment calls (server log lane counters).

## Rollout

Single branch off main; full test suite; production rebuild + install with backup
(same procedure as 2026-07-12). The model swap (change 1) is trivially revertible
independent of the rest.
