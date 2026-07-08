# Graph Connections v2 — synonym resolution, margin adjudication, edge facts

2026-07-07. Follows the RAG/memory research assessment (HippoRAG 2 synonym
edges, A-MEM link adjudication, Graphiti fact-carrying edges). Temporal
validity explicitly deferred.

## Goals

1. **Entity/topic synonym resolution** — shared-vocabulary edges stop being
   exact-string: term strings are embedded and soft-linked into canonical
   clusters above a similarity threshold, so "recommendation-engine" /
   "recommendation_engine" / "Recommendation Engine" build edges together.
2. **LLM link adjudication at the margin** — deepseek-v4-pro (with the full
   user card as context) judges only the pairs where cheap mechanisms are
   uncertain: near-misses that just failed the edge thresholds (may admit)
   and weak kept edges (may veto). Never O(n²) LLM work.
3. **Facts on edges** — the same adjudication call writes a one-sentence
   fact for each pair it confirms, and additionally for each note's
   strongest kept edges. Facts persist on the edge row, survive rebuilds via
   a judgment cache, and surface in /graph, note previews, and the map.

## Design

### A. Term resolution (no LLM)

- Vocabulary = all enrichment `entities` (case-preserved, stripped) and
  `topics` (lowercased) across active notes.
- Each unique normalized term is embedded once (document encoding) and
  cached in a `term_vectors` sqlite table; only new terms embed on rebuild.
- Union-find clustering over pairs with cosine ≥ `TERM_SYNONYM_MIN_SIM`
  (0.95 — calibrated on the real corpus 2026-07-07: true variants ≥ 0.98,
  ShopLC↔Shopify 0.67) **with a digit guard**: two terms whose extracted
  digit sequences differ never merge ('PR #370' ≠ 'PR #371',
  'voice_agent_v3' ≠ 'voice_agent_v4', '24 Darling St' ≠ '25 Darling St').
- Canonical form per cluster = most frequent member (ties: shortest, then
  lexicographic). `_add_shared_edges` maps every term to its canonical form
  before intersection; df/rarity weights and `shared` labels use canonical
  forms.
- Pre-pass cheap normalization stays: lowercase topics, collapse
  whitespace; additionally fold `-`/`_` to space for matching.

### B. Margin adjudication + edge facts (one LLM surface)

Bands, computed from the same candidate machinery that builds edges:

| band | definition | LLM authority |
|---|---|---|
| strong kept | kept edge, weight ≥ VETO_MAX_W | fact only (top FACT_TOP_EDGES_PER_NOTE per note) |
| weak kept | kept edge, weight < VETO_MAX_W (0.45) | veto or confirm + fact |
| margin | failed keep-rule but semantic ≥ MARGIN_SEMANTIC_SIM (0.50) or shared-term weight ≥ MARGIN_SHARED_W (0.15); capped MARGIN_MAX_PER_NOTE (5) | admit + fact, or reject |

- **Judgment cache** (`edge_judgments` table): key (src, dst), payload
  (sig, related, fact, judged_at). `sig` = sha256 of both notes'
  content_hash — either note changing invalidates the judgment. Rejected
  pairs cache `related=0` so they are never re-asked.
- **Cheap rebuilds apply cached judgments synchronously**: graph rebuild
  (rebuild_all / rebuild_for) consults the cache — vetoed pairs dropped,
  admitted margin pairs added as edges (mechanism-typed, judged fact
  attached), facts attached to kept edges. Rebuild stays LLM-free and
  deterministic.
- **Async adjudication pass** (`indexer/adjudicate.py`) fills the cache:
  collects band pairs lacking a valid judgment, batches
  ADJUDICATE_BATCH_PAIRS (20) pairs per call to ADJUDICATE_MODEL
  (deepseek/deepseek-v4-pro, reasoning disabled, response_format json,
  ~1.5s/call verified live), then triggers a rebuild to apply.
- **Prompt**: full user card at the top; per pair: both titles + gists +
  shared canonical terms + cosine. Output keyed by pair id:
  `{"pairs": {"<id>": {"related": bool, "fact": "<one sentence>"}}}`.
  Missing/invalid entries → pair simply stays unjudged (asked again next
  pass); no per-pair LLM fallback.
- **Where it runs**: after enrichment + rebuild inside connector sync and
  re-enrich (both already async-capable), same progress plumbing
  ("adjudicating" phase). First full pass on the current corpus ≈ 989 kept
  edges (facts for top-per-note subset) + a few hundred margin pairs ≈
  ~60–80 calls; incremental passes only touch pairs whose sig changed.

### C. Surfacing

- `GraphEdge.fact: str = ""` (schemas), edges table `fact` column
  (migration), /graph payload carries it.
- Map: edge tooltip shows the fact when present.
- Note preview (`NotePreview.neighbors`): `NeighborRef.fact`.
- Presenter consumption of facts = future work (the "low-cost traversal"
  path), not in this batch.

## Non-goals

- Temporal validity / fact invalidation windows.
- LLM-written facts for every pair (bands only).
- Cross-note entity KG nodes (notes stay the only node type).

## Execution split

- Contracts (config knobs, schemas, store tables/columns/helpers): Claude,
  pre-written before dispatch.
- Unit E (Codex): term resolution in graph.py + tests.
- Unit F (Codex, after E): margin bands + judgment application in graph.py,
  adjudicate.py module, sync/reenrich integration + tests.
- Frontend (Claude): map tooltip + preview facts.
- Verification (Claude): merge-list review on real corpus, live adjudication
  run on a DB copy, cost/latency measurement, full suite.
