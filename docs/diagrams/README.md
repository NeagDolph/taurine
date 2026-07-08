# Architecture diagrams

Self-contained SVGs (embedded stylesheet, no external assets) — open in a
browser or view inline on GitHub.

## [Live voice retrieval pipeline](live-voice-retrieval-pipeline.svg)

![Live voice retrieval pipeline](live-voice-retrieval-pipeline.svg)

**Stale as of 2026-07-08** — the SVG still shows the pin-centric loop. The
current pipeline (see
[`../superpowers/specs/2026-07-08-live-summary-graph-retrieval.md`](../superpowers/specs/2026-07-08-live-summary-graph-retrieval.md)):
mic → Deepgram → C1 exact aliases + C2 heads (dense window, GLiNER2 spans +
snap + context, BM25) → RRF → meeting seed accumulator → Personalized
PageRank over the fact-boosted note graph → blend with recency decay,
source rank, and quality rank → grounding set + edge facts. A change
trigger (grounding-set Jaccard, topic drift, staleness) fires ONE
deepseek-v4-flash pass that maintains a rolling, citation-grounded meeting
summary and conservative quoted-span actionables. No reranker, no
presenter, no pins.

## [Enrichment & graph connection pipeline](enrichment-graph-connection-pipeline.svg)

![Enrichment and graph connection pipeline](enrichment-graph-connection-pipeline.svg)

Connector sources are extracted and hash-diffed so only changed notes are
re-enriched. Notes are clustered by embedding similarity and enriched in
one keyed GLM-5.2 call per batch (user card as shared context, per-note
validation with a per-note fallback), then embedded with Qwen3. The cheap,
LLM-free graph build resolves synonym terms to canonical forms (digit +
lexical guards), draws semantic kNN and rarity-weighted shared-term edges,
and merges them per pair by noisy-or. deepseek-v4-pro then adjudicates only
the band pairs the cheap signals flag — writing facts onto strong edges,
vetoing weak ones, admitting near-miss margins — caching every verdict by
note-content signature so future rebuilds apply it for free.

See also [`../superpowers/specs/2026-07-07-graph-connections-v2.md`](../superpowers/specs/2026-07-07-graph-connections-v2.md).
