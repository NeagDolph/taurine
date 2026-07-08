# Architecture diagrams

Self-contained SVGs (embedded stylesheet, no external assets) — open in a
browser or view inline on GitHub.

## [Live voice retrieval pipeline](live-voice-retrieval-pipeline.svg)

![Live voice retrieval pipeline](live-voice-retrieval-pipeline.svg)

Mic → Deepgram → transcript chunks feed the two collectors. C1 matches
word-boundary aliases straight into the slate's exact lane. C2 fans the
window into four query heads (dense, salient-rarity, GLiNER2 spans +
sentence context, BM25), fuses them with RRF, traverses graph space
(graph-vote as a fifth RRF list, 1-hop neighbor expansion), reranks with
Cohere, then protects the top fused/dense hits past demotion into the
search lane. The cadence gate and the presenter (whole transcript +
`read_note` tool) turn the slate into at most three pinned cards.

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
