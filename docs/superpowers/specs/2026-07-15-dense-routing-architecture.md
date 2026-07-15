# Dense-Default Routing Architecture — Design

**Date:** 2026-07-15
**Status:** Approved direction (Neil): "implement the routing architecture since dense is clearly better for retrieval… all retrieved documents have dates… chunk iMessage texts into daily dates… each LLM chat query gets the current date and time."
**Grounding:** three ranked ablation rounds (docs/superpowers/specs/2026-07-15-graph-last-chance-investigation.md). Dense+rerank won breadth consistently; the graph's wins were the adjudicated fact layer (temporal, attribution, exclusivity) and abstention — never PPR traversal. Matches HippoRAG's own published results: graph wins are multi-hop-only, and this corpus/question mix is not multi-hop.

## 1. Chat retrieval becomes dense-default + fact overlay (no PPR)

New chat path (`CHAT_RETRIEVAL_MODE = "dense_facts"`, old `"graph"` kept as fallback flag):

1. **Dense sweep:** query embedding (existing backend) → cosine top `DENSE_SWEEP_K = 48` over all passages → Cohere rerank → top `CHAT_PACK_BUDGET = 12` passages. No dense acceptance curve for candidacy (round-3 finding: no separating floor exists); the reranker is the gate. BM25 keeps feeding the sweep as today (union pre-rerank).
2. **Fact overlay (the graph's proven half, no traversal):** resolve query terms against entity aliases (existing resolver); attach ALL active facts touching resolved entities, ordered by adjudication score then recency, capped at `CHAT_FACT_OVERLAY_MAX = 12`, as the `[F#]` block. Temporal reconciliation already guarantees currency (e.g., single active `works_at`). This is a SQL lookup, not PPR — O(ms).
3. **Pack assembly:** passages `[E#]` + facts `[F#]`, existing registries and citation validation unchanged. Low-confidence fill and abstention accounting retired from chat (the reranker + rerank floors alone label confidence); chat abstains only when the sweep itself is empty.
4. **Live mode: UNTOUCHED** — strict graph lane, calibrated abstention, byte-identical. Map/exploration UI: untouched. PPR code stays for live; it simply no longer serves chat.
5. Speculative pre-retrieval and the chat tool loop keep working — they call the new path.

## 2. Dates everywhere

- **Per-passage dates:** every retrieved passage renders with its date: iMessage passages use the per-day date (§3); all other sources use note modified (fallback created) as `YYYY-MM-DD`. Format in evidence block: `[E3] (2026-05-14) <text>`. Fact rows render their observed_at/valid window as today.
- **Current datetime in every chat request:** the chat system prompt gains a first line rendered at request time: `Current date and time: {ISO local, e.g. 2026-07-15 22:41 EDT}.` — never cached across turns (assert in test). Also added to the live-summary prompt for consistency.

## 3. Daily iMessage chunking

- The iMessage connector inserts day-boundary marker lines into rendered transcripts: `— YYYY-MM-DD —` before the first message of each local calendar day.
- `passages.py` gains a source-aware rule: for `imessage` notes, day markers are HARD passage boundaries (a passage never spans two days); within a day, existing token/sentence splitting applies (target 220 / max 300 unchanged). Day-header lines belong to the day's first passage.
- Each passage records `passage_date` (new column, nullable): for iMessage, the day marker's date; for others, null (renderers fall back to note dates). Char offsets stay note-global and exact; the marker lines are part of note text (searchable, quotable) so no offset trickery.
- Effect: retrieval granularity becomes "what was said that day in that chat," dense matches sharpen (one day's topic per vector), and date rendering is exact instead of thread-level.

## 4. Consequences & rollout

- **Rebuild required** (chunking change → new passages/embeddings/extraction; fingerprints already cover chunking knobs). ~$0.10 embeddings; extraction on MPS post-watermark-fix.
- **Validation gate:** the 15-question ablation set (Neil-ranked) + 16-question battery are the regression gate. Success = dense-side parity or better on every round-3 dense win, fact-layer wins retained (graduation, employer-now, docker-projects recover via the `[F#]` overlay), live abstention controls still pass.
- Latency target: ≤3s warm chat retrieval (dense sweep + one rerank call + SQL fact lookup).
- Cleanups rendered obsolete IF this ships as default: round-3's DENSE_UNGATED_CANDIDACY, head reservations, and fill-trigger machinery in the chat path (superseded by dense_facts mode; leave code dormant behind the graph flag for live).

## Risks

- Fact overlay depends on entity resolution catching query aliases ("Klaviyo", "Mahith") — misses degrade to dense-only, which is the measured-good baseline.
- Daily chunking multiplies passage count (~5k → est. 8-12k); embedding cost scales linearly (cents) and rerank input is unchanged (top-48 cap). Cohere rerank per query unchanged.
- Marker lines appear in quoted evidence text — acceptable (they're honest dates); UI may later strip them cosmetically.
