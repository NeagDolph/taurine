# Live Meeting Summary v1 — latent graph retrieval, grounded facts, actionables

**Date:** 2026-07-08 · **Status:** approved for codex delegation
**Supersedes:** the pin-centric live loop of the MVP spec (retrieval sections of
`2026-07-06-taurine-mvp-design.md`). Builds directly on
`2026-07-07-graph-connections-v2.md` (synonym resolution, margin adjudication,
facts on edges) — that upfront graph investment is what this spec cashes in.

## Product pivot

Taurine's live deliverable changes from **source pinning** (0–2 pinned note
cards chosen by a presenter LLM) to **a rolling meeting summary grounded in the
corpus**: a continuously maintained summary of the meeting so far, every claim
cited to corpus facts/notes, plus **conservatively surfaced actionables** with a
full grounding interface (what sources support it, why it's new). Sources stop
being the product and become evidence — Taurine is a copilot, not a search box.

Three systemic changes make this fast and cheap:

1. **No LLM in retrieval.** The Cohere reranker and the presenter's
   multi-call `read_note` tool loop are removed. Retrieval is fully latent
   (local embeddings + BM25 + exact aliases) with the graph doing the heavy
   lifting via Personalized PageRank (PPR) traversal. One LLM call remains in
   the live loop: the summarizer.
2. **One summarizer pass, change-triggered.** `deepseek/deepseek-v4-flash`
   updates the rolling summary only when the *grounding set* (the retrieved
   ambient sources) materially changes — not on keywords, not on a fixed
   clock. The model itself may also answer "no update needed."
3. **Static priors on every note.** Two new ranks — a hardcoded **source
   rank** and a graph-derived **quality rank** (weighted PageRank with a
   trust+recency teleport) — plus **exponential age decay** gate what enters
   the grounding set, so stale or low-value items stop surfacing.

Research grounding (full citations in the appendix): HippoRAG 2's LLM-free
query-time PPR (arXiv:2502.14802), ACL forward-push / power-iteration PPR
(sub-10 ms at our scale), ProCIS proactive-retrieval triggering (SIGIR 2024),
ReSum rolling-summary state (arXiv:2509.13313), generative-agents / Zep
temporal decay (arXiv:2304.03442, 2501.13956), Craswell et al. static-prior
blending (SIGIR 2005), TrustRank teleport priors (VLDB 2004), Grolmusz on
undirected PageRank ≈ degree unless the teleport is non-uniform
(arXiv:1205.1960), MiniCheck-style constrained citation validation
(arXiv:2404.10774), conservative action-item extraction (ICASSP 2023,
arXiv:2303.16763).

## Architecture

```
 transcript chunks (Deepgram finals)
        │
        ├─► C1 EXACT (unchanged matcher) ──────────────┐  exact hits = max-strength seeds
        │                                              │  (instant-pin cards RETIRED)
        ├─► C2 HEADS (throttle 8s, all local)          ▼
        │     dense(window) · dense(GLiNER spans+snap) · BM25   [salient head REMOVED]
        │     → RRF fuse (k=60)                        │
        │                                              ▼
        │            MEETING SEED ACCUMULATOR  (decayed sum of fused lists —
        │            the "entire transcription so far" as a personalization vector)
        │                                              │
        │                                              ▼
        │            PPR TRAVERSAL over the note graph (damping 0.5, scipy CSR,
        │            fact/entity-boosted edge weights, <10 ms)
        │                                              │
        │                                              ▼
        │            BLEND: relevance ⊕ PPR mass, × recency decay × source rank
        │            × quality rank  →  GROUNDING SET (top-K notes + edge facts)
        │                                              │
        ▼                                              ▼
   transcript UI                       CHANGE TRIGGER (Jaccard vs last summarized
                                       set + topic-centroid drift + gates)
                                                       │ fires
                                                       ▼
                                       SUMMARIZER — ONE deepseek-v4-flash pass
                                       prev summary + transcript delta + fact IDs
                                       → updated summary (cited) + actionables
                                       → or {"no_update": true}
                                                       │
                                                       ▼
                                       UI: summary panel · actionable cards w/
                                       grounding drawer · grounding source list
```

## 1. Data model & store changes

### 1.1 Timestamps (requirement: age-aware retrieval)

`NoteRecord` gains `created: float` alongside the existing `modified`:

- **apple_notes**: `creation_date` from macnotesapp/AppleScript.
- **obsidian / claude_memories / codex_memories / file-based**:
  `st_birthtime` (macOS); fallback `st_mtime`.
- **imessage**: `created` = timestamp of the *oldest* message in the extracted
  window; `modified` stays the newest (already implemented).
- Migration: `ALTER TABLE notes ADD COLUMN created REAL NOT NULL DEFAULT 0`;
  backfill `created = modified` for existing rows; connectors overwrite on
  next sync.

`modified` is the recency anchor for decay (a note actively edited is fresh
regardless of creation). `created` is stored for display ("note from March")
and future bi-temporal work.

### 1.2 Static ranks on notes

```sql
ALTER TABLE notes ADD COLUMN quality_rank REAL NOT NULL DEFAULT 0.0;
-- quality_rank = log(pagerank_i / median_pagerank), clipped to [-QUALITY_CLIP, +QUALITY_CLIP]
```

Source rank is NOT stored — it's a config lookup by `source` (§2.1).

### 1.3 Fact embeddings (query-to-fact matching, HippoRAG 2 pattern)

Adjudicated edge facts become retrievable units. New embedding kind:

```sql
-- embeddings.kind gains 'fact'; fact rows carry the edge identity
ALTER TABLE embeddings ADD COLUMN dst TEXT NOT NULL DEFAULT '';
-- kind='fact': note_id = edge.src, dst = edge.dst, text = edge.fact
```

Fact rows are (re)built during graph rebuild whenever an edge's fact changes;
deleted when the edge or either note goes. A dense hit on a fact credits
**both** endpoint notes (half weight each) in the head's ranked list, and the
fact itself is remembered for grounding payload assembly (§4.3).

## 2. Static priors

### 2.1 Source rank (hardcoded)

```python
SOURCE_RANK = {
    "apple_notes": 1.00,       # written notes: highest
    "obsidian": 1.00,
    "imessage": 0.85,          # text messages: a bit lower
    "claude_memories": 0.70,   # agent memories: a bit lower still
    "codex_memories": 0.70,
}
SOURCE_RANK_DEFAULT = 0.85     # future connectors until explicitly ranked
```

Used in two places: as teleport mass in the quality rank (§2.2) and as a
direct multiplier at blend time (§4.4). Deliberately mild spreads — this is a
prior, not a filter.

### 2.2 Quality rank (weighted PageRank, trust+recency teleport)

Computed at the end of every graph rebuild (`indexer/rank.py`, pure
numpy/scipy — no new deps beyond scipy, which sentence-transformers already
pulls in). Per the research: on an undirected graph, PageRank with a
*uniform* teleport is just smoothed degree (Grolmusz 2012); a **non-uniform
teleport is the lever** — it makes the rank a principled composite of graph
structure, source trust, and freshness (TrustRank / T-Rank pattern).

Algorithm:

1. Build CSR transition matrix over active notes: `P[j→i] = w'_ij / s_j`
   where `w'` are the boosted edge weights of §4.2 and `s_j` = strength.
2. Teleport vector `t_i ∝ SOURCE_RANK[source_i] × recency_i` (the same decay
   formula as §4.4, evaluated at rebuild time), normalized.
3. Power iteration: `r ← QUALITY_DAMPING·P·r + (1−QUALITY_DAMPING)·t`,
   `QUALITY_DAMPING = 0.85`, tol `1e-6` (L1), max 100 iters, warm-started
   from the previous rank vector. Isolated notes receive pure teleport mass.
4. Store `quality_rank_i = clip(log(r_i / median(r)), ±QUALITY_CLIP)`,
   `QUALITY_CLIP = 2.0`. Log-transform before blending is mandatory —
   raw PageRank in a linear blend is worse than no prior (Craswell 2005).

Pathology guards (from the research):

- **Near-duplicate cliques** inflate each other's rank (collusion bound
  1/(1−α) ≈ 6.7× at α=0.85). Guard: pairs with cosine ≥ `DUP_EDGE_SIM = 0.95`
  within the same source contribute **zero** transition weight to the quality
  computation (edge kept for display/traversal, excluded from `P`). If top
  ranks still cluster on near-identical notes, drop damping to 0.7 before
  anything fancier.
- **Hub bias**: strength normalization + the existing `GRAPH_MAX_DEGREE = 12`
  cap already bound it. Eigenvector centrality is explicitly rejected
  (localization pathology, Martin/Newman 2014).
- At current scale (~300 notes, ~1.1k edges) this converges in single-digit
  milliseconds; at 100k notes it's still < 1 s — always recompute fully, no
  incremental machinery.

## 3. What is removed

| removed | replaced by |
|---|---|
| Cohere rerank (`live/rerank.py`, `RERANK_*`, `COHERE_API_KEY` usage in live path) | PPR + prior blend; the summarizer adjudicates |
| Presenter (`live/presenter.py`), `read_note` tool loop, gemini-3.5-flash | single-pass summarizer (§5) |
| Salient-rarity head (`_salient_query`, `SALIENT_*`) | GLiNER + dense + BM25 cover it; audit showed net noise |
| Graph-vote 5th RRF list + 1-hop expansion (`GRAPH_VOTE_*`, `EXPAND_SEEDS`, `HOP_DECAY`) | PPR traversal (multi-hop, consensus-aware by construction) |
| Instant-pin cards, pinned stack, `pin`/`unpin` events, `MAX_PINNED` | summary panel + actionables; exact hits stay as retrieval signal |
| Slate views/decay/quotas/suppression lifecycle (`slate.py` semantics) | grounding set with hysteresis (§4.5) |
| Presenter cadence semantics (`COOLDOWN_S`, promos/min, reoffer) | summary trigger policy (§4.6) |

Session logging keeps working: `c2_pass` extends with PPR/prior fields; new
record kinds `grounding` (set + scores + trigger decision) and `summary`
(model, latency, prompt token counts, no_update, picks). The `retrieval`
event feeding the map view stays (PPR arrivals get `hop`-like provenance:
`via="ppr"` with mass).

## 4. Latent retrieval → grounding set

`live/ambient.py` slims into heads + fusion; new `live/grounding.py` owns
seeds → PPR → blend → grounding set → trigger. All local, target < 50 ms per
pass; the 8 s `C2_THROTTLE_S` stays as the pass cadence.

### 4.1 Heads and fusion (kept, minus salient)

Per pass over the rolling window: `dense(window)`, `dense(GLiNER spans, raw +
fuzzy-snapped + span-context)`, `bm25(window + snapped terms)` — RRF-fused
(`RRF_K = 60`) exactly as today. C1 exact alias hits (word-boundary matcher
unchanged) are injected as a fourth list at rank 1. `FUSED_PROTECT` /
`DENSE_PROTECT` / `MIN_COSINE` floor semantics carry over unchanged to the
fused list.

### 4.2 Edge weights for traversal ("weigh connections more heavily")

Traversal (and quality rank) use boosted weights, recomputed at rebuild:

```python
w' = min(1.0, w × (FACT_EDGE_BOOST if edge.fact else 1.0)
              × (SHARED_EDGE_BOOST if edge.type in ("entity","topic","mixed") else 1.0))
FACT_EDGE_BOOST = 1.5     # adjudicated, LLM-confirmed edges are trustworthy
SHARED_EDGE_BOOST = 1.35  # audit F3.1: entity/topic edges carry precise
                          # relevance but were the weakest — corrected here
```

Vetoed edges are already gone; margin-admitted edges carry facts and get the
fact boost. This is where graph-connections-v2 pays off: the traversal
substrate is adjudicated, synonym-resolved, and fact-annotated.

### 4.3 Meeting seed accumulator + PPR traversal

**Accumulator** — the personalization vector representing the *entire
transcription so far*, recency-weighted:

```python
seeds = SEED_ACCUM_DECAY * seeds_prev            # γ = 0.85 per C2 pass
for note_id, fused_score in fused_top(SEED_TOP_K=10):
    seeds[note_id] += fused_score                # exact hits enter at list-max
```

**PPR** (`indexer/rank.py` shares the CSR machinery): personalization =
normalized accumulator; damping `PPR_DAMPING = 0.5` (HippoRAG's value — keeps
mass within ~2 hops of seeds, right for grounding facts, wrong for the
classic 0.15 which diffuses too far); power iteration, tol `1e-8`, ≤ 50
iters, warm-started from the previous pass's PPR vector (seed sets between
consecutive passes overlap heavily, so this converges in a handful of
iterations). Runtime at our scale: well under 10 ms.

### 4.4 Blend: relevance ⊕ traversal, × priors

```python
rel     = (1 - PPR_BLEND) * norm(fused_rrf) + PPR_BLEND * norm(ppr_mass)   # PPR_BLEND = 0.35
recency = RECENCY_FLOOR + (1 - RECENCY_FLOOR) * exp(-ln2 * max(0, age_days - RECENCY_OFFSET_DAYS)
                                                     / RECENCY_HALF_LIFE_DAYS[source])
final   = rel * recency * SOURCE_RANK[source] * exp(QUALITY_LAMBDA * quality_rank)
```

- `norm` = min-max over the current candidate pool (fused ∪ PPR top-30).
- `RECENCY_OFFSET_DAYS = 2` — a 48 h grace so yesterday's note isn't
  penalized vs. one from an hour ago (Elasticsearch decay `offset` pattern).
- `RECENCY_HALF_LIFE_DAYS = {"imessage": 30, "claude_memories": 60,
  "codex_memories": 60, "apple_notes": 90, "obsidian": 120}` — episodic
  sources decay fast, written notes slowly (per-class decay, Zep/generative-
  agents pattern). Age from `modified`.
- `RECENCY_FLOOR = 0.15` — decay is a downweight, never an exclusion: an old
  note that is strongly relevant *and* graph-central can still surface, but
  old-and-marginal items are pushed below the cut. The summarizer also sees
  each note's age (§5.2) and adjudicates staleness explicitly.
- `QUALITY_LAMBDA = 0.20` — with `quality_rank ∈ [−2, 2]` this gives a
  multiplier in [0.67, 1.49]: roughly "one good query-term's worth" of
  influence, matching Craswell's tuned static-prior magnitudes. Tune later;
  the optimum is flat.

### 4.5 Grounding set

The top `GROUNDING_TOP_K = 12` notes by `final`, with hysteresis to prevent
churn: a member is only displaced when the challenger's score exceeds the
incumbent's by `GROUNDING_HYSTERESIS = 1.15×`. No views/decay/quotas — this
is a ranked evidence set, not a promotion queue. Each entry carries
`{note_id, title, gist, source, age_days, score, why: {heads_hit, ppr_mass,
exact_alias?}}`.

**Grounding facts:** all edge facts on edges *within* the grounding set or
from a grounding note to a top-5 seed, plus facts whose fact-embedding hit
this pass — capped at `GROUNDING_FACTS_MAX = 20`, each assigned a stable
short ID (`F1`, `F2`, …) persisted for the session (IDs never reused — the
summarizer's citations must stay valid across revisions).

The ambient UI list becomes this grounding set (score bar = `final`, plus a
source badge and age chip).

### 4.6 Summary trigger (change in ambient sources, not keywords)

Evaluated after every C2 pass (`live/trigger.py`, replacing cadence.py
semantics):

```
fire iff  (Jaccard(top-8 grounding ids, last-summarized top-8) < SUMMARY_CHANGE_JACCARD = 0.6
           OR topic drift: cos(window_emb, topic_centroid) < SUMMARY_DRIFT_COSINE = 0.70
           OR forced: now - last_pass > SUMMARY_MAX_STALENESS_S = 240 AND new transcript exists)
      AND now - last_pass > SUMMARY_MIN_INTERVAL_S = 25
      AND new final transcript chars since last pass ≥ SUMMARY_MIN_NEW_CHARS = 200
```

`topic_centroid` = EMA (`TOPIC_CENTROID_DECAY = 0.8`) of window embeddings;
a drift firing resets the centroid. A `no_update` reply from the summarizer
applies `SUMMARY_EMPTY_BACKOFF_S = 40` before the same grounding set can
re-fire. This is the ProCIS "opportune moment" recipe: fire on evidence
change and topic boundaries, debounced — never on raw keywords.

## 5. Summarizer — one pass, deepseek-v4-flash

### 5.1 Model

```python
SUMMARY_MODEL = "deepseek/deepseek-v4-flash"
SUMMARY_FALLBACK_MODEL = "deepseek/deepseek-chat"
SUMMARY_TIMEOUT_S = 20.0
LLM_REASONING_BY_MODEL["deepseek/deepseek-v4-flash"] = {"enabled": False}
```

JSON response_format via the existing `llm.py` client (fence-tolerant
parsing already handled). No tools, no second pass, no retry loops beyond the
single fallback-model retry.

### 5.2 Input (ReSum-style rolling state — bounded prompt, not the whole transcript)

1. **Previous summary state** — the last `MeetingSummary` JSON (or "none").
2. **Transcript delta** — final chunks since the last *accepted* pass
   (channel-labeled), capped `SUMMARY_DELTA_MAX_CHARS = 6000`.
3. **Grounding block** — per note: `[N3] "title" (source, age: 4mo, score
   0.61) — gist`; per fact: `[F7] (N3↔N9) "one-sentence fact"`. Ages stated
   explicitly so the model can discount stale evidence.
4. **Open actionables** — previously surfaced, with status.

### 5.3 Output contract

```json
{
  "no_update": false,
  "summary_md": "## Meeting so far\n- Discussing Shopify integration rollout [N3][F7]...",
  "highlights": [{"fact_id": "F7", "why": "directly answers the open staffing question"}],
  "actionables": [{
    "title": "Send Ashish the integration timeline by Friday",
    "quote": "I'll get you the timeline before Friday",
    "assignee": "me",
    "due": "Friday",
    "grounding": ["N3", "F7"],
    "why_new": "Commitment made just now; no existing actionable covers the timeline deliverable",
    "confidence": 0.9
  }]
}
```

`no_update: true` (with all else omitted) is a first-class, correct answer —
the model is explicitly instructed that silence is preferred when the summary
would not materially change (requirement 3; the trigger layer's backoff
handles the rest).

### 5.4 Validation (constrained citation — engineering guarantees, not vibes)

- Every `[N#]`/`[F#]` in `summary_md` must exist in the grounding block;
  claims citing unknown IDs have the citation stripped; a summary where > 30%
  of citations are invalid is rejected (keep previous summary, log).
- **Actionables are conservative by construction** (precision over recall):
  - `quote` must fuzzy-match the transcript delta at ratio ≥
    `ACTIONABLE_QUOTE_MIN_RATIO = 0.85` (rapidfuzz, already a dep) — no
    quote, no actionable;
  - `confidence ≥ ACTIONABLE_MIN_CONF = 0.8`;
  - deduped against open actionables (embedding cosine ≥ 0.90 on title →
    treated as update, not new);
  - at most `ACTIONABLE_MAX_PER_PASS = 2` accepted per pass;
  - system prompt requires an explicit commitment surface form (assignee +
    verb + deliverable), rejecting aspirational chatter — AMI-corpus work
    shows recall is unattainable anyway; a copilot that shows wrong
    actionables loses trust faster than one that misses some.

### 5.5 Cost/latency envelope

Per accepted pass: ~2.5–4k prompt tokens, ~400 output tokens, one
flash-tier call ≈ 1–2 s. Trigger gates yield roughly 1–2 passes/minute in
active conversation, near-zero during unrelated chatter. Removed per-pass
costs: Cohere rerank call, gemini presenter + up to 3 `read_note` tool
round-trips. Retrieval itself: zero API calls.

## 6. Protocol, REST, UI

**WS events** (replacing `pin`/`unpin`):

- `{type:"summary", rev, summary_md, highlights, actionables, grounding:[...]}`
  — full replacement on every accepted pass.
- `{type:"actionable", op:"new"|"update"|"dismiss", item}` — for toast/badge UX.
- `ambient` event now carries the grounding set (with `source`, `age_days`,
  `score`, `why`). `retrieval` (map) unchanged plus `via:"ppr"` arrivals.
- client → server: `{type:"actionable_status", id, status:"confirmed"|"dismissed"}`.

**REST:** `GET /summary` → current `MeetingSummary`; actionable status
mutations mirror the ws message.

**UI (Live tab restructure):**

- **Summary panel** (top, replaces pinned stack): rendered `summary_md`,
  citation chips (`[N3]`, `[F7]`) — hover shows the fact/note, click opens
  the existing note preview window.
- **Actionable cards** (right of / under summary): title, quote, assignee/due
  chips, confidence. Expanding opens the **grounding drawer**: the verbatim
  transcript quote, each grounding source (title, source badge, age, gist,
  the edge facts involved), and the model's `why_new` — the full "why should
  this be a new actionable" interface. Confirm / dismiss buttons drive
  `actionable_status`.
- **Grounding list** (sidebar, evolves AmbientList): score bar, source badge,
  age chip, head/PPR provenance icon.
- Map view unchanged (still consumes `retrieval`).

## 7. Config summary (new/changed knobs)

```python
# summarizer
SUMMARY_MODEL = "deepseek/deepseek-v4-flash"
SUMMARY_FALLBACK_MODEL = "deepseek/deepseek-chat"
SUMMARY_TIMEOUT_S = 20.0
SUMMARY_MIN_INTERVAL_S = 25.0
SUMMARY_MAX_STALENESS_S = 240.0
SUMMARY_MIN_NEW_CHARS = 200
SUMMARY_CHANGE_JACCARD = 0.6
SUMMARY_DRIFT_COSINE = 0.70
TOPIC_CENTROID_DECAY = 0.8
SUMMARY_EMPTY_BACKOFF_S = 40.0
SUMMARY_DELTA_MAX_CHARS = 6000
ACTIONABLE_MIN_CONF = 0.8
ACTIONABLE_QUOTE_MIN_RATIO = 0.85
ACTIONABLE_MAX_PER_PASS = 2

# grounding / traversal
GROUNDING_TOP_K = 12
GROUNDING_FACTS_MAX = 20
GROUNDING_HYSTERESIS = 1.15
SEED_TOP_K = 10
SEED_ACCUM_DECAY = 0.85
PPR_DAMPING = 0.5
PPR_TOL = 1e-8
PPR_MAX_ITERS = 50
PPR_BLEND = 0.35
FACT_EDGE_BOOST = 1.5
SHARED_EDGE_BOOST = 1.35
DUP_EDGE_SIM = 0.95

# priors
SOURCE_RANK = {...}                      # §2.1
SOURCE_RANK_DEFAULT = 0.85
QUALITY_DAMPING = 0.85
QUALITY_CLIP = 2.0
QUALITY_LAMBDA = 0.20
RECENCY_OFFSET_DAYS = 2
RECENCY_HALF_LIFE_DAYS = {"imessage": 30, "claude_memories": 60,
                          "codex_memories": 60, "apple_notes": 90,
                          "obsidian": 120}
RECENCY_HALF_LIFE_DEFAULT_DAYS = 90
RECENCY_FLOOR = 0.15

# removed: RERANK_*, PRESENTER_* (except transcript logging aids),
# SALIENT_*, GRAPH_VOTE_*, EXPAND_SEEDS, HOP_DECAY, MAX_PINNED,
# COOLDOWN_S/MAX_PROMOS_PER_MIN/REOFFER_S/EMPTY_BACKOFF_S (old semantics),
# QUOTA_*/DECAY_*/SUPPRESS_S
```

## 8. Testing

`pytest` in `core/`, fixture corpus, no prompt-substring assertions
(project convention). New/changed suites:

- `test_index_rank.py`: quality rank — teleport composition (source×recency),
  convergence, log/clip transform, dup-edge exclusion, isolated notes,
  warm-start equivalence, incremental rebuild stability.
- `test_live_grounding.py`: seed accumulator decay, PPR mass vs hand-computed
  values on a toy graph, blend formula (recency offset/floor/half-life per
  source, source rank, quality multiplier), hysteresis, fact-ID stability,
  grounding-fact selection.
- `test_live_trigger.py`: Jaccard/drift/staleness firing matrix, min-interval
  and min-new-chars gates, no_update backoff — fake clock.
- `test_live_summarizer.py`: prompt assembly (mocked LLM), output validation —
  unknown citation stripping, >30% invalid rejection, quote fuzzy-gate,
  confidence gate, dedupe-as-update, per-pass cap, no_update handling,
  fallback model retry.
- `test_index_store.py` additions: `created` migration + backfill, fact
  embedding rows lifecycle (edge fact change → row replaced; note delete →
  cascades).
- Connector tests: `created` extraction per connector (birthtime fallback,
  iMessage oldest-in-window).
- Removal hygiene: rerank/presenter/salient/graph-vote code and events gone;
  ws protocol tests updated for `summary`/`actionable`.

## 9. Execution split (codex units)

Contracts first (Claude, pre-written before dispatch): config knobs, schemas
(`MeetingSummary`, `Actionable`, `GroundingItem`, ws events), store
migrations + helper signatures, module skeletons with docstrings.

- **Unit G (codex):** store migrations (`created`, `quality_rank`, fact
  embeddings) + connector `created` extraction + `indexer/rank.py` (quality
  rank) + tests. No live-loop dependencies.
- **Unit H (codex, after G):** `live/grounding.py` — heads reuse, seed
  accumulator, PPR, blend, grounding set + facts, hysteresis; strip salient/
  graph-vote/expansion/rerank from `ambient.py`; `retrieval` event `via:"ppr"`;
  tests.
- **Unit I (codex, parallel with H):** `live/trigger.py` + `live/summarizer.py`
  + validation layer + session-log record kinds + tests (mocked LLM).
- **Unit J (codex, after H+I):** `ingest.py` rewiring, ws protocol + REST
  `GET /summary` + actionable status, removal of pin/presenter paths, protocol
  tests.
- **Frontend (Claude):** summary panel, actionable cards + grounding drawer,
  grounding list rework, event handling.
- **Verification (Claude):** live run against the real corpus on a DB copy —
  measure retrieval pass latency (< 50 ms target), summarizer latency/cost per
  pass, trigger fire-rate during a scripted conversation, and a
  precision-check on actionables; full suite; update
  `docs/diagrams/live-voice-retrieval-pipeline.svg`.

## Non-goals

- Bi-temporal fact validity / edge invalidation windows (Graphiti-style
  `t_valid`/`t_invalid`) — next round; §1.1's `created` is the substrate.
- NLI-based citation verification (MiniCheck) — the constrained-ID scheme
  covers v1; add a local verifier only if hallucinated citations appear.
- Learned blend weights / LTR; usage-based priors (note-access counts).
- Multi-hop PPR beyond what damping 0.5 naturally reaches; entity-level KG
  nodes (notes remain the only node type).
- Post-meeting export of summary/actionables (obvious follow-on).

## Appendix — research the design leans on

- **HippoRAG 2** (arXiv:2502.14802): LLM-free query-time PPR over a
  precomputed graph; damping 0.5; query-to-triple dense matching beats
  entity linking; their droppable LLM triple-filter is worth only ~0.7
  recall pts — we drop it. **HippoRAG** (NeurIPS'24, arXiv:2405.14831):
  PPR seeding with specificity weighting; 10–30× cheaper, 6–13× faster than
  iterative LLM retrieval.
- **PPR at small scale** (survey arXiv:2403.05198; Andersen-Chung-Lang FOCS'06):
  power iteration is sub-10 ms at ≤100k nodes; approximate/incremental
  machinery is for 10M+ graphs. Warm-starting across overlapping seed sets
  is the only "incremental" trick needed.
- **Triggering** (ProCIS, SIGIR'24 arXiv:2405.06460; FLARE arXiv:2305.06983;
  topic segmentation arXiv:2106.12978): fire on evidence change and topic
  drift, not clocks or keywords; debounce; judge the moment jointly with the
  content (npDCG).
- **Rolling summaries** (ReSum arXiv:2509.13313): previous-summary + delta +
  evidence → updated state; bounded prompts regardless of meeting length.
- **Temporal decay** (generative agents arXiv:2304.03442; Zep arXiv:2501.13956;
  Li & Croft CIKM'03; Elasticsearch decay functions): exponential decay with
  per-class half-lives, an offset grace window, multiplied into relevance.
- **Static priors & blending** (Craswell et al. SIGIR'05; Kraaij et al.
  SIGIR'02; Elasticsearch `rank_feature`): log/saturate the prior, single
  tuned weight, prior contributes ~one query-term of influence; raw priors
  are harmful. **Richardson et al. 2006**: graph centrality is a weak quality
  proxy — keep it one factor among several.
- **Undirected PageRank** (Grolmusz arXiv:1205.1960): equals degree iff
  teleport is degree-distributed — a non-uniform trust/recency teleport is
  what makes it informative. **TrustRank** (VLDB'04) / **T-Rank**: teleport
  as trust/freshness prior. **Collusion bound** (Zhang et al. 2004): dup
  cliques amplify rank ≤ 1/(1−α) — dedup before ranking. **Eigenvector
  centrality rejected** (localization, Martin/Zhang/Newman arXiv:1401.5093).
- **Grounded generation** (ALCE EMNLP'23; ReClaim arXiv:2407.01796; MiniCheck
  arXiv:2404.10774): constrained citation to closed fact-ID sets + post-hoc
  validation. **Action items** (ICASSP'23 arXiv:2303.16763; arXiv:2312.17581):
  class imbalance makes recall hopeless — bias to precision, require quoted
  spans.
