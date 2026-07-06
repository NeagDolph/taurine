# Taurine MVP — Design Spec

**Date:** 2026-07-06
**Status:** Draft for review

Taurine is a live meeting copilot: an Electron app that listens to the mic,
streams audio to the Taurine ASR server, and surfaces relevant Apple Notes in
real time — a continuously updating ambient ranking plus gated "pinned" cards
with a one-line "why this matters now" summary. The live loop is a direct
descendant of the wink voice_agent_v4 slate paradigm (managers decide *what's
a candidate*; presenter + cadence decide *what's worth showing right now*),
re-targeted from a product catalog to a personal document corpus.

On top of retrieval, Taurine maintains a **note graph** (Reflect-style):
edges between notes that share topics, entities, or semantic similarity,
built at index time. The graph is (a) rendered as a live visual map — the
current transcription as a node whose edges to retrieved notes appear in
real time, colored by retrieval rank and by hop distance — and (b) used in
retrieval itself: neighbors of retrieved notes are pulled into the candidate
pool before reranking.

## Goals

- Talk about a topic covered in your Notes → the right note appears in the
  ambient list within ~2s and is promoted to the pinned slot with a sensible
  one-liner within ~10s.
- No garbage promotions during unrelated chatter — empty selection is the
  default, silence is a feature.
- Everything off-the-shelf: local embedding model, cloud flash-tier LLMs
  (all via OpenRouter), Cohere Rerank for the retrieval path — no
  custom-trained models anywhere (we're data-poor until real usage exists;
  a custom reranker can replace Cohere later behind the same interface).
- The note graph makes cross-note structure visible (map view) and useful
  (graph-expanded retrieval).

## Non-goals (explicitly out of MVP scope)

- ~~**No logging / data capture of any kind.**~~ REVERSED 2026-07-06 (post-MVP,
  by user request): full per-session JSONL logging now exists — see the
  addendum at the bottom of this spec.
- No replay harness.
- No custom-trained models (no gate, no custom reranker, no fine-tuned
  embeddings). Cohere Rerank (hosted, off-the-shelf) is in scope.
- No system audio capture (but the ingest is N-channel-shaped so it can be
  added without refactoring).
- No external connectors (Google Docs, Linear, email) and no agent actions.
- No conversational presenter memory (one-shot calls only).
- No packaging/signing — the app runs in dev mode.

## Architecture

```
┌────────────────────────┐   binary PCM frames (channel-tagged)   ┌──────────────────────────────┐
│  Electron app (app/)   │ ─────────── localhost ws ────────────► │  taurine-core (core/)        │
│  mic capture · UI      │ ◄────────── JSON events ────────────── │  Python sidecar (FastAPI)    │
└────────────────────────┘   transcript · ambient · pin · status  └──────┬───────────────────────┘
                                                                         │ wss (16 kHz PCM)
                                                                         ▼
                                                          Taurine ASR server (existing)
                                                          wss://transcription.trywink.io
```

Two processes. All intelligence, credentials, and the ASR connection live in
the Python sidecar; Electron stays thin (audio capture + rendering). The
sidecar owns one ASR websocket per audio channel.

### Live loop (inside taurine-core)

```
 transcript chunks (channel-tagged, final)
        │
        ├──► C1 EXACT COLLECTOR ── alias hits ──► instant pin (deterministic, no LLM)
        │                                └──────► slate add (source=exact)
        │
        ├──► C2 AMBIENT RETRIEVER (throttle 8s)
        │      rolling window → local embed → hybrid search (dense + BM25, RRF)
        │      → 1-hop graph expansion (note-graph neighbors join the pool)
        │      → Cohere Rerank over the pooled candidates
        │      top rows ──► slate add (source=search, decay 2)
        │      + retrieval event → live map (rank-colored edges)
        │
        ▼
   ┌─────────────────────────────┐         ┌──────────────────┐
   │  SLATE (max 11)             │◄───────►│  CADENCE POLICY  │
   │  quotas: exact 3 · search 8 │  decide │  debounce 3s     │
   │  decay · views · renewal    │         │  cooldown 20s    │
   │  revision counter           │         │  ≤3 promos/min   │
   └──────┬──────────────────────┘         └────────┬─────────┘
          │ every revision                          │ presenter pass
          ▼                                         ▼
   ambient feed → UI                    PRESENTER (flash LLM via
   (top-k, no LLM)                      OpenRouter, model fallback)
                                        picks 0–2, writes one-liners
                                        → pinned cards
```

## Components

### 1. Electron app (`app/`)

Stack: electron-vite + React + TypeScript.

- **Audio capture** (renderer): `getUserMedia` mic → AudioWorklet →
  downsample to 16 kHz mono s16le → binary frames to the sidecar ws every
  ~250 ms. Each binary frame is prefixed with a 1-byte channel id (`0` =
  mic). Adding system audio later = a second channel id on the same socket.
- **UI** — single window: two switchable main views (Feed / Map) plus
  persistent chrome:
  - **Live transcript** (Feed view, main column): finalized chunks,
    channel-labeled.
  - **Ambient list** (sidebar, always visible): the current slate (≤11
    notes), sorted by relevance — title, gist, score bar. Updates on every
    slate revision. Low-visual-weight; no LLM involvement.
  - **Pinned "now" slot** (top, always visible): promoted cards — note
    title, presenter's one-liner, and an "Open in Notes" button. Max 3
    stacked, newest first. Each card has a dismiss (×) button.
  - **Map view**: the live note graph, detailed below.
- **Map view** — force-directed canvas graph (`react-force-graph-2d`):
  - **Nodes** = notes (label = title, size ∝ degree). Loaded once from
    `GET /graph`, patched after index re-syncs.
  - **Static edges** = note-graph connections (semantic / shared-topic /
    shared-entity), drawn faint, opacity ∝ weight; hover shows edge type
    and the shared topics/entities behind it.
  - **The "now" node**: one distinguished node for the current
    transcription window (label = last few words heard). On every C2 pass
    the sidecar emits a `retrieval` event and the map draws live edges:
    - now-node → directly retrieved note, colored on a **rank gradient**
      (rank 1 hottest → rank K coolest);
    - retrieved note → graph-expanded neighbor, colored on a separate
      **hop scale** with the hop count on the edge — so it's visible which
      candidates arrived through connections rather than direct retrieval,
      and how many steps from a retrieved note they sit.
  - Live edges from the previous pass fade when the next `retrieval` event
    replaces them; touched nodes flash-highlight; notes that got pinned
    keep a persistent ring for the session.
- **Session controls**: start/stop listening; status indicator (ASR
  connected / reconnecting, index freshness, note count).
- **Open in Notes**: clicking a card asks the sidecar to run AppleScript
  `show note id …` so macOS Notes opens the note.

Electron ↔ sidecar protocol (`ws://127.0.0.1:8930/session`):

- client → server, JSON text frames:
  `{type:"start", channels:[{id:0, label:"mic"}]}` · `{type:"stop"}` ·
  `{type:"dismiss", card_id}` · `{type:"open_note", note_id}`
- client → server, binary frames: `[channel_id u8][s16le 16 kHz mono PCM]`
- server → client, JSON events:
  - `{type:"transcript", channel, text, final, ts}`
  - `{type:"ambient", items:[{note_id, title, gist, score}]}` (full
    replacement list, sent on slate revision)
  - `{type:"pin", card:{card_id, note_id, title, summary, provenance, heard}}`
  - `{type:"unpin", card_id}` (dismiss ack / expiry)
  - `{type:"retrieval", heard, hits:[{note_id, rank, score}],
     expansions:[{from_note_id, note_id, hop, score}]}` (one per C2 pass;
     drives the map's live edges)
  - `{type:"status", asr, index:{notes, last_sync}}`

REST (same server): `GET /graph` →
`{nodes:[{note_id, title, degree}], edges:[{src, dst, type, weight,
shared:[…]}]}` · `POST /index/sync`.

### 2. taurine-core (`core/`)

Python 3.11+, managed with `uv`. FastAPI + uvicorn + websockets. Layout:

```
core/taurine_core/
├── server.py          # FastAPI app, /session ws, /index REST
├── config.py          # ALL tunables as in-code constants
├── schemas.py         # pydantic models for ws protocol + internal types
├── asr.py             # Taurine ASR ws client (one connection per channel)
├── llm.py             # OpenRouter client (ALL LLM calls go through this)
├── indexer/
│   ├── notes.py       # Apple Notes extraction (macnotesapp / AppleScript)
│   ├── enrich.py      # per-note LLM enrichment (gist, utterances, aliases, topics)
│   ├── embed.py       # sentence-transformers wrapper
│   ├── graph.py       # note-graph builder (semantic kNN + shared topic/entity)
│   └── store.py       # sqlite + FTS5 + in-memory embedding matrix + edges
└── live/
    ├── ingest.py      # chunk router: transcript → C1/C2, UI fanout
    ├── exact.py       # C1 exact-reference collector
    ├── ambient.py     # C2 ambient retriever
    ├── slate.py       # slate + lifecycle
    ├── presenter.py   # LLM promotion pass
    └── cadence.py     # cadence policy
```

Config lives in `config.py` as constants (project convention: no env vars
for config). The only env vars are secrets: `TAURINE_ASR_API_KEY`,
`OPEN_ROUTER_API_KEY`, `COHERE_API_KEY` (the latter two copied from
`~/wink/server/.env` into `core/.env` at setup — both confirmed present
there under exactly these names).

**All LLM calls — enrichment and presenter — go through the OpenRouter
chat-completions API** (`llm.py`, single client). Model strings are config
constants; provider fallback is just a second model string on the same
client. Cohere Rerank is the one non-OpenRouter remote call (OpenRouter
does not proxy rerank endpoints), hit directly with `COHERE_API_KEY`.

### 3. ASR client (`asr.py`)

Connects to the existing Taurine transcription server
(`wss://transcription.trywink.io`, API-key header) and streams 16 kHz PCM;
receives partial/final text. **Reference implementation:** the client
protocol in `wink/transcription-service/scripts/sherpa_loadtest.py` — the
implementer reads that harness and mirrors its framing/auth/result parsing
exactly rather than reinventing it. One ASR connection per audio channel;
auto-reconnect with exponential backoff (1s → 30s cap); connection state
surfaced to the UI via `status` events. Only **final** chunks enter the live
loop; partials go to the transcript UI only.

### 4. Indexer

**Extraction** (`notes.py`): all notes in the default Apple Notes account
via the `macnotesapp` package (AppleScript-backed — first run triggers the
macOS automation permission prompt; a denied permission is surfaced in the
UI, not fatal). HTML body → plain text. Incremental sync keyed on
`(note_id, content_hash)`: full pass at sidecar startup, then re-sync every
5 minutes and on an explicit `POST /index/sync`. Sync failures are non-fatal
— the live loop keeps serving the last good index.

**Enrichment** (`enrich.py`): one LLM call per new/changed note (flash-tier
model via OpenRouter, JSON output) producing:
- `gist`: one line, ≤120 chars — shown in the ambient list and given to the
  presenter.
- `utterances`: 4 synthetic spoken sentences someone might say in a meeting
  when this note is relevant (the inverse-HyDE register bridge — matching
  speech-against-speech instead of speech-against-notes).
- `aliases`: up to 6 short strings people would call this note or its
  subject (feeds C1).
- `topics`: 2–4 normalized topic tags (lowercase, canonical phrasing) and
  `entities`: named people/projects/products in the note — both feed the
  note graph.

Cached in sqlite by content hash; only changed notes re-enrich. Enrichment
failures degrade gracefully: the note is still indexed on raw text, with
title-only aliases, a truncated first line as gist, and no topic/entity
edges (semantic edges still apply).

**Embedding** (`embed.py`): `BAAI/bge-small-en-v1.5` via
sentence-transformers (local, CPU/MPS, 384-dim). Embedded units per note:
each text chunk (~300 tokens, sentence-boundary splits), the gist, and each
synthetic utterance. All units map back to their note_id.

**Store** (`store.py`): single sqlite file (`taurine.db`) — notes, aliases,
enrichment cache, embedding blobs, and an `edges` table — plus an FTS5 table
over title+body for BM25. At startup the embedding matrix loads into one
numpy array; the corpus is small enough that dense search is brute-force
cosine over the full matrix (microseconds at this scale — no vector DB, no
ANN index).

**Note graph** (`graph.py`): built at index time, rebuilt incrementally —
when a note changes, only its incident edges are recomputed. Two edge
mechanisms, both cheap and off-the-shelf (no extra LLM pass beyond the
enrichment call that already ran):
- **Semantic edges**: per-note embedding = the gist embedding (fallback:
  mean of chunk embeddings). Each note connects to its top
  `GRAPH_KNN=6` nearest notes with cosine ≥ `GRAPH_MIN_SIM=0.60`;
  weight = cosine.
- **Topic/entity edges**: notes sharing a topic tag or entity get an edge;
  weight = shared-term count damped by term frequency across the corpus
  (a topic held by half the notes contributes ~nothing; a rare shared
  entity is a strong link). Floor `GRAPH_MIN_SHARED_W=0.25`.

Parallel edges merge into one edge carrying `type` (semantic | topic |
entity | mixed), combined `weight = 1 - Π(1 - wᵢ)`, and the `shared` terms
for hover display. Undirected. Per-note edge cap `GRAPH_MAX_DEGREE=12`
(keep strongest) so hub notes don't turn the map into a hairball or flood
retrieval expansion.

### 5. Live loop

**Ingest** (`ingest.py`): receives final transcript chunks
`{channel, text, ts}`, maintains the rolling window, fans out to C1
(every chunk) and C2 (throttled), forwards transcript events to the UI, and
after any slate revision emits the ambient feed and offers the cadence
policy a presenter attempt.

**C1 — exact-reference collector** (`exact.py`): normalized substring match
of every alias (note titles + enrichment aliases, lowercased,
punctuation-stripped; aliases under 4 characters require word-boundary
match) against each chunk. A hit:
- adds the note to the slate (`source=exact`, decay 2), and
- if the matched alias is the note's *title*, triggers an **instant pin**
  (deterministic, no LLM — the v4 reference-card analog): summary is the
  note's gist, provenance `exact`. Instant pins consume rate budget but
  start no cooldown.

**C2 — ambient retriever** (`ambient.py`): at most once per
`C2_THROTTLE_S=8`, takes the rolling window (last 3 final chunks or ~12s of
speech, whichever is more text), embeds it, and runs a four-step pass:

1. **Hybrid search** — dense: cosine against all embedding units, per-note
   score = max over its units; sparse: FTS5 BM25 over title+body with the
   window's content words; fused with reciprocal-rank fusion (k=60).
2. **Graph expansion** — the top `EXPAND_SEEDS=5` fused notes seed a 1-hop
   walk over the note graph: each neighbor joins the candidate pool with
   `score = seed_fused_score × edge_weight × HOP_DECAY (0.5)`, tagged
   `{from_note_id, hop:1}`. Already-retrieved notes keep their direct
   ranking (a note reached both ways counts as direct). One hop only at
   MVP.
3. **Cohere Rerank** — the pooled candidates (direct ∪ expanded, ≤ ~20) go
   to Cohere Rerank (`rerank-v3.5`): query = the window text, document =
   title + gist + best-matching chunk per note. Rerank scores replace
   fusion scores for ordering and for the slate/ambient/map displays. On
   Cohere error or timeout (`RERANK_TIMEOUT_S=2`), fall back to the
   fused+expanded order — the pass never dies on the reranker. Latency
   (~100–300 ms) is fine inside an 8 s throttle.
4. **Slate add** — top `C2_TOP_K=5` reranked notes above a loose floor
   (`MIN_COSINE=0.35` on the dense side, applied only to notes with
   neither a BM25 hit nor a graph path) are added to the slate
   (`source=search`, decay 2, `heard=` the window text, carrying
   `via_note_id`/`hop` when they arrived through the graph). The floor is
   deliberately permissive — precision is the presenter's job.

Every pass also emits the `retrieval` event (direct hits with ranks +
expansions with hops) that drives the map view's live edges — this is UI
fanout, not persistence.

**Slate** (`slate.py`): one per session, in-memory (no Redis — single local
session). `SlateItem: {note_id, title, gist, score, source, heard, views,
decay, added_rev, via_note_id?, hop?}` (`via_note_id`/`hop` set when the
item entered the pool through graph expansion).
- Quotas: `exact` 3 · `search` 8 (max 11 total). A lane over quota evicts
  its own stalest item (most-viewed, then oldest).
- Lifecycle: add → each presenter pass increments `views` on every item it
  was shown → kick on the first of: **pinned** (never re-show), `views ≥
  decay`, or **dismissed** (kick + suppress that note_id for
  `SUPPRESS_S=300` so it can't bounce straight back in).
- Re-adding a slated note **renews** it (views reset; provenance upgrades
  to `exact` if the new source is stronger) — a re-mention is a recency
  signal, not a duplicate.
- Revision counter bumps on every add/kick; keys the ambient feed and the
  cadence policy's change detection.
- The ambient feed is the slate sorted by (provenance strength, score),
  full list on every revision.

**Presenter** (`presenter.py`): one-shot structured call through OpenRouter
(`PRESENTER_MODEL` = a flash-tier model, e.g. `deepseek/deepseek-chat`; on
any error, one retry with `PRESENTER_FALLBACK_MODEL` = e.g.
`google/gemini-2.5-flash` on the same client; if both fail, the pass is
skipped silently — no card is always an acceptable outcome). Input:
- `<recent_transcript>`: last 15 final chunks, channel-labeled.
- `<slate>`: one line per item —
  `[exact: title spoken] {note_id} "title" — gist` /
  `[search: matched "…window text…"] {note_id} "title" — gist` /
  `[linked: connected to "seed note title", matched "…"] {note_id} "title"
  — gist` for graph-expanded items, so the model knows the evidence is
  indirect.

Contract: pick **0–2** notes worth putting in front of the user *right
now*; for each, write `why` (≤140 chars) grounded in what was just said;
copy `note_id` exactly (a pick whose note_id isn't on the slate is
dropped); empty pick is the correct output when the moment is wrong —
provenance strength `exact > search > linked`; prefer the moment over the
score.
After the call returns: promoted items become `pin` events, get kicked from
the slate; every item that was in the prompt gets `views += 1`.

**Cadence** (`cadence.py`): decides when a presenter pass runs. A pass is
attempted when the slate revision has changed since the last pass AND all
of: `DEBOUNCE_S=3` since last attempt, `COOLDOWN_S=20` since last
promotion, under `MAX_PROMOS_PER_MIN=3`. An empty selection starts
`EMPTY_BACKOFF_S=15` for the current revision (same pool won't be retried
until it changes or the backoff lapses). An unchanged slate may be
re-offered once after `REOFFER_S=60`. Instant pins (C1) bypass everything
except the per-minute rate cap and start no cooldown.

Pinned cards persist in the UI until dismissed or displaced by newer pins
past the 3-card stack; displacement emits `unpin`.

## Error handling

| failure | behavior |
|---|---|
| ASR ws drops | auto-reconnect w/ backoff; UI status chip; audio frames dropped meanwhile (no buffering) |
| Notes automation permission denied | status event with remediation hint; loop runs with empty/stale index |
| Enrichment LLM error | note indexed un-enriched (raw text, title alias, first-line gist, semantic-only edges) |
| Cohere Rerank error/timeout | fall back to fused+expanded order; pass continues |
| Presenter both-model failure (via OpenRouter) | pass skipped; next trigger per cadence |
| Empty index | ambient list shows empty state; C1/C2 no-op |
| Sidecar not running | Electron shows connect error + retry |

## Testing

`pytest` in `core/`, unique test basenames. No replay harness. Deterministic
unit tests against a fixture corpus of ~15 fake notes (fixtures, not
captured data):

- slate lifecycle: quotas, eviction order, decay/views, renewal semantics,
  dismiss suppression, revision counter;
- cadence: debounce/cooldown/rate-cap/empty-backoff/re-offer transitions
  under a fake clock;
- C1 matcher: alias normalization, word-boundary rule, title-hit → instant
  pin;
- C2 + store: hybrid fusion ordering on the fixture corpus, cosine floor,
  BM25 exact-title behavior; incremental sync (changed hash re-enriches,
  unchanged skips);
- graph builder: semantic kNN edges, topic/entity edge weights and
  frequency damping, parallel-edge merge, degree cap, incremental edge
  recompute on note change;
- graph expansion: neighbor scoring (seed × weight × hop decay),
  direct-beats-expanded rule, `retrieval` event payload shape;
- rerank integration: candidate-pool assembly and fallback-on-error with a
  mocked Cohere client;
- presenter: prompt assembly + response validation (unknown note_id
  dropped, >2 picks truncated) with a mocked LLM client — **no
  prompt-substring assertions** (project convention);
- ws protocol: binary frame parsing (channel byte + PCM), event
  serialization.

The Electron app is verified manually (it's a thin shell); no e2e harness
in the MVP.

## Build process

Implementation is delegated to **Codex subagents** (`codex exec`, model
GPT-5.5, reasoning effort `xhigh` — exact CLI flags verified at build
time), launched detached with output to log files and explicit DONE/ERROR
markers, polled rather than awaited. Claude orchestrates: decomposes into
the component boundaries above, writes task briefs referencing this spec,
reviews diffs, and integrates. Suggested delegation units: (1) store +
indexer + embed + graph builder, (2) slate + cadence + C1/C2 (incl. graph
expansion + rerank), (3) asr client + server + ingest wiring, (4) llm
client + presenter, (5) Electron app: feed view + chrome, (6) Electron map
view. Units 1–4 are testable headless before the app exists.

## Milestones

1. Repo scaffold: `core/` (uv, FastAPI skeleton, config, schemas) +
   `app/` (electron-vite skeleton).
2. Indexer end-to-end: real Apple Notes → enriched, embedded, graph-built,
   queryable store (CLI smoke: `python -m taurine_core.indexer.sync`).
3. Live loop headless: C1/C2 (with graph expansion + rerank)/slate/cadence/
   presenter passing unit tests; manual smoke via a stdin chunk-feeder dev
   entrypoint.
4. ASR client verified against the live Taurine server with a wav file.
5. Electron app: mic → sidecar → transcript + ambient + pins rendering
   (Feed view).
6. Map view: static graph render + live retrieval/expansion edges.
7. Demo pass: live conversation about topics in Neil's real Notes, watched
   from the map.

## Addendum (2026-07-06, post-MVP)

**Session logging (reverses the original non-goal).** One JSONL file per ws
session under `core/sessions/` (`session_log.SessionLogger`; never raises
into the live loop). Record kinds: `session_start/end`, `chunk` (finals +
interims), `asr_status`, `control`, `c1_hit`, `c2_pass` (full retrieval
internals: dense/bm25/fused rows, expansions, rerank status, floor drops,
slate adds), `slate` (every mutation w/ op+reason+revision), `cadence`
(decision + first blocking gate via `Cadence.explain`), `presenter` (model,
raw response, picks, latency), `pin`/`unpin`. REST: `GET /sessions`,
`GET /sessions/{id}`. devfeed logs too.

**Trigger provenance (debug UI).** Final chunks carry monotonic `seq`.
C1 hits and C2 passes emit `TriggerEvent` (trigger_id, kind exact|search,
seqs, span, matches incl. rank/via/hop, action pinned|slated|none);
`RetrievalEvent` carries the correlating `retrieval_id` + `seqs`. The feed
renders exact-match spans as clickable highlights and retrieval windows as
edge accents + chips; clicking opens the trigger inspector (matches table,
Open in Notes, trigger_id/seqs footer for log cross-reference).

**Fixes of note.** OpenRouter calls set `provider.require_parameters` and
parse fence-wrapped JSON (some providers ignore response_format); failed
enrichments are never cached (they retry on later syncs). The REST API has
CORS enabled (the ws never needed it). The app shell is a flex column so the
workspace keeps its height regardless of which conditional banners are
mounted; `/graph` fetch retries every 4s on failure; an initial status event
is pushed on ws connect.
