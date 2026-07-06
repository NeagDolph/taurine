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

## Goals

- Talk about a topic covered in your Notes → the right note appears in the
  ambient list within ~2s and is promoted to the pinned slot with a sensible
  one-liner within ~10s.
- No garbage promotions during unrelated chatter — empty selection is the
  default, silence is a feature.
- Everything off-the-shelf: local embedding model, cloud flash-tier LLMs,
  no custom-trained models anywhere.

## Non-goals (explicitly out of MVP scope)

- **No logging / data capture of any kind.** No decision logs, no feedback
  capture for future training. Dismiss buttons exist purely as UX.
- No replay harness.
- No custom-trained models (no gate, no reranker, no fine-tuned embeddings).
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
        │      top rows ──► slate add (source=search, decay 2)
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
   ambient feed → UI                    PRESENTER (DeepSeek Flash,
   (top-k, no LLM)                      Gemini fallback) picks 0–2,
                                        writes one-liners → pinned cards
```

## Components

### 1. Electron app (`app/`)

Stack: electron-vite + React + TypeScript.

- **Audio capture** (renderer): `getUserMedia` mic → AudioWorklet →
  downsample to 16 kHz mono s16le → binary frames to the sidecar ws every
  ~250 ms. Each binary frame is prefixed with a 1-byte channel id (`0` =
  mic). Adding system audio later = a second channel id on the same socket.
- **UI** — single window, three regions:
  - **Live transcript** (main column): finalized chunks, channel-labeled.
  - **Ambient list** (sidebar): the current slate (≤11 notes), sorted by
    relevance — title, gist, score bar. Updates on every slate revision. Low-visual-weight;
    no LLM involvement.
  - **Pinned "now" slot** (top): promoted cards — note title, presenter's
    one-liner, and an "Open in Notes" button. Max 3 stacked, newest first.
    Each card has a dismiss (×) button.
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
  - `{type:"status", asr, index:{notes, last_sync}}`

### 2. taurine-core (`core/`)

Python 3.11+, managed with `uv`. FastAPI + uvicorn + websockets. Layout:

```
core/taurine_core/
├── server.py          # FastAPI app, /session ws, /index REST
├── config.py          # ALL tunables as in-code constants
├── schemas.py         # pydantic models for ws protocol + internal types
├── asr.py             # Taurine ASR ws client (one connection per channel)
├── indexer/
│   ├── notes.py       # Apple Notes extraction (macnotesapp / AppleScript)
│   ├── enrich.py      # per-note LLM enrichment (gist, utterances, aliases)
│   ├── embed.py       # sentence-transformers wrapper
│   └── store.py       # sqlite + FTS5 + in-memory embedding matrix
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
`DEEPSEEK_API_KEY`, `GEMINI_API_KEY`.

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

**Enrichment** (`enrich.py`): one LLM call per new/changed note
(DeepSeek Flash, JSON output) producing:
- `gist`: one line, ≤120 chars — shown in the ambient list and given to the
  presenter.
- `utterances`: 4 synthetic spoken sentences someone might say in a meeting
  when this note is relevant (the inverse-HyDE register bridge — matching
  speech-against-speech instead of speech-against-notes).
- `aliases`: up to 6 short strings people would call this note or its
  subject (feeds C1).

Cached in sqlite by content hash; only changed notes re-enrich. Enrichment
failures degrade gracefully: the note is still indexed on raw text, with
title-only aliases and a truncated first line as gist.

**Embedding** (`embed.py`): `BAAI/bge-small-en-v1.5` via
sentence-transformers (local, CPU/MPS, 384-dim). Embedded units per note:
each text chunk (~300 tokens, sentence-boundary splits), the gist, and each
synthetic utterance. All units map back to their note_id.

**Store** (`store.py`): single sqlite file (`taurine.db`) — notes, aliases,
enrichment cache, embedding blobs — plus an FTS5 table over title+body for
BM25. At startup the embedding matrix loads into one numpy array; the corpus
is small enough that dense search is brute-force cosine over the full
matrix (microseconds at this scale — no vector DB, no ANN index).

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
speech, whichever is more text), embeds it, and runs hybrid search:
- dense: cosine against all embedding units; per-note score = max over its
  units;
- sparse: FTS5 BM25 over title+body with the window's content words;
- fusion: reciprocal-rank fusion (k=60) over the two lists.

Top `C2_TOP_K=5` fused notes above a loose floor (`MIN_COSINE=0.35` on the
dense side, applied only to notes with no BM25 hit) are added to the slate
(`source=search`, decay 2, `heard=` the window text). The floor is
deliberately permissive — precision is the presenter's job; the floor only
keeps pure noise out of the ambient list.

**Slate** (`slate.py`): one per session, in-memory (no Redis — single local
session). `SlateItem: {note_id, title, gist, score, source, heard, views,
decay, added_rev}`.
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

**Presenter** (`presenter.py`): one-shot structured call to
`deepseek-chat` (DeepSeek native API; on any error, one retry via
`gemini-flash`; if both fail, the pass is skipped silently — no card is
always an acceptable outcome). Input:
- `<recent_transcript>`: last 15 final chunks, channel-labeled.
- `<slate>`: one line per item —
  `[exact: title spoken] {note_id} "title" — gist` /
  `[search: matched "…window text…"] {note_id} "title" — gist`.

Contract: pick **0–2** notes worth putting in front of the user *right
now*; for each, write `why` (≤140 chars) grounded in what was just said;
copy `note_id` exactly (a pick whose note_id isn't on the slate is
dropped); empty pick is the correct output when the moment is wrong —
provenance strength `exact > search`; prefer the moment over the score.
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
| Enrichment LLM error | note indexed un-enriched (raw text, title alias, first-line gist) |
| Presenter both-provider failure | pass skipped; next trigger per cadence |
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
indexer + embed, (2) slate + cadence + C1/C2, (3) asr client + server +
ingest wiring, (4) presenter, (5) Electron app. Units 1–4 are testable
headless before the app exists.

## Milestones

1. Repo scaffold: `core/` (uv, FastAPI skeleton, config, schemas) +
   `app/` (electron-vite skeleton).
2. Indexer end-to-end: real Apple Notes → enriched, embedded, queryable
   store (CLI smoke: `python -m taurine_core.indexer.sync`).
3. Live loop headless: C1/C2/slate/cadence/presenter passing unit tests;
   manual smoke via a stdin chunk-feeder dev entrypoint.
4. ASR client verified against the live Taurine server with a wav file.
5. Electron app: mic → sidecar → transcript + ambient + pins rendering.
6. Demo pass: live conversation about topics in Neil's real Notes.
