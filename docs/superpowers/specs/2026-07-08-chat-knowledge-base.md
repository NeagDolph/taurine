# Chat with the Knowledge Base — agentic retrieval + streaming thinking UI

**Date:** 2026-07-08 · **Status:** draft for review (implementation plan +
delegation follow after sign-off)
**Builds on:** `2026-07-08-live-summary-graph-retrieval.md` (latent retrieval:
heads → RRF → PPR → priors; citation registries; fact embeddings). Nothing in
the live voice loop changes; chat is a parallel consumer of the same index,
graph, and blend machinery.

## Product

A **Chat** tab in the app: a modern LLM chat interface over the user's corpus.
The user types a question; Taurine retrieves broadly from the knowledge graph,
lets **GLM-5.2 with thinking enabled** reason over the evidence (visibly —
thinking streams into the UI), optionally refine retrieval through tools, and
answer in streamed markdown with `[N#]`/`[F#]` citations that open note
previews. Chats persist across restarts and are managed as vertical tabs on
the left, like any modern chat application.

Two deliberate contrasts with the live voice loop:

| | Voice (live summary) | Chat |
|---|---|---|
| query | ambient, noisy ASR window — no intent | typed, intentional question |
| retrieval | zero-shot latent only, LLM never steers it | latent **context pack** first, then the LLM iterates via tools |
| LLM budget | one flash pass, latency-bound | thinking model, quality-bound; streaming hides latency |
| trigger | evidence-change detection | the user pressing Enter |
| recall posture | precise top-12 (copilot must not distract) | broad ("find many relevant sources") — top-16 notes + 24 facts + tool expansion |
| context continuity | meeting seed accumulator | per-chat seed accumulator + conversation history |

---

# Section 1 — Retrieval for typed prompts (latent context pack + LLM tooling)

## 1.1 Why text retrieval differs from the voice path

The voice path compensates for *queryless, noisy* input: GLiNER span
extraction, fuzzy ASR snapping, throttles, hysteresis, and change triggers all
exist because nobody asked a question. A typed prompt is the opposite: clean,
short, intentional — but it may be **underspecified relative to the corpus**
("what did we decide about the rollout?") and it deserves **iteration**: a
thinking model can notice the context pack lacks the answer and go look again.
So chat retrieval is two-stage: a fast latent Stage A that reuses our upfront
graph investment verbatim, and an agentic Stage B where the model steers.

## 1.2 Stage A — latent context pack (zero LLM, target < 300 ms warm)

On every user message, before the model is called:

1. **Heads** (reused from ambient, speech-specific parts dropped):
   - `dense(message)` — Qwen3 asymmetric query embedding. A chat-specific
     instruct prompt (`CHAT_EMBED_QUERY_PROMPT`: "Given a question about the
     user's personal notes, retrieve notes that answer it") replaces the
     spoken-conversation prompt.
   - `bm25(message)` — unchanged.
   - `exact` — alias word-boundary matches over the message, injected at
     rank 1 (typed entity names are high-precision signals).
   - GLiNER2 spans and fuzzy ASR snapping are **not used**: typed text has no
     spelling drift to bridge, and dense+BM25+exact cover clean entity
     mentions. (Cheap to add later behind a flag if entity-dense queries
     underperform.)
   - `fact_dense_search(message, CHAT_CONTEXT_FACTS)` — query-to-fact
     matching; endpoint notes get the half-score relevance credit exactly as
     in grounding §1.3.
2. **RRF fusion** (`RRF_K=60`) over the head lists.
3. **Per-chat seed accumulator** — same mechanism as the meeting accumulator
   but scoped to the chat thread and decayed **per turn**
   (`CHAT_SEED_DECAY = 0.7`, faster than the meeting's 0.85 — conversations
   pivot harder than meetings). Fused top-`SEED_TOP_K` add in; exact hits at
   list-max. This is what makes "what about the compliance part?" retrieve in
   the context of the previous question without any query rewriting.
4. **PPR traversal** over the boosted graph (`PPR_DAMPING`, warm-started per
   chat, matrix shared/rebuilt on `graph_version` change — reuse
   `NoteGraphMatrix` and the GroundingEngine cache pattern).
5. **Blend** — identical formula: `rel = 0.65·norm(RRF) ⊕ 0.35·norm(PPR) +
   fact bonus`, `final = rel × recency × source_rank × exp(0.2·quality)`.
   Recency/priors matter just as much here: an old duplicate of a fresh note
   should lose.
6. **Context pack**: top `CHAT_CONTEXT_NOTES = 16` notes and
   `CHAT_CONTEXT_FACTS = 24` facts — broader than the voice top-12 because
   the failure mode inverts (a chat that misses a source is worse than one
   that cites an extra note; the model adjudicates relevance). **No
   hysteresis** — each turn retrieves fresh; continuity lives in the seed
   accumulator, not in set stickiness.

**Implementation shape:** a `ChatRetriever` in `core/taurine_core/chat/retrieval.py`
wrapping a per-chat `GroundingEngine` variant. Refactor guidance: extract the
head-running + RRF from `AmbientRetriever` into a shared helper rather than
duplicating; `GroundingEngine` gains constructor knobs
(`top_k`, `facts_max`, `seed_decay`, `hysteresis: float | None`,
`query_prompt`) instead of a parallel class. The `N#`/`F#` registries are
**per chat and persistent** (stored with the chat, restored on load) so
citations in old messages keep resolving after restart.

## 1.3 Stage B — LLM tooling (the model steers)

The chat model receives the context pack in its prompt AND three tools; every
call and result is surfaced to the UI as a retrieval-activity step (§3.4):

- `search_notes(query: str, k: int = 8)` — runs the FULL Stage-A latent
  pipeline (heads → RRF → PPR seeded by the current accumulator ∪ this query
  → blend) on a model-formulated query. Returns per note: short_id, title,
  source, age, score, gist. New notes join the chat's citation registry.
  This is the workhorse: coreference ("her timeline") gets resolved by the
  model writing a concrete query ("Ashish integration timeline"), not by a
  separate rewriting pass.
- `read_note(short_id_or_note_id)` — title, source, age, full text capped at
  `CHAT_NOTE_READ_MAX_CHARS = 6000`, plus its edge facts and neighbor titles.
- `expand_note(short_id_or_note_id)` — 1-hop traversal: neighbors with edge
  weight, type, and adjudicated fact, sorted by boosted weight. Gives the
  model deliberate graph navigation ("what else connects to this?") — the
  agentic complement to PPR's automatic diffusion.

Budget: `CHAT_MAX_TOOL_CALLS = 6` per turn (voice presenter precedent: hard
cap, then the model must answer with what it has). Tool responses are JSON;
tool *summaries* (one line) stream to the UI immediately, full payloads are
session-logged.

**Grounding contract:** answers cite `[N#]`/`[F#]` from the context pack ∪
tool results only. Validation is *lenient* compared to the summarizer:
unknown citations are stripped and logged, but the answer is never rejected —
in an interactive setting the user sees and judges the text; blocking the
reply is worse than an uncited claim. A `sources` array (resolved citations
with metadata) accompanies the final message so the UI renders a source rail
without re-parsing markdown.

## 1.4 Model, streaming, and the llm.py extension

- `CHAT_MODEL = "z-ai/glm-5.2"` with **reasoning enabled per-call** —
  `LLM_REASONING_BY_MODEL` globally disables GLM reasoning (enrichment wants
  that); the streaming API therefore takes an explicit `reasoning` override
  parameter instead of the per-model map. `CHAT_REASONING = {"enabled": True}`.
- `CHAT_FALLBACK_MODEL = "deepseek/deepseek-v4-pro"` (reasoning-capable; one
  retry on stream-open failure only — mid-stream errors surface as
  `chat_error` with a retry affordance, no silent model switch mid-answer).
- **New `OpenRouterClient.stream_chat(...)`** — the one substantive llm.py
  addition:
  ```python
  async def stream_chat(
      self, model: str, messages: list[dict], *,
      tools: list[dict] | None = None,
      reasoning: dict | None = None,        # overrides LLM_REASONING_BY_MODEL
      timeout_s: float,
  ) -> AsyncIterator[StreamEvent]
  ```
  SSE (`"stream": true`) via `httpx.AsyncClient.stream`; yields typed events:
  `reasoning_delta(text)` (OpenRouter normalizes GLM thinking into
  `delta.reasoning` / `reasoning_details`), `content_delta(text)`,
  `tool_call(id, name, arguments)` (accumulated from argument fragments,
  yielded complete), `usage(prompt, completion, reasoning_tokens)`, `done`.
  Cancellation = closing the stream (caller cancels the task). Provider
  routing keeps `require_parameters: true`.
- **Turn loop** (`chat/engine.py`): system prompt + history + context pack →
  `stream_chat` → forward reasoning/content deltas to the ws as they arrive;
  on `tool_call`, execute locally, append the tool message, re-open the
  stream (each round trip streams its own thinking). Loop ends on a content
  completion with no tool calls or at the tool budget. History budget:
  last `CHAT_HISTORY_MAX_TURNS = 12` turns verbatim; older turns drop (no
  summarization pass in v1 — non-goal).

## 1.5 Persistence

New sqlite tables (same `taurine.db`, migrations follow house style):

```sql
CREATE TABLE chats (
    chat_id TEXT PRIMARY KEY,            -- "chat-<8 hex>"
    title TEXT NOT NULL DEFAULT '',      -- auto-titled after first exchange
    created REAL NOT NULL,
    updated REAL NOT NULL,
    registry_json TEXT NOT NULL DEFAULT '{}'   -- N#/F# registries + seed accumulator
);
CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content TEXT NOT NULL,               -- markdown (assistant) / raw text (user)
    thinking TEXT NOT NULL DEFAULT '',   -- full reasoning transcript (assistant)
    sources_json TEXT NOT NULL DEFAULT '[]',
    activity_json TEXT NOT NULL DEFAULT '[]',  -- tool/retrieval steps for replay
    usage_json TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL
);
```

Auto-title: after the first completed exchange, one non-streamed
`complete_json` call to `SUMMARY_MODEL` (flash-tier — cheap, not the thinking
model): "≤6-word title for this conversation". Failure → first 40 chars of
the first user message.

## 1.6 Concurrency & isolation

- Retrieval is read-only against the shared `Index` (atomic snapshot tuples
  make it safe under connector-sync threads — already verified).
- Chat `GroundingEngine` instances are per-chat: no interference with a live
  voice session's accumulator or registries. Graph matrix rebuilds are cached
  per engine on `graph_version`, same as live.
- One in-flight turn per chat (`chat_send` while streaming → rejected with
  `chat_error code="busy"`); different chats may stream concurrently
  (`CHAT_MAX_CONCURRENT_TURNS = 2`).
- Session-log: new record kinds `chat_turn` (chat_id, retrieval stats, tool
  calls with latencies, usage, ttft_ms, total_ms) and `chat_retrieval`
  (per-search head/blend diagnostics, mirroring `c2_pass`).

## 1.7 Config (new knobs)

```python
# ── chat (knowledge-base Q&A) ──────────────────────────────────────────
CHAT_MODEL = "z-ai/glm-5.2"
CHAT_REASONING = {"enabled": True}          # per-call override: thinking ON for chat
CHAT_FALLBACK_MODEL = "deepseek/deepseek-v4-pro"
CHAT_TIMEOUT_S = 120.0                      # whole-stream budget per round trip
CHAT_MAX_TOOL_CALLS = 6
CHAT_MAX_CONCURRENT_TURNS = 2
CHAT_CONTEXT_NOTES = 16                     # Stage-A pack: broad recall posture
CHAT_CONTEXT_FACTS = 24
CHAT_SEED_DECAY = 0.7                       # per-turn accumulator decay (chats pivot fast)
CHAT_SEARCH_TOOL_K = 8                      # default k for search_notes
CHAT_NOTE_READ_MAX_CHARS = 6000
CHAT_HISTORY_MAX_TURNS = 12
CHAT_EMBED_QUERY_PROMPT = (
    "Instruct: Given a question about the user's personal notes, retrieve "
    "notes relevant to answering it\nQuery: "
)
CHAT_TITLE_MAX_WORDS = 6
```

---

# Section 2 — Chat UI (streaming, thinking, chat management)

## 2.1 WS protocol — dedicated `/chat` endpoint

A separate websocket (`ws://127.0.0.1:8930/chat`) rather than multiplexing
`/session`: the live socket is audio-framed and session-scoped; chat is
request/response-streamed and multi-thread. One `/chat` socket serves all
chats (events carry `chat_id`).

**client → server**
```jsonc
{"type": "chat_send",   "chat_id": "chat-1a2b3c4d", "text": "..."}   // chat_id "" → server creates chat, first event returns it
{"type": "chat_cancel", "chat_id": "..."}                             // abort in-flight turn (partial text is kept + persisted, marked cancelled)
{"type": "chat_regenerate", "chat_id": "..."}                         // re-run last user message (previous assistant message replaced)
```

**server → client** (all carry `chat_id`; `message_id` = the assistant row)
```jsonc
{"type": "chat_ack",       "message_id", "user_message_id", "title"}          // user msg persisted; title present when just auto-created
{"type": "chat_retrieval", "message_id", "sources": [GroundingItem...], "facts": [GroundedFact...]}  // Stage-A pack, before first token
{"type": "chat_thinking",  "message_id", "delta": "..."}                       // reasoning stream
{"type": "chat_activity",  "message_id", "step": {"kind": "search|read|expand", "label": "Searched “shopify timeline” → 8 notes", "status": "running|done", "note_ids": [...]}}
{"type": "chat_delta",     "message_id", "delta": "..."}                       // answer markdown stream
{"type": "chat_done",      "message_id", "content", "thinking", "sources", "usage": {"prompt","completion","reasoning","ttft_ms","total_ms"}, "cancelled": false}
{"type": "chat_error",     "message_id?", "code": "busy|llm|timeout", "detail"}
{"type": "chat_meta",      "chat": {"chat_id","title","updated"}}              // title/rename/updated fanout
```

**REST** (list/history/manage — ws is for streaming only):
`GET /chats` → `[{chat_id, title, created, updated, message_count}]` ·
`POST /chats` · `GET /chats/{id}` → full messages (content, thinking,
sources, activity — enough to fully re-render) · `PATCH /chats/{id}`
(rename) · `DELETE /chats/{id}`.

Ordering guarantee: per `message_id`, events arrive
`chat_ack → chat_retrieval → (chat_thinking* | chat_activity* | chat_delta*)*
→ chat_done|chat_error`. Thinking and activity may interleave (each tool
round trip thinks again); the UI renders them in arrival order as one
timeline.

## 2.2 Layout — new top-level "Chat" tab

```
┌──────────────────────────────────────────────────────────────┐
│ Taurine   [Live] [Chat] [Data] [Connectors]        ● session │
├───────────────┬──────────────────────────────────────────────┤
│ + New chat    │  ┌ thread ────────────────────────────────┐  │
│───────────────│  │        (user bubble, right-aligned)  ⧉ │  │
│ ▸ Shopify     │  │  ┌ assistant turn ───────────────────┐ │  │
│   rollout     │  │  │ ▸ Thought for 12s        (collaps.)│ │  │
│   2m ago      │  │  │ 🔍 Searched "…" → 8 notes          │ │  │
│ ▸ Whisper     │  │  │ 📄 Read: Ashish Parik — Integration│ │  │
│   capacity    │  │  │ Answer markdown streams here with  │ │  │
│   1h ago      │  │  │ [N3] [F7] citation chips…      ⧉ ↻ │ │  │
│   …           │  │  │ ── sources: [N3] [N7] [F7] +4 ──── │ │  │
│               │  │  └────────────────────────────────────┘ │  │
│               │  └──────────────────────────────────────────┘ │
│               │  [ composer …                       ⏎ / ⏹ ]  │
└───────────────┴──────────────────────────────────────────────┘
```

### Chat sidebar (left, vertical tabs)
- **New chat** button pinned at top (creates locally; chat persists
  server-side on first send — no empty-chat rows).
- One row per chat: title (single line, ellipsized), relative timestamp
  ("2m", "1h", "Mar 3"), active state (accent left-border, matching the main
  tab affordance), sorted by `updated` desc.
- Hover actions: **rename** (inline edit, Enter/Escape) and **delete**
  (two-step: click → "Delete?" confirm state, no modal).
- A subtle streaming indicator (pulsing dot) on chats with an in-flight turn
  that aren't focused.
- Collapsible on narrow windows (same responsive breakpoint discipline as the
  existing workspace grid).

### Thread pane
- **User messages**: compact right-aligned bubbles, raw text preserved
  (whitespace-pre-wrap), copy button on hover.
- **Assistant turn** — a single card containing, in arrival order:
  1. **Thinking accordion**: while `chat_thinking` streams, shows header
     "Thinking…" with a shimmer animation and the live reasoning text in a
     muted, smaller monospace-adjacent style, auto-scrolled, max-height with
     internal scroll. On first `chat_delta` (or `chat_done`) it collapses to
     **"Thought for {N}s"** (wall-clock from first to last thinking delta);
     click re-expands the full transcript. Persisted thinking re-renders
     collapsed on chat load.
  2. **Activity steps**: one line per tool call — icon by kind
     (search/read/expand), label from `chat_activity`, spinner while
     `running`, subdued check when `done`. Clicking a step with `note_ids`
     opens the first note's preview window.
  3. **Answer**: streamed markdown via the existing `react-markdown` +
     `markdown-body` styling. Citation chips reuse the SummaryPanel
     link-rewrite pattern (`[N#]`/`[F#]` → chip → `api.openPreview(note_id)`,
     hover tooltip = title/fact text). Code blocks: styled `<pre>` with a
     per-block **copy** button (no syntax-highlight dependency in v1).
     Streaming renders progressively (re-render on delta batch; batch deltas
     with a ~30 ms flush like the transcript feed to avoid re-render churn).
  4. **Footer row** (on completion): **copy** (raw markdown to clipboard,
     button flashes "Copied"), **regenerate** (last assistant message only),
     token/latency subtext ("2.1k tok · 14s"), and a **sources row** — the
     context-pack + tool-found chips (`[N3] Ashish Parik…`, source badge,
     age chip — reuse `GroundingList` chip components), collapsed to one
     line with "+N more" expander.
- **Streaming affordances**: auto-follow scroll while pinned to bottom;
  "↓ Jump to latest" pill when the user scrolls up (reuse FeedView's
  auto-follow discipline); blinking cursor block at the stream tail.
- **Cancel**: composer's send button morphs to **stop** (⏹) while streaming
  → `chat_cancel`; partial answer stays with a subdued "stopped" marker.
- **Error state**: inline card-level error with `detail` and a **retry**
  button (re-sends the same user message); `busy` errors toast without
  clearing the composer.
- **Empty states**: no chats → centered "Ask your notes anything" with 3
  example prompts (clicking fills the composer); empty thread → same.

### Composer
- Auto-growing textarea (1→8 rows), Enter sends, Shift+Enter newline,
  disabled-with-spinner never — input stays editable while streaming (send
  is what's gated), ⌘K clears. Focus on chat switch.

## 2.3 Frontend architecture

- `lib/chat.ts` — `ChatClient`: second `SidecarClient`-style ws wrapper for
  `/chat` (same reconnect/backoff discipline), typed event parsers (house
  style: hand-rolled guards), REST helpers (`fetchChats`, `fetchChat`,
  `createChat`, `renameChat`, `deleteChat`).
- State: a `useChats` reducer in `ChatView` — `chats[]`, `activeChatId`,
  `messagesByChat`, `streamingByChat` (thinking buffer, activity steps,
  content buffer, phase: `retrieving|thinking|answering|done`). Delta events
  append to buffers; `chat_done` swaps buffer → persisted message. History
  loads lazily per chat on first focus (REST), cached thereafter.
- Components: `views/ChatView.tsx` (layout + reducer), `components/chat/`
  `ChatSidebar.tsx`, `ChatThread.tsx`, `AssistantTurn.tsx` (thinking
  accordion + activity + markdown + footer), `Composer.tsx`,
  `MessageMarkdown.tsx` (shared citation-chip markdown renderer — extract
  from SummaryPanel and reuse in both).
- App.tsx: add the tab; `/chat` ws connects lazily on first tab focus and
  stays connected (background chats keep streaming).
- CSS: extend `main.css` house palette; thinking shimmer = existing score-bar
  gradient animation pattern; no new UI dependencies (react-markdown +
  existing stack only).

## 2.4 Modern-chat element checklist (acceptance)

Streaming markdown ✓ · visible+collapsible thinking with duration ✓ ·
retrieval/tool activity timeline ✓ · per-message and per-code-block copy ✓ ·
regenerate ✓ · cancel mid-stream ✓ · vertical chat tabs with rename/delete/
auto-title/relative time ✓ · citation chips → note preview ✓ · sources row ✓ ·
auto-follow + jump-to-latest ✓ · empty/error/busy states ✓ · token/latency
footer ✓ · keyboard: Enter/Shift+Enter/⌘K/focus management ✓.

---

## Testing

- `test_chat_retrieval.py`: context-pack assembly (pack sizes, no hysteresis,
  per-turn seed decay, registry stability across turns and across
  save/restore), search-tool pipeline parity with Stage A, fact credit,
  exact-at-rank-1.
- `test_chat_engine.py` (mocked stream): turn loop — deltas forwarded in
  order, tool round trips (call → execute → re-stream), tool budget, unknown
  citation stripped, lenient non-rejection, cancellation persists partial,
  busy rejection, fallback on stream-open failure only, auto-title fallback.
- `test_llm_stream.py`: SSE parsing against canned OpenRouter streams —
  reasoning deltas, split tool-call argument fragments, usage extraction,
  provider error mid-stream.
- `test_chat_store.py`: chat/message CRUD, cascade delete, registry
  round-trip, ordering.
- `test_server_chat_ws.py`: protocol round-trip, event ordering guarantee,
  REST endpoints, concurrent-chat streaming, cancel.
- No prompt-substring assertions (house rule). Frontend verified via
  typecheck + manual run (house rule).

## Non-goals (v1)

- Cross-chat memory or chat→graph writeback; message editing/branching;
  attachments; voice input in chat; syntax highlighting; export; search over
  chats; history summarization beyond the turn cap; mobile layout.

## Execution split (preview — final plan after spec review)

- Contracts (Claude): config, schemas (ChatMeta, ChatMessage, ws events),
  store migrations + helpers, module skeletons, `StreamEvent` types.
- Unit K (codex): `llm.py stream_chat` + SSE parsing + tests.
- Unit L (codex): chat store + `ChatRetriever`/engine-knob refactor of
  GroundingEngine + tools + tests.
- Unit M (codex, after K+L): turn engine + `/chat` ws + REST + session-log +
  tests.
- Frontend (Claude): ChatView + components + CSS.
- Verification (Claude): live Q&A against the real corpus (grounding quality,
  TTFT, thinking stream), full suite, cost check (GLM-5.2 thinking tokens per
  answer).
