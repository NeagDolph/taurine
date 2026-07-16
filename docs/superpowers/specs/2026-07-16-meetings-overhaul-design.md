# Meetings Overhaul — Granola-Style Notepad, System Audio, Call Detection

**Date:** 2026-07-16
**Status:** Approved direction (Neil, 2026-07-16): system-audio second Deepgram stream with other-party labeling; unified chat/notes/meeting interface inspired by Granola (WYSIWYG markdown notepad, live summaries insertable into notes, user notes feeding the live summary LLM, end-of-call final summary via Gemini 3.5 Flash with graph access, Google-Docs-style AI edit suggestions, Taurine notes as their own implicit source); call-start global notification with countdown auto-enable. Neil delegated all design decisions ("completely drive a new feature to completion within the main branch").
**Grounding:** full architecture maps of `app/` and `core/` (2026-07-16), plus external research on macOS audio capture, editor libraries, and Granola UX.

## 0. What exists today (relevant seams)

- **Audio protocol is already multi-channel.** Renderer sends `[channelByte][pcm s16le @16kHz]` frames over the `/session` WS ([audio.ts:67](app/src/renderer/src/lib/audio.ts), [server.py:1334](core/taurine_core/server.py)); `ASRClient` opens one Deepgram socket per channel int ([asr.py:48-77](core/taurine_core/asr.py)); `Chunk.channel` flows end-to-end into the ledger and `TranscriptEvent`. Only channel 0 (mic) is wired.
- **Live mode is fully manual** (header "Start listening" button). No call detection, no in-app prompt exists on main.
- **No session persistence.** Transcripts go to a debug JSONL only; `MeetingSummary` dies with the socket. No meetings store, no session→note ingestion, no `taurine` source.
- The live tab is a fixed 2×2 grid (SummaryPanel / ActionableRail / FeedView / AmbientList) inside `App.tsx`; all views stay mounted, state lives in `App.tsx` useState hooks.
- Live summarizer: `deepseek/deepseek-v4-flash`, strict JSON + citation validation, prompt assembled in `Summarizer._build_user_prompt` ([summarizer.py:280](core/taurine_core/live/summarizer.py)) — the natural injection point for user notes.
- Chat: `/chat` WS, `glm-5.2`, injection points for extra context blocks already exist in `_build_messages` ([engine.py:935](core/taurine_core/chat/engine.py)); `seed_note_ids` biases retrieval.
- Notes enter the index only as `NoteRecord`s keyed by `source` (= connector id) via `replace_source_records` → global sync rebuild. Source registries for UI: `SOURCE_COLORS/LABELS` (graph.ts), `sourceLabel` (labels.ts), `.source-{id}` chips (main.css), `SOURCE_RANK`/half-lives (config.py).

## 1. System audio → second Deepgram stream ("Them" channel)

**Capture: CoreAudio process tap via the `audiotee` npm package (main process), not getDisplayMedia loopback.**
Rationale: the tap path needs only the soft "System Audio Recording Only" TCC prompt (`NSAudioCaptureUsageDescription`), shows no purple screen-recording indicator, needs no app restart, and avoids Electron 39/40's loopback regressions (audio-only `getDisplayMedia` unreliable; silent zero-filled streams). `audiotee` spawns a small signed Swift binary using `CATapDescription`/`AudioHardwareCreateProcessTap` and emits PCM chunks in the main process. Requires macOS 14.2+; on older macOS the feature is hidden and mic-only behavior is unchanged.

**Pipeline:**
1. Main process: `SystemAudioTap` wrapper around `audiotee` at 48 kHz mono s16; chunks forwarded to the renderer over IPC (`sysaudio:chunk`), start/stop via `sysaudio:start`/`sysaudio:stop` handles exposed in preload.
2. Renderer: `SystemAudioCapture` (sibling of `PcmAudioCapture`) decimates 48 kHz → 16 kHz (exact 3:1, average-of-3), frames 250 ms, prefixes **channel byte 1**, sends on the same `/session` socket. All audio sending stays in the renderer — one socket owner, symmetric with the mic path.
3. Session start declares both channels: `{"type":"start","channels":[{"id":0,"label":"mic"},{"id":1,"label":"system"}]}`. The sidecar already spins up a second, independent Deepgram connection (nova-3, linear16, 16 kHz) for channel 1 — no core transport changes.

**Labeling (UI + agents):** channel 0 = **You**, channel 1 = **Them**.
- Renderer: `FeedView.channelLabel` maps 0→"You", 1→"Them"; per-channel interims already exist.
- Ledger: transcript evidence rendering ([ledger.py:174](core/taurine_core/live/ledger.py)) becomes `[T#] You: "text"` / `[T#] Them: "text"` / `[T#] Ch{n}: "text"` — the `[T#]` tokens are load-bearing (the summarizer's citation validator parses them) and must be preserved exactly.
- Summarizer + final-summary prompts state the convention: "Transcript lines are labeled You (the user's mic) and Them (other call participants)." Attribution in summaries/actionables follows from that.

**Permissions/build:** add `NSAudioCaptureUsageDescription` to `electron-builder.yml` `extendInfo`. Known risk: TCC attribution for the helper binary in `electron-vite dev` (unsigned dev parent) — verified by an early spike task; fallback is running the tap binary with explicit responsibility or documenting a one-time grant.

**Echo caveat (accepted for v1):** without headphones, the mic also hears "Them", producing near-duplicate lines on both channels. Deepgram-side echo cancellation is out of scope; a cross-channel near-duplicate suppressor in the ledger is a logged follow-up.

## 2. Meetings: unified notes + chat + live summary interface

### 2.1 Persistence (sidecar)

New `meetings` store (same SQLite state DB as chats), REST + events:

- Table `meetings(meeting_id TEXT PK, title TEXT, created_at REAL, updated_at REAL, started_at REAL, ended_at REAL, notes_md TEXT, notes_rev INTEGER, final_summary_md TEXT, suggestions_json TEXT, transcript_json TEXT, chat_id TEXT, status TEXT)` — `status ∈ {draft, live, summarizing, done}`, stored on the existing `Index` store (`indexer/store.py`, same connection/DDL-init as chats). `transcript_json` holds the labeled transcript (list of `{channel,text,ts,seq}`) for reload/final-summary input; it is not indexed in v1 and is written in batches (every ~20 finals or 10 s, and on stop), not per chunk.
- REST: `GET /meetings` (list, newest first), `POST /meetings` (create draft), `GET /meetings/{id}`, `PATCH /meetings/{id}` (title, notes_md — debounced autosave target; bumps `notes_rev`/`updated_at`), `DELETE /meetings/{id}`.
- `/updates` bus gains kind `meetings` (list invalidation; published on meeting create/delete/title/status change — not per transcript chunk).
- The `start` control message gains `meeting_id`; a live session is always bound to a meeting (the renderer creates one first if needed). Each `start` gets a fresh live-session state (a `stop`→`start` sequence on one socket must not leak ledger/summary state between meetings). `stop` triggers the end-of-call pipeline (§2.4). **Disconnect grace:** socket loss while live starts a 60 s grace timer instead of finalizing; the client's reconnect replays its full last start message (meeting_id + channels), cancelling the timer and resuming the same meeting; timer expiry finalizes. The client reconnect replay must carry the real last `StartMsg`, not a hard-coded mic-only start.

### 2.2 Renderer: the Meetings tab (rename from Live)

`MainTab 'live'` → `'meetings'`, label **Meetings**. New layout (Granola-inspired):

- **Left sidebar:** meetings list (title, date, status dot; New-note button) — same pattern as ChatSidebar.
- **Center: the notepad.** Editable title + **Plate** WYSIWYG markdown editor.
  - Why Plate: the only MIT-licensed option where WYSIWYG editing, markdown in/out (`@platejs/markdown`), and first-class Google-Docs-style suggestions (`@platejs/suggestion` marks with accept/reject) ship open-source. Tiptap's equivalent is paywalled; BlockNote's is GPL; Milkdown/Lexical would need custom suggestion machinery.
  - Notes persist as markdown (`notes_md`): deserialize on open, serialize + `PATCH` on debounced autosave (800 ms idle). Minimal hand-rolled toolbar; no shadcn dependency.
  - **Canonical content = accepted content only.** Pending AI suggestions live as suggestion-marked nodes/decorations that are ALWAYS excluded from the serialized markdown; autosave can never persist un-accepted suggestion text. Suggestion projection into the editor is idempotent (re-running it after reload produces the same state from `suggestions_json`).
  - **Stop flushes notes first:** stopping a session flushes the pending autosave PATCH and the final notepad control message before sending `stop`, so the final summary never sees stale notes.
- **Top: live summary card** (visible when a session is live or a summary exists): rolling `summary_md` with citation chips (reuse SummaryPanel internals), action items inline (absorbing ActionableRail), and a **"Add to notes"** button that inserts the current summary markdown at the end of the notepad as regular user-owned content.
- **Bottom: "Ask anything" chat bar** expanding into a drawer. Reuses Composer/AssistantTurn/ChatClient. Each meeting lazily creates one chat thread; `chat_send` carries a new optional `meeting_id`.
- **Right (hideable): transcript panel** — FeedView with You/Them labels, toggled by a waveform button; the grounding/evidence view (AmbientList + TriggerInspector) moves behind an "inspect" toggle inside this panel.
- Header SessionControls stays; starting from the header binds to (or creates) today's meeting and opens it.

### 2.3 User notes feed the live summary LLM

- Renderer sends `{"type":"notepad","meeting_id":…,"text_md":…}` on the `/session` socket, debounced 2 s after edits during a live session.
- `LiveSession` keeps the latest notepad text; `Summarizer._build_user_prompt` gains a `<user_notes>` block (capped `MEETING_NOTES_PROMPT_MAX_CHARS = 8000`, head+tail truncation) with instructions: user notes signal what the user considers important — weight related transcript content higher; do not restate the notes back.
- **Live summary model becomes a new `LIVE_SUMMARY_MODEL = "google/gemini-3.5-flash"`** (Neil: Gemini 3.5 Flash is the main summary and note-edit LLM), used ONLY by the live summarizer. `SUMMARY_MODEL` keeps its current value because `IDEA_SEARCH_MODEL`/`CONTEXT_ENRICH_MODEL` and chat utility calls alias it — those paths must not change model or cost. Gemini reasoning is pinned low/minimal for the latency-sensitive live lane via `LLM_REASONING_BY_MODEL`. Existing JSON + citation validation unchanged; `SUMMARY_FALLBACK_MODEL` stays deepseek. Verified on OpenRouter: streaming, tools, structured outputs supported.

### 2.4 End-of-call final summary + suggested edits (Gemini 3.5 Flash, graph access)

New module `core/taurine_core/live/final_summary.py`, triggered on `stop`:

- **Lifecycle (exact order):** stop control → `await asr.stop()` (flush finals) → settle/cancel the in-flight rolling summary → snapshot meeting state (transcript flush + freshest notes: latest of the WS notepad state and the meeting row) → launch a **runtime-owned, idempotent** finalization task keyed by meeting_id (double-finalize impossible; survives socket teardown). Event emission is best-effort to whatever `/session` socket is alive; persistence + a `meetings` update-kind push are the source of truth, so the UI recovers missed events by refetching.
- **Inputs:** full labeled transcript, freshest `notes_md` (+ its `notes_rev`), last live summary + confirmed/dismissed actionables, `user_profile_block`.
- **Tools:** the existing chat tool set (`search_evidence`, `read_note`, `expand_evidence` — [chat/tools.py](core/taurine_core/chat/tools.py)) via a new small public tool-agent helper (`llm_agent.run_tool_agent`: non-streaming tool loop that carries full message history, then a final strict-JSON turn with `response_format` over the same history — the existing `complete_json` only takes system+user and cannot carry tool results). Model `FINAL_SUMMARY_MODEL = "google/gemini-3.5-flash"`, bounded by `FINAL_SUMMARY_MAX_TOOL_CALLS = 6` and `FINAL_SUMMARY_TIMEOUT_S = 120`.
- **Output (strict JSON, validated fail-closed like the live summarizer):**
  - `title` — auto-title for untitled meetings;
  - `final_summary_md` — the polished summary;
  - `suggested_edits[]` — `{anchor, occurrence, insert_md, rationale}` where `anchor` is an exact substring of the user's notes (`""` = append at end) and `occurrence` disambiguates repeated anchors (validated server-side against the snapshotted notes; violators dropped). The result carries `base_rev` = the `notes_rev` it was computed against, so the client re-validates anchors if the user kept typing. Edits are additive insertions/expansions of the user's own notes (Granola's "enhanced notes" behavior), not rewrites.
- **Event flow over `/session`:** `final_summary_status` (`running`/`done`/`error`) → `final_summary` (payload). Status also lands on the meeting row (`summarizing` → `done`) so a reload can re-fetch; suggestions persist in `suggestions_json` until resolved.
- **Renderer:** `final_summary_md` renders as a distinct card (insertable like the live summary); each `suggested_edit` becomes a Plate suggestion mark at the anchor match (fallback: appended suggestion block when the anchor no longer matches), shown Google-Docs-style with rationale tooltip and accept/deny controls as they arrive. Accept applies the insertion as normal text; deny discards; both update `suggestions_json` via PATCH.

### 2.5 Taurine notes as an implicit source

- New source id **`taurine`** — implicit: it never appears in `/connectors` responses or the Connectors view, cannot be disabled, and has no sync toggle. **Mechanism:** it IS registered in the connector registry (a hidden, always-enabled connector whose `extract()` builds `NoteRecord`s from `done` meetings) so the existing sync/build pipeline includes it without special-casing — global sync only extracts registered connectors, so an unregistered source would silently never reach the graph. The `/connectors` REST listing and UI filter it out.
- Records: `NoteRecord(note_id=f"taurine:{meeting_id}", source="taurine", title (or "Meeting YYYY-MM-DD"), text = notes_md + "\n\n## Meeting summary\n" + final_summary_md, modified=meeting.updated_at, created=meeting.created_at, content_hash=<stable hash of title+text>)`. Deleted meetings drop their record on the next extract (replace semantics).
- **Sync trigger:** one coalesced global sync is queued ~60 s after a meeting finalizes (skipped if a sync is already running) — finishing a meeting is user intent. Post-hoc note edits do NOT auto-trigger syncs; they ride the next manual/scheduled sync.
- Priors: `SOURCE_RANK["taurine"] = 0.95` (first-party authored), `RECENCY_HALF_LIFE_DAYS["taurine"] = 60`.
- UI registries: `SOURCE_COLORS["taurine"] = "#ffd75e"` + `SOURCE_LABELS["taurine"] = "Taurine"` (graph.ts), `labels.ts` case, `.source-taurine` chip (main.css). Map legend/coloring then picks it up automatically.

### 2.6 Meeting-scoped chat

- `ChatSendMsg` gains optional `meeting_id`. When set, the meeting's context enters the turn as **citable meeting-local evidence `[M#]`** — the notepad (one item) plus recent transcript windows and the current live/final summary — registered in the turn's evidence registry so the chat citation validator accepts answers grounded in the meeting (a non-citable XML block would be stripped by validation, which requires every factual sentence to cite supplied evidence). Retrieval also seeds `taurine:{meeting_id}` once indexed. Total meeting context capped at `MEETING_CONTEXT_MAX_CHARS`.
- **Durable binding:** the meeting↔chat link persists server-side (`meetings.chat_id`, set when the meeting's first chat message creates a chat; the chat is titled `Meeting: {title}` via the existing rename path). `chat_regenerate` resolves the meeting through this link so regenerated turns keep their context. Non-meeting chats are byte-identical to today.

## 3. Call detection + auto-enable notification

**Detection (main process):** a tiny Swift helper `app/resources/native/micwatch` (source `micwatch.swift` in-repo, built by `npm run build:native` via `swiftc`; prebuilt binary committed so dev works without a toolchain step) subscribes to CoreAudio `kAudioDevicePropertyDeviceIsRunningSomewhere` on the default input device (re-subscribing on default-device changes) and prints `{"active":bool}` lines on state change. Electron main spawns it at launch, debounces blips (<1.5 s), and forwards edges to the renderer (`micwatch:active`).

**Suppression rules:** events are ignored while Taurine itself is listening (our own mic use); after a notification is dismissed or acted on, re-arm only after the mic has been idle ≥5 s (one prompt per call).

**Notification UI (renderer, global):** a bottom-right overlay card rendered above all tabs:
- Top edge: a countdown bar that shrinks over **10 s**.
- Body: "Call detected — start meeting notes?"
- Buttons: **Start notes** (primary) and **Dismiss**.
- At zero (or on Start notes): create a meeting, switch to the Meetings tab, start listening (mic + system audio when available) — identical to pressing the header button. Dismiss hides it until the re-arm condition.

## 4. Config additions (core)

```
LIVE_SUMMARY_MODEL = "google/gemini-3.5-flash"   # live summarizer only; SUMMARY_MODEL untouched
                                                 # (IDEA_SEARCH/CONTEXT_ENRICH/chat utilities alias it)
FINAL_SUMMARY_MODEL = "google/gemini-3.5-flash"
FINAL_SUMMARY_MAX_TOOL_CALLS = 6
FINAL_SUMMARY_TIMEOUT_S = 120
MEETING_NOTES_PROMPT_MAX_CHARS = 8000            # cap includes the truncation marker
MEETING_CONTEXT_MAX_CHARS = 12000                # meeting-scoped chat evidence
MEETING_FINALIZE_GRACE_S = 60                    # disconnect-while-live grace before finalizing
MEETING_SYNC_DELAY_S = 60                        # coalesced auto-sync after finalization
TAURINE_SOURCE_ID = "taurine"
SOURCE_RANK["taurine"] = 0.95
RECENCY_HALF_LIFE_DAYS["taurine"] = 60
LLM_REASONING_BY_MODEL["google/gemini-3.5-flash"] = low/minimal   # latency-sensitive live lane
```

## 5. Rollout phases (each lands green on main)

1. **Meetings foundation:** meetings table/REST/updates-kind; tab rename; sidebar + Plate notepad + autosave; session binds to meeting; transcript persisted. (No behavior change for live summaries.)
2. **System audio:** audiotee integration, IPC relay, channel-1 capture, You/Them labels through FeedView/ledger/summarizer prompts, Info.plist key. Spike first: dev-mode TCC prompt verification.
3. **Notes ↔ live summary:** notepad control message, `<user_notes>` block, Gemini live summary model swap, summary card + Add-to-notes.
4. **Final summary + suggestions + source:** final_summary module + events, suggestion marks UI, `taurine` source indexing + registries + auto-sync queue, meeting-scoped chat.
5. **Call detection:** micwatch helper, main-process watcher, notification card with countdown + auto-start.

## 6. Testing & validation

- **Core (pytest):** meetings store CRUD + status transitions; control-message parsing (`start.meeting_id`, `notepad`); summarizer prompt contains capped `<user_notes>`; final-summary JSON validation (bad anchors, oversized output, tool-loop cap) fail-closed; `taurine` source records upsert + exclusion from `/connectors`; meeting chat context block capped and absent without `meeting_id`; channel-1 chunks labeled in ledger rendering. Full suite stays green.
- **App:** `tsc --noEmit` + eslint + production build per phase; manual smoke via dev app (Electron) with sidecar logs — verified by orchestrator before each phase commit.
- **End-to-end (orchestrator-run):** scripted fake session over `/session` WS (synthetic PCM on channels 0/1 or injected transcript chunks) driving: live summary with notes block → stop → final summary with ≥1 suggestion → meeting note appears under source `taurine` after sync → map shows the source.
- **Battery/ablation regression:** 16-question battery after the Gemini summary swap (live abstention controls unaffected — retrieval untouched); chat regression is N/A except meeting-context injection, covered by unit tests.

## 7. Risks

- **Gemini via OpenRouter `require_parameters` + `response_format`:** verify `complete_json`/tool streaming against `google/gemini-3.5-flash` early (phase-3 spike); relax `require_parameters` for that model if needed.
- **TCC in dev mode** for the audiotee helper (see §1) — spike before committing to phase 2.
- **Echo duplicates** without headphones (§1 caveat, follow-up: ledger cross-channel dedup).
- **Suggestion anchoring** breaks if the user edits during summarization — mitigated by exact-substring match with append fallback.
- **Auto-sync after each meeting** costs a rebuild pass; acceptable (caches make it incremental), and it's queued not forced.
- **Plate integration surface** (Slate model, markdown round-trip lossiness on exotic structures) — notes are simple prose/bullets; acceptable.
- **Summary model swap** changes live-summary tone/costs for all sessions ($1.50/M in, $9/M out); fallback path unchanged if Gemini errors.

## 8. Rev 2 — pilot feedback rework (2026-07-16, implemented directly by orchestrator)

Neil's first-pilot feedback replaced several v1 decisions:
- **Editor: BlockNote (MPL-2.0), not Plate.** Plate lacked working markdown input shortcuts; BlockNote has them natively (`* `, `# `, code blocks) plus a Notion-grade feel. Suggestions are a custom `suggestion` block type rendered INLINE in the document at their anchors (hover accept ✓ / deny ✕, rationale tooltip), always excluded from serialization until accepted.
- **Final summary lands in the notepad** as one large pending suggestion appended below the user's notes (server appends it to `suggested_edits`; rides suggestions persistence). No separate final-summary card; the card is a slim live-only rolling summary strip.
- **Granola-faithful layout:** meetings home list ↔ note page (serif title, chip row, centered 680px column), floating pill bottom bar (waveform transcript toggle + stop, "Ask anything", "What did I miss"), transcript and chat as floating overlay sheets. Transcript = compact chat bubbles (mic right, system audio left), search, copy, Insights toggle; stays mounted so scroll position survives open/close (fixes forced bottom-scroll).
- **Call notification is an OS-level always-on-top panel window** (frameless, bottom-right of the screen, all Spaces, shown without focus steal) — visible when Taurine is hidden/minimized. Suppression state machine moved to the main process.
- **Models:** `LIVE_SUMMARY_MODEL = deepseek/deepseek-v4-flash` (gemini rolling summaries were garbled), `FINAL_SUMMARY_MODEL = z-ai/glm-5.2`.
- **Note links:** chat/summary/insights citation chips center the note on the map (never external/localhost); the notepad supports `note://<id>` deep links to the map.
