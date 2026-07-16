# Meetings Overhaul Implementation Plan (rev 2 — post codex review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Granola-style Meetings section: persistent meeting notepads (WYSIWYG markdown + suggestions), system-audio second Deepgram stream with You/Them labeling, user notes feeding the live summarizer, Gemini 3.5 Flash end-of-call summary with graph access + suggested edits, meetings indexed as an implicit `taurine` source, and a call-detection notification with countdown auto-start.

**Spec:** `docs/superpowers/specs/2026-07-16-meetings-overhaul-design.md` — read it first; authoritative on behavior. This rev 2 incorporates the 2026-07-16 codex review (16 findings): corrected store location, session lifecycle state machine, runtime-owned finalization, hidden-connector sync path, `llm_agent` tool-loop API, citable `[M#]` meeting evidence, suggestion-serialization safety, native packaging/cleanup, and an E2E harness task.

**Architecture:** Core (Python sidecar) gains meetings tables on the existing `Index` store, a per-start live-session lifecycle on `/session`, a runtime-owned finalization pipeline, and a hidden `taurine` connector. App (Electron 39 + React 19 + Tailwind 4) gains a Meetings tab (Plate editor) coordinated from App.tsx, a main-process system-audio tap (audiotee/CATap) feeding channel 1 of the existing PCM protocol, and a micwatch Swift helper driving a global notification.

**Tech Stack:** FastAPI sidecar (`core/taurine_core`), Electron 39/React 19, Plate (`platejs` + markdown + suggestion packages — MIT only, never `xl-*`/paid/GPL), `audiotee` npm, Deepgram nova-3, OpenRouter `google/gemini-3.5-flash`, vitest (new, renderer unit tests).

## Global Constraints

- Everything lands on `main` in /Users/neil/taurine; each phase leaves the repo green (core: `cd /Users/neil/taurine/core && .venv/bin/python -m pytest -q`; app: `cd /Users/neil/taurine/app && npx tsc --noEmit -p tsconfig.web.json && npm run lint && npx electron-vite build && npx vitest run`).
- Python: match existing style (type hints, pydantic models in `schemas.py`, config constants in `config.py`). TS: match existing idioms (plain useState/props, hand-rolled tolerant parsers in `sidecar.ts`, Tailwind utilities + `@layer components` classes). React runs under StrictMode — every effect/subscription must be cleanup-correct and double-invoke-safe.
- Live retrieval/abstention/calibration code is OFF-LIMITS except where a task names a file. Do not touch `retrieval/`, calibration artifacts, or chat routing defaults.
- Wire contracts (message/table/REST shapes) in this plan are exact; do not rename fields. Wire = snake_case; renderer types = camelCase mapped in lib parsers.
- Commits: conventional prefix, trailer `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.
- macOS-only features (audiotee, micwatch) must degrade gracefully: feature-detect, never crash on unsupported platforms.
- Chats persist via `Index` in `core/taurine_core/indexer/store.py` (NOT a top-level store.py). All meetings persistence lives there too, same connection + `CREATE TABLE IF NOT EXISTS` in the existing schema-init path.

---

## Phase 1 — Meetings foundation

### Task 1.1 (core): meetings store + REST + per-start session lifecycle

**Files:**
- Modify: `core/taurine_core/indexer/store.py` (meetings tables + CRUD on `Index`), `core/taurine_core/server.py` (routes near the chats block ~:670; `/session` handler :773-856; `_parse_control` :1376; UpdateBus KINDS :134), `core/taurine_core/schemas.py` (StartMsg ~:666, Meeting models), `core/taurine_core/live/ingest.py` (final-chunk hook), `core/taurine_core/config.py` (`MEETING_FINALIZE_GRACE_S = 60`)
- Test: `core/tests/test_meetings.py`

**Interfaces (produced — exact):**
- Tables (in `Index` schema init):
  ```sql
  CREATE TABLE IF NOT EXISTS meetings (
    meeting_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    started_at REAL, ended_at REAL,
    notes_md TEXT NOT NULL DEFAULT '', notes_rev INTEGER NOT NULL DEFAULT 0,
    final_summary_md TEXT NOT NULL DEFAULT '',
    suggestions_json TEXT NOT NULL DEFAULT '[]',
    transcript_json TEXT NOT NULL DEFAULT '[]',
    chat_id TEXT, status TEXT NOT NULL DEFAULT 'draft')
  ```
  `status ∈ {'draft','live','summarizing','done'}`; `meeting_id = uuid4().hex`.
- `Index` methods: `meeting_create(title='') -> dict`, `meeting_get(meeting_id) -> dict | None`, `meetings_list() -> list[dict]` (no notes/transcript/suggestions bodies; newest-first), `meeting_update(meeting_id, *, title=None, notes_md=None, suggestions=None) -> dict` (any notes_md change bumps `notes_rev` and `updated_at`; title change bumps `updated_at`), `meeting_delete(meeting_id)`, `meeting_append_transcript(meeting_id, chunks: list[dict])` (batched — chunk = `{"channel":int,"text":str,"ts":float,"seq":int}`; appends to the JSON array in one read-modify-write, single writer: the live session), `meeting_set_status(meeting_id, status, *, started_at=None, ended_at=None)`, `meeting_set_chat(meeting_id, chat_id)`, `meetings_done() -> list[dict]` (full rows for source extraction).
- REST (mirror the chats handlers' style): `GET /meetings` → `{"meetings":[{meeting_id,title,created_at,updated_at,started_at,ended_at,status}]}`; `POST /meetings` body `{"title"?}` → full meeting JSON (`suggestions` and `transcript` as parsed lists); `GET /meetings/{id}` (404 if missing); `PATCH /meetings/{id}` body any of `{"title","notes_md","suggestions"}`; `DELETE /meetings/{id}` → `{"ok":true}`. Full meeting JSON always includes `notes_rev`.
- `/updates`: new kind `"meetings"` in `UpdateBus.KINDS` + snapshot sender (`server.py:1692` area) sending `{"type":"meetings","meetings":[…list rows…]}`; published on create/delete/title/status change ONLY (never per transcript batch).
- Control: `StartMsg` gains `meeting_id: str | None = None`. Old messages parse unchanged.
- **Per-start lifecycle (inside the `/session` handler):** `start` with a meeting_id while a previous meeting on this socket is still live → finalize the previous one first (phase 1: just `meeting_set_status(...,'done', ended_at=now)`; phase 4 swaps in the pipeline). Each `start` creates a FRESH `LiveSession` + ASR client (tear down the old pair exactly like the socket-close path does) so ledger/summary/actionable state never leaks across meetings. `stop` → flush transcript buffer + finalize the bound meeting. **Disconnect while live:** do NOT finalize inline; schedule a grace timer (`MEETING_FINALIZE_GRACE_S`) on the runtime; a new socket's `start` for the same meeting_id cancels it (meeting stays `live`, transcript keeps appending); expiry finalizes. Timer + "already finalized" guard are runtime-owned and idempotent.
- **Transcript buffering:** `LiveSession` receives an `on_final_chunk(chunk_dict)` callback (injected via `make_live_session`); the server-side buffer flushes to `meeting_append_transcript` every 20 finals or 10 s (whichever first) and always on stop/disconnect.

**Steps:**
- [ ] Failing tests in `core/tests/test_meetings.py` (mirror existing FastAPI test-client fixtures from the server tests): store CRUD + `notes_rev` bump semantics; list excludes bodies; batched transcript append preserves order; status transitions set timestamps; REST 404/PATCH-partial/DELETE; `_parse_control` accepts start with and without `meeting_id`; fresh-LiveSession-per-start (two starts on one socket → second meeting's ledger does not contain first meeting's chunks — assert via the session objects the factory produced); disconnect-grace (simulate: bind meeting, drop socket, assert status still `live`, fire the timer callback, assert `done`); double-finalize is a no-op.
- [ ] Run: `cd /Users/neil/taurine/core && .venv/bin/python -m pytest tests/test_meetings.py -q` → FAIL.
- [ ] Implement. Keep the grace timer on the runtime object (e.g. `runtime.meeting_finalizers: dict[str, asyncio.TimerHandle|Task]`).
- [ ] Full suite green.
- [ ] Commit: `feat(core): meetings store, REST, per-start live-session lifecycle`.

### Task 1.2 (app): Meetings tab shell, App-level coordinator, Plate notepad

**Files:**
- Create: `app/src/renderer/src/lib/meetings.ts`, `app/src/renderer/src/lib/meetings.test.ts`, `app/src/renderer/src/views/meetings/MeetingsView.tsx`, `MeetingSidebar.tsx`, `NotepadEditor.tsx`, `TranscriptPanel.tsx`, `SummaryCard.tsx` (same dir), `app/vitest.config.ts`
- Modify: `app/src/renderer/src/App.tsx` (tab rename :32-40; live grid :362-391 → `<MeetingsView/>`; coordinator state), `app/src/renderer/src/lib/sidecar.ts` (StartMsg meeting_id + **reconnect replays the full last StartMsg** — today :485 replays a hard-coded mic-only start; store the last start payload and resend it verbatim), `app/src/renderer/src/lib/updates.ts` (`meetings` kind), `app/package.json` (Plate deps + vitest)

**Interfaces:**
- Consumes: Task 1.1 REST/updates/StartMsg.
- **App.tsx is the coordinator and owns:** `selectedMeetingId`, `liveMeetingId`, `meetingsIndex` (list rows from the `meetings` updates kind + refetch), the existing clients, and the header start/stop. It passes ONE props object down:
  ```ts
  interface MeetingsCoordinator {
    meetings: MeetingListRow[]
    selectedMeetingId: string | null
    liveMeetingId: string | null
    listeningState: ListeningState
    micError: string | null
    live: { chunks: TranscriptEvent[]; interims: Record<number, TranscriptEvent>;
            summary: SummaryEvent | null; actionables: Actionable[];
            liveGrounding: LiveGroundingEvent | null }
    selectMeeting(id: string | null): void
    createMeeting(): Promise<string>            // returns new id, selects it
    deleteMeeting(id: string): Promise<void>
    renameMeeting(id: string, title: string): Promise<void>
    startMeeting(id?: string): Promise<void>    // creates+selects if id omitted; binds session
    stopMeeting(): Promise<void>
    saveNotes(id: string, notesMd: string): Promise<void>   // PATCH; App tracks in-flight for flush
    setActionableStatus(id: string, status: 'confirmed'|'dismissed'): void
  }
  ```
  Header SessionControls calls `startMeeting()`/`stopMeeting()`. Live event state (`chunks` etc.) is **cleared on every `startMeeting`** and belongs to `liveMeetingId` only.
- `lib/meetings.ts`: `interface MeetingListRow {meetingId,title,createdAt,updatedAt,startedAt,endedAt,status}`, `interface Meeting extends MeetingListRow {notesMd; notesRev: number; finalSummaryMd; suggestions: SuggestedEdit[]; transcript: TranscriptChunk[]; chatId: string | null}`, `interface SuggestedEdit {id; anchor; occurrence: number; insertMd; rationale; status: 'pending'|'accepted'|'rejected'}`, `interface TranscriptChunk {channel; text; ts; seq}`; functions `listMeetings/createMeeting/getMeeting/patchMeeting/deleteMeeting` with tolerant snake→camel parsing (style of `graph.ts`).
- `NotepadEditor` props: `{ valueMd: string; onChangeMd(md: string): void; suggestions: SuggestedEdit[]; onResolveSuggestion(id: string, accept: boolean): void; readOnly?: boolean }`. Plate internally; markdown at the boundary. **Serialization invariant: output markdown NEVER contains pending-suggestion content** (suggestion projection arrives in Task 4.3; the invariant + a unit test for "no suggestion leakage with empty suggestions" is established now).
- MeetingsView renders: sidebar (list rows, status dot, New note), center title input + editor (autosave: debounce 800 ms → `saveNotes`; flush on meeting switch/unmount via effect cleanup), top SummaryCard (live summary + actionables; hidden when empty and idle), right hideable TranscriptPanel (**live meeting:** `live.chunks`/`interims` via FeedView; **non-live selection:** `meeting.transcript` rendered statically with the same You/Them labels; AmbientList + TriggerInspector behind an "inspect" toggle), bottom disabled composer placeholder "Ask about this meeting" (enabled in Task 4.4).

**Steps:**
- [ ] Add vitest (`vitest` + config; `npx vitest run` green on a trivial test) — renderer-lib tests only, no DOM.
- [ ] Install Plate: resolve current MIT package set for markdown WYSIWYG + suggestions from https://platejs.org/docs; pin versions.
- [ ] Failing vitest: meetings.ts parses a full meeting payload (snake→camel, malformed suggestions tolerated → []), and NotepadEditor's serialize helper round-trips heading/bold/list markdown (export the md<->value helpers from NotepadEditor's module so they're unit-testable).
- [ ] Implement lib + components + App coordinator + sidecar.ts reconnect fix.
- [ ] Verify: tsc, lint, `npx electron-vite build`, `npx vitest run`; dev smoke — create/rename/edit/delete meetings; autosave persists across app restart; start binds meeting (sidecar log shows meeting_id), transcript lands in `transcript_json`, reselecting a done meeting shows its stored transcript.
- [ ] Commit: `feat(app): Meetings tab with Plate notepad, App-level meeting coordinator`.

**Phase 1 gate (orchestrator):** core suite green; app checks green; manual smoke of the full create→listen→stop→reload cycle.

---

## Phase 2 — System audio channel

### Task 2.1 (app main/preload): audiotee tap + IPC relay

**Files:**
- Create: `app/src/main/sysaudio.ts`
- Modify: `app/src/main/index.ts`, `app/src/preload/index.ts`, `app/src/preload/index.d.ts`, `app/electron-builder.yml` (extendInfo + asarUnpack), `app/package.json` (dep `audiotee`)

**Interfaces (produced):**
- Preload `window.api.sysAudio`:
  ```ts
  { available(): Promise<boolean>;          // darwin && macOS >= 14.2 && helper present
    start(): Promise<void>; stop(): Promise<void>;
    onChunk(cb: (chunk: ArrayBuffer) => void): () => void;   // returns unsubscribe
    onError(cb: (message: string) => void): () => void }
  ```
  Preload NORMALIZES ipc payloads: copy the incoming Buffer/typed array into a fresh `ArrayBuffer` before invoking callbacks. All `on*` return unsubscribe fns (StrictMode-safe).
- Main `sysaudio.ts`: lazy AudioTee `{ sampleRate: 48000 }` (mono s16le); `data` → `webContents.send('sysaudio:chunk', chunk.data)`; error/exit → `sysaudio:error` + auto-stop; `ipcMain.handle('sysaudio:available'|'sysaudio:start'|'sysaudio:stop')`; stop the tap when the owning webContents is destroyed or reloaded (`'destroyed'`/`'did-start-navigation'`), and on `app.will-quit`. Start/stop idempotent.
- `electron-builder.yml`: mac.extendInfo gains `NSAudioCaptureUsageDescription: Taurine records system audio to transcribe the other side of your meetings.`; `asarUnpack` gains `node_modules/audiotee/**` (the helper binary must exist unpacked at runtime).

**Steps:**
- [ ] `npm install audiotee` (check its README for the exact chunk event shape and supported rates — it captures at 48 kHz mono 16-bit; if it supports direct 16 kHz output, still emit 48 kHz here — the renderer owns decimation and the contract stays fixed).
- [ ] Implement main + preload + builder config.
- [ ] Verify: tsc/lint/build; dev smoke on this Mac — `await window.api.sysAudio.available()` true; `start()` triggers the one-time "System Audio Recording Only" TCC prompt; while playing audio, chunks arrive with NON-ZERO energy (log RMS of a few chunks in devtools — silent zero-filled chunks = TCC misattribution = **BLOCKER: stop and report**, do not proceed to 2.2).
- [ ] Commit: `feat(app): system-audio tap via CoreAudio process tap (audiotee)`.

### Task 2.2 (app renderer): channel-1 capture + You/Them labels

**Files:**
- Create: `app/src/renderer/src/lib/sysaudio.ts` (`SystemAudioCapture`), `app/src/renderer/src/lib/sysaudio.test.ts`
- Modify: `app/src/renderer/src/lib/audio.ts` (export `prefixChannel`), `app/src/renderer/src/App.tsx` (start/stop lifecycle), `app/src/renderer/src/views/FeedView.tsx` (`channelLabel` :322-324)

**Interfaces:**
- `class SystemAudioCapture { constructor(onFrame: (frame: ArrayBuffer) => void, onError: (msg: string) => void); async start(): Promise<void>; stop(): void }` — subscribes `window.api.sysAudio.onChunk/onError`, decimates 48 kHz→16 kHz by averaging each 3 s16 samples (carry remainder samples across chunks), packs 4000-sample (250 ms) frames, prefixes channel byte `1`. `stop()` unsubscribes everything and calls `sysAudio.stop()`.
- `startMeeting`: if `await window.api.sysAudio.available()` → start SystemAudioCapture and declare `channels: [{id:0,label:'mic'},{id:1,label:'system'}]`; else mic-only exactly as today. `sysaudio` errors surface on the existing micError strip (distinct copy: "System audio stopped — mic still live") and stop only the system capture; mic keeps running. `stopMeeting` and every error path stop both captures.
- FeedView: `channelLabel` → `0:'You'`, `1:'Them'`, else `Ch ${n}`.

**Steps:**
- [ ] Failing vitest for the decimator: 48 kHz sine input chunks of awkward lengths (e.g. 1000, 1001, 999 samples) → output frame boundaries exact, 3:1 ratio, remainder carried, channel byte = 1.
- [ ] Implement; wire App lifecycle.
- [ ] Verify: tsc/lint/build/vitest; dev smoke — play a YouTube video during a session: channel-1 lines labeled **Them**, mic lines **You**; kill system audio mid-session → mic unaffected.
- [ ] Commit: `feat(app): stream system audio as channel 1 with You/Them labels`.

### Task 2.3 (core): speaker labels in ledger + summarizer awareness

**Files:**
- Modify: `core/taurine_core/live/ledger.py` (rendering ~:169-174), `core/taurine_core/live/summarizer.py` (SYSTEM_PROMPT :38-59)
- Test: extend the existing ledger/summarizer test files (locate via `grep -rl "TranscriptLedger" core/tests`)

**Interfaces:** transcript evidence rendering becomes `[T#] You: "text"` (channel 0) / `[T#] Them: "text"` (channel 1) / `[T#] Ch{n}: "text"` (other). **`[T#]` tokens are load-bearing** — the summarizer's citation validator parses them (`summarizer.py:351`); only the `[channel N]` segment changes. SYSTEM_PROMPT gains: `Transcript lines are labeled You (the user's own mic) and Them (other call participants); attribute statements accordingly.`

**Steps:**
- [ ] Failing tests: rendering for channels 0/1/2; existing `[T#]` citation validation still passes on a summarizer round-trip fixture.
- [ ] Implement; full suite green.
- [ ] Commit: `feat(core): speaker-side labels in live transcript evidence`.

**Phase 2 gate (orchestrator):** dev-app smoke with real system audio (both channels transcribing, labels correct); core suite green; packaged-build check deferred to Neil's next packaging cycle (documented).

---

## Phase 3 — Notes ↔ live summary (Gemini)

### Task 3.1 (core): notepad block + LIVE_SUMMARY_MODEL

**Files:**
- Modify: `core/taurine_core/schemas.py` (NotepadMsg), `core/taurine_core/server.py` (`_parse_control`/`_handle_control`), `core/taurine_core/live/ingest.py` (`set_notepad`, threading into summarize), `core/taurine_core/live/summarizer.py` (`summarize(..., notepad_md: str = "")` → `_build_user_prompt(..., notepad_md)`), `core/taurine_core/config.py`
- Test: summarizer prompt tests + control parse tests

**Interfaces:**
- Control: `{"type":"notepad","meeting_id": str,"text_md": str}` → `LiveSession.set_notepad(text_md)` (ignored with a debug log if meeting_id ≠ the bound meeting; no reply event).
- Config: `LIVE_SUMMARY_MODEL = "google/gemini-3.5-flash"` — **used ONLY at the summarizer's model-selection site** (`summarizer.py:113-130`). `SUMMARY_MODEL` stays `deepseek/deepseek-v4-flash` (`IDEA_SEARCH_MODEL`/`CONTEXT_ENRICH_MODEL`/chat utilities alias it — verify with `grep -rn "SUMMARY_MODEL" core/taurine_core` that only the live summarizer call sites change). `SUMMARY_FALLBACK_MODEL` unchanged. `LLM_REASONING_BY_MODEL["google/gemini-3.5-flash"] = {"effort": "low"}` (or the minimal supported variant — check `_reasoning_for` in llm.py and OpenRouter's reasoning param for Gemini; live latency matters). `MEETING_NOTES_PROMPT_MAX_CHARS = 8000`.
- Prompt: `_build_user_prompt` emits, before `<transcript_evidence>`, a `<user_notes>` block: instruction line per spec §2.3 + the notepad text. Truncation when over cap: `head + "\n…[notes truncated]…\n" + tail` where head=5,800 chars, tail=2,000 chars, and **total including the marker ≤ 8,000**. Block omitted entirely when the notepad is empty/whitespace.

**Steps:**
- [ ] Failing tests: control parse (valid; missing text_md tolerated as no-op); prompt contains `<user_notes>` when set, absent when empty; truncation total-length assertion; summarizer JSON/citation validation fixtures unchanged; grep-test that no non-live consumer of `SUMMARY_MODEL` changed (assert `config.SUMMARY_MODEL == "deepseek/deepseek-v4-flash"`).
- [ ] Implement.
- [ ] Gemini smoke (needs `OPEN_ROUTER_API_KEY` from core/.env — if unavailable in sandbox, STOP and report; the orchestrator runs it): one `complete_json(model="google/gemini-3.5-flash", …)` call and one `stream_chat` with a dummy tool. If `require_parameters` + `response_format` fails for this model, add a per-model relaxation in `llm.py` (narrow, commented) rather than dropping validation.
- [ ] Full suite green. Commit: `feat(core): user-notes block in live summary; Gemini 3.5 Flash live summarizer`.

### Task 3.2 (app): notepad push + flush-before-stop + Add-to-notes

**Files:**
- Modify: `app/src/renderer/src/lib/sidecar.ts` (`sendNotepad(meetingId, textMd)` control), `app/src/renderer/src/App.tsx` (coordinator), `app/src/renderer/src/views/meetings/MeetingsView.tsx`, `SummaryCard.tsx`

**Interfaces:**
- `SidecarClient.sendNotepad(meetingId: string, textMd: string)` sends the notepad control message (no-op when socket down).
- Coordinator behavior: while `listeningState==='listening'` and the edited meeting is `liveMeetingId`, notepad edits ALSO send `sendNotepad` debounced 2 s (independent of the 800 ms REST autosave). **`stopMeeting()` order:** flush pending REST autosave (`await saveNotes`) → `sendNotepad` with the final text → send `stop`. WS ordering on one socket guarantees the sidecar sees the final notepad before the stop control.
- SummaryCard `onAddToNotes(summaryMd)`: MeetingsView appends `\n\n${summaryMd}\n` through the normal editor-change path (so autosave + notepad push both fire).

**Steps:**
- [ ] Implement; tsc/lint/build/vitest.
- [ ] Dev smoke: jot notes mid-session → sidecar session log shows `<user_notes>` in the next summary prompt; Add-to-notes appends + persists; type-then-immediately-stop → final text present on the meeting row.
- [ ] Commit: `feat(app): notepad feeds live summary; insert summary into notes`.

**Phase 3 gate (orchestrator):** live E2E on this machine; orchestrator re-runs its retrieval battery (live abstention controls) — retrieval untouched but verify; summary latency spot-check with Gemini (target: comparable to deepseek path).

---

## Phase 4 — Final summary, suggestions, taurine source, meeting chat

### Task 4.1 (core): tool-agent helper + final-summary pipeline

**Files:**
- Create: `core/taurine_core/llm_agent.py`, `core/taurine_core/live/final_summary.py`
- Modify: `core/taurine_core/llm.py` (`complete_json_messages`), `core/taurine_core/server.py` (stop path + runtime-owned task registry + grace-timer finalization calls into the same pipeline), `core/taurine_core/schemas.py` (events), `core/taurine_core/config.py` (`FINAL_SUMMARY_MODEL="google/gemini-3.5-flash"`, `FINAL_SUMMARY_MAX_TOOL_CALLS=6`, `FINAL_SUMMARY_TIMEOUT_S=120`)
- Test: `core/tests/test_llm_agent.py`, `core/tests/test_final_summary.py` (fake LLM — mirror existing summarizer fakes)

**Interfaces:**
- `OpenRouterClient.complete_json_messages(model, messages: list[dict], *, timeout_s, temperature=0.2, max_tokens=None) -> dict` — same as `complete_json` but takes a full message list (system/user/assistant/tool roles preserved).
- `llm_agent.run_tool_agent(llm, *, model, system: str, user: str, tools: list[dict], execute_tool: Callable[[str, dict], Awaitable[str]], max_tool_calls: int, timeout_s: float) -> dict`:
  non-streaming loop — `complete_tools`-style call (with per-call timeout derived from the remaining budget, NOT the global 15 s default) → append assistant tool_calls + tool results to messages → repeat until no tool call or budget exhausted → final `complete_json_messages` turn appending `{"role":"user","content":"Now return the final JSON result exactly as instructed."}`. Raises `TimeoutError` past `timeout_s`.
- Tool executor for final summary: build the SAME executor the chat engine uses. Extract chat's private tool-execution into a small public seam — e.g. a `ChatEngine.tool_executor()` method or a module-level `make_tool_executor(retriever, index)` in `chat/engine.py` — reusing the existing implementations of `search_evidence`/`read_note`/`expand_evidence` with a `mode="tool"` retriever from `knowledge_runtime.make_engine`. Do NOT reimplement the tools.
- `final_summary.run_final_summary(meeting: dict, llm, execute_tool, profile_block: str, emit: Callable[[Event], Awaitable[None]], index) -> None` — assembles inputs (labeled transcript rendered `You:`/`Them:` lines, notes_md + notes_rev, last live summary text, actionable statuses, profile block; prompt per spec §2.4 incl. the You/Them convention and additive-edits rule), runs `run_tool_agent`, validates, persists, emits.
- Validation (fail-closed): ≤10 edits kept; each non-empty `anchor` must occur in the snapshotted notes and `occurrence` (1-based) must exist — violators dropped, rest kept; `insert_md` non-empty; server assigns `id = uuid4().hex[:8]`, `status="pending"`. Garbage/timeout ⇒ emit `error` status, meeting still → `done`, nothing persisted except status.
- Events: `{"type":"final_summary_status","meeting_id","state":"running"|"done"|"error","detail":""}` and `{"type":"final_summary","meeting_id","title","final_summary_md","base_rev":int,"suggested_edits":[{"id","anchor","occurrence","insert_md","rationale","status":"pending"}]}`.
- **Stop path (exact order in `_handle_control`/socket-close):** `await asr.stop()` → settle in-flight rolling summary (existing `aclose` machinery) → flush transcript buffer → snapshot meeting (freshest notes = WS notepad state if newer than row) → mark `summarizing`, publish `meetings` kind → spawn ONE runtime-tracked task (`runtime.finalization_tasks[meeting_id]`, idempotent guard) → task persists results (`final_summary_md`, `suggestions_json` incl. `base_rev`, auto-title only if title empty) → `done` + `meetings` publish + best-effort events (emit wrapper no-ops once the socket is gone). Grace-timer expiry runs the same pipeline (emit = no-op).

**Steps:**
- [ ] Failing tests — llm_agent: tool loop executes N calls then returns final JSON (fake llm); budget exhaustion; timeout. final_summary: happy path persists + emits + auto-titles untitled meeting; titled meeting keeps title; anchor/occurrence violations dropped; >10 edits truncated; garbage → error event + status done; duplicate finalization no-op; disconnect-before-done → results persisted, no emit crash.
- [ ] Implement; full suite green.
- [ ] Commit: `feat(core): end-of-call final summary agent with KB tool access`.

### Task 4.2 (core): hidden `taurine` connector + coalesced sync

**Files:**
- Create: `core/taurine_core/connectors/taurine_meetings.py`
- Modify: `core/taurine_core/connectors/catalog.py` (register), `core/taurine_core/server.py` (`/connectors` listing excludes it :527; enable/disable/delete endpoints 404 it; coalesced post-finalize sync), `core/taurine_core/config.py` (`TAURINE_SOURCE_ID="taurine"`, `SOURCE_RANK["taurine"]=0.95`, `RECENCY_HALF_LIFE_DAYS["taurine"]=60`, `MEETING_SYNC_DELAY_S=60`)
- Test: `core/tests/test_taurine_source.py`

**Interfaces:**
- `TaurineMeetingsConnector(ConnectorPort)`: `connector_id="taurine"`, `kind="local"`, plus a class attr `hidden = True`. `extract(config)` reads `index.meetings_done()` and yields one `NoteRecord` per done meeting with non-empty content: `note_id=f"taurine:{meeting_id}"`, `source="taurine"`, `title = meeting title or f"Meeting {date(created_at):%Y-%m-%d}"`, `text = notes_md + ("\n\n## Meeting summary\n" + final_summary_md if final_summary_md else "")`, `modified=updated_at`, `created=created_at`, `content_hash = <the same stable hash helper other connectors use — find it in connectors/base.py>`. Deleted meetings disappear via replace semantics on the next extract.
- Registry: registered like any connector but **always enabled** (seed the enabled ledger on registration or special-case `hidden` in `sync_all_enabled_collecting` — pick whichever is smaller against `registry.py:198-218`), so every global sync includes taurine records in the build extraction. `/connectors` REST filters `hidden` connectors; enable/disable/config/delete-data endpoints return 404 for them.
- Post-finalize sync: after a meeting reaches `done` with a final summary, schedule the existing global-sync entrypoint (`_run_global_sync` path used by `POST /index/sync`) after `MEETING_SYNC_DELAY_S`, coalesced (a pending timer is not duplicated; skipped when a sync is already running). PATCH edits to done meetings do NOT schedule syncs.

**Steps:**
- [ ] Failing tests: extract shape/id/hash/timestamps; empty meetings skipped; `/connectors` response never contains `taurine`; enable endpoint 404s; global sync's collected extraction includes taurine records (fake two done meetings); finalize schedules exactly one coalesced sync.
- [ ] Implement; full suite green.
- [ ] Commit: `feat(core): index meetings as implicit taurine source`.

### Task 4.3 (app): suggestions UX + final card + source branding

**Files:**
- Modify: `app/src/renderer/src/views/meetings/NotepadEditor.tsx`, `MeetingsView.tsx`, `SummaryCard.tsx`, `app/src/renderer/src/lib/sidecar.ts` (parse both events), `app/src/renderer/src/lib/suggestions.ts` (new — pure anchor/apply logic), `app/src/renderer/src/lib/suggestions.test.ts`, `app/src/renderer/src/lib/graph.ts` (`SOURCE_COLORS["taurine"]="#ffd75e"`, `SOURCE_LABELS["taurine"]="Taurine"`), `app/src/renderer/src/lib/labels.ts` (`taurine`→`'Taurine'`), `app/src/renderer/src/assets/main.css` (`.source-taurine` chip)

**Interfaces:**
- `lib/suggestions.ts` (pure, unit-tested): `locateAnchor(notesMd, anchor, occurrence) -> number | null` (char index after the occurrence; `""` anchor → end) and `applySuggestion(notesMd, edit) -> string` (insert `insertMd` at the located point with sensible newline handling; null location → append `\n\n${insertMd}`).
- Event handling (App coordinator): `final_summary_status` → progress state on SummaryCard ("Summarizing…" shimmer / error note). `final_summary` → refetch meeting (authoritative), show final card. If `base_rev < meeting.notesRev`, re-validate every pending anchor client-side via `locateAnchor`; misses degrade to append-mode suggestions.
- Editor: pending suggestions project as suggestion-marked insertions at their anchor points (Plate suggestion plugin), each with inline accept (accent) / deny (subtle) controls + rationale on hover. **Autosave serialization filters all pending-suggestion content** (serialize from a value with suggestion nodes stripped — add a vitest asserting md-out excludes pending text). Projection is idempotent: re-projecting the same `suggestions_json` after reload yields the same editor state. Accept → `applySuggestion` result becomes canonical content (normal change path), edit status `accepted`; deny → status `rejected`; both `patchMeeting(id, {suggestions})`.
- If the Plate suggestion plugin fights the filtered-serialization invariant, fall back to widget-decoration rendering (suggestion cards anchored inline but NOT part of the document tree) — the invariant wins over the mechanism.

**Steps:**
- [ ] Failing vitest for `suggestions.ts`: occurrence disambiguation (3× same anchor), empty anchor append, missing anchor → null, apply idempotence, unicode safety.
- [ ] Implement events + projection + branding entries.
- [ ] Verify: tsc/lint/build/vitest; dev smoke with the Task 4.5 harness — suggestions appear as they arrive, accept inserts + persists, deny removes, pending suggestions survive app reload and never leak into saved markdown (inspect `notes_md` via `GET /meetings/{id}` while suggestions pending).
- [ ] Commit: `feat(app): AI suggested edits with accept/deny; taurine source branding`.

### Task 4.4 (core+app): meeting-scoped chat with citable [M#] evidence

**Files:**
- Modify: `core/taurine_core/schemas.py` (`ChatSendMsg.meeting_id: str | None = None`), `core/taurine_core/chat/engine.py` (meeting evidence registration + `_build_messages`), `core/taurine_core/server.py` (chat path gets meeting lookup; regenerate resolves via `meetings.chat_id`; first meeting send links + titles the chat), `core/taurine_core/config.py` (`MEETING_CONTEXT_MAX_CHARS=12000`); app: `app/src/renderer/src/lib/chat.ts` (send `meeting_id`), `app/src/renderer/src/views/meetings/MeetingChat.tsx` (new), `MeetingsView.tsx`
- Test: `core/tests/test_chat_meeting_context.py`

**Interfaces:**
- **Citable meeting evidence:** when `meeting_id` resolves, the engine registers meeting-local items in the turn's evidence registry as `[M#]` refs: `[M1]` = notepad (as one item), `[M2..]` = transcript windows (chunk consecutive same-channel lines into ~800-char windows, newest ~60 lines) and the live/final summary, all capped to `MEETING_CONTEXT_MAX_CHARS` total (drop oldest transcript windows first). They render in the evidence block like other evidence (`[M3] (2026-07-16, meeting) You: …`), the system prompt names them as citable meeting context, and the citation validator accepts registered `M` ids exactly like other registered kinds (follow how `[T#]` transcript refs are registered/validated in live — reuse that kind or add an `M` kind to the registry + validator + renderer chips). Renderer `MessageMarkdown.tsx` citation regex already matches `[T\d+]`-style patterns — extend the regex + chip kind for `M`.
- Live summary access: while live, the engine reads the bound `LiveSession`'s current summary via `runtime.active_live_session`; after finalize, uses `final_summary_md` from the row.
- **Durable link:** first `chat_send` with `meeting_id` and empty `chat_id` → create chat, then set its title `Meeting: {title or date}` through the existing rename path, then `meeting_set_chat(meeting_id, chat_id)`. `chat_regenerate` (no meeting_id on the wire) resolves the meeting by reverse lookup (`Index` method `meeting_for_chat(chat_id)`) so regenerated turns keep context. Without `meeting_id` and no linked chat: byte-identical behavior to today (test asserts message list equality).
- Retrieval seeds `taurine:{meeting_id}` when that note exists in the index.
- App: `MeetingChat` bottom drawer (collapsed bar "Ask about this meeting…" → expands; Esc collapses) reusing Composer/AssistantTurn/ChatClient; chat id restored from `meeting.chatId` on load.

**Steps:**
- [ ] Failing tests: `[M#]` items registered + capped + oldest-dropped; citations to `M` ids survive validation; no meeting_id → byte-identical messages; regenerate re-injects context via reverse lookup; first send links chat + titles it; unknown meeting_id ignored gracefully.
- [ ] Implement core, then app drawer.
- [ ] Full core suite + app checks green. Commit: `feat: meeting-scoped chat with citable meeting evidence`.

### Task 4.5 (core): deterministic meeting E2E harness

**Files:**
- Create: `core/scripts/e2e_meeting_flow.py`
- Modify: `core/taurine_core/server.py` (`_parse_control`/`_handle_control`: dev-only `inject_transcript`), `core/taurine_core/config.py` (`TEST_CONTROLS_ENABLED = os.environ.get("TAURINE_TEST_CONTROLS") == "1"`)
- Test: `core/tests/test_inject_transcript.py`

**Interfaces:**
- Control (only when `TEST_CONTROLS_ENABLED`, otherwise rejected+logged): `{"type":"inject_transcript","chunks":[{"channel":0,"text":"…"},…]}` — each chunk runs through the exact `on_chunk` path as ASR finals (`final=True`, ts=now, seq assigned normally).
- `e2e_meeting_flow.py` (run against a live sidecar: `TAURINE_TEST_CONTROLS=1 .venv/bin/python scripts/e2e_meeting_flow.py`): POST /meetings → WS /session start(meeting, channels 0+1) → inject a scripted 2-speaker transcript → send notepad → stop → await `final_summary` event (timeout 180 s) → assert: meeting row `done`, `final_summary_md` non-empty, ≥0 suggestions with valid anchors, transcript persisted; print a PASS/FAIL summary. Exit non-zero on failure.
- Purpose: phase-4 gate + regression tool; uses the real Gemini path (cheap) — NOT part of pytest.

**Steps:**
- [ ] Failing test: inject control rejected when flag off; accepted + flows to ledger when on.
- [ ] Implement control + script; run the script end-to-end against a local sidecar.
- [ ] Commit: `feat(core): deterministic meeting E2E harness`.

**Phase 4 gate (orchestrator):** `e2e_meeting_flow.py` PASS; note appears under source `taurine` after the coalesced sync (map legend shows Taurine); orchestrator retrieval battery re-run; suggestions accept/deny + reload smoke in the dev app.

---

## Phase 5 — Call detection + auto-start notification

### Task 5.1 (app main): micwatch helper + watcher

**Files:**
- Create: `app/resources/native/micwatch.swift`, `app/scripts/build-native.mjs`, `app/src/main/micwatch.ts`
- Modify: `app/src/main/index.ts` (spawn on ready), `app/package.json` (`build:native` script; `postinstall` builds if binary missing AND swiftc present), `app/src/preload/index.ts` + `index.d.ts`

**Interfaces:**
- Swift helper: subscribes `kAudioDevicePropertyDeviceIsRunningSomewhere` on the default input device AND `kAudioHardwarePropertyDefaultInputDevice` (re-subscribe on device change); prints one JSON line `{"active":true|false}` per state edge plus the initial state; exits on SIGTERM. No deps, ~60-90 lines.
- Build: `swiftc -O micwatch.swift -o micwatch` — build a **universal binary** (`-target arm64-apple-macos14 ` + `-target x86_64-apple-macos14` slices, `lipo -create`). Commit BOTH source and the universal binary at `app/resources/native/micwatch` (dev works with no toolchain; `resources/**` is already asarUnpack'd so packaging works).
- `micwatch.ts`: spawn (path branches on `app.isPackaged`), parse lines tolerantly, debounce activations <1.5 s, forward edges via `webContents.send('micwatch:active', bool)`; respawn with backoff (max 3) on crash; disabled silently on non-darwin/missing binary; kill on quit.
- Preload: `window.api.micWatch: { onActive(cb: (active: boolean) => void): () => void }` (unsubscribe fn, StrictMode-safe).

**Steps:**
- [ ] Write helper + build script; `npm run build:native`; standalone test: run binary, start a Voice Memos recording → `{"active":true}`; stop → `{"active":false}`.
- [ ] Wire main/preload; tsc/lint/build.
- [ ] Commit: `feat(app): mic-activity watcher (CoreAudio) for call detection`.

### Task 5.2 (app renderer): notification card + auto-start

**Files:**
- Create: `app/src/renderer/src/components/CallNotification.tsx`
- Modify: `app/src/renderer/src/App.tsx`

**Interfaces:**
- `CallNotification` props `{ seconds: number; onStart(): void; onDismiss(): void }` — fixed bottom-right (z above all view layers), top edge = full-width 3 px countdown bar animating `scaleX 1→0` linearly over `seconds` (CSS transition started on mount; do NOT re-trigger on re-render), body "Call detected — start meeting notes?", buttons **Start notes** (primary) / **Dismiss** (secondary). One `setTimeout(seconds)` fires `onStart`; cleanup on unmount (StrictMode-safe: timer set in effect with cleanup).
- App wiring: show when `micwatch:active === true` AND `listeningState === 'idle'` AND not suppressed. `onStart`/timeout → `startMeeting()` (the coordinator creates/binds a meeting, switches `activeTab='meetings'`, starts mic + system audio) → hide. `onDismiss` → hide + suppress until `micwatch:active === false` has held continuously ≥5 s (re-arm). While Taurine is listening, micwatch events are ignored. Countdown default 10 s.

**Steps:**
- [ ] Implement card + wiring.
- [ ] Verify: tsc/lint/build/vitest; dev smoke — Voice Memos recording → card appears; let it expire once (auto-start fires, Meetings tab opens listening); trigger again and Dismiss (no re-prompt until idle 5 s).
- [ ] Commit: `feat(app): call-detected notification with countdown auto-start`.

**Phase 5 gate (orchestrator):** full scenario — external app grabs mic → notification → auto-start → both channels live → notes → stop → final summary + suggestions → indexed under `taurine` in map after sync.

---

## Self-review record (rev 2)

- All 16 codex-review findings addressed: B1→Task 1.2 coordinator contract; B2→Task 1.1 per-start lifecycle + grace + sidecar.ts replay fix; B3→Task 4.1 stop-path ordering + runtime-owned tasks + best-effort emit; B4→Task 4.2 hidden registered connector; B5→Task 4.1 `llm_agent`/`complete_json_messages`; B6→Task 4.4 `[M#]` citable evidence; M7→store location corrected everywhere; M8→content_hash/updated_at/delete handling; M9→`LIVE_SUMMARY_MODEL` + reasoning pin + alias grep-test; M10→`meetings.chat_id` + reverse lookup + regenerate; M11→`[T#]`-preserving label format; M12→flush-before-stop + freshest-notes snapshot + truncation math; M13→live-state keying + stored-transcript rendering + batched appends + update-kind rules; M14→onError/Buffer-normalization/asarUnpack/destroyed-cleanup/universal binary/TCC-as-blocker; M15→canonical-content invariant + occurrence + base_rev + filtered serialization + fallback mechanism; M16→vitest setup, Task 4.5 harness, battery referenced as orchestrator-owned.
- Type consistency pass: `SuggestedEdit {id, anchor, occurrence, insertMd, rationale, status}` matches wire `suggested_edits` + `suggestions_json`; `meeting_id` identical across StartMsg/NotepadMsg/ChatSendMsg/events; `notes_rev`/`base_rev` int end-to-end; channel byte 1 + label `system` everywhere.
- No placeholder steps; every task names exact files, shapes, commands, and failure modes.
