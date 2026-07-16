# Meetings Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Granola-style Meetings section: persistent meeting notepads (WYSIWYG markdown + suggestions), system-audio second Deepgram stream with You/Them labeling, user notes feeding the live summarizer, Gemini 3.5 Flash end-of-call summary with graph access + suggested edits, meetings indexed as an implicit `taurine` source, and a call-detection notification with countdown auto-start.

**Spec:** `docs/superpowers/specs/2026-07-16-meetings-overhaul-design.md` — read it first; it is authoritative on behavior.

**Architecture:** Core (Python sidecar) gains a meetings store + REST, notepad/final-summary lanes on the existing `/session` WS, and a `taurine` note source. App (Electron 39 + React 19 + Tailwind 4) gains a Meetings tab (Plate editor), a main-process system-audio tap (audiotee/CATap) feeding channel 1 of the existing PCM protocol, and a micwatch Swift helper driving a global notification.

**Tech Stack:** FastAPI sidecar (`core/taurine_core`), Electron 39/React 19, Plate (`platejs`, `@platejs/markdown`, `@platejs/suggestion` — MIT packages only, never `xl-*`/paid), `audiotee` npm, Deepgram nova-3, OpenRouter `google/gemini-3.5-flash`.

## Global Constraints

- Everything lands on `main` in /Users/neil/taurine; each phase leaves the repo green (core: `cd /Users/neil/taurine/core && .venv/bin/python -m pytest -q`; app: `cd /Users/neil/taurine/app && npx tsc --noEmit -p tsconfig.web.json && npm run lint && npm run build` — use `npx electron-vite build` if `npm run build` is absent).
- Python: match existing style (type hints, pydantic models in `schemas.py`, config constants in `config.py`). TS: match existing idioms (plain useState/props, hand-rolled tolerant parsers in `sidecar.ts`, Tailwind utilities + `@layer components` classes).
- Live retrieval/abstention/calibration code is OFF-LIMITS except where a task names a file. Do not touch `retrieval/`, calibration artifacts, or chat routing defaults.
- Wire contracts (message/table/REST shapes) in this plan are exact; do not rename fields.
- Commits: conventional prefix, trailer `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.
- macOS-only features (audiotee, micwatch) must degrade gracefully: feature-detect, never crash on unsupported platforms.

---

## Phase 1 — Meetings foundation

### Task 1.1 (core): meetings store + REST + session binding

**Files:**
- Create: `core/taurine_core/meetings.py` (store class)
- Modify: `core/taurine_core/store.py` (only if the chats tables live there — mirror wherever `GET /chats` persistence lives), `core/taurine_core/server.py` (routes ~line 660 area, `/session` handler ~773-856, `_parse_control` ~1376, UpdateBus KINDS ~134), `core/taurine_core/schemas.py` (control msgs ~666, events), `core/taurine_core/live/ingest.py` (transcript append hook)
- Test: `core/tests/test_meetings.py`

**Interfaces (produced — later tasks and the app rely on these exactly):**
- Table: `meetings(meeting_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, started_at REAL, ended_at REAL, notes_md TEXT NOT NULL DEFAULT '', final_summary_md TEXT NOT NULL DEFAULT '', suggestions_json TEXT NOT NULL DEFAULT '[]', transcript_json TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL DEFAULT 'draft')`; `status ∈ {'draft','live','summarizing','done'}`; `meeting_id = uuid4().hex`.
- `MeetingStore` methods: `create(title='') -> Meeting`, `get(meeting_id) -> Meeting | None`, `list_summaries() -> list[MeetingSummaryRow]` (no bodies, newest-first by created_at), `update(meeting_id, *, title=None, notes_md=None, suggestions=None) -> Meeting`, `delete(meeting_id)`, `append_transcript(meeting_id, chunk: dict)` (chunk = `{"channel":int,"text":str,"ts":float,"seq":int}`), `set_status(meeting_id, status, *, started_at=None, ended_at=None)`.
- REST: `GET /meetings` → `{"meetings":[{"meeting_id","title","created_at","started_at","ended_at","status"}]}`; `POST /meetings` body `{"title"?: str}` → full meeting JSON (all columns, `suggestions` as parsed list, `transcript` as parsed list); `GET /meetings/{id}` → full meeting JSON (404 if missing); `PATCH /meetings/{id}` body any of `{"title","notes_md","suggestions"}` → full meeting JSON; `DELETE /meetings/{id}` → `{"ok": true}`.
- Control message: `StartMsg` gains `meeting_id: str | None = None`. When a session starts with a meeting_id: `set_status(id,'live',started_at=now)`; every **final** transcript chunk also goes to `append_transcript`; on `stop` control **or socket disconnect while live**: `set_status(id,'done', ended_at=now)` (phase 4 changes 'done' to the summarizing pipeline).
- `/updates` bus: new kind `"meetings"` pushed after any meeting mutation, payload `{"type":"meetings"}` (list invalidation only).

**Steps:**
- [ ] Write failing tests in `core/tests/test_meetings.py`: store CRUD round-trip (create → update notes_md → get returns it; list_summaries excludes `notes_md`); `append_transcript` accumulates ordered chunks; status transitions (`draft→live` sets `started_at`, `→done` sets `ended_at`); REST endpoints via the existing FastAPI test-client pattern (find it in `core/tests/test_server*.py` and mirror fixtures) including 404 and PATCH partial-update; `_parse_control` accepts `{"type":"start","channels":[{"id":0,"label":"mic"}],"meeting_id":"abc"}` and old messages without `meeting_id` still parse.
- [ ] Run: `cd /Users/neil/taurine/core && .venv/bin/python -m pytest tests/test_meetings.py -q` → FAIL (module missing).
- [ ] Implement `meetings.py` + wiring. Persistence in the same SQLite DB the chats endpoints use (find via the `GET /chats` handler at `server.py:670`); `CREATE TABLE IF NOT EXISTS` on init. Live-session binding in the `/session` websocket handler: hold `meeting_id` per connection; transcript hook where finals reach the ledger in `ingest.py:111-139` (callback injected via `make_live_session`, keep `LiveSession` decoupled — pass an `on_final_chunk` callable).
- [ ] Run tests + full suite → PASS, no regressions.
- [ ] Commit: `feat(core): meetings store, REST, live-session binding`.

### Task 1.2 (app): Meetings tab shell with Plate notepad

**Files:**
- Create: `app/src/renderer/src/lib/meetings.ts`, `app/src/renderer/src/views/meetings/MeetingsView.tsx`, `app/src/renderer/src/views/meetings/MeetingSidebar.tsx`, `app/src/renderer/src/views/meetings/NotepadEditor.tsx`, `app/src/renderer/src/views/meetings/TranscriptPanel.tsx`, `app/src/renderer/src/views/meetings/SummaryCard.tsx`
- Modify: `app/src/renderer/src/App.tsx` (tab rename `'live'`→`'meetings'` at :32-40; replace the live 2×2 grid at :362-391 with `<MeetingsView …/>`; session start gains meeting binding), `app/src/renderer/src/lib/sidecar.ts` (StartMsg meeting_id; `meetings` update kind in `updates.ts`), `app/package.json` (Plate deps)
- Delete from layout (files stay, imported by new components): direct grid usage of `SummaryPanel`/`ActionableRail`/`FeedView`/`AmbientList` in App.tsx.

**Interfaces:**
- Consumes: Task 1.1 REST + `meetings` update kind + `start.meeting_id`.
- Produces: `lib/meetings.ts` exports `interface Meeting {meetingId, title, createdAt, startedAt, endedAt, notesMd, finalSummaryMd, suggestions: SuggestedEdit[], transcript: TranscriptChunk[], status}`, `interface SuggestedEdit {id: string; anchor: string; insertMd: string; rationale: string; status: 'pending'|'accepted'|'rejected'}`, functions `listMeetings()`, `createMeeting()`, `getMeeting(id)`, `patchMeeting(id, patch)`, `deleteMeeting(id)` (snake_case↔camelCase mapping, tolerant parsing like `graph.ts`).
- `NotepadEditor` props: `{ valueMd: string; onChangeMd: (md: string) => void; suggestions: SuggestedEdit[]; onResolveSuggestion: (id: string, accept: boolean) => void; readOnly?: boolean }`. Internally Plate; markdown (de)serialization at the boundary via `@platejs/markdown`; suggestion marks rendered from `suggestions` with inline accept/deny controls (suggestion UI may be minimal in this task — full behavior lands in Task 4.3, but the props contract is fixed now).
- `MeetingsView` props (from App): `{ live: { listeningState, chunks, interims, summary, actionables, liveGrounding, micError }, callbacks: { startListening(meetingId), stopListening(), setActionableStatus(), inspectGrounding() } }` — exact prop names may follow existing App.tsx conventions, but MeetingsView owns ALL meetings UI state (selected meeting, editor content, autosave, panel toggles).

**Steps:**
- [ ] Install Plate: resolve the current MIT package set for a markdown WYSIWYG with suggestions from https://platejs.org/docs (core `platejs` packages + markdown + suggestion; NO `xl-`/paid/GPL packages). Pin exact versions in package.json.
- [ ] Build `lib/meetings.ts` + `NotepadEditor` + view components. Layout per spec §2.2: left sidebar (meetings list + New note), center title input + editor, top `SummaryCard` (wraps SummaryPanel content + actionables; hidden when no summary and not listening), right hideable `TranscriptPanel` (FeedView + AmbientList/TriggerInspector behind an "inspect" toggle), bottom chat bar placeholder (non-functional this phase — a disabled composer with "Ask about this meeting" placeholder). Autosave: debounce 800 ms after editor change → `patchMeeting`; flush on meeting switch/unmount.
- [ ] Wire App.tsx: tab rename (label **Meetings**), pass live state down, `startListening` creates/binds a meeting (`createMeeting()` if none selected) and sends `meeting_id` in start; keep all other tabs untouched.
- [ ] Verify: `npx tsc --noEmit -p tsconfig.web.json`, `npm run lint`, `npx electron-vite build` all clean. Launch dev app; create/rename/edit/delete meetings; markdown round-trips (type heading/bold/list, switch meetings, come back).
- [ ] Commit: `feat(app): Meetings tab with Plate notepad and meeting persistence`.

**Phase 1 gate (orchestrator):** core suite green; app builds; manual smoke — start listening binds a meeting, transcript lands in `transcript_json`, notes autosave.

---

## Phase 2 — System audio channel

### Task 2.1 (app main/preload): audiotee tap + IPC relay

**Files:**
- Create: `app/src/main/sysaudio.ts`
- Modify: `app/src/main/index.ts`, `app/src/preload/index.ts`, `app/src/preload/index.d.ts`, `app/electron-builder.yml` (extendInfo), `app/package.json` (dep `audiotee`)

**Interfaces (produced):**
- Preload `window.api` gains: `sysAudio: { available(): Promise<boolean>; start(): Promise<void>; stop(): Promise<void>; onChunk(cb: (chunk: ArrayBuffer) => void): () => void }`. `available()` = darwin && macOS ≥ 14.2 && binary present. Chunks are raw PCM s16le mono 48 kHz as produced by audiotee.
- `electron-builder.yml` mac.extendInfo gains `NSAudioCaptureUsageDescription: Taurine records system audio to transcribe the other side of your meetings.`

**Steps:**
- [ ] `npm install audiotee` (main-process usage). Implement `sysaudio.ts`: lazy AudioTee instance `{ sampleRate: 48000 }`, start/stop idempotent, `data` events → `webContents.send('sysaudio:chunk', chunk.data)`, errors → `webContents.send('sysaudio:error', message)` and auto-stop. `ipcMain.handle('sysaudio:available'|'sysaudio:start'|'sysaudio:stop')`. macOS version gate via `process.getSystemVersion()`.
- [ ] Preload bridge with `ipcRenderer.on('sysaudio:chunk')` subscription returning an unsubscribe fn; extend `TaurineApi` types.
- [ ] Verify: typecheck+lint+build; on this Mac run dev app and confirm `await window.api.sysAudio.available()` is true and start() triggers the one-time "System Audio Recording Only" TCC prompt, then chunks flow (log byteLength in devtools). If TCC misattributes in dev, document the observed behavior in the commit body — do not block the task on it.
- [ ] Commit: `feat(app): system-audio tap via CoreAudio process tap (audiotee)`.

### Task 2.2 (app renderer): channel-1 capture + You/Them labels

**Files:**
- Create: `app/src/renderer/src/lib/sysaudio.ts` (`SystemAudioCapture`)
- Modify: `app/src/renderer/src/App.tsx` (start/stop lifecycle), `app/src/renderer/src/views/FeedView.tsx` (`channelLabel` :322-324)

**Interfaces:**
- Consumes: `window.api.sysAudio`, `SidecarClient.sendAudioFrame`, `client.start(channels)` (already accepts arrays).
- Produces: `class SystemAudioCapture { constructor(onFrame: (frame: ArrayBuffer) => void); async start(): Promise<void>; stop(): void }` — decimates 48 kHz→16 kHz by averaging each 3 samples, buffers into 4000-sample (250 ms) frames, prefixes channel byte `1` (reuse/export `prefixChannel` from `lib/audio.ts`).

**Steps:**
- [ ] Implement `SystemAudioCapture` (pure TS; the 3:1 decimation and framing logic mirrors `pcm-worklet.js` but runs on IPC chunks). Handle non-multiple-of-3 chunk tails by carrying remainder samples.
- [ ] `startListening`: when `await window.api.sysAudio.available()`, start it and declare `channels: [{id:0,label:'mic'},{id:1,label:'system'}]`; otherwise mic-only exactly as today. `stopListening`/error paths stop both captures. `sysaudio:error` shows the existing micError surface with a distinct message but does NOT kill the mic channel.
- [ ] FeedView: `channelLabel` → `0: 'You'`, `1: 'Them'`, else `Ch ${n}`.
- [ ] Verify: typecheck+lint+build; dev smoke — play a YouTube video during a session, confirm channel-1 transcripts appear labeled **Them** while mic lines show **You**.
- [ ] Commit: `feat(app): stream system audio as channel 1 with You/Them labels`.

### Task 2.3 (core): speaker labels in ledger + summarizer awareness

**Files:**
- Modify: `core/taurine_core/live/ledger.py` (transcript evidence rendering ~:174), `core/taurine_core/live/summarizer.py` (SYSTEM_PROMPT :38-59)
- Test: extend `core/tests/test_ledger*.py` / summarizer tests (find existing files by `grep -rl "ledger" core/tests`)

**Interfaces:** transcript evidence lines render `You: <text>` (channel 0), `Them: <text>` (channel 1), `Ch{n}: <text>` (other). SYSTEM_PROMPT gains one sentence: `Transcript lines are labeled You (the user's own mic) and Them (other call participants); attribute statements accordingly.`

**Steps:**
- [ ] Failing tests: ledger window rendering shows `You:`/`Them:` prefixes for channels 0/1; existing citation/offset behavior unchanged.
- [ ] Implement; run full core suite → green.
- [ ] Commit: `feat(core): speaker-side labels in live transcript evidence`.

**Phase 2 gate (orchestrator):** dev-app smoke with real system audio; core suite green.

---

## Phase 3 — Notes ↔ live summary (Gemini)

### Task 3.1 (core): notepad block + Gemini summary model

**Files:**
- Modify: `core/taurine_core/schemas.py` (new control msg), `core/taurine_core/server.py` (`_parse_control`/`_handle_control`), `core/taurine_core/live/ingest.py` (hold notepad text), `core/taurine_core/live/summarizer.py` (`_build_user_prompt` :280-298), `core/taurine_core/config.py` (SUMMARY_MODEL swap + `MEETING_NOTES_PROMPT_MAX_CHARS = 8000`)
- Test: summarizer prompt tests + control parse tests

**Interfaces:**
- Control: `{"type":"notepad","meeting_id": str,"text_md": str}` → `LiveSession.set_notepad(text_md)` (stored, no event reply).
- Prompt: `_build_user_prompt` emits, before `<transcript_evidence>`, a `<user_notes>` block containing the notepad capped at `MEETING_NOTES_PROMPT_MAX_CHARS` (keep head 6000 + tail 2000 with `…[truncated]…` marker when over), plus instruction text per spec §2.3 (notes signal importance; don't restate them).
- Config: `SUMMARY_MODEL = "google/gemini-3.5-flash"`; `SUMMARY_FALLBACK_MODEL` unchanged.

**Steps:**
- [ ] Failing tests: control message parses (and rejects missing text_md tolerantly); prompt contains `<user_notes>` when set and omits the block entirely when empty; truncation boundaries exact; existing summarizer JSON validation untouched.
- [ ] Implement. Then verify Gemini structured output ONCE for real: `cd /Users/neil/taurine/core && OPENROUTER smoke — .venv/bin/python -c` script calling `OpenRouterClient.complete_json(model="google/gemini-3.5-flash", system="Return JSON {\"ok\":true}", user="go")` — if `require_parameters` rejects `response_format` for this model, add a per-model relaxation in `llm.py` (documented in code) rather than dropping validation. Requires OPEN_ROUTER_API_KEY from core/.env; if the key is unavailable in your sandbox, STOP and report — the orchestrator runs the smoke.
- [ ] Full suite green.
- [ ] Commit: `feat(core): user-notes block in live summary; Gemini 3.5 Flash summarizer`.

### Task 3.2 (app): notepad push + Add-to-notes

**Files:**
- Modify: `app/src/renderer/src/views/meetings/MeetingsView.tsx`, `app/src/renderer/src/views/meetings/SummaryCard.tsx`, `app/src/renderer/src/lib/sidecar.ts` (send `notepad` control)

**Interfaces:** `SidecarClient.sendNotepad(meetingId: string, textMd: string)`. SummaryCard gains `onAddToNotes(summaryMd: string)` → MeetingsView appends `\n\n${summaryMd}\n` to the editor content as regular text (through the same path as user edits → autosave).

**Steps:**
- [ ] Implement debounced (2 s) `sendNotepad` while `listeningState==='listening'` and the edited meeting is the live one. Add-to-notes button on SummaryCard.
- [ ] Verify: typecheck+lint+build; dev smoke — jot notes during a session, confirm next summary references them (sidecar logs show `<user_notes>`); Add-to-notes appends and persists.
- [ ] Commit: `feat(app): notepad feeds live summary; insert summary into notes`.

**Phase 3 gate (orchestrator):** live session E2E on this machine; battery live controls re-run (`battery-v5.py`) — abstention must hold (summarizer model change does not touch retrieval, but verify).

---

## Phase 4 — Final summary, suggestions, taurine source, meeting chat

### Task 4.1 (core): final-summary agent + events

**Files:**
- Create: `core/taurine_core/live/final_summary.py`
- Modify: `core/taurine_core/server.py` (trigger on stop/disconnect-while-live; event sender), `core/taurine_core/schemas.py` (events), `core/taurine_core/config.py` (`FINAL_SUMMARY_MODEL = "google/gemini-3.5-flash"`, `FINAL_SUMMARY_MAX_TOOL_CALLS = 6`, `FINAL_SUMMARY_TIMEOUT_S = 120`), `core/taurine_core/meetings.py` (persist results)
- Test: `core/tests/test_final_summary.py` (fake LLM client — mirror how summarizer tests fake `stream_chat`/`complete_json`)

**Interfaces:**
- `async def run_final_summary(meeting, llm, engine_factory, store, profile_block, emit) -> FinalSummaryResult` where `emit(event)` sends `/session` events; uses `stream_chat` with the chat tool definitions (`chat/tools.py`) for KB access, then a final strict-JSON turn (or tool-forced JSON) validated fail-closed.
- Events: `{"type":"final_summary_status","meeting_id":str,"state":"running"|"done"|"error","detail":str}` and `{"type":"final_summary","meeting_id":str,"title":str,"final_summary_md":str,"suggested_edits":[{"id":str,"anchor":str,"insert_md":str,"rationale":str,"status":"pending"}]}`.
- Output JSON schema from the model: `{"title": str, "final_summary_md": str, "suggested_edits": [{"anchor": str, "insert_md": str, "rationale": str}]}` — validation rules: ≤10 edits; every non-empty `anchor` MUST be an exact substring of the meeting's `notes_md` (drop violators, keep the rest); `insert_md` non-empty; ids assigned server-side (`uuid4().hex[:8]`). Empty/invalid overall JSON → status `error`, meeting still marked `done`, title/notes untouched.
- Meeting flow: `stop` → status `summarizing` + `final_summary_status running` → on completion persist `final_summary_md`, `suggestions_json`, auto-title if meeting title empty → status `done` + events. Timeout/exception → `error` event, status `done`.

**Steps:**
- [ ] Failing tests: happy path persists + emits (fake LLM returns canned tool call then JSON); anchor-violation edits dropped; >10 edits truncated; LLM garbage → error event, status done; timeout path; untitled meeting gets auto-title, titled meeting keeps its title.
- [ ] Implement (inputs per spec §2.4: labeled transcript, notes_md, last live summary + actionable statuses, profile block; prompt states the You/Them convention and that edits must be additive enhancements anchored to exact note text).
- [ ] Full suite green. Commit: `feat(core): end-of-call final summary agent with KB tool access`.

### Task 4.2 (core): implicit `taurine` source

**Files:**
- Create: `core/taurine_core/indexer/taurine_source.py` (builds `NoteRecord`s from meetings)
- Modify: `core/taurine_core/server.py` (upsert + queue sync after final summary; also on PATCH of a `done` meeting's notes), `core/taurine_core/config.py` (`TAURINE_SOURCE_ID="taurine"`, `SOURCE_RANK["taurine"]=0.95`, `RECENCY_HALF_LIFE_DAYS["taurine"]=60`), `/connectors` listing (exclude `taurine` — it must never appear)
- Test: `core/tests/test_taurine_source.py`

**Interfaces:**
- `def meeting_note_records(meetings: Iterable[Meeting]) -> list[NoteRecord]`: one record per `done` meeting with non-empty content; `note_id=f"taurine:{meeting_id}"`, `source="taurine"`, `title=meeting title or 'Meeting YYYY-MM-DD'`, `text = notes_md + ("\n\n## Meeting summary\n" + final_summary_md if final_summary_md else "")`, `modified/created` from meeting timestamps.
- After final summary (and after PATCH to a done meeting): `index.replace_source_records("taurine", meeting_note_records(store.all_done()))` then queue the existing global sync IF none running (reuse `_run_global_sync` machinery; a simple "sync pending" debounce flag is fine).

**Steps:**
- [ ] Failing tests: record shape/id/source; empty meetings skipped; `/connectors` response never contains `taurine`; replace-records called after a (faked) final summary completes.
- [ ] Implement; full suite green. Commit: `feat(core): index meetings as implicit taurine source`.

### Task 4.3 (app): suggestions UX + final card + source branding

**Files:**
- Modify: `app/src/renderer/src/views/meetings/NotepadEditor.tsx` (real suggestion marks), `MeetingsView.tsx`, `SummaryCard.tsx` (final summary card variant), `app/src/renderer/src/lib/sidecar.ts` (parse the two new events), `app/src/renderer/src/lib/graph.ts` (`SOURCE_COLORS["taurine"]="#ffd75e"`, `SOURCE_LABELS["taurine"]="Taurine"`), `app/src/renderer/src/lib/labels.ts` (case `taurine` → `'Taurine'`), `app/src/renderer/src/assets/main.css` (`.source-taurine` chip)

**Interfaces:** consumes Task 4.1 events + Task 1.2 `SuggestedEdit`/`onResolveSuggestion`. Behavior: on `final_summary` event for the open meeting, show final card + map each pending edit to a Plate suggestion (anchor exact-match; no match → append a suggestion block at document end). Accept = apply `insert_md` after the anchor (or at end) as normal text; deny = discard. Both update the meeting via `patchMeeting(id, {suggestions})` with the new status, and accepted content flows through autosave. `final_summary_status` renders a small progress state on the card ("Summarizing…").

**Steps:**
- [ ] Implement event parsing (tolerant, like the rest of `parseSidecarEvent`), suggestion rendering with accept/deny controls styled like the app (accent for accept, subtle for deny, rationale on hover/expand).
- [ ] Verify: typecheck+lint+build; dev smoke with a scripted short meeting (orchestrator provides E2E script) — suggestions appear, accept inserts text, deny removes, state survives app reload (suggestions_json).
- [ ] Commit: `feat(app): AI suggested edits with accept/deny; taurine source branding`.

### Task 4.4 (core+app): meeting-scoped chat

**Files:**
- Modify: `core/taurine_core/schemas.py` (`ChatSendMsg.meeting_id: str | None = None`), `core/taurine_core/chat/engine.py` (`_build_messages` injects `<meeting_context>`), `core/taurine_core/server.py` (pass meeting store into chat path), `core/taurine_core/config.py` (`MEETING_CONTEXT_MAX_CHARS = 12000`); app: `app/src/renderer/src/lib/chat.ts` (send meeting_id), `app/src/renderer/src/views/meetings/MeetingChat.tsx` (new — bottom drawer reusing Composer/AssistantTurn), `MeetingsView.tsx`
- Test: `core/tests/test_chat_meeting_context.py`

**Interfaces:** when `meeting_id` set and meeting exists: a non-citable `<meeting_context>` block (notepad md + last 60 transcript lines labeled You/Them + current final/live summary, total capped at `MEETING_CONTEXT_MAX_CHARS`) is inserted next to the profile block; `seed_note_ids` gains `taurine:{meeting_id}`. Without `meeting_id`: byte-identical messages to today (test this). App: each meeting lazily creates one chat via existing `POST /chats` (store the chat_id in renderer state keyed by meeting; title `Meeting: {title}`), drawer streams via existing ChatClient.

**Steps:**
- [ ] Failing tests: block present/capped/absent; non-meeting chat unchanged; unknown meeting_id ignored gracefully.
- [ ] Implement core, then app drawer (collapsed bar “Ask about this meeting…” → expands to thread; Esc collapses).
- [ ] Full core suite + app build green. Commit: `feat: meeting-scoped chat with notepad and transcript context`.

**Phase 4 gate (orchestrator):** scripted E2E (fake transcript session → stop → final summary with ≥1 suggestion → note under `taurine` source after sync → map legend shows Taurine); battery re-run; chat latency spot-check.

---

## Phase 5 — Call detection + auto-start notification

### Task 5.1 (app main): micwatch helper + watcher

**Files:**
- Create: `app/resources/native/micwatch.swift`, `app/scripts/build-native.mjs` (swiftc build; skipped gracefully if swiftc missing), `app/src/main/micwatch.ts`
- Modify: `app/src/main/index.ts` (spawn on ready), `app/package.json` (`build:native` script + `postinstall` hook if binary absent), `app/src/preload/index.ts` + `index.d.ts`

**Interfaces:**
- Swift helper: subscribes `kAudioDevicePropertyDeviceIsRunningSomewhere` on the default input device AND `kAudioHardwarePropertyDefaultInputDevice` (re-subscribe on device change); prints one JSON line `{"active":true|false}` on every state edge (plus the initial state on launch); exits cleanly on SIGTERM. ~60-90 lines, no deps.
- `micwatch.ts`: spawn `resources/native/micwatch` (path via `app.isPackaged` branch), parse lines, debounce activations <1.5 s, forward via `webContents.send('micwatch:active', bool)`. Missing binary/non-darwin → silently inactive.
- Preload: `window.api.micWatch: { onActive(cb: (active: boolean) => void): () => void }`.
- Commit BOTH the source and the built binary (`app/resources/native/micwatch`, arm64) so dev works without a build step; `build:native` re-produces it.

**Steps:**
- [ ] Write the Swift helper + build script; build it (`npm run build:native`); test standalone: run binary, start a Voice Memo recording → `{"active":true}` line; stop → false.
- [ ] Wire main/preload; typecheck+lint+build.
- [ ] Commit: `feat(app): mic-activity watcher (CoreAudio) for call detection`.

### Task 5.2 (app renderer): notification card + auto-start

**Files:**
- Create: `app/src/renderer/src/components/CallNotification.tsx`
- Modify: `app/src/renderer/src/App.tsx`

**Interfaces:** `CallNotification` props `{ onStart(): void; onDismiss(): void; seconds: number }` — fixed bottom-right card (z-50, above all view layers), top edge hosts a full-width 3px countdown bar animating `scaleX 1→0` over `seconds` (10), buttons **Start notes** (primary) / **Dismiss** (secondary); reaching zero calls `onStart`.

**App wiring (state machine in App.tsx):**
- Show card when `micwatch:active=true` AND `listeningState==='idle'` AND not suppressed.
- `onStart` / countdown-zero: create meeting (or reuse today's live draft), switch `activeTab='meetings'`, run the exact `startListening` path (mic + system audio), hide card.
- `onDismiss`: hide; suppress until `micwatch:active=false` has been observed continuously ≥5 s (re-arm), so one prompt per call.
- While Taurine itself is listening, micwatch events are ignored entirely.

**Steps:**
- [ ] Implement card + wiring; countdown driven by a single `setTimeout` + CSS transition (pause not required).
- [ ] Verify: typecheck+lint+build; dev smoke — start a Voice Memo recording → card appears; let it expire once (auto-start fires), dismiss once (no re-prompt until mic idle 5 s).
- [ ] Commit: `feat(app): call-detected notification with countdown auto-start`.

**Phase 5 gate (orchestrator):** full manual scenario — incoming "call" (any app using mic) → notification → auto-start → meeting with both channels → notes → stop → final summary → suggestions → indexed note visible in map after sync.

---

## Self-review record

- Spec coverage: §1→Tasks 2.1-2.3; §2.1→1.1; §2.2→1.2; §2.3→3.1-3.2; §2.4→4.1+4.3; §2.5→4.2+4.3; §2.6→4.4; §3→5.1-5.2; §4 constants distributed to their owning tasks; §6 testing embedded per-task + phase gates; §7 risks → Task 2.1 (TCC note), 3.1 (Gemini smoke), 4.1 (anchor validation), 4.2 (sync queue guard).
- Contracts consistent: `SuggestedEdit` field names match between core events (snake_case on wire) and `lib/meetings.ts` (camelCase mapped); `meeting_id` spelled identically in StartMsg/notepad/ChatSendMsg; channel byte 1 + label `system` everywhere.
- No placeholder steps; every task names exact files, shapes, and commands.
