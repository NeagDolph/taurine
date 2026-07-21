# Taurine Knowledge Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Taurine's five-tab Electron renderer with the complete Knowledge Atlas workstation while preserving durable meeting data, making every generated note edit reviewable, and using one serialized actor for resumable live sessions.

**Architecture:** Approved Tasks 1–4 remain the durable prerequisite. Six end-to-end slices now carry the branch from a serialized `SessionActor` and bounded desktop boundary through a working shell, meetings, inquiry/evidence, knowledge operations, and one exhaustive cutover. FastAPI/SQLite remain the durable spine; Redux Toolkit, one typed sidecar gateway, and one `MeetingDocumentController` own renderer coordination; the actor alone owns ephemeral live-session state.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLite, pytest/pytest-asyncio; Electron 39, React 19, TypeScript 5.9, Redux Toolkit/RTK Query, BlockNote, Canvas with d3-force, React Virtual, Vitest, Testing Library, axe-core, and Playwright Electron.

**Authoritative design:** Read both documents before executing a slice:

- `docs/superpowers/specs/2026-07-20-ui-first-session-architecture-reset-design.md`
- `docs/superpowers/specs/2026-07-20-taurine-ui-knowledge-atlas-design.md`

This plan supersedes the former 28-task sequence in this file. It does not narrow any visual, interaction, evidence, accessibility, privacy, or release requirement from the Knowledge Atlas design. The reset specification controls lifecycle ownership and task sequencing wherever the two specifications differ.

## Global constraints

- Execute in the current `codex/ui-knowledge-atlas` worktree. Do not reset, rebase, rewrite, or replay completed branch history.
- Approved prerequisite state is Task 1 commit `0d4827d` plus Tasks 2–4 and their approved repairs through `bb4da82`. Those commits already provide the harness, bundle budgets, revision-aware meeting notes, durable proposals/commands/undo, producer persistence, and referential chat cleanup.
- Commits after `bb4da82` and the current uncommitted `core/taurine_core/live/session_registry.py` repair are superseded implementation material. Do not continue its lock, detached-entry, callback-handoff, retry-driver, or cleanup-successor design. Slice 1 replaces the file and its tests wholesale; useful snapshot limits and user-visible contract assertions may be rewritten against the actor.
- The paused repair archive `/private/tmp/taurine-atlas-paused-task5-20260720.tar.gz` is reference-only. Do not extract it over the worktree.
- Do not add a data migration task or an ephemeral-session migration. Existing additive meeting proposal/operation DDL from Tasks 2–4 remains initialized in place; existing `taurine.db` and `taurine-knowledge.db` open in place. The actor table, deadlines, snapshot, audio state, and task ownership are memory-only.
- Release one completed Knowledge Atlas interface: no feature flag, dual renderer, legacy route, fallback CSS, compatibility UI, or alternate successful test path.
- Preserve at least the verified baselines of 981 passing Core tests with 2 skipped and 61 passing Vitest tests; new tests add to those floors.
- Python remains `>=3.12`; Electron remains `^39.2.6`, React `^19.2.1`, and TypeScript `^5.9.3` unless a separately reviewed dependency change is unavoidable.
- Redux Toolkit is the only renderer global-state library. Sockets, fetches, capture sequencing, and editor writes do not live in React views.
- `MeetingDocumentController` is the only frontend caller allowed to mutate canonical meeting Markdown. Every AI-authored path first persists a `MeetingNoteProposal`.
- `SessionActor` is the only component allowed to read or mutate the live-session table. Public commands cross one bounded admission semaphore and one private FIFO `asyncio.Queue`; only the actor loop changes records.
- Audio is never queued or replayed. Disconnect suspends bounded reasoning state for 60 seconds; Stop rejects new audio immediately and uses a 10-second resumable deadline or a zero-second finalization request.
- Keep the existing wire error `meeting_session_conflict`. It is the transport rendering of the reset specification's Knowledge Atlas `session_conflict` outcome. `state_reset` is reserved for a missing in-memory record after process restart.
- Do not add graph pagination, a unified WebSocket, replay IDs, final-summary retry, profile cancellation, hard retrieval scoping, or a projected-edge call to `/graph/edge-reason`.
- Main-window default/minimum bounds are 1280×820 and 960×640 CSS px. Ordinary Live is two lanes; opt-in three-lane Live begins at 1440 px. Compact Live is the same BrowserWindow at 440×680 default, 420×560 minimum, and 560×900 maximum.
- Initial renderer shell is at most 1.25 MB uncompressed JavaScript and 140 KB uncompressed CSS. Each lazy feature chunk is at most 900 KB without a checked-in analysis.
- Atlas sustains at least 30 fps at 5,000 projected nodes on the reference Mac; virtualized lists sustain 60 fps; selection/command response p95 is below 100 ms excluding model/network work.
- Every surface meets WCAG 2.2 AA, keyboard-only use, 200% zoom, reduced motion, forced colors, 35% string expansion, and RTL. Every target meets 24×24 px; primary, destructive, and meeting controls target 40–44 px.
- Privacy copy is exactly Protection requested, Limited on this Mac, Unavailable, or Privacy curtain on. Never claim Taurine is invisible or uncapturable.
- Evidence copy keeps Cited, Consulted, Retrieved, fallback, abstained, interrupted, historical, conflicted, and unavailable distinct. Never render “verified,” “fact checked,” or numeric confidence as a primary truth claim.
- Determinate progress requires real `done/total`. Runtime, model, finalization, and profile operations without totals remain indeterminate.
- Production Redux DevTools are off. Logs and breadcrumbs exclude note, transcript, profile, passage, prompt, answer, token, credential, and raw retrieval content.
- `app/src/renderer/src/app/App.tsx` stays below approximately 120 substantive lines and owns providers plus `AppShell` only.
- Each slice has one combined RED checkpoint, one combined GREEN checkpoint, one runnable inspectable journey, one frozen commit, and one independent review in this order: GREEN → commit/freeze → stop → independent review. Begin no next-slice edit until that review is clean; repair findings in the frozen slice and submit a new frozen commit for re-review.
- Every checkbox below is one 2–5 minute action. If local code shape makes an action larger, split it into adjacent checkboxes before editing; do not merge behavioral steps into an unreviewable batch.

## Execution preflight

- [ ] **Locate and validate the current Atlas worktree.**

  Run:

  ```bash
  export WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
  cd "$WORKTREE_ROOT"
  test "$(git branch --show-current)" = "codex/ui-knowledge-atlas"
  git merge-base --is-ancestor bb4da82 HEAD
  git status --short
  ```

  Expected: the branch test and ancestry check exit 0. Status may show only the known superseded Task 5 actor-predecessor files; preserve any unrelated user changes.

- [ ] **Record the prerequisite commits without replaying them.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT"
  git log --oneline --reverse b84877d..bb4da82
  ```

  Expected: the range begins with `0d4827d test: establish knowledge atlas quality gates`, includes durable note/proposal work, and ends with `bb4da82 fix(core): harden durable meeting proposal producers`.

- [ ] **Record the superseded Task 5 delta.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT"
  git diff --name-status bb4da82..HEAD
  git diff --stat
  ```

  Expected: committed Task 5 changes are limited to `core/taurine_core/live/session_registry.py`, `core/tests/test_session_registry.py`, and progress metadata; current uncommitted repair is visible. Do not stage or commit it.

## Canonical cross-slice interfaces

These names and shapes are fixed. If implementation discovers an incompatible checked-in contract, update every producer, consumer, parser, fixture, and test in the same slice before freezing it.

```python
SessionState = Literal["active", "suspended", "finalizing"]
SessionDisposition = Literal["new", "resumed", "state_reset", "stopped_resumable"]
AsrState = Literal["disconnected", "connecting", "connected", "error"]

@dataclass(frozen=True, slots=True)
class SessionChannel:
    id: int
    label: str

@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    meeting_id: str
    client_session_id: str
    disposition: SessionDisposition
    resume_deadline: float | None
    asr: AsrState
    channels: tuple[SessionChannel, ...]
    transcript_seq: int
    summary: SummaryEvent | None
    grounding: LiveGroundingEvent | None
    actionables: tuple[Actionable, ...]
    note_suggestions: tuple[NoteSuggestionEvent, ...]

@dataclass(slots=True)
class LiveSessionRecord:
    meeting_id: str
    client_session_id: str
    state: SessionState
    context: SessionContext
    generation: int
    resume_deadline: float | None
    expiry_task: asyncio.Task[None] | None
    finalization_task: asyncio.Task[None] | None

class SessionSnapshotUpdate(TypedDict, total=False):
    transcript_seq: int
    summary: SummaryEvent | None
    grounding: LiveGroundingEvent | None
    actionables: tuple[Actionable, ...]
    note_suggestions: tuple[NoteSuggestionEvent, ...]
    asr: AsrState

@dataclass(frozen=True, slots=True)
class FinalizationInput:
    meeting_id: str
    client_session_id: str
    generation: int
    notes_md: str
    notes_rev: int
    transcript: tuple[Mapping[str, Any], ...]
    last_live_summary: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class FinalizationOutcome:
    status: Literal["success", "failure", "cancelled"]
    error_kind: str | None

```

`SessionActor` has exactly these public calls:

- `start(meeting_id: str, client_session_id: str, channels: Sequence[SessionChannel], sink: EventSink) -> Awaitable[SessionSnapshot]`
- `disconnect(meeting_id: str, client_session_id: str) -> Awaitable[SessionSnapshot]`
- `stop(meeting_id: str, client_session_id: str, resume_window_s: Literal[0, 10]) -> Awaitable[SessionSnapshot | None]`; 10 returns `stopped_resumable`, while 0 returns `None` after the actor has atomically entered finalizing and started its single wrapper.
- `delete(meeting_id: str) -> Awaitable[None]`
- `snapshot(meeting_id: str, client_session_id: str) -> Awaitable[SessionSnapshot]`
- `update_snapshot(meeting_id: str, client_session_id: str, update: SessionSnapshotUpdate) -> Awaitable[SessionSnapshot]`
- `shutdown(finalize: bool) -> Awaitable[None]`

Actor constructor dependencies are exact: `create_context`, `finalize_context`, `delete_meeting`, `initial_disposition`, `monotonic`, `sleep`, `diagnostics`, and `on_loop_failure`. Constructor creation occurs inside the FastAPI lifespan and starts one actor-loop task before `/session` accepts work.

`SessionContext` is the actor-owned adapter over the existing `LiveSession`, ASR client, transcript buffer, rebindable sink, latest notepad, and bounded projection. It has exactly these calls:

- `disable_audio_and_flush() -> Awaitable[SessionSnapshotUpdate]`: reject new frames first, stop ASR, flush final transcript, and return the actual bounded state even when the flush reports a safe diagnostic.
- `rebind_transport(channels: Sequence[SessionChannel], sink: EventSink) -> Awaitable[None]`: bind the new sink and create only the ASR/channel transport; it does not recreate the retained `LiveSession`.
- `apply_snapshot(update: SessionSnapshotUpdate) -> None`: validate/merge bounded projection data and never accept audio or credentials.
- `materialize_snapshot(disposition: SessionDisposition, resume_deadline: float | None) -> SessionSnapshot`: return an immutable copy for the wire mapper.
- `freeze_finalization(generation: int) -> FinalizationInput`: capture durable notes/transcript/summary input once before starting finalization.
- `close_without_output() -> Awaitable[None]`: cooperatively stop ASR/live helpers without generating a meeting record.

```ts
export type KnowledgeCoordinate = "atlas" | "chronicle" | "inquiry"
export type ControlSection = "sources" | "profile" | "privacy" | "appearance" | "shortcuts" | "diagnostics"
export type CapturePhase =
  | "idle" | "preparing" | "live_full" | "live_mic_only" | "reconnecting"
  | "stopped_resumable" | "ending" | "finalizing" | "done"
  | "finalization_error" | "failed"

export type SessionIdentity = {
  meetingId: string
  clientSessionId: string
  channels: Array<{ id: number; label: string }>
}

export type SessionStartWire = {
  type: "start"
  meeting_id: string
  client_session_id: string
  channels: Array<{ id: number; label: string }>
}

export type SessionStopWire = {
  type: "stop"
  meeting_id: string
  client_session_id: string
  resume_window_s: 10 | 0
}

export type SessionNotepadWire = {
  type: "notepad"
  meeting_id: string
  client_session_id: string
  text_md: string
}

export type SessionSnapshotWire = {
  type: "session_snapshot"
  meeting_id: string
  client_session_id: string
  disposition: "new" | "resumed" | "state_reset" | "stopped_resumable"
  resume_deadline: number | null
  asr: "disconnected" | "connecting" | "connected" | "error"
  channels: Array<{ id: number; label: string }>
  transcript_seq: number
  summary: SummaryEventWire | null
  grounding: LiveGroundingEventWire | null
  actionables: ActionableWire[]
  note_suggestions: NoteSuggestionEventWire[]
}

export type SessionErrorWire = {
  type: "session_error"
  meeting_id: string | null
  client_session_id: string | null
  code: "meeting_session_conflict" | "meeting_not_resumable" | "invalid_session_control"
  detail: string
}

export interface MeetingDocumentController {
  replaceFromEditor(notesMd: string): void
  insertUserText(command: UserNoteCommand): Promise<Meeting>
  decideProposal(proposalId: string, decision: "accept" | "reject"): Promise<Meeting>
  supersedeProposal(proposalId: string, placement: ProposalPlacement): Promise<MeetingNoteProposal>
  undo(commandId: string): Promise<Meeting>
  flush(): Promise<void>
  resolveConflict(choice: "mine" | "server" | "copy_both"): Promise<void>
}

export interface SidecarGateway {
  start(): void
  stop(): void
  readonly session: SessionConnection
  readonly updates: UpdatesConnection
  readonly chat: ChatConnection
  subscribe(listener: (event: GatewayEvent) => void): () => void
  getEpoch(protocol: "session" | "updates" | "chat"): number
}

export type CapabilityReport = {
  platform: NodeJS.Platform
  microphone: "unknown" | "granted" | "denied" | "restricted"
  systemAudio: "unsupported" | "unknown" | "granted" | "denied" | "error"
  callDetection: "supported" | "unsupported" | "error"
  contentProtection: "requested_limited" | "requested" | "unavailable"
  appVersion: string
}

export type RecoveryDraft = {
  meetingId: string
  baseNotesRev: number
  contentHash: string
  notesMd: string
  updatedAt: number
}

export type CaptureRequest = {
  microphoneDeviceId: string | null
  includeSystemAudio: boolean
}

export type CaptureResult = {
  microphone: "started" | "denied" | "restricted" | "no_device" | "error"
  systemAudio: "started" | "not_requested" | "unsupported" | "denied" | "error"
}

export type DesktopEvent =
  | { type: "callState"; state: "detected" | "ask" | "countdown" | "paused" | "starting" | "error" | "dismissed"; secondsRemaining: number | null }
  | { type: "contextSuggestion"; state: "queued" | "visible" | "opened" | "dismissed" | "timed_out"; suggestionId: string; title: string | null; snippet: string | null }
  | { type: "systemAudioChunk"; pcm: ArrayBuffer }
  | { type: "privacyChanged"; curtain: boolean; protection: CapabilityReport["contentProtection"] }
  | { type: "windowIntent"; intent: "show" | "start_meeting" | "open_live_meeting" | "stop_meeting" | "new_inquiry" }

export interface DesktopBridge {
  capabilities(): Promise<CapabilityReport>
  requestPermission(kind: "microphone" | "systemAudio"): Promise<CapabilityReport>
  openSystemSettings(kind: "microphone" | "systemAudio" | "fullDiskAccess"): Promise<void>
  setPrivacyCurtain(enabled: boolean): Promise<void>
  setCompactLive(options: { enabled: boolean; alwaysOnTop?: boolean }): Promise<void>
  openExternal(url: string): Promise<void>
  chooseDirectory(): Promise<string | null>
  recoveryDraft: {
    read(meetingId: string): Promise<RecoveryDraft | null>
    write(draft: RecoveryDraft): Promise<void>
    delete(meetingId: string): Promise<void>
  }
  liveSession: {
    read(): Promise<SessionIdentity | null>
    write(identity: SessionIdentity): Promise<void>
    delete(): Promise<void>
  }
  capture: {
    start(input: CaptureRequest): Promise<CaptureResult>
    stop(): Promise<void>
  }
  events: {
    subscribe(listener: (event: DesktopEvent) => void): () => void
  }
}
```

Wire values remain snake_case; normalized renderer state is camelCase. `MeetingNoteProposal`, `ProposalSource`, `UserNoteCommand`, and note-command REST shapes are the approved definitions already implemented by Tasks 2–4 and must not be renamed.

---

### Task 1: Live-session and desktop foundation

**Runnable journey:** Start a durable meeting from the tray or nonactivating call prompt, capture mic/system audio without raising a hidden main window, lose and resume the session honestly, Stop immediately with a 10-second Resume option, and finalize at most once.

**Files:**

- Delete: `core/taurine_core/live/session_registry.py`
- Delete: `core/tests/test_session_registry.py`
- Create: `core/taurine_core/live/session_actor.py`
- Create: `core/tests/test_session_actor.py`
- Create: `core/tests/test_meeting_session_resume.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/tests/test_server_ws.py`
- Modify: `core/tests/test_live_ingest.py`
- Modify: `core/tests/test_final_summary.py`
- Create: `app/src/main/windows/window-state.ts`
- Create: `app/src/main/windows/main-window.ts`
- Create: `app/src/main/windows/window-privacy.ts`
- Create: `app/src/main/windows/call-prompt-window.ts`
- Create: `app/src/main/windows/context-suggestion-window.ts`
- Create: `app/src/main/capture/system-audio.ts`
- Create: `app/src/main/capture/micwatch.ts`
- Create: `app/src/main/permissions/capabilities.ts`
- Create: `app/src/main/permissions/system-settings.ts`
- Create: `app/src/main/recovery/drafts.ts`
- Create: `app/src/main/ipc.ts`
- Rewrite: `app/src/main/index.ts`
- Rewrite: `app/src/main/tray.ts`
- Create: `app/src/preload/contracts.ts`
- Rewrite: `app/src/preload/index.ts`
- Modify: `app/src/preload/index.d.ts`
- Create: `app/src/main/ipc.test.ts`
- Create: `app/src/main/windows/window-state.test.ts`
- Create: `app/src/main/windows/utility-windows.test.ts`
- Create: `app/src/main/tray.test.ts`
- Create: `app/src/main/recovery/drafts.test.ts`
- Create: `app/src/renderer/src/services/sidecar/contracts.ts`
- Create: `app/src/renderer/src/services/sidecar/contracts.test.ts`
- Create: `app/src/renderer/src/services/sidecar/updates-connection.ts`
- Create: `app/src/renderer/src/services/sidecar/session-connection.ts`
- Create: `app/src/renderer/src/services/sidecar/chat-connection.ts`
- Create: `app/src/renderer/src/services/sidecar/gateway.ts`
- Create: `app/src/renderer/src/services/sidecar/gateway.test.ts`
- Create: `app/src/renderer/src/services/sidecar/reconciliation.ts`
- Create: `app/src/renderer/src/services/sidecar/rest-api.ts`
- Create: `app/src/renderer/src/services/desktop/bridge.ts`
- Create: `app/src/renderer/src/services/desktop/capabilities.ts`
- Create: `app/src/renderer/src/utility/call-prompt-main.tsx`
- Create: `app/src/renderer/src/utility/context-suggestion-main.tsx`
- Create: `app/src/renderer/src/utility/utility.css`
- Create: `app/src/renderer/src/utility/utility.test.tsx`
- Modify: `app/src/renderer/notif.html`
- Modify: `app/src/renderer/suggest.html`
- Modify: `app/electron.vite.config.ts`

**Interfaces:** Consumes approved meeting/proposal/finalization services and the existing three protocols. Produces `SessionActor`, the exact session wire types above, one `SidecarGateway`, frozen `window.desktop`, three native renderer entries, and no socket-owned lifecycle task.

#### RED checkpoint

- [ ] **Replace the registry import in the unit-test file name only.** Create `core/tests/test_session_actor.py` importing `SessionActor`, `SessionChannel`, `SessionConflict`, and `SessionNotResumable` from `taurine_core.live.session_actor`.

- [ ] **Add the actor creation test.**

  ```python
  @pytest.mark.asyncio
  async def test_new_start_creates_one_active_record(actor, sink):
      snapshot = await actor.start("m1", "s1", [SessionChannel(0, "Microphone")], sink)
      assert snapshot.disposition == "new"
      assert snapshot.meeting_id == "m1"
      assert await actor.snapshot("m1", "s1") == snapshot
  ```

- [ ] **Add the 59.9/60-second disconnect cases using the fake monotonic scheduler.** Assert the same context identity resumes at 59.9 and exactly one immutable finalization input is observed at 60.

- [ ] **Add the 9.9/10-second Stop cases.** Assert `stopped_resumable` at Stop, `resumed` at 9.9, and one finalization at 10.

- [ ] **Add Stop-zero and repeated-Stop cases.** Assert Stop-zero returns `None` after exactly one finalizer begins; repeated resumable Stop never extends the deadline or creates another expiry/finalizer.

- [ ] **Add stale `_Expire` and `_Finalized` cases.** Queue captured generations directly through the actor test seam and assert a newer record remains.

- [ ] **Add the Knowledge Atlas conflicting-token case.** Assert `SessionConflict` for a different token against active and suspended records.

- [ ] **Add delete and shutdown cases.** Assert delete closes without finalization output and shutdown with both `finalize=False` and `finalize=True` leaves zero actor-owned expiry/finalization tasks.

- [ ] **Add public-caller cancellation cases.** Cancel once before semaphore admission and once after command enqueue; assert the first command never appears and the second transition completes without corrupting the table.

- [ ] **Add finalizer-failure and process-restart cases.** Assert the ephemeral record is removed, durable notes/transcript fixtures remain, and a new actor returns `state_reset` through `initial_disposition` for a durable row left live.

- [ ] **Add socket ordering tests in `core/tests/test_meeting_session_resume.py`.**

  ```python
  @pytest.mark.asyncio
  async def test_resumed_snapshot_precedes_audio_acceptance(live_socket):
      await live_socket.start("m1", "s1")
      await live_socket.disconnect()
      resumed = await live_socket.reconnect("m1", "s1")
      assert resumed[0]["type"] == "session_snapshot"
      assert resumed[0]["disposition"] == "resumed"
      assert resumed[1] == {"type": "status", "asr": "connected"}
  ```

- [ ] **Add the post-Stop binary rejection test.** Assert `invalid_session_control` and a zero-length audio replay/buffer fixture.

- [ ] **Add pre-suspension flush-failure coverage.** Assert the dead sink is unbound, state is suspended, safe diagnostics contain only meeting ID/error kind, and persisted transcript/notes remain.

- [ ] **Add actor-loop failure coverage.** Force the command handler to raise; assert queued and future public requests receive the same safe unavailable error and application shutdown closes remaining contexts.

- [ ] **Create Electron boundary tests.** Assert exact bridge channels, URL/settings allowlists, geometry clamp, safeStorage-only recovery, and no raw `ipcRenderer`, shell, path, URL, or generic invoke capability.

- [ ] **Create native-window and tray tests.** Assert 372×136/372×188 nonactivating utilities, queue cap three, 12-second suggestion expiry, one-prompt-per-call/five-second rearm, Ask default, opt-in 10-second auto-start, live tray labels, and no `show()` call when capture begins from a hidden window.

- [ ] **Create minimal utility-renderer tests.** Assert the call renderer exposes Start now/Mic only/Dismiss without activating the main window, the context renderer exposes Open/Dismiss, curtain payloads contain no title/snippet, and neither entry imports Redux, gateway, meeting, transcript, editor, or connector modules.

- [ ] **Create gateway tests.** Assert one socket per protocol, epoch rejection, no audio/chat replay, matching-token session reconciliation, coalesced notepad, full six-domain refetch after updates reconnect, and content-free logs.

- [ ] **Run the combined Slice 1 RED checkpoint.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/core"
  uv run pytest tests/test_session_actor.py tests/test_meeting_session_resume.py tests/test_server_ws.py -q
  cd "$WORKTREE_ROOT/app"
  npx vitest run src/main/ipc.test.ts src/main/windows/window-state.test.ts src/main/windows/utility-windows.test.ts src/main/tray.test.ts src/main/recovery/drafts.test.ts src/renderer/src/services/sidecar src/renderer/src/utility
  ```

  Expected: FAIL because `session_actor.py`, bounded desktop modules, and renderer gateway modules do not exist. Save the output as the slice's one RED review artifact.

#### Minimal implementation

- [ ] **Delete the superseded Task 5 implementation and tests.** Remove `session_registry.py` and `test_session_registry.py`; do not preserve aliases, detached-entry helpers, callback handoffs, task registries, retry loops, or cancellation shielding.

- [ ] **Define actor record, snapshots, typed public commands, `_Expire`, and `_Finalized`.** Use one record key per `meeting_id`, a monotonically increasing generation, and no client-token index or retired-token table.

- [ ] **Implement bounded submission.**

  ```python
  async def _submit(self, command: PublicCommand[T]) -> T:
      if self._failure is not None or not self._accepting:
          raise SessionServiceUnavailable()
      await self._admission.acquire()
      try:
          self._queue.put_nowait(command)
          return await asyncio.shield(command.response)
      finally:
          self._admission.release()
  ```

  The semaphore size is exactly 256. Caller cancellation before admission submits nothing; cancellation after `put_nowait` leaves the command to complete.

- [ ] **Implement the one actor loop.** Dispatch one command at a time; set a response only when its future is pending; on unexpected loop failure store one sanitized error, stop admission, fail queued public futures, invoke `on_loop_failure`, and let lifespan shutdown close contexts.

- [ ] **Implement missing→active Start.** Ask `initial_disposition(meeting_id)` for `new` versus `state_reset`, create one context/sink, store the record, and return a bounded snapshot before audio can be accepted.

- [ ] **Implement matching suspended Start.** Cancel and await the concrete expiry task, clear it, rebind the sink, recreate only ASR/channel transport, set active with a new generation, and return `resumed` before status connected.

- [ ] **Implement conflicting Start.** Reject any different token while a record exists with `SessionConflict`; reject same-token active duplicate as a conflict rather than creating a second context.

- [ ] **Implement disconnect suspension.** Disable audio acceptance, attempt ASR/transcript flush, always unbind the dead sink, record only a safe flush diagnostic on failure, set a 60-second deadline, and create one expiry task.

- [ ] **Implement Stop suspension.** Reuse the disconnect transition with 10 seconds; on repeated Stop choose `min(existing_deadline, now + 10)` and do not create a second timer.

- [ ] **Implement Stop-zero finalization claim.** Set `finalizing` and capture an immutable context snapshot before spawning exactly one finalization wrapper.

- [ ] **Implement expiry completion without callback ownership.**

  ```python
  async def expire_after(meeting_id: str, generation: int, delay: float) -> None:
      await self._sleep(delay)
      self._queue.put_nowait(_Expire(meeting_id, generation))
  ```

  `_Expire` first awaits the now-returning recorded expiry task, clears its field, checks generation/state/deadline, then claims finalization.

- [ ] **Implement finalization completion without successor tasks.**

  ```python
  async def finalize(record: FinalizationInput) -> None:
      outcome: FinalizationOutcome
      try:
          await self._finalize_context(record)
          outcome = FinalizationOutcome("success", None)
      except asyncio.CancelledError:
          outcome = FinalizationOutcome("cancelled", None)
          raise
      except Exception as exc:
          outcome = FinalizationOutcome("failure", type(exc).__name__)
      finally:
          self._queue.put_nowait(_Finalized(record.meeting_id, record.generation, outcome))
  ```

  `_Finalized` awaits the recorded wrapper, validates generation, reports failure safely, closes the context, and removes the record. A stale completion changes nothing.

- [ ] **Implement typed snapshot updates.** Merge only transcript sequence, latest summary/grounding revisions, bounded actionables/suggestions, channels, and ASR state. Enforce existing count/byte/depth bounds and reject audio, credentials, traces, and unbounded payloads.

- [ ] **Implement actor delete.** Remove the record first, cancel/await its expiry and finalization tasks, close without meeting output, then call the durable meeting deletion callback. Let the application deletion deadline report cooperative cancellation failure.

- [ ] **Implement actor shutdown.** Stop public admission; for `finalize=False`, cancel expiry, close, clear; for `finalize=True`, claim each nonfinalizing record once, allow `_Finalized` messages to drain, and resolve shutdown only when the table is empty.

- [ ] **Replace Runtime lifecycle fields.** Remove `active_sessions`, `meeting_finalizers`, `finalization_tasks`, `active_unbound_sessions`, lifecycle epoch maps, and socket-owned timer handles. Add `session_actor: SessionActor | None` plus one composition-owned actor-loop failure state.

- [ ] **Start and stop the actor in FastAPI lifespan.** Construct it after Runtime is ready and before `/session` accepts work; call `shutdown(finalize=True)` under the existing outer quit deadline.

- [ ] **Extend/retain Pydantic session controls.** Require `meeting_id` and `client_session_id`; Stop accepts only 10 or 0; Notepad carries the token; keep the approved `session_snapshot` and `session_error` schema.

- [ ] **Integrate Start with the actor.** Validate durable meeting status, ask the actor to start/resume, emit its snapshot first, then start ASR and emit connected. Map `SessionConflict` to `meeting_session_conflict` and missing/terminal ownership to `meeting_not_resumable`.

- [ ] **Integrate disconnect and Stop with the actor.** Flip the socket's binary-acceptance flag before awaiting either command; never let the handler own a grace timer or finalization task.

- [ ] **Route Notepad and live deltas through `update_snapshot`.** Retain stable IDs/revisions and persisted transcript authority; do not log payload text.

- [ ] **Route meeting deletion through `SessionActor.delete`.** Invoke durable `meeting_delete` only after actor-owned cooperative work stops; keep linked inquiry preservation from Task 4.

- [ ] **Implement `DesktopBridge` validators and `safeStorage` recovery.** Store only meeting ID, token, channels, base revision, content hash, Markdown, and timestamps. If encryption is unavailable, return the memory-only recovery status and write no plaintext file.

- [ ] **Implement exact main-window geometry and privacy status.** Preserve hide-on-close, dock reveal, full screen, clamped restoration, content-protection Requested/Limited/Unavailable reporting, compact bounds restoration, and opt-in always-on-top.

- [ ] **Move system-audio and micwatch behavior behind capture services.** Preserve helper protocols, macOS 14.2 support check, 1.5-second debounce, one prompt per call, and five-second idle rearm; remove imports from old module paths only after tests pass.

- [ ] **Implement nonactivating utility windows.** Use sandbox/context isolation/no Node integration, sanitize strings, prevent navigation, clamp to active work area, redact suggestion title/snippet under curtain, and never connect them to sidecar.

- [ ] **Implement dynamic tray behavior.** Produce Show Knowledge Atlas, Start/Open/Stop meeting by phase, New inquiry, curtain state, informational protection state, and Quit. Start from tray does not reveal the main window.

- [ ] **Implement minimal stateless utility renderers and safe CSS.** Bind only the typed sanitized preload event/action surface, render call Start now/Mic only/Dismiss and context Open/Dismiss, replace title/snippet with `Relevant context available` under curtain, use no private store, and switch `notif.html`/`suggest.html` to these new entry modules immediately.

- [ ] **Implement exhaustive frontend wire parsing.** Parse every existing `/updates`, `/session`, and `/chat` discriminant plus snapshots/errors. Unknown names log only protocol/discriminant/epoch and request bounded reconciliation.

- [ ] **Implement connection classes and `SidecarGateway`.** Open updates/session after readiness, chat lazily, use capped jittered backoff/visibility wakeups, reject stale epochs, and never replay binary audio or uncertain chat sends.

- [ ] **Implement reconciliation.** Updates reconnect refetches meetings, chats, connectors, profile, graph metadata/status, and runtime summary. Session reconnect presents the stored token, consumes snapshot before audio, and chat reconnect refetches active histories without resending.

- [ ] **Configure exactly three renderer entries.** Keep one stateful main entry plus stateless call-prompt and context-suggestion entries; named lazy chunks remain editor, atlas, and markdown.

#### GREEN checkpoint

- [ ] **Run the complete actor matrix repeatedly.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/core"
  uv run pytest tests/test_session_actor.py tests/test_meeting_session_resume.py tests/test_server_ws.py tests/test_live_ingest.py tests/test_final_summary.py -q
  ```

  Expected: PASS. The 14 reset cases pass, race-sensitive cases repeat at least 20 times, finalization occurs at most once, and pytest emits no pending-task warning.

- [ ] **Run desktop/gateway and build gates.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npm test -- src/main src/renderer/src/services src/renderer/src/utility
  npm run typecheck
  npm run build
  ```

  Expected: PASS with one gateway owner per protocol, exact native geometry/actions, bounded IPC, no utility socket, and a production build with three renderer entries.

- [ ] **Prove the superseded ownership design is absent.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT"
  test ! -e core/taurine_core/live/session_registry.py
  test ! -e core/tests/test_session_registry.py
  rg -n 'SessionRegistry|_detached_entries|finalizer_handoff|finalizer_driver|_schedule_finalization_cleanup|meeting_finalizers|active_sessions' core/taurine_core core/tests
  ```

  Expected: both file checks exit 0 and `rg` exits 1 with no matches. Actor expiry/finalization completion is queue-message-only.

- [ ] **Run the Slice 1 inspectable journey.** Start from tray while the main window is hidden, confirm it stays hidden, disconnect/reconnect with the same token, observe snapshot before resumed audio, Stop and Resume inside 10 seconds, Stop again and Finalize now, then open the durable dossier. Expected: one meeting, one finalization, persisted notes/transcript intact, and no audio from either gap.

- [ ] **Freeze Slice 1 after GREEN.**

  ```bash
  cd "$WORKTREE_ROOT"
  git diff --check
  git status --short
  git add -A core/taurine_core/live/session_registry.py core/tests/test_session_registry.py
  git add core/taurine_core/live/session_actor.py core/taurine_core/schemas.py core/taurine_core/server.py core/tests/test_session_actor.py core/tests/test_meeting_session_resume.py core/tests/test_server_ws.py core/tests/test_live_ingest.py core/tests/test_final_summary.py app/src/main app/src/preload app/src/renderer/src/services app/src/renderer/src/utility app/src/renderer/notif.html app/src/renderer/suggest.html app/electron.vite.config.ts
  git commit -m "feat: establish atlas live-session and desktop foundation"
  ```

  Expected: commit succeeds and becomes the immutable Slice 1 review target.

- [ ] **Stop and request independent Slice 1 review.** Provide the commit SHA, RED/GREEN output, inspectable-journey evidence, and actor absence scan. Expected: no Slice 2 path changes until review returns clean; any blocking finding is fixed in Slice 1 and re-frozen first.

---

### Task 2: Atlas shell

**Runnable journey:** Launch immediately into the paper-and-ink shell, observe real readiness, use keyboard commands to move among Atlas/Chronicle/Inquiry while preserving context, search/select/open/pin objects, and inspect a deterministic territory/list baseline without a pointer.

**Files:**

- Modify: `app/package.json`
- Modify: `app/package-lock.json`
- Create: `app/src/renderer/src/app/App.tsx`
- Create: `app/src/renderer/src/app/AppShell.tsx`
- Create: `app/src/renderer/src/app/store.ts`
- Create: `app/src/renderer/src/app/listeners.ts`
- Create: `app/src/renderer/src/app/commands.ts`
- Create: `app/src/renderer/src/app/hydration.ts`
- Create: `app/src/renderer/src/features/runtime/RuntimeGate.tsx`
- Create: `app/src/renderer/src/features/runtime/FirstRunChecklist.tsx`
- Create: `app/src/renderer/src/features/runtime/OperationLedger.tsx`
- Create: `app/src/renderer/src/features/runtime/state.ts`
- Create: `app/src/renderer/src/features/runtime/selectors.ts`
- Create: `app/src/renderer/src/features/workspace/CoordinateRail.tsx`
- Create: `app/src/renderer/src/features/workspace/CommandDeck.tsx`
- Create: `app/src/renderer/src/features/workspace/CommandCenter.tsx`
- Create: `app/src/renderer/src/features/workspace/QuickOpen.tsx`
- Create: `app/src/renderer/src/features/workspace/ContextShelf.tsx`
- Create: `app/src/renderer/src/features/workspace/ControlWorkbench.tsx`
- Create: `app/src/renderer/src/features/workspace/state.ts`
- Create: `app/src/renderer/src/features/workspace/selectors.ts`
- Create: `app/src/renderer/src/features/atlas/AtlasWorkbench.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasCanvas.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasOutline.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasIndex.tsx`
- Create: `app/src/renderer/src/features/atlas/atlas-worker.ts`
- Create: `app/src/renderer/src/features/atlas/layout.ts`
- Create: `app/src/renderer/src/features/atlas/render.ts`
- Create: `app/src/renderer/src/features/atlas/storage.ts`
- Create: `app/src/renderer/src/features/atlas/contracts.ts`
- Create: `app/src/renderer/src/features/atlas/state.ts`
- Create: `app/src/renderer/src/features/atlas/selectors.ts`
- Create: `app/src/renderer/src/components/primitives/`
- Create: `app/src/renderer/src/components/layout/`
- Create: `app/src/renderer/src/components/knowledge/ObjectRow.tsx`
- Create: `app/src/renderer/src/components/knowledge/SourceMark.tsx`
- Create: `app/src/renderer/src/i18n/en.ts`
- Create: `app/src/renderer/src/i18n/format.ts`
- Create: `app/src/renderer/src/i18n/pseudo.ts`
- Create: `app/src/renderer/src/i18n/useMessage.ts`
- Create: `app/src/renderer/src/styles/tokens.css`
- Create: `app/src/renderer/src/styles/reset.css`
- Create: `app/src/renderer/src/styles/typography.css`
- Create: `app/src/renderer/src/styles/shell.css`
- Create: `app/src/renderer/src/styles/primitives.css`
- Create: `app/src/renderer/src/styles/print.css`
- Create: `app/src/renderer/src/assets/sources/`
- Create: colocated `*.test.ts` and `*.test.tsx` files for the modules above

**Interfaces:** Consumes Slice 1 gateway/desktop services and Task 1 harness. Produces normalized `RootState`, one gateway subscription, exact command IDs, the responsive shell, exception-only operation ledger, max-three context shelf, deterministic Atlas snapshot/Outline baseline, and catalogued visible copy.

#### RED checkpoint

- [ ] **Add store ownership tests.** Assert Control does not overwrite the active coordinate, pins cap at three, removed restored pins become notices, production DevTools are off, and StrictMode mounts one gateway subscription.

- [ ] **Add hydration tests.** Assert `/startup` backoff at 250 ms, 500 ms, 1 s, then 2 s; 503/1013 returns to Starting; hidden polling pauses only after 30 seconds; ready opens sockets then hydrates six domain summaries.

- [ ] **Add command tests.** Assert `⌘1/2/3`, `⌘,`, `⌘K`, `⌘P`, `⌘[`, `⌘]`, `⌘B`, `⌘\`, `F6`, and `Shift+F6`; editor/modal/menu scopes win over global commands.

- [ ] **Add shell semantic/breakpoint tests.** Assert exactly three coordinate controls plus separate Control, region landmarks, ≥1440 docked inspector, 1180–1439 inspector sheet, 960–1179 mutually exclusive index/inspector sheets, and the 200%-zoom compact-pane policy.

- [ ] **Add primitive tests.** Assert labelled fields, menu roving focus, dialog/sheet focus trap and restoration, dismissible tooltip, truthful determinate progress, reduced-motion skeleton, 12 px divider hit region, and 24/40 px target rules.

- [ ] **Add token/catalog tests.** Assert exact light/dark/source tokens, bundled IBM Plex weights, no gradients/emoji control glyphs, 35% pseudo expansion, `Intl` formatting, RTL logical properties, and no visible JSX string outside the catalog.

- [ ] **Add deterministic Atlas tests.** Assert stable hash positions, prior-position reuse, new-node community-centroid placement, sanitized coordinate persistence, worker epoch rejection, reduced-motion snap, and four exact LOD thresholds.

- [ ] **Add accessible baseline tests.** Assert Canvas is `aria-hidden`, Outline is the screen-reader default, global tree selection is not activation, Enter opens, Space previews, Shift+Enter pins, and hidden tail nodes remain searchable.

- [ ] **Run the combined Slice 2 RED checkpoint.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npx vitest run src/renderer/src/app src/renderer/src/features/runtime src/renderer/src/features/workspace src/renderer/src/features/atlas src/renderer/src/components src/renderer/src/i18n
  ```

  Expected: FAIL because store, shell, design system, and Atlas baseline modules do not exist. Save this output as the slice's one RED artifact.

#### Minimal implementation

- [ ] **Install/verify the exact Slice 2 renderer dependencies and lock them.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npm install @reduxjs/toolkit react-redux lucide-react @tanstack/react-virtual d3-force d3-polygon @fontsource/ibm-plex-sans @fontsource/ibm-plex-mono
  npm install --save-dev @types/d3-force @types/d3-polygon
  ```

  Expected: `package.json` contains those direct runtime/dev dependencies at the resolved versions and `package-lock.json` matches. On the current branch this may be a verified no-op because approved Task 1 commit `0d4827d` already introduced the same packages; do not remove or replace them.

- [ ] **Create normalized slices.** Keep server entities, operational state, and workspace state separate; store no socket, DOM node, editor instance, promise, or content-bearing Error.

- [ ] **Configure the store and one listener registration.**

  ```ts
  export const createAppStore = (services: AppServices) => configureStore({
    reducer: rootReducer,
    middleware: (defaults) => defaults({ serializableCheck: true })
      .prepend(listenerMiddleware.middleware)
      .concat(sidecarApi.middleware),
    devTools: import.meta.env.DEV,
  })
  ```

- [ ] **Implement readiness polling and hydration ordering.** Keep shell/preferences/curtain available during startup and fatal error; lazy-load graph only when Atlas or a restored graph pin needs it.

- [ ] **Implement restored-context validation.** Restore coordinate, Control section, history, pins, widths, density, and theme only when identifiers still exist; represent invalid pins as removed-item notices.

- [ ] **Implement the UI tokens exactly.** Use the full §9 hex/OKLCH values, source colors plus glyph/label, spacing `2,4,6,8,12,16,20,24,32,40,48,64`, radii `4/6/3/0`, motion `80/140/220`, and no gradients.

  ```css
  :root {
    --paper: #f3f1eb; --surface: #fbfaf7; --raised: #ffffff;
    --ink: #1e2732; --muted: #566170; --subtle: #788391;
    --rule: #c8cdd3; --rule-strong: #8c96a2;
    --accent: #315bcb; --accent-hover: #2548a5; --accent-soft: #e1e8fc;
    --focus: #1463e6; --positive: #19734b; --warning: #956018;
    --negative: #b43b3b; --live: #a8325a; --generated: #6555a5;
  }
  [data-theme="dark"] {
    --paper: #151a20; --surface: #1d232b; --raised: #252d37;
    --ink: #f1f3f5; --muted: #aab2bd; --rule: #414b57;
    --accent: #8eaaff; --accent-soft: #283657;
  }
  ```

  Source marks are Apple Notes `#B46B00`, Obsidian `#7352B8`, iMessage `#187854`, Gmail `#B43B3B`, Outlook `#23699C`, Google Calendar `#2F7D67`, Google Docs `#315BCB`, Folders `#66717E`, Claude `#8B5A2B`, Codex `#34404C`, and Taurine meetings `#9B4763`; every mark also has its specified glyph/shape and text label.

- [ ] **Implement typography and local assets.** Bundle IBM Plex Sans 400/500/600 and Mono 400/500; forbid remote fonts; normalize the eleven connector SVGs to 20 px.

  Use display 28/34 weight 500, title 20/26 weight 600, section 15/20 weight 600, UI body 14/21, editor 16/25, control 13/18 weight 500, metadata 12/16, and micro 11/14. The micro style is evidence IDs/annotations only, never long prose.

- [ ] **Implement one catalog/formatter path.** Use namespaced complete messages, `Intl.DateTimeFormat`, `Intl.RelativeTimeFormat`, `Intl.NumberFormat`, plural rules, platform key labels, and LTR isolation for evidence IDs/hashes.

- [ ] **Implement primitive controls.** Match §10.1 measurements and semantics; `Progress` throws for determinate mode without finite nonnegative `done` and positive `total`.

  Exact geometry: Button 34/40 px with 0×12 padding; IconButton 32 px visual inside 40×40 hit; fields/selects 34/40; SearchField 36; checkbox 16 inside 32/40 row; switch 32×18 inside 40; menu rows 32/40 with 320 px max height; tooltip delay 400 ms for pointer and immediate for keyboard; focus ring 2 px plus 2 px offset. Toasts are transient success only, while errors remain at the point of action.

- [ ] **Implement SplitPane and focus zones.** Clamp index 230–360 and inspector 320–480, resize by 16 px or 48 px with Shift, Enter/double-click reset, and retain last focus in each F6 zone.

- [ ] **Implement command registry as the sole activation path.**

  ```ts
  export type Command = {
    id: string
    labelKey: MessageKey
    shortcut?: Shortcut
    isAvailable(state: RootState): boolean
    execute(context: CommandContext): void
  }
  ```

  Buttons, menus, tray intents, and shortcuts resolve the same command ID.

- [ ] **Implement the 48 px command deck and 52 px rail.** Show Atlas/Chronicle/Inquiry, separate bottom Control, back/forward, rectangular locator, capture/privacy state, inspector toggle, and one scope-specific primary action.

- [ ] **Implement Index, Workbench, Inspector, Shelf, and Operations regions.** Virtualize after 100 rows, cap pins at three, provide compare/Ask these entry points, and remove the 28 px ledger from layout when healthy/idle unless pinned.

- [ ] **Implement command center and Quick Open.** Group objects/commands/recent contexts for `⌘K`; make `⌘P` object-only; use arrow selection, Enter open, Shift+Enter pin, and `⌘Enter` open-and-close.

- [ ] **Implement first-run checklist.** Store only UI schema version/preferences; report runtime credential status, request mic/system permissions just in time, expose recommended sources, Off/Ask/Auto-start, and privacy limitation/shortcut.

- [ ] **Implement Atlas contracts and immutable snapshot selectors.** Use existing `GET /graph`, `GET /graph/local`, search, facts, notes, graph status/version; never call edge reason from projected edges.

- [ ] **Implement deterministic layout.** Seed from community ID, stable note-ID hash, degree, and date band; run d3-force/hulls in the worker; publish settled snapshots no faster than animation frame.

- [ ] **Implement sanitized layout storage.** Persist build ID, note ID, x/y, and user-pinned flag only; store no title, snippet, passage, source text, or private content.

- [ ] **Implement territory-first Canvas rendering.** Use flat 6–10% tints, horizontal labels, shape/text/pattern plus color, exact LOD bands, no perpetual force motion, and 140 ms removal fade unless reduced motion.

- [ ] **Implement baseline Atlas Index and Outline.** Global Outline is territory→notes tree with source/date/degree/match. Search selection pans/previews; Enter opens local depth one; all essential actions remain outside Canvas.

- [ ] **Keep `App.tsx` composition-only.** It creates providers and renders `AppShell`; no socket constructor, fetch, capture listener, graph reducer, chat reducer, or editor state appears there.

#### GREEN checkpoint

- [ ] **Run the Slice 2 unit/type/build gate.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npm test -- src/renderer/src/app src/renderer/src/features/runtime src/renderer/src/features/workspace src/renderer/src/features/atlas src/renderer/src/components src/renderer/src/i18n
  npm run lint
  npm run typecheck
  npm run build
  npm run check:bundle:report
  ```

  Expected: PASS for tests/lint/type/build. The report-only bundle command prints the unchanged approved limits and may still report the legacy production entry until Slice 6; it must not change the thresholds or create a waiver.

- [ ] **Run the Slice 2 inspectable journey.** Launch with a delayed fake startup, use shell/curtain while starting, reach Ready, traverse all focus zones, select/open/pin in Outline, pivot Atlas→Chronicle→Inquiry and back, and confirm selection/pins/history persist. Expected: no pointer required and the operation ledger collapses after success without moving focus.

- [ ] **Freeze Slice 2 after GREEN.**

  ```bash
  cd "$WORKTREE_ROOT"
  git diff --check
  git add app/package.json app/package-lock.json app/src/renderer/src/app app/src/renderer/src/features/runtime app/src/renderer/src/features/workspace app/src/renderer/src/features/atlas app/src/renderer/src/components app/src/renderer/src/i18n app/src/renderer/src/styles app/src/renderer/src/assets
  git commit -m "feat: deliver the knowledge atlas shell"
  ```

  Expected: commit succeeds and becomes the immutable Slice 2 review target.

- [ ] **Stop and request independent Slice 2 review.** Provide the commit SHA, RED/GREEN output, bundle report, and shell/Outline journey evidence. Expected: no Slice 3 path changes until review returns clean; blocking findings are repaired and re-frozen in Slice 2.

---

### Task 3: Meeting journey

**Runnable journey:** Create a draft, pass mic/system preflight, capture in the two-lane Chronicle Live desk, freely edit durable notes, survive reconnect or revision conflict, Stop immediately, review final recap/proposals, and open meeting-scoped Ask without any generated text mutating notes before Accept.

**Files:**

- Create: `app/src/renderer/src/features/meetings/contracts.ts`
- Create: `app/src/renderer/src/features/meetings/state.ts`
- Create: `app/src/renderer/src/features/meetings/selectors.ts`
- Create: `app/src/renderer/src/features/meetings/listeners.ts`
- Create: `app/src/renderer/src/features/meetings/capture-machine.ts`
- Create: `app/src/renderer/src/features/meetings/MeetingsIndex.tsx`
- Create: `app/src/renderer/src/features/meetings/MeetingDossier.tsx`
- Create: `app/src/renderer/src/features/meetings/LiveDesk.tsx`
- Create: `app/src/renderer/src/features/meetings/CompactLive.tsx`
- Create: `app/src/renderer/src/features/meetings/CaptureHeader.tsx`
- Create: `app/src/renderer/src/features/meetings/TranscriptLane.tsx`
- Create: `app/src/renderer/src/features/meetings/NotesDocument.tsx`
- Create: `app/src/renderer/src/features/meetings/MeetingDocumentController.ts`
- Create: `app/src/renderer/src/features/meetings/ProposalQueue.ts`
- Create: `app/src/renderer/src/features/meetings/LiveIntelligence.tsx`
- Create: `app/src/renderer/src/features/meetings/FinalRecord.tsx`
- Create: `app/src/renderer/src/features/meetings/ProposalDiff.tsx`
- Create: `app/src/renderer/src/features/meetings/editor/blocknote-adapter.ts`
- Create: `app/src/renderer/src/features/meetings/editor/provenance.ts`
- Create: `app/src/renderer/src/features/meetings/editor/markdown-source.tsx`
- Create: `app/src/renderer/src/features/meetings/MeetingDocumentController.test.ts`
- Create: `app/src/renderer/src/features/meetings/capture-machine.test.ts`
- Create: `app/src/renderer/src/features/meetings/MeetingDossier.test.tsx`
- Create: `app/src/renderer/src/features/meetings/LiveDesk.test.tsx`
- Create: `app/src/renderer/src/features/privacy/PrivacyCurtain.tsx`
- Create: `app/src/renderer/src/features/privacy/PrivacyStatus.tsx`
- Create: `app/src/renderer/src/features/privacy/state.ts`
- Create: `app/src/renderer/src/features/notifications/ContextQueue.tsx`
- Create: `app/src/renderer/src/features/notifications/state.ts`
- Modify: `app/src/renderer/src/app/listeners.ts`
- Modify: `app/src/renderer/src/app/AppShell.tsx`

**Interfaces:** Consumes Tasks 2–4 durable proposal/note-command APIs, Slice 1 actor/session/desktop services, and Slice 2 shell. Produces the exact capture finite-state machine, one document controller per mounted dossier, Chronicle, Live desk/compact mode, transcript/intelligence, proposal history, final record, meeting deletion flow, and meeting-scoped inquiry seed.

#### RED checkpoint

- [ ] **Add the capture transition table test.** Assert every legal transition in §12.3 and reject stale `meetingId`, `clientSessionId`, `attemptId`, or `connectionEpoch` events.

- [ ] **Add preflight tests.** Cover runtime/socket unavailable, mic denied/restricted/no device, system unsupported/denied/helper error, durable row create failure, same/different active meeting, saved device, and exact Start mic + system audio/Start mic only labels.

- [ ] **Add immediate Stop ordering.**

  ```ts
  it("stops capture before network stop and document flush", async () => {
    await stopMeeting({ resumeWindowS: 10 })
    expect(desktop.capture.stop.mock.invocationCallOrder[0])
      .toBeLessThan(session.sendStop.mock.invocationCallOrder[0])
    expect(session.sendStop.mock.invocationCallOrder[0])
      .toBeLessThan(controller.flush.mock.invocationCallOrder[0])
  })
  ```

- [ ] **Add reconnect tests.** Assert no binary buffering, Reconnecting copy, resumed snapshot merge before ASR connected, audio-gap divider, `state_reset` ephemeral clear/refetch/restart divider, token conflict, after-grace durable reconciliation, and no artificial summary epoch.

- [ ] **Add document-controller debounce/order tests.** Assert recovery write precedes 800 ms REST autosave, 2-second token-bearing notepad while active/suspended, stale save success cannot clear a later edit, and Stop never waits for storage.

- [ ] **Add revision conflict tests.** Assert both local/server Markdown survive 409, automatic writes pause, and Keep mine/Keep server/Copy both have exact outcomes.

- [ ] **Add sole-writer tests.** Spy on REST note PATCH and note-command calls; assert only `MeetingDocumentController` can issue them and `ProposalQueue`, socket handlers, transcript, Live intelligence, and views cannot.

- [ ] **Add proposal tests for all five origins.** Assert rolling summary, actionable, meeting chat, inline Ask, and final summary remain absent from canonical Markdown until a durable proposal exists and Accept succeeds.

- [ ] **Add proposal decision tests.** Cover accept/reject, immutable payload, stale base revision, ambiguous/missing anchor, explicit append, superseding placement, latest-operation undo, undo conflict, failed creation, retry/copy, and collapsed decision history.

- [ ] **Add transcript tests.** Cover final/interim replacement, localized You/Them/Speaker N, channel grouping, 32 px auto-follow, Jump count, Copy, Insert transcript excerpt, Ask excerpt, 300-block virtualization, and no interim persistence/citation/live spam.

- [ ] **Add Live desk breakpoint tests.** Assert Notes ≥620 plus one 300–360 lane at 1280, optional remembered three-lane only ≥1440, focus-trapped mutually exclusive sheets below 1180, and preserved lane scroll/selection/disclosure.

- [ ] **Add capture/privacy/compact tests.** Assert 44 px header, textual channels, no decorative waveform, curtain keeps elapsed/channels/Stop and removes private DOM, compact uses the same window and restores exact bounds.

- [ ] **Add final dossier tests.** Assert strata order Notes→Generated recap→Proposed additions→Actions→Transcript→Inquiry, generated title fills only empty title, indexing states, missed final event refetch, and finalization error offers no nonexistent retry.

- [ ] **Add meeting deletion tests.** Assert exact permanent consequences, actor cleanup first, linked inquiry retained, Taurine-derived note stale until clean sync, and no trash/undo claim.

- [ ] **Add meeting-scoped Ask tests.** Assert composer reads `This meeting + pinned context`, carries `meeting_id` and seeds indexed `taurine:{meetingId}` only when available, and `edit_notes` output persists a proposal rather than editing.

- [ ] **Run the combined Slice 3 RED checkpoint.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npx vitest run src/renderer/src/features/meetings src/renderer/src/features/privacy src/renderer/src/features/notifications
  ```

  Expected: FAIL because the meeting controller, capture state machine, and Chronicle surfaces do not exist. Save this as the slice's one RED artifact.

#### Minimal implementation

- [ ] **Define normalized meeting operational state.** Store active meeting ID, capture identity/phase, transcript buffers, summary/actionables/context queue, note save metadata, and proposal metadata; keep editor nodes outside Redux.

- [ ] **Implement capture reducer with identity guards.**

  ```ts
  const isCurrent = (state: CaptureState, event: CaptureEvent) =>
    event.meetingId === state.meetingId &&
    event.clientSessionId === state.clientSessionId &&
    event.attemptId === state.attemptId &&
    event.connectionEpoch === state.connectionEpoch
  ```

  `resumeDeadline` exists only in `stopped_resumable`; local expiry requests reconciliation and never overrules the server snapshot.

- [ ] **Implement draft creation idempotency.** New meeting creates one durable draft and opens its dossier; a duplicate pending intent reveals preparation instead of creating a row.

- [ ] **Implement inline preflight.** Inspect permissions without prompting, request from labelled action, require mic/device and durable row, let system audio degrade, and redirect a conflicting active meeting to the existing record.

- [ ] **Persist live-session metadata before Start.** Generate one client token per capture attempt, write meeting/token/channels through `desktop.liveSession`, then send Start; clear only after finalization/deletion.

- [ ] **Start mic before optional system audio.** Abort on mic failure; if system start fails after mic, enter `live_mic_only`, keep mic, and render remediation.

- [ ] **Implement snapshot-first reconnect.** Merge summary by revision, actionables by ID with user status precedence, grounding/evidence by stable ID/revision, suggestions by note ID, and transcript sequence monotonically; restart only preflight-approved channels after ASR connected.

- [ ] **Implement immediate Stop and grace.** Stop desktop capture, write recovery, send one Stop(10), enter stopped-resumable without moving focus, show 44 px Resume plus secondary Finalize now, and flush REST/notepad during grace.

- [ ] **Implement Resume and Finalize now.** Resume reuses meeting/token and creates no new meeting/epoch; Finalize now sends Stop(0). Navigation does not own or cancel runtime finalization.

- [ ] **Implement the controller state machine.**

  ```ts
  async function persistSnapshot(snapshot: LocalNotesSnapshot) {
    await desktop.recoveryDraft.write(snapshot.recovery)
    const meeting = await api.updateMeeting(snapshot.meetingId, {
      notes_md: snapshot.notesMd,
      base_notes_rev: snapshot.baseNotesRev,
    })
    applySaveSuccessOnlyWhenHashMatches(snapshot.contentHash, meeting)
  }
  ```

- [ ] **Implement conflict resolution without last-write-wins.** Keep both strings, suspend autosave, let Keep mine retry from server revision, Keep server replace local/controller state, and Copy both copy without mutating.

- [ ] **Implement editor adapters and source-mode parity.** Serialize Markdown deterministically, keep pending proposal content outside serialization, represent operation provenance in a 2 px generated gutter, and transfer edited spans to user-owned styling.

- [ ] **Implement user-authored atomic commands.** Typing/paste uses revision-aware save; Insert transcript excerpt uses `insert`; no generated origin may call these paths.

- [ ] **Implement `ProposalQueue`.** Hydrate durable proposals, create renderer-originated rolling/actionable/inline proposals idempotently, expose Accept only after persistence, and refetch on chat/final proposal events.

- [ ] **Implement controller decisions and undo.** Send proposal ID/decision/base revision only; use server-returned immutable text and operation; on stale/ambiguous placement create a new superseding proposal.

- [ ] **Implement Chronicle rows and dossier states.** Render draft, preparing, live full/mic-only, reconnecting, stopped, ending, summarizing, done, finalization error, deleted elsewhere, and indexing consequences with exact actions from §8.3.

- [ ] **Implement Live desk and compact layouts.** Notes remain primary; Transcript/Intelligence state persists when switched or sheeted; compact contains header, 120 px transcript, notes, and 44 px collapsed context/action queue.

- [ ] **Implement transcript behavior.** Keep interims replaceable and noncitable; group finals, virtualize after 300 blocks with overscan 10, disclose reconnect gaps, and route Insert/Ask through typed commands.

- [ ] **Implement Live intelligence.** One-open disclosure among rolling summary, actionables, context queue; `Review for notes` persists a proposal; actionable user status wins; retrieval disclosure dispatches the typed `workspace/evidenceSelected` intent consumed by Slice 4 without duplicating truth logic.

- [ ] **Implement final record and proposal history.** Generated recap is visible but not canonical notes; exact insertion diff/rationale/time/base revision and Accept/Reject/Undo are keyboard reachable; decisions remain collapsed history.

- [ ] **Implement meeting-scoped Ask entry.** Pass meeting/pins/selection into the shared inquiry scope contract used in Slice 4; until that workbench exists, render a typed shell destination rather than a second chat client.

- [ ] **Implement deletion sequencing.** Ask actor/server to stop cooperative live work, then delete durable row; preserve linked inquiry and show stale-index consequence.

#### GREEN checkpoint

- [ ] **Run the Slice 3 unit/type/build gate.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npm test -- src/renderer/src/features/meetings src/renderer/src/features/privacy src/renderer/src/features/notifications src/renderer/src/services
  npm run lint
  npm run typecheck
  npm run build
  ```

  Expected: PASS for every capture phase, immediate Stop ordering, honest resume/reset, controller conflict/durability, all proposal origins, responsive Live, transcript, final record, and deletion.

- [ ] **Run the Slice 3 inspectable journey.** Create a draft, degrade to mic-only once, start full capture, type notes, insert a transcript excerpt, force reconnect, Stop/Resume, Stop/Finalize, accept and undo a proposal, trigger a revision conflict, resolve it, then enter scoped Ask. Expected: capture truth is exact and no authored text is lost or generated text silently inserted.

- [ ] **Freeze Slice 3 after GREEN.**

  ```bash
  cd "$WORKTREE_ROOT"
  git diff --check
  git add app/src/renderer/src/features/meetings app/src/renderer/src/features/privacy app/src/renderer/src/features/notifications app/src/renderer/src/app
  git commit -m "feat: deliver the knowledge atlas meeting journey"
  ```

  Expected: commit succeeds and becomes the immutable Slice 3 review target.

- [ ] **Stop and request independent Slice 3 review.** Provide the commit SHA, RED/GREEN output, meeting journey evidence, and sole-writer scan. Expected: no Slice 4 path changes until review returns clean; blocking findings are repaired and re-frozen in Slice 3.

---

### Task 4: Inquiry and evidence

**Runnable journey:** Ask from all knowledge, the current meeting, pinned objects, or the current Atlas neighborhood; watch one streaming thread; inspect every citation/consulted/retrieved object at claim level; cancel/reconnect honestly; and navigate the same evidence back into Atlas/Chronicle.

**Files:**

- Create: `app/src/renderer/src/features/evidence/normalize.ts`
- Create: `app/src/renderer/src/features/evidence/selectors.ts`
- Create: `app/src/renderer/src/features/evidence/EvidenceInspector.tsx`
- Create: `app/src/renderer/src/features/evidence/EvidenceLedger.tsx`
- Create: `app/src/renderer/src/features/evidence/PassageView.tsx`
- Create: `app/src/renderer/src/features/evidence/FactView.tsx`
- Create: `app/src/renderer/src/features/evidence/PathView.tsx`
- Create: `app/src/renderer/src/features/evidence/normalize.test.ts`
- Create: `app/src/renderer/src/features/evidence/EvidenceInspector.test.tsx`
- Create: `app/src/renderer/src/components/knowledge/EvidenceRef.tsx`
- Create: `app/src/renderer/src/components/knowledge/DecisionLabel.tsx`
- Create: `app/src/renderer/src/components/knowledge/FactRow.tsx`
- Create: `app/src/renderer/src/components/knowledge/PathView.tsx`
- Create: `app/src/renderer/src/components/knowledge/DiffProposal.tsx`
- Create: `app/src/renderer/src/features/inquiry/contracts.ts`
- Create: `app/src/renderer/src/features/inquiry/state.ts`
- Create: `app/src/renderer/src/features/inquiry/selectors.ts`
- Create: `app/src/renderer/src/features/inquiry/listeners.ts`
- Create: `app/src/renderer/src/features/inquiry/InquiryIndex.tsx`
- Create: `app/src/renderer/src/features/inquiry/InquiryWorkbench.tsx`
- Create: `app/src/renderer/src/features/inquiry/InquiryComposer.tsx`
- Create: `app/src/renderer/src/features/inquiry/AssistantTurn.tsx`
- Create: `app/src/renderer/src/features/inquiry/ProcessTrace.tsx`
- Create: `app/src/renderer/src/features/inquiry/MessageMarkdown.tsx`
- Create: `app/src/renderer/src/features/inquiry/inquiry.test.tsx`
- Modify: `app/src/renderer/src/app/AppShell.tsx`
- Modify: `app/src/renderer/src/app/listeners.ts`
- Modify: `app/src/renderer/src/features/meetings/MeetingDossier.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasWorkbench.tsx`

**Interfaces:** Consumes the full existing `/chat` protocol, meeting scope from Slice 3, workspace pins/selection, and graph/note/fact contracts. Produces one normalized `EvidenceRecord` union, one renderer-wide inspector, complete inquiry state/event reduction, and navigation intents shared by all coordinates.

#### RED checkpoint

- [ ] **Add evidence precedence tests.** Assert an ID appears once at its highest relationship `cited > consulted > retrieved`, with lower relationships retained only in metadata.

- [ ] **Add exact trust-language tests.** Assert Cited in this answer, Consulted but not cited, Retrieved only, Evidence policy passed, Low-confidence fallback, No supported evidence, and Interrupted/unvalidated; reject Verified/fact checked copy and primary numeric confidence.

- [ ] **Add evidence-type tests.** Cover passage exact text/date/range/scores Advanced, fact direction/status/interval/confidence/source, path route/edges/fact IDs/weight, meeting/transcript speaker/time/sequence, note navigation/passages, missing and unavailable records.

- [ ] **Add temporal/fallback/interruption tests.** Assert inactive current-tense warning, prior-context label, fallback with facts/paths withheld, abstained empty support, and interrupted/cancelled partial with no support even when raw text contains citation-looking tokens.

- [ ] **Add inquiry scope tests.** Assert All, Current meeting, Pinned context, and Current Atlas selection/neighborhood map to `meeting_id`/`seed_note_ids` and render Start from these objects rather than Search only these.

- [ ] **Add first-send and acknowledgement tests.** Assert an empty local thread is not created server-side until send, optimistic user text says Sending, `chat_ack` replaces temporary IDs, and authoritative history prevents uncertain retry duplication.

- [ ] **Add interleaved streaming tests.** Assert retrieval, thinking, activity, and answer deltas may interleave; process events replace by stable index; answer deltas batch 30–50 ms; `chat_done.message` replaces buffers.

- [ ] **Add turn outcome tests.** Cover accepted, fallback, abstained, cancelled, interrupted, busy, timeout, LLM error, stripped claims, usage/build metadata, original/rewritten query, and calibration identity.

- [ ] **Add cancel/reconnect/regenerate tests.** Assert one `chat_cancel`, Stopping until terminal event, no auto-resend, refetch active history, temporary Replacing snapshot for regenerate, and rollback on failure.

- [ ] **Add management/background tests.** Assert rename rollback, destructive delete consequences, meeting link cleanup, background streaming coordinate count, composer preservation, and coordinate changes do not cancel turns.

- [ ] **Add navigation tests.** Assert citation selection updates inspector without workbench navigation; note/meeting opens Atlas/Chronicle; View evidence in Atlas sets evidence lens; Open in source appears only for supported adapter.

- [ ] **Run the combined Slice 4 RED checkpoint.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npx vitest run src/renderer/src/features/evidence src/renderer/src/features/inquiry src/renderer/src/components/knowledge
  ```

  Expected: FAIL because shared evidence normalization and Inquiry workbench modules do not exist. Save this output as the slice's one RED artifact.

#### Minimal implementation

- [ ] **Define normalized evidence unions.** Separate record kind, relationship, policy outcome, temporal state, availability, and prior-context/low-confidence flags; never infer one from another.

- [ ] **Implement relationship normalization.**

  ```ts
  const rank = { retrieved: 0, consulted: 1, cited: 2 } as const
  export function mergeEvidence(rows: EvidenceRecord[]): EvidenceRecord[] {
    const byId = new Map<string, EvidenceRecord>()
    for (const row of rows) {
      const current = byId.get(row.id)
      if (!current || rank[row.relationship] > rank[current.relationship]) byId.set(row.id, row)
    }
    return [...byId.values()]
  }
  ```

- [ ] **Implement `EvidenceRef` and `EvidenceLedger`.** Use inline focusable tokens E/F/P/T/M with accessible type names; order Cited, Consulted, Retrieved; compute counts from unique IDs.

- [ ] **Implement inspector record views.** Render exact provided fields, Advanced scores/build/calibration, linear path alternative under 420 px, textual temporal/availability states, and no semantic claims stronger than the payload.

- [ ] **Implement native-source actions.** Route Apple Notes through session `open_note`; allow Copy text/path/link only when present; keep failure inline; never guess a source-app capability.

- [ ] **Define inquiry state.** Store normalized threads/messages, active thread, local composer drafts/scopes, ordered process events, content buffers, active operation ID, and background stream counts; keep socket out of Redux.

- [ ] **Implement first-send semantics.** Create server chat on first send/ack, retain input on busy/failure until authoritative reconciliation, and never replay a write after socket uncertainty.

- [ ] **Implement exact scope serialization.** Resolve meeting ID and seed note IDs from current meeting, pinned notes/meetings, or Atlas selection/neighborhood; label all seed scopes as retrieval starting points, not security filters.

- [ ] **Implement the stream reducer.** Preserve full retrieval pack, reasoning/activity in Advanced, 30–50 ms answer batching, stable activity replacement, cancellation/interruption, stripped claims, usage, build/calibration IDs, and authoritative done swap.

- [ ] **Implement flat turn anatomy.** Render phase line, process disclosure, progressive Markdown, evidence ledger, and actions under one document section with 2 px rule; user turns are full-width labelled blocks, never bubbles/cards.

- [ ] **Implement composer behavior.** Minimum 44 px, grow to eight lines, Enter send, Shift+Enter newline, `⌘.` stop focused turn, editable next-turn text while streaming with Send disabled.

- [ ] **Implement regenerate/rename/delete.** Preserve old rendered answer during server replacement, refetch on failure, roll back optimistic title, and state permanent message/evidence deletion.

- [ ] **Implement shared navigation intents.** Inspector selection is workspace state; note/meeting/evidence links update coordinate/selection/history; `View evidence in Atlas` applies an evidence-set lens without duplicating graph state.

- [ ] **Wire meeting Ask and inline Ask.** Reuse the Inquiry workbench with meeting scope; `chat_notes_edit` upserts the durable proposal and never calls the editor.

#### GREEN checkpoint

- [ ] **Run the Slice 4 unit/type/build gate.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npm test -- src/renderer/src/features/evidence src/renderer/src/features/inquiry src/renderer/src/components/knowledge src/renderer/src/features/meetings src/renderer/src/features/atlas
  npm run lint
  npm run typecheck
  npm run build
  ```

  Expected: PASS for every chat event/outcome, exact evidence semantics, no-resend reconnect, claim-level inspection, and cross-coordinate navigation.

- [ ] **Run the Slice 4 inspectable journey.** Ask with three cited, two consulted, and four retrieved-only records; inspect a passage/fact/path; run fallback and interrupted scenarios; stop and regenerate a turn; navigate one citation to Atlas and one meeting citation to Chronicle. Expected: groups/counts are exact and no unsupported support claim appears.

- [ ] **Freeze Slice 4 after GREEN.**

  ```bash
  cd "$WORKTREE_ROOT"
  git diff --check
  git add app/src/renderer/src/features/evidence app/src/renderer/src/features/inquiry app/src/renderer/src/components/knowledge app/src/renderer/src/app app/src/renderer/src/features/meetings app/src/renderer/src/features/atlas
  git commit -m "feat: deliver atlas inquiry and evidence"
  ```

  Expected: commit succeeds and becomes the immutable Slice 4 review target.

- [ ] **Stop and request independent Slice 4 review.** Provide the commit SHA, RED/GREEN output, evidence grouping fixtures, and inquiry journey evidence. Expected: no Slice 5 path changes until review returns clean; blocking findings are repaired and re-frozen in Slice 4.

---

### Task 5: Knowledge operations

**Runnable journey:** Investigate with Atlas time/lenses/neighborhood/Outline/comparison, authorize and sync a source without losing the old Atlas, purge with exact consequences, edit/generate the Interpretation profile, inspect redacted diagnostics, use native note opening and utilities, and activate a deterministic privacy curtain during capture.

**Files:**

- Modify: `app/src/renderer/src/features/atlas/AtlasWorkbench.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasCanvas.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasOutline.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasIndex.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasControls.tsx`
- Create: `app/src/renderer/src/features/atlas/TimeBand.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasWorkbench.test.tsx`
- Create: `app/src/renderer/src/features/workspace/ComparisonBoard.tsx`
- Create: `app/src/renderer/src/features/workspace/ComparisonBoard.test.tsx`
- Create: `core/taurine_core/connectors/credentials.py`
- Create: `core/taurine_core/connectors/oauth.py`
- Modify: `core/taurine_core/connectors/gmail.py`
- Modify: `core/taurine_core/connectors/outlook.py`
- Modify: `core/taurine_core/connectors/google_common.py`
- Modify: `core/taurine_core/connectors/google_calendar.py`
- Modify: `core/taurine_core/connectors/google_docs.py`
- Modify: `core/taurine_core/connectors/registry.py`
- Modify: `core/taurine_core/connectors/catalog.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/pyproject.toml`
- Modify: `core/uv.lock`
- Create: `core/tests/test_connector_oauth.py`
- Modify: `core/tests/test_connectors.py`
- Create: `app/src/renderer/src/features/sources/descriptors.ts`
- Create: `app/src/renderer/src/features/sources/state.ts`
- Create: `app/src/renderer/src/features/sources/selectors.ts`
- Create: `app/src/renderer/src/features/sources/listeners.ts`
- Create: `app/src/renderer/src/features/sources/SourcesIndex.tsx`
- Create: `app/src/renderer/src/features/sources/SourcesLedger.tsx`
- Create: `app/src/renderer/src/features/sources/SourceDetail.tsx`
- Create: `app/src/renderer/src/features/sources/SourceConfig.tsx`
- Create: `app/src/renderer/src/features/sources/Authorization.tsx`
- Create: `app/src/renderer/src/features/sources/sources.test.tsx`
- Create: `app/src/renderer/src/features/profile/state.ts`
- Create: `app/src/renderer/src/features/profile/listeners.ts`
- Create: `app/src/renderer/src/features/profile/ProfileWorkbench.tsx`
- Create: `app/src/renderer/src/features/profile/ProfileEditor.tsx`
- Create: `app/src/renderer/src/features/profile/GenerationRail.tsx`
- Create: `app/src/renderer/src/features/profile/profile.test.tsx`
- Create: `app/src/renderer/src/features/diagnostics/DiagnosticsWorkbench.tsx`
- Create: `app/src/renderer/src/features/diagnostics/SessionLog.tsx`
- Create: `app/src/renderer/src/features/diagnostics/diagnostics.test.tsx`
- Modify: `app/src/renderer/src/features/privacy/PrivacyCurtain.tsx`
- Modify: `app/src/renderer/src/features/privacy/PrivacyStatus.tsx`
- Create: `app/src/renderer/src/features/privacy/privacy.test.tsx`
- Modify: `app/src/renderer/src/utility/call-prompt-main.tsx`
- Modify: `app/src/renderer/src/utility/context-suggestion-main.tsx`
- Modify: `app/src/renderer/src/utility/utility.css`
- Modify: `app/src/renderer/src/utility/utility.test.tsx`
- Modify: `app/src/renderer/notif.html`
- Modify: `app/src/renderer/suggest.html`
- Modify: `app/src/renderer/src/features/runtime/FirstRunChecklist.tsx`
- Modify: `app/src/renderer/src/app/AppShell.tsx`

**Interfaces:** Consumes the immutable graph/evidence shell, connector/index/profile/session-log contracts, bounded desktop bridge, and existing native adapters. Produces the completed Atlas, `CredentialStore`/PKCE OAuth REST surface, Sources/Profile/Diagnostics workbenches, truthful privacy/utility behavior, and exact Control state inventory.

#### RED checkpoint

- [ ] **Add complete Atlas interaction tests.** Cover global/local depth 1–3, ego history, incoming/outgoing direction, exact edge patterns/labels, selection/pin/live marks, user pin/release/reset, old graph retention, and graph/list fallback.

- [ ] **Add time-band tests.** Cover 56 px band, All/30 days/90 days/1 year/Custom, brush/fields equivalence, invalid/custom dates, absolute localized tooltip/name, and synchronized Atlas/Outline/Chronicle filtering.

- [ ] **Add lens tests.** Cover source/community/recency/connections/minimum links default 2/session/evidence set/AI idea, removable text clauses, search override for hidden tail, idea input preservation on result/abstain/error, and no graph mutation.

- [ ] **Add live-overlay tests.** Cover Conversation now diamond, ranked direct retrieval, striped traversal, grounded ring, three-second visual recession, event-list retention, and reconnect copy `Live retrieval trail restarted after reconnect`.

- [ ] **Add Outline parity tests.** Assert every zoom/pan/fit/search/history/depth/pin/release/reset/time/context action is reachable by keyboard/menu and Canvas contains no exclusive behavior.

- [ ] **Add comparison tests.** Cover exactly two/three objects, type-specific dossier content, shared entity/date/source/citation text markers, optional synchronized scroll, and explicit Ask these as the only synthesis action.

- [ ] **Add OAuth core tests.** Cover supported-provider guard, PKCE S256/state/exact loopback, five-minute expiry, replay rejection, credential vault read/write/delete, env compatibility precedence, refresh rotation, provider error redaction, and zero token/code/verifier leakage.

- [ ] **Add OAuth route tests.** Cover POST authorize 202, GET flow waiting/authorized/expired/error, exact callback page, DELETE authorization, connector update publication, and no automatic enable/sync/purge.

- [ ] **Add Sources state tests.** Cover priority syncing/cancelling→Needs attention→Active→Off→System, hidden Taurine row, enable-and-sync/batch enable, disable consequences, critical-boundary disabled controls, and old Atlas during build.

- [ ] **Add descriptor tests.** Cover Apple Notes, Obsidian, Folders, iMessage, Claude, Codex, Gmail, Outlook, Calendar, and Docs controls/ranges plus unknown-key round trip and directory-chooser-only paths.

- [ ] **Add sync/purge tests.** Cover real `done/total`, indeterminate absent totals, extraction versus build, cancel terminal wait, stale on cancel/failure, atomic graph swap, two-click purge, document count/provider boundary/no undo, and removed open object state.

- [ ] **Add Profile tests.** Cover empty/populated/editing/dirty/saving/error, nonempty validation, overwrite confirmation, all five server phases, background navigation, disabled manual save reason, success, and failure retaining prior content.

- [ ] **Add Diagnostics tests.** Cover session list/detail filters ASR/chunk/retrieval/grounding/summary/chat/control/error, absent logs, operational-not-durable copy, and redacted diagnostics export.

- [ ] **Add privacy tests.** Cover requested/limited/unavailable/curtain, neutral private-free main DOM, curtain-safe call/context utilities, disabled preview, exact state restoration, and Stop availability while capture continues.

- [ ] **Add native utility tests.** Cover call states, pause/start/dismiss, one prompt/rearm, no focus raise, context cap/timeout/open/dismiss, native Apple Notes open/failure, source settings remediation, and utility absence of store/sidecar imports.

- [ ] **Run the combined Slice 5 RED checkpoint.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/core"
  uv run pytest tests/test_connector_oauth.py tests/test_connectors.py -q
  cd "$WORKTREE_ROOT/app"
  npx vitest run src/renderer/src/features/atlas src/renderer/src/features/workspace/ComparisonBoard.test.tsx src/renderer/src/features/sources src/renderer/src/features/profile src/renderer/src/features/diagnostics src/renderer/src/features/privacy src/renderer/src/utility
  ```

  Expected: FAIL because OAuth, completed Atlas operations, Control workbenches, and utility renderers do not exist. Save this as the slice's one RED artifact.

#### Minimal implementation

- [ ] **Complete local/global Atlas navigation.** Use existing graph endpoints, keep note-edge weight labelled Projection weight, list fact/provenance/mechanisms exactly as returned, and omit Explain this edge.

- [ ] **Implement the time band and shared date selector.** Use locale/time zone `Intl`; update Atlas, Outline, and Chronicle from one range; provide fields/presets so drag is never required.

- [ ] **Implement removable lens clauses.** Store explicit filter values, render a sentence-like clause summary, keep hidden tail reachable via search/Outline, and retain user input on idea no-result/error.

- [ ] **Implement live conversation overlay.** Keep route history renderer-memory-only, decay marks without deleting current event rows, and reset only the route trail after renderer/session reconnect.

- [ ] **Complete Outline parity.** Use tree/treegrid semantics, correct set position/count, separate incoming/outgoing groups, keyboard context menu, and every Canvas operation via named command.

- [ ] **Implement bounded comparison.** Render two/three rule-separated columns with object-specific fields and textual shared markers; do not call an LLM until Ask these.

- [ ] **Add keyring through the lockfile.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/core"
  uv add keyring
  ```

  Expected: `core/pyproject.toml` and `core/uv.lock` change; no application database or migration file is created.

- [ ] **Implement `CredentialStore`.** Provide OS-keyring and memory test backends using versioned JSON; vault credentials override read-only environment compatibility; `repr`, errors, and logs redact all secrets.

- [ ] **Implement provider-neutral OAuth flow state.** Generate cryptographic state/verifier, S256 challenge, fixed loopback redirect from bound sidecar port, five-minute expiry, single consumption, injected exchange/clock in tests, and stable safe error codes.

- [ ] **Inject credential providers into OAuth connectors.** Share Google refresh plumbing, isolate Microsoft, persist rotated refresh tokens, retry once after auth failure, and retain existing environment developer setup.

- [ ] **Implement authorization REST endpoints.** Accept only registry members with `kind="oauth"`; open a minimal callback page; publish connector updates on auth/disconnect; never put tokens in ConnectorInfo, config, Redux, logs, errors, or IPC.

- [ ] **Implement source descriptor state.** Preserve unknown flat config keys; derive field ranges from constants/descriptors; accept folder/vault values only from `chooseDirectory`.

- [ ] **Implement the 56 px Sources ledger.** Group Needs attention/Active/Off/System; show glyph/name/state/detail/count/last sync/enable/disclosure; use the specified priority and one hidden immutable Taurine system source.

- [ ] **Implement enable/disable and authorization.** Enable offers sync now or batch; disable states stored/excluded/stale/rebuild consequences; Connect opens only returned allowlisted HTTPS URL and polls; environment setup stays Advanced.

- [ ] **Implement global/per-source sync.** Show percent only for real totals, distinguish source read from index build, cancel and await terminal state, retain old graph, and swap only after version fetch succeeds.

- [ ] **Implement purge.** Two actions, current count, provider untouched, source disabled, derived/local rows removed where supported, stale until rebuild, no undo, conflict while critical sync, and affected object refetch.

- [ ] **Implement Interpretation profile.** State that it resolves names/projects/interests and is not citable evidence; retain current content through generation; use only Surveying, Interests, Projects & people, Career & studies, Synthesizing plus server detail/counts/time.

- [ ] **Implement Diagnostics.** Treat session logs as optional operational traces, filter structured records, omit absent private passage text, and export runtime/app/socket/build/calibration/safe errors with default redaction.

- [ ] **Complete literal privacy state.** Map desktop report to exact copy; curtain unmounts private main content, substitutes generic utility payloads, disables preview, and restores prior coordinate/layout/live/compact focus state.

- [ ] **Implement stateless utility renderers.** Consume only sanitized preload events; call prompt and context suggestion render exact states/actions; import no Redux, gateway, meeting entity, transcript, editor, or connector service.

- [ ] **Implement native source opening and remediation.** Offer Apple Notes open only when supported; show inline failure; use allowlisted System Settings for mic/system/Full Disk Access; show copy/path/link alternatives only from payload.

#### GREEN checkpoint

- [ ] **Run Core OAuth/connector and full targeted regression.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/core"
  uv run ruff check taurine_core tests
  uv run pytest tests/test_connector_oauth.py tests/test_connectors.py tests/test_server_startup.py tests/test_session_log.py -q
  ```

  Expected: PASS; logs/responses contain no token, code, verifier, or credential; authorization never silently enables, syncs, disables, or purges.

- [ ] **Run the Slice 5 app gate.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npm test -- src/renderer/src/features/atlas src/renderer/src/features/workspace src/renderer/src/features/sources src/renderer/src/features/profile src/renderer/src/features/diagnostics src/renderer/src/features/privacy src/renderer/src/utility src/main
  npm run lint
  npm run typecheck
  npm run build
  ```

  Expected: PASS for Atlas parity/time/lenses/comparison, every source/profile/diagnostic/privacy state, and utility/native behavior.

- [ ] **Run the Slice 5 inspectable journey.** Pin and compare three notes without pointer; Ask these; enable/authorize/sync a fixture source while using the old Atlas; cancel once and resync; purge after reading consequences; edit/generate profile; export redacted diagnostics; activate curtain during capture and inspect utility redaction. Expected: every consequence/state is exact and private content is absent under curtain.

- [ ] **Freeze Slice 5 after GREEN.**

  ```bash
  cd "$WORKTREE_ROOT"
  git diff --check
  git add core/taurine_core/connectors core/taurine_core/schemas.py core/taurine_core/server.py core/pyproject.toml core/uv.lock core/tests/test_connector_oauth.py core/tests/test_connectors.py app/src/renderer/src/features/atlas app/src/renderer/src/features/workspace app/src/renderer/src/features/sources app/src/renderer/src/features/profile app/src/renderer/src/features/diagnostics app/src/renderer/src/features/privacy app/src/renderer/src/features/runtime app/src/renderer/src/utility app/src/renderer/notif.html app/src/renderer/suggest.html
  git commit -m "feat: complete atlas knowledge operations"
  ```

  Expected: commit succeeds and becomes the immutable Slice 5 review target.

- [ ] **Stop and request independent Slice 5 review.** Provide the commit SHA, both GREEN outputs, OAuth redaction evidence, and knowledge/privacy journey evidence. Expected: no Slice 6 path changes until review returns clean; blocking findings are repaired and re-frozen in Slice 5.

---

### Task 6: Cutover and proof

**Runnable journey:** Launch the production/package entry with only Knowledge Atlas, complete acceptance stories A–N across stable visual/accessibility states, prove native/privacy/performance/bundle behavior, remove every old UI/native owner, and pass the complete Core/Electron release gate.

**Files:**

- Create: `app/src/renderer/src/test/accessibility-contracts.test.tsx`
- Create: `app/src/renderer/src/i18n/i18n.test.ts`
- Modify: all stable renderer components only where tests demonstrate accessibility/localization failures
- Modify: `app/src/renderer/src/main.tsx`
- Modify: `app/src/renderer/src/env.d.ts`
- Create: `app/src/renderer/src/app/cutover.test.tsx`
- Create: `app/playwright.config.ts`
- Create: `app/e2e/fixtures/fake-sidecar.ts`
- Create: `app/e2e/fixtures/scenarios.ts`
- Create: `app/e2e/knowledge-atlas.spec.ts`
- Create: `app/e2e/accessibility.spec.ts`
- Create: `app/e2e/visual.spec.ts`
- Create: `app/e2e/performance.spec.ts`
- Create: `app/scripts/check-old-ui-removed.mjs`
- Create: `app/scripts/check-sensitive-logs.mjs`
- Create: `docs/superpowers/verification/2026-07-20-knowledge-atlas-release-checklist.md`
- Delete: `app/src/renderer/src/App.tsx`
- Delete: `app/src/renderer/src/assets/main.css`
- Delete: `app/src/renderer/src/assets/electron.svg`
- Delete: `app/src/renderer/src/assets/wavy-lines.svg`
- Delete: `app/src/renderer/src/views/`
- Delete: `app/src/renderer/src/components/ActionableRail.tsx`
- Delete: `app/src/renderer/src/components/AmbientList.tsx`
- Delete: `app/src/renderer/src/components/ConnectionsPanel.tsx`
- Delete: `app/src/renderer/src/components/SessionControls.tsx`
- Delete: `app/src/renderer/src/components/StatusBar.tsx`
- Delete: `app/src/renderer/src/components/SummaryPanel.tsx`
- Delete: `app/src/renderer/src/components/TriggerInspector.tsx`
- Delete: `app/src/renderer/src/components/UserSummary.tsx`
- Delete: `app/src/renderer/src/components/Versions.tsx`
- Delete: `app/src/renderer/src/components/chat/`
- Delete: `app/src/renderer/src/lib/`
- Delete: `app/src/renderer/src/notif-main.ts`
- Delete: `app/src/renderer/src/suggest-main.ts`
- Delete: `app/src/main/notifier.ts`
- Delete: `app/src/main/suggest-notifier.ts`
- Delete: `app/src/main/sysaudio.ts`
- Delete: `app/src/main/micwatch.ts`
- Modify: `app/electron.vite.config.ts`
- Modify: `app/package.json`
- Modify: `app/package-lock.json`

**Interfaces:** Consumes all five frozen slices. Produces one production entry, three total renderer entries (main plus two stateless utilities), release fixtures/scanners/evidence, no old architecture, and one recorded exhaustive acceptance run.

#### RED checkpoint

- [ ] **Add stable-workbench axe tests.** Cover startup, Atlas, Chronicle draft/live/stopped/final/error, Inquiry outcomes, Sources states, Profile states, Diagnostics, privacy, empty/loading/error, comparison, and utility windows.

- [ ] **Add keyboard/focus tests.** Cover every primary job, all global/local shortcuts, F6 zones, modal/sheet/menu precedence, focus restoration, no hidden focusable descendants, and no essential pointer-only action.

- [ ] **Add zoom/i18n tests.** Cover 200% at 960×640, 35% pseudo expansion, RTL, localized dates/numbers/plurals/key names/speakers, evidence/hash LTR isolation, and no destructive-label ellipsis.

- [ ] **Add cutover source test.** Assert `main.tsx` imports `./app/App`, providers, local fonts, and target CSS only; reject legacy/old UI/interface selector names and alternate main entry.

- [ ] **Add fake-sidecar REST and three-socket fixture.** Model approved current contracts plus note proposals/commands, actor session snapshot/error, and OAuth; expose deterministic network/startup/permission/conflict/finalization faults through fixture methods, never by bypassing gateway code.

- [ ] **Add acceptance stories A–N one-for-one.** Use exact test names from the design: start/finish, mic-only, reconnect, note loss, proposal undo, evidence, fallback/interruption, pointer-free Atlas, source sync, purge, profile, curtain, sole replacement, and reviewable AI edits.

- [ ] **Add performance scenarios.** Generate 5,000 Atlas nodes, 10,000 list rows, 3,000 transcript blocks, and 200 inquiry messages; assert ≥30 Atlas fps, 60 fps virtual lists, p95 command/selection <100 ms, bounded DOM rows, and no unbounded heap growth over coordinate switches.

- [ ] **Add visual scenarios.** Capture 1440/1280/960, compact 440×680, light/dark/system, compact/comfortable, two/three-lane Live, 100/200% zoom, pseudo, RTL, reduced motion, privacy, stable empty/loading/error, and both utility windows with stabilized time/fonts/animations.

- [ ] **Add old-UI scanner rules.** Reject old root/views/components/lib owners, old Feed/Map/Data/Connectors routes, old yellow/dark token signatures, duplicate stateful entry, interface flags, and moved native module imports.

- [ ] **Add sensitive-log scanner rules.** Inspect source and built source maps for logger/breadcrumb calls containing note/transcript/profile/passage/prompt/answer/token/credential/raw retrieval fields.

- [ ] **Add the manual release checklist.** Require date/build/OS/device/result/evidence for VoiceOver, NVDA where supported, 200% zoom, pseudo/RTL, reduced motion, forced colors, ScreenCaptureKit requested/limited/curtain, permissions/audio, call/context utilities, tray/dock/compact, source adapters, sleep/wake/network/sidecar restart/renderer reload, signing, and packaged launch.

- [ ] **Run the combined Slice 6 RED checkpoint before cutover/deletion.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npx vitest run src/renderer/src/test/accessibility-contracts.test.tsx src/renderer/src/i18n/i18n.test.ts src/renderer/src/app/cutover.test.tsx
  npm run check:old-ui
  npm run test:e2e
  ```

  Expected: FAIL because production still mounts the old root, old files remain, and release fixtures/snapshots/checklist evidence are incomplete. Save this as the slice's one RED artifact.

#### Minimal implementation

- [ ] **Fix only demonstrated unit accessibility failures.** Repair names, roles, heading/order, tree/treegrid metadata, focus trap/return, inert backgrounds, live-region cadence, target sizes, focus visibility, status text, and reader ordering without changing approved product behavior.

- [ ] **Fix only demonstrated zoom/localization failures.** Use logical CSS, pane collapse, scroll padding, wrapping, catalogue completion, `Intl`, RTL mirroring of UI chrome but not fact direction, and reduced/forced-color equivalents.

- [ ] **Audit BlockNote with the required source-mode fallback.** If the WYSIWYG cannot pass heading/list/menu/undo/selection/proposal/provenance/focus tests, make the persistent Markdown source mode equally reachable and complete before cutover.

- [ ] **Switch the sole production entry.**

  ```ts
  import "@fontsource/ibm-plex-sans/400.css"
  import "@fontsource/ibm-plex-sans/500.css"
  import "@fontsource/ibm-plex-sans/600.css"
  import "@fontsource/ibm-plex-mono/400.css"
  import "@fontsource/ibm-plex-mono/500.css"
  import "./styles/reset.css"
  import "./styles/tokens.css"
  import "./styles/typography.css"
  import "./styles/primitives.css"
  import "./styles/shell.css"
  import { App } from "./app/App"
  ```

  Mount one Redux provider and one `App`; expose no environment/query/route selection mechanism.

- [ ] **Implement fake-sidecar scenarios through real contracts.** Serve REST and WebSocket order exactly; fault injection changes transport/server state, not renderer store/actions.

- [ ] **Implement stories A–N.** Assert observable state and durable fixture records, including no audio replay, notes revision behavior, exact evidence groups, old immutable Atlas, and canonical notes unchanged before proposal Accept.

- [ ] **Implement automated keyboard/axe/visual/performance suites.** Stabilize time/font/motion; collect traces/screenshots only on configured failures; report measured frame/input/heap and bundle values.

- [ ] **Implement old-UI and sensitive-log scanners.** Use explicit path/import/token rules and exact allowlists only for catalog copy that is not an old route/selector.

- [ ] **Delete all old renderer ownership.** Remove the root App/theme/assets, `views/`, old UI components, socket/state coordinators, and old utility entries; move no alias/re-export forward.

- [ ] **Delete all superseded native ownership.** Remove notifier/suggest-notifier/sysaudio/micwatch only after target windows/capture imports and tests pass; keep `public/pcm-worklet.js` because capture still uses it.

- [ ] **Remove unused old dependencies.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/app"
  npm uninstall react-force-graph-2d @tailwindcss/vite tailwindcss
  ```

  Expected: package/lock update; Canvas/d3 and checked-in CSS remain; no runtime or database migration occurs.

- [ ] **Leave exactly three renderer entries.** Configure main Knowledge Atlas, call prompt, and context suggestion; ensure only main imports Redux/gateway/domain code.

- [ ] **Complete the signed manual checklist on packaged builds.** Record requested/limited ScreenCaptureKit reality without overclaim, curtain neutral pixels, VoiceOver full jobs, Windows NVDA non-native-audio jobs where supported, and native lifecycle tests.

- [ ] **Treat unavailable external proof as a release blocker.** A missing reference Mac/ScreenCaptureKit app, unavailable required assistive-technology run, unsigned/unlaunchable package, or unexecuted native permission/audio check leaves its checklist row `BLOCKED`, prevents the Slice 6 GREEN checkpoint, and must not be replaced by a unit, axe, mocked Electron, screenshot, or source-review result.

#### GREEN checkpoint

- [ ] **Run the clean comprehensive release gate in one recorded run.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT/core"
  uv run ruff check taurine_core tests
  uv run pytest tests -q
  cd "$WORKTREE_ROOT/app"
  npm run lint
  npm run typecheck
  npm test
  npm run build
  npm run check:bundle
  npm run check:old-ui
  npm run check:sensitive-logs
  npm run test:e2e
  npm run test:visual
  ```

  Expected: every command exits 0; Core reports at least 981 passed and exactly the existing 2 skipped plus additions; Vitest reports at least 61 plus additions; A–N, axe, keyboard, performance, visual, privacy, and native checklist gates pass; bundle output is ≤1.25 MB initial JS, ≤140 KB CSS, and ≤900 KB per lazy feature. No manual/moderated checklist row is unchecked, waived, inferred from automation, or marked unavailable.

- [ ] **Run structural invariants.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT"
  test "$(sed '/^[[:space:]]*$/d' app/src/renderer/src/app/App.tsx | wc -l | tr -d ' ')" -le 120
  rg -n 'new WebSocket' app/src/renderer/src
  rg -n 'setMarkdown|updateMeeting\(|note-commands' app/src/renderer/src
  rg -n 'edge-reason|SessionRegistry|_detached_entries|finalizer_handoff|interfaceFlag|window\.api' core/taurine_core app/src
  ```

  Expected: the line-count check exits 0; WebSocket constructors appear only in the three sidecar connection modules; canonical note mutation appears only in `MeetingDocumentController.ts`, its API adapter, and tests; the final `rg` exits 1.

- [ ] **Run source/removal/privacy scans.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT"
  rg -n 'FeedView|MapView|DataView|ConnectorsView|legacy|oldUi|yellow|screenshare invisible|uncapturable' app/src app/electron.vite.config.ts app/package.json
  rg -n 'T[O]DO|T[B]D|F[I]XME|place[h]older|coming[ ]soon|implement[ ]later' core/taurine_core app/src app/e2e docs/superpowers/verification
  ```

  Expected: both commands exit 1 with no matches. Any legitimate source word must live behind a reviewed catalog key and an exact scanner allowlist, never an old route/theme marker.

- [ ] **Inspect the final diff and checklist.**

  Run:

  ```bash
  cd "$WORKTREE_ROOT"
  git diff --check
  git status --short
  git diff --stat
  ```

  Expected: no whitespace errors, one target implementation, old paths are deletions, verification evidence is present, and no unrelated user change is staged.

- [ ] **Freeze Slice 6 after comprehensive GREEN.**

  ```bash
  cd "$WORKTREE_ROOT"
  git add -A app/src/renderer/src app/src/main/notifier.ts app/src/main/suggest-notifier.ts app/src/main/sysaudio.ts app/src/main/micwatch.ts app/electron.vite.config.ts app/package.json app/package-lock.json app/playwright.config.ts app/e2e app/scripts/check-old-ui-removed.mjs app/scripts/check-sensitive-logs.mjs docs/superpowers/verification
  git commit -m "feat: cut over and prove the knowledge atlas"
  ```

  Expected: commit succeeds and becomes the immutable final review target.

- [ ] **Stop and request independent final review.** Provide the commit SHA, complete RED/GREEN record, A–N report, bundle/performance/visual artifacts, scanner outputs, and signed manual checklist. Expected: release remains blocked until review is clean; any blocking finding requires a new Slice 6 commit and a repeated affected gate before re-review.

---

## Design coverage audit

| Approved requirement | Owning slice | Direct proof |
|---|---:|---|
| Reset §§1–4 one serial owner, admission 256, queue completions, simple cancellation, actor failure/unavailability | 1 | actor matrix, absence scan, socket journey |
| Reset §5 bounded projection, snapshot-before-deltas, immediate frame rejection, no audio replay | 1, 3 | actor/socket tests, meeting reconnect journey |
| Reset §6 all 14 lifecycle cases | 1 | repeated `test_session_actor.py` and socket integration |
| Reset §7 retain approved Tasks 1–4, replace Task 5, no detached repair continuation | prerequisite, 1 | ancestry checks, deleted registry/tests, absence scan |
| Reset §§8–9 six runnable vertical slices, one RED/GREEN and review per slice | 1–6 | exact headings/checkpoints/commits in this plan |
| Design §§0–2 thesis, capability preservation, trust vocabulary, metrics/non-goals | 1–6 | slice journeys, evidence tests, final A–N/release gate |
| §3 three coordinates, separate Control, history, selection, pins, compare, exception ledger | 2, 5 | shell/store/comparison tests |
| §4 one stateful window, exact breakpoints/compact/utilities/tray/privacy | 1–3, 5–6 | desktop/shell/live/privacy/native tests |
| §5 selection/preview/open/pin grammar, command center, primary actions, focus zones | 2, 5–6 | command, shell, Outline, keyboard E2E |
| §6 territory LOD, stable geography, time, lenses, local direction, live overlay, accessible Outline, errors | 2, 5 | Atlas unit/performance/keyboard/visual tests |
| §7.1 startup/readiness/first run/permissions | 2 | hydration/runtime tests and inspectable journey |
| §7.2 complete meeting lifecycle, notes, transcript, intelligence, resume/Stop/final/review/delete | prerequisite, 1, 3 | actor/controller/dossier tests and stories A–E/N |
| §7.3 Inquiry threads, scopes, turns, evidence outcomes, management/reconnect | 4 | inquiry/evidence tests and stories F–G |
| §7.4 Sources config/auth/sync/cancel/purge and immutable old Atlas | 5 | OAuth/Sources tests and stories I–J |
| §7.5 Profile states and server-only phases | 5 | Profile tests and story K |
| §§7.6–7.7 Diagnostics and capability-gated native opening | 4–5 | inspector/diagnostic/native tests |
| §8 complete runtime/permission/meeting/notes/chat/evidence/connector/profile/utility state inventory | 1–6 | colocated state matrices and visual fixtures |
| §9 exact tokens/type/spacing/density/motion/no decorative AI styling | 2, 6 | token tests, visual matrix, old-theme scanner |
| §10 primitives/layout/knowledge/meeting/Ask measurements and behavior | 2–5 | component tests and final axe/visual matrix |
| §11 keyboard, semantics, announcements, contrast, zoom, editor, i18n/RTL | 2–6 | unit accessibility, Playwright, manual checks |
| §12 Redux ownership, finite states, one gateway, hydration/reconciliation, privacy, splitting/performance | 1–6 | gateway/store/controller tests, scanners, bundle/perf gates |
| §§13.1–13.5 retained REST, three sockets, desktop contracts, all discriminants | 1–5 | parser/OpenAPI/gateway/native tests |
| §13.6 proposals/commands/revision/idempotency/undo/legacy reads | approved Tasks 2–4, 3 | retained Core suites, controller tests, stories D/E/N |
| §13.7 live continuation as superseded by reset actor | 1, 3 | actor 14-case matrix, socket/state-machine tests, story C |
| §13.8 desktop OAuth/vault/env compatibility | 5 | OAuth/vault/IPC/log tests and Sources journey |
| §§13.9–13.10 chat cleanup and deliberately absent contracts | approved Task 4, 4–6 | retained delete tests, API/diff/edge scan |
| §14 target decomposition and total old-architecture removal | 1–6 | exact file lists, cutover/deletion scanner |
| §15 automated/manual/native/performance/visual acceptance | 6 | one comprehensive recorded GREEN run/checklist |
| §§16–18 risk mitigations, completeness, final product decisions | 1–6 | requirement matrix, A–N, final invariant scans |

## Acceptance-story ownership

| Story | Slice | Required test name in `app/e2e/knowledge-atlas.spec.ts` |
|---|---:|---|
| A. Start and finish a meeting | 1, 3 | `A start and finish a meeting` |
| B. Degrade to mic only | 1, 3 | `B degrade to mic only` |
| C. Reconnect honestly | 1, 3 | `C reconnect honestly` |
| D. Prevent note loss | prerequisite, 3 | `D prevent note loss` |
| E. Accept and undo a final proposal | prerequisite, 3 | `E accept and undo a final proposal` |
| F. Ask with transparent evidence | 4 | `F ask with transparent evidence` |
| G. Handle low-confidence and interrupted turns | 4 | `G low confidence and interrupted turns` |
| H. Investigate spatially without a pointer | 2, 5 | `H investigate spatially without a pointer` |
| I. Sync sources without losing the old Atlas | 5 | `I sync sources without losing the old Atlas` |
| J. Purge with exact boundaries | 5 | `J purge with exact boundaries` |
| K. Generate profile safely | 5 | `K generate profile safely` |
| L. Use privacy curtain during capture | 1, 3, 5 | `L privacy curtain during capture` |
| M. Replace, not coexist | 6 | `M replace not coexist` |
| N. Keep every AI note change reviewable | prerequisite, 3, 4 | `N every ai note change is reviewable` |

## Final plan audit for the executing worker

- [ ] **Confirm slice commits/reviews.** Six slice commits exist in order after the approved prerequisite; each has its saved RED/GREEN output and review disposition.
- [ ] **Confirm no migration work appeared.** No database copy/reset, user-facing conversion, actor-state persistence, migration screen, or ephemeral-session schema exists; existing additive proposal DDL still opens old stores in place.
- [ ] **Confirm canonical type consistency.** Python and TypeScript session dispositions, wire codes, snapshot fields, meeting proposal fields, capture phases, coordinate/control values, and gateway/controller method names match the canonical block.
- [ ] **Confirm no incomplete-marker language.** Run the exact red-flag scan from Slice 6 and resolve every match before release.
- [ ] **Confirm path accountability.** Every Create/Modify/Delete path in the six slices is present or intentionally removed in the owning frozen commit; no old alias masks a missing target module.
- [ ] **Confirm design completeness.** Every row in the design coverage and A–N tables points to passing automated evidence plus manual evidence where the operating system or assistive technology requires it.
- [ ] **Confirm external evidence availability.** Any unavailable reference device, assistive-technology session, ScreenCaptureKit capture, native permission/audio run, signed package, or moderated workflow is a release blocker recorded as `BLOCKED`; completion cannot infer or waive it from automated coverage.
