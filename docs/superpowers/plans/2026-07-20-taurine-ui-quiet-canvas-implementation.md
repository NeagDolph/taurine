# Taurine Quiet Canvas UI-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task in the assigned worktree. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Quiet Canvas as Taurine's sole production Electron interface: a calm, meeting-note-centered workspace with explicit AI proposals, complete evidence semantics, accessible Connections, deterministic privacy, and no legacy UI.

**Architecture:** This plan supersedes the original 30-task sequencing. It implements the approved Quiet Canvas design as six end-to-end slices and uses the serialized `SessionActor` from the UI-first reset instead of adding more ownership layers to the approved reconnect implementation. Electron main owns native windows, encrypted recovery, privacy epochs, tray/menu behavior, and bounded capabilities; one typed renderer composition root owns repositories, transports, routing, documents, and conversations.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic 2, SQLite, pytest/pytest-asyncio; Electron 39, React 19, TypeScript 5.9, Vite/electron-vite, Tailwind CSS 4, BlockNote 0.51, react-markdown 10, react-force-graph-2d 1.29 lazy-only, Vitest 4, Testing Library, Playwright Electron, axe-core, and Electron `safeStorage`.

## Authoritative Inputs and Current State

- Read `docs/superpowers/specs/2026-07-20-taurine-ui-quiet-canvas-design.md` completely before editing UI code.
- Read `docs/superpowers/specs/2026-07-20-ui-first-session-architecture-reset-design.md` completely before editing session lifecycle code.
- Commits through Quiet Canvas Task 5 are approved on branch `codex/ui-quiet-canvas`; do not rewrite the approved durable meeting/proposal work.
- The prior Task 6 WIP is preserved at `/private/tmp/taurine-quiet-paused-task6-20260720.tar.gz`. Audit and reuse it; do not blindly overwrite newer branch changes.
- This plan has no data migration. Ephemeral session state is regenerated; existing durable Taurine data stays in its current schema except additive DDL explicitly required by the approved design.

## Global Constraints

- Work only in a path ending `/.worktrees/ui-quiet-canvas`. Abort if `git rev-parse --show-toplevel` does not match.
- Quiet Canvas and Knowledge Atlas remain independent designs. Do not import Atlas UI code or converge their visual systems.
- The final renderer has no persistent sidebar, five-tab row, dashboard, status footer, old route, compatibility entry, or runtime UI flag.
- Default window is `1120×780`, minimum `840×600`; user-selected Compact meeting is `840×660`; capture never raises, moves, or resizes the primary window.
- The title bar is `48 px`. Responsive thresholds are `840`, `960`, `1180`, `1320`, and `1440` CSS px. A docked editor remains at least `620 px`; Compare requires at least `400 px` per dossier.
- Light canvas/sheet/ink/accent/focus are `#F4F2EC`, `#FFFEFA`, `#20231F`, `#245E50`, `#145DAD`. Dark canvas/sheet/ink/accent/focus are `#171916`, `#20231F`, `#F1F0EA`, `#78B8A2`, `#80B7FF`. Copy every remaining token exactly from design §2.2.
- Use system UI type and bundled Source Serif 4 Variable for editorial content. Meeting body is `17/28`; meeting title is `32/38` weight `550`; no user text is below `12 px`.
- Motion uses opacity/translate only, `140 ms` standard and `180 ms` sheets with `cubic-bezier(.2,.8,.2,1)`; reduced motion removes force, traversal, countdown, and sheet motion.
- Meet WCAG 2.2 AA. Canvas is never the sole graph interface. Privacy curtain removes private DOM, title, portal, and live-region content rather than covering it.
- Exact evidence labels are `Cited`, `Consulted`, `Retrieved`, `Evidence policy passed`, `Low-confidence fallback`, `No supported evidence`, `From earlier context`, `Citation checks passed`, `Interrupted / stopped before verification`, and `Calibration not validated`. Never claim `Verified`, `Fact checked`, or guaranteed screen-capture invisibility.
- Only `MeetingDocumentController` mutates canonical BlockNote content. AI features create durable `NoteProposal` objects; acceptance is explicit and undoable.
- `safeStorage` recovery is persistent only when encryption is available and Linux backend is not `basic_text`; otherwise it is memory-only. Never write plaintext recovery to disk, IndexedDB, or localStorage.
- One `SidecarGateway`, one `/updates` projection, one `SessionTransport`, and one `ConversationHub` exist per renderer. Feature components never construct `fetch`, `WebSocket`, raw IPC, or editor insertions.
- Initial Desk budget is ≤`1.25 MB` uncompressed JS, ≤`120 KB` CSS, ≤`350 KB` gzip JS, and ≤`35 KB` gzip CSS. BlockNote, force graph, and Diagnostics stay out of initial imports.
- Warm-sidecar first content is `<500 ms`; Desk keyboard interactivity is `<1.5 s` p95 across 20 launches; route fallback appears within `100 ms`; ordinary editor input remains within one `16 ms` frame.
- Each slice follows RED → GREEN → focused/full verification → one frozen commit → independent review. P1/P2 correctness, privacy, accessibility, security, or data-loss findings block the next slice. P3 polish is recorded for the final proof slice unless it breaks the primary journey.

## Target Ownership Boundaries

```text
core/taurine_core/live/session_actor.py          serialized session table and lifecycle queue
core/taurine_core/server.py                      HTTP/WS composition; no session task ownership
app/src/main/index.ts                            Electron composition root only
app/src/main/{application-menu,window-state}.ts  native commands and bounds
app/src/main/{recovery-journal,privacy-curtain}.ts encrypted recovery and privacy epoch
app/src/preload/index.ts                         frozen, allowlisted Taurine API only
app/src/renderer/src/bootstrap/                  service composition and startup
app/src/renderer/src/data/                       contracts, gateway, repositories, transports
app/src/renderer/src/workspace/                  title bar, router, Open, curtain
app/src/renderer/src/meetings/                   sole document/capture/live/review journey
app/src/renderer/src/ask/                        conversations and evidence
app/src/renderer/src/knowledge/                  notes, Connections, comparison
app/src/renderer/src/settings/                   sources/profile/privacy/diagnostics
app/src/renderer/src/design/                     tokens, typography, primitives
```

---

### Task 1: Desktop and Recovery Foundation

**Runnable outcome:** Taurine starts reliably into a typed startup surface; tray/call capture does not raise a hidden window; the narrow preload has no raw Electron API; live sessions use one serialized actor; draft recovery and privacy curtain work across every Taurine window.

**Files:**

- Create: `core/taurine_core/live/session_actor.py`
- Create: `core/tests/test_session_actor.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/tests/test_server_ws.py`
- Create: `app/src/main/application-menu.ts`
- Create: `app/src/main/window-state.ts`
- Create: `app/src/main/recovery-journal.ts`
- Create: `app/src/main/privacy-curtain.ts`
- Modify: `app/src/main/index.ts`
- Modify: `app/src/main/tray.ts`
- Modify: `app/src/main/notifier.ts`
- Modify: `app/src/main/suggest-notifier.ts`
- Create: `app/src/main/__tests__/application-menu.test.ts`
- Create: `app/src/main/__tests__/window-state.test.ts`
- Create: `app/src/main/__tests__/tray.test.ts`
- Create: `app/src/main/__tests__/preload.test.ts`
- Create: `app/src/main/__tests__/recovery-journal.test.ts`
- Create: `app/src/main/__tests__/privacy-curtain.test.ts`
- Modify: `app/src/preload/index.ts`
- Modify: `app/src/preload/index.d.ts`
- Create: `app/src/renderer/src/platform/electronBridge.ts`
- Create: `app/src/renderer/src/platform/recoveryBridge.ts`
- Create: `app/src/renderer/src/data/DraftJournal.ts`
- Create: `app/src/renderer/src/data/contracts/decode.ts`
- Create: `app/src/renderer/src/data/contracts/startup.ts`
- Create: `app/src/renderer/src/data/contracts/updates.ts`
- Create: `app/src/renderer/src/data/contracts/meeting.ts`
- Create: `app/src/renderer/src/data/contracts/session.ts`
- Create: `app/src/renderer/src/data/contracts/evidence.ts`
- Create: `app/src/renderer/src/data/contracts/chat.ts`
- Create: `app/src/renderer/src/data/contracts/knowledge.ts`
- Create: `app/src/renderer/src/data/contracts/connector.ts`
- Create: `app/src/renderer/src/data/contracts/profile.ts`
- Create: `app/src/renderer/src/data/contracts/diagnostics.ts`
- Create: `app/src/renderer/src/data/SidecarGateway.ts`
- Create: `app/src/renderer/src/data/SidecarProjectionStore.ts`
- Create: `app/src/renderer/src/data/MeetingRepository.ts`
- Create: `app/src/renderer/src/data/KnowledgeRepository.ts`
- Create: `app/src/renderer/src/data/ConnectorRepository.ts`
- Create: `app/src/renderer/src/data/ProfileRepository.ts`
- Create: `app/src/renderer/src/data/DiagnosticsRepository.ts`
- Create: `app/src/renderer/src/bootstrap/RendererBootstrap.tsx`
- Create: `app/src/renderer/src/bootstrap/StartupScene.tsx`
- Create: `app/src/renderer/src/bootstrap/services.ts`
- Modify: `app/package.json`

**Interfaces:**

```python
class SessionActor:
    async def start(self, meeting_id: str, client_session_id: str,
                    channels: Sequence[SessionChannel], sink: EventSink) -> SessionLease: ...
    async def disconnect(self, meeting_id: str, client_session_id: str) -> SessionLease: ...
    async def stop(self, meeting_id: str, client_session_id: str,
                   resume_window_s: Literal[0, 10]) -> SessionLease: ...
    async def delete(self, meeting_id: str) -> None: ...
    async def snapshot(self, meeting_id: str, client_session_id: str) -> SessionSnapshot: ...
    async def update_snapshot(self, meeting_id: str, client_session_id: str,
                              update: SessionSnapshotUpdate) -> SessionSnapshot: ...
    async def shutdown(self, finalize: bool) -> None: ...
```

```ts
type WorkspaceCommand =
  | { type: 'open-anything' | 'ask' | 'start-stop-meeting' | 'back' | 'forward' |
      'find' | 'toggle-transcript' | 'toggle-context' | 'accept-proposal' |
      'reject-proposal' | 'undo' | 'hide-now' | 'toggle-curtain' |
      'toggle-connections-view' | 'toggle-comparison' }
  | { type: 'open-place'; place: PlaceId }

interface DesktopBridge {
  onCommand(listener: (command: WorkspaceCommand) => void): () => void
  reportWorkspaceState(state: WorkspaceState): void
}
```

- [ ] **Step 1: Verify and preserve the assigned worktree state**

Run: `WORKTREE_ROOT="$(git rev-parse --show-toplevel)"; case "$WORKTREE_ROOT" in */.worktrees/ui-quiet-canvas) ;; *) exit 1;; esac; git status --short`

Expected: the paused Task 6 WIP only; if its archive checksum differs from `178274e9ac385af924a391aa705f8a74ef624771438e8540f0e308f059314e80`, stop and report corruption.

- [ ] **Step 2: Replace repair-specific registry tests with the actor's 14-case RED matrix**

Write `test_new_start`, `test_matching_resume_at_59_9`, `test_expiry_at_60_finalizes_once`, `test_stop_resume_at_9_9`, `test_stop_expiry_at_10`, `test_stop_zero`, `test_duplicate_stop_shortens_only`, `test_stale_expiry`, `test_stale_completion`, `test_wrong_token_state_reset`, `test_delete_cancels_tasks`, `test_shutdown_modes`, `test_cancelled_caller_does_not_cancel_command`, and `test_finalizer_failure_preserves_durable_meeting` in `core/tests/test_session_actor.py` using an injected clock and real cooperative `asyncio.Task` objects.

- [ ] **Step 3: Run actor tests and record RED**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_session_actor.py -q`

Expected: FAIL because `taurine_core.live.session_actor` does not exist.

- [ ] **Step 4: Implement the one-loop actor without callbacks or detached owners**

Use a private `asyncio.Queue`, a 256-permit public admission semaphore, one records dictionary, and internal `_Expire`/`_Finalized` messages. Timer/finalizer wrappers use `put_nowait` then return. The actor loop is the only table mutator. Do not port `_detached_entries`, `_track_task`, cleanup handoffs, task factories, or cancellation-resistant fake-task support.

- [ ] **Step 5: Integrate the actor as the server's sole lifecycle owner**

Make Start/Stop/disconnect/delete/shutdown call the actor. Emit `session_snapshot` before resumed deltas, reject binary audio until Start succeeds and after Stop, and preserve `new | resumed | state_reset`. Remove runtime-owned suspended/finalizer dictionaries and socket-owned expiry tasks.

- [ ] **Step 6: Run focused and full Core GREEN gates**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_session_actor.py tests/test_server_ws.py tests/test_live_ingest.py tests/test_final_summary.py -q && uv run pytest -q`

Expected: all pass with no pending-task warnings; the actor matrix is exactly the documented concrete dependency model.

- [ ] **Step 7: Expand the restored desktop tests to exact RED coverage**

Test the full Places/shortcut matrix, undersized/off-screen bounds, compact normal-bound restoration, hidden tray/call start, live quit confirmation/flush grace, command validation/unsubscribe, one frozen preload global, absence of raw `ipcRenderer`/`process`, encrypted/memory-only recovery, corrupt quarantine, and curtain epochs across main/call/suggestion windows.

- [ ] **Step 8: Run desktop/recovery/privacy tests and record RED**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main/__tests__ src/renderer/src/data/__tests__/DraftJournal.test.ts`

Expected: FAIL on unimplemented index wiring, recovery, privacy coordination, and gateway contracts.

- [ ] **Step 9: Complete native window/menu/tray composition**

Create the `1120×780` window with `840×600` minimum, `#F4F2EC` background, clamped atomic window state, user-only Compact mode, and best-effort `setContentProtection(true)`. Build all §3.5 native commands. Tray Start sends the command without `show()`. Confirm live Quit, request Stop, await the renderer's bounded flush acknowledgement for at most three seconds, then quit.

- [ ] **Step 10: Replace preload exposure with one frozen allowlisted API**

Remove `@electron-toolkit/preload`'s generic `electronAPI`. Validate command/state discriminants at runtime. Retain only the existing narrow audio and notification operations plus `desktop`, `recovery`, and `privacy`; every listener returns an unsubscribe function.

- [ ] **Step 11: Implement encrypted recovery and the curtain epoch**

Persist `{schemaVersion,meetingId,writtenAt,ciphertextBase64}` only via sibling temp/`fsync`/rename. Use memory-only when encryption is unsafe. Curtain enable/disable increments a main-owned epoch, sends state before a window is shown, waits for acknowledgements, and hides any window that misses the bounded acknowledgement deadline.

- [ ] **Step 12: Implement strict contracts, one gateway, repositories, and startup**

Decoders reject invalid discriminants, retain additive fields in `extensions`, preserve every evidence trust field, and map 503/409 to typed errors. `SidecarGateway` owns URL, request ID, abort, readiness, and safe errors. Repositories coalesce keyed reads and abort on final subscriber removal. `StartupScene` keeps retry/settings/diagnostics available without rendering private feature routes.

- [ ] **Step 13: Run the foundation GREEN gates**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main src/preload src/renderer/src/data src/renderer/src/bootstrap && npm run typecheck && npm run lint && npm run build`

Expected: PASS; no generic Electron global, no raw feature transport, and a runnable startup surface.

- [ ] **Step 14: Freeze and commit Slice 1**

Run: `git add core app docs/superpowers/plans/2026-07-20-taurine-ui-quiet-canvas-implementation.md && git commit -m "feat(quiet): establish serialized desktop foundation"`

Expected: one frozen commit and clean worktree, ready for independent review.

---

### Task 2: Quiet Shell

**Runnable outcome:** A user can open Taurine, navigate the Desk and meetings collection using labeled native/renderer commands, use Open anything, and operate every shell state at all supported widths without seeing legacy chrome.

**Files:**

- Create: `app/src/renderer/src/design/tokens.css`
- Create: `app/src/renderer/src/design/typography.css`
- Create: `app/src/renderer/src/design/globals.css`
- Create: `app/src/renderer/src/design/primitives/Button.tsx`
- Create: `app/src/renderer/src/design/primitives/IconButton.tsx`
- Create: `app/src/renderer/src/design/primitives/Field.tsx`
- Create: `app/src/renderer/src/design/primitives/Menu.tsx`
- Create: `app/src/renderer/src/design/primitives/Dialog.tsx`
- Create: `app/src/renderer/src/design/primitives/Drawer.tsx`
- Create: `app/src/renderer/src/design/primitives/Disclosure.tsx`
- Create: `app/src/renderer/src/design/primitives/ObjectRow.tsx`
- Create: `app/src/renderer/src/design/primitives/StatusText.tsx`
- Create: `app/src/renderer/src/design/primitives/EvidenceLabel.tsx`
- Create: `app/src/renderer/src/design/primitives/a11y.tsx`
- Create: `app/src/renderer/src/design/primitives/LiveRegion.tsx`
- Create: `app/src/renderer/src/design/primitives/index.ts`
- Create: `app/src/renderer/src/workspace/WorkspaceShell.tsx`
- Create: `app/src/renderer/src/workspace/WorkspaceTitlebar.tsx`
- Create: `app/src/renderer/src/workspace/WorkspaceRouter.ts`
- Create: `app/src/renderer/src/workspace/routeState.ts`
- Create: `app/src/renderer/src/workspace/OpenAnything.tsx`
- Create: `app/src/renderer/src/workspace/StatusPopover.tsx`
- Create: `app/src/renderer/src/workspace/PrivacyCurtain.tsx`
- Create: `app/src/renderer/src/workspace/WorkspaceContext.tsx`
- Create: `app/src/renderer/src/meetings/DeskScene.tsx`
- Create: `app/src/renderer/src/meetings/MeetingsCollectionScene.tsx`
- Create: `app/src/renderer/src/platform/PreferenceStore.ts`
- Create: `app/src/renderer/src/platform/windowPreferences.ts`
- Modify: `app/src/renderer/src/bootstrap/RendererBootstrap.tsx`
- Modify: `app/src/renderer/src/bootstrap/services.ts`
- Modify: `app/src/renderer/src/main.tsx`
- Add colocated `__tests__` files for every module above.

**Interfaces:** `WorkspaceRouter` owns typed route/history snapshots; `OpenAnything` always returns fixed Places before optional ranked data; `WorkspaceServices` constructs one instance of each shared owner and disposes them in reverse order.

- [ ] **Step 1: Write shell RED tests**

Cover every light/dark token, forced colors, reduced motion, primitive focus/name/target geometry, fixed Place order, full shortcuts with editable guards, degraded Open catalog, exact Back/Forward route state, curtain DOM exclusion, Desk empty/active/recent states, meetings keyboard navigation/search/delete focus, and all five responsive thresholds.

- [ ] **Step 2: Run shell RED**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/design src/renderer/src/workspace src/renderer/src/meetings`

Expected: FAIL because target primitives, router, and scenes are absent.

- [ ] **Step 3: Encode exact visual tokens and primitives**

Implement only the approved four button variants and semantic primitives. Use CSS custom properties, visible 2 px focus, 36/44 px targets, semantic HTML, focus traps/return, stable coalesced live regions, and no gradients/blur/glow/bounce/tiny pills.

- [ ] **Step 4: Implement typed history and the 48 px title bar**

Store route plus scroll/focus/drawer snapshots. Open remains text-labeled. Breadcrumb exposes fixed Places. Status shows only actionable exceptions. Ask adapts from labeled control to icon with accessible name. Capture remains a textual safety control.

- [ ] **Step 5: Implement Open anything**

Render fixed Places immediately, then recent objects/commands/ranked results. Use combobox/listbox semantics, stable active descendant, deterministic ranking, safe empty/error states, and no dependence on knowledge readiness for navigation.

- [ ] **Step 6: Implement Desk and meetings collection**

Desk contains Start meeting, optional active meeting, and at most eight recents—no graph/status dashboard. Meetings is a searchable semantic list with Up/Down/Enter/Shift+F10 and inline named deletion confirmation that returns focus to the next row.

- [ ] **Step 7: Compose one service tree and target shell**

Construct shared services once after readiness. Feature hooks receive narrowed ports. The target shell can be launched directly in tests while the production cutover remains atomic in Task 6.

- [ ] **Step 8: Run shell GREEN, axe, type, and bundle-report gates**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/design src/renderer/src/workspace src/renderer/src/meetings && npm run typecheck && npm run lint && npm run build && node scripts/bundle-budget.mjs --report-only`

Expected: PASS; no horizontal document scroll and no BlockNote/force-graph/Diagnostics in the initial graph.

- [ ] **Step 9: Freeze and commit Slice 2**

Run: `git add app && git commit -m "feat(quiet): build the quiet canvas shell"`

Expected: one frozen runnable shell commit, ready for independent review.

---

### Task 3: Meeting Document

**Runnable outcome:** A complete meeting journey works from Start through free editing, capture, reconnect, transcript/context, Stop, durable proposal review, accept/reject, undo, reload, and encrypted draft reconciliation.

**Files:**

- Create: `app/src/renderer/src/data/SessionTransport.ts`
- Create: `app/src/renderer/src/meetings/MeetingScene.tsx`
- Create: `app/src/renderer/src/meetings/MeetingDocument.tsx`
- Create: `app/src/renderer/src/meetings/MeetingDocumentController.ts`
- Create: `app/src/renderer/src/meetings/CaptureController.ts`
- Create: `app/src/renderer/src/meetings/LiveMeetingStore.ts`
- Create: `app/src/renderer/src/meetings/ContextMargin.tsx`
- Create: `app/src/renderer/src/meetings/TranscriptDrawer.tsx`
- Create: `app/src/renderer/src/meetings/ReviewChanges.tsx`
- Create: `app/src/renderer/src/meetings/ProposalQueue.ts`
- Create: `app/src/renderer/src/meetings/proposalAnchor.ts`
- Create: `app/src/renderer/src/meetings/__tests__/MeetingDocumentController.test.ts`
- Create: `app/src/renderer/src/meetings/__tests__/CaptureController.test.ts`
- Create: `app/src/renderer/src/meetings/__tests__/MeetingScene.test.tsx`
- Modify: `app/src/renderer/src/data/MeetingRepository.ts`
- Modify: `app/src/renderer/src/data/DraftJournal.ts`
- Modify: `app/src/renderer/src/workspace/WorkspaceTitlebar.tsx`
- Modify: `app/src/renderer/src/bootstrap/services.ts`

**Interfaces:** `MeetingDocumentController` owns `load`, `applyLocalEdit`, `save`, `acceptProposal`, `rejectProposal`, `undoAcceptedEdit`, and `dispose`. `SessionTransport` is the only `/session` socket owner and emits decoded events into `LiveMeetingStore`.

- [ ] **Step 1: Write controller/transport RED tests**

Test one canonical writer, revision conflicts, idempotent mutations, stale acknowledgements, out-of-order session revisions, same-token resume, Quiet `state_reset`, rejected post-Stop audio, ASR/channel loss, stop flush, journal schedule/flush/replay/clear/conflict, anchor resolution, and no direct chat-to-editor insertion.

- [ ] **Step 2: Write meeting UI RED tests**

Cover empty/loading/error/live/preparing/review/completed states, editor geometry, optional context, transcript follow/search/copy, stable interim live region, actionable controls, tentative retrieval, proposal accept/reject/accept-all confirmation/undo, keyboard shortcuts, focus restoration, and compact/drawer behavior.

- [ ] **Step 3: Run meeting RED**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings src/renderer/src/data/__tests__/SessionTransport.test.ts`

Expected: FAIL on absent controller, transport, and scene modules.

- [ ] **Step 4: Implement the single-writer document controller**

Decode server revisions and mutation receipts; serialize saves; use stable mutation IDs; rebase or surface explicit conflicts; create distinct editor transactions for accepted proposals; persist rejection; expose undo; coordinate dirty-only encrypted recovery; never treat AI text as canonical before acceptance.

- [ ] **Step 5: Implement session and capture controllers**

Create one socket, issue the stable client token, wait for session snapshot before audio, merge transcript by `seq`, summary by `rev`, grounding by `revision`, and actionables by ID. Stop mic/system audio immediately, flush note/journal/notepad context, send resumable Stop, and reconcile resumed/state-reset state without duplicating deltas.

- [ ] **Step 6: Implement the editorial meeting scene**

Render 680–760 px Source Serif note text, optional 264 px context, the title-bar capture control, local save exceptions, requested transcript/context surfaces, and no automatic panel opening. Below thresholds, use accessible drawers/sheets without shrinking the editor under 620 px.

- [ ] **Step 7: Implement review and proposal semantics**

Display durable proposed changes separately from notes. Accept/reject each; confirm Accept all visible; persist actionable decisions; make accepted edits undoable; keep rejected text out after reload; label preparation/failure truthfully without claiming retry where the backend has none.

- [ ] **Step 8: Run meeting GREEN and Core integration**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings src/renderer/src/data && npm run typecheck && npm run lint`

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_session_actor.py tests/test_server_ws.py tests/test_meeting_notes.py tests/test_meeting_patch.py -q`

Expected: PASS with no multiple writer/socket owners and no plaintext recovery.

- [ ] **Step 9: Freeze and commit Slice 3**

Run: `git add app core && git commit -m "feat(quiet): deliver the complete meeting document"`

Expected: one runnable complete meeting journey commit, ready for independent review.

---

### Task 4: Conversation and Evidence

**Runnable outcome:** Meeting Ask and durable standalone conversations stream through one hub, can be stopped/reconnected/regenerated, and render complete provenance-honest evidence that opens the exact source and restores navigation state.

**Files:**

- Create: `app/src/renderer/src/data/ConversationHub.ts`
- Create: `app/src/renderer/src/ask/ObjectAskSheet.tsx`
- Create: `app/src/renderer/src/ask/ConversationScene.tsx`
- Create: `app/src/renderer/src/ask/ConversationTurn.tsx`
- Create: `app/src/renderer/src/ask/EvidenceDisclosure.tsx`
- Create: `app/src/renderer/src/ask/EvidenceLedger.tsx`
- Create: `app/src/renderer/src/ask/EvidenceRef.tsx`
- Create: `app/src/renderer/src/ask/ProcessDisclosure.tsx`
- Create: `app/src/renderer/src/ask/evidenceSemantics.ts`
- Create colocated tests under `app/src/renderer/src/ask/__tests__/` and `app/src/renderer/src/data/__tests__/ConversationHub.test.ts`
- Modify: `app/src/renderer/src/data/contracts/chat.ts`
- Modify: `app/src/renderer/src/data/contracts/evidence.ts`
- Modify: `app/src/renderer/src/workspace/WorkspaceRouter.ts`
- Modify: `app/src/renderer/src/workspace/routeState.ts`
- Modify: `app/src/renderer/src/meetings/MeetingScene.tsx`

**Interfaces:** `ConversationHub` owns the only `/chat` socket and keyed thread reducers. `deriveEvidenceLedger` deduplicates by `Cited > Consulted > Retrieved`, with interrupted partials owning no cited-support set.

- [ ] **Step 1: Write conversation/evidence RED tests**

Test one socket across scopes, out-of-order events, stop/reconnect/resync, durable thread CRUD/rename/delete/regenerate, bounded retained buffers, meeting scope payloads, exact evidence precedence, fallback/prior-context/low-confidence/interrupted/calibration states, unknown additive fields, and proposal-only note edits.

- [ ] **Step 2: Write Ask/navigation RED tests**

Cover sheet focus/Escape/stop behavior, stream-in-place, process summary, exact cited/consulted/retrieved groups, sentence citation activation, source route state and Back restoration, no unsupported verification claims, and responsive bottom-sheet geometry.

- [ ] **Step 3: Run conversation RED**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/ask src/renderer/src/data/__tests__/ConversationHub.test.ts`

Expected: FAIL because the hub and evidence components are absent.

- [ ] **Step 4: Implement one renderer-wide ConversationHub**

Decode and reduce all stream events by thread/turn IDs; reconnect with authoritative resync; abort final subscribers; cap transient activity; persist only through repository endpoints; expose selectors/actions through narrowed hooks.

- [ ] **Step 5: Implement contract-honest evidence**

Preserve passage date, fallback, prior context, low confidence, evidence policy, citation check, message provenance, fact/path support, and extensions. Withhold unsupported facts/paths under fallback. Never elevate consulted/retrieved items to sentence support.

- [ ] **Step 6: Implement Ask and conversation scenes**

Meeting Ask names its scope; standalone conversations are 760 px centered documents. Render completed/partial/interrupted states, Stop, reconnect, regenerate, rename/delete, evidence disclosure, and durable note proposals without editor mutation.

- [ ] **Step 7: Run conversation GREEN and accessibility gates**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/ask src/renderer/src/data && npm run typecheck && npm run lint`

Expected: PASS; exact evidence vocabulary and route restoration are covered.

- [ ] **Step 8: Freeze and commit Slice 4**

Run: `git add app && git commit -m "feat(quiet): add conversations and truthful evidence"`

Expected: one runnable Ask/conversation commit, ready for independent review.

---

### Task 5: Knowledge and Settings

**Runnable outcome:** Users can inspect knowledge notes, operate a complete accessible Connections List and lazy stable visual graph, compare two or three objects, manage Sources and personal context, configure privacy/appearance/shortcuts, inspect redacted diagnostics, and use safe native peripheral prompts.

**Files:**

- Create: `app/src/renderer/src/knowledge/KnowledgeNoteScene.tsx`
- Create: `app/src/renderer/src/knowledge/ConnectionsScene.tsx`
- Create: `app/src/renderer/src/knowledge/ConnectionsList.tsx`
- Create: `app/src/renderer/src/knowledge/ConnectionsVisual.lazy.tsx`
- Create: `app/src/renderer/src/knowledge/ConnectionFilters.tsx`
- Create: `app/src/renderer/src/knowledge/connectionsModel.ts`
- Create: `app/src/renderer/src/knowledge/CompareScene.tsx`
- Create: `app/src/renderer/src/knowledge/comparisonSet.ts`
- Create: `app/src/renderer/src/knowledge/connectionLayout.worker.ts`
- Create: `app/src/renderer/src/settings/SettingsIndexScene.tsx`
- Create: `app/src/renderer/src/settings/SourcesScene.tsx`
- Create: `app/src/renderer/src/settings/PersonalContextScene.tsx`
- Create: `app/src/renderer/src/settings/PrivacyCaptureScene.tsx`
- Create: `app/src/renderer/src/settings/AppearanceScene.tsx`
- Create: `app/src/renderer/src/settings/ShortcutsScene.tsx`
- Create: `app/src/renderer/src/settings/DiagnosticsScene.lazy.tsx`
- Create: `app/src/main/local-telemetry.ts`
- Create: `app/src/renderer/src/telemetry/productEvents.ts`
- Modify: `app/src/main/call-notify-state.ts`
- Modify: `app/src/main/notifier.ts`
- Modify: `app/src/main/suggest-notifier.ts`
- Modify: `app/src/renderer/src/notif-main.ts`
- Modify: `app/src/renderer/src/suggest-main.ts`
- Modify: `app/src/renderer/notif.html`
- Modify: `app/src/renderer/suggest.html`
- Modify: `app/src/renderer/src/data/KnowledgeRepository.ts`
- Modify: `app/src/renderer/src/data/ConnectorRepository.ts`
- Modify: `app/src/renderer/src/data/ProfileRepository.ts`
- Modify: `app/src/renderer/src/data/DiagnosticsRepository.ts`
- Modify: `app/src/renderer/src/workspace/WorkspaceRouter.ts`
- Modify: `app/src/renderer/src/workspace/routeState.ts`
- Add colocated unit/component/main-process tests.

**Interfaces:** Connections List is the canonical graph interface. Visual consumes the same semantic model and stable worker positions. `comparisonSet` holds 0–3 route-independent ephemeral object refs. Telemetry accepts only the closed content-free event vocabulary from design §14.2.

- [ ] **Step 1: Write knowledge model/list RED tests**

Test graph generation swap, stable keys/positions, depth 1–3 local requests, cluster/source/date/mechanism filters, relation/source/date accessibility text, keyboard list navigation/expand/open/compare, idea/no-result/error/abort, and no edge-reason action without passage IDs.

- [ ] **Step 2: Write visual/comparison RED tests**

Test lazy imports, no canvas in accessibility tree, stable layout independent of input order, selected/pinned/focus state across generation swap, reduced motion, list/visual parity, cap of three comparisons, ≥1320 three-column rule, ≥400 px dossiers, and contract-valid Ask seeds.

- [ ] **Step 3: Write settings/peripheral/telemetry RED tests**

Cover connector auth/config/sync/delete/purge truth, profile edit/generate conflicts, privacy wording, appearance, full shortcuts, lazy redacted diagnostics, telemetry key/event rejection and rotation, explicit-start call policy, default-off suggestions, curtain/presentation redaction, and no emoji/private peripheral titles.

- [ ] **Step 4: Run Slice 5 RED**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge src/renderer/src/settings src/renderer/src/telemetry src/main`

Expected: FAIL on absent target scenes and policies.

- [ ] **Step 5: Implement knowledge notes and canonical Connections List**

Render source text/provenance/why-related/open/compare. List exposes every essential operation semantically, including scoped expansion and typed filters. Use exact `GET /graph/local?center={note_id}&depth={1..3}` and do not fabricate edge provenance.

- [ ] **Step 6: Implement lazy stable visual and comparison**

Load force graph/CSS only on Visual. Settle layout in a worker from stable keys and preserve interaction across active-build swaps. Keep canvas decorative to semantic controls. Compare 2–3 dossiers responsively and send at most one meeting ID plus distinct valid note seeds.

- [ ] **Step 7: Implement settings, diagnostics, and preferences**

Use plain settings rows. Persist only versioned theme, graph representation/pins/filters, auto-start/countdown, and notification policy; drop legacy tab/panel state. Load Diagnostics only on route and exclude titles, IDs, paths, transcript/query/evidence/model/OAuth content from default copy/export.

- [ ] **Step 8: Implement bounded local telemetry and peripheral parity**

Validate the closed event/payload allowlists in renderer, preload, and main; rotate local-only logs; provide no upload. Call prompt is 356×116, explicit Start by default, non-focusable, and does not raise main. Suggestion prompt is 356×132, default off, queue-bounded, and neutral under curtain/presentation-safe mode.

- [ ] **Step 9: Run Slice 5 GREEN, type, lint, and lazy-bundle report**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge src/renderer/src/settings src/renderer/src/telemetry src/main && npm run typecheck && npm run lint && npm run build && node scripts/bundle-budget.mjs --report-only`

Expected: PASS; force graph and Diagnostics are absent from initial imports.

- [ ] **Step 10: Freeze and commit Slice 5**

Run: `git add app && git commit -m "feat(quiet): complete knowledge and settings journeys"`

Expected: one runnable complete feature commit, ready for independent review.

---

### Task 6: Cutover and Proof

**Runnable outcome:** Quiet Canvas is the only shipped renderer; all legacy UI is deleted; complete automated and manual evidence proves functionality, accessibility, privacy, platform behavior, performance, bundle limits, and data preservation.

**Files:**

- Create: `app/scripts/verify-quiet-canvas.mjs`
- Create: `app/scripts/verify-quiet-canvas.test.mjs`
- Create: `app/scripts/bundle-budget.mjs`
- Create: `app/scripts/bundle-budget.test.mjs`
- Create: `app/playwright.config.ts`
- Create: `app/e2e/fake-sidecar.ts`
- Create: `app/e2e/electron-fixture.ts`
- Create: `app/e2e/quiet-canvas.spec.ts`
- Create: `app/e2e/accessibility.spec.ts`
- Create: `app/e2e/performance.spec.ts`
- Create: `app/e2e/manual-platform-matrix.md`
- Create: `docs/verification/quiet-canvas-release.md`
- Modify: `app/src/renderer/src/main.tsx`
- Modify: `app/src/renderer/src/assets/main.css`
- Modify: `app/src/renderer/index.html`
- Modify: `app/package.json`
- Delete: every legacy file enumerated in design §15 and rejected by `verify-quiet-canvas.mjs`.

**Interfaces:** `main.tsx` renders only `RendererBootstrap`. The verifier rejects legacy paths/imports/copy/selectors, raw transports, direct editor mutations, forbidden evidence/privacy claims, missing target modules, or multiple shared owners.

- [ ] **Step 1: Write verifier and bundle-budget RED fixture tests**

Each rule gets a positive and negative fixture: legacy entry/path/copy, generic Electron API, feature fetch/WebSocket, multiple transport owners, direct editor insertion, forbidden `Verified`/invisibility claims, initial heavy imports, oversized initial/feature chunks, and missing lazy chunks.

- [ ] **Step 2: Write full Electron E2E stories before cutover**

Cover startup/down/retry; tray/call/Desk start; type/capture/transcript/Ask/proposal/reconnect/Stop/review/reload; recovery/conflict/corrupt memory-only; conversations/evidence/source/Back; comparison; Connections list/visual/build swap; Sources/profile/privacy/diagnostics; curtain across windows; service and graph failures; call debounce/rearm.

- [ ] **Step 3: Write accessibility/performance RED suites**

Run axe and keyboard journeys across every scene/state/theme at 840/960/1180/1320/1440 widths; add 200% zoom, forced colors, reduced motion, live regions, target geometry, no horizontal page scroll. Measure 20 warm launches, route fallback, editor input, long note/transcript DOM bounds, and Connections interaction.

- [ ] **Step 4: Run cutover/E2E RED**

Run: `cd "$(git rev-parse --show-toplevel)/app" && node --test scripts/verify-quiet-canvas.test.mjs && node scripts/verify-quiet-canvas.mjs && npm run test:e2e`

Expected: verifier fixtures pass while repository/E2E fail on the legacy entry and incomplete cutover.

- [ ] **Step 5: Cut over atomically and delete legacy UI**

Make `main.tsx` construct target services and render only `RendererBootstrap`. Limit `assets/main.css` to target design imports. Delete legacy App/tabs/views/components/libs/assets and obsolete dependencies/preferences. Do not retain a hidden compatibility component or feature flag.

- [ ] **Step 6: Complete package gates and deterministic E2E fixture**

Add `test:all`, `verify:quiet-canvas`, `check:bundle`, `test:e2e`, `test:e2e:a11y`, and `test:e2e:perf`. Fake sidecar must serve production-shaped REST plus `/updates`, `/session`, `/chat` with controllable clock, disconnect, conflicts, finalization, failures, and graph generations.

- [ ] **Step 7: Complete platform and privacy matrix**

Record minimum/latest macOS and Windows behavior for permissions, hidden/visible prompts, tray, bounds/displays, all-Spaces, restart/recovery modes, sleep/wake/provider loss, Zoom/Meet/Teams/OBS/native screenshot/share preview. Every required row is Pass or a linked release blocker; privacy wording remains best-effort regardless of observed capture behavior.

- [ ] **Step 8: Run complete app release gates**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm run format:check && npm run lint && npm run typecheck && npm run verify:quiet-canvas && npm run test:all && npm run build && npm run check:bundle && npm run test:e2e && npm run test:e2e:a11y && npm run test:e2e:perf`

Expected: PASS with exact counts, zero critical axe violations, all budgets met, and all 20-launch percentiles within targets.

- [ ] **Step 9: Run complete Core release gate**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest -q && uv run ruff check taurine_core tests && uv run python -m compileall -q taurine_core`

Expected: PASS with no pending session tasks and no new undocumented skips.

- [ ] **Step 10: Audit durable data and content-free outputs**

Run the app against a copy of a pre-redesign database/knowledge fixture. Open meetings, notes, transcripts, suggestions, chats, profile, connectors, and graph data without migration. Exercise persistent/memory-only/corrupt recovery. Search ciphertext text, default diagnostics, and telemetry for fixture secrets.

Expected: durable objects open unchanged; fixture secrets appear in none of those outputs.

- [ ] **Step 11: Write the release record**

Record command outputs, versions, counts, bundle table, performance percentiles, platform matrix, commit SHA, and every design §17.3 acceptance item mapped to measured automated/manual proof. The record contains no private content.

- [ ] **Step 12: Freeze and commit Slice 6**

Run: `git add -A app core docs/verification docs/superpowers/plans/2026-07-20-taurine-ui-quiet-canvas-implementation.md && git commit -m "feat(quiet): cut over and verify quiet canvas"`

Expected: clean worktree, sole Quiet Canvas renderer, complete proof artifacts, ready for final independent review.

## Completion Audit

Before claiming the branch complete, map every normative section of `2026-07-20-taurine-ui-quiet-canvas-design.md` to a production file and an automated/manual proof. A passing suite is insufficient if a design requirement lacks direct evidence. Confirm the reset acceptance criteria: one serialized actor, no detached/callback cleanup machinery, all 14 lifecycle cases, six reviewed slices, visible UI delivery, and exhaustive final verification.

| Quiet Canvas design coverage | Owning slice | Required direct proof |
|---|---:|---|
| §§1–3 product thesis, objects, visual language, fixed navigation, shortcuts | 2 | token/primitive/router tests, native-menu parity, responsive shell E2E |
| §4 complete before/during/after/return meeting journey | 3 | controller/session tests and complete meeting E2E |
| §§5.1–5.5 Desk, meetings, document, transcript, Ask | 2–4 | component/axe tests and keyboard-only journeys |
| §§5.6–5.10 conversations, notes, Connections, comparison, settings | 4–5 | hub/evidence/knowledge/settings tests and E2E |
| §§6–8 AI boundaries, evidence language, responsive behavior | 2–5 | proposal-only mutation scan, evidence table tests, viewport/zoom suite |
| §§9–10 navigation, persistence, encrypted recovery | 1–3 | route restoration and recovery byte/quarantine/conflict tests |
| §§11–13 accessibility, privacy, platform behavior | 1–6 | axe/keyboard/curtain suites and signed manual matrix |
| §§14–16 telemetry, file boundaries, phased cutover | 1–6 | allowlist tests, ownership scan, sole-entry verifier |
| §17 acceptance and release proof | 6 | release record with one evidence link per acceptance item |
