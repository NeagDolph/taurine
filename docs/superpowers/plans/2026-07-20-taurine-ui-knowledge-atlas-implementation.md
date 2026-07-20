# Taurine Knowledge Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Taurine's five-tab Electron renderer with the production Knowledge Atlas workstation while preserving every current capability and adding durable note proposals, resumable live sessions, and desktop OAuth.

**Architecture:** Keep FastAPI/SQLite as the durable product spine, add transactional meeting-mutation and OAuth services, and retain the three existing WebSocket protocols behind one typed renderer gateway. Replace `App.tsx` coordination with Redux Toolkit domain slices, listener-owned side effects, a sole meeting document writer, three shared-context coordinates, and lazy Atlas/editor/Markdown feature chunks; Electron main owns windows, permissions, privacy, recovery, and bounded IPC.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLite, pytest/pytest-asyncio, keyring; Electron 39, React 19, TypeScript 5.9, Redux Toolkit/RTK Query, BlockNote, Canvas + d3-force, React Virtual, Vitest, Testing Library, axe-core, and Playwright Electron.

## Global Constraints

- Release one completed Knowledge Atlas interface: no runtime feature flag, no dual renderer, no legacy route, no compatibility UI, and no user-facing migration program.
- Preserve readable `taurine.db`, `taurine-knowledge.db`, meeting/chat/source rows, and current REST/WebSocket data; database work is additive DDL initialized in place.
- Preserve the verified floors of 981 passing core tests with 2 skipped and 61 passing Vitest tests; all new tests add to those floors.
- Python remains `>=3.12`; Electron remains `^39.2.6`, React `^19.2.1`, and TypeScript `^5.9.3` unless an explicit dependency update is reviewed separately.
- Use Redux Toolkit as the only renderer global-state library; sockets, fetches, capture sequencing, and editor writes do not live in React views.
- `MeetingDocumentController` is the only frontend caller allowed to mutate canonical meeting Markdown; every AI-authored note path first persists a `MeetingNoteProposal`.
- Audio is never queued or replayed. Disconnect resume preserves bounded live reasoning state for 60 seconds; explicit Stop ceases capture immediately and offers a 10-second server-owned resume window.
- Do not add graph pagination, a unified WebSocket, event replay IDs, final-summary retry, profile cancellation, or hard retrieval scoping.
- `/graph/edge-reason` is not called from projected note edges because `GraphEdge` lacks exact passage endpoints.
- Default/minimum main-window bounds are 1280×820 and 960×640 CSS px; ordinary Live is two lanes, and optional three-lane Live begins at 1440 px.
- Initial renderer shell is at most 1.25 MB uncompressed JavaScript and 140 KB uncompressed CSS; each lazy feature chunk is at most 900 KB without a checked-in bundle analysis.
- Atlas sustains at least 30 fps at 5,000 projected nodes on the reference Mac; virtualized lists sustain 60 fps; selection/command response p95 is below 100 ms excluding model/network work.
- Every rendered surface meets WCAG 2.2 AA, supports keyboard-only operation, 200% zoom, reduced motion, forced colors, 35% string expansion, and RTL; primary/destructive/meeting controls target 40–44 px and all targets meet 24×24 px.
- Copy never claims content protection makes Taurine invisible or uncapturable. Privacy state is exactly Requested, Limited on this Mac, Unavailable, or Privacy curtain on.
- Evidence copy preserves Cited, Consulted, Retrieved, fallback, abstained, interrupted, historical, conflicted, and unavailable distinctions. Never render “verified,” “fact checked,” or numeric confidence as a primary truth claim.
- Never fabricate progress. Determinate progress requires real `done/total`; model/runtime operations without totals remain indeterminate.
- Production Redux DevTools are disabled; logs and error breadcrumbs exclude note, transcript, profile, passage, prompt, answer, token, and credential content.
- Keep `app/src/renderer/src/app/App.tsx` below approximately 120 substantive lines after the final cutover.
- Execute only inside the assigned `ui-knowledge-atlas` linked worktree. At the start of the implementation session, discover and validate its root exactly once:

  ```bash
  export WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
  case "$(git rev-parse --git-dir)" in
    *"/worktrees/ui-knowledge-atlas") ;;
    *) echo "Refusing to run outside the assigned ui-knowledge-atlas linked worktree" >&2; exit 1 ;;
  esac
  cd "$WORKTREE_ROOT"
  ```

- Before every command block in this plan, run `cd "$WORKTREE_ROOT"`; then execute the displayed relative `cd app`, `cd core`, Git, and path commands verbatim. Never substitute the primary checkout or another worktree for `WORKTREE_ROOT`.

---

## Repository baseline and delivery boundaries

The current renderer is coordinated by `app/src/renderer/src/App.tsx` (852 lines), with old views under `views/`, old chrome under `components/`, and three separate clients in `lib/sidecar.ts`, `lib/updates.ts`, and `lib/chat.ts`. The current main window is built in `app/src/main/index.ts`; utility windows live in `notifier.ts` and `suggest-notifier.ts`; preload exposes one broad `window.api` object. Core meeting storage is in `core/taurine_core/indexer/store.py`, WebSocket/finalization ownership is in `core/taurine_core/server.py`, and schemas are in `core/taurine_core/schemas.py`. OAuth token reads are embedded in `gmail.py`, `outlook.py`, and `google_common.py`.

Implementation is internally staged but externally atomic. Tasks 1–25 build reviewed modules and contract tests without exposing a second UI. Task 26 switches the sole renderer entry to Knowledge Atlas. Tasks 27–28 prove the full flow, remove every old UI file, and record the release gate.

## Target file map

### Core files

| Path | Action | Responsibility |
|---|---|---|
| `core/taurine_core/schemas.py` | Modify | Pydantic proposal, command, session-resume, OAuth, and enriched connector contracts |
| `core/taurine_core/indexer/store.py` | Modify | additive proposal/operation tables, revision-checked writes, chat referential cleanup |
| `core/taurine_core/meetings/__init__.py` | Create | meeting-mutation package boundary |
| `core/taurine_core/meetings/note_mutations.py` | Create | proposal validation, atomic command application, inverse/undo, retention |
| `core/taurine_core/live/session_registry.py` | Create | active/suspended ownership, deadlines, snapshot, one-winner resume/finalize claims |
| `core/taurine_core/live/final_summary.py` | Modify | materialize final recap/additions as durable proposals in the final transaction |
| `core/taurine_core/chat/engine.py` | Modify | persist meeting-chat proposals before `chat_notes_edit` emit |
| `core/taurine_core/server.py` | Modify | REST routes, session controls, registry lifecycle, OAuth routes, cleanup |
| `core/taurine_core/connectors/credentials.py` | Create | `CredentialStore`, keyring implementation, in-memory test store, env fallback |
| `core/taurine_core/connectors/oauth.py` | Create | provider registry, PKCE/state, exact loopback callback, flow expiry/exchange |
| `core/taurine_core/connectors/gmail.py` | Modify | credential-provider token refresh |
| `core/taurine_core/connectors/outlook.py` | Modify | credential-provider token refresh and rotated-token persistence |
| `core/taurine_core/connectors/google_common.py` | Modify | shared Google credential-provider token refresh |
| `core/taurine_core/connectors/google_calendar.py` | Modify | injected shared Google token provider |
| `core/taurine_core/connectors/google_docs.py` | Modify | injected shared Google token provider |
| `core/taurine_core/connectors/registry.py` | Modify | OAuth provider identity/status and credential injection |
| `core/taurine_core/connectors/catalog.py` | Modify | materialize OAuth connectors from runtime credential services |
| `core/pyproject.toml` / `core/uv.lock` | Modify | keyring dependency lock |
| `core/tests/test_meeting_note_mutations.py` | Create | proposal/revision/command/undo/storage coverage |
| `core/tests/test_session_registry.py` | Create | registry unit and race coverage |
| `core/tests/test_meeting_session_resume.py` | Create | `/session` continuation integration |
| `core/tests/test_connector_oauth.py` | Create | OAuth routes/provider/vault redaction |
| `core/tests/test_meetings.py` | Modify | new REST detail/revision/final/chat-delete behavior |
| `core/tests/test_chat_meeting_context.py` | Modify | chat proposal persistence-before-emit |
| `core/tests/test_final_summary.py` | Modify | final proposals share final-record transaction |

### Electron main and preload files

| Path | Action | Responsibility |
|---|---|---|
| `app/src/main/index.ts` | Rewrite | lifecycle composition only |
| `app/src/main/windows/main-window.ts` | Create | main bounds/restore/hide/fullscreen/compact |
| `app/src/main/windows/window-state.ts` | Create | geometry-only persistence and visible-work-area clamp |
| `app/src/main/windows/call-prompt-window.ts` | Create | 372×136 non-activating call prompt |
| `app/src/main/windows/context-suggestion-window.ts` | Create | 372×188 bounded/redactable suggestion queue |
| `app/src/main/windows/window-privacy.ts` | Create | content-protection request/report and curtain broadcast |
| `app/src/main/capture/system-audio.ts` | Create | moved `sysaudio.ts` implementation |
| `app/src/main/capture/micwatch.ts` | Create | moved micwatch process/debouncer |
| `app/src/main/permissions/capabilities.ts` | Create | microphone/system/call/content capability report |
| `app/src/main/permissions/system-settings.ts` | Create | allowlisted OS settings launch |
| `app/src/main/recovery/drafts.ts` | Create | safeStorage recovery draft and live-session metadata |
| `app/src/main/ipc.ts` | Create | narrow validated IPC registration |
| `app/src/main/tray.ts` | Rewrite | exact dynamic Knowledge Atlas tray state/actions |
| `app/src/preload/contracts.ts` | Create | complete `DesktopBridge` types |
| `app/src/preload/index.ts` | Rewrite | validated bridge implementation |
| `app/src/preload/index.d.ts` | Modify | `window.desktop` global declaration |
| `app/src/main/notifier.ts`, `suggest-notifier.ts`, `sysaudio.ts`, `micwatch.ts` | Delete in Task 28 | superseded old ownership |

### Renderer application and services

| Path | Action | Responsibility |
|---|---|---|
| `app/src/renderer/src/app/App.tsx` | Create | providers and `AppShell` only |
| `app/src/renderer/src/app/AppShell.tsx` | Create | coordinate rail/index/workbench/inspector/shelf/operations composition |
| `app/src/renderer/src/app/store.ts` | Create | Redux store, listener middleware, redaction, typed hooks |
| `app/src/renderer/src/app/listeners.ts` | Create | cross-domain start/stop/hydrate/reconcile orchestration |
| `app/src/renderer/src/app/commands.ts` | Create | command registry and shortcut precedence |
| `app/src/renderer/src/app/hydration.ts` | Create | startup backoff, domain hydration, restore validation |
| `app/src/renderer/src/services/sidecar/contracts.ts` | Create | exhaustive wire/domain types and parsers |
| `app/src/renderer/src/services/sidecar/rest-api.ts` | Create | RTK Query endpoints/cache tags/conflict errors |
| `app/src/renderer/src/services/sidecar/gateway.ts` | Create | one sidecar supervisor and connection epochs |
| `app/src/renderer/src/services/sidecar/updates-connection.ts` | Create | `/updates` lifecycle |
| `app/src/renderer/src/services/sidecar/session-connection.ts` | Create | `/session`, binary audio, resume controls |
| `app/src/renderer/src/services/sidecar/chat-connection.ts` | Create | lazy `/chat`, no resend |
| `app/src/renderer/src/services/sidecar/reconciliation.ts` | Create | snapshot and reconnect merge rules |
| `app/src/renderer/src/services/desktop/bridge.ts` | Create | `window.desktop` adapter |
| `app/src/renderer/src/services/desktop/capabilities.ts` | Create | capability-to-copy selectors |
| `app/src/renderer/src/main.tsx` | Modify | load fonts/tokens, Redux provider, target `App` |

### Renderer features and shared UI

Create the feature tree specified below. Each directory owns `state.ts`, `selectors.ts`, and tests next to the behavior it implements.

```text
app/src/renderer/src/
  features/
    runtime/        RuntimeGate.tsx FirstRunChecklist.tsx OperationLedger.tsx state.ts selectors.ts
    workspace/      CoordinateRail.tsx CommandDeck.tsx CommandCenter.tsx QuickOpen.tsx
                    ContextShelf.tsx ComparisonBoard.tsx ControlWorkbench.tsx state.ts selectors.ts
    atlas/          AtlasWorkbench.tsx AtlasCanvas.tsx AtlasOutline.tsx AtlasIndex.tsx
                    AtlasControls.tsx TimeBand.tsx atlas-worker.ts layout.ts render.ts
                    contracts.ts state.ts selectors.ts storage.ts
    meetings/       MeetingsIndex.tsx MeetingDossier.tsx LiveDesk.tsx CompactLive.tsx
                    CaptureHeader.tsx TranscriptLane.tsx NotesDocument.tsx
                    MeetingDocumentController.ts ProposalQueue.ts LiveIntelligence.tsx
                    FinalRecord.tsx ProposalDiff.tsx state.ts listeners.ts selectors.ts contracts.ts
      editor/       blocknote-adapter.ts provenance.ts markdown-source.tsx
    inquiry/        InquiryIndex.tsx InquiryWorkbench.tsx InquiryComposer.tsx
                    AssistantTurn.tsx ProcessTrace.tsx MessageMarkdown.tsx
                    state.ts listeners.ts selectors.ts contracts.ts
    evidence/       EvidenceInspector.tsx EvidenceLedger.tsx PassageView.tsx
                    FactView.tsx PathView.tsx normalize.ts selectors.ts
    sources/        SourcesIndex.tsx SourcesLedger.tsx SourceDetail.tsx SourceConfig.tsx
                    Authorization.tsx descriptors.ts state.ts listeners.ts selectors.ts
    profile/        ProfileWorkbench.tsx ProfileEditor.tsx GenerationRail.tsx state.ts listeners.ts
    privacy/        PrivacyCurtain.tsx PrivacyStatus.tsx state.ts
    notifications/  ContextQueue.tsx state.ts
    diagnostics/    DiagnosticsWorkbench.tsx SessionLog.tsx
  components/
    primitives/     Button.tsx Field.tsx Menu.tsx Dialog.tsx Sheet.tsx Notice.tsx
                    Tooltip.tsx Progress.tsx Skeleton.tsx Divider.tsx
    layout/         SplitPane.tsx VirtualList.tsx IndexPane.tsx FocusZone.tsx
    knowledge/      ObjectRow.tsx SourceMark.tsx EvidenceRef.tsx DecisionLabel.tsx
                    FactRow.tsx PathView.tsx DiffProposal.tsx
  i18n/             en.ts format.ts pseudo.ts useMessage.ts
  styles/           tokens.css reset.css typography.css shell.css primitives.css print.css
  assets/sources/   apple-notes.svg obsidian.svg imessage.svg gmail.svg outlook.svg
                    google-calendar.svg google-docs.svg folders.svg claude.svg codex.svg taurine.svg
```

### Test and build infrastructure

| Path | Action | Responsibility |
|---|---|---|
| `app/package.json` / `app/package-lock.json` | Modify | state/UI/test/performance dependencies and scripts |
| `app/vitest.config.ts` | Modify | setup, coverage groups, deterministic browser globals |
| `app/src/renderer/src/test/setup.ts` | Create | Testing Library cleanup and DOM shims |
| `app/electron.vite.config.ts` | Modify | manifest, named lazy chunks, three renderer entries |
| `app/playwright.config.ts` | Create | Electron projects, viewport/theme/zoom matrix, artifacts |
| `app/e2e/fixtures/fake-sidecar.ts` | Create | deterministic REST and three-WebSocket fixture |
| `app/e2e/fixtures/scenarios.ts` | Create | meetings/evidence/graph/source/profile datasets and faults |
| `app/e2e/knowledge-atlas.spec.ts` | Create | acceptance stories A–N |
| `app/e2e/accessibility.spec.ts` | Create | axe, keyboard, focus, zoom, pseudo-locale, RTL |
| `app/e2e/visual.spec.ts` | Create | stabilized screenshot matrix |
| `app/e2e/performance.spec.ts` | Create | 5k/10k/3k/200 datasets and p95/frame/heap assertions |
| `app/scripts/check-bundle-budget.mjs` | Create | uncompressed initial/feature/CSS budget enforcement |
| `app/scripts/check-old-ui-removed.mjs` | Create | reject deleted imports/routes/selectors/theme motifs |
| `app/scripts/check-sensitive-logs.mjs` | Create | static rejection of content in logger/breadcrumb calls |
| `app/src/renderer/src/App.tsx`, `assets/main.css`, old `views/`, old `components/`, superseded `lib/` UI coordinators | Delete in Task 28 | remove old interface completely |

## Canonical cross-task interfaces

These names are fixed. A task that changes one must update every consumer and contract test in the same commit.

```ts
type KnowledgeCoordinate = "atlas" | "chronicle" | "inquiry"
type ControlSection = "sources" | "profile" | "privacy" | "appearance" | "shortcuts" | "diagnostics"
type CapturePhase =
  | "idle" | "preparing" | "live_full" | "live_mic_only" | "reconnecting"
  | "stopped_resumable" | "ending" | "finalizing" | "done" | "finalization_error" | "failed"

type SessionIdentity = {
  meetingId: string
  clientSessionId: string
  channels: Array<{id: number; label: string}>
}

type ProposalSource = {
  chatId?: string
  messageId?: number
  summaryRevision?: number
  actionableId?: string
  evidenceIds?: string[]
}

type MeetingNoteProposal = {
  proposalId: string
  meetingId: string
  origin: "final_summary" | "meeting_chat" | "inline_ask" | "rolling_summary" | "actionable"
  baseNotesRev: number
  anchor: string | null
  occurrence: number | null
  deleteMd: string
  insertMd: string
  rationale: string
  supersedesProposalId: string | null
  source: ProposalSource
  status: "pending" | "accepted" | "rejected" | "superseded"
  createdAt: number
  decidedAt: number | null
  decisionCommandId: string | null
}

type ProposalPlacement = {anchor: string | null; occurrence: number | null}
type UserNoteCommand = {
  commandId: string
  kind: "insert" | "replace"
  baseNotesRev: number
  anchor: string | null
  occurrence: number | null
  deleteMd: string
  insertMd: string
}

type SessionStartWire = {
  type: "start"
  meeting_id: string
  client_session_id: string
  channels: Array<{id: number; label: string}>
}
type SessionStopWire = {
  type: "stop"
  meeting_id: string
  client_session_id: string
  resume_window_s: 10 | 0
}
type SessionNotepadWire = {
  type: "notepad"
  meeting_id: string
  client_session_id: string
  text_md: string
}
type SessionSnapshotWire = {
  type: "session_snapshot"
  meeting_id: string
  client_session_id: string
  disposition: "new" | "resumed" | "state_reset" | "stopped_resumable"
  resume_deadline: number | null
  asr: "disconnected" | "connecting" | "connected" | "error"
  channels: Array<{id: number; label: string}>
  transcript_seq: number
  summary: SummaryEventWire | null
  grounding: LiveGroundingEventWire | null
  actionables: ActionableWire[]
  note_suggestions: NoteSuggestionEventWire[]
}
type SessionErrorWire = {
  type: "session_error"
  meeting_id: string | null
  client_session_id: string | null
  code: "meeting_session_conflict" | "meeting_not_resumable" | "invalid_session_control"
  detail: string
}

interface MeetingDocumentController {
  replaceFromEditor(notesMd: string): void
  insertUserText(command: UserNoteCommand): Promise<Meeting>
  decideProposal(proposalId: string, decision: "accept" | "reject"): Promise<Meeting>
  supersedeProposal(proposalId: string, placement: ProposalPlacement): Promise<MeetingNoteProposal>
  undo(commandId: string): Promise<Meeting>
  flush(): Promise<void>
  resolveConflict(choice: "mine" | "server" | "copy_both"): Promise<void>
}

interface SidecarGateway {
  start(): void
  stop(): void
  session: SessionConnection
  updates: UpdatesConnection
  chat: ChatConnection
  subscribe(listener: (event: GatewayEvent) => void): () => void
  getEpoch(protocol: "session" | "updates" | "chat"): number
}
```

```py
class ProposalSource(BaseModel):
    chat_id: str | None = None
    message_id: int | None = None
    summary_revision: int | None = None
    actionable_id: str | None = None
    evidence_ids: list[str] | None = None

class NotesConflict(Exception):
    meeting_id: str
    expected_rev: int
    actual_rev: int

class MeetingNoteProposal(BaseModel):
    proposal_id: str
    meeting_id: str
    origin: Literal["final_summary", "meeting_chat", "inline_ask", "rolling_summary", "actionable"]
    base_notes_rev: int
    anchor: str | None
    occurrence: int | None
    delete_md: str
    insert_md: str
    rationale: str
    supersedes_proposal_id: str | None
    source: ProposalSource
    status: Literal["pending", "accepted", "rejected", "superseded"]
    created_at: float
    decided_at: float | None
    decision_command_id: str | None

class NoteCommandRequest(BaseModel):
    command_id: str
    kind: Literal["insert", "replace", "proposal_decision", "undo"]
    base_notes_rev: int
    anchor: str | None = None
    occurrence: int | None = None
    delete_md: str = ""
    insert_md: str = ""
    proposal_id: str | None = None
    decision: Literal["accept", "reject"] | None = None
    target_command_id: str | None = None

class CredentialStore(Protocol):
    def get(self, provider: str, account_id: str) -> OAuthCredential | None: ...
    def put(self, provider: str, account_id: str, credential: OAuthCredential) -> None: ...
    def delete(self, provider: str, account_id: str) -> None: ...

class SessionRegistry(Protocol):
    async def start(self, meeting_id: str, client_session_id: str, channels: list[SessionChannel]) -> SessionLease: ...
    async def suspend(self, meeting_id: str, client_session_id: str, reason: str, resume_window_s: int) -> SessionLease: ...
    async def stop(self, meeting_id: str, client_session_id: str, resume_window_s: Literal[0, 10]) -> SessionLease: ...
    async def finalize_due(self, meeting_id: str, client_session_id: str, generation: int) -> bool: ...
```

Wire messages remain snake_case and are exhaustively parsed into camel-case domain objects. Unknown message discriminants are logged by name and epoch only, then trigger a bounded snapshot reconciliation; message payloads never enter logs.

---

### Task 1: Lock the test harness, dependencies, and budgets

**Files:**
- Modify: `app/package.json`
- Modify: `app/package-lock.json`
- Modify: `app/vitest.config.ts`
- Modify: `app/electron.vite.config.ts`
- Create: `app/src/renderer/src/test/setup.ts`
- Create: `app/src/renderer/src/test/baseline.test.ts`
- Create: `app/scripts/check-bundle-budget.mjs`

**Interfaces:** Produces the `npm test`, `npm run test:e2e`, `npm run test:visual`, `npm run check:bundle:report`, and `npm run check:bundle` commands consumed by later renderer tasks. Enable Vite's manifest and make both bundle modes read `app/out/renderer/.vite/manifest.json`. The enforcing command fails on initial JS >1,250,000 bytes, aggregate CSS >140,000 bytes, or any lazy feature chunk >900,000 bytes unless `app/bundle-analysis/<chunk>.md` exists. The report-only command measures and prints the identical assets and violations but exits zero so Task 1 can characterize the intentionally still-active legacy renderer. It is not a waiver: the unchanged enforcing command must pass after the sole-entry cutover in Tasks 26–28.

- [ ] **Step 1: Write the failing harness contract test.**

```ts
// app/src/renderer/src/test/baseline.test.ts
import { describe, expect, it } from "vitest"
// @ts-expect-error The executable JavaScript checker intentionally has no TypeScript surface.
import { BUNDLE_BUDGETS } from "../../../../scripts/check-bundle-budget.mjs"

describe("Knowledge Atlas build contracts", () => {
  it("pins the approved uncompressed budgets", () => {
    expect(BUNDLE_BUDGETS).toEqual({ initialJs: 1_250_000, css: 140_000, lazyFeature: 900_000 })
  })
})
```

- [ ] **Step 2: Run the contract test and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/test/baseline.test.ts`

Expected: FAIL with `Failed to resolve import "../../../../scripts/check-bundle-budget.mjs"`.

- [ ] **Step 3: Install and configure the exact toolchain, then implement the checker.**

Run:

```bash
cd app
npm install @reduxjs/toolkit react-redux lucide-react @tanstack/react-virtual d3-force d3-polygon @fontsource/ibm-plex-sans @fontsource/ibm-plex-mono
npm install --save-dev @testing-library/react @testing-library/user-event @testing-library/jest-dom axe-core @axe-core/playwright @playwright/test ws @types/ws @types/d3-force @types/d3-polygon
```

Add package scripts exactly as follows and configure Vitest `test.projects` with a `node` project for `src/main/**/*.test.ts` and a `renderer` project using `jsdom`, `setupFiles: ["src/renderer/src/test/setup.ts"]`, CSS support, and `src/renderer/src/**/*.test.{ts,tsx}`. This preserves the existing main-process suite instead of applying DOM globals to it:

```json
{
  "test": "vitest run",
  "test:watch": "vitest",
  "test:e2e": "playwright test",
  "test:visual": "playwright test e2e/visual.spec.ts",
  "check:bundle:report": "node scripts/check-bundle-budget.mjs --report-only",
  "check:bundle": "node scripts/check-bundle-budget.mjs",
  "check:old-ui": "node scripts/check-old-ui-removed.mjs",
  "check:sensitive-logs": "node scripts/check-sensitive-logs.mjs"
}
```

Export the exact `BUNDLE_BUDGETS` object used by the test. Classify entry-reachable chunks as initial, files with `isDynamicEntry` as lazy, and CSS assets separately. Print every measured file, byte count, and violation in both modes. Exit nonzero on a violation unless `--report-only` is present; report-only changes only the exit status, never the measured graph, limits, or output. Add deterministic named manual chunks for `editor`, `atlas`, and `markdown`; do not force them into the initial graph.

- [ ] **Step 4: Run the new contract and all existing frontend gates.**

Run:

```bash
cd app
npm test -- src/renderer/src/test/baseline.test.ts
npm run typecheck
npm run build
npm run check:bundle:report
```

Expected: the contract passes, typecheck/build exit 0, and report-only exits 0 while showing the current legacy entry and approximately 381 KB stylesheet above the final target. Then run `npm run check:bundle` separately and observe the expected nonzero exit on those same violations. Record both outputs as the RED bundle characterization. Do not add a waiver or weaken a limit. Task 26 switches the sole entry; Tasks 27–28 require this unchanged enforcing command to pass.

- [ ] **Step 5: Commit the harness.**

```bash
git add app/package.json app/package-lock.json app/vitest.config.ts app/electron.vite.config.ts app/src/renderer/src/test/setup.ts app/src/renderer/src/test/baseline.test.ts app/scripts/check-bundle-budget.mjs
git commit -m "test: establish knowledge atlas quality gates"
```

### Task 2: Add revisioned meeting proposals and operation storage

**Files:**
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/indexer/store.py`
- Create: `core/tests/test_meeting_note_mutations.py`
- Modify: `core/tests/test_meetings.py`

**Interfaces:** Preserves the existing `notes_rev` column and produces `Meeting.note_proposals`, `Meeting.note_operations`, `MeetingNoteProposal`, `MeetingNoteOperation`, `create_note_proposal`, `list_note_proposals`, `apply_note_operation`, and `NotesConflict`. Proposal and operation IDs are client/server idempotency keys; no migration screen or copied database is introduced.

- [ ] **Step 1: Write failing additive-schema and optimistic-concurrency tests.**

```py
def test_proposal_survives_store_reopen(store_path):
    store = Index(store_path)
    meeting = store.meeting_create(title="Review")
    proposal = store.create_note_proposal(
        meeting["meeting_id"], origin="meeting_chat", proposal_id="proposal-1",
        base_notes_rev=meeting["notes_rev"],
        anchor=None, occurrence=None, delete_md="", insert_md="- Ship it", rationale="Requested in chat",
        supersedes_proposal_id=None, source={"chat_id": "chat-1", "message_id": 1},
    )
    store.close()
    reopened = Index(store_path)
    assert reopened.list_note_proposals(meeting["meeting_id"])[0] == proposal

def test_stale_note_operation_is_rejected_without_writing(store):
    meeting = store.meeting_create(title="Review")
    meeting_id = meeting["meeting_id"]
    store.apply_note_operation(
        meeting_id, "command-1", 0, "insert",
        {"anchor": None, "occurrence": None, "delete_md": "", "insert_md": "one"},
    )
    with pytest.raises(NotesConflict) as conflict:
        store.apply_note_operation(
            meeting_id, "command-2", 0, "insert",
            {"anchor": None, "occurrence": None, "delete_md": "", "insert_md": "stale"},
        )
    assert conflict.value.actual_rev == 1
    assert store.meeting_get(meeting_id)["notes_md"] == "one"
```

- [ ] **Step 2: Run the focused tests and confirm the expected failure.**

Run: `cd core && uv run pytest tests/test_meeting_note_mutations.py tests/test_meetings.py -q`

Expected: FAIL because proposal schemas/tables/store methods and revision compare-and-swap do not exist.

- [ ] **Step 3: Implement the additive DDL and typed store API.**

Retain the existing `meetings.notes_rev` DDL/read behavior. Add `meeting_note_proposals` and `meeting_note_operations` through the current idempotent initializer, with meeting foreign keys, status/origin checks, unique IDs, timestamps, serialized source/payload/inverse, and indexes on `(meeting_id, status, created_at)`. Proposal-decision operation rows reference the proposal and never duplicate its private Markdown. All note-changing mutation methods use one `BEGIN IMMEDIATE` transaction and this compare-and-swap:

```sql
UPDATE meetings
SET notes_md = ?, notes_rev = notes_rev + 1, updated_at = ?
WHERE meeting_id = ? AND notes_rev = ?
```

If `rowcount != 1`, roll back and raise `NotesConflict(expected_rev, actual_rev)`. Reusing a command ID returns its stored result without incrementing the revision. Proposal reads are ordered by `created_at, id`; parsers validate every stored enum/source shape.

- [ ] **Step 4: Run focused and schema-regression tests.**

Run:

```bash
cd core
uv run pytest tests/test_meeting_note_mutations.py tests/test_meetings.py tests/test_index_store.py -q
```

Expected: PASS, including reopening a pre-change fixture with revision 0 and cascade-deleting its proposals/operations.

- [ ] **Step 5: Commit the storage spine.**

```bash
git add core/taurine_core/schemas.py core/taurine_core/indexer/store.py core/tests/test_meeting_note_mutations.py core/tests/test_meetings.py
git commit -m "feat(core): persist revisioned meeting note proposals"
```

### Task 3: Expose atomic note commands, decisions, and undo

**Files:**
- Create: `core/taurine_core/meetings/__init__.py`
- Create: `core/taurine_core/meetings/note_mutations.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/tests/test_meeting_note_mutations.py`
- Modify: `core/tests/test_meetings.py`

**Interfaces:** Consumes Task 2 storage. Produces `POST /meetings/{meeting_id}/note-commands`, `POST /meetings/{meeting_id}/note-proposals`, proposal/operation metadata in meeting detail, HTTP 409 `{detail:"notes_conflict", meeting}`, HTTP 409 `proposal_stale`, and HTTP 409 `undo_conflict`. Accepted proposals are inserted exactly once; reject changes status without changing `notes_rev`; supersede occurs in proposal creation; undo applies the stored inverse as a new operation.

- [ ] **Step 1: Write failing API tests for accept, idempotency, conflict, and undo.**

```py
async def test_accept_proposal_is_atomic_and_idempotent(client, meeting):
    meeting_id = meeting["meeting_id"]
    proposal = await create_append_proposal(client, meeting_id, "proposal-1", "## Decision\nShip", 0)
    body = {"command_id": "accept-1", "kind": "proposal_decision", "proposal_id": proposal["proposal_id"],
            "decision": "accept", "base_notes_rev": 0}
    first = await client.post(f"/meetings/{meeting_id}/note-commands", json=body)
    second = await client.post(f"/meetings/{meeting_id}/note-commands", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()["meeting"]["notes_rev"] == second.json()["meeting"]["notes_rev"] == 1
    assert first.json()["meeting"]["notes_md"].count("Ship") == 1
```

- [ ] **Step 2: Run the API tests and confirm the expected failure.**

Run: `cd core && uv run pytest tests/test_meeting_note_mutations.py tests/test_meetings.py -q`

Expected: FAIL with 404 for `/note-commands` and `/note-proposals`.

- [ ] **Step 3: Implement command validation and routes.**

Implement a `NoteMutationService` with pure functions for insert-at-heading, replace-range, proposal placement, and inverse generation. Validate heading/range anchors against the base revision, constrain payload Markdown to the configured meeting note limit, reject unknown payload keys, and never concatenate on conflict. Return `MeetingMutationResult(meeting, proposal, operation)` from one transaction. Map `NotesConflict` to 409 and invalid placement to 422. Retain the newest 100 operations per meeting after a successful transaction; never remove proposal audit rows.

```py
@router.post("/meetings/{meeting_id}/note-commands", response_model=MeetingMutationResult)
def apply_note_command(meeting_id: str, request: NoteCommandRequest) -> MeetingMutationResult:
    return note_mutations.apply(meeting_id, request)
```

- [ ] **Step 4: Run mutation, meeting, and OpenAPI tests.**

Run: `cd core && uv run pytest tests/test_meeting_note_mutations.py tests/test_meetings.py tests/test_server_startup.py -q`

Expected: PASS; generated OpenAPI contains both new routes, explicit request/response models, 409, and 422 responses. Tests also prove exact idempotent replay, conflicting command/proposal ID reuse, required `base_notes_rev` on `notes_md` PATCH, no revision increment for reject/undo-of-reject, one increment for accept/undo-of-accept, latest-operation-only undo, immutable proposal content on Accept, and deterministic on-read adaptation of stored `suggestions`.

- [ ] **Step 5: Commit the mutation API.**

```bash
git add core/taurine_core/meetings core/taurine_core/server.py core/taurine_core/schemas.py core/tests/test_meeting_note_mutations.py core/tests/test_meetings.py
git commit -m "feat(core): add transactional meeting note commands"
```

### Task 4: Route every AI note edit through durable proposals

**Files:**
- Modify: `core/taurine_core/chat/engine.py`
- Modify: `core/taurine_core/live/final_summary.py`
- Modify: `core/taurine_core/indexer/store.py`
- Modify: `core/tests/test_chat_meeting_context.py`
- Modify: `core/tests/test_final_summary.py`
- Modify: `core/tests/test_meetings.py`

**Interfaces:** Consumes `create_note_proposal`. Produces proposal-bearing `chat_notes_edit` updates and final meeting records. `source` carries only durable reference IDs; proposal audit rows survive deletion of a referenced chat and render an unresolved reference as unavailable. Deleting a chat clears `meetings.chat_id` transactionally. Final proposal creation and final meeting status commit together.

- [ ] **Step 1: Write failing persistence-before-emit and final-transaction tests.**

```py
async def test_chat_edit_is_persisted_before_event(chat_engine, store, events):
    await chat_engine.run_edit_notes(meeting_id="m1", chat_id="c1", instruction="add decision")
    event = next(event for event in events if event.type == "chat_notes_edit")
    persisted = store.get_note_proposal(event.proposal.proposal_id)
    assert persisted.status == "pending"
    assert event.proposal == persisted

def test_final_record_never_exposes_unpersisted_note_additions(finalizer, store):
    finalizer.complete("m1", recap="Recap", additions=["Action"])
    meeting = store.meeting_get("m1")
    assert meeting["status"] == "done"
    assert [p.origin for p in store.list_note_proposals("m1")] == ["final_summary"]
```

- [ ] **Step 2: Run focused producer tests and confirm the expected failure.**

Run: `cd core && uv run pytest tests/test_chat_meeting_context.py tests/test_final_summary.py tests/test_meetings.py -q`

Expected: FAIL because chat emits transient Markdown and finalization writes additions outside the proposal ledger.

- [ ] **Step 3: Replace direct/transient edits with proposal writes.**

Create the proposal before publishing an update, emit only the stored proposal DTO, and use the same database transaction as the final meeting record. Deleting a chat clears every matching `meetings.chat_id` in the same transaction and publishes both chats and meetings updates; deleting a meeting still preserves its linked inquiry. Proposal audit rows and their source IDs remain immutable; a later unresolved chat ID renders as unavailable. Add origin mappings: meeting chat → `meeting_chat`, dossier inline Ask → `inline_ask`, live rolling summary → `rolling_summary`, live actionable item → `actionable`, final recap/additions → `final_summary`. No producer mutates `notes_md`.

- [ ] **Step 4: Run all chat/final/meeting tests.**

Run: `cd core && uv run pytest tests/test_chat_meeting_context.py tests/test_server_chat_ws.py tests/test_final_summary.py tests/test_meetings.py -q`

Expected: PASS; a forced publish exception still leaves a queryable pending proposal, while a forced final transaction exception leaves neither final state nor final proposals.

- [ ] **Step 5: Commit durable AI proposals.**

```bash
git add core/taurine_core/chat/engine.py core/taurine_core/live/final_summary.py core/taurine_core/indexer/store.py core/tests/test_chat_meeting_context.py core/tests/test_final_summary.py core/tests/test_meetings.py
git commit -m "feat(core): persist all ai-authored meeting proposals"
```

### Task 5: Build the resumable session registry

**Files:**
- Create: `core/taurine_core/live/session_registry.py`
- Create: `core/tests/test_session_registry.py`

**Interfaces:** Produces a lock-protected registry keyed by `(meeting_id, client_session_id)`, with active/suspended/finalizing states, a monotonic deadline, generation token, retained `LiveSession`, and one-winner matching-Start/`finalize_due`. A different token cannot take over an active or suspended meeting. The snapshot projection holds transcript sequence, current summary, grounding, actionables, note suggestions, channels, ASR state, and deadline; it never stores or replays audio bytes.

- [ ] **Step 1: Write failing deterministic time and race tests.**

```py
async def test_disconnect_resumes_before_sixty_seconds(fake_clock, registry):
    await registry.start("m1", "s1", channels())
    await registry.suspend("m1", "s1", reason="disconnect", resume_window_s=60)
    fake_clock.advance(59.9)
    lease = await registry.start("m1", "s1", channels())
    assert lease.state == "active"

async def test_only_resume_or_expiry_finalization_wins(registry):
    await registry.start("m1", "s1", channels())
    lease = await registry.suspend("m1", "s1", reason="disconnect", resume_window_s=60)
    results = await asyncio.gather(
        registry.start("m1", "s1", channels()),
        registry.finalize_due("m1", "s1", lease.generation),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) and result for result in results) == 1
```

- [ ] **Step 2: Run the registry test and confirm the expected failure.**

Run: `cd core && uv run pytest tests/test_session_registry.py -q`

Expected: FAIL because `taurine_core.live.session_registry` is absent.

- [ ] **Step 3: Implement explicit-stop and disconnect leases.**

Inject `monotonic`, `create_task`, and finalizer callbacks. Use 60 seconds for disconnect and the requested 10 or 0 seconds for explicit Stop. Stop returns the suspended/finalizing lease immediately so the client ceases capture; matching Start validates meeting/token/generation/deadline, cancels the one timer, rebinds the retained event sink, and resumes; `finalize_due` atomically claims the generation before invoking finalization. Duplicate Stop is idempotent and never extends a deadline; Stop on a disconnect-suspended entry shortens it to at most 10 seconds. Delete and explicit app quit claim/remove entries. Bound current state with existing limits and clear it after successful finalization or terminal failure.

- [ ] **Step 4: Run registry tests under repeat and race stress.**

Add `@pytest.mark.parametrize("iteration", range(20))` to the race case, then run: `cd core && uv run pytest tests/test_session_registry.py -q`

Expected: PASS across all 20 race instances with exactly one owner and no pending-task warnings.

- [ ] **Step 5: Commit the registry.**

```bash
git add core/taurine_core/live/session_registry.py core/tests/test_session_registry.py
git commit -m "feat(core): add resumable live session registry"
```

### Task 6: Integrate session start, stop, reconnect, and snapshot reconciliation

**Files:**
- Modify: `core/taurine_core/server.py`
- Modify: `core/taurine_core/schemas.py`
- Create: `core/tests/test_meeting_session_resume.py`
- Modify: `core/tests/test_server_ws.py`

**Interfaces:** Consumes Task 5. `/session` extends existing `start`, `stop`, and `notepad` controls with `client_session_id`; Stop also carries `meeting_id` and `resume_window_s: 10|0`. It emits exact `session_snapshot` and `session_error` contracts while retaining current status/transcript/summary/grounding/actionable/suggestion events. A matching Start is the only resume control. Binary frames are accepted only after the resumed snapshot and ASR-connected acknowledgement.

- [ ] **Step 1: Write failing socket lifecycle tests.**

```py
async def test_explicit_stop_is_immediate_and_resumable(live_socket, fake_clock):
    start = {"type": "start", "client_session_id": "s1", "meeting_id": "m1", "channels": []}
    await live_socket.send_json(start)
    await live_socket.send_json({"type": "stop", "client_session_id": "s1", "meeting_id": "m1",
                                 "resume_window_s": 10})
    stopped = await live_socket.receive_json()
    assert stopped["type"] == "session_snapshot"
    assert stopped["disposition"] == "stopped_resumable"
    assert stopped["resume_deadline"] is not None
    await live_socket.send_json(start)
    assert (await live_socket.receive_json())["disposition"] == "resumed"

async def test_audio_after_stop_is_rejected_without_buffering(live_socket):
    await start_then_stop(live_socket, "s1", "m1")
    await live_socket.send_bytes(b"audio-must-not-be-retained")
    assert (await live_socket.receive_json())["code"] == "invalid_session_control"
```

- [ ] **Step 2: Run socket tests and confirm the expected failure.**

Run: `cd core && uv run pytest tests/test_meeting_session_resume.py tests/test_server_ws.py -q`

Expected: FAIL because stop finalizes directly and a reconnect creates a new context.

- [ ] **Step 3: Make the registry the sole lifecycle owner.**

Replace `Runtime.active_sessions`/`meeting_finalizers` ownership with `SessionRegistry` and a `RebindableEventSink`. On disconnect, flush ASR and suspend the retained `LiveSession` for 60 seconds. On explicit Stop, stop accepting frames and flush ASR before responding; use the requested 10-second lease or immediate finalization. A matching Start restores the same live summary/grounding/actionable/notepad objects and sends `session_snapshot{disposition:"resumed"}` before ASR connected; never replay audio or already-consumed controls. Process restart returns `state_reset` after durable meeting/transcript refetch. Preserve current event shapes except for required session identity additions. Delete meeting cancels its registry entry before durable deletion; explicit app quit requests immediate finalization.

- [ ] **Step 4: Run integration and existing audio/finalization suites.**

Run: `cd core && uv run pytest tests/test_meeting_session_resume.py tests/test_server_ws.py tests/test_live_ingest.py tests/test_final_summary.py -q`

Expected: PASS; tests prove 59.9-second resume, 60-second expiry, 9.9-second stopped resume, 10-second finalization, stale-generation rejection, and exactly-once finalization.

- [ ] **Step 5: Commit live continuation.**

```bash
git add core/taurine_core/server.py core/taurine_core/schemas.py core/tests/test_meeting_session_resume.py core/tests/test_server_ws.py
git commit -m "feat(core): resume live sessions across disconnects"
```

### Task 7: Create OS-backed credential storage and OAuth flow state

**Files:**
- Create: `core/taurine_core/connectors/credentials.py`
- Create: `core/taurine_core/connectors/oauth.py`
- Modify: `core/pyproject.toml`
- Modify: `core/uv.lock`
- Create: `core/tests/test_connector_oauth.py`

**Interfaces:** Produces `CredentialStore`, `KeyringCredentialStore(service_name="Taurine")`, `MemoryCredentialStore`, environment fallback, `OAuthProvider`, `OAuthFlow`, and `OAuthFlowRegistry`. Provider app registration remains configuration; access/refresh tokens and expiry are stored by provider/account in the OS vault. Flow state uses PKCE S256, cryptographic state, exact loopback redirect, five-minute expiry, and single consumption.

- [ ] **Step 1: Write failing vault and flow-security tests.**

```py
def test_flow_uses_pkce_exact_redirect_and_single_use(oauth, clock):
    flow = oauth.begin("gmail", callback_origin="http://127.0.0.1:43119")
    assert flow.authorization_url.startswith("https://accounts.google.com/")
    assert flow.code_challenge_method == "S256"
    assert flow.redirect_uri == f"http://127.0.0.1:43119/oauth/callback/{flow.flow_id}"
    credential = oauth.complete(flow.flow_id, state=flow.state, code="code")
    assert credential.refresh_token == "refresh"
    with pytest.raises(InvalidOAuthState):
        oauth.complete(flow.flow_id, state=flow.state, code="again")

def test_credentials_are_never_represented_with_tokens(memory_credentials):
    memory_credentials.put("google", "me@example.com", oauth_credential("secret-token"))
    assert "secret-token" not in repr(memory_credentials)
```

- [ ] **Step 2: Run the OAuth unit tests and confirm the expected failure.**

Run: `cd core && uv run pytest tests/test_connector_oauth.py -q`

Expected: FAIL because credentials and OAuth flow modules do not exist.

- [ ] **Step 3: Add keyring and implement the provider-neutral core.**

Run: `cd core && uv add keyring`

Serialize credentials as versioned JSON in keyring, never in SQLite or logs. Support read-only environment fallback for existing token variables, with the vault taking precedence. Inject HTTP exchange functions and clocks in tests. Validate provider, redirect origin/path/port, state, expiry, and response content before persisting. Redact exception bodies and map errors to stable codes. Do not ship OAuth client secrets in renderer code.

- [ ] **Step 4: Run security-focused and connector baseline tests.**

Run: `cd core && uv run pytest tests/test_connector_oauth.py tests/test_connectors.py -q`

Expected: PASS; keyring is mocked, environment fallback still works, expired/mismatched/replayed flows fail, and captured logs contain no token/code/verifier values.

- [ ] **Step 5: Commit credential and flow services.**

```bash
git add core/taurine_core/connectors/credentials.py core/taurine_core/connectors/oauth.py core/pyproject.toml core/uv.lock core/tests/test_connector_oauth.py
git commit -m "feat(core): add secure connector oauth services"
```

### Task 8: Expose OAuth and inject credentials into every connector

**Files:**
- Modify: `core/taurine_core/server.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/connectors/gmail.py`
- Modify: `core/taurine_core/connectors/outlook.py`
- Modify: `core/taurine_core/connectors/google_common.py`
- Modify: `core/taurine_core/connectors/google_calendar.py`
- Modify: `core/taurine_core/connectors/google_docs.py`
- Modify: `core/taurine_core/connectors/registry.py`
- Modify: `core/taurine_core/connectors/catalog.py`
- Modify: `core/tests/test_connector_oauth.py`
- Modify: `core/tests/test_connectors.py`

**Interfaces:** Consumes Task 7. Produces `POST /connectors/{id}/authorize -> 202 {flow_id, authorization_url, expires_at}`, `GET /connectors/{id}/authorize/{flow_id} -> {state, detail}`, `DELETE /connectors/{id}/authorization -> ConnectorInfo`, and `GET /oauth/callback/{flow_id}` minimal browser HTML. Enriches connector descriptors with authorization state and redacted detail. Each connector-registry member owns an authorization flow; shared Google credential plumbing prevents duplicate token code without inventing an undocumented provider-wide UI action.

- [ ] **Step 1: Write failing route and shared-provider tests.**

```py
async def test_google_connector_authorization_completes_via_polled_flow(client, oauth_exchange):
    started = await client.post("/connectors/gmail/authorize")
    flow_id = started.json()["flow_id"]
    callback = await client.get(
        f"/oauth/callback/{flow_id}",
        params={"state": oauth_exchange.state_for(flow_id), "code": "valid"},
    )
    assert callback.status_code == 200
    status = await client.get(f"/connectors/gmail/authorize/{flow_id}")
    assert status.json()["state"] == "authorized"

async def test_disconnect_revokes_local_credential_without_returning_token(client):
    response = await client.delete("/connectors/gmail/authorization")
    assert response.status_code == 200
    assert response.json()["id"] == "gmail"
    assert "token" not in response.text.lower()
```

- [ ] **Step 2: Run route/connector tests and confirm the expected failure.**

Run: `cd core && uv run pytest tests/test_connector_oauth.py tests/test_connectors.py -q`

Expected: FAIL with 404 for the OAuth routes and constructors that do not accept a credential provider.

- [ ] **Step 3: Implement routes, refresh, and descriptor projection.**

Use fixed loopback callbacks derived from the bound sidecar port and refuse a renderer-provided redirect. Accept authorization only for registry members with `kind="oauth"`. Callback exchanges the code, stores credentials, publishes the existing connector update event, and returns a small success/failure page. Connector fetches request a current token from injected credential services, persist rotated refresh tokens, retry once after an authorization failure, and surface stable errors. Successful auth does not auto-enable or auto-sync. Disconnect removes the vault entry, clears provider-derived caches, and publishes a connector update; Disable and Delete indexed data remain separate actions.

- [ ] **Step 4: Run all connector, API, and logging tests.**

Run:

```bash
cd core
uv run pytest tests/test_connector_oauth.py tests/test_connectors.py tests/test_server_startup.py tests/test_session_log.py -q
```

Expected: PASS; registry validation, flow polling, shared Google credential code, Microsoft isolation, refresh rotation, expired authorization, no automatic enable/sync, disconnect without purge, and response/log redaction are asserted.

- [ ] **Step 5: Commit desktop authorization support.**

```bash
git add core/taurine_core/server.py core/taurine_core/schemas.py core/taurine_core/connectors core/tests/test_connector_oauth.py core/tests/test_connectors.py
git commit -m "feat(core): expose connector oauth authorization"
```

### Task 9: Replace the Electron boundary with bounded capabilities

**Files:**
- Create: `app/src/main/windows/window-state.ts`
- Create: `app/src/main/windows/main-window.ts`
- Create: `app/src/main/windows/window-privacy.ts`
- Create: `app/src/main/permissions/capabilities.ts`
- Create: `app/src/main/permissions/system-settings.ts`
- Create: `app/src/main/recovery/drafts.ts`
- Create: `app/src/main/ipc.ts`
- Create: `app/src/preload/contracts.ts`
- Rewrite: `app/src/preload/index.ts`
- Modify: `app/src/preload/index.d.ts`
- Create: `app/src/main/ipc.test.ts`
- Create: `app/src/main/windows/window-state.test.ts`
- Create: `app/src/main/recovery/drafts.test.ts`

**Interfaces:** Produces one frozen `window.desktop` bridge matching §13.5: `capabilities`, `requestPermission`, `openSystemSettings`, `setPrivacyCurtain`, `setCompactLive`, allowlisted `openExternal`, `chooseDirectory`, `recoveryDraft`, `liveSession`, retained capture/call/context methods, and bounded event subscription. Each method maps to one validated IPC channel. It does not expose raw `ipcRenderer`, shell, arbitrary paths/URLs, subprocesses, or a generic invoke method.

- [ ] **Step 1: Write failing bridge, geometry, privacy, and recovery tests.**

```ts
// app/src/main/ipc.test.ts
// @vitest-environment node
it("registers only the allowlisted bridge channels", () => {
  registerIpc(deps)
  expect([...registeredChannels]).toEqual([
    "desktop:capabilities", "desktop:request-permission", "desktop:open-system-settings",
    "desktop:set-privacy-curtain", "desktop:set-compact-live", "desktop:open-external",
    "desktop:choose-directory", "desktop:recovery-read", "desktop:recovery-write",
    "desktop:recovery-delete", "desktop:live-session-read", "desktop:live-session-write",
    "desktop:live-session-delete", "desktop:capture-start", "desktop:capture-stop",
    "desktop:call-prompt-action", "desktop:context-suggestion-action",
    "desktop:session-listening", "desktop:show-main",
  ])
})

it("clamps restored geometry into the visible work area", () => {
  expect(clampWindowState({ x: -5000, y: 12, width: 700, height: 400 }, workArea))
    .toMatchObject({ width: 960, height: 640, x: workArea.x })
})
```

- [ ] **Step 2: Run the Electron boundary tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/main/ipc.test.ts src/main/windows/window-state.test.ts src/main/recovery/drafts.test.ts`

Expected: FAIL because the bounded modules and `window.desktop` contract are absent.

- [ ] **Step 3: Implement the bridge and main-window services.**

Create the main window at 1280×820 with 960×640 minimum; persist geometry/fullscreen/compact state only and clamp against `screen.getDisplayMatching(...).workArea`. Preserve close-to-hide and explicit Quit semantics. Compute privacy status from the actual `setContentProtection` request/result and publish exactly `requested`, `limited`, `unavailable`, or `curtain_on`. Encrypt recovery data with `safeStorage`; store only unsaved Markdown, meeting/session identity, base revision, and timestamps, never transcript/audio/token content. Permit external navigation only for validated `https:` OAuth origins returned by the sidecar and exact OS settings routes. Use schema validators on every IPC argument and return serializable result unions rather than throwing content-bearing errors.

```ts
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
  capture: { start(input: CaptureRequest): Promise<CaptureResult>; stop(): Promise<void> }
  events: { subscribe(listener: (event: DesktopEvent) => void): () => void }
}
```

- [ ] **Step 4: Run Electron tests and static type checks.**

Run:

```bash
cd app
npm test -- src/main/ipc.test.ts src/main/windows/window-state.test.ts src/main/recovery/drafts.test.ts
npm run typecheck
```

Expected: PASS; tests reject unapproved schemes/channels, malformed drafts, off-screen geometry, token-shaped recovery fields, and content-protection overclaims.

- [ ] **Step 5: Commit the bounded desktop boundary.**

```bash
git add app/src/main/windows app/src/main/permissions app/src/main/recovery app/src/main/ipc.ts app/src/main/ipc.test.ts app/src/preload
git commit -m "refactor(electron): bound desktop capabilities"
```

### Task 10: Build native prompt windows, capture services, and tray state

**Files:**
- Create: `app/src/main/windows/call-prompt-window.ts`
- Create: `app/src/main/windows/context-suggestion-window.ts`
- Create: `app/src/main/capture/system-audio.ts`
- Create: `app/src/main/capture/micwatch.ts`
- Rewrite: `app/src/main/tray.ts`
- Rewrite: `app/src/main/index.ts`
- Create: `app/src/main/windows/utility-windows.test.ts`
- Create: `app/src/main/tray.test.ts`
- Modify: `app/electron.vite.config.ts`

**Interfaces:** Consumes Task 9. Produces nonactivating 372×136 call prompt and 372×188 context suggestion windows, exact tray actions, moved capture/microphone watchers, and thin lifecycle composition. Prompt actions dispatch typed desktop events; the renderer owns no `BrowserWindow` references.

- [ ] **Step 1: Write failing utility-window and tray tests.**

```ts
it("constructs bounded nonactivating utility windows", () => {
  const call = createCallPromptWindow(deps)
  const context = createContextSuggestionWindow(deps)
  expect(call.options).toMatchObject({ width: 372, height: 136, show: false, focusable: false, resizable: false })
  expect(context.options).toMatchObject({ width: 372, height: 188, show: false, focusable: false, resizable: false })
})

it("changes tray actions with capture state", () => {
  expect(labels(buildTrayMenu({ capturePhase: "live_full" }))).toEqual([
    "Show Knowledge Atlas", "Stop meeting", "New inquiry", "Privacy curtain Off",
    "Privacy protection: Requested", "Quit Taurine",
  ])
})
```

- [ ] **Step 2: Run utility tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/main/windows/utility-windows.test.ts src/main/tray.test.ts`

Expected: FAIL because the new window factories and state-derived tray menu do not exist.

- [ ] **Step 3: Move native services and compose the lifecycle.**

Move current system-audio and microphone-watch behavior without changing subprocess/protocol semantics, then delete imports of their old paths. Configure utility windows with sandbox, context isolation, no Node integration, no activation, no arbitrary navigation, and explicit screen-edge clamping. Render only event-derived sanitized strings. The call prompt offers Start full, Mic only, Dismiss; the suggestion prompt offers Open, Dismiss, and a bounded queue count. `index.ts` composes windows, sidecar, tray, capture, permissions, recovery, single-instance activation, and quit cleanup; it contains no feature business logic.

- [ ] **Step 4: Run main-process, packaging, and type gates.**

Run:

```bash
cd app
npm test -- src/main/windows/utility-windows.test.ts src/main/tray.test.ts src/main/ipc.test.ts
npm run typecheck
npm run build
```

Expected: PASS; the production build contains three renderer entries and no old capture-module import.

- [ ] **Step 5: Commit native composition.**

```bash
git add app/src/main/index.ts app/src/main/tray.ts app/src/main/windows app/src/main/capture app/electron.vite.config.ts
git commit -m "feat(electron): compose knowledge atlas native windows"
```

### Task 11: Create exhaustive sidecar contracts and one connection gateway

**Files:**
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

**Interfaces:** Consumes the existing REST, `/updates`, `/session`, `/chat`, Tasks 3/6/8 additions, and `window.desktop`. Produces one `SidecarGateway`, exhaustive parsers, RTK Query `sidecarApi`, connection epochs, and `GatewayEvent`. Exactly one updates socket and one session socket exist for app lifetime; chat connects lazily. Reconnect never resends a chat message or audio frame.

- [ ] **Step 1: Write failing parser, ownership, and stale-epoch tests.**

```ts
it("normalizes a meeting update and rejects unknown discriminants", () => {
  expect(parseUpdatesMessage({ type: "meeting_updated", meeting_id: "m1", notes_rev: 3 }))
    .toMatchObject({ type: "meetingUpdated", meetingId: "m1", notesRev: 3 })
  expect(() => parseUpdatesMessage({ type: "new_server_event", body: "secret" }))
    .toThrowError(UnknownWireMessageError)
})

it("drops late events from a replaced socket epoch", () => {
  const gateway = makeGateway(fakeSockets)
  gateway.start()
  fakeSockets.updates.reconnect()
  fakeSockets.updates.emitFromEpoch(1, { type: "sources_changed" })
  expect(receivedEvents).toEqual([])
})
```

- [ ] **Step 2: Run gateway tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/services/sidecar/contracts.test.ts src/renderer/src/services/sidecar/gateway.test.ts`

Expected: FAIL because the sidecar service modules do not exist.

- [ ] **Step 3: Implement explicit wire/domain parsing and socket supervision.**

Define discriminated unions for every currently emitted message plus Tasks 3/6/8, including updates `connectors`, `sync`, `graph`, `profile`, `chats`, and `meetings`, and legacy session ambient/pin/unpin/retrieval/trigger seams. Parse required/optional keys without `as` casting at the network boundary. Gateway backoff is jittered/bounded, visibility-aware, and reset after a healthy interval. Updates reconnect requests REST snapshots for affected domains; session reconnect sends matching `SessionStartWire` with the retained meeting/token/channels and accepts the server snapshot; chat reconnect marks an interrupted turn and waits for user retry. `rest-api.ts` maps 409 to typed notes/proposal/undo conflicts and declares tags for Meeting, Proposal, Source, Profile, Graph, Chat, and Diagnostics.

- [ ] **Step 4: Run gateway, legacy protocol, and type tests.**

Run:

```bash
cd app
npm test -- src/renderer/src/services/sidecar
npm run typecheck
```

Expected: PASS; one-owner counts, epoch dropping, backoff, snapshot reconciliation, error redaction, and no-resend behavior are deterministic.

- [ ] **Step 5: Commit the typed gateway.**

```bash
git add app/src/renderer/src/services
git commit -m "feat(renderer): add typed sidecar gateway"
```

### Task 12: Establish the Redux domain model and safe hydration

**Files:**
- Create: `app/src/renderer/src/app/store.ts`
- Create: `app/src/renderer/src/app/listeners.ts`
- Create: `app/src/renderer/src/app/hydration.ts`
- Create: `app/src/renderer/src/app/store.test.ts`
- Create: `app/src/renderer/src/features/runtime/state.ts`
- Create: `app/src/renderer/src/features/workspace/state.ts`
- Create: `app/src/renderer/src/features/meetings/state.ts`
- Create: `app/src/renderer/src/features/inquiry/state.ts`
- Create: `app/src/renderer/src/features/atlas/state.ts`
- Create: `app/src/renderer/src/features/sources/state.ts`
- Create: `app/src/renderer/src/features/profile/state.ts`
- Create: `app/src/renderer/src/features/privacy/state.ts`
- Create: `app/src/renderer/src/features/notifications/state.ts`

**Interfaces:** Consumes Task 11 gateway/API and Task 9 recovery. Produces `RootState`, typed hooks, normalized domain slices, listener middleware, hydration phases, and restore validation. Durable truth remains server-owned; Redux stores projections, selection, bounded UI history, and operation metadata only.

- [ ] **Step 1: Write failing reducer/listener/hydration tests.**

```ts
it("keeps the three coordinates separate from Control", () => {
  const state = workspaceReducer(undefined, workspaceActions.openControl("sources"))
  expect(state.coordinate).toBe("atlas")
  expect(state.control).toBe("sources")
})

it("restores only a matching meeting revision", async () => {
  desktop.recovery.load.mockResolvedValue(recovery({ meetingId: "m1", baseNotesRev: 2 }))
  api.getMeeting.mockResolvedValue(meeting({ id: "m1", notesRev: 3 }))
  await hydrate(store, deps)
  expect(selectRecoveryConflict(store.getState())).toEqual({ meetingId: "m1", draftRev: 2, serverRev: 3 })
})
```

- [ ] **Step 2: Run state tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/app/store.test.ts`

Expected: FAIL because store, slices, and hydration do not exist.

- [ ] **Step 3: Implement normalized slices and listener-owned effects.**

Configure RTK serializability checks and production `devTools: false`; do not store sockets, DOM nodes, errors with content, editor instances, or promises. Register gateway subscription once in listener middleware. Hydration sequence is capabilities → recovery metadata → meetings/sources/profile → graph summary; a domain failure marks only that domain unavailable and creates an exception-ledger entry. Validate restored coordinate/control/selection IDs against hydrated entities. Persist only geometry through main and safe recovery through `window.desktop`; do not add browser localStorage for knowledge content.

- [ ] **Step 4: Run state, gateway, and type tests.**

Run: `cd app && npm test -- src/renderer/src/app src/renderer/src/services/sidecar && npm run typecheck`

Expected: PASS; strict mode double-mount leaves one subscription, recovery mismatch is explicit, and production store construction disables DevTools.

- [ ] **Step 5: Commit the application state spine.**

```bash
git add app/src/renderer/src/app app/src/renderer/src/features/*/state.ts
git commit -m "feat(renderer): establish knowledge atlas state model"
```

### Task 13: Implement tokens, typography, localization, and accessible primitives

**Files:**
- Create: `app/src/renderer/src/styles/tokens.css`
- Create: `app/src/renderer/src/styles/reset.css`
- Create: `app/src/renderer/src/styles/typography.css`
- Create: `app/src/renderer/src/styles/primitives.css`
- Create: `app/src/renderer/src/styles/print.css`
- Create: `app/src/renderer/src/i18n/en.ts`
- Create: `app/src/renderer/src/i18n/format.ts`
- Create: `app/src/renderer/src/i18n/pseudo.ts`
- Create: `app/src/renderer/src/i18n/useMessage.ts`
- Create: `app/src/renderer/src/components/primitives/Button.tsx`
- Create: `app/src/renderer/src/components/primitives/Field.tsx`
- Create: `app/src/renderer/src/components/primitives/Menu.tsx`
- Create: `app/src/renderer/src/components/primitives/Dialog.tsx`
- Create: `app/src/renderer/src/components/primitives/Sheet.tsx`
- Create: `app/src/renderer/src/components/primitives/Notice.tsx`
- Create: `app/src/renderer/src/components/primitives/Tooltip.tsx`
- Create: `app/src/renderer/src/components/primitives/Progress.tsx`
- Create: `app/src/renderer/src/components/primitives/Skeleton.tsx`
- Create: `app/src/renderer/src/components/primitives/Divider.tsx`
- Create: `app/src/renderer/src/components/primitives/primitives.test.tsx`
- Create: source SVG files enumerated in the target tree

**Interfaces:** Produces CSS variables and accessible primitives consumed by every surface. IBM Plex Sans is UI type; IBM Plex Mono is timestamps, IDs, and technical metadata. Theme supports light, dark, system, forced colors, reduced motion, and print. `Progress` requires real `done/total` for determinate mode.

- [ ] **Step 1: Write failing keyboard, progress, and pseudo-locale tests.**

```tsx
it("does not permit invented determinate progress", () => {
  expect(() => render(<Progress label="Indexing" mode="determinate" />)).toThrow()
})

it("expands strings by at least thirty-five percent", () => {
  expect(pseudo("Open meeting").length).toBeGreaterThanOrEqual(Math.ceil("Open meeting".length * 1.35))
})

it("returns focus after a modal closes", async () => {
  render(<DialogHarness />)
  await user.click(screen.getByRole("button", { name: "Open" }))
  await user.keyboard("{Escape}")
  expect(screen.getByRole("button", { name: "Open" })).toHaveFocus()
})
```

- [ ] **Step 2: Run primitive tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/components/primitives/primitives.test.tsx`

Expected: FAIL because primitives, message catalog, and pseudo-localizer are absent.

- [ ] **Step 3: Implement the visual foundation.**

Define neutral paper/ink/sage/cobalt/amber/red tokens, 4/8 px spacing rhythm, 1 px rules, restrained radii, semantic focus/selection/status colors, and z-index layers. Use Lucide icons with text/accessible names; use checked-in source marks only for sources, never hand-authored UI glyphs. Implement roving focus menus, trapped modal dialogs, nonmodal sheets, labelled fields, polite notices, reduced-motion skeletons, and 24×24 minimum hit areas with 40–44 px primary/destructive/meeting controls. Catalog every user-facing string in `en.ts`; formatting uses `Intl` and logical CSS properties.

Use these canonical token values rather than approximations:

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

Source marks use Apple Notes `#B46B00`, Obsidian `#7352B8`, iMessage `#187854`, Gmail `#B43B3B`, Outlook `#23699C`, Google Calendar `#2F7D67`, Google Docs `#315BCB`, Folders `#66717E`, Claude `#8B5A2B`, Codex `#34404C`, and Taurine `#9B4763`, always paired with glyph and label. Bundle Sans weights 400/500/600 and Mono 400/500. Use the exact spacing scale `2,4,6,8,12,16,20,24,32,40,48,64`, radii 4/6/3/0, motion 80/140/220 ms with `cubic-bezier(.2,.8,.2,1)`, and the spec's reduced-motion override. No gradients are permitted.

- [ ] **Step 4: Run primitive, accessibility-unit, and style build tests.**

Run:

```bash
cd app
npm test -- src/renderer/src/components/primitives
npm run typecheck
npm run build
```

Expected: PASS; no missing accessible names, modal focus leaks, invalid progress, physical left/right layout declarations, or missing font assets.

- [ ] **Step 5: Commit the design system.**

```bash
git add app/src/renderer/src/styles app/src/renderer/src/i18n app/src/renderer/src/components/primitives app/src/renderer/src/assets/sources
git commit -m "feat(renderer): add knowledge atlas visual system"
```

### Task 14: Compose the workspace shell and command model

**Files:**
- Create: `app/src/renderer/src/app/App.tsx`
- Create: `app/src/renderer/src/app/AppShell.tsx`
- Create: `app/src/renderer/src/app/commands.ts`
- Create: `app/src/renderer/src/app/AppShell.test.tsx`
- Create: `app/src/renderer/src/features/runtime/RuntimeGate.tsx`
- Create: `app/src/renderer/src/features/runtime/OperationLedger.tsx`
- Create: `app/src/renderer/src/features/runtime/selectors.ts`
- Create: `app/src/renderer/src/features/workspace/CoordinateRail.tsx`
- Create: `app/src/renderer/src/features/workspace/CommandDeck.tsx`
- Create: `app/src/renderer/src/features/workspace/CommandCenter.tsx`
- Create: `app/src/renderer/src/features/workspace/QuickOpen.tsx`
- Create: `app/src/renderer/src/features/workspace/ContextShelf.tsx`
- Create: `app/src/renderer/src/features/workspace/ControlWorkbench.tsx`
- Create: `app/src/renderer/src/features/workspace/selectors.ts`
- Create: `app/src/renderer/src/components/layout/SplitPane.tsx`
- Create: `app/src/renderer/src/components/layout/VirtualList.tsx`
- Create: `app/src/renderer/src/components/layout/IndexPane.tsx`
- Create: `app/src/renderer/src/components/layout/FocusZone.tsx`
- Create: `app/src/renderer/src/styles/shell.css`

**Interfaces:** Consumes Tasks 12–13. Produces the sole shell hierarchy: coordinate rail (Atlas, Chronicle, Inquiry), coordinate-specific index, primary workbench, evidence inspector, persistent context shelf, exception-only operation ledger, command deck, quick open, and separate Control workbench. Commands have stable IDs, availability selectors, shortcut labels, and one execution path shared by menus/keys/buttons.

- [ ] **Step 1: Write failing semantic, breakpoint, and command-precedence tests.**

```tsx
it("renders exactly three coordinates and separate Control", () => {
  renderShell({ width: 1280 })
  expect(screen.getByRole("navigation", { name: "Knowledge coordinates" }))
    .toHaveTextContent("AtlasChronicleInquiry")
  expect(screen.getByRole("button", { name: "Control" })).toBeVisible()
  expect(screen.queryByRole("tab", { name: "Sources" })).not.toBeInTheDocument()
})

it("gives modal escape precedence over global shortcuts", async () => {
  renderShellWithOpenCommandCenter()
  await user.keyboard("{Escape}")
  expect(screen.queryByRole("dialog", { name: "Command center" })).not.toBeInTheDocument()
  expect(selectCoordinate(store.getState())).toBe("atlas")
})
```

- [ ] **Step 2: Run shell tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/app/AppShell.test.tsx`

Expected: FAIL because the shell and command registry are absent.

- [ ] **Step 3: Implement shell composition and exact responsive modes.**

At ≥1440 px show rail + index + workbench + inspector and allow Live three-lane. At 1180–1439 show rail + index + workbench with inspector as a sheet; Live remains two-lane. At 960–1179 collapse index and inspector to sheets while preserving coordinate identity. Below 960 is compact-window content only, never a squeezed full shell. Persist split ratios within bounded ranges. Operation ledger stays hidden when operations succeed and shows only failures, blocked/paused work, required decisions, and explicit retry state. App-level shortcuts defer to editor, modal, and menu scopes.

- [ ] **Step 4: Run shell/state/type tests.**

Run: `cd app && npm test -- src/renderer/src/app src/renderer/src/features/workspace src/renderer/src/features/runtime && npm run typecheck`

Expected: PASS for 1440/1280/960 widths, keyboard navigation, inspector sheets, hidden healthy ledger, and strict-mode listener ownership.

- [ ] **Step 5: Commit the shell.**

```bash
git add app/src/renderer/src/app app/src/renderer/src/features/runtime app/src/renderer/src/features/workspace app/src/renderer/src/components/layout app/src/renderer/src/styles/shell.css
git commit -m "feat(renderer): compose knowledge atlas workspace shell"
```

### Task 15: Normalize evidence semantics and build the shared inspector

**Files:**
- Create: `app/src/renderer/src/features/evidence/normalize.ts`
- Create: `app/src/renderer/src/features/evidence/normalize.test.ts`
- Create: `app/src/renderer/src/features/evidence/selectors.ts`
- Create: `app/src/renderer/src/features/evidence/EvidenceInspector.tsx`
- Create: `app/src/renderer/src/features/evidence/EvidenceLedger.tsx`
- Create: `app/src/renderer/src/features/evidence/PassageView.tsx`
- Create: `app/src/renderer/src/features/evidence/FactView.tsx`
- Create: `app/src/renderer/src/features/evidence/PathView.tsx`
- Create: `app/src/renderer/src/components/knowledge/ObjectRow.tsx`
- Create: `app/src/renderer/src/components/knowledge/SourceMark.tsx`
- Create: `app/src/renderer/src/components/knowledge/EvidenceRef.tsx`
- Create: `app/src/renderer/src/components/knowledge/DecisionLabel.tsx`
- Create: `app/src/renderer/src/components/knowledge/FactRow.tsx`
- Create: `app/src/renderer/src/components/knowledge/PathView.tsx`
- Create: `app/src/renderer/src/components/knowledge/DiffProposal.tsx`
- Create: `app/src/renderer/src/features/evidence/EvidenceInspector.test.tsx`

**Interfaces:** Consumes passage/fact/path/grounding/chat contracts. Produces one normalized `EvidenceRecord` union and the shared inspector used by Atlas, meetings, and Inquiry. Evidence relationship is `cited|consulted|retrieved`; policy outcome is `accepted|fallback|abstained|cancelled|interrupted|error`; temporal fact state is separate. Numeric confidence appears only in Advanced with calibration wording.

- [ ] **Step 1: Write failing truth-language and normalization tests.**

```ts
it.each([
  ["cited", "Cited in this answer"],
  ["consulted", "Consulted but not cited"],
  ["retrieved", "Retrieved only"],
] as const)("renders %s without strengthening the claim", (relationship, label) => {
  render(<EvidenceInspector record={passage({ relationship, score: 0.87 })} />)
  expect(screen.getByRole("heading", { name: label })).toBeVisible()
  expect(screen.queryByText(/verified|fact checked/i)).not.toBeInTheDocument()
  expect(screen.queryByText("87%")).not.toBeInTheDocument()
})

it("withholds support lists for an interrupted partial", () => {
  expect(normalizeTurn(interruptedTurnWithCitationTokens()).support).toEqual([])
})
```

- [ ] **Step 2: Run evidence tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/evidence`

Expected: FAIL because the evidence normalization and inspector modules are absent.

- [ ] **Step 3: Implement shared evidence records and views.**

Render passage ID/title/exact text/date/range, fact direction/status/interval/source support, resolved path nodes/edge kinds/fact IDs, meeting transcript sequence, and note passage sets. Preserve prior-context/low-confidence and fallback warnings. Label graph edge weight `Projection weight`; do not expose an edge-explanation command or call `/graph/edge-reason`. Show Open in source app only when the adapter supports it, route Apple Notes through existing `/session open_note`, offer payload-backed Copy text/path/link otherwise, and keep failures inline. Use a linear list alternative for paths under 420 px. Announce inspector selection once, not on streaming token changes. Advanced numeric values carry `Bootstrap calibration; no held-out metrics` when the active build reports degraded calibration.

- [ ] **Step 4: Run evidence, semantics, and type tests.**

Run: `cd app && npm test -- src/renderer/src/features/evidence src/renderer/src/components/knowledge && npm run typecheck`

Expected: PASS across cited/consulted/retrieved, fallback/abstained/interrupted, active/historical/superseded/conflicted/candidate facts, and missing/unavailable records.

- [ ] **Step 5: Commit shared evidence semantics.**

```bash
git add app/src/renderer/src/features/evidence app/src/renderer/src/components/knowledge
git commit -m "feat(renderer): unify evidence inspection semantics"
```

### Task 16: Implement Inquiry threads, streaming turns, and scoped Ask

**Files:**
- Create: `app/src/renderer/src/features/inquiry/contracts.ts`
- Create: `app/src/renderer/src/features/inquiry/selectors.ts`
- Create: `app/src/renderer/src/features/inquiry/listeners.ts`
- Create: `app/src/renderer/src/features/inquiry/InquiryIndex.tsx`
- Create: `app/src/renderer/src/features/inquiry/InquiryWorkbench.tsx`
- Create: `app/src/renderer/src/features/inquiry/InquiryComposer.tsx`
- Create: `app/src/renderer/src/features/inquiry/AssistantTurn.tsx`
- Create: `app/src/renderer/src/features/inquiry/ProcessTrace.tsx`
- Create: `app/src/renderer/src/features/inquiry/MessageMarkdown.tsx`
- Create: `app/src/renderer/src/features/inquiry/inquiry.test.tsx`

**Interfaces:** Consumes chat REST/socket, Task 15 evidence, and workspace scope. Produces Inquiry index/workbench with scopes All indexed knowledge, Current meeting, Pinned context, and Current Atlas selection/neighborhood. Scope maps to `meeting_id` and `seed_note_ids` and is described as “Start from these objects,” never as hard retrieval isolation.

- [ ] **Step 1: Write failing scope, streaming, cancel, and reconnect tests.**

```tsx
it("sends pinned objects as retrieval seeds without claiming hard scope", async () => {
  renderInquiry({ pins: [note("n1"), meeting("m1")] })
  await user.type(screen.getByRole("textbox", { name: "Ask Taurine" }), "What changed?{Enter}")
  expect(chat.send).toHaveBeenCalledWith(expect.objectContaining({ seed_note_ids: ["n1", "taurine:m1"] }))
  expect(screen.getByText("Start from these objects · 2")).toBeVisible()
})

it("never resends an interrupted user turn on reconnect", () => {
  reconnectActiveChat()
  expect(chat.send).not.toHaveBeenCalled()
  expect(screen.getByText("Unvalidated partial")).toBeVisible()
})
```

- [ ] **Step 2: Run Inquiry tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/inquiry/inquiry.test.tsx`

Expected: FAIL because Inquiry components/listeners are absent.

- [ ] **Step 3: Implement the chat finite-state UI.**

Keep a new inquiry client-empty until first send. Composer is at least 44 px, grows to eight lines, Enter sends, Shift+Enter inserts newline, and `⌘.` stops the focused active turn. Batch Markdown deltas at 30–50 ms. Render phase, ordered process disclosure, answer, evidence ledger, and actions as a flat document section; raw reasoning stays under Advanced process and is not evidence. Preserve input on busy, distinguish fallback/abstained/cancelled/interrupted/error, support server-authoritative regenerate replacement, rename rollback, delete consequences, background-stream count, and meeting-linked chat cleanup. Chat reconnection refetches authoritative history and marks persisted partials interrupted/cancelled without sending again.

- [ ] **Step 4: Run Inquiry, gateway, and evidence tests.**

Run: `cd app && npm test -- src/renderer/src/features/inquiry src/renderer/src/features/evidence src/renderer/src/services/sidecar && npm run typecheck`

Expected: PASS; tests cover process replacement by stable index, stop, fallback/abstain, stripped claims, Retry/Regenerate, rename/delete rollback, background turns, citation focus, and interrupted no-resend.

- [ ] **Step 5: Commit Inquiry.**

```bash
git add app/src/renderer/src/features/inquiry
git commit -m "feat(renderer): build inquiry and evidence workflow"
```

### Task 17: Build deterministic Atlas layout and Canvas rendering

**Files:**
- Create: `app/src/renderer/src/features/atlas/contracts.ts`
- Create: `app/src/renderer/src/features/atlas/layout.ts`
- Create: `app/src/renderer/src/features/atlas/layout.test.ts`
- Create: `app/src/renderer/src/features/atlas/atlas-worker.ts`
- Create: `app/src/renderer/src/features/atlas/render.ts`
- Create: `app/src/renderer/src/features/atlas/storage.ts`
- Create: `app/src/renderer/src/features/atlas/storage.test.ts`
- Create: `app/src/renderer/src/features/atlas/AtlasCanvas.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasCanvas.test.tsx`

**Interfaces:** Consumes global/local graph snapshots and live grounding. Produces immutable `AtlasSnapshot`, worker messages `{type:"layout", buildId, graphVersion, nodes, edges, priorPositions, reducedMotion}` and `{type:"settled", buildId, graphVersion, positions, hulls}`, four cartographic levels of detail, hit-test index, and a Canvas that is `aria-hidden`. Persisted layout contains only build ID, note ID, x/y, and user-pinned flag—never titles, snippets, passages, or source content.

- [ ] **Step 1: Write failing determinism, graph-version, and storage tests.**

```ts
it("derives identical initial geography for identical stable inputs", () => {
  expect(initialPosition(node, "build-7")).toEqual(initialPosition(node, "build-7"))
})

it("keeps unchanged nodes and places additions at their territory centroid", () => {
  const next = seedGraph(nextGraph, previousPositions)
  expect(next.n1).toEqual(previousPositions.n1)
  expect(distance(next.newNode, territoryCentroid("community-a"))).toBeLessThan(24)
})

it("persists no knowledge text", () => {
  saveLayout(storage, layoutWithPrivateNodeText())
  expect(storage.lastValue).not.toMatch(/private title|private snippet/)
})
```

- [ ] **Step 2: Run Atlas engine tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/atlas/layout.test.ts src/renderer/src/features/atlas/storage.test.ts src/renderer/src/features/atlas/AtlasCanvas.test.tsx`

Expected: FAIL because layout, storage, worker, and Canvas modules are absent.

- [ ] **Step 3: Implement stable worker geography and Canvas LOD.**

Seed positions from community ID, stable note-ID hash, degree, and date band. Run d3-force and hull computation in the worker, settle, and publish no faster than animation frame. Reuse prior coordinates by build/note ID, place new nodes near their territory centroid, fade removed nodes for exactly 140 ms unless reduced motion, and persist user pins locally. Render flat 6–10% territory tints, horizontal labels, shape-plus-label selection/pin/live states, and exact scale bands `<0.55`, `0.55–1.05`, `1.05–1.80`, `>1.80`. Static edges appear only on hover/selection until the highest band. Encode semantic/topic/entity/mixed/fact and live states by the specified line/shape patterns. Never mutate graph data or keep a perpetual animation.

- [ ] **Step 4: Run engine, worker, Canvas, and type tests.**

Run: `cd app && npm test -- src/renderer/src/features/atlas && npm run typecheck`

Expected: PASS; deterministic snapshots, stale worker result rejection, reduced-motion snap, pin/release/reset, four LODs, line patterns, and sanitized persistence are asserted.

- [ ] **Step 5: Commit the Atlas engine.**

```bash
git add app/src/renderer/src/features/atlas/contracts.ts app/src/renderer/src/features/atlas/layout.ts app/src/renderer/src/features/atlas/layout.test.ts app/src/renderer/src/features/atlas/atlas-worker.ts app/src/renderer/src/features/atlas/render.ts app/src/renderer/src/features/atlas/storage.ts app/src/renderer/src/features/atlas/storage.test.ts app/src/renderer/src/features/atlas/AtlasCanvas.tsx app/src/renderer/src/features/atlas/AtlasCanvas.test.tsx
git commit -m "feat(renderer): add deterministic atlas canvas engine"
```

### Task 18: Complete Atlas navigation, Outline, time, lenses, and comparison

**Files:**
- Create: `app/src/renderer/src/features/atlas/AtlasWorkbench.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasOutline.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasIndex.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasControls.tsx`
- Create: `app/src/renderer/src/features/atlas/TimeBand.tsx`
- Create: `app/src/renderer/src/features/atlas/selectors.ts`
- Create: `app/src/renderer/src/features/atlas/AtlasWorkbench.test.tsx`
- Create: `app/src/renderer/src/features/workspace/ComparisonBoard.tsx`
- Create: `app/src/renderer/src/features/workspace/ComparisonBoard.test.tsx`

**Interfaces:** Consumes Task 17 engine, graph/search/idea APIs, Task 15 inspector, and shared workspace selection/pins. Produces synchronized Canvas/Outline, global and neighborhood modes, lenses, 56 px time band, selection history, and two/three-column comparison. Selection is not activation: move/click selects, Enter opens, Space previews, Shift+Enter pins.

- [ ] **Step 1: Write failing keyboard, outline, filter, and comparison tests.**

```tsx
it("provides every Canvas action through the Outline", async () => {
  renderAtlas({ screenReader: true })
  const outline = screen.getByRole("tree", { name: "Atlas outline" })
  await user.click(within(outline).getByText("Project Cedar"))
  expect(selectSelection(store.getState())).toBe("n1")
  await user.keyboard("{Enter}")
  expect(api.getLocalGraph).toHaveBeenCalledWith({ center: "n1", depth: 1 })
})

it("filters Canvas, Outline, and Chronicle with one time range", () => {
  setTimePreset("90_days")
  expect(selectAtlasNodes(store.getState())).toEqual([recentNode])
  expect(selectOutlineRows(store.getState())).toEqual([recentNode])
  expect(selectChronicleRows(store.getState())).toEqual([recentMeeting])
})
```

- [ ] **Step 2: Run Atlas workbench tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/atlas/AtlasWorkbench.test.tsx src/renderer/src/features/workspace/ComparisonBoard.test.tsx`

Expected: FAIL because Outline, time band, controls, and comparison are absent.

- [ ] **Step 3: Implement complete spatial and comparative interaction.**

Make Outline the screen-reader default and Canvas `aria-hidden`. Global Outline is territory→notes tree; local Outline is selected note→incoming/outgoing treegrid with relationship kind/weight. Add sorting, `/`, `V`, `F`, `+/-`, arrow pan/navigation, `[ ]` neighborhood history, depth 1/2/3, context-menu keyboard access, 40×40 zoom controls, pin/release/reset, and non-drag alternatives. Time presets are All/30 days/90 days/1 year/Custom with localized fields/tooltips. Render lenses as removable text clauses for source/community/recency/connections/minimum links/session/evidence set/AI idea; preserve idea query on abstain/error. Keep prior graph on sync/fetch failure with build identity. Comparison pins at most three objects, aligns shared entities/dates/sources/citations with textual “shared,” and synthesizes only when the user invokes Ask these.

- [ ] **Step 4: Run complete Atlas/workspace tests.**

Run: `cd app && npm test -- src/renderer/src/features/atlas src/renderer/src/features/workspace src/renderer/src/features/evidence && npm run typecheck`

Expected: PASS for keyboard parity, LOD-independent outline reachability, neighborhood directionality, filters, live overlay/restart disclosure, stale graph, idea abstain/error, and two/three-object comparison.

- [ ] **Step 5: Commit the Atlas workbench.**

```bash
git add app/src/renderer/src/features/atlas app/src/renderer/src/features/workspace/ComparisonBoard.tsx app/src/renderer/src/features/workspace/ComparisonBoard.test.tsx
git commit -m "feat(renderer): complete atlas navigation and comparison"
```

### Task 19: Make one meeting document controller the sole notes writer

**Files:**
- Create: `app/src/renderer/src/features/meetings/contracts.ts`
- Create: `app/src/renderer/src/features/meetings/MeetingDocumentController.ts`
- Create: `app/src/renderer/src/features/meetings/MeetingDocumentController.test.ts`
- Create: `app/src/renderer/src/features/meetings/editor/blocknote-adapter.ts`
- Create: `app/src/renderer/src/features/meetings/editor/provenance.ts`
- Create: `app/src/renderer/src/features/meetings/editor/markdown-source.tsx`
- Create: `app/src/renderer/src/features/meetings/NotesDocument.tsx`
- Create: `app/src/renderer/src/features/meetings/ProposalQueue.ts`
- Create: `app/src/renderer/src/features/meetings/ProposalDiff.tsx`
- Create: `app/src/renderer/src/features/meetings/listeners.ts`
- Create: `app/src/renderer/src/features/meetings/selectors.ts`

**Interfaces:** Consumes Tasks 3/4 APIs and safe recovery. Produces exactly one `MeetingDocumentController` per mounted dossier. All editor typing/paste, transcript insert, proposal decision, conflict resolution, and undo enter typed controller commands. `ProposalQueue` can create/refetch but cannot call editor mutation or note-command endpoints.

- [ ] **Step 1: Write failing autosave, proposal, conflict, and ownership tests.**

```ts
it("persists a recovery draft before revision-aware autosave", async () => {
  controller.replaceFromEditor("changed")
  await clock.advanceByAsync(800)
  expect(desktop.recovery.save.mock.invocationCallOrder[0])
    .toBeLessThan(api.updateMeeting.mock.invocationCallOrder[0])
  expect(api.updateMeeting).toHaveBeenCalledWith("m1", { notes_md: "changed", base_notes_rev: 4 })
})

it("does not put a proposal into the document before Accept succeeds", async () => {
  queue.upsert(proposal({ insertMd: "assistant text" }))
  expect(editor.getMarkdown()).not.toContain("assistant text")
  await controller.decideProposal("p1", "accept")
  expect(editor.getMarkdown()).toContain("assistant text")
})
```

- [ ] **Step 2: Run controller tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/meetings/MeetingDocumentController.test.ts`

Expected: FAIL because controller/editor/proposal adapters are absent.

- [ ] **Step 3: Implement the finite-state save boundary.**

Controller states are clean/editing/saving/queued_offline/conflict. Autosave begins 800 ms after idle, writes encrypted recovery first, and supplies `base_notes_rev`. While live, send token-bearing `/session notepad` after 2 seconds idle. Stop capture never awaits save; `flush()` runs during grace. On 409 preserve local/server text, stop automatic writes, and open resolution with Keep mine, Keep server, Copy both. Proposal Accept/Reject sends immutable server-owned content through `proposal_decision`; stale proposals offer Choose location, explicit Append, or Reject by creating a superseding proposal. Undo obeys server conflict response. Track AI provenance from operations/blocks; editing an AI span marks the changed span user-authored without erasing operation history. If safeStorage is unavailable, keep recovery memory-only and show the exact restart-unavailable state.

- [ ] **Step 4: Run controller, API-contract, and editor tests.**

Run: `cd app && npm test -- src/renderer/src/features/meetings/MeetingDocumentController.test.ts src/renderer/src/services/sidecar && npm run typecheck`

Expected: PASS for debounce, offline queue, reload recovery, base-revision conflict, accept/reject, stale re-anchor, latest undo, direct transcript insertion, final notepad, and strict one-writer spies.

- [ ] **Step 5: Commit the notes boundary.**

```bash
git add app/src/renderer/src/features/meetings app/src/renderer/src/services/sidecar
git commit -m "feat(renderer): enforce sole meeting document writer"
```

### Task 20: Implement capture preflight and the resumable live state machine

**Files:**
- Modify: `app/src/renderer/src/features/meetings/state.ts`
- Modify: `app/src/renderer/src/features/meetings/listeners.ts`
- Create: `app/src/renderer/src/features/meetings/capture-machine.ts`
- Create: `app/src/renderer/src/features/meetings/capture-machine.test.ts`
- Create: `app/src/renderer/src/features/meetings/CaptureHeader.tsx`
- Create: `app/src/renderer/src/features/meetings/CompactLive.tsx`
- Create: `app/src/renderer/src/features/privacy/PrivacyCurtain.tsx`
- Create: `app/src/renderer/src/features/privacy/PrivacyStatus.tsx`
- Create: `app/src/renderer/src/features/notifications/ContextQueue.tsx`

**Interfaces:** Consumes desktop capabilities/capture, session gateway, document controller, and session snapshots. Produces the exact `CapturePhase` machine, stable renderer-generated `client_session_id`, capability preflight, immediate Stop, 10-second Resume/Finalize now, 60-second disconnect continuation, gap/reset disclosure, compact live, and curtain-safe capture controls.

- [ ] **Step 1: Write failing transition and sequencing tests.**

```ts
it("stops local capture before network stop and document flush", async () => {
  await listeners.stopMeeting({ resumeWindowS: 10 })
  expect(desktop.capture.stop.mock.invocationCallOrder[0])
    .toBeLessThan(session.sendStop.mock.invocationCallOrder[0])
  expect(session.sendStop.mock.invocationCallOrder[0])
    .toBeLessThan(controller.flush.mock.invocationCallOrder[0])
})

it("resumes only after snapshot then ASR connected", () => {
  machine.send({ type: "SOCKET_RECONNECTED" })
  machine.send({ type: "SESSION_SNAPSHOT", disposition: "resumed" })
  expect(machine.state).toBe("reconnecting")
  machine.send({ type: "ASR_STATUS", status: "connected" })
  expect(machine.state).toBe("live_full")
})
```

- [ ] **Step 2: Run lifecycle tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/meetings/capture-machine.test.ts`

Expected: FAIL because the capture machine and listeners are absent.

- [ ] **Step 3: Implement exact preflight and continuation rules.**

Preflight checks session socket, microphone/device, optional system audio, durable meeting row, and active-meeting conflict. Mic failure aborts; system failure degrades to mic-only. Declare actual channels on Start. On socket loss pause/stop sending audio and never buffer; show `Reconnecting · audio may be missing`. Matching Start uses the same meeting/token, merges snapshot by summary revision/actionable ID/status/evidence ID/suggestion note ID/transcript sequence, then restarts only channels passing preflight after ASR connected. Resumed state inserts an audio-gap divider; `state_reset` also clears ephemeral grounding/context and inserts `Live context restarted`. Stop saves recovery and ceases capture first, sends idempotent Stop with 10 seconds, continues notes/notepad flush, and offers 44 px Resume plus secondary Finalize now (`resume_window_s:0`). Apply originating `final_summary_status`/`final_summary` only when meeting ID and base revision match the open record, then refetch; missed events reconcile through meetings updates and detail REST. Never show model percent/phases.

- [ ] **Step 4: Run lifecycle, desktop, controller, and gateway tests.**

Run: `cd app && npm test -- src/renderer/src/features/meetings src/renderer/src/features/privacy src/renderer/src/services src/main && npm run typecheck`

Expected: PASS for full/mic-only, duplicate Start, permission recovery, immediate Stop, 10-second resume, disconnect resume/reset/conflict, no audio replay, stale event rejection, and finalization navigation.

- [ ] **Step 5: Commit meeting capture lifecycle.**

```bash
git add app/src/renderer/src/features/meetings app/src/renderer/src/features/privacy app/src/renderer/src/features/notifications
git commit -m "feat(renderer): add resumable meeting capture lifecycle"
```

### Task 21: Build Chronicle, live desk, dossier, and proposal review

**Files:**
- Create: `app/src/renderer/src/features/meetings/MeetingsIndex.tsx`
- Create: `app/src/renderer/src/features/meetings/MeetingDossier.tsx`
- Create: `app/src/renderer/src/features/meetings/LiveDesk.tsx`
- Create: `app/src/renderer/src/features/meetings/TranscriptLane.tsx`
- Create: `app/src/renderer/src/features/meetings/LiveIntelligence.tsx`
- Create: `app/src/renderer/src/features/meetings/FinalRecord.tsx`
- Create: `app/src/renderer/src/features/meetings/MeetingDossier.test.tsx`
- Create: `app/src/renderer/src/features/meetings/LiveDesk.test.tsx`

**Interfaces:** Consumes Tasks 19–20, meeting/chat/update contracts, and evidence inspector. Produces Chronicle index, draft/preflight/live/final dossier, two-lane default Live, opt-in three-lane ≥1440, sheet layout <1180, transcript, live intelligence, durable proposal review/history, final record, index status, delete consequences, and meeting-scoped Ask.

- [ ] **Step 1: Write failing lane, transcript, proposal, and deletion tests.**

```tsx
it("uses two lanes at the ordinary 1280 width", () => {
  renderLiveDesk({ width: 1280, threeLanePreference: true })
  expect(screen.getByRole("document", { name: "Meeting notes" })).toBeVisible()
  expect(screen.getByRole("region", { name: /Transcript|Live intelligence/ })).toBeVisible()
  expect(screen.queryByRole("region", { name: "Third live lane" })).not.toBeInTheDocument()
})

it("keeps decided proposals in collapsed history", async () => {
  renderDossier({ proposals: [acceptedProposal, rejectedProposal] })
  expect(screen.getByRole("button", { name: "Decision history · 2" })).toBeVisible()
})
```

- [ ] **Step 2: Run meeting surface tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/meetings/MeetingDossier.test.tsx src/renderer/src/features/meetings/LiveDesk.test.tsx`

Expected: FAIL because Chronicle/dossier/live surfaces are absent.

- [ ] **Step 3: Implement every meeting state and action.**

Chronicle rows distinguish Draft · not recorded, preparing, live channel mode, ending, generating final record, failed, done, and indexing state. Live header is 44 px with elapsed/status/channel text/save state/Compact/Privacy curtain/Stop and no decorative waveform. At 1280 render Notes ≥620 px plus a 300–360 px switchable Transcript/Intelligence lane preserving scroll/selection; enable remembered three-lane only at ≥1440; use focus-trapped sheets below 1180. Virtualize final transcript, replace interims, group final rows, use 32 px auto-follow threshold, and expose Copy/Insert excerpt/Ask. Intelligence renders rolling summary, actionables, context queue, evidence, user-wins actionable status, proposal creation failures, and no direct note writes. Final strata are notes, generated recap, proposed additions, actions, transcript, inquiry. Show exact diff/rationale/time/base revision; retain decision history. Preserve generated titles only when user title is empty. Delete cancels live/finalization and states permanent storage, linked-inquiry survival, and derived-index staleness.

- [ ] **Step 4: Run all meeting and shared-evidence tests.**

Run: `cd app && npm test -- src/renderer/src/features/meetings src/renderer/src/features/evidence src/renderer/src/features/inquiry && npm run typecheck`

Expected: PASS across create/detect/preflight/live/compact/stop/final/review/delete, responsive lanes, transcript keyboard actions, proposal failure/retry, note conflict, delayed index state, and meeting-scoped Ask.

- [ ] **Step 5: Commit Chronicle and meetings.**

```bash
git add app/src/renderer/src/features/meetings
git commit -m "feat(renderer): complete chronicle meeting dossiers"
```

### Task 22: Complete Sources, authorization, sync, and purge

**Files:**
- Create: `app/src/renderer/src/features/sources/descriptors.ts`
- Create: `app/src/renderer/src/features/sources/selectors.ts`
- Create: `app/src/renderer/src/features/sources/listeners.ts`
- Create: `app/src/renderer/src/features/sources/SourcesIndex.tsx`
- Create: `app/src/renderer/src/features/sources/SourcesLedger.tsx`
- Create: `app/src/renderer/src/features/sources/SourceDetail.tsx`
- Create: `app/src/renderer/src/features/sources/SourceConfig.tsx`
- Create: `app/src/renderer/src/features/sources/Authorization.tsx`
- Create: `app/src/renderer/src/features/sources/sources.test.tsx`

**Interfaces:** Consumes connector/sync REST and updates, OAuth routes, directory chooser, graph version, and Task 14 Control. Produces a 56 px source ledger grouped Needs attention/Active/Off/System, typed configuration that round-trips unknown keys, explicit Connect/poll/Disconnect, coherent global sync, stale-index state, and two-step purge.

- [ ] **Step 1: Write failing priority, round-trip, OAuth, and sync tests.**

```tsx
it("prioritizes active work over availability-derived groups", () => {
  expect(deriveSourceGroup(connector({ syncing: true, enabled: true, available: false }))).toBe("active_work")
  expect(deriveSourceGroup(connector({ syncing: false, enabled: true, available: false }))).toBe("needs_attention")
})

it("round-trips config keys unknown to the target renderer", async () => {
  renderSourceConfig({ config: { sync_window_days: 30, provider_extension: "keep" } })
  await user.clear(screen.getByLabelText("Sync window in days"))
  await user.type(screen.getByLabelText("Sync window in days"), "60")
  await user.click(screen.getByRole("button", { name: "Save" }))
  expect(api.updateConnector).toHaveBeenCalledWith(expect.objectContaining({
    config: { sync_window_days: 60, provider_extension: "keep" },
  }))
})
```

- [ ] **Step 2: Run Sources tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/sources/sources.test.tsx`

Expected: FAIL because Sources descriptors/listeners/views are absent.

- [ ] **Step 3: Implement the complete source ledger.**

Derive state in this order: syncing/cancelling; enabled unavailable; enabled available; disabled; hidden Taurine System. Enabling offers Enable and sync or Enable without syncing. Disable copy states stored rows retained, immediate exclusion/derived staleness, and clean-sync recommendation. Descriptors cover Apple Notes permission, Obsidian vault, ordered Folders, iMessage Full Disk Access/contact/advanced DB paths, Claude glob, Codex glob list, Gmail/Outlook authorization + sync days + contacts, Calendar authorization + past/future days + system calendars, and Docs authorization + modified-within days + maximum documents; ranges come from server/product constants, unknown keys round-trip, and folder paths come only from `chooseDirectory`. Connect opens only the returned allowlisted HTTPS URL, polls waiting/authorized/expired/error, never receives tokens, and leaves enable/sync user-controlled; environment setup lives under Advanced developer setup. Use server `done/total` only when total >0, distinguish extraction from knowledge build, cancel and await terminal state, keep the old immutable Atlas during build, and atomically swap on graph-version fetch. Purge requires two clicks, names document count/provider boundary/no undo, disables the source, marks index stale, refetches affected objects, and is blocked at noncancellable critical work.

- [ ] **Step 4: Run Sources, OAuth, graph reconciliation, and type tests.**

Run: `cd app && npm test -- src/renderer/src/features/sources src/renderer/src/services/sidecar src/renderer/src/features/atlas && npm run typecheck`

Expected: PASS for all ledger groups, known descriptors, unknown-key preservation, enable/disable, OAuth states, sync/cancel/failure, old-Atlas retention, hidden Taurine source, purge boundaries, and removed-open-object state.

- [ ] **Step 5: Commit Sources.**

```bash
git add app/src/renderer/src/features/sources
git commit -m "feat(renderer): complete sources control workflow"
```

### Task 23: Complete Profile and Diagnostics control workbenches

**Files:**
- Create: `app/src/renderer/src/features/profile/listeners.ts`
- Create: `app/src/renderer/src/features/profile/ProfileWorkbench.tsx`
- Create: `app/src/renderer/src/features/profile/ProfileEditor.tsx`
- Create: `app/src/renderer/src/features/profile/GenerationRail.tsx`
- Create: `app/src/renderer/src/features/profile/profile.test.tsx`
- Create: `app/src/renderer/src/features/diagnostics/DiagnosticsWorkbench.tsx`
- Create: `app/src/renderer/src/features/diagnostics/SessionLog.tsx`
- Create: `app/src/renderer/src/features/diagnostics/diagnostics.test.tsx`

**Interfaces:** Consumes profile REST/updates/generation and session-log REST. Produces Interpretation profile empty/populated/editing/confirm/generating/failed states and operational Diagnostics. Profile is explicitly noncitable. Diagnostics exports safe runtime metadata and filters ASR/chunk/retrieval/grounding/summary/chat/control/error.

- [ ] **Step 1: Write failing generation-retention and redaction tests.**

```tsx
it("retains the saved profile while replacement generation fails", async () => {
  renderProfile({ markdown: "Existing profile" })
  await user.click(screen.getByRole("button", { name: "Generate replacement" }))
  emitProfileGeneration({ status: "error", detail: "Generation failed" })
  expect(screen.getByText("Existing profile")).toBeVisible()
  expect(screen.getByRole("button", { name: "Retry" })).toBeVisible()
})

it("redacts diagnostics paths and credentials by default", async () => {
  renderDiagnostics({ records: [unsafeDiagnosticRecord] })
  await user.click(screen.getByRole("button", { name: "Copy diagnostics" }))
  expect(await clipboard.readText()).not.toMatch(/\/Users\/private|access_token|refresh_token/)
})
```

- [ ] **Step 2: Run Profile/Diagnostics tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/profile src/renderer/src/features/diagnostics`

Expected: FAIL because the workbenches are absent.

- [ ] **Step 3: Implement exact states without invented progress.**

Label Profile `Interpretation profile` and state that it resolves names/projects/interests and is not citable evidence. Require nonempty manual content. Confirm overwrite; retain current content until generation succeeds. Render only server phases Surveying, Interests, Projects & people, Career & studies, Synthesizing plus activity/nodes/tool calls/elapsed; no percent and no navigation cancellation. Disable manual Save during generation with an accessible reason. Diagnostics lists sessions by timestamp/ID, filters structured events, states logs are operational rather than durable meetings and may be absent, omits duplicated private text, and copies app/runtime/socket/build/calibration/safe errors with path/token redaction.

- [ ] **Step 4: Run Profile, Diagnostics, updates, and type tests.**

Run: `cd app && npm test -- src/renderer/src/features/profile src/renderer/src/features/diagnostics src/renderer/src/services/sidecar && npm run typecheck`

Expected: PASS for empty/edit/save/error, overwrite/generation/background/failure/success, every server phase, absent logs, all filters, and safe diagnostic copy.

- [ ] **Step 5: Commit Profile and Diagnostics.**

```bash
git add app/src/renderer/src/features/profile app/src/renderer/src/features/diagnostics
git commit -m "feat(renderer): add profile and diagnostics workbenches"
```

### Task 24: Finish runtime readiness, first run, privacy, compact mode, and utilities

**Files:**
- Modify: `app/src/renderer/src/features/runtime/RuntimeGate.tsx`
- Create: `app/src/renderer/src/features/runtime/FirstRunChecklist.tsx`
- Create: `app/src/renderer/src/features/runtime/runtime.test.tsx`
- Modify: `app/src/renderer/src/features/privacy/PrivacyCurtain.tsx`
- Modify: `app/src/renderer/src/features/privacy/PrivacyStatus.tsx`
- Create: `app/src/renderer/src/features/privacy/privacy.test.tsx`
- Create: `app/src/renderer/src/utility/call-prompt-main.tsx`
- Create: `app/src/renderer/src/utility/context-suggestion-main.tsx`
- Create: `app/src/renderer/src/utility/utility.css`
- Create: `app/src/renderer/src/utility/utility.test.tsx`
- Modify: `app/src/renderer/notif.html`
- Modify: `app/src/renderer/suggest.html`

**Interfaces:** Consumes startup, capabilities, privacy/call/context events, and native window factories. Produces exact startup polling, dismissible UI-schema-versioned first run, compact bounds/state, literal privacy statuses, neutral curtain, and sanitized utility renderers. Utility entries have no store, sidecar, notes, transcript, connector, or editor imports.

- [ ] **Step 1: Write failing readiness, curtain, and utility redaction tests.**

```tsx
it("uses capped startup backoff and pauses only after hidden for thirty seconds", async () => {
  await driveStartupPolls([false, false, false, false])
  expect(startupCallTimes()).toEqual([0, 250, 750, 1750, 3750])
  hideFor(30_001)
  await clock.advanceByAsync(4_000)
  expect(startupCallTimes()).toHaveLength(5)
})

it("removes private content from the rendered tree while curtain is on", () => {
  renderCurtainedLiveMeeting({ notes: "private notes", transcript: "private speech" })
  expect(document.body).not.toHaveTextContent("private notes")
  expect(document.body).not.toHaveTextContent("private speech")
  expect(screen.getByRole("button", { name: "Stop" })).toBeVisible()
})

it("redacts context utility content while curtain is on", () => {
  renderContextUtility({ curtain: true, title: "Secret", snippet: "Private" })
  expect(screen.getByText("Relevant context available")).toBeVisible()
  expect(document.body).not.toHaveTextContent("Secret")
})
```

- [ ] **Step 2: Run runtime/privacy/utility tests and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/features/runtime src/renderer/src/features/privacy src/renderer/src/utility`

Expected: FAIL because first-run and utility entry modules are absent and privacy state is incomplete.

- [ ] **Step 3: Implement exact readiness and privacy behavior.**

Poll `/startup` at 250 ms, 500 ms, 1 second, then 2 seconds; after 30 seconds hidden pause and resume immediately on show. Render actual phase/detail/elapsed; on error retain shell/preferences/curtain and offer safe diagnostics, retry, setup guide. On ready connect sockets before hydrating all domains and lazy-load graph/editor/Markdown only when opened. First run covers credential readiness status, just-in-time mic, macOS 14.2 system-audio note, recommended sources, Off/Ask/Auto-start call detection, and privacy limitation/shortcut; it is dismissible and stores only checklist/schema preference.

Map capability states to exact copy: Protection requested, Limited on this Mac, Unavailable, Privacy curtain on. Curtain replaces all main private DOM with elapsed/channel/Stop/Turn curtain off, redacts utilities, and disables preview. Compact uses the same BrowserWindow and exact 440×680 default, 420×560 minimum, 560×900 maximum; always-on-top is opt-in and off by default. Call detection defaults to Ask after upgrade/first run; utility states cover detected/ask/10-second auto-start/starting/failure/dismissed with Pause/Start now/Dismiss, one prompt per call, and the retained five-second idle rearm rule. Context utility caps three, lasts 12 seconds, and offers Open/Dismiss. Both mirror actions in tray/command center and never connect to sidecar.

- [ ] **Step 4: Run runtime/privacy/utility/native tests.**

Run: `cd app && npm test -- src/renderer/src/features/runtime src/renderer/src/features/privacy src/renderer/src/utility src/main && npm run typecheck && npm run build`

Expected: PASS for exact polling, protected 503/socket 1013 restart, checklist persistence, permission timing, all privacy copy, curtain DOM removal/restoration, compact geometry, call rearm/countdown, context cap/expiry/redaction, and zero utility sockets.

- [ ] **Step 5: Commit readiness and privacy completion.**

```bash
git add app/src/renderer/src/features/runtime app/src/renderer/src/features/privacy app/src/renderer/src/utility app/src/renderer/notif.html app/src/renderer/suggest.html
git commit -m "feat(renderer): complete readiness privacy and utility states"
```

### Task 25: Enforce accessibility, keyboard, zoom, RTL, and localization contracts

**Files:**
- Create: `app/src/renderer/src/test/accessibility-contracts.test.tsx`
- Create: `app/src/renderer/src/i18n/i18n.test.ts`
- Modify: `app/src/renderer/src/app/AppShell.tsx`
- Modify: `app/src/renderer/src/app/commands.ts`
- Modify: `app/src/renderer/src/components/primitives/Button.tsx`
- Modify: `app/src/renderer/src/components/primitives/Dialog.tsx`
- Modify: `app/src/renderer/src/components/primitives/Menu.tsx`
- Modify: `app/src/renderer/src/components/primitives/Sheet.tsx`
- Modify: `app/src/renderer/src/components/layout/SplitPane.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasOutline.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasControls.tsx`
- Modify: `app/src/renderer/src/features/meetings/NotesDocument.tsx`
- Modify: `app/src/renderer/src/features/meetings/TranscriptLane.tsx`
- Modify: `app/src/renderer/src/features/meetings/ProposalDiff.tsx`
- Modify: `app/src/renderer/src/features/inquiry/InquiryComposer.tsx`
- Modify: `app/src/renderer/src/features/inquiry/AssistantTurn.tsx`
- Modify: `app/src/renderer/src/features/evidence/EvidenceInspector.tsx`
- Modify: `app/src/renderer/src/features/sources/SourcesLedger.tsx`
- Modify: `app/src/renderer/src/features/profile/ProfileWorkbench.tsx`
- Modify: `app/src/renderer/src/features/privacy/PrivacyCurtain.tsx`
- Modify: `app/src/renderer/src/styles/reset.css`
- Modify: `app/src/renderer/src/styles/primitives.css`
- Modify: `app/src/renderer/src/styles/shell.css`

**Interfaces:** Consumes the completed stable workbenches. Produces unit-level semantic/keyboard/reflow guarantees before Electron E2E. Global shortcut meanings are stable and platform labels localized; single-character Atlas shortcuts can be disabled.

- [ ] **Step 1: Write the complete failing contract matrix.**

```tsx
it.each(stableWorkbenchFixtures)("has no automated axe violations: $name", async ({ renderWorkbench }) => {
  const { container } = renderWorkbench()
  const results = await axe.run(container)
  expect(results.violations).toEqual([])
})

it.each(["atlas", "chronicle", "inquiry", "sources", "profile", "privacy", "diagnostics"])(
  "retains every action at 200%% zoom in %s", (workbench) => {
    setViewport({ width: 960, height: 640, zoom: 2 })
    renderNamedWorkbench(workbench)
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(document.documentElement.clientWidth)
    expect(allRequiredActions(workbench).every((element) => element.tabIndex >= 0)).toBe(true)
  },
)
```

- [ ] **Step 2: Run accessibility/i18n tests and record each real failure.**

Run: `cd app && npx vitest run src/renderer/src/test/accessibility-contracts.test.tsx src/renderer/src/i18n/i18n.test.ts`

Expected: FAIL on the first incomplete accessible name, focus order, reflow, pseudo-locale, RTL, target-size, or catalog assertion; keep the test output with the implementing commit.

- [ ] **Step 3: Fix only demonstrated contract failures.**

Cover landmarks/headings, tree/treegrid/list semantics, modal focus containment/return, live-region throttling, editor labels, proposal diff reading order, status text beyond color, 24×24 targets, 40–44 px primary/destructive/meeting targets, 200% reflow, 35% pseudo expansion, RTL logical layout, forced colors, reduced motion, localized date/number/key names, and complete string-catalog lookup. Shortcut precedence is editor → modal/menu → focused workbench → global; `⌘1/2/3`, `⌘,`, `⌘K`, `⌘⇧M`, `⌘⇧L`, `⌘⇧P`, `⌘.`, and Atlas local keys remain available through named controls.

- [ ] **Step 4: Run accessibility/i18n plus the full frontend unit suite.**

Run: `cd app && npm test && npm run lint && npm run typecheck`

Expected: PASS with at least the 61-test pre-redesign floor plus all added tests; no axe violations in stable fixtures and no uncatalogued visible string failures.

- [ ] **Step 5: Commit accessibility completion.**

```bash
git add app/src/renderer/src app/src/renderer/src/styles
git commit -m "fix(renderer): satisfy accessibility and localization contracts"
```

### Task 26: Cut the sole renderer entry over atomically

**Files:**
- Modify: `app/src/renderer/src/main.tsx`
- Modify: `app/src/renderer/src/env.d.ts`
- Create: `app/src/renderer/src/app/cutover.test.tsx`

**Interfaces:** Consumes Tasks 11–25. The production entry imports only `app/App`, local fonts, and target token/style sheets. There is no runtime switch, query parameter, route, environment variable, or alternate entry that mounts the old UI.

- [ ] **Step 1: Write the failing sole-entry contract test.**

```ts
it("mounts the Knowledge Atlas app with no interface selector", async () => {
  const source = await readSource("src/renderer/src/main.tsx")
  expect(source).toContain('from "./app/App"')
  expect(source).not.toMatch(/legacy|oldUi|interfaceFlag|FeedView|MapView|DataView/)
})
```

- [ ] **Step 2: Run the cutover test and confirm the expected failure.**

Run: `cd app && npx vitest run src/renderer/src/app/cutover.test.tsx`

Expected: FAIL because `main.tsx` still imports the old root `App.tsx` and old theme.

- [ ] **Step 3: Switch the one production entry.**

Import `@fontsource/ibm-plex-sans`, `@fontsource/ibm-plex-mono`, target reset/tokens/typography/primitives/shell styles, and `./app/App`. Mount one Redux Provider and one App. Remove old root App/theme imports from the entry. Do not expose a selection mechanism and do not keep an alternate main renderer entry. Old unreachable files remain only until Task 28's deletion gate so this commit is reviewable and bisectable.

- [ ] **Step 4: Run cutover smoke, full unit, type, and build gates.**

Run: `cd app && npm test && npm run lint && npm run typecheck && npm run build`

Expected: PASS; opening the production index entry renders Knowledge coordinates, does not construct old sidecar clients, and the build manifest has one stateful main renderer plus two stateless utilities.

- [ ] **Step 5: Commit the atomic cutover.**

```bash
git add app/src/renderer/src/main.tsx app/src/renderer/src/env.d.ts app/src/renderer/src/app/cutover.test.tsx
git commit -m "feat(renderer): switch production to knowledge atlas"
```

### Task 27: Prove acceptance stories, performance, visuals, native behavior, and privacy

**Files:**
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

**Interfaces:** Consumes the production Electron build and deterministic fake sidecar. Produces acceptance stories A–N, viewport/theme/zoom/RTL/forced-color/reduced-motion screenshots, renderer budgets, safe-log checks, and a signed manual checklist. Test fixtures implement current REST and all three WebSockets plus proposal/resume/OAuth additions; they do not bypass the gateway.

- [ ] **Step 1: Write the failing end-to-end and performance assertions.**

```ts
test("C. reconnects honestly without replaying audio", async ({ taurine, sidecar }) => {
  await taurine.startMeeting("full")
  await sidecar.dropSessionSocket()
  await expect(taurine.header).toContainText("Reconnecting · audio may be missing")
  expect(sidecar.binaryFramesAfterDrop).toHaveLength(0)
  await sidecar.resumeWithSnapshot()
  await expect(taurine.transcript).toContainText("Audio may be missing during reconnect")
})

test("Atlas holds the 5k-node frame budget", async ({ taurine }) => {
  await taurine.loadScenario("atlas-5000")
  expect(await taurine.measureAtlasFps(5_000)).toBeGreaterThanOrEqual(30)
  expect(await taurine.measureCommandP95("atlas.select", 200)).toBeLessThan(100)
})
```

- [ ] **Step 2: Run the new suites and confirm the expected failures before fixture/test completion.**

Run: `cd app && npm run test:e2e`

Expected: FAIL because Playwright configuration, fake-sidecar behaviors, scenarios, and snapshots are not yet complete.

- [ ] **Step 3: Implement deterministic full-story verification.**

Encode stories A–N one-for-one: start/finish; mic-only degradation; honest reconnect; note-loss prevention; accept/undo final proposal; transparent evidence; fallback/interrupted; pointer-free Atlas; source sync retaining old Atlas; purge boundaries; safe profile generation; privacy curtain; sole replacement; every AI edit reviewable. Scenario sizes are Atlas 5,000 nodes, source/Chronicle lists 10,000 rows, transcript 3,000 chunks, and inquiry 200 turns. Assert ≥30 Atlas fps, virtualized-list 60 fps, p95 selection/command <100 ms excluding network/model, bounded DOM row counts, and no unbounded heap growth over repeated coordinate switches. Visual matrix covers 1440/1280/960, compact 440×680, light/dark, 200% zoom, pseudo 35%, RTL, reduced motion, privacy, every stable empty/loading/error state, and utility windows.

Implement old-UI scanning for root legacy imports/routes/selectors, old dark/yellow token signatures, Feed/Map/Data labels, duplicate renderer entries, and interface flags. Implement sensitive-log scanning over logger/breadcrumb arguments and built source maps for note/transcript/profile/passage/prompt/answer/token/credential payload fields. The manual checklist records date/build/OS/device/result/evidence for VoiceOver Canvas→Outline parity, keyboard-only A–N, 200% zoom, forced colors, actual ScreenCaptureKit capture in requested/limited/curtain states, mic-only/full capture, call prompt focus behavior, compact restoration, tray/dock/single-instance, sleep/wake, hide/show, and packaged-app signing/launch.

- [ ] **Step 4: Run automated release gates and execute the manual checklist on the reference Mac.**

Run:

```bash
cd app
npm run build
npm run check:bundle
npm run check:sensitive-logs
npm run test:e2e
npm run test:visual
```

Expected: every automated story/performance/accessibility/visual/privacy gate passes; the checklist contains no unchecked release-blocking row, includes ScreenCaptureKit evidence, and records any environment-only nonblocking observation explicitly. The newly written old-UI scanner is intentionally first required in Task 28, where its red result proves the deletion test before removal.

- [ ] **Step 5: Commit verification assets and evidence.**

```bash
git add app/playwright.config.ts app/e2e app/scripts/check-old-ui-removed.mjs app/scripts/check-sensitive-logs.mjs docs/superpowers/verification/2026-07-20-knowledge-atlas-release-checklist.md
git commit -m "test: verify knowledge atlas end to end"
```

### Task 28: Remove the old UI and run the final release gate

**Files:**
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

**Interfaces:** Consumes the complete target and Task 27 scanners. Produces one shippable interface with no old renderer, CSS fallback, alternate route/entry, or native ownership duplicate. Existing databases remain opened in place by additive core initialization.

- [ ] **Step 1: Run the deletion scanner before removal and confirm the expected failure.**

Run: `cd app && npm run check:old-ui`

Expected: FAIL and list each remaining old root, view, component, library coordinator, native module, selector/theme signature, and utility entry.

- [ ] **Step 2: Delete the enumerated files and resolve imports into target ownership.**

Delete every listed path. If a current low-level behavior is still imported, move its tested behavior into the target module named in the file map before deletion; do not retain an alias/re-export. Remove obsolete tests with their implementation only after equivalent target tests are passing. Keep `public/pcm-worklet.js` because capture still loads it. Remove Tailwind's Vite plugin and run `cd app && npm uninstall react-force-graph-2d @tailwindcss/vite tailwindcss`; the target uses Canvas/d3 and checked-in CSS. Leave exactly main, call prompt, and context suggestion renderer entries.

- [ ] **Step 3: Prove deletion and privacy invariants.**

Run:

```bash
cd app
npm run check:old-ui
npm run check:sensitive-logs
rg -n 'FeedView|MapView|DataView|ConnectorsView|legacy|interfaceFlag|yellow|window\.api' src electron.vite.config.ts package.json
```

Expected: both scripts PASS and `rg` exits 1 with no matches. Any legitimate copy containing one of those bare words must be expressed through the message catalog without recreating an old selector/route marker and explicitly allowlisted in the scanner with its exact file/key.

- [ ] **Step 4: Run the complete release gate from a clean build.**

Run:

```bash
cd core
uv run ruff check taurine_core tests
uv run pytest tests -q
cd ../app
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

Expected: Ruff/lint/typecheck/build/check scripts exit 0; core reports at least 981 passed and exactly the existing 2 skipped plus all new tests; Vitest reports at least the verified 61-test floor plus all new tests; stories A–N, accessibility, performance, visual, and native checklist gates pass; bundle output remains within 1.25 MB initial JS, 140 KB CSS, and 900 KB per lazy feature.

- [ ] **Step 5: Review the final diff for atomic replacement and commit deletion.**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors, only Knowledge Atlas implementation/verification paths are changed, old paths are deletions, and there are no unrelated working-tree changes staged.

```bash
git add -A app/src app/electron.vite.config.ts app/package.json app/package-lock.json
git commit -m "refactor: remove superseded taurine interface"
```

---

## Spec-requirement-to-task completion matrix

| Finalized design requirement | Implemented in | Completion evidence |
|---|---:|---|
| §0–1 product model, power-user density, keyboard parity, trust, success gates, and non-goals | 13–18, 21–28 | shell/feature contracts; A–N; final gates |
| §2 preserve existing capabilities and Python trust boundary while removing five-tab coordination | 2–12, 15–28 | core regression floor; gateway parsers; story M; deletion scanner |
| §3 three coordinates, separate Control, persistent context/history/pins, exception-only operations, max-three comparison | 12, 14, 18 | reducer/shell/command/comparison tests |
| §4 one stateful main window, 1280×820/960×640, exact breakpoints, compact bounds, nonactivating utilities, tray/dock, privacy states | 9, 10, 14, 20, 24, 27 | geometry/native/shell/utility/privacy tests and native checklist |
| §5 selection ≠ activation, preview/open/pin grammar, command center, focus zones, shortcuts | 14, 18, 25 | keyboard and command-precedence tests; keyboard E2E |
| §6 territory-first Atlas, deterministic stable geography, four LODs, time band, lenses, local direction, live overlay, equal Outline, no invalid edge-reason call | 15, 17, 18, 27 | Atlas/evidence/performance/a11y tests; story H |
| §7.1 startup phases/backoff, domain hydration, nonblocking first run, just-in-time permissions | 12, 24 | runtime/hydration/capability tests |
| §7.2 complete meeting lifecycle, preflight, two-lane ordinary Live, transcript/intelligence, sole notes writer, honest resume, immediate Stop, final review/delete | 2–6, 9, 19–21, 24, 27 | proposal/session/controller/dossier tests; stories A–E/L/N |
| §7.3 Inquiry scope wording, turn anatomy, evidence outcomes, inspector, reconnect/no resend, thread management | 4, 11, 15, 16, 27 | Inquiry/gateway/evidence tests; stories F–G |
| §7.4 source ledger, exact config, desktop OAuth, global sync/cancel, old Atlas, purge boundary | 7, 8, 22, 27 | core OAuth/source tests; Sources tests; stories I–J |
| §7.5 Interpretation profile states and server-only phase rail | 12, 23, 27 | profile tests; story K |
| §7.6–7.7 Diagnostics/session logs and capability-gated native source open | 9, 11, 15, 23 | diagnostic redaction and open-source failure tests |
| §8 complete capability/state inventory with specific rendered recovery | 11, 12, 14–24 | exhaustive parser/slice/fixture matrices; stable-state screenshots |
| §9 visual system, exact light/dark tokens, typography, density, no gradients/sparkles/emoji/cards | 13, 25, 27, 28 | token/visual/pseudo/forced-color tests; old-theme scanner |
| §10 primitives, layout, knowledge, meeting, and Ask component behavior; Lucide plus local source marks | 13–16, 18, 21–23 | component unit/a11y/visual tests |
| §11 shortcut map, semantics, announcements, AA/reflow/targets/editor accessibility/i18n | 13–16, 18–25, 27 | axe, keyboard, 200% zoom, pseudo, RTL, VoiceOver checklist |
| §12 Redux-only global state, finite states, one gateway, hydration/reconciliation, optimistic writes, redaction, code splitting/performance | 1, 11, 12, 16–20, 25, 27 | store/gateway/controller tests; sensitive-log/bundle/performance gates |
| §13.1–13.5 retained REST and three socket/desktop contracts, all update/event variants, chats newly consumed | 9–12, 15–16, 20–24 | exhaustive parser/OpenAPI/gateway/native tests |
| §13.6 additive proposals/commands/revisions/idempotency/undo, legacy suggestion and `ai_lines` reads | 2–4, 19, 21, 27 | core transaction/conflict/legacy tests; stories D/E/N |
| §13.7 in-memory retained live session, 60-second disconnect, 10/0 Stop, matching-token ownership, one-winner finalize | 5, 6, 20, 27 | deterministic registry/socket/state-machine race tests; story C |
| §13.8 connector authorization endpoints, PKCE/state/loopback, OS vault, no token exposure, environment compatibility | 7, 8, 9, 22, 27 | OAuth/vault/IPC/log/E2E tests |
| §13.9 meeting-linked inquiry referential cleanup; §13.10 intentionally absent graph/pagination/socket/replay/retry additions | 4, 16, 28 | delete/refetch tests and final API/diff review |
| §14 exact final module decomposition and complete old-architecture removal | 9–24, 26, 28 | target path checks; one entry; deletion scanner; story M |
| §15 all automated/accessibility/native/performance gates and acceptance stories A–N | 1, 25, 27, 28 | release commands and signed checklist |
| §16 risk mitigations: Canvas/Outline equality, immutable graph swaps, durable notes, race ownership, vault/redaction, literal privacy, atomic replacement | 2–8, 11–28 | risk-specific tests and scanners |
| §17 existing-capability completeness matrix | 11, 15–24, 27 | contract parser coverage and story/scenario matrix |
| §18 final decisions: Atlas as orientation, three coordinates, generated edit review, immediate recoverable Stop, atomic release | 14, 17–21, 26–28 | shell/Atlas/controller/session/cutover gates |
| §19 design self-review record | 1–28 | this execution matrix, per-task Interfaces, A–N ownership, and Final execution audit |

### Acceptance-story ownership

| Story | Owning test in `app/e2e/knowledge-atlas.spec.ts` | Primary implementation tasks |
|---|---|---:|
| A. Start and finish a meeting | `A start and finish a meeting` | 5–6, 19–21 |
| B. Degrade to mic only | `B degrade to mic only` | 9–10, 20–21 |
| C. Reconnect honestly | `C reconnect honestly` | 5–6, 11, 20 |
| D. Prevent note loss | `D prevent note loss` | 2–3, 9, 19–20 |
| E. Accept and undo a final proposal | `E accept and undo a final proposal` | 2–4, 19, 21 |
| F. Ask with transparent evidence | `F ask with transparent evidence` | 15–16 |
| G. Handle low-confidence and interrupted turns | `G low confidence and interrupted turns` | 11, 15–16 |
| H. Investigate spatially without a pointer | `H investigate spatially without a pointer` | 17–18, 25 |
| I. Sync sources without losing the old Atlas | `I sync sources without losing the old Atlas` | 11, 18, 22 |
| J. Purge with exact boundaries | `J purge with exact boundaries` | 22 |
| K. Generate profile safely | `K generate profile safely` | 23 |
| L. Use privacy curtain during capture | `L privacy curtain during capture` | 9, 20, 24 |
| M. Replace, not coexist | `M replace not coexist` | 26, 28 |
| N. Keep every AI note change reviewable | `N every ai note change is reviewable` | 2–4, 19–21 |

## Final execution audit

Before declaring implementation complete, the executing agent must run `cd "$WORKTREE_ROOT"`, revalidate that `git rev-parse --git-dir` ends in `/worktrees/ui-knowledge-atlas`, and verify all of the following there:

1. Every task commit exists in order or an explicitly documented squash preserves the same review boundaries.
2. `git diff --check` is empty and `git status --short` contains no unrelated staged path.
3. `app/src/renderer/src/app/App.tsx` remains below approximately 120 substantive lines and contains no socket, fetch, capture, graph, chat, or editor ownership.
4. `rg -n 'setMarkdown|updateMeeting\(|note-commands' app/src/renderer/src` shows canonical note writes only in `MeetingDocumentController.ts`/its API adapter and tests; proposal producers never call them.
5. `rg -n 'new WebSocket' app/src/renderer/src` shows constructors only in the three connection modules; utility renderers show none.
6. `rg -n 'edge-reason' app/src/renderer/src` exits 1.
7. `rg -n 'T[O]DO|T[B]D|F[I]XME|place[h]older|coming[ ]soon|implement[ ]later' core/taurine_core app/src app/e2e docs/superpowers/verification` exits 1.
8. All Markdown fences in this plan and the release checklist are balanced, headings are unique within their intended hierarchy, and every path named as Create/Modify/Delete is either present or intentionally deleted at the corresponding commit.
9. The Task 28 command block passes with the stated test floors and bundle/performance limits.
10. The requirement matrix and A–N ownership table have no uncovered row, and the manual checklist contains dated evidence for every §15.4–15.5 check.
