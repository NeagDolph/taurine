# Taurine Quiet Canvas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Taurine's five-tab dashboard with the Quiet Canvas desktop workspace: a meeting-note-centered object UI with truthful proposals and evidence, optional accessible Connections, bounded comparison, progressive disclosure, deterministic privacy, and no shipped legacy interface.

**Architecture:** The FastAPI sidecar remains authoritative and receives only revision/idempotency, live-resume, review-actionable, and note-edit truthfulness repairs. Electron main owns window/menu/tray behavior, encrypted crash recovery, and the cross-window privacy-curtain epoch; a narrow preload exposes typed commands. The React renderer becomes a typed object router over one gateway, one updates projection, one session transport, one conversation hub, focused repositories, and a single meeting-document writer; feature scenes are lazy and hidden routes are unmounted.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic 2, SQLite, pytest/pytest-asyncio; Electron 39, React 19, TypeScript 5.9, Vite/electron-vite, Tailwind CSS 4, BlockNote 0.51, react-markdown 10, react-force-graph-2d 1.29 (lazy only), Vitest 4, Testing Library, Playwright Electron, axe-core, and Electron `safeStorage`.

## Global Constraints

- Read `docs/superpowers/specs/2026-07-20-taurine-ui-quiet-canvas-design.md` before Task 1; it is authoritative when wording or behavior is not repeated here.
- The final renderer has no persistent app-wide sidebar, tab row, status footer, dashboard, cockpit, old route, alternate entry, runtime interface flag, or compatibility UI. Intermediate work is testable on the implementation branch, but the only production entry after Task 29 is Quiet Canvas.
- The sidecar remains authoritative for meetings, transcripts, final records, suggestions, chats, connector state, profile, and knowledge builds. Renderer state is a replaceable projection; Electron's recovery journal is encrypted, dirty-only, and noncanonical.
- Only `MeetingDocumentController` may mutate canonical BlockNote content. Chat, inline Ask, final review, and live suggestions create `NoteProposal`; content enters the document only through explicit Accept and remains undoable.
- Wire names are snake_case; renderer names are camelCase. Unknown additive wire fields are retained where feasible; unknown event discriminants are content-free diagnostics and never crash the shell.
- Exact evidence vocabulary is `Cited`, `Consulted`, `Retrieved`, `Evidence policy passed`, `Low-confidence fallback`, `No supported evidence`, `From earlier context`, `Citation checks passed`, `Interrupted / stopped before verification`, and `Calibration not validated`. Never render `Verified`, `Fact checked`, or `True` as an evidence result.
- Deduplicate evidence IDs by precedence `Cited > Consulted > Retrieved`. Interrupted partials have no cited-support set even if their raw text contains citation-shaped tokens. Fallback passages stay `Tentative`; facts/paths remain withheld when the sidecar says fallback.
- `seed_note_ids` are retrieval starting points, not an exclusive corpus. Comparison Ask sends at most one `meeting_id` plus distinct valid note seeds; it never invents arbitrary-object or multi-meeting hard scope.
- The local graph request is exactly `GET /graph/local?center={note_id}&depth={1..3}`. `POST /graph/edge-reason` requires passage IDs; current `GraphEdge` exposes note endpoints, so no edge-reason action ships without an additive passage-pair contract.
- Default main-window bounds are `1120×780`; minimum is `840×600`; optional user-selected compact meeting bounds are `840×660`. Taurine never automatically raises, moves, or resizes the main window for call detection, capture, sync, suggestions, or review completion.
- Responsive thresholds are `840`, `960`, `1180`, `1320`, and `1440` CSS px. No supported scene creates horizontal document scroll. A docked meeting surface never shrinks the editor below `620 px`; three-column Compare requires `≥1320 px` and `≥400 px` per dossier.
- The persistent title bar is `48 px`. Open stays text-labeled at every supported width. Meeting/editor geometry, Context, Transcript, Compare, Settings, and Connections follow spec §§5.1–5.10 exactly.
- Light tokens: canvas `#F4F2EC`, sheet `#FFFEFA`, sheet-muted `#EBE8E0`, ink `#20231F`, ink-muted `#646960`, line `#D4D1C8`, line-strong `#A4A79F`, accent `#245E50`, accent-ink `#FFFFFF`, accent-soft `#DCEBE5`, info `#315E78`, warning `#7A5716`, danger `#963F39`, recording `#B4463B`, focus `#145DAD`.
- Dark tokens: canvas `#171916`, sheet `#20231F`, sheet-muted `#292D28`, ink `#F1F0EA`, ink-muted `#ADB2AA`, line `#3B403A`, line-strong `#646A62`, accent `#78B8A2`, accent-ink `#102B24`, accent-soft `#294139`, info `#7FB0C9`, warning `#D3A653`, danger/recording `#E18B82`, focus `#80B7FF`.
- UI type uses the system stack. Editorial titles/body use bundled Source Serif 4 Variable with Georgia fallback. No user-facing text is below `12 px`; editor body is `17/28`; scene heading is `24/30`; meeting title is `32/38` weight `550`.
- Primary controls are `40–44 px` high; icon-only controls are at least `36×36`; recording and destructive controls are `44×44`; rows are at least `44 px`. Focus is a `2 px` `--focus` outline with `2 px` offset.
- Motion is opacity/translate only: standard `140 ms`, large sheet `180 ms`, easing `cubic-bezier(.2,.8,.2,1)`. Reduced motion removes force/traversal/countdown motion and sliding sheets. No gradients, blur, glow, bounce, decorative agent animation, emoji controls, tiny pills, or all-uppercase prose labels.
- Target WCAG 2.2 AA. Canvas is `aria-hidden`; Connections List contains every essential graph operation. Privacy curtain removes private DOM/accessibility-tree content rather than covering it. Every dialog/sheet has title, initial focus, contained Tab order, Escape policy, and focus return.
- `setContentProtection(true)` is retained where available but UI says `Protection requested`. macOS copy states ScreenCaptureKit limitations. Presentation-safe is best effort; privacy curtain is deterministic only inside Taurine.
- `safeStorage` recovery is permitted only when `isEncryptionAvailable()` is true and Linux backend is not `basic_text`. Never call `setUsePlainTextEncryption(true)` and never write note recovery to IndexedDB, localStorage, or plaintext.
- No remote analytics by default. Pilot events are content-free and local: never record note/transcript/query/title text, note IDs, evidence passages, file paths, model content, or OAuth data.
- Startup Desk budget is `≤1.25 MB` uncompressed JS, `≤120 KB` uncompressed CSS, `≤350 KB` gzip JS, and `≤35 KB` gzip CSS. No feature chunk exceeds `1.5 MB` uncompressed without a checked-in waiver. BlockNote is absent from Desk's initial graph unless an active-meeting idle preload runs; force-graph/CSS load only after Visual; Diagnostics loads only on its route.
- Warm-sidecar first content is `<500 ms`; Desk keyboard interactivity is `<1.5 s` p95 over 20 launches; route fallback appears within `100 ms`; normal editor input stays within one `16 ms` frame.
- Existing regression floor is 61 app Vitest tests, node/web typechecks, production build, and 981 core tests passing with 2 documented skips. Do not remove coverage to meet the floor.
- Execute this plan only from the assigned linked worktree. At execution start define `WORKTREE_ROOT="$(git rev-parse --show-toplevel)"` and require its path to end in `/.worktrees/ui-quiet-canvas`; abort before Task 1 if it does not. Every standalone command below deliberately re-resolves `git rev-parse --show-toplevel`, so a fresh shell cannot redirect writes to the primary checkout.
- Python commands run from `"$WORKTREE_ROOT/core"` with `uv run`; app commands run from `"$WORKTREE_ROOT/app"` with npm. Every task ends with a conventional scoped commit and leaves its scoped tests green.
- Do not modify retrieval ranking, enrichment, graph construction, ASR provider, model routing, calibration artifacts, or connector semantics except where a task names an exact contract repair.

---

## Target File Structure

Files marked `modify` exist now. Files marked `create` are target modules. Task 29 deletes the legacy files listed after the tree; no target file imports them afterward.

```text
app/
  package.json                                      modify: scripts/dependencies/gates
  package-lock.json                                 modify: locked dependency graph
  electron.vite.config.ts                           modify: renderer manifest/chunks
  vitest.config.ts                                  modify: setup and test isolation
  playwright.config.ts                              create: Electron E2E projects
  scripts/
    bundle-budget.mjs                               create: initial/chunk budget checker
    bundle-budget.test.mjs                          create: manifest traversal tests
    verify-quiet-canvas.mjs                         create: forbidden legacy/copy/file scan
  e2e/
    fake-sidecar.ts                                 create: deterministic REST/WS fixture
    electron-fixture.ts                             create: launch/control helper
    quiet-canvas.spec.ts                            create: full user stories
    accessibility.spec.ts                           create: keyboard/axe/zoom/curtain stories
    performance.spec.ts                             create: launch/route/long-document gates
  src/main/
    index.ts                                        modify: composition root only
    application-menu.ts                             create: labeled native destinations/commands
    window-state.ts                                 create: clamped bounds/compact preference
    recovery-journal.ts                             create: safeStorage ciphertext lifecycle
    privacy-curtain.ts                              create: global epoch/window acknowledgements
    local-telemetry.ts                              create: content-free rotating log
    tray.ts                                         modify: live label/start/stop/show/quit
    notifier.ts                                     modify: 356×116 explicit-start prompt
    suggest-notifier.ts                             modify: off-default/redacted 356×132 prompt
    call-notify-state.ts                            modify: auto-start preference policy
    sysaudio.ts                                     modify: unchanged capture, typed health
  src/preload/
    index.ts                                        modify: narrow typed IPC only
    index.d.ts                                      modify: exact TaurineApi
  src/renderer/
    index.html                                      modify: generic title/theme bootstrap
    notif.html                                      modify: quiet accessible prompt, no emoji
    suggest.html                                    modify: quiet/redactable prompt, no emoji
    src/
      main.tsx                                      modify: target bootstrap entry
      notif-main.ts                                 modify: curtain/policy events
      suggest-main.ts                               modify: curtain/redaction events
      bootstrap/
        RendererBootstrap.tsx                       create: service composition + readiness
        StartupScene.tsx                            create: phase/error/recovery UI
        services.ts                                 create: WorkspaceServices interface/factory
      workspace/
        WorkspaceShell.tsx                          create: titlebar/router/overlays branch
        WorkspaceTitlebar.tsx                       create: breadcrumb/Open/Ask/status/capture
        WorkspaceRouter.ts                          create: typed history store
        routeState.ts                               create: routes/refs/restore snapshots
        OpenAnything.tsx                            create: fixed Places + ranked results
        StatusPopover.tsx                           create: exception-only operations
        PrivacyCurtain.tsx                          create: neutral rendered subtree
        WorkspaceContext.tsx                        create: typed service/context hooks
      platform/
        electronBridge.ts                           create: browser-safe TaurineApi adapter
        windowPreferences.ts                        create: renderer commands for main bounds/theme
        recoveryBridge.ts                           create: DraftJournal port
        PreferenceStore.ts                          create: versioned nonprivate preferences
      data/
        SidecarGateway.ts                           create: REST decoder/error/abort boundary
        SidecarProjectionStore.ts                   create: one /updates projection
        MeetingRepository.ts                        create: meeting CRUD/conflict API
        KnowledgeRepository.ts                      create: graph/note/fact/idea cache
        ConnectorRepository.ts                      create: source CRUD/sync/config API
        ProfileRepository.ts                        create: profile read/write/generate API
        DiagnosticsRepository.ts                    create: startup/session log APIs
        ConversationHub.ts                          create: one /chat socket + stream reducer
        SessionTransport.ts                         create: one /session socket + audio controls
        DraftJournal.ts                             create: dirty recovery orchestration
        contracts/
          decode.ts                                 create: shared strict decoder helpers
          startup.ts                                create: StartupStatus
          updates.ts                                create: six UpdateEvent variants
          meeting.ts                                create: meeting/revision/actionables
          session.ts                                create: session controls/events/snapshot
          evidence.ts                               create: complete evidence payload
          chat.ts                                   create: complete chat messages/events
          knowledge.ts                              create: graph/note/facts/idea
          connector.ts                              create: connector/sync contracts
          profile.ts                                create: profile contracts
          diagnostics.ts                            create: session-log contracts
      meetings/
        DeskScene.tsx                               create: start/active/recent surface
        MeetingsCollectionScene.tsx                 create: searchable semantic list
        MeetingScene.tsx                            create: object scene/coordinator
        MeetingDocument.tsx                         create: BlockNote adapter/view
        MeetingDocumentController.ts                create: only canonical note writer
        CaptureController.ts                        create: mic/system lifecycle and stop flush
        LiveMeetingStore.ts                         create: transcript/summary/grounding reducer
        ContextMargin.tsx                           create: bounded live/review context
        TranscriptDrawer.tsx                        create: searchable/follow-safe transcript
        ReviewChanges.tsx                           create: proposal/actionable review
        ProposalQueue.ts                            create: proposal projection/accept/reject/undo
        proposalAnchor.ts                           create: deterministic anchor resolution
      ask/
        ObjectAskSheet.tsx                          create: meeting/note Ask surface
        ConversationScene.tsx                       create: durable standalone thread
        ConversationTurn.tsx                        create: streamed/persisted turn rendering
        EvidenceDisclosure.tsx                      create: evidence/process container
        EvidenceLedger.tsx                          create: cited/consulted/retrieved precedence
        EvidenceRef.tsx                             create: passage/fact/path link control
        ProcessDisclosure.tsx                       create: concise activity + advanced detail
        evidenceSemantics.ts                        create: trust-state derivation
      knowledge/
        KnowledgeNoteScene.tsx                      create: source document/why/open/compare
        ConnectionsScene.tsx                       create: scoped List/Visual scene
        ConnectionsList.tsx                        create: canonical semantic operations
        ConnectionsVisual.lazy.tsx                 create: lazy canvas enhancement
        connectionLayout.worker.ts                 create: deterministic settled geometry
        ConnectionFilters.tsx                      create: source/cluster/date/mechanism fields
        connectionsModel.ts                        create: relationships/LOD/stable keys
        CompareScene.tsx                            create: two/three-object dossiers
        comparisonSet.ts                           create: ephemeral capped selection/Ask scope
      settings/
        SettingsIndexScene.tsx                     create: labeled settings rows
        SourcesScene.tsx                           create: source health/config/sync/delete
        PersonalContextScene.tsx                   create: profile edit/generation
        PrivacyCaptureScene.tsx                    create: capture/privacy/curtain truth
        AppearanceScene.tsx                        create: light/dark/system + density-safe prefs
        ShortcutsScene.tsx                         create: complete shortcut reference
        DiagnosticsScene.lazy.tsx                  create: redacted status/session logs
      design/
        tokens.css                                 create: exact light/dark tokens
        typography.css                             create: bundled Source Serif/UI scale
        globals.css                                create: reset/layout/reduced-motion/forced colors
        primitives/
          Button.tsx                               create: primary/secondary/quiet/danger
          IconButton.tsx                           create: enforced accessible name
          Field.tsx                                create: TextField/SearchField
          Menu.tsx                                 create: keyboard menu
          Dialog.tsx                               create: focus-trapped modal
          Drawer.tsx                               create: dock/overlay behavior
          Disclosure.tsx                           create: semantic details
          ObjectRow.tsx                            create: 44 px object row
          StatusText.tsx                           create: textual state priority
          EvidenceLabel.tsx                        create: exact evidence vocabulary
          a11y.tsx                                 create: VisuallyHidden/FocusReturn
          LiveRegion.tsx                           create: coalesced announcements
          index.ts                                 create: public primitive exports
      telemetry/
        productEvents.ts                           create: closed renderer event vocabulary
      test/
        setup.ts                                   create: jest-dom/cleanup
        fixtures.ts                                create: complete wire fixtures
        renderWithServices.tsx                     create: deterministic service harness
core/
  taurine_core/
    schemas.py                                      modify: patch/resume/actionable contracts
    server.py                                       modify: atomic PATCH + suspend/resume ownership
    indexer/store.py                                modify: review JSON + mutation receipts
    chat/engine.py                                  modify: proposed-not-applied result/copy
  tests/
    test_meetings.py                                modify: store/REST/finalization coverage
    test_server_ws.py                               modify: resume/race/snapshot coverage
    test_chat_engine.py                             modify: edit truthfulness coverage
```

Task 29 deletes `app/src/renderer/src/App.tsx`, the entire old `components/` tree, `views/ChatView.tsx`, `views/ConnectorsView.tsx`, `views/DataView.tsx`, `views/FeedView.tsx`, the entire old `views/meetings/` tree, the entire old `views/map/` tree, the entire old `lib/` tree, `assets/wavy-lines.svg`, and `assets/electron.svg`. Capture/audio helpers and equivalent assertions must be ported into `CaptureController`, `SessionTransport`, and their tests before the old `lib/` tree is deleted. `assets/main.css` becomes a three-import entry for `tokens.css`, `typography.css`, and `globals.css`; it contains no legacy selector block.

## Execution Gates

Before running Task 1, execute this preflight in the shell that will coordinate implementation:

```bash
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
case "$WORKTREE_ROOT" in
  */.worktrees/ui-quiet-canvas) printf 'assigned-worktree=%s\n' "$WORKTREE_ROOT" ;;
  *) printf 'Refusing non-assigned checkout: %s\n' "$WORKTREE_ROOT" >&2; exit 1 ;;
esac
```

Expected: one line whose value ends with `/.worktrees/ui-quiet-canvas`. Any other result stops execution. Keep `WORKTREE_ROOT` exported or defined in the coordinating shell for documentation and tooling, while still using each standalone command's inline root resolution.

- Tasks 1–5 are the sidecar and verification foundation. Complete them before renderer mutation depends on the new wire contract.
- Tasks 6–8 establish Electron APIs before renderer recovery/privacy work.
- Tasks 9–12 establish target renderer infrastructure; feature tasks consume only those interfaces.
- Tasks 13–17 complete the full meeting journey before Ask/knowledge/settings scenes.
- Tasks 18–27 complete capability destinations without exposing any production route from the old shell.
- Task 28 proves full stories against the target bootstrap. Task 29 performs the single production cutover and deletes the old UI. Task 30 runs the complete release gate.
- A task is complete only after its named red test failed for the expected missing behavior, the scoped implementation passed, type/lint checks named in that task passed, and the scoped commit exists.

---

## Phase A — Characterization and Sidecar Contracts

### Task 1: Lock the test, build-manifest, and bundle-budget harness

**Files:**
- Modify: `app/package.json`
- Modify: `app/package-lock.json`
- Modify: `app/electron.vite.config.ts`
- Modify: `app/vitest.config.ts`
- Create: `app/src/renderer/src/test/setup.ts`
- Create: `app/scripts/bundle-budget.mjs`
- Create: `app/scripts/bundle-budget.test.mjs`

**Interfaces:**
- Consumes: current npm/electron-vite layout and `out/renderer/.vite/manifest.json`.
- Produces: `collectInitialAssets(manifest, entryKey): { js: string[]; css: string[] }`, `measureAssets(root, assets): BudgetMeasurement`, and immediately callable scripts `test`, `test:unit`, `format:check`, and `check:bundle:report`. Tasks 28–29 add E2E and enforcing cutover scripts when their files exist.

- [ ] **Step 1: Install deterministic test and UI support packages**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/app"
npm install --save-dev @testing-library/react @testing-library/user-event @testing-library/jest-dom @playwright/test @axe-core/playwright jest-axe @types/jest-axe ws @types/ws
npm install lucide-react @fontsource-variable/source-serif-4 @tanstack/react-virtual
```

Expected: `package.json` and `package-lock.json` change; npm exits 0 with no peer-resolution error.

- [ ] **Step 2: Write the failing manifest traversal test**

Create `app/scripts/bundle-budget.test.mjs` with this complete fixture:

```js
import assert from 'node:assert/strict'
import test from 'node:test'
import { collectInitialAssets } from './bundle-budget.mjs'

test('collectInitialAssets follows static imports and excludes dynamic imports', () => {
  const manifest = {
    'src/renderer/index.html': {
      file: 'assets/index.js',
      css: ['assets/index.css'],
      imports: ['src/renderer/src/bootstrap.ts'],
      dynamicImports: ['src/renderer/src/ConnectionsVisual.tsx']
    },
    'src/renderer/src/bootstrap.ts': {
      file: 'assets/bootstrap.js',
      css: ['assets/bootstrap.css']
    },
    'src/renderer/src/ConnectionsVisual.tsx': {
      file: 'assets/visual.js',
      css: ['assets/visual.css']
    }
  }
  assert.deepEqual(collectInitialAssets(manifest, 'src/renderer/index.html'), {
    js: ['assets/bootstrap.js', 'assets/index.js'],
    css: ['assets/bootstrap.css', 'assets/index.css']
  })
})
```

- [ ] **Step 3: Run the test and confirm the missing module failure**

Run: `cd "$(git rev-parse --show-toplevel)/app" && node --test scripts/bundle-budget.test.mjs`

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `scripts/bundle-budget.mjs`.

- [ ] **Step 4: Implement the budget collector and checker**

Create `app/scripts/bundle-budget.mjs` with exported functions and CLI behavior:

```js
import { gzipSync } from 'node:zlib'
import { readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

export function collectInitialAssets(manifest, entryKey) {
  const seen = new Set()
  const js = new Set()
  const css = new Set()
  function visit(key) {
    if (seen.has(key)) return
    seen.add(key)
    const row = manifest[key]
    if (!row) throw new Error(`Manifest entry not found: ${key}`)
    if (row.file) js.add(row.file)
    for (const file of row.css ?? []) css.add(file)
    for (const dependency of row.imports ?? []) visit(dependency)
  }
  visit(entryKey)
  return { js: Array.from(js).sort(), css: Array.from(css).sort() }
}

export async function measureAssets(root, assets) {
  async function sum(files, gzip) {
    let total = 0
    for (const file of files) {
      const bytes = await readFile(resolve(root, file))
      total += gzip ? gzipSync(bytes).byteLength : (await stat(resolve(root, file))).size
    }
    return total
  }
  return {
    js: await sum(assets.js, false),
    css: await sum(assets.css, false),
    gzipJs: await sum(assets.js, true),
    gzipCss: await sum(assets.css, true)
  }
}

export async function checkBundle(root = resolve('out/renderer'), enforce = true) {
  const manifest = JSON.parse(await readFile(resolve(root, '.vite/manifest.json'), 'utf8'))
  const assets = collectInitialAssets(manifest, 'src/renderer/index.html')
  const value = await measureAssets(root, assets)
  const limits = { js: 1_250_000, css: 120_000, gzipJs: 350_000, gzipCss: 35_000 }
  const failures = Object.entries(limits).filter(([key, limit]) => value[key] > limit)
  const forceGraph = assets.js.some((file) => file.includes('force-graph'))
  const blockNote = assets.js.some((file) => file.toLowerCase().includes('blocknote'))
  if (enforce && (failures.length || forceGraph || blockNote)) {
    throw new Error(JSON.stringify({ value, failures, forceGraph, blockNote }))
  }
  return value
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  checkBundle(resolve('out/renderer'), !process.argv.includes('--report-only'))
    .then((value) => console.log(JSON.stringify(value)))
}
```

Modify `electron.vite.config.ts` so renderer `build.manifest` is `true`. Add npm scripts exactly:

```json
{
  "test": "vitest",
  "test:unit": "vitest run",
  "format:check": "prettier --check .",
  "check:bundle:report": "node scripts/bundle-budget.mjs --report-only"
}
```

Set `vitest.config.ts` `setupFiles` to `src/renderer/src/test/setup.ts`; that setup imports `@testing-library/jest-dom/vitest` and registers `afterEach(cleanup)`.

- [ ] **Step 5: Prove the harness passes and characterize the old bundle failure**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/app"
node --test scripts/bundle-budget.test.mjs
npm run build
npm run check:bundle:report
```

Expected: node test and build PASS; the report command exits 0 and prints the old renderer measurement, which exceeds the later enforcing limits and/or shows BlockNote/force-graph statically reachable. Record it in the commit body; do not weaken limits. Task 29 turns the same checker into a CI failure after the sole-entry cutover.

- [ ] **Step 6: Run existing app regression checks**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npx vitest run && npm run typecheck && npm run lint`

Expected: all existing tests PASS (at least 61), both TypeScript projects PASS, ESLint exits 0.

- [ ] **Step 7: Commit the harness**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/package.json app/package-lock.json app/electron.vite.config.ts app/vitest.config.ts app/src/renderer/src/test/setup.ts app/scripts/bundle-budget.mjs app/scripts/bundle-budget.test.mjs
git commit -m "test(app): lock Quiet Canvas verification harness"
```

### Task 2: Add revision-safe, idempotent meeting mutations

**Files:**
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/indexer/store.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/tests/test_meetings.py`

**Interfaces:**
- Consumes: existing `Index.meeting_update`, `MeetingUpdateRequest`, `PATCH /meetings/{meeting_id}`.
- Produces: `MeetingNotesConflict`, `Index.meeting_patch_revisioned(meeting_id, notes_md, expected_notes_rev, mutation_id, title, suggestions, ai_lines, review_actionables)`, `meeting_mutation_receipts`, required `expected_notes_rev` and `mutation_id` for API note writes, and the exact 409 body below.

```python
class MeetingUpdateRequest(BaseModel):
    title: str | None = None
    notes_md: str | None = None
    suggestions: list[dict[str, Any]] | None = None
    ai_lines: list[str] | None = None
    review_actionables: list[dict[str, Any]] | None = None
    expected_notes_rev: int | None = None
    mutation_id: str | None = None

    @model_validator(mode="after")
    def require_note_precondition(self) -> "MeetingUpdateRequest":
        if self.notes_md is not None and (
            self.expected_notes_rev is None or not (self.mutation_id or "").strip()
        ):
            raise ValueError("notes_md requires expected_notes_rev and mutation_id")
        return self
```

- [ ] **Step 1: Write failing store and REST tests**

Add tests that execute these assertions (use the existing `index` fixture and `_runtime(index)` helper):

```python
def test_revisioned_patch_is_atomic_idempotent_and_conflict_safe(index):
    meeting = index.meeting_create("Review")
    first = index.meeting_patch_revisioned(
        meeting["meeting_id"], notes_md="alpha", ai_lines=["alpha"],
        expected_notes_rev=0, mutation_id="mutation-1",
    )
    duplicate = index.meeting_patch_revisioned(
        meeting["meeting_id"], notes_md="alpha", ai_lines=["alpha"],
        expected_notes_rev=0, mutation_id="mutation-1",
    )
    assert first == duplicate
    assert first["notes_rev"] == 1
    with pytest.raises(MeetingNotesConflict) as raised:
        index.meeting_patch_revisioned(
            meeting["meeting_id"], notes_md="stale",
            expected_notes_rev=0, mutation_id="mutation-2",
        )
    assert raised.value.current["notes_md"] == "alpha"
```

Add TestClient cases: note body without either precondition returns 422; stale base returns status 409 with `detail.code == "notes_conflict"`, echoed mutation ID, and complete current notes/suggestions/AI lines; title-only and status-only suggestion patches remain accepted without note preconditions.

- [ ] **Step 2: Run the focused tests and verify red**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_meetings.py -q -k 'revisioned or precondition or conflict_safe'`

Expected: FAIL because `meeting_patch_revisioned` and `MeetingNotesConflict` do not exist and the current REST route accepts unguarded notes.

- [ ] **Step 3: Implement the receipt table and atomic method**

Add schema initialization:

```sql
CREATE TABLE IF NOT EXISTS meeting_mutation_receipts (
  meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  mutation_id TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY(meeting_id, mutation_id)
);
```

Add this exception in `store.py`:

```python
class MeetingNotesConflict(RuntimeError):
    def __init__(self, current: dict[str, Any]) -> None:
        super().__init__("meeting notes revision conflict")
        self.current = current
```

Add `meeting_patch_revisioned` with required positional `meeting_id`, keyword-only required `notes_md: str`, `expected_notes_rev: int`, `mutation_id: str`, optional `title: str | None`, `suggestions: list[dict[str, Any]] | None`, `ai_lines: list[str] | None`, and `review_actionables: list[dict[str, Any]] | None`, returning `dict[str, Any]`.

Inside one `with self._conn:` transaction: return `json.loads(response_json)` for an existing receipt; select the meeting; raise `KeyError` if absent; compare `notes_rev`; raise `MeetingNotesConflict(_meeting_from_row(row))` on mismatch; update title/notes/suggestions/AI/actionables together and increment `notes_rev` exactly once; select the updated row; add echoed `mutation_id`; persist that complete JSON response; retain only the newest 64 receipts for that meeting using `created_at`; return the response. Never implement last-write-wins retry.

In `server.py`, call the revisioned method whenever `payload.notes_md is not None`; catch `MeetingNotesConflict` and raise:

```python
raise HTTPException(status_code=409, detail={
    "code": "notes_conflict",
    "mutation_id": payload.mutation_id,
    "current": conflict.current,
})
```

Non-note PATCHes continue through `meeting_update`; API validation enforces the model validator.

- [ ] **Step 4: Run focused and full core tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/core"
uv run pytest tests/test_meetings.py -q
uv run pytest -q
```

Expected: meeting tests PASS; complete suite PASS with only the two documented skips.

- [ ] **Step 5: Commit the mutation contract**

```bash
cd "$(git rev-parse --show-toplevel)"
git add core/taurine_core/schemas.py core/taurine_core/indexer/store.py core/taurine_core/server.py core/tests/test_meetings.py
git commit -m "feat(core): make meeting note mutations revision safe"
```

### Task 3: Persist review actionables and validate status-only edits

**Files:**
- Modify: `core/taurine_core/indexer/store.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/tests/test_meetings.py`

**Interfaces:**
- Consumes: Task 2 request field `review_actionables`, existing `Actionable`, `_capture_finalization_snapshot`, and suggestion status-patch validation.
- Produces: `review_actionables_json TEXT NOT NULL DEFAULT '[]'`, full meeting field `review_actionables`, `Index.meeting_update(meeting_id, review_actionables=rows)`, and durable open/confirmed/dismissed status transitions.

- [ ] **Step 1: Write the failing migration/finalization/status tests**

Add cases for a pre-existing meetings table without the column, a new meeting default, finalization capture, and invalid mutation:

```python
def test_review_actionable_status_patch_preserves_immutable_fields(index):
    meeting = index.meeting_create("Actions")
    original = {"id": "a1", "title": "Send draft", "quote": "send the draft",
                "assignee": "Neil", "due": "", "grounding": [], "why_new": "",
                "confidence": 0.8, "status": "open"}
    index.meeting_update(meeting["meeting_id"], review_actionables=[original])
    with TestClient(create_app(_runtime(index))) as client:
        response = client.patch(f"/meetings/{meeting['meeting_id']}", json={
            "review_actionables": [{**original, "title": "tampered", "status": "confirmed"}]
        })
    assert response.status_code == 200
    assert response.json()["review_actionables"][0]["title"] == "Send draft"
    assert response.json()["review_actionables"][0]["status"] == "confirmed"
```

Also assert `_capture_finalization_snapshot` writes the latest live summary's actionables before final-agent execution and that reload returns them.

- [ ] **Step 2: Run the focused tests and verify red**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_meetings.py -q -k review_actionable`

Expected: FAIL because the column/field and status validator do not exist.

- [ ] **Step 3: Implement durable actionables**

Add the column to new DDL and `_ensure_meetings_columns`; parse it defensively in `_meeting_from_row`. Extend `meeting_update` and Task 2's atomic method. Add `_review_actionable_status_patch(previous, requested)` that matches by `id`, permits only `open`, `confirmed`, or `dismissed`, copies every immutable field from the stored row, rejects unknown/duplicate IDs with HTTP 422, and permits reversing a previous decision.

In `_capture_finalization_snapshot`, immediately after `summary = context.live.current_summary()`, persist:

```python
runtime.index.meeting_update(
    meeting_id,
    review_actionables=[item.model_dump(mode="json") for item in summary.actionables],
)
```

Then refresh the meeting row before building the snapshot. Do not persist ephemeral `T#/E#/F#` registry objects; durable review locates each saved `quote` in the transcript.

- [ ] **Step 4: Run meeting and full suites**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_meetings.py -q && uv run pytest -q`

Expected: both commands PASS; old databases acquire the additive column without content transformation.

- [ ] **Step 5: Commit durable review state**

```bash
cd "$(git rev-parse --show-toplevel)"
git add core/taurine_core/indexer/store.py core/taurine_core/server.py core/tests/test_meetings.py
git commit -m "feat(core): persist meeting review actionables"
```

### Task 4: Repair meeting-note tool truthfulness

**Files:**
- Modify: `core/taurine_core/chat/engine.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/tests/test_chat_engine.py`

**Interfaces:**
- Consumes: `_execute_edit_notes`, `ChatNotesEditEvent`, `ChatActivityStep`.
- Produces: model-visible `{"ok": true, "proposed": true}`, activity label `Proposed a meeting-note change`, unchanged `chat_notes_edit` wire discriminant.

- [ ] **Step 1: Write the failing exact-copy test**

Add to `test_chat_engine.py`:

```python
def test_edit_notes_reports_proposal_before_renderer_acceptance(index):
    meeting = index.meeting_create("Truth")
    result, step, event = _execute_edit_notes(
        {"anchor": "", "insert_md": "Proposed text"},
        meeting_id=meeting["meeting_id"], index=index,
        chat_id="chat-1", message_id=7,
    )
    assert json.loads(result) == {"ok": True, "proposed": True}
    assert step.label == "Proposed a meeting-note change"
    assert event is not None and event.type == "chat_notes_edit"
    assert index.meeting_get(meeting["meeting_id"])["notes_md"] == ""
```

- [ ] **Step 2: Run the test and verify the copy mismatch**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_chat_engine.py -q -k reports_proposal`

Expected: FAIL showing current `applied` and `Edited the meeting notes` values.

- [ ] **Step 3: Change only the result and activity language**

Replace the success result with:

```python
step = ChatActivityStep(
    kind="edit", label="Proposed a meeting-note change", status="done",
    tool_name="edit_notes", arguments=dict(arguments),
)
return json.dumps({"ok": True, "proposed": True}), step, event
```

Update the `ChatNotesEditEvent` docstring to say the renderer converts the event into a proposal and only `MeetingDocumentController` can accept it.

- [ ] **Step 4: Run chat and full suites**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_chat_engine.py -q && uv run pytest -q`

Expected: both PASS.

- [ ] **Step 5: Commit the truthfulness repair**

```bash
cd "$(git rev-parse --show-toplevel)"
git add core/taurine_core/chat/engine.py core/taurine_core/schemas.py core/tests/test_chat_engine.py
git commit -m "fix(core): report meeting note edits as proposals"
```

### Task 5: Preserve live intelligence across a session reconnect

**Files:**
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/live/ingest.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/tests/test_server_ws.py`
- Modify: `core/tests/test_live_ingest.py`

**Interfaces:**
- Extends `StartMsg` with `client_session_id: str`; a meeting-bound start without it is invalid.
- Adds `SessionSnapshotEvent` with the exact `new | resumed | state_reset` disposition and snapshot fields in the design.
- Adds `RebindableEventSink.bind(send)`, `.unbind()`, and async `__call__(event)`.
- Adds runtime-owned `suspended_sessions: dict[str, SuspendedSession]`; nothing from the evidence registry is persisted.

- [ ] **Step 1: Write failing contract and lifecycle tests**

Add tests that use the existing fake ASR/session helpers and an injectable finalization scheduler:

```python
def test_meeting_start_requires_client_session_id():
    with pytest.raises(ValidationError):
        StartMsg(type="start", meeting_id="m1", channels=[])

@pytest.mark.asyncio
async def test_matching_reconnect_rebinds_same_live_session(runtime, ws_pair):
    first = await start_live(runtime, ws_pair.first, "m1", "client-1")
    first.live.summary = MeetingSummary(rev=3, summary_md="Kept", actionables=[])
    await disconnect(runtime, first)
    second = await start_live(runtime, ws_pair.second, "m1", "client-1")
    assert second.live is first.live
    assert snapshot(ws_pair.second)["disposition"] == "resumed"
    assert snapshot(ws_pair.second)["summary"]["rev"] == 3

```

Add three separate async tests named `test_wrong_token_resets_live_context_without_erasing_durable_meeting`, `test_grace_expiry_finalizes_suspended_session_once`, and `test_old_expiry_callback_cannot_finalize_rebound_epoch`. Use the same `runtime` and `ws_pair` fixtures: the first asserts `state_reset` plus unchanged persisted notes/transcript, the second advances the injected scheduler to 60 seconds and asserts one finalization/close, and the third reconnects before advancing the obsolete scheduler callback and asserts zero finalizations for the rebound epoch. Also assert disconnect stops ASR and flushes the transcript, but does not call `LiveSession.aclose()` until the 60-second grace expires.

- [ ] **Step 2: Run the focused tests and verify the missing resume behavior**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_server_ws.py tests/test_live_ingest.py -q -k 'client_session_id or reconnect or grace_expiry or old_expiry'`

Expected: FAIL because `StartMsg` has no token, disconnect closes live state, and no snapshot is emitted.

- [ ] **Step 3: Add the typed contract and a rebindable sink**

In `schemas.py`, add the required token and validator, importing `model_validator` from Pydantic:

```python
class StartMsg(BaseModel):
    type: Literal["start"]
    meeting_id: str | None = None
    client_session_id: str = ""
    channels: list[ChannelSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_bound_session_token(self) -> "StartMsg":
        if self.meeting_id and not self.client_session_id.strip():
            raise ValueError("client_session_id is required for meeting sessions")
        return self

class SessionSnapshotEvent(BaseModel):
    type: Literal["session_snapshot"] = "session_snapshot"
    meeting_id: str
    client_session_id: str
    disposition: Literal["new", "resumed", "state_reset"]
    asr: Literal["connecting", "connected", "error"]
    channels: list[dict[str, Any]]
    summary: MeetingSummary
    grounding: dict[str, Any]
```

In `live/ingest.py`, implement an async callable sink whose target is replaced under an `asyncio.Lock`. A missing target drops transport-only events; authoritative summary and grounding remain on `LiveSession` for the next snapshot.

- [ ] **Step 4: Move live context ownership from the socket to the runtime**

Introduce explicit state (names may match local server conventions, but fields may not be omitted):

```python
@dataclass
class SuspendedSession:
    meeting_id: str
    client_session_id: str
    context: SessionContext
    epoch: int
    expires_at: float
    finalizer: asyncio.Task[None]

@dataclass
class SessionContext:
    live: LiveSession
    sink: RebindableEventSink
    transcript_buffer: list[TranscriptChunk]
    notepad_md: str
    requested_channels: list[ChannelSpec]
```

On disconnect: stop ASR, flush `transcript_buffer`, unbind the sink, increment the meeting epoch, and park the context for 60 seconds. On a matching meeting/token: cancel the matching finalizer, remove the suspended entry, bind the new socket, recreate only the ASR/channel layer, and emit `resumed`. On an unmatched token with an existing durable live meeting: finalize/retire any prior context safely, build a fresh one, and emit `state_reset`. A genuinely new draft emits `new`. At expiry, verify the captured epoch still matches before closing/finalizing once.

Emit `session_snapshot` immediately after binding, before ordinary live deltas. Populate it from `live.summary`, the latest grounding revision/display arrays, ASR state, and actual channel states. Keep transcript merge identity as `seq`, summary as `rev`, grounding as `revision`, and actionables as ID.

- [ ] **Step 5: Run race, websocket, and full core suites**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest tests/test_server_ws.py tests/test_live_ingest.py -q && uv run pytest -q`

Expected: PASS, including exact same-token identity, wrong-token reset, expiry, and stale-finalizer race assertions.

- [ ] **Step 6: Commit the resume contract**

```bash
cd "$(git rev-parse --show-toplevel)"
git add core/taurine_core/schemas.py core/taurine_core/live/ingest.py core/taurine_core/server.py core/tests/test_server_ws.py core/tests/test_live_ingest.py
git commit -m "feat(core): resume live meeting context within grace"
```

## Phase B — Desktop Boundary, Recovery, and Privacy

### Task 6: Establish the main-window, menu, tray, and command boundary

**Files:**
- Create: `app/src/main/window-state.ts`
- Create: `app/src/main/application-menu.ts`
- Modify: `app/src/main/tray.ts`
- Create: `app/src/main/__tests__/window-state.test.ts`
- Create: `app/src/main/__tests__/application-menu.test.ts`
- Create: `app/src/main/__tests__/tray.test.ts`
- Modify: `app/src/main/index.ts`
- Modify: `app/src/preload/index.ts`
- Modify: `app/src/preload/index.d.ts`
- Create: `app/src/renderer/src/platform/electronBridge.ts`
- Modify: `app/package.json`

**Interfaces:**
- `WorkspaceCommand` is a closed union: `open-anything`, `ask`, `start-stop-meeting`, `back`, `forward`, `find`, `toggle-transcript`, `toggle-context`, `accept-proposal`, `reject-proposal`, `undo`, `hide-now`, `toggle-curtain`, `toggle-connections-view`, `toggle-comparison`, and `open-place` with a fixed `PlaceId`.
- Preload exposes only `desktop.onCommand(listener)`, `desktop.reportWorkspaceState(state)`, and existing narrow audio/notification operations; no raw IPC.
- Primary window defaults to 1120×780, minimum 840×600, restores bounded user size/position, and never gets raised by tray/call-triggered capture.

- [ ] **Step 1: Write failing pure menu/window/tray tests**

Test the complete fixed Places order and accelerators, bounds clamping against an injected display work area, explicit user-selected Compact meeting window at 840×660 with prior bounds restored on exit, and hidden capture behavior:

```ts
expect(placeItems(menu)).toEqual([
  'Desk', 'Meetings', 'Conversations', 'Connections', 'Sources',
  'Personal context', 'Privacy & Capture', 'Diagnostics', 'Settings…',
])
expect(accelerator(menu, 'Open anything')).toBe('CommandOrControl+K')
expect(accelerator(menu, 'Start/stop meeting')).toBe('CommandOrControl+Shift+R')
expect(windowFactory.lastOptions).toMatchObject({ width: 1120, height: 780, minWidth: 840, minHeight: 600 })
await tray.startNotes()
expect(primaryWindow.show).not.toHaveBeenCalled()
```

- [ ] **Step 2: Run tests and verify they fail before modules exist**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main/__tests__/window-state.test.ts src/main/__tests__/application-menu.test.ts src/main/__tests__/tray.test.ts`

Expected: FAIL with unresolved module errors.

- [ ] **Step 3: Implement typed commands, fixed menus, bounds, and tray state**

Make menu construction a pure function over platform + `sendCommand`. Repeat Open and every fixed Place in the native App/View navigation menu. Add every shortcut from design §3.5, including platform-specific Back/Forward and Hide Taurine. Ensure no one-character handler is registered globally.

Persist only `{x,y,width,height,maximized,compactMeeting}` in Electron's user-data directory using atomic temp-file rename. Clamp restored bounds so at least 80×48 px remain visible on a current display. Compact meeting is user-selected only, changes bounds to 840×660, and restores prior normal bounds when disabled; capture never toggles it. In `index.ts`, replace 1500×950/minimum 1180×760 with the required dimensions, retain the frameless/traffic-light-safe titlebar arrangement, set the web content background to `#F4F2EC`, and call `setContentProtection(true)` without claiming it guarantees exclusion on unsupported platforms.

The tray menu has `Show Taurine`, `Start meeting` or `Stop meeting`, `Privacy curtain`, and `Quit`. `reportWorkspaceState` supplies only live/curtained/dirty booleans and the safe active-channel label. Quit while live requires a native confirmation; confirmed quit requests Stop and a 3-second note flush before closing. Starting from tray/call notification sends the capture command while leaving a hidden primary window hidden.

- [ ] **Step 4: Expose and validate the narrow preload API**

Add `contextBridge` methods backed by allowlisted channels and matching declarations. Freeze the returned object. Validate incoming command discriminants before invoking renderer listeners, return unsubscribe functions, and never expose `ipcRenderer`, arbitrary channel names, filesystem paths, or `BrowserWindow` operations.

- [ ] **Step 5: Run tests, typecheck, and existing Electron tests**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main && npm run typecheck`

Expected: PASS; snapshots contain every menu label/shortcut and no legacy navigation item.

- [ ] **Step 6: Commit the desktop command boundary**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/main app/src/preload app/src/renderer/src/platform/electronBridge.ts app/package.json
git commit -m "feat(desktop): add quiet canvas window and command boundary"
```

### Task 7: Add encrypted draft recovery with no plaintext fallback

**Files:**
- Create: `app/src/main/recovery-journal.ts`
- Create: `app/src/main/__tests__/recovery-journal.test.ts`
- Create: `app/src/renderer/src/platform/recoveryBridge.ts`
- Create: `app/src/renderer/src/data/DraftJournal.ts`
- Create: `app/src/renderer/src/data/__tests__/DraftJournal.test.ts`
- Modify: `app/src/preload/index.ts`
- Modify: `app/src/preload/index.d.ts`

**Interfaces:**
- Exact `DraftJournalRecord` from design §10.5; `DraftJournalHeader` contains only `meetingId`, `schemaVersion`, and `writtenAt`.
- Preload operations: `putEncryptedDraft(record)`, `readEncryptedDraft(meetingId)`, `deleteEncryptedDraft(meetingId)`, `listEncryptedDraftHeaders()`.
- Encryption mode: `persistent-encrypted | memory-only`; never plaintext on disk, IndexedDB, or localStorage.

- [ ] **Step 1: Write failing encryption, quarantine, and scheduler tests**

Use injected `SafeStorageLike`, filesystem adapter, clock, and user-data root. Assert:

```ts
await journal.put(recordWith('SECRET TITLE', 'SECRET BODY'))
expect(await fs.readFile(cipherPath, 'utf8')).not.toContain('SECRET')
expect(await journal.listHeaders()).toEqual([{ meetingId: 'm1', schemaVersion: 1, writtenAt: 1000 }])
```

Cover macOS/Windows encryption available; Linux `basic_text`; encryption unavailable; corrupt ciphertext quarantine; unknown schema quarantine; 400 ms coalescing; pre-close flush; acknowledgement cleanup; meeting deletion; server-equal cleanup; and conflict leaves the record intact until a user choice.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main/__tests__/recovery-journal.test.ts src/renderer/src/data/__tests__/DraftJournal.test.ts`

Expected: FAIL because the journal and bridge do not exist.

- [ ] **Step 3: Implement encrypted main-process storage**

Store `{schemaVersion, meetingId, writtenAt, ciphertextBase64}` only. Validate meeting IDs before deriving filenames, write a sibling temp file, `fsync`, then rename. Encryption is persistent only when `isEncryptionAvailable()` and, on Linux, `getSelectedStorageBackend() !== 'basic_text'`; otherwise retain records in a process-local `Map` and report `memory-only`. Never call `setUsePlainTextEncryption`. Move undecipherable files to an opaque `.quarantine` name, emit only code `recovery_decrypt_failed`, and do not erase it.

- [ ] **Step 4: Implement the renderer scheduler and exact reconciliation outcomes**

`DraftJournal.schedule(record)` replaces the pending record and writes after 400 ms idle. `flush()` is awaitable. `reconcile(server, recovered)` returns exactly `replay` when revisions match, `clear` when Markdown and AI lines match, and `conflict` otherwise. Expose mode so the save disclosure can say `Crash recovery is available only while Taurine stays open` in memory-only mode.

- [ ] **Step 5: Run focused tests, inspect fixture bytes, and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main/__tests__/recovery-journal.test.ts src/renderer/src/data/__tests__/DraftJournal.test.ts && npm run typecheck`

Expected: PASS; the byte-level plaintext assertions and all cleanup/quarantine cases pass.

- [ ] **Step 6: Commit encrypted recovery**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/main/recovery-journal.ts app/src/main/__tests__/recovery-journal.test.ts app/src/preload app/src/renderer/src/platform/recoveryBridge.ts app/src/renderer/src/data/DraftJournal.ts app/src/renderer/src/data/__tests__/DraftJournal.test.ts
git commit -m "feat(desktop): add encrypted meeting draft recovery"
```

### Task 8: Make the privacy curtain deterministic across every product window

**Files:**
- Create: `app/src/main/privacy-curtain.ts`
- Create: `app/src/main/__tests__/privacy-curtain.test.ts`
- Create: `app/src/renderer/src/workspace/PrivacyCurtain.tsx`
- Create: `app/src/renderer/src/workspace/__tests__/PrivacyCurtain.test.tsx`
- Modify: `app/src/main/index.ts`
- Modify: `app/src/main/notifier.ts`
- Modify: `app/src/main/suggest-notifier.ts`
- Modify: `app/src/preload/index.ts`
- Modify: `app/src/preload/index.d.ts`

**Interfaces:**
- Main owns `CurtainCoordinator` with monotonic epoch, `enable()`, `disable()`, `registerWindow()`, `acknowledge(webContentsId, epoch)`, and safe `stopCapture()`.
- Preload exposes `privacy.onState(listener)`, `privacy.acknowledge(epoch)`, `privacy.show()`, and `privacy.stopCapture()` only.
- Curtained renderer mounts no workspace route, overlay, private live region, or object-derived title.

- [ ] **Step 1: Write failing epoch and DOM-exclusion tests**

Test two visible product windows, one late-created window, a stale acknowledgement, a renderer crash/reload, timeout-to-hide, and restore focus. In DOM tests render a secret in the route, portal, document title, and live region; after `active=true`, assert none are present and the neutral subtree contains `Privacy curtain`, safe capture channel health/elapsed time, `Show Taurine`, and live-only `Stop meeting`.

- [ ] **Step 2: Run tests and verify private subtrees remain mounted today**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main/__tests__/privacy-curtain.test.ts src/renderer/src/workspace/__tests__/PrivacyCurtain.test.tsx`

Expected: FAIL because there is no epoch/acknowledgement protocol or exclusion branch.

- [ ] **Step 3: Implement main-owned epoch coordination**

Register `CommandOrControl+Shift+P` and keep the active epoch independent of any renderer. Enabling sends `{active:true, epoch}` to every existing Taurine `webContents`; new windows receive it before `show()`. Do not resolve `enable()` until every shown window acknowledges or is hidden after a bounded timeout. Ignore stale acknowledgements. Reloaded renderers query the current state before rendering private content. Disabling sends the next epoch and waits for the same acknowledgement discipline.

- [ ] **Step 4: Implement the neutral renderer branch and peripheral policy**

At the top of bootstrap, branch on curtain state before creating `WorkspaceRouter`, portals, live regions, or title synchronization. Store only a safe focus token, not arbitrary private accessible names. Restore the prior route already held in memory; after its editor/heading is ready, restore a still-valid safe element or focus the route heading. The call prompt and suggestion window must either render their own neutral curtain branch and acknowledge or remain hidden. Keep capture running; Stop uses the dedicated safe IPC action.

- [ ] **Step 5: Run privacy tests and full app checks**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main src/renderer/src/workspace && npm run typecheck && npm run lint`

Expected: PASS, including source/portal/title absence and late-window/crash cases.

- [ ] **Step 6: Commit the privacy curtain**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/main/privacy-curtain.ts app/src/main/__tests__/privacy-curtain.test.ts app/src/main/index.ts app/src/main/notifier.ts app/src/main/suggest-notifier.ts app/src/preload
git add app/src/renderer/src/workspace/PrivacyCurtain.tsx app/src/renderer/src/workspace/__tests__/PrivacyCurtain.test.tsx
git commit -m "feat(privacy): add deterministic desktop curtain"
```

## Phase C — Typed Data Foundation and Quiet Shell

### Task 9: Build strict sidecar contracts, gateway, and repositories

**Files:**
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
- Create: `app/src/renderer/src/data/MeetingRepository.ts`
- Create: `app/src/renderer/src/data/KnowledgeRepository.ts`
- Create: `app/src/renderer/src/data/ConnectorRepository.ts`
- Create: `app/src/renderer/src/data/ProfileRepository.ts`
- Create: `app/src/renderer/src/data/DiagnosticsRepository.ts`
- Create: `app/src/renderer/src/data/__tests__/SidecarGateway.test.ts`
- Create: `app/src/renderer/src/data/__tests__/repositories.test.ts`
- Modify: `app/src/renderer/src/lib/sidecar.ts`
- Modify: `app/eslint.config.mjs`

**Interfaces:**
- `SidecarGateway.request<T>(request: DecodedRequest<T>): Promise<T>` owns URL, abort, request ID, readiness and error decoding.
- `RuntimeNotReady` carries safe `phase/detail`; `SidecarHttpError` carries status/code; `NotesConflict` carries `mutationId/current`.
- Repositories implement `getSnapshot`, `subscribe`, keyed request coalescing, explicit invalidation, and abort on final subscriber removal.
- Evidence decoding retains `fallback`, `passage_date`, `prior_context`, `low_confidence`, message provenance, and unknown additive fields.

- [ ] **Step 1: Add failing decoder and access-boundary tests**

Use literal fixtures captured from current Pydantic serializers for `/startup`, meetings, chats, graph global/local/idea, facts, notes, and streamed chat/session events. Assert a 503 decodes to `RuntimeNotReady`, 409 to `NotesConflict`, invalid discriminants are rejected, unknown fields remain in `extensions`, and evidence retains all trust fields currently dropped by `lib/sidecar.ts`.

Add an ESLint `no-restricted-imports` rule for feature directories so only `data/*` and `platform/*` may import `lib/sidecar`, construct `WebSocket`, call `fetch`, or use the preload bridge.

- [ ] **Step 2: Run tests/lint and verify current parsing loses data**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data/__tests__/SidecarGateway.test.ts src/renderer/src/data/__tests__/repositories.test.ts && npm run lint`

Expected: FAIL on missing modules and evidence fields; lint identifies current direct transports but is scoped to new target feature roots until Task 29 deletes legacy files.

- [ ] **Step 3: Implement strict decoders and gateway**

Use hand-written type guards with exhaustive discriminant switches; do not add a runtime schema library. Preserve an `extensions: Record<string, unknown>` bag after consuming known fields. `SidecarGateway` attaches a UUID request ID, combines caller and timeout abort signals, decodes JSON/content type, maps readiness 503 and note 409, and reports content-free decoder diagnostics. It never retries mutations.

Keep `lib/sidecar.ts` temporarily as a legacy-only adapter; new files may not import it. Put every endpoint path exactly as it exists now, including `/graph/local?center={noteId}&depth={1|2|3}` and facts batches of at most 50. Implement focused Connector/Profile/Diagnostics repositories over the existing endpoint shapes at the same time; none may create an additional websocket.

- [ ] **Step 4: Implement meeting and knowledge caches**

Meeting list and full meeting records are separate cache slots so a list projection can never overwrite an open document. Knowledge invalidates global/local graph, note preview, facts, clusters, and idea results atomically on graph-version change. A stale local graph remains readable until a same-scope replacement commits; never merge generations. Expose all stores through `useSyncExternalStore`-compatible subscription semantics.

- [ ] **Step 5: Run data tests, typecheck, and lint**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data && npm run typecheck && npm run lint`

Expected: PASS; invalid fixtures become diagnostics instead of exceptions escaping to React.

- [ ] **Step 6: Commit the data boundary**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/data app/src/renderer/src/lib/sidecar.ts app/eslint.config.mjs
git commit -m "feat(renderer): add typed sidecar data boundary"
```

### Task 10: Centralize readiness and full update projections

**Files:**
- Create: `app/src/renderer/src/data/SidecarProjectionStore.ts`
- Create: `app/src/renderer/src/data/__tests__/SidecarProjectionStore.test.ts`
- Create: `app/src/renderer/src/bootstrap/RendererBootstrap.tsx`
- Create: `app/src/renderer/src/bootstrap/StartupScene.tsx`
- Create: `app/src/renderer/src/bootstrap/__tests__/RendererBootstrap.test.tsx`
- Modify: `app/src/renderer/src/main.tsx`
- Modify: `app/src/renderer/src/lib/updates.ts`

**Interfaces:**
- Readiness: `launching → sidecar-starting → ready`, with `startup-error`, `sidecar-reconnecting`, and `unavailable` exactly as design §11.1.
- One `SidecarProjectionStore` owns `/updates` and full projections for connectors, sync, graph version, profile, chats, and meetings.
- Reconnect backoff: 500 ms to 8 seconds with injected ±20% jitter; exception becomes visible after 2 seconds except immediately during live capture.

- [ ] **Step 1: Write failing fake-clock and real-websocket integration tests**

Assert `/startup` polling at 750 ms, detail withheld until 2 seconds, 503 decoding, 15-second unavailable transition, Retry/Open diagnostics/Quit actions, every update kind including `chats`, full replacement on reconnect, optimistic overlay by entity ID + mutation ID, and capped backoff. Use a local `ws` fixture for initial event ordering and reconnection; use fake timers only for policy delays.

- [ ] **Step 2: Run tests and verify chats are currently ignored**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data/__tests__/SidecarProjectionStore.test.ts src/renderer/src/bootstrap/__tests__/RendererBootstrap.test.tsx`

Expected: FAIL on missing store/readiness states and the `chats` projection assertion.

- [ ] **Step 3: Implement the store and readiness controller**

Each update reducer atomically replaces its server projection. Reapply acknowledged-or-pending local overlays without mutating server snapshots. Meeting-list events cannot replace full meeting/editor state. Graph-version changes call the single knowledge invalidator from Task 9. Unknown event types generate a content-free diagnostic and are ignored. On reconnect, accept the server's initial full events as authoritative so missed events need no replay protocol.

`RendererBootstrap` first queries curtain state, then renders static app name/progress immediately. Poll startup; after ready, hydrate projections and mount the workspace. Keep local editor state mounted while sidecar reconnects. Never show an endless spinner or blank frame.

- [ ] **Step 4: Remove the legacy updates owner**

Move all decoding into the new store. Leave `updates.ts` as a re-export only until Task 29 switches the entry, ensuring no second websocket is created. Add a test that instantiating bootstrap twice under Strict Mode still produces one live socket after cleanup/restart.

- [ ] **Step 5: Run renderer data/bootstrap tests and checks**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data src/renderer/src/bootstrap && npm run typecheck && npm run lint`

Expected: PASS.

- [ ] **Step 6: Commit the unified projection owner**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/data/SidecarProjectionStore.ts app/src/renderer/src/data/__tests__/SidecarProjectionStore.test.ts app/src/renderer/src/bootstrap app/src/renderer/src/main.tsx app/src/renderer/src/lib/updates.ts
git commit -m "feat(renderer): centralize readiness and update projections"
```

### Task 11: Encode the visual system in tokens and accessible primitives

**Files:**
- Create: `app/src/renderer/src/design/tokens.css`
- Create: `app/src/renderer/src/design/typography.css`
- Create: `app/src/renderer/src/design/primitives/Button.tsx`
- Create: `app/src/renderer/src/design/globals.css`
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
- Create: `app/src/renderer/src/design/primitives/__tests__/primitives.test.tsx`
- Modify: `app/package.json`

**Interfaces:**
- Buttons: `primary | secondary | quiet | danger`; `IconButton` requires `label` in its TypeScript props.
- Dialog/Drawer/Menu/Popover/Disclosure use focus return, Escape policy, outside-click policy, inert background, and reduced-motion tokens.
- Pointer targets are ≥36×36; primary/capture targets are 44 px; visible `:focus-visible` is never disabled.

- [ ] **Step 1: Add failing semantic, keyboard, contrast, and target tests**

Render each primitive with Testing Library and `jest-axe`; assert names, roles, focus trap/return, disabled + busy semantics, Escape, arrow-key menus, ≥36 px computed targets, and forced-colors/reduced-motion classes. Add a compile-failure fixture proving `<IconButton />` without `label` does not typecheck.

- [ ] **Step 2: Run tests and verify missing primitives**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/design && npm run typecheck`

Expected: FAIL with missing modules.

- [ ] **Step 3: Add bundled fonts, exact tokens, and restrained geometry**

Import bundled Source Serif 4 Variable from `@fontsource-variable/source-serif-4`, never remote fonts; use the exact system UI and monospace stacks from design §7.3. Encode every light/dark color, typography, spacing, radius, border, elevation, and motion value from design §§7.2–7.6 as custom properties. The sheet is warm/quiet, status uses shape + words, and no primitive adds glow, glass blur, indiscriminate pills, or card chrome. Use Lucide icons only; product navigation remains text-labeled where specified.

- [ ] **Step 4: Build primitives on native semantics**

Prefer `<button>`, `<input>`, `<dialog>`/ARIA dialog patterns, lists, and headings. `LiveRegion` maintains one stable node. `FocusReturn` captures a concrete element and falls back to a supplied route heading. Overlay width is `min(calc(100vw - 32px), var(--designed-width))`, maximum height is viewport minus 32 px, and scrolling stays inside the overlay. Honor `prefers-reduced-motion` and forced colors.

- [ ] **Step 5: Run primitive tests, axe, and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/design && npm run typecheck && npm run lint`

Expected: PASS with zero axe violations.

- [ ] **Step 6: Commit the design foundation**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/design app/package.json app/package-lock.json
git commit -m "feat(renderer): add quiet canvas design primitives"
```

### Task 12: Implement typed navigation, title bar, Open anything, and status

**Files:**
- Create: `app/src/renderer/src/workspace/WorkspaceRouter.ts`
- Create: `app/src/renderer/src/workspace/routeState.ts`
- Create: `app/src/renderer/src/workspace/WorkspaceShell.tsx`
- Create: `app/src/renderer/src/workspace/WorkspaceTitlebar.tsx`
- Create: `app/src/renderer/src/workspace/OpenAnything.tsx`
- Create: `app/src/renderer/src/workspace/StatusPopover.tsx`
- Create: `app/src/renderer/src/workspace/__tests__/router.test.ts`
- Create: `app/src/renderer/src/workspace/__tests__/open-anything.test.tsx`
- Create: `app/src/renderer/src/workspace/__tests__/shell.test.tsx`
- Modify: `app/src/renderer/src/bootstrap/RendererBootstrap.tsx`

**Interfaces:**
- Use the exact `ComparisonRef` and `WorkspaceRoute` unions in design §3.2; define `PlaceId`/`SettingsPage` as closed unions.
- Router provides `push`, `replace`, `back`, `forward`, `getSnapshot`, `subscribe`, and route-keyed ephemeral scene-state serialization.
- Fixed Places order is the single constant shared by breadcrumb, Open, and menu parity tests.

- [ ] **Step 1: Write failing router and navigation contract tests**

Cover typed round-trips for every route, capped back/forward stacks, scroll + drawer restoration, focus restoration, unmounted inactive scenes, and comparison persistence for renderer lifetime. Assert empty Open shows exactly the fixed Places before the catalog loads; typed results rank commands, active meeting, recents, metadata matches, then settings. Assert private note bodies are never passed to the search index and `Knowledge search is preparing` appears while metadata is idle-loading.

- [ ] **Step 2: Run focused tests and verify the old tab shell fails them**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/workspace`

Expected: FAIL; old five-tab navigation and mounted hidden scenes violate assertions.

- [ ] **Step 3: Implement history and 48 px global chrome**

The title bar contains Back/Forward, one breadcrumb, always-labeled `Open`, width-permitting labeled `Ask`, exception-only status, and live-only capture slot. At widths below 960 px Ask moves to object More; Open never becomes icon-only. Breadcrumb menu begins with history followed by fixed Places. Keep traffic-light drag/safe regions and mark interactive descendants `-webkit-app-region: no-drag`.

Only the active route component is mounted. Persist serializable scene data keyed by route (scroll, drawer, selected node), not DOM/editor/chat trees. On cited-source navigation, preserve the Ask sheet state and restore the exact meeting position on Back.

- [ ] **Step 4: Implement Open and exception-only status**

Use meeting/chat projections immediately; schedule graph title/gist/entity catalog via `requestIdleCallback` with a timeout fallback. Empty Open is a 640×min(560, viewport−96) dialog with fixed Places first. Escape returns focus to the invoker; selection closes then focuses the destination heading/editor. Status stays a neutral dot or exception badge; healthy source count, sync timestamps, websocket/ASR labels, model activity, and avatar never appear in chrome. The popover exposes only current exceptions/recovery links.

Map native commands from Task 6 to the same router/actions. Guard editor/composer/search/editable targets before any shortcut action.

- [ ] **Step 5: Run workspace tests, axe, and responsive typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/workspace && npm run typecheck && npm run lint`

Expected: PASS; tests at 840, 960, 1180, and 1440 CSS px show no persistent categories/footer and no horizontal document overflow.

- [ ] **Step 6: Commit the quiet shell modules (still not production entry)**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/workspace app/src/renderer/src/bootstrap/RendererBootstrap.tsx
git commit -m "feat(renderer): add typed quiet canvas workspace shell"
```

### Task 13: Add Desk and the meetings collection

**Files:**
- Create: `app/src/renderer/src/meetings/DeskScene.tsx`
- Create: `app/src/renderer/src/meetings/MeetingsCollectionScene.tsx`
- Create: `app/src/renderer/src/meetings/__tests__/DeskScene.test.tsx`
- Create: `app/src/renderer/src/meetings/__tests__/MeetingsCollectionScene.test.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`

**Interfaces:**
- Desk consumes meeting/chat projections and capture command; it never reads graph clusters, connector totals, profile progress, or healthy system status.
- Collection is a semantic 760 px list with `aria-activedescendant`; delete confirmation names title/date and returns focus.

- [ ] **Step 1: Write failing before/empty/loading/error/list keyboard tests**

Assert Desk's 840 px centered content, 44 px Start action, active meeting row, exactly eight Today/Earlier recents, `Open all…`, fixed-height skeleton lines, and exact empty sentence. Assert zero enabled sources adds only a secondary Sources link. Verify Start remains usable when recents fail. For collection, test `⌘/Ctrl+F`, Up/Down, Enter, Shift+F10, one nonterminal status word, named inline delete confirmation, and focus on next row after deletion.

- [ ] **Step 2: Run tests and verify scenes do not exist**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings/__tests__/DeskScene.test.tsx src/renderer/src/meetings/__tests__/MeetingsCollectionScene.test.tsx`

Expected: FAIL with unresolved modules.

- [ ] **Step 3: Implement Desk as a starting place, not a dashboard**

Route successful Start directly to the created/bound meeting, but respect hidden-window behavior because desktop start is owned by Task 6. Group recents by local human date and use separators, not cards. Move compact secondary actions to a labeled row menu. Never render insight recommendations, connector counts, graph content, recent source documents, or healthy service machinery.

- [ ] **Step 4: Implement search, delete, and focus repair**

Filter title/date metadata locally and preserve stable row IDs. Delete uses the existing endpoint and explains transcript, notes, summary, and linked meeting context are deleted; generic chat deletion stays a separate choice. On success, announce `Meeting deleted`, remove any recovery record, and focus the next row (previous when last). Do not invent Undo unless the backend operation becomes reversible.

- [ ] **Step 5: Run scene tests, axe, and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings && npm run typecheck`

Expected: PASS with zero axe violations.

- [ ] **Step 6: Commit Desk and collection**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/meetings/DeskScene.tsx app/src/renderer/src/meetings/MeetingsCollectionScene.tsx app/src/renderer/src/meetings/__tests__ app/src/renderer/src/workspace/WorkspaceShell.tsx
git commit -m "feat(renderer): add desk and meeting collection scenes"
```

## Phase D — Meeting Document, Capture, and Review

### Task 14: Implement the single-writer meeting document controller

**Files:**
- Create: `app/src/renderer/src/meetings/MeetingDocumentController.ts`
- Create: `app/src/renderer/src/meetings/ProposalQueue.ts`
- Create: `app/src/renderer/src/meetings/proposalAnchor.ts`
- Create: `app/src/renderer/src/meetings/__tests__/MeetingDocumentController.test.ts`
- Create: `app/src/renderer/src/meetings/__tests__/ProposalQueue.test.ts`
- Modify: `app/src/renderer/src/data/MeetingRepository.ts`

**Interfaces:**
- Exact `NoteMutationIntent` and `NoteProposal` unions from design §§10.3–10.4.
- Controller public API: `subscribe`, `getSnapshot`, `userEdited`, `submitProposal`, `acceptProposal`, `rejectProposal`, `undo`, `retrySave`, `resolveConflict`, `flush(deadlineMs)`, `dispose`.
- One in-flight note PATCH; 700 ms debounce; journal after 400 ms; mutation UUID + expected revision on every body write.

- [ ] **Step 1: Write failing deterministic queue and recovery tests**

Cover edits during a request, late/mismatched acknowledgement, duplicate mutation response, offline then retry, 409 with both versions preserved, title-only/status-only updates, stop flush at 3 seconds, renderer crash journal replay, sidecar restart, server-equal cleanup, encrypted recovery conflict, meeting deletion, and disposal. Assert no user keystroke waits for network and no 409 causes an automatic overwrite.

- [ ] **Step 2: Run focused tests and verify current autosave lacks revision safety**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings/__tests__/MeetingDocumentController.test.ts src/renderer/src/meetings/__tests__/ProposalQueue.test.ts`

Expected: FAIL because current note writers bypass a central revision queue.

- [ ] **Step 3: Implement the controller state machine**

Represent `clean | dirty | saving | offline | conflict | error` explicitly plus `serverRevision`, `lastAcknowledgedMarkdown`, `currentMarkdown`, `localVersion`, and queue head. Serialize newest accepted editor state only. An ack advances the server revision only when its mutation ID matches the queue head; if localVersion advanced, immediately schedule the next mutation. A 409 stores server/current/recovered versions and exposes only explicit choices: `Keep recovered draft`, `Use saved version`, `Copy both to clipboard`; never arbitrary auto-merge.

Healthy save chrome is absent. Expose `Saving…` only after 1.5 seconds, fade on ack, and retain `Not saved — Retry` on error.

- [ ] **Step 4: Make all proposal state route through the controller**

Validate `anchor` + `occurrence` against the live editor snapshot. Empty anchor is the only append instruction. Anchor miss becomes `conflict` with `Choose location`; it never silently appends. Accept is one editor transaction, writes Markdown + `ai_lines` + suggestion/actionable status atomically, and adds one undo transaction ID. Reject persists status without changing document text. Pending proposal Markdown is excluded from canonical serialization and live notepad frames.

- [ ] **Step 5: Run controller tests and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings/__tests__/MeetingDocumentController.test.ts src/renderer/src/meetings/__tests__/ProposalQueue.test.ts && npm run typecheck`

Expected: PASS, including zero silent-overwrite and pending-text serialization assertions.

- [ ] **Step 6: Commit the single-writer core**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/meetings/MeetingDocumentController.ts app/src/renderer/src/meetings/ProposalQueue.ts app/src/renderer/src/meetings/__tests__ app/src/renderer/src/data/MeetingRepository.ts
git commit -m "feat(renderer): centralize meeting note mutations"
```

### Task 15: Build the editorial meeting document and proposal review UI

**Files:**
- Create: `app/src/renderer/src/meetings/MeetingScene.tsx`
- Create: `app/src/renderer/src/meetings/MeetingDocument.tsx`
- Create: `app/src/renderer/src/meetings/ReviewChanges.tsx`
- Create: `app/src/renderer/src/meetings/__tests__/MeetingDocument.test.tsx`
- Create: `app/src/renderer/src/meetings/__tests__/ReviewChanges.test.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`
- Modify: `app/package.json`

**Interfaces:**
- One BlockNote instance per open meeting route, owned by the controller; approved slash blocks only.
- `ReviewChanges` presents durable note proposals and actionables as separate groups.
- Accepted AI provenance is an annotation, not disabled-looking text; editing an annotated line removes its provenance marker.

- [ ] **Step 1: Write failing editor, proposal, undo, and responsive tests**

Test optional `Untitled meeting`, 32/38 title expansion on focus, 17/28 body, 74-character line measure, metadata/More, allowed slash block list, selection-only toolbar, absence of persistent AI controls, and editor availability in every lifecycle state. Test readable diff, exact origin labels, Applying state, anchor conflict sheet, accept/reject, 30-second visible Undo, ordinary editor undo, actionables statuses and omission of unknown assignee/due fields, and standalone Meeting record after rejection.

- [ ] **Step 2: Run tests and verify the legacy editor/direct-AI UI fails**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings/__tests__/MeetingDocument.test.tsx src/renderer/src/meetings/__tests__/ReviewChanges.test.tsx`

Expected: FAIL; current editor exposes direct Ask insertion/emoji controls and lacks unified proposal review.

- [ ] **Step 3: Build the document layout and constrained BlockNote schema**

Lazy-load BlockNote on a meeting route. Permit paragraph, H2/H3, bullet/number/check list, quote, code, divider, and Ask Taurine only. Central column is 620–760 px within a 1120 px outer bound; secondary lane never shrinks it below 620 px. At 960–1179 secondary content is a drawer; at 840–959 it is a modal sheet. No horizontal document scroll. Title and note remain editable during live, reconnecting, summarizing, review error, and sidecar unavailable states.

- [ ] **Step 4: Build proposal/actionable review without canonical leakage**

Render unchanged anchor context plus a 3 px moss rule on added content, rationale, and source (`Final review`, `Ask`, `Inline Ask`). `Accept all visible` first confirms that each visible proposal becomes a separate undoable editor transaction; then process sequentially, stopping on conflict. Actionable Confirm/Dismiss is independent and reversible. Show `Preparing review…`, `Review changes`, `Review could not be prepared`, or neutral `No prepared review` exactly from lifecycle truth. The error state offers Meeting record, Diagnostics, and `Ask for a recap` labeled as a new conversation; never a final-review retry.

- [ ] **Step 5: Run editor/review tests, axe, and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings && npm run typecheck && npm run lint`

Expected: PASS; pending proposal text never appears in persisted Markdown fixtures.

- [ ] **Step 6: Commit the meeting document UI**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/meetings app/src/renderer/src/workspace/WorkspaceShell.tsx app/package.json app/package-lock.json
git commit -m "feat(renderer): add editorial meeting document and review"
```

### Task 16: Unify capture, session transport, and live-state reconciliation

**Files:**
- Create: `app/src/renderer/src/data/SessionTransport.ts`
- Create: `app/src/renderer/src/data/__tests__/SessionTransport.test.ts`
- Create: `app/src/renderer/src/meetings/CaptureController.ts`
- Create: `app/src/renderer/src/meetings/LiveMeetingStore.ts`
- Create: `app/src/renderer/src/meetings/CaptureControl.tsx`
- Create: `app/src/renderer/src/meetings/__tests__/CaptureController.test.ts`
- Create: `app/src/renderer/src/meetings/__tests__/LiveMeetingStore.test.ts`
- Modify: `app/src/renderer/src/workspace/WorkspaceTitlebar.tsx`

**Interfaces:**
- `SessionTransport` alone owns `/session`, binary frames, start/replay/stop, and matching `clientSessionId` for a live epoch.
- `CaptureController` owns mic/system media and reports actual per-channel `connecting | connected | error | stopped`.
- `LiveMeetingStore` merges transcript by `seq`, summary by `rev`, grounding by `revision`, actionables by ID, and derives the exact lifecycle in design §11.2.

- [ ] **Step 1: Add failing real-websocket, reducer, and partial-channel tests**

Test start payload including UUID token, binary channel framing, no frames while socket is closed, reconnect with same token, snapshot dispositions, stale revision rejection, actual/requested channel truth, ASR error, 60-second grace, 10-second Resume capture, stop order, and state-reset copy. Assert local `sessionActive` intent alone never produces `Listening`.

- [ ] **Step 2: Run tests and verify duplicate/current session logic fails**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data/__tests__/SessionTransport.test.ts src/renderer/src/meetings/__tests__/CaptureController.test.ts src/renderer/src/meetings/__tests__/LiveMeetingStore.test.ts`

Expected: FAIL because session/audio state is coupled to legacy App and cannot reconcile snapshots.

- [ ] **Step 3: Implement ownership and lifecycle reducers**

Generate one `clientSessionId` for each live epoch; reconnect reuses it, while safe Resume after Stop creates a new epoch/token on the same meeting and cancels finalization only if the server accepts it. Start creates/binds meeting, acquires mic, attempts system audio, opens transport, then truthfully derives `live-full` or `live-partial`. On websocket loss stop sending audio immediately and display `Capture interrupted — reconnecting`. `state_reset` displays `Capture resumed; live context restarted` once without erasing durable transcript/notes.

- [ ] **Step 4: Implement stop and the title-bar safety control**

Stop sequence is: await controller queue up to 3 seconds; send newest canonical notepad frame; if deadline expires journal and show `Notes are still saving`; send session Stop; close media tracks; transition to summarizing. Do not confirm Stop. Offer `Resume capture` for 10 seconds. The title-bar control is ≥136×44, uses recording shape + text, elapsed time, `You + call audio`/`You only`, and a popover listing Mic, Call audio, Session link, recovery actions, and Stop.

- [ ] **Step 5: Run capture tests and checks**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data/__tests__/SessionTransport.test.ts src/renderer/src/meetings && npm run typecheck`

Expected: PASS, including exact stop ordering and partial-channel cases.

- [ ] **Step 6: Commit capture reconciliation**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/data/SessionTransport.ts app/src/renderer/src/data/__tests__/SessionTransport.test.ts app/src/renderer/src/meetings app/src/renderer/src/workspace/WorkspaceTitlebar.tsx
git commit -m "feat(renderer): unify live capture and session reconciliation"
```

### Task 17: Add live context and an accessible transcript drawer

**Files:**
- Create: `app/src/renderer/src/meetings/ContextMargin.tsx`
- Create: `app/src/renderer/src/meetings/TranscriptDrawer.tsx`
- Create: `app/src/renderer/src/meetings/__tests__/ContextMargin.test.tsx`
- Create: `app/src/renderer/src/meetings/__tests__/TranscriptDrawer.test.tsx`
- Modify: `app/src/renderer/src/meetings/MeetingScene.tsx`

**Interfaces:**
- Context order: Live understanding, Action items, Related, More context; maximum three related notes and two action titles.
- Transcript has one semantic finalized list and one stable polite interim region; virtualization must preserve semantic row access.
- Breakpoints: simultaneous margin + 380 px transcript ≥1440; one 320 px lane at 1180–1439; overlay ≤92vw below 1180.

- [ ] **Step 1: Write failing content, abstention, follow, and keyboard tests**

Assert first meaningful summary line/revision, action limits, three related-note cap, Tentative fallback language, absence of empty insight cards on abstention, and evidence-drawer diagnostic only. For transcript assert You/Others, timestamp disclosure, search/no result, copy visible transcript, local channel errors, stable interim live region, follow only at bottom, `Jump to latest` after upward scroll, focus trap in overlay, and no position theft as chunks arrive.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings/__tests__/ContextMargin.test.tsx src/renderer/src/meetings/__tests__/TranscriptDrawer.test.tsx`

Expected: FAIL with missing surfaces.

- [ ] **Step 3: Implement quiet context with exact trust labels**

Use a bordered lane, never a floating card. Related items show title, one selected snippet, source/date, and no score. Confirmation controls are visible on keyboard focus as well as hover. Retrieval abstention renders nothing in the margin. Fallback/low confidence is labeled `Tentative`; never use `Verified`, `Strong`, or `Weak`.

- [ ] **Step 4: Implement the transcript semantic/virtualized split**

Use `@tanstack/react-virtual` for long visible rows while preserving a complete accessible semantic list strategy validated with screen readers (nonvisual offscreen text or paged DOM, not canvas). Interim updates replace text in one stable polite region. Track distance from the scroll bottom; only follow when already at bottom. Copy uses filtered visible finalized rows plus current interim text, never hidden search misses.

- [ ] **Step 5: Run tests, axe, and long-list measurement**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/meetings && npm run typecheck`

Expected: PASS; a 10,000-row fixture keeps rendered visual rows bounded and keyboard/accessible search remains correct.

- [ ] **Step 6: Commit meeting secondary surfaces**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/meetings/ContextMargin.tsx app/src/renderer/src/meetings/TranscriptDrawer.tsx app/src/renderer/src/meetings/__tests__ app/src/renderer/src/meetings/MeetingScene.tsx app/package.json app/package-lock.json
git commit -m "feat(renderer): add quiet live context and transcript"
```

## Phase E — Ask, Evidence, Knowledge Notes, and Comparison

### Task 18: Create one renderer-wide ConversationHub

**Files:**
- Create: `app/src/renderer/src/data/ConversationHub.ts`
- Create: `app/src/renderer/src/data/__tests__/ConversationHub.test.ts`
- Modify: `app/src/renderer/src/data/contracts/chat.ts`
- Modify: `app/src/renderer/src/data/MeetingRepository.ts`

**Interfaces:**
- One `/chat` websocket; streams keyed by `{chatId,messageId}`.
- Commands: `send`, `stop`, `regenerate`, `rename`, `delete`, `retryInterrupted`, `subscribeConversation`, `getSnapshot`.
- `chat_notes_edit` becomes a `NoteProposal` submitted to the meeting controller; hub never edits notes.

- [ ] **Step 1: Write failing ordering, multiplexing, and reconnect tests**

Cover ack replacing the provisional client row; interleaved two-chat events; stable retrieval/activity slots; content deltas only for matching message; raw thinking retained but not public-rendered; done replaced by persisted decoded message; error/history refresh; socket loss; terminal-history-wins; unfinished partial labeled `Connection lost`; Retry; regenerate acknowledgement; and edit event proposal conversion.

- [ ] **Step 2: Run tests and verify multiple current chat owners fail**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data/__tests__/ConversationHub.test.ts`

Expected: FAIL because current global/meeting/inline clients cannot multiplex consistently.

- [ ] **Step 3: Implement a keyed reducer and single transport lifecycle**

Instantiate the hub once in bootstrap. Exhaustively switch known events and log/ignore unknown types. Deduplicate activity by stable server step identity instead of appending. `chat_done` refreshes/replaces the entire terminal message. On reconnect, fetch each affected chat once; do not infer cited evidence from partial citation-shaped text. Regenerate retains the prior answer until server acknowledgement, matching current backend semantics.

- [ ] **Step 4: Convert note edits to proposals**

Map `chat_notes_edit` to origin `meeting-ask` or `inline-ask`, a UUID proposal ID, visible turn/tool rationale, server anchor/occurrence/insert Markdown, and the controller's current base revision. Do not create a `user-edit` mutation until Accept. Add an invariant test that no method on the hub receives a BlockNote editor instance.

- [ ] **Step 5: Run hub/data tests and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/data && npm run typecheck`

Expected: PASS with one websocket in the Strict Mode lifecycle test.

- [ ] **Step 6: Commit the conversation owner**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/data/ConversationHub.ts app/src/renderer/src/data/__tests__/ConversationHub.test.ts app/src/renderer/src/data/contracts/chat.ts app/src/renderer/src/data/MeetingRepository.ts
git commit -m "feat(renderer): centralize conversation streaming"
```

### Task 19: Render complete, contract-honest evidence semantics

**Files:**
- Create: `app/src/renderer/src/ask/evidenceSemantics.ts`
- Create: `app/src/renderer/src/ask/EvidenceLedger.tsx`
- Create: `app/src/renderer/src/ask/EvidenceRef.tsx`
- Create: `app/src/renderer/src/ask/ProcessDisclosure.tsx`
- Create: `app/src/renderer/src/ask/EvidenceDisclosure.tsx`
- Create: `app/src/renderer/src/ask/__tests__/evidenceSemantics.test.ts`
- Create: `app/src/renderer/src/ask/__tests__/EvidenceDisclosure.test.tsx`

**Interfaces:**
- `EvidenceLedger` contains distinct ordered `cited`, `consulted`, `retrieved` items; precedence is Cited > Consulted > Retrieved.
- Retains fallback, low confidence, passage date, prior context, decision category, interruption, stripped claims, fact direction/temporal status, path mechanisms/weights, policy/check status, and unavailable fields.
- Evidence link callback carries exact `ComparisonRef`/knowledge note origin; IDs are never coerced into note IDs.

- [ ] **Step 1: Write failing exhaustive trust-language fixtures**

Build fixtures for duplicate IDs across sets, fallback below floor, abstention, earlier context, interrupted raw `[E1]`, stripped claims, missing fields, relative+absolute dates, fact states (`active`, `historical`, `superseded`, `conflicted`, `candidate`), path order, zero-sample calibration, and policy/check-pass flags. Assert exact permitted labels and repository-wide absence within this surface of `Verified`, `Fact checked`, `True`, and scalar-derived `Strong`/`Weak`.

- [ ] **Step 2: Run evidence tests and verify current parser loses fields**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/ask/__tests__/evidenceSemantics.test.ts src/renderer/src/ask/__tests__/EvidenceDisclosure.test.tsx`

Expected: FAIL on missing fields and precedence rules.

- [ ] **Step 3: Implement the evidence ledger as a pure projection**

Deduplicate by evidence ID at highest state, then preserve server order within each group. Interrupted messages have no cited set regardless of citation-shaped text. Low-confidence fallback passages always read `Tentative evidence`; no-floor results omit an empty Sources block and state insufficiency only where it affects the answer. Prior items say `From earlier context`. Unknown confidence/date/temporal values render `Unavailable`, never defaults.

- [ ] **Step 4: Implement collapsed accessible disclosure and source navigation**

Render Cited, Also consulted, then Retrieved. Put `Evidence policy passed` and `Citation checks passed` inside Process details only. When stripped claims exist, show `Some unsupported draft text was removed`. Facts retain subject → predicate → object and direction; paths retain ordered objects and mechanisms. Sentence citations navigate to exact evidence; consulted-only rows never claim sentence support.

- [ ] **Step 5: Run tests, axe, and vocabulary scan**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/ask && npm run typecheck && ! rg -n '\b(Verified|Fact checked|Strong evidence|Weak evidence)\b' src/renderer/src/ask`

Expected: PASS and `rg` prints no matches.

- [ ] **Step 6: Commit evidence semantics**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/ask
git commit -m "feat(renderer): preserve honest evidence semantics"
```

### Task 20: Build meeting Ask and standalone conversation documents

**Files:**
- Create: `app/src/renderer/src/ask/ObjectAskSheet.tsx`
- Create: `app/src/renderer/src/ask/ConversationScene.tsx`
- Create: `app/src/renderer/src/ask/ConversationTurn.tsx`
- Create: `app/src/renderer/src/ask/__tests__/ObjectAskSheet.test.tsx`
- Create: `app/src/renderer/src/ask/__tests__/ConversationScene.test.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`

**Interfaces:**
- Object Ask accepts an explicit scope descriptor and supported `meeting_id`/`seed_note_ids`; it never fabricates hard scope.
- Sheet is ≤760 px, ≤70vh, focuses composer, and preserves its route state when opening evidence.
- Conversation document is a 760 px thread with editable title and no persistent thread sidebar.

- [ ] **Step 1: Write failing sheet/stream/conversation tests**

Assert named noninteractive meeting scope, streamed text in one stable region, collapsed `Looked through N sources`, exact evidence groups, proposal copy, citation navigation + Back restoration, and Escape behavior: close when idle, first Escape offers `Stop response` when active. Test global conversation rename/delete focus, stop/retry/regenerate, interrupted `Stopped before verification`, and no permanent list/sidebar.

- [ ] **Step 2: Run tests and verify legacy Ask paths violate proposal ownership**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/ask/__tests__/ObjectAskSheet.test.tsx src/renderer/src/ask/__tests__/ConversationScene.test.tsx`

Expected: FAIL; current inline Ask writes directly and meeting chat owns another socket.

- [ ] **Step 3: Implement both surfaces over ConversationHub**

Lazy-load Markdown/evidence-heavy UI on first Ask/conversation entry. Meeting Ask passes exactly its meeting ID. Inline Ask completes one bounded response and emits one proposal; it never streams into BlockNote. Collapse search/read/expand activities to one count, while Process details exposes stable steps. Preserve Ask sheet serialized state in the meeting route entry when navigating to a source.

- [ ] **Step 4: Add deletion, interruption, and error truth**

Chat errors are turn-local. After transport loss, keep persisted history and label unfinished partials; retry starts a new attempt. Delete and rename use document menus and return focus predictably. Never expose raw thinking or stripped draft text by default.

- [ ] **Step 5: Run Ask tests, axe, and checks**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/ask src/renderer/src/data/__tests__/ConversationHub.test.ts && npm run typecheck`

Expected: PASS.

- [ ] **Step 6: Commit Ask and conversation scenes**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/ask app/src/renderer/src/workspace/WorkspaceShell.tsx
git commit -m "feat(renderer): add contextual ask and conversation documents"
```

### Task 21: Add knowledge-note documents and the bounded comparison set

**Files:**
- Create: `app/src/renderer/src/knowledge/KnowledgeNoteScene.tsx`
- Create: `app/src/renderer/src/knowledge/comparisonSet.ts`
- Create: `app/src/renderer/src/knowledge/CompareScene.tsx`
- Create: `app/src/renderer/src/knowledge/__tests__/KnowledgeNoteScene.test.tsx`
- Create: `app/src/renderer/src/knowledge/__tests__/comparisonSet.test.ts`
- Create: `app/src/renderer/src/knowledge/__tests__/CompareScene.test.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`

**Interfaces:**
- Comparison set is renderer-memory-only, exactly 0–3 distinct `ComparisonRef`; tray appears only at 1–3 and Compare only at 2–3.
- `deriveComparisonAskPayload(items)` returns supported `{meeting_id?: string, seed_note_ids: string[], excluded: string[]}` using design §5.7 exactly.
- Knowledge note uses current graph generation; missing current body is never rendered from stale cache.

- [ ] **Step 1: Write failing knowledge/missing-object/cap/payload tests**

Cover source/date/folder metadata, conservative body, Ask/Explore/Add, supported-only native open, Why this appeared, short related list, and generation-swap missing copy. Test duplicate adds, fourth disabled with `Comparison is limited to 3 objects`, tray zero/one/two/three states, removal/focus, restart clear, and missing-object privacy.

Table-test Ask derivation for notes; one meeting indexed/unindexed; two meetings; answer/fact/path with distinct originating note IDs; and no valid seeds. Assert prose/fact/path IDs are never sent and caveat is exactly `Retrieval may consult other knowledge.`

- [ ] **Step 2: Run tests and verify modules are absent**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge/__tests__/KnowledgeNoteScene.test.tsx src/renderer/src/knowledge/__tests__/comparisonSet.test.ts src/renderer/src/knowledge/__tests__/CompareScene.test.tsx`

Expected: FAIL.

- [ ] **Step 3: Implement note documents and transient tray**

Use a 760 px reading sheet and current generation fetch. On center disappearance show `This source is no longer in the current knowledge index`, Back, Sync status, and nonprivate cached title/source only. The 40 px comparison tray names every item and exposes Remove/Clear/Compare. It never reserves space at zero or persists across restart.

- [ ] **Step 4: Implement responsive dossiers and honest Ask**

Render native dossier content for note/meeting/answer/fact/path without model synthesis. Shared exact labels/sources/dates/evidence IDs receive a 3 px rule plus text. ≥1320 allows 2–3 columns at ≥400 px; 960–1319 allows two only at ≥440 px; otherwise use one labeled `N of M` dossier. At 200% zoom force one-at-a-time and no page horizontal scroll. `Align headings` synchronizes section jumps only. Disable Ask with the exact contract-explanation copy when no valid context exists.

- [ ] **Step 5: Run tests, axe, and responsive checks**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge/__tests__/KnowledgeNoteScene.test.tsx src/renderer/src/knowledge/__tests__/comparisonSet.test.ts src/renderer/src/knowledge/__tests__/CompareScene.test.tsx && npm run typecheck`

Expected: PASS at 840, 960, 1320 and 200% fixtures.

- [ ] **Step 6: Commit note and comparison scenes**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/knowledge app/src/renderer/src/workspace/WorkspaceShell.tsx
git commit -m "feat(renderer): add knowledge documents and bounded comparison"
```

## Phase F — Connections

### Task 22: Ship Connections List as the complete semantic baseline

**Files:**
- Create: `app/src/renderer/src/knowledge/ConnectionsScene.tsx`
- Create: `app/src/renderer/src/knowledge/ConnectionsList.tsx`
- Create: `app/src/renderer/src/knowledge/ConnectionFilters.tsx`
- Create: `app/src/renderer/src/knowledge/__tests__/ConnectionsList.test.tsx`
- Create: `app/src/renderer/src/knowledge/__tests__/connectionsData.test.ts`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`

**Interfaces:**
- `ConnectionScope` supports global, cluster/idea, or center note plus depth 1–3; local requests use query name `center`.
- Edge detail uses returned `GraphEdge` fields and `/facts?ids=` batches ≤50 only; there is no `Explain this edge` action.
- List offers every essential operation: search, focus/traverse, filter, fact inspection, open note, compare, and supported Ask seeding.

- [ ] **Step 1: Write failing request, language, keyboard, and generation tests**

Assert object/evidence opens depth 1; global opens search/idea first; explicit Explore all loads global territory data. Test fixed disclosure, breadcrumbs, 1–3 depth, source/cluster/date/mechanism filters including custom date fields, title/gist/entity search, Projection weight/Recent/Title sorts, grouping by fact/entity/semantic explanation, Enter/Right/Space/Shift+F10, facts provenance/temporal state, unknown center, filter-empty, skeleton, 5–20 second idea progress + Cancel, and generation swap without mixed builds.

- [ ] **Step 2: Run tests and verify current map has no complete list equivalent**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge/__tests__/ConnectionsList.test.tsx src/renderer/src/knowledge/__tests__/connectionsData.test.ts`

Expected: FAIL.

- [ ] **Step 3: Implement contract-exact data loading and list interactions**

Use `/graph/local?center={noteId}&depth={1..3}`, `/graph`, `/graph/idea`, note detail, and `/facts?ids=` only. Explain relationships from existing mechanisms/entity metadata/facts. Label the scalar `Projection weight — not confidence`; ordinary rows never call it Strong/Weak. Do not call `/graph/edge-reason`: projection edges contain note endpoints, not required passage IDs. Cycles remain rows, not a fake tree.

- [ ] **Step 4: Add honest global/idea/empty/degraded states**

Keep the center heading while loading. Preserve a stale same-scope list until a new graph generation resolves, then replace atomically and announce `Connections updated`; missing center changes to not-found and discards pins only for missing IDs. `Start from selected notes` accepts Taurine note IDs only and says `Starting points, not an exclusive search boundary`.

- [ ] **Step 5: Run list tests, axe, and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge && npm run typecheck`

Expected: PASS; all essential actions work without canvas.

- [ ] **Step 6: Commit Connections List**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/knowledge app/src/renderer/src/workspace/WorkspaceShell.tsx
git commit -m "feat(renderer): add accessible connections list"
```

### Task 23: Add the lazy, stable Connections visual enhancement

**Files:**
- Create: `app/src/renderer/src/knowledge/ConnectionsVisual.lazy.tsx`
- Create: `app/src/renderer/src/knowledge/connectionLayout.worker.ts`
- Create: `app/src/renderer/src/knowledge/connectionsModel.ts`
- Create: `app/src/renderer/src/knowledge/__tests__/connectionsLayout.test.ts`
- Create: `app/src/renderer/src/knowledge/__tests__/ConnectionsVisual.test.tsx`
- Modify: `app/src/renderer/src/knowledge/ConnectionsScene.tsx`
- Modify: `app/electron.vite.config.ts`

**Interfaces:**
- Layout cache key `{buildId,noteId}`; deterministic seed uses cluster ID, stable note ID, degree, and date.
- Canvas is `aria-hidden`; adjacent DOM region owns selection, Open/Add/Focus/Pin actions.
- Representation preference is versioned local preference; List remains default/accessibility baseline.

- [ ] **Step 1: Write failing deterministic geometry, LOD, and parity tests**

Assert repeatable positions, unchanged IDs reuse prior coordinates after build swap, new nodes begin at cluster centroids, missing pins drop, Reset works without drag, and reduced motion freezes after initial placement. Test fit/medium/close LOD caps, fit edges absent unless selected, node sizes, line patterns and square fact marker, neutral source markers, DOM detail parity, forced-colors, and lazy chunk absence before Visual selection.

- [ ] **Step 2: Run tests and verify legacy force behavior fails stability/accessibility**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge/__tests__/connectionsLayout.test.ts src/renderer/src/knowledge/__tests__/ConnectionsVisual.test.tsx`

Expected: FAIL.

- [ ] **Step 3: Implement stable territory-first geometry and LOD**

At fit render cluster hulls, 2–4 word names/counts, selected objects, and at most three representative hubs per territory. At medium add recent/filter matches and bridges; close adds visible labels/selection-adjacent edges. Center is 36 px outlined with always-visible title; neighbors are 20–28 px by degree. Use exact semantic/topic/entity/mixed line styles and directed-fact midpoint marker. No rainbow source modes or perpetual heat.

- [ ] **Step 4: Isolate the visual dependency**

Load `react-force-graph-2d` and visual CSS through `React.lazy` only after explicit Visual selection. Configure Rollup manual chunk names `connections-visual` and `blocknote-editor` so budget tooling can prove dependency isolation. Keep DOM search/List available throughout canvas errors. At 840–959 require an explicit full-scene Visual switch.

- [ ] **Step 5: Run tests and inspect production manifest**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/knowledge && npm run build && node scripts/bundle-budget.mjs --manifest out/renderer/.vite/manifest.json --report-only`

Expected: tests/build PASS; initial entry imports neither `connections-visual` nor force-graph files.

- [ ] **Step 6: Commit the visual enhancement**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/knowledge app/electron.vite.config.ts
git commit -m "feat(renderer): add lazy stable connections visual"
```

## Phase G — Settings, Diagnostics, and Local Telemetry

### Task 24: Port Sources with truthful connector and sync semantics

**Files:**
- Create: `app/src/renderer/src/settings/SettingsIndexScene.tsx`
- Create: `app/src/renderer/src/settings/SourcesScene.tsx`
- Create: `app/src/renderer/src/settings/__tests__/SourcesScene.test.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`
- Modify: `app/src/renderer/src/platform/electronBridge.ts`

**Interfaces:**
- Settings index rows: Sources, Personal context, Privacy & Capture, Appearance, Shortcuts, Diagnostics.
- Connector status priority: Syncing > Needs attention > Ready > Off.
- Existing connector configuration/sync/cancel/delete/choose-folder contracts only; no fake OAuth button or new endpoint.

- [ ] **Step 1: Write failing projection, copy, destructive, and reachability tests**

Assert aggregate `N sources on · N documents`, textual priority status, expanded config/last report/actions, exact On/Off consequence copy, Sync now/Cancel, error recovery, folder chooser cancellation, and named Delete data confirmation with document count/rebuild effect. Assert `Advanced setup` contains terminal authorization instructions for current env-based providers and no Sign in button. Verify fixed Open/native/settings-index paths.

- [ ] **Step 2: Run tests and verify the old connector destination fails IA/copy**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/settings/__tests__/SourcesScene.test.tsx`

Expected: FAIL.

- [ ] **Step 3: Implement source projections and existing mutations**

Derive one status from current connector/sync payloads. Preserve last truthful rows while refreshing. Enable means only enabled; say `On — update knowledge to include this source`. Disable says stored data remains and current view excludes it. Purge confirmation names source/document count and explains derived rebuild. Route all chooser use through the allowlisted preload bridge.

- [ ] **Step 4: Add settings-only breadcrumb and error focus**

Use plain rows, not permanent category nav. After mutation errors retain expanded row/data and focus its recovery action. Do not show routine-success toasts. Do not implement §13.6 authorization endpoints in this redesign.

- [ ] **Step 5: Run settings tests, axe, and typecheck**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/settings && npm run typecheck`

Expected: PASS.

- [ ] **Step 6: Commit Sources**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/settings app/src/renderer/src/workspace/WorkspaceShell.tsx app/src/renderer/src/platform/electronBridge.ts
git commit -m "feat(renderer): move sources into quiet settings"
```

### Task 25: Port personal context, privacy preferences, shortcuts, and diagnostics

**Files:**
- Create: `app/src/renderer/src/settings/PersonalContextScene.tsx`
- Create: `app/src/renderer/src/settings/PrivacyCaptureScene.tsx`
- Create: `app/src/renderer/src/settings/AppearanceScene.tsx`
- Create: `app/src/renderer/src/settings/ShortcutsScene.tsx`
- Create: `app/src/renderer/src/settings/DiagnosticsScene.lazy.tsx`
- Create: `app/src/renderer/src/settings/__tests__/settings-scenes.test.tsx`
- Modify: `app/src/renderer/src/settings/SettingsIndexScene.tsx`

**Interfaces:**
- Uses current profile, session-log, startup, sync/build, ASR, call-detection, and capture preference contracts.
- New-install defaults: call auto-start off and relevant-note OS notifications off.
- Diagnostic export defaults content-free; explicit include-content action is separately confirmed.

- [ ] **Step 1: Write failing state/copy/privacy/redaction tests**

Cover profile view/edit/generate/regenerate/overwrite confirmation, concise current phase plus counts disclosure, retained profile on failure, and uncitable-context explanation. Cover mic/system capability, call detection, opt-in countdown, suggestion notifications default-off, Presentation-safe mode, curtain, Hide shortcut, retention links, light/dark/system appearance, and complete shortcut table. Cover diagnostics readiness/history/sync/build/ASR/session list, lazy loading, redacted bundle, explicit content inclusion, and warned raw JSONL disclosure.

- [ ] **Step 2: Run tests and verify legacy settings/diagnostics are fragmented**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/settings/__tests__/settings-scenes.test.tsx`

Expected: FAIL.

- [ ] **Step 3: Implement profile, privacy, appearance, and shortcuts**

Show one current research activity sentence and disclose nodes/tool counts; no decorative research rail. Preserve old content through generation failure. Privacy copy says platform content protection is best-effort and may not prevent ScreenCaptureKit/third-party capture; never say invisible/guaranteed. Presentation-safe suppresses private peripheral content, closes transcript/evidence drawers, suppresses note previews and suggestion notifications without stopping capture, and adds the persistent quiet shield whose tooltip is `Best-effort privacy mode`. Keep the exact global shortcuts from §3.5 visible and consistent with native menu tests.

- [ ] **Step 4: Implement lazy, copy-safe diagnostics**

Load the session-log/JSON viewer only on Diagnostics. Default copy includes event times/durations, coarse codes, versions/platform/window/route kinds and excludes titles, IDs, paths, note/transcript/query/evidence/model/OAuth content. Raw JSONL remains Advanced with warning. Add a manifest assertion that diagnostics code is absent from initial imports.

- [ ] **Step 5: Run settings tests, axe, typecheck, and manifest report**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/settings && npm run typecheck && npm run build && node scripts/bundle-budget.mjs --manifest out/renderer/.vite/manifest.json --report-only`

Expected: PASS; diagnostics remains lazy.

- [ ] **Step 6: Commit remaining settings and diagnostics**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/settings
git commit -m "feat(renderer): add truthful privacy profile and diagnostics settings"
```

### Task 26: Add bounded content-free local product telemetry

**Files:**
- Create: `app/src/main/local-telemetry.ts`
- Create: `app/src/main/__tests__/local-telemetry.test.ts`
- Create: `app/src/renderer/src/telemetry/productEvents.ts`
- Create: `app/src/renderer/src/telemetry/__tests__/productEvents.test.ts`
- Modify: `app/src/preload/index.ts`
- Modify: `app/src/preload/index.d.ts`

**Interfaces:**
- Closed `ProductEventName` union is exactly design §14.2.
- Allowed payload fields only: monotonic duration, boolean outcome, coarse error code, window mode, route kind, app/platform version.
- Local-only rotating log; no remote analytics; diagnostic export is explicit.

- [ ] **Step 1: Write failing allowlist, forbidden-content, and rotation tests**

Table-test every vocabulary name. At compile time and runtime reject note/transcript/query/title/note ID/evidence/model/path/OAuth keys and arbitrary event names. Assert bounded file count/bytes, atomic rotation, telemetry-disabled behavior, and exported default redaction.

- [ ] **Step 2: Run tests and verify telemetry module is absent**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main/__tests__/local-telemetry.test.ts src/renderer/src/telemetry/__tests__/productEvents.test.ts`

Expected: FAIL.

- [ ] **Step 3: Implement closed event emitters and narrow IPC**

Expose `telemetry.record({name,durationMs?,outcome?,errorCode?,windowMode?,routeKind?})`. Validate and drop unknown keys in preload and main; never serialize content-adjacent objects. Use monotonic renderer timings and append app/platform versions in main. Rotate locally and keep remote upload nonexistent.

- [ ] **Step 4: Wire only decision-point events from §14.2**

Emit at controller/store action boundaries, not React renders. Do not optimize UI for context-open, acceptance, time-in-app, clicks, chat-turn, or sync-frequency guardrail metrics. Unit-test representative meeting/save/resume/review/Ask/Connections/sync/privacy events.

- [ ] **Step 5: Run telemetry tests, typecheck, and redaction scan**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main/__tests__/local-telemetry.test.ts src/renderer/src/telemetry && npm run typecheck`

Expected: PASS with no forbidden key accepted.

- [ ] **Step 6: Commit local telemetry**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/main/local-telemetry.ts app/src/main/__tests__/local-telemetry.test.ts app/src/renderer/src/telemetry app/src/preload
git commit -m "feat(privacy): add bounded local product telemetry"
```

### Task 27: Close service composition, preferences, and peripheral-window capability parity

**Files:**
- Create: `app/src/renderer/src/bootstrap/services.ts`
- Create: `app/src/renderer/src/workspace/WorkspaceContext.tsx`
- Create: `app/src/renderer/src/platform/PreferenceStore.ts`
- Create: `app/src/renderer/src/platform/windowPreferences.ts`
- Create: `app/src/renderer/src/bootstrap/__tests__/services.test.ts`
- Modify: `app/src/main/call-notify-state.ts`
- Modify: `app/src/main/notifier.ts`
- Modify: `app/src/main/suggest-notifier.ts`
- Modify: `app/src/renderer/src/notif-main.ts`
- Modify: `app/src/renderer/src/suggest-main.ts`
- Modify: `app/src/renderer/notif.html`
- Modify: `app/src/renderer/suggest.html`

**Interfaces:**
- `WorkspaceServices` is the only composition interface for gateway/stores/repositories/router/controllers/telemetry/platform ports.
- `PreferenceStore` schema persists only theme, graph representation/pins/filters, auto-start/countdown, notification policy, and version; it discards old active-tab/panel keys.
- Call prompt is non-focusable and explicit-start by default; auto-start requires the enabled preference and visible countdown. Suggestion popup is off by default and curtain/presentation-safe redacted.

- [ ] **Step 1: Write failing composition, migration, call-policy, and peripheral tests**

Assert one instance each of updates/chat transport and repositories, deterministic disposal, and no feature imports of Electron/transport globals. Test preference v0/v1 migration retains theme/pins but drops tab/panel state and invalid values. Test call debounce/rearm, explicit Start notes/Not now, opt-in countdown/cancel, hidden-main no-focus behavior, supported/unsupported system audio, notification default-off, redaction, curtain-before-paint, accessible names, and no emoji/private titles.

- [ ] **Step 2: Run tests and verify current prompt auto-start/popup policy differs**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/renderer/src/bootstrap/__tests__/services.test.ts src/main/call-notify-state.test.ts src/main/__tests__/privacy-curtain.test.ts`

Expected: FAIL on missing service composition/preference policy and old prompt defaults.

- [ ] **Step 3: Implement target services/repository adapters and preferences**

Construct services once after readiness and dispose in reverse ownership order. Repositories wrap only their exact existing endpoints. Version local preferences; preserve meaningful theme/pins and discard obsolete UI location state. Keep recovery and telemetry out of browser storage. Feature components receive narrowed hooks, not the service bag.

- [ ] **Step 4: Port peripheral UI and native behavior**

Use 356×116 call prompt and 356×132 suggestion prompt. Call detection offers one-click Start without raising main; Not now rearms according to existing debounce. Countdown exists only when explicitly enabled. Suggestion notifications default off for new installs, queue/dismiss safely, redact under presentation-safe/curtain, and never become the default relevant-knowledge surface. Both windows acknowledge curtain before first show.

- [ ] **Step 5: Run full main/peripheral/bootstrap checks**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm test -- --run src/main src/renderer/src/bootstrap && npm run typecheck && npm run lint`

Expected: PASS; process/socket-count tests show one updates and one chat owner.

- [ ] **Step 6: Commit composition and peripheral parity**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/src/renderer/src/bootstrap app/src/renderer/src/workspace/WorkspaceContext.tsx app/src/renderer/src/platform app/src/renderer/src/data
git add app/src/main/call-notify-state.ts app/src/main/notifier.ts app/src/main/suggest-notifier.ts app/src/renderer/src/notif-main.ts app/src/renderer/src/suggest-main.ts app/src/renderer/notif.html app/src/renderer/suggest.html
git commit -m "feat(desktop): complete quiet canvas service and peripheral parity"
```

## Phase H — Full-Story Verification and Single Cutover

### Task 28: Prove complete journeys, accessibility, platforms, and performance

**Files:**
- Create: `app/playwright.config.ts`
- Create: `app/e2e/fake-sidecar.ts`
- Create: `app/e2e/electron-fixture.ts`
- Create: `app/e2e/quiet-canvas.spec.ts`
- Create: `app/e2e/accessibility.spec.ts`
- Create: `app/e2e/performance.spec.ts`
- Create: `app/e2e/manual-platform-matrix.md`
- Modify: `app/package.json`

**Interfaces:**
- Fake sidecar serves production-shaped REST plus `/updates`, `/session`, `/chat`, controllable clock/disconnect/revision conflicts/finalization/graph generations.
- E2E launches the target `RendererBootstrap` entry directly, without exposing legacy UI or adding a runtime feature flag.
- Manual matrix records OS/app/version/result and preserves best-effort privacy wording regardless of outcome.

- [ ] **Step 1: Write full-story tests before wiring the target entry**

Add journeys: call/tray/Desk start → type → partial capture → transcript → meeting Ask → proposal accept/undo → disconnect resumed/state-reset → stop flush → review reload; fixed Places and native commands; all meetings/delete; conversation interruption/evidence; note/source navigation + exact Back state; add three comparison items + responsive Ask payload; Connections List full keyboard parity + visual switch/build swap; Sources/profile/privacy/diagnostics; curtain across main/peripherals. Include sidecar down/startup/error, save conflict/recovery, final-review failure with no retry, graph/source failures, and call detection rearm.

- [ ] **Step 2: Run the E2E suite and verify target-entry gaps fail**

Add these scripts before running the red test:

```json
{
  "test:e2e": "playwright test --project=electron --grep-invert @a11y --grep-invert @performance",
  "test:e2e:a11y": "playwright test --project=electron --grep @a11y",
  "test:e2e:perf": "playwright test --project=electron --grep @performance"
}
```

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm run test:e2e`

Expected: FAIL until the fixture entry/service injection and every target journey are wired; no test may fall back to legacy selectors.

- [ ] **Step 3: Add accessibility and responsive automation**

Run axe on every scene/state/theme at 840, 960, 1180, 1320, and 1440 widths. Keyboard-only test the complete meeting journey, Connections List, comparison, fixed Places/settings, and curtain. Test 200% zoom one-dossier Compare/no horizontal page scroll, reduced motion, forced colors, stable live regions, accessible icon names, and ≥36 px target geometry. Add documented manual VoiceOver/NVDA rows for titlebar/editor/live/dialog/evidence/Sources/Connections.

- [ ] **Step 4: Add measurable performance gates**

Over 20 warm-sidecar launches on the pilot profile, capture first-content `<500 ms` and Desk keyboard-interactive `<1.5 s` p95. Assert purposeful route fallback within 100 ms, no blank frame, typical editor input frame `<16 ms`, and bounded DOM/interaction with a long note/transcript. Use trace markers from content-free telemetry; do not log private fixtures.

- [ ] **Step 5: Complete the platform/manual matrix**

Record macOS minimum/latest and Windows minimum/latest: hidden/visible call prompt, mic first-run/denied/revoked, macOS system audio supported/unsupported, tray start/stop/quit, hide-on-close, multi-display bounds, all-Spaces prompt, restart/reload/safeStorage modes, sleep/wake/provider loss, and Zoom/Meet/Teams/OBS/native screenshot/share preview. Each row has environment, exact observed result, issue link if failed, and reviewer/date; every required row must be Pass or a release-blocking issue.

- [ ] **Step 6: Run full target E2E/a11y/performance gates**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm run test:e2e && npm run test:e2e:a11y && npm run test:e2e:perf`

Expected: PASS, including the 20-launch percentile assertions and zero critical axe violations.

- [ ] **Step 7: Commit full-story verification**

```bash
cd "$(git rev-parse --show-toplevel)"
git add app/playwright.config.ts app/e2e app/package.json app/package-lock.json
git commit -m "test(app): prove quiet canvas user journeys"
```

### Task 29: Make Quiet Canvas the sole renderer and delete the old interface

**Files:**
- Create: `app/scripts/verify-quiet-canvas.mjs`
- Create: `app/scripts/verify-quiet-canvas.test.mjs`
- Modify: `app/src/renderer/src/main.tsx`
- Modify: `app/src/renderer/src/assets/main.css`
- Modify: `app/src/renderer/index.html`
- Modify: `app/package.json`
- Delete: every legacy path enumerated under **Target File Structure**

**Interfaces:**
- `main.tsx` renders only `RendererBootstrap`; there is no legacy import, compatibility route, old-interface command, feature flag, or runtime fallback.
- `verify-quiet-canvas.mjs` fails on legacy paths/imports/copy/selectors, forbidden privacy/evidence claims, missing target files, or a second transport owner.
- Existing SQLite meetings/chats/profile/connectors/knowledge and Markdown/transcript/suggestion data remain untouched except additive DDL already tested.

- [ ] **Step 1: Write the failing cutover verifier**

Its own fixture tests prove each rule fails independently. Repository rules include: all enumerated legacy files absent; no `App` import; no old five-tab labels/selectors/status footer; no `legacy`, `old UI`, `classic`, or Quiet Canvas feature flag; no feature-level `fetch`, `WebSocket`, raw IPC, or direct editor insertion; one `/updates`, `/chat`, `/session` owner; no `Verified`/guaranteed-invisibility copy; target files present; asset CSS has only three imports.

- [ ] **Step 2: Run verifier and capture the expected legacy failures**

Run: `cd "$(git rev-parse --show-toplevel)/app" && node --test scripts/verify-quiet-canvas.test.mjs && node scripts/verify-quiet-canvas.mjs`

Expected: verifier unit tests PASS; repository scan FAIL listing current `App.tsx`, tabs, old views/components/libs/assets, and entry import.

- [ ] **Step 3: Switch the single entry and delete legacy UI in one commit scope**

Change `main.tsx` to create target services and render `RendererBootstrap` only. Make `assets/main.css` import `design/tokens.css`, `design/typography.css`, and `design/globals.css` only. Remove every exact legacy file listed in Target File Structure after migrating only capture helpers still referenced. Remove dead dependencies and obsolete tab/panel preference reads. Do not retain a hidden or unreachable compatibility component.

- [ ] **Step 4: Add final verification scripts and budget enforcement**

Only now add these enforcing package scripts (so earlier tasks never point at missing files):

```json
{
  "test:all": "vitest run",
  "verify:quiet-canvas": "node scripts/verify-quiet-canvas.mjs",
  "check:bundle": "node scripts/bundle-budget.mjs"
}
```

Keep the E2E scripts from Task 28. Make the budget script enforce initial ≤1.25 MB JS/120 KB CSS, gzip ≤350/35 KB, max feature chunk ≤1.5 MB; assert BlockNote/force graph/Diagnostics absent from initial imports and named lazy chunks present.

- [ ] **Step 5: Run cutover scan, app suites, and production bundle**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm run verify:quiet-canvas && npm run test:all && npm run typecheck && npm run lint && npm run build && npm run check:bundle`

Expected: all PASS; initial manifest satisfies every dependency/budget assertion and verifier finds zero legacy runtime paths.

- [ ] **Step 6: Commit the irreversible branch-local cutover**

```bash
cd "$(git rev-parse --show-toplevel)"
git add -A app
git commit -m "refactor(app): cut over exclusively to quiet canvas"
```

The branch/commit is the rollback unit before merge. Do not ship a runtime fallback.

### Task 30: Run the complete release gate and preserve evidence

**Files:**
- Create: `docs/verification/quiet-canvas-release.md`
- Modify only if a gate exposes a defect: the smallest owning production/test file from Tasks 1–29

**Interfaces:**
- Release record contains commands, versions, counts, bundle table, 20-launch percentiles, platform-matrix status, and commit SHA; no private content.
- Required regression floor is preserved: ≥61 app Vitest behaviors and the existing complete core suite baseline (981 pass/2 documented skip before additions), with all newly added tests passing.

- [ ] **Step 1: Run formatting and static repository gates**

First name and verify the assigned worktree in the final audit:

```bash
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
case "$WORKTREE_ROOT" in
  */.worktrees/ui-quiet-canvas) printf 'final-audit-worktree=%s\n' "$WORKTREE_ROOT" ;;
  *) printf 'Wrong final-audit checkout: %s\n' "$WORKTREE_ROOT" >&2; exit 1 ;;
esac
git -C "$WORKTREE_ROOT" status --short
```

Expected: `final-audit-worktree` names the assigned `ui-quiet-canvas` linked worktree, never the primary checkout, followed only by this implementation's intentional changes.

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm run format:check && npm run lint && npm run typecheck && npm run verify:quiet-canvas`

Expected: PASS with no errors or legacy matches.

- [ ] **Step 2: Run all app tests and production/bundle gates from a clean build**

Run: `cd "$(git rev-parse --show-toplevel)/app" && npm run test:all && npm run build && npm run check:bundle && npm run test:e2e && npm run test:e2e:a11y && npm run test:e2e:perf`

Expected: PASS; report the exact Vitest/Playwright counts and budget sizes, and confirm the three heavy feature areas are absent from initial imports.

- [ ] **Step 3: Run the complete core suite**

Run: `cd "$(git rev-parse --show-toplevel)/core" && uv run pytest -q`

Expected: PASS; record the exact new pass/skip totals. Any new skip is a blocker unless documented as platform-only in the matrix.

- [ ] **Step 4: Audit data preservation and privacy fixtures**

Run the app against a copy of a pre-redesign SQLite/knowledge fixture; open meetings/chats/profile/connectors/notes without transforming Markdown/transcript/suggestion/chat/graph data. Exercise encrypted replay, conflict, memory-only, corrupt quarantine, and deletion. Inspect ciphertext and default diagnostics/telemetry exports for fixture secrets.

Expected: all durable objects open; fixture secrets occur in none of the on-disk recovery ciphertext text, telemetry, or default diagnostic export.

- [ ] **Step 5: Write and review the release record**

Record the exact outputs and link the completed `app/e2e/manual-platform-matrix.md`. Include every functional acceptance item 1–23 from design §17.3 as a checked line mapped to an automated or manual proof. Every entry must contain measured evidence or a release-blocking issue.

- [ ] **Step 6: Commit release evidence**

```bash
cd "$(git rev-parse --show-toplevel)"
git add docs/verification/quiet-canvas-release.md app/e2e/manual-platform-matrix.md
git commit -m "docs(verification): record quiet canvas release gates"
```

## Spec Requirement → Task and Proof Matrix

| Finalized design requirement | Implementing task(s) | Required release proof |
|---|---:|---|
| Product thesis, six object types, no persistent tabs/sidebar/footer/cockpit | 12, 13, 20–25, 29 | Shell/route E2E; cutover repository scan |
| Typed routes, history, exact scene-state/focus restoration | 12 | Router unit tests; cited-source Back E2E |
| 48 px title bar; text Open; responsive Ask; exception-only status; live safety control | 12, 16 | Responsive shell and capture E2E |
| Fixed Places in Open, breadcrumb, native menus before catalog readiness | 6, 12 | Menu parity and degraded-catalog tests |
| Exact shortcut table; editable-target guards | 6, 12, 25 | Native snapshot + keyboard E2E |
| Transient comparison tray/cap and contract-honest two/three-item scene | 21 | Payload table tests; responsive comparison E2E |
| Desk and meetings collection states/keyboard/delete semantics | 13 | Component/axe and Desk/list E2E |
| Meeting editorial geometry, BlockNote restrictions, always-editable states | 15 | Editor component tests; meeting E2E |
| One canonical note writer, revisions/idempotency, encrypted recovery, conflicts | 2, 7, 14 | Core atomic tests; queue/recovery E2E |
| AI proposals only; accept/reject/undo/provenance; truthful backend wording | 4, 14, 15, 18 | Proposal serialization + meeting Ask/review E2E |
| Capture full/partial/reconnect/state-reset/stop/resume and no false Listening | 5, 16 | WS race tests; capture E2E/platform matrix |
| Context margin bounds/abstention/tentative evidence | 17, 19 | Context/evidence fixtures and axe |
| Transcript layout, stable live region, follow/search/copy/errors | 17 | Long-list/component and meeting E2E |
| Durable final proposals/actionables; final-review failure has no retry | 3, 15 | Restart/failure E2E + repository scan |
| Contextual/global Ask; one chat owner; stream/reconnect behavior | 18, 20, 29 | Hub WS tests; Ask/conversation E2E; owner scan |
| Exact Cited/Consulted/Retrieved/fallback/prior/interrupted/temporal/stripped language | 9, 19, 20 | Exhaustive fixtures + forbidden-copy scan |
| Knowledge note current-generation/missing/native-open/Why this appeared | 21 | Knowledge-note generation-swap E2E |
| Connections complete List, exact current APIs, no invalid edge-reason call | 22 | Network assertion + keyboard parity E2E |
| Stable territory-first lazy Visual, LOD, reduced motion, DOM parity | 23 | Layout unit tests, manifest, visual E2E |
| Sources semantics, sync/config/delete, Advanced authorization only | 24 | Source mutation/copy E2E; no fake OAuth scan |
| Personal context, truthful privacy/capture, appearance, shortcuts, diagnostics | 25 | Settings states/axe; redaction tests |
| Main window bounds, hidden lifecycle, tray/call/suggestion peripherals | 6, 8, 27 | Electron unit/E2E and platform matrix |
| Deterministic main-owned curtain across all windows/accessibility tree | 8 | Epoch/crash/late-window tests; curtain E2E |
| Readiness/degraded matrix and last-truthful-content behavior | 9, 10, 13–25 | Gateway/store fault fixtures; full degraded E2E |
| Exact visual tokens/type/geometry/motion/accessibility/reflow | 11, 12, 15, 17, 21–25, 28 | Primitive tests; axe/keyboard/zoom/forced-color/manual AT |
| Content-free local telemetry and explicit diagnostic export | 26 | Runtime allowlist/rotation/redaction tests |
| Startup/lazy chunk/bundle/performance budgets | 1, 23, 25, 28–30 | Manifest budget CI + 20-launch trace gate |
| Existing durable data preserved; additive DDL only; OAuth deferred | 2, 3, 24, 29, 30 | Old-data fixture audit; endpoint/flag scan |
| No compatibility UI/gradual rollout; complete legacy deletion | 29 | Sole-entry and forbidden-path scan |
| Full automated/core/accessibility/platform/release acceptance | 28, 30 | Green release record with exact counts/results |
