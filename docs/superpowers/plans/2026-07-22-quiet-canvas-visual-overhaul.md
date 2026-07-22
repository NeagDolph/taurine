# Quiet Canvas Visual Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Quiet Canvas into the approved warm living-notebook UI with a 49 px rail, composed Today screen, document-centered meetings, and quiet contextual AI while preserving every existing workflow.

**Architecture:** Keep the existing `WorkspaceRouter`, repositories, projections, `MeetingRuntime`, and `ConversationHub`. Replace only renderer composition: `WorkspaceShell` owns a slim destination rail and one scene host; route-local headers own breadcrumbs and tabs; scene components consume the existing typed data and expose the same callbacks.

**Tech Stack:** Electron 39, React 19, TypeScript 5.9, `lucide-react`, CSS custom properties, Vitest, Testing Library, Playwright.

## Global Constraints

- Implement the approved Quiet Canvas direction in `docs/superpowers/specs/2026-07-22-two-track-ui-overhaul-design.md`.
- Use a 49 px global rail containing only Today, Meetings, Ask, Knowledge, and Open Anything.
- Keep Settings and secondary destinations inside Open Anything and the native menu.
- Preserve all backend, persistence, meeting, capture, chat, evidence, graph, connector, profile, privacy, and diagnostics contracts.
- Use existing `lucide-react`; add no UI framework or new runtime dependency.
- No feature flag, compatibility theme, data migration, or legacy shell in the DOM.
- Verify with the populated worktree corpus and deterministic fixtures.

---

### Task 1: Quiet design foundation and destination rail

**Files:**
- Create: `app/src/renderer/src/workspace/QuietRail.tsx`
- Create: `app/src/renderer/src/workspace/QuietRail.test.tsx`
- Create: `app/src/renderer/src/workspace/SceneHeader.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceTitlebar.tsx`
- Modify: `app/src/renderer/src/design/tokens.css`
- Modify: `app/src/renderer/src/design/typography.css`
- Modify: `app/src/renderer/src/design/globals.css`
- Test: `app/src/renderer/src/workspace/WorkspaceShell.test.tsx`

**Interfaces:**
- `QuietRail({ route, onNavigate, onOpen }: QuietRailProps)` consumes the current route and existing router callbacks.
- `SceneHeader({ back, title, meta, tabs, actions }: SceneHeaderProps)` provides route-local navigation without owning data.
- `WorkspaceShell` remains the sole composition owner and retains its public props.

- [ ] **Step 1: Write the rail behavior test**

```tsx
render(<QuietRail route={{ kind: 'desk' }} onNavigate={navigate} onOpen={open} />)
expect(screen.getByRole('navigation', { name: 'Taurine' })).toBeVisible()
expect(screen.getAllByRole('button')).toHaveLength(5)
await user.click(screen.getByRole('button', { name: 'Meetings' }))
expect(navigate).toHaveBeenCalledWith({ kind: 'meetings', query: '' })
expect(screen.queryByText('Settings')).not.toBeInTheDocument()
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `cd app && npx vitest run src/renderer/src/workspace/QuietRail.test.tsx`

Expected: FAIL because `QuietRail.tsx` does not exist.

- [ ] **Step 3: Implement the rail and route-local header**

```tsx
export interface QuietRailProps {
  route: WorkspaceRoute
  onNavigate(route: WorkspaceRoute): void
  onOpen(): void
}

const destinations = [
  { label: 'Today', icon: Home, route: { kind: 'desk' } },
  { label: 'Meetings', icon: CalendarDays, route: { kind: 'meetings', query: '' } },
  { label: 'Ask Taurine', icon: MessageCircle, route: { kind: 'conversation' } },
  { label: 'Knowledge', icon: Library, route: { kind: 'connections', scope: { kind: 'all' } } }
] as const
```

Render 18 px Lucide icons in 36 px buttons inside a 49 px rail. Use `aria-current="page"`, native `title`, and an accessible tooltip. Move back/forward, local title, status, and route actions into `SceneHeader`; remove large `Open` and `Ask` text controls from permanent top chrome.

- [ ] **Step 4: Replace shell layout and tokens**

Set the approved `--qc-*` tokens, warm dark equivalents, 4 px spacing base, 8–12 px radii, and Source Serif document typography. Compose `.workspace-shell` as `grid-template-columns: 49px minmax(0, 1fr)` with one `.workspace-scene-host`. Ensure the rail drag exclusion and title-region drag behavior remain correct.

- [ ] **Step 5: Run focused tests and typecheck**

Run: `cd app && npx vitest run src/renderer/src/workspace/QuietRail.test.tsx src/renderer/src/workspace/WorkspaceShell.test.tsx`

Run: `cd app && npm run typecheck`

Expected: PASS.

- [ ] **Step 6: Commit the foundation**

```bash
git add app/src/renderer/src/workspace app/src/renderer/src/design
git commit -m "feat(quiet): introduce living notebook shell"
```

### Task 2: Composed Today and meetings collection

**Files:**
- Create: `app/src/renderer/src/meetings/TodayScene.tsx`
- Create: `app/src/renderer/src/meetings/TodayScene.test.tsx`
- Modify: `app/src/renderer/src/workspace/WorkspaceShell.tsx`
- Modify: `app/src/renderer/src/meetings/MeetingsCollectionScene.tsx`
- Modify: `app/src/renderer/src/design/globals.css`
- Delete: `app/src/renderer/src/meetings/DeskScene.tsx`

**Interfaces:**
- `TodayScene` consumes `MeetingListItem[]`, `onStart`, `onOpen`, `onOpenAll`, `onAsk`, and `onConnectSource`.
- `featuredMeeting()` chooses a live meeting first, then the newest non-done draft, then the newest completed meeting. It never labels a meeting as upcoming because `MeetingListItem` has no scheduled-time contract.
- `meetingPreview()` uses a non-empty `extensions.final_summary_md` only when the server supplies it; otherwise it renders the truthful meeting state and date.

- [ ] **Step 1: Write populated and empty Today tests**

```tsx
render(<TodayScene meetings={meetings} onStart={start} onOpen={open} onOpenAll={all} onAsk={ask} onConnectSource={sources} />)
expect(screen.getByRole('heading', { name: /good (morning|afternoon|evening)/i })).toBeVisible()
expect(screen.getByText('Recommendation engine & voice of customer')).toBeVisible()
expect(screen.getByRole('button', { name: /ask across your notes/i })).toBeVisible()

render(<TodayScene meetings={[]} onStart={start} onOpen={open} onOpenAll={all} onAsk={ask} onConnectSource={sources} />)
expect(screen.getByRole('button', { name: 'Start a new note' })).toBeVisible()
expect(screen.getByRole('button', { name: 'Connect a source' })).toBeVisible()
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `cd app && npx vitest run src/renderer/src/meetings/TodayScene.test.tsx`

Expected: FAIL because `TodayScene` does not exist.

- [ ] **Step 3: Implement Today**

Create a semantic scene with greeting/date, live-or-featured meeting card, eight recent rows, composed empty state, and Ask composer button. Use real `MeetingListItem` fields and optional extension values only. Route Connect a source to `{ kind: 'settings', page: 'sources' }` through a callback supplied by `WorkspaceShell`.

- [ ] **Step 4: Restyle the collection**

Group meetings with pure `meetingDateGroup(updatedAt): 'Today' | 'This week' | 'Earlier'`. Keep the existing search and inline delete behavior. Add a sticky scene header and stable row actions that appear on `:hover` and `:focus-within` without layout shift.

- [ ] **Step 5: Remove `DeskScene` and verify**

Run: `cd app && npx vitest run src/renderer/src/meetings/TodayScene.test.tsx src/renderer/src/workspace/WorkspaceShell.test.tsx src/renderer/src/meetings`

Expected: PASS and no import of `DeskScene` remains.

- [ ] **Step 6: Commit Today and meetings**

```bash
git add app/src/renderer/src/meetings app/src/renderer/src/workspace/WorkspaceShell.tsx app/src/renderer/src/design/globals.css
git commit -m "feat(quiet): compose today and meeting library"
```

### Task 3: Document-centered meeting scene

**Files:**
- Create: `app/src/renderer/src/meetings/MeetingContextMargin.tsx`
- Create: `app/src/renderer/src/meetings/MeetingContextMargin.test.tsx`
- Modify: `app/src/renderer/src/meetings/MeetingScene.tsx`
- Modify: `app/src/renderer/src/meetings/MeetingDocument.tsx`
- Modify: `app/src/renderer/src/meetings/TranscriptDrawer.tsx`
- Modify: `app/src/renderer/src/meetings/ReviewChanges.tsx`
- Modify: `app/src/renderer/src/design/globals.css`
- Test: `app/src/renderer/src/meetings/__tests__/MeetingScene.test.tsx`

**Interfaces:**
- `MeetingContextMargin({ grounding, actionables, suggestions, onOpenEvidence })` renders at most three contextual items.
- Existing `MeetingRuntimeSnapshot`, `MeetingDocumentController`, proposal, transcript, and capture callbacks remain unchanged.

- [ ] **Step 1: Write composition tests**

```tsx
expect(screen.getByRole('tab', { name: 'Notes' })).toHaveAttribute('aria-selected', 'true')
expect(screen.getByRole('tab', { name: 'Transcript' })).toBeVisible()
expect(screen.getByRole('status', { name: /capture/i })).toHaveTextContent('Listening')
expect(screen.getByRole('complementary', { name: 'Live context' })).toBeVisible()
```

Add a compact-width case that expects a `Live context` button instead of a permanently visible complementary region.

- [ ] **Step 2: Run the test and confirm RED**

Run: `cd app && npx vitest run src/renderer/src/meetings/__tests__/MeetingScene.test.tsx src/renderer/src/meetings/MeetingContextMargin.test.tsx`

- [ ] **Step 3: Implement local tabs and context margin**

Keep Notes mounted as the default document. Transcript, Review, and Details use local tab state and the existing drawer/review content. Render the context margin only when it contains real grounding, actionables, or suggestions; cap it to three items ordered by actionability. Generated items retain source/provenance labels.

- [ ] **Step 4: Apply the approved document composition**

Use a 620–760 px editorial note measure, a 220–264 px optional context margin, a quiet sticky meeting Ask composer, and local save/capture state. Remove the old flex header/body styling that gives every region equal weight.

- [ ] **Step 5: Verify meeting behavior**

Run: `cd app && npx vitest run src/renderer/src/meetings`

Run: `cd app && npm run typecheck`

Expected: all existing capture, controller, transcript, proposal, and meeting journey tests pass.

- [ ] **Step 6: Commit meeting composition**

```bash
git add app/src/renderer/src/meetings app/src/renderer/src/design/globals.css
git commit -m "feat(quiet): make meetings authored documents"
```

### Task 4: Ask, Knowledge, and Settings polish

**Files:**
- Modify: `app/src/renderer/src/ask/ConversationScene.tsx`
- Modify: `app/src/renderer/src/ask/ConversationTurn.tsx`
- Modify: `app/src/renderer/src/ask/EvidenceDisclosure.tsx`
- Modify: `app/src/renderer/src/ask/EvidenceLedger.tsx`
- Modify: `app/src/renderer/src/knowledge/ConnectionsScene.tsx`
- Modify: `app/src/renderer/src/knowledge/KnowledgeNoteScene.tsx`
- Modify: `app/src/renderer/src/settings/SettingsIndexScene.tsx`
- Modify: `app/src/renderer/src/settings/SourcesScene.tsx`
- Modify: `app/src/renderer/src/settings/PersonalContextScene.tsx`
- Modify: `app/src/renderer/src/settings/PrivacyCaptureScene.tsx`
- Modify: `app/src/renderer/src/design/globals.css`
- Test: existing Ask, Knowledge, and Settings tests

**Interfaces:** No data interfaces change. Route-aware scope and evidence semantics remain owned by existing services and components.

- [ ] **Step 1: Add semantic layout assertions**

Assert Ask exposes its scope before the composer, evidence opens in a named complementary region/sheet, Knowledge defaults to the list/document view, and Settings retains all current routes without adding rail items.

- [ ] **Step 2: Run Ask/Knowledge/Settings tests and confirm RED where new composition is absent**

Run: `cd app && npx vitest run src/renderer/src/ask src/renderer/src/knowledge src/renderer/src/settings`

- [ ] **Step 3: Implement the reading-column Ask layout**

Use a centered thread, tinted user prompts, editorial answer typography, sticky scoped composer, numbered citations, and evidence side sheet. Keep stop/regenerate/proposal behavior and current evidence status language.

- [ ] **Step 4: Implement list-first Knowledge and unified Settings surfaces**

Make the note/connection list the default Knowledge surface and retain visual graph mode as the secondary toggle. Apply one Settings pattern: scene intro, section heading, quiet rows, field help, status, and isolated destructive zone.

- [ ] **Step 5: Verify all feature tests**

Run: `cd app && npm run test:unit`

Expected: PASS.

- [ ] **Step 6: Commit secondary scenes**

```bash
git add app/src/renderer/src/ask app/src/renderer/src/knowledge app/src/renderer/src/settings app/src/renderer/src/design/globals.css
git commit -m "feat(quiet): finish notebook knowledge surfaces"
```

### Task 5: Quiet Canvas cutover and proof

**Files:**
- Modify: `app/e2e/quiet-canvas.spec.ts`
- Modify: `app/e2e/accessibility.spec.ts`
- Modify: `app/e2e/performance.spec.ts`
- Modify: `app/scripts/verify-quiet-canvas.mjs`
- Modify: screenshot expectations under the existing E2E output mechanism

**Interfaces:** Production composition remains `RendererBootstrap -> WorkspaceShell`; no alternate renderer or feature flag is introduced.

- [ ] **Step 1: Update E2E assertions for the new shell**

Cover Today with populated meetings, rail navigation, live meeting document, transcript/review tabs, Ask evidence, Knowledge list/visual toggle, Settings via Open Anything, privacy curtain, narrow layout, 200% zoom, and reduced motion.

- [ ] **Step 2: Run full automated verification**

Run: `cd app && npm run test:all`

Run: `cd app && npm run typecheck`

Run: `cd app && npm run lint`

Run: `cd app && npm run build`

Run: `cd app && npm run test:e2e`

Run: `cd app && npm run test:e2e:a11y`

Run: `cd core && uv run pytest -q`

Expected: all commands exit zero.

- [ ] **Step 3: Run populated manual verification**

Launch `./run.sh`, confirm 724 notes, 9 meetings, 25 chats, and the projected graph are visible, then exercise every route and capture screenshots at 1120×780, compact width, dark mode, and 200% zoom.

- [ ] **Step 4: Commit proof and final fixes**

```bash
git add app/e2e app/scripts app/src
git commit -m "test(quiet): verify living notebook cutover"
```
