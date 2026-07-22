# Knowledge Atlas Visual Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Knowledge Atlas into the approved light cartographic research workspace with a real projected graph, contextual floating tools, selected-object dossiers, and evidence-led Inquiry.

**Architecture:** Preserve the existing Redux/service boundary but hydrate the complete graph projection instead of discarding edges. Replace the four-column shell with a command bar and one full workbench. Atlas field, Explore panel, dossier, time control, and Inquiry evidence registry remain independently testable presentation units.

**Tech Stack:** Electron 39, React 19, Redux Toolkit, TypeScript 5.9, `lucide-react`, SVG/canvas graph rendering, existing deterministic layout functions, CSS custom properties, Vitest, Testing Library, Playwright.

## Global Constraints

- Implement the approved light cartographic direction in `docs/superpowers/specs/2026-07-22-two-track-ui-overhaul-design.md`.
- Do not implement the rejected dark immersive Atlas mockup.
- The map owns the workbench; there is no permanent empty inspector, index column, pin footer, or operation ledger.
- Preserve existing graph, evidence, meeting, chat, connector, profile, privacy, desktop, and runtime contracts.
- Use existing `lucide-react`; add no UI framework or new runtime dependency.
- Stable node geography, no ambient drift, semantic zoom, and accessible outline parity are mandatory.
- No feature flag, compatibility theme, data migration, or legacy shell in the DOM.

---

### Task 1: Preserve the complete projected graph in renderer state

**Files:**
- Modify: `app/src/renderer/src/features/atlas/contracts.ts`
- Modify: `app/src/renderer/src/features/atlas/state.ts`
- Modify: `app/src/renderer/src/app/hydration.ts`
- Modify: `app/src/renderer/src/app/hydration.test.tsx`

**Interfaces:**
- `AtlasEdge { source: string; target: string; weight: number; kind: string; label?: string }`
- `AtlasGraph { nodes: AtlasNode[]; edges: AtlasEdge[]; communities: Record<string, unknown> }`
- `parseAtlasGraph(value: unknown): AtlasGraph`
- Redux state exposes `state.atlas.nodes`, `state.atlas.edges`, and `state.atlas.positions`.

- [ ] **Step 1: Write graph parsing and state tests**

```ts
const graph = parseAtlasGraph({
  nodes: [{ id: 'a', title: 'Alpha', community: 'Research', degree: 2, source: 'notes' }],
  edges: [{ source: 'a', target: 'b', weight: 0.84, kind: 'semantic' }]
})
expect(graph.nodes).toHaveLength(1)
expect(graph.edges).toEqual([
  { source: 'a', target: 'b', weight: 0.84, kind: 'semantic', label: undefined }
])
```

Also assert malformed edges and edges referencing missing nodes are removed.

- [ ] **Step 2: Run hydration tests and confirm RED**

Run: `cd app && npx vitest run src/renderer/src/app/hydration.test.tsx`

Expected: FAIL because edges are not preserved and `parseAtlasGraph` is absent.

- [ ] **Step 3: Implement typed parsing and hydration**

Add `AtlasEdge` and `AtlasGraph`. Parse API edge endpoint keys defensively (`source`/`src`, `target`/`dst`) and normalize weight/kind/label. Dispatch one `graphHydrated(AtlasGraph)` action. Preserve existing node compatibility for tests that inject `nodes` directly.

- [ ] **Step 4: Run focused and contract tests**

Run: `cd app && npx vitest run src/renderer/src/app/hydration.test.tsx src/renderer/src/services/sidecar`

Expected: PASS.

- [ ] **Step 5: Commit graph hydration**

```bash
git add app/src/renderer/src/features/atlas app/src/renderer/src/app/hydration.ts app/src/renderer/src/app/hydration.test.tsx
git commit -m "feat(atlas): preserve projected graph topology"
```

### Task 2: Replace pane grid with cartographic shell

**Files:**
- Create: `app/src/renderer/src/features/workspace/AtlasCommandBar.tsx`
- Create: `app/src/renderer/src/features/workspace/AtlasCommandBar.test.tsx`
- Modify: `app/src/renderer/src/app/AppShell.tsx`
- Delete: `app/src/renderer/src/features/workspace/CommandDeck.tsx`
- Delete: `app/src/renderer/src/features/workspace/CoordinateRail.tsx`
- Modify: `app/src/renderer/src/features/runtime/OperationLedger.tsx`
- Modify: `app/src/renderer/src/styles/tokens.css`
- Modify: `app/src/renderer/src/styles/typography.css`
- Modify: `app/src/renderer/src/styles/shell.css`
- Test: `app/src/renderer/src/app/shell-journey.test.tsx`

**Interfaces:**
- `AtlasCommandBar({ coordinate, status, onCoordinate, onOpenCommand, onAsk, onOpenTasks })` owns presentation only.
- `AppShell` keeps its current public props and command execution behavior.

- [ ] **Step 1: Write shell semantic tests**

```tsx
expect(screen.getByRole('navigation', { name: 'Knowledge coordinates' })).toBeVisible()
expect(screen.getByRole('button', { name: 'Atlas' })).toHaveAttribute('aria-current', 'page')
expect(screen.queryByTestId('index-region')).not.toBeInTheDocument()
expect(screen.queryByTestId('inspector-region')).not.toBeInTheDocument()
expect(screen.queryByText('Pinned objects appear here.')).not.toBeInTheDocument()
```

- [ ] **Step 2: Run shell tests and confirm RED**

Run: `cd app && npx vitest run src/renderer/src/app/shell-journey.test.tsx src/renderer/src/features/workspace/AtlasCommandBar.test.tsx`

- [ ] **Step 3: Implement the command bar**

Use brand, compact Atlas/Chronicle/Inquiry segmented navigation, a command locator, corpus/runtime status, task popover, and Ask. Preserve shortcuts and focus zones. Move Control to command/menu access.

- [ ] **Step 4: Replace shell composition and tokens**

Compose `.atlas-shell` as `grid-template-rows: 50px minmax(0,1fr)` with a single `.atlas-workbench`. Remove permanent index, inspector, shelf, and operation rows. Set the approved light cartographic `--ka-*` tokens, IBM Plex typography, 8–10 px contextual surface radii, and restrained elevation.

- [ ] **Step 5: Verify shell and accessibility tests**

Run: `cd app && npx vitest run src/renderer/src/app src/renderer/src/features/workspace`

Run: `cd app && npm run typecheck`

Expected: PASS.

- [ ] **Step 6: Commit shell cutover**

```bash
git add app/src/renderer/src/app app/src/renderer/src/features/workspace app/src/renderer/src/features/runtime app/src/renderer/src/styles
git commit -m "feat(atlas): introduce cartographic workbench shell"
```

### Task 3: Build the real Atlas field and contextual tools

**Files:**
- Create: `app/src/renderer/src/features/atlas/AtlasField.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasField.test.tsx`
- Create: `app/src/renderer/src/features/atlas/ExplorePanel.tsx`
- Create: `app/src/renderer/src/features/atlas/ObjectDossier.tsx`
- Create: `app/src/renderer/src/features/atlas/AtlasTimeControl.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasWorkbench.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasOutline.tsx`
- Modify: `app/src/renderer/src/features/atlas/AtlasControls.tsx`
- Modify: `app/src/renderer/src/features/atlas/layout.ts`
- Modify: `app/src/renderer/src/features/atlas/layout.test.ts`
- Modify: `app/src/renderer/src/styles/shell.css`
- Test: `app/src/renderer/src/features/atlas/knowledge-operations.test.tsx`

**Interfaces:**
- `AtlasField({ graph, positions, selectedId, filters, onSelect, onOpen, onLayout })`
- `ExplorePanel({ graph, filters, onFilters, collapsed, onCollapsed })`
- `ObjectDossier({ node, edges, onOpen, onPin, onAsk, onClose })`
- `AtlasTimeControl({ range, bounds, onRange })`

- [ ] **Step 1: Write field rendering tests**

```tsx
render(<AtlasField graph={graph} positions={positions} selectedId="a" onSelect={select} onOpen={open} />)
expect(screen.getByRole('img', { name: /knowledge atlas/i })).toBeVisible()
expect(screen.getByRole('button', { name: /alpha/i })).toHaveAttribute('aria-pressed', 'true')
expect(screen.getByTestId('atlas-edge-a-b')).toBeVisible()
await user.click(screen.getByRole('button', { name: /beta/i }))
expect(select).toHaveBeenCalledWith('b')
```

Add stable-layout, semantic-label, filtered-node, empty-graph, and outline-parity cases.

- [ ] **Step 2: Run Atlas tests and confirm RED**

Run: `cd app && npx vitest run src/renderer/src/features/atlas`

- [ ] **Step 3: Implement field rendering**

Render graph edges and nodes from the typed projection using an SVG viewBox over stable positions. Use weight for opacity/width within bounded ranges, degree for bounded node size, territory for labeled contour/group treatment, and selection for the strongest route. Cull labels by zoom and priority. Do not animate unchanged positions.

- [ ] **Step 4: Implement contextual tools**

Move search, territory, layer, and time state out of the old index. Render Explore over the field, render Object Dossier only for a selection, and render Ask with its explicit scope. Retain Outline as a synchronized accessible alternative rather than rendering a fake canvas inside it.

- [ ] **Step 5: Verify Atlas behavior**

Run: `cd app && npx vitest run src/renderer/src/features/atlas src/renderer/src/app`

Run: `cd app && npm run typecheck`

Expected: PASS with stable graph positions and no empty permanent regions.

- [ ] **Step 6: Commit the map**

```bash
git add app/src/renderer/src/features/atlas app/src/renderer/src/styles/shell.css
git commit -m "feat(atlas): render the projected knowledge field"
```

### Task 4: Evidence-led Inquiry, Chronicle, and Control

**Files:**
- Modify: `app/src/renderer/src/features/inquiry/InquiryWorkbench.tsx`
- Modify: `app/src/renderer/src/features/evidence/EvidenceInspector.tsx`
- Modify: `app/src/renderer/src/features/evidence/EvidenceLedger.tsx`
- Modify: `app/src/renderer/src/features/meetings/ChronicleWorkbench.tsx`
- Modify: `app/src/renderer/src/features/workspace/ControlWorkbench.tsx`
- Modify: `app/src/renderer/src/styles/shell.css`
- Test: existing inquiry, evidence, chronicle, and knowledge-operation tests

**Interfaces:** Existing Inquiry, Chronicle, Evidence, and Control service props remain unchanged.

- [ ] **Step 1: Add evidence-workbench semantic assertions**

Assert Inquiry names scope/corpus size, answers and evidence are separate labeled regions, evidence statuses remain exact, citation selection opens the corresponding record, and compact layout moves evidence into a sheet.

- [ ] **Step 2: Run feature tests and confirm RED for the new composition**

Run: `cd app && npx vitest run src/renderer/src/features/inquiry src/renderer/src/features/evidence src/renderer/src/features/meetings`

- [ ] **Step 3: Implement Inquiry and evidence composition**

Build the approved two-column answer/evidence workbench. Use numbered inline citations, status-labeled evidence cards, and a single high-contrast Evidence Route card. Keep streaming, stop, regenerate, scope, proposal, and evidence normalization behavior unchanged.

- [ ] **Step 4: Restyle Chronicle and Control**

Give Chronicle a temporal rail/list and selected meeting dossier within the one-workbench shell. Give Control a conventional local settings index and content surface. Neither route may recreate the old four-pane grid.

- [ ] **Step 5: Verify all renderer tests**

Run: `cd app && npm test -- --run`

Expected: PASS.

- [ ] **Step 6: Commit secondary workbenches**

```bash
git add app/src/renderer/src/features app/src/renderer/src/styles/shell.css
git commit -m "feat(atlas): finish evidence research workbenches"
```

### Task 5: Atlas cutover and proof

**Files:**
- Modify: `app/e2e/knowledge-atlas.spec.ts`
- Modify: `app/src/renderer/src/app/accessibility.test.tsx`
- Modify: `app/scripts/check-old-ui-removed.mjs`
- Modify: screenshot expectations under the existing visual E2E mechanism

**Interfaces:** Production composition remains `App -> AppShell`; no alternate renderer or feature flag is introduced.

- [ ] **Step 1: Update E2E journeys**

Cover populated Atlas field, Explore search/territory/layers, node selection, dossier, outline parity, time control, Ask scope, Inquiry citations/evidence route, Chronicle, Control, privacy curtain, compact sheets, dark mode, reduced motion, and 200% zoom.

- [ ] **Step 2: Run full automated verification**

Run: `cd app && npm test -- --run`

Run: `cd app && npm run typecheck`

Run: `cd app && npm run lint`

Run: `cd app && npm run build`

Run: `cd app && npm run test:e2e`

Run: `cd app && npm run test:visual`

Run: `cd core && uv run pytest -q`

Expected: all commands exit zero.

- [ ] **Step 3: Run populated manual verification**

Launch `./run.sh`; confirm 724 nodes and 2,224 projected graph edges render, then exercise Atlas, Outline, Chronicle, Inquiry, evidence, comparison, Control, meeting capture, privacy, compact width, dark mode, and 200% zoom. Confirm the UI matches the approved light design and not the rejected dark immersive design.

- [ ] **Step 4: Commit proof and final fixes**

```bash
git add app/e2e app/scripts app/src
git commit -m "test(atlas): verify cartographic workspace cutover"
```
