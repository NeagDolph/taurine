# Taurine Two-Track UI Overhaul Design

**Date:** 2026-07-22
**Status:** Approved visual direction; implementation-grade design
**Applies to:** `codex/ui-quiet-canvas` and `codex/ui-knowledge-atlas`
**Product reference:** Granola's calm, document-centered desktop experience, adapted rather than copied

## 1. Decision

Taurine will retain two intentionally distinct desktop UI experiments.

- **Quiet Canvas** becomes a warm, personal **living notebook**. Its center of gravity is the user's authored meeting document. Navigation is a 49 px icon rail, the home screen foregrounds the next meeting and recent notes, and AI appears in a restrained contextual margin or composer rather than as a dashboard.
- **Knowledge Atlas** becomes a cool, spatial **research landscape**. Its center of gravity is the projected knowledge graph. The map owns the canvas; search, territory filters, selected-object details, time controls, and evidence paths appear as contextual floating surfaces rather than permanent bordered columns.

The two branches share reliability, accessibility, data contracts, iconography quality, and interaction polish. They do not share a shell layout, visual palette, navigation metaphor, density, or page composition.

The approved Knowledge Atlas direction is the earlier light cartographic design. The later dark immersive experiment is explicitly rejected and must not be implemented.

## 2. Why the Current UI Fails

The current implementations expose application scaffolding rather than a finished product.

### Quiet Canvas

- The home screen is a heading, one button, a sentence, and large unused space.
- The title bar presents text commands without visual grouping or hierarchy.
- Navigation is discoverable only through Open Anything and history controls.
- Meeting and conversation screens have good semantic foundations but minimal composition, affordance, or personality.
- Empty states describe the product without helping the user begin or understand what existing data contains.

### Knowledge Atlas

- The shell is visibly divided into empty grid regions even when no region has useful content.
- The 52 px rail attempts to render full text labels, causing cramped navigation.
- Empty inspector and pinned-object regions remain permanently visible.
- The graph canvas looks like an unimplemented placeholder instead of the primary experience.
- Inquiry is rendered as a basic form inside another bordered pane.
- The interface has almost no elevation, spatial hierarchy, object preview, or polished transition behavior.

These are not token-level defects. The composition, hierarchy, and progressive disclosure model must change.

## 3. Goals and Non-Goals

### Goals

1. Make each branch feel like a deliberate, shippable desktop product on first launch.
2. Make the primary action and current context understandable in under five seconds.
3. Use the complete imported corpus so home, meeting, chat, and graph screens are evaluated with realistic density.
4. Preserve every existing workflow and backend contract while changing its presentation.
5. Remove permanent empty chrome and reveal secondary machinery only when it is relevant.
6. Establish coherent typography, spacing, icons, interaction states, empty states, and motion.
7. Keep every workflow keyboard-accessible and WCAG 2.2 AA compliant.
8. Maintain responsive behavior at current Electron window sizes and 200% zoom.

### Non-goals

- Changing graph generation, retrieval, chat, meeting capture, connector, or profile algorithms.
- Adding collaboration, cloud sync, calendar planning, or new product areas.
- Copying Granola's branding, proprietary assets, wording, or exact layouts.
- Converging the branches into one themeable application shell.
- Reintroducing the rejected dark immersive Atlas treatment.
- Migrating databases. The worktree stores are disposable snapshots and can be recopied from main.

## 4. Shared Product Principles

### 4.1 One dominant surface

Each route has one obvious working surface. Quiet Canvas prioritizes the authored document. Knowledge Atlas prioritizes the map or the evidence-backed answer. Secondary controls never compete at equal visual weight.

### 4.2 Progressive disclosure

- Empty inspectors are not rendered.
- Detailed retrieval traces are behind Evidence or Process disclosures.
- Destructive and administrative controls live in menus or Settings.
- Transcript, review, and meeting details are document-level tabs, not separate global destinations.
- Filters remain visible only while the user is actively exploring the graph.

### 4.3 Authored content outranks generated content

User notes use the primary ink color and document typography. Generated summaries, suggestions, and contextual insights use a distinct muted surface and retain their provenance labels. AI suggestions never visually masquerade as authored text.

### 4.4 State is visible but quiet

Healthy state stays subdued. Recording, reconnection, unsaved edits, conflicts, weak evidence, sync failure, and unavailable sources become prominent only when they can affect the user's next action.

### 4.5 Native desktop behavior

The top drag region, traffic-light clearance, menus, keyboard shortcuts, privacy curtain, notification surfaces, back/forward history, and hidden-window behavior remain functional. Custom chrome must not block window dragging or keyboard focus.

### 4.6 Icons and language

Use the existing `lucide-react` dependency at 16–18 px with 1.75 px strokes. Every icon-only control has a tooltip and accessible name. No emoji controls, decorative sparkles, or unexplained symbols are permitted.

## 5. Quiet Canvas — Living Notebook

### 5.1 Visual character

Quiet Canvas is warm, tactile, and editorial without simulating physical paper. It should feel private and approachable rather than technical.

#### Core tokens

| Token | Light value | Purpose |
|---|---:|---|
| `--qc-canvas` | `#F6F2E9` | Main notebook background |
| `--qc-rail` | `#EEE9DF` | Navigation rail |
| `--qc-sheet` | `#FFFDF8` | Elevated note and composer surfaces |
| `--qc-ink` | `#2D2A25` | Primary content |
| `--qc-muted` | `#7D7569` | Metadata and helper copy |
| `--qc-line` | `#DED7CA` | Quiet dividers |
| `--qc-accent` | `#3E4B3C` | Primary actions and active navigation |
| `--qc-accent-soft` | `#D9D2C5` | Selected rail items |
| `--qc-generated` | `#ECE7DC` | Generated context surfaces |
| `--qc-recording` | `#B85A4C` | Capture state only |

Dark mode uses warm charcoal rather than neutral black. It retains the same hierarchy and contrast ratios.

Use Source Serif 4 for document titles and long-form note content. Use the platform UI font stack for navigation, controls, metadata, and status.

Spacing follows a 4 px base. Major scene gutters are 32–40 px, document rhythm is 20–28 px, surface radii are 8–12 px, and shadows are limited to floating composers, menus, dialogs, and the next-meeting card.

### 5.2 Shell

The shell has three regions:

1. A native 30–48 px drag/title region with traffic-light clearance.
2. A persistent **49 px icon rail**.
3. One scrollable scene.

The rail contains only:

- Today (`Home`)
- Meetings (`NotebookTabs` or `CalendarDays`)
- Ask Taurine (`Search` or `MessageCircle`)
- Knowledge (`Library`)
- Open Anything (`Ellipsis`) at the bottom

The active item uses `--qc-accent-soft`. Labels appear on hover and keyboard focus in a dark tooltip. Settings, Sources, Personal Context, Privacy, Appearance, Shortcuts, and Diagnostics are reached through Open Anything or the native application menu. They do not expand the rail.

Back and forward move into the scene header only when history exists. They are icon buttons, not permanent large text commands.

### 5.3 Today

Today replaces the current bare Desk.

- Header: contextual greeting, date, and one-line product orientation.
- Next meeting: a single elevated card with time, title, capture configuration, and `Open note` or `Start meeting`.
- Active meeting: replaces the next-meeting card and clearly shows `Listening`, elapsed time, channel state, and `Return to note`.
- Recent notes: five to eight compact rows with title, one-line summary/gist, source or participant context, and relative date.
- Empty state: a composed first-note invitation with `Start a new note`, `Connect a source`, and `Ask what Taurine can do`; never only prose.
- Bottom composer: `Ask across your notes…`, available from Today without turning the page into a chat dashboard.

### 5.4 Meetings collection

The collection uses a calm list, not cards for every meeting.

- Sticky header with title, search, and `New meeting`.
- Sections: Today, This week, Earlier.
- Rows show title, participants/source context, date, duration, and state.
- Hover/focus reveals the row menu without shifting layout.
- Search results preserve date grouping where possible.
- Deletion remains a two-step inline confirmation with consequence copy.

### 5.5 Meeting document

The meeting is one durable document.

- Header: breadcrumb/back, editable title, participants/date, capture status, and overflow actions.
- Document tabs: Notes, Transcript, Review, Details. Tabs are local to the meeting header and never appear in the global rail.
- Notes: 620–760 px readable measure using editorial typography. The editor remains the visual center.
- Live Context: a 220–264 px optional right margin showing at most three timely items. It is open during capture on sufficiently wide windows and collapses into a button below the breakpoint.
- Generated material uses `--qc-generated`, a provenance label, and explicit accept/dismiss actions where applicable.
- Ask composer remains at the bottom and inherits the meeting scope automatically.
- Save state is terse: `Saving…`, `Saved`, or an actionable conflict/error. Healthy `Saved` fades after confirmation.

### 5.6 Ask

Ask is a reading and writing surface, not a chat bubble stack.

- A narrow centered thread with human-readable scope above the title.
- User prompts use a quiet tinted block aligned with the reading column.
- Answers use document typography with inline numbered citations.
- The composer is sticky and shows its scope before submission.
- Evidence opens as a side sheet on wide windows and a modal sheet on compact windows.
- Empty Ask shows three grounded starter prompts based on the available corpus, not generic AI suggestions.

### 5.7 Knowledge and Settings

Knowledge is list/document-first in Quiet Canvas. Connections visualization remains available as a secondary view, but the default is searchable notes grouped by source, project, or recent activity. Settings retain the current scene boundaries and data behaviors while adopting the same title, section, field, disclosure, and destructive-zone patterns.

## 6. Knowledge Atlas — Research Landscape

### 6.1 Visual character

Knowledge Atlas is cool, precise, and cartographic. It is denser than Quiet Canvas but avoids dashboard chrome.

#### Core tokens

| Token | Light value | Purpose |
|---|---:|---|
| `--ka-field` | `#E8EEEC` | Map field |
| `--ka-surface` | `#F7FAF8` | Floating tools and dossier |
| `--ka-ink` | `#19272E` | Primary content |
| `--ka-muted` | `#66777C` | Metadata |
| `--ka-line` | `#C6D1CF` | Dividers and quiet graph edges |
| `--ka-grid` | `rgba(42,71,80,.055)` | Cartographic reference grid |
| `--ka-accent` | `#3C7380` | Knowledge nodes and primary actions |
| `--ka-selection` | `#B36C44` | Selected object and active evidence route |
| `--ka-territory-a` | `#4F7F8A` | Knowledge systems |
| `--ka-territory-b` | `#9C744F` | Audio and meetings |
| `--ka-territory-c` | `#687FA1` | Product research |

The approved design is light-first. Dark mode may use a deep slate field, but it must preserve the same composition and must not adopt the rejected luminous immersive treatment as the default.

Use IBM Plex Sans for UI and dossier text and IBM Plex Mono sparingly for technical IDs, dates, and path metadata. Object titles may use the existing editorial face only inside dossiers and Inquiry answers.

### 6.2 Shell

The permanent shell is a 41–52 px top command bar above a full workbench.

- Brand at left.
- Atlas, Chronicle, and Inquiry as a compact segmented coordinate switch.
- Search/command locator centered when space permits.
- Corpus/build status and Ask at right.
- Control opens from the command locator or application menu; it is not a permanent coordinate.

There is no permanent left rail, empty inspector column, pinned-object footer, or always-visible operation ledger. Operations become toasts or a compact task popover in the command bar. Pinned context appears inside the active Ask scope or comparison tray only when pins exist.

### 6.3 Atlas field

The graph fills the workbench beneath the command bar.

- A subtle 25 px reference grid and contour rings establish geography without decorative noise.
- Communities render as labeled territories at wide zoom, clusters at medium zoom, and individual notes at close zoom.
- Stable node positions and restrained motion remain mandatory.
- Selected nodes use `--ka-selection`; the explanatory route to related evidence becomes the strongest line in the field.
- Labels appear by semantic zoom and collision priority, not for every node simultaneously.
- The accessible synchronized outline remains available from `View as outline` and is used automatically for reduced-motion or unsupported-canvas scenarios.

Contextual floating surfaces:

- **Explore panel:** search, visible territories, active layers, and result count. Default width 220–250 px; collapsible.
- **Object dossier:** selected source/date, title, gist, tags, dates, connection explanation, and open/pin/ask actions. It is absent until selection.
- **Time control:** compact floating scrubber across the lower canvas. It expands only while being manipulated.
- **Ask composer:** appears above the time control and names the active territory/object/pin scope.
- **Legend:** minimal category legend; hidden when color is not contributing meaningful distinctions.

### 6.4 Chronicle

Chronicle uses the same cool system but changes the workbench into a temporal surface.

- Left-aligned timeline with date groups and meeting density marks.
- Main dossier for the selected meeting.
- Live meetings gain a prominent capture strip, editable note, and optional context drawer.
- Timeline selection and Atlas selection remain synchronized.
- Returning to Atlas restores the previous camera, filters, selected object, and time range.

### 6.5 Inquiry

Inquiry is evidence-led and intentionally distinct from Quiet Canvas Ask.

- Two-column workbench: answer/thread on the left and evidence registry on the right.
- The header names the active scope and corpus size.
- Answers use long-form typography with inline numbered citations.
- Evidence cards clearly label Cited, Consulted, Retrieved, Historical, Fallback, or Unavailable.
- A dark, high-contrast Evidence Route card may visualize question → recognized triple → fact/path → passage. This is the only dark surface in the light default and is functional, not decorative.
- Clicking a citation selects the evidence card and can pivot the route back into Atlas.
- The right column collapses to a sheet at compact widths.

### 6.6 Control

Control is a conventional settings workspace. It uses a local settings index and calm content surface rather than the graph. Sources, Profile, Privacy & Capture, Appearance, Shortcuts, and Diagnostics retain their complete current functionality.

## 7. Shared Components and Boundaries

The branches may reuse concepts but not copy shell components across worktrees.

### Quiet Canvas target modules

- `workspace/QuietRail.tsx`: the 49 px global destination rail.
- `workspace/SceneHeader.tsx`: breadcrumbs, title actions, and local tabs.
- `workspace/WorkspaceShell.tsx`: rail + one scene host + overlays.
- `meetings/TodayScene.tsx`: composed home experience replacing `DeskScene`.
- `meetings/MeetingDocument.tsx`: authored document layout.
- `meetings/MeetingContextMargin.tsx`: bounded generated context.
- `ask/ConversationScene.tsx`: reading-column chat and evidence sheet.
- `design/tokens.css`, `design/globals.css`, and focused scene stylesheets.

### Knowledge Atlas target modules

- `features/workspace/AtlasCommandBar.tsx`: coordinates, locator, status, Ask.
- `features/atlas/AtlasField.tsx`: full-workbench graph composition.
- `features/atlas/ExplorePanel.tsx`: search, territory, and layer controls.
- `features/atlas/ObjectDossier.tsx`: selected object and path explanation.
- `features/atlas/AtlasTimeControl.tsx`: temporal filter control.
- `features/inquiry/InquiryWorkbench.tsx`: answer/evidence composition.
- `features/evidence/EvidenceRegistry.tsx`: evidence cards and route display.
- `styles/tokens.css`, `styles/shell.css`, and focused feature stylesheets.

Existing repositories, Redux slices, controllers, state machines, sidecar gateways, and WebSocket supervisors remain authoritative. Presentation components consume their existing typed projections. UI components do not fetch directly when a repository/service boundary already exists.

## 8. Data and Runtime Behavior

The overhaul does not change server contracts.

- Quiet Canvas continues to consume `SidecarProjectionStore`, repositories, `MeetingRuntime`, and `ConversationHub`.
- Knowledge Atlas continues to hydrate normalized domains through its existing services and Redux store.
- Both continue to use `/startup`, `/meetings`, `/chats`, `/profile`, `/graph`, `/connectors`, `/updates`, `/session`, and `/chat` as currently implemented.
- Route changes, view selection, drawers, filters, and panel visibility are local UI state.
- Authored notes, capture lifecycle, chat turns, evidence payloads, graph projection, connectors, and profile remain durable/server-owned where they are today.

The worktree development stores are populated from the main Taurine snapshot. The current verification corpus contains 724 notes, 9 meetings, 25 chat threads, 6,206 passages, 1,750 facts, and 2,224 projected note edges. Design and performance acceptance must be run against this populated corpus as well as deterministic fake-sidecar fixtures.

## 9. Loading, Empty, Degraded, and Error States

### Startup

Retain the readiness gate. Use a centered branded startup surface with a real phase label and concise detail. Temporary sidecar connection refusal continues to retry automatically. A fatal error offers Retry, Diagnostics, Settings when useful, and Quit.

### Loading

- Preserve layout dimensions with subtle skeletons for lists and document metadata.
- Do not render fake graph nodes. Atlas shows a field progress state until a real projection is available.
- Streaming answers reserve the evidence column but do not show citations until validated evidence arrives.

### Empty

Every empty state contains a title, one-sentence explanation, and one or two relevant actions. Empty regions are not rendered merely to say they are empty.

### Degraded

- Disconnected: preserve readable local content and show one compact reconnecting status.
- Stale graph: keep the current projection visible with a stale badge and rebuild action.
- Missing capture permission: meeting document remains editable; capture action explains which permission is absent.
- Weak retrieval: show the existing abstention language and suggested scope changes.
- Evidence unavailable: keep the answer state truthful and identify unavailable citations without inventing support.
- Save conflict: keep both authored versions available and use the existing conflict-resolution workflow.

## 10. Motion and Interaction

- Standard transitions: 140–180 ms using ease-out.
- Sheets and floating surfaces animate opacity plus at most 8 px translation.
- Graph camera transitions: 220–320 ms and cancellable by direct manipulation.
- Node motion communicates layout or changed data only; there is no ambient drift.
- Recording indicators may pulse gently, but must stop under `prefers-reduced-motion`.
- Reduced motion removes nonessential transforms and uses immediate graph camera changes.
- Hover never reveals functionality that cannot also be reached by keyboard focus.

## 11. Responsive Rules

### Quiet Canvas

- ≥ 900 px: 49 px rail, main document, optional context margin.
- 680–899 px: rail retained; context becomes a right sheet.
- < 680 px or 200% zoom: rail remains; titles compact; meeting tabs horizontally scroll; all secondary surfaces become modal sheets.

### Knowledge Atlas

- ≥ 1180 px: Explore panel and selected dossier may coexist over the field.
- 820–1179 px: only the active floating panel remains open; evidence is a sheet.
- < 820 px or 200% zoom: graph uses full field with one bottom sheet; the outline is the default alternative navigation surface.

No primary action, title, or authored content may be horizontally clipped at supported widths.

## 12. Accessibility

- Preserve correct landmarks and one `h1` per scene.
- Maintain 44 px preferred pointer targets and a 36 px absolute minimum for compact desktop controls.
- Every icon-only button has an accessible name and tooltip.
- Focus indicators use a two-pixel outline with sufficient contrast.
- Dialogs and sheets trap focus, restore focus on close, and make the background inert.
- Live transcript and streaming output use throttled announcements; the whole stream is never repeatedly announced.
- Atlas has a complete outline/list equivalent with matching selection, open, pin, and Ask actions.
- Color is never the sole carrier of source, evidence, temporal, or connection meaning.
- Both branches pass axe scans, keyboard-only primary journeys, 200% zoom, reduced motion, and light/dark contrast checks.

## 13. Testing and Acceptance

### Automated

1. Preserve all current core and renderer tests.
2. Add component tests for each new shell, rail/command bar, contextual panel, and empty/error state.
3. Update existing journey tests to assert semantic behavior rather than obsolete class names or pane structure.
4. Add visual screenshots for populated Today, meeting document, Ask/Inquiry, Atlas field, selected dossier, evidence sheet, compact layout, dark mode, startup, and fatal startup.
5. Run TypeScript checks, ESLint, Prettier, production build, and existing bundle budgets.
6. Run Playwright/Electron journeys with both fake fixtures and the populated local corpus.

### Manual

- Start both apps through their worktree `run.sh` scripts.
- Confirm imported meetings, chats, graph nodes, passages, facts, and edges are visible.
- Complete start/stop meeting, note edit/save, transcript, proposal review, standalone Ask, meeting Ask, evidence open, graph select/filter/time, source settings, profile, diagnostics, privacy curtain, and quit/reopen flows.
- Inspect all main routes at default window size, narrow width, 200% zoom, dark mode, and reduced motion.
- Confirm no blank permanent inspector, pinned footer, oversized unused canvas, raw placeholder copy, or cramped text label rail remains.

### Visual acceptance

Quiet Canvas passes when it reads as a warm notebook with a slim rail, composed populated home, authored meeting document, and quiet contextual AI.

Knowledge Atlas passes when it reads as a light cartographic research environment with the map owning the workbench, contextual floating tools, clear territory depth, a selected-object dossier, and evidence-led Inquiry. It must not resemble the rejected dark immersive mockup.

## 14. Cutover

This is a direct cutover in each feature branch. No feature flag, gradual rollout, legacy theme, or compatibility CSS is required.

Implementation removes obsolete shell markup and styles after each replacement route is covered by tests. It must not leave both old and new layouts in the DOM. Existing data and backend contracts remain in place; worktree databases are disposable snapshots and require no migration.

## 15. Final Approved Direction

- Two visually distinct products are required.
- Quiet Canvas: approved reduced 49 px rail and living-notebook composition.
- Knowledge Atlas: approved earlier light cartographic map and evidence-led Inquiry.
- Dark immersive Atlas: rejected.
- Both branches: complete styling and composition overhaul, populated-data verification, existing functionality preserved, no staged cutover, and no data migration.
