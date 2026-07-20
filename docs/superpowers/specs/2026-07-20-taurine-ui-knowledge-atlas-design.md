# Taurine Knowledge Atlas — desktop product and frontend architecture specification

**Date:** 2026-07-20
**Status:** Design decision; implementation-grade specification, no application code
**Track:** B — power, dense legibility, keyboard operation, multi-context comparison
**Product posture:** Local-first personal intelligence workstation

## 0. Decision in one page

Taurine will no longer present meetings, chat, a force graph, data, and connectors as separate products behind tabs. It will become a single **Knowledge Atlas**: a workstation in which the same personal knowledge can be read spatially, temporally, and evidentially.

- **Spatially:** the Atlas field shows stable territories, neighborhoods, relationships, and the current conversation's route through them.
- **Temporally:** Chronicle places capture, notes, transcripts, summaries, and follow-up in a chronological record rather than an isolated live screen.
- **Evidentially:** every generated claim can open the passage, fact, path, date, and retrieval decision behind it; “retrieved,” “consulted,” and “cited” never collapse into one source label.

The shell is one interactive main window with a command deck, coordinate rail, context-sensitive index, central workbench, optional evidence inspector, persistent context shelf, and an exception-driven operation ledger. A user can search in the spatial Atlas, pivot the same selection into the temporal Chronicle, pin three notes or meetings for comparison, start an explicitly seeded inquiry from those objects, enter a live meeting without losing that context, then return to the same arrangement. Sources, Profile, Privacy, and Diagnostics live in Control rather than posing as equal knowledge coordinates. A meeting is a durable dossier, not a transient session. The graph is a navigable atlas with an accessible outline, not a hairball that only a pointer user can operate.

The design is light-first, ink-on-mineral-paper, square-edged, compact, and typographically restrained. It uses no decorative gradients, sparkle motifs, emoji controls, excessive pills, or generic dashboard cards. Color conveys category only in combination with shape, label, or line pattern.

The Python sidecar and its evidence contracts remain the product spine. The frontend is reorganized around normalized domain state, one connection supervisor, explicit meeting and chat state machines, and revision-aware writes. Three narrowly justified backend additions are specified: atomic/revision-aware meeting note commands, resumable live context inside the existing disconnect/stop grace windows, and desktop-owned OAuth authorization. The graph, retrieval, chat, live summary, profile generation, connector sync, and final-summary algorithms are not redesigned here.

One trust rule overrides visual polish: Taurine must describe what it can prove. `setContentProtection(true)` is best-effort, not guaranteed invisibility on modern macOS. “Accepted” retrieval is not the same as a true claim, especially while the checked-in calibration artifact has zero labeled samples. The UI must disclose both facts plainly.

## 1. Product thesis, users, and success

### 1.1 Product thesis

Personal intelligence becomes useful when a user can hold **where an idea lives, when it changed, and why the system believes it** in one working context. Taurine's advantage is not a prettier note list or another chat box. It is the ability to move without translation among a live conversation, the user's authored record, the surrounding knowledge territory, and the exact evidence that supported an intervention.

The primary object is therefore a **working context**, not a page. A working context contains:

1. a scope: all knowledge, a meeting, a note neighborhood, a source, a date range, or pinned objects;
2. a coordinate: Atlas, Chronicle, or Inquiry, with administrative state in Control;
3. a selection and optional comparison set;
4. a history of inspected objects and evidence;
5. an inquiry whose retrieval scope is visible before it is sent.

This model lets Taurine serve as a spatial/temporal “knowledge atlas” while remaining legible to users who never want to touch a graph. Search, outline, dossiers, and evidence inspection are equal first-class paths through the same objects.

### 1.2 Primary users

#### The meeting operator

Runs several calls per day, needs low-friction capture and notes, wants a reliable post-call record, and cannot tolerate the app stealing focus. They value obvious mic/system-audio state, fast note entry, recoverable drafts, and a concise finalization state more than novelty.

#### The knowledge investigator

Searches across notes, messages, email, calendar, documents, and previous meetings. They compare sources, follow paths, inspect dates, and ask multi-part questions. They care whether a passage was merely retrieved, actually read, or cited, and they use the keyboard heavily.

#### The reflective planner

Returns to meeting history, relationships, recurring projects, and the generated profile to understand change over time. They need chronology, stable spatial geography, and the ability to collect a small evidence set before asking or writing.

#### The cautious professional

Uses Taurine while screen sharing or handling sensitive local sources. They need truthful permission and privacy state, a deterministic privacy curtain, clear destructive-action boundaries, and no surprise recording.

### 1.3 Core jobs

- Start a correctly configured meeting capture in under ten seconds without losing the current workspace.
- Take notes continuously while transcription, grounding, and summaries update around them.
- Understand what Taurine surfaced and why, without mistaking correlation or retrieval confidence for truth.
- Ask a question across all knowledge or from a deliberately seeded context and inspect support at claim level.
- Move from a note, meeting, answer, fact, or source into its spatial and temporal neighborhood.
- Compare up to three objects without opening a forest of windows.
- Recover from sidecar startup, socket loss, permission denial, stale indexes, conflicts, and interrupted model turns without losing authored work.
- Enable, configure, sync, disable, or purge a source with consequences stated before action.
- Maintain the non-citable interpretation profile that helps Taurine resolve names and interests.

### 1.4 Design principles

1. **Authored content outranks generated content.** User notes remain visually and operationally primary. Generated summaries, edits, actions, and profile text are labeled and reversible.
2. **Selection is not activation.** One click or arrow movement selects; Enter opens; Space previews; Shift+Enter pins for comparison. This rule is consistent in lists, the outline, and the map.
3. **Evidence has states, not decoration.** Retrieved, consulted, cited, fallback, interrupted, historical, conflicted, and unavailable are distinct labeled states.
4. **Temporal and spatial views stay synchronized.** A date brush filters the Atlas and Chronicle; a selected meeting highlights its knowledge neighborhood; a node dossier shows its dates and source lineage.
5. **Power is visible but staged.** The default layout has strong hierarchy. Dense controls live in the index and inspector, not in every row. Advanced retrieval traces are one disclosure away, never the first thing a meeting operator must parse.
6. **Keyboard and pointer are peers.** No graph, timeline, resize, filter, context, or destructive action is pointer-only.
7. **Connectivity is explicit.** “Reconnecting” never masquerades as live. Local edits show queued, saved, or conflicted state. Capture gaps are disclosed.
8. **Privacy claims are literal.** Best-effort OS APIs are described as best effort. A privacy curtain works by ceasing to render private content.
9. **Stable geography beats perpetual motion.** Unchanged nodes retain coordinates across refreshes and builds where possible. Motion communicates transition, not ambience.
10. **No fabricated progress.** If the server reports only running/done/error, the UI shows an indeterminate operation rather than invented percentages or phases.

### 1.5 Success metrics and release gates

Product success is measured with representative tasks, not only subjective visual review:

| Measure | Release target |
|---|---:|
| Start a new mic + system-audio meeting from a ready app | median ≤ 10 s; 90th percentile ≤ 18 s |
| Open the evidence behind a cited answer sentence | one action from citation or keyboard equivalent |
| Recover a meeting dossier after renderer reload | 100% of persisted notes/transcript/final state restored; no silent overwrite |
| Core workflows operable without pointer | 100% of start/stop, note editing, meeting selection, ask, evidence opening, graph outline navigation, source sync, and profile edit |
| Destructive actions with consequence preview | 100% of meeting/chat deletion, source purge, profile overwrite |
| Accessibility | WCAG 2.2 AA for rendered application surfaces; no known keyboard traps |
| UI response to selection/command | p95 < 100 ms excluding network/model work |
| Atlas interaction | ≥ 30 fps at 5,000 projected nodes on the reference Mac; lists remain 60 fps through virtualization |
| Note durability | zero lost authored edits in disconnect, conflict, reload, stop, and final-summary races in automated scenarios |
| Unsupported privacy claims | zero uses of “invisible,” “uncapturable,” or equivalent guarantee |
| Evidence semantics | zero cases where consulted-only items are labeled as support for a claim |

Engineering gates treat today's verified baseline as a floor, not an endorsement:

- preserve the complete core suite: **981 tests passing, 2 skipped**;
- preserve and expand the current **61 Vitest tests**, both Node and Web TypeScript checks, and production build;
- introduce contract, accessibility, and Electron end-to-end coverage described in §15;
- reduce and segment the current renderer startup payload. Today's approximately **3.64 MB JavaScript and 381 KB CSS** are regression baselines to beat. The initial shell chunk must be ≤ 1.25 MB uncompressed JavaScript and ≤ 140 KB uncompressed CSS; Atlas, editor, and Markdown rendering load as separate chunks. Total lazy-loaded code may exceed the initial limit when justified, but no single feature chunk may exceed 900 KB without a recorded bundle analysis.

### 1.6 Non-goals

- Replacing the shared retrieval engine, evidence registry, graph builder, ASR provider, summarizer, or model routing.
- Building Projects, People CRM, Calendar planning, Coach, Daily Briefing, Agentic Canvas, or Gap Finder as new product areas. Existing calendar/doc sources remain searchable.
- Collaborative multi-user editing, cloud sync, web/mobile clients, or shared workspaces.
- Arbitrary floating panels, unlimited split panes, or detachable stateful feature windows.
- A graph editor that lets users create or correct semantic/fact edges. Inspection is in scope; correction needs its own data design.
- Persisting the entire live grounding/retrieval trail into meeting records. Current session logs remain diagnostic; the post-meeting dossier uses persisted notes, transcript, summary, durable note proposals, chat, and index evidence.
- Claiming semantic entailment for citations. Current hard validation checks identifier validity, availability, fact provenance, and temporal consistency; coverage/NLI enforcement was deliberately removed.
- In-product management of model providers, API keys, or model selection.

## 2. Evidence from the current product

This design follows an audit of every file under `app/src`, application configuration and package contracts, `README.md`, current product specs and retrieval audits, recent meeting/map/notification history, and the relevant sidecar schemas, endpoints, sockets, chat/live/retrieval/connectors/session logging modules.

### 2.1 Capabilities that must survive the redesign

- One Electron main window that restores/hides from the tray, plus non-activating call and context suggestion windows.
- Mic capture at 16 kHz and macOS system-audio capture at 48 kHz decimated 3:1, carried on channel-prefixed binary frames to `/session`.
- Call detection, one prompt per call, dismissal/rearm behavior, and user-selectable automatic start.
- Draft/live/summarizing/done meetings with persisted title, Markdown notes, transcript, final summary, proposed edits, linked chat, and AI provenance.
- Rolling transcript, grounding, summary, actionables, suggested relevant notes, final-summary research, and meeting-scoped Ask.
- General multi-thread chat with streaming retrieval, reasoning, tool activity, answer deltas, stop, regeneration, auto-title, rename/delete, and evidence payloads.
- Global and local graph, communities, facts, explanatory paths, idea lens, search, source/time/community filters, timeline, graph coloring, ego depth, hide/pin, and live retrieval overlays.
- Apple Notes, Obsidian, iMessage, Claude memories, Codex memories, local folders, Gmail, Google Calendar, Google Docs, Outlook, and the hidden Taurine meeting source.
- Source enable/disable/config/sync/purge, global sync/cancel, stale index status, and immutable graph refreshes.
- Editable/generated user profile with server-reported research phases and background injection into chat/live summaries.
- Startup progress, index status, session diagnostics, external Apple Notes opening, note preview, tray actions, and content-protection requests.

### 2.2 Structural liabilities to remove

1. `App.tsx` currently coordinates unrelated runtime, meeting, audio, notes, chat, updates, graph, notification, and navigation lifecycles. Feature views cannot be reasoned about or tested independently.
2. Three reconnecting sockets implement different recovery semantics. The session client replays a start request but cannot replay audio; note updates can be dropped; chat turns are socket-task-bound; update events only invalidate some domains.
3. Meeting notes have competing REST, session-notepad, AI-line, suggestion, inline-AI, and chat-edit states. Text-normalized per-line provenance is ambiguous when lines repeat or move.
4. The canvas map has no semantic keyboard/screen-reader equivalent. Timeline brushing, resize, context menus, and several graph operations are pointer-dependent.
5. Views remain mounted while visually hidden. `aria-hidden` does not remove descendants from keyboard order; overlays and drawers do not have a uniform inert/focus policy.
6. Small 24–36 px controls, hover-only actions, low-opacity text, emojis, decorative gradients, glass effects, sparkles, and many pill shapes undermine dense legibility and accessibility.
7. The current UI drops or underuses evidence already returned by the sidecar: query rewrite, retrieval decision/confidence, build identity, temporal fact status, explanatory paths, tool arguments/evidence IDs/latency, stripped claims, and interruption state.
8. Meeting finalization events are best-effort on the originating session socket. The meeting row is the durable truth, but the UI does not consistently reconcile from it after reconnect.
9. Source disable/purge and global sync have important immediate-versus-rebuild consequences that are hard to infer from current controls.
10. `setContentProtection(true)` is applied broadly but cannot justify a promise that windows are absent from screen capture.

### 2.3 Trust boundary

The interface uses the following exact language:

- **Cited:** the accepted output contains this evidence identifier and it was available to the turn.
- **Consulted:** the model read this evidence, but the accepted output did not cite it.
- **Retrieved:** the retrieval system considered or selected it; it may never have been read by the answer model.
- **Accepted evidence:** the retrieval decision passed the configured policy. It does not mean the generated sentence is independently proven true.
- **Low-confidence fallback:** chat received below-floor passages for best-effort exploration; facts and paths were withheld by the sidecar.
- **Citation checks passed:** identifiers, availability, fact provenance, and temporal consistency passed. It must never be rendered as “fact checked” or “verified true.”
- **Calibration not validated:** the active artifact reports zero labeled samples/held-out metrics. Confidence is shown as a technical score only in the inspector; primary UI uses decision categories.
- **Interrupted:** the streamed partial was persisted without final evidence validation and cites nothing, even if citation-looking text remains in the raw partial.

## 3. Information architecture and mental model

### 3.1 Three knowledge coordinates and Control

The persistent 52 px rail contains three coordinate switches, not five product destinations:

| Coordinate | Shortcut | Index pane | Workbench | Default inspector |
|---|---|---|---|---|
| **Atlas** (where) | `⌘1` | search results, territories, saved/pinned contexts | spatial field or synchronized outline | selected note, edge, fact, or path |
| **Chronicle** (when) | `⌘2` | chronological meeting list grouped by date | meeting dossier or live desk | transcript/evidence/record metadata |
| **Inquiry** (why/what next) | `⌘3` | inquiry threads | selected investigation | cited/consulted evidence and process |

The coordinate switch changes the representation of the same working context. A selected/pinned meeting in Chronicle remains highlighted as a Taurine note in Atlas and remains the named scope in Inquiry. Back/forward history, context shelf, comparison, inspector, time range, and active meeting survive the pivot. This is why the rail is not the current tab taxonomy rotated vertically.

A separated bottom **Control** button (`⌘,`) opens an administrative workbench with an internal index: Sources, Profile, Privacy & Capture, Appearance, Shortcuts, and Diagnostics. Control does not retain a place among the three knowledge coordinates, and healthy source/profile machinery does not compete with ordinary work. Direct commands `Open Sources` and `Open Profile` remain in `⌘K`; optional accelerators `⌘4` and `⌘5` may invoke them without creating rail items.

“Atlas” and “Chronicle” are introduced once in the first-run orientation as spatial and temporal views. Inquiry and Control remain plain functional labels.

### 3.2 Persistent shell regions

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Command deck: back/forward · locator/command · live state · privacy · actions│ 48
├────┬───────────────┬───────────────────────────────────┬─────────────────────┤
│    │               │                                   │                     │
│ 52 │ Index         │ Workbench                         │ Inspector           │
│rail│ 230–360       │ min 540                           │ 320–480             │
│    │               │                                   │ optional            │
│    │               │                                   │                     │
├────┴───────────────┴───────────────────────────────────┴─────────────────────┤
│ Context shelf: 0–3 pinned objects · inquiry scope · compare controls       │ 44
├──────────────────────────────────────────────────────────────────────────────┤
│ Operation ledger (only while active/degraded, or when user pins it)          │ 28
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Command deck

The center locator reads the current scope and selection and is also the `⌘K` command entry. It is a rectangular 36 px field, not a decorative omnibox. Back/forward are object-history operations. The right side contains live capture state, privacy state, inspector toggle, and one scope-specific primary action.

#### Coordinate rail

Uses 20 px line icons and visible text tooltips on hover/focus. The active coordinate has a 3 px ink/accent bar and surface change, never a pill. An active meeting adds a labeled live mark to Chronicle; background inquiry activity adds a count, not an animated glow. Control is visually separated at the bottom and never shows routine source/profile counts.

#### Index pane

Provides fast object navigation and filters. It is a virtualized semantic list with a sticky query/header and a single selected row. Rows are 40 px compact or 48 px comfortable. Actions appear on focus as well as hover and are present in the row context menu.

#### Workbench

Hosts one primary representation or a two/three-column comparison board. It owns document scrolling. It never becomes an unbounded draggable canvas of panels.

#### Inspector

Shows evidence and metadata for the current selection. It is persistent when width permits, an inert focus-trapped sheet on smaller widths, and toggled with `⌘\`. Selecting a citation updates the inspector without navigating away from the answer.

#### Context shelf

Holds zero to three pinned notes, meetings, answers, facts, or paths. Each rectangular item has type, title, source/date, and remove action. `Shift+Enter` pins the selection. “Compare” lays pinned objects into 2–3 columns. “Ask these” opens Ask with explicit seed IDs/current meeting scope. Removing an item changes only workspace context, never source data.

#### Operation ledger

The 28 px ledger is **not persistent when the system is healthy and idle**. It appears when an operation is active or user action is required: runtime recovery, capture interruption, sync/index work, note save beyond 1.5 seconds, conflict, profile generation, privacy curtain, or a failed operation. Completed success remains for four seconds, then the ledger collapses without changing focus. Routine note counts, socket labels, ASR state, build hashes, and last-sync times never appear there.

Power users can pin the ledger open from the command center. Pinned healthy state shows only `Ready` plus user-selected operation categories, never five mandatory telemetry cells. When collapsed, the command deck has one 40 px **Operations** button only while an exception/background operation exists; its accessible name includes the highest-priority state. Capture truth remains in the Live header even if the ledger is collapsed.

### 3.3 Object vocabulary

The UI consistently distinguishes:

- **Note:** a source document from any connector.
- **Meeting:** a durable authored record with transcript and generated material; indexed later as a hidden Taurine-source note.
- **Inquiry:** a general or meeting-linked chat thread.
- **Passage (`E#`):** exact answer-bearing source text.
- **Fact (`F#`):** directed temporal assertion with source passages.
- **Path (`P#`):** explanatory graph route.
- **Transcript evidence (`T#`) / meeting evidence (`M#`):** turn/session-local material.
- **Territory:** a server-provided community rendered as an Atlas region.
- **Working context:** current scope, filters, selection, pins, comparison, and inquiry scope.

Evidence IDs remain visible because they connect generated text to inspection, but raw internal note IDs and build hashes stay in the advanced metadata disclosure.

### 3.4 Multi-context comparison

Comparison supports exactly two or three pinned objects. This cap preserves readable columns and prevents a second window-management problem.

- Notes show title, source/date, gist, full preview, outgoing/incoming relationships, and evidence passages.
- Meetings show title/time/status, notes, generated recap, actionables, and transcript excerpts.
- Answers show query, answer, cited set, consulted set, retrieval decision, and process.
- Facts show subject/predicate/object, direction, temporal status, validity interval, confidence, and source passages.
- Paths show rendered route, edge kinds, weight, and each supporting object.

Shared entities, dates, sources, and citations are highlighted with a left-margin marker and textual “shared” label. There is no automatic LLM synthesis merely because comparison opens. “Ask these” is the explicit synthesis action.

## 4. Window and layout model

### 4.1 Stateful windows

Only the **main workstation window** owns product state, sockets, audio sending, editor state, and navigation. This prevents duplicate session/chat owners and divergent caches.

| Property | Main workstation |
|---|---|
| Default bounds | 1280 × 820 CSS px, centered on first launch |
| Minimum | 960 × 640 |
| Restored | last normal bounds and display; clamp to visible work area |
| Close | hide to tray unless explicit Quit |
| Content protection | requested on supported platforms, represented as best-effort state |
| Native frame | platform frame; macOS traffic-light inset respected by 48 px command deck |
| Full screen | supported; compact mode exits first |

No note preview, transcript, chat, or connector configuration opens a stateful secondary BrowserWindow. Local note previews use the inspector/workbench; supported sources may still open in their native application.

### 4.2 Layout breakpoints

| Width | Layout |
|---:|---|
| ≥ 1440 | rail 52 + index 280 default + workbench + inspector 360 default |
| 1180–1439 | rail 52 + index 248 + workbench; inspector closed by default and opens as a 360 px overlay |
| 960–1179 | rail 52 + workbench; index and inspector are mutually exclusive overlay sheets |

Index width resizes from 230–360; inspector from 320–480. Dividers are 1 px visually with a 12 px hit region, keyboard focus, and arrow-key adjustment in 16 px increments (`Shift` = 48 px). Double-click/Enter on a divider restores its default. Widths persist locally by coordinate and Control section.

At 200% text zoom, the app switches to the smallest-width layout even if physical width is larger. No coordinate or Control section disappears; overlays become full-height and focus-trapped.

### 4.3 Workbench modes

The main window has four mutually exclusive layout modes:

1. **Standard:** the shell described above.
2. **Compare:** the workbench becomes two or three synchronized dossier columns; index and inspector remain available.
3. **Live desk:** Chronicle with transcript, notes, and live intelligence arranged for capture (§7.2).
4. **Compact live:** the same main BrowserWindow temporarily becomes a narrow capture desk. Previous bounds/layout are saved and restored exactly on exit.

Compact live uses 440 × 680 default bounds, 420 × 560 minimum, 560 × 900 maximum. It can be kept above other windows only after the user enables **Keep compact meeting above other windows**; default is off. Layout:

- 44 px capture header with elapsed time, mic/system channel state, stop, and return-to-workstation;
- 120 px last-final transcript region with explicit interim styling;
- remaining space for the note editor;
- 44 px collapsed context/action queue; opening it overlays the editor and traps focus;
- no Atlas canvas or persistent inspector; citations open a temporary internal evidence sheet.

Compact mode does not create a second renderer and cannot outlive the main window.

### 4.4 Non-activating utility windows

#### Call prompt

- 372 × 136, bottom-right of the active display with 16 px work-area inset.
- non-activating/non-focus-stealing; clickable pointer controls; every action is mirrored in the tray and main-window command center for keyboard/assistive access.
- visible on all workspaces and above ordinary windows, never above system security dialogs.
- states: detected, ask, auto-start countdown, starting, start failed, dismissed.
- does not contain transcript, contact, or source content.

Call-detection preference has three values: **Off**, **Ask** (default after upgrade/first run), and **Auto-start after 10 seconds**. Auto-start is opt-in. The countdown has Pause, Start now, and Dismiss. Dismiss/rearm retains the current one-prompt-per-call and five-second idle rule.

#### Context suggestion

- 372 × 188, same placement discipline, queue cap three, 12-second visible duration.
- shows source, note title, and bounded snippet; Open sends an IPC intent that reveals the existing main window and selects the note in Atlas. Dismiss changes only the ephemeral suggestion queue.
- while Privacy curtain is on, it renders “Relevant context available” without title/snippet and offers Open privately/Dismiss.

These utilities never open their own sidecar connections.

### 4.5 Tray and dock

Tray menu state is live and exact:

- Show Knowledge Atlas
- Start meeting… / Open live meeting / Stop depending on capture state
- New inquiry
- Privacy curtain On/Off
- Privacy protection: Limited/Requested/Unavailable (informational item opens settings)
- Quit Taurine

Dock activation reveals the existing main window. Call/context utilities never raise it. Explicit Open/Show does.

### 4.6 Privacy modes

Taurine requests Electron content protection, but the UI reports one of:

- **Protection requested:** platform API enabled; user should still verify with the actual sharing/recording app.
- **Limited on this Mac:** content protection requested, but newer macOS applications using ScreenCaptureKit may capture the window despite it.
- **Unavailable:** platform/runtime cannot request protection.
- **Privacy curtain on:** private content is not rendered in any Taurine window; capture/listening may continue.

Electron explicitly documents the ScreenCaptureKit limitation in [`BrowserWindow.setContentProtection`](https://www.electronjs.org/docs/latest/api/browser-window#winsetcontentprotectionenable). No UI copy may say “invisible to screen share.”

Privacy curtain (`⌘⇧P`) replaces main-window content with a neutral surface containing only elapsed capture state, channel health, Stop, and “Turn curtain off.” It also redacts utility windows and disables native note preview popovers. Because private content is not in the rendered tree, captured Taurine pixels are neutral even when OS capture protection fails. The user cannot inspect notes while the curtain is active; that tradeoff is stated before first activation.

## 5. Navigation and interaction grammar

### 5.1 History, selection, preview, open, pin

- Arrow keys or one click change selection and inspector.
- Space opens/closes Quick Look in the inspector without changing history.
- Enter opens the object as the workbench subject and pushes object history.
- `Shift+Enter` toggles it in the context shelf.
- `⌘[` / `⌘]` move backward/forward through opened objects, coordinates, Control sections, and filter contexts.
- Escape closes the topmost menu/sheet/Quick Look first, then clears transient selection; it never silently discards editor changes.
- A double click is equivalent to Enter but is never required.

### 5.2 Command center and quick open

`⌘K` opens a command center with three result groups:

1. objects: notes, meetings, inquiries, sources, facts;
2. commands: switch coordinate, open Control section, start/stop meeting, sync, toggle inspector/index/curtain, density/theme;
3. recent contexts: selections and pin sets from this session.

`⌘P` opens object-only Quick Open. Both use one search field, arrow navigation, Enter to open, `Shift+Enter` to pin, and `⌘Enter` to open then close overlays. Search results expose type/source/date and never rely on icon alone.

### 5.3 Coordinate-specific primary actions

- Atlas: **Ask selection** when something is selected, otherwise **Search Atlas**.
- Chronicle: **New meeting**; while live, **Stop**.
- Inquiry: **New inquiry**.
- Control → Sources: **Sync enabled sources**; while syncing, **Cancel sync**.
- Control → Profile: **Edit** or **Generate profile** depending on mode.

Only one filled primary button appears in the command deck. Secondary actions live in menus/toolbars with text tooltips.

### 5.4 Focus zones

`F6` cycles coordinate rail → index → workbench → inspector → shelf → operation ledger, skipping closed or absent regions. `Shift+F6` reverses. Each region has a visible focus entry and retains its last internal focus. Opening a modal sheet stores and restores origin focus. Hidden regions use the HTML `inert` attribute and are unmounted when expensive; `aria-hidden` alone is prohibited as a focus-management strategy.

## 6. The Atlas: spatial, temporal, and evidentiary knowledge

### 6.1 Purpose

The Atlas is an orientation and investigation tool. It is not the retrieval base graph and does not claim that every drawn relationship is a verified fact. It renders the existing note projection, community assignments, semantic/topic/entity/mixed edges, facts, mechanisms, and current session signals. The inspector explains which layer a relationship came from.

### 6.2 Cartographic levels of detail

The initial global view opens on **territories**, not every label and edge.

| Camera scale | Visible content |
|---:|---|
| < 0.55 | territory hulls, 2–4 word community labels, counts, selected/pinned objects |
| 0.55–1.05 | top three degree/value hubs per territory, recent selected-range objects, territory bridges |
| 1.05–1.80 | nodes passing current filters/default min degree 2; labels for hubs, matches, selection, and pins; static edges only for hover/selection |
| > 1.80 | all visible-node labels, edge line patterns/mechanism markers, date/source micro-labels when density allows |

Territory fills are flat 6–10% categorical tints with a 1 px boundary. There are no gradient heatmaps. Community labels remain horizontal. Search matches use an outer square focus mark; pinned objects use a double ring; the current conversation uses a labeled diamond anchor. Shape and label accompany color.

### 6.3 Stable geography

- Initial positions derive deterministically from community ID, stable note ID hash, degree, and date band.
- The force simulation runs in a worker, settles, and stores coordinates keyed by active build ID plus note ID.
- On graph version change, unchanged nodes begin at prior coordinates; new nodes enter near their community centroid; removed nodes fade for 140 ms before deletion.
- A user-dragged/pinned coordinate persists locally and is visually marked. “Release position” and “Reset layout” are available by keyboard/menu.
- Global fit never starts a perpetual animation. Reduced-motion mode snaps to the settled state.

### 6.4 Time band

A 56 px time band anchors the bottom of Atlas workbench. It plots note/meeting counts as 1 px vertical ticks grouped by week/month, not an area gradient. Presets: All, 30 days, 90 days, 1 year, Custom. Custom exposes start/end date fields beside the brush so dragging is never required. The active range filters Atlas, outline, and Chronicle index; clearing restores All.

Relative labels always have an absolute local date/time tooltip and accessible name. Dates use the user's locale/time zone through `Intl.DateTimeFormat`.

### 6.5 Search and lenses

Atlas search covers title, entity, gist, source, and server-supported query results. Results appear in the index, sorted by match relevance with date/source. Moving through results pans without opening; Enter opens; `Shift+Enter` pins.

Lenses are explicit filter sets, not navigation tabs:

- **Source:** one or more sources; glyph/label stays visible.
- **Community:** one or more territories.
- **Recency:** active time range.
- **Connections:** semantic, topic, entity, mixed/fact-backed mechanism; edge patterns differ.
- **Minimum links:** 0–20, default 2 for the global canvas. Search/outline can still reach hidden tail nodes.
- **Session:** objects retrieved/grounded/traversed during the current live session; ephemeral by current contract.
- **Evidence set:** cited/consulted objects from a selected answer.
- **Idea lens:** natural-language `/graph/idea` result set. It is labeled “AI lens,” includes the input query, can be cleared, and never mutates the graph.

Lens changes are summarized as removable text clauses above the field, for example “Last 90 days · Gmail + Meetings · Entity relationships.” They are not a row of rounded chips.

### 6.6 Neighborhood and connections

Opening a node can switch to a local neighborhood using existing `/graph/local?center=&depth=1..3`. The selected node is central; incoming and outgoing relationships are separately listed in the outline and inspector even when canvas edges are visually symmetric.

Edge line language:

| Kind | Line | Inspector label |
|---|---|---|
| semantic | 1 px solid | Similar passage content |
| topic | 1 px long-dash | Shared topic |
| entity | 1 px dot-dash | Shared entity |
| mixed | 2 px solid with mechanism ticks | Multiple relationship mechanisms |
| fact/provenance | 2 px solid + square midpoint | Directed fact support |
| live direct retrieval | 2 px solid, rank number at target | Retrieved for current speech |
| live graph traversal | no extra edge; striped outer node ring | Reached through graph traversal |

Selecting an edge opens endpoints, type, weight, shared terms, fact sentence, fact IDs/provenance, and mechanisms. Weight is labeled “projection weight,” not “confidence.” The projected `GraphEdge` identifies note endpoints, while the existing `/graph/edge-reason` request requires an exact source/destination **passage** pair that the projection does not expose. Knowledge Atlas therefore does not render an **Explain this edge** action and never substitutes note IDs for passage IDs. A future explanation action requires an additive reviewed graph contract that returns the exact representative passage pair.

### 6.7 Live conversation overlay

During a meeting, the Atlas can show a temporary diamond **Conversation now** anchor. Direct retrieved notes connect to it with rank labels; traversal arrivals receive striped rings scaled in two discrete widths. Grounded/displayed notes receive a solid outer ring. Previous pass marks recede over three seconds but stay available in an in-memory event list for the current renderer session.

No post-meeting persistence is implied. After a resumable renderer/socket reconnect, the current grounding snapshot is restored but the route-event history says “Live retrieval trail restarted after reconnect”; it does not invent intermediate events from transcript or session logs.

### 6.8 Accessible Atlas Outline

The canvas is `aria-hidden` and has no essential exclusive action. A synchronized **Outline** is available with `V` and is the default for screen readers.

- Global outline is a tree: territory → notes, with count, source, date, degree, and match state.
- Neighborhood outline is a treegrid: selected note → incoming/outgoing relationship groups → neighbor rows with relationship kind/weight.
- Results can be sorted by relevance, title, date, source, or connections.
- Arrow keys navigate tree/grid semantics; Enter opens; Space Quick Looks; `Shift+Enter` pins; context menu key exposes every canvas context action.
- `+`/`-` zoom, arrows pan, `F` fit, `/` search, `[`/`]` ego history, and Escape clear are available when canvas has focus.

Dragging is never required: zoom controls are 40 × 40, position pin/release is a menu action, time range has fields/presets, and comparison uses commands.

### 6.9 Empty, loading, and error states

- Runtime starting: Atlas skeleton with the real startup phase in the operation ledger.
- No indexed notes: a structured source setup view with Open Sources; no empty decorative illustration.
- Filters yield none: show the active clauses and one-action clear per clause.
- Graph stale during sync: keep prior immutable snapshot visible with “Index update in progress; results use build …”. Swap only after graph version changes and fetch succeeds.
- Graph fetch fails: retain prior graph if present, show retry; otherwise provide search/list fallback.
- Idea lens fails/abstains: preserve the user's query, distinguish request error from no supported result.

## 7. Complete product workflows

### 7.1 Startup, readiness, and first-run setup

The renderer appears immediately with the shell and a real runtime phase from `GET /startup`. It does not hammer protected endpoints while the sidecar is not ready.

1. Main process starts/locates the sidecar as today.
2. Renderer polls `/startup` with capped backoff: 250 ms, 500 ms, 1 s, then every 2 s; pause while app is hidden after 30 seconds, resume immediately on show.
3. `ready=false,error=""` shows phase/detail and elapsed time.
4. `error` shows a persistent runtime incident panel with safe error detail, Copy diagnostics, Retry connection, and Open setup guide. The application shell, privacy curtain, and local preferences remain usable.
5. On ready, the state layer connects `/updates` and `/session`, then hydrates meetings, chats, connectors, profile, summary/status, and the Atlas. The graph/editor/Markdown chunks load only when first used.
6. If a protected REST endpoint returns 503 or a socket closes 1013, runtime returns to Starting and retries; it is not shown as generic offline.

First-run setup is a dismissible checklist stored locally by UI schema version, not a blocking carousel:

- runtime credentials/service readiness (status only; values are never rendered);
- microphone permission;
- system audio availability/permission, with macOS 14.2 requirement;
- recommended sources and their permissions;
- call detection preference (Off/Ask/Auto-start);
- privacy limitation acknowledgement and curtain shortcut.

Mic and system-audio permission requests occur just in time from a labeled user action. A preflight may inspect status without triggering a prompt. iMessage Full Disk Access and similar OS settings show Open System Settings through preload IPC; the UI never claims it granted access.

### 7.2 Meeting lifecycle

#### 7.2.1 Create or detect

A meeting can begin from New meeting, tray, command center, keyboard, or call prompt.

- **New meeting** creates a draft immediately, opens its dossier, focuses title only when invoked from the main window, and leaves capture off.
- **Start immediately** creates a draft and enters preflight.
- **Call detection Ask** displays the non-activating prompt; Start creates/opens the draft without moving focus until the user explicitly opens the main window.
- **Auto-start** exists only after opt-in. At zero, it creates the meeting and begins capture; Pause freezes the countdown; Dismiss obeys rearm rules.
- Duplicate Start intents for the same locally pending action are idempotent in the frontend. A second user action while preparing reveals the existing preparation state.

Drafts are legitimate records and remain in Chronicle if preflight fails. Their row reads Draft · not recorded and offers Resume setup or Delete.

#### 7.2.2 Preflight

Preflight is an inline panel, not a modal that hides notes:

| Check | Ready | Degraded/recovery |
|---|---|---|
| Sidecar/session socket | Connected | Wait/retry; capture cannot begin |
| Microphone | Granted + device label | Denied/unavailable blocks start; Open settings / choose another device |
| System audio | Supported + granted | Continue mic-only; Open settings; do not block |
| Meeting row | Created/refetched | Retry create; no capture without durable meeting ID |
| Existing active meeting | None/same | Open current meeting or explicitly stop it first |

The user chooses the microphone from available devices. The last device ID is stored locally; when absent, the system default is used and labeled. System audio is an explicit per-meeting toggle defaulting to the saved preference when supported.

The final button states exactly “Start mic + system audio” or “Start mic only.” Starting sends the actual channel declaration and meeting ID, opens mic capture first, then system capture. Mic failure aborts the session start and retains the draft. System failure after mic start produces **Live · mic only** with remediation; it does not tear down working mic capture.

#### 7.2.3 Live desk

At the 1280 px ordinary default, Live desk is deliberately a two-lane document surface:

- **Notes document:** primary and at least 620 px wide;
- **Secondary lane:** 300–360 px and switchable between Transcript and Live intelligence without losing either lane's scroll, selection, or disclosure state.

At ≥1440 px the user may enable three-lane live: Transcript 280–380 px, Notes at least 620 px, and Live intelligence 300–360 px. It is remembered as a density preference, not forced merely because the display is wide. Below 1180 px, Notes remains primary and Transcript/Intelligence open as mutually exclusive, focus-trapped sheets. Compact live follows §4.3. This keeps capture comfortable on an ordinary laptop while preserving the high-density operator layout when it is genuinely useful.

The 44 px live header contains:

- live status and elapsed time;
- microphone and system-audio channel health, each with text label;
- note save state;
- Compact toggle;
- Privacy curtain;
- Stop.

There is no animated waveform decoration. Channel meters, if present, are two low-frequency level bars with textual “receiving / quiet / unavailable” state and are `aria-hidden` supplements.

##### Transcript

- Final chunks are grouped by channel and temporal proximity into You/Them blocks; unknown channels use localized “Speaker N.”
- Interims are visually lighter and replaced in place. They are never cited, persisted by the renderer, or announced token-by-token.
- Final rows show local time on focus/hover and support Copy, Insert transcript excerpt, and Ask about this excerpt. Insert is an explicit user-authored atomic document command; it is not a hidden socket mutation. The current API lacks stable renderer-visible `T#` mappings in transcript events beyond `seq`; generated summary citations still open through grounding payloads/registry metadata when available.
- Auto-follow stays on only while the user is within 32 px of the bottom. When they scroll away, a rectangular “New transcript · Jump” control appears with count.
- A reconnect divider marks either a known audio gap or an explicit live-context restart. It says that audio during the disconnected interval may be missing; a normal resumed snapshot does not claim a summary epoch restart.

##### Notes

- BlockNote remains the WYSIWYG editor unless its accessibility audit fails the release gate; Markdown is the persisted interchange.
- Editor body is 16/25 px, maximum reading width 760 px, with a stable left provenance gutter.
- Autosave begins 800 ms after idle. The local draft is mirrored to renderer persistence before network write, keyed by meeting ID and server `notes_rev`.
- While live, the latest saved/locally changed text is also sent as `/session` `notepad` after 2 seconds idle. Stop never waits on storage: capture ceases first, the recovery draft is written, and `stop` opens the 10-second server grace. The document controller then flushes revision-aware REST save and the final notepad during that grace; the finalizer claims the freshest available values at expiry/Finalize now.
- Save states are Editing, Saving, Saved at time, Queued offline, and Conflict. “Saved” is never inferred from a timer.
- A revision conflict keeps both local and server text, stops automatic writes, and opens a two-pane text resolution sheet with Keep mine, Keep server, and Copy both. It never last-write-wins silently.

AI provenance is attached to durable note operations/blocks, not a normalized set of line strings. User edits inside an AI-authored block transfer that changed span to user-owned styling while retaining operation history in the undo stack.

##### Live intelligence

The panel contains three stacked sections with one-open-at-a-time disclosures:

1. **Rolling summary:** current accepted summary revision, last-updated time, Markdown, and evidence citations. **Review for notes** creates a durable proposal; it never inserts generated text directly.
2. **Actionables:** proposed/confirmed/dismissed. Each row shows text, owner if present, transcript quote/evidence IDs, Confirm, Dismiss, Review for notes, and Undo for the latest status decision. Confirm/dismiss sends `actionable_status`; user status wins over in-flight summary updates. Review for notes creates a durable proposal independently of actionable status.
3. **Context queue:** relevant note suggestions from `note_suggestion`, with source, title, snippet, reason “Surfaced during this conversation,” Open, Pin, and Dismiss. The in-app queue mirrors utility notifications so a timeout never destroys discoverability during the current session.

An **Inspect retrieval** disclosure opens the same evidence inspector used elsewhere: decision, confidence category, passages, facts, paths, revision, new support IDs, no-new-evidence, accepted/rejected heads, build/calibration identities, and latency. Debug trace is visually separate from the user-facing summary.

##### Ask in a meeting

Ask opens in the inspector or full Inquiry coordinate while preserving `meeting_id`. Meeting notes, transcript windows, and summaries are citable `M#` evidence; indexed meeting note `taurine:{meeting_id}` is additionally seeded once available.

- The composer visibly reads “This meeting + pinned context.”
- `edit_notes` is available only in the meeting scope.
- An explicit command such as “add this to my notes” creates a durable **Ask proposal** with Accept and Reject. Intent is not consent to mutate the canonical document.
- Inline Ask on selected editor content creates the same durable proposal type and returns a replacement/insertion diff. Accept and Reject are equally reachable; no response mutates the note merely because it streamed.
- If an anchor is missing or ambiguous at application time, the edit is not appended silently. It remains a pending proposal with Choose location, Append here, and Reject.

#### 7.2.4 Disconnect and resume

The target sidecar preserves the meeting's live reasoning context through a narrow 60-second suspension contract; it does not preserve audio. Each start carries a renderer-generated `client_session_id`, stable for one capture attempt.

1. Session socket loss immediately changes the header to **Reconnecting · audio may be missing**. Mic/system capture stops sending and is paused rather than buffering private audio. The local editor stays active and queues its latest draft.
2. The sidecar detaches the socket event sink, stops ASR input, flushes ASR finals, and moves the `LiveSession` into a bounded in-memory suspended registry for 60 seconds. It retains rolling summary revision/text, actionable decisions, grounding/evidence registry, context suggestions, transcript sequence, and latest notepad under `{meeting_id, client_session_id}`.
3. A matching Start within the deadline rebinds a new sink, cancels pending finalization, restarts ASR, and emits `session_snapshot{disposition:"resumed"}` before accepting audio. The renderer replaces/merges summary, actionables, grounding, and sequence by their stable IDs/revisions, then resumes capture. It inserts an audio-gap divider but does **not** claim the summary or grounding restarted.
4. A sidecar restart or expired registry entry yields `disposition:"state_reset"`; an already-finalizing row yields its durable status. The UI then inserts a **Live context restarted** divider, clears only ephemeral grounding/context state, refetches the dossier, and never implies continuity it cannot prove. A mismatched token while an entry still exists is a session conflict, not permission to take it over; the UI opens the existing live meeting or asks the user to finalize it.
5. After the 60-second deadline, the suspended session enters the normal finalization path. A summarizing/done meeting cannot be resumed; the UI opens its finalization/review state and offers Start a new meeting.
6. Persisted transcript and notes remain durable across either disposition. Audio produced while disconnected is never backfilled, and the gap remains disclosed in the final dossier.

#### 7.2.5 Stop and finalization

Stop is immediate and recoverable, with no confirmation that prolongs capture:

1. Activating **Stop** by pointer, tray, or `⌘⇧M` stops new mic/system capture immediately and persists the renderer recovery draft.
2. The renderer sends one idempotent `stop` with a 10-second resume window. The sidecar stops ASR input, flushes final transcript chunks, detaches the event sink, suspends the live context, and delays the final-summary job.
3. A 10-second **Stopped · Resume** strip replaces the live controls without moving focus. Resume is a 44 px action; **Finalize now** is secondary. Note autosave and the freshest notepad flush continue during the grace period. A save failure preserves the local recovery draft and does not restart recording.
4. Resume sends Start with the same `{meeting_id, client_session_id}`. Before the deadline it atomically cancels delayed finalization, rebinds the preserved context, emits the resumed snapshot, and restarts only the channels that pass preflight. It creates no new meeting and no artificial summary epoch.
5. Finalize now or deadline expiry snapshots the freshest persisted meeting notes plus suspended notepad, transitions to **Ending** while final transcript commits, then to **Generating final record** with indeterminate progress. The backend does not expose a percent or internal phases.
6. If the socket is unavailable, capture still stops locally. On reconnect the client sends the same stop intent; an expired/missing suspended session reconciles from the meeting row and cannot resume. The user is never required to keep recording merely to save notes.
7. The user may navigate away or edit notes. Finalization is runtime-owned and recovers through meeting refetch/update events.

If the originating socket emits `final_summary_status`/`final_summary`, apply them only if meeting ID and base revision match the open record, then refetch. If events are missed, the `meetings` update and `GET /meetings/{id}` are source of truth.

#### 7.2.6 Review and proposals

A completed dossier has these document strata in order:

1. **Your notes** — editable canonical record.
2. **Generated recap** — always visible when `final_summary_md` exists, labeled generated and not yet part of notes unless accepted.
3. **Proposed additions** — pending final-summary edits and the “Add meeting summary” proposal.
4. **Actions** — confirmed/open/dismissed status.
5. **Transcript** — collapsible, searchable, virtualized.
6. **Inquiry** — linked meeting chat.

Each proposal shows exact insertion preview with unchanged context, inserted content, rationale, generated timestamp, and base revision:

- **Accept** atomically applies content and marks the proposal accepted.
- **Reject** marks it rejected without changing notes.
- **Undo** reverses the most recent decision if notes have not diverged; otherwise opens a safe inverse diff for confirmation.
- **Stale** appears when current `notes_rev` differs from `base_rev`; anchor is revalidated. Missing/ambiguous anchors require Choose location, Append, or Reject.
- Accepted/rejected proposals remain in a collapsed decision history rather than vanishing.

The generated title fills only an empty title. A user title is never overwritten.

Finalized meetings enter the hidden Taurine source after the delayed/coalesced global sync. The dossier shows one of: Not yet indexed, Index update queued, Indexing, Indexed in build/time, or Index stale. Post-hoc note edits do not claim immediate reindexing; they state “Included on next source sync.”

#### 7.2.7 Delete and recovery

Delete meeting opens an inline dialog naming the meeting and consequences:

- meeting notes, transcript, summary, and note proposals are deleted from product storage;
- active capture/finalization is stopped/cancelled;
- the linked inquiry is **not** deleted by the current backend and remains in Ask;
- the derived Taurine-source note disappears on the next clean sync; until then the index is marked stale/previous build may still contain it.

Confirmation requires the explicit Delete meeting button, not typing the title. There is no trash/undo in the current store, so copy states that deletion is permanent.

### 7.3 Ask: inquiry and evidence workflow

#### 7.3.1 Thread and scope

Ask is a research workspace. The index lists inquiries by updated time with title, meeting link if any, message count, and streaming/interrupted state. New inquiry stays client-empty until first send, matching server semantics.

The composer always exposes scope on its first line:

- All indexed knowledge;
- Current meeting;
- Pinned context (lists count and object titles);
- Current Atlas selection/neighborhood.

Scope determines `meeting_id` and `seed_note_ids`; it does not imply a hard security filter because current seeds bias retrieval rather than restrict it. The UI says **Start from these objects**, not “Search only these,” unless a future server contract implements hard scoping.

Composer is 44 px minimum, grows to eight lines, Enter sends, Shift+Enter inserts a newline. While a turn streams, text remains editable for the next turn but Send is disabled and the stop action is explicit. `⌘.` stops the focused active turn.

#### 7.3.2 Turn anatomy

Assistant turns use a flat document section with a 2 px left rule, not a floating chat card:

1. **Phase line:** Retrieving, Reasoning, Reading, Expanding, Editing notes, Answering, Stopping, Done, Interrupted, or Failed.
2. **Process disclosure:** tool activity in arrival order with status, arguments summary, evidence IDs, latency, and note links. Raw model reasoning is collapsed under Advanced process and is never presented as evidence.
3. **Answer:** progressively rendered Markdown. Deltas batch at 30–50 ms. Citation tokens are keyboard-focusable inline links that select the inspector.
4. **Evidence ledger:** Cited, Consulted, and Retrieved counts/sets; decision category; temporal/fallback warnings; stripped claims if any.
5. **Actions:** Copy answer, Regenerate (last answer only), Pin answer, View evidence in Atlas, and More metadata.

User turns are full-width text blocks aligned to the same reading column with a “You” label, not speech bubbles.

#### 7.3.3 Evidence inspector

Selecting a citation opens the exact record:

- passage: ID, source/note title, exact text, date, character range, retrieval/rerank scores in Advanced, path IDs, prior-context/low-confidence flags;
- fact: subject → predicate → object, direction, active/historical/superseded/conflicted/candidate status, observed/valid interval, confidence, source evidence; current-tense warning for inactive facts;
- path: rendered route, node IDs resolved to labels, edge kinds, fact IDs, weight, source passages;
- meeting/transcript: speaker, timestamp/sequence, excerpt, meeting link;
- note: navigation metadata and all cited passages from that note.

The header says one of **Cited in this answer**, **Consulted but not cited**, or **Retrieved only**. Selecting a factual sentence with multiple citations shows the set in one inspector stack.

Primary turn state never displays a raw percentage confidence while calibration is unvalidated. Advanced metadata shows the numeric value with “Bootstrap calibration; no held-out metrics” when the active build reports the known degraded state.

#### 7.3.4 Retrieval outcomes

| Contract state | UI treatment |
|---|---|
| accepted | “Evidence policy passed” in disclosure; normal answer |
| fallback | persistent warning “Low-confidence passages were provided; facts and paths were withheld” |
| abstained/no evidence | answer may state insufficiency; evidence ledger says “No evidence cleared the retrieval floor” |
| cancelled | partial marked Stopped; persisted `cancelled`; citations treated as absent |
| interrupted/error | raw partial marked Unvalidated partial; no source support list even if tokens resemble citations |
| stripped claims | disclosure names count and server reason; removed text is not reconstructed into answer |
| busy | composer retains input; inline notice identifies active chat/global turn limit |
| timeout/LLM error | persisted partial if present, Retry and Regenerate rules visible |

Regenerate deletes/replaces the last assistant row server-side and restores retriever pre-turn state. The UI keeps the old rendered answer in a temporary “Replacing…” snapshot until ack, then swaps; failure refetches authoritative history.

#### 7.3.5 Thread management

- Rename is inline with Enter save/Escape cancel and server rollback on failure.
- Delete confirmation states messages/evidence history are permanent; a meeting-linked chat deletion leaves the meeting but removes its usable linked thread according to server behavior on next fetch.
- Background streaming appears as a textual count on the Inquiry coordinate. Changing coordinate or retrieval scope does not cancel it.
- Reconnect refetches every chat with a locally active turn. Since socket-owned engine tasks cancel on disconnect, any persisted partial returns as interrupted/cancelled; the client never resends a user message automatically.

### 7.4 Control → Sources and index workflow

Sources is a ledger, not a grid of integration cards. The index groups **Needs attention**, **Active**, **Off**, and **System**. The workbench row anatomy is 56 px: source glyph, name, exact state/detail, document count, last sync, enable switch, disclosure.

Derived state priority:

1. syncing/cancelling;
2. enabled but unavailable → Needs attention;
3. enabled and available → Active;
4. disabled → Off, with “N documents stored but excluded”;
5. hidden Taurine meeting source → System, always enabled, no switch.

#### Enable

Enabling changes source ledger participation and marks the index stale. The UI offers:

- **Enable and sync** (primary): enable, then start the global sync so all enabled sources produce one coherent build;
- **Enable without syncing**: for deliberate batching; warning remains “Index is stale.”

#### Disable

Disable confirmation explains: stored connector rows are kept, source is excluded immediately, derived knowledge rows are removed/marked stale by current server behavior, and a clean sync is recommended. The action is disabled while that connector or global index mutation is at a non-cancellable critical boundary; otherwise the server response decides.

#### Configure

Known config is rendered by a typed frontend descriptor while unknown keys round-trip unchanged:

| Source | Controls |
|---|---|
| Apple Notes | permission/availability only |
| Obsidian | vault folder chooser; auto-detected value shown |
| Folders | ordered attached-folder list; Add, Remove, Reveal; supported file types/help |
| iMessage | Full Disk Access status; require known contact; advanced DB/contacts paths |
| Claude memories | detected glob, advanced read-only/edit field |
| Codex memories | detected glob list, advanced edit |
| Gmail / Outlook | account authorization, sync window days, contacts source |
| Google Calendar | account authorization, past/future day windows, include system calendars |
| Google Docs | account authorization, modified-within days, maximum documents |

Numeric fields have ranges derived from product constants/descriptors and show server errors inline. Folder selection uses existing `chooseDirectory` IPC and never accepts a fabricated path.

#### Authorize OAuth

OAuth sources use the desktop authorization addition in §13.8: Connect account opens the system browser, workbench shows Waiting/Authorized/Expired/Error, and tokens are stored outside renderer-readable state. The existing Terminal bootstrap command moves to an **Advanced developer setup** disclosure as an environment-credential compatibility path; it is not the primary or only general-user authorization workflow.

#### Sync

Global sync has exact stages from `SyncStatus`: syncing source data, indexing with server `detail`, completed, cancelled, or failed. Connector-local extraction progress uses connector `sync_phase/done/total`. Percent is shown only when total > 0. Cancel calls `/index/sync/cancel`, changes to Cancelling, and waits for terminal state. Cancellation/failure marks the index stale because source snapshots may have advanced.

The old immutable graph remains usable during a build. On successful graph-version update, Atlas fetches and atomically swaps. While visible, the operation ledger distinguishes source-read work from the subsequent knowledge-build work; it collapses after success.

#### Purge data

Delete source data is inside a bordered danger section and requires two clicks. The confirmation includes current document count and states:

- source is disabled;
- cached/source rows and derived knowledge for that source are removed immediately where supported;
- the global index is stale until rebuilt;
- provider data is not deleted;
- there is no undo.

Completion refetches connectors, graph version/status, and affected open objects. An open purged note dossier becomes “Source data removed” with Copy local selection if still available; it does not render stale text as current.

### 7.5 Control → Profile workflow

Profile is labeled **Interpretation profile** with the subtitle “Background Taurine uses to resolve names, projects, and interests. It is not citable evidence.”

States:

- **Empty:** Generate from knowledge or Write manually.
- **Populated:** rendered Markdown with Edit and Generate replacement.
- **Editing:** full-document Markdown editor, dirty/saving/error state; API requires non-empty content.
- **Confirm overwrite:** names that generation replaces the saved profile only on success; current profile remains during work/failure.
- **Generating:** server phase rail (Surveying, Interests, Projects & people, Career & studies, Synthesizing), current activity, nodes read, tool calls, elapsed time. Navigation away does not cancel.
- **Failed:** error and Retry; prior content remains visible under a retained label.

The phase rail uses server state only. No percent is inferred. Manual Save during generation is disabled because generation is single-flight and may replace it; the reason is in the disabled control description. A successful update affects the next chat/live summary without restart.

### 7.6 Diagnostics and session logs

`GET /sessions` and `/sessions/{id}` are preserved under **Control → Diagnostics**, not promoted to ordinary meeting history. The view lists timestamp/session ID and renders structured records with filters for ASR, chunk, retrieval, grounding, summary, chat, control, and error. It avoids duplicating private passage text not present in logs. Copy diagnostics redacts paths/tokens by default and includes runtime/app versions, socket state, build/calibration IDs, and recent safe errors.

Diagnostics explicitly says session logs are operational traces, not the durable meeting record, and may be absent if logging failed.

### 7.7 Native-source opening

Every note can open a Taurine dossier. “Open in source app” appears only when a current adapter supports it. Apple Notes uses the existing `/session` `open_note` control; unsupported sources show Copy text/path/link where the note payload provides it. Failure is an inline notice and does not close the dossier.

## 8. Capability and state inventory

This section is normative. Every listed state must have a rendered representation and a tested transition; generic “something went wrong” may supplement but not replace the specific state.

### 8.1 Runtime, connectivity, and index

| Domain | States | Required representation/recovery |
|---|---|---|
| Sidecar startup | not started, starting phase/detail, ready, fatal error | shell remains; safe detail; elapsed; retry/setup/diagnostics |
| REST | ready, request pending, 503 starting, unreachable, 4xx validation, 404 deleted, 409 conflict, 5xx | retain prior data when safe; targeted retry; no blank entire app |
| Updates socket | connecting, open, reconnecting, closed by startup 1013, failed | ledger state; rehydrate affected domains after reconnect; no duplicate toasts |
| Session socket | connecting, ready, ASR disconnected/connecting/connected/error, reconnecting, resumed, state reset | capture header priority; exact audio-gap warning; reconcile by meeting/session token and snapshot before audio restarts |
| Chat socket | unopened, connecting, open, reconnecting, unavailable | lazy open; preserve composer; refetch active histories; never auto-resend |
| Index | empty, ready, stale, syncing sources, indexing detail, cancelling, failed, completed | old snapshot stays; clear stale consequence; cancel/retry; active build identity in Advanced |
| Graph refresh | no graph, loading, ready, version invalidated, swap pending, fetch failed | atomic swap; retain prior; list/search fallback |
| Calibration health | release-ready, degraded/bootstrap, identity mismatch/unavailable | primary decision categories; advanced warning; runtime mismatch is incident |

### 8.2 Permissions and devices

| Capability | States | Behavior |
|---|---|---|
| Microphone | unknown, promptable, granted, denied, restricted, no device, device lost | start blocks except granted/device; settings/device recovery |
| System audio | unsupported OS, unknown, promptable, granted, denied, helper error, tap lost | degrade mic-only; never label system channel active when lost |
| Call detection | unsupported/helper missing, off, ask, auto, watching, active call, rearm wait, error | settings and tray mirror; one prompt per call |
| iMessage/contacts | available, Full Disk Access missing, DB missing/schema error | source attention with direct remediation |
| Folder/vault | ready, empty config, missing path, permission denied, partial file failures | row-level detail; preserve other valid roots |
| OAuth | disconnected, authorizing, waiting callback, connected, expired/revoked, provider/network error | explicit connect/retry/disconnect; never show token |

### 8.3 Meeting and capture

| State | Allowed primary actions | Prohibited/messaging |
|---|---|---|
| draft | edit, preflight/start, delete | no “recorded” implication |
| preparing | cancel preparation, edit | duplicate start disabled |
| live mic + system | edit, Ask, compact, curtain, stop | new meeting redirects to current |
| live mic only | same + fix system audio | degraded banner persists |
| reconnecting | edit locally, stop locally, retry | “audio may be missing”; no live badge |
| conflict | resolve/copy, stop | autosave paused; no silent overwrite |
| stopped · resumable | Resume, Finalize now, edit | capture is off; 10-second server deadline shown; no new meeting or summary epoch on successful resume |
| ending | navigate, inspect | audio already stopped; no restart same record |
| summarizing | edit, navigate, Ask existing material | generated recap pending; no fabricated progress |
| done | edit/review/accept/reject/Ask/delete | post-edit indexing caveat |
| finalization error | review existing record, copy notes/transcript, open Diagnostics, or ask for a separate recap in the inquiry | current backend has no explicit retry endpoint; UI preserves the record and does not show a nonfunctional Retry |
| deleted elsewhere | close dossier, copy unsaved recovery if any | no further writes |

The UI does not expose a “retry final summary” control until a server contract exists. It may offer “Ask for a recap” in the linked inquiry, clearly distinct from rebuilding `final_summary_md`/suggestions.

### 8.4 Notes and generated edits

| State | Representation |
|---|---|
| pristine/saved | quiet saved timestamp on focus/hover |
| dirty | “Editing” |
| saving | progress text/spinner, editor enabled |
| queued offline | local recovery icon + “Queued locally” |
| conflict | persistent document banner + resolution sheet |
| REST validation/server failure | local draft retained + Retry |
| user-owned text | normal ink |
| AI-authored accepted operation | 2 px violet gutter + “Added by Ask” on focus; normal readable text color |
| pending proposal | durable inserted/replacement diff preview, not canonical editor text |
| accepted/rejected | decision history with actor/time |
| stale/ambiguous anchor | no automatic apply; location workflow |
| undo available | operation-scoped action with clear target |
| undo conflicts | inverse preview, manual confirm |

### 8.5 Chat/inquiry

| State | Representation/recovery |
|---|---|
| no threads / empty local thread | example tasks that fill but do not auto-send |
| history loading / loaded / 404 | skeleton; cached prior; close deleted thread |
| sending before ack | optimistic user text marked Sending; retry only after authoritative check |
| retrieval / thinking / activity / answer | exact phase line; ordered process |
| cancel requested | Stopping; one cancel; wait terminal/socket result |
| done | persisted authoritative message replaces buffers |
| cancelled | partial + Stopped; no cited set |
| interrupted | Unvalidated partial; no support claim |
| busy | composer retained; identify concurrent limit |
| timeout/LLM error | partial if server persisted; refetch; Retry/Regenerate as applicable |
| reconnect | do not replay; refetch active chat; surface interruption |
| rename/delete pending/failure | optimistic label only with rollback; destructive state named |

### 8.6 Evidence and graph

| State | Representation |
|---|---|
| accepted retrieval | text category; advanced score |
| fallback | warning rule and low-confidence label on every passage |
| abstained | no supported evidence, no empty “Sources” rail |
| retrieved/read/cited | separate groups and counts |
| fact active/historical/superseded/conflicted/candidate | text status + icon/pattern; current-tense warning |
| prior-context evidence | “Carried from earlier context,” not current-query result |
| no new live evidence | keep previous summary/context, label no change only in inspector |
| selected node/edge/path | inspector and outline synchronized |
| hidden by LOD/filter | still reachable in outline/search; filter clause visible |
| graph empty/no results/error/stale | §6.9 states |
| idea lens running/result/abstained/error | input preserved and explicitly AI-derived |

### 8.7 Connectors, profile, and utilities

| Domain | States |
|---|---|
| Connector | loading; off; available; enabled active; enabled unavailable; per-source syncing with/without total; error; disable pending; purge confirmation/pending/complete; config dirty/saving/invalid |
| Global sync | idle clean; idle stale; syncing; indexing; cancel requested; cancelled stale; failed stale; complete clean |
| Profile | loading; empty; populated; editing; dirty; saving; overwrite confirm; each server generation phase; failed with preserved prior |
| Call utility | hidden; detected; ask; countdown paused/running; starting; error; dismissed/rearm |
| Context utility | hidden; queued; visible; opened; dismissed; timed out; privacy-redacted |
| Privacy | requested; limited; unavailable; curtain on |
| Main visibility | shown; hidden to tray; compact; full screen; quitting |

## 9. Visual system

### 9.1 Visual direction

The visual metaphor is a technical field notebook crossed with a cartographic instrument: warm mineral paper, cool ink, rules, precise source marks, flat territory tints, and compact annotations. It should feel calm under dense information, not luxurious, playful, or “AI magical.”

- Light is the default; theme follows system after the user chooses Follow system.
- No CSS gradients in application UI, graphs, progress, focus, or generated-state styling.
- No sparkle/star/bull/emoji interface controls. Branding may use the app icon outside operational controls.
- Panels are separated by rules and surface values, not floating glass cards.
- Pills are reserved for immutable identity tokens that read inline (evidence IDs and short source marks). Filters, buttons, status, and navigation are rectangular.
- Shadows are reserved for menus, dialogs, and overlay sheets.

### 9.2 Color tokens

All values below are canonical sRGB hex with equivalent CSS OKLCH. Components consume semantic variables, not raw source colors.

#### Light

| Token | Hex | OKLCH | Use |
|---|---|---|---|
| `--paper` | `#F3F1EB` | `oklch(0.958 0.008 91.5)` | window/canvas base |
| `--surface` | `#FBFAF7` | `oklch(0.985 0.004 91.4)` | index/editor/inspector |
| `--raised` | `#FFFFFF` | `oklch(1.000 0.000 89.9)` | menu/dialog only |
| `--ink` | `#1E2732` | `oklch(0.269 0.024 253.7)` | primary text |
| `--muted` | `#566170` | `oklch(0.489 0.028 256.2)` | secondary text, 5.57:1 on paper |
| `--subtle` | `#788391` | `oklch(0.606 0.025 254.5)` | decorative/large metadata only; never normal body text |
| `--rule` | `#C8CDD3` | `oklch(0.846 0.010 252.8)` | ordinary dividers |
| `--rule-strong` | `#8C96A2` | `oklch(0.669 0.021 252.9)` | controls/graph outlines |
| `--accent` | `#315BCB` | `oklch(0.508 0.180 265.0)` | primary/action/selection |
| `--accent-hover` | `#2548A5` | `oklch(0.434 0.156 264.9)` | hover/pressed |
| `--accent-soft` | `#E1E8FC` | `oklch(0.932 0.028 270.4)` | selected row/region |
| `--focus` | `#1463E6` | `oklch(0.539 0.212 260.7)` | keyboard ring only |
| `--positive` | `#19734B` | `oklch(0.494 0.104 158.8)` | ready/accepted |
| `--warning` | `#956018` | `oklch(0.535 0.107 68.7)` | degraded/stale |
| `--negative` | `#B43B3B` | `oklch(0.527 0.157 24.3)` | error/destructive |
| `--live` | `#A8325A` | `oklch(0.504 0.156 4.1)` | recording state only |
| `--generated` | `#6555A5` | `oklch(0.504 0.124 290.4)` | AI-authored/proposed |

White on accent is 6.02:1. Ink/muted/accent/status colors meet normal-text contrast on their intended light surfaces. `--subtle` does not and is restricted accordingly.

#### Dark

| Token | Hex | OKLCH | Use |
|---|---|---|---|
| `--paper` | `#151A20` | `oklch(0.215 0.014 253.1)` | base |
| `--surface` | `#1D232B` | `oklch(0.254 0.018 255.7)` | panes |
| `--raised` | `#252D37` | `oklch(0.294 0.021 254.2)` | overlay |
| `--ink` | `#F1F3F5` | `oklch(0.963 0.003 247.9)` | primary |
| `--muted` | `#AAB2BD` | `oklch(0.761 0.018 256.3)` | secondary |
| `--rule` | `#414B57` | `oklch(0.409 0.024 253.1)` | divider/control |
| `--accent` | `#8EAAFF` | `oklch(0.751 0.127 269.3)` | primary/action |
| `--accent-soft` | `#283657` | `oklch(0.337 0.061 265.8)` | selected |

Status hues in dark mode are separate lighter variants verified against `--paper`; they are not produced with opacity. Canvas territory fills use solid derived tokens precomputed for each theme.

#### Source categorical tokens

Source color is always accompanied by glyph and label. These tokens apply to marks and graph nodes, not body text:

| Source | Color | Shape/glyph |
|---|---|---|
| Apple Notes | `#B46B00` | folded sheet |
| Obsidian | `#7352B8` | faceted diamond |
| iMessage | `#187854` | speech square |
| Gmail | `#B43B3B` | envelope M |
| Outlook | `#23699C` | divided envelope |
| Google Calendar | `#2F7D67` | calendar square |
| Google Docs | `#315BCB` | lined document |
| Folders | `#66717E` | folder tab |
| Claude memories | `#8B5A2B` | C document |
| Codex memories | `#34404C` | terminal document |
| Taurine meetings | `#9B4763` | transcript diamond |

### 9.3 Typography

Bundle only WOFF2 weights actually used:

- `IBM Plex Sans` 400/500/600: application, documents, editor;
- `IBM Plex Mono` 400/500: evidence IDs, times, hashes, code, diagnostics.

Fallbacks are `-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif` and `ui-monospace, "SFMono-Regular", Consolas, monospace`. No remote font request is allowed.

| Style | Size/line | Weight | Use |
|---|---|---:|---|
| display | 28/34 | 500 | empty/setup headings only |
| title | 20/26 | 600 | dossier title/view title |
| section | 15/20 | 600 | section heads |
| body/editor | 14/21 UI; 16/25 editor | 400 | reading/content |
| control | 13/18 | 500 | buttons/rows/inputs |
| metadata | 12/16 | 400/500 | source/date/status |
| micro | 11/14 | 500 | evidence IDs, graph annotations; never long prose |

Tabular numerals are enabled for time, counts, confidence, and progress. Headings use sentence case. All-caps is limited to two-to-four-character evidence prefixes in monospace.

### 9.4 Spacing, geometry, elevation, motion

Spacing scale: `2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64` px.

- pane padding: 16 compact / 20 comfortable;
- document column horizontal padding: 24, reduced to 16 below 1180;
- row gaps: 6; section gaps: 20; document section gaps: 28;
- control radii: 4 px; menus/dialogs: 6 px; evidence ID token: 3 px; panels: 0 px;
- borders: 1 px; selected left rule: 3 px; focus ring: 2 px plus 2 px offset;
- popover shadow: `0 8px 24px rgba(30,39,50,.16)`; dialog adds `0 2px 6px rgba(30,39,50,.10)`; no card shadows;
- motion durations: 80 ms press, 140 ms disclosure/selection, 220 ms sheet/layout, `cubic-bezier(.2,.8,.2,1)`;
- loading spinner: 800 ms linear rotation; live mark: static filled center plus text, not pulsing by default;
- reduced motion: duration 0–1 ms, no smooth pan/zoom, no spinner rotation (use changing text/dots no faster than 1 Hz), no node-entry transition.

### 9.5 Density modes

Compact is default for Track B; Comfortable is a user setting.

| Element | Compact | Comfortable |
|---|---:|---:|
| object row | 40 | 48 |
| source row | 52 | 60 |
| toolbar/button | 32 visual, 40 hit | 36 visual, 44 hit |
| input | 34 | 40 |
| menu item | 32 | 40 |
| transcript block vertical padding | 6 | 10 |

Every pointer target contains at least a 24 × 24 CSS px square, the WCAG 2.2 AA minimum; primary/destructive/meeting controls target 40–44 px. The W3C rationale and exceptions are documented in [Understanding Target Size (Minimum)](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum).

## 10. Component system and exact behavior

### 10.1 Primitives

| Component | Measurement/behavior |
|---|---|
| `Button` | 34 px compact/40 comfortable; padding 0 12; primary, secondary, quiet, danger; label required except toolbar icon with tooltip/accessibility name |
| `IconButton` | 32 visual inside 40 × 40 hit; icon 16; 4 px radius |
| `TextField` | 34/40 high; persistent label outside except locator; error below; clear control 40 hit |
| `SearchField` | 36 high; search icon, query, result count, clear; `/` hint only when empty |
| `Checkbox` | 16 square inside 32/40 row; text label handles click |
| `Switch` | 32 × 18 inside 40 target; explicit On/Off accessible value; never sole status indicator |
| `Select/MenuButton` | rectangular 34/40; native-like keyboard; menu max 320 high then scroll |
| `Menu` | 6 radius, 8 outer padding, 32/40 rows, destructive group separated by rule |
| `Tooltip` | text only, 400 ms pointer delay, immediate keyboard; dismissible; never carries essential-only content |
| `InlineNotice` | 1 px left status rule, icon, title/detail/actions; no tinted full card unless error needs contrast |
| `Dialog/Sheet` | native `<dialog>` or proven focus trap; heading/description; Escape unless destructive operation committed; origin focus restore |
| `Toast` | transient success only; errors also persist at point of action; max 3; pause on hover/focus |
| `Progress` | determinate only with real done/total; 2 px flat bar plus text; indeterminate text/spinner |
| `Skeleton` | static blocks in reduced motion; no gradient shimmer |
| `Divider` | 1 px visible/12 hit; focusable resize semantics and keyboard alternatives |

### 10.2 Layout and navigation components

- `CommandDeck` 48 px, platform drag regions explicitly exclude controls.
- `CoordinateRail` 52 px; each coordinate is 48 px high; Control is visually separated at the bottom; badges are text/count with accessible labels.
- `IndexPane` virtualizes after 100 rows; sticky header 48; optional filter summary 36.
- `OperationLedger` 28 px when present; absent from layout when healthy/idle unless pinned; entries are buttons only when detail/action exists.
- `ContextShelf` 44 px collapsed, 84 px when object subtitles need display; pin items 180–260 px, square-edged with separators.
- `InspectorDock` uses sections with 32 px disclosure headers; selected evidence header stays sticky.
- `ComparisonBoard` gaps 1 px rule, not floating cards; columns min 320, synchronized optional scroll toggled explicitly.

### 10.3 Knowledge components

#### `ObjectRow`

40/48 px, 20 px type/source glyph, two-line area only in comfortable mode, date/status right. Selected uses `--accent-soft` plus 3 px bar. Focus uses ring independent of selection. Hover actions duplicate in context menu and appear on keyboard focus.

#### `SourceMark`

Inline 18 px glyph + source label. Compact evidence rows may show a 14 px glyph with an accessible offscreen label. Color alone is never sufficient.

#### `EvidenceRef`

Inline rectangular token, mono 11/14, 3 px radius, 1 px border. Prefix shape varies: E square, F arrow-corner, P linked corners, T transcript rule, M meeting diamond. It behaves as a link/button and exposes full type in accessible name.

#### `EvidenceLedger`

Three ordered groups: Cited, Consulted, Retrieved. Cited opens by default; others collapsed. Duplicates appear once at their highest state, with a note “also consulted/retrieved” only in metadata. Counts are computed from IDs, not payload row count.

#### `DecisionLabel`

Text + icon: Evidence policy passed, Low-confidence fallback, No supported evidence, Interrupted/unvalidated, or Unknown contract. It never says Verified. Numeric confidence is Advanced only.

#### `FactRow`

Subject, directional arrow with text accessible name, predicate, object; temporal state/date second line; source refs. Conflicted/historical use pattern/icon plus word.

#### `PathView`

Linear ordered route with nodes and labeled edges; a compact list alternative at widths <420. No force graph inside inspector.

#### `DiffProposal`

Unchanged context in muted ink, insertion with 2 px generated gutter and `+` marker, never green-only; rationale and base revision; Accept/Reject 40 px actions; Undo after decision. Pending generated text is excluded from canonical editor serialization.

### 10.4 Meeting components

- `CaptureHeader` 44 px, Stop is danger-secondary with text; live status always textual.
- `TranscriptLane` virtualized after 300 final blocks; overscan 10; final/interim separate nodes.
- `NotesDocument` maximum text width 760; toolbar 36; slash menu searchable and keyboard complete.
- `RollingSummary` flat document section with revision/timestamp; citation links.
- `ActionRow` minimum 44; action text first; status/actions second at narrow width.
- `ContextQueue` source-marked rows; no utility notification is the sole copy.
- `FinalRecord` strata use 1 px section rules and 28 px spacing.

### 10.5 Ask components

- `InquiryComposer` fixed to workbench bottom but uses scroll padding so it never obscures focused content, satisfying the focus visibility intent described in [WCAG 2.2 Focus Not Obscured](https://www.w3.org/WAI/WCAG22/Understanding/focus-not-obscured-minimum).
- `AssistantTurn` is a document section, no speech bubble/card/shimmer.
- `ProcessTrace` shows server activity; model reasoning is Advanced/collapsed.
- `MessageMarkdown` owns safe note links/evidence refs, code copy, lists/tables, and streaming batching.
- `JumpToLatest` is 40 px high and participates in focus order only while visible.

### 10.6 Icons and illustrations

Use one outlined icon set (`lucide-react`) with 1.75 px stroke, 16 px controls, 20 px coordinate rail, 24 px empty-state maximum. Connector glyphs are custom monochrome SVGs normalized to 20 px. Icons with no adjacent text always have tooltips and accessible names. Empty states use typography, rules, and a small diagram only when it teaches the Atlas model; no generic AI illustrations.

## 11. Keyboard, accessibility, and internationalization

### 11.1 Global shortcut map

Shortcuts are registered through one command registry and shown in menus/command center. Platform labels use `⌘` on macOS and `Ctrl` elsewhere.

| Shortcut | Command |
|---|---|
| `⌘1` / `⌘2` / `⌘3` | Atlas, Chronicle, Inquiry coordinates |
| `⌘4` / `⌘5` | open Control directly at Sources / Profile |
| `⌘,` | open Control at its last section |
| `⌘K` | command center |
| `⌘P` | quick open object |
| `⌘N` | new object for current coordinate (draft meeting or inquiry); Atlas opens Inquiry |
| `⌘[` / `⌘]` | object/context back/forward |
| `⌘B` | toggle index |
| `⌘\` | toggle inspector |
| `⌘⇧M` | start meeting preflight; while live, stop immediately and expose 10-second Resume |
| `⌘⇧L` | enter/exit compact live while live |
| `⌘⇧P` | privacy curtain |
| `⌘.` | cancel focused active inquiry or global sync; never stops recording |
| `F6` / `Shift+F6` | cycle focus zones |
| `Escape` | dismiss top transient layer/clear transient selection |
| `Space` | Quick Look selected object in lists/outline |
| `Enter` | open/activate selected item |
| `Shift+Enter` | pin/unpin selected object |
| context-menu key / `Shift+F10` | full context menu |

Editable text contexts take precedence for standard editing shortcuts. Single-letter Atlas shortcuts operate only when canvas/outline owns focus and no text field/editor is active. Users can disable single-character shortcuts to satisfy WCAG 2.1.4.

### 11.2 Coordinate-local keys

- Atlas canvas/outline: `/` search, `V` Canvas/Outline, `F` fit, `+`/`-` zoom, arrows pan/navigate, `[`/`]` neighborhood history, `1`/`2`/`3` depth when neighborhood controls own focus.
- Chronicle index: Up/Down select, Enter open, `⌘Delete` opens meeting deletion confirmation.
- Transcript: `J`/`K` next/previous final block only when transcript owns focus and character shortcuts enabled; `A` opens the Insert transcript excerpt action menu.
- Inquiry: Enter send and Shift+Enter newline in composer; `⌘.` stop; `⌘Shift+R` regenerate last only after confirmation if focus is in turn actions.
- Control → Sources: Space toggles row disclosure, not enable; enable switch requires explicit focus/activation.

### 11.3 Semantics and announcements

- App regions use landmark labels: Knowledge coordinates, Index, Workbench, Evidence inspector, Context shelf, Operations.
- The coordinate rail uses navigation links/buttons with current state; it is not ARIA tabs because a coordinate changes how the same persistent context is represented rather than selecting one panel from a tablist. Control is a separate settings entry.
- Virtualized lists retain correct `aria-setsize`/`aria-posinset` or use a semantic table/treegrid where needed.
- Live transcript does not put all deltas in a global live region. A polite announcement says “New final transcript from Them” at a user-configurable cadence, default off. Critical capture errors are assertive once.
- Streaming answer announces phase changes and completion, not every token. The finished answer receives a heading/focus target without moving focus automatically if the user is elsewhere.
- Save/sync statuses are polite and coalesced; repeated reconnect ticks do not spam.
- Canvas has an accessible Outline equivalent (§6.8). Pointer hover panels are also available on focus and dismissible without moving the pointer, satisfying hover/focus content requirements.
- Color, opacity, motion, or position never carries the only state. Every live/error/generated/fallback/historical distinction has text/icon/pattern.
- Focus is never entirely obscured by sticky command/composer/status surfaces. Workbench uses `scroll-padding-block: 64px 96px`; overlays make background inert.

### 11.4 Contrast, zoom, reflow, and targets

- Normal text ≥4.5:1; large text ≥3:1; control boundaries/focus/meaningful graphics ≥3:1 against adjacent colors.
- Text supports 200% browser zoom and OS text scaling without clipped actions. The shell collapses panes; no horizontal page scroll except code/table/graph workbench where explicitly appropriate.
- User-generated Markdown tables scroll inside a labeled region and have a linearized copy/reader option.
- All controls meet WCAG 2.2 24 × 24 minimum; primary controls aim for 40–44. Inline evidence links use the inline exception but also appear as ≥40 px rows in the inspector.
- Focus ring is 2 px `--focus`, 2 px offset, and never removed based on pointer modality when `:focus-visible` applies.
- Drag operations have click/keyboard alternatives, including split resizing, Atlas panning/positioning, time range, and context collection.

### 11.5 Editor accessibility

Before release, BlockNote is tested with VoiceOver + Safari/Chromium accessibility tree and NVDA + Electron/Chromium for:

- heading/list semantics and reading order;
- slash menu announcement and keyboard selection;
- undo/redo and selection preservation;
- pending proposal exclusion from canonical document and accessible diff reading;
- provenance gutter not interrupting sentence reading;
- focus restoration after inline Ask and proposal actions.

If a blocking defect cannot be fixed, the release uses a textarea/Markdown source mode as an equal, persistent editing mode rather than shipping an inaccessible WYSIWYG-only editor.

### 11.6 Internationalization

- All user-facing strings live in namespaced message catalogs; JSX concatenation of sentence fragments is prohibited.
- Lists, plurals, elapsed durations, relative dates, absolute dates, numbers, and percentages use `Intl` formatters.
- Source/brand names remain proper nouns; status/action text is localized.
- Layout tolerates 35% string expansion. Buttons size to text up to reasonable max and wrap in dialogs; destructive action labels never ellipsize.
- RTL mirrors index/inspector order, directional chevrons, and text alignment. Graph factual direction remains semantically explicit through labels; it is not blindly mirrored as data direction.
- Evidence identifiers (`E3`, `F2`, `P1`, `T4`, `M2`), raw hashes, and code remain LTR-isolated.
- Keyboard display uses platform conventions and localized key names; physical command meanings stay stable.
- Transcript speaker labels You/Them/Speaker N are localized at presentation only; stored channel IDs and evidence remain unchanged.
- User/profile/note Markdown carries the document's language; Taurine does not translate authored text implicitly.

## 12. Frontend state and side-effect architecture

### 12.1 Architectural decision

Use **Redux Toolkit** as the single new state dependency, including `createEntityAdapter`, listener middleware, and RTK Query for REST. Do not add a second global state library or a router solely to model three coordinates plus Control.

Reasons:

- normalized meeting/chat/note metadata and explicit operation IDs are a better match than nested component state;
- listener middleware can sequence start/stop/save/reconcile without putting side effects in views;
- RTK Query owns fetch caching/invalidation while sockets patch or invalidate the same authoritative records;
- reducers and selectors provide deterministic tests for the many lifecycle states;
- Redux DevTools can be disabled/redacted in production so private content is not exposed accidentally.

The store separates **server entities**, **operational state**, and **workspace state**. Draft editor documents remain in feature-local editor adapters; Redux stores only the current Markdown snapshot/revision needed for coordination, not editor internals. One `MeetingDocumentController` per mounted dossier is the sole writer to canonical meeting Markdown. Editor input, transcript insertion, proposal acceptance, conflict resolution, and undo all enter that controller as typed commands; views and socket handlers cannot call `setMarkdown` or PATCH notes directly. A `ProposalQueue` adapter hydrates durable proposals but never injects their content into the editor until Accept succeeds. Cross-reload recovery drafts are written through main-process `safeStorage` IPC. If OS encryption is unavailable, recovery is memory-only and the save state says “Recovery unavailable after app restart”; Taurine never silently writes transcript/notes to a new plaintext recovery store.

### 12.2 Domain slices

| Slice/API | Owns | Does not own |
|---|---|---|
| `runtime` | startup, REST reachability, three socket states, ASR/index/sync status, versions/capabilities | feature entities |
| `workspace` | active coordinate/Control section, history, selection, index/inspector widths/open state, pins, compare, density/theme | meeting/chat lifecycle |
| `meetings` | normalized summaries/details and durable proposals, active meeting ID, capture finite state/session token, transcript buffers, summaries/actionables/context queue, note operation metadata | WebSocket implementation/editor nodes |
| `inquiries` | chat metadata/messages, active thread, per-chat stream reducer, composer drafts/scopes | socket object |
| `atlas` | graph version/snapshot, layout positions, filters, timeline, selection/ego/idea/session overlay | retrieval engine data mutation |
| `sources` | connector entities, typed config drafts, authorization operations | active graph build internals |
| `profile` | card, edit draft, generation status | chat/live prompt injection |
| `privacy` | content-protection report, curtain, notification redaction | secret/private content copies |
| `notifications` | call prompt, suggestion queue, toasts/notices | main-process window handles |
| `sidecarApi` | REST endpoints/cache tags | socket stream state |

Selectors derive composite UI state (`meetingCanStart`, `sourceDisplayState`, `evidenceGroups`, `privacyCopy`) in one place. Components do not reconstruct status priority ad hoc.

### 12.3 Finite state models

#### Capture

```text
idle → preparing → live_full | live_mic_only
preparing → failed → idle
live_* → reconnecting
reconnecting → live_* | ending | failed
live_* | reconnecting → stopped_resumable
stopped_resumable → live_* | ending
ending → finalizing → done | finalization_error
```

Events carry `meetingId`, `clientSessionId`, `attemptId`, and `connectionEpoch`; stale async results are ignored when identifiers differ. `state_reset` is a Start acknowledgement disposition handled during `reconnecting`, not a dangling capture state: it clears ephemeral state and then enters the returned durable/live status. `stopRequestedAt` makes Stop idempotent. `resumeDeadline` exists only in `stopped_resumable`; expiry dispatches one Finalize transition and is reconciled against the server snapshot rather than trusted as a local clock authority.

#### Chat turn

```text
local_pending → acknowledged → retrieving → thinking/activity/answering
→ stopping → cancelled
→ done
→ interrupted | failed
```

Thinking, activity, and answer deltas may interleave after retrieval, so the state stores an ordered process-event array plus separate content buffers rather than assuming a single linear visual phase. Only `chat_done.message` becomes authoritative complete content.

#### Note save

```text
clean → dirty → saving → clean
dirty/saving → queued_offline → saving
saving → conflict | failed
conflict → resolving → dirty | clean
```

Every request carries local `saveId`, server `baseNotesRev`, and content hash. A later local edit prevents an earlier success from marking the document clean.

### 12.4 Connection supervisor

`SidecarGateway` is a service object created once outside React. It owns:

- one `UpdatesConnection`, opened after runtime ready;
- one `SessionConnection`, opened after runtime ready because ASR/index status and control are global;
- one lazy `ChatConnection`, opened on first Ask/meeting inquiry and retained;
- capped exponential backoff with jitter and visibility/network wakeups;
- typed parse/dispatch adapters using exhaustive discriminated unions;
- connection epochs so stale events from replaced sockets are rejected;
- one outbound policy per protocol.

Outbound policy:

- Audio is never queued or replayed. Binary sends only while the current session socket epoch is open and capture state is live.
- Session `start` may replay only with the same `client_session_id` through the reconciliation procedure in §7.2.4; generic socket reconnect does not blindly send it.
- The latest `notepad` state is coalesced to one pending value and sent after meeting/start reconciliation. During stopped-resumable grace it is allowed to update the suspended session; finalization claims the freshest persisted notes plus suspended notepad atomically.
- Chat sends are never automatically replayed after write uncertainty. Refetch history first.
- Cancel/stop/resume controls are idempotent at the frontend and sent at most once per epoch/operation; server token ownership resolves deadline races.

### 12.5 Hydration and reconciliation

On runtime ready:

1. Open updates/session sockets.
2. Fetch lightweight lists/status in parallel: meetings, chats, connectors, profile, graph metadata/status.
3. Restore the locally active coordinate/Control section, history, and pins only for objects that still exist; invalid pins become removed-item notices.
4. Lazy-fetch active meeting/chat/note details.
5. Load Atlas graph only when Atlas is first visible or a restored pin requires it.
6. Restore a renderer recovery draft only after comparing meeting ID, saved base revision, and content hash; prompt if it differs.

On `/updates`:

- connectors payload upserts normalized rows;
- sync payload replaces runtime sync status;
- graph version invalidates graph query but old snapshot remains until replacement succeeds;
- profile payload replaces card/status unless a local edit is dirty, in which case it becomes a conflict notice;
- chats payload is consumed (the current app ignores it) and updates metadata without replacing fetched messages;
- meetings payload updates list metadata/status; detail is refetched for the active/changed meeting when status, revision, or timestamps indicate need.

After any updates-socket reconnect, refetch all six domain summaries rather than assuming missed invalidations can be reconstructed.

### 12.6 Optimistic writes

Allowed optimistic operations:

- workspace-only pins/filter/layout;
- chat/meeting title, with visible Saving and rollback/refetch;
- actionable status, with operation ID and server-event reconciliation;
- source enable switch only after consequence confirmation, with Pending label and rollback.

Not optimistically destroyed:

- chat/meeting deletion;
- source purge;
- accepted proposal content;
- profile overwrite;
- global sync completion.

Delete rows remain disabled with “Deleting…” until success, then leave the list. This prevents focus jumping and false undo expectations.

### 12.7 Privacy in state and logs

- Production Redux DevTools are off.
- Error breadcrumbs contain endpoint/type/IDs/status/duration, never note, transcript, profile, passage, prompt, or answer text.
- Analytics, if later added, are off by default and cannot include content/source object IDs without a separate consent design.
- Utility-window IPC payloads are bounded to necessary title/snippet and cleared after dismissal; curtain substitutes generic payloads.
- Clipboard actions are explicit and show success locally; clipboard contents are not retained by app state.

### 12.8 Performance and code splitting

- `AppShell`, runtime, workspace, Meetings list, and basic source/profile skeleton are initial.
- Lazy chunks: `atlas` (force graph + renderer + worker), `editor` (BlockNote), `markdown` (ReactMarkdown/render plugins), `inquiry`, `sources-config`, `diagnostics`.
- Preload editor chunk when a meeting row receives intent/hover for 250 ms; preload Atlas after runtime idle only if memory pressure allows.
- CSS is feature-scoped and imported with its lazy entry; `main.css` is replaced by tokens/reset/shell essentials plus feature files.
- Graph layout/render calculations run in worker/canvas; React receives stable snapshots no faster than animation frame.
- Transcript, object lists, evidence ledgers, and long inquiry histories virtualize; streaming answer itself stays non-virtual until complete for selection stability.
- RTK selectors are memoized; high-frequency transcript/audio level actions use external buffered stores and 10 Hz UI snapshots rather than dispatching per audio frame.

## 13. Backend and desktop contract mapping

### 13.1 Existing REST contracts retained

The redesign consumes existing sidecar contracts rather than renaming them to match frontend language.

| Contract | Product use | Reconciliation/important semantics |
|---|---|---|
| `GET /startup` → `StartupStatus` | readiness shell | only endpoint assumed before ready; detail/error/elapsed are authoritative |
| `GET /graph` → `GraphResponse` | global Atlas | nodes already include source/gist/dates/folder/community/value/entities; edges include type/shared/fact/provenance/mechanisms; clusters are territories |
| `GET /graph/local?center&depth` | neighborhood depth 1–3 | 404/empty handled in selected context; depth clamped server-side |
| `POST /graph/idea` | AI idea lens | input/result ephemeral; 503 distinct from abstained/empty |
| `POST /graph/edge-reason` | not called from projected-edge UI | request requires exact passage IDs; current `GraphEdge` exposes note endpoints only, so the renderer must not guess or substitute IDs |
| `GET /facts?ids=` | graph fact drilldown | maximum 50; current payload is less rich than chat evidence facts, so inspector labels only supplied fields |
| `GET /notes/{note_id:path}` → `NotePreview` | dossier/Quick Look | contains source/gist/text/neighbors; path ID must remain URL-encoded safely |
| `POST /index/sync`, `GET /index/sync`, `POST /index/sync/cancel` | global source read + clean build | 409 means already running; cancellation cooperative; stale remains on failure/cancel |
| `GET /connectors` | visible source ledger | hidden Taurine connector excluded by server |
| `POST /connectors/{id}/enable` | include source | marks stale; does not read/index |
| `POST /connectors/{id}/disable` | exclude source | removes derived source rows immediately where possible and marks stale; source documents retained |
| `POST /connectors/{id}/data/delete` | purge | conflicts while sync runs; disables/purges source and derived data, marks stale |
| `POST /connectors/{id}/sync` | one-source sync/rebuild | separate per-connector task; updates connector progress and graph version |
| `PUT /connectors/{id}/config` | typed source config | entire flat config; frontend preserves unknown keys |
| `GET /profile`, `PUT /profile`, `POST /profile/generate` | interpretation profile | PUT requires non-empty; generate 409 single-flight; old card survives failure |
| `GET /summary` | current rolling summary recovery while live | returns empty if no current live session; never substitutes for a meeting final record |
| `GET/POST /chats` | inquiry list/manual empty chat when needed | primary UI still creates on first send; list may update through socket |
| `GET/PATCH/DELETE /chats/{id}` | history/rename/delete | full `ChatMessage` is enough to rebuild completed turns |
| `GET/POST /meetings` | chronology/new draft | list is summary rows; create title optional |
| `GET/PATCH/DELETE /meetings/{id}` | dossier/title/notes/current suggestion compatibility/delete | current PATCH increments notes revision on content; existing suggestion patch may change status only, whole-list positional |
| `GET /sessions`, `GET /sessions/{id}` | diagnostics | raw JSONL records, not product meeting history |

The target UI does not expose the legacy `GET /summary` as a global “Data” card. It is only a reconciliation helper for an active session.

### 13.2 Existing `/updates` WebSocket

The socket sends an initial snapshot for all kinds, then coalesced latest values:

| Event | Consumption |
|---|---|
| `{type:"connectors", connectors}` | upsert source ledger; reconcile config drafts carefully |
| `{type:"sync", status}` | replace global sync/index stale operation state |
| `{type:"graph", version}` | invalidate/fetch/swap graph snapshot |
| `{type:"profile", card, generation}` | profile card/generation unless local edit conflict |
| `{type:"chats", chats}` | update inquiry index; this is newly consumed by target UI |
| `{type:"meetings", meetings}` | update chronology and identify active detail refetches |

There are no durable monotonic event IDs. A reconnect therefore triggers full summary refetches; it cannot claim gapless event replay.

### 13.3 Existing `/session` WebSocket

#### Client to sidecar

- binary `[channel u8][PCM s16le 16 kHz mono]`; channel 0 mic, channel 1 system;
- current `start {channels, meeting_id}` and `stop` are extended in the completed target with `client_session_id` and resume semantics from §13.7;
- `notepad {meeting_id,text_md}`;
- `actionable_status {id,status}`;
- `open_note {note_id}`;
- `inject_transcript` exists only when test controls are enabled and never appears in production UI.

#### Sidecar to client

The target parser supports every schema event, including older ones still emitted by test seams:

- `status`: ASR state plus index/sync detail;
- `transcript`: channel/text/final/time/sequence;
- `grounding`: accepted decision, evidence pack/display set, revision, no-new-evidence/new-support, trace;
- `summary`: rolling summary Markdown/highlights/actionables/grounding;
- `actionable`: new/update/dismiss;
- `note_suggestion`: note/source/score/snippet/time;
- additive `session_snapshot`: new/resumed/state-reset/stopped-resumable bounded state from §13.7;
- `final_summary_status`: running/done/error;
- `final_summary`: meeting/title/Markdown/base revision/suggested edits;
- legacy ambient/pin/unpin/retrieval/trigger events remain parseable for existing test and diagnostic paths even when the new primary layout does not recreate the old card feed.

Session events are applied only to the current `connectionEpoch` and matching meeting where applicable. Status events without meeting ID are operational socket state, not proof that a particular meeting is bound.

### 13.4 Existing `/chat` WebSocket

#### Client controls

- `chat_send {chat_id:""|id,text,seed_note_ids,meeting_id?}`;
- `chat_cancel {chat_id}`;
- `chat_regenerate {chat_id}`.

#### Server events

| Event | Target behavior |
|---|---|
| `chat_ack` | replace local send with persisted user/assistant IDs; new chat metadata |
| `chat_retrieval` | store full `EvidencePackPayload`, retrieved and consulted IDs before answer |
| `chat_thinking` | append Advanced reasoning buffer; announce phase only |
| `chat_activity` | replace step by stable index; use kind/label/status plus tool/arguments/evidence IDs/latency |
| `chat_notes_edit` | upsert/refetch the server-persisted proposal; Accept may later enter the document controller, never raw string mutation in the view |
| `chat_delta` | append batched answer buffer |
| `chat_done` | replace with authoritative persisted `ChatMessage`; group cited/consulted/retrieved; show usage/interruption/stripped claims |
| `chat_error` | busy/llm/timeout exact state; refetch partial when message ID exists |
| `chat_meta` | upsert thread title/timestamps |

The persisted `ChatMessage` fields `original_query`, `rewritten_query`, `retrieval_decision`, `retrieval_confidence`, `active_build_id`, `activity`, `usage`, `cancelled`, `interrupted`, and `stripped_claims` are all represented in the target inspector/process UI. They are not discarded by a narrow frontend parser.

### 13.5 Desktop/preload contracts retained and extended

Retained:

- directory chooser;
- call-prompt show/dismiss/start/detection/listening IPC;
- context-suggestion show/open/dismiss IPC;
- system-audio supported/start/stop/chunks;
- micwatch reporting;
- hide/show/tray/window behavior.

Additive main/preload contract required by the target UI:

```text
desktop.capabilities() -> {
  platform,
  microphone: unknown|granted|denied|restricted,
  systemAudio: unsupported|unknown|granted|denied|error,
  callDetection: supported|unsupported|error,
  contentProtection: requested_limited|requested|unavailable,
  appVersion
}
desktop.requestPermission("microphone"|"systemAudio")
desktop.openSystemSettings("microphone"|"systemAudio"|"fullDiskAccess")
desktop.setPrivacyCurtain(boolean)
desktop.setCompactLive({enabled, alwaysOnTop?})
desktop.openExternal(url) // allowlisted http/https authorization/help only
desktop.recoveryDraft.read(meetingId)
desktop.recoveryDraft.write({meetingId, baseNotesRev, contentHash, notesMd})
desktop.recoveryDraft.delete(meetingId)
desktop.liveSession.read() // -> {meetingId, clientSessionId, channels} | null
desktop.liveSession.write({meetingId, clientSessionId, channels})
desktop.liveSession.delete()
```

The renderer cannot send arbitrary shell commands, paths, or URLs. Permission state is advisory and must be rechecked by the actual capture/source operation. The live-session record is bounded metadata, written before Start and cleared after finalization/deletion; it lets a renderer reload present the same token during the 60-second resume window. Recovery draft content remains `safeStorage`-protected as specified in §12.1.

### 13.6 Minimal backend addition A: one durable proposal boundary and revision-aware note commands

The current full-string PATCH, transient `chat_notes_edit`, and whole-list suggestion status patch cannot guarantee that every AI-authored change remains a reviewable proposal, nor can they provide atomic accept/reject, conflict-safe insertion, or operation-scoped undo. The target therefore has one universal boundary:

```text
AI output → durable MeetingNoteProposal → explicit Accept/Reject
explicit Accept → MeetingDocumentController → atomic note command → canonical Markdown
```

No live summary, actionable, meeting chat, inline Ask, or final-summary path bypasses it. A direct user action such as typing, pasting, or choosing **Insert transcript excerpt** may submit an atomic user-authored command without becoming an AI proposal.

#### Revision-aware manual save

Extend `MeetingUpdateRequest` with `base_notes_rev`, structurally optional only because title-, suggestion-, or status-only PATCH requests do not write note content. If `notes_md` is present and `base_notes_rev` is absent, the server returns HTTP 422. The target frontend always supplies it for a note-content write.

- If current revision equals base, update and return meeting.
- If it differs, return HTTP 409:

```json
{
  "detail": "notes_conflict",
  "meeting": { "meeting_id": "…", "notes_md": "…", "notes_rev": 14 }
}
```

No caller receives last-write-wins behavior for `notes_md`. Revision enforcement is part of the backend contract before the target renderer begins issuing content writes.

#### Durable proposal

Meeting detail adds `note_proposals: MeetingNoteProposal[]`:

```ts
type MeetingNoteProposal = {
  proposal_id: string
  meeting_id: string
  origin: "final_summary" | "meeting_chat" | "inline_ask" | "rolling_summary" | "actionable"
  base_notes_rev: number
  anchor: string | null
  occurrence: number | null
  delete_md: string
  insert_md: string
  rationale: string
  supersedes_proposal_id: string | null
  source: {
    chat_id?: string
    message_id?: number
    summary_revision?: number
    actionable_id?: string
    evidence_ids?: string[]
  }
  status: "pending" | "accepted" | "rejected" | "superseded"
  created_at: number
  decided_at: number | null
  decision_command_id: string | null
}
```

Add `POST /meetings/{meeting_id}/note-proposals` for renderer-originated inline, rolling-summary, and actionable proposals. The request is the type above without server-owned `meeting_id`, `status`, `created_at`, `decided_at`, and `decision_command_id`; `proposal_id` is required and idempotent. Exact replay returns the stored proposal, conflicting reuse returns 409, a nonexistent/foreign source returns 422, and no notes revision changes. `anchor:null` means an explicit append proposal and requires `occurrence:null`/`delete_md:""`; an anchored proposal requires a positive occurrence. If `supersedes_proposal_id` is present, the old pending proposal belongs to the same meeting and is marked superseded in the create transaction.

Meeting chat does not rely on the renderer to persist a socket event: the chat engine calls the same proposal service before emitting additive `chat_notes_edit {chat_id, message_id, proposal}`. Final-summary generation materializes its edits and “Add recap” as proposals in the same transaction that stores the final record. Existing stored `suggestions` remain readable through a deterministic compatibility adapter and appear in meeting `note_proposals` with `origin:"final_summary"`; this is an on-read contract adapter, not a second UI or a data migration program.

The UI does not label generated content **Proposal** or enable Accept until the durable proposal response/event exists. A failed proposal write leaves canonical notes unchanged and offers Retry creation or Copy text.

#### Atomic note command

Add `POST /meetings/{meeting_id}/note-commands`:

```jsonc
{
  "command_id": "uuid",
  "base_notes_rev": 14,
  "kind": "insert|replace|proposal_decision|undo",
  "anchor": "exact markdown substring",
  "occurrence": 1,
  "delete_md": "",               // replace only; exact text after anchor
  "insert_md": "…",              // insert/replace
  "proposal_id": "…",          // required for proposal_decision
  "decision": "accept|reject",   // proposal_decision only
  "target_command_id": "…"       // undo only
}
```

Response:

```jsonc
{
  "meeting": {"meeting_id":"…", "notes_md":"…", "notes_rev":15, "note_proposals":[…]},
  "operation": {
    "command_id":"…", "from_rev":14, "to_rev":15,
    "kind":"proposal_decision", "authored_by":"assistant", "source":{…},
    "undoable":true, "created_at":1784730000.0
  }
}
```

Rules:

- `command_id` is idempotent; exact replay returns the stored response, conflicting reuse returns 409.
- Base revision mismatch returns `notes_conflict` with current meeting; no mutation.
- Anchors/occurrences and `delete_md` are validated against canonical Markdown; missing/ambiguous targets return 422 with a reason.
- `insert` and `replace` are user-authored only. The server derives attribution: direct commands are `user`; accepted proposal content is `assistant`. Attribution cannot be asserted by a renderer-supplied string.
- Proposal Accept reads immutable anchor/delete/insert/source fields from the stored proposal, applies content, and marks it accepted in one SQLite transaction. Reject changes status only and records a decision command. Client-supplied replacement text is ignored for proposal decisions.
- Reject and undo-of-reject leave `notes_rev` unchanged, so their operation `from_rev` and `to_rev` are equal. Accept and undo-of-accept increment the revision exactly once.
- Proposal acceptance against a changed revision returns 409 with `proposal_stale` and current meeting; the controller offers re-anchor/choose location, which creates a new superseding proposal rather than mutating the old one.
- Undo is allowed only when the target operation is the latest note-changing operation and current revision equals its `to_rev`; otherwise 409 `undo_conflict`. Decision undo restores both text and proposal status atomically.
- Store the last 100 operation records per meeting plus all still-undoable proposal decisions. Meeting detail adds `note_operations` metadata without duplicating full private document snapshots.
- Existing `ai_lines` remains readable for old persisted meetings but the target renderer writes new attribution through operations and does not use normalized line equality as identity.

Storage is two additive SQLite tables: `meeting_note_proposals` keyed by `proposal_id` with meeting/status/source/anchor/content/decision columns, and `meeting_note_operations` keyed by `command_id` with request hash, revisions, inverse data, attribution, and timestamps. Both have a meeting/time index and foreign-key cascade on meeting deletion. Private proposal text is not duplicated in operation rows; operation records reference the proposal.

The `MeetingDocumentController` is the only frontend caller of revision-aware note PATCH and note-command mutation. `ProposalQueue` may create/refetch proposals but cannot edit the document. This makes the boundary structural rather than a convention in five separate views.

### 13.7 Minimal backend addition B: resumable live context

Extend session Start and Stop while retaining the existing WebSocket:

```ts
type SessionStart = {
  type: "start"
  meeting_id: string
  client_session_id: string
  channels: Array<{id: number; label: string}>
}

type SessionStop = {
  type: "stop"
  meeting_id: string
  client_session_id: string
  resume_window_s: 10 | 0
}

type SessionNotepad = {
  type: "notepad"
  meeting_id: string
  client_session_id: string
  text_md: string
}

type SessionSnapshot = {
  type: "session_snapshot"
  meeting_id: string
  client_session_id: string
  disposition: "new" | "resumed" | "state_reset" | "stopped_resumable"
  resume_deadline: number | null
  asr: "disconnected" | "connecting" | "connected" | "error"
  channels: Array<{id: number; label: string}>
  transcript_seq: number
  summary: SummaryEvent | null
  grounding: LiveGroundingEvent | null
  actionables: Actionable[]
  note_suggestions: NoteSuggestionEvent[]
}

type SessionError = {
  type: "session_error"
  meeting_id: string | null
  client_session_id: string | null
  code: "meeting_session_conflict" | "meeting_not_resumable" | "invalid_session_control"
  detail: string
}
```

Server rules:

- The registry key is `{meeting_id, client_session_id}` and contains only one owner. A different token cannot take over an active/suspended meeting and receives `SessionError{code:"meeting_session_conflict"}`.
- Disconnect suspends for 60 seconds; explicit Stop uses the requested 10-second window; `resume_window_s:0` finalizes immediately. Both paths stop accepting audio and flush ASR before suspension.
- A matching Start atomically moves the entry from suspended to active ownership, cancels its single delayed-finalization task, attaches a `RebindableEventSink`, restarts ASR, and emits the snapshot before `status.asr="connected"`. Audio received before that acknowledgement is rejected, not silently dropped.
- Duplicate Stop for the same token is idempotent and never extends its deadline. Stop applied to a disconnect-suspended entry shortens the remaining deadline to at most 10 seconds. Target notepad messages carry the token and may update either its active or suspended entry.
- Summary/grounding/actionable/notepad objects stay owned by the retained `LiveSession`; the frontend merges by summary revision, actionable ID/status, evidence ID, suggestion note ID, and transcript sequence. The snapshot is a bounded current-state representation, not replay of every live event.
- Expiry atomically claims the entry for finalization. A simultaneous resume has one winner: resume returns `resumed`, or the client receives the durable summarizing/done status; two final-summary jobs cannot be created.
- The registry is intentionally in memory. Process restart returns `state_reset`, refetches durable meeting/transcript state, and discloses the live-context break. No false continuity is reconstructed.
- Delete meeting cancels and removes an active/suspended entry before deleting durable data. App quit follows the existing explicit-quit policy and requests immediate finalization for a live session.

This is a narrow continuation contract, not durable audio buffering or a general event-replay service.

### 13.8 Minimal backend addition C: desktop OAuth authorization

Current Gmail/Outlook/Google availability depends on environment variables and tells users to run a Python module. A complete desktop Sources workflow needs one narrow authorization surface without exposing tokens to the renderer.

Add:

```text
POST /connectors/{id}/authorize
  -> 202 {flow_id, authorization_url, expires_at}
GET /connectors/{id}/authorize/{flow_id}
  -> {state: waiting|authorized|expired|error, detail}
DELETE /connectors/{id}/authorization
  -> ConnectorInfo
GET /oauth/callback/{flow_id}?code=…&state=…
  -> minimal browser success/failure page
```

Rules:

- only connector registry members with `kind="oauth"` are accepted;
- PKCE/state and exact local callback binding are mandatory;
- refresh/access tokens are stored through a sidecar credential-store abstraction backed by the OS credential vault; they never enter connector `config`, `ConnectorInfo`, logs, Redux, renderer storage, or utility IPC;
- `available()` consults the credential store first and retains environment variables as a developer compatibility source;
- renderer asks main process to open only the returned allowlisted HTTPS authorization URL;
- authorization state publishes a connectors update; successful auth does not auto-enable or auto-sync without the user's selected action;
- disconnect removes the stored credential but does not purge already indexed data; the UI offers Disable and Delete data separately.

This addition is required for the completed desktop Sources workflow. Environment credentials and the Terminal bootstrap remain supported for development/existing personal setups, but they do not replace Connect/Disconnect acceptance.

### 13.9 Referential cleanup for meeting-linked inquiries

The current store can delete a chat while a meeting still points at its ID. As a contract-integrity correction, `DELETE /chats/{chat_id}` also clears any `meetings.chat_id` equal to that ID in the same transaction and publishes both chats and meetings updates. Deleting a meeting continues to preserve its linked inquiry. The next meeting Ask send may then create and bind a new inquiry through existing semantics.

### 13.10 Contracts deliberately not added

- No new graph endpoint: existing full/local graph, clusters, idea, facts, notes, and edge reason support the Atlas; scaling remains a measured future boundary.
- No unified WebSocket: the three protocols have materially different lifecycles; one frontend supervisor is sufficient.
- No event replay IDs: domain refetch on reconnect is adequate for this local single-window product, with known live-trail limitations disclosed.
- No final-summary retry endpoint: UI does not promise one.
- No profile cancel endpoint: generation stays runtime-owned and continues across navigation.
- No source hard-scope query endpoint: seed scopes are labeled accurately as bias.

## 14. Target file and module decomposition

The final worktree completely replaces the old UI architecture. It ships no runtime feature flag, no old-interface fallback, no dual renderer, and no migration program. Existing `taurine.db`, `taurine-knowledge.db`, meeting/chat/source rows, and current contracts remain readable; only additive schema changes from §13 may migrate in place through normal sidecar DDL initialization.

### 14.1 Renderer

```text
app/src/renderer/src/
  app/
    App.tsx                       providers + AppShell only
    AppShell.tsx                  command/coordinate rail/index/workbench/inspector/shelf/operations
    store.ts                      configureStore + redacted middleware
    listeners.ts                  cross-domain orchestration
    commands.ts                   command/shortcut registry
    hydration.ts                  startup/restore/reconcile
  services/
    sidecar/
      gateway.ts                  one connection supervisor
      rest-api.ts                 RTK Query endpoints/tags
      updates-connection.ts
      session-connection.ts
      chat-connection.ts
      contracts.ts                exhaustive frontend contract types/parsers
      reconciliation.ts
    desktop/
      bridge.ts                   typed preload adapter
      capabilities.ts
  features/
    runtime/                       slice/selectors/readiness/error UI
    workspace/                     slice/history/pins/compare/layout persistence
    atlas/
      AtlasWorkbench.tsx
      AtlasCanvas.tsx
      AtlasOutline.tsx
      AtlasIndex.tsx
      AtlasControls.tsx
      TimeBand.tsx
      atlas-worker.ts
      layout.ts
      render.ts
      slice.ts selectors.ts contracts.ts
    meetings/
      MeetingsIndex.tsx
      MeetingDossier.tsx
      LiveDesk.tsx
      CompactLive.tsx
      CaptureHeader.tsx
      TranscriptLane.tsx
      NotesDocument.tsx
      MeetingDocumentController.ts
      ProposalQueue.ts
      LiveIntelligence.tsx
      FinalRecord.tsx
      ProposalDiff.tsx
      state.ts listeners.ts selectors.ts contracts.ts
      editor/                      BlockNote adapter, serialization, provenance
    inquiry/
      InquiryIndex.tsx
      InquiryWorkbench.tsx
      InquiryComposer.tsx
      AssistantTurn.tsx
      ProcessTrace.tsx
      MessageMarkdown.tsx
      state.ts listeners.ts selectors.ts contracts.ts
    evidence/
      EvidenceInspector.tsx
      EvidenceLedger.tsx
      PassageView.tsx FactView.tsx PathView.tsx
      normalize.ts selectors.ts
    sources/
      SourcesIndex.tsx
      SourcesLedger.tsx
      SourceDetail.tsx
      SourceConfig.tsx
      Authorization.tsx
      descriptors.ts state.ts listeners.ts selectors.ts
    profile/
      ProfileWorkbench.tsx ProfileEditor.tsx GenerationRail.tsx
      state.ts listeners.ts
    privacy/
      PrivacyCurtain.tsx PrivacyStatus.tsx state.ts
    notifications/
      ContextQueue.tsx state.ts
    diagnostics/
      DiagnosticsWorkbench.tsx SessionLog.tsx
  components/
    primitives/                    Button, Field, Menu, Dialog, Sheet, Notice…
    layout/                        SplitPane, VirtualList, IndexPane…
    knowledge/                     ObjectRow, SourceMark, EvidenceRef, FactRow…
  styles/
    tokens.css reset.css typography.css shell.css
    primitives.css print.css
  main.tsx
```

`App.tsx` has no socket constructors, fetch calls, capture coordination, graph state, chat reducer, or editor state. It should stay below approximately 120 substantive lines.

### 14.2 Main process and preload

```text
app/src/main/
  index.ts                         lifecycle composition only
  windows/
    main-window.ts
    call-prompt-window.ts
    context-suggestion-window.ts
    window-privacy.ts
  capture/
    micwatch.ts
    system-audio.ts
  permissions/
    capabilities.ts
    system-settings.ts
  recovery/
    drafts.ts                       safeStorage-backed meeting recovery
  tray.ts
  ipc.ts                           narrow validated handlers

app/src/preload/
  index.ts
  contracts.ts
```

The existing notifier/suggest-notifier/sysaudio/micwatch logic moves behind these ownership boundaries; behavior and native helpers are preserved. Utility renderer entry points remain minimal and receive sanitized state from main, not direct sidecar access.

### 14.3 Styles and assets

- Delete the monolithic old theme/layout rules after all target components move to feature CSS.
- Retain no old selector namespace as a fallback.
- Bundle IBM Plex Sans/Mono WOFF2 and normalized connector SVGs locally.
- Use CSS custom properties from `tokens.css`; theme override is `[data-theme="dark"]` plus system-resolution code.
- Canvas colors are exported from the same token source through a typed theme adapter, not duplicated hex maps in `graph.ts` and CSS.

### 14.4 Core additions

Only if §13 additions are implemented:

```text
core/taurine_core/
  schemas.py                       proposal/note command/live resume/auth contracts
  server.py                        endpoint wiring only
  indexer/store.py                 transactional proposal/command/idempotency storage
  meetings/note_mutations.py       proposal, anchor, decision, inverse/undo rules
  live/session_registry.py         bounded suspension/rebind/finalize ownership
  connectors/credentials.py        credential-store port + platform backend
  connectors/oauth.py              PKCE flow registry/callback lifecycle
```

OAuth provider-specific URL/scope/token behavior stays in current connector/auth modules behind the shared flow. Retrieval/live/chat algorithms are not forked.

## 15. Verification and acceptance

### 15.1 Mandatory automated gates

Every acceptance build must pass, in one recorded run:

1. `core`: complete pytest suite, preserving at least the verified **981 passed, 2 skipped** baseline; new proposal/note-command/live-resume/OAuth tests add to it.
2. `app`: all current and new Vitest tests, preserving the verified **61-test** baseline; both Node and Web TypeScript projects; ESLint; production Electron build.
3. Accessibility: automated axe checks on every stable workbench state plus keyboard traversal tests. Automated results do not replace manual screen-reader checks.
4. Electron end-to-end: packaged or production renderer through Playwright's Electron support with fake/local sidecar fixtures.
5. Bundle budget: generated bundle report enforces §1.5 initial/feature chunk and CSS limits and demonstrates Atlas/editor/Markdown splitting.
6. Performance: scripted 5,000-node Atlas, 10,000-row source/note list, 3,000-block transcript, and 200-message inquiry datasets with recorded frame/input/heap results.
7. Visual regression: light/dark, compact/comfortable, 960/1280/1440 widths, two- and three-lane Live desk, 100/200% zoom, and compact live screenshots. Animations/fonts/time are stabilized.

There is one completed target interface. Tests do not accept a feature flag or the old UI as an alternate successful path.

### 15.2 Unit and reducer coverage

#### Runtime/gateway

- startup backoff, ready transition, 503/1013 return to Starting;
- one socket instance per protocol, epoch rejection, reconnect caps/jitter;
- no audio or chat-send replay; coalesced notepad policy;
- full domain rehydrate after updates reconnect;
- sanitized logging excludes content.

#### Workspace

- selection/open/pin history; max three pins; removed-object restore;
- coordinate transitions preserve shelf and per-coordinate selection; Control does not erase the working context;
- divider keyboard/resets/persistence; responsive pane policy;
- command contexts do not fire inside editors.

#### Meeting

- every state transition in §12.3, including duplicate start/stop/resume and deadline races;
- mic block/system degradation and device loss;
- 800 ms save, 2 s notepad, immediate local stop, recovery write, grace-period flush ordering;
- stale save success cannot clear later edit;
- local recovery vs server revision restore;
- conflict keep mine/server/copy both;
- transcript final/interim grouping, channel labels, auto-follow;
- reconnect resume/state-reset/after-grace outcomes, snapshot merge, and capture-gap disclosure;
- actionable user status wins over in-flight summary event;
- final event missed then meeting update/refetch;
- generated title does not replace user title;
- all five AI origins persist pending proposals before presentation; accepted/rejected/stale/ambiguous/superseded proposal and undo conflict;
- `chat_notes_edit` carries a persisted proposal and never mutates the editor; only `MeetingDocumentController` can issue note PATCH/commands;
- deletion preserves linked inquiry and marks index consequence.

#### Inquiry/evidence

- exact event ordering/interleaving and stable activity-step replacement;
- created-on-first-send, ack ID replacement, background stream counts;
- done authoritative swap, cancel, interrupted, busy, timeout, disconnect/refetch;
- regeneration old-snapshot behavior and failure recovery;
- original/rewritten query, decision/confidence/build/usage/stripped claims survive parse and render;
- cited/consulted/retrieved deduplication and precedence;
- fallback, abstention, prior-context, temporal fact, path, and unavailable citation presentations;
- interrupted partial never receives a support list;
- note/meeting/evidence links route to the inspector/Atlas.

#### Atlas

- deterministic initial coordinates and unchanged-node reuse;
- LOD thresholds, default min degree 2, search override, filter clauses;
- territory/node/edge source and pattern encodings;
- global/local depth, ego history, idea states, session overlay expiry;
- timeline presets/fields/brush equivalence;
- canvas action parity in Outline/context menu;
- no essential object becomes unreachable when hidden by canvas LOD.

#### Sources/profile/privacy

- status priority and exact disable/purge consequences;
- typed config parsing/ranges and unknown-key round trip;
- global/per-source sync/cancel/stale/graph swap;
- authorization waiting/success/expiry/error and no token in renderer payload;
- profile empty/edit/overwrite/generation/failure preservation;
- privacy platform copy, curtain redaction across main/call/context utilities;
- curtain keeps Stop accessible while capture continues.

### 15.3 Core contract coverage for additions

#### Proposals and note commands

- base revision success/conflict response;
- proposal create idempotency/conflicting reuse and no revision change;
- meeting-chat persistence precedes socket emit; final proposals share the final-record transaction;
- rolling/actionable/inline origins survive refetch; failed creation never enables Accept;
- immutable proposal payload used on Accept; client replacement injection rejected;
- legacy suggestion adapter returns deterministic proposal identity without destructive migration;
- duplicate command ID idempotency and conflicting reuse;
- exact anchor, repeated occurrence, missing anchor, delete mismatch;
- insert/replace assistant/user attribution;
- proposal accept/reject as one transaction;
- undo latest operation, undo decision, intervening-revision conflict;
- rollback on storage error and concurrent command serialization;
- 100-record retention without dropping undoable pending decisions;
- legacy meeting rows/`ai_lines` still deserialize.

#### Live continuation

- disconnect suspension retains bounded summary/grounding/actionable/suggestion/notepad state and never buffers audio;
- matching meeting/token rebind emits snapshot before listening; mismatched token is rejected;
- summary revision, evidence IDs, actionable status, suggestion IDs, and transcript sequence merge without duplication;
- process restart/missing entry returns state reset and durable meeting truth;
- Stop immediately closes capture, exposes 10-second resume, and one-click resume cancels delayed finalization;
- Resume-versus-expiry race has one winner and creates at most one final-summary job;
- Finalize now, natural expiry, meeting delete, and explicit app quit clean up the registry exactly once.

#### OAuth

- supported/unsupported connector guard;
- state/PKCE/callback validation, expiry, replay rejection;
- safe provider error and callback browser page;
- credential store write/read/delete, environment compatibility precedence;
- no tokens in connector info, logs, REST errors, or update events;
- disconnect does not purge/disable implicitly;
- auth success publishes connector update but does not sync.

### 15.4 Accessibility/manual assistive checks

Run on a packaged build:

- VoiceOver on macOS: complete each primary job without pointer, including Atlas Outline, start/stop, notes, Ask/citation inspection, source sync, profile edit, privacy curtain.
- NVDA on Windows build where supported: same non-native-audio workflows and content protection copy.
- 200% zoom at minimum window; 35% pseudo-localized expansion; RTL catalog.
- Reduced motion, high contrast/forced colors, light/dark/system theme.
- Pointer target inspection and keyboard focus not obscured by command deck/composer/status.
- BlockNote edit/menu/proposal flow per §11.5.

### 15.5 Native/manual integration checks

- macOS microphone permission: unknown → prompt → granted, denied → settings, device lost.
- macOS 14.2+ system audio grant/deny/tap loss and mic-only fallback; older macOS hides unsupported control.
- call helper: 1.5 s debounce, one prompt/call, five-second idle rearm, Ask and opt-in 10 s auto-start, no app focus/raise.
- context suggestion queue/timeouts/open/dismiss and curtain redaction.
- tray hide/show/start/open/stop/quit, compact bounds restore, optional always-on-top.
- Apple Notes native opening and failure.
- iMessage Full Disk Access remediation, Obsidian discovery, folder picker/missing roots.
- screen-sharing verification in at least one ScreenCaptureKit-based macOS app: confirm the UI says Limited even if a particular capture omits the window; curtain capture must show only neutral content.
- sleep/wake, network loss, sidecar restart, renderer reload during live meeting/finalization/sync/profile generation/chat.

### 15.6 End-to-end acceptance stories

#### A. Start and finish a meeting

Given runtime ready, mic granted, and system audio supported, starting from the command center creates one meeting, declares channels 0/1, shows both healthy, and persists notes while live. One Stop action ceases audio immediately and exposes Stopped · Resume for 10 seconds. Finalize now or deadline expiry flushes the freshest notes/notepad, moves through summarizing, recovers final data by meeting refetch, and renders recap/proposals without adding them to canonical notes.

#### B. Degrade to mic only

Given system audio denied, preflight clearly offers Start mic only; capture is labeled mic-only for the entire live state; no “Them” channel health is shown; finalization remains available.

#### C. Reconnect honestly

Given a live socket disconnect, the header stops saying Live, notes remain editable/locally recoverable, and no audio frames are queued. A matching reconnect within 60 seconds receives a resumed snapshot that preserves summary, grounding, actionables, and suggestions, inserts an audio-gap divider, then restarts capture. A missing/expired context instead says Live context restarted and never describes the lost interval or ephemeral context as recovered.

#### D. Prevent note loss

Given local notes based on revision 8 and a server update to 9, autosave receives 409, preserves both versions, pauses writes, and requires an explicit resolution. Stop still stops audio and retains local recovery.

#### E. Accept and undo a final proposal

Given a durable pending proposal at the current revision, Accept performs one command that inserts immutable proposal text and marks accepted. Undo restores both exact notes and pending status. With intervening edits, Undo opens/returns conflict rather than deleting the wrong text.

#### F. Ask with transparent evidence

Given an answer with three cited, two consulted, and four retrieved-only records, the turn shows those exact groups. Selecting a citation opens the passage/fact/path. Consulted records never appear as “supports this claim.” Original/rewritten query and bootstrap-calibration warning are available in Advanced.

#### G. Handle low-confidence and interrupted turns

Given a fallback pack, the persistent warning explains low confidence and withheld facts/paths. Given socket interruption, the raw persisted partial is labeled unvalidated and has no support list even if its text includes `[E1]`.

#### H. Investigate spatially without a pointer

Given an Atlas with communities and filtered tail nodes, a keyboard/screen-reader user can search, enter Outline, expand a territory, inspect a relationship and fact, pin three notes, compare them, and open Ask seeded from those notes without using canvas drag/hover.

#### I. Sync sources without losing the old Atlas

Given an active immutable graph, enabling a source marks stale; Enable and sync reports real source/index detail, keeps the old Atlas usable, and swaps only after a new graph version fetch succeeds. Cancel/failure leaves stale and gives recovery.

#### J. Purge with exact boundaries

Given a source with N documents, Delete data names N, provider data untouched, source disable, immediate local/derived removal, stale index, and no undo. It is blocked during conflicting sync. After success, open removed objects no longer masquerade as current.

#### K. Generate profile safely

Given an existing profile, Generate requires overwrite confirmation, preserves the old profile while server phases run, continues across navigation, atomically shows the new card on success, and leaves the old one with a retryable failure state on error.

#### L. Use privacy curtain during capture

Given a live meeting and Limited macOS protection, activating curtain removes private rendered content from main and utilities while keeping elapsed/channel/Stop controls. Capturing the Taurine window shows only that neutral surface. Exiting restores the exact Live desk/compact state.

#### M. Replace, not coexist

Given a production build, only the Knowledge Atlas shell exists. There is no old tab bar, old dark/yellow theme, legacy Feed/Map/Data UI route, runtime interface flag, alternate renderer, or fallback CSS. Existing meetings/chats/sources/profile/knowledge databases open in the target interface without destructive conversion.

#### N. Keep every AI note change reviewable

Given generated note text from rolling summary, actionable, meeting chat, inline Ask, or final summary, the UI first persists and displays one `MeetingNoteProposal`. Reload preserves it. Before Accept, canonical Markdown and `notes_rev` are unchanged. Accept enters the sole document controller and one atomic command; Reject records the decision without editing notes. No socket handler or feature view has a direct generated-text mutation path.

## 16. Risks and mitigations

| Risk | Consequence | Mitigation/decision |
|---|---|---|
| Atlas remains a visually impressive but low-value hairball | power users abandon spatial view | territory-first LOD, stable geography, search/outline equality, edges on selection, task tests |
| Existing projection does not scale past full-graph fetch | memory/startup cost | lazy Atlas, canvas/worker, 5k acceptance; measure before adding paged API; old snapshot retained |
| Redux/RTK Query becomes another monolith | logic only moves out of `App.tsx` | strict slice ownership, listener side effects, feature selectors, no component fetch/socket |
| BlockNote bundle/accessibility | startup cost or unusable editor | lazy chunk/budget, source mode parity, assistive gate |
| Revision-aware full Markdown conflicts feel heavy | interrupted note flow | local recovery, infrequent conflict sheet, operation commands for generated mutations |
| Durable proposal/operation attribution complicates store | schema/storage surface | one proposal type, one controller, additive bounded log, no editor-node persistence, legacy suggestion/`ai_lines` read compatibility |
| Three sockets still produce race conditions | stale/duplicate state | epoch supervisor, domain refetch, idempotent controls, explicit state machines |
| Session reconnect loses audio or a process restart loses live context | false confidence in record | bounded in-memory context resume, no audio replay claim, gap divider, explicit state-reset disposition, durable dossier refetch |
| Stop/resume races delayed finalization | duplicate recap or reopened capture | registry ownership claim, stable session token, one delayed task, one winner tests, immediate local capture stop |
| Final-summary event missed | record appears stuck | meeting row/update is source of truth, active detail refetch |
| OAuth credential storage expands security surface | token exposure | OS vault abstraction, no renderer/log payload, PKCE/state tests; Terminal compatibility if excluded |
| Privacy protection marketed too strongly | harmful disclosure | Limited/Requested/Unavailable copy, Electron source link, deterministic curtain, real sharing test |
| Bootstrap calibration score appears scientific | misplaced trust | decision categories primary, numeric Advanced only, explicit zero-sample warning |
| Current citation checks do not entail support | “verified” overclaim | exact vocabulary §2.3, no fact-checked label, evidence passage inspection |
| Source colors exceed categorical discrimination | inaccessible/ambiguous graph | glyph + shape + label + pattern, filters/outline, color not sole state |
| Dense UI becomes cramped on ordinary laptops or at text zoom | inaccessible workflow | 1280 default two-lane Live desk, optional three-lane only at ≥1440, overlay breakpoints, 40/44 key targets, reflow gates |
| Full replacement has broad regression surface | lost hidden capabilities | completeness matrix, existing test floor, exact end-to-end stories, no old UI fallback masking failures |

## 17. Completeness matrix

| Existing capability/contract | Target destination | State/error coverage | Acceptance evidence |
|---|---|---|---|
| main hide-on-close/tray/dock | window model/tray | shown/hidden/compact/fullscreen/quitting | native check + story M |
| call detection/debounce/countdown/rearm | call utility + preferences | unsupported/off/ask/auto/starting/error/rearm | native check + A/B |
| context suggestion popup/queue | utility + Live context queue | queued/open/dismiss/timeout/redacted | native check + L |
| content protection | privacy status + curtain | requested/limited/unavailable/curtain | native sharing + L |
| mic/system audio/channels | preflight/Live desk | permissions/devices/full/mic-only/lost/reconnect | A/B/C/native |
| `/session` status/transcript/resume snapshot | CaptureHeader/Transcript | all ASR/final/interim/gap/resumed/state-reset/stopped-resumable states | meeting reducer/E2E/core continuation |
| live grounding/trigger inspector | Live intelligence/Evidence inspector/Atlas overlay | accepted/abstained/no-new/degraded/ephemeral | evidence/Atlas tests |
| rolling summary/highlights | Rolling summary | empty/revision/update/rejected/no change | A + reducer |
| actionables | Live/Final Actions | proposed/confirmed/dismissed/in-flight user precedence | meeting unit/E2E |
| note suggestions/snippets | utility + Context queue | normal/error fallback/timed/redacted | native + reducer |
| meeting drafts/list/title/delete | Chronicle index/dossier | CRUD/pending/failure/deleted elsewhere | A/M + unit |
| notes autosave/notepad push | NotesDocument/listeners | dirty/save/queued/conflict/stop flush | A/C/D |
| all generated note edits | meeting Ask/Live/Final + durable proposals/commands | create/reload/accept/reject/stale/anchor error/undo | E/N + proposal tests |
| per-line AI provenance | operation/block gutter, legacy read | generated/user-transferred/legacy | command/editor tests |
| finalization/status/final summary | Final record | ending/running/done/error/missed event/stale revision | A/E |
| proposal accept/reject | ProposalDiff/note commands | pending/accepted/rejected/stale/ambiguous/undo | E/N/core |
| transcript overlay/search | dossier Transcript | loading/virtualized/search/no results | perf/component |
| Taurine hidden meeting source | Sources System + Atlas source | queued/indexing/indexed/stale | I + connector tests |
| global chat threads/stream | Ask | every §8.5 state | F/G + reducer/E2E |
| reasoning/activity/tools | ProcessTrace Advanced | running/done/error/interleaved | inquiry tests |
| cancel/regenerate/copy/rename/delete | Ask actions/index | pending/rollback/interrupted/busy | inquiry tests |
| retrieved/consulted/cited | EvidenceLedger | distinct/dedup/fallback/interrupted | F/G |
| original/rewrite/decision/confidence/build/usage/stripped | Advanced metadata | missing/degraded/known | parse/render tests |
| graph global/local | Atlas/Outline | loading/empty/ready/stale/error | H/I/perf |
| graph search/entity/gist | Atlas Index | query/results/no results/error | Atlas tests |
| community/source/recency/min-degree colors/filters | lenses/time band/tokens | active clauses/clear/hidden-tail access | H |
| timeline brushing | Time band + fields/presets | all/custom/invalid/no dates | accessibility/Atlas tests |
| node/edge hover/select/context/pin/hide | canvas + outline + inspector | pointer/focus/menu parity | H |
| facts/edge reason/provenance/mechanisms | Evidence inspector | missing/generated/historical/conflicted | evidence tests |
| live rank/PPR/grounding/session heat | conversation overlay | current/fading/reload restart | Atlas tests |
| idea lens | Atlas lens | idle/running/result/empty/error | Atlas tests |
| note preview/native Apple Notes open | dossier/open source | loading/404/unsupported/open error | native/component |
| connector list/status/config | Sources ledger/detail | all §8.7 connector states | I/J/unit |
| enable/disable/per-source sync | Sources actions | stale/consequence/progress/conflict | I/J |
| global sync/cancel | operation ledger/Control → Sources | phases/cancel/fail/stale/complete/collapsed success | I |
| purge | danger section | confirm/conflict/pending/complete/no undo | J |
| folder chooser | typed config/preload | chosen/cancel/missing/permission | native |
| OAuth authorization + developer env compatibility | Authorization | disconnected/waiting/success/error/disconnect | OAuth tests + Sources E2E |
| profile manual/generated/progress | Profile | all §7.5 states | K |
| startup progress/503/1013 | readiness shell | all §8.1 states | gateway/E2E |
| `/updates` all six kinds | normalized store | initial/coalesced/reconnect/dirty conflict | gateway tests |
| session JSONL endpoints | Diagnostics | loading/empty/filter/error | diagnostics tests |
| responsive panes/hidden mounted views | shell overlays/inert/unmount | breakpoints/zoom/focus restore | accessibility/visual |
| current dark/yellow CSS and emoji controls | fully removed | not applicable | story M + visual review |

## 18. Final design decisions

Three directions were considered:

1. **Polished multi-page application:** meetings, chat, graph, and sources redesigned separately. Rejected because it preserves mode loss and leaves evidence/context translation to the user.
2. **Freeform agentic canvas:** draggable/generated panels over the knowledge graph. Rejected because it creates window management, accessibility, and persistence problems before the stable objects/workflows are trustworthy.
3. **Knowledge Atlas workstation:** one shell, fixed regions, synchronized spatial/temporal/evidence representations, bounded comparison. Chosen because it creates a new mental model without sacrificing operational precision.

The chosen design is intentionally adversarial to the current UI's assumptions:

- the five-tab product taxonomy becomes three persistent knowledge coordinates over one context, with administration separated into Control;
- the graph becomes a territory-first atlas plus semantic outline;
- chat becomes scoped inquiry with a first-class evidence ledger;
- meetings become durable chronological dossiers with live mode;
- generated content becomes proposed/attributed/reversible operations;
- connector cards become a source/index ledger;
- decorative AI styling becomes restrained technical typography;
- “screenshare invisible” becomes a truthful limitation plus a deterministic curtain;
- component-local socket/state coordination becomes explicit domain state and reconciliation.

Six cross-track decisions were resolved explicitly:

1. **Keep the rail, change its meaning.** Atlas/Chronicle/Inquiry are orthogonal coordinates that preserve one selection, pin set, time range, and history; they are not the current tabs rotated. Sources/Profile/Privacy/Diagnostics move to Control so administration no longer masquerades as knowledge navigation. Persistent coordinates preserve spatial orientation and make comparison/thread discovery visible to keyboard and pointer users.
2. **Demote operational chrome to exceptions.** The former always-on status concept becomes an operation ledger that is absent when healthy and idle. Capture truth stays in the live header; technical counters and build identity live in diagnostics/Advanced. A power user may pin selected categories, but ordinary work does not carry a telemetry footer.
3. **Shrink ordinary geometry.** Default/minimum change to 1280 × 820 and 960 × 640. Ordinary Live is Notes plus one switchable secondary lane. Three lanes are an optional ≥1440 density choice, preserving the investigator layout without requiring a large monitor for daily capture.
4. **Preserve live reasoning through transient failure.** A bounded `{meeting_id, client_session_id}` suspension/rebind contract retains summary, grounding, actionables, suggestions, and notepad for 60 seconds while never buffering or claiming recovery of audio. Process loss is an explicit state reset.
5. **Make proposal review universal.** One durable `MeetingNoteProposal` type covers final recap edits, meeting chat, inline Ask, rolling summaries, and actionables. One `MeetingDocumentController` owns every canonical write; generated content cannot enter notes from a view or socket handler.
6. **Stop immediately, then permit recovery.** Confirmation is removed because it keeps capture running at the moment a user asked it to stop. Stop ceases capture in one action and offers a server-backed 10-second Resume/Finalize window with single-winner race semantics.

The implementation may be staged internally, but the release target replaces the old interface atomically. There is no product feature flag, dual renderer, old-route fallback, or user-facing migration program.

The final product should let a demanding user answer four questions at any moment: **Where am I? What am I looking at? What is Taurine doing? What evidence and user action caused this state?** If any major surface cannot answer those questions without opening developer tools, it does not meet this specification.

## 19. Self-review record

The final draft was checked against the requested design brief and the audited product surface:

| Requirement | Evidence in this specification |
|---|---|
| thesis, users, principles, metrics | §§0–1 |
| current architecture/capability evidence | §2 |
| information architecture and multi-context mental model | §3 |
| exact main, utility, live, compact, responsive, tray, and privacy window behavior | §4 |
| navigation and command grammar | §5 |
| reimagined graph/Atlas, temporal band, lenses, live overlay, accessible outline | §6 |
| complete startup, meeting, Ask/evidence, connector, profile, diagnostics, and native-open workflows | §7 |
| loading/empty/error/degraded/conflict/interrupted/permission states | §8 |
| exact light/dark tokens, type, spacing, density, geometry, motion, source marks | §9 |
| reusable components and measurements | §10 |
| keyboard, accessibility, editor, zoom, target, RTL/i18n behavior | §11 |
| frontend store, state machines, sockets, reconciliation, privacy, performance/code splitting | §12 |
| every REST/WebSocket/desktop contract plus narrowly justified additions | §13 |
| implementation ownership/file decomposition and full-replacement constraint | §14 |
| complete core/app/accessibility/native/performance verification and acceptance stories | §15 |
| risks and mitigations | §16 |
| current-capability-to-destination completeness | §17 |
| considered alternatives and final decision | §18 |

No application code, configuration, database, or existing design document is changed by this specification. There are no unresolved design decisions; optional behavior is expressed only where the current platform genuinely differs (for example unsupported system audio), not as an undecided product branch.
