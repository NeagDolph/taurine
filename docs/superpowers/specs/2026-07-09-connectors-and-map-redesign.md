# Connectors page redesign + Data-panel map upgrade

Date: 2026-07-09. Scope: UI/UX. Two independent work streams; small backend
change (expose `gist` on `/graph` nodes) supports the second.

## 1. Connectors page — redesigned from the ground up

### Problems with the current page

- A grid of identical chunky cards with four equal-weight buttons each
  (toggle / Sync / Re-enrich / Delete data) — no hierarchy, button soup.
- Composite state (`enabled` × `available` × `syncing` × `detail`) is never
  distilled: the "Available" dot and the toggle disagree visually, and OAuth
  connectors print a raw CLI command as wrapped plain text.
- Config is hardcoded to Obsidian and **broken**: the UI reads/writes
  `vault_path` but the backend key is `vault`, so the field never round-trips.
- `kind` badge ("local"/"oauth") is jargon; stats pills are cramped uppercase
  noise; dead `visible` prop.

### New design

**Layout.** Replace the card grid with a single vertical list of connector
rows inside one raised panel (integrations-page pattern à la Linear/Notion).
Page header keeps the intro; the panel header shows aggregate stats
(`N active · N docs indexed`) and global actions: **Sync all**
(`POST /index/sync`) and **Re-enrich all** (existing).

**Status model.** Derive one status per connector, in priority order:

| status | condition | presentation |
|---|---|---|
| `syncing` | `syncing` | pulsing accent dot + phase label + slim progress bar |
| `attention` | `enabled && !available` | amber dot, `detail` as subtitle ("Needs attention" / "Sign in required") |
| `active` | `enabled && available` | green dot, subtitle "N docs · synced X ago" |
| `off` | `!enabled` | gray dot, subtitle "Off — N docs stored" |

**Row anatomy.** Left: connector glyph (inline SVG per connector, tinted with
the same source color the map uses — cross-view consistency). Middle: name +
one-line status subtitle. Right: enable/disable switch + expand chevron. The
whole row header toggles expansion.

**Expanded panel** (per row):
- *OAuth sign-in block* (oauth + not signed in): explanation + the bootstrap
  command in a code chip with a copy button, instead of raw wrapped text.
- *Config* — generic, driven by the connector's `config` dict (works for all
  seven connectors, not just Obsidian). Strings/numbers get text inputs;
  string lists render comma-separated and save back as lists. Known keys get
  friendly labels/placeholders (`vault` → "Vault path", `glob`/`globs`,
  `window_days` → "Sync window (days)", `contacts_glob`, `db`). **Fixes the
  Obsidian `vault` key bug.**
- *Actions*: Sync now, Re-enrich, and a danger-zone Delete data with inline
  confirm.
- Per-row errors render inside the row; sync progress renders as a slim bar
  at the bottom edge of the row.

Loading state: skeleton rows. Sorting: Apple Notes first, then alphabetical
(unchanged).

**Files.** Rewrite `app/src/renderer/src/components/ConnectionsPanel.tsx`
(drop the `visible` prop), trim `ConnectorsView.tsx`, replace the
`.connections-panel`/`.connector-*` CSS block in `assets/main.css`. No API
changes; all seven existing endpoints reused.

## 2. Data panel — graph view upgrade

### a) Zoom-aware labels (LOD)

Labels currently draw for every node at every zoom, so a zoomed-out graph is
a mass of text chips. Add alpha-faded level-of-detail in `drawNode` using the
`globalScale` the draw callback already receives:

- Hub nodes (degree ≥ 75th percentile of the current graph) fade in first
  (scale ≈ 0.55 → 0.95), everything else fades in later (≈ 1.1 → 1.9).
- The `Now` node label and the hovered node's label always render.
- During active search/source filtering, matched nodes keep labels (that's
  the point of filtering).

### b) Retire the hop-expansion overlay

The PPR traversal (spec 2026-07-08) replaced 1-hop expansion; `Expansion`
events now arrive with `from_note_id=""`, `hop=0`, `via="ppr"`, so the dashed
hop edges and "hop N" labels are dead code. Remove `LiveHopLink`,
`hopColor`, `drawHopLabel`, and the "hop expansion" legend row.

Replace with a **traversal glow**: each PPR arrival paints a soft amber halo
on its node, intensity scaled by normalized mass within the event, fading
over the same 3 s window as rank edges. Legend row: "graph traversal" glow
swatch. "rank hot → cold" edges (real `hits[]`) stay as-is.

### c) Hover preview overlay (top-right)

Replace the native browser tooltips (`nodeLabel`/`linkLabel`) with a sleek
glass panel pinned top-right of the map stage (~300 px, backdrop-blur,
pointer-events none, fade/slide in):

- **Node hover**: source chip (colored dot + label), title, the note's LLM
  gist (fallback: first line — same field), meta row "N connections", and an
  "open note" hint. Hovering `Now` shows the live transcript tail.
- **Edge hover (static)**: "Connection" header with edge-type chip + weight;
  the two endpoint notes as mini rows (dot, title, 2-line gist); the edge's
  LLM `fact` sentence when present, else shared terms.
- **Live rank edge hover**: "Retrieval · rank N" + the target note block.
- Hovered node also gets an accent ring on canvas.

**Backend change**: `GraphNode` gains `gist: str = ""` (schemas.py) and
`store.graph()` populates it from `NoteRecord.gist` — additive, no consumers
break. Frontend `graph.ts` parses it onto `MapNode`.

**Files.** `core/taurine_core/schemas.py`, `core/taurine_core/indexer/store.py`,
`app/src/renderer/src/lib/graph.ts`, `views/MapView.tsx`, `assets/main.css`.

## Verification

- `cd app && npm run typecheck && npm run lint`
- `cd core && uv run pytest tests/test_index_store.py tests/test_index_graph.py`
- Visual pass in the running app (dev server) on both views.
