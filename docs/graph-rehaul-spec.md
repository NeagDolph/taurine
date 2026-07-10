# Graph View Rehaul Spec (post-HippoRAG cutover)

Date: 2026-07-09. Owner: orchestrated build (backend agent + frontend agent in parallel).

## Context

The index cutover is complete: `GET /graph` serves the deterministic note projection from
the knowledge runtime (`server.py:335` → `_knowledge_graph`, `server.py:951`), reading
`snapshot.projection` (tuple of `ProjectionEdge` from `note_projection_edges`). The legacy
`Index.graph()` path remains only as a fallback when `knowledge_runtime is None`.

This rehaul turns the map from a passive global hairball into the app's connective tissue:
local/ego exploration, a selection side panel (replacing the preview window), temporal and
community lenses, and round-trips to Chat and the Live summary.

## API contract (backend implements, frontend codes against)

All new fields are additive. The legacy fallback path must still work by defaulting them.

### Extended `GraphNode` (schemas.py)

```python
class GraphNode(BaseModel):
    note_id: str
    title: str
    degree: int
    source: str = "apple_notes"
    gist: str = ""
    created: float = 0.0        # unix seconds, 0 when unknown
    modified: float = 0.0       # unix seconds, 0 when unknown
    folder: str = ""
    community: int = -1         # stable label-propagation community id, -1 = none
    value_score: float = 0.0    # from notes.value_score
    entities: list[str] = []    # top ≤8 traversal-eligible canonical labels, by mention count
```

### Extended `GraphEdge`

```python
class GraphEdge(BaseModel):
    src: str
    dst: str
    type: EdgeType              # semantic | entity | mixed (topic legacy-only)
    weight: float
    shared: list[str] = []
    fact: str = ""              # legacy only
    fact_provenance: str = ""
    fact_ids: list[str] = []
    mechanisms: dict[str, float] = {}   # {"semantic": w, "entity": w, "fact": w}
```

### New: `GET /graph/local?center=<note_id>&depth=<1|2|3>` (default depth=2)

Server-side BFS over projection edges from `center`. Returns:

```json
{ "center": "<note_id>", "depth": 2, "nodes": [GraphNode...], "edges": [GraphEdge...] }
```

404 if center unknown. Edges included iff both endpoints are in the result set.

### New: `GET /facts?ids=<comma-separated fact_ids>` (cap 50)

```json
{ "facts": [ { "fact_id": str, "subject": str, "predicate": str, "object": str,
               "source_note_id": str, "confidence": float, "observed_at": float } ] }
```

Unknown ids are silently dropped.

### Communities → persisted clusters (2026-07-10 update)

Clustering now runs at build time (deterministic weighted greedy modularity over the
projection with de-saturated entity weights) and persists to `clusters` /
`note_clusters` tables. `GraphNode.community` is the persisted contiguous cluster id
(0..N-1, size-ordered; -1 = unclustered). `GraphResponse` and `LocalGraphResponse`
gain an additive field:

```json
"clusters": [ { "id": int, "name": str, "description": str, "size": int } ]
```

Names/descriptions are LLM-written during sync (specific, ≤5 words) with deterministic
fallbacks ("Ernest Chae, Adi Yadhav +2", "Group Chats"). Against a pre-cluster database
the server still boots: the snapshot falls back to activation-time label propagation
remapped to contiguous ids with `"Cluster 1 (24 notes)"`-style names, so the contract
always holds. `/graph` returns every cluster; `/graph/local` returns only clusters with
a member among the returned nodes.

### New: `POST /graph/idea` (idea lens — agentic concept search)

Request: `{ "idea": str, "limit": int = 40 }` (limit clamped 5–80). Runs 5–20s:
dense+BM25 seeding, 1-hop projection expansion, LLM per-note adjudication, and one
agentic second hop around strong (≥0.7) matches. Response:

```json
{ "idea": str,
  "matches": [ { "note_id": str, "title": str, "source": str,
                 "strength": float, "reason": str } ],
  "considered": int, "expanded": int, "latency_ms": float }
```

Matches are aligned-only, sorted by strength desc. Errors: 400 empty idea, 503 when no
LLM/knowledge index is configured. Malformed LLM rows are skipped (fail-open).

### Chat seeding (for "Ask about this")

Chat WS `chat` message and/or `POST /chats` accepts optional `seed_note_ids: list[str]`.
Seeds pre-populate the `ChatRetriever` per-chat seed accumulator so the first turn is
biased toward those notes. Frontend sends it when a chat is opened from a map node.

## Frontend scope (MapView rehaul)

All in `app/`; MapView + new sibling files own the bulk. Non-map file edits must stay
minimal and surgical (a concurrent agent is restyling the rest of the app).

1. **Map becomes a top-level tab** (Live | Chat | Map | Data | Connectors). Remove it from
   DataView.
2. **Selection side panel replaces the preview window.** Single click selects a node and
   opens a right-side panel: title, source chip, gist, created/modified, entity chips,
   full text (fetch `GET /notes/{id}`), neighbors list (click → select/center), and
   actions: **Focus** (ego mode), **Ask about this** (chat handoff), **Open note**
   (existing `open_note` WS for apple_notes; hide otherwise). Delete the preview window
   entirely: `preview:open` IPC + window creation in `app/src/main/index.ts`, the preload
   `openPreview` bridge, `views/PreviewView.tsx`, and its route.
3. **Local/ego mode with depth slider (1–3)** fetching `/graph/local`, with exploration
   history: breadcrumb trail of visited centers, back button, and "Global" to exit. Node
   click inside ego mode re-centers (pushes history).
4. **Hover neighborhood highlighting**: hovered node + incident edges + 1-hop neighbors at
   full opacity, everything else dimmed (reuse the existing 0.06-dim machinery).
5. **Drag + pin**: re-enable node drag; on drop set `fx/fy` and show a pin marker;
   unpin via context menu; persist pinned positions (localStorage keyed by note_id) and
   rehydrate on load so returning users keep their layout.
6. **Right-click context menu** (nodes: Open, Focus here, Ask about this, Pin/Unpin,
   Hide node; edges: show supporting facts via `/facts?ids=`). Custom lightweight menu,
   no library.
7. **Search upgrade**: match over title + gist + entity labels; dropdown of top results
   with click-to-center; Enter centers + selects the top hit; matched nodes get a
   distinct halo in addition to always-on labels.
8. **Degree/orphan control**: "hide leaf nodes" toggle or min-degree slider.
9. **Timeline scrubber** (temporal lens): horizontal brush under the canvas over
   `modified` (fall back to `created`) timestamps; nodes outside the brushed range dim;
   presets: All / Today / Week / Month. Histogram sparkline of note counts over time in
   the scrubber if cheap.
10. **Color modes** toggle: **Source** (current) | **Community** (palette by
    `community` id) | **Recency** (warm→cool by modified). Legend updates per mode.
11. **Decaying heat trail**: retrieval flash/glow currently fades in ~3s; add a
    session-scoped residue — touched nodes keep a faint warm tint decaying over ~5
    minutes, so a glance at the map shows where the session's conversation has walked.
12. **Citations open the graph**: SummaryPanel `[N#]` citation chips and chat source
    pills/citations get a "show in map" affordance → switch to Map tab, center + pulse
    that node (ego mode depth 1 centered on it). Requires small App.tsx plumbing
    (e.g. `mapFocus` state passed down; keep it surgical).
13. **Ask about this**: from side panel / context menu → switch to Chat tab, open a new
    chat with `seed_note_ids=[note_id]` and the composer prefilled with a reference to
    the note title (user edits/sends).
14. **Bigger default window**: raise `BrowserWindow` default size in
    `app/src/main/index.ts` to ~1500×950 (min ~1180×760).
15. **Design language** for all new map UI: match the app's new direction — dark neutral
    grays, subtle translucency (backdrop blur on panels), compact controls (small
    buttons, 12–13px UI text), sparing yellow `#f5c518` accents for active/selected
    states. Keep map styles self-contained (own CSS file or co-located styles), since
    the rest of the app is being migrated to Tailwind concurrently.

## Explicit non-goals (follow-ups — do NOT implement now)

- Edge-type / mechanism filtering + legend-as-filter (B10)
- Keyboard navigation (B13)
- Session subgraph view (D19)
- Chat-answer → graph retrieval-subgraph highlighting (E24)
- Multi-select / lasso → chat actions (E23)
- Structural gap / bridge surfacing (C16)
- LLM-assisted community labeling & cluster refinement (see follow-ups doc)
- Tailwind-ification of map UI (after the app-wide Tailwind migration lands)
