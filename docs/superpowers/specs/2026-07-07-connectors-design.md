# Taurine Connectors — Design Spec

**Date:** 2026-07-07 · **Status:** approved (implement immediately)

Generalize the index from "Apple Notes" to **Connectors**: pluggable local data
sources feeding the same store/enrichment/graph/retrieval pipeline. Ship four
new local connectors (Claude memories, Codex memories, Obsidian, iMessage)
plus the existing Apple Notes as the first-class reference connector, a
**Data tab** in the app (map on top, connections underneath), **graph search
by node type**, and a **note preview window** (separate Electron window).

## Principles

- **Local-only for now** (no OAuth) but the abstraction must extend to
  Google Docs / Linear / Jira later: a connector is anything that can
  `extract()` a list of documents; auth is a connector-internal concern.
- Everything downstream (enrichment incl. user card, embeddings, graph,
  C1/C2, slate, presenter) stays **source-agnostic**: connectors produce
  `NoteRecord`s. Sync/re-enrich move under the connectors framework.
- Document identity: `note_id` is namespaced per connector
  (`imessage:<chat-guid>`, `obsidian:<vault-rel-path>`, `claude-mem:<path>`,
  `codex-mem:<path>`); existing Apple Notes keep their x-coredata ids
  (`source` field distinguishes them — no id migration).

## Core

### Connector abstraction (`core/taurine_core/connectors/`)

`ConnectorPort` (frozen in ports.py): `connector_id`, `label`, `kind`
("local" | "oauth"), `available() -> (bool, detail)`,
`default_config() -> dict`, `extract(config) -> list[NoteRecord]`.

- `base.py` helpers (file→NoteRecord, hashing, char caps) ·
  `apple_notes.py` (wraps existing extractor) · `claude_memories.py` ·
  `codex_memories.py` · `obsidian.py` · `imessage.py` ·
  `catalog.py` (the CONNECTORS list) · `registry.py` (state + orchestration).
- **State** (store `meta` kv or a `connectors` table): per-connector
  `enabled` (apple_notes defaults on, others off), `config` (JSON, e.g.
  obsidian vault path), `last_sync`, `last_report`, doc count derived from
  notes table by source.
- **Sync**: per-connector `sync_connector(id)` = extract → diff by
  (note_id, content_hash) **scoped to that source** → enrich (user card as
  today) → embed → graph rebuild_for. "Sync all" iterates enabled
  connectors. Documents of a disabled connector stay indexed but a
  **disable purge option** (`?purge=true`) deletes them. Re-enrich is
  index-wide and unchanged.

### The four new connectors

- **Claude memories**: glob `~/.claude/projects/*/memory/*.md` (+ config
  override). One doc per file; title = first heading or filename; folder =
  project dir name.
- **Codex memories**: default globs `~/.codex/AGENTS.md`,
  `~/.codex/memor{y,ies}/**/*.md` — tolerate absence (available=false with
  detail when nothing exists).
- **Obsidian**: vault path from config; default discovery via
  `~/Library/Application Support/obsidian/obsidian.json` (first vault).
  All `*.md` excluding `.obsidian/` and templates dirs; per-file char cap
  `CONNECTOR_DOC_MAX_CHARS`.
- **iMessage** (macOS): read `~/Library/Messages/chat.db` — **copy to a
  temp file first** (the live db is locked/WAL) and open read-only. One
  document per conversation (chat), titled "iMessage: <display name or
  handle>". Content = transcript lines `[me]|[them] <text>` newest-last,
  taking messages from newest backwards bounded by
  `IMESSAGE_WINDOW_DAYS=14` **or** `IMESSAGE_MAX_CHARS=5000`, whichever
  bound hits first — the raw db bulk is never stored. Apple epoch
  (2001-01-01, ns) date conversion; `message.text` with `attributedBody`
  fallback decoding for newer macOS rows. Requires Full Disk Access →
  `available()` returns false with a remediation detail when unreadable.

### REST

- `GET /connectors` → `list[ConnectorInfo]`
- `POST /connectors/{id}/enable` · `/disable` (`?purge=true` optional) ·
  `/sync` (single-flight per connector) · `PUT /connectors/{id}/config`
- `GET /notes/{note_id}` → `NotePreview` (title, source, gist, text,
  neighbors with titles/weights/sources) — powers the preview window.
- `POST /index/sync` becomes "sync all enabled"; `/index/reenrich` unchanged.
- `GET /graph` nodes gain `source`.

## App

### Tabs

Top-level tabs replace the Feed/Map switch: **Live** (transcript feed +
pinned + ambient, as today) and **Data** (the map on top, connections panel
underneath). Status bar stays global.

### Data tab

- **Map (top ~60%)**: as today plus — node colors by `source` (legend
  updated); a **search box** filtering/highlighting nodes by title substring
  with **type filter chips** (one per connector source, toggleable); nodes
  failing the filter fade to near-invisible; matches keep full opacity
  (and the view zooms to fit matches when a query is active).
- **Connections panel (below)**: one row/card per connector — icon/label,
  availability state (with remediation detail when unavailable), enable
  toggle, doc count, last sync, per-connector **Sync** button with spinner,
  config affordance where relevant (Obsidian vault path input). Apple Notes
  row carries the Re-enrich control (moved from the status bar).

### Preview window

Clicking a map node (or a search result) opens a **separate Electron
window** (single reusable preview window): renderer hash route
`#/preview/<note_id>` → fetches `GET /notes/{id}` → shows title + source
badge + gist + scrollable content preview + a scrolling list of connected
notes (title, source, edge weight) — clicking one switches the preview to
that note in place. Main process exposes `api.openPreview(noteId)` via
preload IPC.

## Out of scope (this round)

OAuth connectors, per-connector enrichment prompts, iMessage attachments,
incremental iMessage tail-updates beyond hash-diff re-extraction, moving
re-enrich off the index level.
