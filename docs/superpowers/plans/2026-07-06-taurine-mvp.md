# Taurine MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Delegation note:** per the user's direction, implementation units are executed by **Codex subagents** (`codex exec`, GPT-5.5, `xhigh` reasoning — both already the defaults in `~/.codex/config.toml`). Claude orchestrates: writes briefs, dispatches, polls logs, reviews diffs, merges. The unit tasks below are written as self-contained briefs.

**Goal:** Electron meeting copilot that streams mic audio to the Taurine ASR server and surfaces relevant Apple Notes live — ambient slate ranking, LLM-gated pinned cards, and a note-graph map showing retrieval connections in real time.

**Architecture:** Thin Electron app (audio capture + React UI) ↔ localhost ws ↔ Python sidecar (`taurine-core`: FastAPI) that owns the ASR connection, the Apple Notes index (sqlite + FTS5 + brute-force embeddings + note graph), and the v4-descended live loop (C1 exact / C2 ambient+graph+rerank / slate / presenter / cadence).

**Tech Stack:** Python 3.11+ (uv, FastAPI, websockets, sentence-transformers `BAAI/bge-small-en-v1.5`, numpy, sqlite FTS5, macnotesapp, httpx) · OpenRouter (all LLM calls) · Cohere Rerank v3.5 (direct REST) · Electron (electron-vite, React, TS, react-force-graph-2d).

**Spec:** `docs/superpowers/specs/2026-07-06-taurine-mvp-design.md` — every unit brief includes it. Where this plan and the spec disagree, the spec wins; flag the conflict instead of guessing.

## Global Constraints

- **No logging / data capture of any kind.** No decision logs, no analytics, no feedback persistence. Dismiss is pure UX.
- No custom-trained models; no replay harness; mic-only audio (but the binary frame protocol is channel-tagged); no connectors/actions; one-shot presenter (no memory); dev-mode app only (no packaging).
- Config = in-code constants in `core/taurine_core/config.py`. Env vars are for secrets ONLY: `TAURINE_ASR_TOKEN`, `OPEN_ROUTER_API_KEY`, `COHERE_API_KEY` (loaded from `core/.env`).
- All LLM calls go through `core/taurine_core/llm.py` (OpenRouter chat-completions). Cohere Rerank is the only other remote AI call.
- Python: type-hinted, `pytest` + `pytest-asyncio`, **unique test file basenames**, no test asserting exact prompt phrasing (behavioral assertions only).
- Frequent small commits per step; message prefix `feat:`/`test:`/`fix:`/`chore:`.
- ASR wire protocol (verified against `docs/reference/mic_test.py`): connect `wss://transcription.trywink.io/taurine?token=<TAURINE_ASR_TOKEN>`; send binary **float32-LE 16 kHz mono PCM**; send text `"Done"` to flush; receive JSON `{"text": str, "segment": int}` (interim) / `{"text": str, "segment": int, "is_final": 1}` (final). Only finals enter the live loop.
- Electron→sidecar binary frames are **s16-LE** 16 kHz mono prefixed with a channel byte; the sidecar converts s16→float32 (`/32768.0`) before forwarding to ASR.

---

## Phase & dependency map

```
Phase A (orchestrator, inline):  Task 0 scaffold → Task 1 contracts
Phase B (3 parallel Codex worktrees):
    Unit 1  index: store+embed+notes+enrich+graph     [core/taurine_core/indexer/*]
    Unit 2  loop: slate+cadence+C1+C2(+rerank)        [core/taurine_core/live/*]
    Unit 4  llm client + presenter                    [core/taurine_core/llm.py, live/presenter.py]
Phase C (Codex, main checkout):  Unit 3  asr + server + ingest wiring (integrates B)
Phase D (Codex, sequential):     Unit 5  Electron feed view → Unit 6 map view
Phase E (orchestrator):          integration smoke + live demo checklist
```

Units 1, 2, 4 touch disjoint files and run in parallel git worktrees off `main` (branches `unit/index`, `unit/loop`, `unit/presenter`); the orchestrator merges in that order after review. `pyproject.toml` and both contract files are frozen in Phase A so parallel units never edit shared files. **Rule for every Codex brief: do not modify `config.py`, `schemas.py`, `ports.py`, or `pyproject.toml`; if a contract is wrong, stop and report.**

## Codex dispatch protocol (orchestrator runbook)

Per unit:

1. Write the brief to `.build/briefs/unitN.md`: unit section below, verbatim, plus the Global Constraints section and the paths of the spec + contract files.
2. Dispatch detached, log-file output, never block on the process:

```bash
cd ~/taurine && mkdir -p .build/logs
nohup codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --cd <worktree-or-repo-path> \
  -o .build/logs/unitN.last.md \
  "$(cat .build/briefs/unitN.md)" \
  > .build/logs/unitN.log 2>&1 &
```

3. Brief instructs Codex to end its final message with exactly `UNIT-N DONE` or `UNIT-N BLOCKED: <reason>`. Poll `.build/logs/unitN.last.md` existence + tail `unitN.log` in short independent reads (never `wait`/`sleep`-block); on BLOCKED, read, fix the brief or contract, redispatch.
4. Review gate before merge: `git -C <worktree> diff main --stat` then full diff read; run `uv run pytest` in the worktree; check the unit's acceptance list. Merge with `git merge --no-ff unit/<name>`.

`.build/` is gitignored.

---

## Phase A — orchestrator inline

### Task 0: Repo scaffold

**Files:**
- Create: `core/pyproject.toml`, `core/.env`, `core/taurine_core/__init__.py`, `.gitignore`, `README.md`, `.build/` (gitignored)
- Create: `app/` via electron-vite scaffold

- [ ] **Step 0.1** `core/` python project:

```bash
cd ~/taurine/core && uv init --name taurine-core --python 3.12 --no-readme
uv add fastapi 'uvicorn[standard]' websockets pydantic numpy sentence-transformers macnotesapp httpx python-dotenv
uv add --dev pytest pytest-asyncio ruff
```

Set in `pyproject.toml`: `[tool.pytest.ini_options] asyncio_mode = "auto"`, `testpaths = ["tests"]`.

- [ ] **Step 0.2** Secrets → `core/.env` (never committed; `.gitignore` gets `.env`, `.build/`, `*.db`, `node_modules/`, `dist/`, `out/`, `__pycache__/`):

```bash
grep -E '^(OPEN_ROUTER_API_KEY|COHERE_API_KEY)=' ~/wink/server/.env > ~/taurine/core/.env
printf 'TAURINE_ASR_TOKEN=%s\n' "$(grep -oE 'token=[0-9a-fA-F]{16,}' ~/wink/transcription-service/Caddyfile | head -1 | cut -d= -f2)" >> ~/taurine/core/.env
awk -F= '{print $1, "len="length($2)}' ~/taurine/core/.env   # verify 3 non-empty keys, no values printed
```

- [ ] **Step 0.3** Electron scaffold:

```bash
cd ~/taurine && npm create @quick-start/electron@latest app -- --template react-ts
cd app && npm install && npm install react-force-graph-2d
```

Verify `npm run dev` opens a window, then close.

- [ ] **Step 0.4** Commit: `chore: scaffold core (uv) and app (electron-vite react-ts)`

### Task 1: Frozen contracts

**Files:**
- Create: `core/taurine_core/config.py`, `core/taurine_core/schemas.py`, `core/taurine_core/ports.py`

These three files are written in full here, committed in Phase A, and are **read-only for all units**.

- [ ] **Step 1.1** `core/taurine_core/config.py`:

```python
"""All tunables. In-code constants by convention — env vars are secrets only."""

# ── sidecar ────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8930

# ── ASR (wire protocol: see docs/reference/mic_test.py) ────────────────
ASR_URL = "wss://transcription.trywink.io/taurine"  # + ?token=<TAURINE_ASR_TOKEN>
ASR_RECONNECT_MIN_S = 1.0
ASR_RECONNECT_MAX_S = 30.0

# ── embedding / chunking ───────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, local, CPU/MPS
EMBED_DIM = 384
CHUNK_MAX_TOKENS = 300                   # sentence-boundary chunking

# ── LLM (ALL calls via OpenRouter) ─────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENRICH_MODEL = "deepseek/deepseek-chat"
PRESENTER_MODEL = "deepseek/deepseek-chat"
PRESENTER_FALLBACK_MODEL = "google/gemini-2.5-flash"
LLM_TIMEOUT_S = 25.0

# ── notes sync ─────────────────────────────────────────────────────────
SYNC_INTERVAL_S = 300.0

# ── note graph ─────────────────────────────────────────────────────────
GRAPH_KNN = 6
GRAPH_MIN_SIM = 0.60
GRAPH_MIN_SHARED_W = 0.25
GRAPH_MAX_DEGREE = 12

# ── C2 retrieval ───────────────────────────────────────────────────────
C2_THROTTLE_S = 8.0
C2_WINDOW_CHUNKS = 3
C2_WINDOW_MIN_S = 12.0
RRF_K = 60
EXPAND_SEEDS = 5
HOP_DECAY = 0.5
RERANK_MODEL = "rerank-v3.5"
RERANK_TIMEOUT_S = 2.0
RERANK_MAX_CANDIDATES = 20
C2_TOP_K = 5
MIN_COSINE = 0.35                        # floor only when no BM25 hit AND no graph path

# ── C1 exact collector ─────────────────────────────────────────────────
ALIAS_WORD_BOUNDARY_UNDER = 4            # aliases shorter than this need word-boundary match

# ── slate ──────────────────────────────────────────────────────────────
QUOTA_EXACT = 3
QUOTA_SEARCH = 8
DECAY_EXACT = 2
DECAY_SEARCH = 2
SUPPRESS_S = 300.0                       # dismissed note re-entry suppression

# ── cadence ────────────────────────────────────────────────────────────
DEBOUNCE_S = 3.0
COOLDOWN_S = 20.0
MAX_PROMOS_PER_MIN = 3
EMPTY_BACKOFF_S = 15.0
REOFFER_S = 60.0

# ── presenter / cards ──────────────────────────────────────────────────
PRESENTER_TRANSCRIPT_CHUNKS = 15
PRESENTER_MAX_PICKS = 2
WHY_MAX_CHARS = 140
MAX_PINNED = 3
```

- [ ] **Step 1.2** `core/taurine_core/schemas.py`:

```python
"""Pydantic models: ws protocol, REST payloads, and cross-unit data shapes."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel

# ── transcript ─────────────────────────────────────────────────────────
class Chunk(BaseModel):
    channel: int
    text: str
    ts: float                      # sidecar wall-clock at finalization
    final: bool = True

# ── index shapes ───────────────────────────────────────────────────────
class Enrichment(BaseModel):
    gist: str                      # <=120 chars
    utterances: list[str]          # 4 synthetic spoken sentences
    aliases: list[str]             # <=6
    topics: list[str]              # 2-4, lowercase normalized
    entities: list[str]

class NoteRecord(BaseModel):
    note_id: str
    title: str
    folder: str = ""
    modified: float
    content_hash: str
    text: str                      # plain text body
    gist: str = ""                 # filled after enrichment (fallback: first line)

EdgeType = Literal["semantic", "topic", "entity", "mixed"]

class GraphEdge(BaseModel):
    src: str
    dst: str                       # src < dst (undirected, canonical order)
    type: EdgeType
    weight: float                  # combined: 1 - prod(1 - w_i)
    shared: list[str] = []         # topics/entities behind the edge

class GraphNode(BaseModel):
    note_id: str
    title: str
    degree: int

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]

# ── retrieval ──────────────────────────────────────────────────────────
class Hit(BaseModel):
    note_id: str
    rank: int                      # 1-based, post-rerank
    score: float

class Expansion(BaseModel):
    from_note_id: str              # the seed it was reached from
    note_id: str
    hop: int                       # 1 at MVP
    score: float

class RetrievalResult(BaseModel):
    heard: str                     # the window text
    hits: list[Hit]
    expansions: list[Expansion]

# ── slate / cards ──────────────────────────────────────────────────────
Source = Literal["exact", "search"]

class SlateItem(BaseModel):
    note_id: str
    title: str
    gist: str
    score: float
    source: Source
    heard: str
    views: int = 0
    decay: int
    added_rev: int
    via_note_id: Optional[str] = None   # set when reached via graph expansion
    hop: Optional[int] = None

class Card(BaseModel):
    card_id: str
    note_id: str
    title: str
    summary: str                   # gist (instant pin) or presenter's why
    provenance: Literal["exact", "search", "linked"]
    heard: str

class Pick(BaseModel):
    note_id: str
    why: str                       # <=WHY_MAX_CHARS, validated

# ── ws events (server → client) ────────────────────────────────────────
class TranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    channel: int
    text: str
    final: bool
    ts: float

class AmbientItem(BaseModel):
    note_id: str
    title: str
    gist: str
    score: float

class AmbientEvent(BaseModel):
    type: Literal["ambient"] = "ambient"
    items: list[AmbientItem]

class PinEvent(BaseModel):
    type: Literal["pin"] = "pin"
    card: Card

class UnpinEvent(BaseModel):
    type: Literal["unpin"] = "unpin"
    card_id: str

class RetrievalEvent(BaseModel):
    type: Literal["retrieval"] = "retrieval"
    heard: str
    hits: list[Hit]
    expansions: list[Expansion]

class IndexStatus(BaseModel):
    notes: int
    last_sync: Optional[float] = None

class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    asr: Literal["disconnected", "connecting", "connected", "error"]
    index: IndexStatus
    detail: str = ""

# ── client → server control messages ───────────────────────────────────
class StartMsg(BaseModel):
    type: Literal["start"]
    channels: list[dict]           # [{"id": 0, "label": "mic"}]

class StopMsg(BaseModel):
    type: Literal["stop"]

class DismissMsg(BaseModel):
    type: Literal["dismiss"]
    card_id: str

class OpenNoteMsg(BaseModel):
    type: Literal["open_note"]
    note_id: str
```

- [ ] **Step 1.3** `core/taurine_core/ports.py` — the cross-unit seams (unit 1 implements `IndexPort`; unit 2 consumes it via fakes; unit 4 implements `LLMPort`):

```python
"""typing.Protocol seams between delegation units. Frozen in Phase A."""
from __future__ import annotations

from typing import Protocol
from .schemas import Enrichment, GraphResponse, NoteRecord

class IndexPort(Protocol):
    """Implemented by indexer.store.Index (unit 1). Read side used by the live loop."""

    def dense_search(self, text: str, k: int) -> list[tuple[str, float]]:
        """Embed text, cosine vs all units, per-note max. -> [(note_id, score)] desc."""

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        """FTS5 BM25 over title+body. -> [(note_id, score)] desc, empty on no match."""

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        """Note-graph neighbors -> [(note_id, edge_weight)] desc."""

    def note(self, note_id: str) -> NoteRecord | None: ...

    def best_chunk(self, note_id: str, query_text: str) -> str:
        """Chunk of the note most similar to query_text (for rerank docs)."""

    def aliases(self) -> list[tuple[str, str, bool]]:
        """[(alias_lowercase, note_id, is_title)] for C1."""

    def graph(self) -> GraphResponse: ...

    def status(self) -> tuple[int, float | None]:
        """(note_count, last_sync_ts)"""

class RerankerPort(Protocol):
    async def rerank(self, query: str, docs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """docs=[(note_id, doc_text)] -> [(note_id, relevance)] desc.
        Raises RerankError on failure/timeout; caller falls back."""

class RerankError(Exception):
    pass

class LLMPort(Protocol):
    async def complete_json(self, model: str, system: str, user: str) -> dict:
        """One chat completion via OpenRouter, response_format json_object.
        Raises LLMError on HTTP/parse failure."""

class LLMError(Exception):
    pass
```

- [ ] **Step 1.4** `uv run python -c "import taurine_core.config, taurine_core.schemas, taurine_core.ports"` → no error. Commit: `feat: frozen contracts (config, schemas, ports)`. Then create the three worktrees:

```bash
cd ~/taurine
git worktree add .build/wt-index -b unit/index
git worktree add .build/wt-loop -b unit/loop
git worktree add .build/wt-presenter -b unit/presenter
```

---

## Phase B — parallel Codex units

### Unit 1: Index — store, embed, notes, enrich, graph  *(worktree `.build/wt-index`, branch `unit/index`)*

**Files:**
- Create: `core/taurine_core/indexer/__init__.py`, `store.py`, `embed.py`, `notes.py`, `enrich.py`, `graph.py`, `sync.py`
- Test: `core/tests/conftest.py` (fixture corpus), `core/tests/test_index_store.py`, `test_index_embed.py`, `test_index_graph.py`, `test_index_sync.py`

**Interfaces:**
- Consumes: `config`, `schemas` (`NoteRecord`, `Enrichment`, `GraphEdge`, `GraphResponse`), `ports.IndexPort`, `ports.LLMPort` (enrichment calls go through an injected `LLMPort` — do NOT implement an OpenRouter client here; tests use a stub).
- Produces: `indexer.store.Index` implementing every `IndexPort` method exactly as documented; `indexer.sync.sync_once(index, notes_source, llm) -> SyncReport(added, changed, removed, enrich_failures)`; CLI `python -m taurine_core.indexer.sync`.

**Requirements (each becomes at least one test; TDD — write the failing test first, then implement, then commit per module):**

1. **Store** (`store.py`): single sqlite file; tables `notes`, `aliases`, `enrich_cache` (keyed `(note_id, content_hash)`), `embeddings` (unit rows: `note_id, kind ∈ {chunk,gist,utterance}, text, vec BLOB`), `edges`, FTS5 virtual table over title+body kept in sync on upsert/delete. In-memory numpy matrix (`float32`, unit rows) rebuilt on `load()` and after sync; `dense_search` = brute-force cosine, per-note max over its units. `Index(db_path)` implements `IndexPort`.
2. **Embed** (`embed.py`): sentence-transformers wrapper, lazy singleton model load, `embed_texts(list[str]) -> np.ndarray (n, EMBED_DIM)`, L2-normalized. Chunking: sentence-boundary packing up to `CHUNK_MAX_TOKENS` (whitespace-token count is fine as the token proxy).
3. **Notes** (`notes.py`): `macnotesapp` extraction of all notes in the default account → `NoteRecord` (HTML→plain text; `content_hash = sha256(title + text)`). Also `open_note(note_id)` via AppleScript `show note id …` (subprocess `osascript`, no assertion tests needed — mark with a `pytest.mark.skipif(not sys.platform == "darwin")` smoke marker, skipped in CI-less runs). A permission/AppleScript failure raises `NotesAccessError` — callers keep the last good index.
4. **Enrich** (`enrich.py`): one `LLMPort.complete_json` call per new/changed note → `Enrichment`; prompt must request the exact JSON keys of the `Enrichment` schema (test the *parsing/validation* — never assert prompt wording). Response validation clamps: gist ≤120 chars, ≤6 aliases, 2–4 topics (lowercase-normalized). On `LLMError`: fallback enrichment = `gist=first line (≤120)`, `aliases=[title]`, empty topics/entities/utterances — note still indexed.
5. **Graph** (`graph.py`): per spec —
   - semantic: per-note vector = gist embedding (fallback: mean of chunk embeddings); top `GRAPH_KNN` neighbors with cosine ≥ `GRAPH_MIN_SIM`, weight = cosine;
   - topic/entity: shared term weight damped by corpus frequency — `w(term) = 1 / (1 + log(1 + n_notes_with_term))`, sum over shared terms, floor `GRAPH_MIN_SHARED_W`;
   - merge parallel edges: `weight = 1 - Π(1 - wᵢ)`, `type = "mixed"` when mechanisms combine, `shared` carries the terms;
   - degree cap `GRAPH_MAX_DEGREE` keeping strongest edges; undirected canonical `(src < dst)`;
   - incremental: `rebuild_for(note_id)` recomputes only edges incident to that note.
6. **Sync** (`sync.py`): full pass = extract → diff by `(note_id, content_hash)` → for added/changed: enrich (cached by hash), chunk, embed, store, `rebuild_for`; for removed: delete note+units+edges. `SyncReport` printed by the CLI. CLI smoke target: `uv run python -m taurine_core.indexer.sync` runs against real Notes.

**Key test cases (write these verbatim, plus your own):**

```python
# test_index_graph.py
def test_semantic_edge_from_similar_gists(fixture_index):
    # notes "roadmap-q3" and "roadmap-q4" have near-identical gists in the fixture
    edges = {(e.src, e.dst): e for e in fixture_index.graph().edges}
    assert any("roadmap-q3" in k and "roadmap-q4" in k for k in edges)

def test_common_topic_is_damped(fixture_index):
    # topic "work" is on >half the fixture notes -> alone it must not clear GRAPH_MIN_SHARED_W
    e = edges_between(fixture_index, "standup-notes", "grocery-list")
    assert e is None

def test_degree_cap(fixture_index_hub):
    # hub note connected to 20 candidates keeps only GRAPH_MAX_DEGREE strongest
    assert degree_of(fixture_index_hub, "hub") <= 12

def test_incremental_rebuild_touches_only_incident_edges(fixture_index):
    before = {frozenset((e.src, e.dst)) for e in fixture_index.graph().edges
              if "roadmap-q3" not in (e.src, e.dst)}
    fixture_index.rebuild_for("roadmap-q3")
    after = {frozenset((e.src, e.dst)) for e in fixture_index.graph().edges
             if "roadmap-q3" not in (e.src, e.dst)}
    assert before == after

# test_index_sync.py
def test_unchanged_hash_skips_enrichment(store, stub_llm):
    sync_once(store, source_with(NOTE_A), stub_llm)
    stub_llm.calls.clear()
    sync_once(store, source_with(NOTE_A), stub_llm)   # same content
    assert stub_llm.calls == []

def test_enrich_failure_still_indexes(store, failing_llm):
    report = sync_once(store, source_with(NOTE_A), failing_llm)
    assert report.enrich_failures == 1
    assert store.note(NOTE_A.note_id) is not None
    assert store.bm25_search(NOTE_A.title, 5)          # searchable on raw text
```

**Fixture corpus** (`conftest.py`): ~15 hand-written fake notes as `NoteRecord`+`Enrichment` literals, covering: two near-duplicate roadmap notes, one hub note sharing entities with many, one note with a ubiquitous topic ("work"), short-title notes ("Q3", for C1 boundary tests in unit 2 — keep names consistent), a grocery list (irrelevant filler). Deterministic stub embeddings for graph unit tests where noted; real model allowed in a marked `slow` test.

**Acceptance:** `uv run pytest core/tests -q` green; `uv run python -m taurine_core.indexer.sync` prints a `SyncReport` against real Apple Notes (run manually by orchestrator at review). End final message with `UNIT-1 DONE`.

### Unit 2: Live loop — slate, cadence, C1, C2  *(worktree `.build/wt-loop`, branch `unit/loop`)*

**Files:**
- Create: `core/taurine_core/live/__init__.py`, `slate.py`, `cadence.py`, `exact.py`, `ambient.py`, `rerank.py`
- Test: `core/tests/test_live_slate.py`, `test_live_cadence.py`, `test_live_exact.py`, `test_live_ambient.py`, `test_live_rerank.py`, `core/tests/fakes.py` (FakeIndex implementing `IndexPort`, FakeClock)

**Interfaces:**
- Consumes: `config`, `schemas`, `ports.IndexPort` (via `fakes.FakeIndex`), `ports.RerankerPort`/`RerankError`.
- Produces (unit 3 wires these):
  - `slate.Slate`: `add(item: SlateItem) -> None` (handles quota eviction, renewal, suppression), `kick(note_id, reason)`, `dismiss(note_id, now)`, `mark_viewed(note_ids)`, `items() -> list[SlateItem]` sorted (provenance strength, score), `revision: int`.
  - `cadence.Cadence(clock)`: `may_present(revision, now) -> bool`, `note_attempt(now)`, `note_promotion(now)`, `note_empty(revision, now)`, `instant_pin_allowed(now) -> bool` (rate cap only, no cooldown started).
  - `exact.ExactCollector(index)`: `on_chunk(chunk: Chunk) -> list[C1Hit]` where `C1Hit = (note_id, matched_alias, is_title)`.
  - `ambient.AmbientRetriever(index, reranker, clock)`: `async def on_window(window_text: str) -> RetrievalResult | None` (None when throttled) — full 4-step pass; also returns the slate adds via `.last_slate_adds: list[SlateItem]`.
  - `rerank.CohereReranker(api_key)` implementing `RerankerPort` (httpx POST `https://api.cohere.com/v2/rerank`, model `RERANK_MODEL`, timeout `RERANK_TIMEOUT_S`, raise `RerankError` on any failure).

**Requirements:**

1. **Slate** per spec: quotas exact 3 / search 8 with same-lane eviction (most-viewed, then oldest); pinned/decayed/dismissed kicks; renewal on re-add (views reset, provenance upgrades search→exact only); dismissed note_ids suppressed `SUPPRESS_S` (re-adds silently dropped); revision bumps on every add/kick.
2. **Cadence** per spec, injected clock, pure state machine: revision-change requirement, `DEBOUNCE_S`, `COOLDOWN_S`, `MAX_PROMOS_PER_MIN` (rolling 60 s window), `EMPTY_BACKOFF_S` keyed to revision, `REOFFER_S` for unchanged slate, instant-pin bypass (rate cap only).
3. **C1**: alias table from `index.aliases()`, normalized (lowercase, punctuation → space, collapse whitespace); aliases with < `ALIAS_WORD_BOUNDARY_UNDER` characters require word-boundary regex match; title hits flagged `is_title=True` (unit 3 turns those into instant pins).
4. **C2** 4-step pass exactly per spec §C2: RRF(k=60) over dense+bm25 → 1-hop expansion from top `EXPAND_SEEDS` seeds (`score = seed_fused × edge_w × HOP_DECAY`; a note reached both ways stays direct) → rerank ≤ `RERANK_MAX_CANDIDATES` pooled candidates (doc text = `title + " — " + gist + "\n" + best_chunk`), `RerankError`/timeout → fused order fallback → top `C2_TOP_K` above the floor (`MIN_COSINE` applied only to notes with neither BM25 hit nor graph path) become `SlateItem`s (`source="search"`, `decay=DECAY_SEARCH`, `via_note_id`/`hop` for expanded arrivals). Returns `RetrievalResult` with 1-based post-rerank ranks.

**Key test cases (verbatim, plus your own):**

```python
# test_live_slate.py
def test_search_lane_evicts_most_viewed_then_oldest(slate_full_search_lane): ...
def test_renewal_resets_views_and_upgrades_provenance(slate): ...
def test_dismissed_note_suppressed_for_300s(slate, clock):
    slate.add(item("n1")); slate.dismiss("n1", now=clock.now())
    slate.add(item("n1"))                       # within suppression
    assert "n1" not in [i.note_id for i in slate.items()]
    clock.advance(300.1); slate.add(item("n1"))
    assert "n1" in [i.note_id for i in slate.items()]

# test_live_cadence.py
def test_same_revision_not_represented_until_reoffer(cadence, clock): ...
def test_empty_backoff_keyed_to_revision(cadence, clock):
    assert cadence.may_present(rev=5, now=clock.now())
    cadence.note_empty(rev=5, now=clock.now())
    clock.advance(4)
    assert not cadence.may_present(rev=5, now=clock.now())   # backoff holds
    assert cadence.may_present(rev=6, now=clock.now())       # new revision escapes
def test_rate_cap_counts_instant_pins(cadence, clock): ...

# test_live_ambient.py
async def test_expanded_neighbor_joins_pool_with_decayed_score(fake_index, ok_reranker):
    # fake_index: "seed" ranks #1 by dense; "neighbor" linked w/ weight 0.8, absent from search
    r = AmbientRetriever(fake_index, ok_reranker, FakeClock())
    res = await r.on_window("talk about the seed topic")
    exp = [e for e in res.expansions if e.note_id == "neighbor"]
    assert exp and exp[0].hop == 1 and exp[0].from_note_id == "seed"

async def test_rerank_failure_falls_back_to_fused_order(fake_index, failing_reranker):
    res = await AmbientRetriever(fake_index, failing_reranker, FakeClock()).on_window("x")
    assert res is not None and res.hits            # pass survived

async def test_direct_beats_expanded_for_dual_path_note(fake_index_dualpath, ok_reranker):
    res = await AmbientRetriever(...).on_window("...")
    assert all(e.note_id != "dual" for e in res.expansions)   # counted as direct hit
```

**Acceptance:** `uv run pytest core/tests -q` green (FakeIndex only — no model download, no network; `CohereReranker` tested against a mocked httpx transport). End with `UNIT-2 DONE`.

### Unit 4: LLM client + presenter  *(worktree `.build/wt-presenter`, branch `unit/presenter`)*

**Files:**
- Create: `core/taurine_core/llm.py`, `core/taurine_core/live/presenter.py`
- Test: `core/tests/test_llm_client.py`, `core/tests/test_live_presenter.py`

**Interfaces:**
- Consumes: `config`, `schemas` (`Chunk`, `SlateItem`, `Pick`, `Card`), `ports.LLMPort`/`LLMError`.
- Produces:
  - `llm.OpenRouterClient(api_key)` implementing `LLMPort`: httpx POST `{OPENROUTER_BASE_URL}/chat/completions`, `response_format={"type":"json_object"}`, timeout `LLM_TIMEOUT_S`, `LLMError` on HTTP/parse failure.
  - `presenter.Presenter(llm: LLMPort)`: `async def present(transcript: list[Chunk], slate: list[SlateItem]) -> list[Pick]`.

**Requirements:**

1. Prompt assembly per spec: `<recent_transcript>` = last `PRESENTER_TRANSCRIPT_CHUNKS` chunks channel-labeled; `<slate>` = one line per item with provenance tags — `[exact: title spoken]`, `[search: matched "…heard…"]`, `[linked: connected to "<via note title>", matched "…"]` — the via title comes from the `via_titles: dict[str, str]` parameter (note_id → title), which the caller (unit 3 ingest) builds from the slate items' `via_note_id`s. Contract lines: pick 0–2, copy note_id exactly, moment-wins, provenance strength exact > search > linked, empty is valid.
2. Response validation: JSON `{"picks": [{"note_id", "why"}]}`; drop picks whose note_id is not on the slate; truncate to `PRESENTER_MAX_PICKS`; clamp `why` to `WHY_MAX_CHARS`.
3. Model ladder: `PRESENTER_MODEL`, on `LLMError` retry once with `PRESENTER_FALLBACK_MODEL`, on second failure return `[]` (silent skip).
4. Tests: mocked `LLMPort` — unknown-note_id dropped, >2 picks truncated, both-models-fail → `[]`, fallback used exactly once. **No prompt-substring assertions**; assert structure only (e.g. every slate note_id appears in the prompt — that's data, not phrasing).

**Acceptance:** `uv run pytest core/tests -q` green, no network in tests. End with `UNIT-4 DONE`.

---

## Phase C — Unit 3: ASR client + server + ingest  *(main checkout, after B merges)*

**Files:**
- Create: `core/taurine_core/asr.py`, `core/taurine_core/server.py`, `core/taurine_core/live/ingest.py`, `core/taurine_core/devfeed.py`
- Test: `core/tests/test_asr_protocol.py`, `test_server_ws.py`, `test_live_ingest.py`

**Interfaces:**
- Consumes: everything Phase B produced (real `Index`, `Slate`, `Cadence`, `ExactCollector`, `AmbientRetriever`, `CohereReranker`, `OpenRouterClient`, `Presenter`) + `config` + `schemas`.
- Produces: `uvicorn taurine_core.server:app` — the complete sidecar; `python -m taurine_core.devfeed` (stdin chunk feeder).

**Requirements:**

1. **ASR client** (`asr.py`), protocol per `docs/reference/mic_test.py` (copied into this repo): one ws connection per channel to `ASR_URL?token=…`; sends float32-LE frames (converted from the s16-LE the app sends: `np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0`); on session stop sends `"Done"` and drains; parses interim/final JSON into `Chunk`s (`final = bool(d.get("is_final"))`); auto-reconnect with exponential backoff `ASR_RECONNECT_MIN_S→MAX_S`, state changes surfaced as a callback for `StatusEvent`s. Audio frames arriving while disconnected are dropped (no buffering).
2. **Ingest** (`ingest.py`): the session object wiring the loop — on final chunk: transcript event out; C1 pass (title hit → instant pin if `cadence.instant_pin_allowed`, card summary = gist, provenance `exact`, and slate add for non-title hits); rolling window update; C2 pass when throttle allows (slate adds + `RetrievalEvent` out + ambient event on revision change); presenter attempt when `cadence.may_present` (picks → `Card`s provenance `search`/`linked` per item, pin events out, `mark_viewed` on all prompted items, kick pinned, `note_promotion`/`note_empty`). Pinned stack bookkeeping: max `MAX_PINNED`, displacement emits `UnpinEvent`. Dismiss control → `slate.dismiss` + unpin.
3. **Server** (`server.py`): FastAPI; `GET /graph` → `GraphResponse`; `POST /index/sync` → `SyncReport` JSON; `WS /session` — text frames parsed into the control messages, binary frames split `[channel u8][pcm s16le]` → ASR client; events serialized from the schema models. Background tasks: periodic sync every `SYNC_INTERVAL_S`, startup index load. `.env` loaded via python-dotenv at startup; missing key → clear startup error naming the key.
4. **Devfeed** (`devfeed.py`): reads lines from stdin, each line becomes a final `Chunk(channel=0)` into a live session (no ASR, no audio) and prints emitted events as JSON lines — the headless smoke path for the whole loop.
5. Tests: binary frame parse/route; ws event serialization round-trip; ingest orchestration with fakes (C1 title hit produces pin event + no cooldown start; presenter empty → no pin + backoff noted; dismiss suppresses). ASR framing tested against a local fake ws server fixture (send two interims + a final, assert one `Chunk` with `final=True`).

**Acceptance:** full `uv run pytest core/tests -q` green; `echo "remind me about the q3 roadmap deck" | uv run python -m taurine_core.devfeed` (with a synced real index + real keys) prints transcript/ambient/retrieval events and — cadence permitting — a pin. Orchestrator verifies live ASR manually with a wav/mic before Phase D. End with `UNIT-3 DONE`.

---

## Phase D — Electron app

### Unit 5: Feed view + chrome + audio  *(main checkout, `app/`)*

**Files:**
- Create/modify: `app/src/renderer/src/` — `App.tsx`, `views/FeedView.tsx`, `components/{PinnedStack,AmbientList,StatusBar,SessionControls}.tsx`, `lib/sidecar.ts` (ws client + event types mirroring `schemas.py`), `lib/audio.ts` (worklet loader), `public/pcm-worklet.js`
- Test: none automated (spec: app verified manually); `lib/sidecar.ts` message parsing gets a small vitest only if trivial to add — otherwise skip, do not build test infra.

**Interfaces:**
- Consumes: sidecar ws protocol verbatim from `schemas.py` / spec §protocol (`ws://127.0.0.1:8930/session`).
- Produces: working Feed view; `lib/sidecar.ts` exports `SidecarClient` with typed event callbacks (`onTranscript`, `onAmbient`, `onPin`, `onUnpin`, `onRetrieval`, `onStatus`) — unit 6 reuses it.

**Requirements:**

1. **Audio**: `getUserMedia({audio: {sampleRate: 48000, channelCount: 1}})` → AudioWorklet downsampling to 16 kHz s16-LE, ~250 ms binary frames prefixed with channel byte `0`, sent over the ws. Start/stop controls send `start`/`stop` control messages and open/close the mic.
2. **Layout**: persistent top pinned stack (≤3 cards: title, summary, provenance chip, Open-in-Notes button → `open_note` msg, dismiss × → `dismiss` msg) + persistent right sidebar ambient list (title, gist, score bar, full replacement on `ambient` events) + main column = Feed view (finalized chunks; interims shown dim inline and replaced by finals) + Feed/Map view switch (Map is a placeholder pane in this unit) + status bar (ASR state, note count, last sync, sync-now button → `POST /index/sync`).
3. Reconnect: sidecar ws retry with backoff + disconnected banner.
4. Styling: minimal, dark, readable; no component library; CSS modules or plain CSS.

**Acceptance:** `npm run dev` against a running sidecar: speak → transcript renders, ambient list moves, a pin renders with working dismiss + Open in Notes (macOS Notes opens). End with `UNIT-5 DONE`.

### Unit 6: Map view  *(main checkout, `app/`, after unit 5)*

**Files:**
- Create: `app/src/renderer/src/views/MapView.tsx`, `lib/graph.ts`
- Modify: `App.tsx` (mount real MapView)

**Interfaces:**
- Consumes: `GET http://127.0.0.1:8930/graph` (`GraphResponse` shape), `SidecarClient.onRetrieval`/`onPin` from unit 5.
- Produces: the live map per spec §Map view.

**Requirements:**

1. `react-force-graph-2d`; nodes from `/graph` (label = title, radius ∝ degree, re-fetched after a `status` event whose `last_sync` changed); static edges faint, opacity ∝ weight, hover tooltip = type + shared terms.
2. A fixed **"now" node** (pinned position, distinct color; label = last ~6 words of the latest final chunk).
3. On each `retrieval` event: replace previous live edges — now→hit edges colored on a rank gradient (rank 1 = `#ff5c5c` hot → rank K = `#4c7dff` cool, interpolated), hit→expansion edges on a separate dimmer hop scale with hop count rendered on the edge; touched nodes flash; previous live edges fade out (CSS-free: animate opacity in the canvas paint via a decay timestamp).
4. Pinned notes (from `pin` events) get a persistent ring for the session.
5. View switch from unit 5 shows/hides Map without losing force-layout state (keep component mounted, toggle visibility).

**Acceptance:** with the sidecar running and someone talking: static graph renders; rank-colored edges appear from the now-node each C2 pass and fade on the next; expanded neighbors visibly distinct (hop label); pinned note ringed. End with `UNIT-6 DONE`.

---

## Phase E — integration smoke + demo (orchestrator inline)

- [ ] **E.1** Fresh sync against real Notes: `cd core && uv run python -m taurine_core.indexer.sync` → SyncReport with real counts, spot-check 3 notes' gists/topics in sqlite.
- [ ] **E.2** Headless loop smoke: pipe 6 conversational lines (3 on-topic for known notes, 3 chitchat) through `devfeed`; verify ambient movement on on-topic lines, at most one pin, **zero pins for chitchat**.
- [ ] **E.3** Live end-to-end: sidecar + `npm run dev`; mic session touching 2 known note topics + 1 spoken note title; verify: title → instant pin ≤1 s; topic → ambient ≤ ~2 s, pin ≤ ~10 s (cadence permitting); map shows rank/hop edges live; Open in Notes works; dismiss suppresses.
- [ ] **E.4** Update README (run instructions: sidecar, app, sync CLI). Final commit + `git tag mvp-0.1`.

---

## Self-review (done at plan time)

- **Spec coverage:** every spec section maps to a unit (indexer→U1, graph→U1, C1/C2/slate/cadence→U2, rerank→U2, presenter/LLM→U4, ASR/server/ingest/devfeed→U3, feed UI→U5, map→U6, error table→U1/U2/U3 requirements, milestones→phases). Spec's "no logging" and env-key names are Global Constraints.
- **Type consistency:** all cross-unit names are pinned in `schemas.py`/`ports.py` (frozen Phase A); unit briefs reference only those names.
- **Placeholder scan:** no TBDs; behavioral requirements carry exact constants from `config.py`.
