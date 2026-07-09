# Agentic User Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace enrichment-coupled profile generation with an explicit GLM-5.2 graph-research agent, a native research-rail UI, and runtime profile context for chat and live summarization.

**Architecture:** A focused `ProfileGenerator` owns the GLM-5.2 tool loop and reports semantic progress through a callback. FastAPI owns single-flight task lifecycle and broadcasts a typed profile payload over the existing update bus. The renderer treats that payload as one state object, while chat and live summarization read the latest persisted card when building each prompt.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, asyncio, OpenRouter streaming tools, SQLite, pytest; React 19, TypeScript 5.9, Electron/Vite, CSS.

## Global Constraints

- User-summary generation must never run from startup, sync, or re-enrichment.
- The model is exactly `z-ai/glm-5.2`, with reasoning enabled for profile research.
- Successful generation overwrites immediately; failure preserves the saved card.
- Existing content requires UI confirmation before generation; an empty card does not.
- Profile context is background only and never becomes citable evidence.
- Use Taurine's existing `#0b0b0c` / `#131316` / `#1c1c20` gray surfaces and `#f5c518` / `#ffd75e` yellow accent tokens.
- Preserve unrelated workspace changes and add no new dependencies.

---

### Task 1: Profile agent tool loop

**Files:**
- Modify: `core/taurine_core/config.py`
- Replace: `core/taurine_core/indexer/profile.py`
- Replace tests: `core/tests/test_index_profile.py`

**Interfaces:**
- Consumes: `OpenRouterClient.stream_chat`, `StreamEvent`, `ChatRetriever`, `TOOL_DEFINITIONS`, `execute_tool`, and `Index`.
- Produces: `ProfileProgress(phase, activity, nodes_read, tool_calls)` and `ProfileGenerator.generate(on_progress) -> str`.

- [ ] **Step 1: Write failing tests for multi-round graph research**

Create a streaming fake that returns search, read, and expand tool calls across rounds followed by Markdown. Assert that `ProfileGenerator` executes the calls, carries tool results into later messages, emits monotonic counters, uses `config.PROFILE_MODEL`, enables `config.PROFILE_REASONING`, and returns clamped Markdown.

```python
progress: list[ProfileProgress] = []
card = await ProfileGenerator(store, llm).generate(progress.append)
assert card.startswith("# User profile")
assert [call["name"] for call in llm.tool_calls] == [
    "search_notes", "read_note", "expand_note"
]
assert progress[-1].phase == "synthesizing"
assert progress[-1].tool_calls == 3
assert progress[-1].nodes_read >= 1
```

- [ ] **Step 2: Run the focused test and confirm red state**

Run: `cd core && uv run pytest tests/test_index_profile.py -q`

Expected: import failure for `ProfileGenerator` or missing stream behavior.

- [ ] **Step 3: Add profile-agent configuration**

Add exact constants:

```python
PROFILE_MODEL = "z-ai/glm-5.2"
PROFILE_REASONING = {"enabled": True}
PROFILE_TIMEOUT_S = 180.0
PROFILE_MAX_TOOL_CALLS = 24
PROFILE_MAX_CHARS = 12_000
PROFILE_CONTEXT_MAX_CHARS = 12_000
```

Remove the obsolete full-corpus `PROFILE_NOTE_CHAR_BUDGET` constant.

- [ ] **Step 4: Implement the focused generator**

`ProfileGenerator.generate()` starts with a research brief, streams GLM rounds,
executes only the existing graph tools, appends OpenAI-compatible assistant tool
calls and tool responses, and removes tools after the budget. It reports phase
from the tool query/activity without exposing reasoning. Its final no-tools
instruction is:

```text
Research is complete. Write the detailed Markdown user profile now using only
what you learned. Do not mention tools, searches, note IDs, or missing data.
```

The final result is `"".join(content_parts).strip()` from the terminal round,
clamped to `PROFILE_MAX_CHARS`; empty content raises `LLMError`.

- [ ] **Step 5: Run profile tests and commit**

Run: `cd core && uv run pytest tests/test_index_profile.py tests/test_chat_tools.py -q`

Expected: all pass.

Commit: `feat: add graph-research user profile agent`

---

### Task 2: Profile generation lifecycle and API

**Files:**
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/server.py`
- Modify tests: `core/tests/test_server_ws.py`

**Interfaces:**
- Consumes: `ProfileGenerator.generate(on_progress)` from Task 1.
- Produces: `ProfileGenerationStatus`, runtime `profile_generation`, `POST /profile/generate`, and `{card, generation}` profile/update payloads.

- [ ] **Step 1: Write failing API and update-stream tests**

Cover idle payload, accepted generation, progress update, successful overwrite,
failure preservation, and `409` while a task is active.

```python
assert client.get("/profile").json()["generation"] == {
    "running": False,
    "phase": "idle",
    "activity": "",
    "nodes_read": 0,
    "tool_calls": 0,
    "started_at": None,
    "elapsed_s": 0.0,
    "error": "",
}
assert client.post("/profile/generate").status_code == 202
assert client.post("/profile/generate").status_code == 409
```

- [ ] **Step 2: Run server tests and confirm red state**

Run: `cd core && uv run pytest tests/test_server_ws.py -q`

Expected: profile response mismatch and missing route.

- [ ] **Step 3: Add the Pydantic status contract**

```python
class ProfileGenerationStatus(BaseModel):
    running: bool = False
    phase: Literal[
        "idle", "surveying", "interests", "projects_people",
        "career_studies", "synthesizing", "failed"
    ] = "idle"
    activity: str = ""
    nodes_read: int = 0
    tool_calls: int = 0
    started_at: Optional[float] = None
    elapsed_s: float = 0.0
    error: str = ""
```

- [ ] **Step 4: Implement runtime single-flight orchestration**

Add `make_profile_generator`, `profile_generator`, `profile_generation`, and
`profile_generation_task` to `Runtime`. `POST /profile/generate` snapshots a
running status, creates `_run_profile_generation(runtime)`, and returns 202.
Progress callbacks update the status on the event loop and publish `profile`.
Success calls `index.set_user_card(card)` before publishing idle terminal state.
Failure records `phase="failed"` and a concise error while retaining the card.
Cancel the task in FastAPI lifespan cleanup.

- [ ] **Step 5: Upgrade all profile payloads compatibly**

Use one helper:

```python
def _profile_payload(runtime: Runtime) -> dict[str, Any]:
    return {
        "card": getattr(runtime.index, "user_card", lambda: None)(),
        "generation": _profile_status(runtime),
    }
```

Return it from GET, PUT, POST, and `/updates` profile events.

- [ ] **Step 6: Run tests and commit**

Run: `cd core && uv run pytest tests/test_server_ws.py tests/test_index_profile.py -q`

Expected: all pass.

Commit: `feat: expose profile generation lifecycle`

---

### Task 3: Remove profile generation from re-enrichment

**Files:**
- Modify: `core/taurine_core/indexer/reenrich.py`
- Modify: `core/taurine_core/schemas.py`
- Modify tests: `core/tests/test_index_reenrich.py`
- Modify: `app/src/renderer/src/lib/sidecar.ts`

**Interfaces:**
- Consumes: existing re-enrichment pipeline.
- Produces: re-enrichment phases limited to `idle | notes | finalizing`; no read or write of `user_card`.

- [ ] **Step 1: Replace profile-era re-enrichment tests**

Assert full re-enrichment never imports/calls `build_user_card`, leaves an
existing card unchanged, begins in `notes`, and returns only total/changed/
failures.

- [ ] **Step 2: Run focused test and confirm red state**

Run: `cd core && uv run pytest tests/test_index_reenrich.py -q`

Expected: existing tests observe the `profile` phase.

- [ ] **Step 3: Delete the profile phase**

Remove the import and `profile_failed` branch from `reenrich_all`, remove the
field from `ReenrichReport`, start progress with `phase="notes"`, and update
server defaults and renderer `ReenrichPhase` parsing accordingly.

- [ ] **Step 4: Run tests and commit**

Run: `cd core && uv run pytest tests/test_index_reenrich.py tests/test_server_ws.py -q`

Expected: all pass.

Commit: `refactor: decouple user profile from reenrichment`

---

### Task 4: Inject profile context into chat and live summarization

**Files:**
- Modify: `core/taurine_core/chat/engine.py`
- Modify: `core/taurine_core/live/summarizer.py`
- Modify tests: `core/tests/test_chat_engine.py`
- Modify tests: `core/tests/test_live_summarizer.py`

**Interfaces:**
- Consumes: `Index.user_card()` and `config.PROFILE_CONTEXT_MAX_CHARS`.
- Produces: `_user_profile_block(index) -> str` semantics in both prompt builders.

- [ ] **Step 1: Write failing prompt-injection tests**

Set `store.set_user_card("Neil builds Taurine.")`, run a chat turn and a live
summary pass, then assert the outbound model input contains:

```xml
<user_profile>
Neil builds Taurine.
</user_profile>
```

Also assert the block is absent for an empty card and clamps oversized cards.

- [ ] **Step 2: Run tests and confirm red state**

Run: `cd core && uv run pytest tests/test_chat_engine.py tests/test_live_summarizer.py -q`

Expected: profile text absent from captured prompts.

- [ ] **Step 3: Add bounded runtime profile blocks**

Chat adds the profile block to the system message and explicitly says it is
uncitable background. Live summarization adds it before `<previous_summary>`
and updates `SYSTEM_PROMPT` to prohibit using it as meeting evidence. Both read
the card while building each turn/pass so edits apply immediately.

- [ ] **Step 4: Run tests and commit**

Run: `cd core && uv run pytest tests/test_chat_engine.py tests/test_live_summarizer.py -q`

Expected: all pass.

Commit: `feat: provide user profile context to runtime agents`

---

### Task 5: Typed renderer state and generation actions

**Files:**
- Modify: `app/src/renderer/src/lib/sidecar.ts`
- Modify: `app/src/renderer/src/App.tsx`
- Modify: `app/src/renderer/src/views/DataView.tsx`

**Interfaces:**
- Consumes: `{card, generation}` payload from Task 2.
- Produces: `ProfileState`, `ProfileGenerationStatus`, `generateProfile()`, and `profile` prop passed to `UserSummary`.

- [ ] **Step 1: Add exact TypeScript contracts and parsers**

```ts
export type ProfileGenerationPhase =
  | 'idle' | 'surveying' | 'interests' | 'projects_people'
  | 'career_studies' | 'synthesizing' | 'failed'

export interface ProfileState {
  card: string | null
  generation: ProfileGenerationStatus
}
```

Change the updates parser to validate both fields, change `fetchProfile()` to
return `ProfileState`, make `saveProfile()` return the server payload, and add
`generateProfile()` using `POST /profile/generate`.

- [ ] **Step 2: Thread one profile object through App and DataView**

Replace `profileCard` with `profile: ProfileState | undefined`, set it from
profile update events, and pass it unchanged into `UserSummary`.

- [ ] **Step 3: Run TypeScript checking and commit**

Run: `cd app && npm run typecheck`

Expected: all TypeScript projects pass.

Commit: `refactor: carry typed profile generation state`

---

### Task 6: Research-rail User summary UI

**Files:**
- Replace: `app/src/renderer/src/components/UserSummary.tsx`
- Modify: `app/src/renderer/src/assets/main.css`

**Interfaces:**
- Consumes: `ProfileState | undefined`, `generateProfile()`, and `saveProfile()`.
- Produces: empty, ready, confirming, editing, generating, and failed UI states.

- [ ] **Step 1: Implement explicit component state transitions**

Use local state only for collapse, edit draft, save error, and overwrite
confirmation. Server `profile.generation` is authoritative for running/failed
states. Empty Generate calls immediately; populated Generate opens the inline
confirmation. `Overwrite & generate` starts the request without clearing the
displayed card.

- [ ] **Step 2: Implement the approved research rail**

Map backend phases to five visual steps:

```ts
const steps = [
  ['surveying', 'Survey graph'],
  ['interests', 'Map interests'],
  ['projects_people', 'Projects & people'],
  ['career_studies', 'Career & studies'],
  ['synthesizing', 'Write summary']
]
```

Use the approved compact header, yellow progress indicator, truthful activity,
nodes-read/tool-call metrics, elapsed time, and failed retry treatment.

- [ ] **Step 3: Add native Taurine CSS**

All new classes use existing variables; do not add a separate palette. Include
responsive rail collapse, reduced-motion handling, visible focus styles, and
`aria-live="polite"` for changing progress copy.

- [ ] **Step 4: Run frontend verification and commit**

Run: `cd app && npm run typecheck && npm run lint`

Expected: both pass.

Commit: `feat: revitalize user summary with research rail`

---

### Task 7: Full verification and documentation cleanup

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-09-user-summary-agent-design.md` only if implementation details require a factual correction.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified feature and accurate setup/product documentation.

- [ ] **Step 1: Update README behavior**

Remove language claiming re-enrichment generates the user card. State that the
Data tab can explicitly launch a GLM-5.2 graph-research profile and that the
result supplies runtime user context to chat and live summaries.

- [ ] **Step 2: Run focused backend suites**

Run:

```bash
cd core
uv run pytest \
  tests/test_index_profile.py \
  tests/test_server_ws.py \
  tests/test_index_reenrich.py \
  tests/test_chat_engine.py \
  tests/test_live_summarizer.py -q
```

Expected: all pass.

- [ ] **Step 3: Run full backend and frontend checks**

Run: `cd core && uv run pytest tests -q`

Run: `cd app && npm run typecheck && npm run lint`

Expected: all pass; report any pre-existing failure separately with exact test
or lint output.

- [ ] **Step 4: Inspect the rendered flow**

Run Taurine locally, open the Data tab, and verify empty generation, populated
overwrite confirmation, live research rail updates, completion replacement,
manual edit, and retry after a simulated provider failure. Confirm chat and live
summary captured prompts include the latest card in backend tests.

- [ ] **Step 5: Commit final documentation and verification fixes**

Commit: `docs: document explicit user profile generation`

