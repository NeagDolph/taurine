# OpenRouter Embedding Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production OpenRouter embedding backend with backend-aware dimensions, atomic build failure, query fail-open, snapshot-space verification, and two separately committed pilot default flips.

**Architecture:** Configuration helpers select the active model identity and dimension. The existing embedding module dispatches to either the unchanged local Qwen path or a focused synchronous OpenRouter client. Builds keep the current atomic exception boundary, while retrieval catches a failed query embedding once and marks dense/triple unavailable before running the independent heads.

**Tech Stack:** Python 3.12, NumPy, httpx, pytest, SQLite build snapshots.

## Global Constraints

- `EMBED_BACKEND` accepts exactly `"openrouter"` and `"local"`; pilot default is `"openrouter"`.
- OpenRouter uses `qwen/qwen3-embedding-8b`, dimension 4096, bearer auth from `OPEN_ROUTER_API_KEY`.
- Local uses the unchanged `Qwen/Qwen3-Embedding-0.6B` SentenceTransformer path, dimension 1024.
- Documents embed bare; queries receive `EMBED_QUERY_PROMPT` exactly once.
- Each API request has at most 64 texts and 100,000 characters.
- Retry 429, 5xx, timeouts, and transport failures for three total attempts with exponential backoff.
- Build embedding failures abort without installing; query failures log and disable dense/triple for that retrieval only.
- All HTTP tests use `httpx.MockTransport`; pytest performs no network access.
- Run `/Users/neil/taurine/core/.venv/bin/python -m pytest` from `/Users/neil/taurine/core` before every commit.
- Stage task files only; never stage `.claude/` or `*.db*`.
- Every commit includes `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.

---

### Task 1: Backend-aware configuration and OpenRouter client

**Files:**
- Modify: `core/taurine_core/config.py`
- Modify: `core/taurine_core/indexer/embed.py`
- Modify: `core/tests/test_index_embed.py`
- Modify: `core/tests/conftest.py`
- Modify: `core/tests/test_stress_threading.py`

**Interfaces:**
- Produces: `config.embed_model_id() -> str`, `config.embed_dim() -> int`.
- Produces: `OpenRouterEmbeddingClient.embed(texts: list[str]) -> np.ndarray`.
- Preserves: `embed_texts(list[str])` and `embed_queries(list[str], prompt=None)`.

- [ ] **Step 1: Write failing config and HTTP behavior tests**

Add tests that monkeypatch both backends and assert model IDs/dimensions; use
`httpx.MockTransport` to assert bearer auth, endpoint, model, bare document
inputs, query prefixes, ordered indexed response parsing, float32 output, and
unit norms.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_index_embed.py -q`

Expected: failures because the backend helpers/client do not exist and the
current constant dimension is 1024.

- [ ] **Step 3: Implement configuration helpers and dispatch**

Add local/API model and dimension constants plus strict helper functions. Keep
the local `_embed_local` implementation behavior intact. Add a synchronous
OpenRouter client that validates the complete response and normalizes rows;
dispatch from `_embed` based on `EMBED_BACKEND`.

- [ ] **Step 4: Add failing batching/retry/error tests**

Cover the 64-text boundary, the cumulative 100,000-character boundary, one
oversize text, retryable 429/500/timeout/transport errors, non-retryable 400,
exponential sleep values, exhausted attempts, missing key, malformed indexes,
wrong vector width, non-finite rows, and zero norms.

- [ ] **Step 5: Run focused tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_index_embed.py -q`

Expected: new edge-case tests fail until batching and retry classification are
implemented.

- [ ] **Step 6: Implement batching and retry behavior**

Partition without splitting semantic inputs, perform three attempts per
request, and raise a sanitized embedding error without logging or exposing the
API key.

- [ ] **Step 7: Replace all direct `EMBED_DIM` consumers**

Use `config.embed_dim()` in empty matrices, shape validation, shared test vector
factories, and stress-test allocation. Confirm with:

Run: `rg -n "EMBED_DIM" taurine_core tests`

Expected: no matches.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_index_embed.py -q`

Expected: all focused embedding tests pass with no network.

### Task 2: Build wiring, cache identity, and atomic failure

**Files:**
- Modify: `core/taurine_core/indexer/build.py`
- Modify: `core/taurine_core/server.py`
- Modify: `core/tests/test_index_build.py`
- Modify: `core/tests/test_build_cache.py`

**Interfaces:**
- `BuildDependencies` resolves active dimension/model at construction time.
- Production cache wrapper uses `config.embed_model_id()`.
- Build manifest `embedding` equals the selected backend model ID.

- [ ] **Step 1: Write failing dependency, manifest, fingerprint, cache, and atomic failure tests**

Assert backend switching changes new `BuildDependencies` defaults, a report and
stored snapshot record the active model, the API fingerprint includes the
remote identity, model-scoped cache rows do not reuse the local vector, and an
exhausted API-style exception leaves the prior target bytes untouched with no
temporary install.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_index_build.py tests/test_build_cache.py -q`

Expected: failures from import-time defaults, calibration equality rejection,
local-only fingerprints, and production cache wiring.

- [ ] **Step 3: Implement dynamic build identity and server wiring**

Use dataclass default factories, write the active embedding ID over the
calibration manifest's embedding field, retain other identity fields, use an
API identity fingerprint for the remote model, and pass the active model ID to
`BuildCache.caching_embedder`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_index_build.py tests/test_build_cache.py -q`

Expected: all focused build/cache tests pass.

### Task 3: Snapshot-space checks and query fail-open

**Files:**
- Modify: `core/taurine_core/retrieval/runtime.py`
- Modify: `core/taurine_core/retrieval/evidence.py`
- Modify: `core/tests/test_retrieval_runtime.py`
- Modify: `core/tests/test_evidence_pack.py`

**Interfaces:**
- `RetrievalRuntime._validate_snapshot` requires active embedding identity and
  non-empty matrix width while retaining calibration/non-embedding checks.
- `SharedRetrievalEngine.retrieve` records dense/triple unavailability when
  `query_embedder` raises and continues independent heads.

- [ ] **Step 1: Write failing incompatible-snapshot tests**

Build a deterministic 16-dimensional fixture with an explicit fixture model
identity, then assert runtime refuses a snapshot whose embedding model differs
from the monkeypatched active model or whose matrix width differs from the
monkeypatched active dimension.

- [ ] **Step 2: Run focused runtime tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_retrieval_runtime.py -q`

Expected: current runtime compares only the snapshot to the static calibration
manifest and does not validate active dimension.

- [ ] **Step 3: Implement active-space snapshot validation**

Compare embedding identity to `config.embed_model_id()`, compare passage/fact
matrix widths when rows exist to `config.embed_dim()`, and compare all remaining
model identities to the calibration artifact.

- [ ] **Step 4: Write failing query-embedding failure test**

Use a real `SharedRetrievalEngine` fixture with a query embedder that raises.
Assert retrieval does not raise, BM25/exact/entity results remain available,
dense and triple appear in `debug.unavailable_heads`, no triple recognizer call
occurs, and a warning is logged without secret-bearing exception text.

- [ ] **Step 5: Run the focused retrieval test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_evidence_pack.py -q`

Expected: current retrieval raises before head isolation.

- [ ] **Step 6: Implement one-shot query failure isolation**

Catch and sanitize the query embedding failure, log a warning, return an
unavailable dense result, skip triple recognition with a degraded result, then
run BM25/exact/entity and pass both unavailable heads to abstention.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_retrieval_runtime.py tests/test_evidence_pack.py -q`

Expected: all focused runtime/retrieval tests pass.

### Task 4: Verify and commit the embedding backend

**Files:**
- Modify: all Task 1-3 files
- Add: `docs/superpowers/specs/2026-07-14-openrouter-embedding-backend-design.md`
- Add: `docs/superpowers/plans/2026-07-14-openrouter-embedding-backend.md`

- [ ] **Step 1: Run direct-consumer and diff audits**

Run: `rg -n "EMBED_DIM" taurine_core tests`

Expected: no direct constant consumers.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; unrelated `.claude/`, database, and prior spec
remain unstaged.

- [ ] **Step 2: Run the full core suite**

Run: `.venv/bin/python -m pytest`

Expected: all tests pass, with only intentional slow-test skips.

- [ ] **Step 3: Stage only embedding task files and commit**

Commit subject: `feat(core): add OpenRouter embedding backend`

Commit trailer: `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`

### Task 5: Pilot extractor default

**Files:**
- Modify: `core/taurine_core/config.py`
- Modify: the focused extractor/default test that asserts the pilot default.

- [ ] **Step 1: Change the focused expectation to `"union"` and verify RED**

Run the focused extractor/config test and confirm it fails against `"gliner2"`.

- [ ] **Step 2: Set `EXTRACTOR_BACKEND = "union"`**

Update the adjacent rationale to the 2026-07-14 benchmark: 20.6% versus 12.7%
silver recall with equal judge-verified novel precision.

- [ ] **Step 3: Run focused test, full suite, stage only the two files, and commit**

Commit subject: `feat(core): default to union extraction`

Commit trailer: `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`

### Task 6: Pilot contextual-enrichment default

**Files:**
- Modify: `core/taurine_core/config.py`
- Modify: `core/tests/test_context_enrich.py`

- [ ] **Step 1: Change the default expectation to `"hybrid"` and verify RED**

Run: `.venv/bin/python -m pytest tests/test_context_enrich.py -q`

Expected: the default assertion fails against `"llm"`.

- [ ] **Step 2: Set `CONTEXT_ENRICH_MODE = "hybrid"`**

Retain existing hybrid behavior: deterministic templates always, LLM top-up
only for notes meeting `CONTEXT_LLM_MIN_CHARS` or
`CONTEXT_LLM_MIN_PASSAGES`.

- [ ] **Step 3: Run focused test, full suite, stage only the two files, and commit**

Commit subject: `feat(core): default to hybrid context enrichment`

Commit trailer: `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`

### Task 7: Final verification and handoff

**Files:**
- Verify only.

- [ ] **Step 1: Run final full suite and inspect commits**

Run: `.venv/bin/python -m pytest`

Run: `git log -3 --format=fuller --stat`

Run: `git status --short`

Expected: full suite green; exactly the three requested commits at HEAD with
trailers; unrelated workspace files still present and unstaged.

- [ ] **Step 2: Report operational handoff**

List commit SHAs/subjects, suite pass/skip counts, every former `EMBED_DIM`
consumer, the exact rebuild CLI, and the query-time embedding call sites in the
shared retrieval engine and live topic-shift adapter.
