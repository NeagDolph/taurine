# Inference Backend and Rerank Candidate Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cap passage-reranker scoring at 24 deterministic candidates and route passage reranking through OpenRouter or unchanged local Qwen inference with calibrated fail-open fallback.

**Architecture:** Candidate assembly keeps its existing live/chat breadth, while `LocalPassageReranker` becomes a fallback-aware, identity-carrying passage-ranking boundary that scores only a deterministic 24-row prefix and preserves the unscored tail in trace order. A focused OpenRouter client implements Cohere's rerank protocol; runtime wiring keeps triple recognition local and selects result-specific Qwen artifact floors or Cohere configuration overrides.

**Tech Stack:** Python 3.12, httpx, Pydantic calibration artifacts, pytest, local Qwen pair scorer.

## Global Constraints

- `INFERENCE_BACKEND` accepts exactly `"openrouter"` and `"local"`; remove every `EMBED_BACKEND` reference.
- `RERANK_MAX_CANDIDATES = 24` applies to live, chat, and tool passage reranking.
- Candidate scoring order is descending finite `recall_score`, then ascending `passage_id`.
- Tail candidates stay in decisions/trace, marked unscored and excluded from accepted evidence.
- OpenRouter passage reranking uses `cohere/rerank-v3.5`; comment that it changes to v4-pro when OpenRouter adds the slug.
- Triple recognition stays on local `Qwen/Qwen3-Reranker-0.6B` in both inference modes.
- Cloud passage rerank failure falls back to local Qwen, then degraded direct-only retrieval.
- Cloud success uses `RERANK_API_FLOOR_OVERRIDES = {"accept": 0.5, "fallback": 0.15}` only; local success uses artifact rerank/evidence floors.
- API calls authenticate with `OPEN_ROUTER_API_KEY`, retry 429/5xx/transport failures three total times with exponential backoff, and expose no provider body or secret in errors.
- HTTP tests use `httpx.MockTransport`; pytest performs no network requests.
- Preserve existing user changes to `evals/graph_retrieval/calibration-v1.json` and unrelated files.
- Run `.venv/bin/python -m pytest` from `/Users/neil/taurine/core` for the full suite.
- Stage only task files; every commit includes `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.

---

### Task 1: Unified inference configuration

**Files:**
- Modify: `taurine_core/config.py`
- Modify: `taurine_core/indexer/embed.py`
- Modify: `tests/test_index_embed.py`
- Modify: `tests/test_index_build.py`
- Modify: `tests/test_retrieval_runtime.py`
- Modify: `tests/test_retrieval_types.py`

**Interfaces:**
- Produces: `config.INFERENCE_BACKEND: str`.
- Preserves: `config.embed_model_id() -> str`, `config.embed_dim() -> int`.
- Produces: `config.rerank_model_id() -> str`.
- Produces: `config.RERANK_MAX_CANDIDATES`, `config.RERANK_API_MODEL`, API retry constants, and `config.RERANK_API_FLOOR_OVERRIDES`.

- [ ] **Step 1: Write failing backend-helper and approved-default tests**

Change embedding tests to monkeypatch `INFERENCE_BACKEND`; assert both embedding helpers and `rerank_model_id()` resolve local/API identities, unknown values raise `ValueError`, the cap is 24, and cloud overrides are exactly `accept=0.5` and `fallback=0.15`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_index_embed.py::test_inference_backend_helpers_resolve_model_dimension_and_reranker tests/test_retrieval_types.py::test_overhaul_configuration_matches_approved_defaults -q`

Expected: FAIL because `INFERENCE_BACKEND`, `rerank_model_id`, and the new rerank constants do not exist.

- [ ] **Step 3: Implement strict helpers and migrate direct consumers**

Replace `EMBED_BACKEND` with `INFERENCE_BACKEND`, add:

```python
RERANK_API_MODEL = "cohere/rerank-v3.5"
RERANK_MAX_CANDIDATES = 24
RERANK_API_FLOOR_OVERRIDES = {"accept": 0.5, "fallback": 0.15}

def rerank_model_id() -> str:
    if INFERENCE_BACKEND == "openrouter":
        return RERANK_API_MODEL
    if INFERENCE_BACKEND == "local":
        return RERANK_MODEL
    raise ValueError(f"unsupported inference backend: {INFERENCE_BACKEND!r}")
```

Update every production and test consumer without retaining a compatibility alias.

- [ ] **Step 4: Verify migration and GREEN**

Run: `rg -n "EMBED_BACKEND" taurine_core tests`

Expected: no matches.

Run: `.venv/bin/python -m pytest tests/test_index_embed.py tests/test_index_build.py tests/test_retrieval_runtime.py tests/test_retrieval_types.py -q`

Expected: all focused configuration/embedding/build/runtime tests pass.

### Task 2: Deterministic 24-row cap and unscored tail

**Files:**
- Modify: `taurine_core/retrieval/rerank.py`
- Modify: `tests/test_retrieval_rerank.py`

**Interfaces:**
- `LocalPassageReranker.rerank(...) -> PassageRerankResult` scores at most 24 documents.
- `PassageRerankResult.decisions` contains scored decisions followed by deterministic unscored tail decisions.
- `PassageRerankResult.accepted` contains only scored decisions clearing the active acceptance floor.

- [ ] **Step 1: Replace old mode-cap test with failing 25-row behavior tests**

Provide 25 candidates including equal and non-finite recall scores. Assert the scorer sees exactly the deterministic first 24, the result contains 25 decisions, the last decision is unaccepted with reason `outside_rerank_candidate_cap`, and no tail recall score is used as its rerank score.

- [ ] **Step 2: Run focused cap test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_retrieval_rerank.py::test_passage_reranker_caps_scoring_at_24_and_preserves_unscored_tail -q`

Expected: FAIL because the current cap is 30/60 and decisions omit truncated rows.

- [ ] **Step 3: Implement the fixed cap and tail decisions**

Sort all candidates once with `(-_safe_recall(row.recall_score), row.passage_id)`, split at `max(0, int(config.RERANK_MAX_CANDIDATES))`, score only the prefix, and append tail decisions with `score=0.0`, `accepted=False`, sequential ranks after the scored block, and `outside_rerank_candidate_cap`.

- [ ] **Step 4: Run rerank tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_retrieval_rerank.py -q`

Expected: all local rerank and local Qwen microbatch tests pass.

### Task 3: OpenRouter Cohere rerank client

**Files:**
- Modify: `taurine_core/retrieval/rerank.py`
- Modify: `tests/test_retrieval_rerank.py`

**Interfaces:**
- Produces: `RerankError(RuntimeError)`.
- Produces: `OpenRouterRerankClient.score_pairs(query, documents, *, instruction, deadline_at=None) -> tuple[float, ...]`.

- [ ] **Step 1: Write failing happy-path HTTP test**

Use `httpx.MockTransport` to return out-of-order results. Assert one POST to `/api/v1/rerank`, bearer auth, payload `{model, query, documents, top_n}`, ignored local instruction framing, and score reconstruction by result index.

- [ ] **Step 2: Run the happy-path test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_retrieval_rerank.py::test_openrouter_rerank_client_posts_one_indexed_request -q`

Expected: FAIL because the client does not exist.

- [ ] **Step 3: Implement request and strict response parsing**

Post with configured timeout, require exactly one unique index in `[0, count)` per document, coerce finite relevance scores, and return them in input order. Missing keys, duplicate/out-of-range indexes, wrong cardinality, invalid JSON, and non-finite values raise sanitized `RerankError`.

- [ ] **Step 4: Write failing retry/error tests**

Cover retryable 429/500 and `httpx.ReadTimeout`, immediate 400, invalid successful payloads, three-attempt exhaustion, missing API key, and expected backoff sleeps `[0.5, 1.0]`.

- [ ] **Step 5: Run retry/error tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_retrieval_rerank.py -q`

Expected: new retry/error cases fail until retry classification is implemented.

- [ ] **Step 6: Implement three-attempt exponential backoff**

Retry only status 429, 5xx, and `httpx.TransportError`; sanitize raised messages to endpoint/status categories without including response bodies or the API key.

- [ ] **Step 7: Run rerank tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_retrieval_rerank.py -q`

Expected: all cap, local, and remote-client tests pass.

### Task 4: Runtime fallback chain, model identity, and floor selection

**Files:**
- Modify: `taurine_core/retrieval/rerank.py`
- Modify: `taurine_core/retrieval/runtime.py`
- Modify: `taurine_core/retrieval/calibration.py`
- Modify: `taurine_core/retrieval/evidence.py`
- Modify: `tests/test_retrieval_rerank.py`
- Modify: `tests/test_retrieval_runtime.py`
- Modify: `tests/test_retrieval_calibration.py`
- Modify: `tests/test_evidence_pack.py`

**Interfaces:**
- `LocalPassageReranker` accepts primary model/floor identity and optional local fallback model/floor identity.
- `PassageRerankResult.model_id` identifies the scorer that actually produced scores.
- `PassageRerankResult.evidence_floor_override` is `0.15` only for Cohere success and `None` for local/Qwen success.
- `AbstentionPolicy.post_rerank(..., evidence_floor_override=None)` uses an explicit override only when supplied.

- [ ] **Step 1: Write failing reranker fallback-chain tests**

Assert cloud success never calls local and reports Cohere/override identity; remote failure followed by local success reports Qwen with artifact acceptance and no evidence override; both failures return degraded status and log warnings. Also freeze a local-only result for a <=24-row fixture and assert its decisions/reasons remain unchanged.

- [ ] **Step 2: Run fallback tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_retrieval_rerank.py -q`

Expected: FAIL because results do not carry model/floor identity and the reranker has no fallback scorer.

- [ ] **Step 3: Implement fallback-aware scoring**

Factor scorer invocation so the primary scorer is attempted first. On a configured cloud primary failure, log once and retry the same bounded documents with local Qwen using the artifact acceptance floor. If both fail, return the existing degraded decision structure and log that no reranker succeeded. Keep local-only exception reasons byte-for-byte unchanged.

- [ ] **Step 4: Write failing runtime and abstention floor-source tests**

In local mode assert the runtime does not create a remote client, both triple and passage reranking use the same local Qwen object, and artifact rerank/evidence floors apply. In OpenRouter mode assert triple recognition still receives local Qwen while passage reranking gets Cohere primary plus local fallback. Assert Cohere success supplies `0.5/0.15`, local fallback supplies artifact rerank/mode evidence floors, and debug trace names the actually-used model and floor source.

- [ ] **Step 5: Run runtime/calibration/evidence tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_retrieval_runtime.py tests/test_retrieval_calibration.py tests/test_evidence_pack.py -q`

Expected: FAIL until runtime wiring and per-result post-rerank floors exist.

- [ ] **Step 6: Wire backend-specific runtime without changing triple recognition**

Always attempt to load local Qwen for `TripleRecognizer`. For local mode, construct the passage reranker exactly as before with artifact floors. For OpenRouter, construct the Cohere client from `OPEN_ROUTER_API_KEY`, configure Qwen as fallback when available, and supply the cloud override floors. Extend post-rerank abstention to accept the result's optional evidence floor override and record actual model/floor source in retrieval debug data.

- [ ] **Step 7: Run all focused retrieval tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_retrieval_rerank.py tests/test_retrieval_runtime.py tests/test_retrieval_calibration.py tests/test_evidence_pack.py tests/test_chat_retrieval.py -q`

Expected: all focused tests pass with no HTTP network access.

### Task 5: Audit, full verification, and commits

**Files:**
- Modify: all task files above.
- Add: `docs/superpowers/specs/2026-07-14-inference-backend-rerank-cap-design.md`.
- Add: `docs/superpowers/plans/2026-07-14-inference-backend-rerank-cap.md`.

**Interfaces:**
- Produces a clean task-only commit history with the required trailer.

- [ ] **Step 1: Run static migration and diff audits**

Run: `rg -n "EMBED_BACKEND" taurine_core tests`

Expected: no matches.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; user-owned calibration, `.claude`, database, and unrelated spec changes remain unstaged.

- [ ] **Step 2: Run the full core suite**

Run: `.venv/bin/python -m pytest`

Expected: at least the 825-test baseline plus new tests passes, with the existing two intentional skips.

- [ ] **Step 3: Commit task implementation files only**

Commit subject: `feat(core): add cloud-aware capped reranking`

Commit trailer: `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.

- [ ] **Step 4: Report evidence**

Report commit hashes, exact passed/skipped counts, and floor sources: artifact mode floors for local and cloud-to-local fallback; `RERANK_API_FLOOR_OVERRIDES` for Cohere success.
