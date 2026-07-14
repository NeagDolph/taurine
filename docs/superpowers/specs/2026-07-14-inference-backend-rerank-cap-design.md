# Inference Backend and Rerank Candidate Cap Design

## Scope

Taurine gets one inference-mode switch for embeddings and passage reranking:
`INFERENCE_BACKEND = "openrouter" | "local"`. The existing embedding helpers
remain the source of the active embedding identity and dimension. Passage
reranking gains a model-independent hard cap of 24 scored candidates in live,
chat, and tool retrieval. Existing larger per-mode candidate-pool caps continue
to bound head recall and expansion; they no longer determine reranker work.

Triple recognition remains on the bundled local Qwen scorer in both inference
modes. This preserves its Qwen-fitted recognition and fallback floors and
ensures OpenRouter receives at most one logical passage-rerank request for a
retrieval query.

## Candidate ordering and cap

`LocalPassageReranker` first orders the complete candidate sequence by
descending finite pre-rerank `recall_score`, then ascending `passage_id`. Only
the first `RERANK_MAX_CANDIDATES = 24` documents are sent to a scorer. Their
decisions are ordered by descending reranker score, then `passage_id`.

Candidates after position 24 remain in the result decisions and retrieval
trace below the scored block, retaining their deterministic pre-rerank order.
They are marked unscored and are not included in accepted evidence. Their
recall scores are never compared with either Qwen or Cohere floors.

## Backend configuration and identity

`EMBED_BACKEND` is replaced everywhere by `INFERENCE_BACKEND`; supported values
remain exactly `"openrouter"` and `"local"`. `embed_model_id()` and
`embed_dim()` keep their existing interfaces and resolve through the new
constant. `rerank_model_id()` exposes the configured passage-reranker identity:

- local: `Qwen/Qwen3-Reranker-0.6B`;
- OpenRouter: `cohere/rerank-v3.5`.

The Cohere constant carries an adjacent comment to replace it with
`cohere/rerank-v4-pro` when OpenRouter adds that slug. The configured and
actually-used reranker model identities travel through the runtime reranker,
rerank result, and retrieval debug trace. Snapshot/build manifest compatibility
continues to describe persisted build inputs; the query-time API scorer does
not invalidate an otherwise compatible knowledge snapshot.

## OpenRouter rerank client

A focused synchronous client posts once to
`https://openrouter.ai/api/v1/rerank` with bearer auth from
`OPEN_ROUTER_API_KEY` and JSON fields `model`, `query`, `documents`, and
`top_n`. The caller has already capped documents at 24. Successful responses
must contain exactly one finite `relevance_score` for every input index; rows
may arrive in ranked order and are restored to input order before the passage
reranker applies its deterministic ranking.

HTTP 429, HTTP 5xx, timeouts, and transport errors receive three total attempts
with exponential backoff matching the embedding client. Other HTTP failures and
invalid success payloads fail immediately with sanitized errors. Unit tests use
`httpx.MockTransport`; no test makes a network call.

## Failure chain and calibration

In local mode, passage reranking uses only Qwen and retains today's scorer,
error, decision, and degraded-direct-fallback behavior apart from the approved
24-candidate cap and trace identity fields.

In OpenRouter mode, passage reranking first calls Cohere. A failed cloud call
logs a warning and tries the already loaded local Qwen scorer. If Qwen is
unavailable or fails, the rerank result is degraded and the shared evidence
engine uses its existing calibrated-direct-only fallback, with a warning that
no reranker succeeded.

The checked-in artifact has one unkeyed rerank calibration belonging to Qwen,
so it must not calibrate Cohere scores. Until `self_calibrate` refits keyed
cloud values after merge, OpenRouter success reads
`RERANK_API_FLOOR_OVERRIDES = {"accept": 0.5, "fallback": 0.15}`. `accept`
filters passage rerank decisions; `fallback` replaces the artifact mode's
post-rerank evidence floor for that cloud result only. A successful local
fallback within OpenRouter mode switches back to the artifact's mode-specific
Qwen rerank acceptance floor and evidence floor. Local inference mode always
uses those artifact floors.

## Testing and commits

Tests cover strict backend helpers and migration away from `EMBED_BACKEND`, the
24-of-25 deterministic cap and unscored tail, mocked OpenRouter request/response
behavior and retries, cloud-to-local-to-degraded fallback, local-mode behavior,
active reranker identity, and result-specific floor selection. The full suite
runs from `core/` with `.venv/bin/python -m pytest`.

Only files changed for this task are staged. Existing modifications to
`core/evals/graph_retrieval/calibration-v1.json`, `.claude/launch.json`, database
artifacts, and unrelated specs remain untouched and unstaged. Every task commit
includes `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.
