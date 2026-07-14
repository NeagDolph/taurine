# OpenRouter Embedding Backend Design

## Scope

Add a configurable embedding backend to the Python sidecar. `EMBED_BACKEND`
defaults to `"openrouter"` for the pilot and accepts `"local"` to retain the
existing Qwen3-Embedding-0.6B MPS path unchanged. The same work changes the
extractor default from `"gliner2"` to `"union"` and contextual enrichment from
`"llm"` to `"hybrid"`, each in its own commit.

## Configuration and identity

The local model remains `Qwen/Qwen3-Embedding-0.6B` at 1024 dimensions. The
OpenRouter model is `qwen/qwen3-embedding-8b` at 4096 dimensions. The config
module exposes `embed_model_id()` and `embed_dim()` as the only production
sources of the active embedding identity and width. Backend-sensitive default
values use factories or call-time resolution so tests and operators can switch
the backend without stale import-time constants.

The active embedding model ID is written into the build model manifest. The
embedding fingerprint records the remote API model identity for OpenRouter and
continues to hash the installed local artifact for the local backend. Snapshot
activation checks the stored embedding model ID and every non-empty embedding
matrix width against the active backend. A snapshot built in another embedding
space is refused cleanly; the existing server incompatibility path preserves it
and starts an empty compatible generation pending rebuild.

The calibration artifact remains unchanged. Its dense/triple curves are the
pilot's current calibration, while snapshot identity is checked independently
against the active backend. Non-embedding model identities, calibration hash,
and config hash retain their existing strict checks.

## OpenRouter client

A focused synchronous OpenRouter embedding client posts to
`https://openrouter.ai/api/v1/embeddings` with bearer authentication. Documents
are sent bare. Queries are prefixed with `EMBED_QUERY_PROMPT` before batching,
matching the existing asymmetric Qwen3 contract.

Requests contain no more than 64 texts and no more than 100,000 Python
characters. A single text over the character ceiling is rejected rather than
silently split into a different semantic input. HTTP 429, HTTP 5xx, timeouts,
and transport failures receive three total attempts with exponential backoff.
Other HTTP errors and invalid successful responses fail immediately. Responses
must contain one indexed 4096-float vector per input, with finite values and no
zero-norm rows. Returned rows are L2-normalized to match the local path.

Tests use `httpx.MockTransport` and injected sleep behavior. Pytest never makes
a network request.

## Build and cache behavior

The production builder wraps the selected embedder with
`BuildCache.caching_embedder(embed_model_id(), ...)`. Cache keys already include
the model ID, so changing to `qwen/qwen3-embedding-8b` produces fresh cache rows
without a schema change. Tests retain and extend the existing model-scoping
proof.

Any exhausted OpenRouter failure propagates into the existing `BuildError`
boundary. The coordinator closes and removes its temporary database and never
replaces the active index, preserving strict non-fail-open build semantics.

## Query failure behavior

The shared retrieval engine catches query-embedding failure before head
execution, logs one warning, and records both `dense` and `triple` as
unavailable. Dense and embedding-dependent triple recall produce no scores.
BM25, exact alias, and entity heads still execute, and abstention receives the
reduced-head set so it can evaluate the surviving evidence honestly. Chat,
tool, and live retrieval therefore never crash solely because the embedding
API is temporarily unavailable.

Live topic-shift embedding keeps its existing best-effort behavior and uses the
same backend-dispatched query embedder.

## Verification and commits

Implementation follows red-green-refactor with mocked HTTP tests covering
dispatch, asymmetric inputs, batching limits, normalization, response
validation, retry classification, atomic build failure, query fail-open, cache
identity, manifests, and incompatible snapshot refusal. The full core suite is
run before each of three commits:

1. OpenRouter embedding backend and its design/plan documentation.
2. Default `EXTRACTOR_BACKEND = "union"`.
3. Default `CONTEXT_ENRICH_MODE = "hybrid"`.

Only task files are staged. `.claude/` and database artifacts are never staged.
Each commit includes `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.
