# Taurine

Live meeting copilot: listens to the mic, transcribes through the Taurine ASR
server, maintains a grounded rolling summary and actionables, answers questions
over the user's connected knowledge, and maps how the current conversation
relates to the knowledge graph.

Architecture: Electron app (`app/`) ↔ localhost ws ↔ Python sidecar (`core/`,
FastAPI). Product state remains in `taurine.db`; the disposable derived
knowledge index is `taurine-knowledge.db`. A clean build creates token-aware
source passages, batched GLiNER2 entities and directed temporal facts, Qwen
embeddings, a passage/entity/fact base graph, and a separate note projection.
Live speech, chat, and profile research all use the same calibrated
HippoRAG-style retrieval engine and stable `T/E/F/P/N` evidence registry.
Design docs: `docs/superpowers/specs/`, plan: `docs/superpowers/plans/`.

## Setup (once)

```bash
# secrets — core/.env with two keys:
#   TAURINE_ASR_TOKEN      (transcription-service Caddyfile token)
#   OPEN_ROUTER_API_KEY    (all LLM calls route through OpenRouter)

cd core && uv sync          # python sidecar deps
uv run python -m taurine_core.download_models  # offline embedding/rerank/NLI/GLiNER assets
cd ../app && npm install    # electron app deps
```

## Index your notes

```bash
cd core
uv run python -m taurine_core.indexer.build \
  --db taurine-knowledge.db \
  --from-enabled-connectors \
  --replace-derived \
  --yes
```

The build reads enabled connectors, writes a temporary generation, validates
referential/embedding/graph invariants, and atomically replaces the active
knowledge database only after success. Regeneration is intentionally clean and
complete—there is no migration or translation of old derived graph rows. The
server uses the same path after connector synchronization and publishes the new
immutable snapshot only after validation.

## Run

```bash
# terminal 1 — sidecar (from core/, so it finds taurine.db and .env)
cd core && uv run uvicorn taurine_core.server:app --host 127.0.0.1 --port 8930

# terminal 2 — app
cd app && npm run dev
```

Click **Start listening**, talk. Live view shows source passages and calibrated
grounding, a citation-validated rolling summary, and transcript-safe
actionables. The inspector exposes accepted/rejected heads, triple decisions,
PPR/path/rerank details, abstention, hashes, and latency without duplicating
private passage text in logs. Chat receives answer-bearing passages in Stage A
and can call `search_evidence`, query-aware paginated `read_note`, and directed
`expand_evidence`. Cited sources and merely consulted notes are labeled
separately. The editable User summary remains uncitable background context.

Headless smoke (no audio, no app):

```bash
cd core && echo "we should ask the shopify consultant about checkout" \
  | uv run python -m taurine_core.devfeed
```

## Development

```bash
cd core && uv run pytest tests -q     # offline (RUN_SLOW=1 adds real-model tests)
cd app && npm run typecheck
```

Model/config identities and confidence floors are pinned by
`core/evals/graph_retrieval/calibration-v1.json`; runtime identity mismatches
fail before ranking. Cadence and bounded resource limits remain in
`core/taurine_core/config.py`. Missing reranking uses conservative direct
evidence fallback; missing NLI fails live factual updates closed and makes chat
return source excerpts instead of unvalidated synthesis.

The checked-in calibration curves are intentionally marked with zero samples
and no held-out metrics. They are conservative bootstrap values, not a claimed
production calibration. Clean builds therefore report `health=degraded`,
`release_ready=false`, and one `uncalibrated_retrieval_head` blocker per
mode/head until the labeled §18 evaluation datasets are populated and the
artifact is fit and evaluated. The build never fabricates release metrics.
