# Taurine

Live meeting copilot: listens to the mic, transcribes through the Taurine ASR
server, and surfaces relevant Apple Notes in real time — an ambient relevance
list, LLM-gated pinned cards with a "why this matters now" line, and a live
note-graph map showing how the conversation connects to your notes.

Architecture: Electron app (`app/`) ↔ localhost ws ↔ Python sidecar (`core/`,
FastAPI) which owns the ASR connection, the notes index (sqlite + FTS5 +
local embeddings + note graph), and the live loop (C1 exact / C2 hybrid
retrieval + graph expansion + Cohere rerank / slate / presenter / cadence).
Design docs: `docs/superpowers/specs/`, plan: `docs/superpowers/plans/`.

## Setup (once)

```bash
# secrets — core/.env with three keys:
#   TAURINE_ASR_TOKEN      (transcription-service Caddyfile token)
#   OPEN_ROUTER_API_KEY    (all LLM calls route through OpenRouter)
#   COHERE_API_KEY         (rerank v3.5)

cd core && uv sync          # python sidecar deps
cd ../app && npm install    # electron app deps
```

## Index your notes

```bash
cd core
uv run python -m taurine_core.indexer.sync --db taurine.db
```

First run extracts every note in the default Apple Notes account (macOS will
ask for automation permission), enriches each via OpenRouter (gist, synthetic
spoken utterances, aliases, topics/entities), embeds locally
(bge-small-en-v1.5), and builds the note graph. Re-runs are incremental by
content hash; the running server also re-syncs every 5 minutes.

## Run

```bash
# terminal 1 — sidecar (from core/, so it finds taurine.db and .env)
cd core && uv run uvicorn taurine_core.server:app --host 127.0.0.1 --port 8930

# terminal 2 — app
cd app && npm run dev
```

Click **Start listening**, talk. Feed view shows the live transcript, ambient
sidebar ranks notes for the current moment, pinned cards appear when the
presenter promotes something (dismiss with ×, or open directly in Notes).
Map view shows the note graph with live rank-colored edges from the "now"
node on every retrieval pass and hop-labeled edges for graph-expanded
candidates.

Headless smoke (no audio, no app):

```bash
cd core && echo "we should ask the shopify consultant about checkout" \
  | uv run python -m taurine_core.devfeed
```

## Development

```bash
cd core && uv run pytest tests -q     # 64 tests, offline (RUN_SLOW=1 adds real-model tests)
cd app && npm run typecheck
```

Tunables are in-code constants in `core/taurine_core/config.py` (cadence,
quotas, retrieval K's, graph thresholds, model ids). Env vars are secrets
only. `core/repair_enrich.py` re-enriches notes that previously fell back to
raw-text indexing.
