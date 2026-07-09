# Taurine

Live meeting copilot: listens to the mic, transcribes through the Taurine ASR
server, maintains a grounded rolling summary and actionables, answers questions
over the user's connected knowledge, and maps how the current conversation
relates to the knowledge graph.

Architecture: Electron app (`app/`) ↔ localhost ws ↔ Python sidecar (`core/`,
FastAPI) which owns the ASR connection, the notes index (sqlite + FTS5 +
local embeddings + heterogeneous graph), and the live loop (C1 exact / C2
hybrid retrieval + personalized graph traversal + grounded summarization).
Design docs: `docs/superpowers/specs/`, plan: `docs/superpowers/plans/`.

## Setup (once)

```bash
# secrets — core/.env with two keys:
#   TAURINE_ASR_TOKEN      (transcription-service Caddyfile token)
#   OPEN_ROUTER_API_KEY    (all LLM calls route through OpenRouter)

cd core && uv sync          # python sidecar deps
cd ../app && npm install    # electron app deps
```

## Index your notes

```bash
cd core
uv run python -m taurine_core.indexer.sync --db taurine.db
```

First run extracts every note in the default Apple Notes account (macOS will
ask for automation permission), extracts typed entities and relations locally,
embeds content with Qwen3-Embedding-0.6B, and builds the knowledge graph.
Limited LLM calls remain for high-value spoken aliases and uncertain graph-edge
adjudication. Re-runs are incremental by content hash; the running server also
re-syncs every 5 minutes.

## Run

```bash
# terminal 1 — sidecar (from core/, so it finds taurine.db and .env)
cd core && uv run uvicorn taurine_core.server:app --host 127.0.0.1 --port 8930

# terminal 2 — app
cd app && npm run dev
```

Click **Start listening**, talk. Live view shows the transcript, grounded
rolling summary, actionables, and ambient sources. Chat uses GLM-5.2 with graph
search/read/traversal tools. Data shows the knowledge map and the editable User
summary card. **Generate** on that card explicitly launches a GLM-5.2 research
agent to traverse the current graph; it never runs automatically during sync or
re-enrichment. The saved summary becomes background user context for subsequent
chat turns and live-summary passes.

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

Tunables are in-code constants in `core/taurine_core/config.py` (cadence,
quotas, retrieval K's, graph thresholds, model ids). Env vars are secrets
only. `core/repair_enrich.py` re-enriches notes that previously fell back to
raw-text indexing.
