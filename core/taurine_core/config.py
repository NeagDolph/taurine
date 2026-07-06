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
