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
# demo-first: gemini-3.5-flash is pricier but much faster than deepseek
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENRICH_MODEL = "google/gemini-3.5-flash"
PRESENTER_MODEL = "google/gemini-3.5-flash"
PRESENTER_FALLBACK_MODEL = "deepseek/deepseek-chat"
LLM_TIMEOUT_S = 15.0

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
ALIAS_MIN_COMPACT_LEN = 3                # aliases shorter than this (spaces stripped) are dropped
# aliases that are generic english/note words trigger constantly in speech —
# dropped at alias-table build regardless of source
ALIAS_STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have he her his i if in is it its me my
    no not of on or our she so that the their them they this to was we what when where which
    who will with you your yes yeah okay ok right well just like really actually
    note notes new list lists todo todos task tasks work stuff things idea ideas item items
    untitled misc random daily journal draft drafts meeting call chat talk plan plans
    """.split()
)

# ── C2 salient-query formulation ───────────────────────────────────────
SALIENT_TERMS = 6                        # top rare window terms forming the salient query
SALIENT_MIN_TERM_LEN = 3

# ── C2 GLiNER span extraction ──────────────────────────────────────────
GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLINER_LABELS = [
    "person",
    "company",
    "project",
    "product",
    "document topic",
    "task",
    "event",
    "place",
]
GLINER_THRESHOLD = 0.38
GLINER_MAX_SPANS = 4

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
