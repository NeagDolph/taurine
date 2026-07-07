"""All tunables. In-code constants by convention — env vars are secrets only."""

# ── sidecar ────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8930

# ── ASR ────────────────────────────────────────────────────────────────
# "deepgram" (general speech, prod-proven) | "taurine" (customer-service-tuned)
ASR_PROVIDER = "deepgram"
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
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
ENRICH_MODEL = "z-ai/glm-5.2"            # enrichment is offline: quality over latency
ENRICH_TIMEOUT_S = 60.0
PRESENTER_MODEL = "google/gemini-3.5-flash"
PRESENTER_FALLBACK_MODEL = "deepseek/deepseek-chat"
LLM_TIMEOUT_S = 15.0

# ── user information card (full-corpus profile fed to every enrichment) ─
PROFILE_MODEL = "z-ai/glm-5.2"           # 1M context: the whole corpus fits
PROFILE_TIMEOUT_S = 300.0
PROFILE_NOTE_CHAR_BUDGET = 2400          # per-note truncation when building card input
PROFILE_MAX_CHARS = 6000                 # cap on the generated card itself
REENRICH_CONCURRENCY = 4

# ── notes sync / connectors ────────────────────────────────────────────
SYNC_INTERVAL_S = 300.0
CONNECTOR_DOC_MAX_CHARS = 12000          # per-document body cap at extraction
IMESSAGE_WINDOW_DAYS = 14
IMESSAGE_MAX_CHARS = 5000                # per-conversation transcript cap
CLAUDE_MEMORY_GLOB = "~/.claude/projects/*/memory/*.md"
CODEX_MEMORY_GLOBS = ("~/.codex/AGENTS.md", "~/.codex/memory/**/*.md", "~/.codex/memories/**/*.md")
OBSIDIAN_APP_JSON = "~/Library/Application Support/obsidian/obsidian.json"
IMESSAGE_DB = "~/Library/Messages/chat.db"
CONTACTS_DB_GLOB = "~/Library/Application Support/AddressBook/**/AddressBook-v22.abcddb"
GIST_MAX_CHARS = 220                     # was 120: mid-word truncation degraded rerank docs
                                         # and card copy (audit finding #5)

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
C2_TOP_K = 8                             # match the search-lane quota (was 5: entity notes
                                         # repeatedly died at ordered[6] — see 2026-07-06 triage)
FUSED_PROTECT = 3                        # top-N fused candidates enter the slate even if the
                                         # reranker demotes them (rerank can't see entity links
                                         # through ASR spelling drift)
DENSE_PROTECT = 2                        # top-N dense-window hits likewise protected (RRF
                                         # buries single-list hits — audit finding #1)
SALIENT_MIN_DF = 1                       # drop df=0 window terms: they are ASR misspellings,
                                         # not rare gold (audit finding #2)
GRAPH_VOTE_TOP = 10                      # fused candidates whose neighbors get vote credit
GRAPH_VOTE_WEIGHT = 0.5                  # vote strength multiplier (5th RRF list)
MIN_COSINE = 0.35                        # floor only when no BM25 hit AND no graph path

# ── fuzzy entity snapping (ASR spelling drift: "Armand" → "Armaan") ────
FUZZY_SNAP_MIN_LEN = 4                   # only snap spans at least this long
FUZZY_SNAP_MAX_DIST = 2                  # max Levenshtein distance to an alias/title term
FUZZY_SNAP_LABELS = ("person", "organization", "project")  # spans eligible for snapping

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

# ── C2 GLiNER2 span extraction (schema-based, label descriptions) ─────
GLINER2_MODEL = "fastino/gliner2-large-v1"
# label → description; descriptions are per-label prompt engineering and the
# main precision lever — edit freely. Schema order = span priority at the cap.
GLINER2_SCHEMA = {
    "person": "Names of people mentioned or addressed in the conversation",
    "organization": "Companies, teams, clients, or vendors discussed",
    "project": "Internal work initiatives, codenames, or features under discussion",
    "topic": "The concrete subject matter being discussed, e.g. 'checkout flow', 'Q3 roadmap'",
    "commitment": "Tasks, action items, or promises someone agrees to do",
    "event": "Meetings, deadlines, launches, or scheduled occurrences",
}
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
PRESENTER_MAX_TOOL_CALLS = 3             # read_note tool-loop budget per pass
PRESENTER_NEIGHBORS_SHOWN = 3            # neighbor ids listed per slate line
WHY_MAX_CHARS = 140
MAX_PINNED = 3
