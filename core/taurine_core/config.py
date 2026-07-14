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
# Qwen3-Embedding-0.6B (2025): 1024-dim, instruction-aware, runs on MPS.
# Documents embed bare; queries get the instruction prefix (asymmetric
# retrieval). Replaced bge-small-en-v1.5 (384-dim) — see 2026-07-07 upgrade.
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 64                     # per-forward-pass cap; bounds MPS peak
EMBED_QUERY_PROMPT = (
    "Instruct: Given a snippet of spoken conversation, retrieve personal "
    "notes relevant to what is being discussed\nQuery: "
)
CHUNK_MAX_TOKENS = 300                   # sentence-boundary chunking

# ── LLM (ALL calls via OpenRouter) ─────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENRICH_TIMEOUT_S = 60.0
LLM_TIMEOUT_S = 15.0
# chain-of-thought off everywhere: enrichment is bulk offline work (GLM-5.2
# reasons by default and multiplies latency; it can also return
# reasoning-only responses with empty content). Values are per-model because
# providers disagree: deepseek-chat rejects the parameter entirely (absent
# from the map → omitted from the request).
LLM_REASONING_BY_MODEL = {
    "z-ai/glm-5.2": {"enabled": False},
    "deepseek/deepseek-v4-pro": {"enabled": False},
    "deepseek/deepseek-v4-flash": {"enabled": False},
    # Grounded edge adjudication is a bounded verify/label task, not open
    # reasoning: gpt-oss defaults to medium effort (thousands of think-tokens
    # per call → ~100s latency and runaway malformed JSON). Low effort is the
    # right tier and cuts per-call cost several-fold.
    "openai/gpt-oss-120b": {"effort": "low"},
}

# ── user information card (explicit agentic graph research) ───────────
PROFILE_MODEL = "z-ai/glm-5.2"
PROFILE_REASONING = {"enabled": True}
PROFILE_TIMEOUT_S = 180.0
PROFILE_MAX_TOOL_CALLS = 24
PROFILE_MAX_CHARS = 12_000
PROFILE_CONTEXT_MAX_CHARS = 12_000
REENRICH_CONCURRENCY = 4

# ── batched enrichment (similarity-clustered, keyed by note_id) ────────
# Related notes enriched in one call converge on consistent entity/topic
# vocabulary, which is what builds graph edges; batching also amortizes the
# user card prefix. Any note missing/invalid in a batch response falls back
# to a per-note call.

# ── notes sync / connectors ────────────────────────────────────────────
SYNC_INTERVAL_S = 300.0
CONNECTOR_DOC_MAX_CHARS = 12000          # per-document body cap at extraction
# each conversation contributes the smaller of its last IMESSAGE_WINDOW_DAYS
# of messages or its last IMESSAGE_MAX_MESSAGES; conversations whose most
# recent message predates the window are skipped entirely
IMESSAGE_WINDOW_DAYS = 183               # ~6 months
IMESSAGE_MAX_MESSAGES = 150              # most-recent-N cap per conversation
CLAUDE_MEMORY_GLOB = "~/.claude/projects/*/memory/*.md"
CODEX_MEMORY_GLOBS = ("~/.codex/AGENTS.md", "~/.codex/memory/**/*.md", "~/.codex/memories/**/*.md")
OBSIDIAN_APP_JSON = "~/Library/Application Support/obsidian/obsidian.json"
IMESSAGE_DB = "~/Library/Messages/chat.db"
CONTACTS_DB_GLOB = "~/Library/Application Support/AddressBook/**/AddressBook-v22.abcddb"
GIST_MAX_CHARS = 220                     # was 120: mid-word truncation degraded retrieval docs
                                         # and card copy (audit finding #5)

# ── email connectors (gmail/outlook — spec 2026-07-08-email-connectors) ─
# Unit of ingestion is the THREAD. A deterministic funnel (no LLM) keeps
# enrichment cost sane: window → header/sender drops → importance cap.
EMAIL_WINDOW_DAYS = 120                  # threads whose newest message is older are skipped
EMAIL_MAX_THREADS = 400                  # hard cap on threads reaching enrichment, per source
EMAIL_FETCH_CEILING = 1500               # raw threads fetched before the funnel (safety valve)
EMAIL_MAX_MESSAGES = 25                  # most-recent-N cap per thread
EMAIL_MESSAGE_MAX_CHARS = 2000           # per-message cap after quote/signature stripping
EMAIL_SYNC_MIN_INTERVAL_S = 3600.0       # in-process extract cache: the 300s sync loop must
                                         # not refetch the whole mailbox every pass
# senders whose local part matches are automated; threads consisting only of
# these survive only when you replied in the thread
EMAIL_NOREPLY_LOCALPARTS = frozenset(
    """noreply no-reply donotreply do-not-reply notifications notification alerts alert
    billing receipts receipt invoice invoices statement statements mailer-daemon bounce
    bounces newsletter newsletters news updates update marketing offers deals promo
    promotions confirm confirmation orders order support help info hello contact
    account accounts security admin system automated digest reminder reminders
    """.split()
)
# subjects matching this AND never replied to → transactional noise (billing,
# receipts, account/security boilerplate, marketing). Replying rescues.
EMAIL_TRANSACTIONAL_SUBJECT_RE = (
    r"(?i)(receipt|invoice|payment\s+(?:received|due|confirmation|failed)"
    r"|order\s+(?:confirm\w*|shipped|delivered|placed)"
    r"|your\s+(?:statement|bill|order|subscription|receipt|invoice|account)"
    r"|verify\s+your|verification\s+code|confirm\s+your\s+(?:email|account)"
    r"|password\s+reset|reset\s+your\s+password|security\s+(?:alert|code|notice)"
    r"|two.factor|sign.in\s+(?:attempt|alert|detected)|new\s+login"
    r"|terms\s+of\s+service|privacy\s+policy|welcome\s+to|free\s+trial"
    r"|%\s*off|sale\s+ends|last\s+chance|don.t\s+miss)"
)
# importance score: S = Σ weight·signal; survivors sorted desc, capped
EMAIL_SCORE_WEIGHTS = {
    "recency": 1.0,                      # exp half-life decay over newest-message age
    "replied": 2.0,                      # you sent a message in the thread: gold tier
    "known_participant": 1.0,            # sender in Contacts or prior-correspondent set
    "flagged": 0.5,                      # starred / flagged / high importance
    "length": 0.5,                       # min(messages, 8)/8 — real back-and-forth
}
EMAIL_RECENCY_SCORE_HALF_LIFE_DAYS = 30.0

# ── note graph / map communities ───────────────────────────────────────
COMMUNITY_LP_ITERS = 20                  # label-propagation passes (legacy-DB fallback communities)
PROJECTION_MAX_DEGREE = 12               # product map only; never retrieval
# Map cleanliness: fill-phase edges below this weight are pruned. The spanning
# forest ignores the floor, so every note keeps its best route (no new
# isolated nodes) — only the mesh of weak incidental links disappears.
PROJECTION_FILL_MIN_WEIGHT = 0.35

# ── build-time note clusters (map lens; persisted per build) ──────────
# Clustering edge weight re-weights projection mechanisms: the entity
# mechanism noisy-or saturates (~76% of production edges ≥ 0.99), so it is
# capped and down-weighted; semantic and fact stay discriminative.
CLUSTER_W_SEMANTIC = 1.0
CLUSTER_W_FACT = 0.7
CLUSTER_W_ENTITY = 0.5                   # entity mechanism now discriminates (was saturated at 1.0)
CLUSTER_MIN_WEIGHT = 0.15                # combined weight below this: edge ignored for clustering
CLUSTER_RESOLUTION = 1.0                 # modularity resolution (higher → smaller clusters)
CLUSTER_MAX_SIZE = 35                    # clusters larger than this are recursively re-clustered
CLUSTER_MIN_SIZE = 3                     # clusters smaller than this merge into their best neighbor
# IDEA_SEARCH_MODEL lives below SUMMARY_MODEL.

# ── evidence-first passage/base graph ─────────────────────────────────
PASSAGE_TARGET_TOKENS = 220
PASSAGE_MAX_TOKENS = 300
PASSAGE_OVERLAP_TOKENS = 40
BASE_GRAPH_PASSAGE_K = 12
FACT_REVERSE_WEIGHT_DEFAULT = 0.35
# contains weight = clip01(idf × confidence × count_bonus × surprise / SCALE):
# realistic max ≈ 3.2 × 1.0 × 2.1 ≈ 6.7 → ~1.0, typical mid values land 0.3-0.7
# (H2 — without the rescale every production edge clipped to exactly 1.0)
CONTAINS_WEIGHT_SCALE = 14.0             # ≈ formula max (idf 6.7 × conf 1.0 × bonus 2.1); sqrt-compressed
# frequency-surprise soft cap (edge-quality item 3): entities seen in more
# notes than this co-occur with any one passage roughly at chance, so their
# contains weights carry surprise = 1/(1+log1p(df − cap)) beyond it
DF_SOFT_CAP = 8
# retrieval-matrix hub dampening (edge-quality item 3): entity nodes with more
# than this many out-neighbors forward only HUB_DEGREE_CAP/degree of their
# mass; the remainder parks on a self-loop so columns stay stochastic
HUB_DEGREE_CAP = 50
# note_salience = ingest_salience × content_quality; notes below this floor
# skip GLiNER/mentions/facts/base-graph entity participation (§0.2). Passages,
# FTS, and embeddings are ALWAYS built; the gate re-evaluates every build.
EXTRACTION_SALIENCE_FLOOR = 0.35
# Fail-safe: abort a build (clean BuildError, prior snapshot kept) if its
# resident set crosses this. Local model inference on this corpus peaks well
# under 2 GB with the MPS allocator watermark cap in taurine_core/__init__.py;
# the ceiling is headroom that still fits the packaged desktop sidecar. 0 = off.
BUILD_MEMORY_CEILING_GB = 16.0
# explicit owner aliases (merged with the name parsed from the user card) —
# owner/self entities above the df cap are traversal hubs (§10.5)
OWNER_ALIASES: frozenset[str] = frozenset()

# ── C2 retrieval ───────────────────────────────────────────────────────
C2_THROTTLE_S = 8.0
LIVE_WINDOW_MIN_S = 12.0
LIVE_WINDOW_MAX_S = 45.0
LIVE_WINDOW_MAX_CHARS = 6000
LIVE_ACCUM_DECAY = 0.85                  # per accepted pass, prior accumulated mass decays by this
LIVE_ACCUM_NOTE_CAP = 1.5                # max accumulated session mass per note
LIVE_ACCUM_EPSILON = 0.02                # accumulated entries below this are pruned
LIVE_TOPIC_SHIFT_SIM = 0.35              # window-vs-EMA cosine below this on an accepted pass → pivot
LIVE_SPANS_BUDGET_S = 0.3                # soft budget for GLiNER query spans; timeout → no spans
LIVE_LEDGER_MAX_CHUNKS = 2000            # hard safety cap on retained transcript chunks
RRF_K = 60
HEAD_TOP_K = 20
C2_TOP_K = 8                             # match the search-lane quota (was 5: entity notes
                                         # repeatedly died at ordered[6] — see 2026-07-06 triage)

# ── fuzzy entity snapping (ASR spelling drift: "Armand" → "Armaan") ────
FUZZY_SNAP_MIN_LEN = 4                   # only snap spans at least this long
FUZZY_SNAP_MAX_DIST = 2                  # max Levenshtein distance to an alias/title term
FUZZY_SNAP_LABELS = ("person", "organization", "project")  # spans eligible for snapping

# ── summarizer (spec 2026-07-08: one pass, change-triggered) ───────────
SUMMARY_MODEL = "deepseek/deepseek-v4-flash"
SUMMARY_FALLBACK_MODEL = "deepseek/deepseek-chat"
IDEA_SEARCH_MODEL = SUMMARY_MODEL        # idea-lens per-note alignment adjudication
CONTEXT_ENRICH_MODEL = SUMMARY_MODEL     # situating prefixes for episodic passages (empty on failure)
CONTEXT_ENRICH_MODE = "llm"             # "llm" (current) | "template" | "hybrid"
CONTEXT_LLM_MIN_CHARS = 2000             # hybrid: long, plausibly multi-topic source text
CONTEXT_LLM_MIN_PASSAGES = 3             # hybrid: multi-passage threads receive an LLM top-up
CONTEXT_ENRICH_BATCH = 10                # passages per enrichment call, grouped by note
CONTEXT_ENRICH_CONCURRENCY = 6           # bounded parallel enrichment calls
SUMMARY_TIMEOUT_S = 20.0
SUMMARY_MIN_INTERVAL_S = 25.0            # hard floor between summarizer passes
SUMMARY_MAX_STALENESS_S = 240.0          # force a pass if new speech and this much elapsed
SUMMARY_MIN_NEW_CHARS = 200              # don't resummarize silence
SUMMARY_CHANGE_JACCARD = 0.6             # top-8 grounding overlap below this → trigger
SUMMARY_CHANGE_TOP_K = 8                 # grounding prefix compared by Jaccard
SUMMARY_DRIFT_COSINE = 0.70              # window-vs-topic-centroid cosine below this → trigger
TOPIC_CENTROID_DECAY = 0.8               # EMA factor for the topic centroid
SUMMARY_EMPTY_BACKOFF_S = 40.0           # after a no_update reply, same set can't re-fire sooner
SUMMARY_DELTA_MAX_CHARS = 6000           # transcript delta cap per pass
SUMMARY_INVALID_CITATION_MAX = 0.30      # reject summary if > this fraction of citations unknown
ACTIONABLE_MIN_CONF = 0.8
ACTIONABLE_QUOTE_MIN_RATIO = 0.85        # rapidfuzz partial ratio vs transcript delta (0-1)
ACTIONABLE_MAX_PER_PASS = 2
ACTIONABLE_DEDUPE_SIM = 0.90             # title-embedding cosine vs open actionables → update, not new

# ── chat (knowledge-base Q&A — spec 2026-07-08-chat-knowledge-base) ────
CHAT_MODEL = "z-ai/glm-5.2"
CHAT_REASONING = {"enabled": True}       # per-call override: thinking ON for chat only
CHAT_FALLBACK_MODEL = "deepseek/deepseek-v4-pro"
CHAT_TIMEOUT_S = 120.0                   # whole-stream budget per round trip
CHAT_MAX_TOOL_CALLS = 6
CHAT_MAX_CONCURRENT_TURNS = 2
CHAT_SEARCH_TOOL_K = 8                   # default k for search_evidence
CHAT_NOTE_READ_MAX_CHARS = 6000
CHAT_HISTORY_MAX_TURNS = 12
CHAT_TITLE_MAX_WORDS = 6

# ── grounded edge adjudication (edge-quality spec items 1/5b/6) ────────
# Optional build lane, same pattern as relation completion: absent LLM ⇒
# skipped, adjudicated=0 everywhere, build completes. Temperature 0 always.
ADJUDICATION_MODEL = "openai/gpt-oss-120b"
ADJUDICATION_CONCURRENCY = 6             # bounded parallel judge calls (all lanes)
# 25-fact packed batches produced truncation-length malformed responses on the
# 2026-07-13 verification build (7 failed batches, 53 malformed rows); 12 keeps
# most of the call reduction with responses safely inside the token cap.
ADJUDICATION_FACTS_PER_CALL = 12         # facts per judge call (packed across notes)
ADJUDICATION_MAX_TOKENS = 4096           # cap runaway judge generations (fail-open)
# gpt-oss-120b collapsed this lane (618→9 facts, 88% offset-parse rejection on
# the 2026-07-13 verification build); flash was the only probed model emitting
# valid grounded rows on the densest completion notes.
COMPLETION_MODEL = "deepseek/deepseek-v4-flash"
COMPLETION_MAX_TOKENS = 8192             # cap runaway completions (one 94KB malformed response observed)
COMPLETION_NO_RELATION_MIN_PERCENTILE = 0.5  # no-relation gate needs median-or-better value
COMPLETION_TOOL_HOP_CAP = 40             # extra tool/anchor notes per build in the completion lane
# 12-note batches drove flash responses past the output cap (7/25 batches
# truncated mid-JSON at ~21KB ≈ 8192 tokens on the 2026-07-13 flash rebuild)
COMPLETION_NOTES_PER_CALL = 6            # completion batch size: response must fit COMPLETION_MAX_TOKENS
COMPLETION_CONCURRENCY = 4               # parallel completion batches
ENTITY_NEIGHBORHOOD_FACTS = 6            # resolved query entities surface their top facts in the pack
ADJUDICATION_TEXT_MAX_CHARS = 4000       # passage text supplied per fact-judge call
ADJUDICATION_REJECT_BELOW = 0.3          # judged score < this ⇒ status 'rejected'
ADJUDICATION_DOWNWEIGHT_BELOW = 0.7      # scores in [reject, this) multiply the fact edge weight
NLI_SCREEN_ENABLED = False               # current behavior until calibration is verified
# 2026-07-13 fit: train precision/calls 0.9167/2.75%; heldout 0.8000/3.70%.
# The 0.95 precision + 40% savings gate was not met, so the lane stays disabled.
NLI_KEEP_FLOOR = 0.99
NLI_AUTO_REJECT = False                  # destructive rejection remains LLM-only
NLI_REJECT_CEIL = 0.05                   # inactive while NLI_AUTO_REJECT is false
SEMANTIC_REASON_BATCH = 20               # cross-note semantic pairs per judge call
SEMANTIC_REASON_SNIPPET_CHARS = 300      # per-passage snippet supplied to the judge
# Reasons are an optional retrieval hint on the strongest cross-note links, not
# a per-edge requirement. Only the top-weighted cross-note semantic edges are
# judged; the long tail keeps an empty reason (never dropped). This is the main
# lever against adjudication call volume — ~8k cross-note edges would otherwise
# be hundreds of judge calls per build for the lowest-value signal.
SEMANTIC_REASON_MAX_EDGES = 600          # 0 = judge every cross-note edge
SEMANTIC_PRUNE_MODE = "llm"              # "local" only after offline fit verification
# Intelligent fact gating: a fact is only worth an LLM grounding check when it
# is genuinely risky. Facts that are high-confidence, from evergreen written
# sources, and on a non-exclusive predicate are trusted (GLiNER already gated
# them); episodic (chat) sources, low confidence, exclusive predicates, and
# LLM-completed relations are always judged — that is where the garbage lives.
ADJUDICATION_TRUST_CONF = 0.85           # ≥ this + evergreen + non-exclusive ⇒ skip judge
ADJUDICATION_EVERGREEN_SOURCES = frozenset(
    {"apple_notes", "local_folders", "claude_mem", "codex_mem", "obsidian", "gmail", "outlook"}
)  # episodic sources (imessage) are always judged
MERGE_GRAY_ZONE_CAP = 200                # gray-zone candidate pairs judged per build
MERGE_GRAY_ZONE_EMBED_BAND = 0.03        # [ENTITY_EMBED_SIM − band, ENTITY_EMBED_SIM) emits a pair
MERGE_ADJUDICATION_MIN_CONF = 0.8        # same_entity verdicts below this never merge
MERGE_ADJUDICATION_BATCH = 16            # gray-zone pairs per judge call

# ── shared retrieval, rerank, and evidence ────────────────────────────
TRIPLE_RECALL_K = 20
TRIPLE_SEED_K = 5
PASSAGE_RESET_W = 1.0
ENTITY_RESET_W = 1.0
QUERY_ENTITY_RESET_W = 0.5
SESSION_RESET_W = 0.35
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
LIVE_RERANK_CANDIDATES = 30
CHAT_RERANK_CANDIDATES = 60
LIVE_EVIDENCE_PASSAGES = 8
CHAT_EVIDENCE_PASSAGES = 12
TEMPORAL_HISTORY_DOWNWEIGHT = 0.5        # historical/superseded facts under "any"/past intents
HUB_ACTIVATION_EDGE_WEIGHT = 0.5         # synthesized contains weight for query-activated hubs (§10.5)
HUB_ACTIVATION_MAX_EDGES = 64            # synthesized contains links per activated hub
RETRIEVAL_FALLBACK_FLOOR_RATIO = 0.75    # chat/tool fallback lane floor = ratio × head acceptance floor
RETRIEVAL_FALLBACK_PASSAGES = 4          # fallback pack cap: 4 passages / 0 facts / 0 paths / ≤1 per note
EVIDENCE_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# ── grounding / graph traversal (latent retrieval, no LLM) ─────────────
GROUNDING_HYSTERESIS = 1.15              # challenger must beat incumbent by this factor
PPR_DAMPING = 0.5                        # HippoRAG value: mass stays within ~2 hops of seeds
PPR_TOL = 1e-8                           # L1 convergence tolerance
PPR_MAX_ITERS = 50
DUP_EDGE_SIM = 0.92                      # same-source semantic mechanism ≥ this → projection edge
                                         # excluded from quality rank + flagged near-duplicate (M14)

# ── static priors (source rank, quality rank, recency decay) ───────────
SOURCE_RANK = {
    "apple_notes": 1.00,                 # written notes: highest
    "obsidian": 1.00,
    "local_folders": 1.00,               # user-attached documents: deliberate, notes tier
    "imessage": 0.85,                    # text messages: a bit lower
    "gmail": 0.85,                       # email threads: iMessage tier
    "outlook": 0.85,
    "claude_mem": 0.70,                  # agent memories: a bit lower still
    "codex_mem": 0.70,
}
SOURCE_RANK_DEFAULT = 0.85               # future connectors until explicitly ranked

# ── value score ────────────────────────────────────────────────────────
VALUE_RECENCY_HALF_LIFE_DAYS = 30.0      # value-score recency decay

QUALITY_DAMPING = 0.85                   # quality-rank PageRank damping
QUALITY_TOL = 1e-6                       # L1 convergence tolerance
QUALITY_MAX_ITERS = 100
QUALITY_CLIP = 2.0                       # quality_rank = clip(log(r/median), ±this)
RECENCY_OFFSET_DAYS = 2.0                # grace window before decay starts
RECENCY_HALF_LIFE_DAYS = {
    "imessage": 30.0,                    # episodic sources decay fast
    "gmail": 30.0,                       # email is episodic too
    "outlook": 30.0,
    "claude_mem": 60.0,
    "codex_mem": 60.0,
    "apple_notes": 90.0,
    "obsidian": 120.0,                   # evergreen notes barely decay
    "local_folders": 120.0,              # attached documents: evergreen tier
}
RECENCY_HALF_LIFE_DEFAULT_DAYS = 90.0
RECENCY_FLOOR = 0.15                     # decay is a downweight, never an exclusion

# ── C1 exact collector ─────────────────────────────────────────────────
# ALL aliases require word-boundary matches: substring matching fired 'comp'
# inside "company" and 'bing' inside unrelated words (2026-07-07 triage)
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

# ── C2 GLiNER2 span extraction (schema-based, label descriptions) ─────
GLINER2_MODEL = "fastino/gliner2-large-v1"
GLINER2_THRESHOLD = 0.6                  # was the 0.5 default with confidence discarded;
                                         # 2026-07-07 triage: ~half of spans were junk
# pronouns/vocatives/generic nouns GLiNER tags as entities in casual speech —
# dropped regardless of label or confidence (lowercase-normalized match)
GLINER_SPAN_STOPLIST = frozenset(
    """i you me he she we they them him her it this that these those
    someone somebody anyone everybody everyone person people persons
    dude bro bruh man guy guys girl girls kid kids friend friends
    parent parents mom dad brother sister family
    thing things stuff okay ok yeah yep nah alright right cool damn
    meeting call session note notes list money data company business
    all the people one person someone else
    """.split()
)
# these labels are proper nouns by definition: a span must contain an
# uppercase character (as spoken/ASR-cased) or be multiword to survive
GLINER_PROPER_LABELS = ("person", "organization", "project")
# span-context queries: dense-search the sentence(s) around a triggered span,
# not just the span words; the context window extends before the C2 window
C2_CONTEXT_CHUNKS = 2                    # extra transcript chunks before the window
SPAN_CONTEXT_SENTENCES = 1               # sentences around the span occurrence, each side
SPAN_CONTEXT_MAX_CHARS = 240             # hard cap per context query
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

# ── GLiNER ingestion (local entities + relations) ─────────────────────
GLINER_INGEST_BATCH = 8                  # notes per forward batch
GLINER_LONG_CHUNK_TOKENS = 384
GLINER_LONG_OVERLAP_TOKENS = 64
GLINER_BATCH_TOKEN_BUDGET = 3072
GLINER_CHUNK_SIZE = GLINER_LONG_CHUNK_TOKENS       # legacy alias until extractor cutover
GLINER_CHUNK_OVERLAP = GLINER_LONG_OVERLAP_TOKENS # legacy alias until extractor cutover
GLINER_ENTITY_MIN_CONF = 0.5             # calibration-pending entity floor
GLINER_RELATION_MIN_CONF = 0.45          # calibration-pending relation floor
# Extraction backend stays on current GLiNER2 behavior until the quality gate
# selects a replacement.
EXTRACTOR_BACKEND = "gliner2"
RELEX_MODEL = "knowledgator/gliner-relex-large-v1.0"
# 3×3 sweep best point (2026-07-13): equal recall to stricter settings with
# +77% novel yield, and judge-sampled precision of the novel pool matched
# GLiNER2's, so the extra volume is real signal for the adjudicator to filter.
RELEX_ENTITY_THRESHOLD = 0.35
RELEX_RELATION_THRESHOLD = 0.7
RELEX_ADJACENCY_THRESHOLD = 0.55
RELEX_MAX_TOKENS = 2048
# entities above this active-note df fraction are connective stopwords
# (owner-name/pronoun hubs); contains edges remain for PPR hub dilution
ENTITY_SHARED_DF_FRACTION_CAP = 0.06
# time words like "today"/"tomorrow" are not connective tissue
SHARED_EDGE_ETYPE_STOPLIST = frozenset({"date_or_deadline"})
# entity-resolution stage 5: embedding cosine floor for merge *candidates*
# (verified by lexical/digit/type guards; never sufficient alone)
ENTITY_EMBED_SIM = 0.95
# generic concept surfaces are provenance, not traversal nodes (§10.5)
CONCEPT_STOPLIST = frozenset(
    {"things", "stuff", "ideas", "notes", "list", "misc", "todo", "update", "meeting"}
)
INGEST_ENTITY_TYPES: dict[str, str] = {
    "person": "A named individual mentioned in the note",
    "organization": "A company, team, client, vendor, or institution",
    "project": "An internal work initiative, codename, feature, or plan",
    "product": "A product, service, app, or tool",
    "place": "A physical location, city, venue, or region",
    "event": "A meeting, launch, trip, conference, or scheduled occurrence",
    "date_or_deadline": "A date, time, deadline, milestone, or time window",
    "concept": "abstract topic, technology, or practice the note is about",
}
INGEST_RELATION_TYPES: dict[str, str] = {
    "works_with": "{head} works with {tail}",
    "works_at": "{head} works at {tail}",
    "member_of": "{head} is a member of {tail}",
    "part_of": "{head} is part of {tail}",
    "located_at": "{head} is located at {tail}",
    "scheduled_for": "{head} is scheduled for {tail}",
    "paid_or_owes": "{head} paid or owes {tail}",
    "owns_or_has": "{head} owns or has {tail}",
    "uses": "{head} uses {tail}",
    "about": "{head} is about {tail}",
    "met_with": "{head} met with {tail}",
    "related_to": "{head} is related to {tail}",
    "decided": "{head} decided {tail}",
    "prefers": "{head} prefers {tail}",
    "blocked_by": "{head} is blocked by {tail}",
    "responsible_for": "{head} is responsible for {tail}",
    "completed": "{head} completed {tail}",
    "supersedes": "{head} supersedes {tail}",
}
REL_FACT_MIN_CONF = 0.7                  # relation fact without LLM
REL_EDGE_W = 0.25                        # relation edge base weight
REL_CATCHALL_FACTOR = 0.5                # related_to down-weight

# ── PII redaction (email bodies: regex layer + GLiNER2 schema layer) ──
# Regex owns fixed-format identifiers (Luhn-checked cards, SSNs, account
# numbers, one-time codes); GLiNER2 catches contextual PII with its own
# schema. person/organization/address are deliberately ABSENT: names and
# addresses stay — entity linking needs them. Applied per message BEFORE
# content_hash, so raw PII never lands in the DB, embeddings, or FTS.
EMAIL_PII_GLINER_THRESHOLD = 0.35        # recall-leaning: a false positive costs one
                                         # redacted span; a false negative leaks PII
GLINER2_PII_SCHEMA = {
    "credit_card_number": "A credit or debit card number, possibly with spaces or dashes",
    "social_security_number": "A social security number or national identity number",
    "bank_account_number": "A bank account number, routing number, or IBAN",
    "password_or_credential": "A password, PIN, access token, or other account credential",
    "government_id": "A passport, driver's license, or other government ID number",
    "date_of_birth": "A person's date of birth",
}
