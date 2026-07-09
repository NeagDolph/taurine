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
EMBED_QUERY_PROMPT = (
    "Instruct: Given a snippet of spoken conversation, retrieve personal "
    "notes relevant to what is being discussed\nQuery: "
)
CHUNK_MAX_TOKENS = 300                   # sentence-boundary chunking

# ── LLM (ALL calls via OpenRouter) ─────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENRICH_MODEL = "z-ai/glm-5.2"            # enrichment is offline: quality over latency
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
ENRICH_BATCH_SIZE = 10                   # max notes per LLM call
ENRICH_BATCH_CHAR_BUDGET = 24_000        # max summed note-text chars per call
ENRICH_BATCH_MIN_NOTES = 3               # below this, per-note calls (batch overhead loses)

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

# ── graph term resolution (synonym soft-links, no LLM) ────────────────
# entity/topic strings are embedded (cached in term_vectors) and clustered;
# canonical forms are what shared-term edges intersect on. Calibrated on the
# real corpus 2026-07-07: true variants ('Panasonic Avionics Corp'↔'Corp.',
# 'recommendation-engine'↔'recommendation_engine') sit ≥0.98; distinct
# entities (ShopLC↔Shopify 0.67) are far below. Terms whose digit sequences
# differ NEVER merge ('PR #370'≠'#371', 'voice_agent_v3'≠'v4').
TERM_SYNONYM_MIN_SIM = 0.95
# merges additionally require lexical compatibility (containment after
# stripping non-alphanumerics, or normalized Levenshtein ≥ this): embeddings
# alone merge distinct referents ('Cologne'↔'Düsseldorf' 0.957) while true
# variants are always string-close too
TERM_LEXICAL_MIN_RATIO = 0.80

# ── graph adjudication (corners LLM pass + edge facts) ────────────────
# GLiNER-confirmed facts skip the queue; remaining uncertainty-band pairs
# are gated by value score or bridge status before the LLM budget.
ADJUDICATE_MODEL = "deepseek/deepseek-v4-pro"
ADJUDICATE_TIMEOUT_S = 90.0
ADJUDICATE_BATCH_PAIRS = 20              # pairs per LLM call
ADJUDICATE_CONCURRENCY = 4
ADJ_VALUE_FRACTION = 0.20                # corners value gate
ADJUDICATE_MAX_PAIRS_PER_SYNC = 100      # sync adjudication budget
BRIDGE_RANK_BOOST = 1.5                  # bridge-pair budget boost
COMMUNITY_LP_ITERS = 20                  # label-propagation passes
ADJ_TEXT_CAP = 700                       # note-text chars per side
# Legacy margin/band knobs stay until Task 9 removes old callers.
MARGIN_SEMANTIC_SIM = 0.50               # legacy near-miss dense floor
MARGIN_SHARED_W = 0.15                   # legacy near-miss shared-term floor
MARGIN_MAX_PER_NOTE = 5                  # legacy margin-pair cap
VETO_MAX_W = 0.45                        # legacy weak-kept edge band
FACT_TOP_EDGES_PER_NOTE = 6              # strong edges fact candidates

# ── note graph ─────────────────────────────────────────────────────────
GRAPH_KNN = 6
GRAPH_MIN_SIM = 0.55                     # percentile-matched to Qwen3-Embedding cosines
                                         # (bge 0.60 = p93.6 of pairwise sims → 0.550)
GRAPH_MIN_SHARED_W = 0.25
GRAPH_MAX_DEGREE = 12

# ── C2 retrieval ───────────────────────────────────────────────────────
C2_THROTTLE_S = 8.0
C2_WINDOW_CHUNKS = 3
C2_WINDOW_MIN_S = 12.0
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
CHAT_CONTEXT_NOTES = 16                  # Stage-A pack: broad recall posture
CHAT_CONTEXT_FACTS = 24
CHAT_SEED_DECAY = 0.7                    # per-turn accumulator decay (chats pivot fast)
CHAT_SEARCH_TOOL_K = 8                   # default k for the search_notes tool
CHAT_NOTE_READ_MAX_CHARS = 6000
CHAT_HISTORY_MAX_TURNS = 12
CHAT_EMBED_QUERY_PROMPT = (
    "Instruct: Given a question about the user's personal notes, retrieve "
    "notes relevant to answering it\nQuery: "
)
CHAT_TITLE_MAX_WORDS = 6

# ── grounding / graph traversal (latent retrieval, no LLM) ─────────────
GROUNDING_TOP_K = 12                     # grounding-set size
GROUNDING_FACTS_MAX = 20                 # edge facts carried per pass
GROUNDING_HYSTERESIS = 1.15              # challenger must beat incumbent by this factor
GROUNDING_PPR_POOL_K = 30                # PPR candidates blended with direct fused hits
GROUNDING_FACT_SEED_TOP_K = 5            # seed-neighbor edge facts carried for grounding notes
SEED_TOP_K = 10                          # fused candidates entering the seed accumulator per pass
SEED_ACCUM_DECAY = 0.85                  # per-pass decay of the meeting seed accumulator
PPR_DAMPING = 0.5                        # HippoRAG value: mass stays within ~2 hops of seeds
PPR_TOL = 1e-8                           # L1 convergence tolerance
PPR_MAX_ITERS = 50
PPR_BLEND = 0.35                         # rel = (1-β)·norm(fused RRF) + β·norm(PPR mass)
PPR_PASSAGE_RESET_W = 1.0                # note-seed reset weight
PPR_ENTITY_RESET_W = 1.0                 # entity-seed reset weight
ENTITY_SEED_W = 0.8                      # GLiNER entity seed weight
FACT_EDGE_BOOST = 1.5                    # traversal weight multiplier for adjudicated fact edges
SHARED_EDGE_BOOST = 1.35                 # entity/topic/mixed edges (audit F3.1 correction)
DUP_EDGE_SIM = 0.95                      # same-source mean chunk cosine ≥ this → edge excluded from quality rank

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

# ── value score / slim LLM call ────────────────────────────────────────
VALUE_RECENCY_HALF_LIFE_DAYS = 30.0      # value-score recency decay
SLIM_LLM_VALUE_FRACTION = 0.15           # top-value changed notes
SLIM_LLM_MIN_NOTES = 5                   # slim-call floor per sync
SLIM_UTTERANCES = True                   # Task 12 ablation may flip off

QUALITY_DAMPING = 0.85                   # quality-rank PageRank damping
QUALITY_TOL = 1e-6                       # L1 convergence tolerance
QUALITY_MAX_ITERS = 100
QUALITY_CLIP = 2.0                       # quality_rank = clip(log(r/median), ±this)
QUALITY_LAMBDA = 0.20                    # blend: multiplier = exp(λ·quality_rank) ∈ [0.67, 1.49]
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
GLINER_CHUNK_SIZE = 384                  # long-note window tokens
GLINER_CHUNK_OVERLAP = 64                # long-note overlap tokens
GLINER_ENTITY_MIN_CONF = 0.5             # calibration-pending entity floor
GLINER_RELATION_MIN_CONF = 0.45          # calibration-pending relation floor
# entities above this active-note df fraction are connective stopwords
# (owner-name/pronoun hubs); contains edges remain for PPR hub dilution
ENTITY_SHARED_DF_FRACTION_CAP = 0.06
# time words like "today"/"tomorrow" are not connective tissue
SHARED_EDGE_ETYPE_STOPLIST = frozenset({"date_or_deadline"})
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
