# Local Extraction & Quality — Implementation Plan

> **For agentic workers:** Each track below is a self-contained brief for one implementation agent. Read the spec first: `docs/superpowers/specs/2026-07-13-local-extraction-quality-design.md`. All commands run from `/Users/neil/taurine/core`; tests via `.venv/bin/python -m pytest`. TDD: failing test → implement → pass → commit per unit. Never use `git add -A`; stage only files you changed. Never stage `.claude/`, `*.db*`, or scratch files. Commit trailer: `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.

**Global constraints**
- Every lane stays fail-open: model-load failure or scorer error must never fail a build (log + degrade to current behavior).
- New behavior ships behind config flags DEFAULTING TO CURRENT behavior (`EXTRACTOR_BACKEND="gliner2"`, `CONTEXT_ENRICH_MODE="llm"`, `NLI_SCREEN_ENABLED=False`, `SEMANTIC_PRUNE_MODE="llm"`). The orchestrator flips them after benchmarks.
- Do not modify `_FACT_PROMPT_VERSION`, `_fact_key`, completion cache keys, or any BuildCache key derivation except where a track explicitly says so.
- Match surrounding code style. No new heavyweight runtime deps (sklearn allowed in offline eval scripts only, it is already installed).

## Track 0 (shared, part of Track A agent's brief): quality_bench harness

Create `taurine_core/evaluation/quality_bench.py` with subcommands (argparse, mirroring self_calibrate.py's CLI shape):
- `extraction --db <knowledge.db> --backends gliner2,relex,hybrid --sample <n>`: loads notes via the store, runs each backend's `extract_notes`, scores silver-fact recall / new-fact yield per spec §Benchmark-1, writes JSON report to `--out`.
- `adjudication --cache <taurine-cache.db>`: loads `kind='fact'` rows, splits train/heldout by key hash %4==0, fits/evaluates NLI thresholds (used by Track C; Track A agent only creates the file with the extraction subcommand + shared report plumbing; Track C agent adds its subcommand).
- Shared: `write_report(path, payload)` with a stable schema `{variant, metrics{}, cost{}, timestamp_source}` (timestamps passed in, never Date.now-style calls inside).
Tests: `tests/test_quality_bench.py` — fixture corpus of ~6 synthetic notes with known facts; assert recall/yield computation correctness and report schema.

## Track A: GLiNER-Relex extractor backend

**Files:** create `taurine_core/indexer/extract_relex.py`, `tests/test_extract_relex.py`; modify `taurine_core/config.py` (add `EXTRACTOR_BACKEND = "gliner2"`, `RELEX_MODEL = "knowledgator/gliner-relex-large-v1.0"`, `RELEX_ENTITY_THRESHOLD = 0.4`, `RELEX_RELATION_THRESHOLD = 0.8`, `RELEX_ADJACENCY_THRESHOLD = 0.55`, `RELEX_MAX_TOKENS = 2048`), `taurine_core/indexer/extract.py` (backend dispatch in a new `extract_notes_dispatch` — server.py and build.py call sites switch to it), `taurine_core/indexer/build_cache.py` (`_schema_fingerprint` must include backend + Relex thresholds when backend != gliner2).

**Contract:** `extract_relex.extract_notes(notes, *, tokenizer=None, progress=None) -> dict[str, Extraction]` — byte-compatible with `extract.py`'s return schema (`Extraction` with `mentions: list[ExtractedMention]`, `relations: list[ExtractedRelation]`, note-global char offsets). Read `extract.py`'s `_extract_family` and schema helpers first and mirror its provenance behavior exactly (observed_at, historical flags, confidence propagation). Relation assertion span = minimal window covering subject+object mention spans when the model gives no relation span.

**Model access:** lazy process-cached like `live/spans.py`. `from gliner import GLiNER` — run the orchestrator's smoke findings (in the brief when dispatched); if the installed `gliner==0.2.27` lacks the Relex head, upgrade the package in `.venv` and record the resolved version + any dependency changes in the commit message. Fail-open: load failure → every note returns `Extraction(failed=True, error="relex unavailable")`.

**Hybrid mode:** `EXTRACTOR_BACKEND="hybrid"` = GLiNER2 mentions + Relex relations. Relex-discovered entities that GLiNER2 missed are kept only when they are relation endpoints.

**Predicate mapping:** derive Relex `relations` schema from `PREDICATES` (facts.py): label = predicate name, description = predicate description, `pairs_filter` from the predicate's declared subject/object etypes.

**Tests (mock the model — never download in tests):** dispatch honors backend flag; fail-open on load error; offset mapping note-global; assertion-window derivation; hybrid merge rules; fingerprint changes with backend+thresholds.

**Bench deliverable:** implement Track 0's `extraction` subcommand. Then run it for real (model download allowed at run time, not test time) over a 100-note sample, sweep the 3×3 threshold grid (entity {0.35,0.45}, relation {0.7,0.8,0.9}), then the full corpus at the best point. Write reports to `core/evals/quality/2026-07-13-extraction-*.json` and commit them. Print the summary table in your final message.

## Track B: hybrid template/LLM context enrichment

**Files:** modify `taurine_core/indexer/context_enrich.py`, `taurine_core/config.py` (`CONTEXT_ENRICH_MODE = "llm"`, `CONTEXT_LLM_MIN_CHARS = 2000`, `CONTEXT_LLM_MIN_PASSAGES = 3`), `tests/test_context_enrich.py`.

**Design:** pure function `template_context(note: NoteRecord, passage: EnrichmentPassage, position: str) -> str` (≤240 chars): source kind ("iMessage thread"/"Email"), participants parsed from title (`"iMessage: A, B, C"` → "between A, B, C"; `"Email: <subj>"` → subject), sender from `display_line`'s leading `[Sender]`, absolute month+year from `note.created`, position ∈ {"start of","middle of","end of"} by passage ordinal fraction. Handles missing metadata gracefully (skip clause, never emit placeholders).

In `enrich_passages`: mode `"template"` → every enrichable passage gets the template, zero LLM calls; mode `"hybrid"` → template for all, then the existing LLM path ONLY for notes with `len(note.text) > CONTEXT_LLM_MIN_CHARS or passage_count >= CONTEXT_LLM_MIN_PASSAGES`, LLM result replacing the template per passage on success (cache behavior unchanged); mode `"llm"` → exactly today. Template contexts are NOT cached in BuildCache (pure function) and must not disturb existing cache keys for LLM contexts.

**Tests:** template formatting for imessage/gmail fixtures incl. missing-metadata degradation; mode selection logic; hybrid gating (small note → no LLM call, large note → LLM call, LLM failure → template survives); llm mode unchanged (existing tests keep passing).

**Bench deliverable:** none in-agent (retrieval benchmark needs full rebuilds — orchestrator runs those). Your definition of done is tests + a printed sample of 10 real template contexts from the production DB (read-only query) in your final message.

## Track C: NLI fact-adjudication pre-screen  (same agent as Track D, do C first)

**Files:** create `taurine_core/indexer/nli_screen.py`, `tests/test_nli_screen.py`; modify `taurine_core/indexer/edge_adjudication.py` (screen inside `adjudicate_facts` between eligibility and batching), `taurine_core/config.py` (`NLI_SCREEN_ENABLED = False`, `NLI_KEEP_FLOOR = 0.9` placeholder — the bench fits the real value, `NLI_AUTO_REJECT = False`, `NLI_REJECT_CEIL = 0.05` placeholder), `taurine_core/evaluation/quality_bench.py` (add `adjudication` subcommand).

**Design:** `nli_screen.verbalize(fact) -> str` — per-predicate hypothesis templates for all 18 predicates (read `PREDICATES` for the list; e.g. works_at → "{subject} works at {object}", paid_or_owes → "{subject} paid or owes money to {object}"). `nli_screen.screen(facts, draft_by_passage, scorer) -> ScreenResult` with `kept: dict[fact_id, FactJudgment]` (support="supported", quote=assertion text, score=min(0.95, entailment)) and `pending: list[FactRecord]`. Scorer from `models_local.local_nli_scorer(config.EVIDENCE_NLI_MODEL)`; scorer-load failure → everything pending (fail-open toward LLM). Never screen: exclusive predicates (`facts.py` exclusivity set), `extraction_method == "llm_relation_completion"`, facts whose passage text is missing. Auto-reject only when `NLI_AUTO_REJECT` is True (leave False).

Wiring: when `NLI_SCREEN_ENABLED` and a scorer loads, screened-kept facts merge into `judgments` exactly like cache hits do (they must flow through `_fact_outcome` identically); pending facts continue to the existing cache/batch path. Screened keeps are NOT written to the judgments cache (they're cheap to recompute and semantics differ from judge verdicts) — instead tag them via a new `FactJudgment` field default or reuse score provenance; keep it minimal.

**Bench deliverable:** `quality_bench adjudication --cache taurine-cache.db` — load the 1,335 `kind='fact'` cached verdicts, recover each fact's premise/hypothesis from the production knowledge DB (join by rebuilding fact verbalizations from the facts table; where the source passage text is needed use the passages table), split train/heldout, sweep entailment floor over [0.7..0.99], report: keep-precision vs LLM keep, calls-eliminated %, at the smallest floor meeting precision ≥0.95. Fit `NLI_KEEP_FLOOR` on train, report heldout, write `core/evals/quality/2026-07-13-nli-adjudication.json`, set the fitted floor in config with heldout numbers in the comment. Print the table.

## Track D: semantic-edge local pruning + lazy reasons  (same agent as Track C, after C)

**Files:** create `taurine_core/indexer/edge_prune.py`, `tests/test_edge_prune.py`, `scripts/fit_edge_prune.py` (offline, sklearn OK); modify `taurine_core/indexer/edge_adjudication.py` / `build.py` (mode dispatch), `taurine_core/config.py` (`SEMANTIC_PRUNE_MODE = "llm"`), `taurine_core/server.py` (on-demand reason endpoint).

**Design:** `edge_prune.decide(pairs: Sequence[SemanticPair], features_by_pair) -> PruneResult` — logistic score from constants (weights fit offline by `scripts/fit_edge_prune.py` against the 557 cached `kind='semantic_reason'` verdicts; embed fitted coefficients as module constants with the fit date + heldout metrics in a comment). Features (compute in build where the pairs are assembled, pass in): cosine, shared_entity_count, min shared-entity IDF, same_source, title Jaccard, length ratio. Skip NLI cross-entailment in v1 (latency); note it as the iterate-once lever. Deterministic reason template: top shared entities or source-pair phrasing.

Mode `"local"`: `adjudicate_semantic_edges` is not called; `edge_prune.decide` drops/keeps and reasons deterministically. Mode `"llm"`: today, untouched.

**Lazy reasons endpoint:** POST `/graph/edge-reason` (src passage id, dst passage id) → if the stored reason is deterministic/template, generate the rich reason with the existing single-pair judge prompt, persist to the projection edge's reason column, return it. Auth/shape: follow the existing idea-lens endpoint pattern in server.py. Never called during build.

**Tests:** decide() on synthetic features (drop vs keep both ways), reason templates, mode dispatch (llm mode byte-identical behavior), endpoint happy-path + LLM-failure fail-open (returns template reason).

**Bench deliverable:** run `scripts/fit_edge_prune.py --cache taurine-cache.db --knowledge taurine-knowledge.db`, report heldout drop-agreement and false-drop rate, write `core/evals/quality/2026-07-13-edge-prune.json`, embed coefficients. Print the table. If heldout agreement < 0.90, still commit the module (flag stays "llm") and report the gap honestly — the orchestrator decides iterate-vs-scrap.

## Orchestrator reserve (not delegated)

Retrieval benchmarks (three context-mode rebuilds + extractor-variant rebuilds), threshold flips, final full-system benchmark, production install decision, outcome appendix on the spec.
