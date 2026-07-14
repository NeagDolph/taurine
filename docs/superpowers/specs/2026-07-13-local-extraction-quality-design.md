# Local Extraction & Quality — Design

**Date:** 2026-07-13
**Status:** Approved direction (Neil, in-session): "Replace gliner with gliner-relex, build out the hybrid context enrichment, build out non-llm fact adjudication and the semantic edge reasoning separately and benchmark all of it… keep the most useful and high value aspects… fix/improve the less value levers… otherwise scrap."
**Goal:** Maximize extraction + retrieval quality while moving work off LLMs. Every track ships behind a benchmark gate: measured better-or-equal → keep; fixable → iterate once; else scrap and document.

## Context

The enrichment pipeline currently makes ~300 LLM calls cold (post call-reduction work, commits e3d5b39..b4984da). Remaining LLM lanes: relation completion (flash), fact adjudication + semantic reasons + merge judge (gpt-oss-120b), context enrichment (flash). Research (2026-07-13, GLiNER-Relex paper arXiv 2605.10108) measured GLiNER2's relation extraction at 11.7 F1 on DocRED vs GLiNER-Relex at 31.3 — the quantified version of why the LLM completion lane exists.

Free labeled data already on disk (taurine-cache.db `adjudication_cache`): **1,335 fact judgments, 557 semantic-edge judgments, 204 merge judgments** — gpt-oss verdicts usable as ground truth for non-LLM replacements. The GPL self-calibration harness (`taurine_core/evaluation/self_calibrate.py`) generates retrieval eval sets on demand; `core/evals/graph_retrieval/calibration_questions.jsonl` holds the 16-question battery.

## Benchmark framework (build first, judge everything with it)

New module `taurine_core/evaluation/quality_bench.py` + CLI, producing one JSON report per variant:

1. **Extraction quality** (Track A): against the production DB's adjudicated-active facts (score ≥ 0.7, both GLiNER2 and completion-lane origins) as silver truth. Metrics: silver-fact recall (subject entity, object entity, predicate matched after alias normalization), new-fact yield, and **judge-sampled precision** — sample ≤100 novel facts per extractor, adjudicate with the existing gpt-oss lane, report acceptance rate. Entity side: mention count/coverage delta vs GLiNER2, entity-resolution stability (merged-entity count delta).
2. **Retrieval accuracy** (Tracks A/B): GPL-generated question set (≥150 questions, train/heldout split as in self_calibrate) + the 16-question battery. Metrics: top-1/top-3 passage hit rate, evidence-pack acceptance, abstention correctness on the 2 control questions.
3. **Adjudication agreement** (Track C): NLI screen verdicts vs the 1,335 cached gpt-oss fact judgments. Metrics: auto-keep precision vs LLM keep (target ≥0.95), auto-reject precision vs LLM reject (target ≥0.95), % of facts screened out of the LLM lane.
4. **Edge-pruning agreement** (Track D): classifier decisions vs the 557 cached semantic-edge judgments (kept-with-reason vs dropped). Metrics: drop-decision agreement (target ≥0.90), false-drop rate on LLM-kept edges (target ≤0.05).
5. **System cost**: LLM calls by lane, build wall-time by stage, peak memory (already in BuildReport).

Gate rule per track: **keep** if quality metrics ≥ targets with cost reduced; **iterate once** with a specific fix if within striking distance (≥80% of target); else **scrap**, keep the LLM lane, record findings in the spec's outcome appendix.

## Track A — GLiNER-Relex extractor (replace GLiNER2's relation head, maybe more)

New module `taurine_core/indexer/extract_relex.py` implementing the exact `extract_notes(notes, *, tokenizer, progress) -> dict[str, Extraction]` contract of `taurine_core/indexer/extract.py`, selected by `config.EXTRACTOR_BACKEND` (`"gliner2"` default until benchmarked, `"relex"`, `"hybrid"`).

- Model: `knowledgator/gliner-relex-large-v1.0` via the `gliner` package (installed 0.2.27 — **first step is a load/API smoke test; upgrade the package if the Relex head needs it**). Lazy process-cached load like `live/spans.py`; MPS placement like the embedder.
- Schema mapping: the 18-predicate registry (`_relation_descriptions` per source family in extract.py) maps to Relex relation labels with `pairs_filter` derived from each predicate's subject/object etype constraints (already encoded in `PREDICATES`).
- Spans: Relex emits entity char spans; if relation-level assertion spans are absent, derive `assertion_start/end` as the minimal window covering the subject and object mentions (the completion parser's tightest-mention-pair logic in `relation_completion.py` is the reference).
- 2048-token context: Relex reads the whole note (up to its window) rather than per-passage — cross-sentence relations are the whole point. Mentions must still be emitted with note-global offsets mapped back to passages downstream (existing behavior in extract.py's offset handling).
- `"hybrid"` mode: GLiNER2 entities (proven) + Relex relations only.
- Cache: `_schema_fingerprint()` in build_cache.py already hashes the extractor model id and schema — confirm the backend choice and Relex thresholds flow into it so variants never share cache rows.
- Thresholds: entities 0.3–0.5, adjacency 0.5–0.65, relations 0.7–0.9 (model card ranges); the eval harness sweeps a small grid (3×3 max) on a 100-note sample before the full run.
- Benchmark: framework metrics 1 + 2 for backends {gliner2, relex, hybrid}; wall-time per full extraction pass (target: within 2× of GLiNER2's).
- Decision: adopt the best backend; if Relex relations beat GLiNER2's, ALSO demote the completion lane's no-relation gate to notes where the winning extractor found ≥2 entities but 0 relations (shrinks LLM completion further).

## Track B — hybrid template/LLM context enrichment

Extend `taurine_core/indexer/context_enrich.py`:

- **Template prefix (always, free):** built from note metadata only — source kind, title (participants for iMessage: "iMessage: Rowan, Doug, Sanj"; subject for email: "Email: Wing consulate APIs document"), sender from `display_line`'s `[Sender]` prefix, absolute month+year from `created`, and passage position ("early/mid/late in the thread"). Deterministic function `template_context(note, passage) -> str` (≤240 chars, same budget as LLM contexts). No LLM, no cache needed (pure function).
- **LLM top-up (gated):** only notes where the template plausibly under-describes — gate = note text length > 2,000 chars OR passage count ≥ 3 (multi-topic threads). LLM contexts keep the existing prompt/cache; when present they replace the template for that passage; otherwise template stands.
- Config: `CONTEXT_ENRICH_MODE` (`"llm"` current, `"template"`, `"hybrid"` proposed default), `CONTEXT_LLM_MIN_CHARS = 2000`, `CONTEXT_LLM_MIN_PASSAGES = 3`.
- Benchmark: framework metric 2 across {llm, template, hybrid} on identical corpora — three scratch builds differing only in this mode (extraction cache shared, so each build only re-embeds). Metric: retrieval hit-rate delta vs current; cost: enrichment calls per cold build (136 today; hybrid target ≤40).
- Decision: hybrid keeps if retrieval within 1 point of full-LLM at ≥60% fewer calls; template-only keeps if within 1 point at zero calls.

## Track C — NLI fact-adjudication pre-screen

New module `taurine_core/indexer/nli_screen.py`; wiring inside `adjudicate_facts` (edge_adjudication.py) BEFORE batching:

- Scorer: bundled `cross-encoder/nli-deberta-v3-base` via `models_local.local_nli_scorer` (`EntailmentScorer.score(premise, hypothesis) -> float`). Premise = the fact's source passage text (the same `draft.text` the judge sees); hypothesis = verbalized fact via per-predicate templates (`works_at` → "{subject} works at {object}", ~18 one-liners in the module).
- Bands (calibrated, not guessed): entailment ≥ `NLI_KEEP_FLOOR` → keep without LLM (adjudicated=True, score = min(0.95, entailment), quote = assertion text); ≤ `NLI_REJECT_CEIL` → **do not auto-reject** — route to the LLM as today (rejection destroys graph edges; the LLM stays the only rejector unless the benchmark proves ≥0.95 auto-reject precision, in which case enable `NLI_AUTO_REJECT=True`). Middle band → LLM as today.
- Exclusions (always LLM): exclusive predicates (works_at etc. — relabel/supersede logic needs the judge), facts from `llm_relation_completion` on their first build, anything needing relabel semantics. NLI cannot relabel.
- Calibration: `quality_bench` metric 3 fits the two thresholds on the 1,335 cached judgments (train/heldout split by fact hash) under the precision targets; thresholds land in config with the fitted values and the heldout numbers in comments.
- History note: per-sentence NLI validation was removed from RETRIEVAL on 2026-07-10 for rejecting honest absence answers. This is a different use — verifying an extracted assertion against its own source text (entailment IS the task) — and it fails toward the LLM, never toward data loss.
- Decision: keep if ≥40% of judge calls eliminated at ≥0.95 keep-precision on heldout; iterate = tighten hypothesis templates once; else scrap.

## Track D — semantic-edge pruning + reasons without build-time LLM

Split `adjudicate_semantic_edges`'s two jobs:

- **Pruning → local classifier.** Features per candidate pair (all already computed or cheap): embedding cosine, shared-entity count and min-IDF of shared entities, same-source flag, title-token Jaccard, passage-length ratio, NLI cross-entailment (both directions, capped to the top ~600 pairs it currently judges). Simple transparent model — logistic regression fit offline on the 557 cached judgments (kept vs dropped), coefficients embedded as constants in `taurine_core/indexer/edge_prune.py` (no sklearn at runtime); benchmark on heldout split.
- **Reasons → deterministic + lazy.** Build-time reason string becomes deterministic: "Both mention {top shared entities}" / "Semantically similar passages from {source A} and {source B}" (template in edge_prune.py). The richer LLM reason moves OUT of the build to an on-demand server endpoint (generate on first edge-click, persist to the existing reason column) — same UX after first click, zero build-time calls. On-demand endpoint reuses the existing judge prompt for a single pair.
- Config: `SEMANTIC_PRUNE_MODE` (`"llm"` current, `"local"` proposed).
- Decision: local mode keeps if drop-agreement ≥0.90 and false-drop ≤0.05 on heldout; else keep LLM pruning but STILL move reasons lazy (that half is safe regardless and removes ~30 calls).

## What stays LLM (explicitly out of scope to remove)

Merge judge (204 judgments, tiny cost, catastrophic failure mode), relation completion for implicit relations (shrinks via Track A's gate change but stays), gray-zone fact adjudication (Track C's middle band), on-demand edge reasons (Track D, user-triggered).

## Execution & verification

- Implementation delegated to codex (gpt-5.6-sol high) subagents, ≤2 concurrent (fleet stability limit), per-track briefs referencing this spec. Waves: (1) Track A + Track B in parallel (disjoint files), (2) Track C + Track D as one agent (both edit edge_adjudication.py), (3) benchmark integration fixes.
- Orchestrator (Claude) verifies every track independently: reruns the track's tests, reruns its benchmark, inspects diffs — codex claims are never trusted unverified.
- Final: full-system benchmark (all accepted tracks on, scratch rebuild vs production baseline): retrieval battery, fact quality, calls, wall-time, memory. Keep/fix/scrap decisions recorded in an outcome appendix appended to this spec.
- Nothing installs to production until the final benchmark passes and Neil sees the report.

## Risks

- Relex is 2 months old: API instability, undocumented relation-span support, possible `gliner` package upgrade with transitive-dependency fallout (torch/transformers pins) — smoke test gates the track.
- DocRED gains may not transfer to iMessage text — that's what the judge-sampled precision metric detects.
- NLI on sarcastic/elliptical chat may under-entail true facts (missed savings, not data loss — fails toward LLM).
- Template contexts may regress retrieval on multi-topic threads — that's the hybrid gate's job.
- Extraction cache invalidation on backend switch = full re-extraction (~10 min, one-time per variant).
