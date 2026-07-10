# Graph overhaul implementation review — remaining items

**Date:** 2026-07-10 (remediation completed same day)

**Scope:** correctness review of the implemented knowledge-graph overhaul against `docs/superpowers/specs/2026-07-09-knowledge-graph-retrieval-overhaul-design.md` and the audit `docs/audits/2026-07-09-knowledge-graph-retrieval-review.md`. Five parallel deep reviews (indexing/build, graph layer, shared retrieval engine, live path, chat path) plus independent verification of every critical/high claim.

**Status:** all critical, high, and medium findings — and all in-scope low findings — were remediated on 2026-07-10 in three waves per `docs/superpowers/specs/2026-07-10-graph-overhaul-fixes-spec.md`, verified with the full suite green (583 passed, 2 skipped) and real-wiring integration tests at the seams that originally broke. The complete annotated findings list (C1-C3, H1-H8/H-B2, M1-M15, L1-L34, S1-S8) with per-fix notes is preserved in git history (this file's prior revision). This document now tracks only what remains.

---

## Remediation summary (what was closed)

- **Criticals:** C1 entity-head crash (fixed pre-wave, `f086180`, verified); C2 chat tool registry conflicts (shared payload helpers + `ref_for` guards, real-registry integration tests); C3 live rolling-summary deadlock + trigger cold-start (accepted-evidence ledger keeps once-validated citations citable without loosening validation; transcript-only summaries now trigger).
- **Highs:** acronym entity merging (uniqueness-gated, non-transitive), contains-weight saturation (rescaled with a degeneracy health gate), accumulator decay/cap/prune + topic-shift wiring, real content-hash calibration integrity with conservative-floor fallback, extraction cache keyed on actual schema+threshold inputs, tiered rerank-cap ordering, interrupted-turn provenance (partial persisted as streamed, nothing cited, accumulator restored), relation-completion multi-mention grounding, per-head failure isolation.
- **Mediums:** assertion-span off-by-one, temporal recall/graph/path coherence (+ conflicted-fact presentation), O(n) bucketed entity resolution with embedding stages wired, salience plumbing + extraction gating (§0.2 of the fixes spec — connector salience × content quality, stricter than raw length, search never gated), retrieved/consulted/cited three-set persistence, fact-support refs rendered and citable, evidence-sufficiency nudge, sentence-classifier bypass closed with a labeled fixture, query-time hub activation + persisted hub provenance, quality-rank duplicate guard, hub path penalty.
- **Lows / loose ends:** projection fact-provenance edges, pure forest phase, dead scripts and knobs deleted, straddling-span acceptance, etype-collision determinism, fail-closed legacy-shape confidences, corrupt-cache-row isolation, registry thread safety, ledger growth bound, triple-hit hygiene, MMR tie-break, required `available_ids`, docstrings, per-family predicates + completion aliases + proposal normalization, owner-alias wiring + concept stoplist, semantic floor moved to the calibration artifact, full §9.4 invariants including activation-time graph-hash verification (fresh installs only; `remove_source` republishes are exempt by design), startup temp sweep, real model fingerprints in the build manifest, single-read resilient sync with connector carry-forward, temporal-graph LRU, §23 legacy deletion (~5,400 lines of dead modules/store methods/knobs/tests removed; `SyncReport`→`schemas.py`, priors→`indexer/priors.py`).
- **Product decisions implemented:** live retrieval stays strict (abstention returns nothing; no unvalidated evidence ever enters a summary); chat runs a liberal fallback profile (below-floor-but-near evidence returned as an explicitly LOW-CONFIDENCE pack with facts/paths omitted; citation validation unchanged, so liberal retrieval never becomes liberal assertion).

The spec-level holes identified in the original review (S1-S8: cutover-before-calibration, unspecified rolling-summary availability, trigger/abstention interaction, weight-degeneracy gates, rerank-cap ordering, registry payload identity, untested head isolation, unpriced feasibility) are addressed by the fixes spec's product decisions and the tests added with each fix. The recurring root cause — unit-test fakes diverging from production shapes with no integration test at the seam — is now countered by real-wiring integration tests in retrieval, chat, and live.

---

## Remaining open items

**P1 — Calibration/evaluation loop (spec Unit 0 / §18) has never been run.** `core/evals/graph_retrieval/calibration-v1.json` remains a hand-authored bootstrap (`sample_count: 0`, `held_out: {}` on every curve); the labeled datasets, curve fitting, ablations, and release gates have not been executed. Every floor (dense 0.70/0.62, BM25 0.75, triple recognition 0.82/0.74, evidence 0.75/0.65, fallback = 0.75× acceptance, NLI 0.8) is an unvalidated guess — the abstention/fallback machinery is architecturally sound but empirically untuned. Requires user labeling effort (spec §18.1: extraction, live-retrieval, chat-retrieval, citation, and entity-resolution sets); the harness exists (`evaluation/graph_retrieval.py`). This is the single highest-value remaining task. `Status: open (user-dependent)`

**L22 — No relative-date resolver.** Fact temporal fields are effectively `observed_at = note.modified` only; explicit `valid_from`/`valid_to` extraction from text ("since March", "until Friday") needs its own design per spec §8.1 (resolve relative dates only against reliable source timestamps, keep the raw mention and resolution confidence, never set validity from low-confidence resolutions). `Status: open (design needed)`

**L30 — Calibration-dependent design shortcuts.** Pack confidence via `max()` instead of the spec's fitted logistic combination; post-rerank head agreement structurally satisfiable only by dense+bm25; hard 2-per-note evidence cap (no third-passage allowance for multi-part answers); one query-embedding instruction for all modes. All deliberately deferred: fixing them without labeled data would be tuning blind. Revisit as part of P1. `Status: open (blocked on P1)`

## Accepted trades (no action planned)

- **L4** — `remove_source` mutates the active generation in place, keeping its `build_id` with a stale stored `graph_hash` until the next clean build. Documented; activation-time hash verification therefore applies only to fresh build installs.
- **L20** — `mention_id` hashes note+etype+offsets (spec §4.4 also wanted surface + content hash). Harmless within fresh-per-build databases.
- **L27** — utterance query-expansion index absent from the knowledge DB; ablation-gated, deferred until evidence of lift.
- **L32** — post-rerank abstention consumes MMR-selected scores (order nuance vs §13.2, strictly conservative); live grounding now carries `prior_context` flags, so the remaining nuance is presentational.
- **L34** — broad exception handling in triple recognition/rerank fails closed into recorded degraded states; debug logging added so programming errors are visible in traces.

## Reference

- Fix plan and product decisions: `docs/superpowers/specs/2026-07-10-graph-overhaul-fixes-spec.md`
- Full annotated findings (prior revision of this file): `git log -- docs/audits/2026-07-10-graph-overhaul-implementation-review.md`
- Architecture spec: `docs/superpowers/specs/2026-07-09-knowledge-graph-retrieval-overhaul-design.md`
