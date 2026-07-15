# Graph last-chance investigation (Neil's ablation rankings, 2026-07-15)

## Verdict from Neil (fair-fight ablation, /Users/neil/taurine/ablation-report.md)
- GRAPH wins: Mahith (temporal accuracy), Quillan (slight), roommates (both wrong, graph closer), drugs, graduation date
- DENSE wins: peptides, TV shows (graph caught fewer), Klaviyo coworkers (graph "too reluctant" — evidence present but hedged), hackathon (graph missed the exact document), 2023 projects, dating (dense found Trisha)
- Both perfect: university/degree. Both wrong: Vineet (not in corpus), anime, age (dense said "turning 23" — stale; needs DATES on evidence)

## Neil's directives
1. Investigate EVERY graph loss separately: why not found / hedged / refused. Stage-attributed root cause per question.
2. Lower confidence gate + adjust prompt to allow unconfident answers when no confident ones exist.
3. Add dates to documents provided as context (age failure).
4. Use these 15 questions as a calibration set.
5. Propose fixes, implement, validate via ablation rerun. Last chance before defaulting chat to dense+rerank.

## Known hypotheses (from prior findings)
- Chat floors post-Cohere refit too strict: graph returned 1-4p vs dense 8p on EVERY question; employer-now battery regression same class.
- 24-candidate rerank cap has no reserved slots for fact-head/triple-anchored passages.
- Chat's LOW-CONFIDENCE fallback lane (liberal by design) may not be engaging — check why packs stop at 1-4 accepted instead of filling budget with flagged fallback evidence.
- Evidence rendering lacks timestamps (age question) — episodic template contexts carry month/year but verify they render into the chat evidence block.
- glm hedges under strict citation prompt when evidence is thin (coworkers: mentioned but reluctant).

## Forensic method per lost question
Run graph retrieval with trace; find the passage(s) dense used to answer correctly (ablation.json in scratchpad has dense excerpts); locate them in graph pipeline: recalled? in top-24 rerank set? floored? Which stage dropped them. Script: compare per-stage candidate lists.

## State
- Installed: pilot v3 (158MB), backups: pre-v3-0714, pre-v2-0714, pre-8b-pilot-0714
- Latency fixed: 1.2-4.1s. Battery 15/16 (employer-now fails, same root cause suspected)
- ablation script: scratchpad/ablation.py; results: scratchpad/ablation.json; report: /Users/neil/taurine/ablation-report.md

## Round 2 rankings (Neil, post-starvation-fix)
- Graph much better: peptides, hackathons (dense missed results this round). Slightly better: Klaviyo, drugs, roommates.
- Dense slightly better: Mahith (FLIPPED from graph win), Quillan (FLIPPED), dates, age, graduation. Dense much better: TV shows (persistent).
- Same: projects, university. Anime both wrong.
- Neil: "try other strategies — increase amount of items, allow more traversal, or issue in indexing process. Keep investigating."

## Round-3 hypotheses
1. FLIPS (Mahith/Quillan): low-confidence fill may DILUTE precision answers — consider fill only when accepted < 4, or rank-aware fill budget.
2. TV shows (persistent dense win): broad-aggregation queries — PPR concentrates on few strong notes; dense sweeps breadth. Check: distinct-note diversity of graph candidate pool vs dense top-24; MMR two-per-note cap; head K values into fusion; consider diversity-aware recall (larger dense-head K + per-note caps earlier).
3. Age/dates: the "turn 23" passage — check if graph dense head even recalls it (head K too small?) vs fusion drowning it.
4. Verify indexing: confirm show-mention passages exist with embeddings in graph candidate paths.

## Round-3 diagnostic (2026-07-15, diversity_diag.py)
SMOKING GUN: graph pack has ~ZERO overlap with pure-dense top-24 (shows 0/24, age 0/24, dating 1/24) and half the note diversity (7-10 notes vs 18-21). The graph's final evidence comes almost entirely from PPR/entity/fact paths — dense-head candidates are drowned before the pack. Next discriminating check: dump the graph's dense HEAD candidates pre-fusion for these queries; if dense-top-24 present there → fusion/PPR weights drown them (fix: fusion rebalance or reserved dense slots); if absent → dense head K too small or query-embedding path differs (fix: raise K / verify embedding parity). Then implement + round-3 ablation.
