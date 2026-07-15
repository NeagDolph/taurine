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
