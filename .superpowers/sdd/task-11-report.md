# Task 11 Report: Utterance Ablation Replay Harness

## Summary

Implemented `python -m scripts.ablate_utterances --db taurine.db --sessions sessions/ [--limit N] [--json out.json]`.

The harness opens the DB read-only, runs baseline and utterance-masked retrieval in process, masks only the loaded dense embedding matrix, and replays stored final transcript chunks through the shared C2 fusion and `GroundingEngine`.

## TDD Evidence

Red:
- `cd core && uv run pytest -q tests/test_ablate_utterances.py`
- Failed with `ModuleNotFoundError: No module named 'scripts.ablate_utterances'`.

Green:
- `cd core && uv run pytest -q tests/test_ablate_utterances.py`
- `3 passed in 0.10s`.

Regression after attribution refinement:
- `cd core && uv run pytest -q tests/test_ablate_utterances.py tests/test_live_ambient.py`
- Initially exposed an empty-kind fallback bug in fake-index GLiNER dense attribution.
- Fixed root cause and reran: `10 passed in 0.08s`.

Final verification:
- `cd core && uv run pytest -q tests/test_index_store.py tests/test_live_ambient.py tests/test_chat_retrieval.py tests/test_ablate_utterances.py`
- `29 passed, 2 warnings in 2.66s`.
- `cd core && uv run pytest -q`
- `311 passed, 2 skipped, 3 warnings in 12.11s`.
- `git diff --check -- core/scripts/__init__.py core/scripts/ablate_utterances.py core/tests/test_ablate_utterances.py core/taurine_core/indexer/store.py core/taurine_core/live/ambient.py` passed.
- `cd core && uv run python -m scripts.ablate_utterances --help` showed the required CLI shape.

## Files Changed

- `core/scripts/__init__.py`
- `core/scripts/ablate_utterances.py`
- `core/tests/test_ablate_utterances.py`
- `core/taurine_core/indexer/store.py`
- `core/taurine_core/live/ambient.py`

## Implementation Notes

- `Index(read_only=True)` opens the SQLite DB in read-only URI mode and skips schema initialization.
- `Index.mask_embedding_kind("utterance")` filters only `_dense_state` rows and matrix entries in memory. BM25, aliases, graph, and fact state are untouched.
- `Index.dense_search_with_kinds(...)` preserves existing per-note max dense search semantics while returning the winning embedding row kind.
- `run_head_fusion(...)` now carries direct dense and GLiNER-dense row-kind attribution into `HeadFusionResult` and debug fused attribution.
- The harness compares baseline vs masked top-5 grounding sets with Jaccard and computes utterance-only grounding from baseline grounding items whose direct dense row kinds are only `utterance` and whose introducing heads do not include BM25, exact, GLiNER, or fact.
- `--json` writes the same machine-readable report returned by `run_ablation(...)`; the CLI always returns 0 after a successful report run regardless of KEEP/ELIMINATE.

## Self-Review

- Production retrieval reuse: yes. Replay calls the real `run_head_fusion(...)` and `GroundingEngine.on_pass(...)`; it does not copy RRF, thresholds, grounding blend, PPR, or hysteresis logic.
- Window determinism: yes. `build_replay_windows(...)` parses only final `chunk` JSONL events in file order and emits one rolling window per final chunk over the last `config.C2_WINDOW_CHUNKS` finals, matching `LiveSession._window_chunks()` and `_window_text()` without partial state.
- DB mutation: guarded by read-only `Index` construction plus in-memory masking. The fixture test hashes DB bytes before and after a run and checks the stored utterance row remains.

## Concerns

- Replay intentionally has no active partial transcript state or GLiNER span extractor. This matches the brief's no-LLM/no-extra-calls constraint and exercises the LLM-free retrieval path, but it is not a byte-for-byte recreation of live wall-clock throttling plus partial windows.
- `--limit` is implemented as a session-file limit, not a window limit.

## Review Fix Addendum: Caveats and Empty-Both Jaccard

Implemented the two review findings on commit `3b5f364`:

- Added fixed replay-fidelity caveats to the machine-readable report under `caveats` and to `render_markdown(...)` under `## Caveats`.
- Added `empty_both_windows` to the report and markdown summary.
- Excluded windows where baseline and masked grounding sets are both empty from the mean top-5 Jaccard denominator.

TDD evidence:

- Red: `cd core && uv run pytest tests/test_ablate_utterances.py -q` failed with `KeyError: 'caveats'` and `assert 0.5 == 0.0` for the empty-both regression fixture.
- Green: `cd core && uv run pytest tests/test_ablate_utterances.py -q` returned `5 passed in 0.11s`.

Final verification:

- `cd core && uv run pytest -q`
- `313 passed, 2 skipped, 3 warnings in 12.13s`.
