# Task 7 Report: Heterogeneous retrieval graph + note projection

## Summary

Implemented the heterogeneous retrieval graph while keeping the existing public graph projection note-only.

- `graph.rebuild_all` / `rebuild_for` now canonicalize stored extraction entities per `etype` before candidate generation, using `resolve_terms` with lowercase normalization only for `concept`.
- Note-note graph generation now adds relation-derived `entity` candidates by sweeping `relation_rows()` and joining through `note_entity_rows()`, including `related_to` catch-all down-weighting.
- Added `rank.HeteroGraphMatrix.build(index)` with note rows followed by entity rows, prefixed row ids, note/entity ids, contains edges, relation edges, projected note-note edges, and column-stochastic normalization.
- Added heterogeneous PPR entity seeds with note-mass-only return semantics. `NoteGraphMatrix` and `compute_quality_ranks` remain note-projection based.

## TDD Evidence

Red:

- `cd /Users/neil/taurine/core && uv run pytest -q tests/test_graph_hetero.py`
- Result after fixing a test helper error: `4 failed, 1 passed`
- Missing behavior covered: canonical entity rows, relation-derived candidates, `HeteroGraphMatrix`, entity-seeded PPR.

Green:

- `cd /Users/neil/taurine/core && uv run pytest -q tests/test_graph_hetero.py`
- Result: `5 passed`

Regression checks:

- `cd /Users/neil/taurine/core && uv run pytest -q tests/test_graph_hetero.py tests/test_graph_terms.py tests/test_index_rank.py tests/test_store_entities.py tests/test_index_graph.py`
- Result: `33 passed, 2 warnings`

Full suite:

- `cd /Users/neil/taurine/core && uv run pytest -q`
- Result: `292 passed, 2 skipped, 3 warnings`

## Files Changed

- `core/taurine_core/indexer/graph.py`
- `core/taurine_core/indexer/rank.py`
- `core/tests/test_graph_hetero.py`
- `.superpowers/sdd/task-7-report.md`

## Self-Review Notes

- PPR warm-start dimension changes are handled defensively: both note-only and hetero PPR only reuse `warm_start` when its shape exactly matches the current matrix size; otherwise they reset from teleport.
- Hetero column normalization handles isolated nodes by only inverting nonzero column strengths, so zero-degree entity columns stay zero instead of producing NaN or inf.
- `index.graph()` remains note-only because entity nodes only exist inside `HeteroGraphMatrix`; the stored graph payload still comes from `notes` plus note-note `edges`.
- No new dependencies and no LLM paths were introduced.
