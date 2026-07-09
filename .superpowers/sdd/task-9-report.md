# Task 9 Report: Corners Adjudication Gate

## TDD Evidence

- Red run: `cd core && uv run pytest -q tests/test_communities.py tests/test_index_extract.py::test_extract_notes_drops_self_and_tiny_endpoint_relations tests/test_graph_adjudicate.py::test_gliner_relation_promotes_fact_with_provenance_and_skips_queue tests/test_graph_adjudicate.py::test_gate_excludes_non_bridge_low_value_and_keeps_bridge_low_value tests/test_graph_adjudicate.py::test_gate_budget_keeps_highest_ranked_pairs tests/test_graph_adjudicate.py::test_value_threshold_quantile_and_zero_behavior`
  - Result: 7 failures for missing `communities`, missing extraction guards, absent GLiNER promotion/provenance, old pending-pair gate, and missing value-threshold helpers.
- Focused green run: same command.
  - Result: 7 passed.
- Affected-file run: `cd core && uv run pytest -q tests/test_communities.py tests/test_index_extract.py tests/test_graph_adjudicate.py tests/test_store_entities.py tests/test_index_rank.py`
  - Result: 34 passed, 2 deprecation warnings.
- Full required run: `cd core && uv run pytest -q`
  - Result: 302 passed, 2 skipped, 3 deprecation warnings.

## Files Changed

- `core/taurine_core/indexer/communities.py`
- `core/taurine_core/indexer/adjudicate.py`
- `core/taurine_core/indexer/graph.py`
- `core/taurine_core/indexer/extract.py`
- `core/taurine_core/indexer/store.py`
- `core/taurine_core/schemas.py`
- `core/tests/test_communities.py`
- `core/tests/test_graph_adjudicate.py`
- `core/tests/test_index_extract.py`
- `core/tests/test_server_ws.py`

## Self-Review

- Promoted-edge idempotence: GLiNER facts are re-derived from current `relations_between(src, dst)` during graph rebuild and are not written into `edge_judgments`, so relation-row changes are reflected on rebuild. If a relation disappears while note content hashes remain unchanged, an existing valid LLM judgment can still reapply because `pair_sig` intentionally ignores extraction state; this preserves the sacred cache semantics.
- LP determinism: label propagation seeds labels from matrix row indexes, iterates rows by sorted `note_id`, reads only prior-iteration labels, and uses `np.unique` with lowest-label tie selection. There is no set iteration in the update path.
- Budget and `MARGIN_MAX_PER_NOTE`: margin candidates are still capped by `_cap_margin_candidates` before `_pending_pairs`; the bridge/value gate and hard sync budget only rank/truncate that already-capped set. Strong fact candidates remain value-filtered by `_fact_eligible_strong_keys` and share the same hard queue budget.

## Concerns

- The new `fact_provenance` field is included in `/graph` payloads. Existing neighbor/fact retrieval surfaces remain fact-text only.
