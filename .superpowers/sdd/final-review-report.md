# Final Whole-Branch Review Fix Report

## Finding 1 - Failed-Extraction Retry Guarantee

RED:
- `uv run pytest -q tests/test_index_sync.py::test_failed_extraction_with_llm_is_not_slim_cached_and_retries_same_hash ...`
- Failed because the slim LLM was called for the failed extraction (`llm.slim_calls` contained the note prompt), proving the failed note could be slim-enriched and recached.

GREEN:
- `sync_source` now treats cache hits as valid only when `hit is not None and not hit.failed`.
- `_apply_slim_enrichment` now filters out missing or `extraction_failed` base enrichments before `select_slim_notes`.
- Regression rerun: `3 passed, 2 warnings`.
- Focused file: `uv run pytest -q tests/test_index_sync.py` -> `10 passed, 2 warnings`.

## Finding 2 - Legacy Enrich Cache Rows False-Skip Extraction

RED:
- `uv run pytest -q tests/test_index_store.py::test_cached_extraction_rejects_legacy_llm_payload_extra_keys ...`
- Failed because a legacy `{"gist", "utterances", "aliases", "topics", "entities"}` payload parsed as `Extraction(entities=[], relations=[], failed=False)`.
- `uv run pytest -q tests/test_migrate_enrichment.py::test_backfill_skips_valid_cached_extractions_and_counts_only_fresh_backfills ...`
- Failed because migration extracted only `beta`, skipping the legacy-empty `gamma` note.

GREEN:
- `Index.cached_extraction` now decodes JSON, rejects non-dicts, rejects keys outside `Extraction.model_fields`, then validates the remaining payload.
- New `Extraction()` rows still round-trip through `cache_extraction`.
- Migration now re-extracts the legacy-empty `gamma` note and writes its fresh extraction.
- Regression rerun: `3 passed, 2 warnings`.
- Focused files:
  - `uv run pytest -q tests/test_index_store.py` -> `13 passed, 2 warnings`.
  - `uv run pytest -q tests/test_migrate_enrichment.py` -> `6 passed, 2 warnings`.

## Full Verification

- `uv run pytest -q` -> `315 passed, 2 skipped, 3 warnings`.
