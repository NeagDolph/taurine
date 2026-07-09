# Task 10 Report

## Files Changed

- `core/migrate_enrichment.py`
- `core/tests/test_migrate_enrichment.py`
- `.superpowers/sdd/task-10-report.md`

## Implementation

- Added `MigrationReport(gist_rows_deleted, utterance_rows_deleted, notes_backfilled, extraction_failures, judgments_written)`.
- Added `migrate(db_path: Path, *, llm: LLMPort | None, dry_run: bool) -> MigrationReport`.
- Implemented the required five-step order:
  1. Delete `gist` embeddings, and `utterance` embeddings only when `not config.SLIM_UTTERANCES`; reload `Index` matrix after deletion.
  2. Recompute `notes.gist` using imported `store._display_line`.
  3. Batch `extract.extract_notes` over active notes, then write successful extractions through `replace_note_extraction` and `cache_extraction`; failed notes are counted and left untouched.
  4. Rebuild graph, compute quality ranks, compute value scores.
  5. With `--llm` and not dry-run, run one adjudication pass, count new judgments by edge-judgment count delta, then refresh fact embeddings.
- `--dry-run` uses a read-only SQLite connection and does not construct `Index`, so it does not delete rows, write cache/entity/rank data, run extraction, rebuild graph, call LLMs, or initialize missing schema tables. It reports exact embedding counts and active note count; extraction failures are reported as `0` because extraction is intentionally not run.

## TDD Evidence

- Red 1: `uv run pytest -q tests/test_migrate_enrichment.py`
  - Failed with 4 `ModuleNotFoundError: No module named 'migrate_enrichment'`.
- Green 1: `uv run pytest -q tests/test_migrate_enrichment.py`
  - `4 passed, 2 warnings`.
- Red 2: added stricter dry-run schema mutation test.
  - `uv run pytest -q tests/test_migrate_enrichment.py`
  - Failed because `Index()` initialized missing schema tables during dry-run.
- Green 2: implemented read-only dry-run.
  - `uv run pytest -q tests/test_migrate_enrichment.py`
  - `5 passed, 2 warnings`.

## Final Verification

- `uv run pytest -q`
  - `307 passed, 2 skipped, 3 warnings in 12.06s`.
- `uv run python migrate_enrichment.py --help`
  - Exited 0 and showed `--db`, `--dry-run`, and `--llm`.

## Self-Review

- Idempotency: safe to rerun. Embedding deletes become zero-count on later runs; display-line recomputation is deterministic; extraction backfill replaces per-note entity/relation rows and cache on success; failed extractions leave old state untouched; graph/rank/value passes are full rebuilds; adjudication skips cached valid judgments and reports only positive cache count delta.
- Matrix consistency: after step 1 deletes embeddings, `index.load()` refreshes the in-memory dense matrix before step 4. The focused test patches `rebuild_graph` and asserts no `gist` rows remain in `index._embedding_rows` when rebuild starts.
- Concern: `rank.refresh_fact_embeddings` is explicitly called after `adjudicate_graph`, even though current `adjudicate_graph` also refreshes internally. The extra call is idempotent and keeps the migration aligned with the task's explicit step 5 requirement.

## Review Fix Follow-Up: Resumable Backfill and Failure Preservation

### Files Changed

- `core/migrate_enrichment.py`
- `core/tests/test_migrate_enrichment.py`
- `.superpowers/sdd/task-10-report.md`

### Implementation

- `_backfill_extractions()` now skips notes before batching when `index.cached_extraction(note_id, note.content_hash)` returns a non-failed extraction. Skipped notes are treated as already migrated and are not included in `notes_backfilled`.
- Documented `notes_backfilled` semantics on `MigrationReport` and `_backfill_extractions`: real migrations count only freshly extracted, successfully written notes from the current run.
- Seeded beta with prior `note_entities`, `entity_relations`, and an `enrich_cache` failed row, then asserted those beta rows remain byte-for-byte equivalent after another failed extraction.
- Added a cache-hit regression test that pre-seeds alpha with a valid current cache entry and asserts the fake extractor receives only beta.
- Removed the dead `dry_run` parameter and unreachable dry-run branches from `_migrate_index()`; dry runs continue through `_dry_run()` only.

### TDD Evidence

- Red: `uv run pytest tests/test_migrate_enrichment.py -q`
  - Failed because the fake extractor saw `["alpha", "beta"]` instead of only `["beta"]`.
  - The beta failure-preservation assertion was also exercised and passed pre-fix because the existing failed-extraction path already preserved prior rows.
- Green: `uv run pytest tests/test_migrate_enrichment.py -q`
  - `6 passed, 2 warnings`.

### Final Verification

- `uv run pytest tests/test_migrate_enrichment.py -q`
  - `6 passed, 2 warnings in 1.82s`.
- `uv run pytest -q`
  - `308 passed, 2 skipped, 3 warnings in 12.16s`.
