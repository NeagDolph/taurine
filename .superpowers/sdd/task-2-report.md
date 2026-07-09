# Task 2 Report: Store Entity Tables, Value Columns, Helpers

## Status

DONE

## Files Changed

- `core/taurine_core/indexer/store.py`
- `core/tests/test_store_entities.py`
- `.superpowers/sdd/task-2-report.md`

## TDD Evidence

### RED

Command:

```bash
cd /Users/neil/taurine/core && uv run pytest tests/test_store_entities.py -q
```

Output:

```text
FFFFF                                                                    [100%]
=================================== FAILURES ===================================
_______ test_replace_note_extraction_round_trips_entities_and_relations ________
E       AttributeError: 'Index' object has no attribute 'replace_note_extraction'

________ test_replace_note_extraction_recounts_df_on_replace_and_delete ________
E       AttributeError: 'Index' object has no attribute 'replace_note_extraction'

______ test_relations_between_returns_max_conf_in_both_note_orientations _______
E       AttributeError: 'Index' object has no attribute 'replace_note_extraction'

_______________ test_ingest_salience_and_value_scores_round_trip _______________
E       AttributeError: 'Index' object has no attribute 'set_ingest_salience'

__________ test_entity_schema_migration_is_idempotent_for_existing_db __________
E       AssertionError: assert False
E        +  where False = <built-in method issubset of set object at 0x10cc059a0>({'content_hash', 'created', 'folder', 'gist', 'modified', 'note_id', ...})
E        +    where <built-in method issubset of set object at 0x10cc059a0> = {'ingest_salience', 'value_score'}.issubset

=========================== short test summary info ============================
FAILED tests/test_store_entities.py::test_replace_note_extraction_round_trips_entities_and_relations
FAILED tests/test_store_entities.py::test_replace_note_extraction_recounts_df_on_replace_and_delete
FAILED tests/test_store_entities.py::test_relations_between_returns_max_conf_in_both_note_orientations
FAILED tests/test_store_entities.py::test_ingest_salience_and_value_scores_round_trip
FAILED tests/test_store_entities.py::test_entity_schema_migration_is_idempotent_for_existing_db
5 failed in 0.07s
```

Note: the first sandboxed run could not initialize `/Users/neil/.cache/uv`; I reran the same command with approval so `uv` could use its cache.

### GREEN: Focused

Command:

```bash
cd /Users/neil/taurine/core && uv run pytest tests/test_store_entities.py -q
```

Output:

```text
.....                                                                    [100%]
5 passed in 0.06s
```

### GREEN: Full Suite

Command:

```bash
cd /Users/neil/taurine/core && uv run pytest -q
```

Output:

```text
........................................................................ [ 26%]
.............................................s....s..................... [ 52%]
.....................................................s.................. [ 79%]
........................................................                 [100%]
=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

.venv/lib/python3.12/site-packages/fastapi/testclient.py:1
  /Users/neil/taurine/core/.venv/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
269 passed, 3 skipped, 3 warnings in 16.89s
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Implementation Notes

- Added `entities`, `note_entities`, and `entity_relations` DDL in the existing schema initialization flow.
- Added guarded migrations for `notes.ingest_salience REAL DEFAULT 1.0` and `notes.value_score REAL DEFAULT 0.0`.
- Added `replace_note_extraction`, row inspection helpers, `relations_between`, `set_ingest_salience`, `set_value_scores`, and `value_scores` with the exact required signatures.
- Extended `delete_note` to delete note extraction rows and refresh touched entity `df` values. The repo does not have a method named `remove_note`; connector `purge_data` delegates to `delete_note`, so the purge path is covered there.

## Self-Review

- Exact helper signatures from the brief are present.
- Entity tables and note value columns are additive and idempotent on existing DBs.
- `df` is recomputed for old and new entities during replacement, including entities that drop to zero.
- `df` is recomputed after note deletion, which also covers connector `purge_data`.
- `relations_between` checks both note orientations and returns grouped triples with max confidence.
- Score setters mirror `set_quality_ranks`: reset defaults and update supplied note IDs in one transaction.

## Concerns

None.
