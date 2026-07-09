## Task 3 Report: GLiNER2 ingestion adapter

STATUS: DONE_WITH_CONCERNS

### GLiNER2 API findings

- Installed package: `gliner2` 1.3.2 from `core/.venv/lib/python3.12/site-packages/gliner2`.
- `GLiNER2` exposes the expected schema/extraction API:
  - `create_schema(self) -> Schema`
  - `extract(self, text, schema, threshold=0.5, format_results=True, include_confidence=False, include_spans=False, max_len=None)`
  - `batch_extract(self, texts, schemas, batch_size=8, threshold=0.5, num_workers=0, format_results=True, include_confidence=False, include_spans=False, max_len=None)`
  - `extract_relations` / `batch_extract_relations` also exist, but this adapter uses one combined schema via `extract` / `batch_extract`.
- `Schema.entities({...})` accepts the configured entity-description dict. `Schema.relations([...])` accepts relation keys; relation descriptions are not materially represented in the built schema in this installed version.
- No separate long-document/chunking API was present. `max_len` is documented in source as truncation: tokens beyond the limit are dropped. The adapter therefore implements local token-window chunking with `GLINER_CHUNK_SIZE` / `GLINER_CHUNK_OVERLAP` and merges chunk outputs.
- Formatted relation output is under `relation_extraction`, keyed by relation type. With `include_confidence=True`, relation instances contain `head` and `tail` span dicts with confidence; there is no relation-level confidence in 1.3.2. The adapter uses an explicit item confidence if present, otherwise `min(head_conf, tail_conf)`.

### TDD evidence

RED:

```text
$ cd /Users/neil/taurine/core && uv run pytest tests/test_index_extract.py -q
ERROR tests/test_index_extract.py
ImportError: cannot import name 'extract' from 'taurine_core.indexer'
1 error in 0.05s
```

GREEN targeted:

```text
$ cd /Users/neil/taurine/core && uv run pytest tests/test_index_extract.py -q
.....                                                                    [100%]
5 passed in 0.01s
```

Full suite:

```text
$ cd /Users/neil/taurine/core && uv run pytest -q
274 passed, 3 skipped, 3 warnings in 16.47s
```

### Files changed

- `core/taurine_core/schemas.py`
  - Added `EntityMention`, `RelationTriple`, and `Extraction`.
  - Added `EntityMention.mentions` because the task behavior explicitly requires dedupe to retain mention count.
- `core/taurine_core/indexer/extract.py`
  - Added `_extractor_model()` seam delegating to `live.spans.shared_model()`.
  - Added `extract_notes(notes: list[NoteRecord]) -> dict[str, Extraction]`.
  - Builds one combined GLiNER2 schema from `INGEST_ENTITY_TYPES` and `INGEST_RELATION_TYPES`.
  - Batches short notes by `GLINER_INGEST_BATCH`.
  - Chunks long notes locally and merges chunk results.
  - Filters entity/relation thresholds, filters relation endpoints against surviving entities, dedupes entities and relations, and marks per-note failures with `Extraction(failed=True)`.
- `core/tests/test_index_extract.py`
  - Added fake-model tests for thresholds, relation endpoint filtering, dedupe max confidence plus mention count, per-note failure fallback after batch failure, configured batch splitting, and long-note chunk merging.
- `.superpowers/sdd/task-3-report.md`
  - This report.

### Concerns

- `EntityMention.mentions` is not in the brief's class sketch, but the same brief explicitly requires keeping mention count. I added it with a default of `1`.
- GLiNER2 1.3.2 has no separate long-document method; local chunking is a best-effort adapter layer until Task 12 validates behavior against the real model.
- Relation confidence is derived from endpoint confidences because the installed formatted relation output does not include a relation-level score.
- Tests intentionally do not load the real model; live model compatibility remains for Task 12.
