# Task 6 Report: LLM-Free Sync Pipeline and Gist Elimination

Status: DONE

## Summary

Implemented the swap from LLM enrichment to local extraction:

- `Enrichment` now carries heuristic display line, aliases, slim utterances, typed `EntityMention`s, `RelationTriple`s, and `extraction_failed`.
- `sync_source` now runs connector extraction -> hash/cache diff -> local `extract.extract_notes` -> chunk embeddings/upsert -> graph/ranks/value scores -> gated `slim_enrich_batch` -> adjudication -> fact embedding refresh.
- `reenrich_all` now keeps profile-card generation, then refreshes local extraction and gated slim aliases/utterances.
- Old LLM enrichment modules/tests were deleted.
- Graph/rank now use typed extraction terms and mean chunk vectors. Gist vectors are no longer produced by `_embedding_units`.
- Store owns display-line generation and reconstructs enrichment from extraction tables.

## TDD Evidence

Red run after rewriting contract tests:

```text
uv run pytest tests/test_index_sync.py tests/test_index_reenrich.py tests/test_graph_terms.py tests/test_index_store.py tests/test_index_reembed.py tests/test_graph_adjudicate.py tests/test_index_rank.py -q

ImportError while loading conftest '/Users/neil/taurine/core/tests/conftest.py'.
...
pydantic_core._pydantic_core.ValidationError: 6 validation errors for Enrichment
topics
  Field required
entities.0
  Input should be a valid string
...
```

Targeted green run:

```text
uv run pytest tests/test_index_sync.py tests/test_index_reenrich.py tests/test_graph_terms.py tests/test_index_store.py tests/test_index_reembed.py tests/test_graph_adjudicate.py tests/test_index_rank.py -q

47 passed, 2 warnings in 3.39s
```

Full-suite green run:

```text
cd /Users/neil/taurine/core && uv run pytest -q

284 passed, 2 skipped, 3 warnings in 11.32s
```

Warnings are existing dependency deprecations from SWIG/FastAPI testclient; no real GLiNER model is loaded in tests after the shared fake extractor fixture.

## Grep Audit

Command requested:

```bash
grep -rn "gist" core/taurine_core --include="*.py" | grep -v display
```

Output captured after cleanup:

```text
core/taurine_core/connectors/email_auth.py:8:device-code flow against your own Azure app registration (public client,
core/taurine_core/connectors/registry.py:27:class ConnectorRegistry:
core/taurine_core/connectors/__init__.py:4:from .registry import ConnectorRegistry, build_connectors
core/taurine_core/connectors/__init__.py:6:__all__ = ["AppleNotesConnector", "ConnectorRegistry", "build_connectors"]
core/taurine_core/connectors/outlook.py:5:against your own Azure app registration, public client, no secret).
core/taurine_core/server.py:25:from taurine_core.connectors.registry import ConnectorRegistry
core/taurine_core/server.py:85:    publish() is thread-safe — registry callbacks fire from sync worker
core/taurine_core/server.py:135:    connector_registry: Any = None:
core/taurine_core/server.py:165:        if active_runtime.connector_registry is not None:
core/taurine_core/server.py:166:            active_runtime.connector_registry.on_change = lambda: active_runtime.updates.publish(
core/taurine_core/server.py:225:        registry = _connector_registry(active_runtime)
core/taurine_core/server.py:226:        return registry.list_info(active_runtime.index)
core/taurine_core/server.py:231:        registry = _connector_registry(active_runtime)
core/taurine_core/server.py:233:            registry.set_enabled(active_runtime.index, connector_id, True)
core/taurine_core/server.py:242:        registry = _connector_registry(active_runtime)
core/taurine_core/server.py:244:            registry.set_enabled(active_runtime.index, connector_id, False)
core/taurine_core/server.py:253:        registry = _connector_registry(active_runtime)
core/taurine_core/server.py:254:        if connector_id in registry.syncing:
core/taurine_core/server.py:257:            await asyncio.to_thread(registry.purge_data, active_runtime.index, connector_id)
core/taurine_core/server.py:266:        registry = _connector_registry(active_runtime)
core/taurine_core/server.py:270:            connector_id in registry.syncing
core/taurine_core/server.py:275:        registry.syncing.add(connector_id)
core/taurine_core/server.py:287:        registry = _connector_registry(active_runtime)
core/taurine_core/server.py:290:            registry.set_config(active_runtime.index, connector_id, config_payload)
core/taurine_core/server.py:595:    connector_registry = ConnectorRegistry()
core/taurine_core/server.py:596:    connector_registry.reconcile_ledger(index)
core/taurine_core/server.py:629:        sync_once=lambda: connector_registry.sync_all_enabled(index, llm),
core/taurine_core/server.py:630:        sync_all_enabled=lambda: connector_registry.sync_all_enabled(index, llm),
core/taurine_core/server.py:636:        connector_registry=connector_registry,
core/taurine_core/server.py:775:    registry = _connector_registry(runtime)
core/taurine_core/server.py:778:            registry.sync_connector,
core/taurine_core/server.py:790:        registry.syncing.discard(connector_id)
core/taurine_core/server.py:870:    if "connectors" in kinds and runtime.connector_registry is not None:
core/taurine_core/server.py:872:            runtime.connector_registry.list_info, runtime.index
core/taurine_core/server.py:932:def _connector_registry(runtime: Runtime) -> Any:
core/taurine_core/server.py:933:    if runtime.connector_registry is None:
core/taurine_core/server.py:935:    return runtime.connector_registry
core/taurine_core/server.py:939:    registry = _connector_registry(runtime)
core/taurine_core/server.py:940:    for info in registry.list_info(runtime.index):
core/taurine_core/chat/tools.py:88:                    "gist": item.gist,
core/taurine_core/chat/engine.py:650:            f"— {item.gist}"
core/taurine_core/chat/engine.py:710:            gist=str(row.get("gist") or ""),
core/taurine_core/chat/engine.py:727:    note_registry, fact_registry = _registry_maps(retriever)
core/taurine_core/chat/engine.py:731:        if ref.startswith("N") and ref not in item_by_id and ref in note_registry:
core/taurine_core/chat/engine.py:732:            item = _materialize_registry_note(ref, retriever)
core/taurine_core/chat/engine.py:736:            key = fact_registry.get(ref)
core/taurine_core/chat/engine.py:739:            fact = _materialize_registry_fact(ref, key, retriever)
core/taurine_core/chat/engine.py:766:def _registry_maps(retriever: Any) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
core/taurine_core/chat/engine.py:784:def _materialize_registry_note(ref: str, retriever: Any) -> GroundingItem | None:
core/taurine_core/chat/engine.py:806:            str(getattr(note, "gist", "")),
core/taurine_core/chat/engine.py:814:def _materialize_registry_fact(
core/taurine_core/chat/retrieval.py:13:Per-chat N#/F# registries and the accumulator persist via
core/taurine_core/chat/retrieval.py:14:Index.chat_registry / set_chat_registry so citations in old messages keep
core/taurine_core/chat/retrieval.py:33:    """One instance per chat_id; state restored from the chat registry."""
core/taurine_core/chat/retrieval.py:47:        registry = index.chat_registry(chat_id)
core/taurine_core/chat/retrieval.py:48:        if isinstance(registry, dict) and registry:
core/taurine_core/chat/retrieval.py:49:            self._grounding.restore_state(registry)
core/taurine_core/chat/retrieval.py:63:        New notes join the chat's citation registry."""
core/taurine_core/chat/retrieval.py:79:        """Persist registries + accumulator to Index.set_chat_registry."""
core/taurine_core/chat/retrieval.py:80:        self.index.set_chat_registry(self.chat_id, self._grounding.export_state())
core/taurine_core/schemas.py:119:    gist: str
core/taurine_core/schemas.py:173:    gist: str
core/taurine_core/schemas.py:283:    gist: str
core/taurine_core/live/ingest.py:421:                        gist=item.gist,
core/taurine_core/live/summarizer.py:201:            f"— {item.gist}"
core/taurine_core/live/grounding.py:8:- stable short-id registries: note_id → "N#" and (src, dst) → "F#"
core/taurine_core/indexer/store.py:35:EmbeddingKind = Literal["chunk", "gist", "utterance"]
core/taurine_core/indexer/store.py:245:                INSERT INTO notes(note_id, title, folder, modified, created, content_hash, text, gist, source)
core/taurine_core/indexer/store.py:254:                    gist=excluded.gist,
core/taurine_core/indexer/store.py:265:                    stored.gist,
core/taurine_core/indexer/store.py:793:    def chat_registry(self, chat_id: str) -> dict:
core/taurine_core/indexer/store.py:794:        """Per-chat retrieval state: N#/F# registries + seed accumulator."""
core/taurine_core/indexer/store.py:796:            "SELECT registry_json FROM chats WHERE chat_id = ?", (chat_id,)
core/taurine_core/indexer/store.py:798:        return json.loads(row["registry_json"]) if row is not None else {}
core/taurine_core/indexer/store.py:800:    def set_chat_registry(self, chat_id: str, registry: dict) -> None:
core/taurine_core/indexer/store.py:803:                "UPDATE chats SET registry_json = ? WHERE chat_id = ?",
core/taurine_core/indexer/store.py:804:                (json.dumps(registry), chat_id),
core/taurine_core/indexer/store.py:954:            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
core/taurine_core/indexer/store.py:968:                SELECT note_id, title, folder, modified, created, content_hash, text, gist,
core/taurine_core/indexer/store.py:977:            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
core/taurine_core/indexer/store.py:1026:            gist=note.gist,
core/taurine_core/indexer/store.py:1144:                gist=note.gist,
core/taurine_core/indexer/store.py:1284:            gist=note.gist,
core/taurine_core/indexer/store.py:1329:                    gist TEXT NOT NULL,
core/taurine_core/indexer/store.py:1375:                    kind TEXT NOT NULL CHECK(kind IN ('chunk', 'gist', 'utterance')),
core/taurine_core/indexer/store.py:1418:                    registry_json TEXT NOT NULL DEFAULT '{}'
core/taurine_core/indexer/store.py:1606:        if kind not in {"chunk", "gist", "utterance"}:
```

Notes:

- Many lines above are substring false positives from `registry`.
- Meaningful `\bgist\b` hits are schema/API field names kept for payload compatibility or DB column/kind compatibility.
- There are no old LLM gist generation, gist vector preference, or gist embedding-unit production paths remaining.

Meaningful word-boundary audit:

```text
grep -rn "\bgist\b" core/taurine_core --include="*.py"

core/taurine_core/chat/tools.py:88:                    "gist": item.gist,
core/taurine_core/chat/engine.py:650:            f"— {item.gist}"
core/taurine_core/chat/engine.py:710:            gist=str(row.get("gist") or ""),
core/taurine_core/chat/engine.py:803:        gist=_display_line_with_entities(
core/taurine_core/chat/engine.py:806:            str(getattr(note, "gist", "")),
core/taurine_core/schemas.py:34:    gist: str = ""                 # heuristic display line; not an LLM artifact
core/taurine_core/schemas.py:56:    gist: str = ""                 # heuristic display line (fallback: first non-empty line)
core/taurine_core/schemas.py:78:    gist: str = ""                 # heuristic display line for map hover previews
core/taurine_core/schemas.py:119:    gist: str
core/taurine_core/schemas.py:173:    gist: str
core/taurine_core/schemas.py:283:    gist: str
core/taurine_core/live/ingest.py:421:                        gist=item.gist,
core/taurine_core/live/summarizer.py:201:            f"— {item.gist}"
core/taurine_core/live/grounding.py:430:            gist=_display_line_with_entities(self.index, note_id, note.gist),
core/taurine_core/indexer/store.py:35:EmbeddingKind = Literal["chunk", "gist", "utterance"]
core/taurine_core/indexer/store.py:239:        stored = note.model_copy(update={"gist": _display_line(note)})
core/taurine_core/indexer/store.py:245:                INSERT INTO notes(note_id, title, folder, modified, created, content_hash, text, gist, source)
core/taurine_core/indexer/store.py:254:                    gist=excluded.gist,
core/taurine_core/indexer/store.py:265:                    stored.gist,
core/taurine_core/indexer/store.py:954:            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
core/taurine_core/indexer/store.py:968:                SELECT note_id, title, folder, modified, created, content_hash, text, gist,
core/taurine_core/indexer/store.py:977:            SELECT note_id, title, folder, modified, created, content_hash, text, gist,
core/taurine_core/indexer/store.py:1026:            gist=note.gist,
core/taurine_core/indexer/store.py:1144:                gist=note.gist,
core/taurine_core/indexer/store.py:1284:            gist=note.gist,
core/taurine_core/indexer/store.py:1329:                    gist TEXT NOT NULL,
core/taurine_core/indexer/store.py:1375:                    kind TEXT NOT NULL CHECK(kind IN ('chunk', 'gist', 'utterance')),
core/taurine_core/indexer/store.py:1606:        if kind not in {"chunk", "gist", "utterance"}:
```

## Files Changed

Production:

- `core/taurine_core/schemas.py`
- `core/taurine_core/indexer/sync.py`
- `core/taurine_core/indexer/store.py`
- `core/taurine_core/indexer/graph.py`
- `core/taurine_core/indexer/rank.py`
- `core/taurine_core/indexer/reenrich.py`
- `core/taurine_core/indexer/adjudicate.py`
- `core/taurine_core/indexer/__init__.py`
- `core/taurine_core/live/grounding.py`
- `core/taurine_core/live/summarizer.py`
- `core/taurine_core/chat/engine.py`

Deleted:

- `core/taurine_core/indexer/enrich.py`
- `core/taurine_core/indexer/batch_enrich.py`
- `core/tests/test_index_enrich_notes.py`
- `core/tests/test_batch_enrich.py`

Tests updated:

- `core/tests/conftest.py`
- `core/tests/test_index_sync.py`
- `core/tests/test_index_reenrich.py`
- `core/tests/test_graph_terms.py`
- `core/tests/test_index_store.py`
- `core/tests/test_index_reembed.py`
- `core/tests/test_graph_adjudicate.py`
- `core/tests/test_index_rank.py`
- `core/tests/test_chat_retrieval.py`
- `core/tests/test_stress_threading.py`
- `core/tests/test_value_scores.py`

Other touched task-adjacent files:

- `core/taurine_core/config.py` comment only: duplicate guard now documents mean chunk vectors.
- `core/taurine_core/server.py` comment only: re-enrichment comment now says extraction/display lines.

## Self-Review

- Old LLM enrichment path is deleted. Searches for `batch_enrich`, `indexer.enrich`, `enrich_all_batched`, `enrich_note`, and `.topics` return no production hits.
- `_embedding_units` emits chunks and slim utterances only; no gist unit is produced.
- `graph._note_vector` and rank duplicate guard use mean chunk vectors.
- `enrich_cache` now caches `Extraction` JSON keyed by `(note_id, content_hash)`. Failed extraction is not cached, so same-hash failed notes are retried on subsequent sync.
- `reenrich_all` still emits `profile`, `notes`, `finalizing`, and final `idle` statuses.
- LLM surfaces remaining by design: `slim_enrich`, `adjudicate_graph`, `profile.build_user_card`.
- Tests use fake extractor seams by default; no real GLiNER model load remains in the suite.

## Concerns

- The raw requested grep command is noisy because it matches `registry`. The meaningful word-boundary audit is clean for old LLM gist semantics.
- The DB schema still permits legacy `gist` embedding rows so old databases can be read/re-embedded; new sync no longer writes them.

---

## Review Fix Report - 2026-07-09

Commit target: `fix: reembed rank refresh, narrow entity-surface lookup, append-only slim phase, restore html test`

### 1. Reembed rank/fact refresh

Changed:
- `core/taurine_core/indexer/reembed.py` now imports `rank` after `index.rebuild_graph()`/`index.load()` and calls `rank.compute_quality_ranks(index)` plus `rank.refresh_fact_embeddings(index)`.
- `core/tests/test_index_reembed.py` spies on both rank refresh functions and verifies quality ranks are persisted by `reembed_all`.

RED evidence:
- `cd core && uv run pytest -q tests/test_index_reembed.py::test_reembed_all_updates_legacy_vectors_rebuilds_graph_and_reloads ...`
- Failed as expected: `rank_calls == []` instead of `["quality", "facts"]`.

GREEN evidence:
- Same focused selector passed in the combined GREEN run: `4 passed, 2 warnings`.
- Focused touched-area files passed: `45 passed, 2 warnings`.

### 2. Narrow entity-surface lookup for grounding suffixes

Changed:
- `core/taurine_core/indexer/store.py` adds `Index.entity_surfaces_for(note_id: str, limit: int = 3) -> list[str]` using the requested single join query ordered by `ne.conf DESC, e.entity_id ASC`.
- `core/taurine_core/live/grounding.py` and `core/taurine_core/chat/engine.py` use `entity_surfaces_for(..., limit=3)` when available, with the old enrichment fallback retained for test doubles/non-store indexes.
- `core/tests/test_store_entities.py` covers confidence ordering, deterministic entity-id tie break, default limit, explicit limit, and missing-note behavior.

RED evidence:
- `cd core && uv run pytest -q tests/test_store_entities.py::test_entity_surfaces_for_orders_by_confidence_then_entity_id_and_limits ...`
- Failed as expected with `AttributeError: 'Index' object has no attribute 'entity_surfaces_for'`.

GREEN evidence:
- Same focused selector passed in the combined GREEN run: `4 passed, 2 warnings`.
- Focused touched-area files passed: `45 passed, 2 warnings`.

### 3. Append-only slim enrichment

Changed:
- `core/taurine_core/indexer/sync.py` now embeds only slim utterance texts in `_utterance_embedding_units` and calls `index.apply_slim_enrichment(...)` instead of re-upserting the full note.
- `core/taurine_core/indexer/store.py` adds `Index.apply_slim_enrichment(...)`, replacing alias rows, deleting/reinserting only `kind='utterance'` embeddings, refreshing extraction cache JSON, and leaving chunk rows untouched.
- `core/tests/test_index_sync.py` records `embed_texts` batches and asserts each gated note chunk text is embedded exactly once per sync while slim utterance text is still embedded.

RED evidence:
- `cd core && uv run pytest -q tests/test_index_sync.py::test_slim_enrichment_embeds_gated_note_chunks_once_per_sync ...`
- Failed as expected with chunk count `2` instead of `1`.

GREEN evidence:
- Same focused selector passed in the combined GREEN run: `4 passed, 2 warnings`.
- Focused touched-area files passed: `45 passed, 2 warnings`.

### 4. Restored Apple Notes HTML coverage

Changed:
- `core/tests/test_connectors.py` restores `test_extract_notes_converts_html_and_hashes_content`, covering `<div>Hello <b>world</b><br>Next line&nbsp;here</div>` to `Hello world\nNext line here` plus exact SHA-256 content hash and length.
- No production code changed for this item.

Evidence:
- Included in the RED/GREEN selector command and passed throughout: `tests/test_connectors.py::test_extract_notes_converts_html_and_hashes_content`.

### Test Runs

RED:
- `cd /Users/neil/taurine/core && uv run pytest -q tests/test_index_reembed.py::test_reembed_all_updates_legacy_vectors_rebuilds_graph_and_reloads tests/test_store_entities.py::test_entity_surfaces_for_orders_by_confidence_then_entity_id_and_limits tests/test_index_sync.py::test_slim_enrichment_embeds_gated_note_chunks_once_per_sync tests/test_connectors.py::test_extract_notes_converts_html_and_hashes_content`
- Output summary: `3 failed, 1 passed, 2 warnings`.

GREEN focused selectors:
- Same command after fixes.
- Output summary: `4 passed, 2 warnings`.

Focused touched-area files:
- `cd /Users/neil/taurine/core && uv run pytest -q tests/test_index_reembed.py tests/test_store_entities.py tests/test_index_sync.py tests/test_connectors.py tests/test_live_grounding.py tests/test_chat_engine.py`
- Output summary: `45 passed, 2 warnings`.

Full suite:
- `cd /Users/neil/taurine/core && uv run pytest -q`
- Output summary: `287 passed, 2 skipped, 3 warnings in 11.26s`.
