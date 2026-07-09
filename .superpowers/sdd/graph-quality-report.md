# Graph Quality Fix Report

## RED/GREEN

- Fix A RED: `tests/test_graph_terms.py` failed because high-df `Neil` and `date_or_deadline` `today` still appeared in shared edges, relation-derived candidates still crossed capped endpoints, and capped-endpoint relations still promoted facts.
- Fix A GREEN: `tests/test_graph_terms.py` passes with df/etype specificity applied to shared terms, relation candidate sweeps, and GLiNER fact promotion.
- Fix B RED: `tests/test_store_entities.py` failed with missing `relations_between_scoped`; `tests/test_graph_terms.py` showed note A/B fact promotion from a relation asserted only in note C.
- Fix B GREEN: scoped relation lookup passes, graph promotion uses it, and relation-derived candidate sweep only connects note pairs when the relation provenance note is one of the two notes.
- Fix C RED: `tests/test_imessage_contacts.py` failed because `default_config()` had no `require_contact` and raw-handle chats were still extracted by default.
- Fix C GREEN: iMessage drops chats with no resolved non-self participant contact by default, keeps them with `require_contact=False`, and keeps chats with a named contact.

## Files Changed

- `core/taurine_core/config.py`
- `core/taurine_core/indexer/graph.py`
- `core/taurine_core/indexer/store.py`
- `core/taurine_core/connectors/imessage.py`
- `core/tests/test_graph_terms.py`
- `core/tests/test_store_entities.py`
- `core/tests/test_imessage_contacts.py`
- `core/tests/test_unit17_connectors_local.py`
- `core/tests/test_graph_adjudicate.py`
- `core/tests/test_index_graph.py`
- `core/tests/test_index_rank.py`
- `core/tests/test_index_reembed.py`

## Verification

- Focused RED runs failed for the expected missing behavior.
- Focused GREEN runs passed for graph, store, and iMessage files.
- Full suite: `cd /Users/neil/taurine/core && uv run pytest -q` -> `322 passed, 2 skipped, 3 warnings`.

## Concerns

- `ENTITY_SHARED_DF_FRACTION_CAP = 0.06` is intentionally strict for tiny synthetic corpuses: any shared term in a two-note fixture exceeds the cap. Existing tests that were not about specificity now set a higher cap locally.
- The old `Index.relations_between` is retained. `rg` showed no production caller remains after graph promotion moved to `relations_between_scoped`; tests still cover the legacy helper.
