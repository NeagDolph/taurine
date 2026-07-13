# Enrichment LLM Call Reduction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut enrichment LLM calls from ~484 cold / ~45 warm to ~300 cold / ~0 warm by caching the relation-completion lane, packing adjudication batches across notes, tightening the completion gate, moving completion to gpt-oss-120b, and deleting the LLM cluster-naming lane.

**Architecture:** All changes live in the Python sidecar (`core/taurine_core/`). The completion cache reuses the existing BuildCache `adjudication_cache` table via `cached_judgments`/`store_judgments` with a `completion:` key prefix — the same pattern the fact/reason/merge judge lanes already use. Adjudication batching keeps per-fact cache keys byte-identical so existing cached judgments stay valid.

**Tech Stack:** Python 3.11+, pytest, sqlite (BuildCache), OpenRouter LLM client (`complete_json`).

**Spec:** `docs/superpowers/specs/2026-07-13-enrichment-call-reduction-design.md` — read it first.

## Global Constraints

- Working dir for all commands: `/Users/neil/taurine/core`. Run tests with `python -m pytest`.
- Every LLM lane is **fail-open**: an LLM error must never fail a build. Preserve this in every change.
- `_FACT_PROMPT_VERSION` stays `"fact-adjudication-v1"` — do NOT bump it (Task 4 must not invalidate existing cached judgments; `_fact_key` output must be byte-identical before/after).
- Completion calls keep `temperature=0.0`.
- No new dependencies. Match surrounding code style (comment density, naming).
- Commit after each task with the message given in the task.

---

### Task 1: Config — completion model swap + dead config removal

**Files:**
- Modify: `taurine_core/config.py` (lines ~31, ~64-66, ~243-245)
- Modify: `taurine_core/indexer/build.py` (`relation_completer_for`, ~line 1726: the `llm.complete_json(config.CHAT_MODEL, ...)` call)

**Interfaces:**
- Produces: `config.COMPLETION_MODEL` (str, = `ADJUDICATION_MODEL`), `config.COMPLETION_NO_RELATION_MIN_PERCENTILE` (float, 0.5 — consumed by Task 2).

- [ ] **Step 1: Verify the dead config is dead**

Run: `grep -rn "ENRICH_MODEL\|ENRICH_BATCH_SIZE\|ENRICH_BATCH_CHAR_BUDGET\|ENRICH_BATCH_MIN_NOTES" --include="*.py" taurine_core/ tests/ | grep -v "config.py\|CONTEXT_ENRICH"`
Expected: no output. (If anything shows up, stop and report — do not delete.)

- [ ] **Step 2: Edit config.py**

Delete these lines (they are legacy of the deleted pre-overhaul enrichment flow):

```python
ENRICH_MODEL = "z-ai/glm-5.2"            # enrichment is offline: quality over latency
ENRICH_BATCH_SIZE = 10                   # max notes per LLM call
ENRICH_BATCH_CHAR_BUDGET = 24_000        # max summed note-text chars per call
ENRICH_BATCH_MIN_NOTES = 3               # below this, per-note calls (batch overhead loses)
```

In the completion block (next to `COMPLETION_TOOL_HOP_CAP` / `COMPLETION_NOTES_PER_CALL` / `COMPLETION_CONCURRENCY`), add:

```python
COMPLETION_MODEL = ADJUDICATION_MODEL    # structured extraction; the judge validates its output
COMPLETION_NO_RELATION_MIN_PERCENTILE = 0.5  # no-relation gate needs median-or-better value
```

(`ADJUDICATION_MODEL` is defined above this block; if not, move the new line below it.)

- [ ] **Step 3: Swap the completion model in build.py**

In `relation_completer_for`, change the `complete_json` first argument:

```python
return await llm.complete_json(
    config.COMPLETION_MODEL,
```

(was `config.CHAT_MODEL`). gpt-oss-120b already gets reasoning effort=low via `LLM_REASONING_BY_MODEL` — no other change needed.

- [ ] **Step 4: Run the affected tests**

Run: `python -m pytest tests/test_relation_completion.py tests/test_index_build.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: relation completion on gpt-oss-120b; drop dead ENRICH_* config"
```

---

### Task 2: Tighten the no-relation completion gate

**Files:**
- Modify: `taurine_core/indexer/relation_completion.py` (`select_completion_notes`, lines 104-122)
- Test: `tests/test_relation_completion.py`

**Interfaces:**
- Consumes: `config.COMPLETION_NO_RELATION_MIN_PERCENTILE` (Task 1).
- Produces: `select_completion_notes` behavior change only; signature unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_relation_completion.py` (reuse the file's existing `CompletionNote` fixture helpers — there is a pattern in `test_completion_selector_includes_value_top_floor_sparse_and_schema_gap`):

```python
def test_no_relation_gate_requires_median_value() -> None:
    # 10 notes, distinct values 1.0 .. 0.1; all have 2 entities and 0 relations
    notes = [
        _note(f"note-{i:02d}", value=1.0 - i * 0.1, entities=2, relations=0)
        for i in range(10)
    ]
    selected = {note.note_id for note in select_completion_notes(notes)}
    # top-15% floor (min 5) admits note-00..04 regardless; the no-relation arm
    # must NOT admit the bottom-half notes beyond that
    assert "note-09" not in selected
    assert "note-08" not in selected
    assert "note-03" in selected  # above median → no-relation arm admits


def test_schema_gap_ignores_value_threshold() -> None:
    notes = [
        _note(f"note-{i:02d}", value=1.0 - i * 0.1, entities=2, relations=0)
        for i in range(10)
    ]
    notes.append(_note("note-gap", value=0.0, entities=0, relations=5, schema_gap=True))
    selected = {note.note_id for note in select_completion_notes(notes)}
    assert "note-gap" in selected
```

Write (or extend) a small `_note(...)` helper constructing a `CompletionNote` with `value_score`, N dummy entities, `active_relation_count`, `schema_gap`, matching how existing tests build notes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_relation_completion.py -q -k "median or ignores_value"`
Expected: FAIL (`note-08`/`note-09` currently selected by the unconditional no-relation arm)

- [ ] **Step 3: Implement**

Replace the no-relation loop in `select_completion_notes`:

```python
def select_completion_notes(notes: Sequence[CompletionNote]) -> tuple[CompletionNote, ...]:
    if not notes:
        return ()
    ordered = sorted(notes, key=lambda row: (-row.value_score, row.note_id))
    value_count = min(len(ordered), max(5, math.ceil(len(ordered) * 0.15)))
    selected = {row.note_id: row for row in ordered[:value_count]}
    # the no-relation arm self-scales: barren bottom-half chatter (≥2 entities,
    # 0 facts) stays out of the LLM call instead of being re-asked every build
    values = sorted(row.value_score for row in ordered)
    pct = min(1.0, max(0.0, float(config.COMPLETION_NO_RELATION_MIN_PERCENTILE)))
    threshold = values[min(len(values) - 1, int(math.floor(pct * (len(values) - 1))))]
    for note in ordered:
        if note.schema_gap or (
            len(note.entities) >= 2
            and note.active_relation_count == 0
            and note.value_score >= threshold
        ):
            selected[note.note_id] = note
    tool_hop_budget = max(0, int(config.COMPLETION_TOOL_HOP_CAP))
    for note in ordered:
        if tool_hop_budget <= 0:
            break
        if note.tool_hop and note.note_id not in selected:
            selected[note.note_id] = note
            tool_hop_budget -= 1
    return tuple(sorted(selected.values(), key=lambda row: (-row.value_score, row.note_id)))
```

- [ ] **Step 4: Run the file's full test set**

Run: `python -m pytest tests/test_relation_completion.py -q`
Expected: PASS. If `test_completion_selector_includes_value_top_floor_sparse_and_schema_gap` fails because its sparse-note fixture now falls below the median, raise that fixture's `value_score` to sit above the median of its candidate set (the test's intent is the sparse arm works, not that it is unconditional) — do not weaken the new tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: no-relation completion gate requires median-or-better note value"
```

---

### Task 3: Content-keyed completion cache

**Files:**
- Modify: `taurine_core/indexer/relation_completion.py` (new: `COMPLETION_SYSTEM_PROMPT`, `completion_note_payload`, `completion_note_key`, `group_completion_rows`)
- Modify: `taurine_core/indexer/build.py` (`relation_completer_for` ~1675-1765; completion stage ~736-750; `BuildDependencies` ~88-108; CLI wiring ~1274-1280)
- Modify: `taurine_core/server.py` (~896-901: dependency wiring)
- Test: `tests/test_relation_completion.py`, `tests/test_index_build.py`

**Interfaces:**
- Consumes: `BuildCache.cached_judgments(keys) -> dict[str, dict]` and `BuildCache.store_judgments(rows: Sequence[tuple[key, kind, dict]])` (existing, `taurine_core/indexer/build_cache.py:293-334`).
- Produces:
  - `COMPLETION_SYSTEM_PROMPT: str` (module constant, moved verbatim from `relation_completer_for`)
  - `completion_note_payload(note: CompletionNote) -> dict` (moved verbatim from the `_note_payload` closure in build.py)
  - `completion_note_key(model_id: str, note: CompletionNote) -> str` (returns `"completion:" + sha256hex`)
  - `group_completion_rows(payload: object, notes: Sequence[CompletionNote]) -> dict[str, dict]` (note_id → `{"facts": [...], "aliases": [...]}`)
  - `relation_completer_for(llm)` now returns callables whose result dict has keys `facts`, `aliases`, **`completed_note_ids`** (list of note ids that were in a batch whose LLM call succeeded)
  - `BuildDependencies.completion_cache: Any | None = None`

- [ ] **Step 1: Move the prompt and payload builder into relation_completion.py**

At module level in `relation_completion.py` (import `hashlib`, `json`, `asdict` as needed):

```python
_COMPLETION_PROMPT_VERSION = "relation-completion-v1"

COMPLETION_SYSTEM_PROMPT = (
    "Return JSON {facts:[...]}. Each fact row MUST contain exactly these "
    "fields: note_id, passage_id, subject_entity_id, object_entity_id, "
    "predicate, assertion_start, assertion_end, confidence (0-1). Offsets "
    "are note-global character positions inside the cited passage, and the "
    "assertion span must contain a listed mention of both entities. Find "
    "only relations directly supported by the supplied passage text; reuse "
    "entity_id and passage_id values exactly; select a registered predicate "
    "when possible. For an unregistered predicate, also provide "
    "predicate_description. Do not infer unsupported relations. Notes "
    "carrying a `focus` field likely assert that a project or organization "
    "uses a tool/technology — extract those `uses` relations when the text "
    "supports them (subject = the project/system, object = the tool)."
)


def completion_note_payload(note: CompletionNote) -> dict:
    # shared by the LLM call and the cache key so they can never drift apart
    return {
        "note_id": note.note_id,
        "observed_at": note.observed_at,
        "entities": [asdict(entity) for entity in note.entities],
        "passages": [asdict(passage) for passage in note.passages],
        **(
            {"focus": "tool/technology usage — check for `uses` relations"}
            if note.tool_hop
            else {}
        ),
    }


def completion_note_key(model_id: str, note: CompletionNote) -> str:
    """Content key: any change to the note's payload, the prompt, the model,
    or the predicate registry busts the cache naturally."""
    identity = json.dumps(
        {
            "version": _COMPLETION_PROMPT_VERSION,
            "model": model_id,
            "prompt": hashlib.sha256(
                COMPLETION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "predicates": [
                [name, spec.description] for name, spec in sorted(PREDICATES.items())
            ],
            "note": completion_note_payload(note),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "completion:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
```

(`PREDICATES` is already imported by this module for `_parse_row`; if not, import from `taurine_core.indexer.facts`.)

In build.py's `relation_completer_for`: delete the local `system_prompt` and `_note_payload`, import `COMPLETION_SYSTEM_PROMPT` and `completion_note_payload` from `taurine_core.indexer.relation_completion`, and use them.

- [ ] **Step 2: Completer reports which notes completed**

In `relation_completer_for`'s `run_batches`, track success per batch. Replace the merge loop:

```python
        async def one(batch: Sequence[CompletionNote]) -> dict[str, Any] | None:
            async with semaphore:
                try:
                    return await llm.complete_json(
                        config.COMPLETION_MODEL,
                        COMPLETION_SYSTEM_PROMPT,
                        json.dumps(
                            {
                                "predicates": predicates,
                                "notes": [
                                    completion_note_payload(note) for note in batch
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        timeout_s=config.SUMMARY_TIMEOUT_S,
                        # deterministic completions keep fact identities
                        # stable across rebuilds so adjudication cache hits
                        temperature=0.0,
                    )
                except Exception as exc:  # noqa: BLE001 — per-batch fail-open
                    _LOG.warning(
                        "relation completion batch failed (%s): %s",
                        type(exc).__name__,
                        exc,
                    )
                    return None

        results = await asyncio.gather(*(one(batch) for batch in batches))
        merged: dict[str, Any] = {"facts": [], "aliases": [], "completed_note_ids": []}
        for batch, result in zip(batches, results):
            if not isinstance(result, dict):
                continue
            # a successful call vouches for every note in its batch — including
            # notes the model returned nothing for (cached as empty, asked once)
            merged["completed_note_ids"].extend(note.note_id for note in batch)
            for key in ("facts", "aliases"):
                rows = result.get(key)
                if isinstance(rows, list):
                    merged[key].extend(rows)
        return merged
```

`parse_completion_response` only reads `facts`/`aliases` keys — it ignores `completed_note_ids`; no change needed there.

- [ ] **Step 3: Add row grouping for storage**

In `relation_completion.py`:

```python
def group_completion_rows(
    payload: object, notes: Sequence[CompletionNote]
) -> dict[str, dict]:
    """Raw model rows grouped by resolved note, for per-note cache storage.

    Fact rows resolve via note_id with the same passage_id fallback the parser
    uses; alias rows attribute to the first supplied note listing the entity.
    Every supplied note gets an entry (empty results are cached too)."""

    grouped: dict[str, dict] = {
        note.note_id: {"facts": [], "aliases": []} for note in notes
    }
    if not isinstance(payload, dict):
        return grouped
    note_ids = set(grouped)
    passage_to_note = {
        passage.passage_id: note.note_id
        for note in notes
        for passage in note.passages
    }
    rows = payload.get("facts")
    for raw in rows if isinstance(rows, list) else ():
        if not isinstance(raw, dict):
            continue
        note_id = str(raw.get("note_id", ""))
        if note_id not in note_ids:
            note_id = passage_to_note.get(str(raw.get("passage_id", "")), "")
        if note_id in note_ids:
            grouped[note_id]["facts"].append(raw)
    alias_rows = payload.get("aliases")
    for raw in alias_rows if isinstance(alias_rows, list) else ():
        if not isinstance(raw, dict):
            continue
        entity_id = str(raw.get("entity_id", ""))
        for note in notes:
            if any(row.entity_id == entity_id for row in note.entities):
                grouped[note.note_id]["aliases"].append(raw)
                break
    return grouped
```

- [ ] **Step 4: Thread the cache through the build stage**

`BuildDependencies` gains (next to `adjudication_cache`):

```python
    # per-note completion cache (BuildCache in production): unchanged notes —
    # including notes that yielded nothing — are never re-asked
    completion_cache: Any | None = None
```

Rewrite the completion call site in the build (currently `~line 736`):

```python
                selected_completion = select_completion_notes(completion_notes)
                if selected_completion:
                    cache = self.dependencies.completion_cache
                    keys = {
                        note.note_id: completion_note_key(
                            config.COMPLETION_MODEL, note
                        )
                        for note in selected_completion
                    }
                    cached_rows: dict[str, dict] = {}
                    if cache is not None:
                        by_key = cache.cached_judgments(sorted(set(keys.values())))
                        cached_rows = {
                            note_id: by_key[key]
                            for note_id, key in keys.items()
                            if key in by_key
                        }
                    pending = tuple(
                        note
                        for note in selected_completion
                        if note.note_id not in cached_rows
                    )
                    fresh: dict[str, Any] = {
                        "facts": [],
                        "aliases": [],
                        "completed_note_ids": [],
                    }
                    try:
                        if pending:
                            raw = self.dependencies.relation_completer(pending)
                            if isinstance(raw, dict):
                                fresh = {
                                    "facts": list(raw.get("facts") or []),
                                    "aliases": list(raw.get("aliases") or []),
                                    "completed_note_ids": list(
                                        raw.get("completed_note_ids") or []
                                    ),
                                }
                        merged_payload = {
                            "facts": [
                                row
                                for rows in cached_rows.values()
                                for row in rows.get("facts", [])
                            ]
                            + fresh["facts"],
                            "aliases": [
                                row
                                for rows in cached_rows.values()
                                for row in rows.get("aliases", [])
                            ]
                            + fresh["aliases"],
                        }
                        completed = parse_completion_response(
                            merged_payload,
                            selected_completion,
                            build_id=build_id,
                        )
                    except Exception as exc:
                        completion_error = f"{type(exc).__name__}: relation completion failed"
                    else:
                        fact_rows.extend(completed.facts)
                        proposals = completed.proposals
                        completion_rejected = completed.rejected
                        completion_aliases = completed.aliases
                        # only notes vouched for by a successful call may store;
                        # failed-batch notes retry next build
                        if cache is not None and fresh["completed_note_ids"]:
                            grouped = group_completion_rows(fresh, pending)
                            completed_ids = {
                                str(note_id)
                                for note_id in fresh["completed_note_ids"]
                            } & set(grouped)
                            cache.store_judgments(
                                [
                                    (keys[note_id], "completion", grouped[note_id])
                                    for note_id in sorted(completed_ids)
                                ]
                            )
```

Add the needed imports (`completion_note_key`, `group_completion_rows`) to the existing `relation_completion` import block in build.py.

- [ ] **Step 5: Wire the cache in server and CLI**

`server.py` (~896): add `completion_cache=build_cache,` next to `adjudication_cache=build_cache,`.
`build.py` CLI `main()` (~1274): if the CLI constructs a BuildCache (check how `adjudication_cache`/`context_cache` are wired there), wire `completion_cache` the same way; if the CLI passes no cache, leave it None.

- [ ] **Step 6: Write the tests**

Unit (in `tests/test_relation_completion.py`):

```python
def test_completion_note_key_stable_and_content_sensitive() -> None:
    note = _note("note-a", value=1.0, entities=2, relations=0)
    assert completion_note_key("m", note) == completion_note_key("m", note)
    assert completion_note_key("m", note) != completion_note_key("m2", note)
    changed = replace(note, tool_hop=True)
    assert completion_note_key("m", note) != completion_note_key("m", changed)


def test_group_completion_rows_resolves_note_id_passage_fallback_and_alias() -> None:
    # note-a has passage p-a and entity e-a; note-b has p-b / e-b
    payload = {
        "facts": [
            {"note_id": "note-a", "passage_id": "p-a"},
            {"note_id": "", "passage_id": "p-b"},   # passage fallback
            {"note_id": "ghost", "passage_id": "ghost"},  # dropped
        ],
        "aliases": [{"entity_id": "e-b", "alias": "B"}],
    }
    grouped = group_completion_rows(payload, [note_a, note_b])
    assert len(grouped["note-a"]["facts"]) == 1
    assert len(grouped["note-b"]["facts"]) == 1
    assert grouped["note-b"]["aliases"] == [{"entity_id": "e-b", "alias": "B"}]
    assert grouped["note-a"]["aliases"] == []
```

Integration (in `tests/test_index_build.py`, following that file's existing build-fixture pattern with a real `BuildCache(tmp_path / "cache.db")`):

```python
def test_completion_cache_skips_repeat_calls_and_caches_empty(tmp_path: Path) -> None:
    calls: list[int] = []

    def completer(notes):
        calls.append(len(notes))
        return {"facts": [], "aliases": [],
                "completed_note_ids": [note.note_id for note in notes]}

    # build twice with the same corpus, same BuildCache, completer wired both times
    ...  # first build: assert calls == [<n>] with n > 0
    ...  # second build: assert calls unchanged (zero new completer invocations)


def test_completion_failed_batch_notes_retry_next_build(tmp_path: Path) -> None:
    # completer returns completed_note_ids=[] (all batches failed) on build 1
    # → build 2 must call the completer again for the same notes
```

Flesh both out with the file's existing helpers for constructing a small corpus that passes the completion gates (≥2 entities, 0 relations, adequate value).

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_relation_completion.py tests/test_index_build.py tests/test_build_cache.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: content-keyed relation-completion cache (empty results cached, failed batches retry)"
```

---

### Task 4: Pack adjudication batches across notes

**Files:**
- Modify: `taurine_core/indexer/edge_adjudication.py` (`_FactBatch` ~152, `_fact_batches` ~334, `_fact_payload` ~404, `_parse_fact_response` ~425, `_FACT_SYSTEM_PROMPT` ~48, the batch-failure log line ~252)
- Test: `tests/test_edge_adjudication.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: internal only. `_fact_key` and `_FACT_PROMPT_VERSION` are UNCHANGED (cache compatibility is a hard requirement).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_edge_adjudication.py` (reuse its existing fact/draft fixture helpers):

```python
def test_fact_batches_pack_small_notes_into_one_call() -> None:
    # 3 notes × 2 facts each, well under ADJUDICATION_FACTS_PER_CALL and the
    # char budget → exactly one batch containing all 6 facts, one section per note
    batches = _fact_batches(facts, draft_by_passage, note_titles)
    assert len(batches) == 1
    assert len(batches[0].facts) == 6
    assert [section.note_id for section in batches[0].sections] == [
        "note-a", "note-b", "note-c",
    ]


def test_fact_batches_still_split_at_fact_cap(monkeypatch) -> None:
    monkeypatch.setattr(config, "ADJUDICATION_FACTS_PER_CALL", 4)
    # same 6 facts → 2 batches, note grouping preserved (a note's facts stay ordered)
    batches = _fact_batches(facts, draft_by_passage, note_titles)
    assert [len(batch.facts) for batch in batches] == [4, 2]


def test_quote_must_ground_in_the_facts_own_note() -> None:
    # judgment quote copied from note-b's text for a note-a fact → malformed
    parsed, malformed = _parse_fact_response(response, batch)
    assert parsed == {}
    assert malformed == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_edge_adjudication.py -q -k "pack or fact_cap or own_note"`
Expected: FAIL (`sections` attribute doesn't exist; per-note batching yields 3 batches)

- [ ] **Step 3: Implement the multi-note batch**

```python
@dataclass(frozen=True)
class _NoteSection:
    note_id: str
    title: str
    text: str


@dataclass(frozen=True)
class _FactBatch:
    """One judge call: one or more notes' relevant passage text plus their facts."""

    sections: tuple[_NoteSection, ...]
    facts: tuple[FactRecord, ...]
    assertions: Mapping[str, str]
```

Rewrite `_fact_batches` to stream facts note-by-note into a shared batch, flushing on either cap. Keep the docstring's spirit and the deterministic ordering:

```python
def _fact_batches(
    facts: Sequence[FactRecord],
    draft_by_passage: Mapping[str, Any],
    note_titles: Mapping[str, str],
) -> tuple[_FactBatch, ...]:
    """Deterministic batches packing multiple notes per call (measured builds
    averaged ~3 facts/call against a 25-fact cap when batches were per-note).

    A note's facts stay contiguous and ordered; the supplied text stays
    bounded at ADJUDICATION_TEXT_MAX_CHARS per call."""

    by_note: dict[str, list[FactRecord]] = {}
    for fact in facts:
        by_note.setdefault(fact.source_note_id, []).append(fact)
    max_chars = int(config.ADJUDICATION_TEXT_MAX_CHARS)
    max_facts = max(1, int(config.ADJUDICATION_FACTS_PER_CALL))
    batches: list[_FactBatch] = []

    current_facts: list[FactRecord] = []
    # note_id → {passage_id: text}, insertion-ordered
    current_notes: dict[str, dict[str, str]] = {}
    current_chars = 0

    def flush() -> None:
        nonlocal current_facts, current_notes, current_chars
        if current_facts:
            sections = tuple(
                _NoteSection(
                    note_id=note_id,
                    title=note_titles.get(note_id, ""),
                    text="\n\n".join(passages.values()),
                )
                for note_id, passages in current_notes.items()
            )
            batches.append(
                _FactBatch(
                    sections=sections,
                    facts=tuple(current_facts),
                    assertions={
                        fact.fact_id: _assertion_text(
                            fact, draft_by_passage[fact.source_passage_id]
                        )
                        for fact in current_facts
                    },
                )
            )
        current_facts, current_notes, current_chars = [], {}, 0

    for note_id in sorted(by_note):
        rows = sorted(
            by_note[note_id],
            key=lambda fact: (
                draft_by_passage[fact.source_passage_id].ordinal,
                fact.assertion_start,
                fact.fact_id,
            ),
        )
        for fact in rows:
            draft = draft_by_passage[fact.source_passage_id]
            passage_text = draft.text[:max_chars]
            seen = draft.passage_id in current_notes.get(note_id, {})
            added = 0 if seen else len(passage_text) + 2
            if current_facts and (
                len(current_facts) >= max_facts
                or current_chars + added > max_chars
            ):
                flush()
                added = len(passage_text) + 2
            current_notes.setdefault(note_id, {})
            if draft.passage_id not in current_notes[note_id]:
                current_notes[note_id][draft.passage_id] = passage_text
                current_chars += added
            current_facts.append(fact)
    flush()
    return tuple(batches)
```

- [ ] **Step 4: Update payload, prompt, grounding, and logging**

`_fact_payload` groups facts under their note's section:

```python
def _fact_payload(batch: _FactBatch) -> dict:
    facts_by_note: dict[str, list[FactRecord]] = {}
    for fact in batch.facts:
        facts_by_note.setdefault(fact.source_note_id, []).append(fact)
    return {
        "predicates": {
            name: spec.description for name, spec in sorted(PREDICATES.items())
        },
        "notes": [
            {
                "note_title": section.title,
                "text": section.text,
                "facts": [
                    {
                        "fact_key": fact.fact_id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "predicate_description": PREDICATES[fact.predicate].description,
                        "object": fact.object,
                        "assertion": batch.assertions.get(fact.fact_id, ""),
                    }
                    for fact in facts_by_note.get(section.note_id, ())
                ],
            }
            for section in batch.sections
        ],
    }
```

`_FACT_SYSTEM_PROMPT`: change the second sentence from "You receive one source text and candidate facts (subject — predicate — object) extracted from it. Judge each fact ONLY against the supplied text, never against world knowledge." to "You receive one or more source notes, each with its own text and the candidate facts (subject — predicate — object) extracted from it. Judge each fact ONLY against its own note's supplied text, never against another note or world knowledge." Everything after is unchanged. **Do not change `_FACT_PROMPT_VERSION`.**

`_parse_fact_response`: ground quotes per note:

```python
    known = {fact.fact_id: fact.source_note_id for fact in batch.facts}
    normalized_by_note = {
        section.note_id: _normalize_ws(section.text) for section in batch.sections
    }
    ...
        if (
            judgment is None
            or judgment.fact_id not in known
            or judgment.fact_id in parsed
            or not judgment.quote
            or _normalize_ws(judgment.quote)
            not in normalized_by_note.get(known[judgment.fact_id], "")
        ):
```

Batch-failure log line (~252): `batch.note_id` no longer exists — log `", ".join(section.note_id for section in batch.sections)`.

- [ ] **Step 5: Verify cache-key stability, then run the full file**

Run: `grep -n "_fact_key\|_FACT_PROMPT_VERSION" taurine_core/indexer/edge_adjudication.py` — confirm neither changed.
Run: `python -m pytest tests/test_edge_adjudication.py tests/test_index_build.py -q`
Expected: PASS. Existing tests that assert one-batch-per-note will now see packed batches — update those assertions to the packed shape (the behavior change is intended); tests asserting cached-judgment reuse must pass UNMODIFIED.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: adjudication batches pack across notes (260→~90 calls, keys unchanged)"
```

---

### Task 5: Remove LLM cluster naming (deterministic fallback names stay)

**Files:**
- Modify: `taurine_core/indexer/cluster_labels.py` (delete `cluster_labeler_for`, `label_clusters`, `_parse_label_row`, the LLM prompt; KEEP `ClusterSummary`, `build_cluster_summaries`, `fallback_label`, `_note_line`)
- Modify: `taurine_core/indexer/build.py` (imports ~32-37; `BuildDependencies.cluster_labeler` ~98; label stage ~988-1010; stats fields ~151-152, ~1013-1022, ~1171-1172; CLI wiring ~1276)
- Modify: `taurine_core/server.py` (import ~40, wiring ~897)
- Modify: `taurine_core/config.py` (delete `CLUSTER_LABEL_BATCH` ~145, `CLUSTER_LABEL_MODEL` ~203, fix the comment at ~146)
- Test: `tests/test_cluster_labels.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `BuildDependencies` no longer has `cluster_labeler`; every `ClusterRecord` gets `label_source="fallback"`. `IDEA_SEARCH_MODEL` and the idea-lens feature are untouched.

- [ ] **Step 1: Delete the lane**

In `cluster_labels.py` remove the LLM pieces listed above. In `build.py`:
- drop `cluster_labeler_for` / `label_clusters` from the import block (keep `build_cluster_summaries`, `fallback_label`, `ClusterSummary`)
- delete `BuildDependencies.cluster_labeler` and its comment
- delete `BuildResult.cluster_labeled_count` / `cluster_label_failed_batches` fields, the stats computation (~1013-1022) and result assignments (~1171-1172), and any log line reporting them
- replace the label stage with:

```python
            cluster_records = []
            for summary in cluster_summaries:
                name, description = fallback_label(summary)
                cluster_records.append(
                    ClusterRecord(
                        cluster_id=summary.cluster_id,
                        size=summary.size,
                        name=name,
                        description=description,
                        label_source="fallback",
                        top_entities=summary.top_entities,
                    )
                )
```

(Keep whatever follows the record construction — persistence — unchanged. `ClusterRecord.label_source` stays in the schema; older DBs contain `"llm"` rows.)

In `server.py`: delete the `cluster_labeler_for` import and the `cluster_labeler=cluster_labeler_for(llm),` line. In build.py `main()`: delete the `cluster_labeler=...` line (~1276). In `config.py`: delete the two knobs and update the pointer comment.

- [ ] **Step 2: Update tests**

In `tests/test_cluster_labels.py`:
- DELETE `test_label_clusters_batches_parses_and_fails_open` and `test_build_survives_labeler_failure_with_fallback_names` (no labeler exists to fail)
- REWRITE `test_build_persists_llm_labeled_clusters_and_membership` as `test_build_persists_fallback_labeled_clusters_and_membership`: same build fixture minus the labeler dependency; assert persisted clusters have non-empty `name` and `label_source == "fallback"`
- KEEP `test_build_cluster_summaries_orders_entities_titles_and_sources`, `test_fallback_label_prefers_entities_then_source`, `test_build_without_labeler_uses_fallback_names` (drop its now-redundant name if you fold it into the rewrite), and all snapshot/endpoint/boot tests

Also sweep: `grep -rn "cluster_labeler\|label_clusters\|CLUSTER_LABEL" --include="*.py" taurine_core/ tests/` — expected: no output except `fallback_label`/`build_cluster_summaries` references.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_cluster_labels.py tests/test_index_build.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: drop LLM cluster naming; deterministic fallback labels everywhere"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire core suite**

Run: `python -m pytest -q`
Expected: all tests pass (baseline before this work: 583+ passed; count will shift with deleted/added tests — zero failures is the requirement).

- [ ] **Step 2: Static sweep for stragglers**

Run: `grep -rn "CHAT_MODEL" taurine_core/indexer/ --include="*.py"`
Expected: no completion-lane usage remains (chat/profile lanes elsewhere are fine).
Run: `grep -rn "ENRICH_MODEL\|CLUSTER_LABEL_MODEL\|cluster_labeler" --include="*.py" taurine_core/ tests/`
Expected: no output.

- [ ] **Step 3: Commit anything outstanding**

```bash
git status --short   # expect clean; commit any test-only stragglers with "test: ..."
```
