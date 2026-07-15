# Dense-Default Routing Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Taurine chat use a dense top-48 plus active-fact overlay by default, with exact dates on evidence and daily iMessage passages, while preserving the graph route as a flag and leaving live retrieval unchanged.

**Architecture:** The indexer will render local-day markers into iMessage notes, split those notes on marker-aligned hard boundaries, and persist a nullable passage date. Chat evidence will carry that date through the store/domain/payload layers and construct request-time system prompts. The shared retriever will select a new early-return `dense_facts` branch for chat/tool modes; the existing graph/PPR pipeline remains the complete fallback branch and the only live branch.

**Tech Stack:** Python 3.12, SQLite/FTS5, NumPy cosine retrieval, Cohere-compatible reranking, pytest.

## Global Constraints

- The authoritative requirements are `docs/superpowers/specs/2026-07-15-dense-routing-architecture.md` and its round-3 context.
- Run tests from `/Users/neil/taurine/core` with `.venv/bin/python -m pytest`.
- Use TDD: add each behavioral test first, observe the expected failure, then implement.
- Run the complete suite before every commit.
- Stage only files changed for the current unit.
- Every commit includes `Co-Authored-By: Codex GPT-5.6 <noreply@openai.com>`.
- Keep `DENSE_UNGATED_CANDIDACY`, head reservations, fill machinery, and all graph/PPR code dormant behind `CHAT_RETRIEVAL_MODE="graph"`.
- Live retrieval must not enter the new dense-facts branch.

---

### Task 1: Daily iMessage chunking and dated passage storage

**Files:**
- Modify: `core/taurine_core/connectors/imessage.py`
- Modify: `core/taurine_core/indexer/passages.py`
- Modify: `core/taurine_core/indexer/knowledge_store.py`
- Modify: `core/taurine_core/indexer/build.py`
- Test: `core/tests/test_imessage_contacts.py`
- Test: `core/tests/test_index_passages.py`
- Test: `core/tests/test_passage_index.py`
- Test: `core/tests/test_index_build.py`

**Interfaces:**
- Produces: `PassageDraft.passage_date: str | None` and `PassageContext.passage_date: str | None`.
- Persists: nullable `passages.passage_date` on fresh disposable indexes.

- [x] Add connector tests proving `— YYYY-MM-DD —` precedes the first message of each local day and remains part of the note text/hash.
- [x] Run the connector tests and confirm they fail because markers are absent.
- [x] Render day markers while reversing the newest-first database rows into chronological transcript order.
- [x] Add passage tests proving markers are hard boundaries, overlap never crosses days, dates are assigned per day, and char offsets remain note-global/source-exact.
- [x] Run the passage tests and confirm they fail because `passage_date` and source-aware splitting are absent.
- [x] Split iMessage token windows independently per marker-delimited day while retaining the existing 220/300/40 token policy inside each day.
- [x] Add store/build tests proving the nullable column round-trips, non-iMessage dates remain null, and the passage-build fingerprint changes.
- [x] Run the store/build tests and confirm the missing column/version failures.
- [x] Persist `passage_date`, bump the passage algorithm version to `passages-v2`, and include it in the build config manifest so identical source snapshots receive a new build identity.
- [x] Run the focused unit-1 tests, then the full suite.
- [x] Commit only unit-1 files with the required trailer.

### Task 2: Dates in every evidence prompt

**Files:**
- Modify: `core/taurine_core/retrieval/types.py`
- Modify: `core/taurine_core/retrieval/evidence.py`
- Modify: `core/taurine_core/chat/retrieval.py`
- Modify: `core/taurine_core/schemas.py`
- Modify: `core/taurine_core/chat/engine.py`
- Modify: `core/taurine_core/live/summarizer.py`
- Test: `core/tests/test_chat_engine.py`
- Test: `core/tests/test_chat_retrieval.py`
- Test: `core/tests/test_live_summarizer.py`

**Interfaces:**
- Consumes: `PassageContext.passage_date` from Task 1.
- Produces: `EvidencePassage.passage_date` and `EvidencePassagePayload.passage_date`.
- Produces: a first-line `Current date and time: YYYY-MM-DD HH:MM ZZZ.` string generated for every chat request and every live-summary request.

- [x] Add tests for exact `YYYY-MM-DD` evidence rendering: passage date first, otherwise modified, otherwise created.
- [x] Run them and confirm current month-only rendering fails.
- [x] Thread `passage_date` into every evidence-passage construction/payload and use one shared date formatter in initial chat evidence, live-summary evidence, and chat tool JSON payloads.
- [x] Add a two-turn chat test with `time.time()` advanced between prompt builds and assert distinct first lines.
- [x] Add a live-summary test asserting the same request-time line is first in both primary and fallback system prompts.
- [x] Run the clock tests and confirm the static prompts fail.
- [x] Construct chat and summary system strings per request using the local timezone and its abbreviation.
- [x] Run focused unit-2 tests, unchanged live tests, then the full suite.
- [ ] Commit only unit-2 files with the required trailer.

### Task 3: Dense-facts default chat routing

**Files:**
- Modify: `core/taurine_core/config.py`
- Modify: `core/taurine_core/retrieval/rerank.py`
- Modify: `core/taurine_core/retrieval/evidence.py`
- Test: `core/tests/test_retrieval_rerank.py`
- Test: `core/tests/test_evidence_pack.py`
- Test: `core/tests/test_retrieval_runtime.py`
- Test: `core/tests/test_graph_retrieval_integration.py`
- Test: `core/tests/test_chat_retrieval.py`

**Interfaces:**
- Adds: `CHAT_RETRIEVAL_MODE="dense_facts"`, `DENSE_SWEEP_K=48`, `CHAT_PACK_BUDGET=12`, and `CHAT_FACT_OVERLAY_MAX=12`.
- Adds: an optional reranker candidate-cap override used only by dense-facts chat/tool retrieval.
- Preserves: `SharedRetrievalEngine.retrieve(...) -> EvidencePack` and existing registry/citation contracts.

- [ ] Add config/routing tests proving chat and tool use dense-facts by default, `graph` preserves the old route, and live never enters dense-facts.
- [ ] Run them and confirm the mode/config branch is absent.
- [ ] Add retrieval tests proving dense top-48 candidacy ignores the dense acceptance curve, BM25 candidates join before one bounded rerank, top 12 are returned, and only an empty union abstains.
- [ ] Run them and confirm the graph pipeline gates/abstains.
- [ ] Add fact-overlay tests proving exact/entity aliases select only active touching facts, grouped variants are ordered by adjudication score then observed recency, capped at 12, registered as `[F#]`, and backed by readable source refs.
- [ ] Run them and confirm no SQL-style overlay exists.
- [ ] Implement an early dense-facts branch after dense/BM25/entity resolution and before triple recognition/PPR; retain rerank-floor labels on passage rows without invoking graph low-confidence fill or graph abstention.
- [ ] Assemble the dense passage pack and active fact overlay with existing registry payload helpers and citation validation identities.
- [ ] Run focused unit-3 tests, all existing live-mode tests, then the full suite.
- [ ] Commit only unit-3 files with the required trailer.

### Task 4: Final verification and handoff

**Files:**
- Inspect only: all committed diffs and repository status.

- [ ] Re-read both authoritative specs and match every requirement to code/tests.
- [ ] Run `.venv/bin/python -m pytest` fresh and record the pass/skip count.
- [ ] Inspect `git status`, `git diff HEAD~3..HEAD`, and the three commit trailers; confirm unrelated pre-existing changes remain unstaged/uncommitted.
- [ ] Report commit hashes, suite count, and any deviations with reasons.
