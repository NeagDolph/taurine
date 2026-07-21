# Taurine UI-First Session Architecture Reset

**Date:** 2026-07-20

**Status:** Approved design direction

**Applies to:** Quiet Canvas and Knowledge Atlas implementation branches

## 1. Decision

Replace the increasingly complex resumable-session ownership machinery with a serialized session actor, and reorganize both UI implementation plans around end-to-end vertical slices.

The redesigns still retain the user-visible behavior promised by their existing specifications:

- a meeting remains the same durable object across a short transport interruption;
- a matching client can reconnect during the grace period and recover current bounded live context;
- capture stops accepting audio immediately after Stop or disconnect;
- notes and transcript data already persisted remain authoritative;
- finalization is idempotent and occurs at most once;
- a hidden Taurine window is never raised merely because capture starts;
- both proposed interfaces remain complete redesigns rather than incremental skins.

The reset removes internal requirements that do not contribute to those outcomes. Taurine will not maintain a graph of detached task owners, callback handoff sentinels, cancellation-resistant cleanup chains, or multiple overlapping lifecycle drivers.

## 2. Why the Current Direction Is Being Replaced

The original implementation plans front-loaded unusually exhaustive asynchronous lifecycle hardening. The resulting registry accumulated generations, detached entries, task registries, callback handoffs, fallback cleanup tasks, cancellation shielding, and repair-specific ownership rules. Independent review continued to uncover new interleavings even after broad Core test suites passed.

That work is disproportionate to the product objective. The objective is to produce two fully realized Electron UI alternatives while keeping the backend mostly stable. Session recovery is supporting infrastructure, not the product itself.

This reset therefore optimizes for:

1. a lifecycle model that can be explained on one page;
2. one serial decision-maker for each meeting;
3. bounded and explicit background work;
4. direct tests of user-visible behavior;
5. earlier delivery of complete, inspectable UI journeys.

## 3. Scope

### 3.1 In scope

- Replace the custom resumable-session registry implementation on both branches with the actor described below.
- Retain the existing typed WebSocket session controls and bounded `session_snapshot` contract where already implemented.
- Preserve current durable meeting, notes, transcript, proposal, evidence, and finalization services.
- Rework the two implementation plans into larger vertical slices that end in working user journeys.
- Keep strict automated testing, accessibility, privacy, performance, and final release gates.
- Keep independent review at architectural boundaries and completed vertical slices.

### 3.2 Out of scope

- Distributed session ownership across processes or hosts.
- Persistence or migration of ephemeral live-session state.
- Audio replay after reconnect.
- Recovery of in-memory live intelligence after a Core process restart.
- An unbounded audit of every theoretically possible `asyncio` cancellation point.
- A second general-purpose task scheduler inside Taurine.
- Changes to the approved visual or interaction designs except where plan sequencing changes.

## 4. Serialized Session Actor

### 4.1 Ownership model

`SessionActor` is the only component allowed to read or mutate the live-session table. Public methods acquire one of 256 admission permits, submit typed commands to one private FIFO `asyncio.Queue`, and await a response future. One actor-loop task processes commands serially across all meetings. Public submission applies backpressure rather than dropping lifecycle commands; actor-owned completion messages bypass admission because their count is already bounded by the records table.

Because lifecycle decisions never run concurrently, the implementation does not need a lock-protected web of intermediate owners. A command either completes its state transition or reports an error before the next command begins.

The actor owns:

```python
@dataclass
class LiveSessionRecord:
    meeting_id: str
    client_session_id: str
    state: Literal["active", "suspended", "finalizing"]
    context: SessionContext
    generation: int
    resume_deadline: float | None
    expiry_task: asyncio.Task[None] | None
    finalization_task: asyncio.Task[None] | None
```

There is at most one record per `meeting_id`, one expiry task per suspended record, and one finalization task per terminal record.

### 4.2 Public command surface

The actor exposes only:

- `start(meeting_id, client_session_id, channels, sink)`
- `disconnect(meeting_id, client_session_id)`
- `stop(meeting_id, client_session_id, resume_window_s)`
- `delete(meeting_id)`
- `snapshot(meeting_id, client_session_id)`
- `update_snapshot(meeting_id, client_session_id, update)`
- `shutdown(finalize: bool)`

Internal timeout and finalization completion messages use the same queue:

- `_Expire(meeting_id, generation)`
- `_Finalized(meeting_id, generation, outcome)`

No background task directly mutates the table. An expiry task sleeps and uses `put_nowait` to enqueue `_Expire` before returning. A finalization wrapper catches and records every outcome, uses `put_nowait` to enqueue `_Finalized` in `finally`, and then returns. Because admission is bounded separately from the private queue, completion messages cannot be rejected by public-command backpressure.

The actor starts under application composition before any session endpoint accepts work. If its loop fails unexpectedly, composition marks the session service unavailable, fails every queued or future public command with one safe internal-service error, and closes the remaining contexts during application shutdown. It does not silently fall back to socket-owned lifecycle state.

### 4.3 State transitions

```text
missing --start--> active
active --disconnect--> suspended(60 s)
active --stop(10)--> suspended(10 s)
active --stop(0)--> finalizing
suspended --matching start before deadline--> active
suspended --stop(10)--> suspended with the earlier deadline
suspended --stop(0) / matching expiry--> finalizing
active or suspended --delete--> removed without meeting finalization
finalizing --finalized--> removed
```

A different client token does not take over an active record. For a suspended record, behavior is exact and branch-specific:

- Knowledge Atlas rejects a different token with `session_conflict`. `state_reset` is used only after process restart, when no in-memory record exists.
- Quiet Canvas stops and closes the suspended context, awaits that cooperative cleanup, creates a fresh context for the same durable meeting, and returns `state_reset`. It does not run the old meeting finalizer or erase durable notes/transcript during this transport reset.

In either case, the behavior is handled as one serial command, not a concurrent swap.

### 4.4 Start and resume

For a new meeting session, `start` constructs the live context, binds the sink, stores the record, and returns a `new` snapshot.

For a matching suspended session before its deadline, `start`:

1. cancels the single expiry task;
2. awaits its completion before replying;
3. clears the task field;
4. rebinds the transport sink;
5. recreates only the ASR/channel transport layer;
6. marks the record active;
7. returns the bounded `resumed` snapshot before accepting binary audio.

The actor loop may await the cancelled expiry task because expiry tasks only sleep, enqueue with `put_nowait`, and return; they never await the actor response. When processing `_Expire`, the actor first awaits that now-completing expiry task and clears its field, then validates the captured generation. This avoids a dependency cycle and leaves no timer owner behind.

### 4.5 Disconnect and resumable Stop

Disconnect and `stop(..., resume_window_s=10)` first disable new audio input and flush the current ASR/transcript boundary through existing services. The sink is then unbound and the record becomes suspended.

Only one expiry task is created. Repeated Stop is idempotent and may shorten but never extend the existing deadline. No audio bytes are retained for replay.

If pre-suspension flushing fails, the socket handler still stops accepting new audio and the actor still unbinds the dead transport and enters `suspended`. The actor records the safe flush error in diagnostics and returns a snapshot with the actual suspended state; it never represents a disconnected transport as active. Already-persisted transcript and notes remain authoritative.

### 4.6 Finalization

The actor claims finalization by changing the record to `finalizing` before starting one task. Duplicate stop, expiry, disconnect, shutdown, or completion messages observe that state and do not start another task.

The finalization task receives an immutable snapshot of the context it owns. Its completion message carries only meeting ID, generation, and success/failure. On `_Finalized`, the actor awaits the wrapper—which has already enqueued the message and is returning—before clearing the task and record. A stale completion cannot remove a newer record.

Finalization failure is terminal for the ephemeral record and is reported through existing diagnostics. Durable notes/transcript remain available. Retry policy belongs to the durable finalization service, not to the session actor.

### 4.7 Delete and shutdown

Delete is deliberately simple for records in any state:

1. remove the record from the actor table so new transport work cannot find it;
2. cancel and await its concrete expiry and finalization tasks, if present;
3. stop/close its live context without generating new meeting output;
4. invoke the existing durable meeting deletion flow after all cooperative actor-owned work has stopped.

Finalization services used by the actor must remain cooperatively cancellable and idempotent. The actor does not attempt to support a finalizer that deliberately suppresses task cancellation. The application-level deletion deadline reports failure rather than deleting durable data while a finalizer is still running.

Shutdown stops accepting public commands, then processes every current record serially:

- `finalize=False`: cancel expiry, close context, clear table;
- `finalize=True`: cancel expiry and invoke the existing finalizer once for each non-finalizing record, then await the actor-owned finalization tasks.

Shutdown has one outer deadline owned by application composition. The actor itself does not build cancellation-recovery chains. If the application cancels shutdown, process termination remains the final cleanup boundary.

### 4.8 Cancellation policy

Public callers may be cancelled while waiting, but cancellation of a response future does not cancel or roll back an already-enqueued actor command. The command completes according to its serial state transition, and the actor checks whether the response future is still pending before resolving it.

Only actor-created expiry and finalization tasks are retained. All are cancelled or awaited during record removal and shutdown. No task uses completion callbacks to publish a successor task; lifecycle completion is expressed only as a queue message.

## 5. Data and Transport Contracts

The actor does not alter durable data models. It retains only the bounded live projection already defined by each design:

- transcript sequence or last committed boundary;
- current summary revision;
- current grounding revision and display projection;
- bounded actionables and note suggestions;
- channel state;
- ASR state;
- resume deadline.

The projection contains no audio frames, credentials, raw retrieval traces, or unbounded transcript buffers.

WebSocket handlers remain responsible for schema validation and transport ordering. They must:

1. reject binary frames until a Start/Resume command succeeds;
2. emit `session_snapshot` before ordinary resumed deltas;
3. stop accepting frames immediately after Stop or disconnect;
4. translate actor ownership errors to the existing safe `session_error` codes;
5. avoid owning lifecycle tasks independently of the actor.

## 6. Required Test Matrix

Testing is strict but bounded. The session actor must prove:

1. new Start creates one active record;
2. matching reconnect at 59.9 seconds resumes the same live context;
3. expiry at 60 seconds finalizes once;
4. explicit Stop with 10 seconds resumes at 9.9 seconds and finalizes at 10 seconds;
5. Stop with zero starts one finalization immediately;
6. repeated Stop never extends the deadline or duplicates finalization;
7. stale expiry and stale completion messages cannot affect a newer generation;
8. conflicting token behavior matches the branch contract;
9. audio after Stop is rejected and never buffered;
10. delete cancels expiry and closes without finalization output;
11. shutdown with and without finalization leaves no actor-owned pending tasks;
12. caller cancellation before and after enqueue does not corrupt actor state;
13. finalizer failure removes the ephemeral record safely and preserves durable meeting data;
14. process restart produces the specified state-reset behavior.

Use a fake monotonic clock or injected expiry scheduler for deadline tests. Run the race-sensitive cases repeatedly, but do not add tests for arbitrary injected task implementations, delayed third-party callback delivery, or cancellation-resistant fake tasks that violate the actor's concrete `asyncio.Task` dependency model.

## 7. Worktree Reset Strategy

### 7.1 Knowledge Atlas

The frozen approved commits through Task 4 remain. The uncommitted Task 5 repair is archived at:

`/private/tmp/taurine-atlas-paused-task5-20260720.tar.gz`

Do not continue patching its detached-entry/task-handoff implementation. Replace the Task 5 implementation with the actor and retain useful contract, snapshot materialization, and behavioral tests only where they apply to the simpler design. Task 6 integrates the actor with the WebSocket surface as part of the same vertical slice.

### 7.2 Quiet Canvas

The approved Task 5 commit remains as historical branch state, but the current server-owned suspended-session implementation must not receive additional repair layers. Before later UI session integration, replace its lifecycle ownership with the shared actor design while preserving Quiet Canvas's `state_reset` behavior and already-approved durable note/proposal semantics.

The paused Task 6 desktop-boundary work is archived at:

`/private/tmp/taurine-quiet-paused-task6-20260720.tar.gz`

That desktop work remains useful and resumes after the plans are rewritten. Its command/preload security checks are not discarded.

## 8. UI-First Plan Structure

The rewritten plans must use vertical slices rather than one commit per infrastructure micro-layer. Each slice ends with a runnable, inspectable user journey.

### 8.1 Knowledge Atlas slices

1. **Live-session and desktop foundation:** actor, WebSocket integration, bounded Electron capabilities, native prompt/tray behavior.
2. **Atlas shell:** tokens, typography, workspace shell, command model, deterministic graph/list baseline, startup/readiness.
3. **Meeting journey:** capture preflight, Chronicle, free-edit meeting document, live state, proposal review, scoped Ask.
4. **Inquiry and evidence:** shared evidence inspector, streaming threads, provenance-complete citations, scoped navigation.
5. **Knowledge operations:** Atlas lenses/time/outline/comparison, Sources authorization/sync/purge, Profile, Diagnostics, privacy and utilities.
6. **Cutover and proof:** sole renderer, old UI removal, accessibility, localization, native behavior, privacy, performance, visuals, and release gate.

### 8.2 Quiet Canvas slices

1. **Desktop and recovery foundation:** current Task 6 work, encrypted draft recovery, privacy curtain, bounded gateway/readiness.
2. **Quiet shell:** visual tokens, accessible primitives, typed navigation, Open anything, Desk, meetings collection.
3. **Meeting document:** single-writer notes controller, editorial document, capture actor integration, live context, transcript, proposal review.
4. **Conversation and evidence:** renderer-wide conversation hub, complete evidence semantics, meeting Ask and standalone conversations.
5. **Knowledge and settings:** knowledge documents, comparison, Connections list/visual modes, Sources, personal context, privacy, diagnostics.
6. **Cutover and proof:** composition, sole renderer, old UI removal, E2E, accessibility, platforms, privacy, performance, visuals, and release gate.

## 9. Review and Delivery Policy

Strict TDD remains required for behavior, but the review process changes:

- One RED checkpoint and one GREEN checkpoint per vertical slice.
- Independent review occurs at the frozen slice commit, not after every repair micro-commit.
- A reviewer may require fixes for P1/P2 correctness, security, privacy, data-loss, accessibility, or spec violations.
- Purely theoretical concurrency concerns outside the documented actor model do not block UI progress.
- P3 polish may be recorded for the slice's final visual/accessibility pass unless it affects the primary journey.
- Each branch must remain runnable at the end of every slice.
- Implementers may not silently narrow the approved interface designs to save time.

The final branch-wide gate remains exhaustive. It must include Core and Electron tests, type checking, linting, build and bundle budgets, full Playwright journeys, axe checks, keyboard-only flows, zoom/reflow, reduced motion, privacy assertions, representative screenshots, and measured launch/interaction/render performance.

## 10. Acceptance Criteria for the Reset

The reset is complete when:

- both implementation plans reference this design and use the six vertical slices above;
- the current complex Atlas Task 5 WIP is replaced rather than incrementally repaired;
- session lifecycle mutation is serialized through one actor queue;
- no callback-created cleanup successor or detached task-owner registry remains;
- the required 14-case lifecycle matrix passes;
- both branches resume visible UI delivery immediately after their first foundation slice;
- all behavior and visual requirements from the approved Quiet Canvas and Knowledge Atlas designs remain represented in the rewritten plans;
- final verification remains broad enough to prove each complete redesign, not merely its session infrastructure.

## 11. Non-Goals and Guardrails

This reset is not permission to weaken data safety, evidence semantics, accessibility, privacy, or final verification. It is also not permission to converge the two designs. Quiet Canvas and Knowledge Atlas remain distinct implementations with separate visual systems and interaction models.

The simplification applies specifically to execution architecture and review granularity: fewer lifecycle mechanisms, fewer infrastructure-only task boundaries, more complete user-facing slices, and faster arrival at interfaces that can be exercised and judged.
