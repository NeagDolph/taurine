# Agentic user summary generation

**Date:** 2026-07-09  
**Status:** approved for implementation

## Context

Taurine previously generated a corpus-wide user information card as part of
full re-enrichment. That card existed primarily to improve per-note LLM
enrichment. Ingestion is now LLM-free by default, so profile generation no
longer belongs in the enrichment pipeline. The stale profile phase leaves the
User summary card empty and does not serve the new runtime use cases.

The current application already has the useful primitives for an agentic
replacement:

- a persisted `user_card` value in the index metadata;
- graph-aware `search_notes`, `read_note`, and `expand_note` chat tools;
- GLM-5.2 streaming tool-call support through OpenRouter;
- an `/updates` websocket for pushed profile state;
- an editable Markdown User summary card;
- chat and live-summary agents that can read the index at prompt-build time.

## Goals

1. Do not generate a user summary during startup, sync, or re-enrichment.
2. Let the user explicitly generate or regenerate the summary from the User
   summary card.
3. Make generation agentic: GLM-5.2 must search, read, and traverse the current
   knowledge graph before synthesizing the profile.
4. Produce a detailed Markdown profile covering identity, interests, projects,
   people and relationships, career, employment, clients, and studies when the
   corpus supports those claims.
5. Show a polished, truthful generation state in the UI using Taurine's current
   gray/yellow visual language.
6. Inject the latest user summary into chat and live meeting-summary prompts as
   background user context.
7. Preserve existing manual write/edit behavior.

## Non-goals

- Automatic regeneration when the graph changes.
- A scheduled profile refresh.
- Persisting the agent's chain-of-thought or exposing raw reasoning.
- Adding a cancel control in the first version.
- Treating the user profile as citable evidence in chat or live summaries.
- Sending the user profile to ASR. The current transcription path has no LLM
  prompt surface; the relevant transcription-time consumer is the live meeting
  summarizer.

## Chosen architecture

Create a dedicated `ProfileGenerator`. It reuses the graph-aware retrieval and
tool execution patterns from chat but has its own prompt, tool budget, lifecycle,
status model, and persistence boundary. It does not create a hidden chat or
write profile research into chat tables.

Rejected alternatives:

- **Hidden chat turn:** expedient but pollutes chat persistence and couples a
  system operation to user conversation semantics.
- **One-shot corpus prompt:** retains the old architecture, is not agentic, and
  scales by dumping note text rather than following relevant paths.

## Profile-agent behavior

### Research loop

The generator runs `z-ai/glm-5.2` with reasoning enabled and the existing
graph-aware tool capabilities:

- `search_notes(query, k)` searches semantic, keyword, fact, and graph-expanded
  retrieval results;
- `read_note(note)` opens source text plus graph facts and neighbors;
- `expand_note(note)` traverses the note's graph neighborhood.

The system prompt gives the agent an explicit research checklist:

1. survey the corpus across identity, interests, current projects, historical
   projects, recurring people, organizations, work, clients, career, and study;
2. search multiple independent aspects rather than anchoring on one cluster;
3. read primary nodes before asserting details;
4. follow useful graph edges to related nodes and people;
5. distinguish current activity from historical material using note age and
   wording;
6. omit unsupported categories instead of guessing;
7. finish with a detailed Markdown profile rather than citations or tool logs.

The profile loop gets a larger tool budget than ordinary chat. On the final
round, tools are disabled and the model is instructed to synthesize the best
supported profile from completed research. Empty or whitespace-only output is
a failure.

### Persistence and overwrite semantics

Generation is single-flight per sidecar. The saved card is not cleared when a
run begins. A successful non-empty result atomically replaces `user_card`; a
failure leaves the existing card unchanged.

Manual `PUT /profile` continues to replace the card directly. Generated and
manual summaries persist across application restarts. No startup, sync, or
re-enrichment path creates or changes the summary.

## Status and API

Add an ephemeral `ProfileGenerationStatus` model with:

- `running`;
- `phase`: `idle`, `surveying`, `interests`, `projects_people`,
  `career_studies`, `synthesizing`, or `failed`;
- `activity`: a concise description of the current search/read/traversal;
- `nodes_read`;
- `tool_calls`;
- `started_at`;
- `elapsed_s`;
- `error`.

The status is held on the runtime because progress is operational state, not
durable user data. Restarting the sidecar returns it to idle.

Endpoints:

- `GET /profile` returns `{card, generation}`;
- `PUT /profile` retains manual replacement behavior and returns the same
  shape;
- `POST /profile/generate` starts the background generation and returns `202`
  with the current profile payload;
- a concurrent generation request returns `409`.

The `/updates` profile event carries both `card` and `generation`. Progress
publishes profile updates at meaningful boundaries and tool completions. The
final update contains the replacement card or retryable error state.

## User interface

The User summary card is rebuilt around four states using existing Taurine
tokens: `--surface-base`, `--surface-raised`, `--surface-card`, the warm gray
text ramp, `--accent: #f5c518`, and existing compact primary/secondary buttons.

### Empty

The card explains what generation does, offers a yellow **Generate** button,
and retains **write it by hand**.

### Populated

The card renders Markdown and shows **Generate** and **Edit** actions. Generated
content is immediately saved and displayed.

### Conditional overwrite confirmation

When content exists, Generate first reveals an inline yellow confirmation:
"Replace your current summary? Generation permanently overwrites the current
content." Actions are **Cancel** and **Overwrite & generate**. No confirmation
is shown for an empty card.

### Generating research rail

The card body switches to the approved research-rail layout. The left rail
shows survey, interests, projects and people, career and studies, and synthesis
phases. The main area shows current activity, nodes read, tool calls, and elapsed
time. Edit and Generate are disabled while a run is active. The rail is driven
by server status rather than fabricated timers.

### Failure

The card shows the concise server error and a **Try again** action. If a prior
summary existed it remains saved, but the card stays in the failed operational
state until retry, edit, or a successful profile update.

## Runtime context injection

### Chat

At every chat turn, the engine reads the latest `index.user_card()`. When
present, it adds a bounded `<user_profile>` block to the system context before
conversation history and the per-turn grounding block.

The chat prompt states that the profile is background for interpreting names,
relationships, interests, and likely referents. It may not be cited as a note
or fact, and grounded answers must still use the retrieved `[N#]` and `[F#]`
evidence.

### Live meeting summarizer

At every live-summary pass, the summarizer reads the latest card from its
existing `Index` reference and adds the same bounded `<user_profile>` block to
the user prompt. The system prompt says this is disambiguation and continuity
context only; meeting claims remain constrained to transcript, grounding notes,
grounded facts, previous summary, and actionables.

Manual edits and successful generation therefore affect the next chat turn or
live-summary pass without restarting sessions.

## Error handling and concurrency

- Only one profile generation task may run at a time.
- Network, provider, parsing, or empty-result failures preserve the saved card.
- Status errors are concise and safe for UI display; full exceptions are logged.
- Websocket subscribers receive progress and terminal state updates.
- Leaving the Data tab does not cancel generation because the task belongs to
  the sidecar runtime.
- Sidecar shutdown cancels the in-memory task through the normal application
  lifespan cleanup.

## Testing

Backend tests cover:

- the multi-round tool loop and final synthesis;
- graph tool execution and progress counters;
- successful overwrite and failure preservation;
- single-flight generation;
- profile endpoint payloads and pushed updates;
- removal of profile generation from full re-enrichment;
- chat prompt injection and absence when the card is empty;
- live-summary prompt injection and grounding-only instructions.

Frontend verification covers:

- empty versus populated Generate behavior;
- conditional overwrite confirmation;
- research-rail progress rendering;
- disabled actions while running;
- success replacement and failure/retry state;
- manual edit/write regression behavior;
- TypeScript typechecking.

