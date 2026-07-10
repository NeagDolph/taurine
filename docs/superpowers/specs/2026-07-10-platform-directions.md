# Taurine Platform Directions — 2026-07-10

Taurine today: live meeting copilot (Deepgram ASR → grounded rolling summary + cited actionables), agentic chat over the knowledge base, 8 connectors (iMessage, Gmail, Outlook, Apple Notes, Obsidian, local folders, Claude/Codex memories), deterministic knowledge-graph index with clustering + map UI, and a profile research agent. The strategic gap: all of this is **reactive** (you talk, you ask) and **session-scoped** (meetings evaporate). The directions below push toward a holistic life-data platform: persistent, proactive, and self-extending.

Every direction below reuses the same spine — `SharedRetrievalEngine` + `ChatRetriever` + `chat/tools.py` + `EvidenceValidator` — which the `ProfileGenerator` already proves is a viable base for new autonomous agents.

## 1. Meeting Memory — sessions as first-class data
Persist every live session (transcript ledger, summary iterations, actionables, grounding trail) into `taurine.db`; add a `sessions` connector so finished meetings are ingested into the knowledge graph on next sync; Sessions history UI (list, detail with recap + transcript + actionables, "ask about this meeting" seeding chat); post-meeting recap generation. Fixes the platform's biggest structural gap: today meetings vanish when the window closes.

## 2. Projects Hub — auto-derived projects with completion tracking
An agent derives a project/subproject tree from clusters, facts, actionables, and email threads; each project gets an evidence-linked page (status narrative, open loops, people involved, recent activity) and an inferred completion level with confidence; board/tree UI with accept/dismiss/merge feedback that persists. Turns the passive graph into a work-management surface without manual data entry.

## 3. People & Personas — a zero-entry personal CRM
Person entities become rich profiles: LLM-generated persona tags (colleague, investor, close friend…), relationship summary, interaction timeline across iMessage/email/meetings, relationship-health signals (frequency trend, reply debt — who's waiting on you and who you're waiting on). People directory UI + map lens coloring nodes by persona. The graph already resolves person entities; this gives them a product surface.

## 4. Calendar & Agenda — time as a data plane
Read the user's calendar (macOS EventKit/`icalBuddy` or ICS); join events to the graph to auto-generate meeting-prep briefs (who, what threads, open actionables with citations); propose calendar events and time blocks from actionables and email commitments (exportable .ics, accept/dismiss); daily agenda view fusing calendar + proposed blocks + prep. Connects the knowledge base to the one place users already plan their day.

## 5. Coach — proactive next-steps agent
A phased research agent (ProfileGenerator pattern) that traverses the whole graph and produces a prioritized, evidence-cited brief: concrete next steps extracted from your actual data (unanswered commitments, stalled projects, follow-ups), plus higher-level opportunity suggestions ("you keep discussing X with Y — consider Z"), plus a weekly review. Accept/snooze/dismiss feedback loop tunes future runs. The "looks at your life and tells you what to work on" direction.

## 6. Agentic Canvas — model-generated custom features
A declarative widget DSL (JSON: stat cards, tables, timelines, lists, markdown blocks — all evidence-cited) rendered by the app; an agent takes a user request ("dashboard for my apartment hunt"), traverses the KB with the standard tools, and emits a widget spec; specs persist as named custom views that re-run/refresh against live data. This is the "agentic UI" direction: the model builds interactive features over your data on demand, safely, because output is data (a spec), not code.

## 7. Daily Briefing — the proactive feed
A generated morning digest: what changed since yesterday (new clusters/facts), reply debt, today's meetings with one-line prep, resurfaced forgotten threads/commitments, one suggested focus. Rendered as a Feed tab of evidence-cited cards with feedback controls, regenerable on demand, pushed over the `/updates` bus. The retention surface — the reason to open Taurine every morning.

## 8. Gap Finder / Idea Studio — creative leverage from graph structure
InfraNodus-style structural analysis of the projection graph: detect weakly-connected cluster pairs, have an LLM adjudicate which gaps are *interesting*, and generate cited "bridge ideas" (what connecting these two areas of your life/work could mean); plus a drafting assistant that writes documents grounded in graph evidence around a chosen gap or topic. Map overlay highlighting gap pairs. Uses the graph's structure — not just its content — as a thinking tool.

## Sequencing note
1 (Meeting Memory) is foundational plumbing others benefit from; 5/7 (Coach, Briefing) are the proactive layer; 2/3/4 are structured product surfaces; 6 is the platform bet; 8 is the differentiator for ideation. All eight are being prototyped in parallel worktrees (Fable orchestrator subagents driving Codex gpt-5.6-sol xhigh implementers) to learn which surfaces earn a place in main.
