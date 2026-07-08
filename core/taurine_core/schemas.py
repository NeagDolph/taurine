"""Pydantic models: ws protocol, REST payloads, and cross-unit data shapes."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


# ── transcript ─────────────────────────────────────────────────────────
class Chunk(BaseModel):
    channel: int
    text: str
    ts: float                      # sidecar wall-clock at finalization
    final: bool = True
    seq: int = 0                   # monotonic per session, assigned at ingest


# ── index shapes ───────────────────────────────────────────────────────
class Enrichment(BaseModel):
    gist: str                      # <=120 chars
    utterances: list[str]          # 4 synthetic spoken sentences
    aliases: list[str]             # <=6
    topics: list[str]              # 2-4, lowercase normalized
    entities: list[str]


class NoteRecord(BaseModel):
    note_id: str
    title: str
    folder: str = ""
    modified: float                # recency anchor for age decay
    created: float = 0.0           # creation ts (0 → store falls back to modified)
    content_hash: str
    text: str                      # plain text body
    gist: str = ""                 # filled after enrichment (fallback: first line)
    source: str = "apple_notes"    # connector_id that produced this document
    quality_rank: float = 0.0      # log(pagerank/median), clipped ±QUALITY_CLIP; set at rebuild


EdgeType = Literal["semantic", "topic", "entity", "mixed"]


class GraphEdge(BaseModel):
    src: str
    dst: str                       # src < dst (undirected, canonical order)
    type: EdgeType
    weight: float                  # combined: 1 - prod(1 - w_i)
    shared: list[str] = []         # canonical topics/entities behind the edge
    fact: str = ""                 # LLM-written one-sentence relationship (adjudicated edges)


class GraphNode(BaseModel):
    note_id: str
    title: str
    degree: int
    source: str = "apple_notes"


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# ── retrieval ──────────────────────────────────────────────────────────
class Hit(BaseModel):
    note_id: str
    rank: int                      # 1-based, post-grounding
    score: float


class Expansion(BaseModel):
    from_note_id: str              # the seed it was reached from ("" for PPR arrivals)
    note_id: str
    hop: int                       # 1 for 1-hop expansion; 0 for PPR arrivals
    score: float
    via: Literal["graph", "ppr"] = "graph"


class RetrievalResult(BaseModel):
    heard: str                     # the window text
    hits: list[Hit]
    expansions: list[Expansion]
    extracted: list[str] = []      # GLiNER spans that served as query heads


# ── grounding / meeting summary (spec 2026-07-08) ──────────────────────
class GroundingWhy(BaseModel):
    heads: list[str] = []          # heads that surfaced the note: dense|gliner|bm25|exact|fact
    ppr_mass: float = 0.0          # PPR probability mass this pass
    exact_alias: str = ""          # alias text when C1 matched


class GroundingItem(BaseModel):
    note_id: str
    short_id: str                  # stable per-session citation id ("N3"), never reused
    title: str
    gist: str
    source: str = "apple_notes"
    age_days: float = 0.0          # now - modified, in days
    score: float = 0.0             # final blended score (rel × recency × source × quality)
    why: GroundingWhy = GroundingWhy()


class GroundedFact(BaseModel):
    fact_id: str                   # stable per-session citation id ("F7"), never reused
    src: str                       # edge endpoints (src < dst)
    dst: str
    text: str                      # the adjudicated one-sentence edge fact


class SummaryHighlight(BaseModel):
    fact_id: str
    why: str


class Actionable(BaseModel):
    id: str
    title: str
    quote: str                     # verbatim transcript span (fuzzy-validated ≥ ACTIONABLE_QUOTE_MIN_RATIO)
    assignee: str = ""
    due: str = ""
    grounding: list[str] = []      # GroundingItem.short_id / GroundedFact.fact_id references
    why_new: str = ""              # model's rationale for surfacing this as new
    confidence: float = 0.0
    status: Literal["open", "confirmed", "dismissed"] = "open"


class MeetingSummary(BaseModel):
    rev: int = 0
    summary_md: str = ""           # markdown with [N#]/[F#] citation chips
    highlights: list[SummaryHighlight] = []
    actionables: list[Actionable] = []
    grounding: list[GroundingItem] = []
    facts: list[GroundedFact] = []
    updated_at: float = 0.0


# ── ws events (server → client) ────────────────────────────────────────
class TranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    channel: int
    text: str
    final: bool
    ts: float
    seq: int = 0                   # 0 for interims (only finals get seqs)


class AmbientItem(BaseModel):
    note_id: str
    title: str
    gist: str
    score: float
    short_id: str = ""             # grounding citation id when in the grounding set
    source: str = "apple_notes"
    age_days: float = 0.0
    why: Optional[GroundingWhy] = None


class AmbientEvent(BaseModel):
    type: Literal["ambient"] = "ambient"
    items: list[AmbientItem]


class SummaryEvent(BaseModel):
    type: Literal["summary"] = "summary"
    rev: int
    summary_md: str
    highlights: list[SummaryHighlight] = []
    actionables: list[Actionable] = []
    grounding: list[GroundingItem] = []
    facts: list[GroundedFact] = []
    updated_at: float = 0.0

    @classmethod
    def from_summary(cls, summary: MeetingSummary) -> "SummaryEvent":
        return cls(
            rev=summary.rev,
            summary_md=summary.summary_md,
            highlights=summary.highlights,
            actionables=summary.actionables,
            grounding=summary.grounding,
            facts=summary.facts,
            updated_at=summary.updated_at,
        )


class ActionableEvent(BaseModel):
    type: Literal["actionable"] = "actionable"
    op: Literal["new", "update", "dismiss"]
    item: Actionable


class RetrievalEvent(BaseModel):
    type: Literal["retrieval"] = "retrieval"
    heard: str
    hits: list[Hit]
    expansions: list[Expansion]
    retrieval_id: str = ""         # equals the search TriggerEvent's trigger_id
    seqs: list[int] = []           # transcript chunks forming the window
    extracted: list[str] = []      # GLiNER spans that served as query heads


# ── trigger provenance (debug UI: spans → notes they brought about) ────
class TriggerMatch(BaseModel):
    note_id: str
    title: str
    rank: Optional[int] = None     # post-grounding rank for search triggers
    score: float = 0.0
    via_note_id: Optional[str] = None
    hop: Optional[int] = None


class TriggerEvent(BaseModel):
    type: Literal["trigger"] = "trigger"
    trigger_id: str
    kind: Literal["exact", "search"]
    seqs: list[int]                # transcript chunks involved
    span: str                      # matched alias (exact) / window text (search)
    matches: list[TriggerMatch]    # notes this trigger brought about
    action: Literal["grounded", "none"]


# ── connectors ─────────────────────────────────────────────────────────
class ConnectorInfo(BaseModel):
    connector_id: str
    label: str
    kind: Literal["local", "oauth"] = "local"
    enabled: bool
    available: bool
    detail: str = ""               # remediation hint when unavailable
    docs: int = 0
    last_sync: Optional[float] = None
    syncing: bool = False
    sync_phase: str = ""           # extracting | enriching | indexing | finalizing
    sync_done: int = 0
    sync_total: int = 0
    config: dict = {}


class ConnectorSyncReport(BaseModel):
    connector_id: str
    added: int = 0
    changed: int = 0
    removed: int = 0
    enrich_failures: int = 0
    error: str = ""


class NeighborRef(BaseModel):
    note_id: str
    title: str
    source: str = "apple_notes"
    weight: float = 0.0
    fact: str = ""                 # edge fact when the connection was adjudicated


class NotePreview(BaseModel):
    note_id: str
    title: str
    source: str
    gist: str
    text: str
    neighbors: list[NeighborRef] = []


class ReenrichStatus(BaseModel):
    running: bool
    phase: Literal["idle", "profile", "notes", "finalizing"] = "idle"
    done: int = 0
    total: int = 0


class IndexStatus(BaseModel):
    notes: int
    last_sync: Optional[float] = None
    reenrich: Optional[ReenrichStatus] = None


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    asr: Literal["disconnected", "connecting", "connected", "error"]
    index: IndexStatus
    detail: str = ""


# ── client → server control messages ───────────────────────────────────
class StartMsg(BaseModel):
    type: Literal["start"]
    channels: list[dict]           # [{"id": 0, "label": "mic"}]


class StopMsg(BaseModel):
    type: Literal["stop"]


class OpenNoteMsg(BaseModel):
    type: Literal["open_note"]
    note_id: str


class ActionableStatusMsg(BaseModel):
    type: Literal["actionable_status"]
    id: str
    status: Literal["confirmed", "dismissed"]
