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
    modified: float
    content_hash: str
    text: str                      # plain text body
    gist: str = ""                 # filled after enrichment (fallback: first line)


EdgeType = Literal["semantic", "topic", "entity", "mixed"]


class GraphEdge(BaseModel):
    src: str
    dst: str                       # src < dst (undirected, canonical order)
    type: EdgeType
    weight: float                  # combined: 1 - prod(1 - w_i)
    shared: list[str] = []         # topics/entities behind the edge


class GraphNode(BaseModel):
    note_id: str
    title: str
    degree: int


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# ── retrieval ──────────────────────────────────────────────────────────
class Hit(BaseModel):
    note_id: str
    rank: int                      # 1-based, post-rerank
    score: float


class Expansion(BaseModel):
    from_note_id: str              # the seed it was reached from
    note_id: str
    hop: int                       # 1 at MVP
    score: float


class RetrievalResult(BaseModel):
    heard: str                     # the window text
    hits: list[Hit]
    expansions: list[Expansion]


# ── slate / cards ──────────────────────────────────────────────────────
Source = Literal["exact", "search"]


class SlateItem(BaseModel):
    note_id: str
    title: str
    gist: str
    score: float
    source: Source
    heard: str
    views: int = 0
    decay: int
    added_rev: int
    via_note_id: Optional[str] = None   # set when reached via graph expansion
    hop: Optional[int] = None


class Card(BaseModel):
    card_id: str
    note_id: str
    title: str
    summary: str                   # gist (instant pin) or presenter's why
    provenance: Literal["exact", "search", "linked"]
    heard: str


class Pick(BaseModel):
    note_id: str
    why: str                       # <=WHY_MAX_CHARS, validated


# ── ws events (server → client) ────────────────────────────────────────
class TranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    channel: int
    text: str
    final: bool
    ts: float


class AmbientItem(BaseModel):
    note_id: str
    title: str
    gist: str
    score: float


class AmbientEvent(BaseModel):
    type: Literal["ambient"] = "ambient"
    items: list[AmbientItem]


class PinEvent(BaseModel):
    type: Literal["pin"] = "pin"
    card: Card


class UnpinEvent(BaseModel):
    type: Literal["unpin"] = "unpin"
    card_id: str


class RetrievalEvent(BaseModel):
    type: Literal["retrieval"] = "retrieval"
    heard: str
    hits: list[Hit]
    expansions: list[Expansion]


class IndexStatus(BaseModel):
    notes: int
    last_sync: Optional[float] = None


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


class DismissMsg(BaseModel):
    type: Literal["dismiss"]
    card_id: str


class OpenNoteMsg(BaseModel):
    type: Literal["open_note"]
    note_id: str
