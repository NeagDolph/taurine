"""Immutable domain contracts shared by live, chat, and retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


RetrievalMode = Literal["live", "chat", "tool"]
TargetKind = Literal["passage", "fact", "entity", "note"]
TemporalKind = Literal["current", "historical", "any", "at_time"]


@dataclass(frozen=True)
class TemporalIntent:
    kind: TemporalKind
    at: float | None = None


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    mode: RetrievalMode
    now: float
    conversation_context: tuple[str, ...] = ()
    temporal_intent: TemporalIntent | None = None


@dataclass(frozen=True)
class HeadHit:
    target_kind: TargetKind
    target_id: str
    raw_score: float
    calibrated_score: float
    rank: int
    head: str
    reason: str


@dataclass(frozen=True)
class RetrievalDecision:
    accepted: bool
    confidence: float
    reason: str
    accepted_heads: tuple[str, ...] = ()
    rejected_heads: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidencePassage:
    evidence_id: str
    passage_id: str
    note_id: str
    title: str
    source: str
    text: str
    char_start: int
    char_end: int
    created: float
    modified: float
    retrieval_score: float
    rerank_score: float
    path_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceFact:
    fact_ref: str
    fact_ids: tuple[str, ...]
    subject: str
    predicate: str
    object: str
    direction: str
    status: str
    observed_at: float
    valid_from: float | None
    valid_to: float | None
    confidence: float
    source_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidencePath:
    path_id: str
    node_ids: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    fact_ids: tuple[str, ...]
    weight: float
    rendered: str


@dataclass(frozen=True)
class EvidenceNote:
    note_ref: str
    note_id: str
    title: str
    source: str
    modified: float
    score: float
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalTrace:
    original_query: str
    rewritten_query: str = ""
    head_hits: tuple[HeadHit, ...] = ()
    rejected_hits: tuple[HeadHit, ...] = ()
    entity_resolutions: tuple[Mapping[str, object], ...] = ()
    triple_decisions: tuple[Mapping[str, object], ...] = ()
    teleport: Mapping[str, float] = field(default_factory=dict)
    ppr: Mapping[str, object] = field(default_factory=dict)
    rerank: tuple[Mapping[str, object], ...] = ()
    mmr_exclusions: tuple[Mapping[str, object], ...] = ()
    prior_contributions: Mapping[str, float] = field(default_factory=dict)
    selected_ids: tuple[str, ...] = ()
    read_ids: tuple[str, ...] = ()
    cited_ids: tuple[str, ...] = ()
    active_build_id: str = ""
    calibration_hash: str = ""
    latency_ms: Mapping[str, float] = field(default_factory=dict)
    debug: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePack:
    decision: RetrievalDecision
    query: RetrievalQuery
    passages: tuple[EvidencePassage, ...]
    facts: tuple[EvidenceFact, ...]
    paths: tuple[EvidencePath, ...]
    notes: tuple[EvidenceNote, ...]
    trace: RetrievalTrace

    @classmethod
    def abstained(cls, query: RetrievalQuery, reason: str) -> "EvidencePack":
        return cls(
            decision=RetrievalDecision(
                accepted=False,
                confidence=0.0,
                reason=reason,
            ),
            query=query,
            passages=(),
            facts=(),
            paths=(),
            notes=(),
            trace=RetrievalTrace(
                original_query=query.text,
                rewritten_query=query.text,
            ),
        )
