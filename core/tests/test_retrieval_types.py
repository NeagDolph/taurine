from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from taurine_core import config
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    EvidencePath,
    HeadHit,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
    TemporalIntent,
)
from taurine_core.schemas import EvidencePackPayload


def test_retrieval_contracts_are_immutable_and_tuple_backed() -> None:
    query = RetrievalQuery(text="who owns launch?", mode="chat", now=12.0)
    hit = HeadHit(
        target_kind="passage",
        target_id="p1",
        raw_score=0.72,
        calibrated_score=0.81,
        rank=1,
        head="dense",
        reason="query embedding",
    )

    assert query.conversation_context == ()
    assert query.temporal_intent is None
    assert hit.calibrated_score == pytest.approx(0.81)
    with pytest.raises(FrozenInstanceError):
        query.text = "changed"  # type: ignore[misc]


def test_abstained_pack_has_no_evidence_and_explicit_reason() -> None:
    query = RetrievalQuery(text="um okay yeah", mode="live", now=20.0)
    pack = EvidencePack.abstained(query, "no_head_cleared_floor")

    assert pack.decision == RetrievalDecision(
        accepted=False,
        confidence=0.0,
        reason="no_head_cleared_floor",
        accepted_heads=(),
        rejected_heads=(),
    )
    assert pack.passages == ()
    assert pack.facts == ()
    assert pack.paths == ()
    assert pack.notes == ()
    assert pack.trace.original_query == "um okay yeah"


def test_evidence_pack_serializes_to_public_payload_without_private_trace_text() -> None:
    query = RetrievalQuery(
        text="who owns launch?",
        mode="chat",
        now=30.0,
        temporal_intent=TemporalIntent(kind="current"),
    )
    passage = EvidencePassage(
        evidence_id="E1",
        passage_id="p1",
        note_id="n1",
        title="Launch",
        source="apple_notes",
        text="Maya owns the launch checklist.",
        char_start=10,
        char_end=41,
        created=1.0,
        modified=2.0,
        retrieval_score=0.8,
        rerank_score=0.9,
        path_ids=("P1",),
        low_confidence=True,
    )
    fact = EvidenceFact(
        fact_ref="F1",
        fact_ids=("fact-1",),
        subject="Maya",
        predicate="responsible_for",
        object="launch checklist",
        direction="forward",
        status="active",
        observed_at=2.0,
        valid_from=None,
        valid_to=None,
        confidence=0.91,
        source_evidence_ids=("E1",),
    )
    path = EvidencePath(
        path_id="P1",
        node_ids=("entity:m", "passage:p1"),
        edge_kinds=("contains",),
        fact_ids=(),
        weight=0.7,
        rendered="Maya -> [E1]",
    )
    note = EvidenceNote(
        note_ref="N1",
        note_id="n1",
        title="Launch",
        source="apple_notes",
        modified=2.0,
        score=0.9,
        evidence_ids=("E1",),
    )
    pack = EvidencePack(
        decision=RetrievalDecision(True, 0.92, "accepted", ("dense", "triple"), ()),
        query=query,
        passages=(passage,),
        facts=(fact,),
        paths=(path,),
        notes=(note,),
        trace=RetrievalTrace(
            original_query=query.text,
            rewritten_query=query.text,
            active_build_id="build-private",
            calibration_hash="cal-private",
            debug={"private_source_text": "must not serialize"},
        ),
    )

    payload = EvidencePackPayload.from_domain(pack).model_dump()

    assert payload["decision"]["accepted"] is True
    assert payload["passages"][0]["evidence_id"] == "E1"
    assert payload["passages"][0]["low_confidence"] is True
    assert payload["facts"][0]["fact_ref"] == "F1"
    assert payload["paths"][0]["path_id"] == "P1"
    assert payload["notes"][0]["note_ref"] == "N1"
    assert "trace" not in payload
    assert "private_source_text" not in str(payload)


def test_overhaul_configuration_matches_approved_defaults() -> None:
    assert config.PASSAGE_TARGET_TOKENS == 220
    assert config.PASSAGE_MAX_TOKENS == 300
    assert config.PASSAGE_OVERLAP_TOKENS == 40
    assert config.GLINER_LONG_CHUNK_TOKENS == 384
    assert config.GLINER_LONG_OVERLAP_TOKENS == 64
    assert config.GLINER_BATCH_TOKEN_BUDGET == 3072
    assert config.BASE_GRAPH_PASSAGE_K == 12
    assert config.FACT_REVERSE_WEIGHT_DEFAULT == pytest.approx(0.35)
    assert config.TRIPLE_RECALL_K == 20
    assert config.TRIPLE_SEED_K == 5
    assert config.PASSAGE_RESET_W == pytest.approx(1.0)
    assert config.ENTITY_RESET_W == pytest.approx(1.0)
    assert config.QUERY_ENTITY_RESET_W == pytest.approx(0.5)
    assert config.SESSION_RESET_W == pytest.approx(0.35)
    assert config.RERANK_MODEL == "Qwen/Qwen3-Reranker-0.6B"
    assert config.RERANK_API_MODEL == "cohere/rerank-v3.5"
    assert config.RERANK_MAX_CANDIDATES == 24
    assert config.RERANK_API_FLOOR_OVERRIDES == {
        "accept": pytest.approx(0.30),
        "fallback": pytest.approx(0.05),
    }
    assert config.LIVE_RERANK_API_FLOOR_OVERRIDES == {
        "accept": pytest.approx(0.5),
        "fallback": pytest.approx(0.15),
    }
    assert config.LIVE_RERANK_CANDIDATES == 30
    assert config.CHAT_RERANK_CANDIDATES == 60
    assert config.LIVE_EVIDENCE_PASSAGES == 8
    assert config.CHAT_EVIDENCE_PASSAGES == 12
    assert config.DENSE_UNGATED_CANDIDACY is True
    assert config.CHAT_FILL_TRIGGER == 4
    assert config.CHAT_FILL_MAX_PASSAGES == 8
    assert config.CHAT_NON_DENSE_RESERVED_CANDIDATES == 6
    assert config.EVIDENCE_NLI_MODEL == "cross-encoder/nli-deberta-v3-base"
    assert config.LIVE_WINDOW_MIN_S == pytest.approx(12.0)
    assert config.LIVE_WINDOW_MAX_S == pytest.approx(45.0)
    assert config.LIVE_WINDOW_MAX_CHARS == 6000
    assert config.PROJECTION_MAX_DEGREE == 12
