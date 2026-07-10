from __future__ import annotations

import json

from taurine_core.live.grounding import GroundingUpdate
from taurine_core.retrieval.types import EvidenceFact, EvidencePassage, RetrievalDecision
from taurine_core.session_log import (
    SessionLogger,
    console_line,
    log_grounding,
    log_summary,
)


def test_session_logger_writes_jsonl_records_and_session_end(tmp_path) -> None:
    logger = SessionLogger.create(tmp_path)

    logger.log("chunk", seq=1, channel=0, text="hello", ts=10.0, final=True)
    logger.log(
        "c2_pass",
        trigger_id="abc123",
        seqs=[1],
        window="hello",
        dense=[["n1", 0.9]],
        gliner_spans=[["shopify checkout", 0.95]],
        dense_gliner=[["n2", 0.87]],
        bm25=[],
        fused=[["n1", 0.016]],
        ppr_top=[["n2", 0.4]],
        blend=[{"note_id": "n1", "final": 0.91}],
        grounding_ids=["n1"],
    )
    logger.close()

    assert logger.path is not None
    records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [record["kind"] for record in records] == [
        "chunk",
        "c2_pass",
        "session_end",
    ]
    assert records[0] | {"ts": records[0]["ts"]} == {
        "ts": records[0]["ts"],
        "kind": "chunk",
        "seq": 1,
        "channel": 0,
        "text": "hello",
        "final": True,
    }
    assert records[1]["trigger_id"] == "abc123"
    assert records[1]["gliner_spans"] == [["shopify checkout", 0.95]]
    assert records[1]["dense_gliner"] == [["n2", 0.87]]
    assert records[1]["grounding_ids"] == ["n1"]
    assert records[-1]["kind"] == "session_end"


def test_session_logger_create_log_and_close_do_not_raise_on_bad_path(tmp_path) -> None:
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("already a file")

    logger = SessionLogger.create(not_a_directory)
    logger.log("chunk", unserializable=object())
    logger.close()


def test_c2_console_line_appends_compact_gliner_spans() -> None:
    line = console_line(
        "c2_pass",
        {
            "seqs": [1, 2],
            "window": "we should ask ashish about checkout",
            "grounding_ids": ["n1"],
            "ppr_top": [],
            "gliner_spans": [
                ["shopify checkout", 0.95],
                ["Ashish", 0.9],
                ["extra", 0.8],
            ],
        },
    )

    assert line is not None
    assert line.endswith(" spans=[shopify checkout; Ashish]")


def test_chat_console_lines_are_compact() -> None:
    retrieval = console_line(
        "chat_retrieval",
        {
            "chat_id": "chat-1234abcd",
            "query": "what changed in roadmap?",
            "top": [{"id": "N1", "score": 0.9}, {"id": "N2", "score": 0.7}],
            "facts": ["F1"],
        },
    )
    turn = console_line(
        "chat_turn",
        {
            "chat_id": "chat-1234abcd",
            "message_id": 7,
            "model": "z-ai/glm-5.2",
            "tools": [{"name": "search_evidence"}],
            "usage": {"prompt": 10, "completion": 20, "reasoning": 5},
            "total_ms": 1234.5,
            "cancelled": False,
            "stripped_citations": 2,
        },
    )

    assert retrieval == "chat chat-1234abcd retrieval “what changed in roadmap?” → 2 notes, 1 facts top=N1"
    assert turn == "chat chat-1234abcd msg=7 z-ai/glm-5.2 1234ms tools=1 usage=10/20+5 stripped=2"


def test_grounding_and_summary_helpers_write_canonical_payloads(tmp_path) -> None:
    logger = SessionLogger.create(tmp_path)
    update = GroundingUpdate(
        decision=RetrievalDecision(True, 0.9, "accepted"),
        revision=7,
        passages=(EvidencePassage("E1", "p1", "n1", "Timeline", "test", "Timeline gist", 0, 13, 1.0, 1.0, 0.9, 0.91),),
        facts=(EvidenceFact("F1", ("f1",), "n1", "related_to", "n2", "forward", "active", 1.0, None, None, 0.9, ("E1",)),),
        paths=(),
        notes=(),
        display_passages=(EvidencePassage("E1", "p1", "n1", "Timeline", "test", "Timeline gist", 0, 13, 1.0, 1.0, 0.9, 0.91),),
        display_notes=(),
        no_new_evidence=False,
        new_support_ids=("E1", "F1"),
    )

    log_grounding(logger, update, decision=True, reason="jaccard")
    log_summary(
        logger,
        model="model-a",
        latency_ms=12.5,
        no_update=False,
        rejected=False,
        reject_reason="",
        citation_stats={"total": 2, "valid": 2, "invalid": 0},
        accepted_actionable_ids=["act-12345678"],
    )
    logger.close()

    assert logger.path is not None
    records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert records[0] | {"ts": records[0]["ts"]} == {
        "ts": records[0]["ts"],
        "kind": "grounding",
        "revision": 7,
        "items": [{"note_id": "n1", "evidence_id": "E1", "score": 0.91}],
        "fact_ids": ["F1"],
        "trigger_decision": True,
        "trigger_reason": "jaccard",
    }
    assert records[1] | {"ts": records[1]["ts"]} == {
        "ts": records[1]["ts"],
        "kind": "summary",
        "model": "model-a",
        "latency_ms": 12.5,
        "no_update": False,
        "rejected": False,
        "reject_reason": "",
        "citation_stats": {"total": 2, "valid": 2, "invalid": 0},
        "accepted_actionable_ids": ["act-12345678"],
    }
