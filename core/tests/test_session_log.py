from __future__ import annotations

import json

from taurine_core.live.grounding import GroundingUpdate
from taurine_core.schemas import GroundedFact, GroundingItem
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
        salient="hello",
        dense=[["n1", 0.9]],
        dense_salient=[["n1", 0.8]],
        gliner_spans=[["shopify checkout", 0.95]],
        dense_gliner=[["n2", 0.87]],
        bm25=[],
        fused=[["n1", 0.016]],
        expansions=[{"from": "n1", "to": "n2", "score": 0.4}],
        rerank_used=True,
        rerank_error=None,
        ordered=[["n1", 0.91]],
        floor_dropped=[],
        slate_adds=["n1"],
    )
    logger.log("slate", op="add", note_id="n1", source="search", reason="added", revision=1)
    logger.log("cadence", decision=False, revision=1, reason="debounce")
    logger.log(
        "presenter",
        revision=1,
        model_used="model-a",
        transcript_chunks=2,
        slate_note_ids=["n1"],
        raw_response={"picks": []},
        picks=[],
        latency_ms=12.5,
        error=None,
    )
    logger.close()

    assert logger.path is not None
    records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [record["kind"] for record in records] == [
        "chunk",
        "c2_pass",
        "slate",
        "cadence",
        "presenter",
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
    assert records[1]["salient"] == "hello"
    assert records[1]["dense_salient"] == [["n1", 0.8]]
    assert records[1]["gliner_spans"] == [["shopify checkout", 0.95]]
    assert records[1]["dense_gliner"] == [["n2", 0.87]]
    assert records[2]["op"] == "add"
    assert records[3]["reason"] == "debounce"
    assert records[4]["picks"] == []
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
            "ordered": [["n1", 0.9]],
            "expansions": [],
            "slate_adds": ["n1"],
            "rerank_used": True,
            "gliner_spans": [
                ["shopify checkout", 0.95],
                ["Ashish", 0.9],
                ["extra", 0.8],
            ],
        },
    )

    assert line is not None
    assert line.endswith(" spans=[shopify checkout; Ashish]")


def test_grounding_and_summary_helpers_write_canonical_payloads(tmp_path) -> None:
    logger = SessionLogger.create(tmp_path)
    update = GroundingUpdate(
        revision=7,
        items=[
            GroundingItem(
                note_id="n1",
                short_id="N1",
                title="Timeline",
                gist="Timeline gist",
                score=0.91,
            )
        ],
        facts=[GroundedFact(fact_id="F1", src="n1", dst="n2", text="related")],
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
        "items": [{"note_id": "n1", "short_id": "N1", "score": 0.91}],
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
