from __future__ import annotations

import json

from taurine_core.session_log import SessionLogger


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
        "ts": records[0]["ts"],
        "final": True,
    }
    assert records[1]["trigger_id"] == "abc123"
    assert records[1]["salient"] == "hello"
    assert records[1]["dense_salient"] == [["n1", 0.8]]
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
