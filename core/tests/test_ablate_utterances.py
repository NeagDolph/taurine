from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ablate_utterances import (
    _run_ablation_with_indexes,
    decision_verdict,
    render_markdown,
    run_ablation,
)
from tests.conftest import enrichment, make_note, shifted_vec, unit_vec
from taurine_core import config
from taurine_core.indexer.store import Index


def test_ablation_finds_utterance_only_grounding_and_masks_without_mutating_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 1)
    monkeypatch.setattr(config, "HEAD_TOP_K", 2)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    monkeypatch.setattr(config, "GROUNDING_HYSTERESIS", None)
    query_vec = unit_vec(201)
    db_path = _ablation_db(tmp_path, query_vec)
    sessions_dir = _sessions_dir(tmp_path, ["spoken launch question"])
    before = db_path.read_bytes()

    def fake_embed_queries(texts: list[str], *, prompt: str | None = None) -> np.ndarray:
        del prompt
        return np.vstack([query_vec for _text in texts]).astype(np.float32)

    monkeypatch.setattr("taurine_core.indexer.store.embed_queries", fake_embed_queries)

    report = run_ablation(db_path, sessions_dir)

    assert db_path.read_bytes() == before
    assert _embedding_kind_counts(db_path)["utterance"] == 1
    assert report["corpus"]["windows"] == 1
    assert report["corpus"]["utterance_only_fraction"] > 0.0
    assert report["corpus"]["mean_top5_jaccard"] < 1.0
    window = report["sessions"][0]["windows"][0]
    assert window["baseline_top5"] == ["utterance-only"]
    assert window["masked_top5"] == ["chunk-fallback"]
    assert window["utterance_only_ids"] == ["utterance-only"]
    assert window["fused"][0]["dense_kinds"] == ["utterance"]


def test_run_ablation_writes_machine_readable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 1)
    monkeypatch.setattr(config, "HEAD_TOP_K", 2)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    monkeypatch.setattr(config, "GROUNDING_HYSTERESIS", None)
    query_vec = unit_vec(301)
    db_path = _ablation_db(tmp_path, query_vec)
    sessions_dir = _sessions_dir(tmp_path, ["spoken launch question"])
    json_path = tmp_path / "report.json"

    monkeypatch.setattr(
        "taurine_core.indexer.store.embed_queries",
        lambda texts, prompt=None: np.vstack([query_vec for _text in texts]).astype(
            np.float32
        ),
    )

    report = run_ablation(db_path, sessions_dir, json_path=json_path)

    assert json.loads(json_path.read_text()) == report


def test_report_discloses_replay_caveats_in_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 1)
    monkeypatch.setattr(config, "HEAD_TOP_K", 2)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    monkeypatch.setattr(config, "GROUNDING_HYSTERESIS", None)
    query_vec = unit_vec(401)
    db_path = _ablation_db(tmp_path, query_vec)
    sessions_dir = _sessions_dir(tmp_path, ["spoken launch question"])
    expected_caveats = [
        (
            "Replay runs without the GLiNER span-extractor head and without "
            "partial-transcript/wall-clock throttling."
        ),
        (
            "Net bias direction is likely conservative (toward KEEP) but not "
            "provably one-directional since production GLiNER queries also hit "
            "utterance rows."
        ),
        (
            "Metric (a) is attribution-based (winning dense row), not "
            "reachability-based."
        ),
    ]

    monkeypatch.setattr(
        "taurine_core.indexer.store.embed_queries",
        lambda texts, prompt=None: np.vstack([query_vec for _text in texts]).astype(
            np.float32
        ),
    )

    report = run_ablation(db_path, sessions_dir)
    markdown = render_markdown(report)

    assert report["caveats"] == expected_caveats
    assert "## Caveats" in markdown
    for caveat in expected_caveats:
        assert f"- {caveat}" in markdown


def test_empty_both_grounding_windows_do_not_dilute_mean_jaccard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions_dir = _sessions_dir(tmp_path, ["diverging window", "empty both"])
    baseline = SimpleNamespace(condition="baseline")
    masked = SimpleNamespace(condition="masked")

    class FakeConditionRunner:
        def __init__(self, index: SimpleNamespace) -> None:
            self._condition = index.condition

        def run(self, window: SimpleNamespace) -> SimpleNamespace:
            note_ids_by_window = {
                1: {"baseline": ["baseline-only"], "masked": ["masked-only"]},
                2: {"baseline": [], "masked": []},
            }
            note_ids = note_ids_by_window[window.index][self._condition]
            items = [
                SimpleNamespace(note_id=note_id, why=SimpleNamespace(heads=["dense"]))
                for note_id in note_ids
            ]
            return SimpleNamespace(
                update=SimpleNamespace(items=items),
                heads=SimpleNamespace(
                    dense_kinds={note_id: {"chunk"} for note_id in note_ids},
                    head_hits={note_id: {"dense"} for note_id in note_ids},
                    dense_gliner_kinds={},
                    fused=[(note_id, 1.0) for note_id in note_ids],
                ),
                exact_hits=[],
            )

    monkeypatch.setattr(
        "scripts.ablate_utterances._ConditionRunner",
        FakeConditionRunner,
    )

    report = _run_ablation_with_indexes(
        baseline,
        masked,
        tmp_path / "taurine.db",
        sessions_dir,
        limit=None,
    )

    assert report["corpus"]["mean_top5_jaccard"] == 0.0
    assert report["empty_both_windows"] == 1


def test_decision_verdict_uses_preregistered_rule() -> None:
    assert decision_verdict(0.049, 0.9) == "ELIMINATE"
    assert decision_verdict(0.05, 0.9) == "KEEP"
    assert decision_verdict(0.049, 0.899) == "KEEP"


def _ablation_db(tmp_path: Path, query_vec: np.ndarray) -> Path:
    db_path = tmp_path / "taurine.db"
    index = Index(db_path)
    index.upsert_note(
        make_note("utterance-only", "Utterance Only", "Body text unrelated to query."),
        enrichment("Unrelated body gist.", utterances=["spoken launch question"]),
        [
            ("chunk", "unrelated chunk", unit_vec(202)),
            ("utterance", "spoken launch question", query_vec),
        ],
    )
    index.upsert_note(
        make_note("chunk-fallback", "Chunk Fallback", "Another unrelated body."),
        enrichment("Fallback gist.", utterances=[]),
        [("chunk", "fallback semantic chunk", shifted_vec(query_vec, 203, noise=0.05))],
    )
    index.close()
    return db_path


def _sessions_dir(tmp_path: Path, texts: list[str]) -> Path:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    with (sessions_dir / "fixture.jsonl").open("w", encoding="utf-8") as handle:
        for seq, text in enumerate(texts, start=1):
            handle.write(
                json.dumps(
                    {
                        "ts": 1_783_375_500.0 + seq,
                        "kind": "chunk",
                        "seq": seq,
                        "channel": 0,
                        "text": text,
                        "final": True,
                    }
                )
                + "\n"
            )
    return sessions_dir


def _embedding_kind_counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            kind: count
            for kind, count in conn.execute(
                "SELECT kind, COUNT(*) FROM embeddings GROUP BY kind"
            )
        }
    finally:
        conn.close()
