"""Idea lens: seed retrieval, graph expansion, LLM adjudication, second hop,
and the POST /graph/idea endpoint contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import make_note
from taurine_core import idea_search
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.retrieval.snapshot import load_index_snapshot
from taurine_core.schemas import SyncReport
from taurine_core.server import Runtime, create_app


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def _grouped_embed(texts: list[str]) -> np.ndarray:
    """16-dim embedder: 'billing' texts on axis 0, 'travel' on axis 1."""

    rows = []
    for text in texts:
        lowered = text.lower()
        row = np.zeros(16, dtype=np.float32)
        noise = np.frombuffer(
            hashlib.sha256(text.encode()).digest()[:14], dtype=np.uint8
        ).astype(np.float32)
        if "billing" in lowered:
            row[0] = 100.0
            row[2:] = noise / 255.0
        elif "travel" in lowered:
            row[1] = 100.0
            row[2:] = noise / 255.0
        else:
            row[2:] = noise + 1.0
        rows.append(row / np.linalg.norm(row))
    return np.vstack(rows)


def _notes():
    billing = [
        make_note(
            f"bill-{index}",
            f"Billing {index}",
            f"billing invoice reconciliation pass number {index} for the billing rollout",
            modified=float(1000 + index),
        )
        for index in range(4)
    ]
    travel = [
        make_note(
            f"trip-{index}",
            f"Travel {index}",
            f"travel packing list and flights for the tokyo travel plan {index}",
            modified=float(2000 + index),
        )
        for index in range(3)
    ]
    return billing + travel


def _snapshot(tmp_path: Path):
    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=_grouped_embed, embedding_dim=16)
    ).build(_notes(), database, replace_derived=True)
    return load_index_snapshot(database)


@dataclass
class FakeLLM:
    """Aligns billing notes with per-note strengths; records every payload."""

    strengths: dict[str, float] = field(default_factory=dict)
    malformed: set[str] = field(default_factory=set)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        timeout_s: float | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        del model, system, timeout_s, temperature
        payload = json.loads(user)
        self.calls.append(payload)
        judgements: list[Any] = []
        for note in payload["notes"]:
            note_id = note["note_id"]
            if note_id in self.malformed:
                judgements.append({"note_id": note_id, "aligned": "yes-but-not-bool"})
                continue
            aligned = "billing" in (note["title"] + " " + note["snippet"]).lower()
            judgements.append(
                {
                    "note_id": note_id,
                    "aligned": aligned,
                    "strength": self.strengths.get(note_id, 0.9 if aligned else 0.1),
                    "reason": "mentions the billing rollout" if aligned else "unrelated",
                }
            )
        return {"judgements": judgements}


def test_search_idea_seeds_judges_and_sorts_matches(tmp_path: Path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(idea_search, "embed_queries", _grouped_embed)
    llm = FakeLLM(strengths={"bill-0": 0.8, "bill-1": 0.95, "bill-2": 0.9, "bill-3": 0.9})

    response = asyncio.run(
        idea_search.search_idea("Ongoing billing reconciliation work", snapshot=snapshot, llm=llm)
    )

    assert [match.note_id for match in response.matches] == [
        "bill-1",
        "bill-2",  # strength tie with bill-3 breaks by note id
        "bill-3",
        "bill-0",
    ]
    assert all(match.source for match in response.matches)
    assert all(match.reason for match in response.matches)
    assert response.considered == 7  # every note reached adjudication
    assert response.latency_ms >= 0.0
    # payloads carry bounded snippets, never full note bodies beyond the cap
    for call in llm.calls:
        for note in call["notes"]:
            assert len(note["snippet"]) <= idea_search.SNIPPET_MAX_CHARS


def test_expansion_adds_projection_neighbors_beyond_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(idea_search, "embed_queries", _grouped_embed)
    # tiny seed pool: only the top 2 dense notes seed; neighbors must arrive
    # via 1-hop expansion and the agentic second hop
    monkeypatch.setattr(idea_search, "SEED_LIMIT", 2)
    monkeypatch.setattr(idea_search, "SEED_EXPAND_FROM", 1)
    monkeypatch.setattr(idea_search, "CANDIDATE_CAP", 3)
    llm = FakeLLM()

    response = asyncio.run(
        idea_search.search_idea("billing reconciliation", snapshot=snapshot, llm=llm)
    )

    assert response.expanded > 0  # expansion actually contributed candidates
    assert response.considered > 3  # second hop went beyond the round-one cap
    matched = {match.note_id for match in response.matches}
    assert matched == {"bill-0", "bill-1", "bill-2", "bill-3"}


def test_second_hop_only_fires_on_strong_matches(tmp_path: Path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(idea_search, "embed_queries", _grouped_embed)
    monkeypatch.setattr(idea_search, "SEED_LIMIT", 2)
    monkeypatch.setattr(idea_search, "SEED_EXPAND_FROM", 1)
    monkeypatch.setattr(idea_search, "CANDIDATE_CAP", 3)
    # aligned but weak: below the 0.7 second-hop floor
    weak = FakeLLM(strengths={f"bill-{index}": 0.5 for index in range(4)})

    response = asyncio.run(
        idea_search.search_idea("billing reconciliation", snapshot=snapshot, llm=weak)
    )

    assert response.considered == 3  # capped round one, no second hop
    judged = {note["note_id"] for call in weak.calls for note in call["notes"]}
    assert len(judged) == 3


def test_malformed_judgement_rows_are_skipped_not_fatal(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(idea_search, "embed_queries", _grouped_embed)
    llm = FakeLLM(malformed={"bill-1"})

    response = asyncio.run(
        idea_search.search_idea("billing reconciliation", snapshot=snapshot, llm=llm)
    )

    matched = {match.note_id for match in response.matches}
    assert "bill-1" not in matched  # malformed row dropped
    assert {"bill-0", "bill-2", "bill-3"} <= matched  # request still succeeded


# ── endpoint contract ──────────────────────────────────────────────────


def _runtime(tmp_path: Path, llm: Any) -> Runtime:
    snapshot = _snapshot(tmp_path)
    return Runtime(
        index=SimpleNamespace(),
        sync_once=lambda: SyncReport(0, 0, 0, 0),
        open_note=lambda note_id: None,
        make_live_session=None,
        make_asr_client=lambda on_chunk, on_status, index_status: None,
        knowledge_runtime=SimpleNamespace(snapshot=snapshot),
        connector_llm=llm,
    )


def test_graph_idea_endpoint_returns_contractual_payload(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(idea_search, "embed_queries", _grouped_embed)
    runtime = _runtime(tmp_path, FakeLLM())

    with TestClient(create_app(runtime)) as client:
        payload = client.post(
            "/graph/idea", json={"idea": "Ongoing billing reconciliation", "limit": 1}
        ).json()

    assert payload["idea"] == "Ongoing billing reconciliation"
    assert payload["considered"] == 7
    assert payload["latency_ms"] >= 0.0
    matches = payload["matches"]
    assert len(matches) == 4  # limit=1 clamps up to 5
    assert matches[0]["note_id"].startswith("bill-")
    assert set(matches[0]) == {"note_id", "title", "source", "strength", "reason"}
    assert matches == sorted(matches, key=lambda row: (-row["strength"], row["note_id"]))


def test_graph_idea_endpoint_rejects_empty_idea_and_missing_llm(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(idea_search, "embed_queries", _grouped_embed)

    with TestClient(create_app(_runtime(tmp_path, FakeLLM()))) as client:
        assert client.post("/graph/idea", json={"idea": "   "}).status_code == 400

    with TestClient(create_app(_runtime(tmp_path, None))) as client:
        response = client.post("/graph/idea", json={"idea": "billing"})
        assert response.status_code == 503
