from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from tests.conftest import make_note
from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.retrieval.calibration import AbstentionPolicy, HeadCalibrator
from taurine_core.retrieval.evidence import SharedRetrievalEngine
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.rerank import LocalPassageReranker
from taurine_core.retrieval.snapshot import load_index_snapshot
from taurine_core.retrieval.types import RetrievalQuery


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class RelevantScorer:
    def score_pairs(
        self,
        query: str,
        documents: tuple[str, ...],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> tuple[float, ...]:
        del query, instruction, deadline_at
        return tuple(0.99 for _ in documents)


def embedder(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:16], dtype=np.uint8)
        vector = raw.astype(np.float32) + 1.0
        rows.append(vector / np.linalg.norm(vector))
    return np.vstack(rows)


def test_clean_records_build_to_citeable_pack_and_null_abstention(tmp_path: Path) -> None:
    database = tmp_path / "taurine-knowledge.db"
    report = BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    ).build(
        [make_note("launch", "Launch plan", "The launch is scheduled for July 18.")],
        database,
        replace_derived=True,
    )
    snapshot = load_index_snapshot(database)
    artifact = load_calibration(
        Path(__file__).resolve().parents[1]
        / "evals"
        / "graph_retrieval"
        / "calibration-v1.json"
    )
    calibrator = HeadCalibrator(
        artifact,
        mode="chat",
        artifact_hash=artifact.hash,
        config_hash=artifact.config_hash,
        model_manifest=artifact.model_manifest,
    )
    registry = EvidenceRegistry()
    store = snapshot.open_knowledge_store()
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        AbstentionPolicy(artifact, mode="chat"),
        query_embedder=lambda _text: snapshot.passage_matrix[0],
        registry=registry,
        reranker=LocalPassageReranker(
            RelevantScorer(),  # type: ignore[arg-type]
            mode="chat",
            acceptance_floor=artifact.rerank.chat_floor,
        ),
    )

    pack = engine.retrieve(RetrievalQuery("When is the launch?", "chat", 2.0))

    assert pack.decision.accepted
    assert pack.passages[0].text == "The launch is scheduled for July 18."
    assert pack.passages[0].evidence_id == "E1"
    assert pack.notes[0].note_ref == "N1"
    assert pack.trace.active_build_id == report.build_id
    assert registry.refs("read") == ("N1", "E1")

    abstained = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        AbstentionPolicy(artifact, mode="chat"),
        query_embedder=lambda _text: np.zeros(16, dtype=np.float32),
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            RelevantScorer(),  # type: ignore[arg-type]
            mode="chat",
            acceptance_floor=artifact.rerank.chat_floor,
        ),
    ).retrieve(RetrievalQuery("unrelated filler", "chat", 2.0))
    assert not abstained.decision.accepted
    assert abstained.passages == ()
    assert abstained.facts == ()
    store.close()
