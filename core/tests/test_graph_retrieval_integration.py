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


class OneHotEmbedder:
    """Deterministic orthogonal embeddings so dense cosine is controllable."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.assigned: dict[str, int] = {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            index = self.assigned.setdefault(text, len(self.assigned) % self.dim)
            row = np.zeros(self.dim, dtype=np.float32)
            row[index] = 1.0
            rows.append(row)
        return np.vstack(rows)


def _real_engine(
    snapshot,
    store,
    registry: EvidenceRegistry,
    *,
    mode: str = "chat",
    query_vector: np.ndarray,
    entity_resolver=None,
    fallback_on_abstain: bool = False,
) -> SharedRetrievalEngine:
    artifact = load_calibration(
        Path(__file__).resolve().parents[1]
        / "evals"
        / "graph_retrieval"
        / "calibration-v1.json"
    )
    calibrator = HeadCalibrator(
        artifact,
        mode=mode,  # type: ignore[arg-type]
        artifact_hash=artifact.hash,
        config_hash=artifact.config_hash,
        model_manifest=artifact.model_manifest,
    )
    return SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        AbstentionPolicy(artifact, mode=mode),  # type: ignore[arg-type]
        query_embedder=lambda _text: query_vector,
        registry=registry,
        reranker=LocalPassageReranker(
            RelevantScorer(),  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            acceptance_floor=(
                artifact.rerank.live_floor if mode == "live" else artifact.rerank.chat_floor
            ),
        ),
        entity_resolver=entity_resolver,
        fallback_on_abstain=fallback_on_abstain,
    )


def test_real_store_with_broken_fts_degrades_instead_of_crashing(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "taurine-knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16)
    ).build(
        [make_note("launch", "Launch plan", "The launch is scheduled for July 18.")],
        database,
        replace_derived=True,
    )
    snapshot = load_index_snapshot(database)
    store = snapshot.open_knowledge_store()
    try:
        def broken_bm25(query: str, k: int):
            raise RuntimeError("fts index corrupted")

        monkeypatch.setattr(store, "search_passages_bm25", broken_bm25)
        registry = EvidenceRegistry()
        engine = _real_engine(
            snapshot, store, registry, query_vector=snapshot.passage_matrix[0]
        )

        pack = engine.retrieve(RetrievalQuery("When is the launch?", "chat", 2.0))

        assert pack.decision.accepted
        assert pack.passages[0].text == "The launch is scheduled for July 18."
        assert "bm25" in pack.trace.debug["unavailable_heads"]
        assert registry.refs("read")
    finally:
        store.close()


def test_hub_seeded_date_entity_traverses_on_a_real_build(tmp_path: Path) -> None:
    from taurine_core.schemas import ExtractedMention, Extraction

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "dates": Extraction(
                mentions=[
                    ExtractedMention(
                        surface="July 18",
                        etype="date_or_deadline",
                        confidence=0.95,
                        char_start=7,
                        char_end=14,
                        extractor="fake",
                        schema_id="s1",
                    )
                ],
                schema_id="s1",
            ),
            "plan": Extraction(mentions=[], schema_id="s1"),
        }

    database = tmp_path / "taurine-knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=OneHotEmbedder(),
            embedding_dim=16,
            extractor=extractor,
        )
    ).build(
        [
            make_note("dates", "Dates", "Due on July 18 for sure."),
            make_note("plan", "Plan", "Completely unrelated gardening tips."),
        ],
        database,
        replace_derived=True,
    )
    snapshot = load_index_snapshot(database)
    store = snapshot.open_knowledge_store()
    try:
        entity_ids = tuple(
            node_id
            for node_id, kind in zip(
                snapshot.base_graph.node_ids, snapshot.base_graph.node_kinds, strict=True
            )
            if kind == "entity"
        )
        assert len(entity_ids) == 1
        date_entity = entity_ids[0]
        rows = store.entity_contexts(entity_ids, build_id=snapshot.build_id)
        assert rows and rows[0].traversal_eligible is False

        from taurine_core.retrieval.heads import RawHeadCandidate

        registry = EvidenceRegistry()
        engine = _real_engine(
            snapshot,
            store,
            registry,
            query_vector=np.zeros(16, dtype=np.float32),
            entity_resolver=lambda surface, etype, query: (
                RawHeadCandidate("entity", date_entity, 1.0, "july 18"),
            ),
        )

        seeded = engine.retrieve(
            RetrievalQuery("What is due?", "chat", 2.0),
            entity_spans=(("July 18", "date_or_deadline"),),
        )

        assert seeded.decision.accepted
        date_passage = next(
            row.passage_id for row in seeded.passages if row.note_id == "dates"
        )
        # The hub was temporarily enabled: PPR mass actually traversed the
        # synthesized contains edges rather than the passage merely being a
        # mention-derived direct candidate.
        assert any(
            node_id == date_passage and score > 0.0
            for node_id, score in seeded.trace.ppr["top_passages"]
        )

        unseeded = engine.retrieve(RetrievalQuery("What is due?", "chat", 2.0))
        assert not unseeded.decision.accepted
        assert unseeded.passages == ()
    finally:
        store.close()


def test_chat_fallback_pack_on_a_real_build(tmp_path: Path) -> None:
    database = tmp_path / "taurine-knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=OneHotEmbedder(), embedding_dim=16)
    ).build(
        [
            make_note("alpha", "Alpha", "Alpha subject matter."),
            make_note("beta", "Beta", "Beta subject matter."),
            make_note("gamma", "Gamma", "Gamma subject matter."),
        ],
        database,
        replace_derived=True,
    )
    snapshot = load_index_snapshot(database)
    store = snapshot.open_knowledge_store()
    try:
        # Engineer the top cosine into the fallback band of the LOADED artifact
        # (curves are data-fitted and change across self-calibration runs):
        # calibrated target = midpoint of (0.75 x floor, floor), inverted
        # through the chat dense knots to a raw cosine, realized exactly
        # against passage 0 with the remainder on an axis orthogonal to every
        # one-hot passage vector.
        from taurine_core.evaluation.graph_retrieval import load_calibration
        from taurine_core.retrieval.runtime import CALIBRATION_PATH

        chat_dense = load_calibration(CALIBRATION_PATH).modes["chat"].heads["dense"]
        floor = chat_dense.acceptance_floor
        target_probability = (0.75 * floor + floor) / 2

        def invert(knots, probability):
            for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
                if y0 <= probability <= y1:
                    if y1 == y0:
                        return x1
                    return x0 + (x1 - x0) * (probability - y0) / (y1 - y0)
            return knots[-1][0]

        raw = float(invert(list(chat_dense.knots), target_probability))
        blended = raw * snapshot.passage_matrix[0].astype(np.float32)
        blended[-1] = float(np.sqrt(max(0.0, 1.0 - raw * raw)))
        blended = blended / np.linalg.norm(blended)

        registry = EvidenceRegistry()
        chat_engine = _real_engine(
            snapshot,
            store,
            registry,
            query_vector=blended,
            fallback_on_abstain=True,
        )
        pack = chat_engine.retrieve(
            RetrievalQuery("loosely reminiscent of it all", "chat", 2.0)
        )

        assert not pack.decision.accepted
        assert pack.decision.fallback is True
        assert pack.decision.reason == "fallback_low_confidence"
        assert 1 <= len(pack.passages) <= 4
        assert pack.facts == () and pack.paths == ()
        assert all(
            registry.resolve(row.evidence_id) is not None
            and "read" in registry.resolve(row.evidence_id).states
            for row in pack.passages
        )

        live_registry = EvidenceRegistry()
        live_engine = _real_engine(
            snapshot,
            store,
            live_registry,
            mode="live",
            query_vector=blended,
            fallback_on_abstain=False,
        )
        live_pack = live_engine.retrieve(
            RetrievalQuery("loosely reminiscent of it all", "live", 2.0)
        )
        assert not live_pack.decision.accepted
        assert live_pack.decision.fallback is False
        assert live_pack.passages == ()
        assert live_registry.refs("read") == ()
    finally:
        store.close()
