from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.models_local import LocalModelUnavailable
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.runtime import RetrievalRuntime


@pytest.fixture(autouse=True)
def _fixture_embedding_space(monkeypatch):
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "openrouter")
    monkeypatch.setattr(config, "EMBED_API_DIM", 16)


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def _embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:16], dtype=np.uint8)
        row = raw.astype(np.float32) + 1.0
        rows.append(row / np.linalg.norm(row))
    return np.vstack(rows)


class PairScorerStub:
    def score_pairs(
        self,
        query: str,
        documents,
        *,
        instruction: str,
        deadline_at: float | None = None,
    ):
        del query, instruction, deadline_at
        return tuple(0.9 for _ in documents)


def test_local_mode_keeps_qwen_for_triples_and_passage_reranking(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "INFERENCE_BACKEND", "local")
    monkeypatch.setattr(config, "EMBED_LOCAL_DIM", 16)
    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16)
    ).build(
        [make_note("n1", "One", "first source passage")],
        database,
        replace_derived=True,
    )
    local = PairScorerStub()
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker", lambda: local
    )
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime._openrouter_pair_reranker",
        lambda: (_ for _ in ()).throw(AssertionError("remote client loaded in local mode")),
    )

    runtime = RetrievalRuntime(database)
    try:
        engine = runtime.make_engine("chat", EvidenceRegistry())
        assert runtime.local_pair_scorer is local
        assert runtime.pair_scorer is local
        assert runtime.pair_scorer_model_id == config.RERANK_MODEL
        assert engine.triple_recognizer.scorer is local
        assert engine.reranker.scorer is local
        assert engine.reranker.model_id == config.RERANK_MODEL
        assert engine.reranker.acceptance_floor == pytest.approx(
            runtime.artifact.rerank.chat_floor
        )
        assert engine.reranker.evidence_floor_override is None
        assert engine.reranker.fallback_scorer is None
    finally:
        runtime.close()


def test_openrouter_mode_uses_provisional_cohere_floors_for_chat_only(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16)
    ).build(
        [make_note("n1", "One", "first source passage")],
        database,
        replace_derived=True,
    )
    local = PairScorerStub()
    remote = PairScorerStub()
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker", lambda: local
    )
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime._openrouter_pair_reranker", lambda: remote
    )

    runtime = RetrievalRuntime(database)
    try:
        engine = runtime.make_engine("chat", EvidenceRegistry())
        live_engine = runtime.make_engine("live", EvidenceRegistry())
        assert runtime.local_pair_scorer is local
        assert runtime.pair_scorer is remote
        assert runtime.pair_scorer_model_id == config.RERANK_API_MODEL
        assert engine.triple_recognizer.scorer is local
        assert engine.reranker.scorer is remote
        assert engine.reranker.model_id == config.RERANK_API_MODEL
        assert engine.reranker.acceptance_floor == pytest.approx(0.30)
        assert engine.reranker.evidence_floor_override == pytest.approx(0.05)
        assert live_engine.reranker.acceptance_floor == pytest.approx(0.5)
        assert live_engine.reranker.evidence_floor_override == pytest.approx(0.15)
        assert engine.reranker.fallback_scorer is local
        assert engine.reranker.fallback_acceptance_floor == pytest.approx(
            runtime.artifact.rerank.chat_floor
        )
        assert engine.reranker.fallback_model_id == config.RERANK_MODEL
    finally:
        runtime.close()


def test_runtime_rejects_snapshot_from_inactive_embedding_model(tmp_path) -> None:
    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=_embed,
            embedding_dim=16,
            embedding_model_id=config.EMBED_MODEL,
        )
    ).build([make_note("n1", "One", "first source passage")], database, replace_derived=True)

    with pytest.raises(ValueError, match="active embedding model"):
        RetrievalRuntime(database)


def test_runtime_rejects_snapshot_with_wrong_active_embedding_dimension(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=_embed,
            embedding_dim=16,
            embedding_model_id=config.EMBED_API_MODEL,
        )
    ).build([make_note("n1", "One", "first source passage")], database, replace_derived=True)
    monkeypatch.setattr(config, "EMBED_API_DIM", 4096)

    with pytest.raises(ValueError, match="embedding dimension"):
        RetrievalRuntime(database)


def test_runtime_checks_manifests_and_publishes_new_complete_build(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "knowledge.db"
    builder = BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16)
    )
    first = builder.build(
        [make_note("n1", "One", "first source passage")],
        database,
        replace_derived=True,
    )
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker",
        lambda: (_ for _ in ()).throw(LocalModelUnavailable("missing")),
    )

    runtime = RetrievalRuntime(database)
    engine = runtime.make_engine("live", EvidenceRegistry())
    same_generation = runtime.make_engine("chat", EvidenceRegistry())
    assert engine.snapshot.build_id == first.build_id
    assert same_generation.store is engine.store
    assert engine.calibrator.mode == "live"

    second = builder.build(
        [make_note("n2", "Two", "second source passage")],
        database,
        replace_derived=True,
    )
    activated = runtime.activate_from_disk()
    assert activated.build_id == second.build_id
    assert runtime.snapshot.build_id == second.build_id
    assert engine.snapshot.build_id == first.build_id
    runtime.close()


def test_activation_rejects_tampered_installed_build(tmp_path, monkeypatch) -> None:
    """A fresh install whose stored rows no longer reproduce the recorded
    graph hash must not activate (L24 activation-time verification)."""
    import sqlite3

    import pytest

    database = tmp_path / "knowledge.db"
    builder = BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16)
    )
    builder.build(
        [make_note("n1", "One", "first source passage")],
        database,
        replace_derived=True,
    )
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker",
        lambda: (_ for _ in ()).throw(LocalModelUnavailable("missing")),
    )
    runtime = RetrievalRuntime(database)

    second = builder.build(
        [make_note("n2", "Two", "second source passage")],
        database,
        replace_derived=True,
    )
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE index_builds SET graph_hash = 'tampered' WHERE build_id = ?",
            (second.build_id,),
        )

    with pytest.raises(ValueError, match="graph-hash verification"):
        runtime.activate_from_disk()
    assert runtime.snapshot.build_id != second.build_id
    runtime.close()


def test_entity_resolver_output_flows_through_query_entity_head(
    tmp_path, monkeypatch
) -> None:
    """Regression: chat/live died with 'RawHeadCandidate' has no 'entity_id'.

    The runtime resolver must return the plain-sequence shape that
    heads._resolution_candidates passes through untouched.
    """
    from taurine_core.retrieval.heads import QueryEntityHead, RawHeadCandidate
    from taurine_core.schemas import ExtractedMention, Extraction

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(
                        surface="Maya",
                        etype="person",
                        confidence=0.95,
                        char_start=0,
                        char_end=4,
                        extractor="fake",
                        schema_id="s1",
                    )
                ],
                schema_id="s1",
            )
        }

    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16, extractor=extractor
        )
    ).build([make_note("n1", "Maya", "Maya owns the launch.")], database, replace_derived=True)
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker",
        lambda: (_ for _ in ()).throw(LocalModelUnavailable("missing")),
    )

    runtime = RetrievalRuntime(database)
    for mode in ("chat", "live"):
        engine = runtime.make_engine(mode, EvidenceRegistry())
        resolution = engine.entity_resolver("Maya", "person", "who is maya?")
        assert isinstance(resolution, tuple)
        assert all(isinstance(row, RawHeadCandidate) for row in resolution)
        assert len(resolution) == 1
        assert resolution[0].raw_score == 1.0

        result = QueryEntityHead(engine.entity_resolver, engine.calibrator).search(
            "who is maya?", (("Maya", "person"), ("Unknown Widget", None))
        )
        hits = list(result.accepted) + list(result.rejected)
        assert [hit.target_id for hit in hits] == [resolution[0].target_id]


def test_tampered_calibration_degrades_to_conservative_floors_without_raising(
    tmp_path, monkeypatch
) -> None:
    import json
    from pathlib import Path

    from taurine_core.retrieval.runtime import CALIBRATION_PATH

    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16)
    ).build([make_note("n1", "One", "first source passage")], database, replace_derived=True)
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker",
        lambda: (_ for _ in ()).throw(LocalModelUnavailable("missing")),
    )

    # Tamper a floor but keep the declared hash: the content check must catch
    # it and the runtime must degrade instead of raising.
    payload = json.loads(Path(CALIBRATION_PATH).read_text(encoding="utf-8"))
    payload["modes"]["chat"]["evidence_floor"] = 0.01
    payload["modes"]["chat"]["heads"]["dense"]["acceptance_floor"] = 0.01
    tampered = tmp_path / "tampered-calibration.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    runtime = RetrievalRuntime(database, calibration_path=tampered)
    try:
        assert runtime.calibration_status == "calibration_mismatch"
        # Conservative table: strictest (live) floors on every mode, not the
        # tampered 0.01 values.
        assert runtime.artifact.modes["chat"].evidence_floor >= 0.75
        assert runtime.artifact.modes["chat"].heads["dense"].acceptance_floor >= 0.7
        engine = runtime.make_engine("chat", EvidenceRegistry())
        assert engine.calibration_status == "calibration_mismatch"
        assert engine.fallback_on_abstain is True
    finally:
        runtime.close()


def test_intact_calibration_is_verified_and_wires_fallback_modes(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "knowledge.db"
    BuildCoordinator(
        BuildDependencies(tokenizer=Tokenizer(), embedder=_embed, embedding_dim=16)
    ).build([make_note("n1", "One", "first source passage")], database, replace_derived=True)
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_pair_reranker",
        lambda: (_ for _ in ()).throw(LocalModelUnavailable("missing")),
    )

    runtime = RetrievalRuntime(database)
    try:
        assert runtime.calibration_status == "verified"
        assert runtime.make_engine("live", EvidenceRegistry()).fallback_on_abstain is False
        assert runtime.make_engine("chat", EvidenceRegistry()).fallback_on_abstain is True
        assert runtime.make_engine("tool", EvidenceRegistry()).fallback_on_abstain is True
    finally:
        runtime.close()
