from __future__ import annotations

import hashlib
import re

import numpy as np

from tests.conftest import make_note
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.models_local import LocalModelUnavailable
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.runtime import RetrievalRuntime


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
    monkeypatch.setattr(
        "taurine_core.retrieval.runtime.local_nli_scorer",
        lambda: (_ for _ in ()).throw(LocalModelUnavailable("missing")),
    )

    runtime = RetrievalRuntime(database)
    engine = runtime.make_engine("live", EvidenceRegistry())
    same_generation = runtime.make_engine("chat", EvidenceRegistry())
    assert engine.snapshot.build_id == first.build_id
    assert same_generation.store is engine.store
    assert engine.calibrator.mode == "live"
    assert runtime.nli_scorer is None

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
