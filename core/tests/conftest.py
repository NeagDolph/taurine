from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from taurine_core import config
from taurine_core.schemas import Enrichment, EntityMention, NoteRecord, RelationTriple


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: requires local model downloads or remote services")


def make_note(
    note_id: str,
    title: str,
    text: str,
    *,
    folder: str = "Fixture",
    modified: float = 1.0,
) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=title,
        folder=folder,
        modified=modified,
        content_hash=hashlib.sha256((title + text).encode("utf-8")).hexdigest(),
        text=text,
    )


def unit_vec(seed: int, *, dim: int = config.EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=dim).astype(np.float32)
    values /= np.linalg.norm(values)
    return values


def shifted_vec(base: np.ndarray, seed: int, *, noise: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = base + rng.normal(scale=noise, size=base.shape).astype(np.float32)
    values /= np.linalg.norm(values)
    return values.astype(np.float32)


def entity(
    surface: str,
    etype: str = "project",
    *,
    conf: float = 0.9,
    mentions: int = 1,
) -> EntityMention:
    return EntityMention(surface=surface, etype=etype, conf=conf, mentions=mentions)


def enrichment(
    gist: str,
    *,
    aliases: list[str] | None = None,
    topics: list[str] | None = None,
    entities: list[str | EntityMention] | None = None,
    relations: list[RelationTriple] | None = None,
    utterances: list[str] | None = None,
    extraction_failed: bool = False,
) -> Enrichment:
    typed_entities = [entity(topic, "concept") for topic in (topics or ["work"])]
    for item in entities or []:
        typed_entities.append(item if isinstance(item, EntityMention) else entity(item))
    return Enrichment(
        gist=gist,
        utterances=utterances
        or [
            f"Can we discuss {gist.lower()}?",
            f"What changed about {gist.lower()}?",
            f"I need the context on {gist.lower()}.",
            f"Let's revisit {gist.lower()}.",
        ],
        aliases=aliases or [],
        entities=typed_entities,
        relations=relations or [],
        extraction_failed=extraction_failed,
    )


def fixture_units(vector: np.ndarray, enrich: Enrichment) -> list[tuple[str, str, np.ndarray]]:
    units = [("chunk", enrich.gist, vector)]
    if enrich.utterances:
        units.append(("utterance", enrich.utterances[0], vector))
    return units


class _DefaultExtractorSchema:
    def entities(self, _types: dict[str, str]) -> "_DefaultExtractorSchema":
        return self

    def relations(self, _types: list[str]) -> "_DefaultExtractorSchema":
        return self


class _DefaultExtractorModel:
    def create_schema(self) -> _DefaultExtractorSchema:
        return _DefaultExtractorSchema()

    def batch_extract(
        self,
        texts: list[str],
        schema: object,
        *,
        batch_size: int,
        threshold: float,
        include_confidence: bool,
        max_len: int | None,
    ) -> list[dict]:
        del schema, batch_size, threshold, include_confidence, max_len
        return [{"entities": {}, "relation_extraction": {}} for _text in texts]


@pytest.fixture(autouse=True)
def no_real_indexer_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    from taurine_core.indexer import extract

    monkeypatch.setattr(extract, "_extractor_model", lambda: _DefaultExtractorModel())


@pytest.fixture
def store(tmp_path: Path):
    from taurine_core.indexer.store import Index

    return Index(tmp_path / "taurine-test.db")
