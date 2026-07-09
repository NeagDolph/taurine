from __future__ import annotations

import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.indexer import extract


class FakeSchema:
    def __init__(self) -> None:
        self.entity_types: dict[str, str] = {}
        self.relation_types: list[str] = []

    def entities(self, entity_types: dict[str, str]) -> "FakeSchema":
        self.entity_types = dict(entity_types)
        return self

    def relations(self, relation_types: list[str] | dict[str, str]) -> "FakeSchema":
        self.relation_types = (
            list(relation_types.keys())
            if isinstance(relation_types, dict)
            else list(relation_types)
        )
        return self


class FakeModel:
    def __init__(self, outputs: dict[str, dict], *, fail_on: set[str] | None = None) -> None:
        self.outputs = outputs
        self.fail_on = fail_on or set()
        self.batch_sizes: list[int] = []
        self.batch_texts: list[list[str]] = []
        self.extract_texts: list[str] = []
        self.schemas: list[FakeSchema] = []

    def create_schema(self) -> FakeSchema:
        schema = FakeSchema()
        self.schemas.append(schema)
        return schema

    def batch_extract(
        self,
        texts: list[str],
        schema: FakeSchema,
        *,
        batch_size: int,
        threshold: float,
        include_confidence: bool,
        max_len: int | None = None,
    ) -> list[dict]:
        assert include_confidence is True
        self.batch_sizes.append(len(texts))
        self.batch_texts.append(list(texts))
        if any(marker in text for marker in self.fail_on for text in texts):
            raise RuntimeError("batch inference failed")
        return [self._output_for(text) for text in texts]

    def extract(
        self,
        text: str,
        schema: FakeSchema,
        *,
        threshold: float,
        include_confidence: bool,
        max_len: int | None = None,
    ) -> dict:
        assert include_confidence is True
        self.extract_texts.append(text)
        if any(marker in text for marker in self.fail_on):
            raise RuntimeError("note inference failed")
        return self._output_for(text)

    def _output_for(self, text: str) -> dict:
        for marker, output in self.outputs.items():
            if marker in text:
                return output
        return {"entities": {}, "relation_extraction": {}}


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeModel,
    *,
    batch: int = 8,
    chunk_size: int = 100,
    chunk_overlap: int = 10,
) -> None:
    monkeypatch.setattr(extract, "_extractor_model", lambda: fake)
    monkeypatch.setattr(config, "GLINER_INGEST_BATCH", batch)
    monkeypatch.setattr(config, "GLINER_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(config, "GLINER_CHUNK_OVERLAP", chunk_overlap)
    monkeypatch.setattr(config, "GLINER_ENTITY_MIN_CONF", 0.5)
    monkeypatch.setattr(config, "GLINER_RELATION_MIN_CONF", 0.45)


def test_extract_notes_filters_thresholds_and_relation_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeModel(
        {
            "alpha": {
                "entities": {
                    "person": [
                        {"text": "Jane Doe", "confidence": 0.91},
                        {"text": "Bob Smith", "confidence": 0.49},
                    ],
                    "organization": [{"text": "Acme", "confidence": 0.8}],
                    "product": [{"text": "Taurine", "confidence": 0.4}],
                },
                "relation_extraction": {
                    "works_at": [
                        {
                            "head": {"text": "Jane Doe", "confidence": 0.9},
                            "tail": {"text": "Acme", "confidence": 0.88},
                        },
                        {
                            "head": {"text": "Bob Smith", "confidence": 0.99},
                            "tail": {"text": "Acme", "confidence": 0.99},
                        },
                    ],
                    "uses": [
                        {
                            "head": {"text": "Jane Doe", "confidence": 0.92},
                            "tail": {"text": "Taurine", "confidence": 0.4},
                        }
                    ],
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)

    result = extract.extract_notes([make_note("n1", "alpha", "body")])["n1"]

    assert result.failed is False
    assert [(entity.surface, entity.etype, entity.conf, entity.mentions) for entity in result.entities] == [
        ("Jane Doe", "person", pytest.approx(0.91), 1),
        ("Acme", "organization", pytest.approx(0.8), 1),
    ]
    assert [(rel.head, rel.rel_type, rel.tail, rel.conf) for rel in result.relations] == [
        ("Jane Doe", "works_at", "Acme", pytest.approx(0.88))
    ]


def test_extract_notes_derives_mentions_from_note_text_occurrences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModel(
        {
            "duplicates": {
                "entities": {
                    "person": [{"text": "Jane Doe", "confidence": 0.93}],
                    "organization": [{"text": "Acme, Inc.", "confidence": 0.8}],
                },
                "relation_extraction": {},
            }
        }
    )
    _install_fake(monkeypatch, fake)

    result = extract.extract_notes(
        [
            make_note(
                "n1",
                "Jane Doe",
                "Jane Doe met jane doe at Acme, Inc. Later ACME, INC. expanded. "
                "Acme, Inc.x is not a full mention. duplicates",
            )
        ]
    )["n1"]

    assert [(entity.surface, entity.etype, entity.conf, entity.mentions) for entity in result.entities] == [
        ("Jane Doe", "person", pytest.approx(0.93), 3),
        ("Acme, Inc.", "organization", pytest.approx(0.8), 2),
    ]


def test_extract_notes_marks_only_failed_notes_after_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModel(
        {
            "good": {
                "entities": {"project": [{"text": "Taurine", "confidence": 0.8}]},
                "relation_extraction": {},
            }
        },
        fail_on={"boom"},
    )
    _install_fake(monkeypatch, fake, batch=2)

    result = extract.extract_notes(
        [
            make_note("good", "good", "body"),
            make_note("bad", "boom", "body"),
        ]
    )

    assert result["good"].failed is False
    assert [(entity.surface, entity.etype) for entity in result["good"].entities] == [
        ("Taurine", "project")
    ]
    assert result["bad"].failed is True
    assert result["bad"].entities == []
    assert result["bad"].relations == []
    assert fake.batch_sizes == [2]
    assert len(fake.extract_texts) == 2


def test_extract_notes_splits_forward_batches_at_configured_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModel({})
    _install_fake(monkeypatch, fake, batch=2)
    notes = [make_note(f"n{i}", f"title-{i}", "body") for i in range(5)]

    result = extract.extract_notes(notes)

    assert list(result) == ["n0", "n1", "n2", "n3", "n4"]
    assert fake.batch_sizes == [2, 2, 1]


def test_extract_notes_chunks_long_notes_and_merges_chunk_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModel(
        {
            "alpha beta gamma": {
                "entities": {"person": [{"text": "gamma", "confidence": 0.7}]},
                "relation_extraction": {},
            },
            "gamma delta epsilon": {
                "entities": {"person": [{"text": "gamma", "confidence": 0.9}]},
                "relation_extraction": {},
            },
        }
    )
    _install_fake(monkeypatch, fake, chunk_size=3, chunk_overlap=1)

    result = extract.extract_notes(
        [make_note("n1", "", "alpha beta gamma delta epsilon")]
    )["n1"]

    assert fake.batch_sizes == []
    assert fake.extract_texts == ["alpha beta gamma", "gamma delta epsilon"]
    assert [(entity.surface, entity.conf, entity.mentions) for entity in result.entities] == [
        ("gamma", pytest.approx(0.9), 1)
    ]
