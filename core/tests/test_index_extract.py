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
        self.include_spans: list[bool] = []

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
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict]:
        assert include_confidence is True
        self.include_spans.append(include_spans)
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
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict:
        assert include_confidence is True
        self.include_spans.append(include_spans)
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

    result = extract.extract_notes(
        [make_note("n1", "alpha", "alpha Jane Doe works at Acme.")]
    )["n1"]

    assert result.failed is False
    assert [(entity.surface, entity.etype, entity.conf, entity.mentions) for entity in result.entities] == [
        ("Jane Doe", "person", pytest.approx(0.91), 1),
        ("Acme", "organization", pytest.approx(0.8), 1),
    ]
    assert [(rel.head, rel.rel_type, rel.tail, rel.conf) for rel in result.relations] == [
        ("Jane Doe", "works_at", "Acme", pytest.approx(0.88))
    ]


def test_extract_notes_drops_self_and_tiny_endpoint_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModel(
        {
            "calibration": {
                "entities": {
                    "person": [
                        {"text": "Sharon", "confidence": 0.94},
                        {"text": "Matthew Tung", "confidence": 0.93},
                    ],
                    "organization": [{"text": "ts", "confidence": 0.91}],
                },
                "relation_extraction": {
                    "works_with": [
                        {
                            "head": {"text": " Sharon ", "confidence": 0.9},
                            "tail": {"text": "sharon", "confidence": 0.9},
                            "confidence": 0.8,
                        }
                    ],
                    "works_at": [
                        {
                            "head": {"text": "Matthew Tung", "confidence": 0.92},
                            "tail": {"text": "ts", "confidence": 0.92},
                            "confidence": 0.72,
                        }
                    ],
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)

    result = extract.extract_notes(
        [make_note("n1", "calibration", "calibration Sharon and Matthew Tung at ts.")]
    )["n1"]

    assert result.failed is False
    assert [(entity.surface, entity.etype) for entity in result.entities] == [
        ("Sharon", "person"),
        ("Matthew Tung", "person"),
        ("ts", "organization"),
    ]
    assert result.relations == []


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
        ("Jane Doe", "person", pytest.approx(0.93), 2),
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
            make_note("good", "good", "good Taurine body"),
            make_note("bad", "boom", "boom body"),
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

    assert fake.batch_sizes == [2]
    assert fake.batch_texts == [["alpha beta gamma", "gamma delta epsilon"]]
    assert [(entity.surface, entity.conf, entity.mentions) for entity in result.entities] == [
        ("gamma", pytest.approx(0.9), 1)
    ]


class OffsetTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        import re

        assert add_special_tokens is False
        assert return_offsets_mapping is True
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class NativeLongFake(FakeModel):
    def __init__(self, output: dict) -> None:
        super().__init__({})
        self.output = output
        self.long_calls: list[dict] = []

    def batch_extract_long(self, texts: list[str], schema: FakeSchema, **kwargs: object) -> list[dict]:
        self.long_calls.append({"texts": list(texts), "schema": schema, **kwargs})
        return [self.output for _ in texts]


def test_short_batch_requests_spans_and_records_note_global_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Maya owns the launch checklist."
    fake = FakeModel(
        {
            "Maya": {
                "entities": {
                    "person": [{"text": "Maya", "confidence": 0.95, "start": 0, "end": 4}],
                    "concept": [
                        {
                            "text": "launch checklist",
                            "confidence": 0.9,
                            "start": 14,
                            "end": 30,
                        }
                    ],
                },
                "relation_extraction": {
                    "responsible_for": [
                        {
                            "head": {"text": "Maya", "confidence": 0.94, "start": 0, "end": 4},
                            "tail": {
                                "text": "launch checklist",
                                "confidence": 0.89,
                                "start": 14,
                                "end": 30,
                            },
                        }
                    ]
                },
            }
        }
    )
    _install_fake(monkeypatch, fake, chunk_size=50)

    result = extract.extract_notes(
        [make_note("n1", "Launch", text)], tokenizer=OffsetTokenizer()
    )["n1"]

    assert fake.include_spans == [True]
    assert [(row.surface, row.char_start, row.char_end) for row in result.mentions] == [
        ("Maya", 0, 4),
        ("launch checklist", 14, 30),
    ]
    relation = result.fact_candidates[0]
    assert (relation.subject_start, relation.subject_end) == (0, 4)
    assert (relation.object_start, relation.object_end) == (14, 30)
    assert text[relation.assertion_start : relation.assertion_end] == text
    assert relation.status == "active"
    assert relation.schema_id == result.schema_id


def test_native_long_api_is_used_with_global_spans_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "alpha beta gamma delta epsilon"
    fake = NativeLongFake(
        {
            "entities": {
                "person": [{"text": "delta", "confidence": 0.91, "start": 17, "end": 22}]
            },
            "relation_extraction": {},
        }
    )
    _install_fake(monkeypatch, fake, chunk_size=3, chunk_overlap=1)

    result = extract.extract_notes(
        [make_note("n1", "Long", text)], tokenizer=OffsetTokenizer()
    )["n1"]

    assert len(fake.long_calls) == 1
    call = fake.long_calls[0]
    assert call["chunk_size"] == 3
    assert call["chunk_overlap"] == 1
    assert call["include_confidence"] is True
    assert call["include_spans"] is True
    assert [(row.surface, row.char_start, row.char_end) for row in result.mentions] == [
        ("delta", 17, 22)
    ]


def test_invalid_returned_entity_span_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeModel(
        {
            "Maya": {
                "entities": {
                    "person": [{"text": "Maya", "confidence": 0.95, "start": 5, "end": 9}]
                },
                "relation_extraction": {},
            }
        }
    )
    _install_fake(monkeypatch, fake)

    result = extract.extract_notes(
        [make_note("n1", "Title", "Maya owns launch.")], tokenizer=OffsetTokenizer()
    )["n1"]

    assert result.mentions == []
    assert result.entities == []


def test_assertion_span_covers_first_paragraph_without_leading_blank_line() -> None:
    """M1: rfind returning -1 must map to paragraph start 0, not 1."""
    text = "Alice joined Acme. She runs the launch with Bob"

    span = extract._assertion_span(text, 0, 5, 44, 47)

    assert span == (0, len(text))


def test_assertion_span_mid_document_paragraph_offsets_are_exact() -> None:
    text = "Intro line.\n\nAlice joined Acme. She runs the launch with Bob"
    subject_start = text.index("Alice")
    object_start = text.index("Bob")

    span = extract._assertion_span(
        text, subject_start, subject_start + 5, object_start, object_start + 3
    )

    assert span is not None
    assert text[span[0] : span[1]] == "Alice joined Acme. She runs the launch with Bob"


def test_relation_with_first_word_subject_spanning_sentences_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Alice joined Acme. She runs the launch with Bob"
    fake = FakeModel(
        {
            "Alice": {
                "entities": {
                    "person": [
                        {"text": "Alice", "confidence": 0.95, "start": 0, "end": 5},
                        {"text": "Bob", "confidence": 0.94, "start": 44, "end": 47},
                    ]
                },
                "relation_extraction": {
                    "works_with": [
                        {
                            "head": {"text": "Alice", "confidence": 0.9, "start": 0, "end": 5},
                            "tail": {"text": "Bob", "confidence": 0.9, "start": 44, "end": 47},
                            "confidence": 0.9,
                        }
                    ]
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)

    result = extract.extract_notes(
        [make_note("n1", "Team", text)], tokenizer=OffsetTokenizer()
    )["n1"]

    assert len(result.fact_candidates) == 1
    relation = result.fact_candidates[0]
    assert relation.assertion_start == 0
    assert text[relation.subject_start : relation.subject_end] == "Alice"
    assert text[relation.object_start : relation.object_end] == "Bob"


def test_tuple_relations_and_bare_string_spans_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L7: legacy shapes without confidence default to 0.0 and fail floors."""
    fake = FakeModel(
        {
            "legacy": {
                "entities": {
                    "person": ["Maya", {"text": "Bob Smith", "confidence": 0.9}],
                },
                "relation_extraction": {
                    "works_with": [["Maya", "Bob Smith"]],
                },
            }
        }
    )
    _install_fake(monkeypatch, fake)

    result = extract.extract_notes(
        [make_note("n1", "Legacy", "legacy Maya and Bob Smith work.")]
    )["n1"]

    assert [(row.surface, row.etype) for row in result.entities] == [
        ("Bob Smith", "person")
    ]
    assert result.fact_candidates == []
    assert result.relations == []


def test_family_schemas_gate_optional_predicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L21a: the core registry is always on; optional extras vary per family."""
    fake = FakeModel({})
    _install_fake(monkeypatch, fake)
    email = make_note("email", "Email", "body").model_copy(update={"source": "gmail"})
    conversation = make_note("chat", "Chat", "body").model_copy(
        update={"source": "imessage"}
    )
    written = make_note("note", "Note", "body")

    extract.extract_notes([email, conversation, written], tokenizer=OffsetTokenizer())

    email_schema, conversation_schema, written_schema = (
        set(schema.relation_types) for schema in fake.schemas
    )
    assert written_schema == set(config.INGEST_RELATION_TYPES)
    assert {"responsible_for", "scheduled_for", "blocked_by", "completed"} <= email_schema
    assert {"met_with", "decided", "prefers"}.isdisjoint(email_schema)
    assert {"met_with", "decided", "prefers"} <= conversation_schema
    assert {"responsible_for", "scheduled_for", "blocked_by", "completed"}.isdisjoint(
        conversation_schema
    )
    # the core registry never varies
    assert {"works_at", "works_with", "about", "related_to"} <= (
        email_schema & conversation_schema & written_schema
    )


def test_source_families_produce_distinct_stable_schema_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModel({})
    _install_fake(monkeypatch, fake)
    email = make_note("email", "Email", "body").model_copy(update={"source": "gmail"})
    conversation = make_note("chat", "Chat", "body").model_copy(update={"source": "imessage"})

    first = extract.extract_notes([email, conversation], tokenizer=OffsetTokenizer())
    second = extract.extract_notes([email, conversation], tokenizer=OffsetTokenizer())

    assert first["email"].schema_id != first["chat"].schema_id
    assert first["email"].schema_id == second["email"].schema_id
    assert len(fake.schemas) == 4
