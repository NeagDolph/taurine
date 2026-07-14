from __future__ import annotations

import re

import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.indexer import extract, extract_relex
from taurine_core.schemas import ExtractedMention, Extraction


class FakeRelexModel:
    def __init__(self, outputs: dict[str, tuple[list[dict], list[dict]]]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    def predict_relations(self, text: str, **kwargs):
        self.calls.append({"text": text, **kwargs})
        for marker, output in self.outputs.items():
            if marker in text:
                return output
        return [], []


class OffsetTokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def _install_model(monkeypatch: pytest.MonkeyPatch, model: FakeRelexModel) -> None:
    monkeypatch.setattr(extract_relex, "_relex_model", lambda: model)
    monkeypatch.setattr(config, "RELEX_ENTITY_THRESHOLD", 0.4)
    monkeypatch.setattr(config, "RELEX_RELATION_THRESHOLD", 0.8)
    monkeypatch.setattr(config, "RELEX_ADJACENCY_THRESHOLD", 0.55)
    monkeypatch.setattr(config, "RELEX_MAX_TOKENS", 2048)


def test_relex_normalizes_note_global_offsets_and_filters_endpoint_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Alice works at Acme. Yesterday Acme worked at Alice."
    alice_start = text.index("Alice")
    acme_start = text.index("Acme")
    second_acme = text.index("Acme", acme_start + 1)
    second_alice = text.index("Alice", alice_start + 1)
    model = FakeRelexModel(
        {
            "Alice": (
                [
                    {
                        "start": alice_start,
                        "end": alice_start + 5,
                        "text": "Alice",
                        "label": "person",
                        "score": 0.96,
                    },
                    {
                        "start": acme_start,
                        "end": acme_start + 4,
                        "text": "Acme",
                        "label": "organization",
                        "score": 0.94,
                    },
                    {
                        "start": second_acme,
                        "end": second_acme + 4,
                        "text": "Acme",
                        "label": "organization",
                        "score": 0.93,
                    },
                    {
                        "start": second_alice,
                        "end": second_alice + 5,
                        "text": "Alice",
                        "label": "person",
                        "score": 0.92,
                    },
                ],
                [
                    {
                        "head": {
                            "start": alice_start,
                            "end": alice_start + 5,
                            "text": "Alice",
                            "type": "person",
                            "entity_idx": 0,
                        },
                        "tail": {
                            "start": acme_start,
                            "end": acme_start + 4,
                            "text": "Acme",
                            "type": "organization",
                            "entity_idx": 1,
                        },
                        "relation": "works_at",
                        "score": 0.91,
                    },
                    {
                        "head": {
                            "start": second_acme,
                            "end": second_acme + 4,
                            "text": "Acme",
                            "type": "organization",
                            "entity_idx": 2,
                        },
                        "tail": {
                            "start": second_alice,
                            "end": second_alice + 5,
                            "text": "Alice",
                            "type": "person",
                            "entity_idx": 3,
                        },
                        "relation": "works_at",
                        "score": 0.99,
                    },
                ],
            )
        }
    )
    _install_model(monkeypatch, model)

    result = extract_relex.extract_notes(
        [make_note("n1", "Employment", text)], tokenizer=OffsetTokenizer()
    )["n1"]

    assert result.failed is False
    assert [
        (row.surface, row.etype, row.char_start, row.char_end)
        for row in result.mentions
    ] == [
        ("Alice", "person", alice_start, alice_start + 5),
        ("Acme", "organization", acme_start, acme_start + 4),
        ("Acme", "organization", second_acme, second_acme + 4),
        ("Alice", "person", second_alice, second_alice + 5),
    ]
    assert len(result.fact_candidates) == 1
    relation = result.fact_candidates[0]
    assert relation.predicate == "works_at"
    assert (relation.assertion_start, relation.assertion_end) == (
        alice_start,
        acme_start + 4,
    )
    assert relation.status == "active"
    assert relation.extraction_method == "relex"
    assert model.calls[0]["labels"] == list(config.INGEST_ENTITY_TYPES)
    assert model.calls[0]["relations"] == list(extract_relex.PREDICATES)
    assert model.calls[0]["flat_ner"] is True
    assert model.calls[0]["threshold"] == pytest.approx(0.4)
    assert model.calls[0]["adjacency_threshold"] == pytest.approx(0.55)
    assert model.calls[0]["relation_threshold"] == pytest.approx(0.8)
    assert model.calls[0]["multi_label"] is False
    assert "pairs_filter" not in model.calls[0]


def test_relex_leaves_historical_false_for_temporal_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Alice worked at Acme in 2020."
    model = FakeRelexModel(
        {
            "Alice": (
                [
                    {
                        "start": 0,
                        "end": 5,
                        "text": "Alice",
                        "label": "person",
                        "score": 0.9,
                    },
                    {
                        "start": 16,
                        "end": 20,
                        "text": "Acme",
                        "label": "organization",
                        "score": 0.9,
                    },
                ],
                [
                    {
                        "head": {
                            "start": 0,
                            "end": 5,
                            "text": "Alice",
                            "type": "person",
                            "entity_idx": 0,
                        },
                        "tail": {
                            "start": 16,
                            "end": 20,
                            "text": "Acme",
                            "type": "organization",
                            "entity_idx": 1,
                        },
                        "relation": "works_at",
                        "score": 0.95,
                    }
                ],
            )
        }
    )
    _install_model(monkeypatch, model)

    result = extract_relex.extract_notes([make_note("n1", "Past", text)])["n1"]

    assert result.fact_candidates[0].status == "active"
    assert result.fact_candidates[0].valid_to is None


def test_relex_respects_max_token_window(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeRelexModel({})
    _install_model(monkeypatch, model)
    monkeypatch.setattr(config, "RELEX_MAX_TOKENS", 2)

    extract_relex.extract_notes(
        [make_note("n1", "Long", "zero one two three")], tokenizer=OffsetTokenizer()
    )

    assert model.calls[0]["text"] == "zero one"


def test_relex_releases_device_cache_after_each_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeRelexModel({})
    _install_model(monkeypatch, model)
    releases: list[None] = []
    monkeypatch.setattr(
        extract_relex,
        "_release_device_cache",
        lambda: releases.append(None),
    )

    extract_relex.extract_notes(
        [make_note("n1", "One", "one"), make_note("n2", "Two", "two")]
    )

    assert len(releases) == 2


def test_relex_retries_mps_oom_on_cpu_with_the_same_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeviceModel(FakeRelexModel):
        def __init__(self):
            super().__init__({})
            self.device = "mps"

        def to(self, device: str):
            self.device = device
            return self

        def predict_relations(self, text: str, **kwargs):
            self.calls.append({"text": text, "device": self.device, **kwargs})
            if self.device == "mps":
                raise RuntimeError("MPS backend out of memory")
            return [], []

    model = DeviceModel()
    _install_model(monkeypatch, model)
    text = " ".join(f"word-{index}" for index in range(1200))

    result = extract_relex.extract_notes(
        [make_note("n1", "Fallback", text)], tokenizer=OffsetTokenizer()
    )["n1"]

    assert result.failed is False
    assert [call["device"] for call in model.calls] == ["mps", "cpu"]
    assert [len(call["text"].split()) for call in model.calls] == [1200, 1200]


def test_relex_model_load_failure_fails_open_for_every_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extract_relex, "_relex_model", lambda: None)

    results = extract_relex.extract_notes(
        [make_note("n1", "One", "one"), make_note("n2", "Two", "two")]
    )

    assert list(results) == ["n1", "n2"]
    assert all(
        row.failed and row.error == "relex unavailable" for row in results.values()
    )


def test_hybrid_keeps_gliner_mentions_plus_relex_relation_endpoints_only() -> None:
    base = Extraction(
        mentions=[
            ExtractedMention(
                surface="Alice",
                etype="person",
                confidence=0.97,
                char_start=0,
                char_end=5,
                extractor="fastino/gliner2-large-v1",
                schema_id="gliner-schema",
            )
        ],
        schema_id="gliner-schema",
    )
    relex = Extraction(
        mentions=[
            ExtractedMention(
                surface="Alice",
                etype="person",
                confidence=0.9,
                char_start=0,
                char_end=5,
                extractor=config.RELEX_MODEL,
                schema_id="relex-schema",
            ),
            ExtractedMention(
                surface="Project X",
                etype="project",
                confidence=0.88,
                char_start=25,
                char_end=34,
                extractor=config.RELEX_MODEL,
                schema_id="relex-schema",
            ),
            ExtractedMention(
                surface="Zebra",
                etype="concept",
                confidence=0.8,
                char_start=40,
                char_end=45,
                extractor=config.RELEX_MODEL,
                schema_id="relex-schema",
            ),
        ],
        fact_candidates=[_fixture_relation()],
        schema_id="relex-schema",
    )

    merged = extract_relex.merge_hybrid(base, relex)

    assert [(row.surface, row.extractor) for row in merged.mentions] == [
        ("Alice", "fastino/gliner2-large-v1"),
        ("Project X", config.RELEX_MODEL),
    ]
    assert merged.fact_candidates[0].extraction_method == "relex_hybrid"
    assert merged.fact_candidates[0].schema_id == merged.schema_id


def _fixture_relation():
    from taurine_core.schemas import ExtractedRelation

    return ExtractedRelation(
        head="Alice",
        predicate="responsible_for",
        tail="Project X",
        confidence=0.9,
        subject_start=0,
        subject_end=5,
        object_start=25,
        object_end=34,
        assertion_start=0,
        assertion_end=34,
        extraction_method="relex",
        schema_id="relex-schema",
        status="active",
    )


def test_dispatch_honors_default_and_explicit_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = make_note("n1", "One", "one")
    calls: list[str] = []

    def gliner(notes, *, tokenizer=None, progress=None):
        del tokenizer, progress
        calls.append("gliner2")
        return {row.note_id: Extraction(schema_id="gliner") for row in notes}

    def relex(notes, *, tokenizer=None, progress=None):
        del tokenizer, progress
        calls.append("relex")
        return {row.note_id: Extraction(schema_id="relex") for row in notes}

    monkeypatch.setattr(extract, "extract_notes", gliner)
    monkeypatch.setattr(extract_relex, "extract_notes", relex)
    monkeypatch.setattr(config, "EXTRACTOR_BACKEND", "gliner2")

    assert extract.extract_notes_dispatch([note])["n1"].schema_id == "gliner"
    assert (
        extract.extract_notes_dispatch([note], backend="relex")["n1"].schema_id
        == "relex"
    )
    assert extract.extract_notes_dispatch([note], backend="hybrid")[
        "n1"
    ].schema_id.startswith("hybrid-")
    assert calls == ["gliner2", "relex", "gliner2", "relex"]


def test_hybrid_relex_failure_degrades_to_gliner_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = make_note("n1", "One", "one")
    gliner_result = Extraction(schema_id="gliner")
    monkeypatch.setattr(config, "EXTRACTOR_BACKEND", "hybrid")
    monkeypatch.setattr(
        extract,
        "extract_notes",
        lambda notes, **kwargs: {row.note_id: gliner_result for row in notes},
    )
    monkeypatch.setattr(
        extract_relex,
        "extract_notes",
        lambda notes, **kwargs: {
            row.note_id: Extraction(failed=True, error="relex unavailable")
            for row in notes
        },
    )

    result = extract.extract_notes_dispatch([note])["n1"]

    assert result == gliner_result


def test_production_build_and_server_use_backend_dispatch() -> None:
    from taurine_core import server
    from taurine_core.indexer import build

    assert build.extract_notes is extract.extract_notes_dispatch
    assert server.extract_notes is extract.extract_notes_dispatch
