from __future__ import annotations

import os

import pytest

from taurine_core import config


class FakeModel:
    def __init__(self, entities: dict[str, list[object]]) -> None:
        self.entities = entities
        self.calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    def extract_entities(
        self,
        text: str,
        schema: dict[str, str],
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append((text, schema, kwargs))
        return {"entities": self.entities}


def test_gliner2_extractor_orders_by_schema_dedupes_and_caps(monkeypatch) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import Gliner2SpanExtractor

    model = FakeModel(
        {
            "topic": [
                {"text": "checkout flow", "confidence": 0.81},
                "Checkout Flow",
                "roadmap",
                "pricing",
                "sync",
            ],
            "person": ["Ashish"],
            "organization": [{"text": "Shopify", "confidence": 0.92}],
        }
    )
    monkeypatch.setattr(spans, "_MODEL", model)
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)
    monkeypatch.setattr(config, "GLINER_MAX_SPANS", 4)

    extractor = Gliner2SpanExtractor()
    result = extractor.extract(" ask ashish about checkout ")

    # schema order: person > organization > ... > topic; dedupe case-insensitive; cap 4
    assert result == [
        ("Ashish", 1.0, "person"),
        ("Shopify", 0.92, "organization"),
        ("checkout flow", 0.81, "topic"),
        ("roadmap", 1.0, "topic"),
    ]
    assert model.calls == [
        (
            "ask ashish about checkout",
            dict(config.GLINER2_SCHEMA),
            {"threshold": config.GLINER2_THRESHOLD, "include_confidence": True},
        )
    ]
    assert extractor.last_dropped == []


def test_gliner2_extractor_filters_junk_and_tracks_drops(monkeypatch) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import Gliner2SpanExtractor

    model = FakeModel(
        {
            "person": [
                "you",
                {"text": "dude", "confidence": 0.71},
                {"text": "Dan Ham", "confidence": 0.96},
            ],
            "organization": [
                {"text": "shopify", "confidence": 0.82},
                {"text": "Shopify", "confidence": 0.93},
            ],
            "project": [{"text": "lowercase initiative", "confidence": 0.74}],
            "topic": [
                {"text": "the people", "confidence": 0.67},
                {"text": "okay", "confidence": 0.66},
                {"text": "invoicing stuff", "confidence": 0.88},
            ],
        }
    )
    monkeypatch.setattr(spans, "_MODEL", model)
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)
    monkeypatch.setattr(config, "GLINER_MAX_SPANS", 10)
    monkeypatch.setattr(
        config,
        "GLINER_SPAN_STOPLIST",
        config.GLINER_SPAN_STOPLIST - {"dude"},
    )

    extractor = Gliner2SpanExtractor()

    assert extractor.extract("talk to Dan Ham about Shopify") == [
        ("Dan Ham", 0.96, "person"),
        ("Shopify", 0.93, "organization"),
        ("lowercase initiative", 0.74, "project"),
        ("invoicing stuff", 0.88, "topic"),
    ]
    assert extractor.last_dropped == [
        ("you", 1.0, "person"),
        ("dude", 0.71, "person"),
        ("shopify", 0.82, "organization"),
        ("the people", 0.67, "topic"),
        ("okay", 0.66, "topic"),
    ]


def test_gliner2_extractor_returns_empty_on_model_exception(monkeypatch) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import Gliner2SpanExtractor

    class RaisingModel:
        def extract_entities(self, *args, **kwargs) -> dict[str, object]:
            del args, kwargs
            raise RuntimeError("prediction failed")

    monkeypatch.setattr(spans, "_MODEL", RaisingModel())
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)

    assert Gliner2SpanExtractor().extract("checkout flow") == []


def test_gliner2_extractor_tolerates_malformed_result(monkeypatch) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import Gliner2SpanExtractor

    class WeirdModel:
        def extract_entities(self, *args, **kwargs) -> object:
            del args, kwargs
            return {"entities": {"person": "not-a-list", "topic": [42, {"no": "text"}]}}

    monkeypatch.setattr(spans, "_MODEL", WeirdModel())
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)

    assert Gliner2SpanExtractor().extract("checkout flow") == []


def test_gliner2_extractor_logs_model_load_failure_once(monkeypatch, caplog) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import Gliner2SpanExtractor

    def fail_load() -> object:
        raise RuntimeError("offline")

    monkeypatch.setattr(spans, "_MODEL", None)
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)
    monkeypatch.setattr(spans, "_load_model", fail_load)

    with caplog.at_level("WARNING", logger="taurine.session"):
        assert Gliner2SpanExtractor().extract("checkout flow") == []
        assert Gliner2SpanExtractor().extract("checkout flow") == []

    messages = [record.message for record in caplog.records]
    assert messages == ["GLiNER2 model load failed: offline"]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW is unset")
def test_real_gliner2_extractor_smoke() -> None:
    from taurine_core.live.spans import Gliner2SpanExtractor

    result = Gliner2SpanExtractor().extract(
        "we should ask ashish about the shopify checkout flow"
    )
    lowered = [span.lower() for span, _score, _label in result]

    assert any("ashish" in span for span in lowered)
    assert any("shopify" in span or "checkout" in span for span in lowered)
