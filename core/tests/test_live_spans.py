from __future__ import annotations

import os

import pytest

from taurine_core import config


class FakeModel:
    def __init__(self, predictions: list[dict[str, object]]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[str, list[str], float]] = []

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float,
    ) -> list[dict[str, object]]:
        self.calls.append((text, labels, threshold))
        return self.predictions


def test_gliner_span_extractor_dedupes_thresholds_sorts_and_caps(monkeypatch) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import GlinerSpanExtractor

    model = FakeModel(
        [
            {"text": "Ashish", "score": 0.41},
            {"text": "ashish", "score": 0.91},
            {"text": "below threshold", "score": 0.37},
            {"text": "Shopify checkout", "score": 0.8},
            {"text": "Roadmap", "score": 0.7},
        ]
    )
    monkeypatch.setattr(spans, "_MODEL", model)
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)
    monkeypatch.setattr(config, "GLINER_THRESHOLD", 0.38)
    monkeypatch.setattr(config, "GLINER_MAX_SPANS", 2)

    result = GlinerSpanExtractor().extract(" ask ashish about checkout ")

    assert result == [("ashish", 0.91), ("Shopify checkout", 0.8)]
    assert model.calls == [
        ("ask ashish about checkout", config.GLINER_LABELS, 0.38)
    ]


def test_gliner_span_extractor_returns_empty_on_model_exception(monkeypatch) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import GlinerSpanExtractor

    class RaisingModel:
        def predict_entities(self, *args, **kwargs) -> list[dict[str, object]]:
            del args, kwargs
            raise RuntimeError("prediction failed")

    monkeypatch.setattr(spans, "_MODEL", RaisingModel())
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)

    assert GlinerSpanExtractor().extract("checkout flow") == []


def test_gliner_span_extractor_logs_model_load_failure_once(monkeypatch, caplog) -> None:
    from taurine_core.live import spans
    from taurine_core.live.spans import GlinerSpanExtractor

    def fail_load() -> object:
        raise RuntimeError("offline")

    monkeypatch.setattr(spans, "_MODEL", None)
    monkeypatch.setattr(spans, "_MODEL_LOAD_FAILED", False)
    monkeypatch.setattr(spans, "_load_model", fail_load)

    with caplog.at_level("WARNING", logger="taurine.session"):
        assert GlinerSpanExtractor().extract("checkout flow") == []
        assert GlinerSpanExtractor().extract("checkout flow") == []

    messages = [record.message for record in caplog.records]
    assert messages == ["GLiNER model load failed: offline"]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW is unset")
def test_real_gliner_span_extractor_smoke() -> None:
    from taurine_core.live.spans import GlinerSpanExtractor

    result = GlinerSpanExtractor().extract(
        "we should ask ashish about the shopify checkout flow"
    )
    lowered = [span.lower() for span, _score in result]

    assert any("ashish" in span for span in lowered)
    assert any("shopify" in span or "checkout" in span for span in lowered)
