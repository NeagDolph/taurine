from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from taurine_core import config
from taurine_core import download_models
from taurine_core import models_local
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.validate import (
    EvidenceValidator,
    _sentences,
    validate_evidence_output,
)


@dataclass
class FakeNLI:
    scores: dict[tuple[str, str], float]

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return self.scores.get((premise, hypothesis), 0.0)


def test_rejects_unknown_and_unsupplied_citations_without_stripping_claim() -> None:
    registry = EvidenceRegistry()
    registry.register("passage", "p1", {"text": "Launch is July 18."})
    validator = EvidenceValidator(registry, FakeNLI({}), entailment_floor=0.8)

    unknown = validator.validate(
        "Launch is July 18. [E99]", mode="chat", available_ids=("E1",)
    )
    unsupplied = validator.validate(
        "Launch is July 18. [E1]", mode="chat", available_ids=()
    )

    assert not unknown.accepted
    assert unknown.accepted_output == ""
    assert unknown.invalid_citations == ("E99",)
    assert unknown.unsupported_sentences == ("Launch is July 18. [E99]",)
    assert not unsupplied.accepted
    assert unsupplied.invalid_citations == ("E1",)
    assert unsupplied.repair_required


def test_explicit_turn_availability_excludes_evidence_read_on_prior_turns() -> None:
    registry = EvidenceRegistry()
    ref = registry.register("passage", "p1", {"text": "Launch is July 18."})
    registry.mark_read(ref)
    validator = EvidenceValidator(registry, FakeNLI({}), entailment_floor=0.8)

    result = validator.validate(
        "Launch is July 18. [E1]",
        mode="chat",
        available_ids=(),
    )

    assert not result.accepted
    assert result.invalid_citations == ("E1",)


@pytest.mark.parametrize("citation", ["", " [N1]", " [P1]"])
def test_factual_sentence_requires_e_f_or_t_coverage(citation: str) -> None:
    registry = EvidenceRegistry()
    registry.register("note", "n1", {"title": "Roadmap"})
    registry.register("path", "path1", {"rendered": "N1 -> E1"})

    result = EvidenceValidator(
        registry, FakeNLI({}), entailment_floor=0.8
    ).validate(
        f"Launch is July 18.{citation}", mode="chat", available_ids=("N1", "P1")
    )

    assert not result.accepted
    assert result.unsupported_sentences == (f"Launch is July 18.{citation}",)


def test_entailed_passage_is_accepted_and_contradiction_requires_repair() -> None:
    evidence = "The launch date is July 18."
    claim = "Launch is July 18."
    registry = _read_passage(evidence)

    entailed = EvidenceValidator(
        registry,
        FakeNLI({(evidence, claim): 0.92}),
        entailment_floor=0.8,
    ).validate(f"{claim} [E1]", mode="chat", available_ids=("E1",))
    contradicted = EvidenceValidator(
        registry,
        FakeNLI({(evidence, claim): 0.1}),
        entailment_floor=0.8,
    ).validate(f"{claim} [E1]", mode="chat", available_ids=("E1",))

    assert entailed.accepted
    assert entailed.accepted_output == f"{claim} [E1]"
    assert entailed.cited_ids == ("E1",)
    assert not contradicted.accepted
    assert contradicted.accepted_output == ""
    assert contradicted.repair_required
    assert contradicted.unsupported_sentences == (f"{claim} [E1]",)


def test_fact_requires_supplied_passage_provenance_and_uses_it_for_nli() -> None:
    evidence = "Maya owns Launch."
    claim = "Maya owns Launch."
    registry = _read_passage(evidence)
    registry.register(
        "fact",
        "owns-launch",
        {"source_evidence_ids": ["E1"], "status": "active"},
    )

    result = EvidenceValidator(
        registry,
        FakeNLI({(evidence, claim): 0.95}),
        entailment_floor=0.8,
    ).validate(f"{claim} [F1]", mode="chat", available_ids=("E1", "F1"))

    assert result.accepted
    assert result.cited_ids == ("F1",)

    broken = EvidenceRegistry()
    broken.register(
        "fact",
        "owns-launch",
        {"source_evidence_ids": ["E9"], "status": "active"},
    )
    invalid = EvidenceValidator(
        broken, FakeNLI({}), entailment_floor=0.8
    ).validate(f"{claim} [F1]", mode="chat", available_ids=("F1",))
    assert not invalid.accepted
    assert "F1" in invalid.invalid_citations


def test_fact_support_group_needs_at_least_one_supplied_source_not_every_source() -> None:
    evidence = "Maya owns Launch."
    registry = _read_passage(evidence)
    registry.register("passage", "p2", {"text": "A duplicate unsupplied assertion."})
    registry.register(
        "fact",
        "owns-launch",
        {"source_evidence_ids": ["E1", "E2"], "status": "active"},
    )

    result = EvidenceValidator(
        registry,
        FakeNLI({(evidence, "Maya owns Launch."): 0.95}),
        entailment_floor=0.8,
    ).validate("Maya owns Launch. [F1]", mode="chat", available_ids=("E1", "F1"))

    assert result.accepted


def test_current_claim_cannot_rely_only_on_superseded_fact() -> None:
    evidence = "Maya formerly worked at Wink."
    registry = _read_passage(evidence)
    registry.register(
        "fact",
        "maya-wink",
        {"source_evidence_ids": ["E1"], "status": "superseded"},
    )
    nli = FakeNLI(
        {
            (evidence, "Maya works at Wink."): 0.95,
            (evidence, "Maya formerly worked at Wink."): 0.95,
        }
    )
    validator = EvidenceValidator(registry, nli, entailment_floor=0.8)

    current = validator.validate(
        "Maya works at Wink. [F1]", mode="chat", available_ids=("E1", "F1")
    )
    historical = validator.validate(
        "Maya formerly worked at Wink. [F1]", mode="chat", available_ids=("E1", "F1")
    )

    assert not current.accepted
    assert current.degraded_reason == "temporal_inconsistency"
    assert historical.accepted


def test_nonfactual_headings_questions_code_and_uncertainty_need_no_citations() -> None:
    output = """# Findings

What should we review next?

```python
launch_date = "July 18"
```

I might be mistaken about the schedule.
"""
    result = EvidenceValidator(
        EvidenceRegistry(), FakeNLI({}), entailment_floor=0.8
    ).validate(output, mode="chat", available_ids=())

    assert result.accepted
    assert result.accepted_output == output
    assert result.unsupported_sentences == ()


def test_missing_nli_fails_live_closed_and_requires_chat_evidence_excerpt() -> None:
    registry = _read_passage("Launch is July 18.")
    validator = EvidenceValidator(registry, None, entailment_floor=0.8)

    live = validator.validate(
        "Launch is July 18. [E1]", mode="live", available_ids=("E1",)
    )
    chat = validator.validate(
        "Launch is July 18. [E1]", mode="chat", available_ids=("E1",)
    )

    assert not live.accepted
    assert live.accepted_output == ""
    assert live.degraded_reason == "nli_unavailable_live_fail_closed"
    assert not live.evidence_excerpt_required
    assert not chat.accepted
    assert chat.degraded_reason == "nli_unavailable_chat_excerpt_required"
    assert chat.evidence_excerpt_required
    assert not chat.repair_required


def test_local_nli_loader_never_falls_back_to_network(monkeypatch) -> None:
    monkeypatch.setattr(models_local, "local_model_path", lambda _repo_id: None)

    with pytest.raises(models_local.LocalModelUnavailable, match="bundled local model"):
        models_local.local_nli_scorer()


def test_local_nli_loader_is_fake_friendly_and_process_cached(monkeypatch) -> None:
    fake = FakeNLI({})
    constructed: list[str] = []
    monkeypatch.setattr(models_local, "_NLI_SCORERS", {})
    monkeypatch.setattr(
        models_local,
        "local_model_path",
        lambda _repo_id: "/bundled/nli-deberta",
    )
    monkeypatch.setattr(
        models_local,
        "_DebertaEntailmentScorer",
        lambda path: constructed.append(path) or fake,
    )

    assert models_local.local_nli_scorer("fake/nli") is fake
    assert models_local.local_nli_scorer("fake/nli") is fake
    assert constructed == ["/bundled/nli-deberta"]


def test_download_manifest_includes_nli_model(monkeypatch, tmp_path) -> None:
    downloaded: list[str] = []
    monkeypatch.setattr(download_models, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        download_models,
        "snapshot_download",
        lambda repo_id, *, local_dir: downloaded.append(repo_id),
    )

    assert download_models.main() == 0
    assert config.EVIDENCE_NLI_MODEL in downloaded


def test_available_ids_is_required_and_never_session_cumulative() -> None:
    """L13: no default availability — callers must scope the citable set."""

    registry = _read_passage("Launch is July 18.")
    validator = EvidenceValidator(registry, FakeNLI({}), entailment_floor=0.8)

    with pytest.raises(TypeError):
        validator.validate("Launch is July 18. [E1]", mode="chat")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        validate_evidence_output(  # type: ignore[call-arg]
            "Launch is July 18. [E1]",
            registry,
            mode="chat",
            nli_scorer=FakeNLI({}),
            entailment_floor=0.8,
        )


def test_validate_evidence_output_threads_available_ids() -> None:
    evidence = "Launch is July 18."
    registry = _read_passage(evidence)
    nli = FakeNLI({(evidence, "Launch is July 18."): 0.95})

    accepted = validate_evidence_output(
        "Launch is July 18. [E1]",
        registry,
        mode="chat",
        nli_scorer=nli,
        entailment_floor=0.8,
        available_ids=("E1",),
    )
    unavailable = validate_evidence_output(
        "Launch is July 18. [E1]",
        registry,
        mode="chat",
        nli_scorer=nli,
        entailment_floor=0.8,
        available_ids=(),
    )

    assert accepted.accepted
    assert not unavailable.accepted
    assert unavailable.invalid_citations == ("E1",)


def test_unsupported_details_carry_per_sentence_reasons() -> None:
    """L31: stripped claims persist with their rejection reason."""

    evidence = "Launch is July 18."
    registry = _read_passage(evidence)
    result = EvidenceValidator(
        registry,
        FakeNLI({(evidence, "The launch is tomorrow."): 0.0}),
        entailment_floor=0.8,
    ).validate(
        "The launch is tomorrow. [E1] The venue is unknown to me for now.",
        mode="chat",
        available_ids=("E1",),
    )

    assert not result.accepted
    assert ("The launch is tomorrow. [E1]", "entailment_below_floor") in result.unsupported_details
    assert (
        "The venue is unknown to me for now.",
        "citation_coverage",
    ) in result.unsupported_details


def _classify(text: str) -> bool:
    sentences, _visible = _sentences(text)
    assert len(sentences) == 1, text
    return sentences[0].factual


def test_sentence_classifier_fixture_accuracy_and_zero_bypass_exempts() -> None:
    """M12: ≥0.95 accuracy on the labeled set; bypass phrasings are NEVER
    exempted ("Here's what I found: <claim>" must classify factual)."""

    fixture = Path(__file__).parent / "fixtures" / "sentence_classes.jsonl"
    rows = [
        json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 40

    correct = 0
    for row in rows:
        predicted = _classify(str(row["text"]))
        if predicted == bool(row["factual"]):
            correct += 1
        if row.get("bypass"):
            assert predicted is True, f"false exempt on bypass: {row['text']}"
    accuracy = correct / len(rows)
    assert accuracy >= 0.95, f"classifier accuracy {accuracy:.3f} below 0.95"


def _read_passage(text: str) -> EvidenceRegistry:
    registry = EvidenceRegistry()
    registry.register("passage", "p1", {"text": text})
    registry.mark_read("E1")
    return registry
