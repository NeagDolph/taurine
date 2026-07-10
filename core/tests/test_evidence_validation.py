from __future__ import annotations

from dataclasses import dataclass

import pytest

from taurine_core import config
from taurine_core import download_models
from taurine_core import models_local
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.validate import (
    EvidenceValidator,
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
    validator = EvidenceValidator(registry)

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
    validator = EvidenceValidator(registry)

    result = validator.validate(
        "Launch is July 18. [E1]",
        mode="chat",
        available_ids=(),
    )

    assert not result.accepted
    assert result.invalid_citations == ("E1",)


@pytest.mark.parametrize("citation", ["", " [N1]", " [P1]"])
def test_uncited_and_navigation_cited_sentences_are_accepted(citation: str) -> None:
    """Per-sentence citation coverage (stage 3) was removed 2026-07-10:
    uncited prose is the model's judgment, not the validator's."""
    registry = EvidenceRegistry()
    registry.register("note", "n1", {"title": "Roadmap"})
    registry.register("path", "path1", {"rendered": "N1 -> E1"})

    result = EvidenceValidator(registry).validate(
        f"Launch is July 18.{citation}", mode="chat", available_ids=("N1", "P1")
    )

    assert result.accepted
    assert result.unsupported_sentences == ()


def test_cited_sentences_are_accepted_without_entailment_checks() -> None:
    """NLI entailment (stage 5) was removed 2026-07-10: a valid, available
    citation is sufficient; claim/evidence agreement is the model's job."""
    registry = _read_passage("The launch date is July 18.")

    result = EvidenceValidator(registry).validate(
        "Launch is cancelled forever. [E1]", mode="chat", available_ids=("E1",)
    )

    assert result.accepted
    assert result.cited_ids == ("E1",)


def test_fact_citation_requires_supplied_passage_provenance() -> None:
    claim = "Maya owns Launch."
    registry = _read_passage("Maya owns Launch.")
    registry.register(
        "fact",
        "owns-launch",
        {"source_evidence_ids": ["E1"], "status": "active"},
    )

    result = EvidenceValidator(registry).validate(
        f"{claim} [F1]", mode="chat", available_ids=("E1", "F1")
    )

    assert result.accepted
    assert result.cited_ids == ("F1",)

    broken = EvidenceRegistry()
    broken.register(
        "fact",
        "owns-launch",
        {"source_evidence_ids": ["E9"], "status": "active"},
    )
    invalid = EvidenceValidator(broken).validate(f"{claim} [F1]", mode="chat", available_ids=("F1",))
    assert not invalid.accepted
    assert "F1" in invalid.invalid_citations


def test_fact_support_group_needs_at_least_one_supplied_source_not_every_source() -> None:
    registry = _read_passage("Maya owns Launch.")
    registry.register("passage", "p2", {"text": "A duplicate unsupplied assertion."})
    registry.register(
        "fact",
        "owns-launch",
        {"source_evidence_ids": ["E1", "E2"], "status": "active"},
    )

    result = EvidenceValidator(registry).validate(
        "Maya owns Launch. [F1]", mode="chat", available_ids=("E1", "F1")
    )

    assert result.accepted


def test_current_claim_cannot_rely_only_on_superseded_fact() -> None:
    evidence = "Maya formerly worked at Wink."
    registry = _read_passage(evidence)
    registry.register(
        "fact",
        "maya-wink",
        {"source_evidence_ids": ["E1"], "status": "superseded"},
    )
    validator = EvidenceValidator(registry)

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
    result = EvidenceValidator(EvidenceRegistry()).validate(
        output, mode="chat", available_ids=()
    )

    assert result.accepted
    assert result.accepted_output == output
    assert result.unsupported_sentences == ()


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


def test_download_manifest_excludes_retired_nli_model(monkeypatch, tmp_path) -> None:
    downloaded: list[str] = []
    monkeypatch.setattr(download_models, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        download_models,
        "snapshot_download",
        lambda repo_id, *, local_dir, ignore_patterns=(): downloaded.append(repo_id),
    )

    assert download_models.main() == 0
    # NLI entailment validation was removed 2026-07-10; its model is no
    # longer part of the offline bundle.
    assert config.EVIDENCE_NLI_MODEL not in downloaded
    assert config.RERANK_MODEL in downloaded


def test_available_ids_is_required_and_never_session_cumulative() -> None:
    """L13: no default availability — callers must scope the citable set."""

    registry = _read_passage("Launch is July 18.")
    validator = EvidenceValidator(registry)

    with pytest.raises(TypeError):
        validator.validate("Launch is July 18. [E1]", mode="chat")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        validate_evidence_output(  # type: ignore[call-arg]
            "Launch is July 18. [E1]",
            registry,
            mode="chat",
        )


def test_validate_evidence_output_threads_available_ids() -> None:
    registry = _read_passage("Launch is July 18.")

    accepted = validate_evidence_output(
        "Launch is July 18. [E1]",
        registry,
        mode="chat",
        available_ids=("E1",),
    )
    unavailable = validate_evidence_output(
        "Launch is July 18. [E1]",
        registry,
        mode="chat",
        available_ids=(),
    )

    assert accepted.accepted
    assert not unavailable.accepted
    assert unavailable.invalid_citations == ("E1",)


def test_unsupported_details_carry_per_sentence_reasons() -> None:
    """L31: stripped claims persist with their rejection reason."""

    registry = _read_passage("Launch is July 18.")
    result = EvidenceValidator(registry).validate(
        "The launch is tomorrow. [E99] The venue is booked. [E1]",
        mode="chat",
        available_ids=(),
    )

    assert not result.accepted
    assert ("The launch is tomorrow. [E99]", "invalid_citation") in result.unsupported_details
    assert ("The venue is booked. [E1]", "evidence_unavailable") in result.unsupported_details


def _read_passage(text: str) -> EvidenceRegistry:
    registry = EvidenceRegistry()
    registry.register("passage", "p1", {"text": text})
    registry.mark_read("E1")
    return registry
