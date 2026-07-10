from __future__ import annotations

import pytest

from taurine_core import config
from taurine_core.live.grounding import GroundingEngine
from taurine_core.schemas import LiveGroundingEvent
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
)


def test_accumulates_only_new_registered_evidence_without_local_ids() -> None:
    registry = EvidenceRegistry()
    first = _pack(registry, "one", passages=(("p1", "n1", 0.9),), fact="f1")
    second = _pack(
        registry,
        "two",
        passages=(("p1", "n1", 0.8), ("p2", "n2", 0.95)),
        fact="f2",
    )
    engine = GroundingEngine(registry)

    one = engine.on_pack(first)
    two = engine.on_pack(second)

    assert one.new_support_ids == ("E1", "F1")
    assert one.revision == 1
    assert two.new_support_ids == ("E2", "F2")
    assert two.revision == 2
    assert {row.evidence_id for row in two.passages} == {"E1", "E2"}
    assert {row.fact_ref for row in two.facts} == {"F1", "F2"}
    assert {row.note_ref for row in two.notes} == {"N1", "N2"}
    assert registry.refs("read") == ("N1", "N2", "E1", "E2", "F1", "F2")


def test_abstention_does_not_add_decay_or_relabel_prior_evidence() -> None:
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry)
    accepted = engine.on_pack(
        _pack(registry, "one", passages=(("p1", "n1", 0.9),), fact="f1")
    )
    state_before = engine.evidence_ids()
    abstained_pack = EvidencePack.abstained(
        RetrievalQuery("filler", "live", now=2.0), "below_live_floor"
    )

    abstained = engine.on_pack(abstained_pack)

    assert abstained.no_new_evidence
    assert abstained.new_support_ids == ()
    assert abstained.revision == accepted.revision
    assert engine.evidence_ids() == state_before
    assert abstained.passages == accepted.passages
    assert abstained.display_passages == accepted.display_passages
    assert abstained.support_passages == ()
    assert abstained.support_facts == ()
    assert abstained.decision.reason == "below_live_floor"


def test_confirmed_topic_shift_clears_prior_accumulator_and_display() -> None:
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry)
    first = engine.on_pack(
        _pack(registry, "one", passages=(("p1", "n1", 0.9),), fact="f1")
    )
    shifted_pack = _pack(
        registry, "two", passages=(("p2", "n2", 0.85),), fact="f2"
    )

    shifted = engine.on_pack(shifted_pack, topic_shift=True)

    assert shifted.revision == first.revision + 1
    assert {row.evidence_id for row in shifted.passages} == {"E2"}
    assert {row.fact_ref for row in shifted.facts} == {"F2"}
    assert [row.evidence_id for row in shifted.display_passages] == ["E2"]
    assert shifted.new_support_ids == ("E2", "F2")
    assert registry.resolve("E1") is not None


def test_hysteresis_changes_display_only_not_evidence_or_support_revision() -> None:
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry, display_limit=1, hysteresis=1.15)
    first = engine.on_pack(
        _pack(registry, "one", passages=(("p1", "n1", 0.9),))
    )
    close = engine.on_pack(
        _pack(
            registry,
            "two",
            passages=(("p1", "n1", 0.8), ("p2", "n2", 0.85)),
        )
    )
    decisive = engine.on_pack(
        _pack(
            registry,
            "three",
            passages=(("p1", "n1", 0.8), ("p2", "n2", 1.0)),
        )
    )

    assert [row.evidence_id for row in first.display_passages] == ["E1"]
    assert [row.evidence_id for row in close.display_passages] == ["E1"]
    assert [row.evidence_id for row in close.passages] == ["E2", "E1"]
    assert close.new_support_ids == ("E2",)
    assert [row.evidence_id for row in decisive.display_passages] == ["E2"]
    assert decisive.new_support_ids == ()
    assert decisive.no_new_evidence
    assert decisive.revision == close.revision


def test_rejects_accepted_pack_whose_evidence_was_not_registered_as_read() -> None:
    registry = EvidenceRegistry()
    pack = _pack(registry, "one", passages=(("p1", "n1", 0.9),))
    record = registry.resolve("E1")
    assert record is not None
    registry = EvidenceRegistry.from_snapshot(
        {
            **registry.snapshot(),
            "entries": [
                {**row, "states": ["retrieved"]} if row["ref"] == "E1" else row
                for row in registry.snapshot()["entries"]  # type: ignore[index]
            ],
        }
    )

    with pytest.raises(ValueError, match="supplied/read"):
        GroundingEngine(registry).on_pack(pack)


def test_accumulator_decays_adds_and_prunes_below_epsilon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "LIVE_ACCUM_DECAY", 0.5)
    monkeypatch.setattr(config, "LIVE_ACCUM_NOTE_CAP", 10.0)
    monkeypatch.setattr(config, "LIVE_ACCUM_EPSILON", 0.3)
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry)

    engine.on_pack(_pack(registry, "one", passages=(("p1", "n1", 1.0),)))
    assert engine.session_reset() == {"p1": 1.0}

    engine.on_pack(_pack(registry, "two", passages=(("p2", "n2", 1.0),)))
    assert engine.session_reset() == {"p1": 0.5, "p2": 1.0}

    # p1 decays to 0.25 < epsilon 0.3 → pruned from reset AND session state.
    update = engine.on_pack(_pack(registry, "three", passages=(("p2", "n2", 1.0),)))
    reset = engine.session_reset()
    assert "p1" not in reset
    assert reset["p2"] == pytest.approx(1.5)
    assert all(row.evidence_id != "E1" for row in update.passages)


def test_accumulator_caps_per_note_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LIVE_ACCUM_DECAY", 1.0)
    monkeypatch.setattr(config, "LIVE_ACCUM_NOTE_CAP", 1.5)
    monkeypatch.setattr(config, "LIVE_ACCUM_EPSILON", 0.02)
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry)

    for text in ("one", "two", "three"):
        engine.on_pack(_pack(registry, text, passages=(("p1", "n1", 1.0),)))

    assert engine.session_reset()["p1"] == pytest.approx(1.5)


def test_abstained_pass_neither_adds_nor_decays_accumulated_mass() -> None:
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry)
    engine.on_pack(_pack(registry, "one", passages=(("p1", "n1", 0.9),)))
    before = engine.session_reset()

    engine.on_pack(
        EvidencePack.abstained(RetrievalQuery("filler", "live", now=2.0), "below_live_floor")
    )

    assert engine.session_reset() == before


def test_topic_shift_clears_accumulated_mass() -> None:
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry)
    engine.on_pack(_pack(registry, "one", passages=(("p1", "n1", 0.9),)))

    engine.on_pack(
        _pack(registry, "two", passages=(("p2", "n2", 0.85),)), topic_shift=True
    )

    assert engine.session_reset() == {"p2": pytest.approx(0.85)}


def test_incumbent_display_entries_are_flagged_prior_context() -> None:
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry, display_limit=2, hysteresis=1.15)
    engine.on_pack(_pack(registry, "one", passages=(("p1", "n1", 0.9),)))
    update = engine.on_pack(_pack(registry, "two", passages=(("p2", "n2", 0.95),)))

    display_ids = {row.evidence_id for row in update.display_passages}
    assert display_ids == {"E1", "E2"}
    assert update.prior_context_ids == ("E1",)

    event = LiveGroundingEvent.from_domain(update)
    flags = {row.evidence_id: row.prior_context for row in event.display_passages}
    assert flags == {"E1": True, "E2": False}
    # accumulated (non-display) sections stay unflagged
    assert all(not row.prior_context for row in event.passages)


def test_abstained_pass_flags_every_display_entry_prior_context() -> None:
    registry = EvidenceRegistry()
    engine = GroundingEngine(registry)
    engine.on_pack(_pack(registry, "one", passages=(("p1", "n1", 0.9),)))

    update = engine.on_pack(
        EvidencePack.abstained(RetrievalQuery("filler", "live", now=2.0), "below_live_floor")
    )

    assert update.prior_context_ids == ("E1",)


def _pack(
    registry: EvidenceRegistry,
    text: str,
    *,
    passages: tuple[tuple[str, str, float], ...],
    fact: str | None = None,
) -> EvidencePack:
    note_refs: dict[str, str] = {}
    evidence_passages: list[EvidencePassage] = []
    evidence_notes: list[EvidenceNote] = []
    for passage_id, note_id, score in passages:
        passage_text = f"Evidence {passage_id}"
        note_ref = note_refs.get(note_id)
        if note_ref is None:
            note_ref = registry.register("note", note_id, {"note_id": note_id})
            registry.mark_read(note_ref)
            note_refs[note_id] = note_ref
            evidence_notes.append(
                EvidenceNote(note_ref, note_id, f"Title {note_id}", "apple_notes", 2.0, score)
            )
        evidence_id = registry.register(
            "passage",
            passage_id,
            {"passage_id": passage_id, "note_id": note_id, "text": passage_text},
        )
        registry.mark_read(evidence_id)
        evidence_passages.append(
            EvidencePassage(
                evidence_id,
                passage_id,
                note_id,
                f"Title {note_id}",
                "apple_notes",
                passage_text,
                0,
                len(passage_text),
                1.0,
                2.0,
                score,
                score,
            )
        )
    evidence_facts: list[EvidenceFact] = []
    if fact is not None:
        source_ids = tuple(row.evidence_id for row in evidence_passages)
        fact_ref = registry.register(
            "fact",
            fact,
            {"fact_ids": [fact], "source_evidence_ids": list(source_ids), "status": "active"},
        )
        registry.mark_read(fact_ref)
        evidence_facts.append(
            EvidenceFact(
                fact_ref,
                (fact,),
                "Maya",
                "owns_or_has",
                "Launch",
                "forward",
                "active",
                2.0,
                None,
                None,
                0.9,
                source_ids,
            )
        )
    query = RetrievalQuery(text, "live", now=2.0)
    return EvidencePack(
        RetrievalDecision(True, 0.9, "accepted"),
        query,
        tuple(evidence_passages),
        tuple(evidence_facts),
        (),
        tuple(evidence_notes),
        RetrievalTrace(original_query=text),
    )
