from __future__ import annotations

import json

import pytest

from taurine_core.retrieval.registry import EvidenceRegistry


def test_registry_assigns_stable_kind_ids_and_never_reuses_them() -> None:
    registry = EvidenceRegistry()

    assert registry.register("note", "note-1", {"title": "Roadmap"}) == "N1"
    assert registry.register("passage", "passage-1", {"note_id": "note-1"}) == "E1"
    assert registry.register("fact", "fact-key-1", {"fact_ids": ["fact-1"]}) == "F1"
    assert registry.register("path", "path-key-1", {}) == "P1"
    assert registry.register("transcript", "chunk-7", {"seq": 7}) == "T1"
    assert registry.register("passage", "passage-1", {"note_id": "note-1"}) == "E1"
    assert registry.register("passage", "passage-2", {}) == "E2"


def test_registry_tracks_retrieved_read_and_cited_as_monotonic_states() -> None:
    registry = EvidenceRegistry()
    ref = registry.register("passage", "passage-1", {})

    assert registry.refs("retrieved") == ("E1",)
    assert registry.refs("read") == ()
    registry.mark_read(ref)
    registry.mark_cited(ref)

    record = registry.resolve(ref)
    assert record is not None
    assert record.states == frozenset({"retrieved", "read", "cited"})
    assert registry.refs("read") == ("E1",)
    assert registry.refs("cited") == ("E1",)


def test_registry_snapshot_round_trip_preserves_counters_and_is_json_safe() -> None:
    registry = EvidenceRegistry()
    registry.register("passage", "passage-1", {"offsets": [10, 20]})
    registry.mark_read("E1")
    snapshot = registry.snapshot()

    restored = EvidenceRegistry.from_snapshot(json.loads(json.dumps(snapshot)))

    assert restored.resolve("E1") == registry.resolve("E1")
    assert restored.register("passage", "passage-2", {}) == "E2"
    assert restored.register("note", "note-1", {}) == "N1"


def test_registry_rejects_unknown_refs_and_identity_payload_mutation() -> None:
    registry = EvidenceRegistry()
    registry.register("fact", "logical-fact", {"fact_ids": ["f1"]})

    with pytest.raises(KeyError, match="unknown evidence reference"):
        registry.mark_read("E9")
    with pytest.raises(ValueError, match="different payload"):
        registry.register("fact", "logical-fact", {"fact_ids": ["f2"]})
