from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from taurine_core import config
from taurine_core.live.grounding import GroundingEngine
from taurine_core.schemas import GraphEdge, GraphNode, GraphResponse, NoteRecord

DAY = 24.0 * 60.0 * 60.0


def _note(
    note_id: str,
    *,
    source: str = "apple_notes",
    modified: float = 1_000.0,
    quality_rank: float = 0.0,
) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=f"Title {note_id}",
        modified=modified,
        created=modified,
        content_hash=f"hash-{note_id}",
        text=f"Body {note_id}",
        gist=f"Gist {note_id}",
        source=source,
        quality_rank=quality_rank,
    )


@dataclass
class GroundingFakeIndex:
    note_rows: dict[str, NoteRecord]
    edge_rows: list[GraphEdge] = field(default_factory=list)
    fact_hits: list[tuple[str, str, float, str]] = field(default_factory=list)
    version: int = 1
    notes_calls: int = 0

    def notes(self) -> list[NoteRecord]:
        self.notes_calls += 1
        return [self.note_rows[note_id] for note_id in sorted(self.note_rows)]

    def note(self, note_id: str) -> NoteRecord | None:
        return self.note_rows.get(note_id)

    def graph_version(self) -> int:
        return self.version

    def graph(self) -> GraphResponse:
        active = set(self.note_rows)
        degree = {note_id: 0 for note_id in active}
        edges: list[GraphEdge] = []
        for edge in self.edge_rows:
            if edge.src not in active or edge.dst not in active:
                continue
            degree[edge.src] += 1
            degree[edge.dst] += 1
            edges.append(edge)
        return GraphResponse(
            nodes=[
                GraphNode(
                    note_id=note.note_id,
                    title=note.title,
                    degree=degree[note.note_id],
                    source=note.source,
                )
                for note in self.note_rows.values()
            ],
            edges=edges,
        )

    def neighbor_edges(self, note_id: str) -> list[tuple[str, float, str]]:
        rows: list[tuple[str, float, str]] = []
        for edge in self.edge_rows:
            if edge.src == note_id and edge.dst in self.note_rows:
                rows.append((edge.dst, edge.weight, edge.fact))
            elif edge.dst == note_id and edge.src in self.note_rows:
                rows.append((edge.src, edge.weight, edge.fact))
        return sorted(rows, key=lambda row: row[1], reverse=True)

    def fact_dense_search(self, text: str, k: int) -> list[tuple[str, str, float, str]]:
        del text
        return self.fact_hits[:k]


def _ids(engine: GroundingEngine, fused: list[tuple[str, float]], now: float) -> list[str]:
    return [item.note_id for item in engine.on_pass("window", fused, [], now).items]


def test_seed_accumulator_decays_exact_hits_and_prunes(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 3)
    index = GroundingFakeIndex(
        {
            "dense": _note("dense"),
            "other": _note("other"),
            "exact": _note("exact"),
        }
    )
    engine = GroundingEngine(index)  # type: ignore[arg-type]

    engine.on_pass(
        "window",
        [("dense", 0.5), ("other", 0.25)],
        [("exact", "exact alias")],
        1_000.0,
    )
    assert engine._seeds["dense"] == pytest.approx(0.5)
    assert engine._seeds["other"] == pytest.approx(0.25)
    assert engine._seeds["exact"] == pytest.approx(0.5)

    engine._seeds["tiny"] = 1e-7
    engine.on_pass("window", [], [], 1_001.0)

    assert engine._seeds["dense"] == pytest.approx(0.5 * config.SEED_ACCUM_DECAY)
    assert engine._seeds["exact"] == pytest.approx(0.5 * config.SEED_ACCUM_DECAY)
    assert "tiny" not in engine._seeds

    engine.on_pass("window", [], [("exact", "exact alias")], 1_002.0)
    assert engine._seeds["exact"] >= 1.0


def test_ppr_lifts_note_connected_to_two_seeds_above_equal_relevance_note(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 4)
    now = 10_000.0
    index = GroundingFakeIndex(
        {
            "seed-a": _note("seed-a", modified=now),
            "seed-b": _note("seed-b", modified=now),
            "bridge": _note("bridge", modified=now),
            "unlinked": _note("unlinked", modified=now),
        },
        [
            GraphEdge(src="bridge", dst="seed-a", type="semantic", weight=1.0),
            GraphEdge(src="bridge", dst="seed-b", type="semantic", weight=1.0),
        ],
    )
    engine = GroundingEngine(index)  # type: ignore[arg-type]

    update = engine.on_pass(
        "window",
        [("seed-a", 1.0), ("seed-b", 1.0), ("bridge", 0.5), ("unlinked", 0.5)],
        [],
        now,
    )
    scores = {item.note_id: item.score for item in update.items}

    assert scores["bridge"] > scores["unlinked"]
    assert next(item for item in update.items if item.note_id == "bridge").why.ppr_mass > 0


def test_blend_applies_recency_source_quality_and_degenerate_minmax(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 4)
    now = 5_000_000.0
    index = GroundingFakeIndex(
        {
            "fresh-imessage": _note("fresh-imessage", source="imessage", modified=now),
            "old-imessage": _note("old-imessage", source="imessage", modified=now - 365 * DAY),
            "fresh-apple": _note("fresh-apple", source="apple_notes", modified=now),
            "quality": _note(
                "quality",
                source="apple_notes",
                modified=now,
                quality_rank=1.0,
            ),
        }
    )
    engine = GroundingEngine(index)  # type: ignore[arg-type]

    update = engine.on_pass(
        "window",
        [
            ("fresh-imessage", 1.0),
            ("old-imessage", 1.0),
            ("fresh-apple", 1.0),
            ("quality", 1.0),
        ],
        [],
        now,
    )
    scores = {item.note_id: item.score for item in update.items}

    assert engine.last_blend["fresh-imessage"]["rel"] == pytest.approx(1.0)
    assert scores["fresh-imessage"] > scores["old-imessage"]
    assert scores["fresh-apple"] > scores["fresh-imessage"]
    assert scores["quality"] > scores["fresh-apple"]
    assert engine.last_blend["quality"]["quality"] == pytest.approx(
        math.exp(config.QUALITY_LAMBDA)
    )


def test_hysteresis_keeps_incumbent_until_threshold_and_drops_deleted(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 1)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    now = 20_000.0
    index = GroundingFakeIndex(
        {
            "incumbent": _note("incumbent", modified=now),
            "challenger": _note("challenger", modified=now, quality_rank=0.5),
        }
    )
    engine = GroundingEngine(index)  # type: ignore[arg-type]

    assert _ids(engine, [("incumbent", 1.0)], now) == ["incumbent"]
    assert _ids(engine, [("incumbent", 1.0), ("challenger", 1.0)], now) == [
        "incumbent"
    ]

    index.note_rows["challenger"] = index.note_rows["challenger"].model_copy(
        update={"quality_rank": 1.0}
    )
    assert _ids(engine, [("incumbent", 1.0), ("challenger", 1.0)], now) == [
        "challenger"
    ]

    del index.note_rows["challenger"]
    index.version += 1
    assert _ids(engine, [("incumbent", 1.0), ("challenger", 1.0)], now) == [
        "incumbent"
    ]


def test_short_ids_are_stable_and_fact_ids_are_never_reused(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 2)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    now = 30_000.0
    index = GroundingFakeIndex(
        {
            "alpha": _note("alpha", modified=now),
            "beta": _note("beta", modified=now),
            "gamma": _note("gamma", modified=now),
        },
        [GraphEdge(src="alpha", dst="beta", type="semantic", weight=1.0, fact="AB fact")],
    )
    engine = GroundingEngine(index)  # type: ignore[arg-type]

    first = engine.on_pass("window", [("alpha", 1.0), ("beta", 1.0)], [], now)
    assert [(item.note_id, item.short_id) for item in first.items] == [
        ("alpha", "N1"),
        ("beta", "N2"),
    ]
    assert [(fact.src, fact.dst, fact.fact_id) for fact in first.facts] == [
        ("alpha", "beta", "F1")
    ]

    index.edge_rows = [
        GraphEdge(src="alpha", dst="gamma", type="semantic", weight=1.0, fact="AC fact")
    ]
    index.version += 1
    second = engine.on_pass("window", [("alpha", 1.0), ("gamma", 1.0)], [], now)
    assert [(fact.src, fact.dst, fact.fact_id) for fact in second.facts] == [
        ("alpha", "gamma", "F2")
    ]

    index.edge_rows = [
        GraphEdge(src="alpha", dst="beta", type="semantic", weight=1.0, fact="AB fact")
    ]
    index.version += 1
    third = engine.on_pass("window", [("alpha", 1.0), ("beta", 1.0)], [], now)
    assert next(item.short_id for item in third.items if item.note_id == "beta") == "N2"
    assert [(fact.src, fact.dst, fact.fact_id) for fact in third.facts] == [
        ("alpha", "beta", "F1")
    ]


def test_facts_payload_prioritizes_dense_hits_then_within_set_edges_and_caps(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 3)
    monkeypatch.setattr(config, "GROUNDING_FACTS_MAX", 2)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    now = 40_000.0
    index = GroundingFakeIndex(
        {
            "alpha": _note("alpha", modified=now),
            "beta": _note("beta", modified=now),
            "gamma": _note("gamma", modified=now),
        },
        [
            GraphEdge(src="alpha", dst="beta", type="semantic", weight=0.9, fact="AB edge"),
            GraphEdge(src="alpha", dst="gamma", type="semantic", weight=0.8, fact="AC edge"),
        ],
        fact_hits=[("beta", "gamma", 0.99, "BC dense")],
    )
    engine = GroundingEngine(index)  # type: ignore[arg-type]

    update = engine.on_pass(
        "window",
        [("alpha", 1.0), ("beta", 1.0), ("gamma", 1.0)],
        [],
        now,
    )

    assert [(fact.src, fact.dst, fact.text) for fact in update.facts] == [
        ("beta", "gamma", "BC dense"),
        ("alpha", "beta", "AB edge"),
    ]


def test_revision_only_changes_when_membership_or_order_changes(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROUNDING_TOP_K", 2)
    monkeypatch.setattr(config, "PPR_BLEND", 0.0)
    now = 50_000.0
    index = GroundingFakeIndex(
        {
            "alpha": _note("alpha", modified=now),
            "beta": _note("beta", modified=now),
        }
    )
    engine = GroundingEngine(index)  # type: ignore[arg-type]

    first = engine.on_pass("window", [("alpha", 1.0), ("beta", 1.0)], [], now)
    second = engine.on_pass("window", [("alpha", 1.0), ("beta", 1.0)], [], now)
    third = engine.on_pass("window", [("beta", 2.0), ("alpha", 1.0)], [], now)

    assert first.revision == 1
    assert second.revision == first.revision
    assert third.revision == first.revision + 1
