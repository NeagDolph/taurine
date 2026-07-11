from __future__ import annotations

import numpy as np
import pytest

from taurine_core.indexer.base_graph import (
    BaseGraphBuilder,
    EntityGraphInput,
    PassageEntityInput,
    PassageGraphInput,
)
from taurine_core.retrieval.ppr import TypedPPR


def test_ppr_redistributes_dangling_mass_and_preserves_total_probability() -> None:
    graph = BaseGraphBuilder(build_id="build-1", semantic_k=0, semantic_floor=1.0).build(
        passages=(
            _passage("p1", "n1", [1.0, 0.0]),
            _passage("p2", "n2", [0.0, 1.0]),
        ),
        entities=(),
        contains=(),
        facts=(),
    )

    result = TypedPPR(graph, damping=0.5, tolerance=1e-12, max_iterations=100).run(
        passage_reset={"p1": 1.0}
    )

    assert result.accepted is True
    assert sum(result.ranks.values()) == pytest.approx(1.0, abs=1e-10)
    assert result.ranks["p1"] == pytest.approx(1.0)
    assert result.ranks["p2"] == pytest.approx(0.0)
    assert result.passage_mass == pytest.approx(1.0)
    assert result.entity_mass == pytest.approx(0.0)
    assert result.contributions["p1"][0][0] == "reset"


def test_typed_reset_weights_compose_one_normalized_teleport() -> None:
    graph = BaseGraphBuilder(build_id="build-1", semantic_k=0, semantic_floor=1.0).build(
        passages=(
            _passage("p1", "n1", [1.0, 0.0]),
            _passage("p2", "n2", [0.0, 1.0]),
        ),
        entities=(EntityGraphInput("e1", "Maya", "person", 1, True),),
        contains=(PassageEntityInput("p1", "e1", 1.0, 1, 1),),
        facts=(),
    )

    result = TypedPPR(graph, damping=0.5).run(
        passage_reset={"p1": 1.0},
        recognized_entity_reset={"e1": 1.0},
        query_entity_reset={"e1": 1.0},
        session_reset={"p2": 1.0},
    )

    denominator = 1.0 + 1.0 + 0.5 + 0.35
    assert result.teleport["p1"] == pytest.approx(1.0 / denominator)
    assert result.teleport["e1"] == pytest.approx(1.5 / denominator)
    assert result.teleport["p2"] == pytest.approx(0.35 / denominator)
    assert sum(result.teleport.values()) == pytest.approx(1.0)
    assert sum(result.ranks.values()) == pytest.approx(1.0, abs=1e-8)


def test_empty_or_unknown_resets_return_explicit_empty_result() -> None:
    graph = BaseGraphBuilder(build_id="build-1", semantic_k=0, semantic_floor=1.0).build(
        passages=(_passage("p1", "n1", [1.0, 0.0]),),
        entities=(),
        contains=(),
        facts=(),
    )
    ppr = TypedPPR(graph)

    empty = ppr.run()
    unknown = ppr.run(passage_reset={"missing": 1.0})

    assert empty.accepted is False
    assert empty.reason == "empty_reset"
    assert empty.ranks == {}
    assert unknown.accepted is False
    assert unknown.reason == "empty_reset"


def _passage(passage_id: str, note_id: str, vector: list[float]) -> PassageGraphInput:
    return PassageGraphInput(
        passage_id=passage_id,
        note_id=note_id,
        ordinal=0,
        vector=np.asarray(vector, dtype=np.float32),
    )


def test_hub_dampening_caps_mass_through_a_synthetic_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 3: with the hub column dampened, less random-walk mass leaks from
    the seed through the hub to unrelated passages, and total probability and
    (sub-)column stochasticity are preserved."""
    from taurine_core import config

    def build_graph() -> object:
        passages = tuple(
            _passage(f"p{index}", f"n{index}", [1.0, float(index) * 0.01])
            for index in range(8)
        )
        entities = (EntityGraphInput("hub", "Neil", "person", 8, True),)
        contains = tuple(
            PassageEntityInput(f"p{index}", "hub", 0.9, 1, 8) for index in range(8)
        )
        return BaseGraphBuilder(
            build_id="build-1", semantic_k=0, semantic_floor=1.0
        ).build(passages, entities, contains, ())

    baseline = TypedPPR(build_graph(), damping=0.5).run(passage_reset={"p0": 1.0})
    monkeypatch.setattr(config, "HUB_DEGREE_CAP", 4)
    dampened_graph = build_graph()
    dampened = TypedPPR(dampened_graph, damping=0.5).run(passage_reset={"p0": 1.0})

    assert dampened.accepted is True
    assert sum(dampened.ranks.values()) == pytest.approx(1.0, abs=1e-8)
    strengths = np.asarray(dampened_graph.matrix.sum(axis=0)).ravel()
    assert all(value == pytest.approx(1.0) for value in strengths if value > 0)
    # unrelated leaves receive strictly less mass through the dampened hub
    assert dampened.ranks["p7"] < baseline.ranks["p7"]
    # the withheld mass parks on the hub's self-loop instead of spraying out
    assert dampened.ranks["hub"] > baseline.ranks["hub"]
