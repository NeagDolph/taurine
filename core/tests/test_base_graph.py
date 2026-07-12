from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from taurine_core.indexer.base_graph import (
    BaseGraphBuilder,
    EntityGraphInput,
    PassageEntityInput,
    PassageGraphInput,
)
from taurine_core.indexer.facts import FactRecord


def test_base_graph_has_passage_entity_nodes_without_projection_or_note_nodes() -> None:
    graph = _builder().build(
        passages=(
            _passage("p1", "n1", 0, [1.0, 0.0]),
            _passage("p2", "n2", 0, [0.99, 0.01]),
            _passage("p3", "n1", 1, [1.0, 0.0]),
        ),
        entities=(EntityGraphInput("e1", "Maya", "person", 2, True),),
        contains=(PassageEntityInput("p1", "e1", 0.9, 1, 2),),
        facts=(),
    )

    assert set(graph.node_kinds) == {"passage", "entity"}
    assert all(not node_id.startswith("note:") for node_id in graph.node_ids)
    semantic_pairs = {
        tuple(sorted((edge.src_id, edge.dst_id)))
        for edge in graph.edges
        if edge.kind == "semantic"
    }
    assert ("p1", "p2") in semantic_pairs
    assert ("p1", "p3") not in semantic_pairs
    assert graph.matrix.shape == (4, 4)
    strengths = np.asarray(graph.matrix.sum(axis=0)).ravel()
    assert all(value == pytest.approx(1.0) for value in strengths if value > 0)


def test_contains_edges_use_idf_and_hubs_require_explicit_activation() -> None:
    builder = _builder()
    passages = (
        _passage("p1", "n1", 0, [1.0, 0.0]),
        _passage("p2", "n2", 0, [0.0, 1.0]),
    )
    entities = (
        EntityGraphInput("eligible", "Maya", "person", 1, True),
        EntityGraphInput("hub", "Neil", "person", 9, False),
    )
    contains = (
        PassageEntityInput("p1", "eligible", 0.8, 2, 1),
        PassageEntityInput("p1", "hub", 0.9, 1, 9),
    )

    default = builder.build(passages, entities, contains, ())
    activated = builder.build(passages, entities, contains, (), activate_entities={"hub"})

    default_contains = [edge for edge in default.edges if edge.kind == "contains"]
    activated_contains = [edge for edge in activated.edges if edge.kind == "contains"]
    assert {edge.dst_id for edge in default_contains if edge.src_id == "p1"} == {"eligible"}
    assert {edge.dst_id for edge in activated_contains if edge.src_id == "p1"} == {
        "eligible",
        "hub",
    }
    eligible = next(edge for edge in default_contains if edge.src_id == "p1")
    assert 0.0 < eligible.weight < 1.0
    assert eligible.metadata["idf"] > 1.0


def test_contains_weights_discriminate_rare_from_common_entities() -> None:
    """H2: after the CONTAINS_WEIGHT_SCALE rescale, idf rarity must survive
    into edge weights instead of every edge clipping to exactly 1.0."""
    note_count = 10
    passages = tuple(
        _passage(f"p{index}", f"n{index}", 0, [1.0, float(index) * 0.01])
        for index in range(note_count)
    )
    entities = (
        EntityGraphInput("rare", "Zephyr", "project", 1, True),
        EntityGraphInput("common", "Standup", "concept", 8, True),
    )
    contains = (
        PassageEntityInput("p0", "rare", 0.9, 1, 1),
        PassageEntityInput("p0", "common", 0.9, 1, 8),
    )

    graph = _builder().build(passages, entities, contains, ())

    weight_by_entity = {
        edge.dst_id: edge.weight
        for edge in graph.edges
        if edge.kind == "contains" and edge.src_id == "p0"
    }
    assert weight_by_entity["rare"] > weight_by_entity["common"]
    assert 0.0 < weight_by_entity["common"] < weight_by_entity["rare"] < 1.0
    # typical mid values land well inside (0, 1) — the old bug clipped both to 1.0
    assert weight_by_entity["rare"] < 0.9


def test_contains_weight_scale_saturates_only_at_the_realistic_maximum() -> None:
    """The formula's true maximum (corpus-rare entity, idf≈7, conf 1.0,
    bonus≈2.4) still reaches ~1.0 after sqrt compression; a rare entity in a
    small fixture corpus lands high but below saturation."""
    passages = tuple(
        _passage(f"p{index}", f"n{index}", 0, [1.0, float(index) * 0.01])
        for index in range(500)
    )
    entities = (EntityGraphInput("unique", "Nadir", "person", 1, True),)
    contains = (PassageEntityInput("p0", "unique", 1.0, 3, 0),)

    graph = _builder().build(passages, entities, contains, ())

    weight = next(
        edge.weight
        for edge in graph.edges
        if edge.kind == "contains" and edge.src_id == "p0"
    )
    assert weight > 0.95


def test_fact_edges_preserve_direction_and_weaker_reverse_weight() -> None:
    fact = _fact()
    graph = _builder().build(
        passages=(_passage("p1", "n1", 0, [1.0, 0.0]),),
        entities=(
            EntityGraphInput("person", "Maya", "person", 1, True),
            EntityGraphInput("org", "Wink", "organization", 1, True),
        ),
        contains=(),
        facts=(fact,),
    )

    forward = next(edge for edge in graph.edges if edge.kind == "fact" and edge.direction == "forward")
    reverse = next(edge for edge in graph.edges if edge.kind == "fact" and edge.direction == "reverse")
    assert (forward.src_id, forward.dst_id) == ("person", "org")
    assert (reverse.src_id, reverse.dst_id) == ("org", "person")
    assert reverse.weight == pytest.approx(forward.weight * 0.35)
    assert forward.fact_ids == ("fact-1",)


def test_fact_edges_group_logical_triples_without_counting_one_source_twice() -> None:
    fact = _fact()
    graph = _builder().build(
        passages=(
            _passage("p1", "n1", 0, [1.0, 0.0]),
            _passage("p2", "n2", 0, [0.0, 1.0]),
        ),
        entities=(
            EntityGraphInput("person", "Maya", "person", 2, True),
            EntityGraphInput("org", "Wink", "organization", 2, True),
        ),
        contains=(),
        facts=(
            replace(fact, fact_id="fact-low", confidence=0.6),
            replace(fact, fact_id="fact-high", confidence=0.9),
            replace(
                fact,
                fact_id="fact-independent",
                source_passage_id="p2",
                source_note_id="n2",
                confidence=0.5,
            ),
            replace(
                fact,
                fact_id="fact-inactive",
                source_passage_id="p2",
                source_note_id="n2",
                confidence=1.0,
                status="historical",
            ),
        ),
    )

    fact_edges = [edge for edge in graph.edges if edge.kind == "fact"]
    assert len(fact_edges) == 2
    forward = next(edge for edge in fact_edges if edge.direction == "forward")
    reverse = next(edge for edge in fact_edges if edge.direction == "reverse")
    assert forward.fact_ids == ("fact-high", "fact-independent", "fact-low")
    assert forward.metadata["source_note_ids"] == ("n1", "n2")
    # Same-note support uses max(0.6, 0.9); the independent note combines by noisy-or.
    assert forward.weight == pytest.approx(0.25 * (1.0 - (1.0 - 0.9) * (1.0 - 0.5)))
    assert reverse.weight == pytest.approx(forward.weight * 0.35)


def _builder() -> BaseGraphBuilder:
    return BaseGraphBuilder(build_id="build-1", semantic_k=1, semantic_floor=0.8)


def _passage(
    passage_id: str,
    note_id: str,
    ordinal: int,
    vector: list[float],
) -> PassageGraphInput:
    return PassageGraphInput(
        passage_id=passage_id,
        note_id=note_id,
        ordinal=ordinal,
        vector=np.asarray(vector, dtype=np.float32),
    )


def _fact() -> FactRecord:
    return FactRecord(
        fact_id="fact-1",
        subject_entity_id="person",
        subject="Maya",
        predicate="works_at",
        object_entity_id="org",
        object="Wink",
        source_passage_id="p1",
        source_note_id="n1",
        assertion_start=0,
        assertion_end=18,
        subject_start=0,
        subject_end=4,
        object_start=14,
        object_end=18,
        confidence=0.8,
        extraction_method="fake",
        schema_id="schema-1",
        observed_at=1.0,
        valid_from=None,
        valid_to=None,
        invalidated_at=None,
        superseded_by=None,
        status="active",
        build_id="build-1",
    )


def test_contains_weights_carry_frequency_surprise_beyond_the_soft_cap() -> None:
    """Item 3: frequent-but-uninformative entities (df far past DF_SOFT_CAP)
    weigh much less than specific ones at equal confidence and mentions."""
    passages = tuple(
        _passage(f"p{index:02d}", f"n{index:02d}", 0, [1.0, float(index) * 0.01])
        for index in range(50)
    )
    entities = (
        EntityGraphInput("specific", "Klaviyo", "organization", 2, True),
        EntityGraphInput("frequent", "Standup", "concept", 40, True),
    )
    contains = (
        PassageEntityInput("p00", "specific", 0.9, 1, 2),
        PassageEntityInput("p00", "frequent", 0.9, 1, 40),
    )

    graph = _builder().build(passages, entities, contains, ())

    edge_by_entity = {
        edge.dst_id: edge
        for edge in graph.edges
        if edge.kind == "contains" and edge.src_id == "p00"
    }
    specific = edge_by_entity["specific"]
    frequent = edge_by_entity["frequent"]
    # below the soft cap the surprise factor is exactly neutral
    assert specific.metadata["surprise"] == pytest.approx(1.0)
    assert frequent.metadata["surprise"] < 0.3
    # discrimination: the df=2 edge is decisively heavier than the df=40 edge
    assert specific.weight > 2 * frequent.weight
    # weights stay non-degenerate in (0, 1) — the H2 health gate keeps holding
    assert 0.0 < frequent.weight < specific.weight < 1.0


def test_semantic_edges_require_mutual_top_k_selection() -> None:
    """Item 5a: one-sided selections are dropped — an edge exists only when
    both endpoints pick each other inside their top-k."""
    import math as _math

    def unit(degrees: float) -> list[float]:
        return [_math.cos(_math.radians(degrees)), _math.sin(_math.radians(degrees))]

    builder = BaseGraphBuilder(build_id="build-1", semantic_k=1, semantic_floor=0.3)
    # pa's best is pb; pb's best is pc; pc's best is pb — the union rule
    # would keep pa-pb, the mutual rule keeps only pb-pc
    graph = builder.build(
        passages=(
            _passage("pa", "na", 0, unit(0.0)),
            _passage("pb", "nb", 0, unit(40.0)),
            _passage("pc", "nc", 0, unit(60.0)),
        ),
        entities=(),
        contains=(),
        facts=(),
    )

    semantic_pairs = {
        tuple(sorted((edge.src_id, edge.dst_id)))
        for edge in graph.edges
        if edge.kind == "semantic"
    }
    assert semantic_pairs == {("pb", "pc")}


def test_entity_hub_columns_forward_capped_mass_with_stochastic_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 3: an entity hub past HUB_DEGREE_CAP forwards only cap/degree of
    its mass; the rest parks on a self-loop so every column still sums to 1
    and the hub's neighbor ordering is preserved."""
    from taurine_core import config
    from taurine_core.indexer.base_graph import materialize_base_graph

    monkeypatch.setattr(config, "HUB_DEGREE_CAP", 4)
    passages = tuple(
        _passage(f"p{index}", f"n{index}", 0, [1.0, float(index) * 0.01])
        for index in range(6)
    )
    entities = (EntityGraphInput("hub", "Neil", "person", 6, True),)
    contains = tuple(
        PassageEntityInput(f"p{index}", "hub", 0.5 + 0.08 * index, 1, 6)
        for index in range(6)
    )

    builder = BaseGraphBuilder(build_id="build-1", semantic_k=0, semantic_floor=1.0)
    graph = builder.build(passages, entities, contains, ())

    strengths = np.asarray(graph.matrix.sum(axis=0)).ravel()
    assert all(value == pytest.approx(1.0) for value in strengths if value > 0)
    hub_row = graph.row_by_id["hub"]
    column = graph.matrix.getcol(hub_row).toarray().ravel()
    assert column[hub_row] == pytest.approx(1.0 - 4.0 / 6.0)
    neighbors = {
        graph.node_ids[index]: value
        for index, value in enumerate(column)
        if value > 0 and index != hub_row
    }
    assert sum(neighbors.values()) == pytest.approx(4.0 / 6.0)
    # uniform scaling preserves the ordering induced by the raw edge weights
    raw = {
        edge.dst_id: edge.weight
        for edge in graph.edges
        if edge.kind == "contains" and edge.src_id == "hub"
    }
    assert sorted(neighbors, key=neighbors.get) == sorted(raw, key=raw.get)
    # passage columns are untouched: each forwards its full unit of mass to
    # its single neighbor (the hub)
    for index in range(6):
        passage_row = graph.row_by_id[f"p{index}"]
        passage_column = graph.matrix.getcol(passage_row).toarray().ravel()
        assert passage_column.sum() == pytest.approx(1.0)
        assert passage_column[graph.row_by_id["hub"]] == pytest.approx(1.0)

    # materialize_base_graph (the query-time path) produces the identical
    # dampened matrix from raw edge rows — dampening is applied exactly once
    rebuilt = materialize_base_graph(
        build_id=graph.build_id,
        node_ids=graph.node_ids,
        node_kinds=graph.node_kinds,
        edges=graph.edges,
    )
    assert (rebuilt.matrix - graph.matrix).nnz == 0


def test_hub_dampening_ignores_small_entities_and_passage_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from taurine_core import config

    monkeypatch.setattr(config, "HUB_DEGREE_CAP", 4)
    passages = tuple(
        _passage(f"p{index}", f"n{index}", 0, [1.0, float(index) * 0.01])
        for index in range(4)
    )
    entities = (EntityGraphInput("e1", "Maya", "person", 4, True),)
    contains = tuple(
        PassageEntityInput(f"p{index}", "e1", 0.9, 1, 4) for index in range(4)
    )

    builder = BaseGraphBuilder(build_id="build-1", semantic_k=0, semantic_floor=1.0)
    graph = builder.build(passages, entities, contains, ())

    entity_row = graph.row_by_id["e1"]
    column = graph.matrix.getcol(entity_row).toarray().ravel()
    # degree 4 == cap: no self-loop, column normalized as before
    assert column[entity_row] == pytest.approx(0.0)
    assert column.sum() == pytest.approx(1.0)
