from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pytest

from taurine_core import config
from taurine_core.indexer import graph
from taurine_core.schemas import Enrichment, EntityMention, GraphEdge, RelationTriple

from .conftest import entity, make_note


def _edge_between(index: object, left: str, right: str) -> GraphEdge | None:
    wanted = frozenset((left, right))
    for edge in index.graph().edges:  # type: ignore[attr-defined]
        if frozenset((edge.src, edge.dst)) == wanted:
            return edge
    return None


def _enrichment(
    gist: str,
    *,
    topics: Iterable[str] = (),
    entities: Iterable[str | EntityMention] = (),
    relations: Iterable[RelationTriple] = (),
) -> Enrichment:
    typed_entities = [entity(topic, "concept") for topic in topics]
    for item in entities:
        typed_entities.append(item if isinstance(item, EntityMention) else entity(item))
    return Enrichment(
        gist=gist,
        utterances=[gist],
        aliases=[],
        entities=typed_entities,
        relations=list(relations),
        extraction_failed=False,
    )


def _basis_vector(index: int) -> np.ndarray:
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    vector[index % config.EMBED_DIM] = 1.0
    return vector


def _hash_vector(term: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=config.EMBED_DIM).astype(np.float32)
    vector /= np.linalg.norm(vector)
    return vector


def _install_notes(index: object, rows: Iterable[tuple[str, Enrichment]]) -> None:
    for offset, (note_id, enrichment) in enumerate(rows):
        note = make_note(note_id, note_id.replace("-", " ").title(), enrichment.gist)
        vector = _basis_vector(offset)
        index.upsert_note(note, enrichment, [("chunk", enrichment.gist, vector)])  # type: ignore[attr-defined]


@pytest.fixture
def term_embedding_stub(monkeypatch: pytest.MonkeyPatch):
    base_vectors = {
        "recommendation engine": _basis_vector(100),
        "PR #370": _basis_vector(101),
        "PR #371": _basis_vector(101),
        "Shop LC": _basis_vector(102),
        "ShopLC": _basis_vector(102),
    }
    calls: list[tuple[str, ...]] = []

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        calls.append(tuple(texts))
        return np.vstack([base_vectors.get(text, _hash_vector(text)) for text in texts]).astype(
            np.float32
        )

    monkeypatch.setattr(graph, "embed_texts", fake_embed_texts, raising=False)
    return calls


def test_entity_separator_variants_share_one_canonical_term(store, term_embedding_stub):
    _install_notes(
        store,
        [
            (
                "engine-hyphen",
                _enrichment(
                    "Hyphenated recommendation engine.",
                    topics=["topic alpha"],
                    entities=["recommendation-engine", "Alpha Filler"],
                ),
            ),
            (
                "engine-underscore",
                _enrichment(
                    "Underscore recommendation engine.",
                    topics=["topic beta"],
                    entities=["recommendation_engine", "Beta Filler"],
                ),
            ),
        ],
    )

    store.rebuild_graph()

    edge = _edge_between(store, "engine-hyphen", "engine-underscore")
    assert edge is not None
    assert edge.type == "entity"
    assert edge.shared == ["recommendation engine"]
    assert len(term_embedding_stub) == 1


def test_concept_entities_create_topic_terms_and_other_entities_keep_case() -> None:
    enrichment = _enrichment(
        "Typed extraction.",
        topics=["Billing Flow"],
        entities=["Stripe Team"],
    )

    assert graph._terms(enrichment, "topic") == {"billing flow"}
    assert graph._terms(enrichment, "entity") == {"Stripe Team"}


def test_digit_guard_prevents_merge_for_terms_with_different_digit_sequences(
    store, term_embedding_stub
):
    _install_notes(
        store,
        [
            (
                "pr-370",
                _enrichment(
                    "Pull request 370.",
                    topics=["topic alpha"],
                    entities=["PR #370"],
                ),
            ),
            (
                "pr-371",
                _enrichment(
                    "Pull request 371.",
                    topics=["topic beta"],
                    entities=["PR #371"],
                ),
            ),
        ],
    )

    store.rebuild_graph()

    assert _edge_between(store, "pr-370", "pr-371") is None
    assert len(term_embedding_stub) == 1


def test_most_frequent_synonym_member_becomes_canonical(
    store, term_embedding_stub, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "ENTITY_SHARED_DF_FRACTION_CAP", 1.0, raising=False)
    rows: list[tuple[str, Enrichment]] = []
    rows.extend(
        (
            f"shop-lc-{index}",
            _enrichment(
                f"Shop LC spaced mention {index}.",
                topics=[f"topic spaced {index}"],
                entities=["Shop LC"],
            ),
        )
        for index in range(3)
    )
    rows.extend(
        (
            f"shoplc-{index}",
            _enrichment(
                f"ShopLC compact mention {index}.",
                topics=[f"topic compact {index}"],
                entities=["ShopLC"],
            ),
        )
        for index in range(5)
    )
    _install_notes(store, rows)

    store.rebuild_graph()

    edge = _edge_between(store, "shop-lc-0", "shoplc-0")
    assert edge is not None
    assert edge.shared == ["ShopLC"]


def test_term_vectors_cache_avoids_embedding_on_second_rebuild(store, term_embedding_stub):
    _install_notes(
        store,
        [
            (
                "shop-lc",
                _enrichment(
                    "Shop LC spaced mention.",
                    topics=["topic alpha"],
                    entities=["Shop LC"],
                ),
            ),
            (
                "shoplc",
                _enrichment(
                    "ShopLC compact mention.",
                    topics=["topic beta"],
                    entities=["ShopLC"],
                ),
            ),
        ],
    )

    store.rebuild_graph()
    first_call_count = len(term_embedding_stub)

    store.rebuild_graph()

    assert first_call_count == 1
    assert len(term_embedding_stub) == first_call_count


def test_lexical_gate_blocks_semantic_neighbors_that_are_distinct_referents() -> None:
    # embeddings put 'Cologne'/'Düsseldorf' at 0.957 — lexical compatibility
    # is what stops the merge (measured 2026-07-08)
    assert graph._lexically_compatible("Richard L Johnson", "Richard L. Johnson")
    assert graph._lexically_compatible("recommendation-engine", "recommendation_engine")
    assert graph._lexically_compatible("Back Bay Station (BBY)", "Back Bay Station")
    assert graph._lexically_compatible("flying dream", "flying dreams")
    assert not graph._lexically_compatible("Calumet Street", "Tremont Street")
    assert not graph._lexically_compatible("Cologne", "Düsseldorf")


def test_resolve_terms_respects_lexical_gate(store, term_embedding_stub, monkeypatch) -> None:
    # identical stub vectors (cosine 1.0) but lexically incompatible → no merge
    counts = {"Cologne": 3, "Düsseldorf": 2}
    monkeypatch.setattr(
        graph,
        "embed_texts",
        lambda texts: np.vstack([_basis_vector(0) for _ in texts]),
        raising=False,
    )
    mapping = graph.resolve_terms(store, counts)
    assert mapping["Cologne"] != mapping["Düsseldorf"]


def test_high_df_and_date_entities_do_not_create_shared_edges(
    store, term_embedding_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ENTITY_SHARED_DF_FRACTION_CAP", 0.5, raising=False)
    _install_notes(
        store,
        [
            (
                "alpha",
                _enrichment(
                    "Alpha note mentions the hub and one specific project.",
                    entities=[
                        entity("Neil", "person"),
                        entity("Launchpad", "project"),
                        entity("today", "date_or_deadline"),
                    ],
                ),
            ),
            (
                "beta",
                _enrichment(
                    "Beta note mentions the same project and date.",
                    entities=[
                        entity("Neil", "person"),
                        entity("Launchpad", "project"),
                        entity("today", "date_or_deadline"),
                    ],
                ),
            ),
            (
                "gamma",
                _enrichment(
                    "Gamma note only repeats the owner name.",
                    entities=[entity("Neil", "person")],
                ),
            ),
            (
                "delta",
                _enrichment(
                    "Delta note has unrelated content.",
                    entities=[entity("Unrelated Vendor", "organization")],
                ),
            ),
        ],
    )

    store.rebuild_graph()

    edge = _edge_between(store, "alpha", "beta")
    assert edge is not None
    assert edge.type == "entity"
    assert edge.shared == ["Launchpad"]
    assert _edge_between(store, "alpha", "gamma") is None
    assert _edge_between(store, "beta", "gamma") is None


def test_relation_candidates_skip_capped_relation_endpoints(
    store, term_embedding_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ENTITY_SHARED_DF_FRACTION_CAP", 0.5, raising=False)
    _install_notes(
        store,
        [
            (
                "assertion",
                _enrichment(
                    "Assertion note contains the relation.",
                    entities=[entity("Pankaj", "person"), entity("Neil", "person")],
                    relations=[
                        RelationTriple(
                            head="Pankaj",
                            rel_type="met_with",
                            tail="Neil",
                            conf=0.95,
                        )
                    ],
                ),
            ),
            (
                "pankaj-only",
                _enrichment(
                    "A note containing only Pankaj.",
                    entities=[entity("Pankaj", "person")],
                ),
            ),
            (
                "neil-only",
                _enrichment(
                    "A note containing only Neil.",
                    entities=[entity("Neil", "person")],
                ),
            ),
            (
                "neil-extra",
                _enrichment(
                    "Another note containing only Neil.",
                    entities=[entity("Neil", "person")],
                ),
            ),
        ],
    )

    store.rebuild_graph()

    assert _edge_between(store, "pankaj-only", "neil-only") is None


def test_relation_with_capped_endpoint_does_not_promote_fact(
    store, term_embedding_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ENTITY_SHARED_DF_FRACTION_CAP", 0.5, raising=False)
    _install_notes(
        store,
        [
            (
                "alpha",
                _enrichment(
                    "Alpha has a real below-cap shared project and a capped relation.",
                    entities=[
                        entity("Neil", "person"),
                        entity("Pankaj", "person"),
                        entity("Launchpad", "project"),
                    ],
                    relations=[
                        RelationTriple(
                            head="Neil",
                            rel_type="met_with",
                            tail="Pankaj",
                            conf=0.95,
                        )
                    ],
                ),
            ),
            (
                "beta",
                _enrichment(
                    "Beta repeats the below-cap shared project.",
                    entities=[
                        entity("Neil", "person"),
                        entity("Pankaj", "person"),
                        entity("Launchpad", "project"),
                    ],
                ),
            ),
            (
                "gamma",
                _enrichment(
                    "Gamma repeats the owner name only.",
                    entities=[entity("Neil", "person")],
                ),
            ),
            (
                "delta",
                _enrichment(
                    "Delta has unrelated content.",
                    entities=[entity("Unrelated Vendor", "organization")],
                ),
            ),
        ],
    )

    store.rebuild_graph()

    edge = _edge_between(store, "alpha", "beta")
    assert edge is not None
    assert "Launchpad" in edge.shared
    assert edge.fact == ""
    assert edge.fact_provenance == ""


def test_relation_fact_promotion_is_scoped_to_connected_notes(
    store, term_embedding_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ENTITY_SHARED_DF_FRACTION_CAP", 1.0, raising=False)
    people = [entity("Neil", "person"), entity("Pankaj", "person")]
    _install_notes(
        store,
        [
            (
                "alpha",
                _enrichment("Alpha contains both people.", entities=people),
            ),
            (
                "beta",
                _enrichment("Beta contains both people.", entities=people),
            ),
            (
                "gamma",
                _enrichment(
                    "Gamma alone asserts the relation.",
                    entities=people,
                    relations=[
                        RelationTriple(
                            head="Neil",
                            rel_type="met_with",
                            tail="Pankaj",
                            conf=0.95,
                        )
                    ],
                ),
            ),
        ],
    )

    store.rebuild_graph()

    edge = _edge_between(store, "alpha", "beta")
    assert edge is not None
    assert edge.fact == ""
    assert edge.fact_provenance == ""

    store.upsert_note(
        make_note("alpha", "Alpha", "Alpha now asserts the relation."),
        _enrichment(
            "Alpha now asserts the relation.",
            entities=people,
            relations=[
                RelationTriple(
                    head="Neil",
                    rel_type="met_with",
                    tail="Pankaj",
                    conf=0.95,
                )
            ],
        ),
        [("chunk", "Alpha now asserts the relation.", _basis_vector(0))],
    )
    store.rebuild_graph()

    edge = _edge_between(store, "alpha", "beta")
    assert edge is not None
    assert edge.fact == "Neil met with Pankaj"
    assert edge.fact_provenance == "gliner"
