from __future__ import annotations

import math
import re
from collections.abc import Iterable

import numpy as np
import pytest

from tests.conftest import entity, fixture_units, make_note, unit_vec
from taurine_core import config
from taurine_core.indexer import graph
from taurine_core.ports import LLMError
from taurine_core.schemas import Enrichment, EntityMention, GraphEdge, RelationTriple


_PAIR_ID_RE = re.compile(r"\bpair-\d+\b")


def _basis(index: int) -> np.ndarray:
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    vector[index % config.EMBED_DIM] = 1.0
    return vector


def _cosine_to_anchor(cosine: float, orthogonal_index: int) -> np.ndarray:
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    vector[0] = cosine
    vector[orthogonal_index % config.EMBED_DIM] = math.sqrt(1.0 - cosine * cosine)
    return vector.astype(np.float32)


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


def _install(
    index: object,
    note_id: str,
    title: str,
    gist: str,
    vector: np.ndarray,
    *,
    topics: Iterable[str] = (),
    entities: Iterable[str] = (),
    relations: Iterable[RelationTriple] = (),
) -> None:
    enrich = _enrichment(gist, topics=topics, entities=entities, relations=relations)
    note = make_note(note_id, title, gist)
    index.upsert_note(note, enrich, [("chunk", enrich.gist, vector)])  # type: ignore[attr-defined]


def _edge_between(index: object, left: str, right: str) -> GraphEdge | None:
    wanted = frozenset((left, right))
    for edge in index.graph().edges:  # type: ignore[attr-defined]
        if frozenset((edge.src, edge.dst)) == wanted:
            return edge
    return None


def _pairs_with(note_id: str, candidates: Iterable[object]) -> set[str]:
    result: set[str] = set()
    for candidate in candidates:
        src = getattr(candidate, "src")
        dst = getattr(candidate, "dst")
        if note_id in (src, dst):
            result.add(dst if src == note_id else src)
    return result


def _candidate(src: str, dst: str, weight: float) -> graph._Candidate:
    candidate = graph._Candidate(*sorted((src, dst)))
    candidate.add("semantic", weight)
    return candidate


def _install_ids(store: object, note_ids: Iterable[str]) -> None:
    for offset, note_id in enumerate(note_ids):
        _install(
            store,
            note_id,
            note_id.upper(),
            f"{note_id} body.",
            _basis(700 + offset),
        )


def _replace_cliques(store: object) -> None:
    store.replace_edges(  # type: ignore[attr-defined]
        [
            GraphEdge(src="a0", dst="a1", type="semantic", weight=0.9),
            GraphEdge(src="a0", dst="a2", type="semantic", weight=0.9),
            GraphEdge(src="a1", dst="a2", type="semantic", weight=0.9),
            GraphEdge(src="b0", dst="b1", type="semantic", weight=0.9),
            GraphEdge(src="b0", dst="b2", type="semantic", weight=0.9),
            GraphEdge(src="b1", dst="b2", type="semantic", weight=0.9),
        ]
    )


class ScriptedAdjudicationLLM:
    def __init__(
        self,
        *,
        related: bool = True,
        fact: str = "Alpha and Beta are part of the same launch plan.",
        omit_after_first: bool = False,
        raise_times: int = 0,
    ) -> None:
        self.related = related
        self.fact = fact
        self.omit_after_first = omit_after_first
        self.raise_times = raise_times
        self.calls: list[tuple[str, str, str, float | None]] = []
        self.pair_counts: list[int] = []

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        self.calls.append((model, system, user, timeout_s))
        if self.raise_times > 0:
            self.raise_times -= 1
            raise LLMError("adjudication unavailable")
        pair_ids = list(dict.fromkeys(_PAIR_ID_RE.findall(user)))
        self.pair_counts.append(len(pair_ids))
        if self.omit_after_first and len(self.calls) == 1:
            pair_ids = pair_ids[:1]
        return {
            "pairs": {
                pair_id: {"related": self.related, "fact": self.fact}
                for pair_id in pair_ids
            }
        }


@pytest.fixture(autouse=True)
def deterministic_term_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    from taurine_core.indexer import store as store_module

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        rows = [_basis(900 + index) for index, _text in enumerate(texts)]
        return np.vstack(rows) if rows else np.empty((0, config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr(graph, "embed_texts", fake_embed_texts, raising=False)
    monkeypatch.setattr(store_module, "embed_texts", fake_embed_texts, raising=False)


def test_gliner_relation_promotes_fact_with_provenance_and_skips_queue(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from taurine_core.indexer.adjudicate import _pending_pairs

    monkeypatch.setattr(config, "ENTITY_SHARED_DF_FRACTION_CAP", 1.0, raising=False)
    relation = RelationTriple(
        head="Jane Doe",
        rel_type="works_at",
        tail="Acme",
        conf=0.82,
    )
    for note_id, relations in (("alpha", [relation]), ("beta", [])):
        _install(
            store,
            note_id,
            note_id.title(),
            f"{note_id} relation note.",
            _basis(100 if note_id == "alpha" else 101),
            entities=["Jane Doe", "Acme"],
            relations=relations,
        )

    store.rebuild_graph()

    edge = _edge_between(store, "alpha", "beta")
    assert edge is not None
    assert edge.fact == "Jane Doe works at Acme"
    assert edge.fact_provenance == "gliner"
    assert _pending_pairs(store) == []


def test_gate_excludes_non_bridge_low_value_and_keeps_bridge_low_value(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from taurine_core.indexer.adjudicate import _pending_pairs

    _install_ids(store, ["a0", "a1", "a2", "b0", "b1", "b2"])
    _replace_cliques(store)
    monkeypatch.setattr(
        graph,
        "_candidates",
        lambda _index: (
            [],
            [_candidate("a0", "a1", 0.52), _candidate("a2", "b0", 0.52)],
        ),
    )

    pending = _pending_pairs(store)

    assert [(pair.src, pair.dst) for pair in pending] == [("a2", "b0")]


def test_gate_budget_keeps_highest_ranked_pairs(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from taurine_core.indexer.adjudicate import _pending_pairs

    _install_ids(store, ["a0", "a1", "a2", "b0", "b1", "b2"])
    _replace_cliques(store)
    store.set_value_scores(
        {
            "a0": 0.90,
            "a1": 0.90,
            "a2": 0.65,
            "b0": 0.65,
            "b1": 0.10,
            "b2": 0.10,
        }
    )
    monkeypatch.setattr(config, "ADJ_VALUE_FRACTION", 1.0)
    monkeypatch.setattr(config, "ADJUDICATE_MAX_PAIRS_PER_SYNC", 2)
    monkeypatch.setattr(config, "BRIDGE_RANK_BOOST", 1.5)
    monkeypatch.setattr(
        graph,
        "_candidates",
        lambda _index: (
            [],
            [
                _candidate("a0", "a1", 0.52),
                _candidate("a2", "b0", 0.52),
                _candidate("b0", "b1", 0.52),
            ],
        ),
    )

    pending = _pending_pairs(store)

    assert [(pair.src, pair.dst) for pair in pending] == [
        ("a2", "b0"),
        ("a0", "a1"),
    ]


def test_value_threshold_quantile_and_zero_behavior(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from taurine_core.indexer.adjudicate import (
        _clears_value_threshold,
        _value_threshold,
    )

    monkeypatch.setattr(config, "ADJ_VALUE_FRACTION", 0.20)

    assert _value_threshold({}, set()) == pytest.approx(0.0)
    assert _value_threshold({"a": 0.0, "b": 0.0}, {"a", "b"}) == pytest.approx(0.0)
    assert not _clears_value_threshold(0.0, 0.0)
    assert _clears_value_threshold(0.01, 0.0)
    assert _value_threshold(
        {"a": 0.0, "b": 0.1, "c": 0.5, "d": 0.9, "e": 1.0},
        {"a", "b", "c", "d", "e"},
    ) == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_bands_and_prompt_caps_margin_and_fact_candidates(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from taurine_core.indexer.adjudicate import adjudicate_graph

    monkeypatch.setattr(config, "MARGIN_MAX_PER_NOTE", 3)
    monkeypatch.setattr(config, "FACT_TOP_EDGES_PER_NOTE", 1)
    store.set_user_card("## User\nWorks on the launch graph.")
    _install(
        store,
        "anchor",
        "Anchor",
        "Central launch graph note.",
        _basis(0),
        entities=[
            "strong-ab-1",
            "strong-ab-2",
            "strong-ac-1",
            "strong-bd-1",
            "strong-bd-2",
        ],
    )
    _install(
        store,
        "strong-b",
        "Strong B",
        "Strongest anchor neighbor.",
        _basis(10),
        entities=["strong-ab-1", "strong-ab-2", "strong-bd-1", "strong-bd-2"],
    )
    _install(
        store,
        "strong-c",
        "Strong C",
        "Lower anchor neighbor.",
        _basis(11),
        entities=["strong-ac-1"],
    )
    _install(
        store,
        "strong-d",
        "Strong D",
        "Competing neighbor for Strong B.",
        _basis(12),
        entities=["strong-bd-1", "strong-bd-2"],
    )
    for index in range(4):
        _install(
            store,
            f"weak-{index}",
            f"Weak {index}",
            f"Weak shared note {index}.",
            _basis(20 + index),
            entities=["weak-shared"],
        )
    for index, cosine in enumerate((0.54, 0.53, 0.52, 0.51, 0.50)):
        _install(
            store,
            f"margin-{index}",
            f"Margin {index}",
            f"Near semantic note {index}.",
            _cosine_to_anchor(cosine, 40 + index),
        )
    store.set_value_scores({"anchor": 1.0, "strong-b": 1.0})

    kept, margin = graph._candidates(store)

    strong_edges = {
        frozenset((candidate.src, candidate.dst)): candidate.edge()
        for candidate in kept
        if candidate.edge().weight >= config.VETO_MAX_W
    }
    weak_edges = [
        candidate.edge()
        for candidate in kept
        if candidate.edge().weight < config.VETO_MAX_W
    ]
    assert frozenset(("anchor", "strong-b")) in strong_edges
    assert any(edge.weight < config.VETO_MAX_W for edge in weak_edges)
    assert _pairs_with("anchor", margin) == {"margin-0", "margin-1", "margin-2"}

    llm = ScriptedAdjudicationLLM(fact="Anchor and Strong B are the highest-ranked fact edge.")

    await adjudicate_graph(store, llm)

    assert llm.calls
    prompt = llm.calls[0][2]
    assert "## User\nWorks on the launch graph." in prompt
    assert "anchor" in prompt
    assert "Anchor" in prompt
    assert "Left text: Central launch graph note." in prompt
    assert "Right text: Strongest anchor neighbor." in prompt
    assert "Left gist:" not in prompt
    assert "Right gist:" not in prompt
    assert "Strong B" in prompt
    assert "Strong C" not in prompt


@pytest.mark.asyncio
async def test_admits_margin_pair_and_attaches_fact(store) -> None:
    from taurine_core.indexer.adjudicate import adjudicate_graph

    _install(store, "alpha", "Alpha", "Alpha rollout.", _basis(0))
    _install(store, "beta", "Beta", "Beta rollout.", _cosine_to_anchor(0.52, 1))
    store.rebuild_graph()
    assert _edge_between(store, "alpha", "beta") is None

    report = await adjudicate_graph(
        store,
        ScriptedAdjudicationLLM(fact="Alpha and Beta describe the same rollout."),
    )

    edge = _edge_between(store, "alpha", "beta")
    assert report.admitted == 1
    assert edge is not None
    assert edge.fact == "Alpha and Beta describe the same rollout."


@pytest.mark.asyncio
async def test_veto_drops_weak_edge_but_strong_edge_survives(store) -> None:
    from taurine_core.indexer.adjudicate import adjudicate_graph

    _install(
        store,
        "strong-a",
        "Strong A",
        "Strong note A.",
        _basis(1),
        entities=["strong-only"],
    )
    _install(
        store,
        "strong-b",
        "Strong B",
        "Strong note B.",
        _basis(2),
        entities=["strong-only"],
    )
    for index in range(4):
        _install(
            store,
            f"weak-{index}",
            f"Weak {index}",
            f"Weak note {index}.",
            _basis(10 + index),
            entities=["weak-only"],
        )
    store.rebuild_graph()
    assert _edge_between(store, "strong-a", "strong-b") is not None
    assert _edge_between(store, "weak-0", "weak-1") is not None
    store.set_value_scores(
        {
            "strong-a": 1.0,
            "strong-b": 1.0,
            **{f"weak-{index}": 1.0 for index in range(4)},
        }
    )

    await adjudicate_graph(store, ScriptedAdjudicationLLM(related=False, fact=""))

    assert _edge_between(store, "strong-a", "strong-b") is not None
    assert _edge_between(store, "weak-0", "weak-1") is None


@pytest.mark.asyncio
async def test_facts_attach_to_strong_edges_and_unjudged_pairs_remain(store) -> None:
    from taurine_core.indexer.adjudicate import adjudicate_graph

    _install(
        store,
        "strong-a",
        "Strong A",
        "Strong note A.",
        _basis(1),
        entities=["strong-fact"],
    )
    _install(
        store,
        "strong-b",
        "Strong B",
        "Strong note B.",
        _basis(2),
        entities=["strong-fact"],
    )
    for index in range(4):
        _install(
            store,
            f"weak-{index}",
            f"Weak {index}",
            f"Weak note {index}.",
            _basis(20 + index),
            entities=["weak-unchanged"],
        )
    store.set_value_scores({"strong-a": 1.0, "strong-b": 1.0})

    await adjudicate_graph(
        store,
        ScriptedAdjudicationLLM(
            fact="Strong A and Strong B belong to the same adjudicated project.",
            omit_after_first=True,
        ),
    )

    strong = _edge_between(store, "strong-a", "strong-b")
    weak = _edge_between(store, "weak-0", "weak-1")
    assert strong is not None
    assert strong.fact == "Strong A and Strong B belong to the same adjudicated project."
    assert weak is not None
    assert weak.fact == ""


@pytest.mark.asyncio
async def test_cached_judgments_skip_llm_until_pair_signature_changes(store) -> None:
    from taurine_core.indexer.adjudicate import adjudicate_graph

    ab_enrichment = _enrichment("A-B project.", entities=["project-ab"])
    cd_enrichment = _enrichment("C-D project.", entities=["project-cd"])
    note_a = make_note("a", "A", "A text.")
    note_b = make_note("b", "B", "B text.")
    note_c = make_note("c", "C", "C text.")
    note_d = make_note("d", "D", "D text.")
    store.upsert_note(note_a, ab_enrichment, fixture_units(unit_vec(1), ab_enrichment))
    store.upsert_note(note_b, ab_enrichment, fixture_units(unit_vec(2), ab_enrichment))
    store.upsert_note(note_c, cd_enrichment, fixture_units(unit_vec(3), cd_enrichment))
    store.upsert_note(note_d, cd_enrichment, fixture_units(unit_vec(4), cd_enrichment))
    store.set_value_scores({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})
    llm = ScriptedAdjudicationLLM()

    await adjudicate_graph(store, llm)
    await adjudicate_graph(store, llm)
    changed_a = make_note("a", "A", "A text changed.")
    store.upsert_note(changed_a, ab_enrichment, fixture_units(unit_vec(1), ab_enrichment))
    await adjudicate_graph(store, llm)

    assert llm.pair_counts == [2, 1]


@pytest.mark.asyncio
async def test_missing_pair_response_stays_unjudged_and_retries_next_run(store) -> None:
    from taurine_core.indexer.adjudicate import adjudicate_graph

    _install(store, "alpha", "Alpha", "Alpha rollout.", _basis(0))
    _install(store, "beta", "Beta", "Beta rollout.", _cosine_to_anchor(0.52, 1))
    _install(store, "gamma", "Gamma", "Gamma rollout.", _cosine_to_anchor(0.51, 2))
    llm = ScriptedAdjudicationLLM(omit_after_first=True)

    await adjudicate_graph(store, llm)
    await adjudicate_graph(store, llm)

    assert llm.pair_counts == [2, 1]


def test_sync_source_runs_adjudication_and_keeps_sync_successful_on_failure(
    store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from taurine_core.indexer import extract
    from taurine_core.indexer import sync
    from taurine_core.indexer.store import Index
    from taurine_core.indexer.sync import sync_source

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        return np.vstack([unit_vec(index + 100) for index, _text in enumerate(texts)])

    class _Schema:
        def entities(self, _types: dict[str, str]) -> "_Schema":
            return self

        def relations(self, _types: list[str]) -> "_Schema":
            return self

    class SharedProjectExtractor:
        def create_schema(self) -> _Schema:
            return _Schema()

        def batch_extract(
            self,
            texts: list[str],
            schema: object,
            *,
            batch_size: int,
            threshold: float,
            include_confidence: bool,
            max_len: int | None,
        ) -> list[dict]:
            del schema, batch_size, threshold, include_confidence, max_len
            return [
                {
                    "entities": {
                        "project": [{"text": "Launch Project", "confidence": 0.95}]
                    },
                    "relation_extraction": {},
                }
                for _text in texts
            ]

    class SyncAndAdjudicateLLM:
        def __init__(self, *, fail_adjudication: bool = False) -> None:
            self.fail_adjudication = fail_adjudication
            self.adjudication_calls = 0

        async def complete_json(
            self, model: str, system: str, user: str, timeout_s: float | None = None
        ) -> dict:
            del system, timeout_s
            if model == config.ADJUDICATE_MODEL:
                self.adjudication_calls += 1
                if self.fail_adjudication:
                    raise LLMError("adjudication failed")
                pair_ids = list(dict.fromkeys(_PAIR_ID_RE.findall(user)))
                return {
                    "pairs": {
                        pair_id: {
                            "related": True,
                            "fact": "Alpha and Beta are connected by the launch plan.",
                        }
                        for pair_id in pair_ids
                    }
                }
            return {
                "notes": {
                    "alpha": {
                        "aliases": ["shared launch"],
                        "utterances": ["Discuss the shared launch work."],
                    },
                    "beta": {
                        "aliases": ["shared launch"],
                        "utterances": ["Discuss the shared launch work."],
                    },
                }
            }

    monkeypatch.setattr(sync, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(extract, "_extractor_model", lambda: SharedProjectExtractor())
    store.set_user_card("## User\nBuilds launch systems.")
    first_llm = SyncAndAdjudicateLLM()
    notes = [
        make_note("alpha", "Alpha", "Alpha body."),
        make_note("beta", "Beta", "Beta body."),
    ]
    progress: list[tuple[str, int, int]] = []

    report = sync_source(store, "test", notes, first_llm, lambda *args: progress.append(args))

    edge = _edge_between(store, "alpha", "beta")
    assert report.added == 2
    assert first_llm.adjudication_calls == 1
    assert edge is not None
    assert edge.fact == "Alpha and Beta are connected by the launch plan."
    assert any(phase == "adjudicating" for phase, _done, _total in progress)

    second_store = Index(tmp_path / "failure.db")
    monkeypatch.setattr(sync, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(extract, "_extractor_model", lambda: SharedProjectExtractor())
    failing_llm = SyncAndAdjudicateLLM(fail_adjudication=True)
    failure_report = sync_source(second_store, "test", notes, failing_llm)

    assert failure_report.added == 2
    assert failing_llm.adjudication_calls == 2
    assert _edge_between(second_store, "alpha", "beta") is not None
