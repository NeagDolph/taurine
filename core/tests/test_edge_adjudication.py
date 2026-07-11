"""Grounded LLM edge adjudication (edge-quality items 1, 5b, 6).

Every lane is fail-open and cache-hard: no judge or a failed batch must
reproduce today's behavior exactly, and a build can never fail from
adjudication. The grounding contract (verbatim quotes, registry relabels,
digit guards) is enforced in code, not in the prompt.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.indexer.build_cache import BuildCache
from taurine_core.indexer.edge_adjudication import (
    SemanticPair,
    adjudicate_facts,
    adjudicate_merges,
    adjudicate_semantic_edges,
    merge_mention_contexts,
    _fact_batches,
)
from taurine_core.indexer.entities import GrayZonePair
from taurine_core.indexer.facts import promote_fact
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.passages import PassageDraft
from taurine_core.schemas import ExtractedMention, ExtractedRelation, Extraction


# -- fixtures ------------------------------------------------------------------

_TEXT = "Maya Chen works at Klaviyo now. Rex owns the Launch project checklist."


def _draft(
    passage_id: str = "p1",
    ordinal: int = 0,
    text: str = _TEXT,
    *,
    char_start: int = 0,
) -> PassageDraft:
    return PassageDraft(
        passage_id=passage_id,
        ordinal=ordinal,
        char_start=char_start,
        char_end=char_start + len(text),
        token_count=len(text.split()),
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def _fact(
    subject: str = "Maya Chen",
    predicate: str = "works_at",
    object_: str = "Klaviyo",
    *,
    note_id: str = "n1",
    passage_id: str = "p1",
    span: tuple[int, int] = (0, 31),
    status: str = "active",
    confidence: float = 0.9,
    observed_at: float = 1.0,
):
    relation = ExtractedRelation(
        head=subject,
        predicate=predicate,
        tail=object_,
        confidence=confidence,
        subject_start=span[0],
        subject_end=span[0] + len(subject),
        object_start=span[1] - len(object_) - 5,
        object_end=span[1] - 5,
        assertion_start=span[0],
        assertion_end=span[1],
        extraction_method="fake",
        schema_id="s1",
        status=status,
    )
    return promote_fact(
        relation,
        subject_entity_id=f"e-{subject}",
        object_entity_id=f"e-{object_}",
        source_passage_id=passage_id,
        source_note_id=note_id,
        observed_at=observed_at,
        build_id="b1",
    )


def _verdict(fact_id: str, *, score: float, quote: str, relabel=None) -> dict:
    return {
        "fact_key": fact_id,
        "support": "supported" if score >= 0.7 else ("partial" if score >= 0.3 else "unsupported"),
        "quote": quote,
        "significant": score >= 0.3,
        "direction_ok": True,
        "relabel": relabel,
        "score": score,
    }


class ScriptedFactJudge:
    """Returns per-fact verdicts from a fact_id → verdict map; counts calls."""

    def __init__(self, verdicts: dict[str, dict]):
        self.verdicts = verdicts
        self.calls = 0

    def __call__(self, system_prompt: str, payload: dict) -> dict:
        assert "verbatim" in system_prompt
        self.calls += 1
        return {
            "facts": [
                self.verdicts[row["fact_key"]]
                for row in payload["facts"]
                if row["fact_key"] in self.verdicts
            ]
        }


# -- item 1: fact adjudication -------------------------------------------------


def test_no_judge_skips_the_lane_and_marks_nothing_adjudicated() -> None:
    fact = _fact()
    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        judge=None,
    )
    assert outcome.facts == (fact,)
    assert outcome.adjudicated == 0
    assert outcome.unadjudicated == 1
    assert outcome.facts[0].adjudicated is False


def test_rejection_downweight_and_keep_bands() -> None:
    rejected = _fact("Maya Chen", "works_at", "Klaviyo")
    downweighted = _fact("Rex", "owns_or_has", "Launch", span=(32, 70))
    kept = _fact("Rex", "uses", "Launch", span=(32, 60))
    judge = ScriptedFactJudge(
        {
            rejected.fact_id: _verdict(rejected.fact_id, score=0.1, quote="Maya Chen works at Klaviyo"),
            downweighted.fact_id: _verdict(downweighted.fact_id, score=0.5, quote="Rex owns the Launch"),
            kept.fact_id: _verdict(kept.fact_id, score=0.9, quote="Rex owns the Launch project"),
        }
    )

    outcome = adjudicate_facts(
        [rejected, downweighted, kept],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        judge=judge,
    )

    by_id = {fact.fact_id: fact for fact in outcome.facts}
    assert by_id[rejected.fact_id].status == "rejected"
    assert by_id[rejected.fact_id].adjudication_score == pytest.approx(0.1)
    assert by_id[rejected.fact_id].adjudication_quote == "Maya Chen works at Klaviyo"
    assert by_id[downweighted.fact_id].status == "active"
    assert by_id[downweighted.fact_id].adjudication_score == pytest.approx(0.5)
    assert by_id[kept.fact_id].status == "active"
    assert all(by_id[f.fact_id].adjudicated for f in (rejected, downweighted, kept))
    assert outcome.rejected_by_predicate == {"works_at": 1}
    assert outcome.downweighted_by_predicate == {"owns_or_has": 1}
    assert outcome.kept_by_predicate == {"uses": 1}
    assert outcome.adjudicated == 3
    assert outcome.unadjudicated == 0


def test_quote_must_appear_verbatim_or_the_row_fails_open() -> None:
    fact = _fact()
    fabricated = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.1, quote="Maya was fired from Klaviyo")}
    )

    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        judge=fabricated,
    )

    (result,) = outcome.facts
    assert result.status == "active"  # fail-open keep
    assert result.adjudicated is False
    assert outcome.malformed_rows == 1
    assert outcome.adjudicated == 0


def test_quote_matching_is_whitespace_normalized() -> None:
    fact = _fact()
    judge = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.9, quote="Maya Chen  works\nat Klaviyo")}
    )

    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        judge=judge,
    )

    assert outcome.facts[0].adjudicated is True


def test_relabel_re_derives_identity_and_reruns_reconciliation() -> None:
    # two active works_at objects for one subject: the relabeled fact joins
    # the exclusive group and temporal reconciliation reruns over it
    old = _fact("Maya Chen", "works_at", "OldCorp", note_id="n1", observed_at=10.0)
    mislabeled = _fact(
        "Maya Chen", "member_of", "Klaviyo", note_id="n2", span=(0, 31), observed_at=20.0
    )
    judge = ScriptedFactJudge(
        {
            old.fact_id: _verdict(old.fact_id, score=0.9, quote="works at"),
            mislabeled.fact_id: _verdict(
                mislabeled.fact_id, score=0.9, quote="works at Klaviyo", relabel="works_at"
            ),
        }
    )

    outcome = adjudicate_facts(
        [old, mislabeled],
        passages_by_note={"n1": [_draft()], "n2": [_draft("p1-n2")]},
        note_titles={},
        judge=judge,
    )

    assert outcome.relabeled_by_predicate == {"member_of": 1}
    relabeled = [f for f in outcome.facts if f.predicate == "works_at" and f.object == "Klaviyo"]
    assert len(relabeled) == 1
    assert relabeled[0].fact_id != mislabeled.fact_id
    assert relabeled[0].status == "active"
    # exclusivity: the older employer is superseded by the relabeled fact
    superseded = [f for f in outcome.facts if f.object == "OldCorp"]
    assert superseded[0].status == "superseded"
    assert superseded[0].superseded_by == relabeled[0].fact_id


def test_relabel_outside_the_registry_is_malformed() -> None:
    fact = _fact()
    judge = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.9, quote="works at", relabel="reports_to")}
    )

    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={},
        judge=judge,
    )

    assert outcome.facts[0].adjudicated is False
    assert outcome.malformed_rows == 1


def test_judge_exception_fails_open_and_counts_the_batch() -> None:
    fact = _fact()

    def broken(system_prompt: str, payload: dict) -> dict:
        raise TimeoutError("judge down")

    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={},
        judge=broken,
    )

    assert outcome.facts[0].status == "active"
    assert outcome.failed_batches == 1
    assert outcome.adjudicated == 0


def test_non_candidate_statuses_are_not_judged() -> None:
    superseded = _fact(status="active")
    superseded = superseded.__class__(**{**superseded.__dict__, "status": "superseded"})
    judge = ScriptedFactJudge({})

    outcome = adjudicate_facts(
        [superseded],
        passages_by_note={"n1": [_draft()]},
        note_titles={},
        judge=judge,
    )

    assert judge.calls == 0
    assert outcome.facts[0].status == "superseded"


def test_batches_group_by_note_cap_facts_and_bound_text(monkeypatch) -> None:
    monkeypatch.setattr(config, "ADJUDICATION_FACTS_PER_CALL", 2)
    monkeypatch.setattr(config, "ADJUDICATION_TEXT_MAX_CHARS", 200)
    drafts = {"n1": [_draft("p1", 0), _draft("p2", 1, char_start=100)], "n2": [_draft("p3")]}
    facts = [
        _fact("Maya Chen", "works_at", "Klaviyo", note_id="n1", passage_id="p1"),
        _fact("Rex", "owns_or_has", "Launch", note_id="n1", passage_id="p1", span=(32, 70)),
        _fact("Rex", "uses", "Launch", note_id="n1", passage_id="p2", span=(132, 160)),
        _fact("Maya Chen", "works_at", "Klaviyo", note_id="n2", passage_id="p3"),
    ]
    draft_by_passage = {d.passage_id: d for rows in drafts.values() for d in rows}

    batches = _fact_batches(facts, draft_by_passage, {"n1": "One", "n2": "Two"})

    assert [batch.note_id for batch in batches] == ["n1", "n1", "n2"]
    assert [len(batch.facts) for batch in batches] == [2, 1, 1]
    assert all(len(batch.text) <= 200 for batch in batches)
    # the shared text is supplied once per call and contains the facts' passages
    assert _TEXT in batches[0].text


def test_cached_judgments_skip_the_llm(tmp_path: Path) -> None:
    fact = _fact()
    cache = BuildCache(tmp_path / "cache.db")
    judge = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.5, quote="works at Klaviyo")}
    )
    kwargs = dict(
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        cache=cache,
    )

    first = adjudicate_facts([fact], judge=judge, **kwargs)
    assert judge.calls == 1
    second = adjudicate_facts([fact], judge=judge, **kwargs)

    assert judge.calls == 1  # replayed from the cache
    assert first.facts == second.facts
    assert second.facts[0].adjudication_score == pytest.approx(0.5)
    cache.close()


def test_trusted_evergreen_facts_skip_the_judge() -> None:
    # high-confidence, non-exclusive predicate, evergreen source -> trusted (skip)
    trusted = _fact("Rex", "owns_or_has", "Launch", span=(32, 70), confidence=0.95)
    # exclusive predicate is always judged even when high-confidence + evergreen
    exclusive = _fact("Maya Chen", "works_at", "Klaviyo", confidence=0.95)
    judge = ScriptedFactJudge(
        {exclusive.fact_id: _verdict(exclusive.fact_id, score=0.9, quote="works at Klaviyo")}
    )
    outcome = adjudicate_facts(
        [trusted, exclusive],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        judge=judge,
        note_sources={"n1": "apple_notes"},
    )
    # only the exclusive fact was sent to the judge
    assert judge.calls == 1
    by_id = {fact.fact_id: fact for fact in outcome.facts}
    assert by_id[trusted.fact_id].adjudicated is False
    assert by_id[exclusive.fact_id].adjudicated is True


def test_episodic_source_is_always_judged() -> None:
    # high-conf, non-exclusive, but from an episodic (chat) source -> judged
    episodic = _fact("Rex", "owns_or_has", "Launch", span=(32, 70), confidence=0.95)
    judge = ScriptedFactJudge({})
    adjudicate_facts(
        [episodic],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Chat"},
        judge=judge,
        note_sources={"n1": "imessage"},
    )
    assert judge.calls == 1


def test_low_confidence_facts_are_always_judged() -> None:
    # evergreen + non-exclusive but low confidence -> judged
    low_conf = _fact("Rex", "owns_or_has", "Launch", span=(32, 70), confidence=0.6)
    judge = ScriptedFactJudge({})
    adjudicate_facts(
        [low_conf],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        judge=judge,
        note_sources={"n1": "apple_notes"},
    )
    assert judge.calls == 1


def test_no_note_sources_judges_everything() -> None:
    # backward compatible: without note_sources, all eligible facts are judged
    fact = _fact("Rex", "owns_or_has", "Launch", span=(32, 70), confidence=0.95)
    judge = ScriptedFactJudge({})
    adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={"n1": "Work"},
        judge=judge,
    )
    assert judge.calls == 1


def test_malformed_verdicts_are_never_cached(tmp_path: Path) -> None:
    fact = _fact()
    cache = BuildCache(tmp_path / "cache.db")
    fabricated = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.1, quote="not in the text at all")}
    )

    adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={},
        judge=fabricated,
        cache=cache,
    )
    honest = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.9, quote="works at Klaviyo")}
    )
    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft()]},
        note_titles={},
        judge=honest,
        cache=cache,
    )

    assert honest.calls == 1  # the malformed verdict did not poison the cache
    assert outcome.facts[0].adjudicated is True
    cache.close()


# -- item 5b: semantic-edge reasons ----------------------------------------------


def _pair(src: str = "pa", dst: str = "pb") -> SemanticPair:
    return SemanticPair(
        src_passage_id=src,
        dst_passage_id=dst,
        src_title="Alpha",
        dst_title="Beta",
        src_text="Potluck planning with Adi on Saturday at the park.",
        dst_text="Bring the grill for the Saturday park potluck.",
        src_content_hash=hashlib.sha256(src.encode()).hexdigest(),
        dst_content_hash=hashlib.sha256(dst.encode()).hexdigest(),
    )


class ScriptedReasonJudge:
    def __init__(self, reason_by_title_pair: dict[tuple[str, str], str]):
        self.reasons = reason_by_title_pair
        self.calls = 0

    def __call__(self, system_prompt: str, payload: dict) -> dict:
        assert "NONE" in system_prompt
        self.calls += 1
        out = {}
        for row in payload["pairs"]:
            key = (row["a"]["title"], row["b"]["title"])
            if key in self.reasons:
                out[row["key"]] = self.reasons[key]
        return {"pairs": out}


def test_semantic_reasons_are_kept_clamped_and_none_drops_the_edge() -> None:
    reasoned = _pair("pa", "pb")
    dropped = SemanticPair(
        src_passage_id="pc",
        dst_passage_id="pd",
        src_title="Gamma",
        dst_title="Delta",
        src_text="x",
        dst_text="y",
        src_content_hash="h3",
        dst_content_hash="h4",
    )
    judge = ScriptedReasonJudge(
        {
            ("Alpha", "Beta"): "both plan the Saturday park potluck with Adi " + "word " * 20,
            ("Gamma", "Delta"): "NONE",
        }
    )

    outcome = adjudicate_semantic_edges([reasoned, dropped], judge=judge)

    assert ("pa", "pb") in outcome.reasons
    assert len(outcome.reasons[("pa", "pb")].split()) <= 14
    assert outcome.dropped == frozenset({("pc", "pd")})
    assert outcome.reasoned_count == 1
    assert outcome.dropped_count == 1


def test_semantic_lane_fails_open_without_judge_and_on_errors() -> None:
    assert adjudicate_semantic_edges([_pair()], judge=None).reasons == {}

    def broken(system_prompt: str, payload: dict) -> dict:
        raise RuntimeError("no")

    outcome = adjudicate_semantic_edges([_pair()], judge=broken)
    assert outcome.reasons == {}
    assert outcome.dropped == frozenset()
    assert outcome.failed_batches == 1


def test_semantic_verdicts_replay_from_the_cache_including_drops(tmp_path: Path) -> None:
    cache = BuildCache(tmp_path / "cache.db")
    judge = ScriptedReasonJudge({("Alpha", "Beta"): "NONE"})

    first = adjudicate_semantic_edges([_pair()], judge=judge, cache=cache)
    second = adjudicate_semantic_edges([_pair()], judge=judge, cache=cache)

    assert judge.calls == 1
    assert first.dropped == second.dropped == frozenset({("pa", "pb")})
    cache.close()


# -- item 6: merge adjudication --------------------------------------------------


def _gray_pair(
    left: str = "Neil",
    right: str = "Neil Agrawal",
    *,
    etype: str = "person",
    reason: str = "token_overlap",
) -> GrayZonePair:
    return GrayZonePair(
        etype=etype,
        left_entity_id=f"e-{left}",
        right_entity_id=f"e-{right}",
        left_surface=left,
        right_surface=right,
        left_normalized=left.casefold(),
        right_normalized=right.casefold(),
        combined_df=4,
        reason=reason,
    )


class ScriptedMergeJudge:
    def __init__(self, verdict: dict):
        self.verdict = verdict
        self.calls = 0

    def __call__(self, system_prompt: str, payload: dict) -> dict:
        assert "same real-world entity" in system_prompt
        self.calls += 1
        return {"pairs": {row["key"]: dict(self.verdict) for row in payload["pairs"]}}


def test_merges_approve_only_confident_yes_verdicts() -> None:
    yes = ScriptedMergeJudge({"same_entity": True, "confidence": 0.95, "why": "same person"})
    outcome = adjudicate_merges([_gray_pair()], contexts={}, judge=yes)
    assert outcome.approved == (("person", "neil", "neil agrawal"),)

    hesitant = ScriptedMergeJudge({"same_entity": True, "confidence": 0.5, "why": "maybe"})
    assert adjudicate_merges([_gray_pair()], contexts={}, judge=hesitant).approved == ()

    no = ScriptedMergeJudge({"same_entity": False, "confidence": 0.99, "why": "different"})
    assert adjudicate_merges([_gray_pair()], contexts={}, judge=no).approved == ()


def test_digit_guard_vetoes_a_yes_verdict() -> None:
    pair = _gray_pair("Launch 2", "Launch")
    yes = ScriptedMergeJudge({"same_entity": True, "confidence": 1.0, "why": "same"})

    outcome = adjudicate_merges([pair], contexts={}, judge=yes)

    assert outcome.approved == ()
    assert outcome.judged_count == 1


def test_merge_verdicts_replay_from_the_cache(tmp_path: Path) -> None:
    cache = BuildCache(tmp_path / "cache.db")
    yes = ScriptedMergeJudge({"same_entity": True, "confidence": 0.9, "why": "same"})

    first = adjudicate_merges([_gray_pair()], contexts={}, judge=yes, cache=cache)
    second = adjudicate_merges([_gray_pair()], contexts={}, judge=yes, cache=cache)

    assert yes.calls == 1
    assert first.approved == second.approved
    cache.close()


def test_merge_lane_fails_open() -> None:
    def broken(system_prompt: str, payload: dict) -> dict:
        raise RuntimeError("no")

    outcome = adjudicate_merges([_gray_pair()], contexts={}, judge=broken)
    assert outcome.approved == ()
    assert outcome.failed_batches == 1


# -- build integration -----------------------------------------------------------


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        import re

        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def hash_embedder(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        row = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) + 1
        row /= np.linalg.norm(row)
        rows.append(row)
    return np.vstack(rows)


def constant_embedder(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 16), dtype=np.float32) / 4.0


def _work_extractor(_notes, *, tokenizer):
    del tokenizer
    text = "Maya Chen works at Klaviyo now."
    return {
        "n1": Extraction(
            mentions=[
                ExtractedMention(surface="Maya Chen", etype="person", confidence=0.95, char_start=0, char_end=9, extractor="fake", schema_id="s1"),
                ExtractedMention(surface="Klaviyo", etype="organization", confidence=0.9, char_start=19, char_end=26, extractor="fake", schema_id="s1"),
            ],
            fact_candidates=[
                ExtractedRelation(head="Maya Chen", predicate="works_at", tail="Klaviyo", confidence=0.9, subject_start=0, subject_end=9, object_start=19, object_end=26, assertion_start=0, assertion_end=len(text), extraction_method="fake", schema_id="s1", status="active")
            ],
            schema_id="s1",
        )
    }


def test_build_persists_rejections_for_audit_and_excludes_them_everywhere(
    tmp_path: Path,
) -> None:
    note = make_note("n1", "Work", "Maya Chen works at Klaviyo now.")

    def judge(system_prompt: str, payload: dict) -> dict:
        return {
            "facts": [
                {
                    "fact_key": row["fact_key"],
                    "support": "unsupported",
                    "quote": "works at Klaviyo",
                    "significant": False,
                    "direction_ok": True,
                    "relabel": None,
                    "score": 0.05,
                }
                for row in payload.get("facts", ())
            ],
            "pairs": {},
        }

    deps = BuildDependencies(
        tokenizer=Tokenizer(),
        embedder=hash_embedder,
        embedding_dim=16,
        extractor=_work_extractor,
        edge_adjudicator=judge,
    )
    target = tmp_path / "knowledge.db"

    report = BuildCoordinator(deps).build([note], target, replace_derived=True)

    assert report.adjudicated_fact_count == 1
    assert report.adjudication_rejected_by_predicate == {"works_at": 1}
    store = KnowledgeStore(target, read_only=True)
    try:
        # audit row survives with score + quote…
        row = store._conn.execute(
            "SELECT status, adjudication_score, adjudication_quote, adjudicated FROM facts"
        ).fetchone()
        assert row["status"] == "rejected"
        assert row["adjudication_score"] == pytest.approx(0.05)
        assert row["adjudication_quote"] == "works at Klaviyo"
        assert row["adjudicated"] == 1
        # …but the fact is invisible to recall, embeddings, and the graph
        assert store.fact_contexts() == ()
        assert store.fact_embeddings() == ()
        graph = store.base_graph(report.build_id)
        assert not any(edge.kind == "fact" for edge in graph.edges)
        assert store.validate_invariants().ok
    finally:
        store.close()


def test_build_downweight_multiplies_the_fact_edge_weight(tmp_path: Path) -> None:
    note = make_note("n1", "Work", "Maya Chen works at Klaviyo now.")

    def judge_for(score: float):
        def judge(system_prompt: str, payload: dict) -> dict:
            return {
                "facts": [
                    {
                        "fact_key": row["fact_key"],
                        "support": "partial",
                        "quote": "works at Klaviyo",
                        "significant": True,
                        "direction_ok": True,
                        "relabel": None,
                        "score": score,
                    }
                    for row in payload.get("facts", ())
                ],
                "pairs": {},
            }

        return judge

    def weight_for(score: float, target: Path) -> float:
        deps = BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=hash_embedder,
            embedding_dim=16,
            extractor=_work_extractor,
            edge_adjudicator=judge_for(score),
        )
        report = BuildCoordinator(deps).build([note], target, replace_derived=True)
        store = KnowledgeStore(target, read_only=True)
        try:
            graph = store.base_graph(report.build_id)
            (edge,) = [
                edge
                for edge in graph.edges
                if edge.kind == "fact" and edge.direction == "forward"
            ]
            return edge.weight
        finally:
            store.close()

    full = weight_for(0.9, tmp_path / "full.db")
    halved = weight_for(0.5, tmp_path / "halved.db")

    assert halved == pytest.approx(full * 0.5, rel=1e-6)


def test_build_semantic_reasons_persist_and_none_drops_the_edge(tmp_path: Path) -> None:
    notes = [
        make_note("a", "Alpha", "Potluck planning with Adi on Saturday."),
        make_note("b", "Beta", "Bring the grill for the Saturday potluck."),
        make_note("c", "Gamma", "Quarterly tax filing checklist for April."),
    ]

    def judge(system_prompt: str, payload: dict) -> dict:
        out = {}
        for row in payload.get("pairs", ()):
            titles = {row["a"]["title"], row["b"]["title"]}
            if titles == {"Alpha", "Beta"}:
                out[row["key"]] = "both plan the Saturday potluck"
            else:
                out[row["key"]] = "NONE"
        return {"pairs": out, "facts": []}

    deps = BuildDependencies(
        tokenizer=Tokenizer(),
        embedder=constant_embedder,  # every pair is a mutual-kNN edge
        embedding_dim=16,
        edge_adjudicator=judge,
    )
    target = tmp_path / "knowledge.db"

    report = BuildCoordinator(deps).build(notes, target, replace_derived=True)

    assert report.semantic_reasoned_edge_count == 1
    assert report.semantic_reason_dropped_count == 2
    store = KnowledgeStore(target, read_only=True)
    try:
        rows = store._conn.execute(
            "SELECT src_passage_id, dst_passage_id, reason FROM passage_semantic_edges"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["reason"] == "both plan the Saturday potluck"
        # the reason rides base-graph metadata into retrieval snapshots
        graph = store.base_graph(report.build_id)
        semantic = [edge for edge in graph.edges if edge.kind == "semantic"]
        assert semantic and all(
            edge.metadata["reason"] == "both plan the Saturday potluck"
            for edge in semantic
        )
        # dropped edges left the stored rows AND the in-memory graph (hash holds)
        assert store.verify_graph_hash(report.build_id)
    finally:
        store.close()


def test_build_merges_neil_via_the_judge(tmp_path: Path) -> None:
    text = "Neil planned the trip. Neil Agrawal booked the flights."

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    ExtractedMention(surface="Neil", etype="person", confidence=0.95, char_start=0, char_end=4, extractor="fake", schema_id="s1"),
                    ExtractedMention(surface="Neil Agrawal", etype="person", confidence=0.9, char_start=23, char_end=35, extractor="fake", schema_id="s1"),
                ],
                fact_candidates=[],
                schema_id="s1",
            )
        }

    def judge(system_prompt: str, payload: dict) -> dict:
        if "pairs" in payload and payload["pairs"] and "surface" in payload["pairs"][0]["a"]:
            return {
                "pairs": {
                    row["key"]: {"same_entity": True, "confidence": 0.95, "why": "same person"}
                    for row in payload["pairs"]
                }
            }
        return {"facts": [], "pairs": {}}

    def build(edge_adjudicator):
        deps = BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=hash_embedder,
            embedding_dim=16,
            extractor=extractor,
            edge_adjudicator=edge_adjudicator,
        )
        target = tmp_path / f"kb-{id(edge_adjudicator)}.db"
        report = BuildCoordinator(deps).build(
            [make_note("n1", "Trip", text)], target, replace_derived=True
        )
        store = KnowledgeStore(target, read_only=True)
        try:
            return report, store.counts()["entities"]
        finally:
            store.close()

    baseline_report, baseline_entities = build(None)
    judged_report, judged_entities = build(judge)

    assert baseline_entities == 2  # deterministic resolution keeps them apart
    assert judged_entities == 1  # the grounded judge merges the gray-zone pair
    assert judged_report.merge_gray_zone_pair_count == 1
    assert judged_report.merge_approved_count == 1
    assert baseline_report.merge_approved_count == 0


def test_builds_with_a_cached_judge_are_deterministic(tmp_path: Path) -> None:
    notes = [make_note("n1", "Work", "Maya Chen works at Klaviyo now.")]

    def judge(system_prompt: str, payload: dict) -> dict:
        return {
            "facts": [
                {
                    "fact_key": row["fact_key"],
                    "support": "partial",
                    "quote": "works at Klaviyo",
                    "significant": True,
                    "direction_ok": True,
                    "relabel": None,
                    "score": 0.5,
                }
                for row in payload.get("facts", ())
            ],
            "pairs": {},
        }

    cache = BuildCache(tmp_path / "cache.db")

    def build(target: Path):
        deps = BuildDependencies(
            tokenizer=Tokenizer(),
            embedder=hash_embedder,
            embedding_dim=16,
            extractor=_work_extractor,
            edge_adjudicator=judge,
            adjudication_cache=cache,
        )
        return BuildCoordinator(deps).build(notes, target, replace_derived=True)

    first = build(tmp_path / "first.db")
    second = build(tmp_path / "second.db")

    assert first.graph_hash == second.graph_hash
    cache.close()


# -- retrieval surfacing (hint plumbing) ------------------------------------------


def test_semantic_reasons_surface_in_path_rendering_and_rerank_hints() -> None:
    from taurine_core.indexer.base_graph import BaseEdge
    from taurine_core.retrieval.evidence import (
        _candidate_hint,
        _render_evidence_path,
        _semantic_reason_index,
    )
    from taurine_core.retrieval.paths import PathCandidate
    from taurine_core.indexer.knowledge_store import PassageContext

    class Graph:
        edges = (
            BaseEdge("pa", "pb", "semantic", 0.8, metadata={"cosine": 0.8, "reason": "both plan the potluck"}),
            BaseEdge("pb", "pa", "semantic", 0.8, metadata={"cosine": 0.8, "reason": "both plan the potluck"}),
            BaseEdge("pa", "e1", "contains", 0.5, metadata={}),
        )

    reasons = _semantic_reason_index(Graph())
    assert reasons == {("pa", "pb"): "both plan the potluck"}

    def context(passage_id: str, title: str) -> PassageContext:
        return PassageContext(
            passage_id=passage_id,
            note_id=f"note-{passage_id}",
            title=title,
            source="apple_notes",
            text="body",
            char_start=0,
            char_end=4,
            created=1.0,
            modified=1.0,
            build_id="b1",
        )

    path = PathCandidate(
        path_key="path-1",
        node_ids=("pa", "pb"),
        edge_kinds=("semantic:symmetric",),
        fact_ids=(),
        weight=0.8,
    )
    passages = {"pa": context("pa", "Alpha"), "pb": context("pb", "Beta")}

    rendered = _render_evidence_path(path, passages, {}, reasons)
    assert "semantically_related (both plan the potluck)" in rendered
    # without a reason the rendering is unchanged
    assert "(" not in _render_evidence_path(path, passages, {}, {})

    hint = _candidate_hint("pb", {}, {}, {"pb": (path,)}, reasons)
    assert "linked because: both plan the potluck" in hint
    assert "explanatory graph path" in hint
    # no reason ⇒ today's hint exactly
    assert _candidate_hint("pb", {}, {}, {"pb": (path,)}, {}) == (
        "reached by an explanatory graph path"
    )


def test_fact_contexts_expose_adjudication_fields(tmp_path: Path) -> None:
    note = make_note("n1", "Work", "Maya Chen works at Klaviyo now.")

    def judge(system_prompt: str, payload: dict) -> dict:
        return {
            "facts": [
                {
                    "fact_key": row["fact_key"],
                    "support": "partial",
                    "quote": "works at Klaviyo",
                    "significant": True,
                    "direction_ok": True,
                    "relabel": None,
                    "score": 0.55,
                }
                for row in payload.get("facts", ())
            ],
            "pairs": {},
        }

    deps = BuildDependencies(
        tokenizer=Tokenizer(),
        embedder=hash_embedder,
        embedding_dim=16,
        extractor=_work_extractor,
        edge_adjudicator=judge,
    )
    target = tmp_path / "knowledge.db"
    BuildCoordinator(deps).build([note], target, replace_derived=True)

    store = KnowledgeStore(target, read_only=True)
    try:
        (fact,) = store.fact_contexts()
        assert fact.adjudicated is True
        assert fact.adjudication_score == pytest.approx(0.55)
        assert fact.adjudication_quote == "works at Klaviyo"
    finally:
        store.close()
