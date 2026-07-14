from __future__ import annotations

from types import SimpleNamespace

import pytest

from taurine_core import config
from taurine_core.indexer.edge_adjudication import adjudicate_facts
from taurine_core.indexer.facts import PREDICATES

from tests.test_edge_adjudication import ScriptedFactJudge, _draft, _fact, _verdict


class Scores:
    def __init__(self, values: dict[str, float | Exception]):
        self.values = values
        self.calls: list[tuple[str, str]] = []

    def score(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        value = self.values[hypothesis]
        if isinstance(value, Exception):
            raise value
        return value


def test_verbalize_has_a_specific_template_for_every_predicate() -> None:
    from taurine_core.indexer.nli_screen import verbalize

    fact = _fact("Maya", "works_with", "Rex", span=(0, 20))
    hypotheses = {
        predicate: verbalize(SimpleNamespace(**{**fact.__dict__, "predicate": predicate}))
        for predicate in PREDICATES
    }

    assert set(hypotheses) == set(PREDICATES)
    assert all(value.startswith("Maya ") and value.endswith(" Rex") for value in hypotheses.values())
    assert hypotheses["works_at"] == "Maya works at Rex"
    assert hypotheses["paid_or_owes"] == "Maya paid or owes money to Rex"
    assert hypotheses["blocked_by"] == "Maya is blocked by Rex"


def test_screen_keeps_only_high_entailment_screenable_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    from taurine_core.indexer.nli_screen import screen, verbalize

    kept = _fact("Maya", "works_with", "Rex", span=(0, 20))
    low = _fact("Maya", "uses", "Taurine", note_id="n2", passage_id="p2", span=(0, 20))
    exclusive = _fact("Maya", "works_at", "Acme", note_id="n3", passage_id="p3", span=(0, 20))
    completion = SimpleNamespace(
        **{**_fact("Maya", "member_of", "Club", note_id="n4", passage_id="p4", span=(0, 20)).__dict__,
           "extraction_method": "llm_relation_completion"}
    )
    missing = _fact("Maya", "about", "Launch", note_id="n5", passage_id="missing", span=(0, 20))
    drafts = {
        "p1": _draft(text="Maya works with Rex."),
        "p2": _draft("p2", text="Maya discussed a roadmap."),
        "p3": _draft("p3", text="Maya works at Acme."),
        "p4": _draft("p4", text="Maya is a member of Club."),
    }
    scorer = Scores({verbalize(kept): 0.97, verbalize(low): 0.04})
    monkeypatch.setattr(config, "NLI_KEEP_FLOOR", 0.9)
    monkeypatch.setattr(config, "NLI_AUTO_REJECT", False)

    result = screen([kept, low, exclusive, completion, missing], drafts, scorer)

    assert set(result.kept) == {kept.fact_id}
    judgment = result.kept[kept.fact_id]
    assert judgment.support == "supported"
    assert judgment.score == 0.95
    assert judgment.quote == "Maya works with Rex."
    assert [fact.fact_id for fact in result.pending] == sorted(
        [low.fact_id, exclusive.fact_id, completion.fact_id, missing.fact_id]
    )
    assert len(scorer.calls) == 2


def test_screen_fails_open_per_fact_when_scorer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from taurine_core.indexer.nli_screen import screen, verbalize

    broken = _fact("Maya", "uses", "Taurine", span=(0, 20))
    scorer = Scores({verbalize(broken): RuntimeError("NLI unavailable")})
    monkeypatch.setattr(config, "NLI_KEEP_FLOOR", 0.9)

    result = screen([broken], {"p1": _draft(text="Maya uses Taurine.")}, scorer)

    assert result.kept == {}
    assert result.pending == [broken]


def test_adjudicate_facts_merges_nli_keeps_without_calling_or_caching_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from taurine_core import models_local
    from taurine_core.indexer.nli_screen import verbalize

    fact = _fact("Maya", "works_with", "Rex", span=(0, 20))
    scorer = Scores({verbalize(fact): 0.93})

    class Cache:
        def __init__(self) -> None:
            self.stored = []

        def cached_judgments(self, keys):
            return {}

        def store_judgments(self, rows):
            self.stored.extend(rows)

    cache = Cache()
    judge = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.1, quote="Maya works with Rex")}
    )
    monkeypatch.setattr(config, "NLI_SCREEN_ENABLED", True)
    monkeypatch.setattr(config, "NLI_KEEP_FLOOR", 0.9)
    monkeypatch.setattr(models_local, "local_nli_scorer", lambda model: scorer)

    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft(text="Maya works with Rex.")]},
        note_titles={"n1": "Work"},
        judge=judge,
        cache=cache,
    )

    assert judge.calls == 0
    assert cache.stored == []
    assert outcome.adjudicated == 1
    assert outcome.facts[0].adjudicated is True
    assert outcome.facts[0].adjudication_score == 0.93


def test_adjudicate_facts_model_load_failure_uses_existing_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from taurine_core import models_local

    fact = _fact("Maya", "works_with", "Rex", span=(0, 20))
    judge = ScriptedFactJudge(
        {fact.fact_id: _verdict(fact.fact_id, score=0.9, quote="Maya works with Rex")}
    )
    monkeypatch.setattr(config, "NLI_SCREEN_ENABLED", True)
    monkeypatch.setattr(
        models_local,
        "local_nli_scorer",
        lambda model: (_ for _ in ()).throw(RuntimeError("model missing")),
    )

    outcome = adjudicate_facts(
        [fact],
        passages_by_note={"n1": [_draft(text="Maya works with Rex.")]},
        note_titles={"n1": "Work"},
        judge=judge,
    )

    assert judge.calls == 1
    assert outcome.adjudicated == 1
