"""Fail-open local NLI pre-screen for fact adjudication."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from taurine_core import config
from taurine_core.indexer.facts import FactRecord, PREDICATES

if TYPE_CHECKING:
    from taurine_core.indexer.edge_adjudication import FactJudgment


_LOG = logging.getLogger("taurine.build")

_TEMPLATES = {
    "works_with": "{subject} works with {object}",
    "works_at": "{subject} works at {object}",
    "member_of": "{subject} is a member of {object}",
    "part_of": "{subject} is part of {object}",
    "located_at": "{subject} is located at {object}",
    "scheduled_for": "{subject} is scheduled for {object}",
    "paid_or_owes": "{subject} paid or owes money to {object}",
    "owns_or_has": "{subject} owns or has {object}",
    "uses": "{subject} uses {object}",
    "about": "{subject} is about {object}",
    "met_with": "{subject} met with {object}",
    "related_to": "{subject} is related to {object}",
    "decided": "{subject} decided {object}",
    "prefers": "{subject} prefers {object}",
    "blocked_by": "{subject} is blocked by {object}",
    "responsible_for": "{subject} is responsible for {object}",
    "completed": "{subject} completed {object}",
    "supersedes": "{subject} supersedes {object}",
}

if set(_TEMPLATES) != set(PREDICATES):
    raise RuntimeError("NLI fact templates must cover the predicate registry exactly")


@dataclass
class ScreenResult:
    kept: dict[str, FactJudgment]
    pending: list[FactRecord]


def verbalize(fact: FactRecord) -> str:
    """Render one registry fact as the NLI hypothesis."""
    return _TEMPLATES[fact.predicate].format(
        subject=fact.subject.strip(), object=fact.object.strip()
    )


def _assertion_text(fact: FactRecord, draft: Any) -> str:
    start = max(0, int(fact.assertion_start) - int(draft.char_start))
    end = max(start, int(fact.assertion_end) - int(draft.char_start))
    return str(draft.text)[start:end][:500]


def _must_use_llm(fact: FactRecord, draft_by_passage: Mapping[str, Any]) -> bool:
    spec = PREDICATES.get(fact.predicate)
    return (
        spec is None
        or spec.exclusive_object
        or fact.extraction_method == "llm_relation_completion"
        or fact.source_passage_id not in draft_by_passage
    )


def screen(
    facts: Sequence[FactRecord],
    draft_by_passage: Mapping[str, Any],
    scorer: Any,
) -> ScreenResult:
    """Auto-decide safe NLI bands and route every uncertainty to the LLM.

    ``kept`` follows the approved contract name. When auto-reject is explicitly
    enabled it also carries low-score judgments so ``_fact_outcome`` can apply
    the existing rejection semantics. The default never auto-rejects.
    """
    from taurine_core.indexer.edge_adjudication import FactJudgment

    decided: dict[str, FactJudgment] = {}
    pending: list[FactRecord] = []
    for fact in sorted(facts, key=lambda row: row.fact_id):
        if _must_use_llm(fact, draft_by_passage):
            pending.append(fact)
            continue
        draft = draft_by_passage[fact.source_passage_id]
        try:
            entailment = float(scorer.score(str(draft.text), verbalize(fact)))
        except Exception as exc:
            _LOG.warning("NLI fact screen failed for %s: %s", fact.fact_id, exc)
            pending.append(fact)
            continue
        if not 0.0 <= entailment <= 1.0:
            pending.append(fact)
            continue
        quote = _assertion_text(fact, draft)
        if entailment >= float(config.NLI_KEEP_FLOOR):
            decided[fact.fact_id] = FactJudgment(
                fact_id=fact.fact_id,
                support="supported",
                quote=quote,
                significant=True,
                direction_ok=True,
                relabel=None,
                score=min(0.95, entailment),
            )
        elif bool(config.NLI_AUTO_REJECT) and entailment <= float(
            config.NLI_REJECT_CEIL
        ):
            decided[fact.fact_id] = FactJudgment(
                fact_id=fact.fact_id,
                support="unsupported",
                quote=quote,
                significant=False,
                direction_ok=True,
                relabel=None,
                score=min(entailment, float(config.NLI_REJECT_CEIL)),
            )
        else:
            pending.append(fact)
    return ScreenResult(kept=decided, pending=pending)
