from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from taurine_core.live.grounding import GroundingUpdate
from taurine_core.live.snippets import SnippetAgent
from taurine_core.retrieval.types import (
    EvidenceNote,
    EvidencePassage,
    RetrievalDecision,
)
from taurine_core.schemas import NoteSuggestionEvent


def _note(ref: str, note_id: str, title: str) -> EvidenceNote:
    return EvidenceNote(ref, note_id, title, "obsidian", 100.0, 0.9, (f"E{ref}",))


def _passage(note_id: str, text: str) -> EvidencePassage:
    return EvidencePassage(
        f"E-{note_id}", f"p-{note_id}", note_id, "t", "obsidian", text,
        0, len(text), 90.0, 100.0, 0.8, 0.7,
    )


def _update(
    notes: tuple[EvidenceNote, ...],
    passages: tuple[EvidencePassage, ...] = (),
) -> GroundingUpdate:
    decision = RetrievalDecision(True, 0.9, "ok", ("dense",), ())
    return GroundingUpdate(
        decision=decision,
        passages=passages,
        facts=(),
        paths=(),
        notes=notes,
        display_passages=passages,
        display_notes=notes,
        no_new_evidence=False,
        new_support_ids=(),
        revision=1,
    )


@dataclass
class FakeLLM:
    responses: list[dict[str, Any]]
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        timeout_s: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del system, timeout_s, temperature, max_tokens, schema
        self.calls.append((model, user))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


async def _drain(agent: SnippetAgent) -> None:
    while agent._tasks:  # noqa: SLF001 — test-only settle
        await asyncio.gather(*tuple(agent._tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_emits_llm_snippet_once_per_note() -> None:
    llm = FakeLLM([{"snippets": [{"note_ref": "N1", "snippet": "Budget is $48k"}]}])
    agent = SnippetAgent(llm, clock=lambda: 123.0)
    events: list[NoteSuggestionEvent] = []
    update = _update((_note("N1", "n1", "Roadmap"),), (_passage("n1", "Budget is $48,000 for Q3."),))

    agent.offer(update, "we were discussing the budget", events.append)
    await _drain(agent)
    agent.offer(update, "same notes again", events.append)
    await _drain(agent)

    assert [event.snippet for event in events] == ["Budget is $48k"]
    assert events[0].note_id == "n1"
    assert events[0].title == "Roadmap"
    assert events[0].ts == 123.0
    assert len(llm.calls) == 1
    assert "Budget is $48,000" in llm.calls[0][1]


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_passage_text() -> None:
    llm = FakeLLM([RuntimeError("provider down")])  # type: ignore[list-item]
    agent = SnippetAgent(llm)
    events: list[NoteSuggestionEvent] = []
    update = _update((_note("N1", "n1", "Roadmap"),), (_passage("n1", "Priya approves backups."),))

    agent.offer(update, "tail", events.append)
    await _drain(agent)

    assert [event.snippet for event in events] == ["Priya approves backups."]


@pytest.mark.asyncio
async def test_metadata_only_note_ships_without_llm_call() -> None:
    llm = FakeLLM([])
    agent = SnippetAgent(llm)
    events: list[NoteSuggestionEvent] = []

    agent.offer(_update((_note("N2", "n2", "Contacts"),)), "tail", events.append)
    await _drain(agent)

    assert [event.snippet for event in events] == [""]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_session_cap_limits_suggestions() -> None:
    agent = SnippetAgent(None, max_per_session=1)
    events: list[NoteSuggestionEvent] = []

    agent.offer(_update((_note("N1", "n1", "A"), _note("N2", "n2", "B"))), "t", events.append)
    await _drain(agent)

    assert [event.note_ref for event in events] == ["N1"]
