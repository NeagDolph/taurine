from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core import config
from taurine_core.llm import StreamEvent
from taurine_core.ports import LLMError
from taurine_core.schemas import GroundingItem


@dataclass
class ProfileRetriever:
    searches: list[tuple[str, int, float]] = field(default_factory=list)

    def search(self, query: str, k: int, *, now: float) -> list[GroundingItem]:
        self.searches.append((query, k, now))
        return [
            GroundingItem(
                note_id="alpha",
                short_id="N1",
                title="Taurine roadmap",
                source="apple_notes",
                age_days=1.0,
                score=0.91,
                gist="Current Taurine product roadmap.",
            )
        ]

    def resolve(self, ref: str) -> str | None:
        return "alpha" if ref in {"N1", "alpha"} else None


class ProfileLLM:
    def __init__(self, streams: list[list[StreamEvent | Exception]]) -> None:
        self.streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        reasoning: dict[str, Any] | None,
        timeout_s: float,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "reasoning": reasoning,
                "timeout_s": timeout_s,
            }
        )
        for event in self.streams.pop(0):
            if isinstance(event, Exception):
                raise event
            yield event


def _tool(call_id: str, name: str, arguments: dict[str, Any]) -> StreamEvent:
    return StreamEvent(
        kind="tool_call",
        tool_call={"id": call_id, "name": name, "arguments": arguments},
    )


@pytest.mark.asyncio
async def test_profile_generator_researches_with_graph_tools_before_synthesis(store) -> None:
    note = make_note(
        "alpha",
        "Taurine roadmap",
        "Neil is building Taurine as a live meeting copilot.",
    )
    enrich = enrichment("Current Taurine product roadmap.")
    store.upsert_note(note, enrich, fixture_units(unit_vec(1), enrich))
    retriever = ProfileRetriever()
    llm = ProfileLLM(
        [
            [
                _tool("call-1", "search_notes", {"query": "current projects", "k": 8}),
                _tool("call-2", "read_note", {"note": "N1"}),
                _tool("call-3", "expand_note", {"note": "N1"}),
                StreamEvent(kind="done"),
            ],
            [
                StreamEvent(
                    kind="content",
                    text="# User profile\n\nNeil builds Taurine and works on live knowledge systems.",
                ),
                StreamEvent(kind="done"),
            ],
        ]
    )
    progress = []

    from taurine_core.indexer.profile import ProfileGenerator

    card = await ProfileGenerator(store, llm, retriever=retriever).generate(progress.append)

    assert card.startswith("# User profile")
    assert len(llm.calls) == 2
    assert all(call["model"] == config.PROFILE_MODEL for call in llm.calls)
    assert all(call["reasoning"] == config.PROFILE_REASONING for call in llm.calls)
    assert all(call["timeout_s"] == config.PROFILE_TIMEOUT_S for call in llm.calls)
    assert llm.calls[0]["tools"] is not None
    assert llm.calls[1]["tools"] is not None
    tool_messages = [
        message for message in llm.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert [message["name"] for message in tool_messages] == [
        "search_notes",
        "read_note",
        "expand_note",
    ]
    assert retriever.searches[0][:2] == ("current projects", 8)
    assert progress[-1].phase == "synthesizing"
    assert progress[-1].tool_calls == 3
    assert progress[-1].nodes_read == 1


@pytest.mark.asyncio
async def test_profile_generator_forces_synthesis_when_tool_budget_is_exhausted(
    store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "PROFILE_MAX_TOOL_CALLS", 1)
    retriever = ProfileRetriever()
    llm = ProfileLLM(
        [
            [
                _tool("call-1", "search_notes", {"query": "interests"}),
                StreamEvent(kind="done"),
            ],
            [StreamEvent(kind="content", text="## Interests\nKnowledge systems."), StreamEvent(kind="done")],
        ]
    )

    from taurine_core.indexer.profile import ProfileGenerator

    card = await ProfileGenerator(store, llm, retriever=retriever).generate(lambda _update: None)

    assert card == "## Interests\nKnowledge systems."
    assert llm.calls[1]["tools"] is None
    assert "Research is complete" in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_profile_generator_rejects_empty_final_content(store) -> None:
    llm = ProfileLLM([[StreamEvent(kind="done")]])

    from taurine_core.indexer.profile import ProfileGenerator

    with pytest.raises(LLMError, match="empty user profile"):
        await ProfileGenerator(store, llm, retriever=ProfileRetriever()).generate(
            lambda _update: None
        )


def test_user_card_round_trips_through_store_metadata(store) -> None:
    assert store.user_card() is None

    store.set_user_card("## User\nWorks on Taurine.")

    assert store.user_card() == "## User\nWorks on Taurine."
