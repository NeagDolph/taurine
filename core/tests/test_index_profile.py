from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.llm import StreamEvent
from taurine_core.ports import LLMError
from taurine_core.retrieval.types import (
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
)


@dataclass
class ProfileRetriever:
    searches: list[tuple[str, int, float]] = field(default_factory=list)

    def search_evidence(self, query: str, *, k: int, temporal: str, now: float) -> EvidencePack:
        self.searches.append((query, k, now))
        del temporal
        return EvidencePack(
            decision=RetrievalDecision(True, .91, "accepted"),
            query=RetrievalQuery(query, "tool", now),
            passages=(EvidencePassage(
                "E1", "p1", "alpha", "Taurine roadmap", "apple_notes",
                "Current Taurine product roadmap.", 0, 32, 0, now, .8, .91,
            ),), facts=(), paths=(),
            notes=(EvidenceNote("N1", "alpha", "Taurine roadmap", "apple_notes", now, .91, ("E1",)),),
            trace=RetrievalTrace(original_query=query, rewritten_query=query, read_ids=("N1", "E1")),
        )

    def read_note(self, note: str, *, query: str, cursor: str | None, max_chars: int):
        del query, cursor, max_chars
        return {"note": note, "passages": [{"evidence_id": "E1", "text": "Current Taurine product roadmap."}], "truncated": False, "next_cursor": None}

    def expand_evidence(self, evidence: str, *, direction: str, max_hops: int, limit: int):
        del direction, max_hops, limit
        return {"from": evidence, "results": [], "paths": []}

    def resolve_note(self, ref: str) -> str | None:
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
    store.replace_source_records("apple_notes", [note])
    retriever = ProfileRetriever()
    llm = ProfileLLM(
        [
            [
                _tool("call-1", "search_evidence", {"query": "current projects", "k": 8}),
                _tool("call-2", "read_note", {"note": "N1", "query": "current projects"}),
                _tool("call-3", "expand_evidence", {"evidence": "E1"}),
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
        "search_evidence",
        "read_note",
        "expand_evidence",
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
                _tool("call-1", "search_evidence", {"query": "interests"}),
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
