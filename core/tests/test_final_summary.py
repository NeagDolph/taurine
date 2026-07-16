from __future__ import annotations

from typing import Any

import pytest

from taurine_core import config
from taurine_core.indexer.store import Index
from taurine_core.live.final_summary import FINAL_SUMMARY_SCHEMA, run_final_summary
from taurine_core.ports import LLMError


@pytest.fixture
def index(tmp_path: Any) -> Index:
    store = Index(tmp_path / "final-summary.db")
    try:
        yield store
    finally:
        store.close()


class FakeLLM:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.tool_calls: list[dict[str, Any]] = []
        self.final_calls: list[dict[str, Any]] = []

    async def complete_tools(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.tool_calls.append(
            {
                "model": model,
                "system": system,
                "messages": list(messages),
                "tools": tools,
                "timeout_s": timeout_s,
            }
        )
        return {"role": "assistant", "content": "Ready to summarize."}

    async def complete_json_messages(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.final_calls.append(
            {
                "model": model,
                "messages": list(messages),
                "timeout_s": timeout_s,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "schema": schema,
            }
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _snapshot(index: Index, *, title: str = "") -> dict[str, Any]:
    meeting = index.meeting_create(title)
    meeting = index.meeting_update(
        meeting["meeting_id"], notes_md="Alpha\nAnchor\nBeta\nAnchor\nOmega"
    )
    index.meeting_append_transcript(
        meeting["meeting_id"],
        [
            {"channel": 0, "text": "I will send the deck", "ts": 1.0, "seq": 1},
            {"channel": 1, "text": "Please send it Friday", "ts": 2.0, "seq": 2},
        ],
    )
    index.meeting_set_status(meeting["meeting_id"], "summarizing", ended_at=3.0)
    snapshot = index.meeting_get(meeting["meeting_id"])
    assert snapshot is not None
    return {
        **snapshot,
        "last_live_summary": {
            "summary_md": "Deck timing discussed.",
            "actionables": [
                {"id": "a1", "title": "Send deck", "status": "confirmed"},
                {"id": "a2", "title": "Old task", "status": "dismissed"},
            ],
        },
    }


async def _unused_tool(name: str, arguments: dict[str, Any]) -> str:
    raise AssertionError(f"unexpected tool call: {name} {arguments}")


async def test_happy_path_persists_emits_and_supplies_all_prompt_inputs(
    index: Index,
) -> None:
    meeting = _snapshot(index)
    llm = FakeLLM(
        {
            "title": "Deck Follow-up",
            "final_summary_md": "## Outcome\nThe deck is due Friday.",
            "suggested_edits": [
                {
                    "anchor": "Anchor",
                    "occurrence": 2,
                    "insert_md": "\n- Send deck Friday",
                    "rationale": "Captures the commitment",
                },
                {
                    "anchor": "",
                    "occurrence": 1,
                    "insert_md": "\n## Follow-up",
                    "rationale": "Adds a follow-up section",
                },
            ],
        }
    )
    events: list[Any] = []

    await run_final_summary(
        meeting,
        llm,
        _unused_tool,
        "<user_profile>\nNeil builds Taurine.\n</user_profile>",
        lambda event: _append(events, event),
        index,
    )

    stored = index.meeting_get(meeting["meeting_id"])
    assert stored is not None
    assert stored["status"] == "done"
    assert stored["title"] == "Deck Follow-up"
    assert stored["final_summary_md"] == "## Outcome\nThe deck is due Friday."
    assert stored["suggestions_base_rev"] == meeting["notes_rev"]
    assert len(stored["suggestions"]) == 2
    assert all(len(row["id"]) == 8 for row in stored["suggestions"])
    assert all(row["status"] == "pending" for row in stored["suggestions"])
    assert [event.type for event in events] == [
        "final_summary_status",
        "final_summary",
        "final_summary_status",
    ]
    assert events[0].state == "running"
    assert events[-1].state == "done"
    assert events[1].base_rev == meeting["notes_rev"]

    assert llm.tool_calls[0]["model"] == config.FINAL_SUMMARY_MODEL
    prompt = llm.tool_calls[0]["messages"][0]["content"]
    assert "You: I will send the deck" in prompt
    assert "Them: Please send it Friday" in prompt
    assert meeting["notes_md"] in prompt
    assert "notes_rev" in prompt and str(meeting["notes_rev"]) in prompt
    assert "Deck timing discussed." in prompt
    assert '"status":"confirmed"' in prompt
    assert '"status":"dismissed"' in prompt
    assert "Neil builds Taurine." in prompt
    assert "additive insertions" in llm.tool_calls[0]["system"]
    assert llm.final_calls[0]["schema"] == FINAL_SUMMARY_SCHEMA
    assert llm.final_calls[0]["max_tokens"] == config.FINAL_SUMMARY_MAX_TOKENS


async def test_titled_meeting_keeps_its_title(index: Index) -> None:
    meeting = _snapshot(index, title="Weekly Planning")
    llm = FakeLLM(
        {
            "title": "Model Replacement",
            "final_summary_md": "Summary",
            "suggested_edits": [],
        }
    )

    await run_final_summary(meeting, llm, _unused_tool, "", _noop_emit, index)

    assert index.meeting_get(meeting["meeting_id"])["title"] == "Weekly Planning"


async def test_title_changed_during_generation_is_preserved_in_event(
    index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = _snapshot(index)
    original_set_final = index.meeting_set_final
    events: list[Any] = []

    def raced_set_final(meeting_id: str, **kwargs: Any) -> dict[str, Any]:
        index.meeting_update(meeting_id, title="User title during generation")
        return original_set_final(meeting_id, **kwargs)

    monkeypatch.setattr(index, "meeting_set_final", raced_set_final)

    await run_final_summary(
        meeting,
        FakeLLM(
            {
                "title": "Generated title",
                "final_summary_md": "Summary",
                "suggested_edits": [],
            }
        ),
        _unused_tool,
        "",
        lambda event: _append(events, event),
        index,
    )

    stored = index.meeting_get(meeting["meeting_id"])
    result_event = next(event for event in events if event.type == "final_summary")
    assert stored["title"] == "User title during generation"
    assert result_event.title == stored["title"]


async def test_invalid_anchors_occurrences_and_edits_are_dropped_then_capped(
    index: Index,
) -> None:
    meeting = _snapshot(index)
    edits = [
        {"anchor": "missing", "occurrence": 1, "insert_md": "x", "rationale": "bad"},
        {"anchor": "Anchor", "occurrence": 3, "insert_md": "x", "rationale": "bad"},
        {"anchor": "Anchor", "occurrence": 1, "insert_md": " ", "rationale": "bad"},
    ] + [
        {
            "anchor": "",
            "occurrence": 1,
            "insert_md": f"valid-{number}",
            "rationale": "valid",
        }
        for number in range(12)
    ]
    llm = FakeLLM(
        {"title": "Capped", "final_summary_md": "Summary", "suggested_edits": edits}
    )

    await run_final_summary(meeting, llm, _unused_tool, "", _noop_emit, index)

    stored = index.meeting_get(meeting["meeting_id"])
    assert [row["insert_md"] for row in stored["suggestions"]] == [
        f"valid-{number}" for number in range(10)
    ]


@pytest.mark.parametrize(
    "result",
    [
        {"title": "Bad", "suggested_edits": []},
        {"title": "Bad", "final_summary_md": "", "suggested_edits": []},
        LLMError("garbage"),
        TimeoutError("slow"),
    ],
)
async def test_garbage_or_timeout_emits_error_and_only_marks_done(
    index: Index, result: dict[str, Any] | Exception
) -> None:
    meeting = _snapshot(index)
    events: list[Any] = []

    await run_final_summary(
        meeting,
        FakeLLM(result),
        _unused_tool,
        "",
        lambda event: _append(events, event),
        index,
    )

    stored = index.meeting_get(meeting["meeting_id"])
    assert stored["status"] == "done"
    assert stored["title"] == ""
    assert stored["final_summary_md"] == ""
    assert stored["suggestions"] == []
    assert [event.type for event in events] == [
        "final_summary_status",
        "final_summary_status",
    ]
    assert events[-1].state == "error"
    assert events[-1].detail


async def test_oversized_output_fails_closed(
    index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "FINAL_SUMMARY_MAX_CHARS", 100)
    meeting = _snapshot(index)
    events: list[Any] = []

    await run_final_summary(
        meeting,
        FakeLLM(
            {
                "title": "Too large",
                "final_summary_md": "x" * 200,
                "suggested_edits": [],
            }
        ),
        _unused_tool,
        "",
        lambda event: _append(events, event),
        index,
    )

    stored = index.meeting_get(meeting["meeting_id"])
    assert stored["status"] == "done"
    assert stored["final_summary_md"] == ""
    assert events[-1].state == "error"
    assert "oversized" in events[-1].detail


async def test_disconnect_during_emission_does_not_block_persistence(index: Index) -> None:
    meeting = _snapshot(index)
    calls = 0

    async def disconnected_emit(event: Any) -> None:
        nonlocal calls
        del event
        calls += 1
        raise RuntimeError("socket closed")

    await run_final_summary(
        meeting,
        FakeLLM(
            {"title": "Persisted", "final_summary_md": "Summary", "suggested_edits": []}
        ),
        _unused_tool,
        "",
        disconnected_emit,
        index,
    )

    stored = index.meeting_get(meeting["meeting_id"])
    assert calls == 3
    assert stored["status"] == "done"
    assert stored["title"] == "Persisted"
    assert stored["final_summary_md"] == "Summary"


async def _append(events: list[Any], event: Any) -> None:
    events.append(event)


async def _noop_emit(event: Any) -> None:
    del event


def test_final_summary_schema_is_closed_and_requires_the_wire_contract() -> None:
    assert FINAL_SUMMARY_SCHEMA["additionalProperties"] is False
    assert set(FINAL_SUMMARY_SCHEMA["required"]) == {
        "title",
        "final_summary_md",
        "suggested_edits",
    }
    edit = FINAL_SUMMARY_SCHEMA["properties"]["suggested_edits"]["items"]
    assert edit["additionalProperties"] is False
    assert set(edit["required"]) == {
        "anchor",
        "occurrence",
        "insert_md",
        "rationale",
    }
