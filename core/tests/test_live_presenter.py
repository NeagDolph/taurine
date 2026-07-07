from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live.presenter import Presenter
from taurine_core.ports import LLMError
from taurine_core.schemas import Chunk, NoteRecord, SlateItem


@dataclass
class FakeLLM:
    responses: list[dict[str, Any] | Exception]
    calls: list[tuple[str, str, str, float | None]] = field(default_factory=list)
    tool_responses: list[dict[str, Any] | Exception] | None = None
    tool_requests: list[
        tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]
    ] = field(default_factory=list)

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict[str, Any]:
        self.calls.append((model, system, user, timeout_s))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def complete_tools(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.tool_requests.append((model, system, deepcopy(messages), deepcopy(tools)))
        if self.tool_responses is None:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return {"role": "assistant", "content": json.dumps(response)}

        response = self.tool_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@dataclass
class FakeIndex:
    notes: dict[str, NoteRecord] = field(default_factory=dict)
    neighbor_rows: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    note_calls: list[str] = field(default_factory=list)

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        return self.neighbor_rows.get(note_id, [])

    def note(self, note_id: str) -> NoteRecord | None:
        self.note_calls.append(note_id)
        return self.notes.get(note_id)


def chunk(index: int) -> Chunk:
    return Chunk(channel=index % 2, text=f"transcript chunk {index}", ts=float(index))


def note_record(
    note_id: str,
    *,
    title: str | None = None,
    gist: str | None = None,
    text: str | None = None,
) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=title or f"Title {note_id}",
        modified=0.0,
        content_hash=f"hash-{note_id}",
        text=text or f"Body {note_id}",
        gist=gist or f"Gist {note_id}",
    )


def slate_item(
    note_id: str,
    *,
    source: str = "search",
    heard: str | None = None,
    via_note_id: str | None = None,
) -> SlateItem:
    return SlateItem(
        note_id=note_id,
        title=f"Title {note_id}",
        gist=f"Gist {note_id}",
        score=0.8,
        source=source,  # type: ignore[arg-type]
        heard=heard or f"heard {note_id}",
        decay=2,
        added_rev=1,
        via_note_id=via_note_id,
        hop=1 if via_note_id else None,
    )


def tool_call(note_id: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read_note",
                    "arguments": json.dumps({"note_id": note_id}),
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_presenter_drops_unknown_ids_truncates_picks_and_clamps_why() -> None:
    long_why = "x" * (config.WHY_MAX_CHARS + 20)
    llm = FakeLLM(
        [
            {
                "picks": [
                    {"note_id": "missing", "why": "not on the slate"},
                    {"note_id": "n1", "why": long_why},
                    {"note_id": "n2", "why": "second"},
                    {"note_id": "n3", "why": "third"},
                ]
            }
        ]
    )
    presenter = Presenter(llm, FakeIndex())
    slate = [slate_item("n1"), slate_item("n2"), slate_item("n3")]

    picks = await presenter.present([chunk(1)], slate)

    assert [pick.note_id for pick in picks] == ["n1", "n2"]
    assert len(picks[0].why) == config.WHY_MAX_CHARS
    assert picks[1].why == "second"


@pytest.mark.asyncio
async def test_presenter_falls_back_once_after_primary_model_error() -> None:
    llm = FakeLLM([LLMError("primary failed"), {"picks": [{"note_id": "n1", "why": "fallback pick"}]}])
    presenter = Presenter(llm, FakeIndex())

    picks = await presenter.present([chunk(1)], [slate_item("n1")])

    assert [call[0] for call in llm.tool_requests] == [
        config.PRESENTER_MODEL,
        config.PRESENTER_FALLBACK_MODEL,
    ]
    assert [(pick.note_id, pick.why) for pick in picks] == [("n1", "fallback pick")]


@pytest.mark.asyncio
async def test_presenter_returns_empty_when_both_models_fail() -> None:
    llm = FakeLLM([LLMError("primary failed"), LLMError("fallback failed")])
    presenter = Presenter(llm, FakeIndex())

    assert await presenter.present([chunk(1)], [slate_item("n1")]) == []
    assert [call[0] for call in llm.tool_requests] == [
        config.PRESENTER_MODEL,
        config.PRESENTER_FALLBACK_MODEL,
    ]


@pytest.mark.asyncio
async def test_presenter_tool_loop_reads_neighbor_note_then_allows_pick() -> None:
    long_text = ("A" * 1200) + "TAIL_SHOULD_NOT_APPEAR"
    index = FakeIndex(
        notes={
            "notes/seed": note_record("notes/seed", title="Seed Title"),
            "notes/armaan": note_record(
                "notes/armaan",
                title="Armaan Hiring Plan",
                gist="How to follow up with Armaan",
                text=long_text,
            ),
        },
        neighbor_rows={"notes/seed": [("notes/armaan", 0.91)]},
    )
    llm = FakeLLM(
        responses=[],
        tool_responses=[
            tool_call("armaan"),
            {
                "role": "assistant",
                "content": json.dumps(
                    {"picks": [{"note_id": "armaan", "why": "Armaan came up by name"}]}
                ),
            },
        ],
    )
    presenter = Presenter(llm, index)

    picks = await presenter.present(
        [Chunk(channel=0, text="Armand is the person we need to follow up with", ts=1.0)],
        [slate_item("notes/seed")],
    )

    assert [(pick.note_id, pick.why) for pick in picks] == [
        ("notes/armaan", "Armaan came up by name")
    ]
    assert "notes/armaan" in index.note_calls
    _model, _system, messages, _tools = llm.tool_requests[1]
    tool_result_text = "\n".join(
        str(message.get("content", "")) for message in messages if message.get("role") == "tool"
    )
    assert "Armaan Hiring Plan" in tool_result_text
    assert long_text[:1200] in tool_result_text
    assert long_text[1200:] not in tool_result_text


@pytest.mark.asyncio
async def test_presenter_drops_unread_non_slate_pick() -> None:
    llm = FakeLLM([{"picks": [{"note_id": "outside", "why": "not seen"}]}])
    presenter = Presenter(llm, FakeIndex())

    picks = await presenter.present([chunk(1)], [slate_item("n1")])

    assert picks == []


@pytest.mark.asyncio
async def test_presenter_tool_budget_exhaustion_forces_final_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PRESENTER_MAX_TOOL_CALLS", 1)
    log_records: list[dict[str, Any]] = []
    index = FakeIndex(
        notes={
            "n1": note_record("n1"),
            "n2": note_record("n2"),
        }
    )
    llm = FakeLLM(
        responses=[{"picks": [{"note_id": "n1", "why": "answer after budget"}]}],
        tool_responses=[tool_call("n1")],
    )
    presenter = Presenter(llm, index, on_log=log_records.append)

    picks = await presenter.present([chunk(1)], [slate_item("n1")])

    assert [(pick.note_id, pick.why) for pick in picks] == [("n1", "answer after budget")]
    assert [call[0] for call in llm.tool_requests] == [config.PRESENTER_MODEL]
    assert [call[0] for call in llm.calls] == [config.PRESENTER_MODEL]
    assert log_records[0]["tool_calls"] == [{"note_id": "n1", "ok": True}]


@pytest.mark.asyncio
async def test_presenter_prompt_contains_current_slate_and_recent_transcript_data() -> None:
    llm = FakeLLM([{"picks": []}])
    presenter = Presenter(llm, FakeIndex())
    transcript = [chunk(i) for i in range(config.PRESENTER_TRANSCRIPT_CHUNKS + 3)]
    slate = [
        slate_item("exact-note", source="exact", heard="exact heard"),
        slate_item("search-note", heard="search heard"),
        slate_item("linked-note", heard="linked heard", via_note_id="seed-note"),
    ]

    await presenter.present(transcript, slate, via_titles={"seed-note": "Seed Title"})

    assert len(llm.tool_requests) == 1
    _model, _system, messages, _tools = llm.tool_requests[0]
    user = messages[0]["content"]
    for item in slate:
        assert item.note_id in user
        assert item.title in user
        assert item.gist in user
        if item.source == "search":
            assert item.heard in user
    assert "Seed Title" in user
    for recent in transcript[-config.PRESENTER_TRANSCRIPT_CHUNKS :]:
        assert recent.text in user
    assert transcript[0].text not in user


@pytest.mark.asyncio
async def test_presenter_prompt_contains_neighbor_ids_and_note_directory_titles() -> None:
    index = FakeIndex(
        notes={
            "notes/neighbor-a": note_record("notes/neighbor-a", title="Neighbor A Title"),
            "notes/neighbor-b": note_record("notes/neighbor-b", title="Neighbor B Title"),
            "notes/neighbor-c": note_record("notes/neighbor-c", title="Neighbor C Title"),
            "notes/neighbor-d": note_record("notes/neighbor-d", title="Neighbor D Title"),
        },
        neighbor_rows={
            "notes/seed": [
                ("notes/neighbor-a", 0.9),
                ("notes/neighbor-b", 0.8),
                ("notes/neighbor-c", 0.7),
                ("notes/neighbor-d", 0.6),
            ]
        },
    )
    llm = FakeLLM([{"picks": []}])
    presenter = Presenter(llm, index)

    await presenter.present([chunk(1)], [slate_item("notes/seed")])

    _model, _system, messages, _tools = llm.tool_requests[0]
    user = messages[0]["content"]
    assert "seed" in user
    assert "neighbor-a" in user
    assert "neighbor-b" in user
    assert "neighbor-c" in user
    assert "neighbor-d" not in user
    assert "Title notes/seed" in user
    assert "Neighbor A Title" in user
    assert "Neighbor B Title" in user
    assert "Neighbor C Title" in user


@pytest.mark.asyncio
async def test_presenter_prompt_includes_partials_after_recent_finals() -> None:
    llm = FakeLLM([{"picks": []}])
    presenter = Presenter(llm, FakeIndex())

    await presenter.present(
        [Chunk(channel=0, text="final words", ts=1.0, final=True, seq=1)],
        [slate_item("n1")],
        partials={2: "second channel partial", 0: "mic partial"},
    )

    _model, _system, messages, _tools = llm.tool_requests[0]
    user = messages[0]["content"]
    final_pos = user.index("channel 0: final words")
    mic_partial_pos = user.index("channel 0 (in progress): mic partial")
    channel_two_pos = user.index("channel 2 (in progress): second channel partial")
    slate_pos = user.index("<slate>")
    assert final_pos < mic_partial_pos < channel_two_pos < slate_pos


@pytest.mark.asyncio
async def test_presenter_on_log_records_success_payload() -> None:
    log_records: list[dict[str, Any]] = []
    llm = FakeLLM([{"picks": [{"note_id": "n1", "why": "useful now"}]}])
    presenter = Presenter(llm, FakeIndex(), on_log=log_records.append)

    picks = await presenter.present([chunk(1), chunk(2)], [slate_item("n1")], revision=7)

    assert [(pick.note_id, pick.why) for pick in picks] == [("n1", "useful now")]
    assert log_records == [
        {
            "revision": 7,
            "model_used": config.PRESENTER_MODEL,
            "transcript_chunks": 2,
            "slate_note_ids": ["n1"],
            "raw_response": {"picks": [{"note_id": "n1", "why": "useful now"}]},
            "picks": [{"note_id": "n1", "why": "useful now"}],
            "tool_calls": [],
            "latency_ms": pytest.approx(log_records[0]["latency_ms"]),
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_presenter_on_log_records_failure_payload() -> None:
    log_records: list[dict[str, Any]] = []
    llm = FakeLLM([LLMError("primary failed"), LLMError("fallback failed")])
    presenter = Presenter(llm, FakeIndex(), on_log=log_records.append)

    assert await presenter.present([chunk(1)], [slate_item("n1")], revision=3) == []

    assert len(log_records) == 1
    payload = log_records[0]
    assert payload["revision"] == 3
    assert payload["model_used"] is None
    assert payload["transcript_chunks"] == 1
    assert payload["slate_note_ids"] == ["n1"]
    assert payload["raw_response"] is None
    assert payload["picks"] == []
    assert payload["tool_calls"] == []
    assert isinstance(payload["latency_ms"], float)
    assert "primary failed" in str(payload["error"])
    assert "fallback failed" in str(payload["error"])
