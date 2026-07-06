from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from taurine_core import config
from taurine_core.live.presenter import Presenter
from taurine_core.ports import LLMError
from taurine_core.schemas import Chunk, SlateItem


@dataclass
class FakeLLM:
    responses: list[dict[str, Any] | Exception]
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    async def complete_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        self.calls.append((model, system, user))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def chunk(index: int) -> Chunk:
    return Chunk(channel=index % 2, text=f"transcript chunk {index}", ts=float(index))


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
    presenter = Presenter(llm)
    slate = [slate_item("n1"), slate_item("n2"), slate_item("n3")]

    picks = await presenter.present([chunk(1)], slate)

    assert [pick.note_id for pick in picks] == ["n1", "n2"]
    assert len(picks[0].why) == config.WHY_MAX_CHARS
    assert picks[1].why == "second"


@pytest.mark.asyncio
async def test_presenter_falls_back_once_after_primary_model_error() -> None:
    llm = FakeLLM([LLMError("primary failed"), {"picks": [{"note_id": "n1", "why": "fallback pick"}]}])
    presenter = Presenter(llm)

    picks = await presenter.present([chunk(1)], [slate_item("n1")])

    assert [call[0] for call in llm.calls] == [config.PRESENTER_MODEL, config.PRESENTER_FALLBACK_MODEL]
    assert [(pick.note_id, pick.why) for pick in picks] == [("n1", "fallback pick")]


@pytest.mark.asyncio
async def test_presenter_returns_empty_when_both_models_fail() -> None:
    llm = FakeLLM([LLMError("primary failed"), LLMError("fallback failed")])
    presenter = Presenter(llm)

    assert await presenter.present([chunk(1)], [slate_item("n1")]) == []
    assert [call[0] for call in llm.calls] == [config.PRESENTER_MODEL, config.PRESENTER_FALLBACK_MODEL]


@pytest.mark.asyncio
async def test_presenter_prompt_contains_current_slate_and_recent_transcript_data() -> None:
    llm = FakeLLM([{"picks": []}])
    presenter = Presenter(llm)
    transcript = [chunk(i) for i in range(config.PRESENTER_TRANSCRIPT_CHUNKS + 3)]
    slate = [
        slate_item("exact-note", source="exact", heard="exact heard"),
        slate_item("search-note", heard="search heard"),
        slate_item("linked-note", heard="linked heard", via_note_id="seed-note"),
    ]

    await presenter.present(transcript, slate, via_titles={"seed-note": "Seed Title"})

    assert len(llm.calls) == 1
    _model, _system, user = llm.calls[0]
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
