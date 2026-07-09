from __future__ import annotations

import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.ports import LLMError


def _notes(count: int):
    return [
        make_note(
            f"note-{index}",
            f"Title {index}",
            f"Body text for note {index}.",
            folder=f"Folder {index % 2}",
        )
        for index in range(count)
    ]


class SlimLLM:
    def __init__(
        self,
        *,
        responses: list[dict] | None = None,
        errors: int = 0,
    ) -> None:
        self.responses = list(responses or [])
        self.errors = errors
        self.calls: list[tuple[str, str, str, float | None]] = []

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        self.calls.append((model, system, user, timeout_s))
        if self.errors > 0:
            self.errors -= 1
            raise LLMError("slim batch unavailable")
        if self.responses:
            return self.responses.pop(0)
        return {
            "notes": {
                note_id: {
                    "aliases": [f"Alias {note_id}"],
                    "utterances": [f"Let's talk about {note_id}."],
                }
                for note_id in _note_ids_in(user)
            }
        }


def _note_ids_in(user: str) -> list[str]:
    return [
        line.removeprefix("note_id: ")
        for line in user.splitlines()
        if line.startswith("note_id: ")
    ]


def test_select_slim_notes_uses_fraction_when_above_floor(monkeypatch) -> None:
    from taurine_core.indexer.slim_enrich import select_slim_notes

    monkeypatch.setattr(config, "SLIM_LLM_VALUE_FRACTION", 0.30)
    monkeypatch.setattr(config, "SLIM_LLM_MIN_NOTES", 5)
    notes = _notes(20)
    scores = {note.note_id: float(index) for index, note in enumerate(notes)}

    selected = select_slim_notes(notes, scores)

    assert [note.note_id for note in selected] == [
        "note-19",
        "note-18",
        "note-17",
        "note-16",
        "note-15",
        "note-14",
    ]


def test_select_slim_notes_uses_ceil_for_fractional_count(monkeypatch) -> None:
    from taurine_core.indexer.slim_enrich import select_slim_notes

    monkeypatch.setattr(config, "SLIM_LLM_VALUE_FRACTION", 0.15)
    monkeypatch.setattr(config, "SLIM_LLM_MIN_NOTES", 5)
    notes = _notes(41)
    scores = {
        note.note_id: float(len(notes) - index) for index, note in enumerate(notes)
    }

    selected = select_slim_notes(notes, scores)

    assert [note.note_id for note in selected] == [
        "note-0",
        "note-1",
        "note-2",
        "note-3",
        "note-4",
        "note-5",
        "note-6",
    ]


def test_select_slim_notes_prefers_earlier_original_index_at_cutoff_tie(
    monkeypatch,
) -> None:
    from taurine_core.indexer.slim_enrich import select_slim_notes

    monkeypatch.setattr(config, "SLIM_LLM_VALUE_FRACTION", 0.50)
    monkeypatch.setattr(config, "SLIM_LLM_MIN_NOTES", 0)
    notes = _notes(6)
    scores = {
        "note-0": 100.0,
        "note-1": 90.0,
        "note-2": 80.0,
        "note-3": 80.0,
        "note-4": 10.0,
        "note-5": 0.0,
    }

    selected = select_slim_notes(notes, scores)

    assert [note.note_id for note in selected] == ["note-0", "note-1", "note-2"]


def test_select_slim_notes_respects_floor_and_short_changed_set(monkeypatch) -> None:
    from taurine_core.indexer.slim_enrich import select_slim_notes

    monkeypatch.setattr(config, "SLIM_LLM_VALUE_FRACTION", 0.10)
    monkeypatch.setattr(config, "SLIM_LLM_MIN_NOTES", 5)
    notes = _notes(20)
    scores = {note.note_id: float(index) for index, note in enumerate(notes)}

    selected = select_slim_notes(notes, scores)

    assert [note.note_id for note in selected] == [
        "note-19",
        "note-18",
        "note-17",
        "note-16",
        "note-15",
    ]

    short_notes = _notes(3)
    short_scores = {note.note_id: float(index) for index, note in enumerate(short_notes)}

    selected = select_slim_notes(short_notes, short_scores)

    assert [note.note_id for note in selected] == ["note-2", "note-1", "note-0"]


@pytest.mark.asyncio
async def test_slim_enrich_batch_chunks_consecutively_and_includes_user_card(
    monkeypatch,
) -> None:
    from taurine_core.indexer.slim_enrich import slim_enrich_batch

    monkeypatch.setattr(config, "ENRICH_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "SLIM_UTTERANCES", True)
    notes = _notes(3)
    llm = SlimLLM()

    result = await slim_enrich_batch(notes, llm, user_card="## User\nWorks on Taurine.")

    assert len(llm.calls) == 2
    assert [call[0] for call in llm.calls] == [config.ENRICH_MODEL, config.ENRICH_MODEL]
    assert [call[3] for call in llm.calls] == [config.ENRICH_TIMEOUT_S, config.ENRICH_TIMEOUT_S]
    assert _note_ids_in(llm.calls[0][2]) == ["note-0", "note-1"]
    assert _note_ids_in(llm.calls[1][2]) == ["note-2"]
    first_user = llm.calls[0][2]
    assert "DATA: USER INFORMATION CARD\n" in first_user
    assert (
        "----- BEGIN USER CARD -----\n## User\nWorks on Taurine.\n"
        "----- END USER CARD -----"
    ) in first_user
    assert (
        "note_id: note-0\nTitle: Title 0\nFolder: Folder 0\n"
        "Body:\nBody text for note 0."
    ) in first_user
    assert (
        "note_id: note-1\nTitle: Title 1\nFolder: Folder 1\n"
        "Body:\nBody text for note 1."
    ) in first_user
    assert set(result) == {"note-0", "note-1", "note-2"}
    assert result["note-0"].aliases == ["Alias note-0"]
    assert result["note-2"].utterances == ["Let's talk about note-2."]


@pytest.mark.asyncio
async def test_slim_enrich_batch_validates_and_clamps_response(monkeypatch) -> None:
    from taurine_core.indexer.slim_enrich import slim_enrich_batch

    monkeypatch.setattr(config, "SLIM_UTTERANCES", True)
    note = _notes(1)[0]
    llm = SlimLLM(
        responses=[
            {
                "notes": {
                    note.note_id: {
                        "aliases": [
                            "",
                            " Alpha ",
                            "Beta",
                            "Gamma",
                            "Delta",
                            "Epsilon",
                            "Zeta",
                            "Eta",
                        ],
                        "utterances": [
                            "",
                            " First ",
                            "Second",
                            "Third",
                            "Fourth",
                            "Fifth",
                        ],
                    }
                }
            }
        ]
    )

    result = await slim_enrich_batch([note], llm, user_card=None)

    assert result[note.note_id].aliases == [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
    ]
    assert result[note.note_id].utterances == ["First", "Second", "Third", "Fourth"]


@pytest.mark.asyncio
async def test_slim_enrich_batch_omits_utterances_when_disabled(monkeypatch) -> None:
    from taurine_core.indexer.slim_enrich import slim_enrich_batch

    monkeypatch.setattr(config, "SLIM_UTTERANCES", False)
    note = _notes(1)[0]
    llm = SlimLLM(
        responses=[
            {
                "notes": {
                    note.note_id: {
                        "aliases": ["Alpha"],
                        "utterances": ["This should be ignored."],
                    }
                }
            }
        ]
    )

    result = await slim_enrich_batch([note], llm, user_card=None)

    assert "utterances" not in llm.calls[0][1].lower()
    assert result[note.note_id].aliases == ["Alpha"]
    assert result[note.note_id].utterances == []


@pytest.mark.asyncio
async def test_slim_enrich_batch_retries_once_then_degrades_without_per_note_calls(
    monkeypatch,
) -> None:
    from taurine_core.indexer import slim_enrich

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(slim_enrich.asyncio, "sleep", no_sleep)
    notes = _notes(2)
    llm = SlimLLM(errors=2)

    result = await slim_enrich.slim_enrich_batch(notes, llm, user_card=None)

    assert len(llm.calls) == 2
    assert set(result) == {"note-0", "note-1"}
    assert result["note-0"].aliases == []
    assert result["note-0"].utterances == []
    assert result["note-1"].aliases == []
    assert result["note-1"].utterances == []
