from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest

from tests.conftest import make_note
from taurine_core import config
from taurine_core.ports import LLMError


class StubLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def complete_json(self, model: str, system: str, user: str) -> dict:
        self.calls.append((model, system, user))
        return self.response


class FailingLLM:
    async def complete_json(self, model: str, system: str, user: str) -> dict:
        raise LLMError("boom")


@pytest.mark.asyncio
async def test_enrich_note_validates_and_clamps_response():
    from taurine_core.indexer.enrich import enrich_note

    note = make_note("alpha", "Alpha", "First line\nSecond line")
    llm = StubLLM(
        {
            "gist": "alpha " * 50,
            "utterances": ["Talk about alpha", 42, ""],
            "aliases": [" A ", "B", "C", "D", "E", "F", "G"],
            "topics": [" Work ", "Planning", "ROADMAP", "Billing", "Extra"],
            "entities": [" Taurine ", "", 123],
            "ignored": "value",
        }
    )

    result = await enrich_note(note, llm)

    assert len(llm.calls) == 1
    _model, system_prompt, _user_prompt = llm.calls[0]
    assert str(config.GIST_MAX_CHARS) in system_prompt
    assert result.gist == ("alpha " * 36).strip()
    assert len(result.gist) <= config.GIST_MAX_CHARS
    assert result.utterances == ["Talk about alpha", "42"]
    assert result.aliases == ["A", "B", "C", "D", "E", "F"]
    assert result.topics == ["work", "planning", "roadmap", "billing"]
    assert result.entities == ["Taurine", "123"]


@pytest.mark.asyncio
async def test_enrich_note_falls_back_on_llm_error():
    from taurine_core.indexer.enrich import enrich_note_with_status

    note = make_note("alpha", "Alpha Title", "First useful line\nSecond line")

    result, failed = await enrich_note_with_status(note, FailingLLM())

    assert failed is True
    assert result.gist == "First useful line"
    assert result.aliases == ["Alpha Title"]
    assert result.topics == []
    assert result.entities == []
    assert result.utterances == []


@dataclass
class FakeFolder:
    name: str


@dataclass
class FakeRawNote:
    id: str
    name: str
    body: str
    modification_date: float
    folder: FakeFolder


class FakeAccount:
    def notes(self) -> list[FakeRawNote]:
        return [
            FakeRawNote(
                id="notes://alpha",
                name="Alpha",
                body="<div>Hello <b>world</b><br>Next line&nbsp;here</div>",
                modification_date=123.0,
                folder=FakeFolder("Projects"),
            )
        ]


class FakeNotesApp:
    @property
    def default_account(self) -> FakeAccount:
        return FakeAccount()


def test_extract_notes_converts_html_and_hashes_content():
    from taurine_core.indexer.notes import extract_notes

    records = extract_notes(app=FakeNotesApp())

    assert len(records) == 1
    assert records[0].note_id == "notes://alpha"
    assert records[0].title == "Alpha"
    assert records[0].folder == "Projects"
    assert records[0].modified == 123.0
    assert records[0].text == "Hello world\nNext line here"
    assert len(records[0].content_hash) == 64


@pytest.mark.skipif(
    sys.platform != "darwin" or not os.environ.get("TAURINE_NOTES_SMOKE_ID"),
    reason="requires macOS and TAURINE_NOTES_SMOKE_ID",
)
def test_open_note_smoke() -> None:
    from taurine_core.indexer.notes import open_note

    open_note(os.environ["TAURINE_NOTES_SMOKE_ID"])
