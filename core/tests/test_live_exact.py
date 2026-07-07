from __future__ import annotations

from taurine_core.live.exact import ExactCollector
from taurine_core.schemas import Chunk

from tests.fakes import FakeIndex


def chunk(text: str) -> Chunk:
    return Chunk(channel=0, text=text, ts=0.0)


def test_aliases_are_normalized_for_punctuation_and_case() -> None:
    index = FakeIndex(
        alias_rows=[
            ("Road-Map", "roadmap", False),
            ("Launch Plan", "launch", True),
        ]
    )
    collector = ExactCollector(index)

    hits = collector.on_chunk(chunk("Can we revisit the road, map and LAUNCH-plan?"))

    assert hits == [
        ("roadmap", "road map", False),
        ("launch", "launch plan", True),
    ]


def test_short_alias_requires_word_boundary_match() -> None:
    index = FakeIndex(alias_rows=[("Q3A", "q3a", True)])
    collector = ExactCollector(index)

    assert collector.on_chunk(chunk("q3a planning")) == [("q3a", "q3a", True)]
    assert collector.on_chunk(chunk("xq3a planning")) == []
    assert collector.on_chunk(chunk("q3alpha planning")) == []


def test_long_alias_allows_normalized_substring_match() -> None:
    index = FakeIndex(alias_rows=[("roadmap", "roadmap", True)])
    collector = ExactCollector(index)

    assert collector.on_chunk(chunk("the product-roadmap-update is ready")) == [
        ("roadmap", "roadmap", True)
    ]


def test_aliases_always_require_word_boundaries() -> None:
    index = FakeIndex(
        alias_rows=[
            ("comp", "comp", False),
            ("dan ham", "dan-ham", True),
            ("wink", "wink", False),
        ]
    )
    collector = ExactCollector(index)

    assert collector.on_chunk(chunk("our company is growing and winking")) == []
    assert collector.on_chunk(chunk("talked to Dan Ham today")) == [
        ("dan-ham", "dan ham", True)
    ]


def test_empty_or_whitespace_aliases_are_ignored() -> None:
    index = FakeIndex(alias_rows=[("  ", "blank", True), ("deck", "deck", False)])
    collector = ExactCollector(index)

    assert collector.on_chunk(chunk("the deck is ready")) == [("deck", "deck", False)]


def test_short_and_all_stopword_aliases_are_filtered() -> None:
    index = FakeIndex(
        alias_rows=[
            ("me", "me", True),
            ("note", "note", True),
            ("New Note", "new-note", True),
            ("Sunil", "sunil", True),
            ("Ashish Parik", "ashish", False),
            ("note to sunil", "note-to-sunil", False),
        ]
    )
    collector = ExactCollector(index)

    hits = collector.on_chunk(chunk("me note new note sunil ashish parik note to sunil"))

    assert hits == [
        ("sunil", "sunil", True),
        ("ashish", "ashish parik", False),
        ("note-to-sunil", "note to sunil", False),
    ]
