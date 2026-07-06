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
    index = FakeIndex(alias_rows=[("Q3", "q3", True)])
    collector = ExactCollector(index)

    assert collector.on_chunk(chunk("q3 planning")) == [("q3", "q3", True)]
    assert collector.on_chunk(chunk("xq3 planning")) == []
    assert collector.on_chunk(chunk("q30 planning")) == []


def test_long_alias_allows_normalized_substring_match() -> None:
    index = FakeIndex(alias_rows=[("roadmap", "roadmap", True)])
    collector = ExactCollector(index)

    assert collector.on_chunk(chunk("the product-roadmap-update is ready")) == [
        ("roadmap", "roadmap", True)
    ]


def test_empty_or_whitespace_aliases_are_ignored() -> None:
    index = FakeIndex(alias_rows=[("  ", "blank", True), ("deck", "deck", False)])
    collector = ExactCollector(index)

    assert collector.on_chunk(chunk("the deck is ready")) == [("deck", "deck", False)]
