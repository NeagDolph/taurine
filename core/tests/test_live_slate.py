from __future__ import annotations

import pytest

from taurine_core.config import DECAY_SEARCH, QUOTA_SEARCH
from taurine_core.live.slate import Slate
from taurine_core.schemas import SlateItem

from tests.fakes import FakeClock


def item(
    note_id: str,
    *,
    source: str = "search",
    score: float = 0.5,
    views: int = 0,
    added_rev: int = 0,
) -> SlateItem:
    return SlateItem(
        note_id=note_id,
        title=f"Title {note_id}",
        gist=f"Gist {note_id}",
        score=score,
        source=source,
        heard="heard",
        views=views,
        decay=DECAY_SEARCH,
        added_rev=added_rev,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def slate(clock: FakeClock) -> Slate:
    return Slate(clock=clock)


@pytest.fixture
def slate_full_search_lane(slate: Slate) -> Slate:
    for idx in range(QUOTA_SEARCH):
        slate.add(item(f"n{idx}", score=1.0 - (idx / 100), views=0, added_rev=idx))
    return slate


def test_search_lane_evicts_most_viewed_then_oldest(slate_full_search_lane: Slate) -> None:
    slate_full_search_lane.mark_viewed(["n2", "n5", "n5"])

    slate_full_search_lane.add(item("new"))

    ids = [i.note_id for i in slate_full_search_lane.items()]
    assert "new" in ids
    assert "n5" not in ids
    assert len(ids) == QUOTA_SEARCH


def test_search_lane_tiebreak_evicts_oldest(slate: Slate) -> None:
    for idx in range(QUOTA_SEARCH):
        slate.add(item(f"n{idx}", added_rev=idx))

    slate.add(item("new"))

    ids = [i.note_id for i in slate.items()]
    assert "new" in ids
    assert "n0" not in ids
    assert len(ids) == QUOTA_SEARCH


def test_renewal_resets_views_and_upgrades_provenance(slate: Slate) -> None:
    slate.add(item("n1", source="search", score=0.2))
    slate.mark_viewed(["n1"])

    slate.add(item("n1", source="exact", score=0.9))

    [renewed] = [i for i in slate.items() if i.note_id == "n1"]
    assert renewed.source == "exact"
    assert renewed.score == 0.9
    assert renewed.views == 0
    assert renewed.added_rev == slate.revision


def test_dismissed_note_suppressed_for_300s(slate: Slate, clock: FakeClock) -> None:
    slate.add(item("n1"))
    slate.dismiss("n1", now=clock.now())

    slate.add(item("n1"))
    assert "n1" not in [i.note_id for i in slate.items()]

    clock.advance(300.1)
    slate.add(item("n1"))
    assert "n1" in [i.note_id for i in slate.items()]


def test_mark_viewed_kicks_decayed_items(slate: Slate) -> None:
    slate.add(item("n1"))

    slate.mark_viewed(["n1"])
    slate.mark_viewed(["n1"])

    assert slate.items() == []


def test_items_sorted_by_provenance_strength_then_score(slate: Slate) -> None:
    slate.add(item("low-exact", source="exact", score=0.1))
    slate.add(item("high-search", source="search", score=0.99))
    slate.add(item("high-exact", source="exact", score=0.9))

    assert [i.note_id for i in slate.items()] == ["high-exact", "low-exact", "high-search"]


def test_slate_on_event_reports_mutations_and_suppressed_readds(clock: FakeClock) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    slate = Slate(clock=clock, on_event=lambda op, payload: events.append((op, payload)))

    slate.add(item("n1", source="search"))
    slate.add(item("n1", source="exact"))
    slate.dismiss("n1", now=clock.now())
    slate.add(item("n1", source="search"))

    assert [op for op, _payload in events] == ["add", "renew", "dismiss", "suppressed_drop"]
    assert events[0][1] == {
        "note_id": "n1",
        "source": "search",
        "reason": "added",
        "revision": 1,
    }
    assert events[1][1] == {
        "note_id": "n1",
        "source": "exact",
        "reason": "renewed",
        "revision": 2,
    }
    assert events[2][1] == {
        "note_id": "n1",
        "source": "exact",
        "reason": "dismissed",
        "revision": 3,
    }
    assert events[3][1] == {
        "note_id": "n1",
        "source": "search",
        "reason": "suppressed",
        "revision": 3,
    }


def test_slate_on_event_reports_quota_evictions(clock: FakeClock) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    slate = Slate(clock=clock, on_event=lambda op, payload: events.append((op, payload)))
    for idx in range(QUOTA_SEARCH):
        slate.add(item(f"n{idx}", source="search"))

    slate.add(item("new", source="search"))

    evictions = [(op, payload) for op, payload in events if op == "evict"]
    assert evictions == [
        (
            "evict",
            {
                "note_id": "n0",
                "source": "search",
                "reason": "quota",
                "revision": QUOTA_SEARCH + 2,
            },
        )
    ]
