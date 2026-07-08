from __future__ import annotations

import numpy as np
import pytest

from taurine_core import config
from taurine_core.live.grounding import GroundingUpdate
from taurine_core.live.trigger import SummaryTrigger
from taurine_core.schemas import GroundingItem

from tests.fakes import FakeClock


@pytest.fixture(autouse=True)
def trigger_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SUMMARY_MIN_INTERVAL_S", 10.0)
    monkeypatch.setattr(config, "SUMMARY_MAX_STALENESS_S", 60.0)
    monkeypatch.setattr(config, "SUMMARY_MIN_NEW_CHARS", 5)
    monkeypatch.setattr(config, "SUMMARY_CHANGE_TOP_K", 3)
    monkeypatch.setattr(config, "SUMMARY_CHANGE_JACCARD", 0.6)
    monkeypatch.setattr(config, "SUMMARY_DRIFT_COSINE", 0.70)
    monkeypatch.setattr(config, "TOPIC_CENTROID_DECAY", 0.8)
    monkeypatch.setattr(config, "SUMMARY_EMPTY_BACKOFF_S", 40.0)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_first_offer_fires_with_jaccard(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)

    assert trigger.offer(_update(["a", "b"], revision=1), None, 5) == (True, "jaccard")


def test_jaccard_compares_against_last_accepted_summary_set(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)
    first = _update(["a", "b", "c"], revision=1)
    changed = _update(["d", "e", "f"], revision=2)

    assert trigger.offer(first, None, 5) == (True, "jaccard")
    trigger.note_summary(first, no_update=False)

    clock.advance(10.1)
    assert trigger.offer(changed, None, 5) == (True, "jaccard")
    trigger.note_summary(changed, no_update=False)

    clock.advance(10.1)
    assert trigger.offer(first, None, 5) == (True, "jaccard")


def test_drift_fire_resets_centroid_to_current_window(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)
    update = _update(["a"], revision=1)
    east = np.array([1.0, 0.0], dtype=np.float32)
    north = np.array([0.0, 1.0], dtype=np.float32)

    assert trigger.offer(update, east, 5) == (True, "jaccard")
    trigger.note_summary(update, no_update=False)

    clock.advance(10.1)
    assert trigger.offer(update, north, 5) == (True, "drift")
    assert trigger._topic_centroid is not None
    assert float(np.dot(trigger._topic_centroid, north)) == pytest.approx(1.0)


def test_staleness_fires_when_same_sources_are_old(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)
    update = _update(["a", "b"], revision=1)

    assert trigger.offer(update, None, 5) == (True, "jaccard")
    trigger.note_summary(update, no_update=False)

    clock.advance(60.1)
    assert trigger.offer(update, None, 5) == (True, "staleness")


def test_min_interval_and_min_new_chars_block_before_change_detection(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)
    first = _update(["a"], revision=1)
    changed = _update(["b"], revision=2)

    assert trigger.offer(first, None, 5) == (True, "jaccard")
    trigger.note_summary(first, no_update=False)

    assert trigger.offer(changed, None, 5) == (False, "min_interval")

    clock.advance(10.1)
    assert trigger.offer(changed, None, 4) == (False, "min_new_chars")


def test_no_change_blocks_when_all_trigger_arms_are_quiet(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)
    update = _update(["a", "b"], revision=1)

    assert trigger.offer(update, np.array([1.0, 0.0]), 5) == (True, "jaccard")
    trigger.note_summary(update, no_update=False)

    clock.advance(10.1)
    assert trigger.offer(update, np.array([1.0, 0.0]), 5) == (False, "no_change")


def test_empty_backoff_blocks_same_revision_but_not_new_revision(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)
    first = _update(["a"], revision=1)
    new_revision = _update(["a"], revision=2)

    assert trigger.offer(first, None, 5) == (True, "jaccard")
    trigger.note_summary(first, no_update=True)

    clock.advance(10.1)
    assert trigger.offer(first, None, 5) == (False, "empty_backoff")
    assert trigger.offer(new_revision, None, 5) == (True, "jaccard")


def test_rejected_pass_records_min_interval_without_accepting_or_backoff(clock: FakeClock) -> None:
    trigger = SummaryTrigger(clock.now)
    update = _update(["a"], revision=1)

    assert trigger.offer(update, None, 5) == (True, "jaccard")
    trigger.note_rejected()

    assert trigger.offer(update, None, 5) == (False, "min_interval")
    clock.advance(10.1)
    assert trigger.offer(update, None, 5) == (True, "jaccard")


def _update(note_ids: list[str], *, revision: int) -> GroundingUpdate:
    return GroundingUpdate(
        items=[
            GroundingItem(
                note_id=note_id,
                short_id=f"N{i}",
                title=f"Note {note_id}",
                gist=f"Gist {note_id}",
                score=float(len(note_ids) - i),
            )
            for i, note_id in enumerate(note_ids, start=1)
        ],
        revision=revision,
    )
