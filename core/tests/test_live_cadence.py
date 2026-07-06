from __future__ import annotations

from copy import deepcopy

import pytest

from taurine_core import config
from taurine_core.live.cadence import Cadence

from tests.fakes import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cadence(clock: FakeClock) -> Cadence:
    return Cadence(clock)


def test_same_revision_not_represented_until_reoffer(
    cadence: Cadence,
    clock: FakeClock,
) -> None:
    assert cadence.may_present(rev=1, now=clock.now())
    cadence.note_attempt(now=clock.now())

    clock.advance(config.REOFFER_S - 0.1)
    assert not cadence.may_present(rev=1, now=clock.now())

    clock.advance(0.2)
    assert cadence.may_present(rev=1, now=clock.now())
    cadence.note_attempt(now=clock.now())

    clock.advance(config.REOFFER_S + 1.0)
    assert not cadence.may_present(rev=1, now=clock.now())


def test_empty_backoff_keyed_to_revision(cadence: Cadence, clock: FakeClock) -> None:
    assert cadence.may_present(rev=5, now=clock.now())
    cadence.note_empty(rev=5, now=clock.now())
    clock.advance(4)
    assert not cadence.may_present(rev=5, now=clock.now())
    assert cadence.may_present(rev=6, now=clock.now())


def test_rate_cap_counts_instant_pins(cadence: Cadence, clock: FakeClock) -> None:
    for _ in range(config.MAX_PROMOS_PER_MIN):
        assert cadence.instant_pin_allowed(now=clock.now())

    assert not cadence.instant_pin_allowed(now=clock.now())

    clock.advance(60.1)
    assert cadence.instant_pin_allowed(now=clock.now())


def test_debounce_applies_between_attempts(cadence: Cadence, clock: FakeClock) -> None:
    assert cadence.may_present(rev=1, now=clock.now())
    cadence.note_attempt(now=clock.now())

    clock.advance(config.DEBOUNCE_S - 0.1)
    assert not cadence.may_present(rev=2, now=clock.now())

    clock.advance(0.2)
    assert cadence.may_present(rev=2, now=clock.now())


def test_promotion_cooldown_blocks_non_instant_presentations(
    cadence: Cadence,
    clock: FakeClock,
) -> None:
    assert cadence.may_present(rev=1, now=clock.now())
    cadence.note_attempt(now=clock.now())
    cadence.note_promotion(now=clock.now())

    clock.advance(config.COOLDOWN_S - 0.1)
    assert not cadence.may_present(rev=2, now=clock.now())

    clock.advance(0.2)
    assert cadence.may_present(rev=2, now=clock.now())


def test_new_revision_escapes_empty_backoff_but_not_debounce(
    cadence: Cadence,
    clock: FakeClock,
) -> None:
    assert cadence.may_present(rev=1, now=clock.now())
    cadence.note_attempt(now=clock.now())
    cadence.note_empty(rev=1, now=clock.now())

    clock.advance(config.DEBOUNCE_S - 0.1)
    assert not cadence.may_present(rev=2, now=clock.now())


def test_explain_is_pure_and_reports_first_blocking_gate(
    cadence: Cadence,
    clock: FakeClock,
) -> None:
    assert cadence.may_present(rev=1, now=clock.now())
    cadence.note_attempt(now=clock.now())
    before = _state_without_clock(cadence)

    assert cadence.explain(revision=2, now=clock.now()) == "debounce"
    assert cadence.explain(revision=2, now=clock.now()) == "debounce"
    assert _state_without_clock(cadence) == before


def _state_without_clock(cadence: Cadence) -> dict[str, object]:
    return {
        key: deepcopy(value)
        for key, value in vars(cadence).items()
        if key != "_clock"
    }
