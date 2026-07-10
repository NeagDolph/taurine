from __future__ import annotations

import pytest

from taurine_core import config
from taurine_core.indexer.priors import recency_multiplier, source_rank


def test_every_connector_has_explicit_source_priors() -> None:
    # SOURCE_RANK/RECENCY_HALF_LIFE_DAYS keys must match real connector ids;
    # a typo silently hands the connector the default prior (2026-07-08 bug:
    # "claude_memories" vs actual "claude_mem")
    from taurine_core.connectors.catalog import CONNECTORS

    for connector in CONNECTORS:
        assert connector.connector_id in config.SOURCE_RANK
        assert connector.connector_id in config.RECENCY_HALF_LIFE_DAYS
    assert source_rank("claude_mem") == config.SOURCE_RANK["claude_mem"] == 0.70


def test_source_rank_falls_back_to_default_for_unknown_sources() -> None:
    assert source_rank("future_source") == config.SOURCE_RANK_DEFAULT


def test_recency_multiplier_applies_grace_offset_half_life_and_floor() -> None:
    # inside the grace window: no decay at all
    assert recency_multiplier("obsidian", 0.0) == pytest.approx(1.0)
    assert recency_multiplier("obsidian", config.RECENCY_OFFSET_DAYS) == pytest.approx(1.0)
    # exactly one half-life past the grace offset: floor + half the headroom
    half_life = config.RECENCY_HALF_LIFE_DAYS["imessage"]
    expected = config.RECENCY_FLOOR + (1.0 - config.RECENCY_FLOOR) * 0.5
    assert recency_multiplier(
        "imessage", config.RECENCY_OFFSET_DAYS + half_life
    ) == pytest.approx(expected)
    # decay never drops below the floor
    assert recency_multiplier("imessage", 100_000.0) == pytest.approx(
        config.RECENCY_FLOOR, abs=1e-6
    )
