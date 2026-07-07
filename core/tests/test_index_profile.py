from __future__ import annotations

import pytest

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core import config


class ProfileLLM:
    def __init__(self, card: str) -> None:
        self.card = card
        self.calls: list[tuple[str, str, str, float | None]] = []

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        self.calls.append((model, system, user, timeout_s))
        return {"card": self.card}


@pytest.mark.asyncio
async def test_build_user_card_assembles_truncated_notes_and_clamps_card(store, monkeypatch) -> None:
    monkeypatch.setattr(config, "PROFILE_NOTE_CHAR_BUDGET", 12)
    monkeypatch.setattr(config, "PROFILE_MAX_CHARS", 20)
    alpha = make_note("alpha", "Alpha Title", "abcdefghijklmnop")
    beta = make_note("beta", "Beta Title", "short")
    store.upsert_note(alpha, enrichment("Alpha gist"), fixture_units(unit_vec(1), enrichment("Alpha gist")))
    store.upsert_note(beta, enrichment("Beta gist"), fixture_units(unit_vec(2), enrichment("Beta gist")))
    llm = ProfileLLM("0123456789" * 3)

    from taurine_core.indexer.profile import build_user_card

    card = await build_user_card(store, llm)

    assert card == "01234567890123456789"
    assert len(llm.calls) == 1
    model, _system, user, timeout_s = llm.calls[0]
    assert model == config.PROFILE_MODEL
    assert timeout_s == config.PROFILE_TIMEOUT_S
    assert "Alpha Title" in user
    assert "abcdefghijkl" in user
    assert "abcdefghijklm" not in user
    assert "Beta Title" in user
    assert "short" in user


def test_user_card_round_trips_through_store_metadata(store) -> None:
    assert store.user_card() is None

    store.set_user_card("## User\nWorks on Taurine.")

    assert store.user_card() == "## User\nWorks on Taurine."
