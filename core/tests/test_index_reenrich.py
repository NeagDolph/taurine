from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core import config
from taurine_core.ports import LLMError


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, float | None]] = []

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        self.calls.append((model, system, user, timeout_s))
        if timeout_s == config.PROFILE_TIMEOUT_S:
            return {"card": "## User\nWorks on Alpha."}
        if "Beta body" in user:
            raise LLMError("beta failed")
        return {
            "gist": "Fresh alpha gist",
            "utterances": ["Fresh alpha question"],
            "aliases": ["fresh alpha"],
            "topics": ["alpha"],
            "entities": ["Alpha"],
        }


@pytest.fixture(autouse=True)
def deterministic_reenrich_embeddings(monkeypatch):
    from taurine_core.indexer import sync

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            row = rng.normal(size=config.EMBED_DIM).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row)
        return np.vstack(rows) if rows else np.empty((0, config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr(sync, "embed_texts", fake_embed_texts)


@pytest.mark.asyncio
async def test_reenrich_all_rebuilds_profile_and_keeps_prior_enrichment_on_note_failure(store) -> None:
    from taurine_core.indexer.reenrich import reenrich_all

    alpha = make_note("alpha", "Alpha", "Alpha body")
    beta = make_note("beta", "Beta", "Beta body")
    prior_alpha = enrichment("Old alpha gist", topics=["old"])
    prior_beta = enrichment("Old beta gist", aliases=["old beta"], topics=["old"])
    store.upsert_note(alpha, prior_alpha, fixture_units(unit_vec(1), prior_alpha))
    store.upsert_note(beta, prior_beta, fixture_units(unit_vec(2), prior_beta))
    progress = []
    llm = ScriptedLLM()

    report = await reenrich_all(store, llm, progress.append)

    assert report.total == 2
    assert report.changed == 1
    assert report.failures == 1
    assert store.user_card() == "## User\nWorks on Alpha."
    assert store.enrichment_for("alpha").gist == "Fresh alpha gist"
    assert store.enrichment_for("beta") == prior_beta
    assert [status.phase for status in progress if status.done == 0][:2] == ["profile", "notes"]
    assert progress[-1].running is False
    assert progress[-1].phase == "idle"
    assert progress[-1].done == 2
    assert progress[-1].total == 2


@pytest.mark.asyncio
async def test_reenrich_all_keeps_old_card_when_profile_fails(store) -> None:
    from taurine_core.indexer.reenrich import reenrich_all

    class ProfileFailingLLM(ScriptedLLM):
        async def complete_json(
            self, model: str, system: str, user: str, timeout_s: float | None = None
        ) -> dict:
            if timeout_s == config.PROFILE_TIMEOUT_S:
                raise LLMError("profile failed")
            return await super().complete_json(model, system, user, timeout_s)

    alpha = make_note("alpha", "Alpha", "Alpha body")
    prior_alpha = enrichment("Old alpha gist", topics=["old"])
    store.upsert_note(alpha, prior_alpha, fixture_units(unit_vec(1), prior_alpha))
    store.set_user_card("## User\nExisting card.")

    report = await reenrich_all(store, ProfileFailingLLM(), lambda _status: None)

    assert report.profile_failed is True
    assert store.user_card() == "## User\nExisting card."
    assert store.enrichment_for("alpha").gist == "Fresh alpha gist"
