from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core import config
from taurine_core.ports import LLMError


class _Schema:
    def entities(self, _types: dict[str, str]) -> "_Schema":
        return self

    def relations(self, _types: list[str]) -> "_Schema":
        return self


class ScriptedExtractorModel:
    def __init__(self, *, fail_titles: set[str] | None = None) -> None:
        self.fail_titles = fail_titles or set()
        self.calls: list[str] = []

    def create_schema(self) -> _Schema:
        return _Schema()

    def batch_extract(
        self,
        texts: list[str],
        schema: object,
        *,
        batch_size: int,
        threshold: float,
        include_confidence: bool,
        max_len: int | None,
    ) -> list[dict]:
        del schema, batch_size, threshold, include_confidence, max_len
        if any(text.splitlines()[0] in self.fail_titles for text in texts):
            raise RuntimeError("batch fallback")
        return [self._raw(text) for text in texts]

    def extract(
        self,
        text: str,
        schema: object,
        *,
        threshold: float,
        include_confidence: bool,
        max_len: int | None,
    ) -> dict:
        del schema, threshold, include_confidence, max_len
        self.calls.append(text)
        title = text.splitlines()[0]
        if title in self.fail_titles:
            raise RuntimeError(f"{title} failed")
        return self._raw(text)

    def _raw(self, text: str) -> dict:
        self.calls.append(text)
        title = text.splitlines()[0]
        return {
            "entities": {
                "project": [{"text": f"{title} Project", "confidence": 0.95}],
                "concept": [{"text": "reenrich topic", "confidence": 0.9}],
            },
            "relation_extraction": {},
        }


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, float | None]] = []

    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        self.calls.append((model, system, user, timeout_s))
        if timeout_s == config.PROFILE_TIMEOUT_S:
            return {"card": "## User\nWorks on Alpha."}
        if model == config.ADJUDICATE_MODEL:
            return {"pairs": {}}
        return {
            "notes": {
                "alpha": {
                    "aliases": ["fresh alpha"],
                    "utterances": ["Fresh alpha question"],
                },
                "imsg": {
                    "aliases": ["fresh chat"],
                    "utterances": ["Fresh chat question"],
                },
            }
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
async def test_reenrich_all_rebuilds_profile_and_keeps_prior_extraction_on_note_failure(
    store, monkeypatch
) -> None:
    from taurine_core.indexer import extract
    from taurine_core.indexer.reenrich import reenrich_all

    model = ScriptedExtractorModel(fail_titles={"Beta"})
    monkeypatch.setattr(extract, "_extractor_model", lambda: model)
    alpha = make_note("alpha", "Alpha", "Alpha body")
    beta = make_note("beta", "Beta", "Beta body")
    prior_alpha = enrichment("Old alpha gist", topics=["old"])
    prior_beta = enrichment("Old beta gist", aliases=["old beta"], topics=["old"])
    store.upsert_note(alpha, prior_alpha, fixture_units(unit_vec(1), prior_alpha))
    store.upsert_note(beta, prior_beta, fixture_units(unit_vec(2), prior_beta))
    progress = []
    llm = ScriptedLLM()

    report = await reenrich_all(store, llm, progress.append)

    alpha_enrichment = store.enrichment_for("alpha")
    beta_enrichment = store.enrichment_for("beta")
    assert report.total == 2
    assert report.changed == 1
    assert report.failures == 1
    assert store.user_card() == "## User\nWorks on Alpha."
    assert [entity.surface for entity in alpha_enrichment.entities] == [
        "Alpha Project",
        "reenrich topic",
    ]
    assert [entity.surface for entity in beta_enrichment.entities] == [
        "old",
    ]
    assert ("fresh alpha", "alpha", False) in store.aliases()
    assert [status.phase for status in progress if status.done == 0][:2] == ["profile", "notes"]
    assert progress[-1].running is False
    assert progress[-1].phase == "idle"
    assert progress[-1].done == 2
    assert progress[-1].total == 2


@pytest.mark.asyncio
async def test_reenrich_all_keeps_old_card_when_profile_fails(store, monkeypatch) -> None:
    from taurine_core.indexer import extract
    from taurine_core.indexer.reenrich import reenrich_all

    class ProfileFailingLLM(ScriptedLLM):
        async def complete_json(
            self, model: str, system: str, user: str, timeout_s: float | None = None
        ) -> dict:
            if timeout_s == config.PROFILE_TIMEOUT_S:
                raise LLMError("profile failed")
            return await super().complete_json(model, system, user, timeout_s)

    monkeypatch.setattr(extract, "_extractor_model", lambda: ScriptedExtractorModel())
    alpha = make_note("alpha", "Alpha", "Alpha body")
    prior_alpha = enrichment("Old alpha gist", topics=["old"])
    store.upsert_note(alpha, prior_alpha, fixture_units(unit_vec(1), prior_alpha))
    store.set_user_card("## User\nExisting card.")

    report = await reenrich_all(store, ProfileFailingLLM(), lambda _status: None)

    assert report.profile_failed is True
    assert store.user_card() == "## User\nExisting card."
    assert [entity.surface for entity in store.enrichment_for("alpha").entities] == [
        "Alpha Project",
        "reenrich topic",
    ]


@pytest.mark.asyncio
async def test_reenrich_scoped_to_source_skips_profile_and_other_sources(
    store, monkeypatch
) -> None:
    from taurine_core.indexer import extract
    from taurine_core.indexer.reenrich import reenrich_all

    monkeypatch.setattr(extract, "_extractor_model", lambda: ScriptedExtractorModel())
    alpha = make_note("alpha", "Alpha", "Alpha body")
    chat = make_note("imsg", "iMessage: Jonah", "chat body").model_copy(
        update={"source": "imessage"}
    )
    prior_alpha = enrichment("Old alpha gist", topics=["old"])
    prior_chat = enrichment("Old chat gist", topics=["old"])
    store.upsert_note(alpha, prior_alpha, fixture_units(unit_vec(1), prior_alpha))
    store.upsert_note(chat, prior_chat, fixture_units(unit_vec(2), prior_chat))
    store.set_user_card("existing card")
    progress = []
    llm = ScriptedLLM()

    report = await reenrich_all(store, llm, progress.append, source="imessage")

    assert report.total == 1
    assert report.changed == 1
    assert report.profile_failed is False
    assert store.user_card() == "existing card"
    assert [entity.surface for entity in store.enrichment_for("alpha").entities] == ["old"]
    assert [entity.surface for entity in store.enrichment_for("imsg").entities] == [
        "iMessage: Jonah Project",
        "reenrich topic",
    ]
    assert ("fresh chat", "imsg", False) in store.aliases()
    assert all(status.phase != "profile" for status in progress)
    assert any("existing card" in call[2] for call in llm.calls if call[0] != config.ADJUDICATE_MODEL)
