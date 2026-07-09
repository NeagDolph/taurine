from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from tests.conftest import enrichment, fixture_units, make_note, unit_vec
from taurine_core import config
from taurine_core.connectors.email_common import (
    EmailMessage,
    EmailThread,
    funnel_threads,
)
from taurine_core import pii
from taurine_core.schemas import EntityMention, Extraction, NoteRecord

_NOW = 1_760_000_000.0
_DAY = 86_400.0


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(pii, "_pii_model", lambda: None)


def _install_note(
    store: object,
    note_id: str,
    *,
    source: str = "apple_notes",
    modified: float = _NOW,
    seed: int = 1,
) -> NoteRecord:
    enrich = enrichment(f"{note_id} gist", topics=["value"])
    note = make_note(note_id, note_id.title(), f"{note_id} body", modified=modified)
    note = note.model_copy(update={"source": source})
    store.upsert_note(note, enrich, fixture_units(unit_vec(seed), enrich))  # type: ignore[attr-defined]
    return note


def _msg(
    *,
    sender: str = "alice@example.com",
    age_days: float = 1.0,
    from_me: bool = False,
    recipients: tuple[str, ...] = (),
) -> EmailMessage:
    return EmailMessage(
        text="checking in",
        sender_name=sender.split("@", 1)[0].title(),
        sender_email=sender,
        timestamp=_NOW - age_days * _DAY,
        is_from_me=from_me,
        recipients=recipients,
    )


def test_ingest_salience_read_helper_round_trips_defaults(store) -> None:
    _install_note(store, "alpha", seed=10)
    _install_note(store, "beta", seed=11)

    store.set_ingest_salience({"alpha": 1.75, "missing": 9.0})

    assert store.ingest_salience() == {
        "alpha": pytest.approx(1.75),
        "beta": pytest.approx(1.0),
    }


def test_compute_value_scores_applies_formula_floor_default_source_and_persists(
    store,
) -> None:
    from taurine_core.indexer.rank import compute_value_scores

    half_life = config.VALUE_RECENCY_HALF_LIFE_DAYS
    _install_note(store, "alpha", source="obsidian", modified=_NOW, seed=20)
    _install_note(
        store,
        "beta",
        source="gmail",
        modified=_NOW - half_life * _DAY,
        seed=21,
    )
    _install_note(
        store,
        "gamma",
        source="future_source",
        modified=_NOW - 2.0 * half_life * _DAY,
        seed=22,
    )
    store.set_quality_ranks({"alpha": 2.0, "beta": -1.0, "gamma": 0.5})
    store.set_ingest_salience({"alpha": 0.5, "beta": 2.0})

    scores = compute_value_scores(store, now=_NOW)

    assert scores == {
        "alpha": pytest.approx(1.0 * 1.0 * config.SOURCE_RANK["obsidian"] * 0.5),
        "beta": pytest.approx(0.1 * 0.5 * config.SOURCE_RANK["gmail"] * 2.0),
        "gamma": pytest.approx(0.55 * 0.25 * config.SOURCE_RANK_DEFAULT * 1.0),
    }
    assert store.value_scores() == pytest.approx(scores)


def test_compute_value_scores_gives_all_equal_quality_ranks_full_quality_weight(
    store,
) -> None:
    from taurine_core.indexer.rank import compute_value_scores

    _install_note(store, "fresh", source="obsidian", modified=_NOW, seed=30)
    _install_note(
        store,
        "older",
        source="obsidian",
        modified=_NOW - config.VALUE_RECENCY_HALF_LIFE_DAYS * _DAY,
        seed=31,
    )
    store.set_quality_ranks({"fresh": 4.0, "older": 4.0})

    scores = compute_value_scores(store, now=_NOW)

    assert scores["fresh"] == pytest.approx(1.0)
    assert scores["older"] == pytest.approx(0.5)


def test_compute_value_scores_excludes_disabled_sources_from_active_set(store) -> None:
    from taurine_core.indexer.rank import compute_value_scores

    _install_note(store, "active", source="obsidian", modified=_NOW, seed=40)
    _install_note(store, "disabled", source="gmail", modified=_NOW, seed=41)
    store.set_quality_ranks({"active": 1.0, "disabled": 2.0})
    store.set_source_enabled("gmail", False)

    scores = compute_value_scores(store, now=_NOW)

    assert scores == {"active": pytest.approx(1.0)}
    assert store.value_scores()["disabled"] == pytest.approx(0.0)


def test_funnel_report_carries_minmax_normalized_salience_scores() -> None:
    low = EmailThread(
        thread_id="low",
        subject="Low",
        messages=[_msg(age_days=60.0)],
    )
    high = EmailThread(
        thread_id="high",
        subject="High",
        flagged=True,
        messages=[
            _msg(
                sender="me@example.com",
                age_days=0.2,
                from_me=True,
                recipients=("bob@corp.com",),
            ),
            _msg(sender="bob@corp.com", age_days=0.1),
        ],
    )

    kept, report = funnel_threads([low, high], contacts={}, now=_NOW)

    assert [thread.thread_id for thread in kept] == ["high", "low"]
    assert report.scores == {"high": pytest.approx(1.0), "low": pytest.approx(0.1)}


@pytest.mark.parametrize(
    ("connector_cls", "module_path", "env", "prefix"),
    [
        (
            "GmailConnector",
            "taurine_core.connectors.gmail",
            {
                "GMAIL_CLIENT_ID": "cid",
                "GMAIL_CLIENT_SECRET": "csec",
                "GMAIL_REFRESH_TOKEN": "rtok",
            },
            "gmail",
        ),
        (
            "OutlookConnector",
            "taurine_core.connectors.outlook",
            {"OUTLOOK_CLIENT_ID": "cid", "OUTLOOK_REFRESH_TOKEN": "rtok"},
            "outlook",
        ),
    ],
)
def test_email_connectors_expose_funnel_scores_for_survivor_note_ids(
    connector_cls: str,
    module_path: str,
    env: dict[str, str],
    prefix: str,
    monkeypatch,
) -> None:
    module = __import__(module_path, fromlist=[connector_cls])
    connector = getattr(module, connector_cls)()
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(connector, "_token", lambda client: "token")
    monkeypatch.setattr(
        module,
        "_fetch_threads",
        lambda client, token, window_days: [
            EmailThread("low", "Low", [_msg(age_days=60.0)]),
            EmailThread(
                "high",
                "High",
                [
                    _msg(
                        sender="me@example.com",
                        age_days=0.2,
                        from_me=True,
                        recipients=("bob@corp.com",),
                    ),
                    _msg(sender="bob@corp.com", age_days=0.1),
                ],
                flagged=True,
            ),
        ],
    )
    monkeypatch.setattr("time.time", lambda: _NOW)

    records = connector.extract({"contacts_glob": "/nonexistent/**/*.abcddb"})

    assert [record.note_id for record in records] == [f"{prefix}:high", f"{prefix}:low"]
    assert connector.ingest_salience_scores == {
        f"{prefix}:high": pytest.approx(1.0),
        f"{prefix}:low": pytest.approx(0.1),
    }


@dataclass
class _FakeEmailConnector:
    records: list[NoteRecord]
    ingest_salience_scores: dict[str, float]
    connector_id: str = "gmail"
    label: str = "Gmail"
    kind: str = "oauth"
    configs_seen: list[dict[str, Any]] = field(default_factory=list)

    def available(self) -> tuple[bool, str]:
        return True, ""

    def default_config(self) -> dict:
        return {}

    def extract(self, config: dict) -> list[NoteRecord]:
        self.configs_seen.append(dict(config))
        return list(self.records)


def test_registry_persists_email_salience_after_sync_upserts(
    store, monkeypatch
) -> None:
    from taurine_core.connectors.registry import ConnectorRegistry
    from taurine_core.indexer import sync

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        rows = [unit_vec(index + 200) for index, _text in enumerate(texts)]
        return (
            np.vstack(rows)
            if rows
            else np.empty((0, config.EMBED_DIM), dtype=np.float32)
        )

    monkeypatch.setattr(sync, "embed_texts", fake_embed_texts)
    _install_note(store, "outlook:thread-2", source="outlook", seed=50)
    store.set_ingest_salience({"outlook:thread-2": 0.77})
    note = make_note("gmail:thread-1", "Thread 1", "Body", modified=_NOW).model_copy(
        update={"source": "gmail"}
    )
    store.cache_extraction(
        note.note_id,
        note.content_hash,
        Extraction(
            entities=[
                EntityMention(surface="email", etype="concept", conf=0.9, mentions=1)
            ],
            relations=[],
            failed=False,
        ),
    )
    connector = _FakeEmailConnector(
        records=[note],
        ingest_salience_scores={"gmail:thread-1": 0.42},
    )

    report = ConnectorRegistry([connector]).sync_connector("gmail", store, None)  # type: ignore[arg-type]

    assert report.added == 1
    assert store.note("gmail:thread-1") is not None
    assert store.ingest_salience()["gmail:thread-1"] == pytest.approx(0.42)
    assert store.ingest_salience()["outlook:thread-2"] == pytest.approx(0.77)
