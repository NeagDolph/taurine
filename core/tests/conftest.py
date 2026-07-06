from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from taurine_core import config
from taurine_core.schemas import Enrichment, NoteRecord


def make_note(
    note_id: str,
    title: str,
    text: str,
    *,
    folder: str = "Fixture",
    modified: float = 1.0,
) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=title,
        folder=folder,
        modified=modified,
        content_hash=hashlib.sha256((title + text).encode("utf-8")).hexdigest(),
        text=text,
    )


def unit_vec(seed: int, *, dim: int = config.EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=dim).astype(np.float32)
    values /= np.linalg.norm(values)
    return values


def shifted_vec(base: np.ndarray, seed: int, *, noise: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = base + rng.normal(scale=noise, size=base.shape).astype(np.float32)
    values /= np.linalg.norm(values)
    return values.astype(np.float32)


def enrichment(
    gist: str,
    *,
    aliases: list[str] | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    utterances: list[str] | None = None,
) -> Enrichment:
    return Enrichment(
        gist=gist,
        utterances=utterances
        or [
            f"Can we discuss {gist.lower()}?",
            f"What changed about {gist.lower()}?",
            f"I need the context on {gist.lower()}.",
            f"Let's revisit {gist.lower()}.",
        ],
        aliases=aliases or [],
        topics=topics or ["work"],
        entities=entities or [],
    )


ROADMAP_BASE = unit_vec(10)
HUB_BASE = unit_vec(20)
GROCERY_BASE = unit_vec(30)


FIXTURE_CORPUS: list[tuple[NoteRecord, Enrichment, np.ndarray]] = [
    (
        make_note(
            "roadmap-q3",
            "Q3 Roadmap",
            "Q3 focuses on onboarding, billing polish, and faster activation.",
        ),
        enrichment(
            "Roadmap prioritizes onboarding, billing polish, and activation.",
            aliases=["Q3", "activation roadmap"],
            topics=["roadmap", "planning", "work"],
            entities=["Taurine", "Billing"],
        ),
        ROADMAP_BASE,
    ),
    (
        make_note(
            "roadmap-q4",
            "Q4 Roadmap",
            "Q4 continues onboarding, billing polish, and activation experiments.",
        ),
        enrichment(
            "Roadmap prioritizes onboarding, billing polish, and activation.",
            aliases=["Q4", "activation roadmap"],
            topics=["roadmap", "planning", "work"],
            entities=["Taurine", "Billing"],
        ),
        shifted_vec(ROADMAP_BASE, 11),
    ),
    (
        make_note(
            "standup-notes",
            "Standup Notes",
            "Daily work updates with blockers, deployment status, and support queue notes.",
        ),
        enrichment(
            "Daily standup captures blockers and deployment updates.",
            aliases=["standup"],
            topics=["work", "operations"],
            entities=[],
        ),
        unit_vec(40),
    ),
    (
        make_note(
            "grocery-list",
            "Grocery List",
            "Milk, oats, lemons, coffee, rice, and dish soap.",
        ),
        enrichment(
            "Personal grocery checklist for the next shop.",
            aliases=["groceries"],
            topics=["work"],
            entities=[],
        ),
        GROCERY_BASE,
    ),
    (
        make_note(
            "q3",
            "Q3",
            "Short title boundary fixture for exact matching tests in the live loop.",
        ),
        enrichment(
            "Short Q3 note used for exact title boundaries.",
            aliases=["quarter three"],
            topics=["work", "planning"],
            entities=["Taurine"],
        ),
        shifted_vec(ROADMAP_BASE, 12, noise=0.05),
    ),
]

for i in range(1, 19):
    FIXTURE_CORPUS.append(
        (
            make_note(
                f"fixture-work-{i}",
                f"Fixture Work {i}",
                f"General work note {i} with a separate subject and no shared roadmap terms.",
            ),
            enrichment(
                f"General work note {i} about an unrelated subject.",
                aliases=[f"work {i}"],
                topics=["work", f"topic-{i}"],
                entities=[],
            ),
            unit_vec(100 + i),
        )
    )


def fixture_units(vector: np.ndarray, enrich: Enrichment) -> list[tuple[str, str, np.ndarray]]:
    return [
        ("gist", enrich.gist, vector),
        ("chunk", enrich.gist, vector),
        ("utterance", enrich.utterances[0], vector),
    ]


def install_fixture(
    index: object,
    rows: Iterable[tuple[NoteRecord, Enrichment, np.ndarray]],
) -> None:
    for note, enrich, vector in rows:
        index.upsert_note(note, enrich, fixture_units(vector, enrich))  # type: ignore[attr-defined]
    index.rebuild_graph()  # type: ignore[attr-defined]


@pytest.fixture
def store(tmp_path: Path):
    from taurine_core.indexer.store import Index

    return Index(tmp_path / "taurine-test.db")


@pytest.fixture
def fixture_index(tmp_path: Path):
    from taurine_core.indexer.store import Index

    index = Index(tmp_path / "fixture.db")
    install_fixture(index, FIXTURE_CORPUS)
    return index


@pytest.fixture
def fixture_index_hub(tmp_path: Path):
    from taurine_core.indexer.store import Index

    index = Index(tmp_path / "hub.db")
    rows: list[tuple[NoteRecord, Enrichment, np.ndarray]] = [
        (
            make_note("hub", "Hub", "Central launch note that mentions every integration."),
            enrichment(
                "Hub note connects many launch integrations.",
                topics=["launch", "work"],
                entities=[f"Project-{i}" for i in range(20)],
            ),
            HUB_BASE,
        )
    ]
    for i in range(20):
        angle = i / 1000
        vector = shifted_vec(HUB_BASE, 500 + i, noise=0.02 + angle)
        rows.append(
            (
                make_note(
                    f"hub-candidate-{i}",
                    f"Hub Candidate {i}",
                    f"Candidate {i} shares the central Project-{i} entity.",
                ),
                enrichment(
                    f"Candidate {i} is linked to the launch hub.",
                    topics=["launch", "work"],
                    entities=[f"Project-{i}"],
                ),
                vector,
            )
        )
    install_fixture(index, rows)
    return index
