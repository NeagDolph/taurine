"""Edge-audit harness tests: stratified sampling, LLM scoring, report
rendering, defensive schema handling, and ablation diagnostics.

No network: the LLM judge is a fake, the embedder is deterministic, and the
retrieval engine for ablation is wired with fakes (pattern:
tests/test_graph_retrieval_integration.py / test_retrieval_runtime.py).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_note
from taurine_core.evaluation.edge_audit import (
    DEFAULT_QUERIES_PATH,
    aggregate_scores,
    build_flags,
    corpus_stats,
    latest_build_id,
    load_queries,
    main,
    render_report,
    run_ablation,
    sample_edges,
    score_edges,
)
from taurine_core.evaluation.graph_retrieval import load_calibration
from taurine_core.indexer.build import BuildCoordinator, BuildDependencies
from taurine_core.ports import LLMError
from taurine_core.retrieval.calibration import AbstentionPolicy, HeadCalibrator
from taurine_core.retrieval.evidence import SharedRetrievalEngine
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.rerank import LocalPassageReranker
from taurine_core.retrieval.snapshot import load_index_snapshot
from taurine_core.schemas import ExtractedMention, ExtractedRelation, Extraction


CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "evals" / "graph_retrieval" / "calibration-v1.json"
)


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def embedder(texts: list[str]) -> np.ndarray:
    """Deterministic embedder; texts mentioning zebras share one vector so the
    build emits semantic edges (cosine 1.0 clears the calibrated floor)."""

    rows = []
    for text in texts:
        seed = b"zebra-cluster" if "zebra" in text.lower() else text.encode()
        raw = np.frombuffer(hashlib.sha256(seed).digest()[:16], dtype=np.uint8)
        row = raw.astype(np.float32) + 1.0
        rows.append(row / np.linalg.norm(row))
    return np.vstack(rows)


def _mention(surface: str, etype: str, start: int, end: int) -> ExtractedMention:
    return ExtractedMention(
        surface=surface,
        etype=etype,
        confidence=0.95,
        char_start=start,
        char_end=end,
        extractor="fake",
        schema_id="s1",
    )


def _relation(
    head: str,
    predicate: str,
    tail: str,
    *,
    subject: tuple[int, int],
    obj: tuple[int, int],
    assertion: tuple[int, int],
) -> ExtractedRelation:
    return ExtractedRelation(
        head=head,
        predicate=predicate,
        tail=tail,
        confidence=0.9,
        subject_start=subject[0],
        subject_end=subject[1],
        object_start=obj[0],
        object_end=obj[1],
        assertion_start=assertion[0],
        assertion_end=assertion[1],
        extraction_method="fake",
        schema_id="s1",
        status="active",
    )


def build_audit_db(tmp_path: Path) -> Path:
    """Small real build with contains edges, one semantic edge pair, and two
    fact predicates (works_at, owns_or_has)."""

    n1 = "Maya works at Klaviyo."
    n2 = "Maya owns the Launch project."
    notes = [
        make_note("n1", "Job", n1, modified=100.0),
        make_note("n2", "Project", n2, modified=200.0),
        make_note("n3", "Zebra A", "The zebra grazed across the delta plains.", modified=300.0),
        make_note("n4", "Zebra B", "A zebra herd moved along the river at dusk.", modified=400.0),
    ]

    def extractor(_notes, *, tokenizer):
        del tokenizer
        return {
            "n1": Extraction(
                mentions=[
                    _mention("Maya", "person", 0, 4),
                    _mention("Klaviyo", "organization", 14, 21),
                ],
                fact_candidates=[
                    _relation(
                        "Maya",
                        "works_at",
                        "Klaviyo",
                        subject=(0, 4),
                        obj=(14, 21),
                        assertion=(0, len(n1)),
                    )
                ],
                schema_id="s1",
            ),
            "n2": Extraction(
                mentions=[
                    _mention("Maya", "person", 0, 4),
                    _mention("Launch", "project", 14, 20),
                ],
                fact_candidates=[
                    _relation(
                        "Maya",
                        "owns_or_has",
                        "Launch",
                        subject=(0, 4),
                        obj=(14, 20),
                        assertion=(0, len(n2)),
                    )
                ],
                schema_id="s1",
            ),
        }

    database = tmp_path / "taurine-knowledge.db"
    BuildCoordinator(
        BuildDependencies(
            tokenizer=Tokenizer(), embedder=embedder, embedding_dim=16, extractor=extractor
        )
    ).build(notes, database, replace_derived=True)
    return database


def _connect(database: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class RelevantScorer:
    def score_pairs(self, query, documents, *, instruction, deadline_at=None):
        del query, instruction, deadline_at
        return tuple(0.99 for _ in documents)


def _make_engine(database: Path):
    """Real snapshot + shared engine with fake embedder/scorer (no models)."""

    snapshot = load_index_snapshot(database)
    artifact = load_calibration(CALIBRATION_PATH)
    calibrator = HeadCalibrator(
        artifact,
        mode="chat",
        artifact_hash=artifact.hash,
        config_hash=artifact.config_hash,
        model_manifest=artifact.model_manifest,
    )
    store = snapshot.open_knowledge_store()
    engine = SharedRetrievalEngine(
        snapshot,
        store,
        calibrator,
        AbstentionPolicy(artifact, mode="chat"),
        query_embedder=lambda text: embedder([text])[0],
        registry=EvidenceRegistry(),
        reranker=LocalPassageReranker(
            RelevantScorer(),  # type: ignore[arg-type]
            mode="chat",
            acceptance_floor=artifact.rerank.chat_floor,
        ),
    )
    return engine, store


class FakeJudge:
    """complete_json fake: contains edges are supported, everything else not."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[tuple[str, str, str, float | None]] = []
        self._fail_first = fail_first

    async def complete_json(self, model, system, user, timeout_s=None, temperature=None):
        del timeout_s
        self.calls.append((model, system, user, temperature))
        if self._fail_first and len(self.calls) == 1:
            raise LLMError("synthetic batch failure")
        payload = json.loads(user)
        judgments = []
        for edge in payload["edges"]:
            supported = edge["kind"] == "contains"
            judgments.append(
                {
                    "id": edge["id"],
                    "score": 0.9 if supported else 0.1,
                    "verdict": "supported" if supported else "unsupported",
                    "note": "grounded" if supported else "not in snippet",
                }
            )
        return {"judgments": judgments}


# ── sampling ───────────────────────────────────────────────────────────


def test_sampling_is_deterministic_and_covers_stratum_minimums(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        first = sample_edges(conn, build_id, sample=6, seed=7)
        second = sample_edges(conn, build_id, sample=6, seed=7)

        assert [edge.edge_id for edge in first] == [edge.edge_id for edge in second]

        by_stratum: dict[str, int] = {}
        for edge in first:
            by_stratum[edge.stratum] = by_stratum.get(edge.stratum, 0) + 1
        # Every non-empty stratum is present with min(5, stratum size) rows
        # even when the requested sample is smaller than the minimums.
        assert by_stratum["fact:works_at"] == 1
        assert by_stratum["fact:owns_or_has"] == 1
        sizes = {
            "contains": conn.execute(
                "SELECT count(*) FROM passage_entity_edges WHERE build_id = ?", (build_id,)
            ).fetchone()[0],
            "semantic": conn.execute(
                "SELECT count(*) FROM passage_semantic_edges WHERE build_id = ?", (build_id,)
            ).fetchone()[0],
        }
        assert by_stratum["contains"] == min(5, sizes["contains"])
        assert by_stratum["semantic"] == min(5, sizes["semantic"])


def test_sampled_edges_render_endpoints_and_grounding_snippets(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        edges = sample_edges(conn, build_id, sample=100, seed=7)

    contains = [edge for edge in edges if edge.kind == "contains"]
    semantic = [edge for edge in edges if edge.kind == "semantic"]
    facts = [edge for edge in edges if edge.kind == "fact"]
    assert contains and semantic and facts

    maya = [edge for edge in contains if "Maya" in edge.endpoints]
    assert maya
    # contains snippet: passage text with the mention highlighted
    assert any("[[Maya]]" in snippet for edge in maya for snippet in edge.snippets)
    assert all("person" in edge.endpoints for edge in maya)

    # semantic edges carry both passage snippets
    assert all(len(edge.snippets) == 2 for edge in semantic)
    assert any("zebra" in snippet for snippet in semantic[0].snippets)

    works = [edge for edge in facts if edge.stratum == "fact:works_at"]
    assert works
    assert "works_at" in works[0].endpoints
    assert any("Klaviyo" in snippet for snippet in works[0].snippets)
    assert all(len(snippet) <= 320 for edge in edges for snippet in edge.snippets)


# ── corpus stats + defensive schema handling ───────────────────────────


def test_corpus_stats_report_adjudication_coverage_when_columns_exist(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    with sqlite3.connect(database) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
        if "adjudication_score" not in columns:
            conn.execute("ALTER TABLE facts ADD COLUMN adjudication_score REAL")
            conn.execute(
                "ALTER TABLE facts ADD COLUMN adjudication_quote TEXT NOT NULL DEFAULT ''"
            )
            conn.execute(
                "ALTER TABLE facts ADD COLUMN adjudicated INTEGER NOT NULL DEFAULT 0"
            )
        edge_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(passage_semantic_edges)")
        }
        if "reason" not in edge_columns:
            conn.execute(
                "ALTER TABLE passage_semantic_edges ADD COLUMN reason TEXT NOT NULL DEFAULT ''"
            )
        conn.execute(
            "UPDATE facts SET adjudicated = 1, adjudication_score = 0.85,"
            " adjudication_quote = 'works at Klaviyo' WHERE predicate = 'works_at'"
        )

    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        stats = corpus_stats(conn, build_id)
        edges = sample_edges(conn, build_id, sample=100, seed=7)

    assert stats["adjudication"]["available"] is True
    assert stats["adjudication"]["adjudicated"] == 1
    assert stats["semantic_reasons"]["available"] is True
    works = [edge for edge in edges if edge.stratum == "fact:works_at"][0]
    fields = dict(works.fields)
    assert fields["adjudication_score"] == "0.85"
    assert "works at Klaviyo" in fields["adjudication_quote"]

    report = render_report(
        database=database, seed=7, stats=stats, edges=edges,
        scores=None, judgments=None, unscored=0, judge_model=None, ablation=None,
        flags=build_flags(edges, None, stats),
    )
    assert "adjudicated" in report


def test_harness_works_on_db_without_adjudication_columns(tmp_path: Path) -> None:
    """Pre-W2D databases (no adjudication columns, no semantic reason) must
    still sample, aggregate stats, and render."""

    database = build_audit_db(tmp_path)
    with sqlite3.connect(database) as conn:
        for table, column in (
            ("facts", "adjudication_score"),
            ("facts", "adjudication_quote"),
            ("facts", "adjudicated"),
            ("passage_semantic_edges", "reason"),
        ):
            present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column in present:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")

    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        stats = corpus_stats(conn, build_id)
        edges = sample_edges(conn, build_id, sample=20, seed=7)

    assert stats["adjudication"]["available"] is False
    assert stats["semantic_reasons"]["available"] is False
    assert edges
    report = render_report(
        database=database, seed=7, stats=stats, edges=edges,
        scores=None, judgments=None, unscored=0, judge_model=None, ablation=None,
        flags=build_flags(edges, None, stats),
    )
    assert "n/a" in report


# ── LLM scoring ────────────────────────────────────────────────────────


def test_llm_scoring_batches_and_aggregates_per_stratum(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        edges = sample_edges(conn, build_id, sample=100, seed=7)

    judge = FakeJudge()
    judgments, unscored = score_edges(
        edges, judge, model="fake/model", batch_size=3, concurrency=2
    )

    assert unscored == 0
    assert len(judgments) == len(edges)
    assert all(temperature == 0.0 for _, _, _, temperature in judge.calls)
    assert all(model == "fake/model" for model, _, _, _ in judge.calls)
    # batches of <= batch_size edges
    assert all(len(json.loads(user)["edges"]) <= 3 for _, _, user, _ in judge.calls)
    # judged only against supplied snippets: the prompt carries them
    assert any("snippets" in user for _, _, user, _ in judge.calls)

    scores = aggregate_scores(edges, judgments)
    by_stratum = {row.stratum: row for row in scores}
    assert by_stratum["contains"].mean_score == pytest.approx(0.9)
    assert by_stratum["contains"].pct_supported == pytest.approx(1.0)
    assert by_stratum["semantic"].mean_score == pytest.approx(0.1)
    assert by_stratum["semantic"].pct_unsupported == pytest.approx(1.0)


def test_llm_scoring_fails_open_per_batch(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        edges = sample_edges(conn, build_id, sample=100, seed=7)

    judge = FakeJudge(fail_first=True)
    judgments, unscored = score_edges(
        edges, judge, model="fake/model", batch_size=3, concurrency=1
    )
    assert unscored == min(3, len(edges))
    assert len(judgments) == len(edges) - unscored
    # aggregation skips unscored edges instead of crashing
    scores = aggregate_scores(edges, judgments)
    assert sum(row.scored for row in scores) == len(judgments)


def test_flags_surface_high_weight_low_score_edges(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        stats = corpus_stats(conn, build_id)
        edges = sample_edges(conn, build_id, sample=100, seed=7)

    judgments, _ = score_edges(edges, FakeJudge(), model="fake/model")
    flags = build_flags(edges, judgments, stats)
    # semantic edges have weight ~1.0 (cosine 1.0) and fake score 0.1
    assert any("weight" in flag and "semantic" in flag for flag in flags)
    assert any("mean score" in flag for flag in flags)


# ── ablation ───────────────────────────────────────────────────────────


def test_ablation_reports_overlap_and_displacement_per_knockout(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    engine, store = _make_engine(database)
    queries = load_queries(DEFAULT_QUERIES_PATH)[:2]
    try:
        ablation = run_ablation(database, queries, engine=engine)
    finally:
        store.close()

    assert len(ablation) == 2
    for row in ablation:
        names = [knockout.name for knockout in row.knockouts]
        assert names == ["full", "no_semantic", "no_contains", "no_fact"]
        full = row.knockouts[0]
        assert full.overlap == pytest.approx(1.0)
        for knockout in row.knockouts:
            assert 0.0 <= knockout.overlap <= 1.0
            assert len(knockout.top3_displacement) <= 3
        assert row.full_top  # the tiny corpus always ranks some notes


def test_checked_in_audit_queries_cover_required_shapes() -> None:
    queries = load_queries(DEFAULT_QUERIES_PATH)
    assert len(queries) >= 12
    shapes = {query.shape for query in queries}
    assert {"entity", "topical", "temporal", "conversational", "multihop"} <= shapes


# ── report + CLI ───────────────────────────────────────────────────────


def test_cli_writes_markdown_report(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    out = tmp_path / "report.md"

    code = main(["--db", str(database), "--sample", "10", "--seed", "7", "--out", str(out)])

    assert code == 0
    report = out.read_text(encoding="utf-8")
    assert report.startswith("# Edge audit")
    assert "## Corpus stats" in report
    assert "## Sampled edges" in report
    assert "### contains" in report
    assert "### semantic" in report
    assert "### fact:works_at" in report
    # no --llm / --ablate: those sections are absent
    assert "## LLM precision" not in report
    assert "## Ablation" not in report


def test_report_includes_diagnostic_disclaimer_for_ablation(tmp_path: Path) -> None:
    database = build_audit_db(tmp_path)
    engine, store = _make_engine(database)
    queries = load_queries(DEFAULT_QUERIES_PATH)[:1]
    try:
        ablation = run_ablation(database, queries, engine=engine)
    finally:
        store.close()

    with _connect(database) as conn:
        build_id = latest_build_id(conn)
        stats = corpus_stats(conn, build_id)
        edges = sample_edges(conn, build_id, sample=10, seed=7)

    report = render_report(
        database=database, seed=7, stats=stats, edges=edges,
        scores=None, judgments=None, unscored=0, judge_model=None,
        ablation=ablation, flags=build_flags(edges, None, stats),
    )
    assert "## Ablation" in report
    assert "diagnostic" in report.lower()
    assert "no_semantic" in report
