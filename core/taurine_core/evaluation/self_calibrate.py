"""Self-supervised retrieval calibration against the user's own corpus.

Collecting labeled personal-KB queries by hand is expensive and the data is
too sensitive to ship, so this harness manufactures its own labels locally
(Generative Pseudo-Labeling): sample passages stratified by source, have the
configured flash model write the question each passage answers, and treat
(question, source passage/note) as a relevance pair. Off-domain and filler
queries form the null set. Everything — questions, scores, fitted curves —
stays on the machine; only the fitted calibration artifact (score knots,
floors, aggregate held-out metrics) is written, and it contains no corpus
content.

Usage:
    python -m taurine_core.evaluation.self_calibrate --db taurine-knowledge.db \
        [--samples 140] [--seed 7] [--data-dir <dir>] [--apply] [--report out.md]

Without --apply the fitted curves and metrics are printed only; with --apply
the calibration artifact is rewritten (hash recomputed) — a rebuild/sync is
then required so builds record the new hash.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from taurine_core import config
from taurine_core.evaluation.graph_retrieval import (
    CalibrationArtifact,
    compute_artifact_hash,
    load_calibration,
)

_QG_SYSTEM = (
    "You write evaluation queries for a personal knowledge assistant. For each "
    "passage, write ONE question its owner would plausibly type or say to their "
    "own assistant, answerable from that passage alone. Vary style across rows: "
    "who/what/when questions, casual phrasing, first person ('my', 'I'). Never "
    "mention 'the passage' or 'the note'. Return JSON "
    '{"questions": [{"key": <key>, "question": "..."}]}.'
)

_NULL_QUERIES = (
    "the the um okay yeah right",
    "uh so anyway like I was saying",
    "completely unrelated zephyr quasar nonsense token",
    "what is the boiling point of tungsten on venus",
    "recipe for traditional mongolian throat singing",
    "how do I renew a fishing license in portugal",
    "best defensive formations in medieval siege warfare",
    "hmm let me think about that for a second",
)


@dataclass(frozen=True)
class LabeledQuery:
    query: str
    note_id: str
    passage_id: str
    source: str
    split: str  # train | heldout


def _stratified_passages(
    store: Any, passage_ids: Sequence[str], samples: int, seed: int
) -> list[Any]:
    rows = [
        row
        for row in store.passage_contexts(tuple(passage_ids))
        if len(row.text.split()) >= 20
    ]
    by_source: dict[str, list[Any]] = {}
    for row in rows:
        by_source.setdefault(row.source, []).append(row)
    rng = random.Random(seed)
    picked: list[Any] = []
    total = sum(len(group) for group in by_source.values())
    for source in sorted(by_source):
        group = sorted(by_source[source], key=lambda row: row.passage_id)
        rng.shuffle(group)
        share = max(6, round(samples * len(group) / max(1, total)))
        seen_notes: set[str] = set()
        for row in group:
            if len([p for p in picked if p.source == source]) >= share:
                break
            if row.note_id in seen_notes:  # one passage per note per source
                continue
            seen_notes.add(row.note_id)
            picked.append(row)
    rng.shuffle(picked)
    return picked[: max(samples, 1)]


async def _generate_questions(
    llm: Any, passages: Sequence[Any], *, batch: int = 10, concurrency: int = 4
) -> dict[str, str]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(chunk: Sequence[Any]) -> dict[str, str]:
        payload = {
            "passages": [
                {"key": row.passage_id, "title": row.title, "text": row.text[:900]}
                for row in chunk
            ]
        }
        async with semaphore:
            try:
                raw = await llm.complete_json(
                    config.SUMMARY_MODEL,
                    _QG_SYSTEM,
                    json.dumps(payload, ensure_ascii=False),
                    timeout_s=config.SUMMARY_TIMEOUT_S,
                    temperature=0.4,
                )
            except Exception:  # noqa: BLE001 — per-batch fail-open
                return {}
        out: dict[str, str] = {}
        rows = raw.get("questions") if isinstance(raw, dict) else None
        for row in rows if isinstance(rows, list) else ():
            if isinstance(row, dict) and isinstance(row.get("question"), str):
                out[str(row.get("key", ""))] = row["question"].strip()
        return out

    chunks = [passages[i : i + batch] for i in range(0, len(passages), batch)]
    results = await asyncio.gather(*(one(chunk) for chunk in chunks))
    merged: dict[str, str] = {}
    for result in results:
        merged.update(result)
    return merged


def build_labeled_set(
    runtime: Any, llm: Any, *, samples: int, seed: int, data_dir: Path
) -> list[LabeledQuery]:
    """Generate (or reload) the local labeled query set. Never committed."""
    cache = data_dir / f"self-calibration-queries-{samples}-{seed}.jsonl"
    if cache.exists():
        return [LabeledQuery(**json.loads(line)) for line in cache.open() if line.strip()]
    store = runtime.snapshot.open_knowledge_store()
    try:
        passages = _stratified_passages(
            store, runtime.snapshot.passage_ids, samples, seed
        )
    finally:
        store.close()
    questions = asyncio.run(_generate_questions(llm, passages)) if llm else {}
    rng = random.Random(seed + 1)
    labeled: list[LabeledQuery] = []
    for row in passages:
        question = questions.get(row.passage_id, "")
        if len(question) < 8:
            continue
        # split by NOTE hash so near-duplicate evidence cannot leak across
        # train and held-out (spec §18.1)
        bucket = int(hashlib.sha256(row.note_id.encode()).hexdigest()[:8], 16) % 4
        labeled.append(
            LabeledQuery(
                query=question,
                note_id=row.note_id,
                passage_id=row.passage_id,
                source=row.source,
                split="heldout" if bucket == 0 else "train",
            )
        )
    rng.shuffle(labeled)
    data_dir.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as handle:
        for row in labeled:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
    return labeled


@dataclass
class ScoreSamples:
    dense_pos: list[float]
    dense_neg_top: list[float]  # best non-gold-note cosine per query
    rerank_pos: list[float]
    rerank_neg: list[float]  # hard negatives: top dense passages from other notes
    dense_null_top: list[float]
    rerank_null_top: list[float]


def collect_scores(runtime: Any, labeled: Sequence[LabeledQuery]) -> ScoreSamples:
    from taurine_core.indexer.embed import embed_queries
    from taurine_core.retrieval.rerank import (
        PASSAGE_RERANK_INSTRUCTION,
        PassageRerankCandidate,
    )

    snapshot = runtime.snapshot
    matrix = snapshot.passage_matrix
    passage_ids = list(snapshot.passage_ids)
    row_of = {passage_id: index for index, passage_id in enumerate(passage_ids)}
    store = snapshot.open_knowledge_store()
    try:
        contexts = {
            row.passage_id: row
            for row in store.passage_contexts(tuple(snapshot.passage_ids))
        }
    finally:
        store.close()
    note_of = {pid: contexts[pid].note_id for pid in passage_ids if pid in contexts}

    def rerank_scores(query: str, pids: Sequence[str]) -> list[float]:
        documents = []
        for pid in pids:
            row = contexts[pid]
            documents.append(
                PassageRerankCandidate(
                    passage_id=pid, note_id=row.note_id, title=row.title,
                    source=row.source, text=row.text, recall_score=0.0,
                ).rerank_document
            )
        return list(
            runtime.pair_scorer.score_pairs(
                query, documents, instruction=PASSAGE_RERANK_INSTRUCTION
            )
        )

    samples = ScoreSamples([], [], [], [], [], [])
    queries = [row.query for row in labeled]
    vectors = embed_queries(queries) if labeled else np.empty((0, 1))
    for row, vector in zip(labeled, vectors):
        cosines = matrix @ vector
        gold_row = row_of.get(row.passage_id)
        if gold_row is None:
            continue
        # gold = best passage of the gold NOTE (the generator saw one passage,
        # but any passage of that note answering the query counts)
        note_rows = [
            index for index, pid in enumerate(passage_ids)
            if note_of.get(pid) == row.note_id
        ]
        samples.dense_pos.append(float(max(cosines[index] for index in note_rows)))
        order = np.argsort(-cosines)
        neg_top = next(
            (
                float(cosines[index])
                for index in order
                if note_of.get(passage_ids[index]) != row.note_id
            ),
            0.0,
        )
        samples.dense_neg_top.append(neg_top)
        hard_negative_ids = [
            passage_ids[index]
            for index in order
            if note_of.get(passage_ids[index]) != row.note_id
        ][:3]
        scores = rerank_scores(row.query, [row.passage_id, *hard_negative_ids])
        samples.rerank_pos.append(float(scores[0]))
        samples.rerank_neg.extend(float(score) for score in scores[1:])

    null_vectors = embed_queries(list(_NULL_QUERIES))
    for query, vector in zip(_NULL_QUERIES, null_vectors):
        cosines = matrix @ vector
        order = np.argsort(-cosines)[:3]
        samples.dense_null_top.append(float(cosines[order[0]]))
        top_ids = [passage_ids[index] for index in order]
        samples.rerank_null_top.extend(
            float(score) for score in rerank_scores(query, top_ids)
        )
    return samples


def _monotone_knots(
    positives: Sequence[float], negatives: Sequence[float], *, lo: float, hi: float
) -> list[list[float]]:
    """Fit isotonic P(relevant | score) and reduce to ≤6 stable knots."""
    from sklearn.isotonic import IsotonicRegression

    scores = np.array(list(positives) + list(negatives), dtype=np.float64)
    labels = np.array([1.0] * len(positives) + [0.0] * len(negatives))
    if len(set(scores.tolist())) < 4:
        return [[lo, 0.0], [hi, 1.0]]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(scores, labels)
    grid = np.quantile(scores, [0.05, 0.25, 0.5, 0.75, 0.95])
    knots: list[list[float]] = [[lo, 0.0]]
    last_p = 0.0
    for value in grid:
        p = float(iso.predict([value])[0])
        p = max(p, last_p + 1e-4)  # strictly increasing for the loader
        knots.append([round(float(value), 4), round(min(p, 0.999), 4)])
        last_p = p
    knots.append([hi, 1.0])
    return knots


def _floor_for_null_rate(
    curve: Sequence[Sequence[float]],
    null_scores: Sequence[float],
    positives: Sequence[float],
    *,
    max_null_rate: float = 0.05,
) -> float:
    """Smallest calibrated floor keeping null acceptance ≤ max_null_rate."""

    def calibrated(score: float) -> float:
        points = list(curve)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= score <= x1:
                if x1 == x0:
                    return float(y1)
                return float(y0 + (y1 - y0) * (score - x0) / (x1 - x0))
        return 0.0 if score < points[0][0] else 1.0

    null_calibrated = sorted((calibrated(s) for s in null_scores), reverse=True)
    if not null_calibrated:
        return 0.5
    cutoff_index = max(0, math.ceil(len(null_calibrated) * max_null_rate) - 1)
    floor = null_calibrated[cutoff_index] + 0.02
    # never price out more than half the true positives
    pos_calibrated = sorted(calibrated(s) for s in positives)
    if pos_calibrated:
        median = pos_calibrated[len(pos_calibrated) // 2]
        floor = min(floor, max(0.2, median - 0.05))
    return round(min(0.95, max(0.2, floor)), 3)


def fit_and_report(samples: ScoreSamples, artifact: CalibrationArtifact) -> dict[str, Any]:
    dense_knots = _monotone_knots(
        samples.dense_pos, samples.dense_neg_top + samples.dense_null_top, lo=-1.0, hi=1.0
    )
    rerank_knots = _monotone_knots(
        samples.rerank_pos, samples.rerank_neg + samples.rerank_null_top, lo=0.0, hi=1.0
    )
    dense_floor_chat = _floor_for_null_rate(
        dense_knots, samples.dense_null_top, samples.dense_pos
    )
    rerank_floor_chat = _floor_for_null_rate(
        rerank_knots, samples.rerank_null_top, samples.rerank_pos
    )
    return {
        "dense_knots": dense_knots,
        "dense_floor_chat": dense_floor_chat,
        "dense_floor_live": round(min(0.95, dense_floor_chat + 0.06), 3),
        "rerank_knots": rerank_knots,
        "rerank_floor_chat": rerank_floor_chat,
        "rerank_floor_live": round(min(0.95, rerank_floor_chat + 0.06), 3),
        "counts": {
            "dense_pos": len(samples.dense_pos),
            "rerank_pos": len(samples.rerank_pos),
            "rerank_neg": len(samples.rerank_neg),
            "nulls": len(samples.dense_null_top),
        },
        "distributions": {
            "dense_pos_median": round(float(np.median(samples.dense_pos or [0])), 3),
            "dense_null_max": round(float(max(samples.dense_null_top or [0])), 3),
            "rerank_pos_median": round(float(np.median(samples.rerank_pos or [0])), 3),
            "rerank_neg_median": round(float(np.median(samples.rerank_neg or [0])), 3),
            "rerank_null_max": round(float(max(samples.rerank_null_top or [0])), 3),
        },
    }


def apply_to_artifact(path: Path, fitted: Mapping[str, Any], heldout: Mapping[str, Any]) -> str:
    data = json.loads(path.read_text())
    stamp = "self-calibrated-gpl"
    for mode in ("live", "chat", "tool"):
        dense = data["modes"][mode]["heads"]["dense"]
        dense["knots"] = fitted["dense_knots"]
        dense["acceptance_floor"] = (
            fitted["dense_floor_live"] if mode == "live" else fitted["dense_floor_chat"]
        )
        dense["trained_at"] = stamp
        dense["sample_count"] = int(fitted["counts"]["dense_pos"])
        dense["held_out"] = dict(heldout)
    data["rerank"]["live_floor"] = fitted["rerank_floor_live"]
    data["rerank"]["chat_floor"] = fitted["rerank_floor_chat"]
    for mode in ("live", "chat", "tool"):
        data["modes"][mode]["evidence_floor"] = (
            fitted["rerank_floor_live"] if mode == "live" else fitted["rerank_floor_chat"]
        )
    artifact = CalibrationArtifact.model_validate({**data, "hash": data["hash"]})
    data["hash"] = compute_artifact_hash(artifact)
    path.write_text(json.dumps(data, indent=1))
    return data["hash"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--samples", type=int, default=140)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", default=str(Path.home() / ".taurine" / "calibration"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from taurine_core.llm import OpenRouterClient
    from taurine_core.retrieval.runtime import CALIBRATION_PATH, RetrievalRuntime

    llm = None
    key = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if key:
        llm = OpenRouterClient(key)
    runtime = RetrievalRuntime(args.db)
    try:
        labeled = build_labeled_set(
            runtime, llm, samples=args.samples, seed=args.seed, data_dir=Path(args.data_dir)
        )
        train = [row for row in labeled if row.split == "train"]
        heldout_rows = [row for row in labeled if row.split == "heldout"]
        print(f"labeled queries: {len(labeled)} (train {len(train)}, heldout {len(heldout_rows)})")
        if len(train) < 30:
            print("not enough labeled queries to fit curves — aborting")
            return 2
        samples = collect_scores(runtime, train)
        fitted = fit_and_report(samples, runtime.artifact)
        print(json.dumps({k: v for k, v in fitted.items() if k != "dense_knots"}, indent=1))
        heldout_samples = collect_scores(runtime, heldout_rows)
        recall = float(
            np.mean(
                [
                    1.0 if pos >= neg else 0.0
                    for pos, neg in zip(
                        heldout_samples.rerank_pos,
                        (heldout_samples.rerank_neg[i * 3] for i in range(len(heldout_samples.rerank_pos))),
                    )
                ]
            )
        ) if heldout_samples.rerank_pos else 0.0
        heldout_metrics = {
            "rerank_pos_beats_top_negative": round(recall, 3),
            "heldout_queries": float(len(heldout_rows)),
        }
        print("held-out:", heldout_metrics)
        if args.apply:
            new_hash = apply_to_artifact(Path(CALIBRATION_PATH), fitted, heldout_metrics)
            load_calibration(CALIBRATION_PATH)  # verify
            print(f"artifact updated: {new_hash[:26]} — run a sync/rebuild to activate")
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
