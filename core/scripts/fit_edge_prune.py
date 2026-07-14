"""Fit the runtime-free semantic-edge pruner from cached LLM judgments."""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taurine_core import config
from taurine_core.evaluation.quality_bench import write_report
from taurine_core.indexer.edge_adjudication import SemanticPair, _reason_key
from taurine_core.indexer.edge_prune import FEATURE_NAMES, EdgeFeatures, make_features


@dataclass(frozen=True)
class FitExample:
    key: str
    features: EdgeFeatures
    llm_keep: bool


@dataclass(frozen=True)
class _Passage:
    passage_id: str
    title: str
    source: str
    text: str
    content_hash: str
    vector: np.ndarray


def _read_only(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        Path(path).resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, *, threshold: float
) -> dict[str, int | float]:
    labels = np.asarray(labels, dtype=np.int64)
    predicted_keep = np.asarray(probabilities, dtype=np.float64) >= float(threshold)
    actual_keep = labels == 1
    false_drops = int(np.sum(~predicted_keep & actual_keep))
    keep_count = int(np.sum(actual_keep))
    predicted_drops = int(np.sum(~predicted_keep))
    agreement = float(np.mean(predicted_keep == actual_keep)) if len(labels) else 0.0
    return {
        "drop_agreement": round(agreement, 6),
        "false_drop_rate": round(false_drops / keep_count, 6)
        if keep_count
        else 0.0,
        "predicted_drops": predicted_drops,
        "llm_drops": int(np.sum(~actual_keep)),
        "llm_keeps": keep_count,
        "examples": len(labels),
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Sequence[float] = tuple(value / 100 for value in range(1, 100)),
    false_drop_cap: float = 0.05,
) -> tuple[float, dict[str, int | float]]:
    candidates = [
        (float(threshold), classification_metrics(labels, probabilities, threshold=threshold))
        for threshold in thresholds
    ]
    safe = [
        row for row in candidates if float(row[1]["false_drop_rate"]) <= false_drop_cap
    ]
    pool = safe or candidates
    if not pool:
        raise ValueError("at least one decision threshold is required")
    return max(
        pool,
        key=lambda row: (
            float(row[1]["drop_agreement"]),
            int(row[1]["predicted_drops"]),
            row[0],
        ),
    )


def _load_examples(
    cache_path: str | Path, knowledge_path: str | Path
) -> tuple[list[FitExample], dict[str, int]]:
    cache = _read_only(cache_path)
    knowledge = _read_only(knowledge_path)
    try:
        labels: dict[str, bool] = {}
        for row in cache.execute(
            "SELECT key, payload FROM adjudication_cache "
            "WHERE kind = 'semantic_reason' ORDER BY key"
        ):
            try:
                payload = json.loads(str(row["payload"]))
            except (TypeError, ValueError):
                continue
            reason = payload.get("reason") if isinstance(payload, Mapping) else None
            if isinstance(reason, str):
                labels[str(row["key"])] = bool(
                    reason.strip() and reason.strip().upper() != "NONE"
                )

        passages: list[_Passage] = []
        for row in knowledge.execute(
            """
            SELECT p.passage_id, p.text, p.content_hash, n.title, n.source,
                   embedding.dim, embedding.vec
            FROM passages AS p
            JOIN notes AS n ON n.note_id = p.note_id
            JOIN passage_embeddings AS embedding
              ON embedding.passage_id = p.passage_id
            ORDER BY p.passage_id
            """
        ):
            vector = np.frombuffer(row["vec"], dtype=np.float32).copy()
            if vector.shape != (int(row["dim"]),):
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            passages.append(
                _Passage(
                    passage_id=str(row["passage_id"]),
                    title=str(row["title"]),
                    source=str(row["source"]),
                    text=str(row["text"]),
                    content_hash=str(row["content_hash"]),
                    vector=vector / norm,
                )
            )

        entities: dict[str, dict[str, tuple[str, float]]] = {}
        for row in knowledge.execute(
            """
            SELECT edge.passage_id, edge.entity_id, edge.idf, entity.canonical
            FROM passage_entity_edges AS edge
            JOIN entities AS entity ON entity.entity_id = edge.entity_id
            ORDER BY edge.passage_id, edge.entity_id
            """
        ):
            entities.setdefault(str(row["passage_id"]), {})[
                str(row["entity_id"])
            ] = (str(row["canonical"]), float(row["idf"]))

        by_hash: dict[str, list[_Passage]] = {}
        for passage in passages:
            by_hash.setdefault(passage.content_hash, []).append(passage)
        examples: list[FitExample] = []
        matched: set[str] = set()
        hashes = sorted(by_hash)
        for left_hash, right_hash in itertools.combinations_with_replacement(hashes, 2):
            if left_hash == right_hash:
                if len(by_hash[left_hash]) < 2:
                    continue
                left, right = by_hash[left_hash][:2]
            else:
                left, right = by_hash[left_hash][0], by_hash[right_hash][0]
            if left.passage_id > right.passage_id:
                left, right = right, left
            pair = SemanticPair(
                src_passage_id=left.passage_id,
                dst_passage_id=right.passage_id,
                src_title=left.title,
                dst_title=right.title,
                src_text=left.text,
                dst_text=right.text,
                src_content_hash=left.content_hash,
                dst_content_hash=right.content_hash,
            )
            key = _reason_key(config.ADJUDICATION_MODEL, pair)
            if key not in labels or key in matched:
                continue
            matched.add(key)
            left_entities = entities.get(left.passage_id, {})
            right_entities = entities.get(right.passage_id, {})
            shared = [
                (
                    left_entities[entity_id][0],
                    min(left_entities[entity_id][1], right_entities[entity_id][1]),
                )
                for entity_id in sorted(left_entities.keys() & right_entities.keys())
            ]
            features = make_features(
                pair,
                cosine=float(np.dot(left.vector, right.vector)),
                shared_entities=shared,
                src_source=left.source,
                dst_source=right.source,
            )
            examples.append(FitExample(key=key, features=features, llm_keep=labels[key]))
            if len(matched) == len(labels):
                break
        return examples, {
            "cached_semantic_rows": len(labels),
            "knowledge_passages": len(passages),
            "matched_cache_keys": len(matched),
        }
    finally:
        cache.close()
        knowledge.close()


def _matrix(examples: Sequence[FitExample]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[row.features.values()[name] for name in FEATURE_NAMES] for row in examples],
        dtype=np.float64,
    )
    labels = np.asarray([int(row.llm_keep) for row in examples], dtype=np.int64)
    return matrix, labels


def fit(examples: Sequence[FitExample]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train = [row for row in examples if int(row.key, 16) % 4 != 0]
    heldout = [row for row in examples if int(row.key, 16) % 4 == 0]
    train_x, train_y = _matrix(train)
    heldout_x, heldout_y = _matrix(heldout)
    if len(set(train_y.tolist())) < 2:
        raise ValueError("edge-prune training labels must contain keep and drop rows")
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=7, solver="lbfgs"
    ).fit(scaler.transform(train_x), train_y)
    train_probabilities = classifier.predict_proba(scaler.transform(train_x))[:, 1]
    heldout_probabilities = classifier.predict_proba(scaler.transform(heldout_x))[:, 1]
    threshold, train_metrics = select_threshold(
        train_y, train_probabilities, false_drop_cap=0.05
    )
    heldout_metrics = classification_metrics(
        heldout_y, heldout_probabilities, threshold=threshold
    )
    scaled_weights = classifier.coef_[0]
    raw_weights = scaled_weights / scaler.scale_
    raw_intercept = float(
        classifier.intercept_[0]
        - np.sum(scaled_weights * scaler.mean_ / scaler.scale_)
    )
    return {
        "decision_threshold": round(threshold, 12),
        "intercept": round(raw_intercept, 12),
        "weights": {
            name: round(float(raw_weights[index]), 12)
            for index, name in enumerate(FEATURE_NAMES)
        },
        "train": train_metrics,
        "heldout": heldout_metrics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--knowledge", required=True)
    parser.add_argument(
        "--out", default="evals/quality/2026-07-13-edge-prune.json"
    )
    parser.add_argument("--timestamp-source", default="approved-plan-2026-07-13")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    examples, coverage = _load_examples(args.cache, args.knowledge)
    fitted = fit(examples)
    report = write_report(
        args.out,
        variant="local_semantic_edge_prune",
        metrics={**fitted, "label_coverage": coverage},
        cost={"runtime_llm_calls": 0, "features": list(FEATURE_NAMES)},
        timestamp_source=args.timestamp_source,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
