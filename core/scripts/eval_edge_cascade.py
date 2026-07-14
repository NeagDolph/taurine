"""Evaluate Flash labels, reranker features, and a semantic-edge LLM cascade.

This is an offline experiment only. Both input databases are opened read-only;
the only writes are the explicitly requested JSON artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_edge_prune import (
    FitExample,
    _load_examples,
    _read_only,
    classification_metrics,
    select_threshold,
)
from taurine_core import config
from taurine_core.evaluation.quality_bench import write_report
from taurine_core.indexer.edge_adjudication import (
    SemanticPair,
    _REASON_SYSTEM_PROMPT,
    _parse_reason_response,
    _reason_key,
    _reason_payload,
)
from taurine_core.retrieval.rerank import (
    PASSAGE_RERANK_INSTRUCTION,
    PassageRerankCandidate,
)


FLASH_MODEL = "deepseek/deepseek-v4-flash"
RERANK_MAX_CHARS = 1000
SAMPLE_MAX_CHARS = 300
BASE_FEATURE_NAMES = (
    "cosine",
    "shared_entity_count",
    "min_shared_entity_idf",
    "same_source",
    "title_jaccard",
    "length_ratio",
)
RERANK_FEATURE_NAMES = ("rerank_src_to_dst", "rerank_dst_to_src")


@dataclass(frozen=True)
class PassageMetadata:
    passage_id: str
    note_id: str
    title: str
    source: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class ExperimentExample:
    key: str
    pair: SemanticPair
    src_note_id: str
    dst_note_id: str
    src_source: str
    dst_source: str
    base_features: Mapping[str, float]
    gpt_oss_keep: bool
    gpt_oss_reason: str
    rerank_src_to_dst: float = 0.0
    rerank_dst_to_src: float = 0.0

    def feature_values(self) -> dict[str, float]:
        return {
            **{name: float(self.base_features[name]) for name in BASE_FEATURE_NAMES},
            "rerank_src_to_dst": float(self.rerank_src_to_dst),
            "rerank_dst_to_src": float(self.rerank_dst_to_src),
        }


@dataclass(frozen=True)
class DirectionalRerank:
    src_to_dst: float
    dst_to_src: float
    scorer_available: bool
    failed_directions: int


@dataclass(frozen=True)
class FittedConfiguration:
    report: Mapping[str, Any]
    probabilities: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray


def _pair_cache_key(pair: SemanticPair) -> str:
    return _reason_key(config.ADJUDICATION_MODEL, pair)


def _flash_signature(pairs: Sequence[SemanticPair]) -> dict[str, Any]:
    return {
        "model": FLASH_MODEL,
        "prompt_sha256": hashlib.sha256(_REASON_SYSTEM_PROMPT.encode()).hexdigest(),
        "pair_keys": [_pair_cache_key(pair) for pair in sorted(pairs, key=lambda row: row.pair)],
    }


def judge_flash_pairs(
    pairs: Sequence[SemanticPair],
    *,
    judge: Callable[[str, dict], Any],
    batch_size: int = config.SEMANTIC_REASON_BATCH,
    concurrency: int = config.ADJUDICATION_CONCURRENCY,
) -> dict[str, Any]:
    """Judge exact production-shaped batches; any failed batch keeps every edge."""

    ordered = tuple(sorted(pairs, key=lambda row: row.pair))
    size = max(1, int(batch_size))
    batches = [ordered[start : start + size] for start in range(0, len(ordered), size)]

    def call(batch: Sequence[SemanticPair]) -> tuple[Any | None, str]:
        try:
            return judge(_REASON_SYSTEM_PROMPT, _reason_payload(batch)), ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    workers = max(1, min(int(concurrency), len(batches))) if batches else 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        responses = list(executor.map(call, batches))

    verdicts: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    failed_batches = 0
    for index, (batch, (response, error)) in enumerate(zip(batches, responses, strict=True)):
        parsed = _parse_reason_response(response, batch)
        if parsed is None:
            failed_batches += 1
        raw_batches.append(
            {
                "batch": index,
                "pair_keys": [_pair_cache_key(pair) for pair in batch],
                "raw_response": response,
                "error": error or ("malformed response" if parsed is None else ""),
            }
        )
        for pair in batch:
            reason = parsed.get(pair.pair, "") if parsed is not None else ""
            fail_open = parsed is None or pair.pair not in parsed
            verdicts.append(
                {
                    "key": _pair_cache_key(pair),
                    "pair": [pair.src_passage_id, pair.dst_passage_id],
                    "verdict": "keep"
                    if fail_open or (reason.strip() and reason.strip().upper() != "NONE")
                    else "drop",
                    "reason": reason,
                    "fail_open": fail_open,
                }
            )
    return {
        **_flash_signature(ordered),
        "batch_size": size,
        "pair_count": len(ordered),
        "failed_batches": failed_batches,
        "batches": raw_batches,
        "verdicts": verdicts,
    }


def _valid_flash_cache(data: Any, pairs: Sequence[SemanticPair]) -> bool:
    if not isinstance(data, Mapping):
        return False
    signature = _flash_signature(pairs)
    if any(data.get(name) != value for name, value in signature.items()):
        return False
    verdicts = data.get("verdicts")
    return isinstance(verdicts, list) and [row.get("key") for row in verdicts if isinstance(row, Mapping)] == signature["pair_keys"]


def load_or_judge_flash(
    pairs: Sequence[SemanticPair],
    *,
    cache_path: str | Path,
    judge: Callable[[str, dict], Any] | None,
    batch_size: int = config.SEMANTIC_REASON_BATCH,
    concurrency: int = config.ADJUDICATION_CONCURRENCY,
) -> dict[str, Any]:
    target = Path(cache_path)
    if target.exists():
        try:
            cached = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            cached = None
        if _valid_flash_cache(cached, pairs):
            return dict(cached)
    if judge is None:
        raise RuntimeError("OPEN_ROUTER_API_KEY is required when the Flash cache is absent or stale")
    artifact = judge_flash_pairs(
        pairs,
        judge=judge,
        batch_size=batch_size,
        concurrency=concurrency,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def reranker_features(
    pair: SemanticPair,
    scorer: Any | None,
    *,
    src_note_id: str = "",
    dst_note_id: str = "",
    src_source: str = "",
    dst_source: str = "",
    max_chars: int = RERANK_MAX_CHARS,
) -> DirectionalRerank:
    if scorer is None:
        return DirectionalRerank(0.0, 0.0, False, 2)
    bound = max(1, int(max_chars))
    passages = (
        (pair.src_passage_id, src_note_id, pair.src_title, src_source, pair.src_text[:bound]),
        (pair.dst_passage_id, dst_note_id, pair.dst_title, dst_source, pair.dst_text[:bound]),
    )
    scores: list[float] = []
    failures = 0
    for query_index, document_index in ((0, 1), (1, 0)):
        query = passages[query_index][4]
        passage_id, note_id, title, source, text = passages[document_index]
        document = PassageRerankCandidate(
            passage_id=passage_id,
            note_id=note_id,
            title=title,
            source=source,
            text=text,
            recall_score=0.0,
        ).rerank_document
        try:
            values = tuple(
                scorer.score_pairs(
                    query,
                    (document,),
                    instruction=PASSAGE_RERANK_INSTRUCTION,
                )
            )
            if len(values) != 1:
                raise ValueError("reranker returned the wrong number of scores")
            score = float(values[0])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("reranker returned an invalid score")
        except Exception:
            score = 0.0
            failures += 1
        scores.append(score)
    return DirectionalRerank(scores[0], scores[1], True, failures)


def cohens_kappa(left: Sequence[bool], right: Sequence[bool]) -> float:
    if len(left) != len(right):
        raise ValueError("label sequences must have equal length")
    if not left:
        return 0.0
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    observed = float(np.mean(a == b))
    expected = float(np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 6)


def cascade_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    lo: float,
    hi: float,
) -> dict[str, int | float | None]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have equal shape")
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("cascade thresholds must satisfy 0 <= lo < hi <= 1")
    actual_keep = labels == 1
    auto_drop = probabilities <= lo
    auto_keep = probabilities >= hi
    middle = ~(auto_drop | auto_keep)
    drop_count = int(np.sum(auto_drop))
    keep_count = int(np.sum(auto_keep))
    correct_drops = int(np.sum(auto_drop & ~actual_keep))
    correct_keeps = int(np.sum(auto_keep & actual_keep))
    false_drops = int(np.sum(auto_drop & actual_keep))
    actual_keeps = int(np.sum(actual_keep))
    auto_resolved = drop_count + keep_count
    errors = false_drops + int(np.sum(auto_keep & ~actual_keep))
    return {
        "lo": round(float(lo), 6),
        "hi": round(float(hi), 6),
        "examples": len(labels),
        "auto_resolved_count": auto_resolved,
        "auto_resolved_rate": round(auto_resolved / len(labels), 6) if len(labels) else 0.0,
        "llm_calls_eliminated_rate": round(auto_resolved / len(labels), 6) if len(labels) else 0.0,
        "auto_keep_count": keep_count,
        "auto_keep_precision": round(correct_keeps / keep_count, 6) if keep_count else None,
        "auto_drop_count": drop_count,
        "auto_drop_precision": round(correct_drops / drop_count, 6) if drop_count else None,
        "middle_band_size": int(np.sum(middle)),
        "overall_false_drop_rate": round(false_drops / actual_keeps, 6) if actual_keeps else 0.0,
        "cascade_agreement": round(1.0 - errors / len(labels), 6) if len(labels) else 0.0,
    }


def select_cascade_bands(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    auto_drop_precision_floor: float = 0.95,
    false_drop_cap: float = 0.05,
    thresholds: Sequence[float] = tuple(value / 100 for value in range(1, 100)),
) -> dict[str, int | float | None]:
    candidates: list[dict[str, int | float | None]] = []
    for lo, hi in itertools.combinations((float(value) for value in thresholds), 2):
        metrics = cascade_metrics(labels, probabilities, lo=lo, hi=hi)
        drop_precision = metrics["auto_drop_precision"]
        if (
            int(metrics["auto_drop_count"]) > 0
            and int(metrics["auto_keep_count"]) > 0
            and drop_precision is not None
            and float(drop_precision) >= auto_drop_precision_floor
            and float(metrics["overall_false_drop_rate"]) <= false_drop_cap
        ):
            candidates.append(metrics)
    if not candidates:
        raise ValueError("no cascade band satisfies the safety constraints")
    return max(
        candidates,
        key=lambda row: (
            int(row["auto_resolved_count"]),
            int(row["auto_drop_count"]),
            float(row["auto_keep_precision"] or 0.0),
            float(row["auto_drop_precision"] or 0.0),
            -float(row["hi"]) + float(row["lo"]),
        ),
    )


def _load_pair_metadata(knowledge_path: str | Path) -> list[PassageMetadata]:
    connection = _read_only(knowledge_path)
    try:
        return [
            PassageMetadata(
                passage_id=str(row["passage_id"]),
                note_id=str(row["note_id"]),
                title=str(row["title"]),
                source=str(row["source"]),
                text=str(row["text"]),
                content_hash=str(row["content_hash"]),
            )
            for row in connection.execute(
                """
                SELECT p.passage_id, p.note_id, p.text, p.content_hash,
                       n.title, n.source
                FROM passages AS p
                JOIN notes AS n ON n.note_id = p.note_id
                ORDER BY p.passage_id
                """
            )
        ]
    finally:
        connection.close()


def _load_gpt_reasons(cache_path: str | Path, keys: set[str]) -> dict[str, str]:
    connection = _read_only(cache_path)
    try:
        reasons: dict[str, str] = {}
        for row in connection.execute(
            "SELECT key, payload FROM adjudication_cache "
            "WHERE kind = 'semantic_reason' ORDER BY key"
        ):
            key = str(row["key"])
            if key not in keys:
                continue
            try:
                payload = json.loads(str(row["payload"]))
            except (TypeError, ValueError):
                continue
            reason = payload.get("reason") if isinstance(payload, Mapping) else None
            if isinstance(reason, str):
                reasons[key] = reason
        return reasons
    finally:
        connection.close()


def load_experiment_examples(
    cache_path: str | Path, knowledge_path: str | Path
) -> tuple[list[ExperimentExample], dict[str, Any]]:
    fitted_examples, coverage, _ = _load_examples(
        cache_path, knowledge_path, nli_scorer=None
    )
    fitted_by_key: dict[str, FitExample] = {row.key: row for row in fitted_examples}
    reasons = _load_gpt_reasons(cache_path, set(fitted_by_key))
    metadata = _load_pair_metadata(knowledge_path)
    by_hash: dict[str, list[PassageMetadata]] = {}
    for passage in metadata:
        by_hash.setdefault(passage.content_hash, []).append(passage)

    examples: list[ExperimentExample] = []
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
        key = _pair_cache_key(pair)
        fitted = fitted_by_key.get(key)
        if fitted is None or key in matched:
            continue
        matched.add(key)
        examples.append(
            ExperimentExample(
                key=key,
                pair=pair,
                src_note_id=left.note_id,
                dst_note_id=right.note_id,
                src_source=left.source,
                dst_source=right.source,
                base_features={
                    name: fitted.features.values()[name] for name in BASE_FEATURE_NAMES
                },
                gpt_oss_keep=fitted.llm_keep,
                gpt_oss_reason=reasons.get(key, ""),
            )
        )
        if len(matched) == len(fitted_by_key):
            break
    return examples, {
        **coverage,
        "experiment_examples": len(examples),
        "gpt_reasons": len(reasons),
    }


def _load_reranker_fail_open() -> tuple[Any | None, str]:
    from taurine_core import models_local

    try:
        return models_local.local_pair_reranker(config.RERANK_MODEL), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def add_reranker_features(
    examples: Sequence[ExperimentExample], scorer: Any | None
) -> tuple[list[ExperimentExample], dict[str, Any]]:
    from dataclasses import replace

    enriched: list[ExperimentExample] = []
    failures = 0
    for row in examples:
        scores = reranker_features(
            row.pair,
            scorer,
            src_note_id=row.src_note_id,
            dst_note_id=row.dst_note_id,
            src_source=row.src_source,
            dst_source=row.dst_source,
        )
        failures += scores.failed_directions
        enriched.append(
            replace(
                row,
                rerank_src_to_dst=scores.src_to_dst,
                rerank_dst_to_src=scores.dst_to_src,
            )
        )
    return enriched, {
        "available": scorer is not None,
        "model": config.RERANK_MODEL,
        "max_chars_per_passage": RERANK_MAX_CHARS,
        "failed_directions": failures,
        "scored_directions": len(examples) * 2 - failures,
    }


def fit_configuration(
    examples: Sequence[ExperimentExample],
    labels: Sequence[bool],
    feature_names: Sequence[str],
) -> FittedConfiguration:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(
        [[row.feature_values()[name] for name in feature_names] for row in examples],
        dtype=np.float64,
    )
    label_array = np.asarray(labels, dtype=np.int64)
    train_mask = np.asarray([int(row.key, 16) % 4 != 0 for row in examples], dtype=bool)
    if len(set(label_array[train_mask].tolist())) < 2:
        raise ValueError("cascade training labels must contain keep and drop rows")
    scaler = StandardScaler().fit(matrix[train_mask])
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=7, solver="lbfgs"
    ).fit(scaler.transform(matrix[train_mask]), label_array[train_mask])
    probabilities = classifier.predict_proba(scaler.transform(matrix))[:, 1]
    threshold, train_metrics = select_threshold(
        label_array[train_mask], probabilities[train_mask], false_drop_cap=0.05
    )
    heldout_metrics = classification_metrics(
        label_array[~train_mask], probabilities[~train_mask], threshold=threshold
    )
    scaled_weights = classifier.coef_[0]
    raw_weights = scaled_weights / scaler.scale_
    raw_intercept = float(
        classifier.intercept_[0]
        - np.sum(scaled_weights * scaler.mean_ / scaler.scale_)
    )
    return FittedConfiguration(
        report={
            "features": list(feature_names),
            "decision_threshold": round(float(threshold), 12),
            "intercept": round(raw_intercept, 12),
            "weights": {
                name: round(float(raw_weights[index]), 12)
                for index, name in enumerate(feature_names)
            },
            "train": train_metrics,
            "heldout": heldout_metrics,
        },
        probabilities=probabilities,
        labels=label_array,
        train_mask=train_mask,
    )


def _flash_labels(artifact: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, str]]:
    labels: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for row in artifact.get("verdicts", []):
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("key", ""))
        labels[key] = row.get("verdict") != "drop"
        reasons[key] = str(row.get("reason", ""))
    return labels, reasons


def _agreement_metrics(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=bool)
    right_array = np.asarray(right, dtype=bool)
    return {
        "examples": len(left_array),
        "agreement": round(float(np.mean(left_array == right_array)), 6),
        "cohens_kappa": cohens_kappa(left, right),
        "both_keep": int(np.sum(left_array & right_array)),
        "both_drop": int(np.sum(~left_array & ~right_array)),
        "gpt_oss_keep_flash_drop": int(np.sum(left_array & ~right_array)),
        "gpt_oss_drop_flash_keep": int(np.sum(~left_array & right_array)),
    }


def disagreement_sample(
    examples: Sequence[ExperimentExample],
    flash_labels: Mapping[str, bool],
    flash_reasons: Mapping[str, str],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    groups: dict[tuple[bool, bool], list[ExperimentExample]] = {}
    for row in sorted(examples, key=lambda value: value.key):
        flash_keep = flash_labels[row.key]
        if row.gpt_oss_keep != flash_keep:
            groups.setdefault((row.gpt_oss_keep, flash_keep), []).append(row)
    selected: list[ExperimentExample] = []
    directions = ((True, False), (False, True))
    per_direction = limit // 2
    for direction in directions:
        selected.extend(groups.get(direction, [])[:per_direction])
    if len(selected) < limit:
        used = {row.key for row in selected}
        remainder = [
            row
            for direction in directions
            for row in groups.get(direction, [])
            if row.key not in used
        ]
        selected.extend(remainder[: limit - len(selected)])
    return [
        {
            "key": row.key,
            "pair": [row.pair.src_passage_id, row.pair.dst_passage_id],
            "passage_a": {
                "title": row.pair.src_title,
                "text": row.pair.src_text[:SAMPLE_MAX_CHARS],
            },
            "passage_b": {
                "title": row.pair.dst_title,
                "text": row.pair.dst_text[:SAMPLE_MAX_CHARS],
            },
            "gpt_oss": {
                "verdict": "keep" if row.gpt_oss_keep else "drop",
                "reason": row.gpt_oss_reason,
            },
            "flash": {
                "verdict": "keep" if flash_labels[row.key] else "drop",
                "reason": flash_reasons.get(row.key, ""),
            },
        }
        for row in selected[:limit]
    ]


def _make_flash_judge() -> Callable[[str, dict], Any] | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass
    key = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not key:
        return None
    from taurine_core.llm import OpenRouterClient

    client = OpenRouterClient(key)

    def judge(system_prompt: str, payload: dict) -> Any:
        return asyncio.run(
            client.complete_json(
                FLASH_MODEL,
                system_prompt,
                json.dumps(payload, ensure_ascii=False),
                timeout_s=config.SUMMARY_TIMEOUT_S,
                temperature=0.0,
                max_tokens=config.ADJUDICATION_MAX_TOKENS,
            )
        )

    return judge


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    examples, coverage = load_experiment_examples(args.cache, args.knowledge)
    if coverage["experiment_examples"] != coverage["cached_semantic_rows"]:
        raise RuntimeError(f"incomplete semantic label reconstruction: {coverage}")

    flash_artifact = load_or_judge_flash(
        [row.pair for row in examples],
        cache_path=args.flash_cache,
        judge=_make_flash_judge(),
    )
    flash_by_key, flash_reasons = _flash_labels(flash_artifact)
    if set(flash_by_key) != {row.key for row in examples}:
        raise RuntimeError("Flash verdict cache does not cover every reconstructed pair")

    scorer, reranker_load_error = _load_reranker_fail_open()
    enriched, reranker_diagnostics = add_reranker_features(examples, scorer)
    reranker_diagnostics["load_error"] = reranker_load_error

    label_sets = {
        "gpt_oss": [row.gpt_oss_keep for row in enriched],
        "flash": [flash_by_key[row.key] for row in enriched],
    }
    feature_sets = {
        "base": BASE_FEATURE_NAMES,
        "base_reranker": BASE_FEATURE_NAMES + RERANK_FEATURE_NAMES,
    }
    fitted: dict[str, FittedConfiguration] = {}
    for label_name, labels in label_sets.items():
        for feature_name, feature_names in feature_sets.items():
            name = f"{label_name}_{feature_name}"
            fitted[name] = fit_configuration(enriched, labels, feature_names)

    eligible = [
        (name, result)
        for name, result in fitted.items()
        if float(result.report["heldout"]["false_drop_rate"]) <= 0.05
    ]
    if not eligible:
        eligible = list(fitted.items())
    best_name, best = max(
        eligible,
        key=lambda row: (
            float(row[1].report["heldout"]["drop_agreement"]),
            -float(row[1].report["heldout"]["false_drop_rate"]),
            int(row[1].report["heldout"]["predicted_drops"]),
            row[0],
        ),
    )
    bands = select_cascade_bands(best.labels, best.probabilities)
    other_label_name = "flash" if best_name.startswith("gpt_oss_") else "gpt_oss"
    other_labels = np.asarray(label_sets[other_label_name], dtype=np.int64)
    robustness = cascade_metrics(
        other_labels,
        best.probabilities,
        lo=float(bands["lo"]),
        hi=float(bands["hi"]),
    )
    train_cascade = cascade_metrics(
        best.labels[best.train_mask],
        best.probabilities[best.train_mask],
        lo=float(bands["lo"]),
        hi=float(bands["hi"]),
    )
    heldout_cascade = cascade_metrics(
        best.labels[~best.train_mask],
        best.probabilities[~best.train_mask],
        lo=float(bands["lo"]),
        hi=float(bands["hi"]),
    )

    gpt_labels = label_sets["gpt_oss"]
    flash_labels = label_sets["flash"]
    report = write_report(
        args.out,
        variant="semantic_edge_flash_reranker_cascade",
        metrics={
            "coverage": coverage,
            "flash_vs_gpt_oss": _agreement_metrics(gpt_labels, flash_labels),
            "disagreement_sample": disagreement_sample(
                enriched, flash_by_key, flash_reasons
            ),
            "reranker": reranker_diagnostics,
            "refits": {name: dict(result.report) for name, result in fitted.items()},
            "best_configuration": best_name,
            "cascade": {
                "selection_scope": "all_reconstructed_pairs",
                "constraints": {
                    "auto_drop_precision_min": 0.95,
                    "overall_false_drop_rate_max": 0.05,
                    "requires_nonempty_auto_keep_and_auto_drop": True,
                },
                "against_selected_labels": bands,
                "train": train_cascade,
                "heldout": heldout_cascade,
                "robustness_label_set": other_label_name,
                "against_other_model_labels": robustness,
            },
            "flash_judge": {
                "model": FLASH_MODEL,
                "batch_size": flash_artifact["batch_size"],
                "batches": len(flash_artifact["batches"]),
                "failed_batches": flash_artifact["failed_batches"],
                "cache_path": str(args.flash_cache),
            },
        },
        cost={
            "flash_judged_pairs": len(enriched),
            "flash_batch_calls": len(flash_artifact["batches"]),
            "reranker_direction_calls": len(enriched) * 2,
            "runtime_changes": 0,
        },
        timestamp_source=args.timestamp_source,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--knowledge", required=True)
    parser.add_argument(
        "--flash-cache",
        default="evals/quality/2026-07-14-flash-edge-verdicts.json",
    )
    parser.add_argument(
        "--out", default="evals/quality/2026-07-14-edge-cascade.json"
    )
    parser.add_argument(
        "--timestamp-source", default="orchestrator-cascade-experiment-2026-07-14"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_experiment(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
