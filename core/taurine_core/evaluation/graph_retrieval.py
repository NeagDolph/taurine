"""Strict evaluation and calibration artifacts for graph retrieval."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Split = Literal["train", "calibration", "held_out"]
ExampleKind = Literal[
    "extraction",
    "retrieval_live",
    "retrieval_chat",
    "citations",
    "entity_resolution",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationExample(_StrictModel):
    example_id: str = Field(min_length=1)
    split: Split
    source_note_ids: tuple[str, ...] = Field(min_length=1)
    kind: ExampleKind
    input: dict
    labels: dict


class CalibrationCurve(_StrictModel):
    knots: tuple[tuple[float, float], ...] = Field(min_length=2)
    acceptance_floor: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    trained_at: str = Field(min_length=1)
    held_out: dict[str, float]

    @model_validator(mode="after")
    def validate_monotonic(self) -> "CalibrationCurve":
        raw = [point[0] for point in self.knots]
        calibrated = [point[1] for point in self.knots]
        if any(not 0.0 <= score <= 1.0 for score in calibrated):
            raise ValueError("calibrated knot probabilities must be within [0, 1]")
        if any(right <= left for left, right in zip(raw, raw[1:], strict=False)):
            raise ValueError("raw-score knots must be strictly increasing")
        if any(right < left for left, right in zip(calibrated, calibrated[1:], strict=False)):
            raise ValueError("calibration probabilities must be monotonic")
        return self


class ModeCalibration(_StrictModel):
    heads: dict[str, CalibrationCurve]
    evidence_floor: float = Field(ge=0.0, le=1.0)
    single_head_override: float = Field(ge=0.0, le=1.0)
    minimum_independent_heads: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_heads(self) -> "ModeCalibration":
        required = {"dense", "bm25", "exact", "entity", "triple"}
        missing = sorted(required - set(self.heads))
        if missing:
            raise ValueError(f"missing calibration heads: {', '.join(missing)}")
        return self


class TripleRecognitionCalibration(_StrictModel):
    live_floor: float = Field(ge=0.0, le=1.0)
    chat_floor: float = Field(ge=0.0, le=1.0)


class RerankCalibration(_StrictModel):
    live_floor: float = Field(ge=0.0, le=1.0)
    chat_floor: float = Field(ge=0.0, le=1.0)


class NLICalibration(_StrictModel):
    entailment_floor: float = Field(ge=0.0, le=1.0)


class CalibrationArtifact(_StrictModel):
    artifact_version: str = Field(min_length=1)
    hash: str = Field(min_length=1)
    model_manifest: dict[str, str]
    config_hash: str = Field(min_length=1)
    modes: dict[Literal["live", "chat", "tool"], ModeCalibration]
    triple_recognition: TripleRecognitionCalibration
    rerank: RerankCalibration
    nli: NLICalibration

    @model_validator(mode="after")
    def validate_modes(self) -> "CalibrationArtifact":
        required = {"live", "chat", "tool"}
        if set(self.modes) != required:
            raise ValueError("calibration modes must be exactly live, chat, and tool")
        return self


class RetrievalMetrics(_StrictModel):
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_12: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    support_set_recall: float = Field(ge=0.0, le=1.0)


def load_jsonl(path: str | Path) -> tuple[EvaluationExample, ...]:
    source = Path(path)
    decoded: list[tuple[int, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            decoded.append((line_number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number} of {source}") from exc

    examples: list[EvaluationExample] = []
    seen: set[str] = set()
    for line_number, payload in decoded:
        try:
            example = EvaluationExample.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"invalid example on line {line_number} of {source}: {exc}") from exc
        if example.example_id in seen:
            raise ValueError(f"duplicate example_id {example.example_id!r} in {source}")
        seen.add(example.example_id)
        examples.append(example)
    return tuple(examples)


def load_calibration(path: str | Path) -> CalibrationArtifact:
    return CalibrationArtifact.model_validate_json(Path(path).read_text(encoding="utf-8"))


def validate_split_isolation(examples: tuple[EvaluationExample, ...]) -> None:
    assignments: dict[str, set[Split]] = {}
    for example in examples:
        for note_id in example.source_note_ids:
            assignments.setdefault(note_id, set()).add(example.split)
    leaked = {note_id: splits for note_id, splits in assignments.items() if len(splits) > 1}
    if leaked:
        detail = "; ".join(
            f"{note_id}: {', '.join(sorted(splits))}"
            for note_id, splits in sorted(leaked.items())
        )
        raise ValueError(f"source-note split leakage: {detail}")


def recall_at(ranking: list[str] | tuple[str, ...], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def ndcg_at(ranking: list[str] | tuple[str, ...], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranking[:k], 1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def mrr(ranking: list[str] | tuple[str, ...], relevant: set[str]) -> float:
    for rank, item in enumerate(ranking, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def support_set_recall(
    ranking: list[str] | tuple[str, ...], relevant: set[str]
) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking) & relevant) / len(relevant)


def citation_coverage(factual_sentences: int, cited_factual_sentences: int) -> float:
    if factual_sentences <= 0:
        return 1.0
    return min(max(cited_factual_sentences, 0), factual_sentences) / factual_sentences
