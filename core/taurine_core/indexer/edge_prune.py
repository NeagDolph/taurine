"""Transparent local classifier for cross-note semantic-edge pruning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FEATURE_NAMES = (
    "cosine",
    "shared_entity_count",
    "min_shared_entity_idf",
    "same_source",
    "title_jaccard",
    "length_ratio",
    "nli_src_to_dst",
    "nli_dst_to_src",
)

# Offline fit 2026-07-13, 557/557 labels matched. Heldout: drop agreement
# 0.806723, false-drop rate 0.031915. The mode remains flag-gated to the LLM
# because agreement missed the 0.90 adoption target.
INTERCEPT = -17.80822951084
WEIGHTS = {
    "cosine": 13.689557337711,
    "shared_entity_count": -0.076275852288,
    "min_shared_entity_idf": 0.099872333155,
    "same_source": 5.1862635243,
    "title_jaccard": -1.011065597425,
    "length_ratio": 1.821353473569,
    "nli_src_to_dst": 0.0,
    "nli_dst_to_src": 0.0,
}
DECISION_THRESHOLD = 0.16
NLI_MAX_CHARS = 1000

_TOKEN_RE = re.compile(r"[\w']+")
_TEMPLATE_PREFIXES = ("Both mention ", "Semantically similar passages from ")


@dataclass(frozen=True)
class EdgeFeatures:
    cosine: float
    shared_entity_count: int
    min_shared_entity_idf: float
    same_source: float
    title_jaccard: float
    length_ratio: float
    nli_src_to_dst: float = 0.0
    nli_dst_to_src: float = 0.0
    shared_entities: tuple[str, ...] = ()
    src_source: str = ""
    dst_source: str = ""

    def values(self) -> dict[str, float]:
        return {
            "cosine": float(self.cosine),
            "shared_entity_count": float(self.shared_entity_count),
            "min_shared_entity_idf": float(self.min_shared_entity_idf),
            "same_source": float(self.same_source),
            "title_jaccard": float(self.title_jaccard),
            "length_ratio": float(self.length_ratio),
            "nli_src_to_dst": float(self.nli_src_to_dst),
            "nli_dst_to_src": float(self.nli_dst_to_src),
        }


@dataclass(frozen=True)
class NLICrossEntailment:
    src_to_dst: float
    dst_to_src: float
    scorer_available: bool
    failed_directions: int


@dataclass(frozen=True)
class PruneResult:
    reasons: dict[tuple[str, str], str]
    dropped: frozenset[tuple[str, str]]
    reasoned_count: int
    dropped_count: int
    failed_batches: int = 0


def make_features(
    pair: Any,
    *,
    cosine: float,
    shared_entities: Sequence[tuple[str, float]] = (),
    src_source: str = "",
    dst_source: str = "",
    nli_src_to_dst: float = 0.0,
    nli_dst_to_src: float = 0.0,
) -> EdgeFeatures:
    """Build the runtime features plus reason-template metadata."""
    ranked = sorted(
        ((str(label), float(idf)) for label, idf in shared_entities if str(label)),
        key=lambda row: (-row[1], row[0].casefold(), row[0]),
    )
    title_left = set(_TOKEN_RE.findall(str(pair.src_title).casefold()))
    title_right = set(_TOKEN_RE.findall(str(pair.dst_title).casefold()))
    union = title_left | title_right
    title_jaccard = len(title_left & title_right) / len(union) if union else 0.0
    lengths = (len(str(pair.src_text)), len(str(pair.dst_text)))
    length_ratio = min(lengths) / max(lengths) if max(lengths) else 0.0
    return EdgeFeatures(
        cosine=float(cosine),
        shared_entity_count=len(ranked),
        min_shared_entity_idf=min((idf for _, idf in ranked), default=0.0),
        same_source=float(bool(src_source) and src_source == dst_source),
        title_jaccard=title_jaccard,
        length_ratio=length_ratio,
        nli_src_to_dst=float(nli_src_to_dst),
        nli_dst_to_src=float(nli_dst_to_src),
        shared_entities=tuple(label for label, _ in ranked[:3]),
        src_source=str(src_source),
        dst_source=str(dst_source),
    )


def nli_cross_entailment(
    pair: Any, scorer: Any | None, *, max_chars: int = NLI_MAX_CHARS
) -> NLICrossEntailment:
    """Score both passage directions, failing each unavailable direction to 0."""
    if scorer is None:
        return NLICrossEntailment(0.0, 0.0, False, 2)
    bound = max(1, int(max_chars))
    src_text = str(pair.src_text)[:bound]
    dst_text = str(pair.dst_text)[:bound]
    scores: list[float] = []
    failures = 0
    for premise, hypothesis in ((src_text, dst_text), (dst_text, src_text)):
        try:
            score = float(scorer.score(premise, hypothesis))
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("invalid NLI score")
        except Exception:
            score = 0.0
            failures += 1
        scores.append(score)
    return NLICrossEntailment(scores[0], scores[1], True, failures)


def keep_probability(features: EdgeFeatures) -> float:
    values = features.values()
    logit = float(INTERCEPT) + sum(
        float(WEIGHTS[name]) * values[name] for name in FEATURE_NAMES
    )
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def decide(
    pairs: Sequence[Any],
    features_by_pair: Mapping[tuple[str, str], EdgeFeatures],
) -> PruneResult:
    """Apply embedded logistic weights; missing/bad features keep fail-open."""
    reasons: dict[tuple[str, str], str] = {}
    dropped: set[tuple[str, str]] = set()
    failures = 0
    for pair in sorted(pairs, key=lambda row: row.pair):
        features = features_by_pair.get(pair.pair)
        if features is None:
            failures += 1
            continue
        try:
            probability = keep_probability(features)
        except Exception:
            failures += 1
            continue
        if probability < float(DECISION_THRESHOLD):
            dropped.add(pair.pair)
        else:
            reasons[pair.pair] = deterministic_reason(pair, features)
    return PruneResult(
        reasons=reasons,
        dropped=frozenset(dropped),
        reasoned_count=len(reasons),
        dropped_count=len(dropped),
        failed_batches=failures,
    )


def deterministic_reason(pair: Any, features: EdgeFeatures) -> str:
    entities = features.shared_entities
    if entities:
        if len(entities) == 1:
            rendered = entities[0]
        elif len(entities) == 2:
            rendered = f"{entities[0]} and {entities[1]}"
        else:
            rendered = f"{entities[0]}, {entities[1]}, and {entities[2]}"
        return f"Both mention {rendered}"
    left = _source_label(features.src_source or str(pair.src_title))
    right = _source_label(features.dst_source or str(pair.dst_title))
    return f"Semantically similar passages from {left} and {right}"


def is_template_reason(reason: str) -> bool:
    value = str(reason).strip()
    return not value or value.startswith(_TEMPLATE_PREFIXES)


def _source_label(value: str) -> str:
    return str(value).replace("_", " ").strip() or "unknown sources"
