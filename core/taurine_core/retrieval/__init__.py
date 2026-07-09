"""Shared evidence-first retrieval subsystem."""

from .calibration import AbstentionPolicy, HeadCalibrator
from .evidence import empty_pack, weighted_score_aware_rrf
from .types import EvidencePack, RetrievalQuery

__all__ = [
    "AbstentionPolicy",
    "EvidencePack",
    "HeadCalibrator",
    "RetrievalQuery",
    "empty_pack",
    "weighted_score_aware_rrf",
]
