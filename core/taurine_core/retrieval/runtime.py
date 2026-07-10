"""Production wiring for one immutable calibrated retrieval generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from rapidfuzz.distance import Levenshtein

from taurine_core.evaluation.graph_retrieval import CalibrationArtifact, load_calibration
from taurine_core.indexer.communities import projection_communities
from taurine_core.indexer.embed import embed_queries
from taurine_core.models_local import (
    LocalModelUnavailable,
    local_nli_scorer,
    local_pair_reranker,
)
from taurine_core.retrieval.calibration import AbstentionPolicy, HeadCalibrator
from taurine_core.retrieval.evidence import SharedRetrievalEngine
from taurine_core.retrieval.heads import RawHeadCandidate
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.rerank import LocalPassageReranker
from taurine_core.retrieval.snapshot import IndexSnapshot, load_index_snapshot
from taurine_core.retrieval.triples import FactRecallIndex, TripleRecognizer
from taurine_core.retrieval.types import RetrievalMode
from taurine_core.retrieval.validate import EvidenceValidator


CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "graph_retrieval"
    / "calibration-v1.json"
)


class _UnavailablePairScorer:
    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> Sequence[float]:
        del query, documents, instruction, deadline_at
        raise LocalModelUnavailable("bundled local reranker is unavailable")


@dataclass(frozen=True)
class _EntityResolution:
    candidates: tuple[RawHeadCandidate, ...]
    ambiguous: bool


class RetrievalRuntime:
    """Own the active snapshot inputs and create session-scoped engines."""

    def __init__(
        self,
        knowledge_database: str | Path,
        *,
        calibration_path: str | Path = CALIBRATION_PATH,
    ) -> None:
        self.knowledge_database = Path(knowledge_database)
        self._lock = RLock()
        self._snapshot: IndexSnapshot = load_index_snapshot(self.knowledge_database)
        self._communities: dict[str, int] = projection_communities(
            self._snapshot.projection
        )
        self._stores: dict[str, Any] = {}
        self.artifact: CalibrationArtifact = load_calibration(calibration_path)
        self._validate_snapshot(self._snapshot)
        try:
            self.pair_scorer: Any = local_pair_reranker()
        except LocalModelUnavailable:
            self.pair_scorer = _UnavailablePairScorer()
        try:
            self.nli_scorer: Any | None = local_nli_scorer()
        except LocalModelUnavailable:
            self.nli_scorer = None

    def make_engine(
        self,
        mode: RetrievalMode,
        registry: EvidenceRegistry,
        *,
        ambiguity_acceptor: Any | None = None,
    ) -> SharedRetrievalEngine:
        with self._lock:
            snapshot = self._snapshot
            store = self._stores.get(snapshot.build_id)
            if store is None:
                store = snapshot.open_knowledge_store()
                self._stores[snapshot.build_id] = store
        calibrator = HeadCalibrator(
            self.artifact,
            mode=mode,
            artifact_hash=self.artifact.hash,
            config_hash=self.artifact.config_hash,
            model_manifest=self.artifact.model_manifest,
        )
        triple_calibration = self.artifact.triple_recognition
        recognition_floor = getattr(triple_calibration, f"{mode}_floor")
        fallback_floor = getattr(
            triple_calibration, f"{mode}_fallback_embedding_floor"
        )
        ambiguity_low = (
            getattr(triple_calibration, f"{mode}_ambiguity_low")
            if mode != "live"
            else None
        )
        ambiguity_high = (
            getattr(triple_calibration, f"{mode}_ambiguity_high")
            if mode != "live"
            else None
        )
        triple_recognizer = TripleRecognizer(
            FactRecallIndex(snapshot, store, calibrator),
            store,
            self.pair_scorer,
            mode=mode,
            recognition_floor=recognition_floor,
            fallback_embedding_floor=fallback_floor,
            ambiguity_low=ambiguity_low,
            ambiguity_high=ambiguity_high,
            ambiguity_acceptor=ambiguity_acceptor,
        )
        rerank_floor = (
            self.artifact.rerank.live_floor
            if mode == "live"
            else self.artifact.rerank.chat_floor
        )
        return SharedRetrievalEngine(
            snapshot,
            store,
            calibrator,
            AbstentionPolicy(self.artifact, mode=mode),
            query_embedder=lambda text: embed_queries([text])[0],
            registry=registry,
            reranker=LocalPassageReranker(
                self.pair_scorer,
                mode=mode,
                acceptance_floor=rerank_floor,
            ),
            entity_resolver=lambda surface, etype, query: self._resolve_entity(
                snapshot,
                store,
                surface,
                etype,
                query,
            ),
            triple_recognizer=triple_recognizer,
        )

    def make_validator(self, registry: EvidenceRegistry) -> EvidenceValidator:
        return EvidenceValidator(
            registry,
            self.nli_scorer,
            entailment_floor=self.artifact.nli.entailment_floor,
        )

    def close(self) -> None:
        with self._lock:
            stores, self._stores = tuple(self._stores.values()), {}
        for store in stores:
            store.close()

    @property
    def snapshot(self) -> IndexSnapshot:
        with self._lock:
            return self._snapshot

    def community_labels(self, snapshot: IndexSnapshot | None = None) -> dict[str, int]:
        """Cached label-propagation communities for the active snapshot.

        Computed once per snapshot activation. Passing an older snapshot (a
        reader that raced a swap) recomputes for exactly that generation, so
        node ids and community ids never mix across builds.
        """
        with self._lock:
            current = self._snapshot
            cached = self._communities
        if snapshot is None or snapshot is current:
            return cached
        return projection_communities(snapshot.projection)

    def activate_from_disk(self) -> IndexSnapshot:
        """Atomically publish a newly completed clean build for future sessions."""

        snapshot = load_index_snapshot(self.knowledge_database)
        self._validate_snapshot(snapshot)
        communities = projection_communities(snapshot.projection)
        with self._lock:
            self._snapshot = snapshot
            self._communities = communities
        return snapshot

    def remove_source(self, source: str) -> int:
        """Surgically drop one source from the active index and republish.

        Instant path for disable/delete: no source is re-read and nothing is
        re-embedded. Surviving rows keep pre-removal weights until the next
        sync rebuilds cleanly. Existing session stores share the same SQLite
        file, so they observe the deletes immediately.
        """
        from taurine_core.indexer.knowledge_store import KnowledgeStore

        store = KnowledgeStore(self.knowledge_database)
        try:
            removed = store.remove_source(source)
        finally:
            store.close()
        if removed:
            self.activate_from_disk()
        return removed

    def _validate_snapshot(self, snapshot: IndexSnapshot) -> None:
        if snapshot.calibration_hash != self.artifact.hash:
            raise ValueError(
                "active knowledge snapshot calibration does not match the runtime artifact"
            )
        if dict(snapshot.model_manifest) != self.artifact.model_manifest:
            raise ValueError("active knowledge snapshot model manifest is incompatible")
        if snapshot.config_manifest.get("config_hash") != self.artifact.config_hash:
            raise ValueError("active knowledge snapshot config manifest is incompatible")

    def _resolve_entity(
        self,
        snapshot: IndexSnapshot,
        store: Any,
        surface: str,
        etype: str | None,
        query: str,
    ) -> _EntityResolution:
        del query
        normalized = surface.strip().casefold()
        ids = tuple(snapshot.alias_index.get(normalized, ()))
        exact = bool(ids)
        matched_alias = normalized
        if not ids:
            distances = [
                (Levenshtein.distance(normalized, alias), alias)
                for alias in snapshot.alias_index
                if _fuzzy_alias_allowed(normalized, alias)
            ]
            if distances:
                best_distance = min(row[0] for row in distances)
                best_aliases = sorted(
                    alias for distance, alias in distances if distance == best_distance
                )
                if len(best_aliases) == 1:
                    matched_alias = best_aliases[0]
                    ids = tuple(snapshot.alias_index[matched_alias])
        contexts = {
            row.entity_id: row
            for row in store.entity_contexts(ids, build_id=snapshot.build_id)
        }
        if etype:
            requested = etype.strip().casefold().replace("-", "_")
            ids = tuple(
                entity_id
                for entity_id in ids
                if entity_id in contexts
                and contexts[entity_id].etype.casefold().replace("-", "_") == requested
            )
        candidates = tuple(
            RawHeadCandidate(
                "entity",
                entity_id,
                1.0 if exact else _fuzzy_score(normalized, matched_alias),
                contexts[entity_id].canonical,
            )
            for entity_id in ids
            if entity_id in contexts
        )
        return _EntityResolution(candidates, len(candidates) != 1)


def _fuzzy_alias_allowed(surface: str, alias: str) -> bool:
    if not surface or not alias or any(char.isdigit() for char in surface + alias):
        return False
    distance = Levenshtein.distance(surface, alias)
    maximum = 2 if max(len(surface), len(alias)) >= 8 else 1
    return min(len(surface), len(alias)) >= 4 and distance <= maximum


def _fuzzy_score(surface: str, canonical: str) -> float:
    target = canonical.casefold()
    denominator = max(len(surface), len(target), 1)
    return max(0.0, 1.0 - (Levenshtein.distance(surface, target) / denominator))
