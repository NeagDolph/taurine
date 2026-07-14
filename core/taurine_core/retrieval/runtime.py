"""Production wiring for one immutable calibrated retrieval generation."""

from __future__ import annotations

import logging

from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from rapidfuzz.distance import Levenshtein

from taurine_core import config
from taurine_core.evaluation.graph_retrieval import (
    CalibrationArtifact,
    CalibrationCurve,
    CalibrationIntegrityError,
    ModeCalibration,
    RerankCalibration,
    TripleRecognitionCalibration,
    load_calibration,
)
from taurine_core.indexer.communities import fallback_clusters
from taurine_core.indexer.embed import embed_queries
from taurine_core.models_local import (
    LocalModelUnavailable,
    local_pair_reranker,
)
from taurine_core.retrieval.calibration import AbstentionPolicy, HeadCalibrator
from taurine_core.retrieval.evidence import SharedRetrievalEngine
from taurine_core.retrieval.heads import RawHeadCandidate
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.rerank import (
    LocalPassageReranker,
    OpenRouterRerankClient,
    RerankError,
)
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


# §11.3 fail-closed table used when the checked-in artifact's content hash
# does not match its declared hash: every mode gets the strictest (live)
# floors, so a tampered/corrupted artifact can only make retrieval stricter.
def _conservative_curve(
    knots: tuple[tuple[float, float], ...], floor: float
) -> CalibrationCurve:
    return CalibrationCurve(
        knots=knots,
        acceptance_floor=floor,
        sample_count=0,
        trained_at="conservative-fallback",
        held_out={},
    )


def _conservative_mode() -> ModeCalibration:
    return ModeCalibration(
        heads={
            "dense": _conservative_curve(
                ((-1.0, 0.0), (0.45, 0.05), (0.62, 0.55), (0.78, 0.95), (1.0, 1.0)),
                0.7,
            ),
            "bm25": _conservative_curve(
                ((0.0, 0.0), (2.0, 0.1), (6.0, 0.7), (20.0, 1.0)), 0.75
            ),
            "exact": _conservative_curve(((0.0, 0.0), (0.5, 0.8), (1.0, 1.0)), 0.9),
            "entity": _conservative_curve(
                ((0.0, 0.0), (0.6, 0.4), (0.85, 0.9), (1.0, 1.0)), 0.85
            ),
            "triple": _conservative_curve(
                ((-1.0, 0.0), (0.55, 0.1), (0.72, 0.75), (1.0, 1.0)), 0.8
            ),
        },
        head_weights={
            "dense": 1.0,
            "bm25": 1.0,
            "exact": 1.0,
            "entity": 1.0,
            "triple": 1.0,
        },
        rrf_k=60,
        mmr_lambda=0.2,
        source_diversity_penalty=0.05,
        note_support_bonus=0.05,
        note_support_bonus_cap=0.1,
        evidence_floor=0.75,
        single_head_override=0.92,
        minimum_independent_heads=2,
        bm25_min_token_coverage=0.5,
    )


def _conservative_artifact(declared: CalibrationArtifact) -> CalibrationArtifact:
    """Keep the declared artifact's identity fields (so the snapshot
    cross-check semantics are unchanged) but replace every runtime floor and
    curve with the in-code conservative table."""

    mode = _conservative_mode()
    return declared.model_copy(
        update={
            "modes": {"live": mode, "chat": mode, "tool": mode},
            "triple_recognition": TripleRecognitionCalibration(
                live_floor=0.82,
                chat_floor=0.82,
                tool_floor=0.82,
                live_fallback_embedding_floor=0.9,
                chat_fallback_embedding_floor=0.9,
                tool_fallback_embedding_floor=0.9,
                chat_ambiguity_low=0.6,
                chat_ambiguity_high=0.9,
                tool_ambiguity_low=0.6,
                tool_ambiguity_high=0.9,
            ),
            "rerank": RerankCalibration(live_floor=0.76, chat_floor=0.76),
            "nli": declared.nli.model_copy(
                update={"entailment_floor": max(0.8, declared.nli.entailment_floor)}
            ),
        }
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


class _UnavailableRemotePairScorer:
    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> Sequence[float]:
        del query, documents, instruction, deadline_at
        raise RerankError("OpenRouter passage reranker is unavailable")


def _openrouter_pair_reranker() -> OpenRouterRerankClient:
    import os

    return OpenRouterRerankClient(
        os.environ.get("OPEN_ROUTER_API_KEY", ""),
        model_id=config.RERANK_API_MODEL,
    )


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
        self._stores: dict[str, Any] = {}
        # H4/§11.3: verify the artifact's content hash on load. On a mismatch
        # fail closed to the in-code conservative floors and surface the
        # condition as runtime status instead of raising: the tampered file's
        # floors are untrusted, but its identity fields still let the snapshot
        # cross-check below distinguish "same declared artifact" from "a build
        # made under a different artifact" (which stays an activation error).
        try:
            self.artifact: CalibrationArtifact = load_calibration(calibration_path)
            self.calibration_status = "verified"
        except CalibrationIntegrityError:
            declared = load_calibration(calibration_path, verify=False)
            self.artifact = _conservative_artifact(declared)
            self.calibration_status = "calibration_mismatch"
        self._validate_snapshot(self._snapshot)
        log = logging.getLogger("taurine.retrieval")
        self.pair_scorer_model_id = config.rerank_model_id()
        try:
            self.local_pair_scorer: Any = local_pair_reranker()
        except LocalModelUnavailable as exc:
            self.local_pair_scorer = _UnavailablePairScorer()
            if config.INFERENCE_BACKEND == "local":
                log.warning(
                    "local reranker unavailable (%s) — rerank quality degraded; "
                    "run: uv run python -m taurine_core.download_models",
                    exc,
                )
            else:
                log.warning(
                    "local Qwen triple recognizer and rerank fallback unavailable (%s); "
                    "run: uv run python -m taurine_core.download_models",
                    exc,
                )
        if config.INFERENCE_BACKEND == "openrouter":
            try:
                self.pair_scorer: Any = _openrouter_pair_reranker()
            except RerankError as exc:
                self.pair_scorer = _UnavailableRemotePairScorer()
                log.warning("OpenRouter passage reranker unavailable (%s)", exc)
        else:
            self.pair_scorer = self.local_pair_scorer
        self.active_model_manifest = {
            **self.artifact.model_manifest,
            "embedding": config.embed_model_id(),
            "reranker": self.pair_scorer_model_id,
        }

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
            self.local_pair_scorer,
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
        if config.INFERENCE_BACKEND == "openrouter":
            api_floors = config.RERANK_API_FLOOR_OVERRIDES
            passage_reranker = LocalPassageReranker(
                self.pair_scorer,
                mode=mode,
                acceptance_floor=float(api_floors["accept"]),
                model_id=self.pair_scorer_model_id,
                evidence_floor_override=float(api_floors["fallback"]),
                fallback_scorer=self.local_pair_scorer,
                fallback_acceptance_floor=rerank_floor,
                fallback_model_id=config.RERANK_MODEL,
            )
        else:
            passage_reranker = LocalPassageReranker(
                self.pair_scorer,
                mode=mode,
                acceptance_floor=rerank_floor,
                model_id=self.pair_scorer_model_id,
            )
        return SharedRetrievalEngine(
            snapshot,
            store,
            calibrator,
            AbstentionPolicy(self.artifact, mode=mode),
            query_embedder=lambda text: embed_queries([text])[0],
            registry=registry,
            reranker=passage_reranker,
            entity_resolver=lambda surface, etype, query: self._resolve_entity(
                snapshot,
                store,
                surface,
                etype,
                query,
            ),
            triple_recognizer=triple_recognizer,
            # §0.1: chat/tool get the liberal fallback lane; live stays strict.
            fallback_on_abstain=mode in {"chat", "tool"},
            calibration_status=self.calibration_status,
        )

    def make_validator(self, registry: EvidenceRegistry) -> EvidenceValidator:
        return EvidenceValidator(registry)

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
        """Contiguous cluster ids per note for one snapshot generation.

        Persisted clusters ride the snapshot itself (loaded per activation),
        so node ids and cluster ids never mix across builds; snapshots
        without cluster data (legacy fakes) fall back to remapped label
        propagation.
        """
        target = self.snapshot if snapshot is None else snapshot
        membership = getattr(target, "note_clusters", None)
        if membership is not None:
            return dict(membership)
        return fallback_clusters(target.projection)[0]

    def activate_from_disk(self) -> IndexSnapshot:
        """Atomically publish a newly completed clean build for future sessions."""

        snapshot = load_index_snapshot(self.knowledge_database)
        self._validate_snapshot(snapshot)
        with self._lock:
            previous_build_id = self._snapshot.build_id
        if snapshot.build_id != previous_build_id:
            # Only fresh installs are hash-verified: remove_source republishes
            # the same build after in-place row deletion, which legitimately
            # diverges from the recorded hash until the next clean build.
            self._verify_installed_hash(snapshot.build_id)
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    def _verify_installed_hash(self, build_id: str) -> None:
        from taurine_core.indexer.knowledge_store import KnowledgeStore

        store = KnowledgeStore(self.knowledge_database)
        try:
            if not store.verify_graph_hash(build_id):
                raise ValueError(
                    "installed knowledge build failed graph-hash verification"
                )
        finally:
            store.close()

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
        snapshot_manifest = dict(snapshot.model_manifest)
        snapshot_embedding = snapshot_manifest.pop("embedding", None)
        if snapshot_embedding != config.embed_model_id():
            raise ValueError(
                "active knowledge snapshot embedding model does not match "
                "the active embedding model"
            )
        artifact_manifest = dict(self.artifact.model_manifest)
        artifact_manifest.pop("embedding", None)
        if snapshot_manifest != artifact_manifest:
            raise ValueError("active knowledge snapshot model manifest is incompatible")
        expected_dim = config.embed_dim()
        for kind, matrix in (
            ("passage", snapshot.passage_matrix),
            ("fact", snapshot.fact_matrix),
        ):
            if matrix.shape[0] and matrix.shape[1] != expected_dim:
                raise ValueError(
                    f"active knowledge snapshot {kind} embedding dimension "
                    f"{matrix.shape[1]} does not match active dimension {expected_dim}"
                )
        if snapshot.config_manifest.get("config_hash") != self.artifact.config_hash:
            raise ValueError("active knowledge snapshot config manifest is incompatible")

    def _resolve_entity(
        self,
        snapshot: IndexSnapshot,
        store: Any,
        surface: str,
        etype: str | None,
        query: str,
    ) -> tuple[RawHeadCandidate, ...]:
        # Returns a plain sequence: heads._resolution_candidates treats a
        # sequence as ready-made RawHeadCandidates and derives ambiguity from
        # len != 1, which matches this resolver's semantics exactly.
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
        return tuple(
            RawHeadCandidate(
                "entity",
                entity_id,
                1.0 if exact else _fuzzy_score(normalized, matched_alias),
                contexts[entity_id].canonical,
            )
            for entity_id in ids
            if entity_id in contexts
        )


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
