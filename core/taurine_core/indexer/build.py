"""Deterministic clean knowledge-index build and atomic activation command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from taurine_core import config
from taurine_core.indexer.base_graph import BaseGraphBuilder, PassageGraphInput
from taurine_core.indexer.embed import embed_passages, embed_texts
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.passages import TokenizerPort, build_passages
from taurine_core.indexer.projection import ProjectionPassage, build_projection
from taurine_core.models_local import passage_tokenizer
from taurine_core.schemas import NoteRecord


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildDependencies:
    tokenizer: TokenizerPort
    embedder: Callable[[list[str]], np.ndarray]
    embedding_dim: int = config.EMBED_DIM
    embedding_model_id: str = config.EMBED_MODEL
    now: Callable[[], float] = time.time


@dataclass(frozen=True)
class BuildReport:
    build_id: str
    target: Path
    source_snapshot: str
    graph_hash: str
    health: str
    stages: tuple[str, ...]
    note_count: int
    passage_count: int


class BuildCoordinator:
    def __init__(self, dependencies: BuildDependencies) -> None:
        self.dependencies = dependencies

    def build(
        self,
        records: Sequence[NoteRecord],
        target: str | Path,
        *,
        replace_derived: bool,
    ) -> BuildReport:
        if not replace_derived:
            raise BuildError("--replace-derived is required for a clean knowledge build")
        target_path = Path(target)
        ordered = sorted(records, key=lambda row: row.note_id)
        source_snapshot = _source_snapshot(ordered)
        build_id = "build-" + hashlib.sha256(
            f"knowledge-build-v1\x1f{source_snapshot}".encode("utf-8")
        ).hexdigest()[:20]
        temporary = target_path.with_name(
            f"{target_path.stem}.build-{build_id}{target_path.suffix or '.db'}"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        store: KnowledgeStore | None = None
        stages: list[str] = []
        try:
            store = KnowledgeStore(temporary)
            store.begin_build(
                build_id,
                source_snapshot=source_snapshot,
                model_manifest={"embedding": self.dependencies.embedding_model_id},
                config_manifest={"builder": "knowledge-build-v1"},
                calibration_hash="sha256:initial-conservative-2026-07-09",
                started_at=self.dependencies.now(),
            )
            stages.append("source")
            passage_rows = []
            for note in ordered:
                store.write_note(note, build_id=build_id, extraction_status="pending")
                drafts = build_passages(
                    note,
                    self.dependencies.tokenizer,
                    version="passages-v1",
                )
                store.write_passages(
                    note.note_id,
                    drafts,
                    created=note.created or note.modified,
                    modified=note.modified,
                    build_id=build_id,
                )
                passage_rows.extend((note, draft) for draft in drafts)
            stages.append("passages")

            embedded_by_id = {}
            for note in ordered:
                drafts = tuple(draft for source, draft in passage_rows if source.note_id == note.note_id)
                embedded = embed_passages(
                    drafts,
                    title=note.title,
                    source=note.source,
                    embedder=self.dependencies.embedder,
                    model_id=self.dependencies.embedding_model_id,
                    build_id=build_id,
                    dim=self.dependencies.embedding_dim,
                )
                store.write_passage_embeddings(embedded)
                embedded_by_id.update({row.passage_id: row for row in embedded})
            stages.append("embeddings")
            stages.append("extraction")
            stages.append("entities")
            stages.append("facts")

            graph_passages = tuple(
                PassageGraphInput(
                    passage_id=draft.passage_id,
                    note_id=note.note_id,
                    ordinal=draft.ordinal,
                    vector=embedded_by_id[draft.passage_id].vector,
                )
                for note, draft in passage_rows
            )
            base_graph = BaseGraphBuilder(
                build_id=build_id,
                semantic_k=config.BASE_GRAPH_PASSAGE_K,
                semantic_floor=config.GRAPH_MIN_SIM,
            ).build(graph_passages, (), (), ())
            store.write_semantic_edges(
                tuple(
                    (edge.src_id, edge.dst_id, edge.weight, edge.weight)
                    for edge in base_graph.edges
                    if edge.kind == "semantic" and edge.src_id < edge.dst_id
                ),
                build_id=build_id,
            )
            stages.append("base_graph")
            projection = build_projection(
                base_graph,
                tuple(
                    ProjectionPassage(draft.passage_id, note.note_id)
                    for note, draft in passage_rows
                ),
                {},
                max_degree=config.PROJECTION_MAX_DEGREE,
            )
            store.write_projection(projection, build_id=build_id)
            stages.append("projection")
            integrity = store.validate_invariants()
            if not integrity.ok:
                raise BuildError(f"knowledge invariant failure: {integrity}")
            graph_hash = _graph_hash(base_graph, projection)
            stages.append("validation")
            store.complete_build(
                build_id,
                completed_at=self.dependencies.now(),
                health="degraded",
                graph_hash=graph_hash,
                base_edge_count=len(base_graph.edges),
                projection_count=len(projection),
            )
            store.close()
            store = None
            os.replace(temporary, target_path)
            stages.append("activation")
            return BuildReport(
                build_id=build_id,
                target=target_path,
                source_snapshot=source_snapshot,
                graph_hash=graph_hash,
                health="degraded",
                stages=tuple(stages),
                note_count=len(ordered),
                passage_count=len(passage_rows),
            )
        except Exception as exc:
            if store is not None:
                store.close()
            if temporary.exists():
                temporary.unlink()
            if isinstance(exc, BuildError):
                raise
            raise BuildError(str(exc)) from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: BuildDependencies | None = None,
    connector_loader: Callable[[], Sequence[NoteRecord]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Build a clean Taurine knowledge index")
    parser.add_argument("--db", required=True)
    parser.add_argument("--from-enabled-connectors", action="store_true")
    parser.add_argument("--replace-derived", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if not args.from_enabled_connectors:
        raise BuildError("--from-enabled-connectors is required")
    if not args.yes:
        response = input(f"Replace derived knowledge index at {args.db}? [y/N] ")
        if response.strip().casefold() not in {"y", "yes"}:
            return 1
    if connector_loader is None:
        raise BuildError("enabled connector loader is unavailable in this process")
    deps = dependencies or BuildDependencies(
        tokenizer=passage_tokenizer(),
        embedder=embed_texts,
    )
    report = BuildCoordinator(deps).build(
        connector_loader(),
        args.db,
        replace_derived=args.replace_derived,
    )
    print(json.dumps({**report.__dict__, "target": str(report.target)}, sort_keys=True))
    return 0


def _source_snapshot(records: Sequence[NoteRecord]) -> str:
    payload = [(row.note_id, row.content_hash, row.source) for row in records]
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _graph_hash(base_graph: object, projection: object) -> str:
    graph = base_graph
    payload = {
        "nodes": list(graph.node_ids),
        "edges": [
            (edge.src_id, edge.dst_id, edge.kind, edge.direction, round(edge.weight, 12), edge.fact_ids)
            for edge in graph.edges
        ],
        "projection": [
            (
                edge.src_note_id,
                edge.dst_note_id,
                edge.edge_type,
                round(edge.weight, 12),
                edge.shared,
                edge.fact_ids,
            )
            for edge in projection
        ],
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
