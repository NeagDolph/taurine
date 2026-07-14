"""Local quality benchmarks for extraction and later non-LLM lanes.

The extraction command opens the production-derived knowledge index read-only,
uses its accepted active facts as silver truth, and writes one stable JSON
report per extractor variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from taurine_core import config
from taurine_core.indexer.knowledge_store import KnowledgeStore
from taurine_core.indexer.relation_completion import _normalize_predicate
from taurine_core.schemas import Extraction, NoteRecord


@dataclass(frozen=True)
class SilverFact:
    note_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str


def extract_notes_dispatch(
    notes: list[NoteRecord],
    *,
    tokenizer: Any = None,
    progress: Any = None,
    backend: str | None = None,
) -> dict[str, Extraction]:
    """Lazy import keeps this harness usable while extractor modules initialize."""
    from taurine_core.indexer.extract import extract_notes_dispatch as dispatch

    return dispatch(notes, tokenizer=tokenizer, progress=progress, backend=backend)


def _normalize_surface(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def score_extractions(
    extractions: Mapping[str, Extraction],
    silver_facts: Sequence[SilverFact],
    aliases: Mapping[str, frozenset[str]],
) -> dict[str, int | float]:
    """Compare extracted logical triples with entity-alias-aware silver truth."""
    silver = {
        (
            row.note_id,
            row.subject_entity_id,
            _normalize_predicate(row.predicate),
            row.object_entity_id,
        )
        for row in silver_facts
    }
    matched: set[tuple[str, str, str, str]] = set()
    novel: set[tuple[str, str, str, str]] = set()
    extracted: set[tuple[str, str, str, str]] = set()

    for note_id, extraction in extractions.items():
        for relation in extraction.fact_candidates:
            predicate = _normalize_predicate(relation.predicate)
            head = _normalize_surface(relation.head)
            tail = _normalize_surface(relation.tail)
            subject_ids = aliases.get(head, frozenset())
            object_ids = aliases.get(tail, frozenset())
            candidates = {
                (note_id, subject_id, predicate, object_id)
                for subject_id in subject_ids
                for object_id in object_ids
            }
            silver_matches = candidates & silver
            if silver_matches:
                matched.update(silver_matches)
                extracted.update(silver_matches)
                continue
            logical = (
                note_id,
                min(subject_ids) if subject_ids else f"surface:{head}",
                predicate,
                min(object_ids) if object_ids else f"surface:{tail}",
            )
            extracted.add(logical)
            novel.add(logical)

    silver_count = len(silver)
    return {
        "notes": len(extractions),
        "failed_notes": sum(row.failed for row in extractions.values()),
        "mentions": sum(len(row.mentions) for row in extractions.values()),
        "extracted_facts": len(extracted),
        "silver_facts": silver_count,
        "silver_facts_matched": len(matched),
        "silver_fact_recall": round(len(matched) / silver_count, 6)
        if silver_count
        else 0.0,
        "new_fact_yield": len(novel),
    }


def write_report(
    path: str | Path,
    *,
    variant: str,
    metrics: Mapping[str, Any],
    cost: Mapping[str, Any],
    timestamp_source: str,
) -> dict[str, Any]:
    """Write the stable report envelope; callers own the timestamp source."""
    payload = {
        "variant": variant,
        "metrics": dict(metrics),
        "cost": dict(cost),
        "timestamp_source": timestamp_source,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _sample_notes(
    notes: Sequence[NoteRecord], sample: int | None, seed: int
) -> list[NoteRecord]:
    if sample is None or sample <= 0 or sample >= len(notes):
        return sorted(notes, key=lambda row: row.note_id)

    def sample_key(note: NoteRecord) -> str:
        return hashlib.sha256(f"{seed}\x1f{note.note_id}".encode()).hexdigest()

    return sorted(sorted(notes, key=sample_key)[:sample], key=lambda row: row.note_id)


def _load_benchmark_data(
    database: str | Path,
    *,
    sample: int | None,
    seed: int,
) -> tuple[list[NoteRecord], list[SilverFact], dict[str, frozenset[str]]]:
    store = KnowledgeStore(database, read_only=True)
    try:
        notes = [
            NoteRecord(
                note_id=row.note_id,
                title=row.title,
                folder=row.folder,
                modified=row.modified,
                created=row.created,
                content_hash=row.content_hash,
                text=row.text,
                gist=row.display_line,
                source=row.source,
                ingest_salience=row.ingest_salience,
            )
            for row in store.source_notes()
        ]
        selected = _sample_notes(notes, sample, seed)
        selected_ids = {row.note_id for row in selected}
        silver = [
            SilverFact(
                note_id=str(row["source_note_id"]),
                subject_entity_id=str(row["subject_entity_id"]),
                predicate=str(row["predicate"]),
                object_entity_id=str(row["object_entity_id"]),
            )
            for row in store._conn.execute(
                """
                SELECT source_note_id, subject_entity_id, predicate, object_entity_id
                FROM facts
                WHERE status = 'active'
                  AND adjudicated = 1
                  AND adjudication_score >= 0.7
                ORDER BY fact_id
                """
            )
            if str(row["source_note_id"]) in selected_ids
        ]
        alias_sets: dict[str, set[str]] = {}
        for row in store._conn.execute(
            """
            SELECT entity_id, canonical AS surface FROM entities
            UNION ALL
            SELECT entity_id, surface FROM entity_aliases
            """
        ):
            alias_sets.setdefault(_normalize_surface(str(row["surface"])), set()).add(
                str(row["entity_id"])
            )
        aliases = {key: frozenset(value) for key, value in alias_sets.items()}
        return selected, silver, aliases
    finally:
        store.close()


def _variant_path(output: Path, backend: str, count: int) -> Path:
    if count == 1:
        return output
    suffix = output.suffix or ".json"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}-{backend}{suffix}")


def _run_extraction(args: argparse.Namespace) -> int:
    notes, silver, aliases = _load_benchmark_data(
        args.db, sample=args.sample, seed=args.seed
    )
    backends = tuple(
        dict.fromkeys(
            value.strip() for value in args.backends.split(",") if value.strip()
        )
    )
    invalid = set(backends) - {"gliner2", "relex", "hybrid"}
    if not backends or invalid:
        raise ValueError(f"unsupported extraction backends: {sorted(invalid)}")

    if args.entity_threshold is not None:
        config.RELEX_ENTITY_THRESHOLD = args.entity_threshold
    if args.relation_threshold is not None:
        config.RELEX_RELATION_THRESHOLD = args.relation_threshold
    if args.adjacency_threshold is not None:
        config.RELEX_ADJACENCY_THRESHOLD = args.adjacency_threshold

    output = Path(args.out)
    for backend in backends:
        started = time.perf_counter()
        try:
            extractions = extract_notes_dispatch(notes, backend=backend)
        except Exception as exc:  # benchmark variants fail open like production
            extractions = {
                note.note_id: Extraction(
                    failed=True, error=f"benchmark dispatch failed: {exc}"
                )
                for note in notes
            }
        elapsed = time.perf_counter() - started
        metrics = score_extractions(extractions, silver, aliases)
        metrics.update(
            {
                "entity_threshold": (
                    float(getattr(config, "RELEX_ENTITY_THRESHOLD", 0.4))
                    if backend != "gliner2"
                    else None
                ),
                "relation_threshold": (
                    float(getattr(config, "RELEX_RELATION_THRESHOLD", 0.8))
                    if backend != "gliner2"
                    else None
                ),
                "adjacency_threshold": (
                    float(getattr(config, "RELEX_ADJACENCY_THRESHOLD", 0.55))
                    if backend != "gliner2"
                    else None
                ),
            }
        )
        report = write_report(
            _variant_path(output, backend, len(backends)),
            variant=backend,
            metrics=metrics,
            cost={"wall_time_seconds": round(elapsed, 6)},
            timestamp_source=args.timestamp_source,
        )
        print(json.dumps(report, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    extraction = subcommands.add_parser(
        "extraction", help="benchmark extraction backends"
    )
    extraction.add_argument("--db", required=True)
    extraction.add_argument("--backends", default="gliner2,relex,hybrid")
    extraction.add_argument("--sample", type=int)
    extraction.add_argument("--seed", type=int, default=7)
    extraction.add_argument("--entity-threshold", type=float)
    extraction.add_argument("--relation-threshold", type=float)
    extraction.add_argument("--adjacency-threshold", type=float)
    extraction.add_argument("--timestamp-source", required=True)
    extraction.add_argument("--out", required=True)
    extraction.set_defaults(run=_run_extraction)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
