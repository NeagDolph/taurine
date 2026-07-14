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
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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


@dataclass(frozen=True)
class AdjudicationExample:
    key: str
    premise: str
    hypothesis: str
    llm_keep: bool
    screenable: bool = True


@dataclass(frozen=True)
class AdjudicationScore:
    key: str
    entailment: float
    llm_keep: bool


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


def evaluate_keep_floor(
    scores: Sequence[AdjudicationScore], keep_floor: float
) -> dict[str, int | float | bool]:
    screened = [row for row in scores if row.entailment >= keep_floor]
    correct = sum(row.llm_keep for row in screened)
    precision = correct / len(screened) if screened else 0.0
    eliminated = len(screened) / len(scores) if scores else 0.0
    return {
        "keep_floor": round(float(keep_floor), 6),
        "keep_precision": round(precision, 6),
        "calls_eliminated": round(eliminated, 6),
        "screened_keeps": len(screened),
        "examples": len(scores),
        "target_met": bool(screened and precision >= 0.95),
    }


def fit_keep_floor(
    scores: Sequence[AdjudicationScore],
    *,
    floors: Sequence[float] = tuple(value / 100 for value in range(70, 100)),
    target_precision: float = 0.95,
) -> dict[str, int | float | bool]:
    candidates: list[dict[str, int | float | bool]] = []
    for floor in sorted(set(float(value) for value in floors)):
        metrics = evaluate_keep_floor(scores, floor)
        metrics["target_met"] = bool(
            metrics["screened_keeps"]
            and float(metrics["keep_precision"]) >= target_precision
        )
        candidates.append(metrics)
    if not candidates:
        raise ValueError("at least one NLI keep floor is required")
    return next(
        (row for row in candidates if row["target_met"]), candidates[-1]
    )


def score_adjudication_examples(
    examples: Sequence[AdjudicationExample], scorer: Any
) -> list[AdjudicationScore]:
    return [
        AdjudicationScore(
            key=row.key,
            entailment=(
                float(scorer.score(row.premise, row.hypothesis))
                if row.screenable
                else -1.0
            ),
            llm_keep=row.llm_keep,
        )
        for row in examples
    ]


def _read_only_connection(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        Path(path).resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def _fact_from_row(row: sqlite3.Row) -> Any:
    from taurine_core.indexer.facts import FactRecord

    return FactRecord(
        fact_id=str(row["fact_id"]),
        subject_entity_id=str(row["subject_entity_id"]),
        subject=str(row["subject"]),
        predicate=str(row["predicate"]),
        object_entity_id=str(row["object_entity_id"]),
        object=str(row["object"]),
        source_passage_id=str(row["source_passage_id"]),
        source_note_id=str(row["source_note_id"]),
        assertion_start=int(row["assertion_start"]),
        assertion_end=int(row["assertion_end"]),
        subject_start=int(row["subject_start"]),
        subject_end=int(row["subject_end"]),
        object_start=int(row["object_start"]),
        object_end=int(row["object_end"]),
        confidence=float(row["confidence"]),
        extraction_method=str(row["extraction_method"]),
        schema_id=str(row["schema_id"]),
        observed_at=float(row["observed_at"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        invalidated_at=row["invalidated_at"],
        superseded_by=row["superseded_by"],
        status=str(row["status"]),
        build_id=str(row["build_id"]),
        adjudication_score=row["adjudication_score"],
        adjudication_quote=str(row["adjudication_quote"]),
        adjudicated=bool(row["adjudicated"]),
    )


def _load_adjudication_examples(
    cache_path: str | Path, knowledge_path: str | Path
) -> tuple[list[AdjudicationExample], dict[str, int]]:
    from taurine_core.indexer.edge_adjudication import _fact_key
    from taurine_core.indexer.facts import PREDICATES
    from taurine_core.indexer.nli_screen import verbalize

    cache = _read_only_connection(cache_path)
    knowledge = _read_only_connection(knowledge_path)
    try:
        labels: dict[str, Mapping[str, Any]] = {}
        for row in cache.execute(
            "SELECT key, payload FROM adjudication_cache WHERE kind = 'fact' ORDER BY key"
        ):
            try:
                payload = json.loads(str(row["payload"]))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, Mapping):
                labels[str(row["key"])] = payload

        rows = knowledge.execute(
            """
            SELECT f.*, subject.canonical AS subject, object.canonical AS object,
                   p.text AS premise, p.content_hash AS passage_content_hash
            FROM facts AS f
            JOIN entities AS subject ON subject.entity_id = f.subject_entity_id
            JOIN entities AS object ON object.entity_id = f.object_entity_id
            JOIN passages AS p ON p.passage_id = f.source_passage_id
            ORDER BY f.fact_id
            """
        ).fetchall()
        examples_by_key: dict[str, AdjudicationExample] = {}
        matched_keys: set[str] = set()
        matched_fact_rows = 0
        excluded_rows = 0
        for row in rows:
            fact = _fact_from_row(row)
            draft = SimpleNamespace(content_hash=str(row["passage_content_hash"]))
            key = _fact_key(
                config.ADJUDICATION_MODEL,
                fact,
                {fact.source_passage_id: draft},
            )
            payload = labels.get(key)
            if payload is None:
                continue
            matched_fact_rows += 1
            matched_keys.add(key)
            spec = PREDICATES.get(fact.predicate)
            screenable = not (
                spec is None
                or spec.exclusive_object
                or fact.extraction_method == "llm_relation_completion"
            )
            if not screenable:
                excluded_rows += 1
            score = payload.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue
            examples_by_key.setdefault(
                key,
                AdjudicationExample(
                    key=key,
                    premise=str(row["premise"]),
                    hypothesis=verbalize(fact),
                    llm_keep=float(score)
                    >= float(config.ADJUDICATION_REJECT_BELOW),
                    screenable=screenable,
                ),
            )
        coverage = {
            "cached_fact_rows": len(labels),
            "knowledge_fact_rows": len(rows),
            "matched_fact_rows": matched_fact_rows,
            "matched_cache_keys": len(matched_keys),
            "excluded_rows": excluded_rows,
            "eligible_examples": sum(
                row.screenable for row in examples_by_key.values()
            ),
        }
        return list(examples_by_key.values()), coverage
    finally:
        cache.close()
        knowledge.close()


def _load_nli_scorer() -> Any:
    from taurine_core import models_local

    return models_local.local_nli_scorer(config.EVIDENCE_NLI_MODEL)


def _run_adjudication(args: argparse.Namespace) -> int:
    examples, coverage = _load_adjudication_examples(args.cache, args.knowledge)
    scorer = _load_nli_scorer()
    scored = score_adjudication_examples(examples, scorer)
    train = [row for row in scored if int(row.key, 16) % 4 != 0]
    heldout = [row for row in scored if int(row.key, 16) % 4 == 0]
    fitted = fit_keep_floor(train, target_precision=args.target_precision)
    floor = float(fitted["keep_floor"])
    heldout_metrics = evaluate_keep_floor(heldout, floor)
    heldout_metrics["target_met"] = bool(
        heldout_metrics["screened_keeps"]
        and float(heldout_metrics["keep_precision"]) >= args.target_precision
    )
    report = write_report(
        args.out,
        variant="nli_fact_screen",
        metrics={
            "keep_floor": floor,
            "target_precision": args.target_precision,
            "llm_keep_definition": (
                f"score >= {config.ADJUDICATION_REJECT_BELOW}"
            ),
            "train": fitted,
            "heldout": heldout_metrics,
            "label_coverage": coverage,
        },
        cost={
            "nli_calls": sum(row.screenable for row in examples),
            "llm_calls": 0,
        },
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
    adjudication = subcommands.add_parser(
        "adjudication", help="fit the local NLI fact-adjudication screen"
    )
    adjudication.add_argument("--cache", required=True)
    adjudication.add_argument("--knowledge", default="taurine-knowledge.db")
    adjudication.add_argument("--target-precision", type=float, default=0.95)
    adjudication.add_argument("--timestamp-source", required=True)
    adjudication.add_argument("--out", required=True)
    adjudication.set_defaults(run=_run_adjudication)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
