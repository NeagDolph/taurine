"""Edge-audit harness (edge-quality spec item 7).

Stratified sampling of graph edges (contains / semantic / per fact predicate)
rendered with their grounding snippets for human spot-checks; optional LLM
precision scoring with the DEG-RAG rubric (``--llm``); optional edge-class
knockout ablation over a checked-in query file (``--ablate``). Output is a
markdown report to stdout or ``--out``.

Read-only: the harness never writes to the knowledge database, and it codes
defensively around schema generations (adjudication columns and semantic-edge
reasons may be absent on databases built before the adjudication lane).

CLI::

    python -m taurine_core.evaluation.edge_audit --db <path> \
        [--sample 100] [--llm] [--ablate] [--out report.md] [--seed 7]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from taurine_core import config
from taurine_core.indexer.base_graph import materialize_base_graph
from taurine_core.retrieval.ppr import TypedPPR
from taurine_core.retrieval.types import RetrievalQuery


DEFAULT_QUERIES_PATH = (
    Path(__file__).resolve().parents[2] / "evals" / "graph_retrieval" / "audit_queries.jsonl"
)

# TODO(W2-D): once the adjudication lane lands this getattr resolves to
# config.ADJUDICATION_MODEL; the literal is the spec'd default until then.
_DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"

_SNIPPET_BOUND = 300
_STRATUM_MINIMUM = 5
_KNOCKOUTS: tuple[tuple[str, str | None], ...] = (
    ("full", None),
    ("no_semantic", "semantic"),
    ("no_contains", "contains"),
    ("no_fact", "fact"),
)

_JUDGE_SYSTEM = (
    "You audit knowledge-graph edges for reliability (DEG-RAG rubric). Judge each "
    "edge ONLY against its supplied snippets — never from world knowledge. Score "
    "semantic accuracy of the connection versus the snippet text, significance "
    "(penalize trivial or incidental connections), endpoint/type compatibility, "
    "and direction. Return JSON "
    '{"judgments": [{"id": "<edge id>", "score": 0.0-1.0, '
    '"verdict": "supported"|"partial"|"unsupported"|"trivial", '
    '"note": "<=15 words"}]} with exactly one entry per supplied edge id.'
)
_VERDICTS = frozenset({"supported", "partial", "unsupported", "trivial"})


# ── shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SampledEdge:
    stratum: str  # "contains" | "semantic" | "fact:<predicate>"
    kind: str  # "contains" | "semantic" | "fact"
    edge_id: str
    endpoints: str
    weight: float
    fields: tuple[tuple[str, str], ...]
    snippets: tuple[str, ...]


@dataclass(frozen=True)
class Judgment:
    edge_id: str
    score: float
    verdict: str
    note: str


@dataclass(frozen=True)
class StratumScore:
    stratum: str
    sampled: int
    scored: int
    mean_score: float
    pct_supported: float  # fraction of scored edges with score >= 0.7
    pct_unsupported: float  # fraction of scored edges with score < 0.3


@dataclass(frozen=True)
class AuditQuery:
    id: str
    shape: str
    text: str


@dataclass(frozen=True)
class KnockoutRow:
    name: str
    overlap: float  # top-N note overlap vs the full graph
    # (note_id, rank in full ranking, rank in knockout ranking or None)
    top3_displacement: tuple[tuple[str, int, int | None], ...]


@dataclass(frozen=True)
class QueryAblation:
    query_id: str
    text: str
    shape: str
    teleport_source: str  # "engine" | "uniform_fallback"
    knockouts: tuple[KnockoutRow, ...]
    full_top: tuple[str, ...]


# ── sqlite helpers ─────────────────────────────────────────────────────


def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})"))


def _rows_for_ids(
    conn: sqlite3.Connection, template: str, ids: Sequence[str], *, chunk: int = 500
) -> list[sqlite3.Row]:
    ordered = sorted(set(ids))
    rows: list[sqlite3.Row] = []
    for start in range(0, len(ordered), chunk):
        part = ordered[start : start + chunk]
        marks = ",".join("?" for _ in part)
        rows.extend(conn.execute(template.format(marks=marks), part).fetchall())
    return rows


def latest_build_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT build_id FROM index_builds WHERE status = 'complete'"
        " ORDER BY completed_at DESC, build_id DESC LIMIT 1"
    ).fetchone()
    return None if row is None else str(row[0])


def corpus_stats(conn: sqlite3.Connection, build_id: str) -> dict[str, Any]:
    tables = (
        "notes",
        "passages",
        "entities",
        "entity_mentions",
        "facts",
        "passage_semantic_edges",
        "passage_entity_edges",
        "note_projection_edges",
    )
    counts = {
        table: int(
            conn.execute(
                f"SELECT count(*) FROM {table} WHERE build_id = ?", (build_id,)
            ).fetchone()[0]
        )
        for table in tables
    }
    fact_columns = _columns(conn, "facts")
    if "adjudicated" in fact_columns:
        adjudication: dict[str, Any] = {
            "available": True,
            "total_facts": counts["facts"],
            "adjudicated": int(
                conn.execute(
                    "SELECT count(*) FROM facts WHERE build_id = ? AND adjudicated = 1",
                    (build_id,),
                ).fetchone()[0]
            ),
            "rejected": int(
                conn.execute(
                    "SELECT count(*) FROM facts WHERE build_id = ? AND status = 'rejected'",
                    (build_id,),
                ).fetchone()[0]
            ),
        }
    else:
        adjudication = {"available": False}
    note_columns = _columns(conn, "notes")
    if "snapshot_document" in note_columns:
        snapshot_documents: dict[str, Any] = {
            "available": True,
            "count": int(
                conn.execute(
                    "SELECT count(*) FROM notes WHERE build_id = ? AND snapshot_document = 1",
                    (build_id,),
                ).fetchone()[0]
            ),
        }
    else:
        snapshot_documents = {"available": False}
    edge_columns = _columns(conn, "passage_semantic_edges")
    if "reason" in edge_columns:
        semantic_reasons: dict[str, Any] = {
            "available": True,
            "total": counts["passage_semantic_edges"],
            "with_reason": int(
                conn.execute(
                    "SELECT count(*) FROM passage_semantic_edges"
                    " WHERE build_id = ? AND reason <> ''",
                    (build_id,),
                ).fetchone()[0]
            ),
        }
    else:
        semantic_reasons = {"available": False}
    fact_predicates = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            "SELECT predicate, count(*) FROM facts WHERE build_id = ?"
            " GROUP BY predicate ORDER BY predicate",
            (build_id,),
        )
    }
    return {
        "build_id": build_id,
        "tables": counts,
        "fact_predicates": fact_predicates,
        "adjudication": adjudication,
        "snapshot_documents": snapshot_documents,
        "semantic_reasons": semantic_reasons,
    }


# ── snippets ───────────────────────────────────────────────────────────


def _snippet(
    text: str,
    highlight: tuple[int, int] | None = None,
    *,
    bound: int = _SNIPPET_BOUND,
) -> str:
    """Bounded window of ``text``, centered on ``highlight`` when given, with
    the visible highlighted span wrapped in ``[[ ]]``."""

    if highlight is not None:
        start, end = highlight
        if not (0 <= start < end <= len(text)):
            highlight = None
    if highlight is None:
        window = text[:bound]
        suffix = "…" if len(text) > bound else ""
        return window + suffix
    start, end = highlight
    span = min(end - start, bound)
    slack = bound - span
    window_start = max(0, start - slack // 2)
    window_end = min(len(text), window_start + bound)
    window_start = max(0, window_end - bound)
    visible_start = max(start, window_start)
    visible_end = min(end, window_end)
    parts = [
        "…" if window_start > 0 else "",
        text[window_start:visible_start],
        "[[",
        text[visible_start:visible_end],
        "]]",
        text[visible_end:window_end],
        "…" if window_end < len(text) else "",
    ]
    return "".join(parts)


def _passage_relative(
    note_start: int, note_end: int, passage_char_start: int, passage_text: str, surface: str
) -> tuple[int, int] | None:
    """Convert a note-relative span to passage-relative offsets, falling back
    to a surface search when the recorded offsets do not line up."""

    start = note_start - passage_char_start
    end = note_end - passage_char_start
    if 0 <= start < end <= len(passage_text):
        if not surface or passage_text[start:end] == surface:
            return (start, end)
    if surface:
        found = passage_text.find(surface)
        if found >= 0:
            return (found, found + len(surface))
    if 0 <= start < end <= len(passage_text):
        return (start, end)
    return None


# ── stratified sampling ────────────────────────────────────────────────


def _allocate(sizes: Mapping[str, int], total: int, *, minimum: int = _STRATUM_MINIMUM) -> dict[str, int]:
    """Proportional allocation with a per-stratum minimum of ``minimum`` (or
    the stratum size when smaller). Minimums always win over ``total``; any
    remainder is distributed proportionally to leftover capacity by largest
    remainder with a deterministic name tie-break."""

    allocation = {name: min(size, minimum) for name, size in sizes.items() if size > 0}
    remaining = total - sum(allocation.values())
    if remaining <= 0 or not allocation:
        return allocation
    capacity = {name: sizes[name] - allocation[name] for name in allocation}
    open_total = sum(capacity.values())
    if open_total <= 0:
        return allocation
    remaining = min(remaining, open_total)
    shares = {name: remaining * capacity[name] / open_total for name in allocation}
    extra = {name: min(capacity[name], int(shares[name])) for name in allocation}
    leftover = remaining - sum(extra.values())
    for name in sorted(allocation, key=lambda name: (-(shares[name] - extra[name]), name)):
        if leftover <= 0:
            break
        if extra[name] < capacity[name]:
            extra[name] += 1
            leftover -= 1
    return {name: allocation[name] + extra[name] for name in allocation}


def _pick(rows: list[Any], count: int, *, seed: int, stratum: str) -> list[Any]:
    if count >= len(rows):
        return list(rows)
    rng = random.Random(f"{seed}:{stratum}")
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), count))]


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):g}"


def sample_edges(
    conn: sqlite3.Connection, build_id: str, *, sample: int = 100, seed: int = 7
) -> tuple[SampledEdge, ...]:
    """Deterministic stratified edge sample: contains, semantic, and one
    stratum per fact predicate present in the build."""

    contains_rows = conn.execute(
        "SELECT passage_id, entity_id, confidence, mentions, idf, weight"
        " FROM passage_entity_edges WHERE build_id = ?"
        " ORDER BY passage_id, entity_id",
        (build_id,),
    ).fetchall()
    semantic_columns = _columns(conn, "passage_semantic_edges")
    has_reason = "reason" in semantic_columns
    semantic_rows = conn.execute(
        "SELECT src_passage_id, dst_passage_id, cosine, weight"
        + (", reason" if has_reason else "")
        + " FROM passage_semantic_edges WHERE build_id = ?"
        " ORDER BY src_passage_id, dst_passage_id",
        (build_id,),
    ).fetchall()
    fact_columns = _columns(conn, "facts")
    has_adjudication = "adjudicated" in fact_columns
    adjudication_select = (
        ", adjudication_score, adjudication_quote, adjudicated" if has_adjudication else ""
    )
    fact_rows = conn.execute(
        "SELECT fact_id, subject_entity_id, predicate, object_entity_id,"
        " source_passage_id, confidence, status, observed_at,"
        " assertion_start, assertion_end"
        + adjudication_select
        + " FROM facts WHERE build_id = ? ORDER BY fact_id",
        (build_id,),
    ).fetchall()
    facts_by_predicate: dict[str, list[sqlite3.Row]] = {}
    for row in fact_rows:
        facts_by_predicate.setdefault(str(row["predicate"]), []).append(row)

    sizes: dict[str, int] = {
        "contains": len(contains_rows),
        "semantic": len(semantic_rows),
        **{f"fact:{predicate}": len(rows) for predicate, rows in facts_by_predicate.items()},
    }
    allocation = _allocate(sizes, sample)

    picked_contains = _pick(
        contains_rows, allocation.get("contains", 0), seed=seed, stratum="contains"
    )
    picked_semantic = _pick(
        semantic_rows, allocation.get("semantic", 0), seed=seed, stratum="semantic"
    )
    picked_facts = {
        predicate: _pick(
            rows, allocation.get(f"fact:{predicate}", 0), seed=seed, stratum=f"fact:{predicate}"
        )
        for predicate, rows in sorted(facts_by_predicate.items())
    }

    passage_ids = (
        [str(row["passage_id"]) for row in picked_contains]
        + [str(row["src_passage_id"]) for row in picked_semantic]
        + [str(row["dst_passage_id"]) for row in picked_semantic]
        + [str(row["source_passage_id"]) for rows in picked_facts.values() for row in rows]
    )
    passages = {
        str(row["passage_id"]): row
        for row in _rows_for_ids(
            conn,
            "SELECT passage_id, note_id, ordinal, char_start, text FROM passages"
            " WHERE passage_id IN ({marks})",
            passage_ids,
        )
    }
    entity_ids = [str(row["entity_id"]) for row in picked_contains] + [
        str(row[key])
        for rows in picked_facts.values()
        for row in rows
        for key in ("subject_entity_id", "object_entity_id")
    ]
    entities = {
        str(row["entity_id"]): row
        for row in _rows_for_ids(
            conn,
            "SELECT entity_id, canonical, etype FROM entities WHERE entity_id IN ({marks})",
            entity_ids,
        )
    }
    note_ids = [str(row["note_id"]) for row in passages.values()]
    notes = {
        str(row["note_id"]): row
        for row in _rows_for_ids(
            conn,
            "SELECT note_id, title FROM notes WHERE note_id IN ({marks})",
            note_ids,
        )
    }

    def passage_label(passage_id: str) -> str:
        passage = passages.get(passage_id)
        if passage is None:
            return passage_id
        note = notes.get(str(passage["note_id"]))
        title = str(note["title"]) if note is not None else str(passage["note_id"])
        return f"{title}#p{int(passage['ordinal'])}"

    edges: list[SampledEdge] = []
    for row in picked_contains:
        passage_id = str(row["passage_id"])
        entity_id = str(row["entity_id"])
        entity = entities.get(entity_id)
        canonical = str(entity["canonical"]) if entity is not None else entity_id
        etype = str(entity["etype"]) if entity is not None else "?"
        passage = passages.get(passage_id)
        snippet = ""
        if passage is not None:
            mention = conn.execute(
                "SELECT surface, char_start, char_end FROM entity_mentions"
                " WHERE entity_id = ? AND passage_id = ? ORDER BY char_start LIMIT 1",
                (entity_id, passage_id),
            ).fetchone()
            highlight = None
            if mention is not None:
                highlight = _passage_relative(
                    int(mention["char_start"]),
                    int(mention["char_end"]),
                    int(passage["char_start"]),
                    str(passage["text"]),
                    str(mention["surface"]),
                )
            snippet = _snippet(str(passage["text"]), highlight)
        edges.append(
            SampledEdge(
                stratum="contains",
                kind="contains",
                edge_id=f"c:{passage_id}:{entity_id}",
                endpoints=f"{passage_label(passage_id)} ⟷ {canonical} ({etype})",
                weight=float(row["weight"]),
                fields=(
                    ("weight", _format_number(float(row["weight"]))),
                    ("confidence", _format_number(float(row["confidence"]))),
                    ("mentions", str(int(row["mentions"]))),
                    ("idf", _format_number(float(row["idf"]))),
                ),
                snippets=(snippet,),
            )
        )

    for row in picked_semantic:
        src_id = str(row["src_passage_id"])
        dst_id = str(row["dst_passage_id"])
        fields: list[tuple[str, str]] = [
            ("cosine", _format_number(float(row["cosine"]))),
            ("weight", _format_number(float(row["weight"]))),
        ]
        if has_reason:
            fields.append(("reason", str(row["reason"]) or "(empty)"))
        snippets = tuple(
            _snippet(str(passages[passage_id]["text"])) if passage_id in passages else ""
            for passage_id in (src_id, dst_id)
        )
        edges.append(
            SampledEdge(
                stratum="semantic",
                kind="semantic",
                edge_id=f"s:{src_id}:{dst_id}",
                endpoints=f"{passage_label(src_id)} ⟷ {passage_label(dst_id)}",
                weight=float(row["weight"]),
                fields=tuple(fields),
                snippets=snippets,
            )
        )

    for predicate, rows in picked_facts.items():
        for row in rows:
            subject = entities.get(str(row["subject_entity_id"]))
            obj = entities.get(str(row["object_entity_id"]))
            subject_label = (
                f"{subject['canonical']} ({subject['etype']})"
                if subject is not None
                else str(row["subject_entity_id"])
            )
            object_label = (
                f"{obj['canonical']} ({obj['etype']})"
                if obj is not None
                else str(row["object_entity_id"])
            )
            fields = [
                ("confidence", _format_number(float(row["confidence"]))),
                ("status", str(row["status"])),
                ("observed_at", _format_number(float(row["observed_at"]))),
            ]
            if has_adjudication:
                fields.append(("adjudicated", str(int(row["adjudicated"] or 0))))
                if row["adjudication_score"] is not None:
                    fields.append(
                        ("adjudication_score", _format_number(float(row["adjudication_score"])))
                    )
                if row["adjudication_quote"]:
                    fields.append(("adjudication_quote", str(row["adjudication_quote"])))
            passage = passages.get(str(row["source_passage_id"]))
            snippet = ""
            if passage is not None:
                highlight = _passage_relative(
                    int(row["assertion_start"]),
                    int(row["assertion_end"]),
                    int(passage["char_start"]),
                    str(passage["text"]),
                    "",
                )
                snippet = _snippet(str(passage["text"]), highlight)
            edges.append(
                SampledEdge(
                    stratum=f"fact:{predicate}",
                    kind="fact",
                    edge_id=f"f:{row['fact_id']}",
                    endpoints=f"{subject_label} —{predicate}→ {object_label}",
                    weight=float(row["confidence"]),
                    fields=tuple(fields),
                    snippets=(snippet,),
                )
            )
    return tuple(edges)


# ── LLM scoring ────────────────────────────────────────────────────────


def _judge_model() -> str:
    return str(getattr(config, "ADJUDICATION_MODEL", _DEFAULT_JUDGE_MODEL))


def score_edges(
    edges: Sequence[SampledEdge],
    client: Any,
    *,
    model: str,
    batch_size: int = 10,
    concurrency: int = 4,
) -> tuple[dict[str, Judgment], int]:
    """Score sampled edges with the grounded DEG-RAG rubric.

    Batches of ``batch_size`` edges, at most ``concurrency`` in-flight calls,
    temperature 0. Fail-open per batch: a failed or malformed batch leaves its
    edges unscored (returned count) instead of aborting the audit.
    """

    batches = [
        list(edges[start : start + batch_size]) for start in range(0, len(edges), batch_size)
    ]

    async def _run() -> list[tuple[list[SampledEdge], Any]]:
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def one(batch: list[SampledEdge]) -> tuple[list[SampledEdge], Any]:
            payload = {
                "edges": [
                    {
                        "id": edge.edge_id,
                        "kind": edge.kind,
                        "edge": edge.endpoints,
                        "weight": edge.weight,
                        "snippets": list(edge.snippets),
                    }
                    for edge in batch
                ]
            }
            async with semaphore:
                try:
                    return batch, await client.complete_json(
                        model,
                        _JUDGE_SYSTEM,
                        json.dumps(payload, ensure_ascii=False),
                        timeout_s=config.LLM_TIMEOUT_S,
                        temperature=0.0,
                    )
                except Exception:  # noqa: BLE001 - fail-open per batch (LLMError included)
                    return batch, None

        return await asyncio.gather(*(one(batch) for batch in batches))

    judgments: dict[str, Judgment] = {}
    unscored = 0
    for batch, response in asyncio.run(_run()):
        parsed: dict[str, Judgment] = {}
        rows = response.get("judgments") if isinstance(response, dict) else None
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            edge_id = str(row.get("id", ""))
            verdict = str(row.get("verdict", "")).strip().casefold()
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                continue
            if not edge_id or verdict not in _VERDICTS or not math.isfinite(score):
                continue
            parsed[edge_id] = Judgment(
                edge_id=edge_id,
                score=min(1.0, max(0.0, score)),
                verdict=verdict,
                note=str(row.get("note", ""))[:120],
            )
        for edge in batch:
            judgment = parsed.get(edge.edge_id)
            if judgment is None:
                unscored += 1
            else:
                judgments[edge.edge_id] = judgment
    return judgments, unscored


def aggregate_scores(
    edges: Sequence[SampledEdge], judgments: Mapping[str, Judgment]
) -> tuple[StratumScore, ...]:
    """Per-stratum precision estimates: mean score, %>=0.7, %<0.3."""

    by_stratum: dict[str, list[SampledEdge]] = {}
    for edge in edges:
        by_stratum.setdefault(edge.stratum, []).append(edge)
    rows: list[StratumScore] = []
    for stratum in _stratum_order(by_stratum):
        members = by_stratum[stratum]
        scores = [
            judgments[edge.edge_id].score for edge in members if edge.edge_id in judgments
        ]
        scored = len(scores)
        rows.append(
            StratumScore(
                stratum=stratum,
                sampled=len(members),
                scored=scored,
                mean_score=(sum(scores) / scored) if scored else 0.0,
                pct_supported=(
                    sum(1 for score in scores if score >= 0.7) / scored if scored else 0.0
                ),
                pct_unsupported=(
                    sum(1 for score in scores if score < 0.3) / scored if scored else 0.0
                ),
            )
        )
    return tuple(rows)


def _stratum_order(by_stratum: Mapping[str, Any]) -> list[str]:
    fixed = [name for name in ("contains", "semantic") if name in by_stratum]
    facts = sorted(name for name in by_stratum if name.startswith("fact:"))
    other = sorted(
        name for name in by_stratum if name not in set(fixed) and not name.startswith("fact:")
    )
    return fixed + facts + other


# ── ablation ───────────────────────────────────────────────────────────


def load_queries(path: str | Path) -> tuple[AuditQuery, ...]:
    queries: list[AuditQuery] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        queries.append(
            AuditQuery(id=str(row["id"]), shape=str(row["shape"]), text=str(row["text"]))
        )
    return tuple(queries)


def _note_ranking(
    graph: Any, teleport: Mapping[str, float], passage_note: Mapping[str, str]
) -> list[str]:
    """Notes ranked by summed PPR mass of their passages (stable tie-break)."""

    result = TypedPPR(graph).run(passage_reset=teleport)
    if not result.accepted:
        return []
    scores: dict[str, float] = {}
    for node_id, rank in result.ranks.items():
        note_id = passage_note.get(node_id)
        if note_id is not None and rank > 0.0:
            scores[note_id] = scores.get(note_id, 0.0) + rank
    return sorted(scores, key=lambda note_id: (-scores[note_id], note_id))


def run_ablation(
    database: str | Path,
    queries: Sequence[AuditQuery],
    *,
    engine: Any | None = None,
    top_n: int = 12,
) -> tuple[QueryAblation, ...]:
    """Edge-class knockout diagnostic (no labels).

    Each query runs through the shared engine ONCE on the untouched graph to
    obtain the exact teleport vector the engine used; each knockout then
    re-materializes the matrix from the snapshot's base-graph rows with one
    edge class filtered out (via ``materialize_base_graph``) and re-runs
    TypedPPR with that same teleport. This never modifies retrieval/ source
    and never mutates the engine. Reported numbers are top-``top_n`` note
    overlap vs the full graph and rank displacement of the full graph's
    top-3 notes.
    """

    close = None
    if engine is None:
        from taurine_core.retrieval.registry import EvidenceRegistry
        from taurine_core.retrieval.runtime import RetrievalRuntime

        runtime = RetrievalRuntime(Path(database))
        engine = runtime.make_engine("chat", EvidenceRegistry())
        close = runtime.close
    try:
        snapshot = engine.snapshot
        with sqlite3.connect(f"file:{Path(database)}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            passage_note = {
                str(row["passage_id"]): str(row["note_id"])
                for row in conn.execute(
                    "SELECT passage_id, note_id FROM passages WHERE build_id = ?",
                    (snapshot.build_id,),
                )
            }
            latest = conn.execute("SELECT max(modified) FROM notes").fetchone()[0]
        now = float(latest or 0.0) + 86400.0

        results: list[QueryAblation] = []
        for query in queries:
            pack = engine.retrieve(RetrievalQuery(query.text, "chat", now))
            teleport = {
                str(node_id): float(mass) for node_id, mass in pack.trace.teleport.items()
            }
            teleport_source = "engine"
            if not teleport:
                # The engine abstained before PPR: keep the diagnostic total by
                # seeding a uniform reset over all passages (flagged as such).
                teleport = {
                    node_id: 1.0
                    for node_id, kind in zip(
                        snapshot.base_graph.node_ids,
                        snapshot.base_graph.node_kinds,
                        strict=True,
                    )
                    if kind == "passage"
                }
                teleport_source = "uniform_fallback"

            rankings: dict[str, list[str]] = {}
            for name, knocked in _KNOCKOUTS:
                edges = tuple(
                    edge
                    for edge in snapshot.base_graph.edges
                    if knocked is None or edge.kind != knocked
                )
                graph = materialize_base_graph(
                    build_id=snapshot.build_id,
                    node_ids=snapshot.base_graph.node_ids,
                    node_kinds=snapshot.base_graph.node_kinds,
                    edges=edges,
                )
                rankings[name] = _note_ranking(graph, teleport, passage_note)

            full = rankings["full"]
            full_top = tuple(full[:top_n])
            knockout_rows: list[KnockoutRow] = []
            for name, _knocked in _KNOCKOUTS:
                ranked = rankings[name]
                top = ranked[:top_n]
                if not full_top:
                    overlap = 1.0 if not top else 0.0
                else:
                    overlap = len(set(top) & set(full_top)) / len(full_top)
                positions = {note_id: rank for rank, note_id in enumerate(ranked, start=1)}
                displacement = tuple(
                    (note_id, rank, positions.get(note_id))
                    for rank, note_id in enumerate(full[:3], start=1)
                )
                knockout_rows.append(
                    KnockoutRow(name=name, overlap=overlap, top3_displacement=displacement)
                )
            results.append(
                QueryAblation(
                    query_id=query.id,
                    text=query.text,
                    shape=query.shape,
                    teleport_source=teleport_source,
                    knockouts=tuple(knockout_rows),
                    full_top=full_top,
                )
            )
        return tuple(results)
    finally:
        if close is not None:
            close()


# ── flags ──────────────────────────────────────────────────────────────


def build_flags(
    edges: Sequence[SampledEdge],
    judgments: Mapping[str, Judgment] | None,
    stats: Mapping[str, Any],
) -> tuple[str, ...]:
    flags: list[str] = []
    if judgments:
        for row in aggregate_scores(edges, judgments):
            if row.scored and row.mean_score < 0.5:
                flags.append(
                    f"stratum '{row.stratum}' mean score {row.mean_score:.2f} < 0.5"
                    f" ({row.scored} scored)"
                )
        for edge in edges:
            judgment = judgments.get(edge.edge_id)
            if judgment is not None and edge.weight >= 0.9 and judgment.score < 0.3:
                flags.append(
                    f"{edge.kind} edge {edge.edge_id} has weight {edge.weight:.2f}"
                    f" but LLM score {judgment.score:.2f} ({edge.endpoints})"
                )
    reasons = stats.get("semantic_reasons", {})
    if reasons.get("available") and int(reasons.get("with_reason", 0)) > 0:
        missing = [
            edge.edge_id
            for edge in edges
            if edge.kind == "semantic" and dict(edge.fields).get("reason") == "(empty)"
        ]
        if missing:
            flags.append(
                "semantic edges without a reason while reasons exist elsewhere: "
                + ", ".join(missing)
            )
    return tuple(flags)


# ── report ─────────────────────────────────────────────────────────────


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ⏎ ")


def render_report(
    *,
    database: str | Path,
    seed: int,
    stats: Mapping[str, Any],
    edges: Sequence[SampledEdge],
    scores: Sequence[StratumScore] | None,
    judgments: Mapping[str, Judgment] | None,
    unscored: int,
    judge_model: str | None,
    ablation: Sequence[QueryAblation] | None,
    flags: Sequence[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# Edge audit — {database}")
    lines.append("")
    lines.append(f"Build `{stats['build_id']}` · seed {seed} · {len(edges)} sampled edges.")
    lines.append("")

    lines.append("## Corpus stats")
    lines.append("")
    lines.append("| table | rows |")
    lines.append("| --- | ---: |")
    for table, count in stats["tables"].items():
        lines.append(f"| {table} | {count} |")
    lines.append("")
    adjudication = stats["adjudication"]
    if adjudication.get("available"):
        lines.append(
            f"- adjudication coverage: {adjudication['adjudicated']}/"
            f"{adjudication['total_facts']} facts adjudicated,"
            f" {adjudication['rejected']} rejected"
        )
    else:
        lines.append("- adjudication coverage: n/a (columns absent — pre-adjudication build)")
    snapshot_docs = stats["snapshot_documents"]
    if snapshot_docs.get("available"):
        lines.append(f"- snapshot documents: {snapshot_docs['count']}")
    else:
        lines.append("- snapshot documents: n/a (column absent)")
    reasons = stats["semantic_reasons"]
    if reasons.get("available"):
        lines.append(
            f"- semantic edges with reason: {reasons['with_reason']}/{reasons['total']}"
        )
    else:
        lines.append("- semantic edge reasons: n/a (column absent)")
    lines.append("")

    lines.append("## Sampled edges")
    lines.append("")
    by_stratum: dict[str, list[SampledEdge]] = {}
    for edge in edges:
        by_stratum.setdefault(edge.stratum, []).append(edge)
    stratum_totals = {
        "contains": stats["tables"]["passage_entity_edges"],
        "semantic": stats["tables"]["passage_semantic_edges"],
        **{
            f"fact:{predicate}": count
            for predicate, count in stats.get("fact_predicates", {}).items()
        },
    }
    for stratum in _stratum_order(by_stratum):
        members = by_stratum[stratum]
        total = stratum_totals.get(stratum)
        suffix = f" ({len(members)} of {total})" if total is not None else f" ({len(members)})"
        lines.append(f"### {stratum}{suffix}")
        lines.append("")
        header = "| id | edge | weight | fields | grounding |"
        if judgments is not None:
            header = "| id | edge | weight | fields | grounding | llm |"
        lines.append(header)
        lines.append("| --- | --- | ---: | --- | --- |" + (" --- |" if judgments is not None else ""))
        for edge in members:
            fields = "; ".join(f"{name}={value}" for name, value in edge.fields)
            grounding = " ‖ ".join(snippet for snippet in edge.snippets if snippet)
            row = (
                f"| {_cell(edge.edge_id)} | {_cell(edge.endpoints)} | {edge.weight:.3f}"
                f" | {_cell(fields)} | {_cell(grounding)} |"
            )
            if judgments is not None:
                judgment = judgments.get(edge.edge_id)
                verdict = (
                    f"{judgment.verdict} {judgment.score:.2f} — {judgment.note}"
                    if judgment is not None
                    else "unscored"
                )
                row += f" {_cell(verdict)} |"
            lines.append(row)
        lines.append("")

    if scores is not None:
        lines.append("## LLM precision (DEG-RAG rubric)")
        lines.append("")
        lines.append(
            f"Model `{judge_model}`, temperature 0, judged only against the supplied"
            " snippets. Scores are reliability estimates per edge type."
        )
        if unscored:
            lines.append(f"Unscored edges (failed/malformed batches, fail-open): {unscored}.")
        lines.append("")
        lines.append("| stratum | sampled | scored | mean score | %>=0.7 | %<0.3 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in scores:
            lines.append(
                f"| {row.stratum} | {row.sampled} | {row.scored} | {row.mean_score:.2f}"
                f" | {row.pct_supported:.0%} | {row.pct_unsupported:.0%} |"
            )
        lines.append("")

    if ablation is not None:
        lines.append("## Ablation diagnostics")
        lines.append("")
        lines.append(
            "Diagnostic only — unlabeled overlap/rank-shift numbers against the full"
            " graph, not quality metrics (P1 labeled evaluation remains the"
            " calibration path). Knockouts re-materialize the query-time matrix"
            " from the snapshot's base-graph rows with one edge class removed and"
            " re-run PPR with the engine's own teleport vector."
        )
        lines.append("")
        for row in ablation:
            lines.append(f"### {row.query_id} — \"{row.text}\" ({row.shape})")
            lines.append("")
            lines.append(f"Teleport: {row.teleport_source}.")
            lines.append("")
            lines.append("| knockout | top-12 overlap vs full | full top-3 displacement |")
            lines.append("| --- | ---: | --- |")
            for knockout in row.knockouts:
                displacement = "; ".join(
                    f"{note_id}: {full_rank}→{ablated_rank if ablated_rank is not None else '–'}"
                    for note_id, full_rank, ablated_rank in knockout.top3_displacement
                )
                lines.append(
                    f"| {knockout.name} | {knockout.overlap:.2f} | {_cell(displacement) or '–'} |"
                )
            lines.append("")

    lines.append("## Flags")
    lines.append("")
    if flags:
        for flag in flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────


def _openrouter_client() -> Any | None:
    """Minimal local replica of the build CLI's optional LLM construction."""

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass
    key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        return None
    from taurine_core.llm import OpenRouterClient

    return OpenRouterClient(key)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit knowledge-graph edge quality (stratified sample, optional"
        " LLM scoring and edge-class ablation)."
    )
    parser.add_argument("--db", required=True, help="path to taurine-knowledge.db")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--llm", action="store_true", help="score sampled edges with the judge")
    parser.add_argument("--ablate", action="store_true", help="run edge-class knockout queries")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES_PATH))
    parser.add_argument("--out", default=None, help="write the markdown report here")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    database = Path(args.db)
    if not database.exists():
        print(f"knowledge database not found: {database}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        build_id = latest_build_id(conn)
        if build_id is None:
            print(f"no complete knowledge build in {database}", file=sys.stderr)
            return 2
        stats = corpus_stats(conn, build_id)
        edges = sample_edges(conn, build_id, sample=args.sample, seed=args.seed)
    finally:
        conn.close()

    judgments: dict[str, Judgment] | None = None
    scores: tuple[StratumScore, ...] | None = None
    unscored = 0
    judge_model: str | None = None
    llm_note: str | None = None
    if args.llm:
        client = _openrouter_client()
        if client is None:
            llm_note = "LLM scoring requested but OPEN_ROUTER_API_KEY is not set — skipped"
        else:
            judge_model = _judge_model()
            judgments, unscored = score_edges(edges, client, model=judge_model)
            scores = aggregate_scores(edges, judgments)

    ablation: tuple[QueryAblation, ...] | None = None
    if args.ablate:
        ablation = run_ablation(database, load_queries(args.queries))

    flags = list(build_flags(edges, judgments, stats))
    if llm_note is not None:
        flags.append(llm_note)
    report = render_report(
        database=database,
        seed=args.seed,
        stats=stats,
        edges=edges,
        scores=scores,
        judgments=judgments,
        unscored=unscored,
        judge_model=judge_model,
        ablation=ablation,
        flags=tuple(flags),
    )
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
