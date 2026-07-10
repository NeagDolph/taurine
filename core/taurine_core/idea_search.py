"""Idea lens: agentic concept search over the knowledge snapshot.

POST /graph/idea pipeline: dense+BM25 seed retrieval over the immutable
snapshot, 1-hop projection expansion, LLM per-note alignment adjudication,
and one agentic second hop around strong matches. Read-only consumer of the
snapshot and knowledge store — it never touches the retrieval engine.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from taurine_core import config
from taurine_core.indexer.embed import embed_queries
from taurine_core.schemas import IdeaMatch, IdeaSearchResponse

_LOG = logging.getLogger("taurine.idea")

SEED_LIMIT = 50               # candidate notes from dense+BM25
SEED_EXPAND_FROM = 15         # top seeds whose neighbors join round one
CANDIDATE_CAP = 80            # hard cap on round-one candidates
SECOND_HOP_CAP = 30           # extra candidates around strong matches
SECOND_HOP_FLOOR = 0.7        # strength gating the second hop
JUDGE_BATCH = 15              # notes per adjudication call
JUDGE_CONCURRENCY = 4
SNIPPET_MAX_CHARS = 350
BM25_SEED_K = 60
_REASON_MAX_CHARS = 200

_JUDGE_SYSTEM = (
    "You judge whether personal notes genuinely relate to the user's idea "
    'or concept. Return JSON {"judgements": [{"note_id": "<id>", '
    '"aligned": <bool>, "strength": <0.0-1.0>, "reason": "<≤18 words, '
    'concrete>"}]} with one entry per input note, reusing each note_id '
    "verbatim. aligned is true only when the note genuinely relates to the "
    "idea; strength reflects how central the idea is to the note (1.0 = the "
    "note is about it). Be strict: topical adjacency is not alignment."
)


@dataclass
class _Candidate:
    note_id: str
    title: str
    source: str
    display_line: str
    snippet: str
    expanded: bool


async def search_idea(
    idea: str,
    *,
    snapshot: Any,
    llm: Any,
    limit: int = 40,
) -> IdeaSearchResponse:
    """Run the full idea-lens pipeline. Never blocks the event loop: the
    sync retrieval parts run via asyncio.to_thread and LLM calls are async."""

    started = time.monotonic()
    text = " ".join(idea.split())
    capped_limit = max(5, min(80, int(limit)))

    scores, notes, passage_note, passage_text = await asyncio.to_thread(
        _load_corpus, snapshot, text
    )
    bm25_note_ids = await asyncio.to_thread(
        _bm25_notes, snapshot, text, passage_note
    )

    best_passage: dict[str, tuple[float, str]] = {}
    for passage_id, score in zip(snapshot.passage_ids, scores):
        note_id = passage_note.get(passage_id)
        if note_id is None:
            continue
        current = best_passage.get(note_id)
        if (
            current is None
            or score > current[0]
            or (score == current[0] and passage_id < current[1])
        ):
            best_passage[note_id] = (float(score), passage_id)

    seeds = sorted(best_passage, key=lambda nid: (-best_passage[nid][0], nid))[
        :SEED_LIMIT
    ]
    for note_id in bm25_note_ids:
        if len(seeds) >= SEED_LIMIT:
            break
        if note_id not in seeds and note_id in notes:
            seeds.append(note_id)
    seeds = [note_id for note_id in seeds if note_id in notes]

    adjacency = _projection_adjacency(snapshot.projection)
    candidates: dict[str, _Candidate] = {}
    for note_id in seeds:
        candidates[note_id] = _candidate(
            note_id, notes, best_passage, passage_text, expanded=False
        )
    for neighbor in _neighbors(seeds[:SEED_EXPAND_FROM], adjacency, candidates):
        if len(candidates) >= CANDIDATE_CAP:
            break
        if neighbor in notes:
            candidates[neighbor] = _candidate(
                neighbor, notes, best_passage, passage_text, expanded=True
            )

    judgements = await _judge(text, list(candidates.values()), llm)

    strong = sorted(
        note_id
        for note_id, (aligned, strength, _reason) in judgements.items()
        if aligned and strength >= SECOND_HOP_FLOOR
    )
    second_hop: dict[str, _Candidate] = {}
    for neighbor in _neighbors(strong, adjacency, candidates):
        if len(second_hop) >= SECOND_HOP_CAP:
            break
        if neighbor in notes:
            second_hop[neighbor] = _candidate(
                neighbor, notes, best_passage, passage_text, expanded=True
            )
    if second_hop:  # one extra round only
        judgements.update(await _judge(text, list(second_hop.values()), llm))
        candidates.update(second_hop)

    matches = [
        IdeaMatch(
            note_id=note_id,
            title=candidates[note_id].title,
            source=candidates[note_id].source,
            strength=strength,
            reason=reason,
        )
        for note_id, (aligned, strength, reason) in judgements.items()
        if aligned
    ]
    matches.sort(key=lambda match: (-match.strength, match.note_id))
    matches = matches[:capped_limit]

    latency_ms = (time.monotonic() - started) * 1000.0
    expanded = sum(1 for row in candidates.values() if row.expanded)
    _LOG.info(
        "idea search %s: considered=%d expanded=%d matched=%d in %.0f ms",
        hashlib.sha256(text.encode("utf-8")).hexdigest()[:8],
        len(candidates),
        expanded,
        len(matches),
        latency_ms,
    )
    return IdeaSearchResponse(
        idea=text,
        matches=matches,
        considered=len(candidates),
        expanded=expanded,
        latency_ms=latency_ms,
    )


def _load_corpus(
    snapshot: Any, text: str
) -> tuple[np.ndarray, dict[str, Any], dict[str, str], dict[str, str]]:
    """Embed the idea and load note metadata + passage texts (one store pass)."""

    query = np.asarray(embed_queries([text])[0], dtype=np.float32).ravel()
    norm = float(np.linalg.norm(query))
    if norm > 0.0:
        query = query / norm
    if len(snapshot.passage_ids):
        scores = np.asarray(snapshot.passage_matrix @ query, dtype=np.float32)
    else:
        scores = np.zeros(0, dtype=np.float32)
    store = snapshot.open_knowledge_store()
    try:
        notes = {row.note_id: row for row in store.source_notes()}
        passage_note: dict[str, str] = {}
        passage_text: dict[str, str] = {}
        ids = list(snapshot.passage_ids)
        for start in range(0, len(ids), 500):  # stay under SQL variable limits
            for context in store.passage_contexts(ids[start : start + 500]):
                passage_note[context.passage_id] = context.note_id
                passage_text[context.passage_id] = context.text
    finally:
        store.close()
    return scores, notes, passage_note, passage_text


def _bm25_notes(
    snapshot: Any, text: str, passage_note: Mapping[str, str]
) -> tuple[str, ...]:
    store = snapshot.open_knowledge_store()
    try:
        hits = store.search_passages_bm25(text, k=BM25_SEED_K)
    finally:
        store.close()
    ordered: list[str] = []
    for hit in hits:
        note_id = passage_note.get(hit.passage_id)
        if note_id is not None and note_id not in ordered:
            ordered.append(note_id)
    return tuple(ordered)


def _projection_adjacency(
    projection: Sequence[Any],
) -> dict[str, list[tuple[float, str]]]:
    adjacency: dict[str, list[tuple[float, str]]] = {}
    for edge in projection:
        weight = float(edge.weight)
        adjacency.setdefault(edge.src_note_id, []).append((weight, edge.dst_note_id))
        adjacency.setdefault(edge.dst_note_id, []).append((weight, edge.src_note_id))
    return adjacency


def _neighbors(
    note_ids: Sequence[str],
    adjacency: Mapping[str, list[tuple[float, str]]],
    known: Mapping[str, Any],
) -> list[str]:
    """Unvisited 1-hop neighbors, strongest edge first (deterministic)."""

    best: dict[str, float] = {}
    for note_id in note_ids:
        for weight, neighbor in adjacency.get(note_id, ()):
            if neighbor in known or neighbor in note_ids:
                continue
            best[neighbor] = max(best.get(neighbor, 0.0), weight)
    return sorted(best, key=lambda neighbor: (-best[neighbor], neighbor))


def _candidate(
    note_id: str,
    notes: Mapping[str, Any],
    best_passage: Mapping[str, tuple[float, str]],
    passage_text: Mapping[str, str],
    *,
    expanded: bool,
) -> _Candidate:
    note = notes[note_id]
    ranked = best_passage.get(note_id)
    snippet = passage_text.get(ranked[1], "") if ranked is not None else ""
    return _Candidate(
        note_id=note_id,
        title=note.title,
        source=note.source,
        display_line=note.display_line,
        snippet=" ".join(snippet.split())[:SNIPPET_MAX_CHARS] or note.display_line,
        expanded=expanded,
    )


async def _judge(
    idea: str,
    candidates: Sequence[_Candidate],
    llm: Any,
) -> dict[str, tuple[bool, float, str]]:
    """Adjudicate candidates in bounded concurrent batches; fail-open."""

    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def judge_batch(
        batch: Sequence[_Candidate],
    ) -> dict[str, tuple[bool, float, str]]:
        payload = {
            "idea": idea,
            "notes": [
                {
                    "note_id": row.note_id,
                    "title": row.title,
                    "source": row.source,
                    "display_line": row.display_line,
                    "snippet": row.snippet,
                }
                for row in batch
            ],
        }
        async with semaphore:
            try:
                response = await llm.complete_json(
                    config.IDEA_SEARCH_MODEL,
                    _JUDGE_SYSTEM,
                    json.dumps(payload, ensure_ascii=False),
                    timeout_s=config.SUMMARY_TIMEOUT_S,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001 - fail-open per batch
                _LOG.warning(
                    "idea adjudication batch failed (%s)", type(exc).__name__
                )
                return {}
        return _parse_judgements(response, {row.note_id for row in batch})

    batches = [
        candidates[start : start + JUDGE_BATCH]
        for start in range(0, len(candidates), JUDGE_BATCH)
    ]
    results: dict[str, tuple[bool, float, str]] = {}
    for parsed in await asyncio.gather(*(judge_batch(batch) for batch in batches)):
        results.update(parsed)
    return results


def _parse_judgements(
    response: Any, allowed_ids: set[str]
) -> dict[str, tuple[bool, float, str]]:
    rows = response.get("judgements") if isinstance(response, Mapping) else None
    if not isinstance(rows, list):
        return {}
    parsed: dict[str, tuple[bool, float, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue  # malformed row: skip the note, never the request
        note_id = row.get("note_id")
        if not isinstance(note_id, str) or note_id not in allowed_ids:
            continue
        aligned = row.get("aligned")
        if not isinstance(aligned, bool):
            continue
        try:
            strength = float(row.get("strength"))
        except (TypeError, ValueError):
            continue
        strength = min(1.0, max(0.0, strength))
        reason = row.get("reason")
        if not isinstance(reason, str):
            reason = ""
        parsed[note_id] = (aligned, strength, " ".join(reason.split())[:_REASON_MAX_CHARS])
    return parsed
