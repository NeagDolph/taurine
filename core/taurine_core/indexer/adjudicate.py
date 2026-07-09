from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from taurine_core import config
from taurine_core.ports import LLMError, LLMPort
from taurine_core.schemas import NoteRecord

from . import graph


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class AdjudicationReport:
    judged: int = 0
    admitted: int = 0
    vetoed: int = 0
    facts: int = 0
    errors: int = 0


@dataclass(frozen=True)
class _PendingPair:
    pair_id: str
    src: str
    dst: str
    sig: str
    band: str
    marker: str
    candidate: graph._Candidate


@dataclass(frozen=True)
class _BatchReport:
    judged: int = 0
    admitted: int = 0
    vetoed: int = 0
    facts: int = 0
    errors: int = 0
    done: int = 0


async def adjudicate_graph(
    index: object,
    llm: LLMPort,
    progress: ProgressCallback | None = None,
) -> AdjudicationReport:
    pending = _pending_pairs(index)
    total = len(pending)
    if total == 0:
        return AdjudicationReport()

    semaphore = asyncio.Semaphore(config.ADJUDICATE_CONCURRENCY)
    batches = [
        pending[offset : offset + config.ADJUDICATE_BATCH_PAIRS]
        for offset in range(0, total, config.ADJUDICATE_BATCH_PAIRS)
    ]
    tasks = [
        asyncio.create_task(_adjudicate_batch(index, llm, batch, semaphore))
        for batch in batches
    ]

    done = 0
    judged = admitted = vetoed = facts = errors = 0
    for task in asyncio.as_completed(tasks):
        batch_report = await task
        done += batch_report.done
        judged += batch_report.judged
        admitted += batch_report.admitted
        vetoed += batch_report.vetoed
        facts += batch_report.facts
        errors += batch_report.errors
        if progress is not None:
            try:
                progress(done, total)
            except Exception:
                pass

    active_ids = {
        note.note_id
        for note in index.notes()  # type: ignore[attr-defined]
    }
    index.prune_edge_judgments(active_ids)  # type: ignore[attr-defined]
    graph.rebuild_all(index)
    from .rank import compute_quality_ranks, refresh_fact_embeddings

    compute_quality_ranks(index)  # type: ignore[arg-type]
    refresh_fact_embeddings(index)  # type: ignore[arg-type]
    index.load()  # type: ignore[attr-defined]

    return AdjudicationReport(
        judged=judged,
        admitted=admitted,
        vetoed=vetoed,
        facts=facts,
        errors=errors,
    )


def _pending_pairs(index: object) -> list[_PendingPair]:
    kept, margin = graph._candidates(index)
    notes = {
        note.note_id: note
        for note in index.notes()  # type: ignore[attr-defined]
    }
    active_ids = _active_note_ids(index, notes)
    values = _active_value_scores(index, active_ids)
    value_threshold = _value_threshold(values, active_ids)
    communities = _note_communities(index)
    note_hashes = {note_id: note.content_hash for note_id, note in notes.items()}
    judgments = index.edge_judgments()  # type: ignore[attr-defined]
    # bands only cover kept candidates that survive the degree cap — pre-cap
    # candidates can never appear in the graph, so judging them burns LLM
    # calls on invisible pairs (9,356 pre-cap vs ~2,400 post-cap on the
    # 2026-07-08 corpus)
    capped_keys = {
        (edge.src, edge.dst)
        for edge in graph._cap_edges([candidate.edge() for candidate in kept])
    }
    capped = [
        candidate for candidate in kept if (candidate.src, candidate.dst) in capped_keys
    ]
    strong_fact_keys = _fact_eligible_strong_keys(capped, values, value_threshold)
    pending: list[tuple[float, bool, str, graph._Candidate]] = []

    def add_pending(band: str, candidate: graph._Candidate, *, gated: bool) -> None:
        if graph.promoted_relation_fact(index, candidate.src, candidate.dst) is not None:
            return
        if _has_valid_judgment(candidate, judgments, note_hashes):
            return
        bridge = _is_bridge(candidate, communities)
        if gated and not bridge and not _pair_clears_value(candidate, values, value_threshold):
            return
        pending.append((
            _rank_score(candidate, values, bridge),
            bridge,
            band,
            candidate,
        ))

    for candidate in capped:
        key = (candidate.src, candidate.dst)
        edge = candidate.edge()
        if edge.weight >= config.VETO_MAX_W:
            if key not in strong_fact_keys:
                continue
            add_pending("strong", candidate, gated=False)
        else:
            add_pending("weak", candidate, gated=True)

    for candidate in margin:
        add_pending("margin", candidate, gated=True)

    pending.sort(
        key=lambda item: (-item[0], item[2], item[3].src, item[3].dst)
    )
    pending = pending[: max(0, int(config.ADJUDICATE_MAX_PAIRS_PER_SYNC))]
    return [
        _PendingPair(
            pair_id=f"pair-{index_ + 1}",
            src=candidate.src,
            dst=candidate.dst,
            sig=graph.pair_sig(index, candidate.src, candidate.dst, note_hashes),
            band=band,
            marker="[KEEP]" if band == "strong" else "[JUDGE]",
            candidate=candidate,
        )
        for index_, (_score, _bridge, band, candidate) in enumerate(pending)
    ]


def _fact_eligible_strong_keys(
    kept: Sequence[graph._Candidate],
    values: Mapping[str, float],
    value_threshold: float,
) -> set[tuple[str, str]]:
    counts: dict[str, int] = defaultdict(int)
    selected: set[tuple[str, str]] = set()
    strong = [
        candidate
        for candidate in kept
        if candidate.edge().weight >= config.VETO_MAX_W
    ]
    strong.sort(key=lambda candidate: (-candidate.edge().weight, candidate.src, candidate.dst))
    for candidate in strong:
        if not _pair_clears_value(candidate, values, value_threshold):
            continue
        if counts[candidate.src] >= config.FACT_TOP_EDGES_PER_NOTE:
            continue
        if counts[candidate.dst] >= config.FACT_TOP_EDGES_PER_NOTE:
            continue
        selected.add((candidate.src, candidate.dst))
        counts[candidate.src] += 1
        counts[candidate.dst] += 1
    return selected


def _active_note_ids(
    index: object,
    notes: Mapping[str, NoteRecord],
) -> set[str]:
    disabled_sources_fn = getattr(index, "disabled_sources", None)
    disabled_sources = (
        disabled_sources_fn() if callable(disabled_sources_fn) else set()
    )
    return {
        note_id
        for note_id, note in notes.items()
        if note.source not in disabled_sources
    }


def _active_value_scores(index: object, active_ids: set[str]) -> dict[str, float]:
    value_scores_fn = getattr(index, "value_scores", None)
    raw_scores = value_scores_fn() if callable(value_scores_fn) else {}
    return {
        note_id: float(raw_scores.get(note_id, 0.0))
        for note_id in active_ids
    }


def _value_threshold(
    values: Mapping[str, float],
    active_ids: set[str],
) -> float:
    if not active_ids:
        return 0.0
    scores = np.asarray(
        [float(values.get(note_id, 0.0)) for note_id in sorted(active_ids)],
        dtype=np.float64,
    )
    if scores.size == 0 or float(np.max(scores)) <= 0.0:
        return 0.0
    fraction = min(max(float(config.ADJ_VALUE_FRACTION), 0.0), 1.0)
    return float(np.quantile(scores, 1.0 - fraction))


def _clears_value_threshold(value: float, threshold: float) -> bool:
    if threshold <= 0.0:
        return value > 0.0
    return value >= threshold


def _pair_clears_value(
    candidate: graph._Candidate,
    values: Mapping[str, float],
    value_threshold: float,
) -> bool:
    return (
        _clears_value_threshold(float(values.get(candidate.src, 0.0)), value_threshold)
        or _clears_value_threshold(float(values.get(candidate.dst, 0.0)), value_threshold)
    )


def _note_communities(index: object) -> dict[str, int]:
    from .communities import label_propagation
    from .rank import NoteGraphMatrix

    return label_propagation(NoteGraphMatrix.build(index))


def _is_bridge(
    candidate: graph._Candidate,
    communities: Mapping[str, int],
) -> bool:
    src_label = communities.get(candidate.src)
    dst_label = communities.get(candidate.dst)
    return src_label is not None and dst_label is not None and src_label != dst_label


def _rank_score(
    candidate: graph._Candidate,
    values: Mapping[str, float],
    bridge: bool,
) -> float:
    score = float(values.get(candidate.src, 0.0)) + float(values.get(candidate.dst, 0.0))
    if bridge:
        score *= float(config.BRIDGE_RANK_BOOST)
    return score


def _has_valid_judgment(
    candidate: graph._Candidate,
    judgments: Mapping[tuple[str, str], tuple[str, bool, str]],
    note_hashes: Mapping[str, str],
) -> bool:
    judgment = judgments.get((candidate.src, candidate.dst))
    if judgment is None:
        return False
    sig, _related, _fact = judgment
    return sig == graph.pair_sig(None, candidate.src, candidate.dst, note_hashes)


async def _adjudicate_batch(
    index: object,
    llm: LLMPort,
    batch: Sequence[_PendingPair],
    semaphore: asyncio.Semaphore,
) -> _BatchReport:
    async with semaphore:
        response: dict | None = None
        for attempt in range(2):
            try:
                response = await llm.complete_json(
                    config.ADJUDICATE_MODEL,
                    _system_prompt(),
                    _user_prompt(index, batch),
                    timeout_s=config.ADJUDICATE_TIMEOUT_S,
                )
                break
            except LLMError:
                if attempt == 1:
                    return _BatchReport(errors=1, done=len(batch))
        if response is None:
            return _BatchReport(errors=1, done=len(batch))

    return _write_response(index, batch, response)


def _write_response(
    index: object,
    batch: Sequence[_PendingPair],
    response: dict,
) -> _BatchReport:
    response_pairs = response.get("pairs") if isinstance(response, dict) else None
    if not isinstance(response_pairs, dict):
        return _BatchReport(done=len(batch))

    judged = admitted = vetoed = facts = 0
    for pair in batch:
        item = response_pairs.get(pair.pair_id)
        if not isinstance(item, dict):
            continue
        related = bool(item.get("related"))
        fact = _clamp_fact(item.get("fact")) if related else ""
        index.set_edge_judgment(  # type: ignore[attr-defined]
            pair.src,
            pair.dst,
            pair.sig,
            related,
            fact,
        )
        judged += 1
        if related and pair.band == "margin":
            admitted += 1
        if not related and pair.band in {"weak", "margin"}:
            vetoed += 1
        if related and fact:
            facts += 1
    return _BatchReport(
        judged=judged,
        admitted=admitted,
        vetoed=vetoed,
        facts=facts,
        done=len(batch),
    )


def _system_prompt() -> str:
    return (
        "Decide whether two personal notes are genuinely related. When related, "
        "state THE relationship in one specific sentence naming the people or "
        "projects involved; do not write that the notes are similar. Return "
        'strict JSON shaped as {"pairs": {"<pair_id>": {"related": bool, '
        '"fact": str}}}. For pairs marked [KEEP], the connection already '
        "exists and only the fact is wanted; treat related as true unless the "
        "pair is marked [JUDGE]."
    )


def _user_prompt(index: object, batch: Sequence[_PendingPair]) -> str:
    notes = {
        note.note_id: note
        for note in index.notes()  # type: ignore[attr-defined]
    }
    user_card = index.user_card() or ""  # type: ignore[attr-defined]
    sections = [
        "USER CARD",
        "```",
        user_card,
        "```",
        "",
        "PAIRS",
    ]
    for pair in batch:
        left = notes[pair.src]
        right = notes[pair.dst]
        sections.extend(_pair_section(pair, left, right))
    return "\n".join(sections)


def _pair_section(pair: _PendingPair, left: NoteRecord, right: NoteRecord) -> list[str]:
    shared = ", ".join(pair.candidate.edge().shared) or "(none)"
    cosine = (
        f"{pair.candidate.semantic_score:.3f}"
        if pair.candidate.semantic_score is not None
        else "(unavailable)"
    )
    return [
        "",
        f"Pair ID: {pair.pair_id}",
        f"Band: {pair.marker} {pair.band}",
        f"Left note id: {left.note_id}",
        f"Left title: {left.title}",
        f"Left text: {left.text[:config.ADJ_TEXT_CAP]}",
        f"Right note id: {right.note_id}",
        f"Right title: {right.title}",
        f"Right text: {right.text[:config.ADJ_TEXT_CAP]}",
        f"Shared canonical terms: {shared}",
        f"Semantic cosine: {cosine}",
    ]


def _clamp_fact(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    sentence_end = re.search(r"(?<=[.!?])\s+", text)
    if sentence_end is not None:
        text = text[: sentence_end.start() + 1]
    if len(text) > 300:
        text = text[:300].rstrip()
    return text
