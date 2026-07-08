from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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
    strong_fact_keys = _fact_eligible_strong_keys(capped)
    pending: list[tuple[str, graph._Candidate]] = []

    for candidate in capped:
        key = (candidate.src, candidate.dst)
        edge = candidate.edge()
        if edge.weight >= config.VETO_MAX_W:
            if key not in strong_fact_keys:
                continue
            band = "strong"
        else:
            band = "weak"
        if _has_valid_judgment(candidate, judgments, note_hashes):
            continue
        pending.append((band, candidate))

    for candidate in margin:
        if _has_valid_judgment(candidate, judgments, note_hashes):
            continue
        pending.append(("margin", candidate))

    pending.sort(
        key=lambda item: (-item[1].edge().weight, item[0], item[1].src, item[1].dst)
    )
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
        for index_, (band, candidate) in enumerate(pending)
    ]


def _fact_eligible_strong_keys(
    kept: Sequence[graph._Candidate],
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
        if counts[candidate.src] >= config.FACT_TOP_EDGES_PER_NOTE:
            continue
        if counts[candidate.dst] >= config.FACT_TOP_EDGES_PER_NOTE:
            continue
        selected.add((candidate.src, candidate.dst))
        counts[candidate.src] += 1
        counts[candidate.dst] += 1
    return selected


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
        f"Left gist: {left.gist}",
        f"Right note id: {right.note_id}",
        f"Right title: {right.title}",
        f"Right gist: {right.gist}",
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
