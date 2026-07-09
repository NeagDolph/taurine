from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from taurine_core import config
from taurine_core.indexer.store import Index
from taurine_core.live.ambient import HeadFusionResult, run_head_fusion
from taurine_core.live.exact import ExactCollector
from taurine_core.live.grounding import GroundingEngine, GroundingUpdate
from taurine_core.schemas import Chunk


DECISION_RULE = (
    "ELIMINATE if utterance-only fraction < 5% AND "
    "mean top-5 Jaccard >= 0.9 else KEEP"
)
OTHER_INTRODUCING_HEADS = {"bm25", "exact", "gliner", "fact"}


@dataclass(frozen=True)
class ReplayChunk:
    seq: int
    channel: int
    text: str
    ts: float


@dataclass(frozen=True)
class ReplayWindow:
    session: str
    index: int
    seqs: list[int]
    text: str
    now: float


@dataclass(frozen=True)
class ConditionWindow:
    update: GroundingUpdate
    heads: HeadFusionResult
    exact_hits: list[tuple[str, str]]


def run_ablation(
    db_path: str | Path,
    sessions_dir: str | Path,
    *,
    limit: int | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Any]:
    db = Path(db_path)
    sessions = Path(sessions_dir)
    baseline = Index(db, read_only=True)
    masked = Index(db, read_only=True)
    try:
        masked.mask_embedding_kind("utterance")
        report = _run_ablation_with_indexes(baseline, masked, db, sessions, limit=limit)
    finally:
        baseline.close()
        masked.close()

    if json_path is not None:
        Path(json_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def decision_verdict(utterance_only_fraction: float, mean_top5_jaccard: float) -> str:
    if utterance_only_fraction < 0.05 and mean_top5_jaccard >= 0.9:
        return "ELIMINATE"
    return "KEEP"


def render_markdown(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    lines = [
        "# Utterance Ablation Replay",
        "",
        f"- DB: `{report['db']}`",
        f"- Sessions: `{report['sessions_dir']}`",
        f"- Windows: {corpus['windows']}",
        f"- Utterance-only grounding fraction: {corpus['utterance_only_fraction']:.3%}",
        f"- Mean top-5 grounding Jaccard: {corpus['mean_top5_jaccard']:.3f}",
        f"- Decision rule: {report['decision_rule']}",
        f"- Verdict: **{report['verdict']}**",
        "",
        "## Per-Session Diffs",
    ]
    diff_lines = report.get("diff_lines") or []
    if diff_lines:
        lines.extend(f"- {line}" for line in diff_lines)
    else:
        lines.append("- No replay windows found.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay stored sessions with and without utterance embeddings.",
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--sessions", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_ablation(
        args.db,
        args.sessions,
        limit=args.limit,
        json_path=args.json_path,
    )
    print(render_markdown(report))
    return 0


def _run_ablation_with_indexes(
    baseline: Index,
    masked: Index,
    db_path: Path,
    sessions_dir: Path,
    *,
    limit: int | None,
) -> dict[str, Any]:
    session_reports: list[dict[str, Any]] = []
    diff_lines: list[str] = []
    corpus_totals = _empty_totals()

    for session_path in _session_paths(sessions_dir, limit):
        windows = build_replay_windows(session_path)
        session_report = _run_session(session_path.name, windows, baseline, masked)
        session_reports.append(session_report)
        _add_totals(corpus_totals, session_report["totals"])
        diff_lines.extend(session_report["diff_lines"])

    corpus = _metrics_from_totals(corpus_totals)
    verdict = decision_verdict(
        corpus["utterance_only_fraction"],
        corpus["mean_top5_jaccard"],
    )
    return {
        "db": str(db_path),
        "sessions_dir": str(sessions_dir),
        "decision_rule": DECISION_RULE,
        "verdict": verdict,
        "corpus": corpus,
        "diff_lines": diff_lines,
        "sessions": session_reports,
    }


def build_replay_windows(session_path: Path) -> list[dict[str, Any]]:
    """Build deterministic C2 replay windows from final transcript chunks only.

    LiveSession uses the last ``config.C2_WINDOW_CHUNKS`` final chunks plus
    active partials when AmbientRetriever is allowed through the 8s throttle.
    Session replay has no active partial state, so each final chunk emits one
    rolling window over the last N final chunks in file order.
    """
    chunks = _read_final_chunks(session_path)
    windows: list[dict[str, Any]] = []
    for index in range(len(chunks)):
        group = chunks[max(0, index + 1 - config.C2_WINDOW_CHUNKS) : index + 1]
        text = " ".join(chunk.text for chunk in group if chunk.text).strip()
        if not text:
            continue
        windows.append(
            {
                "session": session_path.name,
                "index": len(windows) + 1,
                "seqs": [chunk.seq for chunk in group],
                "text": text,
                "now": group[-1].ts,
            }
        )
    return windows


def _run_session(
    session_name: str,
    window_rows: list[dict[str, Any]],
    baseline: Index,
    masked: Index,
) -> dict[str, Any]:
    baseline_runner = _ConditionRunner(baseline)
    masked_runner = _ConditionRunner(masked)
    totals = _empty_totals()
    windows: list[dict[str, Any]] = []
    changed = 0

    for row in window_rows:
        window = ReplayWindow(
            session=session_name,
            index=int(row["index"]),
            seqs=list(row["seqs"]),
            text=str(row["text"]),
            now=float(row["now"]),
        )
        baseline_result = baseline_runner.run(window)
        masked_result = masked_runner.run(window)
        baseline_top5 = _top_ids(baseline_result.update, 5)
        masked_top5 = _top_ids(masked_result.update, 5)
        if baseline_top5 != masked_top5:
            changed += 1
        utterance_only_ids = _utterance_only_ids(
            baseline_result.update,
            baseline_result.heads,
        )
        jaccard = _jaccard(baseline_top5, masked_top5)
        window_record = {
            "session": window.session,
            "index": window.index,
            "seqs": window.seqs,
            "text": window.text,
            "baseline_top5": baseline_top5,
            "masked_top5": masked_top5,
            "top5_jaccard": jaccard,
            "utterance_only_ids": utterance_only_ids,
            "fused": _fused_rows(baseline_result.heads),
        }
        windows.append(window_record)
        totals["windows"] += 1
        totals["grounding_items"] += len(baseline_result.update.items)
        totals["utterance_only_items"] += len(utterance_only_ids)
        totals["jaccard_sum"] += jaccard

    metrics = _metrics_from_totals(totals)
    diff_line = (
        f"{session_name}: {metrics['windows']} windows, "
        f"utterance-only {totals['utterance_only_items']}/{totals['grounding_items']} "
        f"({metrics['utterance_only_fraction']:.1%}), "
        f"mean top-5 Jaccard {metrics['mean_top5_jaccard']:.3f}, "
        f"changed {changed}/{metrics['windows']}"
    )
    return {
        "session": session_name,
        "metrics": metrics,
        "totals": totals,
        "diff_lines": [diff_line] if metrics["windows"] else [],
        "windows": windows,
    }


class _ConditionRunner:
    def __init__(self, index: Index) -> None:
        self._index = index
        self._exact = ExactCollector(index)
        self._grounding = GroundingEngine(index)

    def run(self, window: ReplayWindow) -> ConditionWindow:
        exact_hits = self._exact_hits(window.text, window.now)
        heads = run_head_fusion(self._index, window.text, exact_hits=exact_hits)
        update = self._grounding.on_pass(
            window.text,
            heads.fused,
            exact_hits,
            window.now,
            head_hits=heads.head_hits,
        )
        return ConditionWindow(update=update, heads=heads, exact_hits=exact_hits)

    def _exact_hits(self, window_text: str, now: float) -> list[tuple[str, str]]:
        chunk = Chunk(channel=0, text=window_text, ts=now, final=True)
        hits: list[tuple[str, str]] = []
        seen: set[str] = set()
        for note_id, alias, _is_title in self._exact.on_chunk(chunk):
            if note_id in seen:
                continue
            seen.add(note_id)
            hits.append((note_id, alias))
        return hits


def _session_paths(sessions_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(path for path in sessions_dir.glob("*.jsonl") if path.is_file())
    if limit is not None:
        return paths[: max(0, int(limit))]
    return paths


def _read_final_chunks(session_path: Path) -> list[ReplayChunk]:
    chunks: list[ReplayChunk] = []
    with session_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "chunk" or not bool(event.get("final")):
                continue
            text = str(event.get("text") or "").strip()
            if not text:
                continue
            seq = int(event.get("seq") or len(chunks) + 1)
            channel = int(event.get("channel") or 0)
            ts = float(event.get("ts") or seq)
            chunks.append(ReplayChunk(seq=seq, channel=channel, text=text, ts=ts))
    return chunks


def _utterance_only_ids(
    update: GroundingUpdate,
    heads: HeadFusionResult,
) -> list[str]:
    ids: list[str] = []
    for item in update.items:
        dense_kinds = set(heads.dense_kinds.get(item.note_id, set()))
        introducing_heads = set(heads.head_hits.get(item.note_id, set()))
        why_heads = set(item.why.heads or [])
        if dense_kinds == {"utterance"} and not (
            (introducing_heads | why_heads) & OTHER_INTRODUCING_HEADS
        ):
            ids.append(item.note_id)
    return ids


def _fused_rows(heads: HeadFusionResult) -> list[dict[str, Any]]:
    return [
        {
            "note_id": note_id,
            "score": score,
            "heads": sorted(heads.head_hits.get(note_id, set())),
            "dense_kinds": sorted(heads.dense_kinds.get(note_id, set())),
            "dense_gliner_kinds": sorted(
                heads.dense_gliner_kinds.get(note_id, set())
            ),
        }
        for note_id, score in heads.fused
    ]


def _top_ids(update: GroundingUpdate, limit: int) -> list[str]:
    return [item.note_id for item in update.items[:limit]]


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _empty_totals() -> dict[str, float | int]:
    return {
        "windows": 0,
        "grounding_items": 0,
        "utterance_only_items": 0,
        "jaccard_sum": 0.0,
    }


def _add_totals(target: dict[str, float | int], source: dict[str, float | int]) -> None:
    for key in target:
        target[key] += source[key]


def _metrics_from_totals(totals: dict[str, float | int]) -> dict[str, Any]:
    windows = int(totals["windows"])
    grounding_items = int(totals["grounding_items"])
    utterance_only_items = int(totals["utterance_only_items"])
    return {
        "windows": windows,
        "grounding_items": grounding_items,
        "utterance_only_items": utterance_only_items,
        "utterance_only_fraction": (
            utterance_only_items / grounding_items if grounding_items else 0.0
        ),
        "mean_top5_jaccard": (
            float(totals["jaccard_sum"]) / windows if windows else 1.0
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
