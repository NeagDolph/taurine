"""LLM cluster naming during sync, with deterministic fallbacks.

Labeling is presentation-only: cluster membership is deterministic build
output, names/descriptions never enter the graph hash, and a labeling
failure degrades to fallback names instead of failing the build. Payloads
carry only titles/display lines/entities — never full note bodies — matching
the privacy envelope of the relation-completion lane, which already sends
the same fields to the same OpenRouter model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from taurine_core import config

_LOG = logging.getLogger("taurine.build")

_NOTE_LINE_MAX_CHARS = 90
_TITLE_LIMIT = 10
_ENTITY_LIMIT = 8
_NAME_MAX_CHARS = 60
_DESCRIPTION_MAX_CHARS = 400

_SYSTEM_PROMPT = (
    "You name clusters of related personal notes. Return JSON "
    '{"clusters": [{"cluster_id": <int>, "name": "<name>", '
    '"description": "<1-2 sentences>"}]} with exactly one entry per input '
    "cluster, reusing each cluster_id verbatim. Names must be at most 5 "
    "words, title-case, and SPECIFIC to what unites the notes (good: "
    '"ShopLC Reranker Work", "Tokyo Trip Planning", "NLP Course Project"), '
    'never generic ("Miscellaneous Notes", "Various Topics"). The '
    "description is 1-2 sentences describing what unites these notes. If a "
    "cluster is genuinely incoherent, name it honestly (e.g. "
    '"Mixed Personal Notes") and say so in the description.'
)

_SOURCE_LABELS = {
    "imessage": "Group Chats",
    "gmail": "Email Threads",
    "outlook": "Email Threads",
    "apple_notes": "Apple Notes",
    "obsidian": "Obsidian Notes",
    "claude_mem": "Agent Memories",
    "codex_mem": "Agent Memories",
    "local_folders": "Documents",
}


@dataclass(frozen=True)
class ClusterSummary:
    """The bounded per-cluster context handed to the naming LLM."""

    cluster_id: int
    size: int
    top_entities: tuple[str, ...]
    note_lines: tuple[str, ...]
    source_mix: str
    dominant_source: str


def build_cluster_summaries(
    membership: Mapping[str, int],
    *,
    titles: Mapping[str, str],
    display_lines: Mapping[str, str],
    sources: Mapping[str, str],
    value_scores: Mapping[str, float],
    note_entities: Mapping[str, Sequence[str]],
) -> tuple[ClusterSummary, ...]:
    """Deterministic cluster summaries. ``note_entities`` lists one canonical
    label per traversal-eligible mention (repetitions carry the count)."""

    members: dict[int, list[str]] = {}
    for note_id in sorted(membership):
        cluster_id = int(membership[note_id])
        if cluster_id >= 0:
            members.setdefault(cluster_id, []).append(note_id)
    summaries: list[ClusterSummary] = []
    for cluster_id in sorted(members):
        note_ids = members[cluster_id]
        entity_counts: dict[str, int] = {}
        for note_id in note_ids:
            for label in note_entities.get(note_id, ()):
                entity_counts[label] = entity_counts.get(label, 0) + 1
        top_entities = tuple(
            sorted(entity_counts, key=lambda label: (-entity_counts[label], label))[
                :_ENTITY_LIMIT
            ]
        )
        ranked = sorted(
            note_ids, key=lambda note_id: (-value_scores.get(note_id, 0.0), note_id)
        )[:_TITLE_LIMIT]
        note_lines = tuple(
            _note_line(titles.get(note_id, ""), display_lines.get(note_id, ""))
            for note_id in ranked
        )
        source_counts: dict[str, int] = {}
        for note_id in note_ids:
            source = sources.get(note_id, "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        dominant = min(
            source_counts, key=lambda source: (-source_counts[source], source)
        )
        if source_counts[dominant] / len(note_ids) >= 0.6:
            source_mix = f"mostly {dominant}"
        else:
            top_sources = sorted(
                source_counts, key=lambda source: (-source_counts[source], source)
            )[:3]
            source_mix = "mixed: " + ", ".join(top_sources)
        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                size=len(note_ids),
                top_entities=top_entities,
                note_lines=note_lines,
                source_mix=source_mix,
                dominant_source=dominant,
            )
        )
    return tuple(summaries)


def fallback_label(summary: ClusterSummary) -> tuple[str, str]:
    """Deterministic (name, description) when the LLM is absent or failed."""

    if summary.top_entities:
        head = ", ".join(summary.top_entities[:2])
        extra = len(summary.top_entities) - 2
        name = f"{head} +{extra}" if extra > 0 else head
    else:
        name = _SOURCE_LABELS.get(
            summary.dominant_source,
            summary.dominant_source.replace("_", " ").title() or "Notes",
        )
    return name[:_NAME_MAX_CHARS], f"{summary.size} related notes ({summary.source_mix})."


def cluster_labeler_for(llm: Any) -> Callable[[Sequence[ClusterSummary]], Any]:
    """Bounded LLM cluster-naming callable shared by the server and CLI."""

    def label_batch(summaries: Sequence[ClusterSummary]) -> Any:
        payload = {
            "clusters": [
                {
                    "cluster_id": summary.cluster_id,
                    "size": summary.size,
                    "top_entities": list(summary.top_entities),
                    "notes": list(summary.note_lines),
                    "source_mix": summary.source_mix,
                }
                for summary in summaries
            ]
        }
        return asyncio.run(
            llm.complete_json(
                config.CLUSTER_LABEL_MODEL,
                _SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                timeout_s=config.SUMMARY_TIMEOUT_S,
                temperature=0.0,
            )
        )

    return label_batch


def label_clusters(
    summaries: Sequence[ClusterSummary],
    labeler: Callable[[Sequence[ClusterSummary]], Any],
) -> tuple[dict[int, tuple[str, str]], int]:
    """Name clusters in bounded batches; fail-open per batch.

    Returns ({cluster_id: (name, description)}, failed_batch_count). A batch
    that raises or returns malformed rows only loses its own labels.
    """
    labels: dict[int, tuple[str, str]] = {}
    errors = 0
    batch_size = max(1, int(config.CLUSTER_LABEL_BATCH))
    for start in range(0, len(summaries), batch_size):
        batch = summaries[start : start + batch_size]
        batch_ids = {summary.cluster_id for summary in batch}
        try:
            response = labeler(batch)
        except Exception as exc:  # noqa: BLE001 - labeling must never fail a build
            _LOG.warning(
                "cluster labeling batch failed (%s): %s", type(exc).__name__, exc
            )
            errors += 1
            continue
        rows = response.get("clusters") if isinstance(response, Mapping) else None
        if not isinstance(rows, list):
            _LOG.warning("cluster labeling batch returned a malformed response")
            errors += 1
            continue
        for row in rows:
            parsed = _parse_label_row(row, batch_ids)
            if parsed is not None:
                cluster_id, name, description = parsed
                labels[cluster_id] = (name, description)
    return labels, errors


def _parse_label_row(
    row: Any, allowed_ids: set[int]
) -> tuple[int, str, str] | None:
    if not isinstance(row, Mapping):
        return None
    try:
        cluster_id = int(row.get("cluster_id"))
    except (TypeError, ValueError):
        return None
    if cluster_id not in allowed_ids:
        return None
    name = row.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    description = row.get("description")
    if not isinstance(description, str):
        description = ""
    return (
        cluster_id,
        " ".join(name.split())[:_NAME_MAX_CHARS],
        " ".join(description.split())[:_DESCRIPTION_MAX_CHARS],
    )


def _note_line(title: str, display_line: str) -> str:
    title = title.strip()
    display_line = display_line.strip()
    if title and display_line and not display_line.startswith(title):
        line = f"{title} — {display_line}"
    else:
        line = title or display_line
    return line[:_NOTE_LINE_MAX_CHARS]
