"""Deterministic cluster naming for build-time presentation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

_NOTE_LINE_MAX_CHARS = 90
_TITLE_LIMIT = 10
_ENTITY_LIMIT = 8
_NAME_MAX_CHARS = 60

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
    """Bounded per-cluster context used for deterministic naming."""

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
    """Deterministic cluster (name, description)."""

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


def _note_line(title: str, display_line: str) -> str:
    title = title.strip()
    display_line = display_line.strip()
    if title and display_line and not display_line.startswith(title):
        line = f"{title} — {display_line}"
    else:
        line = title or display_line
    return line[:_NOTE_LINE_MAX_CHARS]
