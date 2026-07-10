"""Canonical registry payload shapes shared by the engine and chat tools.

The evidence registry rejects re-registration under the same identity with a
different payload, so every producer of a passage or grouped-fact registration
must build byte-identical payloads. These helpers are the single source of
those shapes (audit finding C2).
"""

from __future__ import annotations

from typing import Any, Sequence


# Append-only provenance fields on a grouped fact registration; everything
# else in the payload is immutable for the identity's lifetime.
FACT_APPEND_FIELDS: frozenset[str] = frozenset({"fact_ids", "source_evidence_ids"})


def passage_payload(passage: Any) -> dict[str, object]:
    """The engine's canonical 5-key passage payload."""

    return {
        "passage_id": str(passage.passage_id),
        "note_id": str(passage.note_id),
        "text": str(passage.text),
        "char_start": int(passage.char_start),
        "char_end": int(passage.char_end),
    }


def fact_source_identity(
    build_id: str,
    logical_key: Sequence[str],
    status: str,
    valid_from: object,
    valid_to: object,
) -> str:
    """One registry identity per logical fact variant within a build."""

    return "\x1f".join(
        (build_id, *logical_key, status, repr(valid_from), repr(valid_to))
    )


def fact_group_payload(
    logical_key: Sequence[str],
    status: str,
    valid_from: object,
    valid_to: object,
    contexts: Sequence[Any],
) -> dict[str, object]:
    """The engine's canonical grouped-fact core payload.

    ``contexts`` must be the variant's provenance rows ordered by fact_id;
    the first row supplies the canonical endpoints and ``observed_at`` is the
    maximum across all provenance.
    """

    first = contexts[0]
    return {
        "logical_key": list(logical_key),
        "subject_entity_id": str(first.subject_entity_id),
        "object_entity_id": str(first.object_entity_id),
        "subject": str(first.subject),
        "predicate": str(first.predicate),
        "object": str(first.object),
        "direction": "forward",
        "status": status,
        "observed_at": max(float(row.observed_at) for row in contexts),
        "valid_from": valid_from,
        "valid_to": valid_to,
    }
