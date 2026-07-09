"""Deterministic token-aware source passage construction."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, TypedDict

from taurine_core import config
from taurine_core.schemas import NoteRecord


class TokenizerResult(TypedDict):
    input_ids: Sequence[int]
    offset_mapping: Sequence[tuple[int, int]]


class TokenizerPort(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> TokenizerResult: ...


@dataclass(frozen=True)
class PassageDraft:
    passage_id: str
    ordinal: int
    char_start: int
    char_end: int
    token_count: int
    text: str
    content_hash: str


def build_passages(
    note: NoteRecord,
    tokenizer: TokenizerPort,
    *,
    version: str,
) -> tuple[PassageDraft, ...]:
    """Split a note into overlapping source-exact passage spans."""
    if not note.text.strip():
        return ()

    encoded = tokenizer(
        note.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = tuple(
        (int(start), int(end))
        for start, end in encoded["offset_mapping"]
        if int(end) > int(start)
    )
    if not offsets:
        return ()

    target = max(1, int(config.PASSAGE_TARGET_TOKENS))
    maximum = max(target, int(config.PASSAGE_MAX_TOKENS))
    overlap = max(0, min(int(config.PASSAGE_OVERLAP_TOKENS), target - 1))
    boundaries = _boundary_token_indices(note.text, offsets)

    rows: list[PassageDraft] = []
    start = 0
    while start < len(offsets):
        end = _window_end(
            start,
            len(offsets),
            target=target,
            maximum=maximum,
            boundaries=boundaries,
        )
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        passage_text = note.text[char_start:char_end]
        if passage_text.strip():
            rows.append(
                PassageDraft(
                    passage_id=_passage_id(
                        note.note_id,
                        note.content_hash,
                        char_start,
                        char_end,
                        version,
                    ),
                    ordinal=len(rows),
                    char_start=char_start,
                    char_end=char_end,
                    token_count=end - start,
                    text=passage_text,
                    content_hash=_sha256(passage_text),
                )
            )
        if end >= len(offsets):
            break
        start = max(start + 1, end - overlap)
    return tuple(rows)


def _window_end(
    start: int,
    total: int,
    *,
    target: int,
    maximum: int,
    boundaries: dict[str, tuple[int, ...]],
) -> int:
    hard_end = min(total, start + maximum)
    if hard_end >= total:
        return total
    target_end = min(hard_end, start + target)
    minimum = min(target_end, start + max(1, math.ceil(target * 0.6)))
    for kind in ("paragraph", "sentence", "clause"):
        candidates = [
            boundary
            for boundary in boundaries[kind]
            if minimum <= boundary <= hard_end
        ]
        if candidates:
            return min(candidates, key=lambda boundary: (abs(boundary - target_end), boundary))
    return target_end


def _boundary_token_indices(
    text: str,
    offsets: tuple[tuple[int, int], ...],
) -> dict[str, tuple[int, ...]]:
    paragraph_chars = [match.start() for match in re.finditer(r"\n\s*\n", text)]
    sentence_chars = [
        match.start() + len(match.group(1))
        for match in re.finditer(r"([.!?][\"')\]]*)(?:\s+|$)", text)
    ]
    clause_chars = [
        match.start() + 1
        for match in re.finditer(r"[,;:]\s+|\n", text)
    ]
    return {
        "paragraph": _char_boundaries_to_token_indices(paragraph_chars, offsets),
        "sentence": _char_boundaries_to_token_indices(sentence_chars, offsets),
        "clause": _char_boundaries_to_token_indices(clause_chars, offsets),
    }


def _char_boundaries_to_token_indices(
    char_boundaries: list[int],
    offsets: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    result: set[int] = set()
    token_index = 0
    for boundary in sorted(char_boundaries):
        while token_index < len(offsets) and offsets[token_index][1] <= boundary:
            token_index += 1
        if token_index > 0:
            result.add(token_index)
    return tuple(sorted(result))


def _passage_id(
    note_id: str,
    note_content_hash: str,
    char_start: int,
    char_end: int,
    version: str,
) -> str:
    identity = "\x1f".join(
        (note_id, note_content_hash, str(char_start), str(char_end), version)
    )
    return f"passage-{_sha256(identity)[:32]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
