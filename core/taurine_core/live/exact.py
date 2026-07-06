from __future__ import annotations

import re
from dataclasses import dataclass

from taurine_core import config
from taurine_core.ports import IndexPort
from taurine_core.schemas import Chunk


C1Hit = tuple[str, str, bool]


@dataclass(frozen=True)
class _Alias:
    alias: str
    note_id: str
    is_title: bool
    needs_boundary: bool
    pattern: re.Pattern[str] | None


class ExactCollector:
    def __init__(self, index: IndexPort) -> None:
        self._aliases = [self._compile_alias(row) for row in index.aliases()]
        self._aliases = [alias for alias in self._aliases if alias is not None]

    def on_chunk(self, chunk: Chunk) -> list[C1Hit]:
        normalized_text = normalize(chunk.text)
        hits: list[C1Hit] = []

        for alias in self._aliases:
            if alias.needs_boundary:
                assert alias.pattern is not None
                matched = alias.pattern.search(normalized_text) is not None
            else:
                matched = alias.alias in normalized_text

            if matched:
                hits.append((alias.note_id, alias.alias, alias.is_title))

        return hits

    def _compile_alias(
        self,
        row: tuple[str, str, bool],
    ) -> _Alias | None:
        raw_alias, note_id, is_title = row
        alias = normalize(raw_alias)
        if not alias:
            return None

        compact_len = len(alias.replace(" ", ""))
        needs_boundary = compact_len < config.ALIAS_WORD_BOUNDARY_UNDER
        pattern = (
            re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            if needs_boundary
            else None
        )
        return _Alias(alias, note_id, is_title, needs_boundary, pattern)


def normalize(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()
