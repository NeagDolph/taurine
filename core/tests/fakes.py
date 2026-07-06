from __future__ import annotations

from dataclasses import dataclass, field

from taurine_core.schemas import GraphResponse, NoteRecord


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@dataclass
class FakeIndex:
    notes: dict[str, NoteRecord] = field(default_factory=dict)
    dense: list[tuple[str, float]] = field(default_factory=list)
    bm25: list[tuple[str, float]] = field(default_factory=list)
    links: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    alias_rows: list[tuple[str, str, bool]] = field(default_factory=list)
    chunks: dict[str, str] = field(default_factory=dict)

    def dense_search(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.dense[:k]

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.bm25[:k]

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        return self.links.get(note_id, [])

    def note(self, note_id: str) -> NoteRecord | None:
        return self.notes.get(note_id)

    def best_chunk(self, note_id: str, query_text: str) -> str:
        return self.chunks.get(note_id, "")

    def aliases(self) -> list[tuple[str, str, bool]]:
        return self.alias_rows

    def graph(self) -> GraphResponse:
        return GraphResponse(nodes=[], edges=[])

    def status(self) -> tuple[int, float | None]:
        return len(self.notes), None


def note(note_id: str, title: str | None = None, gist: str | None = None) -> NoteRecord:
    title = title or f"Note {note_id}"
    return NoteRecord(
        note_id=note_id,
        title=title,
        modified=0.0,
        content_hash=f"hash-{note_id}",
        text=f"Body for {title}",
        gist=gist or f"Gist for {title}",
    )
