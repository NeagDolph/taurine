from __future__ import annotations

from dataclasses import dataclass, field

from taurine_core.schemas import GraphEdge, GraphNode, GraphResponse, NoteRecord


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _NotesDict(dict[str, NoteRecord]):
    def __call__(self, source: str | None = None) -> list[NoteRecord]:
        rows = list(self.values())
        if source is not None:
            rows = [note for note in rows if note.source == source]
        return rows


@dataclass
class FakeIndex:
    notes: dict[str, NoteRecord] = field(default_factory=dict)
    dense: list[tuple[str, float]] = field(default_factory=list)
    bm25: list[tuple[str, float]] = field(default_factory=list)
    links: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    edge_facts: dict[tuple[str, str], str] = field(default_factory=dict)
    fact_hits: list[tuple[str, str, float, str]] = field(default_factory=list)
    version: int = 1
    alias_rows: list[tuple[str, str, bool]] = field(default_factory=list)
    chunks: dict[str, str] = field(default_factory=dict)
    doc_frequencies: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.notes = _NotesDict(self.notes)

    def dense_search(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.dense[:k]

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.bm25[:k]

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        return self.links.get(note_id, [])

    def neighbor_edges(self, note_id: str) -> list[tuple[str, float, str]]:
        rows: list[tuple[str, float, str]] = []
        for other_id, weight in self.links.get(note_id, []):
            key = tuple(sorted((note_id, other_id)))
            rows.append((other_id, weight, self.edge_facts.get(key, "")))
        return rows

    def note(self, note_id: str) -> NoteRecord | None:
        return self.notes.get(note_id)

    def best_chunk(self, note_id: str, query_text: str) -> str:
        return self.chunks.get(note_id, "")

    def aliases(self) -> list[tuple[str, str, bool]]:
        return self.alias_rows

    def doc_frequency(self, term: str) -> int:
        return self.doc_frequencies.get(term.lower(), 0)

    def graph(self) -> GraphResponse:
        degree = {note_id: 0 for note_id in self.notes}
        edges: list[GraphEdge] = []
        seen: set[tuple[str, str]] = set()
        for src, neighbors in self.links.items():
            for dst, weight in neighbors:
                if src not in self.notes or dst not in self.notes:
                    continue
                key = tuple(sorted((src, dst)))
                if key in seen:
                    continue
                seen.add(key)
                degree[key[0]] += 1
                degree[key[1]] += 1
                edges.append(
                    GraphEdge(
                        src=key[0],
                        dst=key[1],
                        type="semantic",
                        weight=weight,
                        fact=self.edge_facts.get(key, ""),
                    )
                )
        return GraphResponse(
            nodes=[
                GraphNode(
                    note_id=note.note_id,
                    title=note.title,
                    degree=degree.get(note.note_id, 0),
                    source=note.source,
                )
                for note in self.notes.values()
            ],
            edges=edges,
        )

    def graph_version(self) -> int:
        return self.version

    def fact_dense_search(self, text: str, k: int) -> list[tuple[str, str, float, str]]:
        del text
        return self.fact_hits[:k]

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
