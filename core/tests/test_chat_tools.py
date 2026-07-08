from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from taurine_core import config
from taurine_core.chat.tools import execute_tool
from taurine_core.schemas import GraphEdge, GraphNode, GraphResponse, GroundingItem, NoteRecord


@dataclass
class ToolRetriever:
    search_rows: list[GroundingItem] = field(default_factory=list)
    refs: dict[str, str] = field(default_factory=dict)
    searches: list[tuple[str, int, float]] = field(default_factory=list)

    def search(self, query: str, k: int, *, now: float) -> list[GroundingItem]:
        self.searches.append((query, k, now))
        return self.search_rows[:k]

    def resolve(self, ref: str) -> str | None:
        return self.refs.get(ref)


@dataclass
class ToolIndex:
    notes: dict[str, NoteRecord]
    edges: list[GraphEdge] = field(default_factory=list)

    def note(self, note_id: str) -> NoteRecord | None:
        return self.notes.get(note_id)

    def neighbor_edges(self, note_id: str) -> list[tuple[str, float, str]]:
        rows: list[tuple[str, float, str]] = []
        for edge in self.edges:
            if edge.src == note_id:
                rows.append((edge.dst, edge.weight, edge.fact))
            elif edge.dst == note_id:
                rows.append((edge.src, edge.weight, edge.fact))
        return sorted(rows, key=lambda row: row[1], reverse=True)

    def graph(self) -> GraphResponse:
        return GraphResponse(
            nodes=[
                GraphNode(note_id=note.note_id, title=note.title, degree=0, source=note.source)
                for note in self.notes.values()
            ],
            edges=self.edges,
        )


def _note(note_id: str, title: str, text: str = "body") -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        title=title,
        modified=86_400.0,
        content_hash=f"hash-{note_id}",
        text=text,
        gist=f"Gist for {title}",
        source="obsidian",
    )


def test_search_notes_returns_compact_rows_and_activity_shape() -> None:
    retriever = ToolRetriever(
        search_rows=[
            GroundingItem(
                note_id="n1",
                short_id="N1",
                title="Roadmap",
                source="obsidian",
                age_days=1.234,
                score=0.98765,
                gist="Roadmap gist",
            )
        ]
    )
    index = ToolIndex(notes={})

    result_json, step = execute_tool("search_notes", {"query": "roadmap"}, retriever, index, now=10.0)

    assert ": " not in result_json
    assert json.loads(result_json) == {
        "rows": [
            {
                "short_id": "N1",
                "title": "Roadmap",
                "source": "obsidian",
                "age_days": 1.2,
                "score": 0.988,
                "gist": "Roadmap gist",
            }
        ]
    }
    assert retriever.searches == [("roadmap", config.CHAT_SEARCH_TOOL_K, 10.0)]
    assert step.kind == "search"
    assert step.status == "done"
    assert step.label == 'Searched "roadmap" → 1 notes'
    assert step.note_ids == ["n1"]


def test_read_note_resolves_caps_text_and_includes_neighbor_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CHAT_NOTE_READ_MAX_CHARS", 12)
    retriever = ToolRetriever(refs={"N1": "n1"})
    index = ToolIndex(
        notes={
            "n1": _note("n1", "Alpha", "abcdefghijklmnop"),
            "n2": _note("n2", "Beta"),
        },
        edges=[GraphEdge(src="n1", dst="n2", type="topic", weight=0.7, fact="Alpha relates to Beta.")],
    )

    result_json, step = execute_tool("read_note", {"note": "N1"}, retriever, index, now=172_800.0)

    assert json.loads(result_json) == {
        "note_id": "n1",
        "title": "Alpha",
        "source": "obsidian",
        "age_days": 1.0,
        "text": "abcdefghijkl",
        "neighbors": [
            {
                "note_id": "n2",
                "title": "Beta",
                "source": "obsidian",
                "weight": 0.7,
                "fact": "Alpha relates to Beta.",
            }
        ],
    }
    assert step.label == "Read: Alpha"
    assert step.note_ids == ["n1"]


def test_expand_note_returns_weight_sorted_neighbors_with_type_and_cap() -> None:
    retriever = ToolRetriever(refs={"N1": "n1"})
    notes = {"n1": _note("n1", "Hub")}
    edges: list[GraphEdge] = []
    for i in range(14):
        note_id = f"n{i + 2}"
        notes[note_id] = _note(note_id, f"Neighbor {i + 2}")
        edges.append(
            GraphEdge(
                src="n1",
                dst=note_id,
                type="mixed" if i == 13 else "semantic",
                weight=i / 10,
                fact=f"Fact {i}",
            )
        )
    index = ToolIndex(notes=notes, edges=edges)

    result_json, step = execute_tool("expand_note", {"note": "N1"}, retriever, index, now=0.0)

    rows = json.loads(result_json)["neighbors"]
    assert len(rows) == 12
    assert rows[0] == {
        "note_id": "n15",
        "title": "Neighbor 15",
        "source": "obsidian",
        "weight": 1.3,
        "type": "mixed",
        "fact": "Fact 13",
    }
    assert rows[-1]["weight"] == 0.2
    assert step.label == "Expanded: Hub"
    assert step.note_ids == ["n1"]


def test_tool_errors_return_error_json_and_failed_step() -> None:
    retriever = ToolRetriever(refs={})
    index = ToolIndex(notes={})

    unknown_json, unknown_step = execute_tool("missing_tool", {}, retriever, index, now=0.0)
    missing_json, missing_step = execute_tool("read_note", {"note": "N404"}, retriever, index, now=0.0)

    assert json.loads(unknown_json) == {"error": "unknown tool: missing_tool"}
    assert unknown_step.label == "Failed: unknown tool: missing_tool"
    assert json.loads(missing_json) == {"error": "could not resolve note: N404"}
    assert missing_step.label == "Failed: could not resolve note: N404"
