from __future__ import annotations

import json
from dataclasses import dataclass, field

from taurine_core.chat.tools import TOOL_DEFINITIONS, execute_tool
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    RetrievalDecision,
    RetrievalQuery,
    RetrievalTrace,
)


@dataclass
class ToolRetriever:
    calls: list[tuple] = field(default_factory=list)

    def search_evidence(self, query: str, *, k: int, temporal: str, now: float) -> EvidencePack:
        self.calls.append(("search", query, k, temporal, now))
        return EvidencePack(
            decision=RetrievalDecision(True, .91, "accepted", ("dense", "triple")),
            query=RetrievalQuery(query, "tool", now),
            passages=(
                EvidencePassage(
                    "E1", "p1", "n1", "Roadmap", "obsidian",
                    "Launch moved to July 18.", 4, 28, 1, 2, .7, .9,
                    passage_date="2026-07-18",
                ),
            ),
            facts=(
                EvidenceFact("F1", ("f1",), "Roadmap", "launch_date", "July 18", "forward", "active", 2, None, None, .9, ("E1",)),
            ),
            paths=(),
            notes=(EvidenceNote("N1", "n1", "Roadmap", "obsidian", 2, .9, ("E1",)),),
            trace=RetrievalTrace(original_query=query, rewritten_query=query, read_ids=("N1", "E1", "F1"), active_build_id="b1"),
        )

    def read_note(self, note: str, *, query: str, cursor: str | None, max_chars: int):
        self.calls.append(("read", note, query, cursor, max_chars))
        return {
            "note": note, "title": "Roadmap", "source": "obsidian",
            "passages": [{"evidence_id": "E2", "text": "Armaan owns review.", "char_start": 40, "char_end": 59}],
            "truncated": True, "next_cursor": "cursor-2",
        }

    def expand_evidence(self, evidence: str, *, direction: str, max_hops: int, limit: int):
        self.calls.append(("expand", evidence, direction, max_hops, limit))
        return {
            "from": evidence,
            "results": [{"evidence_id": "E3", "note": "N2", "text": "Review depends on API.", "fact_refs": ["F2"], "path": "P1"}],
            "paths": [{"path_id": "P1", "edge_kinds": ["fact:depends_on"], "fact_refs": ["F2"]}],
        }

    def resolve_note(self, ref: str) -> str | None:
        return {"N1": "n1", "N2": "n2"}.get(ref)


def test_tool_registration_is_evidence_native() -> None:
    definitions = {row["function"]["name"]: row for row in TOOL_DEFINITIONS}

    assert set(definitions) == {"search_evidence", "read_note", "expand_evidence"}
    assert definitions["search_evidence"]["function"]["parameters"]["properties"]["temporal"]["enum"] == ["current", "historical", "any"]
    assert set(definitions["read_note"]["function"]["parameters"]["required"]) == {"note", "query"}
    assert definitions["expand_evidence"]["function"]["parameters"]["properties"]["direction"]["enum"] == ["outgoing", "incoming", "both"]


def test_search_evidence_returns_exact_registered_evidence() -> None:
    retriever = ToolRetriever()

    result_json, step = execute_tool(
        "search_evidence", {"query": "roadmap", "k": 4, "temporal": "current"}, retriever, now=10.0,
    )

    payload = json.loads(result_json)
    assert payload["decision"]["accepted"] is True
    assert payload["passages"][0]["evidence_id"] == "E1"
    assert payload["passages"][0]["text"] == "Launch moved to July 18."
    assert payload["passages"][0]["passage_date"] == "2026-07-18"
    assert payload["facts"][0]["fact_ref"] == "F1"
    assert retriever.calls == [("search", "roadmap", 4, "current", 10.0)]
    assert step.kind == "search"
    assert step.note_ids == ["n1"]


def test_read_note_requires_query_and_exposes_pagination_truthfully() -> None:
    retriever = ToolRetriever()

    result_json, step = execute_tool(
        "read_note", {"note": "N1", "query": "owner", "max_chars": 1200}, retriever, now=0.0,
    )

    payload = json.loads(result_json)
    assert payload["passages"][0]["evidence_id"] == "E2"
    assert payload["truncated"] is True
    assert payload["next_cursor"] == "cursor-2"
    assert retriever.calls == [("read", "N1", "owner", None, 1200)]
    assert step.kind == "read"


def test_expand_evidence_preserves_direction_and_registered_paths() -> None:
    retriever = ToolRetriever()

    result_json, step = execute_tool(
        "expand_evidence", {"evidence": "F1", "direction": "incoming", "max_hops": 2, "limit": 5}, retriever, now=0.0,
    )

    payload = json.loads(result_json)
    assert payload["results"][0]["evidence_id"] == "E3"
    assert payload["results"][0]["fact_refs"] == ["F2"]
    assert payload["paths"][0]["path_id"] == "P1"
    assert retriever.calls == [("expand", "F1", "incoming", 2, 5)]
    assert step.kind == "expand"


def test_search_evidence_fallback_rows_are_marked_low_confidence() -> None:
    """§0.1: fallback pack rows carry low_confidence and an honest label."""

    class FallbackRetriever(ToolRetriever):
        def search_evidence(self, query: str, *, k: int, temporal: str, now: float) -> EvidencePack:
            pack = super().search_evidence(query, k=k, temporal=temporal, now=now)
            from dataclasses import replace

            return replace(
                pack,
                decision=RetrievalDecision(
                    False, .44, "fallback_low_confidence", fallback=True
                ),
                facts=(),
                paths=(),
            )

    result_json, step = execute_tool(
        "search_evidence", {"query": "roadmap"}, FallbackRetriever(), now=10.0,
    )

    payload = json.loads(result_json)
    assert payload["decision"]["accepted"] is False
    assert payload["decision"]["fallback"] is True
    assert all(row["low_confidence"] is True for row in payload["passages"])
    assert "low-confidence" in step.label

    accepted_json, _step = execute_tool(
        "search_evidence", {"query": "roadmap"}, ToolRetriever(), now=10.0,
    )
    accepted = json.loads(accepted_json)
    assert accepted["decision"]["fallback"] is False
    assert all(row["low_confidence"] is False for row in accepted["passages"])


def test_tool_errors_are_model_visible_and_do_not_raise() -> None:
    retriever = ToolRetriever()

    result_json, step = execute_tool("missing", {}, retriever, now=0.0)
    bad_read_json, bad_step = execute_tool("read_note", {"note": "N1"}, retriever, now=0.0)

    assert json.loads(result_json) == {"error": "unknown tool: missing"}
    assert json.loads(bad_read_json) == {"error": "query is required"}
    assert step.label.startswith("Failed:")
    assert bad_step.label.startswith("Failed:")
