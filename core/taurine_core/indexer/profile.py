"""Explicit GLM-5.2 research agent for the persisted user information card."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from taurine_core import config
from taurine_core.chat.retrieval import ChatRetriever
from taurine_core.chat.tools import TOOL_DEFINITIONS, execute_tool
from taurine_core.indexer.store import Index
from taurine_core.llm import OpenRouterClient
from taurine_core.ports import LLMError


ProfilePhase = Literal[
    "surveying",
    "interests",
    "projects_people",
    "career_studies",
    "synthesizing",
]


@dataclass(frozen=True)
class ProfileProgress:
    phase: ProfilePhase
    activity: str
    nodes_read: int
    tool_calls: int


SYSTEM_PROMPT = """You are Taurine's user-profile research agent.

Build a detailed, accurate understanding of the owner of this personal
knowledge graph. Research agentically before writing: search independent
aspects, read primary source nodes, and traverse useful graph connections.

Cover, when supported by the graph:
- identity and defining background;
- recurring interests, technical areas, practices, and hobbies;
- current and historical projects, including their purpose and status;
- important people and the user's relationship or working connection to them;
- employers, roles, clients, collaborators, career history, and current work;
- education, courses, research, or current studies;
- patterns in how the user works and what they care about.

Distinguish current activity from historical material using dates, note ages,
and wording. Triangulate important claims across nodes. Never guess or fill a
category merely to make the profile look complete.

Your terminal response must be detailed Markdown written as durable background
context for other agents. Do not mention searches, tools, note IDs, graph
mechanics, citations, or missing categories. Do not emit JSON.
"""

RESEARCH_BRIEF = """Research the user thoroughly across the graph now.

Start by surveying multiple independent aspects instead of anchoring on the
first cluster. Follow promising project, person, organization, career, and
education connections into their source nodes. Continue until you can write a
coherent, detailed profile of who the user is and what they do.
"""

SYNTHESIS_INSTRUCTION = """Research is complete. Write the detailed Markdown
user profile now using only what you learned. Do not mention tools, searches,
note IDs, graph mechanics, or missing data."""


class ProfileGenerator:
    def __init__(
        self,
        index: Index,
        llm: OpenRouterClient,
        *,
        retriever: Any | None = None,
    ) -> None:
        self.index = index
        self.llm = llm
        self.retriever = retriever or ChatRetriever(index, "__profile_generation__")

    async def generate(
        self,
        on_progress: Callable[[ProfileProgress], None],
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RESEARCH_BRIEF},
        ]
        tool_count = 0
        touched_nodes: set[str] = set()
        current_phase: ProfilePhase = "surveying"
        synthesis_notice_sent = False
        on_progress(ProfileProgress(current_phase, "Surveying the knowledge graph", 0, 0))

        while True:
            tools = (
                TOOL_DEFINITIONS
                if tool_count < config.PROFILE_MAX_TOOL_CALLS
                else None
            )
            if tools is None and not synthesis_notice_sent:
                synthesis_notice_sent = True
                current_phase = "synthesizing"
                messages.append({"role": "user", "content": SYNTHESIS_INSTRUCTION})
                on_progress(
                    ProfileProgress(
                        current_phase,
                        "Writing the user profile",
                        len(touched_nodes),
                        tool_count,
                    )
                )

            content_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            async for event in self.llm.stream_chat(
                config.PROFILE_MODEL,
                messages,
                tools=tools,
                reasoning=config.PROFILE_REASONING,
                timeout_s=config.PROFILE_TIMEOUT_S,
            ):
                if event.kind == "content" and event.text:
                    content_parts.append(event.text)
                elif event.kind == "tool_call" and event.tool_call is not None:
                    tool_calls.append(event.tool_call)
                elif event.kind == "done":
                    break

            if not tool_calls or tools is None:
                if current_phase != "synthesizing":
                    current_phase = "synthesizing"
                    on_progress(
                        ProfileProgress(
                            current_phase,
                            "Writing the user profile",
                            len(touched_nodes),
                            tool_count,
                        )
                    )
                card = "".join(content_parts).strip()
                if not card:
                    raise LLMError("empty user profile returned by GLM-5.2")
                return _clamp_card(card)

            assistant_tool_calls: list[dict[str, Any]] = []
            tool_messages: list[dict[str, Any]] = []
            for call in tool_calls:
                if tool_count >= config.PROFILE_MAX_TOOL_CALLS:
                    break
                name = str(call.get("name") or "")
                arguments = (
                    call.get("arguments")
                    if isinstance(call.get("arguments"), dict)
                    else {}
                )
                current_phase = _phase_for_tool(name, arguments, current_phase)
                activity = _running_activity(name, arguments)
                on_progress(
                    ProfileProgress(
                        current_phase,
                        activity,
                        len(touched_nodes),
                        tool_count,
                    )
                )
                result_json, step = execute_tool(
                    name,
                    arguments,
                    self.retriever,
                    self.index,
                    now=time.time(),
                )
                touched_nodes.update(step.note_ids)
                tool_count += 1
                on_progress(
                    ProfileProgress(
                        current_phase,
                        step.label,
                        len(touched_nodes),
                        tool_count,
                    )
                )
                assistant_tool_calls.append(
                    {
                        "id": str(call.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "name": name,
                        "content": result_json,
                    }
                )

            if assistant_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": assistant_tool_calls,
                    }
                )
                messages.extend(tool_messages)


def _phase_for_tool(
    name: str,
    arguments: dict[str, Any],
    current: ProfilePhase,
) -> ProfilePhase:
    query = str(arguments.get("query") or "").casefold()
    if any(term in query for term in ("career", "employ", "job", "role", "stud", "education")):
        return "career_studies"
    if any(term in query for term in ("interest", "hobby", "topic", "practice")):
        return "interests"
    if any(
        term in query
        for term in ("project", "people", "person", "relationship", "client", "collabor", "work")
    ):
        return "projects_people"
    if name in {"read_note", "expand_note"} and current == "surveying":
        return "projects_people"
    return current


def _running_activity(name: str, arguments: dict[str, Any]) -> str:
    if name == "search_notes":
        query = str(arguments.get("query") or "").strip()
        return f'Searching for "{query}"' if query else "Searching the knowledge graph"
    ref = str(arguments.get("note") or "").strip()
    if name == "read_note":
        return f"Reading {ref}" if ref else "Reading a source node"
    if name == "expand_note":
        return f"Following connections from {ref}" if ref else "Following graph connections"
    return "Researching the knowledge graph"


def _clamp_card(card: str) -> str:
    if len(card) <= config.PROFILE_MAX_CHARS:
        return card
    return card[: config.PROFILE_MAX_CHARS].rstrip()
