"""typing.Protocol seams between delegation units. Frozen in Phase A."""
from __future__ import annotations

from typing import Protocol

from .schemas import GraphResponse, NoteRecord


class IndexPort(Protocol):
    """Implemented by indexer.store.Index (unit 1). Read side used by the live loop."""

    def dense_search(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, float]]:
        """Embed text, cosine vs all units, per-note max. -> [(note_id, score)] desc."""
        ...

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        """FTS5 BM25 over title+body. -> [(note_id, score)] desc, empty on no match."""
        ...

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        """Note-graph neighbors -> [(note_id, edge_weight)] desc."""
        ...

    def neighbor_edges(self, note_id: str) -> list[tuple[str, float, str]]:
        """Note-graph neighbors with facts -> [(note_id, edge_weight, fact)] desc."""
        ...

    def fact_dense_search(
        self,
        text: str,
        k: int,
        *,
        prompt: str | None = None,
    ) -> list[tuple[str, str, float, str]]:
        """Fact dense search -> [(src, dst, score, fact_text)] desc."""
        ...

    def note(self, note_id: str) -> NoteRecord | None:
        ...

    def notes(self, source: str | None = None) -> list[NoteRecord]:
        ...

    def best_chunk(self, note_id: str, query_text: str) -> str:
        """Chunk of the note most similar to query_text."""
        ...

    def aliases(self) -> list[tuple[str, str, bool]]:
        """[(alias_lowercase, note_id, is_title)] for C1."""
        ...

    def doc_frequency(self, term: str) -> int:
        """Number of notes containing the term."""
        ...

    def graph(self) -> GraphResponse:
        ...

    def graph_version(self) -> int:
        ...

    def status(self) -> tuple[int, float | None]:
        """(note_count, last_sync_ts)"""
        ...


class SpanExtractorPort(Protocol):
    def extract(self, text: str) -> list[tuple[str, float, str]]:
        """Salient spans in the text -> [(span, score, label)], deduped, capped,
        strongest first. Must never raise; returns [] on any failure."""
        ...


class ConnectorPort(Protocol):
    """A local (or later OAuth) data source producing NoteRecords."""

    connector_id: str              # stable id; also the NoteRecord.source value
    label: str                     # human name for the UI
    kind: str                      # "local" | "oauth"

    def available(self) -> tuple[bool, str]:
        """(usable, remediation detail when not). Must never raise."""
        ...

    def default_config(self) -> dict:
        ...

    def extract(self, config: dict) -> list["NoteRecord"]:
        """Full snapshot of this source's documents. Raises ConnectorError."""
        ...


class ConnectorError(Exception):
    pass


class LLMPort(Protocol):
    async def complete_json(
        self, model: str, system: str, user: str, timeout_s: float | None = None
    ) -> dict:
        """One chat completion via OpenRouter, response_format json_object.
        timeout_s overrides config.LLM_TIMEOUT_S (long offline calls).
        Raises LLMError on HTTP/parse failure."""
        ...


class LLMError(Exception):
    pass
