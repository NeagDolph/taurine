"""typing.Protocol seams between delegation units. Frozen in Phase A."""
from __future__ import annotations

from typing import Protocol

from .schemas import GraphResponse, NoteRecord


class IndexPort(Protocol):
    """Implemented by indexer.store.Index (unit 1). Read side used by the live loop."""

    def dense_search(self, text: str, k: int) -> list[tuple[str, float]]:
        """Embed text, cosine vs all units, per-note max. -> [(note_id, score)] desc."""
        ...

    def bm25_search(self, text: str, k: int) -> list[tuple[str, float]]:
        """FTS5 BM25 over title+body. -> [(note_id, score)] desc, empty on no match."""
        ...

    def neighbors(self, note_id: str) -> list[tuple[str, float]]:
        """Note-graph neighbors -> [(note_id, edge_weight)] desc."""
        ...

    def note(self, note_id: str) -> NoteRecord | None:
        ...

    def best_chunk(self, note_id: str, query_text: str) -> str:
        """Chunk of the note most similar to query_text (for rerank docs)."""
        ...

    def aliases(self) -> list[tuple[str, str, bool]]:
        """[(alias_lowercase, note_id, is_title)] for C1."""
        ...

    def doc_frequency(self, term: str) -> int:
        """Number of notes containing the term (for salient-query rarity scoring)."""
        ...

    def graph(self) -> GraphResponse:
        ...

    def status(self) -> tuple[int, float | None]:
        """(note_count, last_sync_ts)"""
        ...


class SpanExtractorPort(Protocol):
    def extract(self, text: str) -> list[tuple[str, float]]:
        """Salient spans in the text -> [(span, score)] desc, deduped, capped.
        Must never raise; returns [] on any failure."""
        ...


class RerankerPort(Protocol):
    async def rerank(self, query: str, docs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """docs=[(note_id, doc_text)] -> [(note_id, relevance)] desc.
        Raises RerankError on failure/timeout; caller falls back."""
        ...


class RerankError(Exception):
    pass


class LLMPort(Protocol):
    async def complete_json(self, model: str, system: str, user: str) -> dict:
        """One chat completion via OpenRouter, response_format json_object.
        Raises LLMError on HTTP/parse failure."""
        ...


class LLMError(Exception):
    pass
