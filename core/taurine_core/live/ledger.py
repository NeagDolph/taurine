"""Sequence-safe transcript evidence, windows, and summary checkpoints."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from taurine_core import config
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.schemas import Chunk


_LOGGER = logging.getLogger("taurine.session")


@dataclass(frozen=True)
class SequencedChunk:
    seq: int
    transcript_ref: str
    channel: int
    text: str
    ts: float


@dataclass(frozen=True)
class SummaryLedgerSnapshot:
    token: int
    through_seq: int
    chunks: tuple[SequencedChunk, ...]


@dataclass(frozen=True)
class LiveQueryWindow:
    query_text: str
    citeable: tuple[SequencedChunk, ...]
    context: tuple[SequencedChunk, ...]
    duration_s: float


class TranscriptLedger:
    def __init__(self, registry: EvidenceRegistry) -> None:
        self.registry = registry
        self.chunks: deque[SequencedChunk] = deque()
        self.last_accepted_summary_seq = 0
        self.in_flight_through_seq: int | None = None
        self._in_flight_token: int | None = None
        self._next_seq = 1
        self._next_summary_token = 1

    def append(self, chunk: Chunk) -> SequencedChunk:
        if not chunk.final:
            raise ValueError("only final transcript chunks enter the evidence ledger")
        text = chunk.text.strip()
        if not text:
            raise ValueError("empty transcript chunks are not evidence")
        seq = self._next_seq
        self._next_seq += 1
        source_id = f"chunk-{seq}"
        transcript_ref = self.registry.register(
            "transcript",
            source_id,
            {
                "seq": seq,
                "channel": int(chunk.channel),
                "text": text,
                "ts": float(chunk.ts),
            },
        )
        row = SequencedChunk(
            seq=seq,
            transcript_ref=transcript_ref,
            channel=int(chunk.channel),
            text=text,
            ts=float(chunk.ts),
        )
        self.chunks.append(row)
        self._enforce_chunk_cap()
        return row

    def _enforce_chunk_cap(self) -> None:
        # Safety bound only: the summary checkpoint normally consumes chunks
        # long before this cap, so hitting it is loud. Oldest chunks drop
        # first, even above the accepted checkpoint if the cap demands it.
        cap = int(config.LIVE_LEDGER_MAX_CHUNKS)
        if cap <= 0 or len(self.chunks) <= cap:
            return
        dropped_unsummarized = 0
        while len(self.chunks) > cap:
            row = self.chunks.popleft()
            if row.seq > self.last_accepted_summary_seq:
                dropped_unsummarized += 1
        if dropped_unsummarized:
            _LOGGER.error(
                "transcript ledger cap %d exceeded: dropped %d unsummarized "
                "chunks (summary checkpoint is stalled)",
                cap,
                dropped_unsummarized,
            )

    def pending(self, *, through_seq: int | None = None) -> tuple[SequencedChunk, ...]:
        upper = through_seq if through_seq is not None else float("inf")
        return tuple(
            row
            for row in self.chunks
            if self.last_accepted_summary_seq < row.seq <= upper
        )

    def begin_summary(
        self,
        *,
        through_seq: int | None = None,
        max_chars: int | None = None,
    ) -> SummaryLedgerSnapshot:
        if self.in_flight_through_seq is not None:
            raise RuntimeError("a transcript summary is already in flight")
        latest = self.chunks[-1].seq if self.chunks else self.last_accepted_summary_seq
        through = latest if through_seq is None else min(int(through_seq), latest)
        rows = self.pending(through_seq=through)
        if not rows:
            raise ValueError("no pending transcript chunks to summarize")
        if max_chars is not None:
            if max_chars <= 0:
                raise ValueError("summary character budget must be positive")
            selected: list[SequencedChunk] = []
            rendered_chars = 0
            for row in rows:
                row_chars = len(row.text) + len(row.transcript_ref) + 24
                if selected and rendered_chars + row_chars > max_chars:
                    break
                selected.append(row)
                rendered_chars += row_chars
            rows = tuple(selected)
            through = rows[-1].seq
        self.in_flight_through_seq = through
        token = self._next_summary_token
        self._next_summary_token += 1
        self._in_flight_token = token
        for row in rows:
            self.registry.mark_read(row.transcript_ref)
        return SummaryLedgerSnapshot(token=token, through_seq=through, chunks=rows)

    def accept_summary(self, snapshot: SummaryLedgerSnapshot) -> None:
        self._finish_summary(snapshot)
        self.last_accepted_summary_seq = max(
            self.last_accepted_summary_seq, snapshot.through_seq
        )

    def reject_summary(self, snapshot: SummaryLedgerSnapshot) -> None:
        self._finish_summary(snapshot)

    def prune_accepted(self, *, retain: int = 0) -> None:
        if retain < 0:
            raise ValueError("retain must be non-negative")
        accepted = [
            row for row in self.chunks if row.seq <= self.last_accepted_summary_seq
        ]
        keep_count = min(retain, len(accepted))
        keep_from = accepted[-keep_count].seq if keep_count else None
        while self.chunks:
            row = self.chunks[0]
            if row.seq > self.last_accepted_summary_seq:
                break
            if keep_from is not None and row.seq >= keep_from:
                break
            self.chunks.popleft()

    def render(self, chunks: Sequence[SequencedChunk]) -> str:
        lines: list[str] = []
        for row in chunks:
            self.registry.mark_read(row.transcript_ref)
            speaker = (
                "You"
                if row.channel == 0
                else "Them" if row.channel == 1 else f"Ch{row.channel}"
            )
            lines.append(f'[{row.transcript_ref}] {speaker}: "{row.text}"')
        return "\n".join(lines)

    def query_window(
        self,
        *,
        now: float,
        partials: Mapping[int, str] | None = None,
        topic_start_seq: int | None = None,
    ) -> LiveQueryWindow:
        del now  # chunk finalization timestamps define the bounded speech interval.
        eligible = [
            row
            for row in self.chunks
            if topic_start_seq is None or row.seq >= topic_start_seq
        ]
        selected: list[SequencedChunk] = []
        characters = 0
        newest_ts = eligible[-1].ts if eligible else 0.0
        for row in reversed(eligible):
            duration = max(0.0, newest_ts - row.ts)
            addition = len(row.text) + (1 if selected else 0)
            if selected and (
                duration > config.LIVE_WINDOW_MAX_S
                or characters + addition > config.LIVE_WINDOW_MAX_CHARS
            ):
                break
            selected.append(row)
            characters += addition
            if len(selected) >= 3 and duration >= config.LIVE_WINDOW_MIN_S:
                break
        citeable = tuple(reversed(selected))
        first_index = (
            eligible.index(citeable[0]) if citeable else len(eligible)
        )
        context = tuple(eligible[max(0, first_index - 2) : first_index])
        parts = [row.text for row in (*context, *citeable)]
        parts.extend(
            text.strip()
            for _, text in sorted((partials or {}).items())
            if text.strip()
        )
        duration_s = (
            max(0.0, citeable[-1].ts - citeable[0].ts) if citeable else 0.0
        )
        query_text = "\n".join(parts)
        if len(query_text) > config.LIVE_WINDOW_MAX_CHARS:
            query_text = query_text[-config.LIVE_WINDOW_MAX_CHARS :]
        return LiveQueryWindow(
            query_text=query_text,
            citeable=citeable,
            context=context,
            duration_s=duration_s,
        )

    def _finish_summary(self, snapshot: SummaryLedgerSnapshot) -> None:
        if self._in_flight_token != snapshot.token:
            raise ValueError("summary checkpoint token does not match the in-flight request")
        if self.in_flight_through_seq != snapshot.through_seq:
            raise ValueError("summary checkpoint does not match the in-flight snapshot")
        self.in_flight_through_seq = None
        self._in_flight_token = None
