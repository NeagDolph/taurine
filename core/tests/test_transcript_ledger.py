from __future__ import annotations

import logging

import pytest

from taurine_core import config
from taurine_core.live.ledger import TranscriptLedger
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.schemas import Chunk


def test_summary_checkpoint_never_consumes_chunks_arriving_in_flight() -> None:
    registry = EvidenceRegistry()
    ledger = TranscriptLedger(registry)
    first = ledger.append(_chunk("first", 1.0))
    second = ledger.append(_chunk("second", 2.0))

    snapshot = ledger.begin_summary()
    third = ledger.append(_chunk("third", 3.0))
    ledger.accept_summary(snapshot)

    assert (first.seq, second.seq, third.seq) == (1, 2, 3)
    assert [row.seq for row in ledger.pending()] == [3]
    assert ledger.last_accepted_summary_seq == 2
    assert ledger.in_flight_through_seq is None


def test_rejected_or_no_update_summary_keeps_every_pending_chunk() -> None:
    ledger = TranscriptLedger(EvidenceRegistry())
    ledger.append(_chunk("one", 1.0))
    ledger.append(_chunk("two", 2.0))

    snapshot = ledger.begin_summary()
    ledger.reject_summary(snapshot)

    assert [row.text for row in ledger.pending()] == ["one", "two"]
    assert ledger.last_accepted_summary_seq == 0


def test_pruning_render_history_cannot_delete_unsummarized_speech() -> None:
    ledger = TranscriptLedger(EvidenceRegistry())
    for index in range(1, 6):
        ledger.append(_chunk(str(index), float(index)))
    snapshot = ledger.begin_summary(through_seq=3)
    ledger.accept_summary(snapshot)

    ledger.prune_accepted(retain=1)

    assert [row.seq for row in ledger.chunks] == [3, 4, 5]
    assert [row.seq for row in ledger.pending()] == [4, 5]


@pytest.mark.parametrize(
    ("channel", "speaker"),
    [(0, "You"), (1, "Them"), (2, "Ch2")],
)
def test_transcript_registry_ids_are_stable_and_render_citeable_speaker_text(
    channel: int,
    speaker: str,
) -> None:
    registry = EvidenceRegistry()
    ledger = TranscriptLedger(registry)
    row = ledger.append(_chunk("Maya moved the review.", 10.0, channel=channel))

    assert row.transcript_ref == "T1"
    assert ledger.render((row,)) == f'[T1] {speaker}: "Maya moved the review."'
    record = registry.resolve("T1")
    assert record is not None
    assert record.payload["text"] == "Maya moved the review."
    assert "read" in record.states


def test_time_aware_window_uses_minimum_and_maximum_bounds_and_partials_for_query_only() -> None:
    registry = EvidenceRegistry()
    ledger = TranscriptLedger(registry)
    for index, ts in enumerate((50.0, 60.0, 70.0, 78.0, 82.0, 90.0), 1):
        ledger.append(_chunk(f"final-{index}", ts))

    window = ledger.query_window(now=92.0, partials={0: "active partial"})

    assert [row.text for row in window.citeable] == ["final-4", "final-5", "final-6"]
    assert [row.text for row in window.context] == ["final-2", "final-3"]
    assert "active partial" in window.query_text
    assert all("active partial" not in row.text for row in window.citeable)
    assert window.duration_s == 12.0


def test_late_completion_cannot_accept_a_new_retry_with_the_same_through_seq() -> None:
    ledger = TranscriptLedger(EvidenceRegistry())
    ledger.append(_chunk("one", 1.0))
    first = ledger.begin_summary()
    ledger.reject_summary(first)
    retry = ledger.begin_summary()

    try:
        ledger.accept_summary(first)
    except ValueError as exc:
        assert "token" in str(exc)
    else:
        raise AssertionError("stale completion accepted a newer checkpoint")

    ledger.accept_summary(retry)
    assert ledger.last_accepted_summary_seq == 1


def test_query_text_is_bounded_even_when_one_final_chunk_is_oversized() -> None:
    from taurine_core import config

    ledger = TranscriptLedger(EvidenceRegistry())
    ledger.append(_chunk("x" * (config.LIVE_WINDOW_MAX_CHARS + 500), 1.0))

    window = ledger.query_window(now=2.0, partials={0: "partial" * 100})

    assert len(window.query_text) <= config.LIVE_WINDOW_MAX_CHARS


def test_summary_budget_checkpoints_only_the_oldest_fully_supplied_chunks() -> None:
    ledger = TranscriptLedger(EvidenceRegistry())
    for index in range(1, 4):
        ledger.append(_chunk(str(index) * 40, float(index)))

    snapshot = ledger.begin_summary(max_chars=90)

    assert [row.seq for row in snapshot.chunks] == [1]
    assert snapshot.through_seq == 1
    ledger.accept_summary(snapshot)
    assert [row.seq for row in ledger.pending()] == [2, 3]


def test_chunk_cap_drops_oldest_first_and_logs_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(config, "LIVE_LEDGER_MAX_CHUNKS", 3)
    ledger = TranscriptLedger(EvidenceRegistry())

    with caplog.at_level(logging.ERROR, logger="taurine.session"):
        for index in range(1, 6):
            ledger.append(_chunk(str(index), float(index)))

    assert [row.seq for row in ledger.chunks] == [3, 4, 5]
    assert "ledger cap" in caplog.text
    # the window keeps working over the surviving chunks
    window = ledger.query_window(now=6.0)
    assert [row.seq for row in window.citeable] == [3, 4, 5]


def test_chunk_cap_drop_of_already_summarized_chunks_is_quiet(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(config, "LIVE_LEDGER_MAX_CHUNKS", 3)
    ledger = TranscriptLedger(EvidenceRegistry())
    for index in range(1, 4):
        ledger.append(_chunk(str(index), float(index)))
    snapshot = ledger.begin_summary()
    ledger.accept_summary(snapshot)

    with caplog.at_level(logging.ERROR, logger="taurine.session"):
        ledger.append(_chunk("4", 4.0))

    assert [row.seq for row in ledger.chunks] == [2, 3, 4]
    assert "ledger cap" not in caplog.text


def _chunk(text: str, ts: float, *, channel: int = 0) -> Chunk:
    return Chunk(channel=channel, text=text, ts=ts, final=True)
