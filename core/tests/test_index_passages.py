from __future__ import annotations

import re

from taurine_core.indexer.passages import build_passages
from taurine_core.schemas import NoteRecord


class OffsetTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_passages_prefer_paragraph_boundaries_and_overlap_exact_tokens() -> None:
    first = " ".join(f"alpha{i}" for i in range(180))
    second = " ".join(f"beta{i}" for i in range(180))
    note = _note(f"{first}\n\n{second}")

    rows = build_passages(note, OffsetTokenizer(), version="passages-v1")

    assert len(rows) == 2
    assert rows[0].text == first
    assert rows[0].token_count == 180
    assert rows[1].token_count == 220
    first_tokens = rows[0].text.split()
    second_tokens = rows[1].text.split()
    assert second_tokens[:40] == first_tokens[-40:]
    assert all(row.text == note.text[row.char_start : row.char_end] for row in rows)


def test_overlong_sentence_uses_hard_bounded_windows_with_overlap() -> None:
    note = _note(" ".join(f"word{i}" for i in range(700)))

    rows = build_passages(note, OffsetTokenizer(), version="passages-v1")

    assert len(rows) >= 4
    assert all(1 <= row.token_count <= 300 for row in rows)
    assert all(row.text == note.text[row.char_start : row.char_end] for row in rows)
    for left, right in zip(rows, rows[1:], strict=False):
        assert right.text.split()[:40] == left.text.split()[-40:]
    covered = {token for row in rows for token in row.text.split()}
    assert covered == set(note.text.split())


def test_unicode_offsets_are_note_global_and_source_exact() -> None:
    text = "Résumé planning — 東京 launch.\n\nMaya owns café follow-up."
    note = _note(text)

    rows = build_passages(note, OffsetTokenizer(), version="passages-v1")

    assert rows
    for row in rows:
        assert row.text == text[row.char_start : row.char_end]
        assert row.char_start >= 0
        assert row.char_end <= len(text)
    assert "Résumé" in rows[0].text
    assert "東京" in rows[0].text


def test_passage_ids_are_deterministic_and_versioned() -> None:
    note = _note("One sentence. Another sentence. Final sentence.")
    tokenizer = OffsetTokenizer()

    first = build_passages(note, tokenizer, version="passages-v1")
    second = build_passages(note, tokenizer, version="passages-v1")
    upgraded = build_passages(note, tokenizer, version="passages-v2")

    assert first == second
    assert [row.passage_id for row in first] != [row.passage_id for row in upgraded]
    assert [row.content_hash for row in first] == [row.content_hash for row in upgraded]


def test_whitespace_only_note_creates_no_passages() -> None:
    assert build_passages(_note(" \n\t "), OffsetTokenizer(), version="passages-v1") == ()


def _note(text: str) -> NoteRecord:
    return NoteRecord(
        note_id="note-1",
        title="Test note",
        modified=10.0,
        content_hash="note-content-hash",
        text=text,
    )
