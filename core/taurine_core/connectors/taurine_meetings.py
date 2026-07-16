from __future__ import annotations

from datetime import datetime
from typing import Any

from taurine_core import config
from taurine_core.connectors.base import content_hash
from taurine_core.schemas import NoteRecord


class TaurineMeetingsConnector:
    connector_id = config.TAURINE_SOURCE_ID
    label = "Taurine"
    kind = "local"
    hidden = True

    def __init__(self, index: Any) -> None:
        self._index = index

    def available(self) -> tuple[bool, str]:
        return True, ""

    def default_config(self) -> dict:
        return {}

    def extract(self, config: dict) -> list[NoteRecord]:
        del config
        records: list[NoteRecord] = []
        for meeting in self._index.meetings_done():
            notes_md = str(meeting.get("notes_md") or "")
            final_summary_md = str(meeting.get("final_summary_md") or "")
            if not notes_md.strip() and not final_summary_md.strip():
                continue
            created_at = float(meeting["created_at"])
            title = str(meeting.get("title") or "") or (
                f"Meeting {datetime.fromtimestamp(created_at).date():%Y-%m-%d}"
            )
            text = notes_md + (
                "\n\n## Meeting summary\n" + final_summary_md
                if final_summary_md
                else ""
            )
            records.append(
                NoteRecord(
                    note_id=f"{self.connector_id}:{meeting['meeting_id']}",
                    source=self.connector_id,
                    title=title,
                    text=text,
                    modified=float(meeting["updated_at"]),
                    created=created_at,
                    content_hash=content_hash(title, text),
                )
            )
        return records
