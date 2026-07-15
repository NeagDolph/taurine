"""Google Calendar / Docs connector units — conversion logic, no network."""
from __future__ import annotations

import pytest

from taurine_core.connectors.google_calendar import (
    GoogleCalendarConnector,
    _events_to_records,
    _self_declined,
)
from taurine_core.connectors.google_docs import (
    GoogleDocsConnector,
    _document_body_text,
    _metadata_header,
)


CAL = {"id": "neil@manymailz.com", "summary": "Work"}


def _event(**overrides) -> dict:
    base = {
        "id": "ev1",
        "status": "confirmed",
        "summary": "Wink integration sync",
        "created": "2026-07-01T10:00:00Z",
        "updated": "2026-07-10T09:00:00Z",
        "start": {"dateTime": "2026-07-14T10:00:00-04:00"},
        "end": {"dateTime": "2026-07-14T10:30:00-04:00"},
        "organizer": {"displayName": "Neil", "email": "neil@manymailz.com"},
        "attendees": [
            {"displayName": "Ashish Parik", "email": "ashish@shoplc.com",
             "responseStatus": "accepted"},
            {"email": "vineet@shoplc.com", "responseStatus": "needsAction"},
        ],
        "location": "Zoom",
        "description": "Discuss <b>AMS integration</b> rollout.",
        "hangoutLink": "https://meet.google.com/abc",
        "htmlLink": "https://calendar.google.com/event?eid=1",
    }
    base.update(overrides)
    return base


class TestCalendarRecords:
    def test_single_event_note_carries_metadata(self):
        records = _events_to_records([(CAL, _event())])
        assert len(records) == 1
        note = records[0]
        assert note.note_id == "google_calendar:neil@manymailz.com/ev1"
        assert note.title == "Wink integration sync"
        assert note.folder == "Work"
        assert note.source == "google_calendar"
        assert "Ashish Parik <ashish@shoplc.com> (accepted)" in note.text
        assert "vineet@shoplc.com" in note.text
        assert "Zoom" in note.text
        assert "meet.google.com/abc" in note.text
        assert "AMS integration" in note.text          # html stripped, content kept
        assert "<b>" not in note.text
        assert "Tue 2026-07-14 10:00–10:30" in note.text

    def test_recurring_instances_collapse_to_one_note(self):
        instances = [
            (CAL, _event(id=f"ev1_{i}", recurringEventId="ev1",
                         start={"dateTime": f"2026-07-{14 + 7 * i:02d}T10:00:00-04:00"},
                         end={"dateTime": f"2026-07-{14 + 7 * i:02d}T10:30:00-04:00"}))
            for i in range(3)
        ]
        records = _events_to_records(instances)
        assert len(records) == 1
        assert "3 occurrences" in records[0].text
        assert records[0].note_id == "google_calendar:neil@manymailz.com/ev1"

    def test_all_day_event(self):
        records = _events_to_records([
            (CAL, _event(start={"date": "2026-07-14"}, end={"date": "2026-07-15"}))
        ])
        assert "(all day)" in records[0].text

    def test_declined_detection(self):
        declined = _event(attendees=[{"self": True, "responseStatus": "declined"}])
        assert _self_declined(declined)
        assert not _self_declined(_event())

    def test_availability_gated_on_env(self, monkeypatch):
        for key in ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        ok, detail = GoogleCalendarConnector().available()
        assert not ok
        assert "email_auth" in detail


class TestDocsParsing:
    DOC = {
        "title": "Roadmap",
        "body": {"content": [
            {"paragraph": {"paragraphStyle": {"namedStyleType": "HEADING_1"},
                           "elements": [{"textRun": {"content": "Q3 Roadmap\n"}}]}},
            {"paragraph": {"elements": [{"textRun": {"content": "Ship the "}},
                                        {"textRun": {"content": "ASR pipeline.\n"}}]}},
            {"paragraph": {"bullet": {"nestingLevel": 0},
                           "elements": [{"textRun": {"content": "Fine-tune WER\n"}}]}},
            {"paragraph": {"bullet": {"nestingLevel": 1},
                           "elements": [{"textRun": {"content": "Collect calls\n"}}]}},
            {"table": {"tableRows": [
                {"tableCells": [
                    {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Metric\n"}}]}}]},
                    {"content": [{"paragraph": {"elements": [{"textRun": {"content": "15.88%\n"}}]}}]},
                ]},
            ]}},
        ]},
    }

    def test_body_markdown(self):
        text = _document_body_text(self.DOC)
        assert "# Q3 Roadmap" in text
        assert "Ship the ASR pipeline." in text
        assert "- Fine-tune WER" in text
        assert "  - Collect calls" in text
        assert "| Metric | 15.88% |" in text

    def test_metadata_header(self):
        file = {
            "owners": [{"displayName": "Neil", "emailAddress": "neil@manymailz.com"}],
            "lastModifyingUser": {"displayName": "Neil"},
            "modifiedTime": "2026-07-10T12:00:00Z",
            "parents": ["folder1"],
            "shared": True,
            "webViewLink": "https://docs.google.com/document/d/x",
        }
        header = _metadata_header(file, {"folder1": "Wink"})
        assert "Owner: Neil <neil@manymailz.com>" in header
        assert "Last edited: 2026-07-10 by Neil" in header
        assert "Folder: Wink" in header
        assert "Shared" in header
        assert "docs.google.com" in header

    def test_availability_gated_on_env(self, monkeypatch):
        for key in ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        ok, detail = GoogleDocsConnector().available()
        assert not ok
        assert "email_auth" in detail


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
