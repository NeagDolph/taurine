"""Google Calendar connector (kind="oauth", shares the Gmail token).

One NoteRecord per event — recurring series collapse into a single note
carrying the schedule and upcoming instances, so a weekly standup is one
graph node, not fifty. Each note packs as much context as the API offers:
when, who (attendees with response status), where, organizer, conference
link, calendar, recurrence, and the full description.

Holiday/birthday system calendars are skipped by default; declined and
cancelled events are always skipped.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx

from taurine_core import config as taurine_config
from taurine_core.connectors.base import content_hash
from taurine_core.connectors.email_common import html_to_text
from taurine_core.connectors.google_common import (
    AUTH_HELP,
    GoogleToken,
    get_json,
    google_env_ready,
)
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord

_API = "https://www.googleapis.com/calendar/v3"
_SYSTEM_CALENDAR_MARKERS = (
    "group.v.calendar.google.com",  # holidays et al.
    "#contacts@",                   # birthdays
)


class GoogleCalendarConnector:
    connector_id = "google_calendar"
    label = "Google Calendar"
    kind = "oauth"

    def __init__(self) -> None:
        self._token = GoogleToken("Google Calendar")
        self._cached_records: list[NoteRecord] | None = None
        self._cached_at = 0.0
        self._client_factory = lambda: httpx.Client(timeout=30.0)

    def available(self) -> tuple[bool, str]:
        if not google_env_ready():
            return False, f"Google sign-in required — {AUTH_HELP}"
        return True, ""

    def default_config(self) -> dict:
        return {
            "past_days": taurine_config.GCAL_PAST_DAYS,
            "future_days": taurine_config.GCAL_FUTURE_DAYS,
            "include_system_calendars": False,
        }

    def extract(self, config: dict) -> list[NoteRecord]:
        if not google_env_ready():
            raise ConnectorError(f"Google sign-in required — {AUTH_HELP}")
        now = time.time()
        if (
            self._cached_records is not None
            and now - self._cached_at < taurine_config.GOOGLE_SYNC_MIN_INTERVAL_S
        ):
            return self._cached_records
        past_days = int(config.get("past_days") or taurine_config.GCAL_PAST_DAYS)
        future_days = int(config.get("future_days") or taurine_config.GCAL_FUTURE_DAYS)
        include_system = bool(config.get("include_system_calendars"))
        try:
            with self._client_factory() as client:
                token = self._token.get(client)
                headers = {"Authorization": f"Bearer {token}"}
                calendars = _list_calendars(client, headers, include_system)
                events = _fetch_events(client, headers, calendars, past_days, future_days)
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Google Calendar API unreachable: {exc}") from exc
        self._cached_records = _events_to_records(events)
        self._cached_at = now
        return self._cached_records


def _list_calendars(
    client: httpx.Client, headers: dict, include_system: bool
) -> list[dict]:
    calendars: list[dict] = []
    page_token: str | None = None
    while True:
        params: dict = {"maxResults": 100}
        if page_token:
            params["pageToken"] = page_token
        data = get_json(
            client, f"{_API}/users/me/calendarList", headers, params,
            service="Google Calendar",
        )
        for entry in data.get("items", []):
            calendar_id = str(entry.get("id", ""))
            if not include_system and any(
                marker in calendar_id for marker in _SYSTEM_CALENDAR_MARKERS
            ):
                continue
            calendars.append(entry)
        page_token = data.get("nextPageToken")
        if not page_token:
            return calendars


def _fetch_events(
    client: httpx.Client,
    headers: dict,
    calendars: list[dict],
    past_days: int,
    future_days: int,
) -> list[tuple[dict, dict]]:
    """Returns (calendar_entry, event) pairs, capped at GCAL_MAX_EVENTS."""
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(days=past_days)).isoformat()
    time_max = (now + timedelta(days=future_days)).isoformat()
    pairs: list[tuple[dict, dict]] = []
    for calendar in calendars:
        calendar_id = str(calendar.get("id", ""))
        page_token = None
        while len(pairs) < taurine_config.GCAL_MAX_EVENTS:
            params = {
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": 250,
            }
            if page_token:
                params["pageToken"] = page_token
            data = get_json(
                client,
                f"{_API}/calendars/{quote(calendar_id, safe='')}/events",
                headers,
                params,
                service="Google Calendar",
            )
            for event in data.get("items", []):
                if event.get("status") == "cancelled":
                    continue
                if _self_declined(event):
                    continue
                pairs.append((calendar, event))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        if len(pairs) >= taurine_config.GCAL_MAX_EVENTS:
            break
    return pairs[: taurine_config.GCAL_MAX_EVENTS]


def _self_declined(event: dict) -> bool:
    for attendee in event.get("attendees", []) or []:
        if attendee.get("self") and attendee.get("responseStatus") == "declined":
            return True
    return False


def _events_to_records(pairs: list[tuple[dict, dict]]) -> list[NoteRecord]:
    """Collapse recurring instances into one note per series."""
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for calendar, event in pairs:
        key = str(event.get("recurringEventId") or event.get("id") or "")
        if not key:
            continue
        groups.setdefault(f"{calendar.get('id','')}/{key}", []).append((calendar, event))

    records: list[NoteRecord] = []
    for group_key, members in groups.items():
        members.sort(key=lambda pair: _start_ts(pair[1]))
        calendar, first = members[0]
        title = str(first.get("summary") or "(untitled event)").strip()
        text = _event_text(calendar, members)
        modified = max(
            (_parse_iso_ts(event.get("updated")) for _, event in members),
            default=time.time(),
        )
        created = min(
            (_parse_iso_ts(event.get("created")) for _, event in members),
            default=modified,
        )
        records.append(
            NoteRecord(
                note_id=f"google_calendar:{group_key}",
                title=title,
                folder=str(calendar.get("summary") or ""),
                modified=modified,
                created=created,
                content_hash=content_hash(title, text),
                text=text[: taurine_config.CONNECTOR_DOC_MAX_CHARS],
                source="google_calendar",
            )
        )
    return records


def _event_text(calendar: dict, members: list[tuple[dict, dict]]) -> str:
    _, first = members[0]
    lines: list[str] = []
    recurring = len(members) > 1 or bool(first.get("recurringEventId"))
    lines.append(f"- Calendar: {calendar.get('summary', '')}")
    if recurring:
        lines.append(f"- Recurring event with {len(members)} occurrences in the sync window:")
        for _, event in members[:12]:
            lines.append(f"  - {_when(event)}")
        if len(members) > 12:
            lines.append(f"  - … and {len(members) - 12} more")
    else:
        lines.append(f"- When: {_when(first)}")
    location = str(first.get("location") or "").strip()
    if location:
        lines.append(f"- Location: {location}")
    organizer = first.get("organizer") or {}
    organizer_line = _person(organizer)
    if organizer_line:
        lines.append(f"- Organizer: {organizer_line}")
    attendees = [a for a in (first.get("attendees") or []) if not a.get("resource")]
    if attendees:
        lines.append(f"- Attendees ({len(attendees)}):")
        for attendee in attendees[:25]:
            status = str(attendee.get("responseStatus") or "")
            suffix = f" ({status})" if status and status != "needsAction" else ""
            lines.append(f"  - {_person(attendee)}{suffix}")
    conference = _conference_link(first)
    if conference:
        lines.append(f"- Conference: {conference}")
    if first.get("hangoutLink"):
        lines.append(f"- Meet: {first['hangoutLink']}")
    if first.get("htmlLink"):
        lines.append(f"- Link: {first['htmlLink']}")
    visibility = str(first.get("visibility") or "")
    if visibility and visibility != "default":
        lines.append(f"- Visibility: {visibility}")
    description = html_to_text(str(first.get("description") or "")).strip()
    if description:
        lines.append("")
        lines.append("## Description")
        lines.append(description)
    return "\n".join(lines)


def _person(entity: dict) -> str:
    name = str(entity.get("displayName") or "").strip()
    email = str(entity.get("email") or "").strip()
    if name and email:
        return f"{name} <{email}>"
    return name or email


def _conference_link(event: dict) -> str:
    conference = event.get("conferenceData") or {}
    for entry in conference.get("entryPoints", []) or []:
        if entry.get("entryPointType") == "video" and entry.get("uri"):
            return str(entry["uri"])
    return ""


def _when(event: dict) -> str:
    start, all_day = _endpoint(event.get("start") or {})
    end, _ = _endpoint(event.get("end") or {})
    if all_day:
        return f"{start:%a %Y-%m-%d} (all day)"
    if start.date() == end.date():
        return f"{start:%a %Y-%m-%d %H:%M}–{end:%H:%M}"
    return f"{start:%a %Y-%m-%d %H:%M} – {end:%a %Y-%m-%d %H:%M}"


def _endpoint(endpoint: dict) -> tuple[datetime, bool]:
    if endpoint.get("dateTime"):
        stamp = datetime.fromisoformat(str(endpoint["dateTime"]))
        return stamp.astimezone(), False
    if endpoint.get("date"):
        return datetime.fromisoformat(str(endpoint["date"])), True
    return datetime.now(), False


def _start_ts(event: dict) -> float:
    stamp, _ = _endpoint(event.get("start") or {})
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def _parse_iso_ts(value: object) -> float:
    if not value:
        return time.time()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()
