"""Gmail connector (kind="oauth", phase 1: env-var tokens).

Env: GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN — produced by
`python -m taurine_core.connectors.email_auth gmail` against your own
Desktop-type OAuth client (scope gmail.readonly).

Stage-0 filtering happens server-side in the threads.list `q` expression;
the shared funnel (email_common) does the rest. Extract results are cached
in-process for EMAIL_SYNC_MIN_INTERVAL_S so the 300s sync loop doesn't
refetch the mailbox every pass — unchanged records are a no-op upstream via
content_hash anyway.
"""
from __future__ import annotations

import base64
import os
import time
from email.utils import getaddresses, parseaddr

import httpx

from taurine_core import config as taurine_config
from taurine_core.connectors.email_common import (
    EmailMessage,
    EmailThread,
    funnel_threads,
    html_to_text,
    log_funnel,
    thread_to_record,
)
from taurine_core.connectors.imessage import _contact_names
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_API = "https://gmail.googleapis.com/gmail/v1/users/me"
_AUTH_HELP = "run: python -m taurine_core.connectors.email_auth gmail"
_ENV_KEYS = ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN")


class GmailConnector:
    connector_id = "gmail"
    label = "Gmail"
    kind = "oauth"

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._access_expiry = 0.0
        self._cached_records: list[NoteRecord] | None = None
        self._cached_at = 0.0
        self.ingest_salience_scores: dict[str, float] = {}
        self._client_factory = lambda: httpx.Client(timeout=30.0)

    def available(self) -> tuple[bool, str]:
        # env presence only — this runs on every UI poll, no network allowed
        if not all(os.environ.get(key) for key in _ENV_KEYS):
            return False, f"Sign in required — {_AUTH_HELP}"
        return True, ""

    def default_config(self) -> dict:
        return {
            "window_days": taurine_config.EMAIL_WINDOW_DAYS,
            "contacts_glob": taurine_config.CONTACTS_DB_GLOB,
        }

    def extract(self, config: dict) -> list[NoteRecord]:
        if not all(os.environ.get(key) for key in _ENV_KEYS):
            raise ConnectorError(f"Gmail sign-in required — {_AUTH_HELP}")
        now = time.time()
        if (
            self._cached_records is not None
            and now - self._cached_at < taurine_config.EMAIL_SYNC_MIN_INTERVAL_S
        ):
            return self._cached_records
        window_days = int(config.get("window_days") or taurine_config.EMAIL_WINDOW_DAYS)
        contacts = _contact_names({"contacts_glob": config.get("contacts_glob")})
        try:
            with self._client_factory() as client:
                token = self._token(client)
                threads = _fetch_threads(client, token, window_days)
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Gmail API unreachable: {exc}") from exc
        survivors, report = funnel_threads(threads, contacts=contacts)
        log_funnel(self.connector_id, report)
        self.ingest_salience_scores = {
            f"{self.connector_id}:{thread_id}": score
            for thread_id, score in report.scores.items()
        }
        records = [thread_to_record(thread, self.connector_id) for thread in survivors]
        self._cached_records = [record for record in records if record is not None]
        self._cached_at = now
        return self._cached_records

    def _token(self, client: httpx.Client) -> str:
        if self._access_token and time.time() < self._access_expiry - 60:
            return self._access_token
        response = client.post(
            _TOKEN_URL,
            data={
                "client_id": os.environ["GMAIL_CLIENT_ID"],
                "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
                "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
                "grant_type": "refresh_token",
            },
        )
        if response.status_code != 200:
            raise ConnectorError(
                f"Gmail token refresh failed ({response.status_code}) — re-{_AUTH_HELP}"
            )
        payload = response.json()
        self._access_token = str(payload["access_token"])
        self._access_expiry = time.time() + float(payload.get("expires_in", 3600))
        return self._access_token


def _fetch_threads(client: httpx.Client, token: str, window_days: int) -> list[EmailThread]:
    headers = {"Authorization": f"Bearer {token}"}
    query = (
        f"newer_than:{window_days}d -in:spam -in:trash "
        "-category:promotions -category:social -category:updates -category:forums"
    )
    thread_ids: list[str] = []
    page_token: str | None = None
    while len(thread_ids) < taurine_config.EMAIL_FETCH_CEILING:
        params: dict = {"q": query, "maxResults": 100}
        if page_token:
            params["pageToken"] = page_token
        data = _get_json(client, f"{_API}/threads", headers, params)
        thread_ids.extend(str(item["id"]) for item in data.get("threads", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    threads: list[EmailThread] = []
    for thread_id in thread_ids[: taurine_config.EMAIL_FETCH_CEILING]:
        data = _get_json(client, f"{_API}/threads/{thread_id}", headers, {"format": "full"})
        thread = _parse_thread(data)
        if thread is not None:
            threads.append(thread)
    return threads


def _get_json(client: httpx.Client, url: str, headers: dict, params: dict) -> dict:
    for attempt in (0, 1):
        response = client.get(url, headers=headers, params=params)
        if response.status_code == 429 and attempt == 0:
            time.sleep(min(float(response.headers.get("Retry-After", 2)), 30.0))
            continue
        if response.status_code == 401:
            raise ConnectorError(f"Gmail authorization expired — re-{_AUTH_HELP}")
        if response.status_code != 200:
            raise ConnectorError(
                f"Gmail API error {response.status_code}: {response.text[:200]}"
            )
        return response.json()
    raise ConnectorError("Gmail API rate-limited after retry")


def _parse_thread(data: dict) -> EmailThread | None:
    messages: list[EmailMessage] = []
    subject = ""
    flagged = False
    for message in data.get("messages", []):
        labels = set(message.get("labelIds") or [])
        if "DRAFT" in labels:
            continue
        payload = message.get("payload") or {}
        headers = {
            str(h.get("name", "")).lower(): str(h.get("value", ""))
            for h in payload.get("headers", [])
        }
        subject = subject or headers.get("subject", "")
        flagged = flagged or bool({"STARRED", "IMPORTANT"} & labels)
        name, address = parseaddr(headers.get("from", ""))
        recipients = tuple(
            addr for _, addr in getaddresses([headers.get("to", "")]) if addr
        )
        messages.append(
            EmailMessage(
                text=_body_text(payload),
                sender_name=name,
                sender_email=address,
                timestamp=float(message.get("internalDate", 0)) / 1000.0,
                is_from_me="SENT" in labels,
                has_unsubscribe=(
                    "list-unsubscribe" in headers
                    or headers.get("precedence", "").lower() in ("bulk", "list")
                ),
                recipients=recipients,
            )
        )
    if not messages:
        return None
    return EmailThread(
        thread_id=str(data.get("id", "")),
        subject=subject,
        messages=messages,
        flagged=flagged,
    )


def _body_text(payload: dict) -> str:
    plain, html = _walk_parts(payload)
    if plain:
        return plain
    if html:
        return html_to_text(html)
    return ""


def _walk_parts(part: dict) -> tuple[str, str]:
    """(text/plain, text/html) — first of each found in a depth-first walk."""
    mime = str(part.get("mimeType", ""))
    data = (part.get("body") or {}).get("data")
    if data and mime.startswith("text/"):
        decoded = _b64url_decode(str(data))
        if mime == "text/plain":
            return decoded, ""
        if mime == "text/html":
            return "", decoded
    plain = html = ""
    for child in part.get("parts") or []:
        child_plain, child_html = _walk_parts(child)
        plain = plain or child_plain
        html = html or child_html
        if plain:
            break
    return plain, html


def _b64url_decode(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        return ""
