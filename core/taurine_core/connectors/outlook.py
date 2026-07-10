"""Outlook connector (kind="oauth", phase 1: env-var tokens).

Env: OUTLOOK_CLIENT_ID, OUTLOOK_REFRESH_TOKEN — produced by
`python -m taurine_core.connectors.email_auth outlook` (device-code flow
against your own Azure app registration, public client, no secret).

Graph has no first-class thread fetch: /me/messages is paged by
receivedDateTime and grouped client-side by conversationId. Messages in
Junk Email / Deleted Items and drafts are skipped; Focused Inbox's "other"
classification marks a thread bulk for the funnel. Extract results are
cached in-process for EMAIL_SYNC_MIN_INTERVAL_S (same rationale as Gmail).

Microsoft rotates refresh tokens on use; the old token stays valid until
its own expiry, so env storage works for personal use — if syncs start
failing with 401s, re-run the auth CLI.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

from taurine_core import config as taurine_config
from taurine_core.connectors.email_common import (
    EmailMessage,
    EmailThread,
    funnel_threads,
    html_to_text,
    log_funnel,
    threads_to_records,
)
from taurine_core.connectors.imessage import _contact_names
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord

_LOGGER = logging.getLogger("taurine.sync")
_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
_GRAPH = "https://graph.microsoft.com/v1.0"
_SCOPE = "offline_access Mail.Read User.Read"
_AUTH_HELP = "run: python -m taurine_core.connectors.email_auth outlook"
_ENV_KEYS = ("OUTLOOK_CLIENT_ID", "OUTLOOK_REFRESH_TOKEN")
_SELECT = (
    "id,conversationId,subject,from,toRecipients,receivedDateTime,isDraft,"
    "inferenceClassification,flag,importance,parentFolderId,body"
)


class OutlookConnector:
    connector_id = "outlook"
    label = "Outlook"
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
            raise ConnectorError(f"Outlook sign-in required — {_AUTH_HELP}")
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
            raise ConnectorError(f"Outlook API unreachable: {exc}") from exc
        survivors, report = funnel_threads(threads, contacts=contacts)
        log_funnel(self.connector_id, report)
        self.ingest_salience_scores = {
            f"{self.connector_id}:{thread_id}": score
            for thread_id, score in report.scores.items()
        }
        self._cached_records = threads_to_records(survivors, self.connector_id)
        self._cached_at = now
        return self._cached_records

    def _token(self, client: httpx.Client) -> str:
        if self._access_token and time.time() < self._access_expiry - 60:
            return self._access_token
        response = client.post(
            _TOKEN_URL,
            data={
                "client_id": os.environ["OUTLOOK_CLIENT_ID"],
                "refresh_token": os.environ["OUTLOOK_REFRESH_TOKEN"],
                "grant_type": "refresh_token",
                "scope": _SCOPE,
            },
        )
        if response.status_code != 200:
            raise ConnectorError(
                f"Outlook token refresh failed ({response.status_code}) — re-{_AUTH_HELP}"
            )
        payload = response.json()
        if payload.get("refresh_token") and payload["refresh_token"] != os.environ.get(
            "OUTLOOK_REFRESH_TOKEN"
        ):
            # never log the token itself; the old one stays valid until expiry
            _LOGGER.info(
                "Outlook rotated the refresh token; if syncs eventually 401, %s",
                _AUTH_HELP,
            )
        self._access_token = str(payload["access_token"])
        self._access_expiry = time.time() + float(payload.get("expires_in", 3600))
        return self._access_token


def _fetch_threads(client: httpx.Client, token: str, window_days: int) -> list[EmailThread]:
    headers = {"Authorization": f"Bearer {token}"}
    me = _get_json(client, f"{_GRAPH}/me", headers, {"$select": "mail,userPrincipalName"})
    my_address = str(me.get("mail") or me.get("userPrincipalName") or "").lower()
    excluded_folders = _excluded_folder_ids(client, headers)

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    params: dict | None = {
        "$select": _SELECT,
        "$filter": f"receivedDateTime ge {cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "$orderby": "receivedDateTime desc",
        "$top": 100,
    }
    url = f"{_GRAPH}/me/messages"
    threads: dict[str, EmailThread] = {}
    fetched = 0
    ceiling = taurine_config.EMAIL_FETCH_CEILING * 4  # messages, not threads
    while url and fetched < ceiling:
        data = _get_json(client, url, headers, params)
        for raw in data.get("value", []):
            fetched += 1
            _add_message(threads, raw, my_address, excluded_folders)
        url = data.get("@odata.nextLink")
        params = None  # nextLink already carries the query
    return list(threads.values())


def _excluded_folder_ids(client: httpx.Client, headers: dict) -> set[str]:
    excluded: set[str] = set()
    for well_known in ("junkemail", "deleteditems"):
        try:
            folder = _get_json(
                client, f"{_GRAPH}/me/mailFolders/{well_known}", headers, {"$select": "id"}
            )
        except ConnectorError:
            continue  # folder lookup is best-effort; the funnel still guards
        folder_id = str(folder.get("id") or "")
        if folder_id:
            excluded.add(folder_id)
    return excluded


def _add_message(
    threads: dict[str, EmailThread],
    raw: dict,
    my_address: str,
    excluded_folders: set[str],
) -> None:
    if raw.get("isDraft"):
        return
    if str(raw.get("parentFolderId") or "") in excluded_folders:
        return
    conversation = str(raw.get("conversationId") or raw.get("id") or "")
    if not conversation:
        return
    sender = (raw.get("from") or {}).get("emailAddress") or {}
    address = str(sender.get("address") or "")
    is_from_me = bool(my_address) and address.lower() == my_address
    body = raw.get("body") or {}
    content = str(body.get("content") or "")
    if str(body.get("contentType") or "").lower() == "html":
        content = html_to_text(content)
    recipients = tuple(
        str((item.get("emailAddress") or {}).get("address") or "")
        for item in raw.get("toRecipients") or []
    )
    message = EmailMessage(
        text=content,
        sender_name=str(sender.get("name") or ""),
        sender_email=address,
        timestamp=_parse_timestamp(str(raw.get("receivedDateTime") or "")),
        is_from_me=is_from_me,
        recipients=tuple(address for address in recipients if address),
    )
    thread = threads.get(conversation)
    if thread is None:
        thread = EmailThread(
            thread_id=conversation,
            subject=str(raw.get("subject") or ""),
            bulk_classified=True,  # cleared by any focused/self message below
        )
        threads[conversation] = thread
    thread.messages.append(message)
    classification = str(raw.get("inferenceClassification") or "").lower()
    if is_from_me or classification != "other":
        thread.bulk_classified = False
    flag_status = str((raw.get("flag") or {}).get("flagStatus") or "").lower()
    if flag_status == "flagged" or str(raw.get("importance") or "").lower() == "high":
        thread.flagged = True


def _parse_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _get_json(client: httpx.Client, url: str, headers: dict, params: dict | None) -> dict:
    for attempt in (0, 1):
        response = client.get(url, headers=headers, params=params)
        if response.status_code == 429 and attempt == 0:
            time.sleep(min(float(response.headers.get("Retry-After", 2)), 30.0))
            continue
        if response.status_code == 401:
            raise ConnectorError(f"Outlook authorization expired — re-{_AUTH_HELP}")
        if response.status_code != 200:
            raise ConnectorError(
                f"Outlook API error {response.status_code}: {response.text[:200]}"
            )
        return response.json()
    raise ConnectorError("Outlook API rate-limited after retry")
