"""Gmail/Outlook connectors: env gating, API parsing, error posture."""
from __future__ import annotations

import base64
import json

import httpx
import pytest

from taurine_core import pii
from taurine_core.connectors.gmail import GmailConnector
from taurine_core.connectors.outlook import OutlookConnector
from taurine_core.ports import ConnectorError


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(pii, "_pii_model", lambda: None)


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


_GMAIL_ENV = {
    "GMAIL_CLIENT_ID": "cid",
    "GMAIL_CLIENT_SECRET": "csec",
    "GMAIL_REFRESH_TOKEN": "rtok",
}


def _set_gmail_env(monkeypatch) -> None:
    for key, value in _GMAIL_ENV.items():
        monkeypatch.setenv(key, value)


def _clear_gmail_env(monkeypatch) -> None:
    for key in _GMAIL_ENV:
        monkeypatch.delenv(key, raising=False)


def _gmail_message(
    *,
    message_id: str = "m1",
    sent: bool = False,
    body: str = "hey, are we still on for thursday?",
    html: bool = False,
    unsubscribe: bool = False,
    starred: bool = False,
    internal_ms: int = 1_760_000_000_000,
    sender: str = "Alice <alice@example.com>",
    to: str = "me@example.com",
) -> dict:
    headers = [
        {"name": "From", "value": sender},
        {"name": "To", "value": to},
        {"name": "Subject", "value": "Thursday plans"},
    ]
    if unsubscribe:
        headers.append({"name": "List-Unsubscribe", "value": "<mailto:u@x.com>"})
    labels = ["INBOX"]
    if sent:
        labels = ["SENT"]
    if starred:
        labels.append("STARRED")
    mime = "text/html" if html else "text/plain"
    payload_body = f"<p>{body}</p>" if html else body
    return {
        "id": message_id,
        "labelIds": labels,
        "internalDate": str(internal_ms),
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": headers,
            "parts": [
                {"mimeType": mime, "body": {"data": _b64url(payload_body)}},
            ],
        },
    }


def _gmail_transport(threads: list[dict], *, fail_get: int | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "oauth2.googleapis.com/token" in url:
            assert b"refresh_token" in request.read()
            return httpx.Response(200, json={"access_token": "atok", "expires_in": 3600})
        if url.endswith("/threads") or "/threads?" in url:
            assert request.headers["Authorization"] == "Bearer atok"
            assert "newer_than" in request.url.params["q"]
            return httpx.Response(
                200, json={"threads": [{"id": t["id"]} for t in threads]}
            )
        for thread in threads:
            if f"/threads/{thread['id']}" in url:
                if fail_get is not None:
                    return httpx.Response(fail_get)
                return httpx.Response(200, json=thread)
        raise AssertionError(f"unexpected request: {url}")

    return httpx.MockTransport(handler)


def _test_config(connector) -> dict:
    config = connector.default_config()
    config["contacts_glob"] = "/nonexistent/**/*.abcddb"  # never touch the real AddressBook
    return config


def _extract(connector: GmailConnector, transport: httpx.MockTransport):
    connector._client_factory = lambda: httpx.Client(transport=transport)
    return connector.extract(_test_config(connector))


def test_gmail_available_requires_env(monkeypatch) -> None:
    _clear_gmail_env(monkeypatch)
    available, detail = GmailConnector().available()
    assert not available
    assert "Sign in required" in detail
    assert "email_auth gmail" in detail


def test_gmail_available_with_env(monkeypatch) -> None:
    _set_gmail_env(monkeypatch)
    assert GmailConnector().available() == (True, "")


def test_gmail_extract_without_env_raises(monkeypatch) -> None:
    _clear_gmail_env(monkeypatch)
    with pytest.raises(ConnectorError, match="sign-in required"):
        GmailConnector().extract({})


def test_gmail_extract_end_to_end(monkeypatch) -> None:
    _set_gmail_env(monkeypatch)
    thread = {
        "id": "t100",
        "messages": [
            _gmail_message(message_id="m1", internal_ms=1_760_000_000_000),
            _gmail_message(
                message_id="m2",
                sent=True,
                body="yes! see you then",
                internal_ms=1_760_000_060_000,
                sender="Me <me@example.com>",
                to="alice@example.com",
            ),
        ],
    }
    monkeypatch.setattr("time.time", lambda: 1_760_000_100.0)
    records = _extract(GmailConnector(), _gmail_transport([thread]))
    assert len(records) == 1
    record = records[0]
    assert record.note_id == "gmail:t100"
    assert record.source == "gmail"
    assert "[Alice] hey, are we still on for thursday?" in record.text
    assert "[me] yes! see you then" in record.text
    assert record.title == "Email: Thursday plans"


def test_gmail_html_only_body_converted(monkeypatch) -> None:
    _set_gmail_env(monkeypatch)
    thread = {
        "id": "t200",
        "messages": [
            _gmail_message(html=True, sent=True, body="rendered <b>fine</b>"),
        ],
    }
    monkeypatch.setattr("time.time", lambda: 1_760_000_100.0)
    records = _extract(GmailConnector(), _gmail_transport([thread]))
    assert "rendered fine" in records[0].text
    assert "<b>" not in records[0].text


def test_gmail_unsubscribe_thread_funneled_out(monkeypatch) -> None:
    _set_gmail_env(monkeypatch)
    thread = {"id": "t300", "messages": [_gmail_message(unsubscribe=True)]}
    monkeypatch.setattr("time.time", lambda: 1_760_000_100.0)
    assert _extract(GmailConnector(), _gmail_transport([thread])) == []


def test_gmail_401_raises_connector_error(monkeypatch) -> None:
    _set_gmail_env(monkeypatch)
    thread = {"id": "t400", "messages": [_gmail_message()]}
    with pytest.raises(ConnectorError, match="re-run"):
        _extract(GmailConnector(), _gmail_transport([thread], fail_get=401))


def test_gmail_extract_cached_within_interval(monkeypatch) -> None:
    _set_gmail_env(monkeypatch)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if "token" in str(request.url):
            return httpx.Response(200, json={"access_token": "atok", "expires_in": 3600})
        return httpx.Response(200, json={"threads": []})

    connector = GmailConnector()
    connector._client_factory = lambda: httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("time.time", lambda: 1_760_000_100.0)
    connector.extract(_test_config(connector))
    first_calls = calls["count"]
    connector.extract(_test_config(connector))
    assert calls["count"] == first_calls  # second extract served from cache


# ── Outlook ────────────────────────────────────────────────────────────

_OUTLOOK_ENV = {"OUTLOOK_CLIENT_ID": "cid", "OUTLOOK_REFRESH_TOKEN": "rtok"}


def _set_outlook_env(monkeypatch) -> None:
    for key, value in _OUTLOOK_ENV.items():
        monkeypatch.setenv(key, value)


def _outlook_message(
    *,
    message_id: str = "om1",
    conversation: str = "conv1",
    sender: str = "alice@example.com",
    sender_name: str = "Alice",
    body: str = "<p>lunch friday?</p>",
    received: str = "2026-07-07T12:00:00Z",
    classification: str = "focused",
    is_draft: bool = False,
    folder: str = "inboxid",
    flagged: bool = False,
) -> dict:
    return {
        "id": message_id,
        "conversationId": conversation,
        "subject": "Friday lunch",
        "from": {"emailAddress": {"name": sender_name, "address": sender}},
        "toRecipients": [{"emailAddress": {"name": "Me", "address": "me@corp.com"}}],
        "receivedDateTime": received,
        "isDraft": is_draft,
        "inferenceClassification": classification,
        "flag": {"flagStatus": "flagged" if flagged else "notFlagged"},
        "importance": "normal",
        "parentFolderId": folder,
        "body": {"contentType": "html", "content": body},
    }


def _outlook_transport(messages: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "login.microsoftonline.com" in url:
            return httpx.Response(
                200,
                json={"access_token": "atok", "expires_in": 3600, "refresh_token": "rtok2"},
            )
        assert request.headers["Authorization"] == "Bearer atok"
        if url.endswith("/me") or "/me?" in url:
            return httpx.Response(200, json={"mail": "me@corp.com", "userPrincipalName": "me@corp.com"})
        if "/mailFolders/junkemail" in url:
            return httpx.Response(200, json={"id": "junkid"})
        if "/mailFolders/deleteditems" in url:
            return httpx.Response(200, json={"id": "delid"})
        if "/me/messages" in url:
            return httpx.Response(200, json={"value": messages})
        raise AssertionError(f"unexpected request: {url}")

    return httpx.MockTransport(handler)


def _extract_outlook(messages: list[dict]):
    connector = OutlookConnector()
    connector._client_factory = lambda: httpx.Client(transport=_outlook_transport(messages))
    return connector.extract(_test_config(connector))


def test_outlook_available_requires_env(monkeypatch) -> None:
    for key in _OUTLOOK_ENV:
        monkeypatch.delenv(key, raising=False)
    available, detail = OutlookConnector().available()
    assert not available
    assert "email_auth outlook" in detail


def test_outlook_groups_by_conversation_and_detects_me(monkeypatch) -> None:
    _set_outlook_env(monkeypatch)
    monkeypatch.setattr("time.time", lambda: 1_783_400_000.0)  # just after 2026-07-07
    messages = [
        _outlook_message(message_id="a1", received="2026-07-06T10:00:00Z"),
        _outlook_message(
            message_id="a2",
            sender="me@corp.com",
            sender_name="Me",
            body="<p>yes, noon works</p>",
            received="2026-07-06T11:00:00Z",
        ),
        _outlook_message(
            message_id="b1",
            conversation="conv2",
            sender="bob@corp.com",
            sender_name="Bob",
            body="<p>standup moved</p>",
            received="2026-07-07T09:00:00Z",
        ),
        _outlook_message(
            message_id="b2",
            conversation="conv2",
            sender="me@corp.com",
            sender_name="Me",
            body="<p>thanks for the heads up</p>",
            received="2026-07-07T09:30:00Z",
        ),
    ]
    records = _extract_outlook(messages)
    by_id = {record.note_id: record for record in records}
    assert set(by_id) == {"outlook:conv1", "outlook:conv2"}
    conv1 = by_id["outlook:conv1"].text
    assert "[Alice] lunch friday?" in conv1
    assert "[me] yes, noon works" in conv1


def test_outlook_skips_drafts_and_junk(monkeypatch) -> None:
    _set_outlook_env(monkeypatch)
    monkeypatch.setattr("time.time", lambda: 1_783_400_000.0)
    messages = [
        _outlook_message(message_id="d1", is_draft=True),
        _outlook_message(message_id="j1", conversation="conv9", folder="junkid"),
    ]
    assert _extract_outlook(messages) == []


# ── auth CLI helpers ───────────────────────────────────────────────────


def test_pkce_pair_is_valid() -> None:
    import hashlib

    from taurine_core.connectors.email_auth import _pkce_pair

    verifier, challenge = _pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert set(verifier) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    assert challenge == expected


def test_outlook_bulk_classification_funneled_out(monkeypatch) -> None:
    _set_outlook_env(monkeypatch)
    monkeypatch.setattr("time.time", lambda: 1_783_400_000.0)
    messages = [
        _outlook_message(message_id="o1", conversation="convX", classification="other"),
    ]
    assert _extract_outlook(messages) == []
