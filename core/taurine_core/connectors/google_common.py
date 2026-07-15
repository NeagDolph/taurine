"""Shared Google OAuth plumbing for the Calendar and Docs connectors.

Both reuse the Gmail refresh token (GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET /
GMAIL_REFRESH_TOKEN) — one Google identity, one consent. The token must have
been minted with the extra scopes (calendar.readonly, documents.readonly,
drive.metadata.readonly); `python -m taurine_core.connectors.email_auth gmail`
requests all of them.
"""
from __future__ import annotations

import os
import time

import httpx

from taurine_core.ports import ConnectorError

TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTH_HELP = "run: python -m taurine_core.connectors.email_auth gmail"
ENV_KEYS = ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN")


def google_env_ready() -> bool:
    return all(os.environ.get(key) for key in ENV_KEYS)


class GoogleToken:
    """Refresh-token → short-lived access token, cached until near expiry."""

    def __init__(self, service: str) -> None:
        self._service = service
        self._access_token: str | None = None
        self._access_expiry = 0.0

    def get(self, client: httpx.Client) -> str:
        if self._access_token and time.time() < self._access_expiry - 60:
            return self._access_token
        response = client.post(
            TOKEN_URL,
            data={
                "client_id": os.environ["GMAIL_CLIENT_ID"],
                "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
                "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
                "grant_type": "refresh_token",
            },
        )
        if response.status_code != 200:
            raise ConnectorError(
                f"{self._service} token refresh failed ({response.status_code}) — re-{AUTH_HELP}"
            )
        payload = response.json()
        self._access_token = str(payload["access_token"])
        self._access_expiry = time.time() + float(payload.get("expires_in", 3600))
        return self._access_token


def get_json(
    client: httpx.Client, url: str, headers: dict, params: dict, *, service: str
) -> dict:
    response = client.get(url, headers=headers, params=params)
    if response.status_code == 403:
        raise ConnectorError(
            f"{service} API returned 403 — the API may be disabled in the Google Cloud "
            f"project or the token lacks its scope; re-{AUTH_HELP}"
        )
    response.raise_for_status()
    return response.json()
