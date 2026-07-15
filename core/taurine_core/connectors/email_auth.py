"""One-time OAuth bootstrap for the email connectors (phase 1).

    python -m taurine_core.connectors.email_auth gmail
    python -m taurine_core.connectors.email_auth outlook

Gmail: RFC 8252 loopback flow with PKCE against your own Desktop-type OAuth
client (Google Cloud Console → APIs & Services → Credentials). Outlook:
device-code flow against your own Azure app registration (public client,
"Allow public client flows" enabled, delegated Mail.Read).

Prints the env lines to paste into core/.env. Tokens go to stdout only —
never logged, never persisted by this tool.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import os
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

# One consent covers every Google connector: mail, calendar events, and docs
# (drive.metadata.readonly is required to LIST docs; documents.readonly only
# permits reading them by id).
_GMAIL_SCOPE = " ".join(
    (
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/documents.readonly",
        "https://www.googleapis.com/auth/drive.metadata.readonly",
    )
)
_GMAIL_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
_MS_DEVICECODE_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/devicecode"
_MS_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
_MS_SCOPE = "offline_access Mail.Read User.Read"


def _pkce_pair() -> tuple[str, str]:
    """(verifier, S256 challenge) per RFC 7636."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _env_or_prompt(key: str, prompt: str) -> str:
    value = os.environ.get(key, "").strip()
    if value:
        print(f"using {key} from environment")
        return value
    return input(f"{prompt}: ").strip()


def _print_env(lines: dict[str, str]) -> None:
    print("\nAdd these lines to core/.env (then enable the connector in the app):\n")
    for key, value in lines.items():
        print(f"{key}={value}")
    print()


# ── Gmail: loopback + PKCE ─────────────────────────────────────────────


class _CodeHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None
    error: str | None = None

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _CodeHandler.code = (params.get("code") or [None])[0]
        _CodeHandler.error = (params.get("error") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Taurine: you can close this tab.</h2></body></html>"
        )

    def log_message(self, *args) -> None:  # silence request logging
        pass


def gmail_flow() -> None:
    client_id = _env_or_prompt("GMAIL_CLIENT_ID", "Google OAuth client id")
    client_secret = _env_or_prompt("GMAIL_CLIENT_SECRET", "Google OAuth client secret")
    verifier, challenge = _pkce_pair()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = http.server.HTTPServer(("127.0.0.1", port), _CodeHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    redirect_uri = f"http://127.0.0.1:{port}"
    auth_url = _GMAIL_AUTH_URL + "?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": _GMAIL_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": secrets.token_urlsafe(16),
        }
    )
    print("\nOpening browser for Google consent…\nIf it doesn't open, visit:\n" + auth_url)
    webbrowser.open(auth_url)
    thread.join(timeout=300)
    server.server_close()
    if _CodeHandler.error:
        sys.exit(f"consent failed: {_CodeHandler.error}")
    if not _CodeHandler.code:
        sys.exit("timed out waiting for the OAuth redirect (5 minutes)")

    response = httpx.post(
        _GMAIL_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": _CodeHandler.code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        sys.exit(f"token exchange failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        sys.exit(
            "Google returned no refresh token — remove the app's prior grant at "
            "https://myaccount.google.com/permissions and re-run"
        )
    _print_env(
        {
            "GMAIL_CLIENT_ID": client_id,
            "GMAIL_CLIENT_SECRET": client_secret,
            "GMAIL_REFRESH_TOKEN": str(refresh_token),
        }
    )
    print(
        "Note: if your OAuth client is in 'Testing' publishing status, Google "
        "expires this token after 7 days — publish the app (unverified is fine "
        "for personal use) to make it persist."
    )


# ── Outlook: device code ───────────────────────────────────────────────


def outlook_flow() -> None:
    client_id = _env_or_prompt("OUTLOOK_CLIENT_ID", "Azure app (client) id")
    start = httpx.post(
        _MS_DEVICECODE_URL,
        data={"client_id": client_id, "scope": _MS_SCOPE},
        timeout=30.0,
    )
    if start.status_code != 200:
        sys.exit(f"device-code request failed ({start.status_code}): {start.text[:300]}")
    device = start.json()
    print(f"\n{device.get('message') or ''}\n")

    interval = float(device.get("interval", 5))
    deadline = time.time() + float(device.get("expires_in", 900))
    while time.time() < deadline:
        time.sleep(interval)
        response = httpx.post(
            _MS_TOKEN_URL,
            data={
                "client_id": client_id,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device["device_code"],
            },
            timeout=30.0,
        )
        payload = response.json()
        if response.status_code == 200:
            _print_env(
                {
                    "OUTLOOK_CLIENT_ID": client_id,
                    "OUTLOOK_REFRESH_TOKEN": str(payload["refresh_token"]),
                }
            )
            return
        error = str(payload.get("error") or "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        sys.exit(f"device-code flow failed: {error}: {payload.get('error_description', '')[:300]}")
    sys.exit("device-code flow timed out")


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass
    provider = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if provider == "gmail":
        gmail_flow()
    elif provider == "outlook":
        outlook_flow()
    else:
        sys.exit("usage: python -m taurine_core.connectors.email_auth gmail|outlook")


if __name__ == "__main__":
    main()
