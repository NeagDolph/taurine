#!/usr/bin/env python3
"""Deterministic real-model meeting flow gate for a running Taurine sidecar."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))


DEFAULT_BASE_URL = "http://127.0.0.1:8930"
FINAL_SUMMARY_TIMEOUT_S = 180.0
SPARSE_NOTES = """# Commerce launch

- Maya owns the rollout.
- Friday is the target.
"""
SCRIPTED_TRANSCRIPT = [
    {
        "channel": 0,
        "text": "Let's lock the commerce launch plan and fill in the details behind my notes.",
    },
    {
        "channel": 1,
        "text": "The production launch date is Friday, August 7, 2026, after a ten percent canary on Monday.",
    },
    {
        "channel": 0,
        "text": "Maya owns the rollout. What exactly is included in her scope?",
    },
    {
        "channel": 1,
        "text": "Maya owns Shopify activation, the merchant email, and the launch dashboard; Priya is the backup approver.",
    },
    {
        "channel": 0,
        "text": "What are the commercial and reliability constraints?",
    },
    {
        "channel": 1,
        "text": "The approved launch budget is forty-eight thousand dollars and rollback begins if checkout errors exceed one point five percent for ten minutes.",
    },
    {
        "channel": 0,
        "text": "I'll confirm the canary metrics with Maya by Wednesday afternoon.",
    },
    {
        "channel": 1,
        "text": "I will send Priya the final dashboard link and the rollback checklist before noon Thursday.",
    },
]


class Checks:
    def __init__(self) -> None:
        self.failed = False

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        suffix = f" — {detail}" if detail else ""
        print(f"{'PASS' if condition else 'FAIL'}: {label}{suffix}")
        if not condition:
            self.failed = True
        return condition


def suggestions_valid(notes_md: str, suggestions: list[dict[str, Any]]) -> bool:
    for suggestion in suggestions:
        anchor = suggestion.get("anchor")
        occurrence = suggestion.get("occurrence")
        if not isinstance(anchor, str) or not isinstance(occurrence, int):
            return False
        if isinstance(occurrence, bool) or occurrence < 1:
            return False
        if anchor == "":
            if occurrence != 1:
                return False
            continue
        if _occurrences(notes_md, anchor) < occurrence:
            return False
    return True


def _occurrences(text: str, needle: str) -> int:
    count = 0
    start = 0
    while True:
        position = text.find(needle, start)
        if position < 0:
            return count
        count += 1
        start = position + len(needle)


def check_source_cache(meeting_id: str, database: Path) -> bool:
    from taurine_core import config
    from taurine_core.connectors.taurine_meetings import TaurineMeetingsConnector
    from taurine_core.indexer.store import Index

    index = Index(database)
    try:
        connector = TaurineMeetingsConnector(index)
        records = connector.extract({})
        index.replace_source_records(connector.connector_id, records)
        expected_id = f"{config.TAURINE_SOURCE_ID}:{meeting_id}"
        note = index.note(expected_id)
        return note is not None and note.source == config.TAURINE_SOURCE_ID
    finally:
        index.close()


async def run_flow(base_url: str, database: Path) -> int:
    from websockets.asyncio.client import connect

    checks = Checks()
    try:
        created = await asyncio.to_thread(
            _request_json,
            f"{base_url}/meetings",
            method="POST",
            payload={"title": "Deterministic commerce launch"},
        )
        meeting_id = str(created["meeting_id"])
        checks.check("meeting created", bool(meeting_id), meeting_id)

        websocket_url = _websocket_url(base_url, "/session")
        async with connect(websocket_url, open_timeout=15, max_size=None) as websocket:
            initial = _json_object(await asyncio.wait_for(websocket.recv(), timeout=15))
            checks.check("session websocket ready", initial.get("type") == "status")
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "meeting_id": meeting_id,
                        "channels": [
                            {"id": 0, "label": "mic"},
                            {"id": 1, "label": "system"},
                        ],
                    },
                    separators=(",", ":"),
                )
            )
            await websocket.send(
                json.dumps(
                    {"type": "inject_transcript", "chunks": SCRIPTED_TRANSCRIPT},
                    separators=(",", ":"),
                )
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "notepad",
                        "meeting_id": meeting_id,
                        "text_md": SPARSE_NOTES,
                    },
                    separators=(",", ":"),
                )
            )
            await websocket.send('{"type":"stop"}')
            final_event = await _await_final_summary(websocket, meeting_id)

        checks.check(
            "final_summary event received",
            final_event.get("type") == "final_summary",
        )
        meeting = await asyncio.to_thread(
            _request_json, f"{base_url}/meetings/{meeting_id}"
        )
        suggestions = meeting.get("suggestions")
        suggestions = suggestions if isinstance(suggestions, list) else []
        transcript = meeting.get("transcript")
        transcript = transcript if isinstance(transcript, list) else []
        notes_md = str(meeting.get("notes_md") or "")

        checks.check("meeting status is done", meeting.get("status") == "done")
        checks.check(
            "final summary is non-empty",
            bool(str(meeting.get("final_summary_md") or "").strip()),
        )
        checks.check(
            "at least one suggested edit exists",
            len(suggestions) >= 1,
            f"count={len(suggestions)}",
        )
        checks.check(
            "every suggestion anchor and occurrence is valid",
            suggestions_valid(notes_md, suggestions),
        )
        checks.check(
            "transcript is persisted",
            len(transcript) >= len(SCRIPTED_TRANSCRIPT),
            f"count={len(transcript)}",
        )
        source_ok = await asyncio.to_thread(check_source_cache, meeting_id, database)
        checks.check(
            f"taurine source cache contains taurine:{meeting_id}", source_ok
        )
    except Exception as exc:
        checks.check("meeting E2E flow completed", False, f"{type(exc).__name__}: {exc}")
    return 1 if checks.failed else 0


async def _await_final_summary(websocket: Any, meeting_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + FINAL_SUMMARY_TIMEOUT_S
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for final_summary")
        event = _json_object(await asyncio.wait_for(websocket.recv(), timeout=remaining))
        if event.get("meeting_id") != meeting_id:
            continue
        if event.get("type") == "final_summary":
            return event
        if (
            event.get("type") == "final_summary_status"
            and event.get("state") == "error"
        ):
            raise RuntimeError(str(event.get("detail") or "final summary failed"))


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"expected JSON object from {url}")
    return decoded


def _json_object(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    decoded = json.loads(raw)
    return decoded if isinstance(decoded, dict) else {}


def _websocket_url(base_url: str, path: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url.removeprefix("https://").rstrip("/") + path
    return "ws://" + base_url.removeprefix("http://").rstrip("/") + path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TAURINE_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("TAURINE_DB_PATH", "taurine.db")),
    )
    parser.add_argument(
        "--check-source",
        metavar="MEETING_ID",
        help="only extract meetings into the taurine source cache and check this id",
    )
    args = parser.parse_args()
    if args.check_source:
        checks = Checks()
        try:
            ok = check_source_cache(args.check_source, args.db)
            checks.check(
                f"taurine source cache contains taurine:{args.check_source}", ok
            )
        except Exception as exc:
            checks.check(
                "taurine source-cache check completed",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        return 1 if checks.failed else 0
    return asyncio.run(run_flow(args.base_url.rstrip("/"), args.db))


if __name__ == "__main__":
    raise SystemExit(main())
