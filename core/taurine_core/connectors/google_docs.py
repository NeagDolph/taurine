"""Google Docs connector (kind="oauth", shares the Gmail token).

Listing uses the Drive API (scope drive.metadata.readonly) because the Docs
API has no list endpoint; content comes from the Docs API (documents.readonly).
One NoteRecord per document with the parsed body (headings preserved as
markdown, lists bulleted, tables flattened) plus a metadata header: owners,
last-modified-by, folder, sharing, and the web link.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import httpx

from taurine_core import config as taurine_config
from taurine_core.connectors.base import content_hash
from taurine_core.connectors.google_common import (
    AUTH_HELP,
    GoogleToken,
    get_json,
    google_env_ready,
)
from taurine_core.ports import ConnectorError
from taurine_core.schemas import NoteRecord

_DRIVE_API = "https://www.googleapis.com/drive/v3"
_DOCS_API = "https://docs.googleapis.com/v1"
_FILE_FIELDS = (
    "id,name,createdTime,modifiedTime,webViewLink,parents,shared,"
    "owners(displayName,emailAddress),lastModifyingUser(displayName,emailAddress),"
    "sharingUser(displayName,emailAddress)"
)


class GoogleDocsConnector:
    connector_id = "google_docs"
    label = "Google Docs"
    kind = "oauth"

    def __init__(self) -> None:
        self._token = GoogleToken("Google Docs")
        self._cached_records: list[NoteRecord] | None = None
        self._cached_at = 0.0
        self._client_factory = lambda: httpx.Client(timeout=30.0)

    def available(self) -> tuple[bool, str]:
        if not google_env_ready():
            return False, f"Google sign-in required — {AUTH_HELP}"
        return True, ""

    def default_config(self) -> dict:
        return {
            "window_days": taurine_config.GDOCS_WINDOW_DAYS,
            "max_docs": taurine_config.GDOCS_MAX_DOCS,
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
        window_days = int(config.get("window_days") or taurine_config.GDOCS_WINDOW_DAYS)
        max_docs = int(config.get("max_docs") or taurine_config.GDOCS_MAX_DOCS)
        try:
            with self._client_factory() as client:
                token = self._token.get(client)
                headers = {"Authorization": f"Bearer {token}"}
                files = _list_documents(client, headers, window_days, max_docs)
                folders = _resolve_folders(client, headers, files)
                records = []
                skipped: list[str] = []
                for file in files:
                    try:
                        records.append(_doc_to_record(client, headers, file, folders))
                    except Exception as exc:  # per-doc fault isolation
                        skipped.append(f"{file.get('name', file.get('id'))}: {exc}")
                if skipped:
                    import logging

                    logging.getLogger("taurine.connectors").warning(
                        "google_docs skipped %d unreadable docs: %s",
                        len(skipped),
                        "; ".join(skipped[:5]),
                    )
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Google Drive/Docs API unreachable: {exc}") from exc
        self._cached_records = records
        self._cached_at = now
        return self._cached_records


def _list_documents(
    client: httpx.Client, headers: dict, window_days: int, max_docs: int
) -> list[dict]:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=window_days)
    ).strftime("%Y-%m-%dT%H:%M:%S")
    query = (
        "mimeType='application/vnd.google-apps.document' "
        f"and trashed=false and modifiedTime > '{cutoff}'"
    )
    files: list[dict] = []
    page_token: str | None = None
    while len(files) < max_docs:
        params: dict = {
            "q": query,
            "orderBy": "modifiedTime desc",
            "pageSize": min(100, max_docs - len(files)),
            "fields": f"nextPageToken,files({_FILE_FIELDS})",
        }
        if page_token:
            params["pageToken"] = page_token
        data = get_json(
            client, f"{_DRIVE_API}/files", headers, params, service="Google Drive"
        )
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return files[:max_docs]


def _resolve_folders(
    client: httpx.Client, headers: dict, files: list[dict]
) -> dict[str, str]:
    """Parent folder id → name, one lookup per unique parent; failures → ''."""
    parent_ids = {
        str(file["parents"][0])
        for file in files
        if file.get("parents")
    }
    names: dict[str, str] = {}
    for parent_id in parent_ids:
        try:
            data = get_json(
                client,
                f"{_DRIVE_API}/files/{parent_id}",
                headers,
                {"fields": "name"},
                service="Google Drive",
            )
            names[parent_id] = str(data.get("name") or "")
        except Exception:
            names[parent_id] = ""
    return names


def _doc_to_record(
    client: httpx.Client, headers: dict, file: dict, folders: dict[str, str]
) -> NoteRecord:
    doc_id = str(file["id"])
    document = get_json(
        client, f"{_DOCS_API}/documents/{doc_id}", headers, {}, service="Google Docs"
    )
    title = str(file.get("name") or document.get("title") or "(untitled doc)")
    body_text = _document_body_text(document)
    header = _metadata_header(file, folders)
    text = (header + "\n\n" + body_text).strip()
    modified = _parse_iso_ts(file.get("modifiedTime"))
    created = _parse_iso_ts(file.get("createdTime")) or modified
    folder = folders.get(str(file["parents"][0]), "") if file.get("parents") else ""
    return NoteRecord(
        note_id=f"google_docs:{doc_id}",
        title=title,
        folder=folder,
        modified=modified,
        created=created,
        content_hash=content_hash(title, text),
        text=text[: taurine_config.CONNECTOR_DOC_MAX_CHARS],
        source="google_docs",
    )


def _metadata_header(file: dict, folders: dict[str, str]) -> str:
    lines: list[str] = []
    owners = ", ".join(
        _person(owner) for owner in file.get("owners", []) if _person(owner)
    )
    if owners:
        lines.append(f"- Owner: {owners}")
    modifier = _person(file.get("lastModifyingUser") or {})
    if modifier:
        stamp = str(file.get("modifiedTime") or "")[:10]
        lines.append(f"- Last edited: {stamp} by {modifier}")
    if file.get("parents"):
        folder = folders.get(str(file["parents"][0]), "")
        if folder:
            lines.append(f"- Folder: {folder}")
    if file.get("shared"):
        sharer = _person(file.get("sharingUser") or {})
        lines.append(f"- Shared{f' by {sharer}' if sharer else ''}")
    if file.get("webViewLink"):
        lines.append(f"- Link: {file['webViewLink']}")
    return "\n".join(lines)


def _person(entity: dict) -> str:
    name = str(entity.get("displayName") or "").strip()
    email = str(entity.get("emailAddress") or "").strip()
    if name and email:
        return f"{name} <{email}>"
    return name or email


_HEADING_PREFIX = {
    "HEADING_1": "# ",
    "HEADING_2": "## ",
    "HEADING_3": "### ",
    "HEADING_4": "#### ",
    "HEADING_5": "##### ",
    "HEADING_6": "###### ",
    "TITLE": "# ",
    "SUBTITLE": "## ",
}


def _document_body_text(document: dict) -> str:
    lines: list[str] = []
    for element in (document.get("body") or {}).get("content", []) or []:
        _render_element(element, lines)
    return "\n".join(lines).strip()


def _render_element(element: dict, lines: list[str]) -> None:
    if "paragraph" in element:
        paragraph = element["paragraph"]
        text = "".join(
            part.get("textRun", {}).get("content", "")
            for part in paragraph.get("elements", []) or []
        ).rstrip("\n")
        if not text.strip():
            return
        style = str((paragraph.get("paragraphStyle") or {}).get("namedStyleType") or "")
        prefix = _HEADING_PREFIX.get(style, "")
        if not prefix and paragraph.get("bullet") is not None:
            depth = int((paragraph["bullet"] or {}).get("nestingLevel") or 0)
            prefix = "  " * depth + "- "
        lines.append(prefix + text)
    elif "table" in element:
        for row in element["table"].get("tableRows", []) or []:
            cells: list[str] = []
            for cell in row.get("tableCells", []) or []:
                cell_lines: list[str] = []
                for cell_element in cell.get("content", []) or []:
                    _render_element(cell_element, cell_lines)
                cells.append(" ".join(cell_lines).strip())
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
    elif "tableOfContents" in element:
        for toc_element in element["tableOfContents"].get("content", []) or []:
            _render_element(toc_element, lines)


def _parse_iso_ts(value: object) -> float:
    if not value:
        return time.time()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()
