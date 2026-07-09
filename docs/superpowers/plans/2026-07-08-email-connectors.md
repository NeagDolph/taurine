# Email Connectors (Gmail + Outlook) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two `kind="oauth"` connectors (`gmail`, `outlook`) that ingest email threads through a deterministic relevance funnel with local PII redaction, authenticated via env-var refresh tokens produced by a bootstrap CLI.

**Architecture:** Provider-agnostic funnel/rendering in `connectors/email_common.py`; PII redaction (regex + GLiNER2) in `pii.py`; thin per-provider fetch/parse modules; a `python -m` auth CLI. Everything upstream of enrichment is LLM-free. Redaction runs before `content_hash` so raw PII never lands anywhere.

**Tech Stack:** Python 3.12, httpx (existing dep), stdlib `http.server`/`webbrowser` for OAuth bootstrap, GLiNER2 (existing vendored model), pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-email-connectors-design.md`

## Global Constraints

- No new dependencies.
- No LLM calls anywhere in the funnel/redaction path.
- Secrets are env vars only (`core/.env`); never written to `taurine.db` or logs.
- Redact PII **before** computing `content_hash`.
- `gmail` and `outlook` must appear in both `SOURCE_RANK` (0.85) and `RECENCY_HALF_LIFE_DAYS` (30.0) — `test_every_connector_has_explicit_source_priors` enforces this.
- `available()` must never raise and must not make network calls (it runs on every UI poll).
- Tests must never load the real GLiNER2 model (monkeypatch `taurine_core.pii._pii_model`).
- Run tests with `cd core && uv run pytest tests/<file> -q`.

---

### Task 1: Config knobs + source priors

**Files:**
- Modify: `core/taurine_core/config.py` (email section after the iMessage block; PII schema after the GLiNER2 section; `SOURCE_RANK` / `RECENCY_HALF_LIFE_DAYS` entries)

**Interfaces:**
- Produces: `EMAIL_WINDOW_DAYS=120`, `EMAIL_MAX_THREADS=400`, `EMAIL_FETCH_CEILING=1500`, `EMAIL_MAX_MESSAGES=25`, `EMAIL_MESSAGE_MAX_CHARS=2000`, `EMAIL_SYNC_MIN_INTERVAL_S=3600.0`, `EMAIL_NOREPLY_LOCALPARTS` (frozenset), `EMAIL_TRANSACTIONAL_SUBJECT_RE` (str), `EMAIL_SCORE_WEIGHTS` (dict), `EMAIL_RECENCY_SCORE_HALF_LIFE_DAYS=30.0`, `EMAIL_PII_GLINER_THRESHOLD=0.35`, `GLINER2_PII_SCHEMA` (dict), `SOURCE_RANK["gmail"|"outlook"]=0.85`, `RECENCY_HALF_LIFE_DAYS["gmail"|"outlook"]=30.0`.

- [ ] **Step 1: Add the email section to config.py** (full content in the implementation; values above are normative)
- [ ] **Step 2: Add `gmail`/`outlook` to both prior dicts**
- [ ] **Step 3: Run `uv run pytest tests/test_index_rank.py -q` — still passes (connectors not yet in catalog)**
- [ ] **Step 4: Commit** `feat: email connector config knobs and source priors`

### Task 2: PII redaction (`pii.py`) + shared model accessor

**Files:**
- Create: `core/taurine_core/pii.py`
- Modify: `core/taurine_core/live/spans.py` (add `shared_model()` public accessor returning `_get_model()`)
- Test: `core/tests/test_email_pii.py`

**Interfaces:**
- Produces: `redact_pii(text: str) -> str` — never raises; regex layer (Luhn-validated cards → `[redacted-card]`, SSNs → `[redacted-ssn]`, keyword-context account numbers → `[redacted-account]`, verification codes → `[redacted-code]`) then GLiNER2 layer with `GLINER2_PII_SCHEMA` (spans ≥4 chars replaced with `[redacted-{label-with-dashes}]`). Degrades to regex-only when the model is unavailable. Internal seam `_pii_model()` for tests.
- Consumes: `live.spans.shared_model()` (cached GLiNER2 instance).

- [ ] **Step 1: Write failing tests** — Luhn-valid card redacted / Luhn-invalid preserved; both SSN forms; account/routing context; OTP codes; names+street addresses preserved; GLiNER span replacement via fake model; regex-only when `_pii_model` returns None.
- [ ] **Step 2: Run — fails (module missing)**
- [ ] **Step 3: Implement `pii.py` + `shared_model()`**
- [ ] **Step 4: `uv run pytest tests/test_email_pii.py -q` — PASS**
- [ ] **Step 5: Commit** `feat: PII redaction — regex layer + GLiNER2 PII schema`

### Task 3: Funnel + rendering (`email_common.py`)

**Files:**
- Create: `core/taurine_core/connectors/email_common.py`
- Test: `core/tests/test_email_common.py`

**Interfaces:**
- Produces:
  - `EmailMessage(text, sender_name, sender_email, timestamp, is_from_me, has_unsubscribe=False, recipients=())` (frozen dataclass)
  - `EmailThread(thread_id, subject, messages, flagged=False, bulk_classified=False)` (dataclass)
  - `funnel_threads(threads, *, contacts: dict[str, str], now: float | None = None) -> tuple[list[EmailThread], FunnelReport]` — sorts messages oldest-first in place; drop order: empty → window → (unless replied) unsubscribe/bulk → noreply-only → transactional subject → unknown single-message sender; scores survivors and caps at `EMAIL_MAX_THREADS`. `FunnelReport(dropped: dict[str, int], kept: int)`.
  - `thread_to_record(thread, connector_id) -> NoteRecord | None` — newest `EMAIL_MAX_MESSAGES`, per-message `clean_body` → `redact_pii` → `EMAIL_MESSAGE_MAX_CHARS` cap, `[me]`/`[Name]` lines, oldest lines dropped to fit `CONNECTOR_DOC_MAX_CHARS`, `note_id=f"{connector_id}:{thread_id}"`, title `Email: {subject minus Re:/Fwd: prefixes}` (participants fallback), folder `"Email"`, hash = sha256(transcript).
  - `clean_body(text) -> str` — cuts at `On … wrote:`, `-----Original Message-----`, Outlook `From:…Sent:` blocks, `__________` rules, `-- ` signatures; drops `>`-quoted lines; collapses whitespace.
  - `html_to_text(markup) -> str`; `log_funnel(connector_id, report)`.
- Consumes: `pii.redact_pii`, config knobs from Task 1.

- [ ] **Step 1: Write failing tests** — one test per drop rule + replied-rescue for each conditional rule; correspondent-set (recipients of your sent mail + senders in replied threads count as known); score ordering (replied recent thread beats unreplied old one) and cap with `EMAIL_MAX_THREADS` monkeypatched to 2; quote/signature stripping cases; `[me]` labeling; Re:/Fwd: title stripping; hash stability; per-message and per-doc caps; `html_to_text`. Fixture monkeypatches `taurine_core.pii._pii_model` to `lambda: None`.
- [ ] **Step 2: Run — fails**
- [ ] **Step 3: Implement**
- [ ] **Step 4: `uv run pytest tests/test_email_common.py -q` — PASS**
- [ ] **Step 5: Commit** `feat: email thread funnel, rendering, and NoteRecord conversion`

### Task 4: Gmail connector

**Files:**
- Create: `core/taurine_core/connectors/gmail.py`
- Test: `core/tests/test_email_connectors.py` (Gmail half)

**Interfaces:**
- Produces: `GmailConnector` (`connector_id="gmail"`, `label="Gmail"`, `kind="oauth"`). Env: `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`. `available()` = env presence only. `extract(config)`: refresh access token (cached in-process until expiry), `threads.list` with `q="newer_than:{window}d -in:spam -in:trash -category:promotions -category:social -category:updates -category:forums"` paginated to `EMAIL_FETCH_CEILING`, `threads.get?format=full` per thread, parse (headers via `email.utils.parseaddr`/`getaddresses`, `SENT` label → `is_from_me`, `STARRED|IMPORTANT` → flagged, `List-Unsubscribe`/`Precedence: bulk|list` → `has_unsubscribe`, base64url bodies, `text/plain` preferred else `html_to_text`), funnel, records. Extract results cached in-process for `EMAIL_SYNC_MIN_INTERVAL_S` so the 300 s sync loop doesn't hammer the API. Client injectable via `_client_factory` for tests. 401 → `ConnectorError("… re-run: python -m taurine_core.connectors.email_auth gmail")`; 429 → retry once honoring `Retry-After`.
- Consumes: Task 3 API, `imessage._contact_names` for the email→name contact map.

- [ ] **Step 1: Failing tests** — `available()` env gating; thread JSON fixture parse (sent detection, unsubscribe header, base64url plain + html-only bodies, flagged); `extract` end-to-end with `httpx.MockTransport` (token refresh POST + list + get); 401 raises `ConnectorError`.
- [ ] **Step 2: Run — fails**  · **Step 3: Implement**  · **Step 4: PASS**  · **Step 5: Commit** `feat: Gmail connector (env-token OAuth, thread ingestion)`

### Task 5: Outlook connector

**Files:**
- Create: `core/taurine_core/connectors/outlook.py`
- Test: `core/tests/test_email_connectors.py` (Outlook half)

**Interfaces:**
- Produces: `OutlookConnector` (`connector_id="outlook"`, `label="Outlook"`, `kind="oauth"`). Env: `OUTLOOK_CLIENT_ID`, `OUTLOOK_REFRESH_TOKEN`. Token refresh against `login.microsoftonline.com/common/oauth2/v2.0/token` (public client, no secret; rotated refresh tokens logged as a warning **without the token value**). Fetch: `GET /me?$select=mail,userPrincipalName` (own address), junk/deleted folder ids, `/me/messages` paginated via `@odata.nextLink` (`$filter=receivedDateTime ge {iso}`, `$orderby=receivedDateTime desc`, `$select` incl. `body,conversationId,inferenceClassification,flag,importance,parentFolderId`), client-side: skip drafts + junk/deleted folders, group by `conversationId`, `is_from_me` = from-address == own address, `bulk_classified` = every non-me message classified `other`, flagged = `flag.flagStatus=="flagged"` or `importance=="high"`, HTML bodies via `html_to_text`, timestamps via `datetime.fromisoformat`. Same funnel, caching, retry, and error posture as Gmail.
- Consumes: Task 3 API, `imessage._contact_names`.

- [ ] **Step 1: Failing tests** — conversation grouping; own-address sent detection; junk-folder + draft skipping; `inferenceClassification` → `bulk_classified`; ISO timestamp parse; `available()` gating.
- [ ] **Step 2: Run — fails**  · **Step 3: Implement**  · **Step 4: PASS**  · **Step 5: Commit** `feat: Outlook connector (device-code OAuth, conversation ingestion)`

### Task 6: Auth bootstrap CLI

**Files:**
- Create: `core/taurine_core/connectors/email_auth.py`
- Test: `core/tests/test_email_connectors.py` (PKCE helper test)

**Interfaces:**
- Produces: `python -m taurine_core.connectors.email_auth gmail|outlook`. Gmail: PKCE pair (`_pkce_pair() -> tuple[str, str]`, verifier + S256 challenge), loopback `http.server` on an ephemeral 127.0.0.1 port, browser consent (`gmail.readonly`, `access_type=offline`, `prompt=consent`), code exchange, prints `GMAIL_*` env lines. Outlook: device-code flow (`offline_access Mail.Read User.Read`, `/common`), polls with `authorization_pending` handling, prints `OUTLOOK_*` env lines. Client id/secret read from env (after `load_dotenv(core/.env)`) or prompted interactively. Tokens printed to stdout only — never logged or persisted by the tool.
- Consumes: nothing from other tasks (standalone).

- [ ] **Step 1: Failing test for `_pkce_pair`** (verifier charset/length, challenge = b64url(sha256(verifier)) without padding)
- [ ] **Step 2–4: Implement, PASS**  · **Step 5: Commit** `feat: email OAuth bootstrap CLI (Gmail loopback+PKCE, Outlook device code)`

### Task 7: Catalog registration + full verification

**Files:**
- Modify: `core/taurine_core/connectors/catalog.py` (append `GmailConnector()`, `OutlookConnector()`)
- Modify: `docs/superpowers/specs/2026-07-08-email-connectors-design.md` (`available()` = env-presence only, no profile call — network on UI poll is wrong)

**Interfaces:**
- Consumes: Tasks 4–5 connector classes. The existing priors regression test now covers the new ids.

- [ ] **Step 1: Register connectors in catalog**
- [ ] **Step 2: `uv run pytest tests/ -q` — full suite green (regression test now exercises gmail/outlook priors)**
- [ ] **Step 3: Ruff:** `uv run ruff check taurine_core tests`
- [ ] **Step 4: Commit** `feat: register gmail/outlook connectors`

## Self-Review

- Spec coverage: stages 0–4 (Tasks 3–5), PII (Task 2), config/priors (Task 1), auth CLI (Task 6), catalog+UI surfacing via existing `kind` plumbing (Task 7), error handling and rate limiting (Tasks 4–5), funnel drop-count logging (Task 3 `log_funnel`). Spec's `available()` profile-call line corrected in Task 7.
- No placeholders: signatures and behaviors are fully specified above; the executor works in-session with the spec and this plan.
- Type consistency: `EmailMessage`/`EmailThread`/`FunnelReport` names match across Tasks 3–5; `_pii_model` seam named consistently in Tasks 2–3.
