# Email connectors (Gmail + Outlook) — design

Date: 2026-07-08. Status: approved (brainstormed in the 2026-07-08 retrieval session,
design confirmed in chat before implementation).

## Goal

Bring email into the corpus as two `kind="oauth"` connectors — `gmail` and `outlook` —
with thread-level ingestion, a deterministic relevance funnel (no LLM anywhere before
enrichment), and local PII redaction. Phase 1 targets personal inboxes: auth is a
one-time bootstrap CLI that produces refresh tokens stored in `core/.env`, consistent
with the "env vars are secrets only" convention. Phase 2 (sidecar OAuth routes +
Keychain + Connect button) is out of scope here and doesn't get foreclosed.

## Why threads, why a funnel

Every document costs batched LLM enrichment, 3–10 embeddings, graph edges, and alias
entries. Thousands of raw emails would wreck enrichment cost, flood the graph with
near-duplicate clutter, and poison the C1 alias table. So:

- **Unit of ingestion = the thread** (Gmail `threadId`, Outlook `conversationId`), one
  `NoteRecord` per conversation, rendered `[me]`/`[Name]`-style like iMessage.
- **A funnel drops ~80–95% of threads deterministically** so only the salient few
  hundred ever reach enrichment.

## Architecture

New files, all under `core/taurine_core/`:

| File | Responsibility |
|---|---|
| `connectors/email_common.py` | Provider-agnostic: `EmailMessage`/`EmailThread` dataclasses, the funnel (filter → score → cap), thread rendering (quote/signature stripping, HTML→text), thread → `NoteRecord`. |
| `pii.py` | `redact_pii(text) -> str`: regex layer (Luhn cards, SSNs, account numbers, verification codes) + GLiNER2 layer with a dedicated PII schema. Never raises; degrades to regex-only if the model can't load. |
| `connectors/gmail.py` | `GmailConnector`: token refresh, `threads.list`/`threads.get` via httpx, Gmail JSON → `EmailThread`. |
| `connectors/outlook.py` | `OutlookConnector`: token refresh, `/me/messages` paging + client-side `conversationId` grouping via httpx, Graph JSON → `EmailThread`. |
| `connectors/email_auth.py` | Bootstrap CLI: `python -m taurine_core.connectors.email_auth gmail\|outlook`. Gmail: RFC 8252 loopback + PKCE. Outlook: device-code flow. Prints the env lines to paste into `core/.env`. |

`catalog.py` gains both connectors (disabled by default — only `apple_notes` defaults
on). No UI changes: `kind: 'oauth'` is already modeled end-to-end, and phase-1
remediation ("run the auth CLI") surfaces through the existing `available()` detail
string in ConnectorsView.

No new dependencies: httpx is already a core dep; the bootstrap uses stdlib
`http.server` + `webbrowser`.

## Data flow

```
provider API ──(stage 0: server-side query filter)──▶ raw threads
  ─▶ stage 1: deterministic drops        (unsubscribe/bulk, noreply, transactional,
                                          unknown single-message senders, window)
  ─▶ stage 2: importance score + cap     (recency × replied × known-participant ×
                                          flagged × length; top EMAIL_MAX_THREADS)
  ─▶ stage 3: render transcript          (strip quoted tails + signatures, [me]/[Name],
                                          newest EMAIL_MAX_MESSAGES, char caps)
  ─▶ stage 4: redact PII per message     (regex + GLiNER2 PII schema; BEFORE hashing)
  ─▶ NoteRecord(content_hash=sha256(redacted transcript)) ─▶ sync_source (unchanged)
```

Redacting before `content_hash` means the hash, `notes.text`, embeddings, FTS, and
enrichment batches only ever see redacted text — raw PII never lands anywhere.
Incremental sync is the existing machinery: a new message changes the hash → that
thread re-enriches; unchanged threads are skipped.

### Stage 0 — server-side filters

- Gmail `threads.list` `q`: `newer_than:{EMAIL_WINDOW_DAYS}d -in:spam -in:trash
  -category:promotions -category:social -category:updates -category:forums`.
- Outlook: `$filter=receivedDateTime ge {cutoff} and isDraft eq false`; messages in
  Junk Email / Deleted Items folders excluded by folder id; messages classified
  `inferenceClassification: other` (Focused Inbox's bulk bucket) are marked and only
  survive stage 1 if you participated in the thread.

### Stage 1 — deterministic drops (per thread, in order)

1. Newest message older than `EMAIL_WINDOW_DAYS` → drop (window).
2. Any message carries `List-Unsubscribe` or `Precedence: bulk/list` **and** you never
   replied in the thread → drop (marketing/newsletters; replying rescues).
3. Every non-you sender has a noreply-style local part (`EMAIL_NOREPLY_LOCALPARTS`)
   **and** you never replied → drop (automated notifications, billing, receipts).
4. Subject matches `EMAIL_TRANSACTIONAL_SUBJECT_RE` (receipt/invoice/verify/password
   reset/security alert/statement/order confirmation/…) **and** you never replied →
   drop (transactional noise; a billing thread you actually discussed survives).

There is deliberately no unknown-sender drop (removed 2026-07-09 after real-inbox
calibration): calendar invites/accepts are single-message threads from addresses the
user may never have written to, and the rule ate exactly those. Unknown senders
instead earn no `known_participant` weight in stage 2, so they sink in the cap
ordering; a future system will keep low-value notes away from expensive LLM
adjudication.

Contacts resolution reuses the iMessage AddressBook reader (`_contact_names` /
`CONTACTS_DB_GLOB`), whose map already contains email → display-name entries. The
funnel returns per-rule drop counts; the connector logs them each sync so the funnel
is auditable against a real inbox.

### Stage 2 — importance score + cap

`S = w·recency_decay(newest_ts) + w·you_replied + w·known_participant + w·flagged
+ w·thread_length` with weights in `EMAIL_SCORE_WEIGHTS`; sort desc, keep top
`EMAIL_MAX_THREADS` (400). All knobs in `config.py`, to be calibrated on real inboxes.

### Stage 3 — rendering

Oldest→newest, newest `EMAIL_MAX_MESSAGES` kept. Per message: prefer `text/plain`
body, else HTML stripped by a small deterministic tag-stripper; cut quoted reply tails
(`>`-prefixed runs, `On … wrote:`, `-----Original Message-----`, Outlook
`From:/Sent:/To:/Subject:` header blocks); cut signatures at `-- `; collapse
whitespace; cap at `EMAIL_MESSAGE_MAX_CHARS`. Total body capped at
`CONNECTOR_DOC_MAX_CHARS`, dropping oldest messages first. Title `Email: {subject}`
(participants as fallback), folder `"Email"`, `note_id` `gmail:{threadId}` /
`outlook:{conversationId}`, `modified` = newest ts, `created` = oldest kept ts.

### Stage 4 — PII redaction (`pii.py`)

Applied per message (short, bounded inputs for the model), before joining/hashing.

- **Regex layer** (runs first, most reliable for fixed formats): card candidates
  (13–19 digits with optional space/dash separators) validated by **Luhn** →
  `[redacted-card]`; SSNs (`ddd-dd-dddd`, and `SSN: ddddddddd` context form) →
  `[redacted-ssn]`; account/routing numbers with a nearby `account/acct/routing`
  keyword → `[redacted-account]`; verification/OTP codes after `code/OTP/passcode` →
  `[redacted-code]`.
- **GLiNER2 layer**: the already-vendored `fastino/gliner2-large-v1` via the shared
  cached model instance (`live.spans`), called with a dedicated
  `GLINER2_PII_SCHEMA` — `credit_card_number`, `social_security_number`,
  `bank_account_number`, `password_or_credential`, `government_id`, `date_of_birth` —
  at `EMAIL_PII_GLINER_THRESHOLD` (0.35, recall-leaning: false positives cost a
  redacted span, false negatives leak PII). Matched spans are replaced with
  `[redacted-{label}]`. **Deliberately excluded**: person, organization, address —
  names and addresses stay, per requirements; they're what entity linking needs.
- Does **not** route through `Gliner2SpanExtractor._normalize_result` — its
  stoplist/proper-noun filters are retrieval-tuned and would eat redaction spans.
- Failure posture: model load/inference failure logs once and degrades to regex-only;
  redaction never blocks a sync.

## Auth (phase 1: personal inboxes, env-var tokens)

Both connectors read tokens from env at `extract()` time and cache the short-lived
access token in-process. `available()` returns
`(False, "Sign in required — run: python -m taurine_core.connectors.email_auth <id>")`
until the env vars exist. It checks env presence only — no network: `available()`
runs on every UI poll, so auth errors surface at sync time instead.

- **Gmail** (bring-your-own OAuth client, Desktop type): env `GMAIL_CLIENT_ID`,
  `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`. Bootstrap runs the loopback flow
  (ephemeral 127.0.0.1 port + PKCE, scope `gmail.readonly`, `access_type=offline`,
  `prompt=consent`). Personal client in testing mode expires refresh tokens after 7
  days unless the app is published (click through the unverified warning once).
- **Outlook** (own Azure app registration, public client): env `OUTLOOK_CLIENT_ID`,
  `OUTLOOK_REFRESH_TOKEN`. Bootstrap runs the device-code flow (scopes
  `offline_access Mail.Read User.Read`, `/common` endpoint). Refresh responses rotate
  the refresh token; the connector logs a reminder to update `.env` when a new one
  arrives (old tokens stay valid until their own expiry, so personal use tolerates
  this).

## Config additions (`config.py`)

Email section: `EMAIL_WINDOW_DAYS=120`, `EMAIL_MAX_THREADS=400`,
`EMAIL_MAX_MESSAGES=25`, `EMAIL_MESSAGE_MAX_CHARS=2000`, `EMAIL_NOREPLY_LOCALPARTS`
(frozenset), `EMAIL_TRANSACTIONAL_SUBJECT_RE`, `EMAIL_SCORE_WEIGHTS`,
`EMAIL_PII_GLINER_THRESHOLD=0.35`, `GLINER2_PII_SCHEMA`. Priors: `gmail` and
`outlook` in **both** `SOURCE_RANK` (0.85, iMessage tier) and
`RECENCY_HALF_LIFE_DAYS` (30.0, episodic) — the
`test_every_connector_has_explicit_source_priors` regression enforces this the moment
the ids land in `CONNECTORS`.

## Error handling

- Provider/API failures raise `ConnectorError` with actionable text (401 → re-run the
  auth CLI; 403 restricted-scope → check OAuth client scopes) — the registry already
  converts these to sync-report errors shown in the UI.
- Malformed messages (undecodable bodies, missing headers) are skipped per-message,
  never fail the thread; threads with zero usable messages are skipped.
- Rate limiting: single-flight sequential fetching with retry-once-on-429 honoring
  `Retry-After`; personal-inbox volumes don't need more.

## Testing

- `test_email_common.py`: each stage-1 rule (drop + replied-rescue), correspondent-set
  construction, scoring order + cap, quote/signature stripping, HTML→text, `[me]`
  labeling, char caps, `content_hash` stability, drop-count reporting.
- `test_email_pii.py`: Luhn-valid card redacted / Luhn-invalid preserved, SSN forms,
  account-number context rule, codes; names/addresses preserved; GLiNER layer with a
  monkeypatched model (span replacement, threshold, label placeholders); regex-only
  degradation when the model is unavailable.
- `test_email_connectors.py`: Gmail/Outlook JSON fixtures → `EmailThread` parsing
  (headers, labels/categories, base64url bodies, conversation grouping, sent
  detection); `available()` env gating; 401 → `ConnectorError`; token refresh via a
  fake httpx transport.
- Existing `test_every_connector_has_explicit_source_priors` covers the config keys.
