# Requirements: recurring research with tracked summaries and email alerts

Each requirement has an ID, a statement, acceptance criteria and how it is checked. All are "must" unless marked "should". `REC-1xx` are functional and `RNFR-xx` non-functional. Phase A is release 3.11.0 (REC-100 – REC-118); phase B is 3.12.0 (REC-130 – REC-139).

**Terms:**

- **Topic:** a saved question with a schedule, notify settings and subscribers.
- **Definition:** a topic's settings file, `data/topics/<id>.json`, written only by the backend.
- **State:** a topic's schedule state, `data/topics/<id>/state.json`, written only by the scheduler.
- **Run:** one scheduled research task for a topic, with its record in `data/topics/<id>/runs/<run_id>.json`.
- **Slot:** one due time computed from the schedule.
- **Hold:** a pause set by the scheduler, for `failures` or `budget`. A **pause** is set by people (`status: paused`).
- **Allowance:** the monthly spending limit for scheduled runs, counting Tavily credits and Together dollars.
- **Scheduler token:** `WEBAGENT_SCHEDULER_TOKEN`, a secret held only by the backend and the scheduler.

## 1. Topics (phase A)

**REC-100 Topic definition (must).** A topic has:

- `topic_id`, a UUID;
- `name`, 1–80 characters with no control characters;
- `query`, 1–2,000 characters with no control characters;
- `schedule`: `{every: day|week, weekday: mon…sun (required for week), time: HH:MM, tz: IANA name}`;
- `notify`: `{when: changes|always, subscribers: [...]}`;
- `status`: `active`, `paused` or `deleted`;
- `created_at`, `created_from_task` (optional UUID), and the counters `run_requests` and `resumes`.

An invalid field answers 422 with the field named.

- Check: unit tests on validation (lengths, control characters, an unknown time zone, `week` without a weekday).

**REC-101 One writer per file (must).**

- The backend alone writes definitions and `data/notify/`.
- The scheduler alone writes `state.json`, run records, `usage-<YYYY-MM>.json` and `counters.json`.
- Every write is atomic (a temp file, then `rename`).
- The backend's read-modify-write of a file happens under an exclusive `fcntl.flock` on a per-file lock, because it runs two workers.
- Check: a test with two concurrent backend edits (subscribe and pause) keeps both; a code test asserts each module writes only its own paths.

**REC-102 Topic API (must).** These need `X-API-Key` (when keys are on) and go through the web app's proxy:

- `GET /api/topics`: list, with the effective state (`active`, `paused`, `held: failures`, `held: budget`), next and last run, and the last run's new-source count.
- `POST /api/topics`: create.
- `GET /api/topics/{id}`.
- `PATCH /api/topics/{id}`: name, schedule, notify settings, `status` (`active` or `paused`).
- `DELETE /api/topics/{id}`.
- `POST /api/topics/{id}/run`: Run now; increments `run_requests`.
- `POST /api/topics/{id}/resume`: sets `status: active` and increments `resumes`.
- `GET /api/topics/{id}/runs` and `GET /api/topics/{id}/runs/{run_id}`.

Creating, or resuming to active, beyond `schedule.max_active_topics` (10) answers 409 `topic_limit`. Unknown IDs answer 404, and a deleted topic answers 404.

- Check: API tests for every route, the cap and 404s.

**REC-103 Intake guard at creation (must).** `POST /api/topics` checks the query through checkpoint 1, as `POST /api/guard/input` does.

- In act mode, `decline`, `escalate` and `consent` refuse the topic: 403, with the guard's refusal body. A scheduled run can't give consent later.
- In shadow mode the outcome is recorded and the topic is created.
- Check: tests in both modes.

**REC-114 Delete (must).** `DELETE` sets `status: deleted` at once (a tombstone), and the API then treats the topic as gone. The tombstone is scrubbed in the same write: the name and query become `deleted` and the subscribers are removed, so no personal text stays on disk while the files are being removed. Afterwards:

- **The scheduler, on its next tick,** removes the topic's folder and the saved results in `data/results/` of the topic's runs (by task ID).
- **The backend** removes the definition and the topic's ledger entries once the folder is gone (checked on access and by the daily cleanup).
- **With no live scheduler** (heartbeat older than 180 s), the backend removes the folder and those results itself, while holding the scheduler lock.
- Check: a test with and without a running scheduler; nothing about the topic remains afterwards.

**REC-115 Retention (must).**

- A topic keeps its last `history_runs_kept` (20) run records; older ones are removed by the scheduler.
- A topic with `status: paused` for `paused_topic_days` (90) is deleted as in REC-114. The scheduler records `paused_since` in the state when it first sees the pause, and requests the deletion through `DELETE /api/topics/{id}`. From phase B, confirmed subscribers get a notice 7 days before (REC-138).
- Unconfirmed subscriber entries expire after 48 h (phase B).
- Check: tests on pruning and the 90-day deletion with a fake clock.

## 2. Scheduler (phase A)

**REC-104 Due runs (must).** Every 60 s the scheduler loads the topics that are `active`, with no hold, and either due (`next_due` ≤ now) or with `run_requests` > `run_requests_seen`. It starts them oldest-due first, at most `max_concurrent_runs` (2) at a time.

- **Next slot:** after a slot is taken, `next_due` becomes the first slot after *now*. Missed slots are never backfilled.
- **Time zones:** slots are computed in the topic's time zone. A local time that doesn't exist (DST gap) moves to the first valid minute after it; a time that occurs twice runs at its first occurrence.
- **Run now:** handling a request increments `run_requests_seen` and doesn't change `next_due`.
- Check: unit tests on slot maths (DST gap, repeated hour, weekly across a year end), no backfill, ordering, the concurrency cap.

**REC-105 Single runner (must).**

- **Lock:** the scheduler holds `data/topics/.scheduler.lock`, with its PID and a heartbeat refreshed every 60 s. A scheduler that finds a heartbeat under 180 s old **waits** for it to go stale (checking every 15 s, for up to 240 s) and then takes it over. A holder that keeps beating is never taken over, and the waiting scheduler exits after 240 s. Waiting instead of exiting at once keeps a restart after a crash or a redeploy from becoming a crash loop (found in the end-to-end check, 2026-10-18).
- **Disabled:** with `schedule.enabled: false` (`WEBAGENT_SCHEDULER_ENABLED`), the scheduler idles and writes a heartbeat with `enabled: false`.
- Check: tests for a second instance, a stale lock, and disabled.

**REC-106 Crash safety (must).**

- **Fixed task ID:** each run's task ID is `uuid5(TOPIC_NAMESPACE, run_id)`, recorded as `status: starting` before the call.
- **Restart:** for every run record in `starting` or `running`, the scheduler asks `GET /api/task/{task_id}`. A known task is polled to the end; an unknown one is marked `failed` (`error_code: not_started`).
- **No duplicates:** a slot never starts two tasks, and the usage of a task that did start is always counted.
- Check: tests that crash after the record and after the POST, then restart.

**REC-107 Scheduler credential and fields (must).**

- **Fields:** `POST /api/task` accepts `origin` (`schedule`), `topic_id`, `task_id` and `token_budget` only with an `X-WebAgent-Scheduler` header equal to `WEBAGENT_SCHEDULER_TOKEN` (a constant-time compare).
  - Without a valid header, any of those fields answers 422.
  - With `WEBAGENT_SCHEDULER_TOKEN` unset, the backend refuses all of them, and the scheduler-only endpoints answer 404. With the token set, those endpoints answer 403 to a missing or wrong header.
- **Task ID:** a caller-chosen `task_id` that already exists answers 409.
- **Web proxy:** the web app's task proxy forwards only the allow-listed body fields: `query`, `mode`, `agentic`, `previous_messages`, `triage_override`, `image`, `document_id`, `consent_token`, `approval_tokens`.
- **Secrecy:** the token is never in the web app's environment, logs or metrics.
- Check: backend tests (header missing, wrong or right; API key alone; token unset); a web test that the proxy drops other fields.

**REC-108 Scheduled run effects (must).** For a task with `origin: schedule`:

- **Budget:** `token_budget` is required and above 0 (422 otherwise), and the run's budget is `token_budget`, not the global one.
- **Triage:** mode must be `always`. Triage is neither run nor collected (`triage_collect.record` is skipped).
- **Drift:** `DriftRecordingEmbeddings` doesn't observe the run's query embeddings.
- **Records:** the saved result and the security collection rows carry `origin` and `topic_id`. `webagent_tasks_total` and `webagent_task_duration_seconds` get an `origin` label (`interactive` or `schedule`). LangSmith runs get `origin` and `topic_id` metadata.
- **No history:** there are no previous turns; `previous_messages` sent with `origin` are dropped. `topic_id` is required with `origin`.
- **Route:** the result records `route: agentic` and `decided_by: schedule`.
- Check: tests for each effect.

**REC-109 Per-task search cap and usage, for every run (must).**

- **Counter:** the run's counter in `task_budget` (a context variable holding a mutable object, shared by parallel agents) counts input tokens, output tokens, searches and Tavily credits: 1 at basic depth, 2 at advanced.
- **Cap:** `web_research` asks it before every search, and a search over `agents.web_research.max_searches_per_task` *per task* is skipped and logged.
- **Saved:** the result gets `usage: {input_tokens, output_tokens, searches, tavily_credits}`, next to the existing `tokens_used`.
- **Interactive runs:** they're bounded the same way. The cap's value is set from shadow week 2's searches per run (task A0).
- Check: tests across 3 passes and the parallel agents; usage on the result.

**REC-110 Standalone question (must).**

- Every research task's result records the supervisor's `standalone_query`, and `GET /api/task/{id}` returns it in `result`.
- Outside dev, it's never logged (only its length).
- Check: test on the result and the response.

**REC-111 Run history and source diff (must).**

- **Run record:** after a run finishes, the scheduler writes `{run_id, task_id, status, error_code, started_at, completed_at, usage, report, sources, change}`. `report` and `sources` are copied from the result, so they outlive the results cleanup.
- **Normalising URLs:** lower-case the host and strip `www.`; drop the fragment and the `utm_*`, `fbclid`, `gclid` parameters; drop a trailing `/`.
- **The diff:** `change.new_sources` are this run's URLs not in the last 3 runs; `change.dropped_sources` are those in the previous run and not this one. The first run marks all its sources new, with `change.first: true`.
- Check: unit tests on normalisation and the diff, and on the first run.

**REC-112 Failures and holds (must).**

- **Failures:** a run that ends failed counts toward `failures_in_a_row`. That covers timeout, budget, cancelled, escalated, an intake refusal (403 or 409 at start), a backend error, and a poll that runs out after 10 minutes. A completed run resets the count to 0.
- **Hold:** at `pause_after_failures` (3) the scheduler sets `hold: failures`. `resumes` > `resumes_seen` clears it and resets the count.
- Check: tests for each failure kind and for resume.

**REC-113 Caps and allowance (must).** Before starting a run the scheduler checks two things:

1. **The daily cap:** scheduled runs started today (in the scheduler's time zone, UTC) must be under `max_runs_per_day` (20).
2. **The allowance:** the remaining allowance must cover the run's worst case plus that of every run in progress. The worst case is `max_follow_up_questions` passes × 3 credits (9 in uat, 15 in prod as configured today, or the per-task search cap's worst case if that's lower), and `token_budget` plus 8,000 summary tokens, priced at `llm.price_output_per_million`.

- **Daily cap over:** the run is skipped (`state.skipped: daily_cap`, with a time), and `next_due` advances as usual.
- **Allowance short:** every topic gets `hold: budget`, until the month changes or the allowance is raised in config (checked every tick).
- **Usage:** recorded after each run, from the result's `usage`, in `usage-<YYYY-MM>.json`.
- Check: tests showing the allowance is never passed with 2 concurrent runs; a skip on the daily cap; the hold clearing next month.

## 3. Web app (phase A)

**REC-116 Web app (must).**

- **Track this:** under each research answer (not direct answers) it opens a form prefilled with `standalone_query` (or the question) and the report title. The form has frequency, weekday, time (in the browser's time zone), notify when, and an optional email (phase B).
- **`/topics`:** list, pause/resume, Run now and delete with confirmation. It shows the effective state, including holds and skips.
- **`/topics/[id]`:** the run timeline with each change (summary or source diff), the stored report rendered like a chat report, and the new sources.
- **Proxies:** `/api/topics/**` proxy to the backend with `X-API-Key`. They allow-list routes and methods, and never forward the scheduler-only routes. The sign-in middleware applies to all of them.
- **No state change on GET.**
- Check: Jest tests for the form, the pages and the proxies.

## 4. Operations (phase A)

**REC-117 Metrics and alerts (must).** The backend exports topic metrics through a custom collector that reads the scheduler's files, so there's one scrape target:

- **Counters** (from `counters.json`, which only grows): `webagent_topic_runs_total{outcome}`, `webagent_topic_changes_total{kind}` (`new_sources`, `summary_changed`), `webagent_topic_held_total{reason}`, `webagent_scheduler_missed_slots_total`.
- **Gauges:** `webagent_topics{state}`, `webagent_topic_budget_used{kind}` (credits, dollars) with `webagent_topic_budget_limit{kind}`, `webagent_scheduler_heartbeat_timestamp`.
- **Histogram:** `webagent_topic_run_seconds`.
- **Alert:** a Grafana rule, "Scheduler down": heartbeat older than 10 min while `schedule.enabled`.
- **Dashboard:** a "Recurring research" row on the WebAgent Backend dashboard.
- Check: metrics test with fixture files; a dashboard and alert JSON lint test.

**REC-118 Deployment and configuration (must).**

- **Config:** the `schedule:` block and `llm.price_*` in `config/{dev,uat,prod}.yaml`, with `schedule.enabled: false` in all three.
- **Environment:** `WEBAGENT_SCHEDULER_ENABLED`, `WEBAGENT_SCHEDULER_TOKEN` and `WEBAGENT_BACKEND_URL` (for the scheduler).
- **Compose:** a `webagent-scheduler` service (same image, `python -m app.scheduler`, the same data volume, `WEBAGENT_BACKEND_URL=http://webagent-backend:8000`).
- **Kubernetes:** a `scheduler` sidecar in the backend pod (`http://127.0.0.1:8000`, same volume, memory request 64 Mi and limit 192 Mi, above RNFR-02's 150 MB target), with the token in the secret through `deploy.sh`.
- **Startup checks:** the scheduler refuses to start, when enabled, without the token or without a reachable backend. The backend logs once when the token is unset.
- **Docs:** `.env.local.example` lists every new setting.
- Check: startup-check tests; a Compose config test; a manifest lint test.

## 5. Change summaries and email (phase B)

**REC-130 Change summary (must).** After a completed run that has a previous completed run, the scheduler calls `POST /api/topics/{id}/summarize` (scheduler token) with both run IDs.

- **Input:** both reports, each cut to 6,000 characters, and the new-source list, sent to the model as `retrieved_doc` context items so checkpoint 1 scores them as untrusted.
- **Model call:** `agents.topic_summary` (`max_tokens` 600) under its own budget of 8,000 tokens, which counts toward the allowance.
- **Output:** at most 5 bullets citing source numbers, or the exact phrase "No material change".
- **Failure:** the run keeps its source diff, with `change.summary_error`.
- Check: tests with a fake model (normal, no change, failure); the guard sees the reports as `retrieved_doc`.

**REC-131 Notify decision (must).**

- **Changed:** `change.changed` is true when there are new sources, or a summary other than "No material change", or it's the first run.
- **When to send:** with `notify.when: changes` the scheduler asks for an email only when changed; with `always`, after every completed run. The request is `POST /api/topics/{id}/notify` (scheduler token, run ID); failed runs never notify.
- Check: tests on the decision table.

**REC-132 Transport (must).**

- **Transports:** `notifications.email.transport` is `log` (the default in dev and tests: the message goes to the log, not sent) or `smtp`.
- **SMTP:** `smtplib` with STARTTLS (587) or TLS (465), from `WEBAGENT_SMTP_HOST`, `_PORT`, `_USERNAME`, `_PASSWORD`, `_FROM` and `_TLS`. Timeout 20 s, with 2 retries and backoff.
- **Who sends:** only the backend sends.
- **Logs:** a failure is logged by topic ID and kind, never with an address.
- Check: tests with a stub SMTP server (`smtplib` patched), including retry and failure.

**REC-133 Double opt-in (must).**

- **Adding an address:** a valid address, at most `max_subscribers_per_topic` (5) per topic, makes an unconfirmed entry and sends one confirmation. That email holds the topic name, a link to `/notify/confirm?token=…`, and nothing else from the topic.
- **Token:** HMAC-SHA256 with `WEBAGENT_SECURITY_SECRET_KEY`, over `(purpose=confirm, topic_id, address, expiry)`; it's valid 48 h and single-use (used tokens are recorded in the ledger).
- **Limits:** at most 3 confirmations per address per day.
- **Until confirmed:** nothing else is sent to the address.
- Check: tests on expiry, reuse, tampering, the caps, and nothing sent unconfirmed.

**REC-134 Confirm and unsubscribe need a POST (must).**

- **Pages:** `/notify/confirm` and `/notify/unsubscribe` show the topic name and a button; opening them changes nothing. The button POSTs the token to `/api/notify/confirm` or `/api/notify/unsubscribe`.
- **Unsubscribe token:** signed, no expiry, safe to repeat.
- **Headers:** every topic email has `List-Unsubscribe: <{WEBAGENT_PUBLIC_URL}/api/notify/unsubscribe?token=…>` and `List-Unsubscribe-Post: List-Unsubscribe=One-Click` (RFC 8058). A one-click POST unsubscribes.
- **Sign-in:** these four routes are the only ones reachable without signing in.
- Check: a GET changes nothing; a POST works; the one-click POST works; a middleware test that nothing else became public.

**REC-135 Content (must).** Email is plain text (UTF-8), with no HTML, images or tracking. Headers are built with `email.message.EmailMessage`.

- **Subject:** `WebAgent: <name> — <YYYY-MM-DD>`, with control characters removed and the name cut to 60 characters.
- **Notice:** the run date, the new-source count, the new sources' **domains only**, as text, and a link to `{WEBAGENT_PUBLIC_URL}/topics/<id>`.
  - Only domains of sources checkpoint 1 didn't flag are listed: not in the result's `security.dropped` (act mode), and not in `security.shadow_flagged` (shadow mode).
  - `security.shadow_flagged` is new: checkpoint 1 records the would-be drops in shadow mode.
- **Summary** (REC-136): the summary bullets, then the new sources as domain and path cut to 100 characters, then the link.
- **Footer:** why the mail was sent, the unsubscribe link, and "Written by an AI research assistant."
- Check: tests on header injection, no model text in notice mode, only unflagged domains.

**REC-136 Summary mode only in act (must).** With `notifications.email.mode: summary` and `security_guard.input_mode` other than `act`, the backend refuses to start (startup check), with a message naming both settings.

- Check: startup-check test.

**REC-137 Caps and ledger (must).**

- **Caps:** at most 1 topic email per topic per run, and 50 emails a day per instance (`max_emails_per_day`). Over the cap, an email is dropped and counted.
- **Ledger:** `data/notify/ledger-<YYYY-MM-DD>.json` records sends by kind and outcome, hashed addresses for the per-address cap, and used confirm tokens. The backend writes it under flock.
- **Metrics:** `webagent_email_sent_total{kind, outcome}`, where kind is confirm, notice, summary, hold or deletion, and outcome is sent, failed or capped.
- Check: tests on the caps and the metrics.

**REC-138 Hold and deletion notices (must).** When a topic gets `hold: failures`, and 7 days before a 90-day-paused topic is deleted, confirmed subscribers get one notice each, through `POST /api/topics/{id}/notify` with a `kind`.

- Check: tests.

**REC-139 Budget alert (must).** A Grafana rule, "Scheduled budget 80%", fires on `webagent_topic_budget_used / webagent_topic_budget_limit ≥ 0.8` for either kind.

- Check: alert JSON test.

## 6. Non-functional

**RNFR-01 Timeliness (must).** While the host is up, a run starts within 15 minutes of its slot, with up to 10 topics due at once. After downtime a topic runs once.

- Check: a test with a fake clock and runs of fake duration.

**RNFR-02 Scheduler footprint (must).** The scheduler imports no ML library (`torch`, `transformers`, `sentence_transformers`, `langchain*`, `chromadb`), and its resident memory stays under 150 MB.

- Check: an import test on `sys.modules`; memory measured in the container (task A11).

**RNFR-03 Privacy (must).**

- Outside dev, no question, topic name, report text or email address appears in logs or metrics labels. Logs carry IDs and lengths only (`describe_text`).
- Email addresses are stored only in definitions (and hashed in the ledger), and removed on unsubscribe, expiry or delete.
- Check: log-capture tests on every new path.

**RNFR-04 Spending (must).** Scheduled spending in a month never passes the allowance (REC-113), and each scheduled run stays within its token budget and the search cap.

- Check: REC-113 tests; usage reviewed in the phase A trial week.

**RNFR-05 Shadow-week isolation (must).** `schedule.enabled` is false in every environment's YAML, and no scheduled run happens until an operator turns it on after shadow week 2.

- Check: config test; release notes.

**RNFR-06 Security (must).**

- **Secrets:** the scheduler token is required for every scheduler-only field and endpoint, compared in constant time, and never sent to the web app.
- **Tokens:** confirm and unsubscribe tokens are HMAC-signed and bound to their purpose.
- **Routes:** no new route is public except REC-134's four.
- **Rendering:** stored reports render with no raw HTML (the existing `react-markdown` path, no `rehype-raw`).
- Check: the tests above, plus a red-team test with a report holding an injected instruction and a `<script>` tag.

**RNFR-07 Offline tests (must).** All new tests run offline: fake model, stub SMTP, fake clock, temp folders, and no real network (the existing conftest guard).

- Check: CI.

**RNFR-08 Interactive behaviour (must).** Apart from the per-task search cap (REC-109), whose value comes from measured search counts, nothing changes for interactive tasks.

- Check: the existing suites pass unchanged; searches per run are compared before and after on the shadow week's tasks (task A0).

**RNFR-09 Documentation (must).** Each phase updates:

- the README: "Using WebAgent" (topics and email), and "What WebAgent doesn't do" (recurring research and email are removed from it);
- `ARCHITECTURE.md`, `api_reference.md`, `RUNBOOK.md` (scheduler, SMTP setup and deliverability check, caps, holds), `.env.local.example`, and the CHANGELOG.
- Check: review in each release PR.

**RNFR-10 Release process (must).** Work on `release/X.Y.Z`, with tests in the same change. Merge through a PR with both CI checks green, then tag and fast-forward `dev`.

- Check: each release.
