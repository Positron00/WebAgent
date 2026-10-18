# Plan: recurring research with tracked summaries and email alerts

Status: v5, aligned with the spec 2026-10-18 (spec in `specs/recurring-research/`, which wins where the two differ), decided 2026-10-18 (D1–D8 and the search cap accepted as recommended; §11) · checked against the code three times, gaps fixed in §13 · 2026-10-17 · Base: `main` at v3.10.1 (`270a012`) · Releases: 3.11.0 (phase A), 3.12.0 (phase B), both deployed after shadow week 2

## 1. Summary

Today a research task runs once, when someone asks. This plan lets a user turn a question into a **tracked topic** that WebAgent researches again on a schedule (daily or weekly). Each run's report is kept, and a short **"what changed"** summary is written against the previous run. Subscribers can get an **email** when something changed.

The plan reuses the existing task path. A scheduled run is an ordinary `POST /api/task` in Always mode, so the security guard, the time limit, the token budget, the search cap (made per task in phase A), the collection and the metrics all apply unchanged. Four parts are new:

1. A **scheduler**: one small process that starts due runs, catches up once after downtime, and never runs a topic twice at the same time.
2. A **topic store**: each topic's runs and summaries, outside the 30-day results cleanup.
3. **Change summaries**: a list of new sources, plus one guarded model call.
4. **Email over SMTP**: double opt-in, unsubscribe, plain text, off by default.

**What makes this harder than it looks:**
- **The guard is in shadow.** It blocks nothing until act mode, so unattended runs would email model-written text built from unchecked web pages (D4).
- **There are no user accounts.** Everyone shares one access code, so topics are instance-wide, and email addresses have to be proven by a confirmation link (D2, D3).
- **Credits are tight.** The Tavily plan's 1,000 credits a month and Together's remaining ~$18 have to cover interactive use too. A monthly allowance just for scheduled runs keeps one busy topic from using them up (D5).
- **The local uat host sleeps.** Schedules on the Mac fire only while it's awake (D1).

## 2. What exists today (3.10.1)

| Piece | Today | What this plan uses |
|---|---|---|
| Task path | `POST /api/task` → `task_manager.run_workflow` under `run_supervisor` (420 s, cancel, progress); report in `data/results/<id>.json` | The scheduler calls it like any API client |
| Searches per run | One research pass makes at most 2 web searches: the main one, plus either a fallback (no results) or one follow-up question. A run makes at most `agents.senior_research.max_follow_up_questions` passes: 3 in dev and uat, 5 in prod. The main search is advanced (2 Tavily credits) only when the plan says "deep"; the others are basic (1). **`max_searches_per_task` (5 or 10) is in fact a per-pass cap and never binds** | At most 6 searches and 9 credits a run in uat, 10 and 15 in prod. Phase A makes the cap a real per-task cap (§6.3) |
| Token budget | `WEBAGENT_TASK_TOKEN_BUDGET`, one global value (0 = off), applied in `task_manager.py` (`task_budget.start`). The counter keeps only total tokens (`together_llm.py` adds `total_tokens`) | Needs a per-run value for scheduled runs, and input and output counted separately to price them (§6.3) |
| Run time | Median 113 s over 22 saved research runs (dev and the first uat week); README says 2.5–3.5 min | Scheduler polls with a 10-minute ceiling |
| Token use | Not measured yet: older reports predate `tokens_used`. Shadow week 2 records it (improvement plan task 2.10) | Sets the default scheduled budget (§7) |
| Results retention | `RESULTS_RETENTION_DAYS` 30, daily cleanup of `data/results/` | Topic history needs its own store and retention |
| Identity | One shared access code; signed session cookie, 30 days; no user records | No owner to attach a topic or an email to (D2) |
| Guard | Checkpoints 1 and 2, shadow in every environment; consent and approval need a person; the intent record is built from the user's turns | Unattended runs can't consent; the topic text is the intent (§6.7) |
| Drift | Embedding drift observes every vector-store query (`DriftRecordingEmbeddings`); model speed and tokens per agent | Repeated scheduled questions would skew embedding drift (§6.10) |
| Rate limits | Per caller; `X-WebAgent-Client` accepted from API-key holders | Scheduler identifies itself as `schedule:<topic>` |
| Web proxy | `src/app/api/task/route.ts` forwards the browser's **raw request body** and adds `X-API-Key` | The API key can't tell the scheduler from a browser; the scheduler needs its own credential (§6.3) |
| Saved result | Report `content`, `title`, `sources`, `route`, `triage`, `security`; **not** the supervisor's standalone question | Track this needs the standalone question saved (§6.3) |
| Kubernetes data volume | One 5 GiB `ReadWriteOnce` claim on the backend pod | The scheduler must share it: a sidecar in the same pod, not its own Deployment (D1) |
| Email | None. The code-safety checks block `smtplib` in **model-written** code; the server itself can use it | New `services/notify_email.py` |

## 3. Use cases

1. "Track this" under a research answer: research it again every Monday and keep a timeline.
2. A daily brief on a fast-moving subject, emailed only when sources changed.
3. Look back: what did the report say three weeks ago, and what changed since?

Out of scope: monitoring a single web page for changes, alerts on words appearing, sharing topics between instances, chat-style follow-ups inside the email.

## 4. Goals and non-goals

**Goals**
- G1. A topic's run starts within 15 minutes of its due time while the host is up, with up to 10 topics due at once (2 at a time, about 3 minutes each). After downtime it runs once on catch-up.
- G2. Every scheduled run goes through the same guard, limits and budget as an interactive one. Nothing bypasses `/api/task`.
- G3. Each run's report and change summary is kept and viewable in the web app, with the sources that were new.
- G4. Email goes only to confirmed addresses, has a working unsubscribe link, and is plain text without remote content.
- G5. Scheduled spending is capped by a monthly allowance, separate from interactive use, and visible in metrics.
- G6. Scheduled runs don't distort the shadow-week data, drift or triage collection.

**Non-goals**
- User accounts and per-user privacy (a separate plan).
- Hourly schedules. Daily is the floor.
- Push, SMS and chat notifications. The design leaves room for them (`notify_*` services).
- An HTML email template.
- Paid or premium sources, such as connecting a user's subscription. Scheduled runs use the same sources as interactive ones. Paid sources need official, licensed API connectors, user accounts and the guard in act mode first (discussed 2026-10-18).

## 5. Decisions (accepted 2026-10-18)

| ID | Decision | Recommended | Why |
|---|---|---|---|
| D1 | Where the scheduler runs | **A separate one-process service** (`python -m app.scheduler`, same image): a Compose service sharing the data folder, and in Kubernetes a **sidecar container in the backend pod**, because the data volume is ReadWriteOnce | Exactly one runner, without coordinating API workers. It catches up after a laptop sleeps, which a CronJob can't. It loads no models (it imports only config, the topic store and an HTTP client, not the agents), because runs execute in the backend. Its imports measured 45 MB; the target is under 150 MB (RNFR-02) |
| D2 | Who sees and manages topics | **Instance-wide**, like loaded documents: anyone signed in can list, create, pause and delete | No accounts exist, and sessions last 30 days, so session-owned topics would be orphaned on a new device. Revisit with accounts |
| D3 | Email transport | **SMTP** (STARTTLS or TLS), any provider (Fastmail, Gmail app password, SES or Resend SMTP), from the standard library. Off by default (`notifications.email.enabled: false`) | No new SDK dependency and no lock-in. The provider handles SPF and DKIM |
| D4 | What an email says while the guard is in shadow | **Notice mode**: "Topic X has a new report with N new sources (domains)" plus a link to the web app, with no model-written text. **Summary mode**, with the change summary in the email, is allowed only when `input_mode: act` | In shadow mode, injected page content isn't removed, and email is a channel the web app's CSP doesn't protect |
| D5 | Spending caps | Weekly by default, daily at most. At most **10 active topics** and **20 scheduled runs a day**. A **monthly allowance for scheduled runs: 400 Tavily credits** (40% of the plan) **and $5 of Together** (§7). A per-run token budget is always set | Interactive use and the shadow week need the rest of the credit |
| D6 | Change detection | **New-source diff** (URLs not seen in the last 3 runs) plus **one guarded model call** comparing the new report with the previous one. "Notify when": `changes` (default) or `always` | The source diff is cheap and exact; the model call explains the change; `always` covers subtle changes |
| D7 | History retention | Keep the **last 20 runs per topic** (full report copies). Deleting a topic deletes its runs, summaries and subscriptions. Topics paused for 90 days are deleted, after a notice | Bounded disk, and the right to delete |
| D8 | When it ships | Implement now; **deploy after shadow week 2**. No scheduled runs during the week | Scheduled runs would mix repeated questions into the week's data |

## 6. Design

### 6.1 Topics and schedules (`app/services/topics.py`, `data/topics/`)

```json
{
  "topic_id": "<uuid>",
  "name": "EU AI Act enforcement",
  "query": "What changed in EU AI Act enforcement this week?",
  "schedule": {"every": "week", "weekday": "mon", "time": "07:30", "tz": "Europe/Paris"},
  "notify": {"when": "changes", "subscribers": [{"email": "a@example.org", "added_at": "...", "confirmed_at": "..."}]},
  "status": "active",
  "created_at": "...",
  "created_from_task": "<task uuid or null>",
  "run_requests": 0,
  "resumes": 0
}
```

The schedule state (`state.json`, scheduler only) holds `{"next_due": "2026-10-20T05:30:00Z", "last_run": {"run_id", "task_id", "status", "at"}, "failures_in_a_row": 0, "hold": null, "skipped": null, "paused_since": null, "schedule_seen": {…}, "run_requests_seen": 0, "resumes_seen": 0}`. Times are stored in UTC. `paused_since` is set by the scheduler when it first sees `status: paused` (for the 90-day rule); `schedule_seen` lets it notice a changed schedule and recompute `next_due`.

- **Fields:**
  - `status` (definition) is `active` or `paused`, set by people, or `deleted`, a tombstone set by a delete.
  - `hold` (state, set by the scheduler) is `null`, `failures` (3 failures in a row) or `budget` (the monthly allowance is used up).
  - A topic runs only when it's `active` with no `hold`; the web app shows "paused", "held: failures" or "held: budget".
  - **Resume** sets `status: active` and increments `resumes` in the definition (`POST /api/topics/{id}/resume`). When `resumes` exceeds `resumes_seen`, the scheduler clears a `failures` hold. A `budget` hold clears itself at the start of the next month, or when the operator raises the allowance.
  - `every` is `day` or `week`; `week` needs `weekday`. `time` is in the topic's IANA time zone, stored with the topic, and converted with `zoneinfo`.
  - On a DST change, a local time that doesn't exist (the hour skipped in spring) moves to the first valid minute after it. A time that occurs twice (the repeated hour in autumn) uses its first occurrence. A topic never runs twice for one slot.
  - `query` is at most 2,000 characters, and `name` at most 80, with no control characters.
- **Storage, with one writer per file**, so the API server and the scheduler never overwrite each other. Every file is written atomically (temp file and rename, as `result_store` does). **The backend runs two uvicorn workers**, so its files (definitions and the email ledger) are changed with read, modify and write under an exclusive `fcntl.flock` on a lock file next to them. `app/core/drift_store.py` already has this helper. The scheduler is a single process and needs no lock for its own files.
  - `data/topics/<id>.json` holds the **definition**: name, query, schedule, notify, status, subscribers, and the `run_requests` and `resumes` counters. Only the **backend API** writes it.
  - `data/topics/<id>/state.json` holds the **schedule state**: `next_due`, `last_run`, `failures_in_a_row`, `hold`, and the request counters it has handled. Only the **scheduler** writes it.
  - `data/topics/<id>/runs/<run_id>.json` holds one record per run. Only the scheduler writes it; the summary comes back from the backend in a response, and the scheduler stores it.
  - `data/topics/usage-<YYYY-MM>.json` holds the month's allowance use, and `data/topics/counters.json` holds monotonic run counters for the metrics. Only the scheduler writes them.
  - `data/notify/` holds the email ledger (sends per day, per address, confirm tokens used). Only the backend writes it, because the backend sends every email (§6.6).
  - A pause sets the definition's `status`.
  - **A delete** sets `status: deleted` (a tombstone), which the API and the web app treat as gone at once. Then each process removes only its own files:
    - The scheduler removes the topic's folder (state and runs) on its next tick. It also deletes the scheduled runs' saved task reports in `data/results/`, by their task IDs, because those hold the question and the report too.
    - The backend removes the definition and the topic's ledger entries when the folder is gone. Its daily cleanup checks this.
    - If no scheduler is running (no fresh heartbeat), for example during a shadow week or with the Mac asleep, the backend removes the folder and those reports itself, under the scheduler lock, so a delete never waits on the scheduler.
  - `data/topics/` sits on the same persistent volume as `data/results/`. The backup CronJob tars all of `/app/data`, so it's covered.
- **Creating a topic:**
  - From the web app's **Track this** button under a research answer, which fills in the standalone question the supervisor wrote (saved with the result from phase A), or from the topics page.
  - **Caps checked by the API** when a topic is created or resumed: at most `max_active_topics` active topics, and `every` of at least a day.
  - **The question goes through the intake guard (checkpoint 1) when the topic is created**, as `POST /api/guard/input` does. In act mode a decline or a consent outcome refuses the topic: an unattended run can't ask for consent later. In shadow mode the outcome is recorded.
  - The first run happens at the first due time.
  - **Run now** increments `run_requests` in the definition. When it exceeds `run_requests_seen`, the scheduler starts one run on its next tick. So every run is started by the scheduler and counts toward the caps, and neither process writes the other's file.
  - **Caps checked by the scheduler** before each run: the daily run count and the monthly allowance (§6.8). A request over a cap is recorded in `state.json` as skipped (`skipped: daily_cap` or `budget`), and the web app shows it, so Run now never fails silently.

### 6.2 Scheduler (`app/scheduler.py`)

- **The loop:** every 60 s, load the topics that are `active` with no hold, and pick those whose `next_due` has passed or that have an unhandled run request. Start at most 2 runs at a time, oldest due first. Advance `next_due` to the next slot after now: missed slots are **not** backfilled, so a topic runs once on catch-up.
- **Single runner:**
  - A lock file `data/topics/.scheduler.lock` holds the PID and a heartbeat refreshed every 60 s. A second scheduler that finds a fresh heartbeat (under 180 s old) exits.
  - Compose and Kubernetes run one instance; the lock guards against a mistaken second one.
  - With `WEBAGENT_SCHEDULER_ENABLED=false` (`schedule.enabled: false`, the default in every environment until an operator turns it on after shadow week 2) the service idles and still writes its heartbeat.
- **Crash safety:**
  - Each run's task ID is fixed before the call: `uuid5(topic namespace, run_id)`. The backend accepts a caller-chosen `task_id` only with the scheduler token, and answers 409 if that task already exists.
  - Before calling the backend, the scheduler writes the run record with `status: starting` and that task ID.
  - On restart, the scheduler asks `GET /api/task/{task_id}` for each `starting` or `running` record. A task the backend knows is polled to the end; an unknown one is marked `failed` and retried only at the next slot.
  - So a crash never starts the same slot twice and never loses a run that did start, and its usage still counts against the allowance.
- **Calling the backend:**
  - `POST {WEBAGENT_BACKEND_URL}/api/task` with `X-API-Key`, **`X-WebAgent-Scheduler: <WEBAGENT_SCHEDULER_TOKEN>`** and `X-WebAgent-Client: schedule:<topic_id>`, so it has its own rate-limit bucket.
  - The body is `{"query", "mode": "always", "origin": "schedule", "topic_id", "task_id", "token_budget"}`.
  - The task goes through the intake guard at every run, as any task does. A 409 (consent needed) or 403 counts as a failed run, so a topic that the guard turns against is held after 3 failures.
  - **The backend address** comes from `WEBAGENT_BACKEND_URL`: `http://webagent-backend:8000` in Compose (the service name) and `http://127.0.0.1:8000` in the Kubernetes sidecar (same pod). The default, `http://127.0.0.1:8000`, also suits a scheduler run by hand next to a local backend.
  - `WEBAGENT_SCHEDULER_TOKEN` is a second secret given only to the backend and the scheduler, never to the web app. The web app's proxy forwards the browser's whole body with the API key, so the API key alone can't prove a caller is the scheduler.
  - It polls `GET /api/task/{id}` every 15 s, for up to 10 minutes. The backend's own 420-second limit ends the run first.
- **Startup checks:**
  - The scheduler refuses to start, when enabled, without `WEBAGENT_SCHEDULER_TOKEN`, a reachable backend, or `WEBAGENT_API_KEY` where the backend requires it.
  - The backend, without the token set, refuses every scheduler field and endpoint (so scheduling is off) and logs that once.
- **Memory:** the scheduler imports only config, the topic store and an HTTP client. Importing `app.core.config` and `app.services.result_store` measured 45 MB and loads no ML library, and a test keeps it that way (no `torch`, `transformers`, `langchain` or `chromadb` in `sys.modules`).
- **Health:**
  - The heartbeat in the lock file (exported by the backend, §6.10), which the container health check also reads.
  - It logs topic and run IDs, never question text outside dev.

### 6.3 Backend changes for scheduled runs (`api/endpoints/task.py`, `task_manager.py`)

- **New fields on `POST /api/task`:** `origin` (`schedule`), `topic_id`, `task_id` and `token_budget`. They're accepted only with a valid `X-WebAgent-Scheduler` token (a constant-time compare with `WEBAGENT_SCHEDULER_TOKEN`); without one they answer 422.
  - The API key isn't enough, because the web proxy adds it to browser requests.
  - The web app's task proxy also gets an allow-list of body fields, so browsers can't send unknown fields at all.
- **Effect of `origin: schedule`:**
  - **Budget:** `task_budget.start(token_budget)`, which **must be > 0**; 422 otherwise.
  - **Triage:** skipped (mode is `always`) and never collected.
  - **Results:** the saved record carries `origin`.
  - **Metrics:** `webagent_tasks_total` and `webagent_task_duration_seconds` get an `origin` label (`interactive` or `schedule`).
  - **Drift:** a context variable stops `DriftRecordingEmbeddings` observing query embeddings for the run. Model speed and tokens still count, since they measure the model rather than the questions.
  - **Security collection:** rows carry `origin: schedule`, so the shadow-week review can separate them.
  - **Usage:** the saved record gets `usage: {"input_tokens", "output_tokens", "searches", "tavily_credits"}`.
    - `task_budget` keeps input and output tokens apart: `together_llm.py` already receives `prompt_tokens` and `completion_tokens`, and today adds only `total_tokens`.
    - The search counter moves from one web-research pass to the run's `task_budget` counter (a context-variable object shared by the parallel agents; `state.context` wouldn't work, because parallel research runs on deep copies of the state), so `max_searches_per_task` becomes a real per-task cap. That also bounds interactive runs, which matters with or without topics.
    - Each search counts 1 credit at basic depth and 2 at advanced.
- **Saved for every research task (interactive too):** the supervisor's `standalone_query`. It's the same kind of data as `query`, and it's returned in `GET /api/task/{id}`'s `result`, so the web app has it when it shows **Track this**.
- **Tracing:** LangSmith runs get `origin` and `topic_id` as metadata, so scheduled runs can be filtered out.
- **No previous turns:** a scheduled run starts fresh. The topic's question already reads as a standalone question; the supervisor rewrote it when the topic was created from an answer.

### 6.4 History and change summaries (`app/services/topic_runs.py`)

- **Run record:** after a run completes, the scheduler writes `{run_id, task_id, status, started_at, completed_at, usage, report, sources, change}`, where `usage` is the result's input and output tokens, searches and credits. The report and sources are copied from the result, so they outlive the 30-day results cleanup.
- **Source diff:** URLs are normalised by lower-case host, stripped `www.`, no fragment, no tracking parameters (`utm_*`, `fbclid`, `gclid`), and no trailing `/`. `change.new_sources` are this run's URLs not in the last 3 runs; `change.dropped_sources` are the previous run's URLs not in this one.
- **Change summary (one model call, `agents.topic_summary` in YAML, `max_tokens` 600):**
  - The input is the previous report and the new report, each truncated to 6,000 characters, plus the new-source list. The output is at most 5 bullet points, each citing a source number, or "No material change".
  - **The call goes through checkpoint 1** like any other model call. Both reports are model output built from web pages, so they're scored as **untrusted, `retrieved_doc`**, not as `agent_output`, which the guard treats as trusted.
  - The scheduler calls a small backend endpoint, `POST /api/topics/{id}/summarize` (scheduler token), with the two reports' run IDs. It runs under its own token budget of 8,000, counted in the allowance, and **returns** the summary; the scheduler stores it in the run record.
  - **First run:** there's no previous report, so there's no summary call. The change is "first report" with all sources new, and with `notify.when: changes` the first run notifies.
  - If it fails, the run keeps its report, and the change is shown as the source diff only.
- **When to notify:** `changed` is true when there are new sources, or when the summary isn't "No material change". With `notify.when: changes`, email goes only when `changed`; with `always`, after every successful run. The scheduler asks the backend to send it: `POST /api/topics/{id}/notify` (scheduler token) with the run ID.
- **Failures:**
  - A failed or stopped run is recorded with its `error_code`: timeout, budget, cancelled, escalated, intake refused (403 or 409 at start), backend unreachable, rate limited, poll timeout, or not started (after a crash). Each counts toward `failures_in_a_row`, and a completed run resets it.
  - After **3 failures in a row** the scheduler sets `hold: failures`. If email is on, subscribers get one notice (through the backend, §6.6), and the operator metric `webagent_topic_held_total{reason}` rises.

### 6.5 Web app (`src/app/topics/`, `src/components/topics/`)

- **Track this:** a button under each research answer (not direct answers). It opens a small form: name, prefilled from the report title; frequency; day and time, in the browser's time zone; notify when; optional email.
- **`/topics`:** a list of name, schedule, next and last run, status, and "N new sources" on the last run. It has pause/resume, Run now and delete (with confirmation).
- **`/topics/[id]`:** a timeline of runs, each with the change summary or source diff, a link to the full stored report (rendered like a chat report, sources panel included), and a list of new sources.
- **Proxies:** `src/app/api/topics/**` proxy to the backend with `X-API-Key`, like `/api/task`, so the middleware's sign-in check applies.
  - They allow-list routes and methods: the scheduler-only endpoints (`summarize`, `notify`) aren't proxied.
  - Nothing changes state on GET. With the session cookie's `SameSite=Lax`, a cross-site POST or DELETE carries no session, so it's refused (CSRF).
- **Stored reports** render with the chat's `react-markdown` path, which renders no raw HTML (no `rehype-raw` anywhere in `src/`). An injected `<script>` in a report stays text.
- **Confirm and unsubscribe pages:** `/notify/confirm` and `/notify/unsubscribe` (and their `/api/notify/*` proxies) are the only routes reachable **without sign-in**, because they come from email. They're added to the middleware matcher's exclusions.
  - **Opening one changes nothing.** The page shows the topic name and a button, and the button's POST, with the signed token, does the confirming or unsubscribing.
  - That's because mail security scanners (Outlook Safe Links and others) open every link in a message. A GET that confirmed would defeat double opt-in, and a GET that unsubscribed would silently unsubscribe people.
- **CSP:** unchanged, since everything is same-origin.

### 6.6 Email (`app/services/notify_email.py`, phase B)

- **The backend sends every email:** confirmations when someone subscribes, and run and hold notices when the scheduler asks. So one process owns the SMTP settings, the send caps and the email ledger (`data/notify/`), and the scheduler needs no SMTP secrets.

- **Transport:** `notifications.email.transport` is `smtp`, or `log`, which writes the message to the log instead of sending it (for dev and tests; the default in dev). SMTP uses `smtplib` with STARTTLS (port 587) or TLS (465). Settings: `WEBAGENT_SMTP_HOST`, `_PORT`, `_USERNAME`, `_PASSWORD`, `_FROM` (a fixed address), `_TLS`. The secrets live in `.env.local` or the deployment's secret store, never in YAML. A send has a 20-second timeout, with 2 retries and backoff; failures are logged by topic ID only.
- **Double opt-in:**
  - Adding an address sends one confirmation email with a link to `/notify/confirm?token=…`. The token is HMAC-signed with `WEBAGENT_SECURITY_SECRET_KEY`, bound to (topic, address, purpose), valid 48 h and single-use.
  - Nothing else is sent until it's confirmed. An unconfirmed address is dropped after 48 h.
  - The confirmation email contains only the topic name and the link, no report text, so it can't be used to send content to strangers.
- **Unsubscribe:**
  - Every email has a `/notify/unsubscribe?token=…` link (signed, no expiry, safe to repeat).
  - It also has the RFC 8058 one-click headers, `List-Unsubscribe: <https://…/api/notify/unsubscribe?token=…>` and `List-Unsubscribe-Post: List-Unsubscribe=One-Click`. Mail clients POST to that address, and Gmail and Yahoo expect it from senders.
  - Unsubscribing removes the address from the topic at once.
- **Content (plain text, UTF-8, no HTML, no images, no tracking):**
  - **Subject:** `WebAgent: <topic name> — <date>`, with the name stripped of CR, LF and other control characters and cut to 60 characters. Headers are built with `email.message.EmailMessage`, so header injection isn't possible.
  - **Notice mode (D4):** the run's date, "N new sources" with their **domains only** (no full URLs), and a link to `/topics/<id>` (needs `WEBAGENT_PUBLIC_URL`).
  - **Summary mode (only with `input_mode: act`):** the change summary bullets, followed by the new sources as domain plus path, cut to 100 characters, then the same link.
  - A footer says why the email was sent, how to unsubscribe, and that it was written by an AI system.
- **Send caps:**
  - At most 1 email per topic per run, and at most **5 subscribers per topic**.
  - At most **50 emails a day per instance**; over that, emails are dropped, counted, and the operator is alerted.
  - At most 3 unconfirmed confirmation emails per address per day.
- **Unconfirmed addresses** are removed by the backend's daily cleanup (with the results cleanup).
- **Topics paused for 90 days** are deleted by the scheduler, which owns `paused_since`: it calls the backend's `DELETE /api/topics/{id}`, and at 83 days it asks the backend for a deletion notice to confirmed subscribers.
- **Startup check:** with `notifications.email.enabled: true` and `transport: smtp`, startup fails unless the SMTP host, From address and `WEBAGENT_PUBLIC_URL` are set and the secret key is set (it's already required in uat and prod). It also fails with `mode: summary` while `security_guard.input_mode` isn't `act` (D4).

### 6.7 Security

| Threat | Handling |
|---|---|
| T1. Injected web content reaches an inbox | Checkpoint 1 drops flagged chunks, **in act mode only**. While in shadow: notice mode (D4), with no model text and only domains. The summary call is itself guarded. Plain text only |
| T2. Email as a spam or harassment channel: anyone with the access code adds a stranger's address | Double opt-in; the confirmation email carries no user content apart from the topic name (cut short, control characters stripped); caps per address and per instance; a fixed From address |
| T3. Header or body injection through the topic name | `EmailMessage` headers; names cleaned on input (no control characters, at most 80) and again for the subject |
| T4. Cost exhaustion: many topics or Run now spam | §5 D5 caps; the monthly allowance is tracked in `data/topics/usage-<YYYY-MM>.json`, and topics are held (`hold: budget`) when it's used up; Run now counts too; rate limits per topic bucket |
| T5. Unattended guard outcomes | `consent`: no person can confirm, so the step is skipped and noted in the run. `escalate`, or an intake refusal at run start: the run fails, and 3 in a row hold the topic. `verify` with the judge off acts as the policy says |
| T6. The intent drifts over repeated runs (rogue subgoals) | Checkpoint 2's intent record is the topic's question at every run; the rules (shadow-only for now) apply unchanged |
| T7. Forged scheduler fields from a browser | `origin`, `topic_id` and `token_budget` need the scheduler token, which the web app never has, so the API key alone isn't enough (the proxy adds it to browser requests). The proxy also allow-lists body fields |
| T8. Forged, replayed or scanner-followed confirm and unsubscribe links | HMAC tokens bound to topic, address and purpose; confirm is single-use (used tokens recorded in the email ledger, `data/notify/`); both need a POST from a button press or a one-click unsubscribe, never a GET |
| T9. Personal data | Emails are stored only with their topic and deleted with it or on unsubscribe; never logged (`describe_text` length only) |
| T10. Notice emails still carry attacker-chosen domain names | Only the domains of sources checkpoint 1 didn't flag, shown as text, not links. In shadow mode this needs the would-be drops recorded (`security.shadow_flagged`, new) |

### 6.8 Cost controls

- **Per run:** the search cap, made per-task in phase A (at most 6 searches in uat and 10 in prod as configured today), plus a token budget that is always set. Its default, `token_budget_default`, is set to `1.5 × p90` of shadow week 2's measured `tokens_used`, and is 60,000 until then. A topic can't change it.
- **Per month:** the scheduled allowance (D5) covers Tavily credits, counted per search at 1 or 2 by depth, and Together dollars, from `tokens_used` at the configured prices. At 80% the operator alert fires.
  - **A run starts only if the remaining allowance covers its worst case,** counting runs already in progress: 15 credits, and `token_budget` plus the summary's 8,000 tokens at the output price. So the allowance is never passed.
  - When the next run can't fit, the scheduler holds every topic (`hold: budget`) until the next month, or until the operator raises the allowance.
  - Prices come from config (`llm.price_input_per_million`, `llm.price_output_per_million`, today $0.35 and $1.50 for the configured Together model). They're also used to report interactive cost, and they're set once per model change.
- **Per day:** 20 scheduled runs, 50 emails.

### 6.9 Privacy and retention

- **What's stored:** topic text and stored reports, the same kind of data as `data/results/`, kept per D7. Subscriber emails are deleted with the topic, on unsubscribe, or 48 h after an unconfirmed add.
- **Logs:** outside dev, logs carry topic and run IDs and text lengths only.
- **Other retention:** the email ledger (hashed addresses, used tokens) is kept 30 days, and monthly usage files 13 months.
- **The README's "What WebAgent doesn't do"** changes: recurring research and email become supported; per-user privacy stays out.

### 6.10 Observability

- **Metrics:** the backend exports them, so Prometheus keeps one scrape target and the scheduler needs no metrics port or shared multiprocess folder. A custom collector reads the scheduler's files on each scrape. Counters come from `counters.json`, which only grows (pruning old runs doesn't change it); gauges come from `state.json` and `usage-*.json`; the heartbeat comes from the lock file. The email counter is the backend's own (multiprocess) counter:
  - `webagent_topic_runs_total{outcome}`, `webagent_topic_run_seconds`, `webagent_topic_changes_total{kind}` (new sources, summary changed);
  - `webagent_topic_budget_used{kind}` and `webagent_topic_budget_limit{kind}` (credits, dollars), `webagent_topic_held_total{reason}`, `webagent_topics{state}` (active, paused, held);
  - `webagent_email_sent_total{kind, outcome}` (confirm, notice, summary, hold, deletion; sent, failed, capped), an ordinary counter incremented by the backend;
  - `webagent_scheduler_heartbeat_timestamp`, `webagent_scheduler_missed_slots_total`.
- **Grafana alert rules (two new):** "scheduler down" (heartbeat older than 10 min) and "scheduled budget 80%". The six drift alerts are unchanged, and scheduled runs don't feed the embedding drift.
- **Dashboard:** a "Recurring research" row on the WebAgent Backend dashboard.

### 6.11 Configuration

```yaml
schedule:
  enabled: false                 # WEBAGENT_SCHEDULER_ENABLED; true only where the scheduler should run
  max_active_topics: 10
  max_runs_per_day: 20
  max_concurrent_runs: 2
  history_runs_kept: 20
  pause_after_failures: 3
  monthly_tavily_credits: 400
  monthly_together_usd: 5.0
  token_budget_default: 60000    # replaced by 1.5 × p90 measured in shadow week 2
  max_token_budget: 200000       # the most /api/task accepts as a scheduled run's token_budget
  paused_topic_days: 90
llm:
  price_input_per_million: 0.35  # $ per million tokens, the configured Together model (2026-10-16)
  price_output_per_million: 1.50
notifications:
  email:
    enabled: false
    transport: log               # log (dev, tests) | smtp
    mode: notice                 # notice | summary (summary needs security_guard.input_mode: act)
    max_subscribers_per_topic: 5
    max_emails_per_day: 50
```

New environment settings: `WEBAGENT_SCHEDULER_TOKEN` (backend and scheduler only; `deploy.sh` and the Kubernetes secret get it, and it's never in the web app's environment), `WEBAGENT_PUBLIC_URL`, `WEBAGENT_SMTP_HOST`, `WEBAGENT_SMTP_PORT`, `WEBAGENT_SMTP_USERNAME`, `WEBAGENT_SMTP_PASSWORD`, `WEBAGENT_SMTP_FROM`, `WEBAGENT_SMTP_TLS`. Every one also goes in `.env.local.example`.

## 7. Cost model

| Item | uat | prod | Source |
|---|---|---|---|
| Research passes per run, at most | 3 | 5 | `max_follow_up_questions` (total passes) |
| Searches per run, at most | 6 | 10 | 2 per pass |
| Tavily credits per run, at most | 6 (all basic) – 9 (main searches advanced) | 10 – 15 | 1 credit basic, 2 advanced; only the main search of a pass can be advanced |
| Runs a month, daily topic | 30 | 30 | — |
| Runs a month, weekly topic | 4.3 | 4.3 | — |
| Tavily credits a month, daily topic, at most | 180 – 270 | 300 – 450 | product of the above |
| Together cost per run | **not measured yet** | | shadow week 2 (3.10.0) records total `tokens_used` only; input and output are split from phase A |

- **Tavily:** with the 400-credit allowance, about 1 daily prod topic at the worst case (450), or 6 weekly ones (65 each), fits. A typical run makes fewer searches than the worst case; shadow week 2's `searches` counts will show how many. The caps make the limit visible rather than a surprise on the bill.
- **Together:** prices are $0.35 per million input tokens and $1.50 per million output (the configured model, 2026-10-16). Until phase A splits input and output, a measured total can be priced at the output price, as an upper bound. As an **assumption to replace with measured numbers**, a run of 60,000 input and 15,000 output tokens costs about $0.044. A daily topic would then cost about $1.30 a month, and the $5 allowance covers about 110 runs.
- **Change summaries:** one more call per run, at most 600 output tokens and about 4,000 input, so under $0.003.

## 8. Phases and releases

| Phase | Release | Contents | Ends when |
|---|---|---|---|
| A | 3.11.0 | The per-task search cap's value from shadow week 2's search counts (spec task A0); Topic store (one writer per file); scheduler service (lock, catch-up, crash safety) with its own token; Compose service and Kubernetes sidecar; `origin`/`topic_id`/`task_id`/`token_budget` on `/api/task` with fixed task IDs; holds and the Run now and Resume counters; an allow-listed task proxy; a real per-task search cap and usage (tokens split, searches, credits) on every result; `standalone_query` saved; intake guard at topic creation; drift and triage exclusions; run history with source diff; web app Track this, `/topics`, `/topics/[id]`; caps and the monthly allowance; metrics, the scheduler-down alert; docs | Tests pass; a weekly and a daily topic run on schedule for a week on the local stack after shadow week 2, with the allowance tracked |
| B | 3.12.0 | `security.shadow_flagged` (checkpoint 1's would-be drops in shadow mode, for notice emails); change-summary call (guarded); SMTP email with double opt-in, unsubscribe, notice mode, and summary mode gated on act; send caps; hold notices; the budget alert | A confirmed address gets notice emails for real changes only; unsubscribe works from a mail client; caps tested |

**Prerequisites:**
- Shadow week 2 finished and reviewed, for D8, the measured `tokens_used`, and the act-mode decision for D4.
- A decision on each item in §11.
- An always-on host for real schedules. The Mac works while it's awake; catch-up runs once when it wakes.

## 9. Tests

- **Scheduler:**
  - due selection and next slot across DST changes;
  - no backfill;
  - the lock (a second scheduler exits; a stale heartbeat is taken over);
  - crash recovery with the fixed task ID (`starting` that the backend knows, `starting` it doesn't, `running`);
  - concurrency cap.
- **Backend:**
  - `origin` fields rejected without the scheduler token, including with the API key, and through the web proxy (allow-list);
  - the search cap counts across passes; usage on the result (input and output tokens, searches, credits);
  - `standalone_query` saved; intake guard at creation (act mode: decline and consent refused);
  - the API and the scheduler never write the same file (definition, state, counters, email ledger), and Run now, resume, delete and pause are honoured on the next tick;
  - the fixed task ID: a crash after the POST resumes polling the same task; the backend answers 409 for an existing ID and refuses a chosen ID without the scheduler token;
  - an intake refusal at run start counts as a failure, and 3 failures hold the topic;
  - two backend workers editing one topic (subscribe and pause at once) lose neither change (flock);
  - delete with and without a running scheduler removes the definition, state, runs and the scheduled tasks' result files;
  - caps: the 11th active topic is refused by the API; a Run now over the daily cap or the allowance is recorded as skipped; a run doesn't start when its worst case doesn't fit;
  - DST: a skipped local time and a repeated one, each run once;
  - the scheduler imports no ML library;
  - the backend refuses scheduler fields when `WEBAGENT_SCHEDULER_TOKEN` is unset;
  - budget required (422 for 0);
  - drift doesn't observe scheduled queries;
  - triage isn't collected;
  - collection rows carry `origin`.
- **History:** source normalisation and diff; the first run (no summary, notifies); the summary call's own budget; counters stay monotonic when old runs are pruned; summary failure falls back to the diff; 3 failures hold the topic and Resume clears it; the monthly allowance holds topics, and the hold clears next month.
- **Email** (with a fake SMTP server, `aiosmtpd` in dev dependencies or a stub):
  - confirm token single-use and expiry; unsubscribe idempotent;
  - a GET of the confirm or unsubscribe page changes nothing (scanner prefetch); one-click POST works;
  - no email to an unconfirmed address;
  - header injection attempts in names;
  - notice mode carries no model text;
  - summary mode refused unless `input_mode: act`;
  - caps per topic, per address and per day;
  - the startup check.
- **Web app:** Track this prefill (from the result's `standalone_query`); the topics proxy doesn't forward `summarize` or `notify`; a cross-site POST has no session; topics list and detail; confirm and unsubscribe pages reachable without sign-in, and nothing else newly reachable (middleware test).
- **Red team:** a run whose stored report contains an injected instruction, under shadow and act. The email in notice mode never contains it, and the summary call is scored by checkpoint 1.

## 10. Risks and limits

- **Shadow mode limits email.** Until checkpoint 1 acts, emails carry only notices. That's a real reduction in usefulness, and the act-mode decision after shadow week 2 unlocks summaries.
- **A laptop isn't a server.** Schedules on the Mac fire only while it's awake and Docker is running; a real deployment needs an always-on host.
- **Change summaries can be wrong:** a model may call a reworded report a change, or miss one. The source diff is exact, and `notify.when: always` exists for users who want every run.
- **Instance-wide topics** mean anyone with the access code sees every topic and its reports, the same exposure as loaded documents today. That's acceptable for a single-operator instance and not for a shared one.
- **Links in email need a reachable web app.** On the local Mac, `WEBAGENT_PUBLIC_URL` is `http://localhost:3002`, so links work only on that machine, and confirmation must be clicked there. Real use needs the web app on a public URL (Vercel or the cluster's Ingress).
- **Email deliverability** depends on the provider's SPF and DKIM setup. A misconfigured From domain lands in spam. The runbook will describe the check.
- **The per-task search cap changes interactive research too:** today's per-pass cap never binds, so making it per task can lower how much a long interactive run searches. Phase A measures the effect on shadow week 2's runs first (searches per run, by pass), and sets the caps so typical runs are unaffected.
- **Costs before measurement:** the Together estimate is an assumption until shadow week 2's `tokens_used` is in. The caps protect the budget either way.

## 11. Decisions (2026-10-18)

The user accepted every recommendation:

1. **D1** The scheduler is a separate one-process service: a Compose service, and a sidecar in the backend pod on Kubernetes.
2. **D2** Topics are instance-wide, until user accounts exist.
3. **D3** Email is sent over provider-neutral SMTP, with a `log` transport in dev. **Still open:** the provider and From address, which must be chosen before phase B ships, with SPF and DKIM set up or the provider's sending domain.
4. **D4** Emails carry a notice only while `input_mode` is shadow; summary emails come only after checkpoint 1 acts, and a startup check enforces it.
5. **D5** The caps are as proposed: 10 active topics, 20 scheduled runs a day, daily at most, and a monthly allowance of 400 Tavily credits and $5 of Together.
6. **D6** Topics notify on changes by default, and each can switch to `always`.
7. **D7** Each topic keeps its last 20 runs. Topics paused for 90 days are deleted, with a notice 7 days before.
8. **D8** Build 3.11.0 and 3.12.0 now; deploy after shadow week 2, with the scheduler disabled until then.
9. **Search cap.** `max_searches_per_task` becomes a real per-task cap for every run, interactive and scheduled, with its value set from shadow week 2's search counts.

Next: the spec (`planning/specs/recurring-research/`), then phase A.

## 12. Files likely to change

- **Backend, new:**
  - `app/scheduler.py`, `app/services/topics.py`, `app/services/topic_files.py`, `app/services/schedule_math.py`, `app/services/topic_runs.py`, `app/core/scheduler_auth.py`, `app/services/notify_email.py`, `app/services/notify_tokens.py`;
  - `app/api/endpoints/topics.py` (CRUD, run, resume, runs, summarize, notify), `app/api/endpoints/notify.py` (lookup, confirm, unsubscribe, one-click), a daily topics and notify cleanup next to the results cleanup.
- **Backend, changed:**
  - `app/api/endpoints/task.py`, `app/services/task_manager.py`, `app/services/task_budget.py` (per-run budget, split counts), `app/services/together_llm.py` (input and output), `app/agents/web_research.py` (per-task search cap and credits), `app/agents/supervisor.py` or `task_manager.py` (`standalone_query` saved);
  - `app/security/hop_guard.py` and `app/security/ledger.py` (`shadow_flagged`, phase B), `app/services/drift_embeddings.py` (skip scheduled queries), `app/services/triage_collect.py`, `app/security/records.py` (`origin`);
  - `app/core/metrics.py`, `app/core/config_def.py`, `app/core/config.py`, `app/core/startup_checks.py`, `config/{dev,uat,prod}.yaml`;
  - `main.py` (routers), `docker-compose.yml` (scheduler service), `kubernetes/deployment.yaml` (scheduler sidecar), `deploy.sh`, `grafana/` (row and two alert rules), a topic-metrics collector in `app/core/metrics.py`.
- **Web app:**
  - `src/app/topics/`, `src/app/notify/{confirm,unsubscribe}/`, `src/app/api/topics/**`, `src/app/api/notify/**`;
  - `src/components/topics/`, a Track this button in `MessageList.tsx`, `src/middleware.ts` (the two public pages and their proxies), `src/app/api/task/route.ts` (allow-listed body fields).
- **Docs:** README ("Using WebAgent", "What WebAgent doesn't do"), `ARCHITECTURE.md`, `api_reference.md`, `RUNBOOK.md` (scheduler, SMTP setup, deliverability check, caps), `.env.local.example`, CHANGELOG.
- **Tests:** `test_scheduler.py`, `test_topics.py`, `test_topic_runs.py`, `test_notify_email.py`, `test_task_origin.py`; web tests for the topics pages, the proxies and the middleware.

## 13. Checks against the code (2026-10-17 and 2026-10-18)

v1 was checked against the 3.10.1 code. These gaps were found and fixed in v2:

1. **Search cost was wrong.** `max_searches_per_task` is counted per research pass, not per task (`web_research.py`, `capped_search`). One pass makes at most 2 searches, so the cap never binds. A run makes at most `max_follow_up_questions` passes (3 or 5; `senior_research.py` stops at that iteration). The real bound is 6 or 10 searches and 9 or 15 credits a run, and §2, §6.8 and §7 now use it. Phase A makes the cap per task. (My earlier answer, "about 40 credits a run", was also too high.)
2. **The API key can't identify the scheduler.** The web proxy forwards the browser's raw body and adds the key, so a browser could have set `origin`, `topic_id` and its own budget. There's now a separate scheduler token, and the proxy allow-lists body fields (T7).
3. **Kubernetes can't attach the data volume to a second pod** (ReadWriteOnce), so the scheduler is a sidecar in the backend pod (D1).
4. **Two writers on one topic file** (the API pausing while the scheduler advances `next_due`) would lose updates. Now there's one writer per file: definition (API), state, runs and usage (scheduler). Run now became a flag that the scheduler picks up.
5. **Mail scanners open links.** A GET that confirmed or unsubscribed would let scanners subscribe or unsubscribe people. Both now need a POST, and one-click unsubscribe headers are added (RFC 8058).
6. **The dollar allowance couldn't be priced:** the budget counter keeps only total tokens. It now keeps input and output, and searches and credits are recorded per run.
7. **A separate process can't share the backend's Prometheus multiprocess folder.** The backend now exports topic metrics from the topic files.
8. **No previous report on the first run.** It's now defined: no summary call, all sources new, and it notifies.
9. **Stored reports would have been scored as trusted.** The guard has no "report" source, and `agent_output` is trusted. They're now scored as `retrieved_doc`.
10. **Track this needed the standalone question,** which results don't save. It's now saved for every research task.
11. **A harmful or consent-needing question could be scheduled.** Topic creation now goes through the intake guard.
12. **There was no way to exercise email in dev and tests without an SMTP server.** A `log` transport is the dev default.

A second reading of v2 found and fixed more:

13. **One writer per file still had exceptions.** The scheduler would have set `paused_budget` and the failure pause in the definition, and cleared `run_requested` there. Holds now live in the scheduler's state, and Run now and Resume are counters the API increments and the scheduler compares.
14. **A crash right after `POST /api/task` succeeded would have lost a running task,** uncounted against the allowance. Task IDs are now fixed before the call (uuid5 of the run ID), so the scheduler finds the task after a restart.
15. **Email had two senders** (backend confirmations, scheduler notices) with separate caps and ledgers. The backend now sends all email.
16. **Metrics counted from run files would go down when old runs are pruned.** Counters now come from a counters file that only grows.
17. **The summary call couldn't use "the rest of the run's budget",** because the run's budget ends with the run. It now has its own budget of 8,000 tokens, counted in the allowance.
18. **G1 promised 10 minutes,** which 10 due topics at 2 at a time can't meet. It now says 15 minutes, for up to 10 topics due at once.
19. **An intake refusal at run start** (consent needed, declined) wasn't handled. It's now a failed run that counts toward the hold.
20. **Who removes unconfirmed addresses and 90-day paused topics** wasn't said. The backend's daily cleanup does it, with a notice 7 days before deleting a topic.

A third reading, of v3, found and fixed these:

21. **The backend runs two uvicorn workers,** so "one writer per file" wasn't true for definitions and the email ledger. They're now changed under `fcntl.flock` (the helper in `drift_store.py`).
22. **Deleting by renaming the topic folder** moved the scheduler's files from under it, and left the scheduled tasks' reports in `data/results/` behind. A delete is now a tombstone; each process removes its own files, including those result files, and the backend does it alone when no scheduler is running.
23. **The scheduler's backend address** differs in Compose (service name) and the Kubernetes sidecar (localhost). It's now stated.
24. **DST** gaps and repeats are now defined, and `week` needs a weekday.
25. **Where each cap is checked** wasn't said. Active topics are checked by the API; daily runs and the allowance by the scheduler, with skips visible in the web app.
26. **Two runs at once could pass the monthly allowance.** A run now starts only if its worst case fits, counting runs in progress.
27. **Prices** existed only in a script. They're now config, also used for interactive cost.
28. **Confirm-token storage** was in two places (topic, ledger). It's now the ledger only.
29. **Track this** needs `standalone_query` in `GET /api/task`'s result, not only in the saved file.
30. **There were no startup checks for the scheduler token,** and no deploy wiring for it. Both are added.
31. **"Override per topic down to that floor"** didn't make sense. Topics can't change the token budget.
32. **The scheduler's memory claim** is now measured (45 MB for its imports) and kept by a test.
33. **Not stated before:** the email-link limitation on a laptop, CSRF for the topic routes, report rendering without raw HTML, and LangSmith metadata.

Writing the spec (2026-10-18) found and fixed more:

34. **The per-task search counter can't live in `state.context`.** Parallel research runs on deep copies of the state, and only reports, logs, history and the ledger are merged back. It lives in the run's `task_budget` counter instead.
35. **In shadow mode, checkpoint 1 records nothing per item,** so "only unflagged domains" in notice emails couldn't be done. Would-be drops are now recorded as `security.shadow_flagged` (spec task B0).

Checking the plan and the spec against each other (2026-10-18) found and fixed these:

36. **The 90-day deletion had two owners:** the plan's backend cleanup, the spec's scheduler. It's now the scheduler, which owns `paused_since`, calling the backend's `DELETE` (and asking for the notice at 83 days).
37. **Metric names:** the plan said `webagent_task_total`; the real counter is `webagent_tasks_total` (and `webagent_task_duration_seconds`). `webagent_topic_changes_total` is now in the spec, and `webagent_topic_budget_limit` in the plan. The email kinds (hold, deletion, not "pause notice") and the counter's source now match.
38. **Topic and state fields:** `status: deleted` (tombstone), the subscriber's `added_at`, and the state's `skipped`, `paused_since` and `schedule_seen` are now in the plan. Resume also sets `status: active`.
39. **Config:** `max_token_budget` and `paused_topic_days` are now in the plan's config block.
40. **Failure kinds and the source diff** (trailing slash; `dropped_sources` against the previous run) now match the spec.
41. **The scheduler's memory:** the plan said "about 100 MB" and the spec's Kubernetes limit was 128 Mi, below the 150 MB requirement. Both now say imports measured at 45 MB, target under 150 MB, limit 192 Mi.
42. **`schedule.enabled` defaults:** the plan said false "in dev"; it's false in every environment (RNFR-05).
43. **The notify pages' name lookup** would have needed a fifth public route. It's now a server-side call, so only the four public routes exist.
44. **Phases and files:** phase A names the search-cap measurement (A0) and phase B `shadow_flagged` (B0). §12 lists the spec's new modules (`topic_files`, `schedule_math`, `scheduler_auth`) and the guard files B0 changes. Retention for the ledger (30 days) and usage files (13 months) is stated.

