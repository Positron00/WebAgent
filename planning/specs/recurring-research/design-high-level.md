# High-level design: recurring research

## 1. Components

```text
 Browser (signed in)                     Mail client (subscriber)
   │  Track this, /topics, /topics/[id]       │  confirm / unsubscribe (button → POST), one-click POST
   ▼                                          ▼
┌─ Web app (Next.js) ───────────────────────────────────────────────────────────────────┐
│ /api/topics/** ── allow-listed proxy, X-API-Key ─┐   /api/notify/* ── public proxy ─┐   │
│ /api/task ─────── allow-listed body fields ──────┤                                  │   │
└──────────────────────────────────────────────────┼──────────────────────────────────┼───┘
                                                   ▼                                  ▼
┌─ Backend (FastAPI, 2 workers) ───────────────────────────────────────────────────────────┐
│ topics API (definitions; flock) · notify API (tokens, ledger; flock) · email sender     │
│ /api/task (+ scheduler fields with X-WebAgent-Scheduler) · summarize, notify (scheduler) │
│ research workflow, security guard, task_budget (tokens, searches, credits)              │
│ metrics: /metrics + topic collector (reads the scheduler's files)                       │
└──────────────▲───────────────────────────────────────────────────────────────────────────┘
               │ HTTP: X-API-Key + X-WebAgent-Scheduler + X-WebAgent-Client: schedule:<topic>
┌─ Scheduler (one process, same image; no ML libraries) ──────────────────────────────────┐
│ 60 s tick · lock + heartbeat · due slots (zoneinfo) · caps and allowance · fixed task   │
│ IDs · polling · run records · source diff · holds · usage · counters                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
               │ shared data folder (/app/data)
               ▼
 data/topics/<id>.json (backend) · <id>/state.json, <id>/runs/*.json (scheduler)
 data/topics/usage-YYYY-MM.json, counters.json, .scheduler.lock (scheduler) · data/notify/ (backend)
```

| Component | Owns | Never does |
|---|---|---|
| Backend | Definitions, the email ledger, sending email, running every task, the guard, metrics | Write a scheduler file (apart from REC-114's no-scheduler delete under the lock) |
| Scheduler | Deciding when runs start, state, run records, usage, counters, holds | Run agents or models, send email, write a definition |
| Web app | Pages and proxies | Hold the scheduler token, or forward unknown fields or scheduler routes |

## 2. Flows

### 2.1 Create a topic (phase A)

1. **Track this** on a research answer opens the form, prefilled with the result's `standalone_query` and title.
2. `POST /api/topics` (through the proxy) validates the fields (REC-100) and the cap (REC-102), and checks the query with checkpoint 1 (REC-103).
3. Under flock, the backend writes `data/topics/<id>.json` with `status: active`, then answers 201.
4. On its next tick the scheduler sees a topic without `state.json`. It writes one with `next_due` set to the first slot after now.

### 2.2 A scheduled run (phase A)

```text
tick ─► due? caps ok? allowance covers worst case? ─► run record: starting, task_id = uuid5(run_id)
     ─► POST /api/task (scheduler fields) ─► 202 ─► record: running
     ─► poll GET /api/task/{id} every 15 s (≤ 10 min) ─► completed | failed
     ─► record: report, sources, usage, change (source diff) ─► usage, counters, state (failures, hold)
     ─► (phase B) summarize ─► notify decision ─► POST /notify ─► backend sends email
```

The backend runs the task like any other, except that it uses `token_budget`, skips triage and its collection, skips embedding drift, and labels results, collection and metrics with `origin` (REC-108).

### 2.3 Run now, pause, resume, delete

- **Run now:** `run_requests += 1` (backend). The scheduler starts one run when the request is unhandled, subject to the caps (a skip is recorded).
- **Pause and resume:** `status: paused` or `active`, plus `resumes += 1` (backend). Resume also clears a `failures` hold (scheduler).
- **Delete:** a tombstone (backend). Then the scheduler removes its files and the runs' results; the backend removes the definition. If there's no scheduler, the backend does both (REC-114).

### 2.4 Subscribe, confirm, unsubscribe (phase B)

1. **Subscribe:** `PATCH /api/topics/{id}` with a new address makes an unconfirmed entry, and the backend sends a confirmation email.
2. **Confirm:** the link opens `/notify/confirm?token=…`, which shows the topic name and a **Confirm** button. Its POST checks the token (signature, expiry, unused, purpose) and marks the address confirmed.
3. **Unsubscribe:** every email's link, or the mail client's one-click POST, removes the address.

## 3. Data layout

| Path | Writer | Contents | Retention |
|---|---|---|---|
| `data/topics/<id>.json` | backend (flock) | Definition (REC-100), subscribers | Until deleted |
| `data/topics/<id>/state.json` | scheduler | `next_due`, `last_run`, `failures_in_a_row`, `hold`, `skipped`, `run_requests_seen`, `resumes_seen`, `paused_since` | With the topic |
| `data/topics/<id>/runs/<run_id>.json` | scheduler | Run record (REC-111) | Last 20 per topic |
| `data/topics/usage-<YYYY-MM>.json` | scheduler | Credits, dollars, runs | 13 months |
| `data/topics/counters.json` | scheduler | Monotonic counters for metrics | Kept |
| `data/topics/.scheduler.lock` | scheduler (the backend only under REC-114) | PID, heartbeat, `enabled` | — |
| `data/notify/ledger-<date>.json` | backend (flock) | Sends by kind and outcome, hashed addresses, used confirm tokens | 30 days (`RESULTS_RETENTION_DAYS`) |
| `data/results/<task_id>.json` | backend (existing) | Each scheduled task's result, with `origin` and `topic_id` | 30 days, or the topic's deletion |

`paused_since` is written by the scheduler when it first sees `status: paused`, so the 90-day rule (REC-115) has a date without the backend writing scheduler state.

## 4. Failure handling

| Failure | Effect | Handling |
|---|---|---|
| Backend unreachable | No run starts | The run is marked `failed` (`backend_unreachable`), counts as a failure, and is retried at the next slot |
| Task fails (timeout, budget, guard, error) | No new report | Recorded with `error_code`; 3 in a row → `hold: failures` (REC-112) |
| Scheduler crash or restart | — | Fixed task IDs; `starting` and `running` records reconciled (REC-106) |
| Mac asleep, host down | Missed slots | Each topic runs once on catch-up; `webagent_scheduler_missed_slots_total` counts the misses |
| Two schedulers | — | The lock; the second one exits (REC-105) |
| Allowance short | No runs | `hold: budget` on every topic until the month changes or the allowance is raised (REC-113) |
| Summary call fails | — | The source diff only (REC-130) |
| SMTP fails | No email | 2 retries, then recorded as `failed` in the ledger and metrics; the run is unaffected |
| Disk full | Writes fail | Writes are atomic (the old file is kept). The scheduler logs and skips the tick; the existing disk alert in the runbook applies |

## 5. Deployment

| Where | Scheduler | Backend URL | Data |
|---|---|---|---|
| Dev (by hand) | `python -m app.scheduler` from `backend/` | `http://127.0.0.1:8000` | `backend/data` |
| Compose | Service `webagent-scheduler`, same image, `restart: unless-stopped` | `http://webagent-backend:8000` | The same volume or bind mount as the backend |
| Kubernetes | Container `scheduler` in the backend pod | `http://127.0.0.1:8000` | The same `ReadWriteOnce` claim |

`schedule.enabled` is false everywhere until an operator turns it on after shadow week 2 (RNFR-05).

## 6. Security summary

- **Credentials:**
  - The scheduler token gates every scheduler field and endpoint; the API key alone isn't enough.
  - The web app forwards only allow-listed fields and routes.
- **Every scheduled run gets the full guard:** an intake check at creation and at every run, then checkpoints 1 and 2 as for any task. Reports read back for summaries count as untrusted.
- **Email:**
  - Notice-only while the guard isn't acting; plain text with domains only.
  - Double opt-in, and POST-only confirm and unsubscribe (mail scanners follow links).
  - Caps per address, topic and day; a fixed From address; headers built safely.
- **Spending:** the allowance can't be passed (REC-113), on top of the per-task budget and search cap.
- **Privacy:** IDs and lengths only in logs; deletes remove everything, including the results of scheduled tasks.

## 7. Limits

- Topics are instance-wide (D2), so anyone with the access code sees every topic and its reports.
- On a laptop, schedules fire only while it's awake, and email links (`WEBAGENT_PUBLIC_URL`) work only on that machine.
- Change summaries are a model's judgement. The source diff is exact, and `always` exists.
- Summary emails wait for act mode (D4).
