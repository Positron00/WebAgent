# Low-level design: backend and scheduler

Paths are relative to `backend/`. Phase A is §1–§10; phase B is §11–§13.

## 1. Modules

| Module | New or changed | Purpose | Requirements |
|---|---|---|---|
| `app/services/topics.py` | new | Topic models, validation, the definition store (flock), effective state | REC-100 – REC-102, REC-114 |
| `app/services/topic_files.py` | new | Paths, atomic write, read helpers shared by the backend and the scheduler; the flock wrapper | REC-101 |
| `app/services/schedule_math.py` | new | Slot computation with `zoneinfo` | REC-104 |
| `app/scheduler.py` | new | The scheduler process: lock, tick, runs, reconciliation, caps, history, holds, usage, counters | REC-104 – REC-106, REC-111 – REC-113, REC-115 |
| `app/services/topic_runs.py` | new | Run records, URL normalisation, source diff, pruning | REC-111, REC-115 |
| `app/api/endpoints/topics.py` | new | Topic routes; the scheduler-only `summarize` and `notify` routes | REC-102, REC-103, REC-130, REC-131 |
| `app/api/endpoints/task.py` | changed | Scheduler fields, the fixed task ID, skipping triage collection | REC-107, REC-108 |
| `app/core/scheduler_auth.py` | new | `require_scheduler` dependency (constant-time compare) | REC-107 |
| `app/services/task_budget.py` | changed | Counter with input and output tokens, searches and credits; the search cap | REC-109 |
| `app/services/together_llm.py` | changed | Adds `prompt_tokens` and `completion_tokens` to the counter | REC-109 |
| `app/agents/web_research.py` | changed | `capped_search` asks `task_budget.charge_search(depth)` | REC-109 |
| `app/services/task_manager.py` | changed | Per-run budget, `origin` and `topic_id` in the result, `usage`, `standalone_query`, LangSmith metadata | REC-108 – REC-110 |
| `app/services/drift_embeddings.py` | changed | Skips `_observe` while `topic_run` is set | REC-108 |
| `app/security/records.py` | changed | `origin` and `topic_id` on collection rows | REC-108 |
| `app/core/metrics.py` | changed | `origin` label; the topic collector | REC-108, REC-117 |
| `app/core/config_def.py`, `config.py` | changed | `ScheduleConfig`, `NotificationsConfig`, `LLMConfig.price_*` | REC-118 |
| `app/core/startup_checks.py` | changed | Scheduler and email checks | REC-118, REC-136 |
| `main.py` | changed | Routers; daily topic and notify cleanup | REC-102, REC-115 |
| `app/services/notify_tokens.py` | new (B) | Signed confirm and unsubscribe tokens | REC-133, REC-134 |
| `app/services/notify_email.py` | new (B) | Messages, transports, caps, ledger | REC-132 – REC-138 |
| `app/api/endpoints/notify.py` | new (B) | Confirm and unsubscribe; one-click | REC-134 |

## 2. Files and locking (`topic_files.py`)

```python
TOPICS_DIR = Path(settings.DATA_DIR) / "topics"
NOTIFY_DIR = Path(settings.DATA_DIR) / "notify"

def definition_path(topic_id: str) -> Path      # topics/<id>.json; UUIDs only
def topic_dir(topic_id: str) -> Path            # topics/<id>/
def state_path(topic_id: str) -> Path           # topics/<id>/state.json
def run_path(topic_id: str, run_id: str) -> Path
def write_json_atomic(path: Path, data: dict) -> None   # tmp + os.replace, as result_store
def read_json(path: Path) -> Optional[dict]
@contextmanager
def locked(path: Path)                           # fcntl.flock on path.with_suffix(".lock"); drift_store._locked
```

- **Invalid IDs:** every path function rejects IDs that aren't UUIDs (as `result_store._result_path` does).
- **Backend edits:** `with locked(definition_path(id)): d = read; change; write_json_atomic(...)`.
- **Scheduler writes:** no lock; the scheduler is single and owns its files.
- **A writer test:** `tests/test_topic_files_ownership.py` greps the modules. The `topics.py` and `notify_*` modules may write only definitions and `NOTIFY_DIR`; `scheduler.py` and `topic_runs.py` may write only state, runs, usage, counters and the lock. REC-114's exception is listed explicitly.

## 3. Models (`topics.py`)

```python
class Schedule(BaseModel):
    every: Literal["day", "week"]
    weekday: Optional[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]] = None   # required if week
    time: constr(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    tz: str                                     # validated with zoneinfo.ZoneInfo

class Subscriber(BaseModel):                    # phase B
    email: str                                  # validated: one @, a domain with a dot, ≤ 254 chars
    confirmed_at: Optional[datetime] = None
    added_at: datetime

class Notify(BaseModel):
    when: Literal["changes", "always"] = "changes"
    subscribers: List[Subscriber] = Field(default_factory=list, max_length=5)

class TopicDefinition(BaseModel):
    topic_id: str
    name: constr(min_length=1, max_length=80)          # no control characters (validator)
    query: constr(min_length=1, max_length=2000)       # no control characters
    schedule: Schedule
    notify: Notify = Notify()
    status: Literal["active", "paused", "deleted"] = "active"
    created_at: datetime
    created_from_task: Optional[str] = None
    run_requests: int = 0
    resumes: int = 0
```

`effective_state(defn, state) -> Literal["active", "paused", "held_failures", "held_budget", "deleted"]`: `deleted` or `paused` from the definition first, then `hold` from the state.

## 4. Topic API (`api/endpoints/topics.py`)

| Route | Body and response | Notes |
|---|---|---|
| `GET /api/topics` | `{topics: [{topic_id, name, schedule, notify.when, subscriber_count, state, next_due, last_run, last_new_sources, skipped}]}` | Excludes deleted topics. The query text is included (instance-wide, D2) |
| `POST /api/topics` | `{name, query, schedule, notify?, created_from_task?}` → 201 the definition | Cap (409 `topic_limit`), intake guard (403 or 409 with the refusal body in act mode) |
| `GET /api/topics/{id}` | The definition (subscribers show addresses and confirmed or not) | 404 if missing or deleted |
| `PATCH /api/topics/{id}` | Any of `name`, `schedule`, `notify.when`, `status` (active or paused); phase B: `add_subscriber`, `remove_subscriber` | Re-activating checks the cap. A changed `schedule` makes the scheduler recompute `next_due`; it compares `schedule` with `state.schedule_seen` |
| `DELETE /api/topics/{id}` | 204 | Tombstone (§9) |
| `POST /api/topics/{id}/run` | 202 `{run_requests}` | Increments the counter |
| `POST /api/topics/{id}/resume` | 202 | `status: active`, `resumes += 1`, with the cap check |
| `GET /api/topics/{id}/runs` | `{runs: [{run_id, status, error_code, completed_at, change: {first, new_sources (count), changed, summary?}}]}` | Newest first, at most 20 |
| `GET /api/topics/{id}/runs/{run_id}` | The run record with `report`, `sources` and `change` | — |
| `POST /api/topics/{id}/summarize` | Scheduler only; `{run_id, previous_run_id}` → `{summary, changed, tokens: {input, output}}` | Phase B (§11) |
| `POST /api/topics/{id}/notify` | Scheduler only; `{run_id, kind: run|hold|deletion}` → `{sent, skipped, reason}` | Phase B (§12) |

The topic routes use the `require_api_key` dependency, as the other `/api` routers do. `summarize` and `notify` use `require_scheduler` instead, and are absent (404) when `WEBAGENT_SCHEDULER_TOKEN` is unset.

**Intake guard at creation (REC-103):** call `check_hop("topic_create", [ContextItem("user_turn", query)], state, chat_id=topic_id)`, as `endpoints/guard.py` does. A `GuardStop` in act mode becomes 403 or 409 with `refusal_body`. Shadow decisions are recorded with collection `origin: topic_create`.

## 5. Scheduler fields on `POST /api/task` (`task.py`, `scheduler_auth.py`)

```python
class TaskRequest(BaseModel):            # added fields
    origin: Optional[Literal["schedule"]] = None
    topic_id: Optional[str] = None
    task_id: Optional[str] = None          # UUID
    token_budget: Optional[int] = None

def require_scheduler(request: Request) -> None:
    token = settings.SCHEDULER_TOKEN
    sent = request.headers.get("X-WebAgent-Scheduler", "")
    if not token or not hmac.compare_digest(sent, token):
        raise HTTPException(404 if not token else 403)
```

- **Field rules:** if any of the four fields is set, then:
  - a missing or wrong header answers 422 `scheduler_fields_need_token`, and so does any field when the token is unset;
  - `origin` must be `schedule` and `mode` must be `always`;
  - `topic_id` and `task_id` must be UUIDs;
  - `token_budget` must be between 1 and `schedule.max_token_budget` (200,000).
- **Duplicate IDs:** `task_id` already in `result_store` answers 409 `task_exists`.
- **Triage:** with `origin` set, the endpoint skips `triage_collect.record(...)` (both the `shadow` and `override` calls) and doesn't classify.
- **Running the task:** `task_manager.run_workflow(..., origin="schedule", topic_id=..., token_budget=...)`, with `task_id` chosen by the caller.

## 6. Usage, budget and the search cap (`task_budget.py`, `together_llm.py`, `web_research.py`, `task_manager.py`)

```python
@dataclass
class _Counter:
    limit: int
    used: int = 0              # total tokens (existing)
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    credits: int = 0
    search_cap: int = 0        # 0 = no cap

def start(limit: int, search_cap: int = 0) -> Token
def add(tokens: int, input_tokens: int = 0, output_tokens: int = 0) -> None
def charge_search(depth: str) -> bool     # False when the cap is reached; else counts 1 or 2 credits
def usage() -> dict                        # {input_tokens, output_tokens, searches, tavily_credits}
```

- **Token counts:** `together_llm.py` calls `task_budget.add(total, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))`.
- **The search cap:** `web_research.capped_search` replaces its local `searches_made` with `if not task_budget.charge_search(kwargs.get("search_depth", "basic")): log; return []`.
  - The counter is the run's context-variable object. `asyncio.gather` copies the context, but the object is shared, so parallel agents and every pass count against one total.
  - The local per-pass counter is removed.
- **Starting a run:** `task_manager.run_workflow` calls `task_budget.start(token_budget or settings.TASK_TOKEN_BUDGET, search_cap=agents.web_research.max_searches_per_task)`.
- **The result** gets `usage: task_budget.usage()`, `origin`, `topic_id` and `standalone_query` (from `state.context["research_plan"]["standalone_query"]`, when there is one). The run passes LangSmith metadata `{origin, topic_id}` through the run config.
- **The direct path** (`run_direct`) also records `usage` (searches 0).

## 7. Drift and collection (`drift_embeddings.py`, `records.py`)

```python
topic_run: ContextVar[Optional[str]] = ContextVar("topic_run", default=None)   # in task_manager; set to topic_id
```

- **Drift:** `DriftRecordingEmbeddings._observe` returns at once when `topic_run.get()` is set. `embed_documents` during ingestion is unaffected, because ingestion never runs under a topic.
- **Collection:** `records.record_collection` adds `"origin": "schedule", "topic_id": ...` to the row when `topic_run.get()` is set.

## 8. Schedule maths (`schedule_math.py`)

```python
def next_slot(schedule: Schedule, after: datetime) -> datetime:
    """First slot strictly after `after` (aware UTC), returned in UTC."""
```

1. Convert `after` to `ZoneInfo(schedule.tz)` and take its local date `d`.
2. For candidate local dates `d, d+1, …`: skip a date that's the wrong weekday when `every == "week"`.
3. Build the local datetime at `time`:
   - **Gap:** if it doesn't exist (`fold` round-trip changes the wall time), use the first valid minute after it, found by adding minutes until the round trip holds;
   - **Repeat:** if it's ambiguous, use `fold=0` (the first occurrence).
4. Return the first candidate whose UTC time is after `after`.

Tests cover Europe/Paris and America/New_York across both DST changes, a weekly schedule across a year end, and 23:59.

## 9. The scheduler (`app/scheduler.py`)

```python
def main() -> None:                 # python -m app.scheduler
    checks()                        # enabled → token set, backend /health reachable (3 tries)
    with SchedulerLock(TOPICS_DIR / ".scheduler.lock") as lock:   # exits if a fresh heartbeat exists
        reconcile()                  # REC-106
        while True:
            lock.beat(enabled=cfg.enabled)
            if cfg.enabled:
                tick(now_utc())
            sleep_until_next_minute()
```

- **`tick(now)`**, step by step:
  1. **Load:** every definition and its state. A topic without state gets `next_due = next_slot(schedule, now)`.
  2. **Tombstones:** for `deleted` topics, `remove_topic_files(id)` (§10), then continue.
  3. **Paused topics:** record `paused_since` the first time `paused` is seen, and clear it on `active`.
  4. **Resume:** when `resumes > resumes_seen`, clear a `failures` hold, reset `failures_in_a_row` and set `resumes_seen`.
  5. **Schedule changes:** when `schedule != state.schedule_seen`, recompute `next_due` and store `schedule_seen`.
  6. **Budget hold:** if `hold == "budget"` and (the month changed or the allowance was raised enough for one worst case), clear it.
  7. **Candidates:** topics `active`, with no hold, and (`next_due <= now` or `run_requests > run_requests_seen`), oldest `next_due` first.
  8. **Starting each candidate,** while in-flight runs are under `max_concurrent_runs`:
     - **Daily cap:** if today's started runs ≥ `max_runs_per_day`, set `state.skipped = {"reason": "daily_cap", "at": now}`, advance `next_due` if it was due, mark the request handled, and continue.
     - **Allowance:** if `remaining() < worst_case() × (in_flight + 1)`, set `hold: budget` on every topic and stop the tick.
     - **Start:** `start_run(topic)` in a thread: run ID `uuid4`, task ID `uuid5(TOPIC_NAMESPACE, run_id)`, record `starting`, POST, record `running`.
     - **Advance:** move `next_due` to `next_slot(schedule, now)` if it was due, and mark the request handled. If `now - next_due > period`, count `missed_slots += floor((now - next_due) / period)`.
  9. **Finished runs** (threads report back): `finish_run(record, result)`.
     - `report`, `sources` and `usage` come from the result.
     - `change` comes from `topic_runs.diff(...)`.
     - Then update usage (`credits += usage.tavily_credits`, `dollars += in × p_in + out × p_out`), counters and state (failures or reset, a hold at 3).
     - Prune to 20 runs, removing the oldest records and their tasks' results.
     - Phase B: summarise, then notify (§11, §12).
  10. **Paused for `paused_topic_days`** (`paused_since` older than 90 days): call the backend's `DELETE /api/topics/{id}`, with `X-API-Key` (when set) and the scheduler token. From phase B, a deletion notice goes out 7 days before.
- **`worst_case()`:** `credits = max_follow_up_questions × 3`, and `dollars = (token_budget + 8000) × price_output / 1e6`.
- **`remaining()`:** the allowance minus `usage-<month>.json`.
- **`reconcile()`:** for each run in `starting` or `running`, `GET /api/task/{task_id}`. On 200 it's polled in a thread; on 404 it's marked `failed`, `not_started`.
- **`start_run`:**
  - The POST has a 30 s timeout; polling is every 15 s for 10 min, after which the run is marked `failed`, `poll_timeout`, and keeps being polled once more after the next tick.
  - **Errors:** 403 and 409 are `failed` (`intake_refused`), 429 is `failed` (`rate_limited`), 5xx or a connection error is `failed` (`backend_unreachable`).
- **Headers:** `X-API-Key` (when set), `X-WebAgent-Scheduler`, and `X-WebAgent-Client: schedule:<topic_id>`.
- **Imports:** `app.core.config`, `app.services.topic_files`, `topics` (models only), `schedule_math`, `topic_runs` and `httpx`. No agents, security or ML (RNFR-02).

## 10. Deletion (`topics.py`, `scheduler.py`)

- **The tombstone:** `DELETE` sets `status: deleted` under flock.
- **The scheduler:** `remove_topic_files(id)` deletes each `data/results/<task_id>.json` named in the topic's run records, then the topic folder.
- **The backend's daily cleanup,** and `GET` on that topic:
  - if the folder is gone, it deletes the definition and the topic's ledger entries;
  - if there's no fresh heartbeat (180 s), it takes `.scheduler.lock` for the duration, writing a heartbeat as `owner: backend` so a starting scheduler waits, and does the scheduler's part itself.
- **Daily cleanup** (`main._startup`, as `result_store.cleanup_daily`; both workers run it, so it runs under flock): expired unconfirmed subscribers (phase B) and tombstones.

## 11. Change summary (phase B; `topics.py` summarize route)

- **Steps:**
  1. Load both run records, read-only.
  2. Build the prompt. The context items are `ContextItem("retrieved_doc", previous_report[:6000])`, `ContextItem("retrieved_doc", new_report[:6000])` and the new-source list. The instruction asks for at most 5 bullets citing `[n]`, or exactly "No material change".
  3. Run `guarded_llm` with `agents.topic_summary` (`max_tokens` 600) under `task_budget.start(8000)`.
- **Response:** `{summary, changed: summary.strip() != "No material change", tokens}`.
- **Failure:** a model or guard failure answers 200 with `{summary: null, error}`, and the scheduler records `change.summary_error`.

## 12. Email (phase B; `notify_email.py`, `notify_tokens.py`, `notify.py`)

```python
def sign(purpose: Literal["confirm", "unsubscribe"], topic_id: str, address: str, expires: Optional[int]) -> str
def verify(token: str, purpose: str) -> Optional[Tuple[str, str]]   # (topic_id, address), or None
```

- **Tokens:** `base64url(json{p, t, a, e})` + `.` + `base64url(hmac_sha256(SECRET, payload))`. Confirm tokens expire in 48 h; unsubscribe tokens don't. Used confirm tokens are recorded in the ledger by SHA-256.
- **Building a message:** `EmailMessage()` with `Subject`, `From` (`WEBAGENT_SMTP_FROM`), `To`, `List-Unsubscribe`, `List-Unsubscribe-Post` and `Auto-Submitted: auto-generated`. The body is plain text; the templates are in §6.6 of the plan and REC-135.
- **Transports:** `LogTransport.send(msg)` logs the subject length and kind. `SmtpTransport.send(msg)` uses `smtplib.SMTP` with `starttls()` or `SMTP_SSL`, `login`, `send_message`, a 20 s timeout and 2 retries (2 s, 8 s).
- **Caps:** checked under flock in `ledger-<today>.json`: per run (1), per instance per day (50), and confirmations per hashed address per day (3).
- **Notice domains:** they come from the run's sources whose URL isn't in the result's `security.dropped` refs (act mode) or `security.shadow_flagged` refs (shadow mode).
  - `security.shadow_flagged` is new (task B0). Today `hop_guard.check_hop` records nothing per item in shadow mode, because the effective outcome is `allow`.
  - B0 adds `{source, ref, policy_id}` entries for items whose would-be outcome was `decline` or `escalate`, to the ledger and the security block. `ledger.merge` must carry the list across the parallel agents.
- **Routes:** `GET /api/notify/lookup?token=…` verifies a confirm or unsubscribe token and returns only `{topic_name, purpose}`. It is read-only, and the web app's notify pages use it to show the name. `POST /api/notify/confirm` and `POST /api/notify/unsubscribe` take `{token}`. The one-click route also takes a form-encoded `List-Unsubscribe=One-Click` body with the token in the query. All are public: no API key, rate-limited per caller.

## 13. Configuration (`config_def.py`)

```python
class ScheduleConfig(BaseModel):
    enabled: bool = False
    max_active_topics: int = 10
    max_runs_per_day: int = 20
    max_concurrent_runs: int = 2
    history_runs_kept: int = 20
    pause_after_failures: int = 3
    paused_topic_days: int = 90
    monthly_tavily_credits: int = 400
    monthly_together_usd: float = 5.0
    token_budget_default: int = 60000
    max_token_budget: int = 200000

class EmailConfig(BaseModel):
    enabled: bool = False
    transport: Literal["log", "smtp"] = "log"
    mode: Literal["notice", "summary"] = "notice"
    max_subscribers_per_topic: int = 5
    max_emails_per_day: int = 50

class LLMConfig(BaseModel):          # added
    price_input_per_million: float = 0.35
    price_output_per_million: float = 1.50
```

- **Environment:** `WEBAGENT_SCHEDULER_ENABLED`, `WEBAGENT_SCHEDULER_TOKEN`, `WEBAGENT_BACKEND_URL`, `WEBAGENT_PUBLIC_URL`, and `WEBAGENT_SMTP_{HOST,PORT,USERNAME,PASSWORD,FROM,TLS}`. They follow the existing precedence (`.env.local` and environment over YAML).
- **Startup checks:**
  - The scheduler, when enabled, needs the token.
  - The backend logs "scheduling off: no WEBAGENT_SCHEDULER_TOKEN" once.
  - With email enabled and the SMTP transport, the backend needs the SMTP host, From and public URL.
  - With email `mode: summary`, the backend needs `security_guard.input_mode: act` (REC-136).

## 14. Metrics (`metrics.py`)

- **`origin` label:** added to `webagent_tasks_total` and `webagent_task_duration_seconds`, with values `interactive` and `schedule`. Existing dashboard queries sum over labels and keep working; the dashboard JSON is checked in task A10.
- **`TopicCollector`:** registered on `WEBAGENT_REGISTRY`. On each scrape it reads `counters.json` (runs by outcome, changes by kind, holds by reason, missed slots), every `state.json` (for `webagent_topics{state}`), `usage-<month>.json` with the config limits, and `.scheduler.lock`. Reads are cached for 15 s, so a burst of scrapes costs one read of each file.
- **Email:** `webagent_email_sent_total{kind, outcome}` is an ordinary counter, incremented by the backend (phase B).

## 15. Tests

| File | Covers |
|---|---|
| `tests/test_topics_api.py` | REC-100, REC-102, REC-103, REC-114 (API side); 404s; caps; intake guard in act and shadow |
| `tests/test_topic_files.py` | Atomic writes, UUID paths, flock under two processes (multiprocessing), the ownership grep |
| `tests/test_schedule_math.py` | REC-104 slot maths with DST and weekly |
| `tests/test_scheduler.py` | REC-104 – REC-106, REC-112, REC-113, REC-115, RNFR-01: a fake clock, a fake backend (httpx `MockTransport`), crash and reconcile, caps, allowance, holds, resume, deletion, pruning, the 90-day rule, missed slots |
| `tests/test_scheduler_imports.py` | RNFR-02: `import app.scheduler` loads no ML library |
| `tests/test_task_scheduler_fields.py` | REC-107, REC-108 (task side): the header matrix, 422 and 409, triage skipped, `origin` on result, rows and metrics |
| `tests/test_task_usage.py` | REC-109, REC-110: input and output counts, the search cap across passes and parallel agents, credits by depth, `standalone_query` in the response |
| `tests/test_topic_runs.py` | REC-111: URL normalisation, the diff, the first run, pruning |
| `tests/test_topic_metrics.py` | REC-117: the collector with fixture files; counters never decrease |
| `tests/test_topic_summary.py` | REC-130: a fake model, `retrieved_doc` items, no change, failure |
| `tests/test_notify_email.py` | REC-131 – REC-138: a patched `smtplib`, tokens, POST-only, one-click, caps, content (header injection, no model text in notice mode, only unflagged domains), startup checks |
| `tests/redteam/test_topic_injection.py` | RNFR-06: a report with an injected instruction and `<script>`, in notice mode and with the summary scored |
| `tests/test_privacy_logs.py` (extended) | RNFR-03 on every new path |
