# WebAgent API Reference

This document provides a comprehensive reference for the WebAgent Backend API (v3.12.1).

## Base URL

All API paths are relative to the base URL:

```
https://api.webagent.example.com/api/v1
```

## Authentication

Send the API key in the `X-API-Key` header:

```
X-API-Key: <WEBAGENT_API_KEY>
```

- Keys are required in uat and prod (`api_key_enabled: true` in their YAML config); the server refuses to start there without `WEBAGENT_API_KEY`.
- In dev, keys are off by default (`WEBAGENT_SECURITY_API_KEY_ENABLED=true` turns them on); when on, `WEBAGENT_API_KEY` is the only valid key.
- The web app never sends the key from the browser: its `/api/task` route adds it on the server.
- `GET /health`, `GET /api/v1/health/` and `GET /metrics` never need a key; `GET /api/v1/health/detailed` does.
- In uat and prod the OpenAPI pages (`/api/v1/docs`, `/api/v1/redoc`, `/api/v1/openapi.json`) are off.

A missing or wrong key returns `401` with `{"error": {"code": "unauthorized", "message": "API key required"}, "detail": "API key required"}` (or "Invalid API key").

## Common Response Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 201  | Created (resource created successfully) |
| 400  | Bad Request (invalid input) |
| 401  | Unauthorized (missing or invalid API key) |
| 403  | Forbidden (insufficient permissions) |
| 404  | Not Found (resource not found) |
| 409  | Conflict (for example, cancelling a task that already finished; or the intake guard needs consent) |
| 413  | Request too large (`WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT`, 10 MB) |
| 415  | Unsupported file type on document upload |
| 422  | Validation error (for example, a question over 20,000 characters) |
| 429  | Too Many Requests (rate limit exceeded) |
| 500  | Internal Server Error |

## Rate Limiting

Each caller may make 60 requests per minute (`WEBAGENT_SECURITY_RATE_LIMIT_MAX_REQUESTS` per `WEBAGENT_SECURITY_RATE_LIMIT_WINDOW_SIZE` seconds; uat/prod YAML can change them).

- The caller is the client's IP address. A request from a trusted proxy (`WEBAGENT_TRUSTED_PROXIES`, default localhost) or one carrying a valid `X-API-Key` (the web app, also on Vercel) may name its user in `X-WebAgent-Client` (`s:<session id>`), so each signed-in user has their own budget. From a trusted proxy without that header, the right-most `X-Forwarded-For` address that isn't a trusted proxy is used.
- Health checks and `/metrics` are never limited.
- Limits are counted per server worker, so with N workers a caller can make up to N × the limit.

When rate limited, you receive `429 Too Many Requests` with `{"error": {"code": "rate_limited", "message": "Too many requests"}}` and a `Retry-After` header giving the window length in seconds.

## Chat API

### Start a Chat Workflow

Initiates a new multi-agent chat workflow.

**Endpoint:** `POST /chat`

**Request Body:**

```json
{
  "message": "What are the benefits of LLM observability?",
  "context": {
    "additional_information": "Focus on LangSmith specifically",
    "previous_messages": [
      {
        "role": "user",
        "content": "Tell me about LLMs"
      },
      {
        "role": "assistant",
        "content": "LLMs are large language models..."
      }
    ]
  },
  "session_id": "optional-session-id"
}
```

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Your request is being processed. Check the status using the task ID."
}
```

### Get Chat Status

Retrieves the status and results of a chat task.

**Endpoint:** `GET /chat/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Task complete",
  "result": {
    "title": "Benefits of LLM Observability with LangSmith",
    "content": "LangSmith provides several benefits for LLM observability...",
    "visualizations": [
      {
        "type": "chart",
        "description": "Performance comparison",
        "image_data": "base64_encoded_image_data",
        "code": "python code used to generate the visualization"
      }
    ],
    "sources": [
      {
        "title": "LangSmith Documentation",
        "content": "Excerpt from LangSmith docs...",
        "url": "https://docs.langchain.com/langsmith/",
        "relevance": 0.95,
        "source_type": "web"
      }
    ]
  }
}
```

## Frontend Compatibility API

These endpoints provide compatibility with the frontend by mimicking other LLM service APIs.

### Chat Completions

Handles chat completions in a format compatible with the frontend.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**

```json
{
  "model": "meta-models/Muse-Glimmer-30B",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "max_tokens": 4096,
  "temperature": 0.7,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "mode": "auto"
}
```

`mode` (`off`, `auto` or `always`) works as on `POST /api/task`; the older `agentic_toggle` flag still works, and with neither the request runs as `off`. Earlier messages in `messages` are passed to the model as history.

**Response:**

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Your request is being processed (Task ID: 550e8400-e29b-41d4-a716-446655440000). Please check back in a moment."
      },
      "finish_reason": "processing"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 0,
    "total_tokens": 10
  }
}
```

### Get Chat Completion Status

Gets the status of a chat completion task.

**Endpoint:** `GET /v1/chat/status/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "status": "completed",
  "response": {
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "The capital of France is Paris. It is known as the 'City of Light' and is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 37,
      "total_tokens": 47
    }
  }
}
```

### Backend Status

Gets the status of the backend services.

**Endpoint:** `GET /v1/status`

**Response:**

```json
{
  "status": "ok",
  "version": "3.12.1",
  "llm": {
    "status": "ok",
    "provider": "together",
    "default_model": "meta-models/Muse-Glimmer-30B",
    "available_models": [
      "meta-models/Muse-Glimmer-30B",
      "meta-llama/Llama-3.3-8B-Instruct"
    ]
  },
  "services": {
    "workflow": "ok",
    "database": "ok"
  }
}
```

## Task and Triage API

These are the endpoints the web app uses. Both need `X-API-Key` when API-key auth is on.

### Start a Task

**Endpoint:** `POST /api/task`

```json
{
  "query": "What changed in EU AI rules this year?",
  "mode": "auto",
  "previous_messages": [
    {"role": "user", "content": "Tell me about AI regulation"},
    {"role": "assistant", "content": "AI regulation covers..."}
  ]
}
```

| Field | Meaning |
|---|---|
| `query` | The question, at most 20,000 characters |
| `mode` | `off`: one direct model call. `always`: the research workflow. `auto`: the triage classifier picks one |
| `agentic` | Older flag: `true` means `always`, `false` means `off`. `mode` wins if both are sent. **With neither, the task runs as `off`** |
| `previous_messages` | Up to 8 earlier turns, oldest first, roles `user` or `assistant`. More than 8 is rejected with 422 |
| `triage_override` | `true` runs the workflow and counts it as a "Research this in depth" click |
| `image` | Accepted but not sent to the model on either path; the result then has `image_ignored: true` |
| `document_id` | Document to extract from, if any |

**Response:** `{"task_id": "...", "status": "processing"}`. Poll `GET /api/task/{task_id}`. The finished `result` includes:

- `route` (`direct` or `agentic`) and `decided_by` (`mode`, `rule`, `model`, `fallback`, `config`, `override`, or `schedule` for scheduled runs);
- `p_agentic`, when the classifier ran;
- `tokens_used`;
- since 3.11.0, `usage` (`input_tokens`, `output_tokens`, `searches`, `tavily_credits`) and, for research runs, the supervisor's `standalone_query`.

**Scheduler-only fields (3.11.0):** `origin` (`schedule`), `topic_id`, `task_id` and `token_budget` are accepted only with an `X-WebAgent-Scheduler` header matching the backend's `WEBAGENT_SCHEDULER_TOKEN`; otherwise they answer 422 `scheduler_fields_need_token`. The API key isn't enough. A chosen `task_id` that already exists answers 409 `task_exists`. The web app's proxy never forwards these fields.

The intake guard checks the question first. A refusal answers 403 (declined) or 409 (consent needed, with a `consent_token` to send back) in its own flat shape: `{"code": "declined" | "consent_required", "outcome", "reason", "policy_id", "policy_ids"}`, plus `consent_token` on a 409.

### Task Status

**Endpoint:** `GET /api/task/{task_id}`

```json
{
  "status": "processing",
  "result": null,
  "error": null,
  "progress": {"step": "Web and internal research", "logs": ["..."], "updated_at": "2026-10-14T10:31:05"}
}
```

- `status` is `processing`, `completed` or `failed`.
- `progress` appears only while a research run is working: the last finished step in plain words and up to 5 recent log lines. Direct answers have none.
- A failed task has `error` and `error_code`: `timeout` (the run passed `WEBAGENT_TASK_MAX_SECONDS`, 420 s), `budget` (it used `WEBAGENT_TASK_TOKEN_BUDGET`), `cancelled`, or `escalated` (the security guard stopped it).

### Cancel a Task

**Endpoint:** `POST /api/task/{task_id}/cancel`

**Response:** `202 {"status": "cancelling"}`. The worker running the task stops it within a few seconds, and the task then reads `failed` with `error_code: cancelled`. A task that already finished answers 409; an unknown task answers 404.

### Triage a Question

**Endpoint:** `POST /api/triage`

```json
{"query": "explain list comprehensions", "previous_user_question": null, "has_image": false}
```

**Response:**

```json
{"route": "direct", "p_agentic": 0.104, "decided_by": "model", "reason": "Looks like: coding help", "classifier_version": "triage-2026-10-a"}
```

`query` may be empty and is limited to 8,000 characters. The endpoint answers 200 even when the classifier can't load (`route: agentic`, `decided_by: fallback`). It counts toward the rate limit.

## Task Management API

### List Tasks

Lists saved tasks from the results folder, newest first, so every worker gives the same answer. Each task shows its question's length, not its text.

**Endpoint:** `GET /tasks?limit=20&offset=0` (`limit` 1–100, default 20)

**Response:**

```json
{
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2026-10-14T10:30:00",
      "completed_at": "2026-10-14T10:31:05",
      "query": "30 chars"
    }
  ],
  "limit": 20,
  "offset": 0
}
```

Saved tasks are deleted after `WEBAGENT_RESULTS_RETENTION_DAYS` (30 days).

### Get Task Details

Gets details about a specific task.

**Endpoint:** `GET /tasks/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2025-03-09T10:30:00Z",
  "completed_at": "2025-03-09T10:31:05Z",
  "message": "Task complete",
  "query": "What is the capital of France?",
  "current_step": null
}
```

`status` is `processing`, `completed` or `error` here (`/api/task` calls the last one `failed`). Any worker can answer: a task another worker ran, or one from before a restart, is read from its saved record.

### Get a Task's Result or Error

**Endpoints:** `GET /tasks/{task_id}/result` and `GET /tasks/{task_id}/error`

- `/result` returns the finished report. It answers 400 while the task is still running, or if it failed.
- `/error` returns `{"message": "...", "error_code": "timeout" | "budget" | "cancelled" | "escalated"}` for a failed task. It answers 400 if the task hasn't failed.
- Both answer 404 for an unknown task.

### Create a Task (deprecated)

**Endpoint:** `POST /tasks/create` with `{"query": "...", "context": {"document_id": "..."}}`. Use `POST /api/task` instead.

It starts a research run in Always mode through the same intake guard and limits. The response is `{"task_id", "status": "processing", "message"}`, or an intake-guard refusal (403 or 409). `POST /chat` is the same.

### Delete Task

Deletes a task and its results.

**Endpoint:** `DELETE /tasks/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "status": "success",
  "message": "Task 550e8400-e29b-41d4-a716-446655440000 deleted"
}
```

## Topics API (3.11.0)

Topics are questions the scheduler researches again daily or weekly. Every route needs `X-API-Key` when keys are on. Unknown or malformed IDs, and deleted topics, answer 404.

| Route | Does |
|---|---|
| `GET /api/topics` | Lists topics: `topic_id`, `name`, `query`, `schedule`, `notify.when`, `subscriber_count`, `state` (`active`, `paused`, `held_failures`, `held_budget`), `next_due`, `last_run`, `last_new_sources`, `skipped` |
| `POST /api/topics` | Creates one; answers 201 with the definition |
| `GET /api/topics/{id}` | The definition, with the same scheduler fields as a list row |
| `PATCH /api/topics/{id}` | Changes `name`, `schedule`, `notify.when` or `status` (`active` or `paused`) |
| `DELETE /api/topics/{id}` | Deletes it, with its runs and their task results; 204 |
| `POST /api/topics/{id}/run` | Run now: queues one extra run; 202 |
| `POST /api/topics/{id}/resume` | Sets it active and clears a failures hold; 202 |
| `GET /api/topics/{id}/runs` | Runs, newest first (at most 20): `run_id`, `status`, `error_code`, times, `change` (`first`, `new_sources` count, `changed`) |
| `GET /api/topics/{id}/runs/{run_id}` | One run with its stored `report`, `sources`, `usage` and `change` (`new_sources` and `dropped_sources` as URL lists) |

**Create:**

```json
{"name": "EU AI rules", "query": "What changed in EU AI rules this week?",
 "schedule": {"every": "week", "weekday": "mon", "time": "07:30", "tz": "Europe/Paris"},
 "notify": {"when": "changes"}}
```

- **Fields:**
  - `name` is 1–80 characters and `query` 1–2,000, with no control characters.
  - `every` is `day` or `week`; `week` needs `weekday`.
  - `time` is `HH:MM` in the IANA time zone `tz`.
- **Errors:**
  - Invalid fields answer 422 with the field named.
  - More than 10 active topics answers 409 `topic_limit`, as does re-activating one.
  - In act mode, a question the security guard refuses answers 403 with the guard's refusal body. A topic can't be created when the question would need consent, because a scheduled run can't ask for it.

### Subscribers and email (3.12.0)

- **`PATCH /api/topics/{id}`** also takes `add_subscriber: "<email>"` and `remove_subscriber: "<email>"`.
  - Adding sends one confirmation email; the response's `confirmation` is `sent`, `failed`, `capped` or `not_needed`. No other email goes to an address until it's confirmed.
  - Errors: 409 `subscriber_limit` (5 per topic), 409 `email_disabled` (email is off), 422 for an invalid address.
- **Scheduler-only routes**, which need `X-WebAgent-Scheduler` as well as the API key:
  - `POST /api/topics/{id}/summarize` with `{run_id, previous_run_id}` → `{summary, changed, tokens, error?}`;
  - `POST /api/topics/{id}/notify` with `{kind: run|hold|deletion, run_id?}` (`run_id` required for `run`) → `{sent, skipped, reason?}`. Email problems answer 200 with a `reason`: `email_disabled`, `no_subscribers`, `already_sent`, `run_not_found`, `run_not_completed`, `daily_cap`, `send_failed`, `topic_not_found` or `error`.

### Notify routes (3.12.0)

These need the API key. Only the web app's `/notify/*` pages and `/api/notify/confirm` and `/api/notify/unsubscribe` proxies are reachable without signing in, and those add the key. Tokens come from the emails.

| Route | Does |
|---|---|
| `GET /api/notify/lookup?token=…` | `{topic_name, purpose}` for a valid token; 404 otherwise. Read-only |
| `POST /api/notify/confirm` | `{token}` → `{status: "confirmed"}`; 404 for an invalid, used or expired token (confirm tokens last 48 h) |
| `POST /api/notify/unsubscribe` | `{token}` (JSON) or the RFC 8058 one-click form (`List-Unsubscribe=One-Click`, token in the query) → `{status: "unsubscribed"}`; safe to repeat |

## Documents API

Documents loaded here are searched by the internal research agent. They are shared by every task on the server. IDs are UUIDs.

### Upload a Document

**Endpoint:** `POST /documents/upload` (multipart form)

| Field | Meaning |
|---|---|
| `file` | The document: `.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.csv` or `.json`. Other types answer 415 |
| `metadata_json` | Optional JSON object, for example `{"category": "reports", "tags": ["financial"]}`. Invalid JSON answers 400 |

**Response:** `{"id", "filename", "metadata", "upload_time", "content_preview"}`.

The document is parsed, summarised, chunked into the vector store, linked in the graph store, and copied to `WEBAGENT_ARTIFACTS_PATH`. The request size limit applies.

### Search Documents

**Endpoint:** `POST /documents/search`

```json
{"query": "quarterly revenue", "limit": 5, "include_content": true}
```

`limit` is 1–50 (default 10). **Response:** a list of `{"id", "score", "metadata", "content"}`, where `content` is present only with `include_content`.

### Get a Document

**Endpoint:** `GET /documents/document/{document_id}?include_content=false`

**Response:** as for upload. Unknown documents answer 404.

### Related Documents

**Endpoint:** `GET /documents/related/{document_id}?relationship_type=&depth=1&limit=10`

`depth` is 1–3 and `limit` is 1–50. **Response:** a list of documents as for upload, from the graph store.

## Direct Chat Guard

**Endpoint:** `POST /api/guard/input`

This is checkpoint 1 of the security guard, for chats that don't run a task. The web app's direct answers call it before calling the model.

```json
{"messages": [{"role": "user", "content": "..."}], "has_image": false, "consent_token": null}
```

- **Fields:** `messages` holds 1–60 messages of at most 100,000 characters each, and the last `user` message is the one checked. Send `consent_token` back after the user confirmed a `consent` outcome.
- **200:** `{"outcome": "allow" | "verify", "reason", "policy_ids", "decision_id"}`. Go ahead; `verify` means a borderline case was noted.
- **403** (declined or escalated) and **409** (consent needed): the flat refusal body, `{"code", "outcome", "reason", "policy_id", "policy_ids", "decision_id"}`. A 409 also carries `consent_token`; ask the user, then send the same messages again with that token.
- **Shadow mode** (the current setting): the would-be outcome is recorded in metrics, logs and collection, and the response is `allow`.

## Health API

### Liveness Probe

Used by the Docker health check and Kubernetes probes. Served at the server root, not under `/api/v1`. No API key, not rate-limited, not logged.

**Endpoint:** `GET /health` (full path, e.g. `http://localhost:8000/health`)

**Response:**

```json
{
  "status": "ok",
  "version": "3.12.1"
}
```

### Basic Health Check

**Endpoint:** `GET /health/` (under the base URL, i.e. `/api/v1/health/`). No API key needed.

**Response:**

```json
{
  "status": "ok",
  "version": "3.12.1",
  "environment": "prod"
}
```

### Detailed Health Check

Performs a detailed health check with information about connected services.

**Endpoint:** `GET /health/detailed`

**Response:**

```json
{
  "api": {
    "status": "ok",
    "version": "3.12.1"
  },
  "llm_service": {
    "status": "ok",
    "provider": "together",
    "latency_ms": 245
  },
  "vector_db": {
    "status": "ok",
    "count": 1250,
    "size_mb": 156
  }
}
```

### Metrics

Prometheus metrics endpoint. No API key needed, not rate-limited.

**Endpoint:** `GET /metrics` at the server root (what Prometheus and the Kubernetes annotations scrape); also available as `/api/v1/metrics`.

**Response:**
Plain text Prometheus metrics format

## Observability API

Needs `X-API-Key`. Prometheus `/metrics` and LangSmith tracing are separate and unchanged; [docs/RUNBOOK.md](../../docs/RUNBOOK.md) explains how they relate.

| Endpoint | What it returns |
|---|---|
| `GET /observability/metrics` | `model_metrics` (`provider`: the configured default provider, since samples don't record one; requests, errors and rate limits in the last hour; tokens in the last 24 hours; `by_label`: per call label, successful replies `requests_24h` and `tokens_24h`), `document_metrics` (loads and errors in the last 24 hours), `system_metrics`, `timestamp`; `pipeline_metrics` and `data_quality_metrics` are empty |
| `GET /observability/drift` | `computed_at`; `detectors` (see below); `global_drift_stats` (same content, the 3.5 key); `llm_service_drift` and `doc_service_drift` (baseline status and the highest-scoring detector of each kind); `retrain_recommendations` (always `[]`) |
| `GET /observability/alerts` | Active alerts: `name`, `severity` (`warning` or `critical`), `service` (`llm` or `document_processing`), `message`, `since`, `timestamp` (equals `since`) |
| `GET /observability/health` | `overall_status` (`unhealthy` with a critical alert, `degraded` with a warning, else `healthy`), `alert_counts` (`critical`, `warning`, `total`; also `high`/`medium`/`low` as in 3.5), `services.llm` and `services.document_processing` (status and figures), `components` (the 3.5 shape) |
| `POST /observability/llm/reset_baseline` | Start new references for the model detectors from the next samples |
| `POST /observability/doc/reset_baseline` | Start new references for the document detectors from the next samples |

All answers come from the shared drift store (`/app/data/drift/`), computed once a minute, so
every worker gives the same answer. With `WEBAGENT_DRIFT_ENABLED=false` they answer as if nothing
had been recorded.

A detector in `detectors` (names such as `llm_speed:supervisor`, `llm_tokens:judge`,
`embeddings:openai`, `doc_processing_time`, `doc_size`, `doc_type`):

```json
{
  "group": "llm",
  "kind": "ks",
  "enough_data": true,
  "score": 0.97,
  "drift": true,
  "reference_count": 50,
  "current_count": 42,
  "reference_start": "2026-10-16T09:12:03.120",
  "last_drift_at": "2026-10-17T14:02:00.000"
}
```

`score` is `1 − p` for the KS and chi-square detectors (drift above 0.95) and `1 − cosine
similarity` for embeddings (drift above 0.7); it is `null` without enough data (a frozen
reference and at least 30 current samples). The stored reference values are not returned.

## Errors

Error responses carry an `error` object:

```json
{"error": {"code": "not_found", "message": "Task not found"}, "detail": "Task not found"}
```

- `code` follows the status: `bad_request`, `unauthorized`, `forbidden`, `not_found`, `conflict`, `too_large`, `unsupported_media_type`, `validation_error`, `rate_limited`, `internal_error`.
- Errors raised as `HTTPException`, and validation errors (422), also keep FastAPI's `detail` field, as before 3.4.0.
- A 500 adds `error_id` and `timestamp`; the message is generic outside debug mode.
- Intake-guard refusals (403, 409 from `/api/task` and `/api/v1/chat/completions`) keep their own flat shape, described under [Start a Task](#start-a-task).

## Support

For API support, please contact:
- Email: api-support@webagent.example.com
- API Status Page: https://status.webagent.example.com 