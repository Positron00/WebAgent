# Requirements: platform improvements

Each requirement has an ID, the plan item it comes from, a statement, acceptance
criteria, and how it is checked. "Must" is required for its release; "should" can slip
to the next one. The release tag (**3.2.7**, **3.3.0**, **3.4.0**, **3.5.0**) matches the
phases in [`tasks.md`](tasks.md).

Terms used throughout:

- **Session**: one sign-in to the web app. It has a random ID carried in the signed
  `wa_session` cookie.
- **Client ID**: the value the backend rate-limits by. `s:<session id>` when the request
  came through the web app; otherwise the caller's address.
- **Run**: one execution of the agentic workflow (or the direct path) for one task.
- **Result file**: `data/results/<task id>.json` on the backend's data volume.
- **Marker file**: `data/results/<task id>.cancel`, written to ask for a cancel.

## 1. Safety net

**IMP-001 CI (must, 3.2.7; plan T1).** Every push and pull request to `main` and
`release/*` runs: `npm ci`, `npm run lint`, `npm run check-types`, `npm run
test:coverage`, a clean Python 3.12 install of `requirements.txt` (with
`backend/constraints.txt`), and `pytest backend/tests`. Git LFS files are not
downloaded. The run fails if any step fails.
- Accept: a pull request that breaks a Jest test, a pytest test, the type check or lint
  shows a failed check.
- Accept: the suite passes on a checkout with LFS pointer files instead of weights;
  tests that need real weights skip with a stated reason.
- Check: open one deliberately failing PR and one passing PR.

**IMP-002 Pre-commit hook (must, 3.2.7; plan T2).** `.husky/pre-commit` runs
`npx lint-staged`, which runs ESLint and Prettier on staged web app files. The existing
LFS hooks are unchanged.
- Check: commit a badly formatted `.ts` file; it is reformatted or the commit stops.

**IMP-003 Python lint (must, 3.3.0; plan T3).** `ruff check backend` runs in CI and as
`npm run lint:backend`, with a config in `backend/ruff.toml`. The first version enables
only rules that pass today or are fixed in the same change (at least `E9`, `F63`, `F7`,
`F82`, bare `except`).
- Check: CI step; a bare `except:` added to a file fails it.

## 2. Web app access and rate limits

**IMP-010 Access code (must, 3.2.7; plan S2, decisions 1 and 7).**
- When `WEBAGENT_ACCESS_CODE` is set, every page and every `/api/*` route except
  `/api/health`, `/api/auth/login`, the sign-in page and the static files it needs
  (icons, SVGs, the manifest) needs a valid `wa_session` cookie. Pages without one
  redirect to `/login?next=<path>`; API routes answer 401 `{"error": "sign_in_required"}`.
- `POST /api/auth/login` with the right code sets `wa_session` (HTTP-only, `Secure`
  in production, `SameSite=Lax`, 30 days) holding a new random session ID, the expiry and
  an HMAC-SHA256 signature made with `WEBAGENT_SESSION_SECRET`. A wrong code answers 401.
  Right and wrong codes both answer no sooner than 500 ms after the request arrives, and
  codes are compared in constant time. After 20 wrong codes in one minute (per server
  process), sign-in answers 429 for the rest of that minute.
- `POST /api/auth/logout` clears the cookie.
- With no access code: in development, everything works and the server logs one
  warning. In production (`NODE_ENV=production`), every API route answers 503
  `{"error": "access_code_not_configured"}`, unless `WEBAGENT_ACCESS_CODE_DISABLED=true`.
- An access code shorter than 12 characters, or a `WEBAGENT_SESSION_SECRET` shorter
  than 32, counts as not configured: API routes answer 503
  `access_code_not_configured`, and the sign-in page says sign-in isn't set up.
- Accept: a request with no cookie, an expired cookie, or a cookie with a changed
  byte gets 401 (API) or a redirect (page).
- Check: Jest (middleware and login route); manual sign-in on Vercel preview and
  `next start`.

**IMP-011 Web app rate limit per session (must, 3.2.7; plan Q1).** `/api/chat` counts
each accepted request against the caller's session, using the existing window and limit
(`CHAT_SETTINGS.rateLimitPerMinute`). One session reaching the limit doesn't affect
another. Without a session (access code off in dev, or `WEBAGENT_ACCESS_CODE_DISABLED=true`),
requests share one bucket, as today.
- Check: Jest: 10 requests from session A pass and the 11th gets 429; a request from
  session B still passes.

**IMP-012 Client ID to the backend (must, 3.2.7; plan S13, decision 6).** The web app
proxy (`src/app/api/task/backend.ts`) no longer copies the caller's `X-Forwarded-For`
or `X-Real-IP`. It sends `X-WebAgent-Client: s:<session id>` when the request has a
valid session. The backend uses `X-WebAgent-Client` as the client ID **only** when the
caller is trusted: the direct peer is in `WEBAGENT_TRUSTED_PROXIES` (a self-hosted web
app), or the request carries a valid `X-API-Key` (the web app on Vercel, whose outgoing
addresses change; only servers hold that key). Otherwise it uses the right-most address
in `X-Forwarded-For` that isn't a trusted proxy, and failing that the peer address.
- Accept: a browser that sends its own `X-Forwarded-For: 1.2.3.4` through the web app
  is still limited as its own session.
- Accept: a direct backend caller with no valid API key that sends `X-WebAgent-Client`
  is limited by its own address.
- Accept: two sessions through a web app on Vercel (peer not in the trusted list, valid
  API key) get separate budgets.
- Check: Jest (proxy headers); pytest (`client_identifier` table test); the Vercel case
  in the release check (task 1.26).

**IMP-013 Spending caps (must, 3.2.7; decision 1).** Monthly spending limits are set
in the Together AI and Tavily dashboards for every key used in uat and prod. The values
and the date set are recorded in `docs/RUNBOOK.md` (or `PRODUCTION_READINESS.md`
until the runbook exists).
- Check: manual; the record exists.

**IMP-014 Dev backend stays local (must, 3.2.7; plan S1).**
- `config/dev.yaml` sets `api.host: "127.0.0.1"`.
- `python -m backend.main` refuses to start in dev when the host isn't a loopback
  address and API keys are off, unless `WEBAGENT_ALLOW_OPEN_DEV=true`. The error says how
  to fix it.
- `backend/docker-compose.yml` publishes ports 8000, 9090 and Grafana's port on
  `127.0.0.1` only.
- Check: pytest (`check_dev_exposure` cases); manual `lsof -i :8000` after `npm run
  stack:up` shows `127.0.0.1`.

## 3. Backend input, guard and small fixes

**IMP-020 Intake guard on chat completions (must, 3.2.7; plan S3, decision 11).**
`POST /api/v1/chat/completions` runs the same intake check as `POST /api/task` before it
starts a task. A refusal answers with the same status (403 or 409) and the same body
shape as `/api/task`.
- Check: pytest: a request that `/api/task` refuses in `act` mode is refused the same
  way here; in `shadow` mode it runs, and the decision is recorded.

**IMP-021 Document IDs (must, 3.2.7; plan Q6).** `GET /api/v1/documents/document/{id}`
and `GET /api/v1/documents/related/{id}` accept only UUIDs; anything else answers 404
without touching the disk.
- Check: pytest: `..`, `%2e%2e`, an empty ID and a non-UUID give 404; a real ID works.

**IMP-022 Input limits (must, 3.2.7; plan Q7).**
- `POST /api/task` `query`: at most 20,000 characters (same as a history turn).
- `POST /api/v1/chat/completions` `messages`: at most 9 (8 earlier turns and the
  question), each at most 20,000 characters, role `user`, `assistant` or `system`
  (`system` messages are not passed to the agents).
- Document search `limit`: 1–50. Related documents: `depth` 1–3, `limit` 1–50.
- Document upload: the extension is checked against the loader's extension table
  (`services/document_loader.py`, the dict used by `_get_loader_for_file`) **before**
  the file is written. An unsupported type answers 415.
- Check: pytest: each limit at, above and below its bounds.

**IMP-023 Guard settings pass through (must, 3.2.7; plan Q8).** Every field of
`SecurityGuardConfig` set in YAML reaches `settings.SECURITY_GUARD`, including
`harm_judge`, `jailbreak_judge` and `prompt_guard_threshold`.
- Check: pytest loads each of dev, uat and prod YAML and compares every field.

**IMP-024 Small fixes (must, 3.2.7; plan Q10–Q14).**
- `npm run research` runs a script that exists, or is removed.
- `backend/models/services/fine_tune_service.py` imports without error, or is removed.
- CORS middleware is added once, with methods, headers and credentials from YAML, and
  uat/prod `allow_headers` include `X-API-Key`.
- `RATE_LIMIT_BURST_LIMIT` is either applied or removed from code, YAML and
  `.env.local.example`.
- Startup and shutdown use a FastAPI `lifespan` handler; no `on_event` warnings remain.
- Check: pytest (CORS headers on a preflight; import of every module under `app/`);
  the warnings summary of the pytest run lists no `on_event`.

## 4. Agent settings

**IMP-030 Per-agent settings apply (must, 3.2.7; plan Q15, decision 10).**
- Senior Research's loop limit comes from `agents.senior_research.max_follow_up_questions`.
- An agent's `temperature` is applied to its model **only when the YAML sets it** for
  that agent; otherwise the global `llm.temperature` stays.
- `web_research.max_searches_per_task` caps the searches one web research pass makes.
- `internal_research.max_documents` caps the documents internal research retrieves.
- Any per-agent key that is still unused after this change is removed from
  `config_def.py` and the YAML files.
- Check: pytest, for each YAML file: build each agent and compare its loop limit,
  temperature and caps with the file. Plus IMP-NFR-07.

**IMP-031 Research evaluation that can't be read (must, 3.2.7; plan Q9).** When Senior
Research can't parse the model's evaluation, the result is `research_status:
"unknown"`, the log records a warning, and the workflow treats it as "needs more
research" until the loop limit, then finishes with the last synthesis. The prompt shows
the real loop limit.
- Check: pytest: a reply without an evaluation block, with a broken one, and a good one.

## 5. Time and cost limits

**IMP-040 Timeouts on outbound calls (must, 3.2.7; plan Q3, Q4).** Async Together calls
time out after `REQUEST_TIMEOUT_SECONDS` (120 s), the same as sync calls. Each Tavily
search (including its retry) is limited to `WEBAGENT_SEARCH_TIMEOUT_SECONDS` (default
30). In 3.2.7 a timed-out search is logged as a warning and treated as no results; from
3.3.0 it is reported as an error (IMP-055).
- Check: pytest with a fake slow server and a fake slow Tavily client.

**IMP-041 Overall run limit (must, 3.2.7; plan R10).** A run that takes longer than
`WEBAGENT_TASK_MAX_SECONDS` (default 420: about twice a typical run, and under the web
app's 8-minute wait, so the backend's message reaches the user first) is cancelled. Its
result file gets `status: "error"`, `error_code: "timeout"` and the message "The research
took too long and was stopped." The web app shows that message. `WEBAGENT_TASK_TIMEOUT_MINUTES` (stale record
check) must be larger than the run limit; startup logs a warning if it isn't.
- Check: pytest with a fake workflow that sleeps, and a limit of 1 s.

**IMP-042 Per-task token budget (must, 3.3.0; plan R11).** Each run counts the total
tokens Together reports. Before each model call, if the count has reached
`WEBAGENT_TASK_TOKEN_BUDGET`, the call raises `TaskBudgetExceeded` and the run ends with
`error_code: "budget"` and the message "The research used its token budget and was
stopped.", which the web app shows. A budget of 0 means no limit, which is the value until task 2.10
sets the default from measurement: three times the p95 of 10 typical agentic runs,
rounded up to the next 10,000. Every final result records `tokens_used`.
- Check: pytest with a fake model that reports usage; measurement recorded in the PR.

**IMP-043 One failed research agent doesn't fail the run (must, 3.2.7; plan Q5).** If
web or internal research raises, the other's report is kept, the failure is added to the
workflow logs and `state.context["research_errors"]`, and Senior Research runs with
what there is. If both fail, the run fails with both messages.
- Check: pytest with one fake agent that raises.

**IMP-044 Web app retry timer (must, 3.2.7; plan Q2).** Each attempt in
`fetchWithRetry` has its own `AbortController` and its own timeout, and the timer is
cleared when that attempt ends.
- Check: Jest with fake timers: attempt 1 times out, attempt 2 starts with a fresh
  signal and succeeds.

## 6. Cancel, progress and failure display

**IMP-050 Cancel endpoint (must, 3.3.0; plan R1, decision 8).** `POST
/api/task/{id}/cancel` (API key required) writes the marker file and answers 202
`{"status": "cancelling"}`. The worker running the task stops it within 5 seconds and
saves `status: "error"`, `error_code: "cancelled"`. Cancelling a finished task answers
409; an unknown task answers 404. After a cancel, a late write from the stopped run
must not overwrite the cancelled result.
- Check: pytest with 2 task managers sharing one results folder (one runs, the other
  receives the cancel).

**IMP-051 Cancel in the web app (must, 3.3.0; plan R1).** While a research run is in
progress, the chat shows a **Stop** button. Clicking it calls the cancel route, stops
polling, and replaces "Thinking…" with "Research stopped." Sending a new message is
possible straight away. The proxy route `POST /api/task/[taskId]/cancel` forwards with
the server's API key.
- Check: Jest (button, request, state); manual run.

**IMP-052 Progress in the status reply (must, 3.3.0; plan R2, decision 9).** While a
run is processing, `GET /api/task/{id}` includes `progress: {step, logs, updated_at}`.
`step` is the last finished graph node in plain words (for example "Web and internal
research"); `logs` holds at most the last 5 workflow log lines. The result file is
written at most once per finished node. The field is missing for direct-path tasks.
- Check: pytest: a fake 3-node workflow produces 3 updates, readable from another
  task manager instance.

**IMP-053 Progress shown (must, 3.3.0; plan R2).** Under "Thinking…", the chat shows
the current step and the latest log line, updated on each poll, and how long the run
has taken so far.
- Check: Jest.

**IMP-054 Clear research failure (must, 3.3.0; plan R3).** When an agentic request
fails (not a guard refusal), the web app shows an error message with **Try again** and
**Answer directly instead** buttons, and does not call `/api/chat` on its own. Timeout,
budget and cancel each have their own wording.
- Check: Jest, one test per failure kind.

**IMP-055 Search errors are reported (must, 3.3.0; plan R4).** A failed or timed-out
Tavily search is recorded in the web research report as `search_errors` and in the
workflow logs, so "search broken" is distinct from "nothing found".
- Check: pytest.

## 7. Web app and backend hardening

**IMP-060 Security headers (must, 3.3.0; plan S4).** Every web app response carries
`Content-Security-Policy` (default-src 'self'; images from self, `data:` and `blob:`;
connect to self only), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: strict-origin-when-cross-origin`, and, in production,
`Strict-Transport-Security: max-age=31536000`.
- Check: Jest on the header config; manual check in the browser console that the app
  works with no CSP errors (voice input and screenshot capture included).

**IMP-061 Image check on the server (must, 3.3.0; plan S5).** `/api/chat` accepts
`image` only as a `data:image/(png|jpeg|gif|webp);base64,` URL of at most 5 MB after
decoding. Anything else answers 400. The browser checks the same size before sending.
- Check: Jest.

**IMP-062 Safe source links (must, 3.3.0; plan S6).** Source links render only for
`http:` and `https:` URLs; other URLs show as plain text.
- Check: Jest.

**IMP-063 Small open endpoints (must, 3.3.0; plan S7).**
- Web app `/api/ping` returns `{status, timestamp}` only.
- Web app `/api/health` returns `{status, version, timestamp}` only.
- Backend `/api/v1/health/detailed` needs the API key.
- In uat and prod, the OpenAPI docs (`/api/v1/docs`, `/redoc`, `/openapi.json`) are off.
- In Kubernetes, a NetworkPolicy lets only the ingress controller and the monitoring
  namespace reach the backend pods, and the public Ingress answers 404 for `/metrics`.
  (A NetworkPolicy works on ports, not paths, hence both parts.)
- Check: Jest; pytest (prod settings: docs 404, detailed health 401 without key);
  manifest review.

**IMP-064 Diagnostics button (must, 3.3.0; plan S8).** The error panel's check calls
`/api/ping`, never `/api/chat`.
- Check: Jest.

**IMP-065 No static keys or unused JWT code (must, 3.3.0; plan S9).** `DEV_API_KEYS`,
`create_jwt`, `decode_jwt` and `validate_auth` are removed, along with `python-jose`
if nothing else uses it. The README no longer mentions JWT. In dev with API keys on,
`WEBAGENT_API_KEY` is the only valid key.
- Check: pytest (dev key `dev_admin_key` gives 401); grep finds no JWT code.

**IMP-066 Image host allowlist (must, 3.3.0; plan S10).** `next.config.js` allows
remote images only from hosts listed in `WEBAGENT_IMAGE_HOSTS` (comma-separated; empty
means none).
- Check: config test.

**IMP-067 Non-root LLM images (should, 3.3.0; plan S11).** `Dockerfile.llm` and
`Dockerfile.llm-mlflow` run as a non-root user.
- Check: `docker run … id -u` is not 0.

**IMP-068 Security policy (must, 3.3.0; plan S12).** `SECURITY.md` says how to report
a vulnerability, what is in scope, and how fast reports get an answer.
- Check: file exists and is linked from the README.

## 8. Privacy and data

**IMP-070 No question text in logs (must, 3.3.0; plan S14, decision 4).** Outside dev,
no log line written by the backend contains the question or history text. Agents log
the task ID (already added by `log_context`) and the question's length. In dev, text
may appear at DEBUG only. This applies to the agents, the task manager, the triage and
guard modules, the web app's server-side logs, and what `apiClient` writes to the browser
console (`apiClient.ts:539`, `:578`).
- Check: pytest runs a fake task with a marker question under `WEBAGENT_ENV=prod` and
  checks `caplog` never contains the marker; Jest does the same for the API routes and
  `apiClient`.

**IMP-071 Result retention (must, 3.3.0; plan C2, decision 4).** Result files (and
any `.cancel` marker) whose `created_at` is older than `WEBAGENT_RESULTS_RETENTION_DAYS`
(default 30) are deleted at startup and once a day, the same way as triage and
security collections. A record whose `created_at` can't be read is judged by the file's
modified time.
- Check: pytest with fake dates.

**IMP-072 Delete and list use the files (must, 3.3.0; plan C2).** Deleting a task
removes its result file. `GET /api/v1/tasks` lists tasks from the result files (any
worker gives the same answer), newest first, with `limit` (1–100, default 20) and
`offset`. The list shows each question's length, not its text (in line with IMP-070).
- Check: pytest with 2 task manager instances sharing a folder.

**IMP-073 Backups (must, 3.3.0; plan C5).** The runbook describes how to back up and
restore `/app/data` in Kubernetes (a VolumeSnapshot or a CronJob that copies the folder),
and the restore was tried once in uat.
- Check: manual; the runbook section and the date of the tried restore.

**IMP-074 Rate limits per worker documented (must, 3.3.0; plan C1, decision 3).**
`PRODUCTION_READINESS.md` says that each worker keeps its own count, so with 2
workers the effective limit is up to twice the setting. The unused `redis` package and
`REDIS_*` settings are removed.
- Check: doc review; pytest config tests pass without the Redis fields.

## 9. Code health

**IMP-080 True duplicate / orphan code removed (must, 3.4.0; plan H1, narrowed).** Delete
only modules that duplicate behaviour already used in production and have no unique callers:
`services/lda_document_extractor.py` (LDA lives in `document_extraction_agent.py`; nothing
imported this file), and any other orphans listed in the backend design's 3.4.0 cleanup table.
Do **not** delete the advanced observability stack (`drift_detection`, `grafana_integration`,
`metrics_registry`, `mlflow_integration`, `llm_service`, `api/routes/observability`,
`self_hosted_llm_service`, demos that exercise them). `services/document_ingestion_service.py`
is the production ingestion path; `doc_ingestion_service.py` is merged in IMP-128, not
deleted here.
- Check: pytest passes; at **3.4.0** the import scan may `xfail` the IMP-128 module set
  (`PENDING_REPAIR_350` in `test_import_all_modules.py`); grep finds no import of a module
  deleted in 3.4.0 (for example `lda_document_extractor.py`).

**IMP-128 Advanced observability repair and integrate (must, 3.5.0).** The stack above is
repaired (imports, missing helpers on `app.core.observability`, drift detectors wired to real
metrics) and integrated into the running product: `api/routes/observability` is mounted under
`/api/v1` behind the API key; document drift lives in `document_ingestion_service.py` as
`DocumentProcessingObservability` (exported as `doc_ingestion_service` for the observability
routes); the standalone `doc_ingestion_service.py` module is removed; optional self-hosted LLM
Docker images use `backend/standalone/llm_server.py`; Grafana dashboard helpers and MLflow
integration are documented in the runbook. Production agent runs continue to use `llm.py` for
chat; `llm_service.py` feeds observability and drift APIs without duplicating routing logic.
Repair at least: `core/drift_detection.py`, `core/grafana_integration.py`,
`core/metrics_registry.py`, `core/mlflow_integration.py`, `services/llm_service.py`,
`services/self_hosted_llm_service.py`, and `api/routes/observability.py`.
- Check: `test_import_all_modules.py` has no `xfail` for the observability import set;
- Check: no `xfail` on that import set; `test_observability_api.py` (or equivalent) hits
  `/api/v1/observability/metrics` with a key and gets 401 without one; drift and LLM management
  endpoints return structured JSON; RUNBOOK describes how this relates to `/metrics` and
  LangSmith.

**IMP-081 Graph wiring (must, 3.4.0; plan H3, H4).** The graph has no nodes that no
edge reaches. `_route_to_team_manager` and the unread `needs_analysis` flag are removed,
or wired into routing with a test.
- Check: pytest asserts every node is reachable from `__start__`.

**IMP-082 One error shape (should, 3.4.0; plan H9).** Every backend error response
(except the guard refusal, whose shape the web app depends on) includes
`"error": {"code": "...", "message": "..."}`. Responses that have a `detail` field today
(FastAPI's `HTTPException` and validation errors) keep it, so existing API clients don't
break (IMP-NFR-06); the plain `{"error": "..."}` bodies from the rate limit, size limit and
global handler become the object form, which is listed in the upgrade notes.
- Check: pytest across 401, 404, 413, 422, 429 and 500.

**IMP-083 One `ApiError` and one settings type (must, 3.4.0; plan H8).** The web app
has a single `ApiError` class and a single accessibility-settings type.
- Check: `npm run check-types`; grep.

**IMP-084 Microservices framework labelled (must, 3.4.0; plan H10).** The README
describes `backend/models/` (gateway, registry, manager) as an optional, separate tool
that the agents don't use.
- Check: doc review.

**IMP-086 One entry point for models (must, 3.4.0; plan H2).** Chat models are built only in
`app/services/llm.py` (`together_llm.py` is imported only by `llm.py`), and embeddings
only through one `get_embeddings()` in `app/services/vectordb.py`.
`agents/internal_research.py:205`, which builds `OpenAIEmbeddings` itself today, uses it.
- Check: pytest scans `backend/app` imports: no module except `llm.py` imports a chat
  model class or `together_llm`, and none except `vectordb.py` imports an embeddings class.

**IMP-085 Repo clutter (must, 3.4.0).** The plan's clutter list is done: backup Babel
files, empty folders, tracked generated files, `test_docs/` moved, unused web app
dependencies removed.
- Check: `git ls-files` review; `npm ci && npm run build` passes.

## 10. Reliability fixes

**IMP-090 Vector search off the event loop (must, 3.4.0; plan R5).** Vector search
and embedding calls run in a worker thread.
- Check: pytest: a slow fake vector store doesn't delay a concurrent `/health` call.

**IMP-091 Document extraction cost (must, 3.4.0; plan R6).** LDA runs only on
documents up to `WEBAGENT_LDA_MAX_CHARS` (default 200,000), with at most 5 passes.
Larger documents skip topic modelling and say so in the result.
- Check: pytest with a large fake document.

**IMP-092 Browser storage full (must, 3.4.0; plan R7).** When saving chat history hits
`QuotaExceededError`, the oldest half of the turns is dropped and saving is retried
once; the user sees "Older messages were removed to save space."
- Check: Jest with a fake `localStorage`.

**IMP-093 No bare `except` (must, 3.3.0; plan R8).** No bare `except:` in
`backend/app`. Enforced by IMP-003's lint rule, which is why this moves up from the plan's
3.4.0: the rule can't be turned on until the six existing ones are fixed.
- Check: CI lint.

**IMP-094 Jest exits cleanly (should, 3.4.0; plan R9).** `forceExit` is removed and
Jest exits on its own.
- Check: `npm test` finishes with no "open handles" warning.

**IMP-095 Embedding provider (should, 3.4.0; plan C4).** `WEBAGENT_EMBEDDINGS_PROVIDER`
selects, in `get_embeddings()` (IMP-086), `openai` (default, today's behaviour) or `local`
(`all-MiniLM-L6-v2`, already baked into `/opt/models` by `backend/Dockerfile:62`). Changing it requires rebuilding the
vector store; the docs say so.
- Check: pytest with each provider faked.

## 11. Dependencies and deployment

**IMP-100 Requirements split (must, 3.4.0; plan D1).** `backend/requirements.txt`
holds only what the server imports. Dev tools go to `backend/requirements-dev.txt`;
demo-only packages go to `backend/requirements-demos.txt`. The image installs only
the runtime file. The root `requirements.txt` installs runtime and dev.
- Check: image builds and starts; CI installs runtime and dev; image size recorded.

**IMP-101 Package cleanup (must, 3.4.0; plan D2).** `py2neo` is replaced by the
`neo4j` driver, `PyPDF2` by `pypdf`, and unused packages (`platform-utils`, `redis`,
`python-jose` and `PyJWT` if IMP-065 removed their last use) are removed.
- Check: pytest; image build.

**IMP-102 Monitoring versions (must, 3.4.0; plan D3).** Grafana and Prometheus images
are on currently supported versions in Compose and Kubernetes, and the dashboards still
load.
- Check: manual: `npm run stack:up`, dashboards render.

**IMP-103 Compose tag and limits (should, 3.4.0; plan D4).** The Compose image tag has
no `latest` fallback, and the backend service has a 3 GiB memory limit.
- Check: Compose config review.

**IMP-104 ESLint config matches Next (should, 3.4.0; plan D5).** `eslint-config-next`
has the same minor version as `next`.
- Check: `npm run lint` passes.

## 12. Tests

**IMP-110 Coverage floors (must, 3.4.0; plan T4).** The Jest floors rise by at least 5
points from 46/45/38/34 (lines/statements/branches/functions). The backend has a
coverage floor in CI at its measured value after IMP-080 (orphan removal only), rounded down.
- Check: CI.

**IMP-111 One copy of each test (must, 3.4.0; plan T5).** The duplicate test pairs are
merged and no test file has a `__main__` block.
- Check: grep; pytest count doesn't drop except for removed duplicates.

**IMP-112 Red-team cases (must, 3.4.0; plan T6).** The red-team suite has at least one
case each for: instructions hidden in a search result, a forged assistant turn in
history, a tool request the user never asked for, and instructions hidden in document
text returned by internal research. Each asserts what the guard decides in shadow mode
and what it would do in `act` mode.
- Check: pytest.

**IMP-113 Browser-to-backend test (should, 3.4.0; plan T7).** One Playwright test signs
in, asks a question in Always mode against a backend started with a fake LLM and fake
search, sees progress, and gets the answer. It runs in CI on pull requests to
`release/*`.
- Check: CI job.

## 13. Docs

**IMP-120 Docs match the code (must, 3.3.0; plan section 11).** Every row of the plan's
"docs that don't match the code" table is fixed, and `PRODUCTION_READINESS.md` gains
the additions listed there (one replica, per-worker limits, restarts lose runs,
retention, access code, spending caps). The `VERCEL_DEPLOYMENT.md` row is fixed earlier,
in 3.2.7, together with IMP-012, which changes what it describes.
- Check: doc review against the table.

**IMP-121 Contributor and ops docs (must, 3.4.0).** `CONTRIBUTING.md` (setup, tests,
LFS, constraints, hooks) and `docs/RUNBOOK.md` (deploy, roll back, rotate secrets and
the access code, disk full, failed probes, backup and restore, spending caps) exist.
CHANGELOG entries before 3.0 move to `docs/CHANGELOG-archive.md`; the `VERSION_2_4_*`
files move to `docs/archive/`.
- Check: doc review; links from the README work.

## 14. Non-functional requirements

| ID | Requirement | Check |
|---|---|---|
| IMP-NFR-01 | All existing tests still pass after each task; new code has tests; CI is green before each release | CI |
| IMP-NFR-02 | No new service to run (no Redis, no database). No new runtime dependency in the backend; the web app adds none for sign-in (Web Crypto is built in) | Diff of requirements and `package.json` |
| IMP-NFR-03 | The middleware's session check adds under 5 ms per request at p95 | Jest timing on 1,000 checks; spot check in the browser |
| IMP-NFR-04 | A typical agentic run takes no longer than in 3.2.6 (2.5–3.5 minutes), measured on the same 5 questions | Container run, before and after 3.3.0 |
| IMP-NFR-05 | New files stay under about 200 lines | Review |
| IMP-NFR-06 | API clients that use `X-API-Key` keep working, apart from the changes listed in the upgrade notes (IMP-NFR-08); status replies only gain fields; no existing response field is removed | pytest contract tests |
| IMP-NFR-07 | After IMP-030, 5 fixed research questions and 5 routine ones are run before and after in uat. A person compares the answers; no answer is clearly worse, and run times stay within IMP-NFR-04 | Results table in the PR |
| IMP-NFR-08 | Each change that alters a default or behaviour is listed under "Upgrade notes" in the CHANGELOG. At least: 3.2.7 — access code needed in production, dev host and Compose ports local only, agents use their YAML temperatures and limits, 7-minute run limit, longer chat-completions inputs rejected; 3.3.0 — no silent fallback to plain chat, OpenAPI docs off and detailed health behind the key in uat/prod, result files deleted after 30 days, token budget, web CSP (`connect-src 'self'`, `WEBAGENT_IMAGE_HOSTS`, same-origin `/api`); 3.4.0 — error bodies gain the `error` object, packages removed; 3.5.0 — `/api/v1/observability/*` available behind the API key, document drift merged into the main ingestion service | Review |
