# WebAgent Architecture

How the parts of WebAgent 3.12.0 fit together: the web app, the backend, the research workflow, the security guard, triage, the stores, and observability. Each section names the code that implements it. Feature designs and requirements are in `planning/specs/`.

## System overview

```text
  Browser ── sign in with the access code, chat, Agentic: Off · Auto · Always
     │
     ▼
┌─ Web app ─ Next.js 15 · React 19 ─ src/ ────────────────────────────────────────────┐
│  middleware.ts: signed session cookie on every page and API route                   │
│  /api/chat ──── direct answer: guard check (checkpoint 1) ──► Together AI           │
│  /api/triage, /api/task ── proxy to the backend, adding X-API-Key and the session ─┐ │
└─────────────────────────────────────────────────────────────────────────────────────┼─┘
                                                                                      ▼
┌─ Backend ─ FastAPI ─ backend/ ──────────────────────────────────────────────────────────┐
│  API key · rate limit per session · size limit · one error format                       │
│                                                                                         │
│  Triage: rules, then a MiniLM classifier ──► direct: one model call                     │
│                                          └─► research: run supervisor (7 min, budget)   │
│                                                 │                                       │
│                                                 ▼                                       │
│                                   LangGraph research workflow                           │
│                                                                                         │
│  Security guard, in process: Cedar policies + local classifiers, shadow mode            │
│    checkpoint 1: every text before a model  ── ContextDefender                          │
│    checkpoint 2: every tool call and subgoal ── intent signals (+ judge if on)          │
│                                                                                         │
│  data/: results/ · vectordb/ · graph · artifacts/ · drift/ · security/ · triage/        │
│  Observability: /metrics · /api/v1/observability · logs · LangSmith                     │
│  Topics API · data/topics/ ◄── Scheduler (separate process): due runs → /api/task       │
│    (scheduler token; topics, schedule state, run history, monthly allowance)            │
└───────┬───────────────────────┬───────────────────────┬─────────────────────────────────┘
        ▼                       ▼                       ▼
   Together AI             Tavily search          Prometheus + Grafana
   (OpenAI if set)         (web research)         (Compose or Kubernetes), LangSmith
```

The web app and the backend are separate processes. The backend is the only thing that runs agents, holds the documents and decides security. The web app never calls Tavily and never sees the backend's stores. For direct answers it calls Together AI itself, after the backend's guard has checked the conversation.

## Web app (`src/`)

- **Stack:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS. React context holds state (`AppContext`, `ChatContext`), and `fetch` makes all HTTP calls.
- **Sign-in (`middleware.ts`, `utils/session.ts`):**
  - The operator sets a shared access code (`WEBAGENT_ACCESS_CODE`). `/login` exchanges it for a session cookie signed with HMAC-SHA256 (`WEBAGENT_SESSION_SECRET`).
  - The middleware checks the cookie on every page and API route. Without one, a page redirects to `/login` and an API route answers 401. Only the static files and `/api/health` stay open.
  - Production refuses to run without the code and the secret (503), unless `WEBAGENT_ACCESS_CODE_DISABLED=true`.
- **API routes (`src/app/api/`):**
  - `chat`: a direct answer, guard check first (see below).
  - `triage` and `task` (with `[taskId]` and `cancel`): proxies to the backend. They add `X-API-Key` and `X-WebAgent-Client: s:<session>`, so the backend rate-limits each session, and they never forward the browser's own forwarding headers.
  - `auth/login`, `auth/logout`, `settings`, `health`, `ping`.
- **Direct answers (`/api/chat`):**
  1. The conversation goes to the backend's `POST /api/guard/input` (checkpoint 1) with a 3-second timeout.
  2. A refusal is returned as is: 403 for declined, 409 for consent needed, with a token to send back.
  3. If the backend can't be reached, a local pattern check (`utils/guardPatterns.ts`) runs instead, and the answer carries a notice that the full check was unavailable.
  4. The model call goes to Together AI with the user's AI style and knowledge focus.
- **Research answers:** `POST /api/task`, then a poll of `GET /api/task/{id}` every few seconds. While a run works, the poll returns its current step; **Stop** calls the cancel route.
- **Security headers (`next.config.js`):** Content Security Policy (scripts from self, `unsafe-inline` because Next bootstraps inline without nonces), `X-Frame-Options: DENY`, `nosniff`, a referrer policy, and HSTS in production. `next/image` loads remote images only from `WEBAGENT_IMAGE_HOSTS`.
- **The rest of the UI:**
  - the Agentic menu, answer labels, **Research this in depth**, **Try again** and **Answer directly instead**;
  - the consent dialog and the security note with **Approve**;
  - source and code side panels, image attachment and screenshots;
  - appearance and high-contrast settings, and an offline banner.

## Backend API layer (`backend/main.py`, `app/api/`, `app/core/`)

- **Authentication:** `X-API-Key` on every route except `/health`, `/api/v1/health/` and `/metrics` (`core/security.py`). It's required in uat and prod, and the server won't start there without `WEBAGENT_API_KEY` and `WEBAGENT_SECURITY_SECRET_KEY`.
- **Middleware (`core/middleware.py`):** CORS, security headers, request logging, and a request size limit, each added once.
- **Rate limiting:**
  - 60 requests a minute per caller by default, counted in memory per worker.
  - Behind the web app, the caller is the sign-in session (`X-WebAgent-Client`). That header is accepted only from `WEBAGENT_TRUSTED_PROXIES` or from a caller holding the API key; other callers are identified by the right-most untrusted `X-Forwarded-For` address.
  - Health checks and `/metrics` are never limited. A limited request gets 429 with `Retry-After`.
- **One error format:** `{"error": {"code", "message"}}` (FastAPI's `detail` is kept too). Intake-guard refusals keep their flat shape, which the web app reads.
- **Startup checks (`core/startup_checks.py`)** refuse to start when anything is unsafe or missing:
  - production settings, task time limits, triage and security model files;
  - a security classifier that hasn't passed its gate (unless it is installed shadow-only);
  - code execution in uat or prod without the Docker sandbox;
  - in dev, binding to a public address.
- **Routes:** listed with their fields in [api_reference.md](api_reference.md). The current ones are `/api/task`, `/api/triage` and `/api/guard/input`. The `/api/v1/chat`, `/api/v1/tasks` and `/api/v1/chat/completions` routes are older, OpenAI-style entry points onto the same task path.

## Tasks (`app/services/task_manager.py`, `run_supervisor.py`, `result_store.py`)

- **Two paths:** `POST /api/task` runs a task on one of them. The workflow (`run_workflow`) builds and runs the LangGraph graph. The direct path (`run_direct`) makes one text-only model call. The result records which one ran (`route`) and why (`decided_by`: `mode`, `rule`, `model`, `fallback`, `config` or `override`).
- **Where tasks run:** in the server process, as FastAPI background tasks.
- **The run supervisor bounds each research run:**
  - it stops after `WEBAGENT_TASK_MAX_SECONDS` (420) and on cancel;
  - it stops at `WEBAGENT_TASK_TOKEN_BUDGET` (0, off by default);
  - it publishes the current step for polls (`services/progress.py`);
  - a failed run gets an `error_code`: `timeout`, `budget`, `cancelled` or `escalated`.
- **Status storage:**
  - `data/results/<task_id>.json` is written when a task starts and again when it ends, so any worker can answer a poll and finished reports survive a restart.
  - Task IDs are UUIDs, and only UUIDs are accepted, so an ID can't point outside the folder.
  - There is no checkpointing: a restart loses runs in progress, and a task still `processing` after `WEBAGENT_TASK_TIMEOUT_MINUTES` (30) is reported as interrupted.
  - Results are deleted after `WEBAGENT_RESULTS_RETENTION_DAYS` (30).
- **Follow-ups:** up to 8 earlier turns (`previous_messages`) travel with a task, and the supervisor rewrites a follow-up as a complete question.

## Research workflow (`app/graph/workflows.py`, `app/agents/`)

```text
 question ─► Supervisor ── plans: web? internal? document? ── rewrites follow-ups as full questions
                 │
                 ├── document only ──► Document Extraction ──► done
                 │
                 └── research ──► ┌─ Web Research ──────┐  (in parallel)
                        ▲         └─ Internal Research ─┘
                        │                   │
                        │                   ▼
                        └── another round ─ Senior Research ── scores 1–10, asks follow-up questions
                                            │
                                            ▼ good enough, or round limit reached
                                       Team Manager ── writes the report
                                         │      ▲
                         needs analysis  ├──► Data Analysis ─────────────────┤
                         needs code      └──► Coding Assistant ─► Coding Review ┘
                                            │
                                            ▼
                                     report saved to data/results/<task_id>.json
```

| Agent | File | Does |
| --- | --- | --- |
| Supervisor | `supervisor.py` | Plans the research (`models/research_plan.py`: web search, internal knowledge, document extraction) and writes the standalone question and its sub-questions |
| Web Research | `web_research.py` | Searches the web with Tavily (`services/web_search.py`) and summarises the results with sources |
| Internal Research | `internal_research.py` | Retrieves from the loaded documents: Chroma vectors plus BM25, merged |
| Senior Research | `senior_research.py` | Scores the findings 1–10 and asks follow-up questions for another round, at most `agents.senior_research.max_follow_up_questions` times (3 in dev and uat, 5 in prod) |
| Document Extraction | `document_extraction_agent.py` | Extracts from one document (`document_id`) when that's all the plan needs |
| Team Manager | `team_manager.py` | Writes the final report; calls Data Analysis or the Coding Assistant when the question needs it |
| Data Analysis | `data_analysis.py` | Patterns, figures and suggested charts from the findings |
| Coding Assistant, Coding Review | `coding_assistant.py`, `coding_review.py` | Write code for the report, and check it |

- **Parallel research:** web and internal research run at the same time (`run_research_agents`); one failing doesn't stop the other. Everything else runs one step at a time.
- **Model calls:** every agent's model call goes through `services/llm.py` to the configured provider. The default is Together AI `meta-models/Muse-Glimmer-30B`; OpenAI is used only where an agent sets `provider: openai`. Each agent can set its own `model` and `max_tokens` under `agents:` in `config/{env}.yaml`. A reply cut off at its limit is logged, and temporary errors (429, 5xx) are retried with backoff.
- **Generated code:** it runs only through `utils/code_sandbox.py`, without the server's API keys. `WEBAGENT_CODE_SANDBOX=local` is for dev; uat and prod require `docker` (`backend/sandbox/`: no network, CPU and memory limits). The server itself never runs model-written code unless it's explicitly enabled.

## Security guard (`app/security/`)

The guard puts two checkpoints in the path of every task. Both ask the same in-process decision point: Cedar policies (`backend/config/security/policies.cedar`, evaluated by `decision.py`) over a context of scores and facts.

| Outcome | Meaning |
| --- | --- |
| `allow` | Go ahead |
| `verify` | Ask the judge (an LLM check, off by default) or a classifier's confident band; if neither is available, treat it as the policy says |
| `consent` | The user must confirm first (a signed, single-use token) |
| `decline` | Refuse, with a reason |
| `escalate` | Stop the task |

- **Checkpoint 1, before every model call (`context_defender.py`, `guard.py`):**
  - Every context item is scored on four axes: the user's turn, earlier turns, web pages, documents, code output.
  - The axes are **injection**, **harm**, **sensitive data** and **jailbreak**. ContextDefender scores them: a MiniLM-L12 classifier, `models/security/` version `cd-2026-10-13-2`, which passed its release gate.
  - A flagged web page chunk is dropped and the task becomes **tainted** (`ledger.py`). A risky user turn is declined, sent for consent or verified. A search that would carry a secret is declined, and one carrying personal data is verified.
  - The web app's direct chat uses the same check through `POST /api/guard/input`.
- **Checkpoint 2, before every tool action and plan subgoal (`tool_guard.py`, `intent.py`, `signals.py`):**
  - Web search, internal retrieval, document extraction, code execution, and the supervisor's and senior research's rewritten questions.
  - Its signals:
    - alignment with the task text and with the user's own turns (MiniLM cosine);
    - **risk above intent** (harm of the call minus harm of the request);
    - private text copied into a web search (salted shingles, kept in memory per task);
    - tools the plan didn't ask for;
    - drift across the task.
  - Acting rules today come from the original policies: on a tainted task, a search goes to verify and code needs consent; dangerous code, a secret in a search, and a document from the wrong source are declined. An action the guard skipped can be approved with the web app's **Approve**, which reruns the task with a signed, single-use approval.
- **The intent classifier (3.9.0).** A fine-tuned cross-encoder for checkpoint 2, `models/security/alignment/` version `ac-2026-10-17-1`.
  - It failed its release gate, so it is installed **shadow-only**: its scores are recorded next to the cosine signals and never decide.
  - A model that passes the gate replaces the cosine signals in the policy context.
- **Modes:** `input_mode` and `tool_mode` are `shadow` (record what would happen, block nothing) or `act`. Every environment runs in shadow now. The nine checkpoint 2 intent rules are shadow-only whatever the mode, until each one's threshold is measured (`planning/specs/checkpoint2-intent/`).
- **Collection:** with `collect: true` (uat), decisions are written as rows to `data/security/collected/` for review. Outside dev, those rows hold scores and decisions, not question text.
- **Design and status:** `planning/specs/agentic-security/` and `planning/specs/checkpoint2-intent/`.

## Agentic triage (`app/services/triage.py`, `triage_model.py`)

The web app's Agentic control has three modes: Off, Auto and Always. In Auto, a question goes to the research workflow only when it needs it.

- **Who decides:** the backend classifies (`POST /api/triage`); the web app routes. A direct answer goes through the web app's `/api/chat`, and a research answer through `/api/task`.
- **How:** rules first (an image or a greeting → direct; "research …" or "write a report …" → research). Then frozen `all-MiniLM-L6-v2` embeddings and a logistic regression (`models/triage/`, about 10 ms on CPU). If the classifier can't load, the answer is research.
- **API clients:** `/api/task` and `/api/v1/chat/completions` take `mode: off | auto | always`. With no mode and no `agentic` flag, the mode is `off`.
- **Learning:**
  - Shadow mode and "Research this in depth" clicks can save questions for review (`data/triage/collected/`). This is off outside dev unless configured, and rows are deleted after 30 days.
  - A new classifier version ships only if it passes its release gate.
  - Shadow mode records Always-mode questions only; in Auto, triage routes for real.
- **Design:** `planning/specs/agentic-triage/`.

## Documents and stores (`app/services/`)

| Store | Where | Holds |
| --- | --- | --- |
| Task results | `data/results/` | One JSON file per task: question, status, timings, report |
| Vector store | `data/vectordb/` (Chroma, embedded in the process) | Document chunks and embeddings for internal research |
| Graph store | Neo4j when `DATABASE_GRAPH_DB_URL` is set, else NetworkX in memory | Document relationships (same source, and links made in code) |
| Artifacts | `WEBAGENT_ARTIFACTS_PATH` (a local folder) | A copy of each loaded document and its metadata |
| Drift | `data/drift/` | Samples and frozen references for the drift detectors, shared by workers |
| Collection | `data/security/collected/`, `data/triage/collected/` | Guard and triage rows for review (when collection is on) |
| MLflow | `data/mlruns/` | Optional experiment tracking for internal research (`MLFLOW_TRACKING_URI`) |

- **How documents get in:** through `POST /api/v1/documents/upload` or `scripts/batch_document_loader.py` (`document_loader.py`). Each one is parsed (PDF, Word, HTML, text, Markdown, CSV, JSON), summarised, chunked into the vector store, linked in the graph store and copied to the artifacts folder. Topics come from LDA, capped by `WEBAGENT_LDA_MAX_CHARS`.
- **Who sees them:** loaded documents are shared by every task on the server. There are no per-user spaces.
- **When they're checked:** at retrieval, by checkpoint 1, like any other context.

## Observability (`app/core/metrics.py`, `app/core/drift*`, `app/api/routes/observability.py`)

- **Metrics:** `/metrics` for Prometheus. It covers requests and latencies, model calls and tokens per agent, task outcomes and durations, guard decisions per checkpoint and policy, shadow decisions, intent signals, the shadow classifier's scores, triage routes, and drift. With several workers it uses the multiprocess collector.
- **Drift:**
  - Measured on real traffic: model speed and tokens per agent, embeddings of questions and results, and document processing time, size and type. Each is compared with a frozen reference taken from the first samples.
  - Six alerts: model drift, model errors, rate limits, token use, document drift and document errors.
  - They appear at `/api/v1/observability/{drift,alerts,health}` (API key needed), on the Grafana "Model Drift" dashboard, and in Grafana's alert list. The references are reset with the `reset_baseline` routes.
- **Logs:**
  - Plain-text logs in `logs/webagent-<env>.log`.
  - Every line written during a task carries `[task <first 8 characters of its ID>]`.
  - Outside dev, a question appears in the logs only as its length.
- **Traces:** with `LANGSMITH_API_KEY` set, each task is one LangSmith trace (project `webagent-research`) with every agent step and token count.
- **Dashboards:** `backend/grafana/` for Compose, and `backend/kubernetes/monitoring/` for a cluster.

## Configuration

- **Environments:** `WEBAGENT_ENV` selects `config/{dev,uat,prod}.yaml`. Those files hold agents, models, limits, the guard and triage modes, and drift settings.
- **Precedence:**
  1. `WEBAGENT_ENV` from the shell or container.
  2. `.env.local` (keys and overrides; git-ignored).
  3. `WEBAGENT_*` environment variables (`WEBAGENT_SECURITY_*` override the YAML security block).
  4. The YAML file.
  5. Code defaults.

  `.env.local.example` lists every setting.
- **The three environments:**
  - **dev:** API key off, debug errors, Swagger on, triage collection on.
  - **uat:** prod-like, with collection on for the shadow week.
  - **prod:** keys required, Swagger off, the Docker sandbox required.

## Deployment

- **Docker image (`backend/Dockerfile`):**
  - A two-stage build on `python:3.12-slim` with the exact versions in `constraints.txt` and the CPU build of torch.
  - It runs as `WEBAGENT_ENV=prod` as a non-root user, with NLTK data and the MiniLM encoder baked in.
  - `GET /health` is its health check, and `WEB_CONCURRENCY` sets the number of uvicorn workers.
  - Run `git lfs pull` before building so the model files are real.
- **Docker Compose (`backend/docker-compose.yml`):** the backend, Prometheus and Grafana, on 127.0.0.1 only. `docs/RUNBOOK.md` shows how to run it as uat with its own data folder.
- **Kubernetes (`backend/kubernetes/`):**
  - One replica with Recreate updates, until `/app/data` is on a shared ReadWriteMany volume.
  - A read-only root filesystem; `emptyDir` for logs and `/tmp`; resource limits (3 GiB of memory).
  - The NetworkPolicy limits port 8000 to the ingress and monitoring namespaces, and the Ingress answers 404 for `/metrics`.
  - An optional backup CronJob.
- **Web app:** `next build` and `next start`, or Vercel (`VERCEL_DEPLOYMENT.md`). It needs `WEBAGENT_BACKEND_URL` and the same `WEBAGENT_API_KEY` as the backend.
- **Dependencies:**
  - CI audits both sides (`pip-audit`, `npm audit`), and a new advisory fails the build.
  - The accepted advisories, which need a major-version migration and can't be reached in WebAgent, are explained in `backend/SECURITY_DEPENDENCIES.md`.

## Recurring research (3.11.0; `app/scheduler.py`, `app/services/topics.py`, `app/api/endpoints/topics.py`)

A **topic** is a saved question that's researched again daily or weekly. The parts:

- **Topics API** (`/api/topics`, API key): create, list, pause, resume, Run now, delete, and read runs.
  - A topic's question passes checkpoint 1 when it's created.
  - Definitions live in `data/topics/<id>.json`, written only by the backend, under `flock` because it runs two workers.
- **Scheduler** (`python -m app.scheduler`: a Compose service, or a sidecar in the backend pod): one process that loads no models (about 40 MB).
  - **Each tick (60 s):**
    - it picks the due topics, oldest first, 2 at a time;
    - it checks the daily cap and the monthly allowance (a run starts only if its worst case fits);
    - it starts each run as `POST /api/task` with a fixed task ID, its own token (`X-WebAgent-Scheduler`) and a token budget.
  - **After each run** it records a run (report, sources, usage, the source diff against the last runs), the month's usage, counters, and holds (3 failures in a row, or the allowance spent).
  - **Its own files:** `state.json`, `runs/`, `usage-*.json` and `counters.json`.
  - **One instance:** a heartbeat lock makes sure only one runs; a restart waits for a dead holder's heartbeat to expire.
- **The task path for scheduled runs** (`origin: schedule`, accepted only with the scheduler token):
  - no triage and no triage collection; no embedding drift;
  - `origin` and `topic_id` on the result, the security collection, the task metrics and LangSmith;
  - `route: agentic`, `decided_by: schedule`.
- **Usage for every task:** each task records `usage` (input and output tokens, searches, Tavily credits). `max_searches_per_task` is a real per-task cap, held in the run's `task_budget` counter, which is shared by the parallel research agents.
- **Web app:**
  - **Track this** under research answers, prefilled with the supervisor's `standalone_query`;
  - `/topics` and `/topics/[id]` (the run timeline and stored reports);
  - `/api/topics/**` proxies, which allow-list routes and fields. The task proxy now forwards only its known fields.
- **Metrics:** the backend's `/metrics` exports the scheduler's numbers from its files: runs, changes, holds, topics by state, budget used and its limit, heartbeat, run time. There's a "Recurring research" row on the backend dashboard, and a "Scheduler down" alert.
- **Change summaries (3.12.0):**
  - After a run with a previous one, the scheduler calls `POST /api/topics/{id}/summarize` (scheduler only). One model call compares the two reports, with both scored by checkpoint 1 as untrusted `retrieved_doc`, under its own 8,000-token budget.
  - The answer is at most 5 bullets citing sources, or "No material change".
- **Email (3.12.0, `notify_email.py`, `notify_tokens.py`, `endpoints/notify.py`, `endpoints/topic_notify.py`):** only the backend sends.
  - **Transport:** SMTP (STARTTLS or TLS), or `log` in dev.
  - **Subscribing:** double opt-in. Confirm and unsubscribe are button presses that POST, because mail scanners open links, plus RFC 8058 one-click unsubscribe. Tokens are HMAC-signed and carry a hash of the address, not the address.
  - **Content:** plain text. Notices list only the domains of sources checkpoint 1 didn't flag. In shadow mode, its would-be drops are recorded as `security.shadow_flagged` for this. Summary emails only in act mode, enforced at startup.
  - **Caps:** per run, per address and per day. A ledger in `data/notify/` holds hashed addresses only.
  - **Logs:** credential query parameters (`token=`) are redacted in the request log, uvicorn's access log and httpx's log.
- **Spec:** `planning/specs/recurring-research/`.

## Limits and what isn't built

- Email is the only user notification, and carries only a notice until checkpoint 1 acts.
- No per-user accounts: one shared access code, and shared documents and topics.
- One server's memory holds the rate limits, and one data folder holds the state. Scaling past one machine needs shared storage for `data/` and a shared rate-limit store.
- No checkpointing of research runs.
- Images reach only direct answers.

## Where the work is going

The plans are in `planning/`:

- `improvement-plan.md` for the platform;
- `checkpoint2-upgrade-plan.md` for the intent checks;
- `observability-plan-3.6.0.md` for drift.

The next steps are a shadow week on 3.10.0, to measure the guard's false-positive rates and collect reviewed real calls. After that come switching the guard's rules from shadow to act one at a time, and retraining the intent classifier.
