# WebAgent Platform

[![version](https://img.shields.io/badge/version-3.11.0-blue.svg)](https://semver.org)
[![last-updated](https://img.shields.io/badge/last%20updated-2026--10--18-lightgrey.svg)](https://github.com/Positron00/WebAgent)

A multi-agent research assistant. Ask a question in the chat. WebAgent then either answers it directly in a few seconds, or sends it to a team of research agents. The agents search the web and your own documents, check each other's work, and return a sourced report in a few minutes. A security guard checks everything the models read and every tool call they make.

WebAgent is non-commercial and licensed under MIT.

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Using WebAgent](#using-webagent)
- [Setting up and running](#setting-up-and-running)
- [Deploying](#deploying)
- [Testing and diagnostics](#testing-and-diagnostics)
- [Troubleshooting](#troubleshooting)
- [What WebAgent doesn't do](#what-webagent-doesnt-do)
- [Documentation map](#documentation-map)

## What it does

- **Answers or researches, per question.** The **Agentic** control picks the path: **Off** always answers directly, **Always** always researches, and **Auto** lets a local classifier decide in about 10 ms. Each answer is labelled "Answered directly" or "Deep research".
- **Researches in rounds.** A supervisor plans the research. Web and internal research run in parallel. A senior research agent scores the findings and asks for further rounds until they are good enough, or the round limit is reached: 3 in dev and uat, 5 in prod. A team manager then writes the report, calling a data analyst or a coding assistant when the question needs one. Model-written code runs in a sandbox, never in the server.
- **Tracks topics.** **Track this** under a research answer turns its question into a topic that WebAgent researches again daily or weekly. Each run's report is kept with the sources that are new since the last runs. A small scheduler starts the runs through the normal task path; it's off until an operator turns it on, and a monthly allowance caps what scheduled runs can spend.
- **Uses your documents.** Documents (PDF, Word, text, Markdown, HTML, CSV, JSON) loaded through the API or the batch loader are chunked into a vector store. Their relationships go into a graph store, and internal research searches them with semantic and keyword (BM25) retrieval.
- **Guards the models.**
  - Checkpoint 1 scores every piece of text before a model reads it: the question, earlier turns, web pages, documents and code output. It drops injected web content and declines harmful requests.
  - Checkpoint 2 checks every tool call and every agent-written subgoal against what the user actually asked.
  - Both run local classifiers and Cedar policies inside the server. They currently run in **shadow mode**: they record what they would do and block nothing.
- **Stays observable.**
  - Prometheus metrics, and drift detection on real traffic (model speed and tokens, embeddings, documents) with six alerts.
  - LangSmith traces of every run.
  - Every log line written during a task carries the task's ID.
  - Saved reports are kept for 30 days.
- **Is bounded.** A research run stops after 7 minutes, on **Stop**, or when it reaches its token budget (off by default). Temporary provider errors are retried. Outside dev, question text never reaches the logs.

## Architecture

```text
  Browser ── sign in with the access code, chat, Agentic: Off · Auto · Always
     │
     ▼
┌─ Web app ─ Next.js 15 · React 19 ─ port 3000 ───────────────────────────────────────┐
│  middleware: signed session cookie on every page and API route                      │
│  /api/chat ──── direct answer: guard check (checkpoint 1) ──► Together AI           │
│  /api/triage, /api/task ── proxy to the backend, adding X-API-Key and the session ─┐ │
└─────────────────────────────────────────────────────────────────────────────────────┼─┘
                                                                                      ▼
┌─ Backend ─ FastAPI ─ port 8000 ─────────────────────────────────────────────────────────┐
│  API key · rate limit per session · size limit · one error format                       │
│                                                                                         │
│  Triage: rules, then a MiniLM classifier ──► direct: one model call                     │
│                                          └─► research: run supervisor (7 min, budget)   │
│                                                 │                                       │
│                                                 ▼                                       │
│                                   LangGraph research workflow (next diagram)            │
│                                                                                         │
│  Security guard, in process: Cedar policies + local classifiers, shadow mode            │
│    checkpoint 1: every text before a model  ── ContextDefender (injection, harm,        │
│                                                sensitive data, jailbreak)               │
│    checkpoint 2: every tool call and subgoal ── intent alignment, risk above intent,    │
│                                                private-text egress, drift              │
│                                                                                         │
│  data/: results/ (task reports) · vectordb/ (Chroma) · graph (Neo4j or NetworkX)        │
│         artifacts/ (document copies) · drift/ · security/ and triage/ collection        │
│  Observability: /metrics · /api/v1/observability (drift, alerts) · logs · LangSmith     │
│  Topics API · data/topics/ ◄── Scheduler (separate process): due runs → /api/task       │
│    (scheduler token; topics, schedule state, run history, monthly allowance)            │
└───────┬───────────────────────┬───────────────────────┬─────────────────────────────────┘
        ▼                       ▼                       ▼
   Together AI             Tavily search          Prometheus + Grafana
   (OpenAI if set)         (web research)         (Compose or Kubernetes), LangSmith
```

The research workflow, as built in `backend/app/graph/workflows.py`:

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

Every model call in the workflow passes checkpoint 1. Every web search, internal retrieval, code execution and plan subgoal passes checkpoint 2. Details: [backend/docs/ARCHITECTURE.md](backend/docs/ARCHITECTURE.md).

**Optional, separate:** `backend/models/` is a gateway-and-registry toolkit for hosting your own transformer models. The agent workflow doesn't use it.

## Using WebAgent

### In the web app

1. **Sign in.** If the operator set an access code, `/login` asks for it once per browser session.
2. **Choose a mode** from the **Agentic** menu in the chat toolbar (it shows the current one, for example "Agentic · Auto"):
   - **Off**: answers directly through Together AI in seconds. The backend isn't needed except for the guard check.
   - **Auto**: the backend's triage decides per question. Routine questions get a direct answer, and research questions go to the agents. Hover an answer's label to see why.
   - **Always**: every question goes to the research agents. A run usually takes 2.5–3.5 minutes.

   A mode you pick is remembered in the browser. New users start on `WEBAGENT_DEFAULT_AGENTIC_MODE` (`off` if unset).
3. **Ask.** Follow-up questions ("what about Europe?") are researched as complete questions, using the conversation so far.
4. **While research runs**, the chat shows the current step. **Stop** ends the run.
5. **Read the report.** With source citations on (the citations button), sources appear in a side panel; code blocks appear in a code panel. Under a direct answer in Auto, **Research this in depth** sends the same question to the agents and adds their report below.
6. **If research fails**, the chat says why, for example "took too long", "budget reached" or "stopped by the security guard". It offers **Try again** or **Answer directly instead**; it never switches to a plain answer on its own.
7. **Security messages.** In act mode, a request the guard refuses is declined with a reason, and a borderline one asks you to confirm first in the consent dialog. Under a research answer, a short note says when the guard left out web results or skipped a search or retrieval. **Approve** runs the research again with the skipped action allowed and adds the new answer below. In shadow mode, the current setting, none of this appears.
8. **Other controls:**
   - attach an image or take a screenshot (images go to direct answers only);
   - **AI style** (balanced, creative, precise, helpful) and **Knowledge focus** (general, medical, legal, science);
   - appearance and high-contrast settings;
   - **New chat**.

### Tracking a topic

1. Under a research answer, choose **Track this**. The form fills in the question the research used.
2. Pick daily or weekly, a day and time (in your browser's time zone), and whether to notify on every run or only when something changes. Notifications by email come in 3.12.0.
3. **Topics** (the clock link in the chat toolbar) lists every topic with its state, next and last run. Each topic's page shows its runs, newest first, with the new sources and the full stored report.
4. From the list:
   - **Run now** queues an extra run.
   - **Pause** stops the schedule.
   - **Resume** restarts it, and also clears a hold after three failed runs.
   - **Delete** removes the topic and everything stored for it.

Topics are shared by everyone using the server, like loaded documents. An operator has to turn scheduling on first ([docs/RUNBOOK.md](docs/RUNBOOK.md#recurring-research)).

### Adding your own documents

Internal research searches documents you load. The web app has no document upload, so load them with the batch loader or the API. They are shared by every user of the server, so don't load private documents on a shared instance.

```bash
# A folder of PDFs, or one file with metadata
python scripts/batch_document_loader.py --dir /path/to/documents --pattern "*.pdf"
python scripts/batch_document_loader.py --file report.pdf --category reports --tags "financial,quarterly"
```

### From your own code (HTTP API)

Every endpoint except the health checks and `/metrics` needs `X-API-Key` when API keys are on. They're always on in uat and prod; in dev, set `WEBAGENT_SECURITY_API_KEY_ENABLED=true`.

```bash
KEY=...   # the backend's WEBAGENT_API_KEY

# Start a research task (mode: off | auto | always) and poll it
curl -s -X POST localhost:8000/api/task -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query": "What changed in EU AI rules this year?", "mode": "always"}'
curl -s localhost:8000/api/task/<task_id> -H "X-API-Key: $KEY"      # processing → completed | failed
curl -s -X POST localhost:8000/api/task/<task_id>/cancel -H "X-API-Key: $KEY"

# Topics: create one, queue a run, read its runs
curl -s -X POST localhost:8000/api/topics -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"name": "EU AI rules", "query": "What changed in EU AI rules this week?", "schedule": {"every": "week", "weekday": "mon", "time": "07:30", "tz": "Europe/Paris"}}'
curl -s -X POST localhost:8000/api/topics/<topic_id>/run -H "X-API-Key: $KEY"
curl -s localhost:8000/api/topics/<topic_id>/runs -H "X-API-Key: $KEY"

# Would this question go to the research agents?
curl -s -X POST localhost:8000/api/triage -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query": "Compare the latest open-weight LLM benchmarks"}'

# Documents
curl -s -X POST localhost:8000/api/v1/documents/upload -H "X-API-Key: $KEY" \
  -F "file=@report.pdf" -F 'metadata_json={"category": "reports"}'
curl -s -X POST localhost:8000/api/v1/documents/search -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query": "quarterly revenue", "limit": 5, "include_content": true}'
```

A finished task's `result` holds the report, `route` (`direct` or `agentic`), `decided_by` and `tokens_used`. Every endpoint, field and error code is in [backend/docs/api_reference.md](backend/docs/api_reference.md), and the dev server serves Swagger at `http://localhost:8000/api/v1/docs`.

### Reading what happened

- **Saved reports:** `backend/data/results/<task_id>.json` holds each task's question, status, timings and report, written when it starts and again when it ends. They are deleted after `WEBAGENT_RESULTS_RETENTION_DAYS` (30).
- **Logs:** `backend/logs/webagent-<env>.log`. Follow one task with `grep "\[task 9e6a4965\]"`, using the first 8 characters of its ID.
- **Traces:** with `LANGSMITH_API_KEY` set, each task is one trace in the LangSmith project `webagent-research` (`LANGSMITH_PROJECT_NAME`), with every agent step and token count. `LANGSMITH_TRACING_ENABLED=false` turns tracing off.
- **Metrics, drift and alerts:** `/metrics` for Prometheus. `/api/v1/observability/{drift,alerts,health}` need the API key. With the Compose stack, the Grafana dashboards are at `http://localhost:3001`. See [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Setting up and running

### Prerequisites

- Python 3.12, Node.js 18.18 or later, Git with Git LFS (the security and triage models are LFS files)
- A Together AI API key (required) and a Tavily API key (for web research). OpenAI is optional.
- Docker, for the Compose stack, the image, and the code sandbox in uat and prod

### Install

```bash
git clone https://github.com/Positron00/WebAgent.git && cd WebAgent
git lfs pull                                   # model weights

conda create -n agents python=3.12 && conda activate agents
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
npm install

cp .env.local.example .env.local               # then set TOGETHER_API_KEY and TAVILY_API_KEY
```

- **Python dependencies:** `requirements.txt` installs the backend with the exact versions in `backend/constraints.txt`. The CPU index keeps torch to its CPU build.
- **`.env.local`:** git-ignored, read by both the backend and the web app, and lists every setting with its default.
- **Environment settings:** agents, models and limits live in `backend/config/{dev,uat,prod}.yaml`, chosen by `WEBAGENT_ENV`. Each agent under `agents:` can set its own `model` and `max_tokens`.
- **Sign-in:** to require it, set `WEBAGENT_ACCESS_CODE` (at least 12 characters) and `WEBAGENT_SESSION_SECRET` (at least 32 random characters, e.g. `openssl rand -hex 32`). In development the web app stays open without them, with a warning. **A production build answers 503 on every API route until both are set**, unless `WEBAGENT_ACCESS_CODE_DISABLED=true`.

### Run (development)

```bash
python -m backend.main        # backend on http://localhost:8000
npm run dev                   # web app on http://localhost:3000 (add -- -p 3002 if 3000 is taken)
```

| What | Where |
| --- | --- |
| Web app | <http://localhost:3000> |
| API | <http://localhost:8000/api/v1> and <http://localhost:8000/api/task> |
| Swagger (dev only) | <http://localhost:8000/api/v1/docs> |
| Health | <http://localhost:8000/health> |

**Local stack with monitoring.** Add `GRAFANA_ADMIN_PASSWORD` to `.env.local` and run `npm run stack:up`. That starts the backend, Prometheus (`:9090`) and Grafana (`:3001`) in Docker, all on 127.0.0.1. To run it as uat, for example for a shadow week, see [docs/RUNBOOK.md](docs/RUNBOOK.md#local-uat-with-docker-compose).

## Deploying

- **Docker:** `docker build --build-arg APP_VERSION=3.11.0 -t webagent/backend:3.11.0 backend/`. The image runs as `prod` and as a non-root user, and starts only with `WEBAGENT_SECURITY_SECRET_KEY` and `WEBAGENT_API_KEY` set. Run `git lfs pull` first. See [backend/README.md](backend/README.md#running-with-docker).
- **Kubernetes:** see [backend/kubernetes/README.md](backend/kubernetes/README.md). `deploy.sh` needs `WEBAGENT_SECURITY_SECRET_KEY`, `WEBAGENT_API_KEY` and `GRAFANA_ADMIN_PASSWORD`.
- **Vercel** (web app only): see [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md). Agentic modes need `WEBAGENT_BACKEND_URL` pointing at a running backend.
- **Operating it:** deploys, rollback, key rotation, backups, spending caps, alerts and the security guard's modes are covered in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Testing and diagnostics

```bash
npm test                      # web app (Jest); npm run test:coverage enforces the coverage floor
npm run check-types && npm run lint
npm run test:backend          # backend (pytest, in the agents conda env)
npm run coverage:backend
npm run lint:backend          # ruff
python -m backend.diagnostics.runners.run_all     # system diagnostics
```

- **Offline tests:** backend tests fail if one opens a network connection, and they never trace to LangSmith.
- **Model tests:** tests that need the real model weights are marked `needs_lfs`. CI skips them, and the weekly `models` workflow runs them.
- **Dependency audits:** CI also audits dependencies (`pip-audit`, `npm audit`), and a new advisory fails the build. See [backend/SECURITY_DEPENDENCIES.md](backend/SECURITY_DEPENDENCIES.md).
- **More:** [docs/TESTING.md](docs/TESTING.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Troubleshooting

1. **An agentic answer says the research failed, or the backend can't be reached.** Check the backend is running at `WEBAGENT_BACKEND_URL` (default `http://127.0.0.1:8000`) with the same `WEBAGENT_API_KEY` as the web app, then read the backend log.
2. **"Unable to access non-serverless model" (HTTP 400).** Together no longer serves that model on demand. Set `WEBAGENT_LLM_DEFAULT_MODEL` to a serverless model (default `meta-models/Muse-Glimmer-30B`).
3. **The backend runs as `dev` although you asked for `prod`.** A `WEBAGENT_ENV` in your shell or container wins over `.env.local`. With `docker run --env-file .env.local`, add `-e WEBAGENT_ENV=prod` after it.
4. **The server refuses to start in uat or prod.** The startup checks name what's missing: a secret key, an API key, the model files (run `git lfs pull`), or a security classifier that hasn't passed its gate and isn't installed shadow-only.
5. **"The research took too long and was stopped."** The backend stops a run after `WEBAGENT_TASK_MAX_SECONDS` (420 s). In the task's LangSmith trace, look for a model call that stops exactly at its `max_tokens` limit, which the log also reports as "stopped at the max_tokens limit". Adjust that agent's `max_tokens` in `backend/config/{env}.yaml`.
6. **LangSmith steps stuck as "pending", or traces in another project.** Usually a `LANGSMITH_PROJECT` exported in your shell; the backend overrides it, so restart the backend.
7. **A backend test fails with "Test tried to open a network connection".** The test reached a real service (LLM, Tavily, LangSmith, Hugging Face, NLTK). Fake it instead; see [docs/TESTING.md](docs/TESTING.md).

## What WebAgent doesn't do

- **No email or other notifications yet.** Topics are researched on schedule and their history is kept (3.11.0), but email arrives in 3.12.0. The operator alerts are in Grafana (drift, errors, rate limits, token use, scheduler down).
- **No per-user accounts or private spaces.** The access code is shared, and loaded documents and topics are shared by every user of the server.
- **No document upload in the web app.** Use the API or the batch loader.
- **Images are used only by direct answers,** not by the research agents.
- **Research progress isn't kept across restarts.** A run in progress when the server restarts is lost and later reported as interrupted (after `WEBAGENT_TASK_TIMEOUT_MINUTES`, 30). Finished reports are kept.

## Documentation map

| Document | For |
| --- | --- |
| [backend/docs/ARCHITECTURE.md](backend/docs/ARCHITECTURE.md) | How the parts fit: web app, backend, workflow, security guard, triage, stores, observability |
| [backend/README.md](backend/README.md) | Backend setup, configuration, project layout, adding an agent |
| [backend/docs/api_reference.md](backend/docs/api_reference.md) | Every endpoint, field and error code |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Deploying and operating: rollback, keys, backups, alerts, the security guard |
| [backend/LOCAL_DEVELOPMENT.md](backend/LOCAL_DEVELOPMENT.md) | Local development details |
| [docs/TESTING.md](docs/TESTING.md), [CONTRIBUTING.md](CONTRIBUTING.md) | Tests, CI and contributing |
| [SECURITY.md](SECURITY.md), [backend/SECURITY_DEPENDENCIES.md](backend/SECURITY_DEPENDENCIES.md) | Reporting a vulnerability; dependency advisories |
| [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md), [backend/kubernetes/README.md](backend/kubernetes/README.md) | Deploying the web app and the backend |
| `planning/specs/` | Designs and requirements for each feature (security guard, triage, observability, platform) |

## Acknowledgments

LangChain and LangGraph, FastAPI, Next.js and React, Together AI, Tavily, Cedar, and the authors of the public datasets the security classifiers are trained on (listed in each model's data README).
