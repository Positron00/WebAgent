# WebAgent Backend

The backend component of the WebAgent platform, built with FastAPI, LangGraph, and LangChain.

## Architecture

The backend runs the research agents, the security guard and triage, and holds the documents and task results. The full picture, including the web app, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The research workflow (`app/graph/workflows.py`):

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
```

### Core Components

1. **Supervisor**: plans the research (web, internal, document) and writes the standalone question and its sub-questions
2. **Web and Internal Research**: run in parallel; Tavily for the web, Chroma vectors plus BM25 for loaded documents
3. **Senior Research**: scores the findings and asks for further rounds, at most `agents.senior_research.max_follow_up_questions` (3 in dev and uat, 5 in prod)
4. **Document Extraction**: extracts from one document when that's all a task needs
5. **Team Manager**: writes the report, calling **Data Analysis** or the **Coding Assistant** (followed by **Coding Review**) when the question needs them
6. **Security guard** (`app/security/`): checkpoint 1 before every model call, checkpoint 2 before every tool call and plan subgoal; shadow mode in every environment today
7. **Triage** (`app/services/triage.py`): decides per question whether it needs the research agents

### Key Features

- **LangGraph Workflow**: Flexible agent orchestration with conditional routing
- **Document Processing**: Extract and analyze information from various document formats
- **Research Loop**: further research rounds until the senior research agent is satisfied, at most 3 (dev, uat) or 5 (prod)
- **API Integration**: FastAPI server with comprehensive endpoint documentation

### Diagnostics & Testing System

The WebAgent platform includes a comprehensive diagnostics and testing system to ensure reliability and proper functioning of all components:

- **Consolidated Diagnostics**: All diagnostics utilities are organized in the `backend/diagnostics` directory
- **System Health Checks**: Verify environment, dependencies, agents, workflow, and network connectivity
- **Performance Monitoring**: Track memory usage, CPU utilization, and execution times for all components
- **Automated Tests**: Comprehensive test suite covering all aspects of the platform
- **LangGraph Testing**: Specialized tests for the LangGraph workflow and agent interactions

#### Running Diagnostics and Tests

The following commands are available to run diagnostics and tests:

```bash
# Run all diagnostics checks
npm run diagnostics

# Run all tests
npm run tests

# Run both diagnostics and tests
npm run check

# Run directly from Python
conda activate agents
python -m backend.diagnostics.runners.run_all

# Additional options
python -m backend.diagnostics.runners.run_diagnostics --check-network --export-json=report.json
python -m backend.diagnostics.runners.run_tests --verbose --test-file=test_langgraph.py
```

The diagnostics can help identify issues with:
- Missing dependencies or environment variables
- LangGraph workflow configuration problems
- Agent initialization or communication issues
- Network connectivity to external services
- LLM availability and configuration

## Prerequisites

- Python 3.12 (the `agents` conda environment)
- Together AI API key (the default provider). An agent can use OpenAI or Anthropic instead with `provider:` in its config; nothing falls back automatically. Groq and Gemini also need their LangChain packages, which aren't installed by default
- Tavily API key (web research)
- LangSmith API key (optional, for tracing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Positron00/WebAgent.git
   cd WebAgent
   ```

2. Fetch the model weights (Git LFS) and install dependencies (exact versions in `backend/constraints.txt`):
   ```bash
   git lfs pull
   pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
   # from the repo root: runtime + dev tools with the constraints; the Docker image installs backend/requirements.txt only
   ```

3. Create `.env.local` in the repo root from the template and add your keys:
   ```bash
   cp .env.local.example .env.local
   ```
   At minimum set `TOGETHER_API_KEY` and `TAVILY_API_KEY`. See [Environment Variables](#environment-variables) below.

## Running the Application

### Development Server

From the repo root:

```bash
python -m backend.main
```

Set `WEBAGENT_API_DEBUG_MODE=true` to auto-reload when files under `backend/` change. A reload drops tasks that are still running.

Access the API documentation at http://localhost:8000/api/v1/docs

### Production Server

Run uvicorn with several workers (the Docker image does this, with `WEB_CONCURRENCY` workers):

```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Several workers are fine: a task's status is saved in `backend/data/results/` as soon as it is created, so any worker can answer a poll. Running on several machines needs that folder on shared storage. In uat and prod, run `python -m app.core.startup_checks` first (the image does), and set `PROMETHEUS_MULTIPROC_DIR` to an empty folder for the metrics.

### Running with Docker

Build and run the Docker container:

From the `backend/` folder:

```bash
# Build the image (the version comes from the repo-root package.json)
docker build --build-arg APP_VERSION=$(node -p "require('../package.json').version") -t webagent/backend .

# Run it as prod: API keys required, debug off
docker run -p 8000:8000 --env-file ../.env.local -e WEBAGENT_ENV=prod \
  -v webagent-data:/app/data webagent/backend
```

- The image sets `WEBAGENT_ENV=prod`. `.env.local` sets `WEBAGENT_ENV=dev`, and `--env-file` overrides the image, so pass `-e WEBAGENT_ENV=prod` after it. In prod the container stops at once with `Startup check failed: ...` if `WEBAGENT_SECURITY_SECRET_KEY` or `WEBAGENT_API_KEY` is missing (the check runs before uvicorn starts).
- `GET /health` is the container health check; it needs no API key.
- NLTK data and the MiniLM encoder are installed at build time. Task results, documents, drift and collection data live in `/app/data`; mount a volume there to keep them. Run `git lfs pull` before building, or the model folders hold pointer files.
- `WEB_CONCURRENCY` sets the number of uvicorn workers (default 4).

## Diagnostics and Monitoring

WebAgent includes built-in diagnostics for monitoring system health and performance:

```bash
# Run diagnostics from the command line
python -m app.utils.diagnostics

# Or use the detailed health endpoint
curl http://localhost:8000/api/v1/health/detailed
```

### Using the Example Script

The repository includes an example script for running various WebAgent tasks:

```bash
# Run diagnostics
python ../scripts/run_webagent.py --mode diagnostics

# Run a workflow
python ../scripts/run_webagent.py --mode workflow --query "Research quantum computing"

# Extract data from a document
python ../scripts/run_webagent.py --mode direct --type document_extraction --document path/to/document.pdf
```

## API Endpoints

The WebAgent backend exposes several API endpoints:

No API key needed:

- **GET /health**: Liveness/readiness probe (not rate-limited)
- **GET /api/v1/health/**: Basic health
- **GET /metrics** (also **/api/v1/metrics**): Prometheus metrics (not rate-limited)

Require `X-API-Key` when API-key auth is on (always in uat/prod):

- **POST /api/task**: Start a task (`{"query": "...", "mode": "off" | "auto" | "always"}`); returns a `task_id`. See [Agentic triage](#agentic-triage) for modes
- **GET /api/task/{task_id}**: Task status (`processing`, `completed`, `failed`) and report, with `route`, `decided_by` and (when the classifier ran) `p_agentic`. While a research run works, `progress` holds its current step and last log lines; a failed run has `error_code` (`timeout`, `budget`, `cancelled`, `escalated`, …); every finished report has `tokens_used`. Any worker can answer, and finished reports survive a restart
- **POST /api/task/{task_id}/cancel**: Stop a running task (202 `{"status": "cancelling"}`; 409 if it already finished, 404 if unknown). The worker running it stops within a few seconds and saves `error_code: cancelled`
- **POST /api/triage**: Decide whether a question needs the research agents (`{"query", "previous_user_question", "has_image"}` → `{"route", "p_agentic", "decided_by", "reason", "classifier_version"}`)
- **POST /api/v1/chat/completions**, **GET /api/v1/chat/status/{task_id}**: OpenAI-style chat request and its status; accepts `mode` like `/api/task`
- **POST /api/v1/chat/** and **POST /api/v1/tasks/create** (both deprecated; use `/api/task`): start a research run, through the same intake guard, limits and run supervisor as `/api/task` in Always mode. Only `context.document_id` is used from the request's context
- **GET /api/v1/chat/{task_id}**, **GET /api/v1/tasks/{task_id}** (plus `/result`, `/error`): a task's status and result
- **GET /api/v1/tasks/?limit=20&offset=0**: Saved tasks, newest first (`limit` 1–100); each shows its question's length, not its text. **DELETE /api/v1/tasks/{task_id}** deletes the task and its saved file
- **POST /api/v1/documents/upload** (415 for an unsupported file type), **POST /api/v1/documents/search**, **GET /api/v1/documents/document/{document_id}**, **GET /api/v1/documents/related/{document_id}**: Documents (IDs are UUIDs)
- **GET /api/v1/status**: LLM and service status
- **GET /api/v1/health/detailed**: Detailed service health
- **GET /api/v1/observability/metrics**, **/drift**, **/alerts**, **/health**; **POST /api/v1/observability/llm/reset_baseline**, **/doc/reset_baseline**: drift per detector, active alerts (`name`, `severity`, `service`, `message`, `since`), health from the alerts and model/document figures, all from the shared drift store (fields in [docs/api_reference.md](docs/api_reference.md); alerts and resets in [docs/RUNBOOK.md](../docs/RUNBOOK.md) "Drift and alerts")

Errors carry `"error": {"code": "...", "message": "..."}`; `HTTPException` and validation errors also keep FastAPI's `detail`. Intake-guard refusals (403, 409) keep their own flat shape, which the web app reads.

Requests are rate-limited per caller (60 per minute by default). Behind the Next.js proxy the caller is the user's sign-in session, sent as `X-WebAgent-Client`; the backend accepts that header only from `WEBAGENT_TRUSTED_PROXIES` or from a caller holding the API key. Other proxies' callers are taken from the right-most untrusted `X-Forwarded-For` address.

See the OpenAPI documentation for complete details and request/response schemas.

## Agentic triage

Triage decides whether a question gets one quick model answer (`direct`) or the research workflow (`agentic`). Rules run first (an image or a greeting → direct; "research …", "write a report …" → agentic), then a small local classifier: an `all-MiniLM-L6-v2` encoder fine-tuned on our routing data, plus logistic regression, about 5 ms per question on a laptop CPU once warmed up. When the classifier can't load or fails, the answer is `agentic` with `decided_by: fallback`. The design is in `planning/specs/agentic-triage/`.

### Modes on `/api/task` and `/api/v1/chat/completions`

| Request | Path |
|---|---|
| `"mode": "off"`, or `"agentic": false`, or **neither field** | Direct: one text-only model call with `previous_messages`, no workflow |
| `"mode": "always"`, or `"agentic": true` | Research workflow |
| `"mode": "auto"` | Classifier decides (same answer as `POST /api/triage`) |
| `"triage_override": true` | Research workflow, counted as an override (the web app's "Research this in depth") |

`mode` wins over the old flag (`agentic` on `/api/task`, `agentic_toggle` on chat completions). **Breaking change:** `/api/task` with neither field used to run the workflow; it now means `off`. The web app always sends `agentic: true` on this endpoint, so it is not affected. Images are not sent to the model on either backend path; results of a request with an image have `image_ignored: true`.

```bash
curl -X POST http://localhost:8000/api/task -H "X-API-Key: $WEBAGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What changed in EU AI rules this year?", "mode": "auto",
       "previous_messages": [{"role": "user", "content": "Tell me about AI regulation"}]}'

curl -X POST http://localhost:8000/api/triage -H "X-API-Key: $WEBAGENT_API_KEY" \
  -H "Content-Type: application/json" -d '{"query": "explain list comprehensions"}'
```

### Configuration

A `triage:` section in `config/{env}.yaml`; these environment variables override it:

| Variable | Meaning |
|---|---|
| `WEBAGENT_TRIAGE_ENABLED` | `false` skips triage: every `auto` decision is `agentic` (`decided_by: config`) and the model never loads |
| `WEBAGENT_TRIAGE_MODE` | `shadow` labels requests sent with `agentic: true` / `mode: always` without changing their route; `act` (default) doesn't. `/api/triage` and `mode: auto` always act |
| `WEBAGENT_TRIAGE_THRESHOLD` | Overrides the threshold stored in `models/triage/metadata.json` (0–1) |
| `WEBAGENT_TRIAGE_COLLECT` | `true` saves shadow decisions and override clicks, **with question text**, for review (on in dev, off in uat/prod) |
| `WEBAGENT_TRIAGE_LLM_TIEBREAK` | `true` lets one LLM call (3 s limit) settle borderline decisions. Off by default |

In uat and prod the server refuses to start unless `models/triage/classifier.joblib`, `metadata.json` and the encoder are on disk and `metadata.json` records a passed release gate (`test.passed: true`), or triage is disabled. The current version, `triage-2026-10-c`, passes the gate; `config/uat.yaml` and `config/prod.yaml` still set `enabled: false` until the shadow week and a human review of the training data. The fine-tuned encoder lives in `models/triage/encoder/` and is copied into the image with the code, so the server never downloads a model. Its weights are stored with Git LFS: run `git lfs pull` after cloning, before building the image. In dev, a missing model logs a warning and triage falls back to agentic.

Collected files are `data/triage/collected/YYYY-MM-DD.jsonl` (in a container, `/app/data/triage/collected/`; copy them out with `docker cp`). They are git-ignored and deleted after 30 days.

Metrics: `webagent_triage_decisions_total{route,decided_by}`, `webagent_triage_latency_seconds`, `webagent_triage_overrides_total`, `webagent_triage_uncertain_total`, shown in the Grafana dashboard's two triage panels.

### Training a new classifier version

Run from `backend/` (see `data/triage/README.md` for how the data is made):

```bash
python scripts/eval_triage.py --check-data              # validate train.jsonl and test.jsonl
python scripts/train_triage.py --version triage-2026-10-d --finetune   # fine-tune MiniLM (~1 min on an Apple GPU)
python scripts/eval_triage.py --write-metadata          # release gate; exit code 1 if it fails
python scripts/eval_triage.py --data data/triage/probe.jsonl   # second check in a different style
```

Without `--finetune` the encoder stays frozen and training takes seconds; metadata.json then names the base encoder instead of `encoder_path`.

Ship a version only if the gate passes: research recall ≥ 95% and routine questions routed direct ≥ 80% on `test.jsonl`.

## Testing

Run the test suite:

From the repo root, in the `agents` conda environment:

```bash
# Run all tests
python -m pytest backend/tests          # or: npm run test:backend

# Run one file
python -m pytest backend/tests/test_senior_research.py

# Coverage report
npm run coverage:backend
```

Notes:
- Tests run offline. `backend/tests/conftest.py` blocks every network connection except localhost, so a test that reaches a real LLM, Tavily or LangSmith fails with "Test tried to open a network connection". Fake the service instead.
- `conftest.py` also turns LangSmith tracing off, makes both `app.*` and `backend.*` imports work from any directory, and points saved task results and the model registry at temporary files, so a test run leaves the working tree unchanged.
- `backend/tests/test_workflow_state_flow.py` runs the real LangGraph workflow with stand-in agents, to check that each agent's report reaches the next step.

## Development

### Project Structure

```
backend/
├── app/                  # Main application package
│   ├── agents/           # Agent implementations
│   │   ├── base_agent.py             # Base agent class
│   │   ├── supervisor.py             # Supervisor agent
│   │   ├── web_research.py           # Web research agent
│   │   ├── internal_research.py      # Internal research agent
│   │   ├── senior_research.py        # Senior research agent
│   │   ├── document_extraction_agent.py # Document extraction
│   │   ├── data_analysis.py          # Data analysis agent
│   │   ├── coding_assistant.py       # Coding assistant agent
│   │   ├── coding_review.py          # Reviews the coding assistant's code
│   │   └── team_manager.py           # Team manager agent
│   ├── api/              # API endpoints
│   ├── core/             # Core functionality
│   │   ├── config.py               # Configuration loading
│   │   ├── logger.py               # Logging setup
│   │   ├── metrics.py              # Performance metrics
│   │   └── middleware.py           # FastAPI middleware
│   ├── graph/            # LangGraph workflows
│   │   └── workflows.py            # Agent workflow definitions
│   ├── models/           # Data models
│   │   ├── task.py                 # Task models and workflow state
│   │   ├── research_plan.py        # Research plan shared by supervisor, routers and research agents
│   │   └── chat.py                 # Chat models
│   ├── security/         # Security guard: checkpoints 1 and 2, ContextDefender, intent signals,
│   │                     #   Cedar decisions, ledger, judge, collection (policies in config/security/)
│   ├── services/         # External service integrations
│   │   ├── llm.py                  # LLM service client (every agent's model call)
│   │   ├── task_manager.py         # Runs workflows; task status
│   │   ├── run_supervisor.py       # Time limit, token budget, cancel, progress
│   │   ├── result_store.py         # Task status and results on disk (shared by workers)
│   │   ├── triage.py, triage_model.py  # Agentic triage: rules and the MiniLM classifier
│   │   ├── web_search.py           # Tavily
│   │   └── vectordb.py, graphdb.py, document_loader.py  # Documents and their stores
│   └── utils/            # Utility functions
│       ├── code_sandbox.py         # Runs generated code (local or docker)
│       ├── nltk_data.py            # NLTK data without startup downloads
│       └── diagnostics.py          # System diagnostics
├── config/               # dev/uat/prod YAML; security policies
├── models/               # ContextDefender, alignment classifier, triage (LFS); optional model services
├── scripts/              # Training, gates, dataset builders, benchmarks
├── sandbox/              # Docker image for generated code
├── tests/                # Test suite
├── main.py               # Application entry point
├── path_setup.py         # Path configuration utility
├── requirements.txt          # Runtime (server image)
├── requirements-dev.txt      # Tests, lint, dev sandbox libs
├── requirements-demos.txt    # Optional demo extras
└── requirements-llm.txt      # LLM-specific dependencies
```

### Adding a New Agent

1. Create a new file in `app/agents/` for your agent
2. Extend the `BaseAgent` class
3. Implement the required methods
4. Add a factory function to get your agent instance
5. Update the workflow in `app/graph/workflows.py`

Example:

```python
from app.agents.base_agent import BaseAgent

class MyNewAgent(BaseAgent):
    def __init__(self, config=None):
        super().__init__(name="my_new_agent", config=config)
        # Initialize your agent
        
    async def run(self, state):
        # Implement agent logic
        return updated_state

# Factory function
def get_my_new_agent():
    return MyNewAgent()
```

Things to know when writing an agent:
- Return the `WorkflowState` you were given. Editing it in place (for example `state.update_with_agent_output(...)` or `state.context[...] = ...`) is fine: `WorkflowState` reports every field to LangGraph, so changes carry to the next step.
- Wrap LangChain chains with `self.trace_chain(chain)` so the step shows up under the agent's name in LangSmith.
- Use `await chain.ainvoke(...)` inside `run()`. Blocking calls stall the whole server while they run.
- Add a test in `backend/tests/` with the model reply faked (see `test_coding_agents.py`).

### Environment Variables

Key environment variables used by the application:

| Variable | Description | Default |
|----------|-------------|---------|
| WEBAGENT_ENV | `dev`, `uat` or `prod`; picks `config/{env}.yaml`. A value set in the shell or container wins over `.env.local` | dev |
| WEBAGENT_API_DEBUG_MODE | Debug logging and auto-reload | false (dev.yaml turns it on) |
| WEBAGENT_LLM_DEFAULT_PROVIDER | First LLM provider to try (`together`, `openai`, `anthropic`, `groq`, `gemini`, `self`) | together |
| WEBAGENT_LLM_DEFAULT_MODEL | Default model | meta-models/Muse-Glimmer-30B |
| TOGETHER_API_KEY | Together AI key (required) | |
| TOGETHER_RATE_LIMIT_SECONDS | Minimum gap between Together calls | 2.0 |
| OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GEMINI_API_KEY | Fallback provider keys | |
| TAVILY_API_KEY | Web search key | |
| LANGSMITH_API_KEY | Turns on LangSmith tracing | |
| LANGSMITH_PROJECT_NAME | LangSmith project | webagent-research |
| LANGSMITH_TRACING_ENABLED | Set `false` to turn tracing off | true |
| WEBAGENT_API_KEY | Key clients send as `X-API-Key` (required in uat/prod) | |
| WEBAGENT_SECURITY_SECRET_KEY | Signs the security guard's consent and approval tokens (required in uat/prod) | insecure dev default |
| WEBAGENT_GUARD_PLAN_CHECK | Check the supervisor's rewrite and plan questions, and senior research's follow-up questions, against the user's own turns (checkpoint 2's intent rules are shadow-only in 3.7.0, so this only records) | true |
| WEBAGENT_SECURITY_API_KEY_ENABLED | Require `X-API-Key` | false (true in uat/prod YAML) |
| WEBAGENT_ENABLE_CODE_EXECUTION | Let agents run generated code in the API server | false |
| WEBAGENT_CODE_SANDBOX | `local` (no secrets, CPU/memory limits) or `docker` (isolated; required in uat/prod) | local |
| WEBAGENT_TRUSTED_PROXIES | Proxies whose `X-WebAgent-Client` (or right-most untrusted `X-Forwarded-For` address) is used for per-user rate limits | 127.0.0.1,::1 |
| WEBAGENT_ALLOW_OPEN_DEV | In dev with API keys off, allow listening on an address other than 127.0.0.1 | false |
| WEBAGENT_TASK_MAX_SECONDS | A research run is stopped after this long (`error_code: timeout`) | 420 |
| WEBAGENT_SEARCH_TIMEOUT_SECONDS | Each Tavily search (with its retry) gives up after this long | 30 |
| WEBAGENT_TASK_TIMEOUT_MINUTES | A task still processing after this long is reported as interrupted (a worker died); keep it above the run limit | 30 |
| WEBAGENT_NLTK_AUTO_DOWNLOAD | Download missing NLTK data on first use | true (false in the Docker image) |
| WEBAGENT_MODEL_REGISTRY_PATH | Model registry state file | backend/data/registry/models_registry.json |
| WEB_CONCURRENCY | Uvicorn workers in the Docker image (task status is shared through `data/results`) | 4 |
| WEBAGENT_ALLOW_FAKE_LLM | Placeholder reply when every provider fails | false |
| WEBAGENT_DATA_DIR | Data folder (vector DB, artifacts, results, drift samples) | backend/data |
| WEBAGENT_DRIFT_ENABLED | Record drift samples from real model calls and document loads, and compute drift and alerts | true |
| WEBAGENT_DRIFT_WINDOW | Reference size, and the most current samples compared with it (minimum 30) | 50 |
| WEBAGENT_DRIFT_INTERVAL_SECONDS | Seconds between drift and alert computations | 60 |
| WEBAGENT_TOKEN_ALERT_DAILY | `token_usage` alert above this many tokens in 24 hours (0 = off) | 0 |
| TOGETHER_API_BASE | Together API address; dev only, for `scripts/fake_together.py` | https://api.together.xyz/v1 |

`.env.local.example` in the repo root lists every setting with comments.

## Troubleshooting

### Common Issues

1. **API Keys**: Ensure your LLM provider API keys are correctly set in `.env.local` in the repo root
2. **Import Errors**: Run from the repo root (`python -m backend.main`) or use `path_setup.py`
3. **Workflow Errors**: Check the logs for details on agent errors; filter by the task's `[task <id>]` tag
4. **Together AI "Error 503" or "Error 429"**: Temporary; calls retry up to 3 times (2s, then 4s). If it persists, raise `TOGETHER_RATE_LIMIT_SECONDS`
5. **Server runs as dev though you set prod**: check `echo $WEBAGENT_ENV`. A value in the shell or container wins over `.env.local`; with `docker run --env-file`, add `-e WEBAGENT_ENV=prod` after it
6. **"Code execution in uat/prod requires the container sandbox"**: either leave `WEBAGENT_ENABLE_CODE_EXECUTION` off, or set `WEBAGENT_CODE_SANDBOX=docker` and build the sandbox image (see Generated code below)
7. **A task shows "Task was interrupted before it finished"**: the worker running it stopped (restart or crash) and the task was still processing after `WEBAGENT_TASK_TIMEOUT_MINUTES` (30). Run it again
8. **Log says "Together reply … stopped at the max_tokens limit"**: that agent's answer was cut off. Raise `max_tokens` for the agent in `config/{env}.yaml` (for example `agents.web_research.max_tokens`); without an agent setting, `llm.max_tokens` applies
9. **Agentic runs are slow**: open the task's LangSmith trace and compare each model call's completion tokens with its limit. A call that always ends exactly at the limit is running away; check that agent's prompt

### Diagnostic Checks

Run system diagnostics to identify issues:

```bash
python -m app.utils.diagnostics
```

This will check:
- System information
- Dependencies and versions
- API keys and environment configuration
- Agent availability
- Workflow configuration
- Performance metrics

### Logging

Logs go to the console and to `backend/logs/webagent-<env>.log` (rotated at 10 MB; daily in prod). uat and prod write JSON lines.

- The level is DEBUG when debug mode is on (`debug_mode` in the YAML config, or `WEBAGENT_API_DEBUG_MODE=true`), otherwise INFO.
- Every API request is logged with an ID and duration; the ID is also returned in the `X-Request-ID` header.
- While a task runs, every log line (agents, parallel steps, libraries) starts with `[task <first 8 chars of task id>]`. This comes from `app/core/log_context.py`.

### Saved results

Each task is written to `backend/data/results/<task_id>.json` when it is created (status `processing`) and again when it finishes (question, status, timings and the full report or error). Every worker reads this folder, so polls work with several workers. The folder is git-ignored.

The model registry (model services and their status) is kept in `backend/data/registry/models_registry.json`, also git-ignored; set `WEBAGENT_MODEL_REGISTRY_PATH` to move it.

### Generated code

The coding agents can run the Python they write. Where it runs depends on two settings:

- `WEBAGENT_ENABLE_CODE_EXECUTION`: off in the API server unless set to `true`; on for demos and scripts.
- `WEBAGENT_CODE_SANDBOX`:
  - `local` (default): a child process with no API keys or other secrets in its environment, and CPU, file-size and (on Linux) memory limits. Fine on your own machine; it is not isolation.
  - `docker`: a throwaway container with no network, a read-only filesystem, and memory, CPU and process limits. Build the image once with `docker build -t webagent/code-sandbox:latest backend/sandbox`. Required whenever code execution is on in uat/prod; the server won't start otherwise.

### Tracing

With `LANGSMITH_API_KEY` set, each task is one LangSmith trace in the project `webagent-research` (`LANGSMITH_PROJECT_NAME`). It covers every agent step and model call, including Together token counts. `app/core/setup_langsmith.py` sets both `LANGSMITH_*` and `LANGCHAIN_*` variables, so settings exported in your shell can't split a trace across projects. Turn tracing off with `LANGSMITH_TRACING_ENABLED=false`.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md). The project is licensed under MIT.

### Services

The backend includes several services for data management and infrastructure:

- **Vector Database Service**: Manages document storage and retrieval using ChromaDB
- **Document Ingestion Service**: Handles generation and ingestion of sample documents into the vector database. Provides unified methods for creating and accessing sample documents across the application.
- **Document Loader Service**: Processes and loads documents into the system
- **Graph Database Service**: Manages knowledge graph relationships between documents
- **LLM Service**: Provides access to large language models

#### Environment Variables

Vector database and data storage locations can be configured using environment variables:

```bash
# Data storage configuration
WEBAGENT_DATA_DIR=/path/to/backend/data             # Base directory for all data
WEBAGENT_VECTOR_DB_PATH=/path/to/backend/data/vectordb   # Vector database location
WEBAGENT_ARTIFACTS_PATH=/path/to/backend/data/artifacts  # Document artifacts location
```

## Environment Variables

The WebAgent backend uses several environment variables for configuration. These are loaded in the following order of priority:

1. `.env.local` - For API keys and sensitive information (highest priority). Exception: `WEBAGENT_ENV` set in the shell or container wins, so `WEBAGENT_ENV=prod` always selects prod
2. YAML configuration in `config/{env}.yaml` - For environment-specific settings
3. `.env` - For default fallback values (lowest priority)

### Environment Files

Two templates live in the repo root:

- `.env.local.example` - Every setting, including API keys. Copy to `.env.local` (git-ignored) and fill in your keys.
- `.env.example` - Non-secret defaults. Copy to `.env` only if you want to change them.

```bash
cp .env.local.example .env.local
```

See the [Environment Variables](#environment-variables) table above for the most important settings.

## Directory Structure

The backend is organized into the following key directories:

### Core Application
- `app/`: Main application code
  - `api/`: API endpoints and routes
  - `agents/`: Agent implementations
  - `core/`: Core utilities and configuration
  - `graph/`: LangGraph workflow definitions
  - `models/`: Data models and schemas
  - `services/`: Service implementations
  - `utils/`: Utility functions

### Testing, Demos, and Diagnostics
- `tests/`: All test code
  - `unit/`: Unit tests for individual components
  - `diagnostics/`: Tests for diagnostic tools
  - `integration/`: Integration tests
  - `mock_utils.py`: Test utilities and mocks

- `demos/`: Demo applications and examples
  - Contains standalone demos for each WebAgent feature
  - Run with `python -m backend.demos.[demo_name]`

- `diagnostics/`: System diagnostics and monitoring
  - `runners/`: Scripts for running diagnostics and tests
  - `system_check.py`: Basic system health check
  - `core.py`: Core diagnostic functionality

### Configuration and Data
- `config/`: Configuration files
- `data/`: Data storage (`results/`, `vectordb/`, `drift/`, `security/collected/`, `triage/collected/`, `mlruns/` are generated and git-ignored; `sample_docs/`, `security/` and `triage/` training data are tracked, partly through LFS)
- `sandbox/`: Docker image for running generated code (`WEBAGENT_CODE_SANDBOX=docker`)
- `kubernetes/`: Deployment manifests and `deploy.sh`

### Documentation
- `docs/`: Documentation files 