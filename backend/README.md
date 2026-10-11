# WebAgent Backend

The backend component of the WebAgent platform, built with FastAPI, LangGraph, and LangChain.

## Architecture

The WebAgent backend is built on a multi-agent architecture that leverages LangGraph for workflow orchestration. The system consists of several specialized agents that work together to process complex research tasks:

```
┌─────────────────────────────────────────────────────┐
│                  LangGraph Workflow                 │
│                                                     │
│  ┌─────────┐   ┌────────────┐   ┌────────────────┐  │
│  │Supervisor│──►│ Research   │──►│ Senior        │  │
│  │ Agent    │   │ Agents     │   │ Research      │  │
│  └─────────┘   └────────────┘   │ Agent         │  │
│       │             ▲           └───────┬────────┘  │
│       │             │                   │           │
│       ▼             └───────────────────┘           │
│  ┌─────────────┐            │                       │
│  │ Document    │            ▼                       │
│  │ Extraction  │    ┌────────────────┐              │
│  │ Agent       │    │ Specialized    │              │
│  └─────────────┘    │ Agents         │              │
│       │             └───────┬────────┘              │
│       │                     │                       │
│       └─────────┐    ┌─────┘                       │
│                 ▼    ▼                              │
│          ┌───────────────────┐                      │
│          │ Team Manager      │                      │
│          │ Agent             │                      │
│          └───────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

### Core Components

1. **Supervisor Agent**: The entry point for all requests, analyzes user queries and orchestrates the workflow
2. **Research Agents**: Gather information from the web and internal knowledge sources
3. **Senior Research Agent**: Evaluates research quality and can request additional research in a feedback loop
4. **Document Extraction Agent**: Processes documents to extract structured information
5. **Specialized Agents**: Handle specific tasks like data analysis and code generation
6. **Team Manager Agent**: Synthesizes outputs from multiple agents into a final response

### Key Features

- **LangGraph Workflow**: Flexible agent orchestration with conditional routing
- **Document Processing**: Extract and analyze information from various document formats
- **Research Loop**: Iterative research capabilities with up to 3 feedback loops
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
- Together AI API key (default LLM provider); OpenAI, Anthropic, Groq or Gemini keys are optional fallbacks
- Tavily API key (web research)
- LangSmith API key (optional, for tracing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Positron00/WebAgent.git
   cd WebAgent
   ```

2. Install dependencies (pinned by `backend/constraints.txt`):
   ```bash
   pip install -r backend/requirements.txt -c backend/constraints.txt
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

For production deployments, use Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
```

Several workers are fine: a task's status is saved in `backend/data/results/` as soon as it is created, so any worker can answer a poll. Running on several machines needs that folder on shared storage.

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

- The image sets `WEBAGENT_ENV=prod`. `.env.local` sets `WEBAGENT_ENV=dev`, and `--env-file` overrides the image, so pass `-e WEBAGENT_ENV=prod` after it. In prod the server refuses to start without `WEBAGENT_SECURITY_SECRET_KEY` and `WEBAGENT_API_KEY`.
- `GET /health` is the container health check; it needs no API key.
- NLTK data is installed at build time. Task results, the model registry and the Hugging Face cache live in `/app/data`; mount a volume there to keep them.
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
- **GET /api/v1/health/**, **GET /api/v1/health/detailed**: Health and detailed diagnostics
- **GET /metrics** (also **/api/v1/metrics**): Prometheus metrics (not rate-limited)

Require `X-API-Key` when API-key auth is on (always in uat/prod):

- **POST /api/task**: Start a multi-agent task from the web app (`{"query": "...", "agentic": true}`); returns a `task_id`
- **GET /api/task/{task_id}**: Task status (`processing`, `completed`, `failed`) and report. Any worker can answer, and finished reports survive a restart
- **POST /api/v1/chat/completions**, **GET /api/v1/chat/status/{task_id}**: OpenAI-style chat request and its status
- **POST /api/v1/chat/**, **GET /api/v1/chat/{task_id}**: Start a chat workflow and read its result
- **POST /api/v1/tasks/create**, **GET /api/v1/tasks/**, **GET /api/v1/tasks/{task_id}** (plus `/result`, `/error`), **DELETE /api/v1/tasks/{task_id}**: Task management
- **POST /api/v1/documents/upload**, **POST /api/v1/documents/search**, **GET /api/v1/documents/document/{document_id}**, **GET /api/v1/documents/related/{document_id}**: Documents
- **GET /api/v1/status**: LLM and service status

Requests are rate-limited per caller (60 per minute by default). Behind the Next.js proxy the caller is the end user, taken from `X-Forwarded-For`, which is trusted only from `WEBAGENT_TRUSTED_PROXIES`.

See the OpenAPI documentation for complete details and request/response schemas.

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
│   ├── services/         # External service integrations
│   │   ├── llm.py                  # LLM service client
│   │   ├── task_manager.py         # Runs workflows; task status
│   │   └── result_store.py         # Task status and results on disk (shared by workers)
│   └── utils/            # Utility functions
│       ├── code_sandbox.py         # Runs generated code (local or docker)
│       ├── nltk_data.py            # NLTK data without startup downloads
│       └── diagnostics.py          # System diagnostics
├── sandbox/              # Docker image for generated code
├── tests/                # Test suite
├── main.py               # Application entry point
├── path_setup.py         # Path configuration utility
├── requirements.txt      # Dependencies
└── requirements-llm.txt  # LLM-specific dependencies
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
| WEBAGENT_SECURITY_SECRET_KEY | JWT signing secret (required in uat/prod) | insecure dev default |
| WEBAGENT_SECURITY_API_KEY_ENABLED | Require `X-API-Key` | false (true in uat/prod YAML) |
| WEBAGENT_ENABLE_CODE_EXECUTION | Let agents run generated code in the API server | false |
| WEBAGENT_CODE_SANDBOX | `local` (no secrets, CPU/memory limits) or `docker` (isolated; required in uat/prod) | local |
| WEBAGENT_TRUSTED_PROXIES | Proxies whose `X-Forwarded-For` is used for per-user rate limits | 127.0.0.1,::1 |
| WEBAGENT_TASK_TIMEOUT_MINUTES | A task still processing after this long is reported as interrupted | 30 |
| WEBAGENT_NLTK_AUTO_DOWNLOAD | Download missing NLTK data on first use | true (false in the Docker image) |
| WEBAGENT_MODEL_REGISTRY_PATH | Model registry state file | backend/data/registry/models_registry.json |
| WEB_CONCURRENCY | Uvicorn workers in the Docker image (task status is shared through `data/results`) | 4 |
| WEBAGENT_ALLOW_FAKE_LLM | Placeholder reply when every provider fails | false |
| WEBAGENT_DATA_DIR | Data folder (vector DB, artifacts, results) | backend/data |

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

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License.

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
- `data/`: Data storage and processing (`results/`, `registry/`, `vectordb/` are generated and git-ignored; `sample_docs/` is tracked)
- `sandbox/`: Docker image for running generated code (`WEBAGENT_CODE_SANDBOX=docker`)
- `kubernetes/`: Deployment manifests and `deploy.sh`

### Documentation
- `docs/`: Documentation files 