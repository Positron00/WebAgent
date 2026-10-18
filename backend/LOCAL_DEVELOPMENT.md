# WebAgent Backend - Local Development

This guide provides instructions for setting up and running the WebAgent backend locally for development purposes.

## Prerequisites

- Python 3.12 (conda environment `agents`)
- Git
- Docker and Docker Compose (optional, for Option 2)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Positron00/WebAgent.git
cd WebAgent
```

2. Create and activate the conda environment:

```bash
conda create -n agents python=3.12
conda activate agents
```

3. Fetch the model weights and install dependencies:

```bash
git lfs pull     # ContextDefender, the alignment classifier and triage are LFS files
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

`requirements.txt` in the repo root adds the dev tools and the exact versions in `backend/constraints.txt`. The CPU index keeps torch to its CPU build.

4. Set up environment variables in the repo root:

```bash
cp .env.local.example .env.local
```

Edit `.env.local` and set at least:

```
TOGETHER_API_KEY=your-together-api-key
TAVILY_API_KEY=your-tavily-api-key
LANGSMITH_API_KEY=your-langsmith-api-key       # optional, for tracing
```

For uat/prod you also need `WEBAGENT_API_KEY` and `WEBAGENT_SECURITY_SECRET_KEY`.

## Running Locally

### Option 1: Running with Python

1. Start the application from the repo root:

```bash
python -m backend.main
```

2. Access the API at http://localhost:8000

3. Access the API documentation at http://localhost:8000/api/v1/docs

4. To use the web app as well, run `npm install && npm run dev` in the repo root. It runs on http://localhost:3000.

### Option 2: Running with Docker Compose

1. Add a Grafana admin password to `.env.local` in the repo root, next to your API keys. Compose reads all its secrets from that file:

```bash
GRAFANA_ADMIN_PASSWORD=choose-a-password   # e.g. from: openssl rand -base64 24
```

Put it on its own line. If you append it from the shell, start with a newline (`printf '\nGRAFANA_ADMIN_PASSWORD=...\n' >> .env.local`), since the file may not end with one. Compose stops with "required variable GRAFANA_ADMIN_PASSWORD is missing" if the password isn't set. The backend uses the `WEBAGENT_ENV` from `.env.local` (normally `dev`); `uat` and `prod` also need `WEBAGENT_SECURITY_SECRET_KEY` and `WEBAGENT_API_KEY`.

2. Start the stack from the repo root:

```bash
npm run stack:up     # docker compose -f backend/docker-compose.yml --env-file .env.local up --build
npm run stack:down   # stop it
```

`stack:up` also tags the backend image with the version from `package.json`. To run the stack as uat with its own data folder (for example for a shadow week), see "Local uat with Docker Compose" in [docs/RUNBOOK.md](../docs/RUNBOOK.md).

This will start:
- WebAgent backend on port 8000
- Prometheus on port 9090
- Grafana on port 3001 (set `GRAFANA_HOST_PORT` in `.env.local` to change it)

3. Access the services:
   - WebAgent API: http://localhost:8000 (health check: http://localhost:8000/health)
   - API Documentation: http://localhost:8000/api/v1/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3001 (login with `admin` and the `GRAFANA_ADMIN_PASSWORD` from `.env.local`)

## Development Workflow

1. Make changes to the code
2. The server reloads automatically when debug mode is on (`debug_mode` in `config/dev.yaml`, or `WEBAGENT_API_DEBUG_MODE=true`). It only watches `backend/`, and a reload drops running tasks.
3. Run tests to ensure your changes don't break existing functionality

## Running Tests

From the repo root:

```bash
python -m pytest backend/tests              # all tests
python -m pytest backend/tests/unit         # unit tests
python -m pytest backend/tests/integration  # integration tests
python -m pytest backend/tests -m needs_lfs # tests that load the real model weights
npm run coverage:backend                    # with a coverage report
```

Tests run offline: `backend/tests/conftest.py` blocks network connections, turns LangSmith tracing off, and sends saved results and the model registry to temporary files. See [docs/TESTING.md](../docs/TESTING.md).

## Monitoring

The local development setup includes Prometheus and Grafana for monitoring:

- Prometheus collects metrics from the WebAgent backend
- Grafana visualizes these metrics with pre-configured dashboards

### Grafana Dashboards

- **WebAgent Backend Dashboard:** HTTP request rates by status code, request duration by endpoint, LLM request rates and durations, token use, task completion rates.
- **Model Drift Dashboard:** drift per detector (model speed and tokens per agent, embeddings, documents) and the six alerts. Grafana's alert list shows the same alerts (see "Drift and alerts" in [docs/RUNBOOK.md](../docs/RUNBOOK.md)).

## Configuration

The application is configured using YAML files in the `config` directory:

- `dev.yaml`: Development configuration
- `uat.yaml`: Pre-production (user acceptance testing) configuration
- `prod.yaml`: Production configuration

The configuration file is selected by the `WEBAGENT_ENV` environment variable (`dev`, `uat` or `prod`; default `dev`). `.env.local` sets it to `dev`, but a value set in your shell wins, so `WEBAGENT_ENV=prod python -m backend.main` runs with production settings. API keys and secrets go in `.env.local` in the repo root, never in the YAML files.

Under `agents:` in each YAML file, every agent can set its `model` and an output limit, `max_tokens`. That limit lowers the shared `llm.max_tokens` for that agent only; web research uses 4,096. If the log says a Together reply "stopped at the max_tokens limit", raise that agent's limit.

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure your API keys are correctly set in `.env.local` in the repo root
   - Check the logs for authentication errors

2. **Docker Issues**:
   - Run `npm run stack:down` (add `-- -v` to also remove volumes) to clean up
   - Rebuild with `docker compose -f backend/docker-compose.yml --env-file .env.local build --no-cache`
   - Container marked unhealthy: the health check calls `GET /health`; check `docker compose logs webagent-backend` for a startup error (in uat/prod, missing `WEBAGENT_SECURITY_SECRET_KEY` or `WEBAGENT_API_KEY` stops startup)

3. **Port Conflicts**:
   - If ports are already in use, modify the port mappings in `docker-compose.yml`
   - The stack's Grafana uses host port 3001, so it doesn't clash with the Next.js dev server or a locally installed Grafana on 3000. Set `GRAFANA_HOST_PORT` in `.env.local` to use another port

### Logs

- When running with Python, logs go to the console and to `backend/logs/webagent-dev.log`
- Lines written during a task are tagged `[task <id>]`, so you can `grep` one task's lines
- Finished task reports are saved in `backend/data/results/<task_id>.json`
- When running with Docker, view logs with `docker compose -f backend/docker-compose.yml logs -f webagent-backend`; log files are in `backend/logs/` (mounted at `/app/logs`) and task results in the `webagent-data` volume

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run tests to ensure everything works
4. Submit a pull request

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/) 