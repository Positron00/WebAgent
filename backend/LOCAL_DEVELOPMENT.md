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

3. Install dependencies:

```bash
pip install -r backend/requirements.txt -c backend/constraints.txt
```

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

1. Create `backend/.env` (git-ignored; Docker Compose reads it) with at least:

```bash
GRAFANA_ADMIN_PASSWORD=choose-a-password
TOGETHER_API_KEY=your_together_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Compose stops with "required variable GRAFANA_ADMIN_PASSWORD is missing" if the password isn't set. The backend runs as `dev` unless you also set `WEBAGENT_ENV=uat` or `prod` there (those need `WEBAGENT_SECURITY_SECRET_KEY` and `WEBAGENT_API_KEY` as well).

2. Start the application from the `backend/` folder:

```bash
docker compose up --build
```

This will start:
- WebAgent backend on port 8000
- Prometheus on port 9090
- Grafana on port 3000

3. Access the services:
   - WebAgent API: http://localhost:8000 (health check: http://localhost:8000/health)
   - API Documentation: http://localhost:8000/api/v1/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (login with `admin` and the `GRAFANA_ADMIN_PASSWORD` from `backend/.env`)

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
npm run coverage:backend                    # with a coverage report
```

Tests run offline: `backend/tests/conftest.py` blocks network connections, turns LangSmith tracing off, and sends saved results and the model registry to temporary files. See [docs/TESTING.md](../docs/TESTING.md).

## Monitoring

The local development setup includes Prometheus and Grafana for monitoring:

- Prometheus collects metrics from the WebAgent backend
- Grafana visualizes these metrics with pre-configured dashboards

### Grafana Dashboards

The WebAgent Backend Dashboard provides:
- HTTP request rates by status code
- Request duration by endpoint
- LLM request rates and durations
- Token usage metrics
- Task completion rates

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
   - Run `docker compose down -v` to clean up volumes and containers
   - Rebuild with `docker compose build --no-cache`
   - Container marked unhealthy: the health check calls `GET /health`; check `docker compose logs webagent-backend` for a startup error (in uat/prod, missing `WEBAGENT_SECURITY_SECRET_KEY` or `WEBAGENT_API_KEY` stops startup)

3. **Port Conflicts**:
   - If ports are already in use, modify the port mappings in `docker-compose.yml`
   - Grafana and the Next.js web app both default to port 3000. Change Grafana's mapping (for example `"3001:3000"`) if you run both

### Logs

- When running with Python, logs go to the console and to `backend/logs/webagent-dev.log`
- Lines written during a task are tagged `[task <id>]`, so you can `grep` one task's lines
- Finished task reports are saved in `backend/data/results/<task_id>.json`
- When running with Docker, view logs with `docker compose logs -f webagent-backend`; log files are in `backend/logs/` (mounted at `/app/logs`) and task results in the `webagent-data` volume

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run tests to ensure everything works
4. Submit a pull request

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/) 