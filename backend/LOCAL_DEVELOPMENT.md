# WebAgent Backend - Local Development

This guide provides instructions for setting up and running the WebAgent backend locally for development purposes.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-org/webagent.git
cd webagent/backend
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# Create a .env file
cp .env.example .env
```

Edit the `.env` file to include your API keys:

```
OPENAI_API_KEY=your-openai-api-key
TOGETHER_API_KEY=your-together-api-key
JWT_SECRET=your-jwt-secret
LANGSMITH_API_KEY=your-langsmith-api-key
```

## Running Locally

### Option 1: Running with Python

1. Start the application:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API at http://localhost:8000

3. Access the API documentation at http://localhost:8000/docs

### Option 2: Running with Docker Compose

1. Start the application with Docker Compose:

```bash
docker-compose up
```

This will start:
- WebAgent backend on port 8000
- Prometheus on port 9090
- Grafana on port 3000

2. Access the services:
   - WebAgent API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (login with admin/WebAgent-Admin-2025!)

## Development Workflow

1. Make changes to the code
2. The server will automatically reload if you're using `uvicorn` with the `--reload` flag
3. Run tests to ensure your changes don't break existing functionality

## Running Tests

### Unit Tests

```bash
pytest tests/unit
```

### Integration Tests

```bash
pytest tests/integration
```

### All Tests

```bash
pytest
```

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
- `test.yaml`: Testing configuration
- `production.yaml`: Production configuration

The configuration file is selected based on the `ENVIRONMENT` environment variable.

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure your API keys are correctly set in the `.env` file
   - Check the logs for authentication errors

2. **Docker Issues**:
   - Run `docker-compose down -v` to clean up volumes and containers
   - Rebuild with `docker-compose build --no-cache`

3. **Port Conflicts**:
   - If ports are already in use, modify the port mappings in `docker-compose.yml`

### Logs

- When running with Python, logs are output to the console and to the `logs` directory
- When running with Docker, view logs with `docker-compose logs -f webagent-backend`

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run tests to ensure everything works
4. Submit a pull request

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/) 