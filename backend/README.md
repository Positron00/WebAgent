# WebAgent Backend

This is the Python-based microservice backend for the WebAgent platform. It uses LangGraph to orchestrate a multi-agent system for research, analysis, and report generation.

**Version: 2.4.6** - Fixed Jest configuration issues for reliable test execution. The update resolves environment teardown problems, properly excludes non-test files from testing, and ensures all 27 tests pass consistently. This maintenance release improves developer experience with more reliable testing infrastructure.

## Key Features

- Complete multi-agent system with seven specialized agents
- Together AI integration with Llama 3.3 70B Instruct Turbo Free model
- Frontend API compatibility layer for seamless integration
- Advanced data analysis and visualization capabilities
- Comprehensive report generation with Team Manager Agent
- Full LangSmith integration for tracing and observability
- Security features including rate limiting, JWT authentication, and input validation
- Prometheus metrics for comprehensive monitoring
- Structured logging with request tracking
- Kubernetes deployment support

## Architecture

The backend uses a microservice architecture with the following components:

- FastAPI server for REST API endpoints
- LangGraph for agent orchestration
- OpenAI and Together AI integration for LLMs
- Redis for task queueing and caching
- ChromaDB for vector storage
- Prometheus client for metrics
- Structured logging with JSON output

## Security Features

- Rate limiting to prevent abuse
- Security HTTP headers to protect against common web vulnerabilities
- Input validation and sanitization
- JWT token-based authentication
- Secure password hashing with bcrypt
- Request size limiting
- Middleware-based security measures
- Environment variable isolation
- Comprehensive error handling and logging

## Monitoring & Observability

- Prometheus metrics for API and LLM requests
- Task duration and status tracking
- Structured logging with JSON format
- Request ID tracking across the application
- Health check endpoints with detailed status
- LangSmith integration for LLM tracing
- Alert rules for critical conditions

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional)
- OpenAI API key
- Together AI API key (optional)
- Tavily API key (optional for web search)
- LangSmith API key (optional for tracing)

### Environment Configuration

The application supports three deployment environments with dedicated configuration files:

1. **Development (`config/dev.yaml`)**: 
   - Debug mode enabled
   - Local database paths
   - Basic search depth
   - Detailed logging

2. **UAT Testing (`config/uat.yaml`)**:
   - Production-like settings
   - Container-aware database configuration
   - Comprehensive search depth
   - Increased task limits

3. **Production (`config/prod.yaml`)**:
   - Production-optimized settings
   - Strict security parameters
   - Cluster-aware database references
   - Higher concurrency limits
   - Minimal logging

#### Selecting an Environment

Set the `WEBAGENT_ENV` environment variable to choose which configuration to use:

```bash
# Development (default if not specified)
export WEBAGENT_ENV=dev

# User Acceptance Testing
export WEBAGENT_ENV=uat

# Production
export WEBAGENT_ENV=prod
```

#### Configuration Priority

Settings are loaded with the following priority (highest to lowest):
1. `.env.local` file (highest priority, specifically for API keys and sensitive information)
2. Environment variables with `WEBAGENT_` prefix 
3. Environment-specific YAML file (`dev.yaml`, `uat.yaml`, or `prod.yaml`)
4. `.env` file values (lowest priority, default fallback)
5. Default values in code

The recommended practice is:
- Store API keys and sensitive information in `.env.local` (never commit to version control)
- Configure environment-specific settings in YAML files
- Use environment variables for deployment-specific overrides
- Keep fallback/default values in `.env` and code

### Running with Docker Compose

The easiest way to start the backend is with Docker Compose:

```bash
# Start with development configuration (default)
docker-compose up

# Specify environment
WEBAGENT_ENV=uat docker-compose up
```

This will start the FastAPI server and Redis container.

### Running without Docker

1. Create a virtual environment and install dependencies:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Run the FastAPI server:

```bash
# With default (dev) configuration
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# With specific environment
WEBAGENT_ENV=uat uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Deployment

### Docker Deployment

The application includes a multi-stage Dockerfile for optimized production builds:

```bash
# Build the Docker image
docker build -t webagent-backend:2.4.0 .

# Run the container
docker run -p 8000:8000 \
  --env-file .env.local \
  --env WEBAGENT_ENV=prod \
  webagent-backend:2.4.0
```

### Kubernetes Deployment

Kubernetes manifests are available in the `kubernetes/` directory:

```bash
# Create the namespace
kubectl create namespace webagent

# Create secrets (replace placeholders first)
kubectl apply -f kubernetes/secrets.yaml

# Apply the configuration
kubectl apply -f kubernetes/configmap.yaml

# Deploy the application
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
```

## Monitoring

### Prometheus Integration

The application exposes Prometheus metrics at the `/api/v1/metrics` endpoint. Kubernetes ServiceMonitor configuration is available in `kubernetes/monitoring/prometheus.yaml`.

Key metrics include:
- HTTP request counts, latencies, and statuses
- Task counts and durations
- LLM request metrics and token usage
- System resource utilization

### Logging

Logs are written to both the console and files with automatic rotation:

- Development: Human-readable format with DEBUG level
- UAT/Production: JSON structured format with INFO/WARNING level
- Log files are stored in the `logs/` directory
- Production logs rotate daily and are kept for 30 days

## API Documentation

When the server is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`
- API Reference: See the `docs/api_reference.md` file for detailed documentation

## Project Structure

```
backend/
├── app/
│   ├── api/                # API endpoints
│   │   ├── endpoints/
│   │   │   ├── chat.py
│   │   │   ├── health.py
│   │   │   ├── tasks.py
│   │   │   └── frontend.py
│   │   └── router.py
│   ├── agents/            # Agent implementations
│   ├── core/              # Core configuration
│   │   ├── config.py      # Settings management
│   │   ├── logger.py      # Structured logging
│   │   ├── metrics.py     # Prometheus metrics
│   │   ├── middleware.py  # Security middleware
│   │   └── security.py    # Security utilities
│   ├── graph/             # LangGraph workflow
│   │   └── workflows.py
│   ├── models/            # Data models
│   ├── services/          # Service integrations
│   └── utils/             # Utility functions
├── config/                # Environment configurations
├── docs/                  # Documentation
├── kubernetes/            # Kubernetes manifests
├── logs/                  # Log files (gitignored)
├── main.py                # FastAPI app entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 