# WebAgent Backend

This is the Python-based microservice backend for the WebAgent platform. It uses LangGraph to orchestrate a multi-agent system for research, analysis, and report generation.

**Version: 2.2.1** - Now with advanced agent capabilities and LangSmith observability.

## Key Features

- Complete multi-agent system with seven specialized agents
- Advanced data analysis and visualization capabilities
- Comprehensive report generation with Team Manager Agent
- Full LangSmith integration for tracing and observability
- Environment-specific configurations with YAML

## Architecture

The backend uses a microservice architecture with the following components:

- FastAPI server for REST API endpoints
- LangGraph for agent orchestration
- OpenAI integration for LLMs
- Redis for task queueing and caching
- ChromaDB for vector storage

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional)
- OpenAI API key
- Tavily API key (for web search)

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
1. Environment variables with `WEBAGENT_` prefix
2. Environment-specific YAML file
3. `.env` file values
4. Default values in code

#### Testing Configuration

You can test the configuration loading with:

```bash
# Test default (dev) configuration
python test_config.py

# Test specific environment
python test_config.py uat
```

### Traditional Environment Setup

You can also use the traditional `.env` file approach (lower priority than YAML):

1. Create a `.env` file in the backend directory:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
DEBUG_MODE=true
```

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

## API Endpoints

- `GET /`: Health check
- `GET /api/v1/health`: Detailed health check
- `POST /api/v1/chat`: Start a new chat workflow
- `GET /api/v1/chat/{task_id}`: Get status and results of a chat workflow
- `GET /api/v1/tasks`: List all tasks
- `GET /api/v1/tasks/{task_id}`: Get details about a specific task
- `DELETE /api/v1/tasks/{task_id}`: Delete a task

## Development

### Project Structure

```
backend/
├── app/
│   ├── api/                # API endpoints
│   │   ├── endpoints/
│   │   │   ├── chat.py
│   │   │   ├── health.py
│   │   │   └── tasks.py
│   │   └── router.py
│   ├── agents/            # Agent implementations
│   ├── core/              # Core configuration
│   │   ├── config.py      # Settings management
│   │   ├── config_def.py  # Pydantic models for config
│   │   └── loadEnvYAML.py # YAML configuration loader
│   ├── graph/             # LangGraph workflow
│   │   └── workflows.py
│   ├── models/            # Data models
│   │   ├── chat.py
│   │   └── task.py
│   ├── services/          # Service integrations
│   │   ├── llm.py
│   │   ├── task_manager.py
│   │   └── vectordb.py
│   └── utils/             # Utility functions
├── config/                # Environment configurations
│   ├── dev.yaml           # Development settings
│   ├── uat.yaml           # UAT testing settings
│   └── prod.yaml          # Production settings
├── main.py                # FastAPI app entry point
├── test_config.py         # Configuration test script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Configuration Categories

Each YAML configuration file defines settings for:

- **API**: Server host, port, debug mode
- **CORS**: Allowed origins, methods, headers
- **Database**: Vector DB and Redis settings
- **LLM**: Models, temperatures, timeouts
- **Web Search**: Provider, depth, result limits
- **Task Management**: Concurrency, TTL settings
- **Agents**: Model configuration for each specialized agent
- **Logging**: Log levels, formats, file settings
- **Security**: Token settings, algorithms

### API Documentation

When the server is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc` 