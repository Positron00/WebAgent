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
- **Diagnostics**: Built-in system monitoring and diagnostics

## Prerequisites

- Python 3.10+
- OpenAI API key or Anthropic API key
- LangSmith API key (optional, for tracing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/webagent.git
   cd webagent/backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your environment variables:
   ```
   # Core settings
   WEBAGENT_ENV=development
   DEBUG_MODE=true
   LOG_LEVEL=INFO
   
   # LLM Provider
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   
   # LangSmith (optional)
   LANGSMITH_API_KEY=your_langsmith_key_here
   LANGSMITH_PROJECT=webagent
   LANGSMITH_ENABLED=false
   ```

## Running the Application

### Development Server

Start the FastAPI server with hot reloading:

```bash
uvicorn main:app --reload
```

Access the API documentation at http://localhost:8000/api/v1/docs

### Production Server

For production deployments, use Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
```

### Running with Docker

Build and run the Docker container:

```bash
# Build the image
docker build -t webagent-backend .

# Run the container
docker run -p 8000:8000 --env-file .env webagent-backend
```

## Diagnostics and Monitoring

WebAgent includes built-in diagnostics for monitoring system health and performance:

```bash
# Run diagnostics from the command line
python -m app.utils.diagnostics

# Or use the diagnostic API endpoint
curl http://localhost:8000/api/v1/health/diagnostics
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

- **POST /api/v1/chat/completions**: Main endpoint for chat interactions
- **POST /api/v1/documents/extract**: Document extraction endpoint
- **GET /api/v1/health**: Health check endpoint
- **GET /api/v1/health/diagnostics**: System diagnostics endpoint
- **POST /api/v1/tasks/create**: Create an asynchronous task
- **GET /api/v1/tasks/{task_id}**: Get task status and results

See the OpenAPI documentation for complete details and request/response schemas.

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_senior_research.py

# Run with coverage report
python -m pytest --cov=app
```

### Testing Environment

For testing, you can use a separate `.env.test` file:

```bash
cp .env .env.test
# Edit .env.test with test configuration
```

Then run tests with:

```bash
WEBAGENT_ENV=test python -m pytest
```

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
│   │   └── chat.py                 # Chat models
│   ├── services/         # External service integrations
│   │   └── llm.py                  # LLM service client
│   └── utils/            # Utility functions
│       └── diagnostics.py          # System diagnostics
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

### Environment Variables

Key environment variables used by the application:

| Variable | Description | Default |
|----------|-------------|---------|
| WEBAGENT_ENV | Environment (development, test, production) | development |
| DEBUG_MODE | Enable debug mode | false |
| LOG_LEVEL | Logging level | INFO |
| LLM_PROVIDER | LLM provider (openai, anthropic) | openai |
| OPENAI_API_KEY | OpenAI API key | |
| ANTHROPIC_API_KEY | Anthropic API key | |
| LANGSMITH_ENABLED | Enable LangSmith tracing | false |
| LANGSMITH_API_KEY | LangSmith API key | |
| LANGSMITH_PROJECT | LangSmith project name | webagent |

## Troubleshooting

### Common Issues

1. **API Keys**: Ensure your LLM provider API keys are correctly set in `.env`
2. **Import Errors**: Make sure you're running from the correct directory or use `path_setup.py`
3. **Workflow Errors**: Check the logs for details on agent errors

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

Logs are stored in the `logs/` directory. For more detailed logs, set:

```
LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License. 