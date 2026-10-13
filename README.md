# WebAgent Platform

[![version](https://img.shields.io/badge/version-3.2.2-blue.svg)](https://semver.org)
[![last-updated](https://img.shields.io/badge/last%20updated-2026--10--10-lightgrey.svg)](https://github.com/Positron00/WebAgent)

A multi-agent platform for web research and document processing.

## Overview

WebAgent is an advanced multi-agent AI platform that orchestrates specialized agents to process complex research tasks, document extraction, and data synthesis. It leverages LangGraph for agent workflows and provides both API access and programmatic interfaces for integration.

### Key Features

- **Multi-Agent System**: Integrated system of specialized agents working together to solve complex tasks
  - Supervisor Agent: Analyzes user queries and orchestrates the workflow
  - Research Agents: Web and internal knowledge retrieval
  - Senior Research Agent: Evaluates research quality and produces summary reports
  - Document Extraction Agent: Processes documents and extracts structured data
  - Team Manager Agent: Synthesizes outputs from multiple agents
  - Specialized Agents: Data analysis and coding assistance
- **Agentic triage**: Off / Auto / Always; in Auto a local classifier sends only research questions to the agents
- **Workflow Logs**: Detailed execution tracking for answers produced by the agent workflow
  - Records important decision points in the workflow
  - Shows step-by-step processing of complex queries
  - Included in final reports for transparency
  - Fully tested with comprehensive diagnostics

- **Document Processing Pipeline**: Comprehensive document ingestion and knowledge management
  - Batch document loading with multiple format support (PDF, docx, txt, md, html, csv, json)
  - Artifact registry integration for document versioning and distribution
  - Intelligent document chunking and embedding for vector database storage
  - Graph database integration for document relationship tracking
  - Data-as-a-Service API for document operations

- **LangGraph Integration**: Uses LangGraph (0.2.x) for complex agent workflow orchestration with optimized direct edge routing, comprehensive state management, and proper error handling throughout the workflow

- **Microservices Model Framework**: Decoupled model services that can be independently scaled and managed
  - Model Registry for discovery and status tracking
  - API Gateway for unified access to all model services
  - Supports transformer-based language and embedding models
  - Configurable via YAML for easy deployment and management

- **Advanced RAG Architecture**: Sophisticated retrieval and processing with built-in document extraction capabilities, enhanced BM25 lexical retrieval for improved results

- **Document Processing**: Extract, analyze, and synthesize information from PDFs, web pages, and other document formats

- **Research Loop Process**: Iterative research capability that evaluates quality and performs additional research as needed:
  1. Initial Research: Web and internal knowledge sources are queried based on the research plan
  2. Research Evaluation: Senior Research Agent evaluates completeness and quality (scoring 1-10)
  3. Follow-up Research: Additional targeted research based on identified gaps
  4. Iteration: Process repeats up to 3 times until quality threshold is met
  5. Final Synthesis: Comprehensive findings consolidated into a structured report

- **Comprehensive Security**: Advanced security module with authentication, authorization, rate limiting, input validation, error sanitization, request size limiting, and protection against common vulnerabilities
  - API keys required in uat/prod; the server won't start there without its secret key and API key
  - Rate limits apply per end user, even behind the web app's proxy
  - Model-written code runs without access to API keys, and in uat/prod only inside a locked-down container

- **Advanced Observability**: Detailed diagnostics and monitoring capabilities for tracking system health, performance metrics, and execution statistics
  - Every log line written during a task is tagged `[task <id>]`
  - Full LangSmith traces of each run, including token counts
  - Finished task reports saved to `backend/data/results/`

- **Reliability**: Temporary Together AI errors (rate limits, 5xx) are retried, and the server keeps answering while tasks run
  - Each agent's model output can be capped in its YAML settings, and a reply cut off at its limit is logged; a typical agentic run takes about 2.5–3.5 minutes
  - Task status is shared between server workers, so polls work with several workers
  - Follow-up questions ("what about Europe?") are researched as a complete question using the conversation so far

## System Architecture

WebAgent uses a modular architecture with a clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WebAgent Platform                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────-┐   │
│  │                                                                          │   │
│  │  ┌───────────┐    ┌───────────────────────────────────────────────--─┐   │   │
│  │  │           │    │               LangGraph Workflow                 │   │   │
│  │  │  FastAPI  │    │  ┌───────────┐        ┌───────────────────┐      │   │   │
│  │  │  Server   │◄───┼─►│ Supervisor│        │ Research Agents   │      │   │   │
│  │  │           │    │  │  Agent    │───────►│ ┌─────────────┐   │      │   │   │
│  │  └───────────┘    │  └───────────┘        │ │Web Research │   │      │   │   │ 
│  │        ▲          │        │              │ └─────────────┘   │      │   │   │
│  │        │          │        │              │ ┌─────────────┐   │      │   │   │
│  │        │          │        │              │ │  Internal   │   │      │   │   │
│  │        │          │        │              │ │  Research   │   │      │   │   │
│  │        │          │        │              │ └─────────────┘   │      │   │   │
│  │  ┌───────────┐    │        │              └─────────┬─────────┘      │   │   │
│  │  │           │    │        │                        │                │   │   │
│  │  │ Frontend  │    │        │              ┌─────────▼─────────┐      │   │   │
│  │  │  Client   │◄───┼────────┼──────────────│  Senior Research  │      │   │   │
│  │  │           │    │        │      ┌───────│      Agent        │      │   │   │
│  │  └───────────┘    │        │      │       └─────────┬─────────┘      │   │   │
│  │                   │        │      │                 │                │   │   │
│  │                   │        │      │       ┌─────────▼─────────┐      │   │   │
│  │                   │        │      │       │  Specialized      │      │   │   │
│  │  ┌───────────┐    │        ▼      │       │     Agents        │      │   │   │
│  │  │           │    │  ┌───────────┐│       │ ┌─────────────┐   │      │   │   │
│  │  │Observa-   │    │  │ Document  ││       │ │    Data     │   │      │   │   │
│  │  │bility &   │◄───┼──│Extraction ││       │ │  Analysis   │   │      │   │   │
│  │  │Monitoring │    │  │   Agent   ││       │ └─────────────┘   │      │   │   │
│  │  │           │    │  └───────────┘│       │ ┌─────────────┐   │      │   │   │
│  │  └───────────┘    │        │      │       │ │   Coding    │   │      │   │   │
│  │                   │        │      │       │ │  Assistant  │   │      │   │   │
│  │                   │        │      │       │ └─────────────┘   │      │   │   │
│  │                   │        │      │       └─────────┬─────────┘      │   │   │
│  │                   │        │      │                 │                │   │   │
│  │                   │        │      │                 │                │   │   │
│  │                   │        ▼      ▼                 ▼                │   │   │
│  │                   │  ┌─────────────────────────────────────────┐     │   │   │
│  │                   │  │               Team Manager Agent        │     │   │   │
│  │                   │  └─────────────────────────────────────────┘     │   │   │
│  │                   │                                                  │   │   │
│  │                   └──────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │   │
│  │  │ Prometheus  │    │   Grafana   │    │ LangSmith   │    │ Security  │  │   │
│  │  │  Metrics    │    │ Dashboards  │    │  Tracing    │    │ Module    │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘  │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Model Microservices Architecture

The WebAgent platform now includes a model microservices framework that allows each model to run as an independent service:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Model Microservices Framework                             │
│                                                                                 │
│  ┌─────────────┐                                                                │
│  │   Client    │                                                                │
│  │ Applications│                                                                │
│  └──────┬──────┘                                                                │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────┐                                                    │
│  │                         │                                                    │
│  │     Model API Gateway   │                                                    │
│  │                         │                                                    │
│  └───┬─────────────┬───────┘                                                    │
│      │             │                                                            │
│      │             │                 ┌───────────────┐                          │
│      │             │                 │               │                          │
│      │             │◄───Discovery───►│ Model Registry│                          │
│      │             │                 │               │                          │
│      │             │                 └───────────────┘                          │
│      │             │                                                            │
│      ▼             ▼                                                            │
│  ┌───────────┐ ┌───────────┐         ┌───────────┐        ┌───────────┐         │
│  │           │ │           │         │           │        │           │         │
│  │ LLM Model │ │ Embedding │         │ Classifier│        │ Other     │         │
│  │ Service   │ │ Model     │         │ Model     │        │ Model     │         │
│  │           │ │ Service   │         │ Service   │        │ Services  │         │
│  └───────────┘ └───────────┘         └───────────┘        └───────────┘         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                         │    │
│  │                         Model Manager                                   │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **API Layer**: FastAPI server exposing REST endpoints for client applications
   - Authentication and authorization middleware
   - Request validation and rate limiting
   - Security headers and input sanitization
   - Swagger documentation with OpenAPI schema

2. **Workflow Engine**: LangGraph orchestration of agent interactions
   - Conditional routing based on context and agent outputs
   - Parallel execution of compatible agents
   - Feedback loops for iterative refinement
   - State management and persistence
   - Built on LangGraph 0.2.x

3. **Agents**: Specialized autonomous agents each with a specific role
   - Supervisor Agent: Orchestrates the workflow and analyzes queries
   - Research Agents: Gather information from web and internal sources
   - Senior Research Agent: Evaluates research quality and requests additional information
   - Document Extraction Agent: Processes and extracts information from documents
   - Specialized Agents: Data analysis and code generation
   - Team Manager Agent: Synthesizes outputs into final response

4. **Model Microservices**: Independent model services that can be managed separately
   - Model Registry: Tracks available models and their status
   - Model API Gateway: Routes requests to appropriate model services
   - Base Model Service: Foundation for all model services
   - Model Manager: Orchestrates model service processes
   - Configuration System: YAML-based configuration for model services

5. **Document Processing Pipeline**: For extracting and processing document content
   - PDF, Word, HTML and text document support
   - Table and image extraction
   - Structural parsing and semantic chunking
   - Hierarchical document representation
   - Enhanced BM25 lexical search for better keyword matching

6. **Integration Layer**: Connectors to external knowledge sources and APIs
   - Web search integration (via Tavily)
   - Internal knowledge base (via vector database)
   - LLM providers: Together AI (default, `meta-models/Muse-Glimmer-30B`), with OpenAI, Anthropic, Groq and Gemini as fallbacks
   - LangSmith for tracing and debugging

7. **Observability & Monitoring**: Tools for system monitoring and performance analysis
   - Prometheus metrics collection
   - Grafana dashboards for visualization
   - Detailed diagnostics for system health
   - Performance profiling and resource tracking
   - Security monitoring and alerting
   - Comprehensive error tracking and reporting

8. **Security Module**: Comprehensive security features
   - JWT authentication and API key authorization
   - Input validation and sanitization
   - Rate limiting and request size validation
   - Error message sanitization to prevent data leakage
   - Protection against common web vulnerabilities
   - Secure configuration management
   - Comprehensive middleware for request protection

## Running WebAgent

Follow these steps to set up and run the WebAgent platform:

### Prerequisites

- Python 3.12 or higher
- Node.js 18 or higher (for the web app)
- Conda for environment management
- Git for version control
- A Together AI API key (required) and a Tavily API key (for web research)

### Setup Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/Positron00/WebAgent.git
   cd webagent
   ```

2. Create and activate conda environment:
   ```bash
   conda create -n agents python=3.12
   conda activate agents
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt   # backend (uses backend/constraints.txt pins)
   npm install                       # web app
   ```

4. Add your API keys:
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local: set TOGETHER_API_KEY and TAVILY_API_KEY at minimum
   ```
   `.env.local` is git-ignored and is read by both the backend and the web app. The template lists every setting with its default.
   Environment-specific settings (agents, models, limits) live in `backend/config/{dev,uat,prod}.yaml`, chosen by `WEBAGENT_ENV`. Each agent under `agents:` can set its own `model` and output limit (`max_tokens`).

### Running the Application

1. Start the backend (FastAPI on port 8000):
   ```bash
   python -m backend.main
   ```

2. In a second terminal, start the web app (Next.js on port 3000):
   ```bash
   npm run dev
   ```
   The **Agentic** control in the chat toolbar has three modes:
   - **Off**: answer directly through Together AI; the backend isn't needed.
   - **Auto**: the backend's triage decides per question. Routine questions are answered directly in seconds; research questions go to the multi-agent backend.
   - **Always**: every question goes to the multi-agent backend (a run takes a few minutes). Images are only used by direct answers.

   Each answer is labelled "Answered directly" or "Deep research"; hover for the reason. In Auto, a direct answer offers **Research this in depth**, which sends the same question to the research agents and adds the new answer below. New users start on the default from `WEBAGENT_DEFAULT_AGENTIC_MODE` in `.env.local` (`off` if unset); a mode you pick is remembered in the browser. See [backend/README.md](backend/README.md#agentic-triage) for the triage service.

3. Run Model Services (optional):
   ```bash
   # Start the model manager
   python -m backend.models.run_manager
   
   # Start a specific model service
   python -m backend.models.run_model_service --model-id mistral-7b   # any model ID in backend/models/config/models.yaml
   ```

4. Access the application:
   - Web UI: http://localhost:3000
   - API: http://localhost:8000/api/v1
   - Swagger UI: http://localhost:8000/api/v1/docs
   - Health check: http://localhost:8000/health
   - Model API Gateway: http://localhost:8100

### Deploying

- **Docker / Docker Compose**: see [backend/README.md](backend/README.md#running-with-docker) and [backend/LOCAL_DEVELOPMENT.md](backend/LOCAL_DEVELOPMENT.md). The image runs as `prod`. For the local stack (backend, Prometheus, Grafana), add `GRAFANA_ADMIN_PASSWORD` to `.env.local` and run `npm run stack:up`.
- **Kubernetes**: see [backend/kubernetes/README.md](backend/kubernetes/README.md). `deploy.sh` requires `WEBAGENT_SECURITY_SECRET_KEY`, `WEBAGENT_API_KEY` and `GRAFANA_ADMIN_PASSWORD`.
- **Vercel** (web app only): see [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md). Agentic mode needs `WEBAGENT_BACKEND_URL` pointing at a running backend.

## Testing and Diagnostics

WebAgent includes comprehensive testing and diagnostic tools.

### Running Tests

1. Run all tests:
   ```bash
   npm test                  # web app (Jest)
   npm run test:backend      # backend (pytest, in the `agents` conda env)
   ```
   Backend tests run offline: they fail if any test opens a network connection, and they never trace to LangSmith or modify tracked files.

2. Run tests with coverage:
   ```bash
   npm run test:coverage     # fails if web app coverage drops below the floor in jest.config.js
   npm run coverage:backend
   ```
   See [docs/TESTING.md](docs/TESTING.md) for details.

3. Run model service tests:
   ```bash
   # Run model registry and API gateway tests
   python -m backend.tests.test_models
   
   # Run model service implementation tests
   python -m backend.tests.test_model_services
   
   # Run model manager tests (simplified version)
   python -m backend.tests.test_model_manager_simplified
   ```

4. Run diagnostics:
   ```bash
   python -m backend.diagnostics.runners.run_all
   ```

### Diagnostics

1. Check system status:
   ```bash
   python -m backend.diagnostics.system_check
   ```

2. Check model services status:
   ```bash
   curl http://localhost:8100/metrics
   ```

3. View logs:
   ```bash
   tail -f backend/logs/webagent-dev.log
   ```
   While a task runs, each line is tagged with the first 8 characters of its task ID, so you can follow one task:
   ```bash
   grep "\[task 9e6a4965\]" backend/logs/webagent-dev.log
   ```

4. Read saved task reports:
   Each task is saved to `backend/data/results/<task_id>.json` when it starts (status `processing`) and again when it finishes, with the question, status, timings and full report. `GET /api/task/<task_id>` reads from there, so any server worker can answer and reports survive a restart.

5. View LangSmith traces (if `LANGSMITH_API_KEY` is set):
   - Visit LangSmith at https://smith.langchain.com and open the project `webagent-research` (change with `LANGSMITH_PROJECT_NAME`)
   - Each task is one trace containing every agent step and model call, with token counts
   - The app's settings override any `LANGSMITH_PROJECT` exported in your shell
   - Set `LANGSMITH_TRACING_ENABLED=false` to turn tracing off (backend tests always run with tracing off)

### Common Issues and Troubleshooting

1. **Connection Errors with Model Services**
   - Ensure model services are running and registered correctly
   - Check ports and host configurations in `backend/models/config/models.yaml`
   - Verify model paths and accessibility

2. **LangGraph Workflow Issues**
   - Check for proper agent implementation and integration
   - Verify state transitions in the workflow
   - Review LangSmith traces for detailed execution information

3. **Security Test Failures**
   - Ensure rate limiting is properly configured
   - Check error message redaction in security module
   - Verify middleware ordering in FastAPI application

4. **Agentic answer starts with "I'm currently operating in fallback mode"**
   - The multi-agent task failed or the backend isn't reachable, so the web app answered with plain chat
   - Check the backend is running on `WEBAGENT_BACKEND_URL` (default `http://127.0.0.1:8000`) and look for errors in `backend/logs/webagent-dev.log`

5. **"Unable to access non-serverless model" (HTTP 400)**
   - Together AI no longer serves that model on demand; set `WEBAGENT_LLM_DEFAULT_MODEL` to a serverless model (default `meta-models/Muse-Glimmer-30B`)

6. **LangSmith shows steps stuck as "pending" or traces in another project**
   - Usually a `LANGSMITH_PROJECT` exported in your shell; the backend now overrides it, so restart the backend

7. **Backend runs as `dev` although you asked for `prod`**
   - A `WEBAGENT_ENV` in your shell or container wins over `.env.local`. With `docker run --env-file .env.local`, add `-e WEBAGENT_ENV=prod` after it

8. **A backend test fails with "Test tried to open a network connection"**
   - The test reached a real service (LLM, Tavily, LangSmith, Hugging Face, NLTK). Fake it instead; see [docs/TESTING.md](docs/TESTING.md)

9. **Agentic runs are slow, or the web app gives up after 8 minutes**
   - A typical run takes 2.5–3.5 minutes. Open the task's LangSmith trace and compare each model call's completion tokens with its limit: a call that stops exactly at the limit is running away, and the backend log says "stopped at the max_tokens limit"
   - Lower or raise that agent's `max_tokens` under `agents:` in `backend/config/{env}.yaml`, and check that agent's prompt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain and LangGraph for agent frameworks
- OpenAI and Anthropic for LLM APIs
- FastAPI for the backend framework
- React and Next.js for the frontend framework

## Working with Documents

### Batch Document Loading

WebAgent now includes a powerful batch document loader for ingesting documents into the system. The documents are processed, chunked, and stored in both vector and graph databases for efficient retrieval.

To use the batch document loader from the command line:

```bash
# Load all PDFs from a directory
python scripts/batch_document_loader.py --dir /path/to/documents --pattern "*.pdf"

# Load a single document with metadata
python scripts/batch_document_loader.py --file /path/to/document.pdf --metadata '{"category": "reports", "tags": ["financial", "quarterly"]}'

# Load documents and publish to an artifact registry
python scripts/batch_document_loader.py --dir /path/to/documents --registry my-registry
```

### Document Data-as-a-Service API

The platform provides a RESTful API for document operations:

- **Upload a Document**: `POST /api/v1/documents/upload`
- **Search Documents**: `POST /api/v1/documents/search`
- **Get Document by ID**: `GET /api/v1/documents/document/{document_id}`
- **Find Related Documents**: `GET /api/v1/documents/related/{document_id}`

Example API usage with curl:

```bash
# Upload a document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F 'metadata_json={"category": "reports", "tags": ["financial"]}'

# Search documents
curl -X POST "http://localhost:8000/api/v1/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "financial report 2024", "limit": 5, "include_content": true}'
```

### Graph Database Integration

WebAgent now tracks document relationships in a graph database, allowing for sophisticated relationship queries. By default, documents from the same source are linked, and additional relationships can be created programmatically.

The system supports both Neo4j for production use and includes a NetworkX-based fallback for development environments. Configure the graph database URL in your environment settings.
