# WebAgent Platform

A comprehensive multi-agent platform for web research, document processing, and knowledge synthesis.

![WebAgent Platform Version](https://img.shields.io/badge/version-2.5.8-blue)
![Last Updated](https://img.shields.io/badge/last%20updated-2025--03--14-brightgreen)

## Overview

WebAgent is an advanced multi-agent AI platform that orchestrates specialized agents to process complex research tasks, document extraction, and data synthesis. It leverages LangGraph for agent workflows and provides both API access and programmatic interfaces for integration.

### Key Features

- **Multi-Agent System**: Integrated system of specialized agents working together to solve complex tasks
  - Supervisor Agent: Analyzes user queries and orchestrates the workflow
  - Research Agents: Web and internal knowledge retrieval
  - Senior Research Agent: Evaluates research quality and produces final reports
  - Document Extraction Agent: Processes documents and extracts structured data
  - Team Manager Agent: Synthesizes outputs from multiple agents
  - Specialized Agents: Data analysis and coding assistance

- **LangGraph Integration**: Uses LangGraph for complex agent workflow orchestration with conditional routing, parallel execution, and feedback loops

- **Advanced RAG Architecture**: Sophisticated retrieval and processing with built-in document extraction capabilities

- **Document Processing**: Extract, analyze, and synthesize information from PDFs, web pages, and other document formats

- **Research Loop Process**: Iterative research capability that evaluates quality and performs additional research as needed:
  1. Initial Research: Web and internal knowledge sources are queried based on the research plan
  2. Research Evaluation: Senior Research Agent evaluates completeness and quality (scoring 1-10)
  3. Follow-up Research: Additional targeted research based on identified gaps
  4. Iteration: Process repeats up to 3 times until quality threshold is met
  5. Final Synthesis: Comprehensive findings consolidated into a structured report

- **Enhanced Security**: Comprehensive security measures for input validation, authentication, rate limiting, and protection against common web vulnerabilities

- **Advanced Observability**: Detailed diagnostics and monitoring capabilities for tracking system health, performance metrics, and execution statistics

## System Architecture

WebAgent uses a modular architecture with a clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WebAgent Platform                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  ┌───────────┐    ┌────────────────────────────────────────────────┐   │   │
│  │  │           │    │               LangGraph Workflow                │   │   │
│  │  │  FastAPI  │    │  ┌───────────┐        ┌───────────────────┐    │   │   │
│  │  │  Server   │◄───┼─►│ Supervisor│        │ Research Agents    │    │   │   │
│  │  │           │    │  │  Agent    │───────►│ ┌─────────────┐    │    │   │
│  │  └───────────┘    │  └───────────┘        │ │Web Research │    │    │   │
│  │        ▲          │        │              │ └─────────────┘    │    │   │
│  │        │          │        │              │ ┌─────────────┐    │    │   │
│  │        │          │        │              │ │  Internal   │    │    │   │
│  │        │          │        │              │ │  Research   │    │    │   │
│  │        │          │        │              │ └─────────────┘    │    │   │
│  │  ┌───────────┐    │        │              └─────────┬─────────┘    │   │   │
│  │  │           │    │        │                        │                │   │   │
│  │  │ Frontend  │    │        │              ┌─────────▼─────────┐     │   │   │
│  │  │  Client   │◄───┼────────┼──────────────│  Senior Research  │     │   │   │
│  │  │           │    │        │      ┌───────│      Agent        │     │   │   │
│  │  └───────────┘    │        │      │       └─────────┬─────────┘     │   │   │
│  │                   │        │      │                 │                │   │   │
│  │                   │        │      │       ┌─────────▼─────────┐     │   │   │
│  │                   │        │      │       │  Specialized      │     │   │   │
│  │  ┌───────────┐    │        ▼      │       │     Agents        │     │   │   │
│  │  │           │    │  ┌───────────┐│       │ ┌─────────────┐   │     │   │   │
│  │  │Observa-   │    │  │ Document  ││       │ │    Data     │   │     │   │   │
│  │  │bility &   │◄───┼──│Extraction ││       │ │  Analysis   │   │     │   │   │
│  │  │Monitoring │    │  │   Agent   ││       │ └─────────────┘   │     │   │   │
│  │  │           │    │  └───────────┘│       │ ┌─────────────┐   │     │   │   │
│  │  └───────────┘    │        │      │       │ │   Coding    │   │     │   │   │
│  │                   │        │      │       │ │  Assistant  │   │     │   │   │
│  │                   │        │      │       │ └─────────────┘   │     │   │   │
│  │                   │        │      │       └─────────┬─────────┘     │   │   │
│  │                   │        │      │                 │                │   │   │
│  │                   │        │      │                 │                │   │   │
│  │                   │        ▼      ▼                 ▼                │   │   │
│  │                   │  ┌─────────────────────────────────────────┐    │   │   │
│  │                   │  │               Team Manager Agent         │    │   │   │
│  │                   │  └─────────────────────────────────────────┘    │   │   │
│  │                   │                                                  │   │   │
│  │                   └──────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │   │
│  │  │ Prometheus  │    │   Grafana   │    │ LangSmith   │    │ Security  │ │   │
│  │  │  Metrics    │    │ Dashboards  │    │  Tracing    │    │ Module    │ │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
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

3. **Agents**: Specialized autonomous agents each with a specific role
   - Supervisor Agent: Orchestrates the workflow and analyzes queries
   - Research Agents: Gather information from web and internal sources
   - Senior Research Agent: Evaluates research quality and requests additional information
   - Document Extraction Agent: Processes and extracts information from documents
   - Specialized Agents: Data analysis and code generation
   - Team Manager Agent: Synthesizes outputs into final response

4. **Document Processing Pipeline**: For extracting and processing document content
   - PDF, Word, HTML and text document support
   - Table and image extraction
   - Structural parsing and semantic chunking
   - Hierarchical document representation

5. **Integration Layer**: Connectors to external knowledge sources and APIs
   - Web search integration (via Tavily)
   - Internal knowledge base (via vector database)
   - LLM provider integration (OpenAI, Anthropic)
   - LangSmith for tracing and debugging

6. **Observability & Monitoring**: Tools for system monitoring and performance analysis
   - Prometheus metrics collection
   - Grafana dashboards for visualization
   - Detailed diagnostics for system health
   - Performance profiling and resource tracking
   - Security monitoring and alerting

7. **Security Module**: Comprehensive security features
   - Input validation and sanitization
   - Authentication and authorization
   - Rate limiting and request validation
   - Protection against common web vulnerabilities
   - Secure configuration management

### Research Loop Architecture

The WebAgent platform features an advanced research loop capability that enables iterative refinement of research results:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Research Loop Architecture                       │
│                                                                         │
│  ┌──────────────┐     ┌────────────────┐     ┌───────────────────────┐  │
│  │  Research    │     │  Research      │     │  Research Evaluation  │  │
│  │   Request    │────►│  Execution     │────►│   & Analysis          │  │
│  └──────────────┘     └────────────────┘     └───────────┬───────────┘  │
│                                                          │              │
│           ┌───────────────────────────────┐              │              │
│           │                               │              │              │
│           │                               ▼              ▼              │
│  ┌────────▼─────────┐     ┌─────────────────────┐     ┌─────────────┐  │
│  │  Final Research  │◄────│ Quality Assessment  │◄────┤  Iteration  │  │
│  │  Synthesis       │  No │ Meets Threshold?    │  No │  Count < 3? │  │
│  └──────────────────┘     └─────────┬───────────┘     └─────┬───────┘  │
│                                     │ Yes                   │ Yes      │
│                                     │                       │          │
│                                     │                 ┌─────▼───────┐  │
│                                     └────────────────►│ Additional  │  │
│                                                       │ Research    │  │
│                                                       └─────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

This loop enables the Senior Research Agent to:
1. Evaluate research completeness with a numerical score (1-10)
2. Identify specific gaps and missing information
3. Formulate targeted follow-up questions
4. Request additional research from the appropriate agents
5. Re-evaluate and refine until quality threshold is met
6. Synthesize the final comprehensive report

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 16+ (for frontend)
- OpenAI API key or Anthropic API key
- LangSmith API key (optional, for tracing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/webagent.git
   cd webagent
   ```

2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd ..
   npm install
   ```

4. Create a `.env` file in the backend directory:
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
   
   # Security settings
   REQUEST_SIZE_LIMIT=10485760  # 10MB
   RATE_LIMIT_ENABLED=true
   RATE_LIMIT_REQUESTS=100
   RATE_LIMIT_TIMEFRAME=60
   
   # Authentication (optional)
   AUTH_ENABLED=false
   JWT_SECRET_KEY=your_jwt_secret_here
   API_KEY_ENABLED=false
   API_KEY=your_api_key_here
   ```

### Running the Application

1. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. Start the frontend (in a new terminal):
   ```bash
   npm run dev
   ```

3. Access the application at `http://localhost:3000`

### Example Usage

Here's an example of using the WebAgent platform programmatically:

```python
# Example script to run a workflow
from backend.path_setup import setup_path
setup_path()

from backend.app.agents.supervisor import get_supervisor_agent
from backend.app.graph.workflows import get_agent_workflow
from backend.app.models.task import WorkflowState

# Get the workflow
workflow = get_agent_workflow()

# Create initial state with a research query
state = WorkflowState(query="What are the latest advancements in large language models?")

# Execute the workflow
final_state = workflow.invoke(state)

# Print the final report
if final_state.final_report:
    print("Final Report:")
    print(final_state.final_report["content"])
```

### Running Diagnostics

The WebAgent platform includes built-in diagnostics for monitoring system health and performance:

```python
# Example script to run diagnostics
from backend.path_setup import setup_path
setup_path()

from backend.app.utils.diagnostics import print_diagnostics_report

# Run full diagnostics and print the report
print_diagnostics_report()
```

Sample output:
```
========== WebAgent Diagnostics Report ==========
WebAgent v2.5.8 | 2025-03-14T12:34:56.789012
Environment: development
==================================================

## System Information
OS: Darwin 24.3.0
Python: 3.10.11
CPU Count: 8
Platform: Darwin-24.3.0-x86_64-i386-64bit
Working Directory: /Users/user/Projects/webagent

## Environment Configuration
Debug Mode: True
LLM Provider: openai

API Keys:
  OPENAI_API_KEY: ✓ Set
  ANTHROPIC_API_KEY: ✓ Set
  LANGSMITH_API_KEY: ✓ Set

## Agent Status
supervisor: active (Avg time: 0.83s)
web_research: active (Avg time: 2.41s)
internal_research: active (Avg time: 1.78s)
senior_research: active (Avg time: 1.92s)
document_extraction: active (Avg time: 3.56s)
team_manager: active (Avg time: 0.95s)

## Workflow
Nodes: __start__, supervisor, web_research, internal_research, senior_research, data_analysis, coding_assistant, team_manager, end
Compiled: True
Node Count: 9
Edge Count: 14

## Network Connectivity
openai: ✓ Available (Latency: 126.3ms)
anthropic: ✓ Available (Latency: 154.8ms)
langsmith: ✓ Available (Latency: 87.2ms)

## Language Model Status
Provider: openai
Default Model: gpt-4-turbo
API Check: check_disabled

## Performance Metrics
Memory (RSS): 156.45 MB
System Memory: 14.32 GB available of 16.00 GB
CPU Usage: 2.3% (process) / 8.7% (system)
Threads: 12
Uptime: 5.2 minutes

LLM Calls:
  gpt-4-turbo: 24 calls, avg 0.92s per call
  claude-3-haiku: 8 calls, avg 1.33s per call

========== End Diagnostics Report ==========
```

### Using the CLI

The WebAgent platform includes a command-line interface for running tasks and diagnostics:

```bash
# Run diagnostics
python scripts/run_webagent.py --mode diagnostics

# Run diagnostics with network and LLM checks
python scripts/run_webagent.py --mode diagnostics --check-network --check-llm

# Run a workflow
python scripts/run_webagent.py --mode workflow --query "Research the impact of AI on healthcare"

# Run a workflow with custom parameters
python scripts/run_webagent.py --mode workflow --query "Research the impact of AI on healthcare" --max-iterations 5 --timeout 600 --output markdown

# Extract data from a document
python scripts/run_webagent.py --mode direct --type document_extraction --document path/to/document.pdf

# Run a research request
python scripts/run_webagent.py --mode direct --type research --query "What are the latest advancements in quantum computing?" --output text

# Run a coding assistant request
python scripts/run_webagent.py --mode direct --type coding_assistant --query "Write a Python function to calculate Fibonacci numbers"
```

The CLI supports the following modes:

1. **diagnostics**: Run system diagnostics to check WebAgent health
   - `--check-network`: Test connectivity to external services
   - `--check-llm`: Verify language model availability

2. **workflow**: Run a full agent workflow for complex queries
   - `--query`: The user query to process (required)
   - `--max-iterations`: Maximum number of research iterations (default: 3)
   - `--timeout`: Maximum time in seconds to wait for completion (default: 300)
   - `--output`: Output format (json, text, markdown)

3. **direct**: Make a direct request to a specific agent
   - `--type`: Request type (document_extraction, research, team_management, data_analysis, coding_assistant)
   - `--query`: Query text for research requests
   - `--document`: Path to document for extraction
   - `--output`: Output format (json, text, markdown)

The script provides detailed error handling and logging to help diagnose issues during execution.

## Security Features

WebAgent includes comprehensive security features to protect against common threats:

1. **Input Validation and Sanitization**:
   - Automatic sanitization of potentially dangerous inputs
   - Prevention of prompt injection attacks
   - Validation of all user inputs and document content

2. **Authentication and Authorization**:
   - JWT-based authentication system
   - API key authentication for programmatic access
   - Fine-grained permission controls

3. **Rate Limiting and Request Validation**:
   - Configurable rate limiting to prevent abuse
   - Request size validation to prevent DoS attacks
   - Automatic blocking of suspicious request patterns

4. **Secure Workflow Execution**:
   - Isolation of agent states to prevent data leakage
   - Enforced agent sequencing to prevent unauthorized transitions
   - Error message sanitization to prevent sensitive data exposure

5. **Web Vulnerability Protection**:
   - Protection against XSS, SQL injection, and CSRF attacks
   - Security headers middleware
   - HTTPS enforcement in production environments

## Observability Features

WebAgent provides advanced observability features to monitor system health and performance:

1. **Detailed Diagnostics**:
   - Comprehensive system information
   - Environment configuration checks
   - Agent status monitoring
   - Network connectivity verification
   - Language model availability checks

2. **Performance Metrics**:
   - Memory usage tracking per component
   - CPU utilization monitoring
   - Execution time tracking for all agent operations
   - LLM call statistics and token usage

3. **Prometheus Integration**:
   - HTTP request metrics
   - LLM call metrics
   - Agent execution metrics
   - Resource utilization metrics

4. **LangSmith Tracing**:
   - Detailed traces of agent executions
   - Input/output recording for debugging
   - Performance profiling
   - Error tracking and reporting

## Development

### Project Structure

```
webagent/
├── backend/                  # Backend Python code
│   ├── app/                  # Main application
│   │   ├── agents/           # Agent implementations
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Core functionality
│   │   ├── graph/            # LangGraph workflows
│   │   ├── models/           # Data models
│   │   ├── services/         # Service integrations
│   │   └── utils/            # Utility functions
│   ├── tests/                # Test cases
│   └── main.py               # Application entry point
├── frontend/                 # Frontend React/Next.js code
├── scripts/                  # Utility scripts
└── docs/                     # Documentation
```

### Running Tests

```bash
cd backend
python -m pytest
```

### Adding a New Agent

1. Create a new agent file in `backend/app/agents/`
2. Implement the agent class extending `BaseAgent`
3. Add a factory function to get the agent instance
4. Update the workflow in `backend/app/graph/workflows.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain and LangGraph for agent frameworks
- OpenAI and Anthropic for LLM APIs
- FastAPI for the backend framework
- React and Next.js for the frontend framework
