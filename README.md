# WebAgent Platform

A comprehensive multi-agent platform for web research, document processing, and knowledge synthesis.

![WebAgent Platform Version](https://img.shields.io/badge/version-2.5.11-blue)
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

- **LangGraph Integration**: Uses LangGraph for complex agent workflow orchestration with optimized direct edge routing, comprehensive state management, and proper error handling throughout the workflow, compatible with LangGraph 0.0.25+

- **Advanced RAG Architecture**: Sophisticated retrieval and processing with built-in document extraction capabilities, enhanced BM25 lexical retrieval for improved results

- **Document Processing**: Extract, analyze, and synthesize information from PDFs, web pages, and other document formats

- **Research Loop Process**: Iterative research capability that evaluates quality and performs additional research as needed:
  1. Initial Research: Web and internal knowledge sources are queried based on the research plan
  2. Research Evaluation: Senior Research Agent evaluates completeness and quality (scoring 1-10)
  3. Follow-up Research: Additional targeted research based on identified gaps
  4. Iteration: Process repeats up to 3 times until quality threshold is met
  5. Final Synthesis: Comprehensive findings consolidated into a structured report

- **Comprehensive Security**: Advanced security module with authentication, authorization, rate limiting, input validation, error sanitization, request size limiting, and protection against common vulnerabilities

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
   - Compatibility with LangGraph 0.0.25 and higher

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
   - Enhanced BM25 lexical search for better keyword matching

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
   - Comprehensive error tracking and reporting

7. **Security Module**: Comprehensive security features
   - JWT authentication and API key authorization
   - Input validation and sanitization
   - Rate limiting and request size validation
   - Error message sanitization to prevent data leakage
   - Protection against common web vulnerabilities
   - Secure configuration management
   - Comprehensive middleware for request protection

### Security Architecture

WebAgent implements a multi-layered security architecture to protect against various threats:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             Security Architecture                                    │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                          Request Processing Pipeline                          │   │
│  │                                                                              │   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────────────┐   │   │
│  │  │ Rate      │    │ Request   │    │ Input     │    │                   │   │   │
│  │  │ Limiting  │───►│ Size      │───►│ Validation│───►│  Authentication   │   │   │
│  │  │ Middleware│    │ Validation│    │ Middleware│    │  & Authorization  │   │   │
│  │  └───────────┘    └───────────┘    └───────────┘    └───────────────────┘   │   │
│  │        │                                                      │              │   │
│  │        ▼                                                      ▼              │   │
│  │  ┌────────────────┐                                   ┌───────────────────┐ │   │
│  │  │                │                                   │                   │ │   │
│  │  │  Rejected if   │                                   │    JWT Token      │ │   │
│  │  │  too many      │                                   │    Verification   │ │   │
│  │  │  requests      │                                   │                   │ │   │
│  │  └────────────────┘                                   └───────────────────┘ │   │
│  │                                                              │              │   │
│  │                                                              ▼              │   │
│  │                                                      ┌───────────────────┐ │   │
│  │                                                      │                   │ │   │
│  │                                                      │    API Key        │ │   │
│  │                                                      │    Validation     │ │   │
│  │                                                      │                   │ │   │
│  │                                                      └───────────────────┘ │   │
│  │                                                                            │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                             │
│                                       ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Protected Resources                             │   │
│  │                                                                              │   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────────────┐   │   │
│  │  │ API       │    │ Agent     │    │ Document  │    │ Knowledge Base    │   │   │
│  │  │ Endpoints │    │ Workflows │    │ Processing│    │ Access Controls   │   │   │
│  │  │           │    │           │    │           │    │                   │   │   │
│  │  └───────────┘    └───────────┘    └───────────┘    └───────────────────┘   │   │
│  │                                                                              │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                           Error Handling & Logging                           │   │
│  │                                                                              │   │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐    │   │
│  │  │ Error Message │    │ Security      │    │ Comprehensive Audit       │    │   │
│  │  │ Sanitization  │    │ Event Logging │    │ Trail                     │    │   │
│  │  │               │    │               │    │                           │    │   │
│  │  └───────────────┘    └───────────────┘    └───────────────────────────┘    │   │
│  │                                                                              │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

This security architecture ensures:
- Prevention of DoS attacks through rate limiting
- Protection against oversized payloads
- Input validation to prevent injection attacks
- Authentication via JWT tokens or API keys
- Proper error handling without leaking sensitive data
- Comprehensive logging for security events
- Clear separation between authentication, authorization, and resource access

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

## Installation and Setup

Follow these steps to set up the WebAgent platform on your local environment:

### Prerequisites

- Python 3.10+ 
- Node.js 18+ (for frontend)
- Conda (recommended for environment management)
- Git

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourorg/WebAgent.git
   cd WebAgent
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n agents python=3.12
   conda activate agents
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp backend/config/example.yaml backend/config/dev.yaml
   ```
   Edit the `dev.yaml` file to set your API keys, database connections, and other configuration parameters.

### Frontend Setup (Optional)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

## Running WebAgent

### Running the Backend

1. Ensure your conda environment is activated:
   ```bash
   conda activate agents
   ```

2. Start the backend server:
   ```bash
   python -m backend.main
   ```

   The server will start on http://localhost:8000 by default, with API endpoints available at http://localhost:8000/api/v1/

3. Access the OpenAPI documentation:
   Open your browser and navigate to http://localhost:8000/docs to view the API documentation.

### Running the Frontend (Optional)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at http://localhost:3000 (or port 3001 if 3000 is already occupied).

### Using WebAgent

1. API Usage Example:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -d '{"query": "What are the benefits of multi-agent AI systems?"}'
   ```

2. Programmatic Usage:
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:8000/api/v1/chat/completions",
       headers={"Authorization": "Bearer YOUR_API_KEY"},
       json={"query": "What are the benefits of multi-agent AI systems?"}
   )
   
   print(response.json())
   ```

## Testing and Diagnostics

WebAgent includes a comprehensive test suite and diagnostic tools to ensure everything is working correctly.

### Running Tests

1. Run the LangGraph workflow tests:
   ```bash
   python -m backend.tests.test_langgraph
   ```

2. Run the security tests:
   ```bash
   python -m backend.tests.test_security
   ```

3. Run all tests with pytest:
   ```bash
   pytest backend/tests/
   ```

### Diagnosing Issues

1. Check the logs:
   Logs are stored in the `logs/` directory. You can examine them for error messages and debugging information:
   ```bash
   tail -f logs/webagent-dev.log
   ```

2. Enable debug mode:
   Set `DEBUG: true` in your `dev.yaml` configuration file to enable more verbose logging.

3. Use LangSmith for tracing:
   WebAgent is integrated with LangSmith for tracing and debugging agent workflows. Configure your LangSmith API key in the `dev.yaml` file:
   ```yaml
   LANGSMITH:
     API_KEY: your_langsmith_api_key
     PROJECT: webagent-research
   ```

4. Monitoring API endpoints:
   The system health can be checked via the `/health` endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

5. Performance monitoring:
   If you've enabled Prometheus metrics, they're available at:
   ```bash
   curl http://localhost:8000/metrics
   ```

### Common Issues and Solutions

1. **Import Errors**: Ensure your conda environment is activated and all dependencies are installed.

2. **API Key Errors**: Verify that all required API keys are correctly set in your configuration file.

3. **LangGraph Workflow Issues**: If you're experiencing issues with the agent workflow:
   - Check that your LangGraph version matches the required version (0.3.5+)
   - Verify that all agents are correctly initialized
   - Look at the workflow state transitions in the logs

4. **Memory or Performance Issues**: 
   - Adjust the chunk sizes for document processing in the configuration
   - Reduce the maximum tokens for LLM calls
   - Implement proper caching for API responses

5. **Security Test Failures**:
   - Ensure that middleware is correctly configured
   - Check that rate limiting is properly set up
   - Verify that authentication is correctly implemented

For more detailed diagnostics, use the built-in observability features and consult the API documentation.

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
