# WebAgent Platform

A comprehensive multi-agent platform for web research, document processing, and knowledge synthesis.

![WebAgent Platform Version](https://img.shields.io/badge/version-2.6.2-blue)
![Last Updated](https://img.shields.io/badge/last%20updated-2025--03--14-brightgreen)

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

- **LangGraph Integration**: Uses LangGraph for complex agent workflow orchestration with optimized direct edge routing, comprehensive state management, and proper error handling throughout the workflow, compatible with LangGraph 0.0.25+

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

- **Advanced Observability**: Detailed diagnostics and monitoring capabilities for tracking system health, performance metrics, and execution statistics

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
   - Compatibility with LangGraph 0.0.25 and higher

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
   - LLM provider integration (OpenAI, Anthropic)
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
- Conda for environment management
- Git for version control

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
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```bash
   cp backend/config/example.yaml backend/config/dev.yaml
   # Edit dev.yaml with your configuration
   ```

### Running the Application

1. Start the main application:
   ```bash
   python -m backend.main
   ```

2. Run Model Services (optional):
   ```bash
   # Start the model manager
   python -m backend.models.run_manager
   
   # Start a specific model service
   python -m backend.models.run_service --model-id llm-model --model-type transformer --model-path /path/to/model
   ```

3. Access the application:
   - Web UI: http://localhost:3000
   - API: http://localhost:8000/api/v1
   - Swagger UI: http://localhost:8000/docs
   - Model API Gateway: http://localhost:8000/model-api

## Testing and Diagnostics

WebAgent includes comprehensive testing and diagnostic tools.

### Running Tests

1. Run all tests:
   ```bash
   python -m unittest discover backend/tests
   ```

2. Run model service tests:
   ```bash
   # Run model registry and API gateway tests
   python -m backend.tests.test_models
   
   # Run model service implementation tests
   python -m backend.tests.test_model_services
   
   # Run model manager tests (simplified version)
   python -m backend.tests.test_model_manager_simplified
   ```

3. Run diagnostics:
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
   curl http://localhost:8000/model-api/metrics
   ```

3. View logs:
   ```bash
   tail -f logs/webagent-dev.log
   ```

4. View LangSmith traces (if configured):
   - Visit LangSmith at https://smith.langchain.com
   - Filter by project "webagent-research"

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

## Contributing

We welcome contributions to the WebAgent platform. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain and LangGraph for agent frameworks
- OpenAI and Anthropic for LLM APIs
- FastAPI for the backend framework
- React and Next.js for the frontend framework
