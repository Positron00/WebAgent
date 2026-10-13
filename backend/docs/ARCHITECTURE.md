# WebAgent Architecture Documentation

This document provides a comprehensive overview of the WebAgent platform architecture, detailing the components, interactions, and design decisions.

## Overview

WebAgent is a sophisticated AI system built as a microservice architecture that combines a modern Next.js frontend with a powerful LangGraph-based multi-agent backend. The platform enables complex research, analysis, and reporting through specialized AI agents.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WebAgent Platform v3.2.4                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────┐                         ┌─────────────────────────────┐ │
│  │                │      REST API           │                             │ │
│  │  Frontend      │◄────────────────────────┤  Backend API Layer          │ │
│  │  Next.js       │                         │  FastAPI + Middleware       │ │
│  │  React         │─────────────────────────►  • Security                 │ │
│  │  TailwindCSS   │                         │  • Rate Limiting            │ │
│  │                │                         │  • Metrics                  │ │
│  └────────────────┘                         └───────────────┬─────────────┘ │
│           │                                                 │               │
│           │        ┌──────────────────┐                     │               │
│           └────────┤ State Management │                     │               │
│                    └──────────────────┘                     ▼               │
│                                                  ┌─────────────────────────┐ │
│  ┌────────────────────────────────────────────┐ │                         │ │
│  │             Monitoring & Observability     │ │  Task Manager           │ │
│  │  • Prometheus Metrics                      │ │  • Async Processing     │ │
│  │  • Grafana Dashboards                      │ │  • Task State Storage   │ │
│  │  • Structured Logging                      │ │  • Result Caching       │ │
│  │  • LangSmith Tracing                       │ │                         │ │
│  └──────────────────┬─────────────────────────┘ └───────────┬─────────────┘ │
│                     │                                       │               │
│                     ▼                                       ▼               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       LangGraph Orchestration                          │ │
│  │                                                                        │ │
│  │  ┌─────────┐        ┌─────────────┐         ┌────────────┐            │ │
│  │  │         │        │             │         │            │            │ │
│  │  │Supervisor◄───────┤Team Manager ◄─────────┤Senior      │            │ │
│  │  │Agent    │        │Agent        │         │Research    │            │ │
│  │  │         ├──┐     │             │         │Agent       │            │ │
│  │  └─────────┘  │     └─────────────┘         └────────────┘            │ │
│  │               │            ▲                      ▲                    │ │
│  │               │            │                      │                    │ │
│  │               ▼            │                      │                    │ │
│  │  ┌─────────────────────────┴──────────────────────┴───────────────┐   │ │
│  │  │                                                                │   │ │
│  │  │  ┌─────────────┐    ┌─────────────┐     ┌────────────┐        │   │ │
│  │  │  │             │    │             │     │            │        │   │ │
│  │  │  │Web Research │    │Internal     │     │Data        │        │   │ │
│  │  │  │Agent        │    │Research     │     │Analysis    │        │   │ │
│  │  │  │             │    │Agent        │     │Agent       │        │   │ │
│  │  │  └─────────────┘    └─────────────┘     └────────────┘        │   │ │
│  │  │         │                  │                   │               │   │ │
│  │  │         └──────────────────┴───────────────────┘               │   │ │
│  │  │                            │                                   │   │ │
│  │  └────────────────────────────┼───────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                  │                                           │
│    ┌─────────────────────────────┼───────────────────────────────────────┐   │
│    │                             │  External Integrations                │   │
│    │                             ▼                                       │   │
│    │  ┌──────────────┐    ┌─────────────┐    ┌─────────────────┐        │   │
│    │  │              │    │             │    │                 │        │   │
│    │  │ Together AI  │    │ OpenAI API  │    │ Tavily Search   │        │   │
│    │  │ Llama 3.3    │    │ GPT-4       │    │                 │        │   │
│    │  │              │    │             │    │                 │        │   │
│    │  └──────────────┘    └─────────────┘    └─────────────────┘        │   │
│    └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Architecture Components

### 1. Frontend Layer

The frontend provides a user-friendly interface for interacting with the WebAgent platform:

- **Technology Stack**:
  - Next.js 14.2+ with React 18
  - TailwindCSS for responsive styling
  - TypeScript for type safety
  - Axios for API requests
  - React Context API for state management

- **Key Features**:
  - Responsive chat interface
  - Real-time updates of task status
  - File upload support
  - Markdown rendering
  - Error handling and retry mechanisms
  - Offline support
  - Accessibility features

### 2. Backend API Layer

The backend API layer handles all incoming requests and provides endpoints for the frontend:

- **Technology Stack**:
  - FastAPI for high-performance async API
  - Middleware components for security and metrics
  - Pydantic for request/response validation
  - JWT for authentication

- **Key Components**:
  - API router for endpoint organization
  - Rate limiting middleware
  - Security headers middleware
  - Input validation and sanitization
  - Error handling with structured logging
  - Health check endpoints
  - Prometheus metrics

### 3. Task Management

The task management system handles the asynchronous processing of user requests:

- **Key Features**:
  - Task creation and tracking
  - State persistence
  - Result caching
  - Progress updates
  - Concurrent task handling
  - Error recovery

- **Implementation**:
  - FastAPI background tasks run each workflow in the server process
  - Each task's status is saved to `backend/data/results/<task_id>.json` when it is created and when it finishes, so every server worker can answer polls and finished reports survive a restart
  - A task still "processing" after `WEBAGENT_TASK_TIMEOUT_MINUTES` (30) is reported as interrupted
  - UUID task IDs (only UUIDs are accepted, so an ID can't point outside the results folder)
  - Each task runs one of two paths: the LangGraph workflow (`run_workflow`) or a direct path (`run_direct`) that makes one text-only model call without building the graph. Results record which one ran (`route`, `decided_by`)

### Agentic triage

The web app's Agentic control has three modes: Off, Auto and Always. In Auto, a question goes to the research workflow only when it needs it.

- **Who decides:** the backend classifies (`POST /api/triage`, `app/services/triage.py`); the web app routes. A direct answer goes through the web app's `/api/chat`, which keeps the user's prompt settings and needs no polling; a research answer goes through `/api/task`.
- **How:** rules first (an image or a greeting → direct; "research …", "write a report …" → research), then frozen `all-MiniLM-L6-v2` embeddings and a logistic regression (`models/triage/`, about 10 ms on CPU). If the classifier can't load, the answer is research.
- **API clients:** `/api/task` and `/api/v1/chat/completions` take `mode: off | auto | always` and use the same classifier. No mode and no `agentic` flag means `off`.
- **Learning:** shadow mode and "Research this in depth" clicks can save questions for review (`data/triage/collected/`, off outside dev, deleted after 30 days). A new classifier version ships only if it passes the release gate on a held-out test set; uat/prod keep triage off until one does.
- Design: `planning/specs/agentic-triage/`.

### 4. LangGraph Orchestration

The core of the WebAgent platform is the LangGraph-based agent orchestration system:

- **Key Components**:
  - Workflow definitions
  - Agent state management
  - Routing logic
  - Error handling and retries
  - Tracing with LangSmith

- **Implementation**:
  - LangGraph for agent orchestration
  - Directed graph for agent interactions
  - State transitions with validation
  - Conditional routing based on agent outputs

### 5. Agent Implementation

The system includes seven specialized agents:

- **Supervisor Agent**:
  - Analyzes user queries
  - Plans research approach
  - Coordinates other agents
  - Determines research needs

- **Web Research Agent**:
  - Performs web searches via Tavily
  - Extracts relevant information
  - Cites sources properly
  - Formats findings for other agents

- **Internal Research Agent**:
  - Queries internal knowledge base
  - Retrieves relevant documents
  - Performs semantic search
  - Synthesizes internal knowledge

- **Senior Research Agent**:
  - Verifies and fact-checks information
  - Resolves conflicting information
  - Requests additional research when needed
  - Creates comprehensive summaries

- **Data Analysis Agent**:
  - Identifies patterns in research data
  - Determines appropriate visualizations
  - Prepares data for presentation
  - Generates insights and conclusions

- **Coding Assistant Agent**:
  - Creates data visualizations
  - Writes Python code for analysis
  - Generates charts and graphs
  - Ensures code correctness

- **Team Manager Agent**:
  - Compiles final reports
  - Integrates findings from all agents
  - Ensures report quality and completeness
  - Formats information for readability

### 6. External Integrations

The system integrates with several external services:

- **Language Models**:
  - Together AI (Llama 3.3 70B Instruct Turbo Free) - primary
  - OpenAI API (GPT-4) - fallback
  - Provider abstraction for easy switching

- **Web Search**:
  - Tavily Search API for web research
  - Configurable depth and result count

- **Knowledge Storage**:
  - Vector database for semantic search
  - Document storage and retrieval
  - Embedding generation

### 7. Monitoring & Observability

Comprehensive monitoring and observability features:

- **Metrics**:
  - Prometheus for metrics collection
  - Request counts and latencies
  - LLM usage statistics
  - Task processing metrics
  - Custom business metrics

- **Logging**:
  - Structured JSON logging
  - Log levels based on environment
  - Request ID tracking
  - Error context capture
  - Rotation and retention policies

- **Tracing**:
  - LangSmith integration for LLM tracing
  - Agent interaction tracing
  - Performance monitoring
  - Error analysis

- **Health Checks**:
  - Basic and detailed health endpoints
  - Dependency status reporting
  - Self-diagnostics

## Communication Flow

The typical flow of a user request through the system:

1. User submits a query through the frontend interface
2. Frontend sends request to backend API
3. Backend validates request and creates a new task
4. Task Manager initiates asynchronous processing
5. Supervisor Agent analyzes query and plans research approach
6. Research Agents (Web and Internal) gather information in parallel
7. Senior Research Agent verifies and synthesizes findings
8. Data Analysis Agent processes information and identifies insights
9. Coding Assistant generates visualizations if needed
10. Team Manager creates final comprehensive report
11. Result is stored in Task Manager
12. Frontend polls for task completion
13. Result is displayed to user with formatting

## Error Handling & Resilience

The system includes comprehensive error handling and resilience features:

- **Rate Limiting**:
  - Per-caller limit (60 requests per minute by default), counted in memory per server worker
  - Behind the web app's proxy, the caller is the end user (`X-Forwarded-For` from `WEBAGENT_TRUSTED_PROXIES`)
  - Health checks and `/metrics` are never limited; a limited request gets `429` with `Retry-After`

- **Circuit Breakers**:
  - Automatic detection of service failures
  - Graceful degradation
  - Fallback mechanisms

- **LLM Fallbacks**:
  - Provider switching on failure
  - Retry mechanisms with backoff
  - Model parameter adjustment
  - Per-agent output limits (`agents.<name>.max_tokens`), with a logged warning when a reply is cut off at its limit

- **Request Validation**:
  - Comprehensive input validation
  - Size limits
  - Content sanitization
  - Type checking

- **Monitoring Alerts**:
  - Error rate thresholds
  - Performance degradation detection
  - Resource utilization alerts

## Deployment Architecture

The system supports multiple deployment options:

- **Docker-based Deployment**:
  - Multi-stage build on Python 3.12, pinned by `backend/constraints.txt`; runs as `WEBAGENT_ENV=prod` as a non-root user
  - NLTK data installed at build time; `GET /health` as the health check
  - Docker Compose for local deployment (backend, Prometheus, Grafana)
  - A separate sandbox image (`backend/sandbox`) for running model-written code with no network

- **Kubernetes Deployment**:
  - Namespace isolation
  - Deployment configurations
  - Service definitions
  - Ingress rules
  - Resource requests and limits
  - Read-only root filesystem with `emptyDir` volumes for `/app/data` and `/app/logs`
  - One replica until `/app/data` is on a shared ReadWriteMany volume

- **Scaling Strategy**:
  - Horizontal scaling for API layer
  - Vertical scaling for compute-intensive agents
  - Caching for repeated requests
  - Resource optimization

## Configuration Management

The system uses a hierarchical configuration system:

- **Environment-based Configuration**:
  - Development (dev)
  - User Acceptance Testing (uat)
  - Production (prod)

- **Configuration Sources** (in priority order):
  - `.env.local` for sensitive information (but `WEBAGENT_ENV` set in the shell or container wins)
  - Environment variables with `WEBAGENT_` prefix (`WEBAGENT_SECURITY_*` override the YAML security block)
  - YAML files by environment
  - Default values in code

- **Key Configuration Categories**:
  - API settings
  - CORS configuration
  - Database connections
  - LLM parameters
  - Security settings
  - Logging configuration
  - Rate limits

## Security Considerations

The system implements multiple security measures:

- **API Security**:
  - Rate limiting to prevent abuse
  - JWT token-based authentication
  - Request size limiting
  - Secure password hashing

- **HTTP Security**:
  - Security headers middleware
  - CORS protection
  - HTTPS enforcement
  - Content security policy

- **Data Protection**:
  - Input sanitization
  - Parameter validation
  - API key protection
  - Environment variable isolation

- **Infrastructure Security**:
  - Container isolation
  - Least privilege principles
  - Dependency scanning
  - Regular updates

## Performance Optimization

Performance is optimized through several techniques:

- **Caching**:
  - Result caching for repeated queries
  - Token count caching
  - Agent state caching

- **Parallel Processing**:
  - Concurrent agent execution
  - Asynchronous API handlers
  - Background task processing

- **Resource Management**:
  - Memory usage monitoring
  - Task prioritization
  - Request throttling under load

## Future Roadmap

Planned enhancements to the architecture:

1. **Enhanced LLM Integration**:
   - Support for more providers
   - Local model deployment options
   - Hybrid approaches

2. **Advanced Caching**:
   - Semantic caching for similar queries
   - Distributed cache with Redis
   - Partial result caching

3. **Improved Scalability**:
   - Kafka-based event system
   - Stateless agent design
   - Distributed tracing

4. **Enhanced Security**:
   - Fine-grained permissions
   - API key rotation
   - Advanced threat detection

5. **User Personalization**:
   - User profiles
   - Customized research preferences
   - History-aware responses

## Conclusion

The WebAgent architecture provides a robust, scalable, and maintainable platform for AI-powered research and analysis workflows. The multi-agent approach allows for complex reasoning and task handling, while the microservice design enables independent scaling and development of components. 