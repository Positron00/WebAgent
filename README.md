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

## System Architecture

WebAgent uses a modular architecture with a clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       WebAgent Platform                             │
│                                                                     │
│  ┌───────────┐   ┌─────────────────────────────────────────────┐   │
│  │           │   │            LangGraph Workflow               │   │
│  │  FastAPI  │   │  ┌─────────┐   ┌────────────┐   ┌────────┐  │   │
│  │  Server   │◄──┼─►│Supervisor│──►│ Research  │──►│ Senior │  │   │
│  │           │   │  │  Agent   │   │  Agents   │   │Research│  │   │
│  └───────────┘   │  └─────────┘   └────────────┘   │  Agent │  │   │
│        ▲         │        │              ▲         └────┬───┘  │   │
│        │         │        │              │              │      │   │
│        │         │        ▼              └──────────────┘      │   │
│        │         │  ┌─────────────┐             │              │   │
│        │         │  │  Document   │             ▼              │   │
│        │         │  │ Extraction  │      ┌────────────┐        │   │
│        │         │  │    Agent    │      │ Specialized│        │   │
│  ┌───────────┐   │  └─────────────┘      │   Agents   │        │   │
│  │           │   │         │             └─────┬──────┘        │   │
│  │ Frontend  │   │         │                   │               │   │
│  │  Client   │◄──┼─────────┼───────────────────┼───────────────┤   │
│  │           │   │         │                   │               │   │
│  └───────────┘   │         ▼                   ▼               │   │
│                  │  ┌────────────────────────────────┐         │   │
│                  │  │        Team Manager Agent      │         │   │
│                  │  └────────────────────────────────┘         │   │
│                  └─────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **API Layer**: FastAPI server exposing REST endpoints for client applications
2. **Workflow Engine**: LangGraph orchestration of agent interactions
3. **Agents**: Specialized autonomous agents each with a specific role
4. **Document Processing Pipeline**: For extracting and processing document content
5. **Integration Layer**: Connectors to external knowledge sources and APIs
6. **Monitoring & Diagnostics**: Tools for system monitoring and performance analysis

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

### Using the CLI

The WebAgent platform includes a command-line interface for running tasks:

```bash
# Run diagnostics
python scripts/run_webagent.py --mode diagnostics

# Run a workflow
python scripts/run_webagent.py --mode workflow --query "Research the impact of AI on healthcare"

# Extract data from a document
python scripts/run_webagent.py --mode direct --type document_extraction --document path/to/document.pdf
```

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
