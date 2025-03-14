# WebAgent Platform v2.5.3

WebAgent is a powerful AI platform that combines multiple specialized agents to perform complex tasks.

## What's New in v2.5.3

This release adds a powerful Document Extraction Agent to the WebAgent platform, enabling advanced document analysis and information extraction:

- **Document Extraction Agent**: Extract structured information from documents using multiple methods:
  - NLP-based extraction with Latent Dirichlet Allocation (LDA) Bayesian Network
  - LLM-based extraction for complex documents
  - Hybrid approach combining both methods
- **Supervisor Integration**: The Supervisor Agent can now directly call the Document Extraction Agent
- **Team Manager Routing**: Document extraction results can be routed to the Team Manager Agent for quick summarization
- **Multi-format Support**: Process PDF, DOCX, TXT, HTML, CSV, and JSON documents
- **Advanced Entity Extraction**: Identify key entities, topics, and relationships in documents

Previous Releases:

- **v2.5.2**: Added MLflow integration for experiment tracking and hyperparameter optimization
- **v2.5.1**: Improved reliability and observability of the self-hosted LLM service
- **v2.5.0**: Added self-hosted LLM support to the backend LLM service framework

## Document Extraction Features

The Document Extraction Agent offers three extraction methods:

1. **LDA-based Extraction**: Uses Latent Dirichlet Allocation with Bayesian Network techniques for topic modeling and information extraction
2. **LLM-based Extraction**: Leverages large language models for complex document understanding and structured extraction
3. **Hybrid Approach**: Combines LDA topic modeling with LLM extraction for enhanced accuracy and performance

The agent automatically selects the best method based on document characteristics or allows manual selection.

### Example Usage

```python
from backend.app.agents.supervisor import SupervisorAgent, get_supervisor_agent

# Initialize the supervisor agent (option 1: direct instantiation)
supervisor = SupervisorAgent()

# Or use the helper function (option 2: recommended for LangGraph integration)
# supervisor = get_supervisor_agent()

# Process a document with automatic method selection
result = supervisor.process_request({
    "request_type": "document_extraction",
    "document_path": "/path/to/document.pdf",
    "extraction_method": "auto",  # Options: "lda", "llm", "hybrid", "auto"
    "route_to_team_manager": True  # Optional routing to Team Manager
})

# Access extraction results
if result["success"]:
    # Access extracted information
    topics = result.get("topics", [])
    summary = result.get("summary", "")
    key_entities = result.get("key_entities", [])
    
    # Print summary
    print(f"Document Summary: {summary}")
    
    # Print top topics
    print("Top Topics:")
    for topic in topics[:3]:
        terms = [term["term"] for term in topic["terms"][:5]]
        print(f"  - Topic {topic['topic_id']}: {', '.join(terms)}")
```

## Architecture

```
┌─────────────┐    HTTP/REST API    ┌───────────────────────────────────────────┐
│ Web Chat App │<------------------>│     LangGraph Multi-Agent Service         │
│ (Frontend)   │                    │                                           │
└─────────────┘                     │  ┌─────────────┐       ┌───────────────┐  │
                                    │  │ Supervisor  │◄─────►│ Team Manager  │  │
                                    │  │   Agent     │       │    Agent      │  │
                                    │  └──────┬──────┘       └───────┬───────┘  │
                                    │         │                      │          │
                                    │         ▼                      ▲          │
                                    │  ┌──────────────┐     ┌───────┴───────┐   │
                                    │  │              │     │               │   │
                                    │  │  Research    │     │  Data         │   │
                                    │  │  Agents      │────►│  Analysis     │   │
                                    │  │              │     │  Agents       │   │
                                    │  └──────────────┘     └───────────────┘   │
                                    │     /          \         /          \     │
                                    │    /            \       /            \    │
                                    │   ▼              ▼     ▼              ▼   │
                                    │ ┌────────┐  ┌─────────┐ ┌─────────┐ ┌────┐│
                                    │ │  Web   │  │Internal │ │  Data   │ │Code││
                                    │ │Research│  │Research │ │Analysis │ │Asst││
                                    │ └────┬───┘  └────┬────┘ └────┬────┘ └──┬─┘│
                                    │      │           │           │         │  │
                                    │      └─────┐     │      ┌────┘         │  │
                                    │            ▼     ▼      ▼              │  │
                                    │          ┌─────────────────┐           │  │
                                    │          │Senior Research  │◄──────────┘  │
                                    │          │     Agent       │              │
                                    │          └─────────────────┘              │
                                    └───────────────────────────────────────────┘
                                                     │
                                     ┌───────────────┼───────────────────┐
                                     │               │                   │
                                     ▼               ▼                   ▼
                             ┌──────────────┐ ┌─────────────┐    ┌─────────────┐
                             │  Tavily API  │ │ Vector DB   │    │ Python      │
                             │ (Web Search) │ │ (Internal   │    │ Runtime     │
                             │              │ │  Knowledge) │    │ (Graphing)  │
                             └──────────────┘ └─────────────┘    └─────────────┘
```

## Features

- 💬 Modern chat interface with real-time communication
- 🔍 Advanced research capabilities via specialized AI agents
- 📊 Data analysis and visualization
- 🌐 Web research with source verification
- 📚 Internal knowledge base integration
- 📝 Comprehensive report generation
- 🧩 Microservice architecture for scalability
- 🔄 Asynchronous task processing
- 🎯 Type-safe implementation
- 🛡️ Comprehensive error handling
- 🎨 Modern UI with Tailwind CSS
- 📱 Responsive design

## Specialized Agents

The system comprises seven specialized agents that work together:

1. **Supervisor Agent**
   - Plans and orchestrates the research workflow
   - Breaks down complex queries into manageable tasks
   - Assigns tasks to specialized agents

2. **Web Research Agent**
   - Searches the web via Tavily API integration
   - Extracts relevant information from search results
   - Creates detailed reports with citations

3. **Internal Research Agent**
   - Queries the vector database for internal knowledge
   - Retrieves relevant documents and information
   - Synthesizes information into a structured report

4. **Senior Research Agent**
   - Fact-checks information from both research agents
   - Requests additional research when necessary
   - Creates a comprehensive, verified research summary

5. **Data Analysis Agent**
   - Identifies patterns and insights in research data
   - Determines appropriate visualization methods
   - Prepares data for visualization

6. **Coding Assistant Agent**
   - Creates data visualizations using Python
   - Generates charts, graphs, and other visual elements
   - Works with Data Analysis Agent for optimal visualization

7. **Team Manager Agent**
   - Compiles the final comprehensive report
   - Integrates research findings, analysis, and visualizations
   - Formats information for clarity and impact

## User Guide

### Getting Started

1. Visit the application at [your-deployed-url.com](https://your-deployed-url.com)
   - Or run locally following the installation instructions below

2. The interface consists of:
   - Chat history area (top)
   - Message input field (bottom)
   - Image upload button (bottom left)

### Basic Usage

1. **Text Chat**:
   - Type your message in the input field
   - Press Enter or click "Send"
   - Wait for the AI's response
   - Messages are automatically saved and persist between sessions

2. **Image Analysis**:
   - Click "Upload Image" button
   - Select an image (JPEG, PNG, GIF, or WebP, max 5MB)
   - Add optional text description
   - Send to get AI's analysis
   - Preview thumbnail appears with option to remove

3. **Accessibility Features**:
   - Toggle dark/light mode
   - Adjust font size (normal/large/larger)
   - Enable high contrast mode
   - Enable reduced motion
   - Full keyboard navigation support
   - Screen reader optimized

4. **Offline Support**:
   - App works offline with limited functionality
   - Previous conversations are available
   - Banner appears when offline
   - Auto-reconnects when back online

### Tips & Tricks

- **Long Conversations**: The app keeps the last 50 messages for context
- **Image Upload**: For best results, use clear images under 5MB
- **Mobile Use**: Fully responsive design works on all devices
- **Keyboard Navigation**: 
  - Tab: Navigate between elements
  - Enter: Send message
  - Space: Select buttons
  - Esc: Clear image selection

### Troubleshooting

1. **Message Not Sending**:
   - Check internet connection
   - Ensure message isn't empty
   - Check for error messages
   - Try refreshing the page

2. **Image Upload Issues**:
   - Verify file format (JPEG, PNG, GIF, WebP)
   - Ensure file is under 5MB
   - Try a different image
   - Clear browser cache

3. **Display Issues**:
   - Try toggling dark/light mode
   - Adjust font size settings
   - Clear browser cache
   - Update your browser

## Prerequisites

- Node.js 18.0.0 or later
- npm or yarn
- Together AI API key (if using Together AI as provider)
- OpenAI API key (if using OpenAI as provider)
- For self-hosted LLM: PyTorch, transformers, and a compatible model

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
npm install
```

3. Set up configuration files:

   a. For frontend API keys, create a `.env.local` file in the root directory:
   ```
   TOGETHER_API_KEY=your_together_api_key_here
   ```

   b. For backend configuration, create both `.env` and `.env.local` files in the `backend` directory:
   ```bash
   # Copy examples as starting points
   cp backend/.env.example backend/.env
   cp backend/.env.example.local backend/.env.local
   
   # Edit .env.local with your API keys
   # nano backend/.env.local
   ```

   c. Review and modify configuration as needed:
   ```bash
   # Set the environment (dev, uat, or prod)
   echo "WEBAGENT_ENV=dev" >> backend/.env
   ```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Environment Variables

### Environment Configuration

The application supports three deployment environments, each with its own configuration:

1. **Development (dev)**: Local development configuration with debug settings
2. **User Acceptance Testing (uat)**: Testing environment with near-production settings
3. **Production (prod)**: Production-ready settings optimized for performance and security

Configuration is managed through a combination of:

1. **YAML Files**: Environment-specific settings in `backend/config/{env}.yaml`
2. **Environment Variables**: Override specific settings via `WEBAGENT_` prefixed variables
3. **Default Values**: Fallback values defined in code

### Configuration Priority

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

### Selecting Environment

Set the environment using the `WEBAGENT_ENV` variable:
```bash
# Development (default)
export WEBAGENT_ENV=dev

# User Acceptance Testing
export WEBAGENT_ENV=uat

# Production
export WEBAGENT_ENV=prod
```

### Configuration Categories

Each environment defines settings for:

- **API**: Server host, port, debug mode
- **CORS**: Allowed origins, methods, headers
- **Database**: Vector DB and Redis settings
- **LLM**: Models, temperatures, timeouts
- **Web Search**: Provider, depth, result limits
- **Task Management**: Concurrency, TTL settings
- **Agents**: Model configuration for each specialized agent
- **Logging**: Log levels, formats, file settings
- **Security**: Token settings, algorithms

### Setting Up Environment Variables

1. **Local Development**:
   Create a `.env.local` file in the root directory:
   ```bash
   # Create .env.local file
   touch .env.local
   ```

2. **Add Required Variables**:
   ```env
   # Together AI API Configuration
   # Get your API key from: https://api.together.xyz/settings/api-keys
   TOGETHER_API_KEY=your_together_api_key_here
   ```

3. **Get Your API Key**:
   - Visit [Together AI Dashboard](https://api.together.xyz/settings/api-keys)
   - Sign up or log in
   - Navigate to API Keys section
   - Create a new API key
   - Copy the key

4. **Security Notes**:
   - Never commit `.env.local` to version control
   - Keep your API key secret
   - Rotate keys periodically
   - Use different keys for development and production

### Production Deployment

For production deployment, set environment variables in your hosting platform:

- **Vercel**:
  - Go to Project Settings → Environment Variables
  - Add `TOGETHER_API_KEY` with your production API key

- **Other Platforms**:
  - Follow platform-specific instructions for setting environment variables
  - Ensure variables are encrypted at rest

### Accessing Environment Variables

The API key is automatically loaded and used in the application. You can verify it's working by:

1. Starting the development server
2. Sending a test message in the chat
3. Checking the network tab for successful API calls

If you get authentication errors, verify that:
- `.env.local` file exists in project root
- API key is correctly copied
- No extra spaces or quotes in the key
- Server was restarted after adding the key

## Project Structure

```
src/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   └── page.tsx           # Main page
├── components/            # React components
│   ├── Chat.tsx          # Chat interface
│   └── ErrorBoundary.tsx # Error handling
├── config/               # Configuration
│   └── env.ts           # Environment config
└── types/               # TypeScript types
    ├── api.ts          # API types
    └── chat.ts         # Chat types
```

## API Integration

The application uses Together AI's API for:
- Text chat completion
- Image analysis
- Multi-modal conversations

## Error Handling

- Custom ApiError class for API errors
- Error Boundary for React component errors
- Form validation
- File upload validation
- Environment variable validation

## Security

- File type validation
- File size limits (5MB max)
- Environment variable protection
- Input sanitization
- API error handling

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Next.js](https://nextjs.org/)
- [Together AI](https://www.together.ai/)
- [Tailwind CSS](https://tailwindcss.com/)
- [TypeScript](https://www.typescriptlang.org/)

## LLM Providers

WebAgent supports multiple LLM providers that can be configured based on your needs:

1. **Together AI** (default)
   - Uses Together AI's hosting of Llama 3 models
   - Requires a Together AI API key
   - Offers a good balance of quality and cost

2. **OpenAI**
   - Uses OpenAI's GPT models 
   - Requires an OpenAI API key
   - Provides high quality but at higher cost

3. **Self-Hosted LLM** (new in v2.5.0)
   - Run your own local language model
   - No API keys required
   - Complete data privacy
   - Lower latency for some deployments
   - See [Self-Hosted LLM Setup](#self-hosted-llm-setup) below

### Self-Hosted LLM Setup

New in version 2.5.0 (improved in 2.5.1), WebAgent can now use your own locally-hosted language models:

1. **Configure WebAgent to use self-hosted LLM**:
   
   Edit your environment YAML file (e.g., `backend/config/dev.yaml`):
   ```yaml
   llm:
     provider: "self"  # Change from "together" or "openai" to "self"
     self_hosted_url: "http://localhost:8080"  # Update if running elsewhere
   ```

2. **Start the self-hosted LLM service**:
   
   Basic usage:
   ```bash
   python backend/app/services/self_hosted_llm_service.py --model_path "/path/to/model" --port 8080
   ```
   
   With memory optimization (new in v2.5.1):
   ```bash
   # Use 8-bit quantization (CUDA only)
   python backend/app/services/self_hosted_llm_service.py --model_path "/path/to/model" --load_in_8bit
   
   # Use 4-bit quantization for larger models (CUDA only)
   python backend/app/services/self_hosted_llm_service.py --model_path "/path/to/model" --load_in_4bit
   ```
   
   Docker deployment:
   ```bash
   # Build the container
   docker build -f backend/app/services/Dockerfile.llm -t webagent-llm:latest .
   
   # Run the container, mounting your model directory
   docker run -p 8080:8080 \
     -v /path/to/model:/models/llama-3 \
     -e MODEL_PATH=/models/llama-3 \
     webagent-llm:latest
   ```

3. **Monitor the service** (new in v2.5.1):
   
   Check health status:
   ```bash
   curl http://localhost:8080/health
   ```
   
   View detailed metrics:
   ```bash
   curl http://localhost:8080/metrics
   ```

4. **Recommended models**:
   - Llama 3 8B Instruct
   - Mistral-7B-Instruct-v0.2
   - Any model compatible with Hugging Face transformers

5. **Advanced configuration**:
   
   For more options, including model quantization, custom prompt formats, and Docker configuration, see the [detailed self-hosted LLM documentation](backend/app/services/README_SELF_HOSTED_LLM.md).

### MLflow Integration for Model Experimentation (New)

The self-hosted LLM service now includes MLflow integration for tracking experiments, fine-tuning and hyperparameter optimization:

1. **Setup MLflow**:

   ```bash
   # Install MLflow (already included in requirements-llm.txt)
   pip install mlflow scikit-learn
   
   # Start MLflow tracking server
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000
   ```

2. **Start the LLM service with MLflow enabled**:

   ```bash
   python backend/app/services/self_hosted_llm_service.py \
     --model_path "/path/to/model" \
     --mlflow_tracking_uri "http://localhost:5000" \
     --mlflow_experiment_name "llm-experiments"
   ```

3. **Track experiments in API calls**:
   
   Include experiment tracking in your requests:
   ```json
   {
     "messages": [{"role": "user", "content": "Your prompt here"}],
     "temperature": 0.7,
     "track_experiment": true,
     "experiment_name": "temperature-comparison",
     "run_name": "temp-0.7-test"
   }
   ```

4. **View experiments**:
   
   Access the MLflow UI at http://localhost:5000 to:
   - Compare different model configurations
   - Analyze performance metrics
   - View generated responses
   - Find optimal hyperparameters

5. **Automated hyperparameter tuning**:
   
   The integration supports systematic optimization of:
   - Temperature
   - Top-p sampling
   - Frequency/presence penalties
   - Model quantization settings

6. **Detailed documentation**:
   
   For comprehensive instructions on MLflow integration, experiment tracking, and hyperparameter optimization, see [MLflow Integration for Self-Hosted LLM](backend/app/services/MLFLOW_INTEGRATION.md).
