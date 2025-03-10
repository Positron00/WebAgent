# WebAgent: Multi-Agent AI Research and Analysis Platform

A sophisticated AI system built as a microservice architecture that combines a modern Next.js frontend with a powerful LangGraph-based multi-agent backend. The platform enables complex research, analysis, and reporting through specialized AI agents.

**Version: 2.4.6** - Fixed Jest configuration issues for reliable test execution. The update resolves environment teardown problems, properly excludes non-test files from testing, and ensures all 27 tests pass consistently. Also fixed Next.js font loading conflicts with Babel configuration and resolved client component issues. This maintenance release improves developer experience with more reliable testing infrastructure and build process.

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
- Together AI API key

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
