# Deploying WebAgent to Vercel

[![version](https://img.shields.io/badge/version-2.9.5-blue.svg)](https://semver.org)

This guide explains how to deploy the WebAgent application to Vercel, including both the Next.js frontend and the Python backend functions.

## Prerequisites

- A [Vercel](https://vercel.com) account
- [Vercel CLI](https://vercel.com/cli) installed (optional for advanced usage)
- [Together AI API Key](https://together.ai)

## Deployment Steps

### 1. Set up Environment Variables

Before deployment, you need to set up the following environment variables in your Vercel project:

- `TOGETHER_API_KEY`: Your Together AI API key (used by `/api/chat`)
- `WEBAGENT_BACKEND_URL`: URL of the FastAPI backend that runs the multi-agent workflow (used by `/api/task`)
- `WEBAGENT_API_KEY`: API key the FastAPI backend expects when API-key auth is on (sent server-side only)

You can add these in the Vercel dashboard under Project Settings > Environment Variables.
`vercel.json` no longer references `@secret` names; set the variables directly in project settings.

### 2. Deploy to Vercel

**Option 1: Deploy from the Dashboard**

1. Import your repository in the [Vercel Dashboard](https://vercel.com/new)
2. Configure your project settings
3. Click "Deploy"

**Option 2: Deploy from CLI**

```bash
# Install Vercel CLI if not already installed
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel
```

### 3. Configuration Details

The deployment uses:

- `vercel.json` - Builds the Next.js app (all `/api/*` routes are served by Next.js)
- `next.config.js` - Configures Next.js behavior
- `src/app/api/chat` - Direct LLM chat with the same system prompts as development
- `src/app/api/task` - Proxies agentic requests to the FastAPI backend at `WEBAGENT_BACKEND_URL`

### 4. Verifying Your Deployment

After deployment:

1. Visit your deployed site
2. Test the API by sending a message through the chat interface
3. Monitor logs in the Vercel dashboard to identify any issues

### 5. Switching to Full-Featured Backend (Optional)

For production deployments requiring the full backend capabilities:

1. Run the FastAPI backend (`python -m backend.main`) on a server you control
2. Set `WEBAGENT_BACKEND_URL` (and `WEBAGENT_API_KEY` if auth is on) in Vercel

## Troubleshooting

- **API Errors**: Check Vercel Function logs for details
- **Environment Variables**: Ensure all required variables are properly set
- **Cold Start**: The first request may take longer due to serverless cold start

## Local Development

To develop locally:

```bash
# Install dependencies
npm install

# Start the Next.js development server
npm run dev

# In a separate terminal, start the FastAPI backend (port 8000)
python -m backend.main
```