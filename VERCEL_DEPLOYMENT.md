# Deploying WebAgent to Vercel

[![version](https://img.shields.io/badge/version-3.1.1-blue.svg)](https://semver.org)

This guide explains how to deploy the WebAgent web app (Next.js) to Vercel. Vercel runs only the web app; the multi-agent FastAPI backend runs on a server you control, and the web app reaches it through `/api/task`.

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
- `WEBAGENT_DEFAULT_AGENTIC_MODE` (optional): Agentic mode for people who haven't picked one: `off` (default), `auto` or `always`. Vercel has no `.env.local` on disk, so a change needs a redeploy
- `WEBAGENT_LLM_DEFAULT_MODEL` (optional): Together model for `/api/chat`; defaults to `meta-models/Muse-Glimmer-30B`

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
- `src/app/api/triage` - Proxies Auto-mode triage to the same backend
- `src/app/api/settings` - Serves the default Agentic mode

### 4. Verifying Your Deployment

After deployment:

1. Visit your deployed site
2. Test the API by sending a message through the chat interface
3. Monitor logs in the Vercel dashboard to identify any issues

### 5. Switching to Full-Featured Backend (Optional)

For production deployments requiring the full backend capabilities:

1. Run the FastAPI backend on a server you control (`python -m backend.main`, or the Docker image described in [backend/README.md](backend/README.md#running-with-docker)) with `WEBAGENT_ENV=prod`
2. Set `WEBAGENT_BACKEND_URL` and `WEBAGENT_API_KEY` in Vercel (the backend requires the key in prod)
3. Rate limits: the `/api/task` proxy forwards each user's address in `X-Forwarded-For`, but the backend trusts that header only from addresses in `WEBAGENT_TRUSTED_PROXIES` (default: localhost). Vercel's outgoing addresses change, so by default every Vercel user shares one budget of `WEBAGENT_SECURITY_RATE_LIMIT_MAX_REQUESTS` (60) per minute. Raise that limit, or put the backend behind a proxy with fixed addresses and list them in `WEBAGENT_TRUSTED_PROXIES`
4. Check the backend from outside with `curl https://<backend-host>/health`

## Troubleshooting

- **API Errors**: Check Vercel Function logs for details
- **Environment Variables**: Ensure all required variables are properly set
- **Cold Start**: The first request may take longer due to serverless cold start
- **Agentic answers start with "operating in fallback mode"**: the web app couldn't reach the backend at `WEBAGENT_BACKEND_URL`, or the backend rejected the API key. Check the backend's `/health` and its logs
- **Answers start with "research routing is temporarily unavailable"**: in Auto mode the web app couldn't reach the backend's `/api/triage`, so it answered directly. Check `WEBAGENT_BACKEND_URL` and the backend's `/health`
- **Agentic requests return 429**: all Vercel users share the backend's rate limit; see step 3 above

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