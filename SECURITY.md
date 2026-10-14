# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability
reporting: open the repository's **Security** tab and choose **Report a vulnerability**
([direct link](https://github.com/Positron00/WebAgent/security/advisories/new)).
Don't open a public issue for a security problem.

Include what you found, how to reproduce it, and what an attacker could do with it.

You'll get a first answer within 5 working days, and a plan or a fix date once the
problem is confirmed.

## In scope

- The web app (`src/`): sign-in, the API routes and their proxies to the backend.
- The FastAPI backend (`backend/`): authentication, rate limits, the task API, document
  upload and search, code execution and its sandbox.
- The agentic security guard (`backend/app/security/`): ways to get instructions past
  it or make it approve a tool call it should stop.
- The Docker images and Kubernetes manifests in this repository.

## Out of scope

- Problems in the hosted services the platform calls (Together AI, Tavily, LangSmith,
  OpenAI and others): report those to the provider.
- Attacks that need a valid access code or API key holder to act against themselves.
- Missing hardening on a deployment that doesn't follow `backend/PRODUCTION_READINESS.md`.

## Supported versions

Only the latest release gets security fixes.
