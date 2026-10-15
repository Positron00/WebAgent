# Contributing to WebAgent

Thank you for improving WebAgent. This guide covers local setup, tests, and what we expect in pull requests.

## Prerequisites

- **Node.js** 20+ and **npm**
- **Python** 3.12 (the CI image uses 3.12)
- **Conda** env named `agents` (recommended for backend work), or any 3.12 venv
- **Git LFS** for model weights used by a few backend tests (optional for day-to-day work)

## First-time setup

1. Clone the repository and install Git LFS hooks:
   ```bash
   git lfs install
   git lfs pull   # only when you need tests marked needs_lfs
   ```

2. Python dependencies (runtime + dev, with pinned versions):
   ```bash
   pip install --extra-index-url https://download.pytorch.org/whl/cpu \
     -r requirements.txt
   ```
   The root `requirements.txt` includes `backend/requirements-dev.txt` and `backend/constraints.txt`.
   Demo-only extras live in `backend/requirements-demos.txt` (install manually if you run demos).
   Then the NLTK data the document agent needs (the tests never download it; the Docker image
   and CI install the same list):
   ```bash
   python -m nltk.downloader punkt punkt_tab stopwords wordnet maxent_ne_chunker words
   ```

3. Web app:
   ```bash
   npm ci
   cp .env.local.example .env.local   # fill in keys as needed
   ```

4. Husky runs `lint-staged` on commit when hooks are installed (`npm run prepare`).

## Running tests

| Command | What it runs |
|---------|----------------|
| `npm test` | Jest (web app) |
| `npm run test:coverage` | Jest with coverage floors (`jest.config.js`) |
| `npm run test:backend` | `pytest backend/tests` in the `agents` conda env |
| `npm run coverage:backend` | Backend coverage report |
| `npm run lint` / `npm run check-types` | ESLint and `tsc` |
| `npm run lint:backend` | `ruff check backend` |

Backend tests must stay **offline**: no real network calls to LLMs, Tavily, LangSmith, or Hugging Face.
See [docs/TESTING.md](docs/TESTING.md).

## Pull requests

- Target `main` or `release/*` branches; CI runs on both.
- Update [CHANGELOG.md](CHANGELOG.md) under **Unreleased** for user-visible changes (follow existing style).
- Keep diffs focused; match surrounding code style.
- Do not commit secrets (`.env.local`, API keys).

## Operations docs

- [docs/RUNBOOK.md](docs/RUNBOOK.md) — deploy, rollback, keys, backups
- [SECURITY.md](SECURITY.md) — vulnerability reports
- [backend/PRODUCTION_READINESS.md](backend/PRODUCTION_READINESS.md) — production limits and design choices
