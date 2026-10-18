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

## What CI runs

`ci` runs on every push and pull request to `main` and `release/*`:

- **web:** `npm ci`, `npm audit --omit=dev --audit-level=high`, lint, type check, Jest with coverage, and `next build`.
- **backend:** install with the constraints, NLTK data, `ruff check`, `pip-audit` on `backend/constraints.txt`, and pytest without the real-model tests, with a coverage floor.

`models` runs the tests marked `needs_lfs` and `real_triage_model` against the real LFS weights, every Monday and when started by hand (`workflow_dispatch`).

## Changing dependencies

- **Python:** change `backend/requirements.txt`, then regenerate `backend/constraints.txt` in a clean `python:3.12-slim` container with the CPU torch index. Don't freeze it from your own environment. Keep the major-version caps listed at the top of the file.
- **Model scores:** after a change to torch, transformers, sentence-transformers or the numeric stack, compare the real models' scores before and after (ContextDefender, triage, the alignment classifier), not just the tests.
- **Advisories:** if `pip-audit` reports something that can't be fixed without a major upgrade and isn't reachable, add it with its reason to `backend/SECURITY_DEPENDENCIES.md`, and to the ignore list in `.github/workflows/ci.yml`.
- **npm:** `npm install <package>@<version>`. If the lock file has to be rebuilt from scratch, do it in a clean `node:20` container (`npm install --package-lock-only`). Rebuilt on a Mac over an existing `node_modules`, it can drop optional platform packages, and `npm ci` in CI then fails.

## Pull requests

- Target `main` or `release/*` branches; CI runs on both.
- Update [CHANGELOG.md](CHANGELOG.md) under **Unreleased** for user-visible changes (follow existing style).
- Keep diffs focused; match surrounding code style.
- Do not commit secrets (`.env.local`, API keys).

## Operations docs

- [docs/RUNBOOK.md](docs/RUNBOOK.md) — deploy, rollback, keys, backups
- [SECURITY.md](SECURITY.md) — vulnerability reports
- [backend/PRODUCTION_READINESS.md](backend/PRODUCTION_READINESS.md) — production limits and design choices
