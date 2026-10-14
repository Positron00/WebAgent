# Low-level design: CI, deployment, dependencies and docs

Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Files

| File | New? | Purpose | Release |
|---|---|---|---|
| `.github/workflows/ci.yml` | new | Web app and backend checks on push and PR | 3.2.7 |
| `.github/workflows/models.yml` | new | Weekly run of tests that need the LFS weights | 3.2.7 |
| `.github/workflows/e2e.yml` | new | Playwright run on PRs to `release/*` | 3.4.0 |
| `.husky/pre-commit` | new | `npx lint-staged` | 3.2.7 |
| `backend/ruff.toml` | new | Python lint rules | 3.3.0 |
| `pytest.ini` | changed | New marker `needs_lfs` | 3.2.7 |
| `backend/docker-compose.yml` | changed | Ports on `127.0.0.1`; no `latest`; memory limit | 3.2.7, 3.4.0 |
| `backend/kubernetes/networkpolicy.yaml` | new | Only the monitoring namespace may reach `/metrics` | 3.3.0 |
| `backend/app/services/Dockerfile.llm`, `Dockerfile.llm-mlflow` | changed | Non-root user | 3.3.0 |
| `backend/requirements.txt`, `requirements-dev.txt`, `requirements-demos.txt` | changed / new | Runtime, dev and demo packages split | 3.4.0 |
| `SECURITY.md`, `CONTRIBUTING.md`, `docs/RUNBOOK.md` | new | Policy, contributor guide, operations | 3.3.0, 3.4.0 |

## 2. CI (`.github/workflows/ci.yml`, IMP-001)

```yaml
name: ci
on:
  push: { branches: [main, 'release/**'] }
  pull_request: { branches: [main, 'release/**'] }
concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }

jobs:
  web:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4            # lfs: false (default): pointer files only
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci
      - run: npm run lint
      - run: npm run check-types
      - run: npm run test:coverage
      - run: npm run build
        env: { TOGETHER_API_KEY: ci-placeholder-key-not-used-0000000000 }

  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12', cache: pip, cache-dependency-path: backend/constraints.txt }
      - run: pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
      - run: python -m pytest backend/tests -q -m "not needs_lfs and not real_triage_model"
      # 3.3.0: - run: ruff check backend
```

- `npm run build` needs `TOGETHER_API_KEY` because `src/config/env.ts` throws at import
  when it's missing and `validateConfig` checks its length (≥ 32). The placeholder is never
  used for a call.
- The CPU wheel index keeps `torch==2.5.1` (`constraints.txt:495`) to a CPU build, which is
  much smaller than the default CUDA wheel.
- The backend suite already blocks the network and LangSmith (`backend/tests/conftest.py`),
  so CI needs no secrets.
- Branch protection on `main` requires both jobs (a manual setting in GitHub).

### 2.1 Tests that need LFS files (IMP-001, decision 12)

The LFS files total about 268 MB (`git lfs ls-files -s`: two encoders, 91 MB and 134 MB,
and 43 MB of security data). Checking them out on every run would use up GitHub's free
1 GB a month of LFS bandwidth in a few days.

- New marker in `pytest.ini`: `needs_lfs: needs real Git LFS files (encoder weights or
  security data)`. `real_triage_model` stays as it is.
- Task 1.1 runs the suite on a fresh `GIT_LFS_SKIP_SMUDGE=1` clone. Every test that fails
  only because a file is an LFS pointer gets `@pytest.mark.needs_lfs`, or a skip when the
  file starts with `version https://git-lfs`. A shared helper,
  `tests/lfs.py::is_lfs_pointer(path)`, does the check.

### 2.2 Weekly model tests (`.github/workflows/models.yml`)

```yaml
on:
  schedule: [{ cron: '0 6 * * 1' }]     # Mondays 06:00 UTC
  workflow_dispatch: {}
jobs:
  models:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: git lfs ls-files -l | cut -d' ' -f1 | sort > .lfs-ids
      - uses: actions/cache@v4
        with: { path: .git/lfs, key: lfs-${{ hashFiles('.lfs-ids') }} }
      - run: git lfs pull
      # same Python setup as ci.yml
      - run: python -m pytest backend/tests -q -m "needs_lfs or real_triage_model"
```

The cache means the weights are downloaded only when they change.

## 3. Git hooks and Python lint (IMP-002, IMP-003)

`.husky/pre-commit` (Husky 8, which `package.json` uses):

```sh
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"
npx lint-staged
```

The existing LFS hooks (`post-checkout`, `post-commit`, `post-merge`, `pre-push`) stay.

`backend/ruff.toml`:

```toml
target-version = "py312"
line-length = 120
extend-exclude = ["demos", "sandbox", "mlruns"]

[lint]
select = ["E9", "F63", "F7", "F82", "E722"]   # syntax errors, undefined names, bare except
```

`E722` (bare `except`) fails today at `coding_assistant.py:164,174,190,199` and
`coding_review.py:327,345`. Task 2.12 fixes those in the same change that turns the rule
on. More rules can be added later, one per change. `ruff` goes into
`requirements-dev.txt` (3.4.0) or, until then, `backend/requirements.txt`'s dev section.
`package.json` gains `"lint:backend": "conda run -n agents ruff check backend"`.

## 4. Compose and Kubernetes

### 4.1 Compose (`backend/docker-compose.yml`, IMP-014, IMP-103)

```yaml
services:
  backend:
    image: webagent/backend:${APP_VERSION:?set APP_VERSION, e.g. via npm run stack:up}
    ports: ["127.0.0.1:8000:8000"]
    mem_limit: 3g
  prometheus:
    ports: ["127.0.0.1:9090:9090"]
  grafana:
    ports: ["127.0.0.1:${GRAFANA_HOST_PORT:-3001}:3000"]
```

`npm run stack:up` already sets `APP_VERSION` from `package.json`. Someone who needs the
stack on the network sets the ports themselves and turns API keys on; the comment at the
top of the file says so.

### 4.2 Metrics NetworkPolicy (`backend/kubernetes/networkpolicy.yaml`, IMP-063)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: webagent-backend-ingress, namespace: default }
spec:
  podSelector: { matchLabels: { app: webagent-backend } }
  policyTypes: [Ingress]
  ingress:
    - from:   # the ingress controller, for the public API
        - namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: ingress-nginx } }
      ports: [{ port: 8000 }]
    - from:   # Prometheus
        - namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: monitoring } }
      ports: [{ port: 8000 }]
```

A NetworkPolicy works on ports, not paths, so the public ingress also stops serving
`/metrics`: the Ingress in `deployment.yaml` gets a rule that answers 404 for `/metrics`
(an nginx `server-snippet` or a separate path rule). NetworkPolicies need a network
plugin that enforces them; `kubernetes/README.md` says to check this.

`deploy.sh` applies the new file. `monitoring` is the namespace the repo's manifests
create (`kubernetes/monitoring/namespace.yaml:4`); the ingress controller's namespace
depends on the cluster (open point O-2 in `tasks.md`), so `deploy.sh` takes it as an
option, defaulting to `ingress-nginx`. If the cluster doesn't enforce NetworkPolicies
(O-3), the Ingress 404 for `/metrics` still keeps it off the public address.

### 4.3 Non-root LLM images (IMP-067)

Both Dockerfiles gain, before `CMD`:

```dockerfile
RUN useradd --create-home --uid 10001 llm && chown -R llm:llm /app
USER llm
```

## 5. Dependency split (IMP-100, IMP-101, 3.4.0)

| File | Holds | Installed by |
|---|---|---|
| `backend/requirements.txt` | Packages the server imports (FastAPI, LangChain/LangGraph, Together, Tavily, sentence-transformers, torch, chromadb, mlflow, gensim, neo4j, pypdf, …) | Docker image; root `requirements.txt` |
| `backend/requirements-dev.txt` | `-r requirements.txt`, pytest and plugins, ruff, black, isort; and the libraries model-written code may import (numpy, pandas, matplotlib, seaborn), because the dev-only `local` code sandbox runs in the server's Python | Root `requirements.txt`; CI |
| `backend/requirements-demos.txt` | Packages only `backend/demos` and `app/services/examples` import | Manually |
| `backend/constraints.txt` | Exact pins for all three | All of them, with `-c` |

How to decide: for each package in today's `requirements.txt`, search `backend/app` and
`backend/main.py` for its import name. No import → dev or demos (or removed). In uat and
prod, model-written code runs only in the `backend/sandbox` image, which installs its own
numpy, pandas, matplotlib and seaborn (`backend/sandbox/Dockerfile`), so the server image
doesn't need them unless the server imports them itself. The result
goes in the PR as a table. Removed in IMP-101: `py2neo` (→ `neo4j` driver in
`graphdb.py`), `PyPDF2` (→ `pypdf`), `platform-utils`, `redis`, and `python-jose` once
IMP-065 has removed JWT. Image size before and after goes in the PR.

## 6. Monitoring versions (IMP-102)

Move Grafana from 8.5.3 to the current supported 11.x, and Prometheus from v2.36.0 to the
current 2.x LTS. Use the same versions in Compose and `kubernetes/monitoring/`. Check that
the provisioned dashboards (`backend/grafana/`, `kubernetes/monitoring/grafana-dashboard.json`)
load. Grafana 9+ changed some panel types; convert any panel that shows "panel plugin not
found".

## 7. Docs

| File | Contents | Release |
|---|---|---|
| `README.md` | Fix every row of the plan's docs table; access code setup; link `SECURITY.md`, `CONTRIBUTING.md`, the runbook; describe `backend/models/` as an optional tool (IMP-084); update the "last updated" badge | 3.3.0, 3.4.0 |
| `backend/PRODUCTION_READINESS.md` | One replica by design and why; per-worker rate limits; restarts lose runs; result retention; access code; spending caps; remove rolling updates, 1.5 GiB, E2E and load claims | 3.3.0 |
| `VERCEL_DEPLOYMENT.md` | Access code and session secret variables; replace the rate-limit paragraph (line 74): limits are per sign-in session, and the backend trusts the web app through its API key, so `WEBAGENT_TRUSTED_PROXIES` isn't needed for Vercel | 3.2.7 |
| `SECURITY.md` | How to report (a private email or GitHub's private advisory), scope (web app, backend, guard), first answer within 5 working days | 3.3.0 |
| `CONTRIBUTING.md` | Conda env, `pip install -r requirements.txt` (with constraints), `npm ci`, `git lfs pull`, running both test suites, hooks, CHANGELOG rules | 3.4.0 |
| `docs/RUNBOOK.md` | Deploy, roll back, rotate `WEBAGENT_API_KEY` / secret key / access code and session secret, disk full on `/app/data`, failed probes, backup and restore (IMP-073), spending caps (IMP-013) | 3.3.0 (backup, caps), 3.4.0 (rest) |
| `.env.local.example` | Every new variable from the two other designs, with its default | each release |
| `CHANGELOG.md` | "Upgrade notes" section per release (IMP-NFR-08); entries before 3.0 moved to `docs/CHANGELOG-archive.md` | each release; archive in 3.4.0 |

## 8. Spending caps (IMP-013)

In the Together AI and Tavily dashboards, set a monthly limit for each key used in uat and
prod. Suggested starting point: twice the highest month so far. Record the key name (not
the key), the limit and the date in the runbook. Recheck every quarter.

## 9. Backup and restore (IMP-073)

If the cluster has a CSI driver with snapshots: a `VolumeSnapshotClass` and a daily
`VolumeSnapshot` of `webagent-data`, keeping 7. Otherwise: a daily CronJob that mounts the
claim read-only and copies it with `tar` to object storage. The runbook gives both, and
the restore steps: scale to 0, restore the claim from the snapshot or archive, scale to 1,
check `/health` and one old task's status. Task 2.18 tries the restore once in uat.
