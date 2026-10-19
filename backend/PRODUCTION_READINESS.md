# WebAgent Backend - Production Readiness

What to check before exposing the backend, and what's already in place. The history is in `CHANGELOG.md`.

## Production checklist (3.12.1)

Before exposing the backend:

1. **Select prod.** Set `WEBAGENT_ENV=prod` (the Docker image and Kubernetes manifest already do). A value set in the shell or container wins over `.env.local`.
2. **Set the secrets.** `WEBAGENT_SECURITY_SECRET_KEY` (for example `openssl rand -hex 48`) and `WEBAGENT_API_KEY`. The server refuses to start in uat/prod without them. Clients send the key as `X-API-Key`; the web app's `/api/task` route adds it on the server side.
3. **Leave code execution off**, or run it in the container sandbox: `WEBAGENT_ENABLE_CODE_EXECUTION=true` requires `WEBAGENT_CODE_SANDBOX=docker` and the `backend/sandbox` image in uat/prod, or the server won't start.
4. **Keep `/app/data` on persistent storage.** Task status and results live there and are shared by the server's workers. **Run one replica, by design**: there is no shared task store or checkpointer, so a second pod couldn't see the first one's tasks. Kubernetes updates use `strategy: Recreate` (the volume can't attach to two pods), so a deploy has a short gap.
5. **Check health and metrics.** `GET /health` (no key) for probes; `GET /api/v1/health/detailed` needs `X-API-Key`. Prometheus scrapes `GET /metrics` inside the cluster only — the public Ingress returns 404 for `/metrics`.
6. **Watch run times.** A typical agentic task takes 2.5–3.5 minutes. The backend stops a run after `WEBAGENT_TASK_MAX_SECONDS` (420 s) with `error_code: timeout`, and the web app waits up to 8 minutes. If runs approach the limit, open the task's LangSmith trace: a model call whose completion tokens equal its `max_tokens` is running away, and the log will say "stopped at the max_tokens limit".
7. **Rate limits.** 60 requests per minute per caller by default (`WEBAGENT_SECURITY_RATE_LIMIT_MAX_REQUESTS`). Limits are **per worker process** (two workers in the image each keep their own counter). Through the web app the caller is the user's sign-in session (`X-WebAgent-Client`), accepted from `WEBAGENT_TRUSTED_PROXIES` or from callers holding the API key. If the backend sits behind another proxy, list it in `WEBAGENT_TRUSTED_PROXIES`.
8. **Runs and retention.** Users can cancel a run (`POST /api/task/{id}/cancel`); stopped runs keep `error_code` `timeout`, `cancelled`, or `budget`. Result JSON files older than `WEBAGENT_RESULTS_RETENTION_DAYS` (default 30) are deleted daily. Pod restarts still lose in-flight runs that haven't been written to disk.
9. **Token budget (optional).** `WEBAGENT_TASK_TOKEN_BUDGET=0` leaves the cap off. When set, Together token usage for one task is summed across parallel agents; overages end the run with `error_code: budget`.
10. **Agentic triage.** prod ships with `triage.enabled: false` (Auto then acts like Always); uat has it on in `mode: shadow` with collection on, for the shadow week. Shadow mode covers only questions sent with Always: in Auto, triage routes for real. Classifier `triage-2026-10-c` passes the release gate, but turn triage on only after the shadow week and a human review of its training data. With triage on, the server refuses to start unless `models/triage/metadata.json` records `test.passed: true` and the fine-tuned encoder is in `models/triage/encoder/` (copied into the image with the code; its weights come from Git LFS, so run `git lfs pull` before `docker build`). Keep `WEBAGENT_TRIAGE_COLLECT` off unless storing users' questions for review is approved. The image runs 2 workers (`WEB_CONCURRENCY=2`), each with its own encoders; Kubernetes allows 3 GiB (request 2 GiB). With the security guard and triage both on, two workers peaked at 1.81 GiB (about 1.25 GiB RSS each) in the 3.2.4 simulated shadow week.
11. **Grafana.** Set `GRAFANA_ADMIN_PASSWORD` (in `.env.local` for Docker Compose, in the environment for `deploy.sh`); no password is stored in the repo. Change any password that was used with the old committed default. `deploy.sh` re-applies the data source, the dashboards and the alert rules and restarts Grafana on every deploy.
12. **Web app sign-in.** A production web app needs `WEBAGENT_ACCESS_CODE` (12+ characters) and `WEBAGENT_SESSION_SECRET` (32+ characters), or every API route answers 503. Rotate both together to sign everyone out.
13. **Drift and alerts.** Drift samples and state live in `/app/data/drift/` (on the persistent volume, deleted after `WEBAGENT_RESULTS_RETENTION_DAYS`). The six alerts show on the Model Drift dashboard and in Grafana's alert list; Grafana sends **no notifications** until a contact point is set up. A fresh install shows "not enough data" until each detector has 50 reference and 30 current samples. See `docs/RUNBOOK.md` "Drift and alerts".
14. **Backups.** See `docs/RUNBOOK.md` for volume snapshots or `backup-cronjob.yaml`.
15. **Spending caps.** Set monthly limits in the Together AI and Tavily dashboards for every key used in uat and prod (suggested: twice the highest month so far, with alerts at 50% and 80%). Record them here:

    | Key (name, not value) | Service | Limit | Recorded on |
    |---|---|---|---|
    | `TOGETHER_API_KEY` | Together AI | $25 in total (not monthly) | 2026-10-15 |
    | `TAVILY_API_KEY` | Tavily | Researcher plan: up to 1,000 API credits a month | 2026-10-15 |

    One key of each serves dev, the local uat runs and any deployment, so they share these limits.
16. **Security guard.** Every environment ships with `input_mode: shadow` and `tool_mode: shadow`: decisions are recorded and nothing is blocked. Switch checkpoint 1 to act only after a shadow week confirms its false-positive rate (`docs/RUNBOOK.md`). Checkpoint 2's nine intent rules stay shadow-only until each one's threshold is measured. The intent classifier in `models/security/alignment/` is installed shadow-only (it hasn't passed its gate) and never decides; to run without it, point `security_guard.alignment_model_dir` at an empty folder.

## What's in place

- **Access:** `X-API-Key` on every backend route except health and `/metrics`, required in uat and prod. The web app adds a shared access code with signed session cookies. There are no user accounts, passwords or JWTs.
- **Abuse limits:** rate limits per caller (per sign-in session through the web app), a request size limit, and a per-task time limit and optional token budget.
- **Security headers:** on the backend (middleware) and the web app (CSP, `X-Frame-Options: DENY`, `nosniff`, HSTS in production).
- **Security guard:** checkpoints 1 and 2 (`app/security/`), in shadow mode in every environment until the shadow week's numbers support act mode (`docs/RUNBOOK.md`).
- **Containers:** a non-root user, a read-only root filesystem, all Linux capabilities dropped, and no privilege escalation (Kubernetes). Model-written code runs only in the separate no-network sandbox image.
- **Dependencies:** exact versions in `constraints.txt`, audited in CI (`pip-audit`, `npm audit`); accepted advisories in `SECURITY_DEPENDENCIES.md`.
- **Monitoring:**
  - Prometheus metrics for requests, model calls and tokens, tasks, the guard and triage.
  - Two Grafana dashboards (WebAgent Backend, Model Drift) and six alert rules.
  - Plain-text logs tagged with the task ID.
  - LangSmith traces.
  - `GET /health` for probes.
- **Deployment:**
  - A multi-stage image with the CPU build of torch.
  - Kubernetes manifests with resource limits, a NetworkPolicy, an Ingress that hides `/metrics`, a persistent volume for `/app/data`, and Recreate updates (a short gap on deploy).
  - `deploy.sh`.

## Testing

- **Unit Tests**: Testing individual components
- **Integration Tests**: Testing component interactions
- **Red-team tests**: Injected search results, forged history, unrequested tool calls and poisoned documents through the security guard (`backend/tests/redteam/`)
- Not yet: a browser-to-backend end-to-end test (planned as IMP-113) and load tests

## Future Improvements

1. **Distributed Tracing**: Implement OpenTelemetry for distributed tracing
2. **Automated Canary Deployments**: Implement progressive rollouts
3. **Chaos Testing**: Test resilience with chaos engineering
4. **Cost Optimization**: Analyze and optimize resource usage
5. **Automated Scaling**: Implement horizontal pod autoscaling based on metrics 