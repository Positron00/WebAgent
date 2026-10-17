# WebAgent operations runbook

Short procedures for production and uat.

## Deploy (Kubernetes)

1. Set secrets in your environment or secret store: `WEBAGENT_SECURITY_SECRET_KEY`, `WEBAGENT_API_KEY`, provider keys (`TOGETHER_API_KEY`, `TAVILY_API_KEY`, …), and `GRAFANA_ADMIN_PASSWORD` for monitoring.
2. Build and push the image (`git lfs pull` first: the encoders are LFS files):
   ```bash
   docker build --build-arg APP_VERSION=3.8.2 -t <registry>/webagent-backend:3.8.2 backend/
   docker push <registry>/webagent-backend:3.8.2
   ```
3. Deploy from `backend/kubernetes/` (`-e` is `prod` or `uat`; `-i` is the ingress controller's namespace, used by the NetworkPolicy):
   ```bash
   cd backend/kubernetes
   ./deploy.sh -e prod -n <ns> -v 3.8.2 -r <registry>/webagent-backend -H <public host> -i ingress-nginx
   ```
4. Confirm `kubectl get pods -n <ns>` shows `webagent-backend` ready and `GET /health` returns 200 through the ingress.
5. Apply monitoring manifests under `backend/kubernetes/monitoring/` if not already applied by `deploy.sh`.

**Docker Compose (local stack):** `npm run stack:up` from the repo root (requires `APP_VERSION` from `package.json`, set by the script). Backend memory is capped at 3 GiB in Compose.

## Roll back

1. Note the previous image tag or git tag that last worked.
2. `kubectl set image deployment/webagent-backend webagent-backend=<registry>/webagent-backend:<previous-tag> -n <ns>` (or re-run `deploy.sh` with `-v <previous-tag>`)
3. Wait for the rollout: `kubectl rollout status deployment/webagent-backend -n <ns>`
4. If the rollback involves data schema changes, restore `/app/data` from backup before scaling up (see below).
5. Verify `/health` and one known `GET /api/task/{id}`.

## Rotate secrets

| Secret | Where | Steps |
|--------|--------|--------|
| `WEBAGENT_API_KEY` | Backend env / K8s Secret | Generate a new key, update the Secret, restart the Deployment (`kubectl rollout restart`; one replica with `Recreate`, so expect a short gap), update the web app's `WEBAGENT_API_KEY` (Vercel or `.env.local`), smoke-test one agentic task. |
| `WEBAGENT_SECURITY_SECRET_KEY` | Backend env | Rotate in Secret; restart backend; invalidates signed tokens tied to the old key. |
| Web access code | `WEBAGENT_ACCESS_CODE` (web app) | Set a new value in the hosting env; share with users; old sessions expire per session TTL. |
| Session signing | `SESSION_SECRET` or platform equivalent | Rotate in the web app env; users sign in again. |
| `GRAFANA_ADMIN_PASSWORD` | Grafana deployment | Update Secret and restart Grafana; update `.env.local` for Compose. |

Record rotation dates in your internal change log (not in git).

## Disk full on `/app/data`

Symptoms: probe failures, tasks stuck in `processing`, errors writing `backend/data/results/`.

1. Check PVC usage: `kubectl exec -it deploy/webagent-backend -n <ns> -- df -h /app/data`
2. Delete old result files past retention (`WEBAGENT_RESULTS_RETENTION_DAYS`, default 30) or expand the volume per your cloud provider.
3. If the guard collection or triage override dirs grew unexpectedly, archive and trim `data/security/collected/` and `data/triage/collected/` per policy.
4. Restart the pod after freeing space; confirm `/health`.

## Failed probes

1. `kubectl describe pod -l app=webagent-backend -n <ns>` — look at `Liveness` / `Readiness` events.
2. Check logs: `kubectl logs deploy/webagent-backend -n <ns> --tail=200`
3. Common causes: missing API keys in uat/prod, NLTK/data volume permissions, startup slower than `start_period` (increase in manifest if needed).
4. For Compose: `docker compose -f backend/docker-compose.yml logs webagent-backend`

## Backup and restore (`/app/data`)

The backend keeps task results, triage overrides, and security-guard collection data on the
`webagent-data` volume (`/app/data` in the pod).

### Option A — volume snapshots (preferred)

If the cluster has a CSI driver with `VolumeSnapshot`:

1. Create a `VolumeSnapshotClass` and a daily `VolumeSnapshot` of `webagent-data`, retaining 7.
2. **Restore:** scale the Deployment to 0, restore the PVC from the snapshot (cluster-specific), scale to 1, check `GET /health` and one old `GET /api/task/{id}`.

### Option B — CronJob archive

When snapshots are not available, apply `backend/kubernetes/backup-cronjob.yaml` after setting
`BACKUP_URI` on the CronJob to a destination your environment can write (for example an S3
prefix). The job tars `/app/data` read-only at 03:00 UTC. Copy the archive off the node or
wire an uploader in your fork of the manifest.

**Restore:**

1. `kubectl scale deployment/webagent-backend --replicas=0 -n <namespace>`
2. Replace the PVC contents from the tar (or attach a restored volume).
3. `kubectl scale deployment/webagent-backend --replicas=1 -n <namespace>`
4. Verify `/health` and an older task status.

Record the date and outcome of at least one restore drill in uat.

## Spending caps (Together AI and Tavily)

Set monthly limits in each provider dashboard for every key used in uat and prod (suggested:
twice the highest month so far, with alerts at 50% and 80%). Record the key **name** (not the
secret), limit, and date in `backend/PRODUCTION_READINESS.md`. Review quarterly.

## Advanced observability (IMP-128)

**JSON API (API key):** `GET /api/v1/observability/metrics`, `/drift`, `/health` and `/alerts`
answer from the shared drift store (below), so every worker and every restart gives the same
answer. `POST /api/v1/observability/llm/reset_baseline` and `/doc/reset_baseline` start new
references. These routes use the same `X-API-Key` (or `Authorization: Bearer`) as other
`/api/v1/*` endpoints. Fields are in `backend/docs/api_reference.md`.

**Prometheus:** in-cluster scrapes still use `/metrics` on port 8000 (not the public Ingress).
**LangSmith:** agent tracing is unchanged (`LANGCHAIN_TRACING_V2` / project settings).

**Grafana dashboards:** `backend/grafana/dashboards/` holds two dashboards, provisioned into
the local stack (`npm run stack:up`, Grafana on `127.0.0.1:3001`, folder "WebAgent") and, since
3.6.0, into Kubernetes by `deploy.sh` (ConfigMaps built from the same files):
- *WebAgent Backend Dashboard*: HTTP traffic, model requests, duration and tokens, tasks, triage.
- *Model Drift Dashboard*: an Alerts row first (active alerts, alerts over time, a detector table
  with enough data / drift / score / current samples), then drift scores per detector (gauges red
  at 0.95 for the KS-based ones and 0.7 for embeddings), model latency (p95), errors and tokens
  per minute, document processing time.

Model-call panels fill from real calls: each Together (or self-hosted) request records
`llm_requests_total`, `llm_request_duration_seconds`, `llm_tokens_total` and
`ml_model_latency_seconds` once (`app/services/llm_call_metrics.py`). Both stacks provision the
Prometheus data source with uid `prometheus`, which the dashboards and alert rules refer to.

## Drift and alerts

Since 3.6.0. (The Grafana alert rules link here: keep this heading's anchor `#drift-and-alerts`.)

**Where the numbers come from.** Real traffic only: every model call adds its speed (seconds per
1,000 completion tokens) and total tokens under its call label (the agent name, `judge`,
`triage` or `other`) and an outcome (`ok`, `error`, `rate_limited`); every document load adds its
processing time, size, file type and outcome; embeddings are added as running sums of at least
30 vectors. No text, prompt or single vector is stored. Samples are appended to
`/app/data/drift/samples-YYYY-MM-DD.jsonl` and deleted after `WEBAGENT_RESULTS_RETENTION_DAYS`;
`state.json` beside them holds the frozen references and the last result. One worker per minute
(`WEBAGENT_DRIFT_INTERVAL_SECONDS`) computes; the others read its result.

**Reference and "not enough data".** A detector's reference is the first 50 samples
(`WEBAGENT_DRIFT_WINDOW`) after install or the last reset; it is then frozen. It is compared with
the latest samples once there are at least 30. A fresh install, or a detector right after a
reset, shows "not enough data" and a score of 0 until then: with little traffic that can take
days. That is expected, not a fault.

**Alerts** (shown on the drift dashboard and in Grafana's alert list, Alerting → Alert rules,
folder "WebAgent"; Grafana sends no notifications until a contact point is set up):

| Alert | Severity | Fires when | What to check |
|---|---|---|---|
| `model_drift` | warning | A model detector (speed, tokens, embeddings) differs from its reference (KS p < 0.05; embeddings cosine < 0.3) | A provider or model change, a prompt change, or a slow provider. Speed drift also fires when Together is slower than usual for a while. Once understood, reset the LLM baseline |
| `model_errors` | critical | Over 5% of at least 20 model requests failed in the last hour (rate limits not counted) | Provider status, API key and spending cap, backend logs (`Error invoking Together LLM`) |
| `rate_limited` | warning | Over 20% of at least 20 model requests were rate-limited (429) in the last hour | Together load or account limits; requests are retried |
| `token_usage` | warning | Tokens in the last 24 hours exceed `WEBAGENT_TOKEN_ALERT_DAILY` (0 = off) | Which call label uses them (`GET /observability/metrics`, `by_label`) |
| `document_drift` | warning | Document processing time, size or file type differs from its reference | What is being uploaded; reset the document baseline once understood |
| `document_errors` | critical | Over 10% of at least 10 document loads failed in the last 24 hours | Backend logs (`Error loading`), the vector store, the embeddings key |

An alert clears by itself when its condition no longer holds. Each new drift and each new alert
is logged once as a WARNING (and drift scores are logged to MLflow).

**Reset a baseline** after a change you understand (new model, new kind of documents):

```bash
curl -X POST -H "X-API-Key: $WEBAGENT_API_KEY" https://<host>/api/v1/observability/llm/reset_baseline
curl -X POST -H "X-API-Key: $WEBAGENT_API_KEY" https://<host>/api/v1/observability/doc/reset_baseline
```

**Turn it off:** `WEBAGENT_DRIFT_ENABLED=false` stops recording and computing; the API then
answers as if nothing had been recorded.

**Check it locally without spending credit:** `python scripts/fake_together.py --port 18200
--delay 0.05` answers like Together (`--delay`, `--error-rate`, `--rate-limit-rate` change its
behaviour). Start the stack with `TOGETHER_API_BASE=http://host.docker.internal:18200/v1` (and,
to go faster, `WEBAGENT_DRIFT_WINDOW=30 WEBAGENT_DRIFT_INTERVAL_SECONDS=10`), reset the LLM
baseline, make 30 calls, then make the fake server slower or failing and watch the alerts. Unset
`TOGETHER_API_BASE` and reset both baselines afterwards.

Grafana dashboard helpers and MLflow experiment logging for model metrics are documented in
`backend/demos/README.md` and `backend/README.md` (demos section).

## Security guard: checks against the user's intent (3.7.0)

Checkpoint 2 now also compares each tool call, and each agent-written subgoal (the supervisor's
rewrite and plan questions, the senior research agent's follow-up questions), with the user's own
turns: intent alignment, risk above intent (is the call more harmful than the request?), private
text copied into a web search, tools the plan didn't ask for, and drift across a task. In 3.7.0
**all nine of these rules are shadow-only**: they are recorded
(`webagent_guard_shadow_policy_total{policy_id}`, `webagent_guard_intent_alignment`,
`webagent_guard_risk_above_intent`, `webagent_guard_egress_overlap`, collection rows) and never
change what an agent does, whatever `WEBAGENT_GUARD_TOOL_MODE` says. A rule starts acting only in
its own later release, after its threshold is measured (`planning/specs/checkpoint2-intent/`,
SEC-130). `WEBAGENT_GUARD_PLAN_CHECK=false` turns the subgoal checks off.

Since 3.8.0 the intent signals can come from a small classifier in `backend/models/security/alignment/`
(`security_guard.alignment_model_dir`). It is used only if its `metadata.json` records a passed gate
(`scripts/eval_alignment_model.py`); in uat and prod the server **refuses to start** if that folder
holds a model whose gate didn't pass (an empty or missing folder is fine: phase A's signals are used).
3.8.0 ships no model: its first candidates failed the gate. Nothing new is stored:
private text is kept only as salted hashes in memory for the task, and outside dev no intent or
subgoal text is logged or collected.

## Metrics exposure

- Prometheus in the `monitoring` namespace scrapes port 8000 inside the cluster.
- The public Ingress answers **404** for `/metrics` (`deployment.yaml` server-snippet).
- `networkpolicy.yaml` limits port 8000 to the ingress controller namespace and `monitoring`.
  If your CNI does not enforce NetworkPolicies, rely on the Ingress rule and document that in
  this file.
