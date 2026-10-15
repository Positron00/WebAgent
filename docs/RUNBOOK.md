# WebAgent operations runbook

Short procedures for production and uat.

## Deploy (Kubernetes)

1. Set secrets in your environment or secret store: `WEBAGENT_SECURITY_SECRET_KEY`, `WEBAGENT_API_KEY`, provider keys (`TOGETHER_API_KEY`, `TAVILY_API_KEY`, …), and `GRAFANA_ADMIN_PASSWORD` for monitoring.
2. Build and push the image (`git lfs pull` first: the encoders are LFS files):
   ```bash
   docker build --build-arg APP_VERSION=3.5.1 -t <registry>/webagent-backend:3.5.1 backend/
   docker push <registry>/webagent-backend:3.5.1
   ```
3. Deploy from `backend/kubernetes/` (`-e` is `prod` or `uat`; `-i` is the ingress controller's namespace, used by the NetworkPolicy):
   ```bash
   cd backend/kubernetes
   ./deploy.sh -e prod -n <ns> -v 3.5.1 -r <registry>/webagent-backend -H <public host> -i ingress-nginx
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

**JSON API (API key):** `GET /api/v1/observability/metrics`, `/drift`, `/health`, and
`/alerts` expose drift status, registry metrics, and LLM/document baselines. `POST`
`/api/v1/observability/llm/reset_baseline` and `/doc/reset_baseline` reset drift windows.
These routes use the same `X-API-Key` (or `Authorization: Bearer`) as other `/api/v1/*`
endpoints.

**Prometheus:** in-cluster scrapes still use `/metrics` on port 8000 (not the public Ingress).
**LangSmith:** agent tracing is unchanged (`LANGCHAIN_TRACING_V2` / project settings).
**Chat routing:** production runs still use `app/services/llm.py` only; `llm_service.py` feeds
the observability API and drift detectors, not user-facing chat.

Grafana dashboard helpers and MLflow experiment logging for model metrics are documented in
`backend/demos/README.md` and `backend/README.md` (demos section).

## Metrics exposure

- Prometheus in the `monitoring` namespace scrapes port 8000 inside the cluster.
- The public Ingress answers **404** for `/metrics` (`deployment.yaml` server-snippet).
- `networkpolicy.yaml` limits port 8000 to the ingress controller namespace and `monitoring`.
  If your CNI does not enforce NetworkPolicies, rely on the Ingress rule and document that in
  this file.
