# WebAgent Kubernetes Deployment

This directory contains Kubernetes manifests and scripts for deploying the WebAgent backend to a Kubernetes cluster.

## Directory Structure

- `deployment.yaml`: the backend's Deployment, Service, Ingress and data volume (the Secret is created by `deploy.sh`)
- `networkpolicy.yaml`: limits port 8000 to the ingress controller and Prometheus (`deploy.sh` applies it; use `-i` for your ingress namespace)
- `backup-cronjob.yaml`: optional daily tar of `webagent-data` when volume snapshots are unavailable (see `docs/RUNBOOK.md`)
- `monitoring/`: Manifests for setting up Prometheus and Grafana monitoring
- `deploy.sh`: Deployment script to simplify the deployment process

## Deployment

### Prerequisites

- Kubernetes cluster with RBAC enabled
- kubectl configured to access your cluster
- Docker image for WebAgent backend (webagent/backend:3.8.1, built from `backend/` with `--build-arg APP_VERSION=3.8.1`)

### Using the Deployment Script

Build the image and push it to a registry the cluster can pull from, then run `deploy.sh`:

```bash
git lfs pull   # the security encoder and dataset are LFS files
docker build --build-arg APP_VERSION=3.8.1 -t <registry>/webagent-backend:3.8.1 backend/
docker push <registry>/webagent-backend:3.8.1

# Required
export OPENAI_API_KEY="your-openai-api-key"
export WEBAGENT_SECURITY_SECRET_KEY="$(openssl rand -hex 48)"
export WEBAGENT_API_KEY="key-the-web-app-sends-as-X-API-Key"
export GRAFANA_ADMIN_PASSWORD="choose-a-strong-password"
# Optional
export TOGETHER_API_KEY="your-together-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"
# Or load them from the repo-root .env.local, where they already are:
#   set -a; . ../../.env.local; set +a

cd backend/kubernetes
./deploy.sh -e uat -n webagent-uat -v 3.8.1 -r <registry>/webagent-backend -H uat-api.your-domain
```

Options:

```
-e, --environment  prod or uat: sets WEBAGENT_ENV, so the pod uses config/<env>.yaml (default: prod)
-n, --namespace    Kubernetes namespace (default: default)
-v, --version      Image tag (default: 3.8.1)
-r, --registry     Image repository the cluster pulls from (default: webagent/backend)
-H, --host         Public host name for the ingress (default: api.webagent.example.com)
```

What the script does:

- Creates or updates the `webagent-secrets` Secret straight from your environment (`kubectl create secret ... | kubectl apply`); keys are never written to a file.
- Fills the environment, image, host and namespace into `deployment.yaml` and applies it: the Deployment, a Service, an Ingress (nginx, TLS from cert-manager's `letsencrypt-prod` issuer) and a 5 GiB PersistentVolumeClaim for `/app/data`.
- Sets up Prometheus and Grafana in the `monitoring` namespace the first time.

What the deployment sets up:

- `WEBAGENT_ENV` from `-e`. In prod and uat the app refuses to start without `WEBAGENT_SECURITY_SECRET_KEY` and `WEBAGENT_API_KEY`.
- Two uvicorn workers; CPU limit 2, request 1; memory limit 3 GiB, request 2 GiB.
- Liveness and readiness probes call `/health`, which needs no API key.
- The root filesystem is read-only. `/app/data` (task results, the security guard's and triage's collection, model registry) is the persistent volume, so it survives restarts and redeploys; `/app/logs` and `/tmp` are `emptyDir` volumes (the workers' metrics live in `/tmp`).
- One replica, updated with the `Recreate` strategy (a ReadWriteOnce volume can't be attached to two pods at once, so a deploy has a short gap). To run more replicas, use a shared ReadWriteMany volume.
- Generated-code execution stays off. If you turn it on (`WEBAGENT_ENABLE_CODE_EXECUTION=true`), you must also set `WEBAGENT_CODE_SANDBOX=docker` and give the pod access to a container runtime; the app refuses to start otherwise.

The web app isn't deployed here: run it on Vercel ([VERCEL_DEPLOYMENT.md](../../VERCEL_DEPLOYMENT.md)) or locally, with `WEBAGENT_BACKEND_URL` set to `https://<host>` and the same `WEBAGENT_API_KEY`.

## Monitoring

The deployment includes Prometheus and Grafana for monitoring. See the [monitoring README](monitoring/README.md) for details.

To access Grafana dashboards:

```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000
```

Then open http://localhost:3000 in your browser and log in with:
- Username: admin
- Password: the `GRAFANA_ADMIN_PASSWORD` you set before running `deploy.sh`

## Configuration

Settings come from `config/<env>.yaml` in the image (`prod.yaml` or `uat.yaml`), chosen by `-e`, plus the environment variables in `deployment.yaml`. Changing a YAML setting needs a new image; most settings can also be overridden with a `WEBAGENT_...` environment variable added to the Deployment.

## Security

The deployment includes several security features:

- **Non-root User**: The container runs as a non-root user (UID 1000)
- **Read-only Filesystem**: The root filesystem is mounted as read-only
- **Resource Limits**: CPU and memory limits are set
- **Network Policies**: Can be added to restrict network traffic
- **Secrets Management**: Sensitive information is stored in Kubernetes secrets

## Troubleshooting

If you encounter issues with the deployment:

1. Check the pod status:

```bash
kubectl get pods -l app=webagent-backend
```

2. Check the pod logs:

```bash
kubectl logs -l app=webagent-backend
```

3. Check the events:

```bash
kubectl get events --sort-by='.lastTimestamp'
```

4. Verify the Secret and the data volume exist:

```bash
kubectl get secret webagent-secrets
kubectl get pvc webagent-data
``` 