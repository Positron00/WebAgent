# WebAgent Kubernetes Deployment

This directory contains Kubernetes manifests and scripts for deploying the WebAgent backend to a Kubernetes cluster.

## Directory Structure

- `deployment.yaml`: Main deployment manifest for the WebAgent backend
- `monitoring/`: Manifests for setting up Prometheus and Grafana monitoring
- `deploy.sh`: Deployment script to simplify the deployment process

## Deployment

### Prerequisites

- Kubernetes cluster with RBAC enabled
- kubectl configured to access your cluster
- Docker image for WebAgent backend (webagent/backend:3.1.1, built from `backend/` with `--build-arg APP_VERSION=3.1.1`)

### Using the Deployment Script

The easiest way to deploy is using the provided `deploy.sh` script:

```bash
# Set required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export WEBAGENT_SECURITY_SECRET_KEY="$(openssl rand -hex 48)"
export WEBAGENT_API_KEY="key-the-web-app-sends-as-X-API-Key"
export GRAFANA_ADMIN_PASSWORD="choose-a-strong-password"
# Or load them from the repo-root .env.local, where they already are:
#   set -a; . ../../.env.local; set +a

# Optional environment variables
export TOGETHER_API_KEY="your-together-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"

# Run the deployment script
./deploy.sh
```

What the deployment sets up:

- `WEBAGENT_ENV=prod`, so the pod uses `config/prod.yaml`: API keys required, debug off. The app refuses to start without `WEBAGENT_SECURITY_SECRET_KEY` and `WEBAGENT_API_KEY`.
- Liveness and readiness probes call `/health`, which needs no API key.
- The root filesystem is read-only; `/app/data` (task results, model registry) and `/app/logs` are `emptyDir` volumes.
- One replica. Task status is shared between the pod's workers through `/app/data/results`, but not between pods. To run more replicas, mount `/app/data` from a shared ReadWriteMany volume.
- Generated-code execution stays off. If you turn it on (`WEBAGENT_ENABLE_CODE_EXECUTION=true`), you must also set `WEBAGENT_CODE_SANDBOX=docker` and give the pod access to a container runtime; the app refuses to start otherwise.

#### Script Options

```
Usage: ./deploy.sh [OPTIONS]

Options:
  -h, --help         Show this help message and exit
  -e, --environment  Environment to deploy (default: staging)
  -v, --version      Version to deploy (default: 3.1.1)
```

### Manual Deployment

If you prefer to deploy manually:

1. Create the namespace if it doesn't exist:

```bash
kubectl create namespace default
```

2. Create the monitoring namespace and deploy monitoring resources:

```bash
kubectl apply -f monitoring/namespace.yaml
kubectl apply -f monitoring/prometheus.yaml
kubectl apply -f monitoring/grafana.yaml
kubectl apply -f monitoring/grafana-datasource.yaml
```

3. Create the secrets:

```bash
kubectl create secret generic webagent-secrets \
  --from-literal=openai-api-key="your-openai-api-key" \
  --from-literal=together-api-key="your-together-api-key" \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=langsmith-api-key="your-langsmith-api-key"
```

4. Deploy the WebAgent backend:

```bash
kubectl apply -f deployment.yaml
```

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

The WebAgent backend is configured using a ConfigMap that contains the `production.yaml` configuration file. You can modify this configuration in the `deployment.yaml` file.

Key configuration options:

- **Rate Limiting**: Controls the number of requests per minute
- **Request Size Limit**: Maximum size of incoming requests (10MB by default)
- **LLM Providers**: Configuration for OpenAI and Together AI
- **Logging**: Log level and format

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

4. Verify the ConfigMap and Secrets exist:

```bash
kubectl get configmap webagent-config
kubectl get secret webagent-secrets
``` 