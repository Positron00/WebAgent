# WebAgent Monitoring Setup

This directory contains Kubernetes manifests for setting up monitoring for the WebAgent backend using Prometheus and Grafana.

## Components

- **Prometheus**: Collects and stores metrics from the WebAgent backend and Kubernetes cluster
- **Grafana**: Visualizes metrics with pre-configured dashboards
- **ServiceMonitor**: Configures Prometheus to scrape metrics from WebAgent services

## Deployment

### Prerequisites

- Kubernetes cluster with RBAC enabled
- kubectl configured to access your cluster
- WebAgent backend deployed with Prometheus metrics enabled

### Installation

1. Create the monitoring namespace:

```bash
kubectl apply -f namespace.yaml
```

2. Deploy Prometheus:

```bash
kubectl apply -f prometheus.yaml
```

3. Deploy Grafana:

```bash
kubectl apply -f grafana.yaml
```

4. Configure Grafana datasource:

```bash
kubectl apply -f grafana-datasource.yaml
```

### Accessing Dashboards

1. Port-forward the Grafana service:

```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000
```

2. Open your browser and navigate to `http://localhost:3000`

3. Log in with the credentials:
   - Username: admin
   - Password: WebAgent-Admin-2025!

4. The WebAgent Backend Dashboard should be available in the dashboards list

## Available Metrics

The WebAgent backend exposes the following metrics:

- **HTTP Request Metrics**:
  - Request counts by status code
  - Request duration by endpoint
  
- **LLM Metrics**:
  - Request counts by provider and model
  - Request duration by provider and model
  - Token usage by provider, model, and type (input/output)
  
- **Task Metrics**:
  - Task completion rates by status
  - Task duration

## Alerting

Alerting can be configured in Grafana based on the metrics collected. Common alerts might include:

- High error rates (5xx status codes)
- Slow response times
- High token usage
- Failed tasks

## Troubleshooting

If metrics are not appearing in Grafana:

1. Check that the WebAgent pods have the correct annotations:
   ```
   prometheus.io/scrape: "true"
   prometheus.io/port: "8000"
   prometheus.io/path: "/metrics"
   ```

2. Verify Prometheus is scraping the targets:
   ```bash
   kubectl port-forward -n monitoring svc/prometheus-server 9090:9090
   ```
   Then open `http://localhost:9090/targets` in your browser

3. Check Prometheus logs:
   ```bash
   kubectl logs -n monitoring -l app=prometheus
   ```

4. Check Grafana logs:
   ```bash
   kubectl logs -n monitoring -l app=grafana
   ``` 