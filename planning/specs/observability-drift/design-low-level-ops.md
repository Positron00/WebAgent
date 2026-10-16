# Low-level design: dashboards, alert rules, deployment and docs

Paths are relative to the repo root. Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Files

| File | New? | Purpose |
|---|---|---|
| `backend/grafana/dashboards/model-drift-dashboard.json` | changed | Alerts row; detector table; gauge thresholds and descriptions (OBS-016) |
| `backend/grafana/provisioning/alerting/webagent-alerts.yaml` | new | One Grafana alert rule per alert (OBS-017) |
| `backend/kubernetes/deploy.sh` | changed | On every deploy: re-apply the Grafana data source, apply the dashboard and alert-rule ConfigMaps, restart Grafana (OBS-018) |
| `backend/kubernetes/monitoring/grafana.yaml` | changed | Drop the inline dashboard ConfigMap; mount the alerting ConfigMap |
| `scripts/fake_together.py` | new | Dev-only fake Together server (backend design §12) |
| `.env.local.example` | changed | Four drift settings and `TOGETHER_API_BASE` (OBS-NFR-04) |
| Docs (§5) | changed | RUNBOOK, API reference, READMEs, production checklist, specs, CHANGELOG |

## 2. Drift dashboard (`model-drift-dashboard.json`)

Panels, top to bottom. Every target uses `{"type": "prometheus", "uid": "prometheus"}`.

**Row "Alerts" (new, first):**

| Panel | Type | Query | Notes |
|---|---|---|---|
| Active alerts | stat | `sum(webagent_alert_active) or vector(0)` | Thresholds: green 0, red 1 |
| Alerts over time | state-timeline | `max by (alert) (webagent_alert_active)` | Legend `{{alert}}`; value mapping 0 → "ok" (green), 1 → "active" (red) |
| Detectors | table | `max by (detector) (webagent_drift_enough_data)`, `max by (detector) (webagent_drift_detected)`, `max by (detector) (label_replace(data_drift_score, "detector", "$1", "component", "(.*)"))`, `max by (detector) (webagent_drift_current_samples)` | Instant queries joined on `detector` (Grafana "Join by field" transformation); columns: detector, enough data, drift, score, current samples. (`webagent_drift_samples_total` restarts with the process, so it isn't used here) |

**Row "Drift" (existing panels, retitled):**

| Panel | Query | Red at |
|---|---|---|
| Model drift scores (timeseries) | `data_drift_score{component=~"llm_.*\|embeddings:.*"}` | — |
| Document drift scores (timeseries) | `data_drift_score{component=~"doc_.*"}` | — |
| Model speed drift (gauge) | `max(data_drift_score{component=~"llm_speed:.*"}) or vector(0)` | 0.95 |
| Embedding drift (gauge) | `max(data_drift_score{component=~"embeddings:.*"}) or vector(0)` | 0.7 |
| Model failure share, last hour (gauge, replaces "performance drift") | `sum(increase(llm_requests_total{status="error"}[1h])) / clamp_min(sum(increase(llm_requests_total[1h])), 1)` | 0.05 |

Each gauge's description states its rule, for example "1 − p-value of a KS test between the
reference and current samples; drift above 0.95 (p < 0.05)". The failure-share gauge includes rate-
limited requests (Prometheus can't tell them apart from `llm_requests_total`'s `status` label);
its description says so, and the `model_errors` alert, which excludes them, is the authoritative
signal.

**Row "Model performance" (unchanged from 3.5.3):** latency p95, errors per minute, tokens per
minute, document processing time.

## 3. Grafana alert rules (`provisioning/alerting/webagent-alerts.yaml`)

Grafana 11 file provisioning, one rule per alert in OBS-012:

```yaml
apiVersion: 1
groups:
  - orgId: 1
    name: webagent
    folder: WebAgent
    interval: 1m
    rules:
      - uid: webagent-model-drift
        title: Model drift
        condition: C
        data:
          - refId: A
            relativeTimeRange: {from: 300, to: 0}
            datasourceUid: prometheus
            model: {expr: 'max(webagent_alert_active{alert="model_drift"}) or vector(0)', instant: true}
          - refId: C
            datasourceUid: __expr__
            model: {type: threshold, expression: A, conditions: [{evaluator: {type: gt, params: [0]}}]}
        noDataState: OK
        execErrState: Error
        for: 0s
        labels: {severity: warning, service: llm}
        annotations:
          summary: Model behaviour changed. See /api/v1/observability/alerts for the detectors.
          runbook: docs/RUNBOOK.md#drift-and-alerts
      # … model_errors (critical), rate_limited, token_usage, document_drift, document_errors (critical)
```

No contact point or notification policy is provisioned; rules appear on Grafana's Alerting page
and in an "Alert list" panel. Compose already mounts `./grafana/provisioning`, so the new folder is
picked up, and Grafana's "can't read alerting provisioning files" log error stops.

## 4. Kubernetes (`deploy.sh`, `monitoring/grafana.yaml`)

- `grafana.yaml`: remove the `grafana-dashboard-definitions` ConfigMap's inline JSON (the
  Deployment keeps mounting a ConfigMap of that name at `/var/lib/grafana/dashboards`); add a
  volume and mount for a `grafana-alerting` ConfigMap at `/etc/grafana/provisioning/alerting`.
- `deploy.sh`, after the monitoring-namespace block (which runs only on a first deploy), on
  **every** deploy:

  ```bash
  GRAFANA_DIR="../grafana"
  # The data source (uid "prometheus", port 9090 since 3.5.3) must reach existing clusters too.
  kubectl apply -f monitoring/grafana-datasource.yaml
  kubectl create configmap grafana-dashboard-definitions -n monitoring \
    --from-file="$GRAFANA_DIR/dashboards" --dry-run=client -o yaml | kubectl apply -f -
  kubectl create configmap grafana-alerting -n monitoring \
    --from-file="$GRAFANA_DIR/provisioning/alerting" --dry-run=client -o yaml | kubectl apply -f -
  # Grafana reloads dashboard files on its own, but reads alert-rule files only at startup.
  kubectl rollout restart deployment/grafana -n monitoring
  ```

  The script already runs from its own folder (`cd "$(dirname "$0")"`, 3.2.6), so the relative
  path is stable.
- Test (`backend/tests/test_deploy_script.py`, new): runs `deploy.sh` with a fake `kubectl` in a
  temp `PATH` (the same technique used to check 3.2.6) and asserts both ConfigMaps contain every
  file in `backend/grafana/dashboards/` and `backend/grafana/provisioning/alerting/`, and that
  Grafana is restarted.

## 5. Docs

| Doc | Change |
|---|---|
| `docs/RUNBOOK.md` | New "Drift and alerts" section: what each alert means, what to check, how to reset a baseline (`curl -X POST … /llm/reset_baseline`), why a fresh install shows "not enough data", where samples live and for how long; the dashboards paragraph from 3.5.3 updated |
| `backend/docs/api_reference.md` | Observability section: `/drift`, `/alerts`, `/health`, `/metrics` fields (OBS-013) |
| `backend/README.md` | Endpoint list (observability fields); settings table (drift settings, `TOGETHER_API_BASE`) |
| `README.md` | "Advanced Observability" feature bullets: real drift, alerts on the dashboard |
| `backend/PRODUCTION_READINESS.md` | Checklist item: drift store on the data volume; alerts visible in Grafana; no notifications until a contact point is set |
| `.env.local.example` | `WEBAGENT_DRIFT_ENABLED`, `WEBAGENT_DRIFT_WINDOW`, `WEBAGENT_DRIFT_INTERVAL_SECONDS`, `WEBAGENT_TOKEN_ALERT_DAILY`, `TOGETHER_API_BASE` |
| `planning/specs/platform-improvements/tasks.md` | Task 4.4 done in 3.6.0 (IMP-128 complete) |
| `planning/observability-plan-3.6.0.md` | Status line set to "implemented in 3.6.0" |
| `backend/requirements.txt` | `scipy` declared (see the backend design) |
| `CHANGELOG.md` | 3.6.0 entry with upgrade notes (plan §8, plus decision 9) |

## 6. Release

Version 3.6.0 in every file that names the current release (package.json and lock, README and
Vercel badges, production checklist heading, architecture doc, API reference, Kubernetes README,
`deploy.sh` default, `deployment.yaml`, security spec README and shadow-week runbook, improvement
plan status line). Then: branch `release/3.6.0`; PR to `main`; both CI checks green; merge through
the PR; tag `v3.6.0` on the merge commit; fast-forward `dev`.
