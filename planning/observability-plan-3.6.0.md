# Observability and model-service plan (3.6.0)

Status: implemented in 3.6.0 (2026-10-16); specified in [`specs/observability-drift/`](specs/observability-drift/README.md), which wins where the two differ · Base: `main` at v3.5.3 (`cac70c6`) · 2026-10-15 ·
Release: 3.6.0, **before** the next shadow week (so the week also produces real drift data)

## 1. Summary

3.5.0 mounted the observability API (`/api/v1/observability/*`) and 3.5.3 made the Grafana
dashboards and per-call model metrics work. But **no real traffic reaches drift detection or
alerts**: the model and document "services" that feed the detectors are simulations nothing calls,
drift state lives in each worker's memory, and the alert checks read keys that are never written.
So drift scores, the drift dashboard's score panels and `/alerts` are always empty.

3.6.0 makes them real and honest:

- Feed drift from real model calls and real document loads (numbers and file types only).
- Keep drift samples and alert state in one shared store, computed by one worker at a time, so
  every worker gives the same answer and it survives restarts.
- Compute alerts from that store and export them as Prometheus metrics, so they show on the drift
  dashboard and as Grafana alert rules.
- Remove the simulations and the duplicate self-hosted client.
- Fix what the dashboards and alerts say.

This finishes spec requirement IMP-128 (task 4.4, "wire drift hooks from production LLM calls"),
which 3.5.0 left undone.

## 2. What's true today (checked in the code, 2026-10-15)

| Part | What it does | Evidence |
|---|---|---|
| Observability API | Mounted behind the API key: `/metrics`, `/drift`, `/alerts`, `/health`, two baseline resets | `app/api/router.py`, `app/api/routes/observability.py` |
| Per-call model metrics | Real since 3.5.3: each Together or self-hosted request records count, duration, tokens, errors | `app/services/llm_call_metrics.py` |
| Model drift detectors | 4 detectors (latency, tokens, embeddings, performance) fed **only** by `LLMService.generate_response` | `app/services/llm_service.py:80-105` |
| `LLMService.generate_response` | **Simulation**: sleeps 0.1 s, returns "This is a simulated response…", counts words as tokens, fixed accuracy 0.95. **No caller** in `app`, demos, tests or scripts | `llm_service.py:62-73`; grep |
| Document drift detectors | 4 detectors (processing time, size, type, quality) fed **only** by `DocumentProcessingObservability.process_document` | `document_ingestion_service.py:656-700` |
| `DocumentProcessingObservability.process_document` | **Simulation**: sleeps 0.1 s, invents entities and a quality score. **No caller**. Real loads go through `DocumentLoaderService.load_document` (upload API, `ingest_documents`, `batch_load_documents`, `scripts/batch_document_loader.py`) | `document_ingestion_service.py:665-676`; grep |
| Drift state | In-process `deque`s and a module-level registry (`get_all_drift_stats`): each of the 2 workers has its own; lost on restart | `app/core/drift_detection.py:46-53, 762, 829` |
| Drift scores | Numerical and categorical: `1 − p-value`, drift when p < 0.05 (score > 0.95). Embedding: `1 − cosine similarity`, drift when similarity < 0.3 (score > 0.7). Performance: weighted relative change, drift when > 0.15 | `drift_detection.py:181-215, 293-330, 429-445, 576-641` |
| Drift dashboard gauges (3.5.3) | Red at 0.05 / 0.3 / 0.15 as score thresholds: **wrong** for the KS-based detectors and for embeddings | `backend/grafana/dashboards/model-drift-dashboard.json` |
| `/alerts` | Drift alerts say a model "requires retraining" (WebAgent uses hosted models it can't retrain). The error-rate, token and document-error alerts read `gpt-4_error_rate`, `gpt-4_total_tokens` and `document_processing_error_rate`, **keys nothing writes**, so they never fire. The token limit is a fixed 1,000,000 per worker since start | `observability.py:74-141`, `core/observability.py:316-328` |
| `/health` | Filters model metrics for names starting with `gpt-4`; real calls are recorded under the provider (`together`), so that section is always empty | `observability.py:167` |
| What feeds `llm_call_metrics` | Every Together request: agent calls, the guard's judges, triage's tie-breaker. One mixed stream: a change in the question mix would look like drift | `services/llm.py`, `security/*`, `services/triage.py` |
| Self-hosted model | Two clients: `SelfHostedChatModel` in `llm.py` (used when `llm.provider: self`) and `self_hosted_llm_service.py` (783 lines, used only by `services/examples/model_fine_tuning.py`) | grep |
| Kubernetes Grafana | Carries an older 6-panel copy of the backend dashboard and no drift dashboard; mounts no alerting provisioning | `kubernetes/monitoring/grafana.yaml` |

## 3. Goals and non-goals

Goals:

1. Drift scores move with real traffic, compared like with like.
2. One drift and alert answer for the whole server: the same from every worker, and after a restart.
3. Alerts that can fire, visible on a dashboard and in Grafana's alert list.
4. Nothing simulated left in a production module; one self-hosted client.
5. Every claim checked on the local stack before release: real calls for the samples, a fake Together server to force drift and alerts.

Non-goals:

- Retraining or switching models automatically. An alert asks a person to look.
- Answer-quality drift (no ground truth at run time). Quality stays a human review.
- Notifications (email, Slack): Grafana can add a contact point later; out of scope.
- A time-series database or new service (improvement plan, decision 3: files, no Redis).
- Changing what the agents do.

## 4. Decisions (all accepted 2026-10-15)

| # | Decision |
|---|---|
| D1 | Drift samples and alert state live in **one shared store on the data folder** (`data/drift/`), the same pattern as results and collections |
| D2 | **Embedding drift is kept, as centroids of at least 30 vectors**; individual vectors are never stored. If the tap costs more than 5 ms per call at p95, the detector is dropped instead |
| D3 | The reference window is the **first 50 samples after a reset** (per detector), then frozen. A fresh install counts as reset at its first sample. `POST /observability/{llm,doc}/reset_baseline` starts a new reference |
| D4 | `self_hosted_llm_service.py` **moves to `app/services/examples/`** beside the example that uses it |
| D5 | Alerts never say "retrain". Drift: "Model behaviour changed: check for a provider, model or prompt change, then reset the baseline once it's understood" |
| D6 | `deploy.sh` **generates the Kubernetes dashboard (and alert-rule) ConfigMaps from `backend/grafana/`**; the stale inline copy goes |
| D7 | **Drift alerts show on a dashboard** (section 5.5): exported as Prometheus metrics, shown in an "Alerts" row on the drift dashboard, and provisioned as Grafana alert rules |
| D8 | Release 3.6.0 **before** the next shadow week |
| D9 | **Failures and rate limits feed alerts, not a drift detector** (made while writing the spec): a failure share against a limit is clearer than a weighted-change score and needs no invented accuracy figure. The "performance drift" detector and gauge become the `model_errors` / `rate_limited` alerts and a failure-share gauge |

## 5. Design

### 5.1 What is measured (numbers only)

Model calls, from `llm_call_metrics.record()` (one call per request attempt):

- `TogetherWithUsage` (and `SelfHostedChatModel`) get a `call_label` field: the agent name when
  built by `get_agent_llm(agent)`; `get_trusted_llm(call_label=…)` takes one too, and its callers
  pass `judge` (`security/judge.py`) and `triage` (`services/triage.py`); anything else is `other`.
  Samples are keyed by label, so each detector compares like with like.
- `llm_speed:<label>`: seconds per 1,000 completion tokens (latency normalised by output length,
  so a longer answer isn't "drift"), for successful replies.
- `llm_tokens:<label>`: total tokens of the request, for successful replies.
- `llm_outcome` (not labelled): `ok`, `error` or `rate_limited`. **Rate limits (429) are recorded as
  `rate_limited`, not as failures**: they say the provider is busy, not that the model changed.
- The Together client takes its URL from `TOGETHER_API_BASE` (read today but ignored), so the
  local check can use a fake server instead of real credit.
- Detectors: `llm_speed` and `llm_tokens` (KS test) per label. Outcomes feed the `model_errors`
  and `rate_limited` alerts, not a drift detector (D9).

Document loads, from `DocumentLoaderService.load_document()` (covers the upload API, ingestion,
directory loads and the batch script; nested calls count once):

- `doc_processing_time` (seconds), `doc_size` (bytes), `doc_type` (file extension, categorical),
  `doc_outcome` (`ok` or `error`). Unsupported types are refused before loading (415) and not
  counted.

Embeddings (D2), from a wrapper around the object `vectordb.get_embeddings()` returns:

- Each worker adds vectors to an in-memory running sum. At 30 vectors it appends one aggregate
  (`sum`, `count`, dimension; the provider is in the detector name) to the store and starts again.
  Single vectors never leave memory. Reference centroid: the first aggregates reaching 50 vectors
  after the reset; current: the latest aggregates, up to 50 vectors (at least 30). Switching
  `WEBAGENT_EMBEDDINGS_PROVIDER` changes the dimension, so it resets the embedding baseline
  automatically.

Nothing records question text, document text, file names or single vectors (OBS-004).

### 5.2 The drift store (`app/core/drift_store.py`, new; under about 200 lines)

- `add(detector, value)`: appends one line (time, detector, value) to
  `data/drift/samples-<date>.jsonl` with `O_APPEND` (number lines are under 512 bytes, so writes from
  the two workers don't interleave; the larger embedding aggregates take a short lock on
  `append.lock`). Never raises; a failed write is logged at DEBUG.
- `reset(group)`: clears the frozen reference for every detector in the group (`llm`: model and
  embedding detectors; `doc`) in `data/drift/state.json`.
- Reference freeze (D3): once a detector has 50 samples after its reset, those values are copied
  into `state.json`, so the 30-day retention can't delete the reference.
- `compute()`: for each detector, current = its latest samples not in the reference, up to 50,
  within the retention period. A detector scores once it has a frozen reference and at least 30
  current samples (80 samples after a reset); otherwise it reports `enough_data: false` and no
  score. Scoring reuses the existing maths, moved into pure functions in `drift_detection.py`.
- **One worker computes per tick:** a loop started in `lifespan` (like `cleanup_daily`) runs every
  60 s and takes a non-blocking `flock` on `data/drift/compute.lock`. The worker that gets it
  computes and writes `state.json` (scores, drift yes/no, alerts, last drift times) atomically;
  the other just reads it. Both set their gauges from the same file, so `/metrics` and the API
  agree.
- Drift transitions (no → yes) are logged once at WARNING and to MLflow by the computing worker
  only.
- Retention: sample files older than `WEBAGENT_RESULTS_RETENTION_DAYS` (30) are deleted at startup
  and then at most once a day by the computing worker. Volume: about 25 lines per research run.
- Settings (added to `.env.local.example`): `WEBAGENT_DRIFT_ENABLED` (default `true`; when false,
  nothing is recorded and the routes answer empty), `WEBAGENT_DRIFT_WINDOW` (50),
  `WEBAGENT_DRIFT_INTERVAL_SECONDS` (60), `WEBAGENT_TOKEN_ALERT_DAILY` (0 = off), and
  `TOGETHER_API_BASE` (now honoured).

### 5.3 Alerts and health, from the store

Computed with the drift state and written to `state.json`, so every worker returns the same list:

| Alert | Fires when | Severity |
|---|---|---|
| `model_drift` | Any `llm_speed`, `llm_tokens` or embedding detector reports drift | warning |
| `model_errors` | Failure share of model requests over the last hour > 5%, with at least 20 requests (rate limits excluded) | critical |
| `rate_limited` | More than 20% of model requests in the last hour were rate-limited, with at least 20 requests | warning |
| `token_usage` | Tokens in the last 24 h exceed `WEBAGENT_TOKEN_ALERT_DAILY` (off by default; the Together key has $25 in total) | warning |
| `document_drift` | Any `doc_*` detector reports drift | warning |
| `document_errors` | Failed document loads over the last 24 h > 10%, with at least 10 loads | critical |

- `/alerts` returns these (`name`, `severity`, `service`, `message`, `since`, and the old
  `timestamp` = `since`). Messages follow D5.
- `/health` derives its status from them (critical → unhealthy, warning → degraded) and lists
  model figures by provider and label from the store; the `gpt-4` filter goes.
- `/drift` returns per detector: score, drift yes/no, `enough_data`, reference and current sample
  counts, reference start, last drift time. Fields are added; none is removed
  (`retrain_recommendations` stays, always empty).
- `/metrics` (observability JSON) reads the same state instead of per-worker registries.

### 5.4 Removing the simulations

- `LLMService` keeps only `get_drift_status()` and `reset_drift_baselines()`, backed by the store;
  `generate_response` and `get_embedding` are deleted.
- `DocumentProcessingObservability` keeps only its status and reset methods; `process_document`
  is deleted.
- The module-level detector instances, the detector classes and the `get_all_drift_stats()` /
  `should_retrain_any_model()` registry are replaced by the store; the invented
  `record_data_drift("llm_response", 0.8)` and the unused `_build_llm_without_tags` go.
- `self_hosted_llm_service.py` moves to `app/services/examples/` (D4); its import paths in the
  example and docs are updated.

### 5.5 Drift alerts on a dashboard (D7)

- **Metrics**, set from `state.json` by each worker (multiprocess mode `max`, so one series per
  label in `/metrics`):
  - `webagent_alert_active{alert, severity}`: 1 while an alert fires, else 0
  - `webagent_drift_detected{detector}`: 1 or 0
  - `webagent_drift_enough_data{detector}`: 1 or 0
  - `webagent_drift_current_samples{detector}`: current samples (vectors for embeddings)
  - `data_drift_score{component}`: the score with enough data, 0 without
  - `webagent_drift_samples_total{detector}`: counter of samples recorded
- **Drift dashboard, new "Alerts" row at the top:**
  - Stat "Active alerts": `sum(webagent_alert_active) or vector(0)`, green at 0, red above.
  - State timeline "Alerts over time": one row per alert, from `webagent_alert_active`.
  - Table "Detectors": detector, enough data, drift, score, current samples.
- **Grafana alert rules** (`backend/grafana/provisioning/alerting/webagent-alerts.yaml`): one rule
  per alert, firing while its `webagent_alert_active` series is 1, so alerts also show in Grafana's
  Alerting page and an
  "Alert list" panel. No contact point is configured (non-goal); adding one later is a Grafana
  setting, not a code change. The folder also stops Grafana's "can't read alerting provisioning
  files" log error.
- **Gauge thresholds:** red at 0.95 for the KS and chi-square detectors and 0.7 for embeddings;
  the "performance drift" gauge becomes the model failure share over the last hour (red at 0.05,
  D9). Each gauge's description says what its value means.

### 5.6 Kubernetes (D6)

On every deploy (not only the first, which is when `deploy.sh` applies the monitoring manifests
today), `deploy.sh` re-applies the Grafana data source (so the 3.5.3 uid and port fix reaches
existing clusters), renders the `grafana-dashboard-definitions` ConfigMap from
`backend/grafana/dashboards/*.json` and a `grafana-alerting` ConfigMap from
`backend/grafana/provisioning/alerting/`, and restarts Grafana (it reads alert-rule files only at
startup). `grafana.yaml` mounts the alerting ConfigMap and drops its inline dashboard copy.

## 6. Requirements

Specified in [`specs/observability-drift/requirements.md`](specs/observability-drift/requirements.md):
OBS-001 – OBS-021 and OBS-NFR-01 – OBS-NFR-07, each with acceptance criteria and its check. The
spec's numbering is the only one; this plan no longer keeps its own list.

## 7. Tasks

Specified in [`specs/observability-drift/tasks.md`](specs/observability-drift/tasks.md): 17 tasks
in dependency order, about 43 h (5.5 days), from the scoring functions to the release.

## 8. Upgrade notes (for the CHANGELOG)

- Drift is computed from real traffic and shared by all workers; samples and state live in
  `data/drift/` and samples are deleted after `WEBAGENT_RESULTS_RETENTION_DAYS`. A fresh install
  collects its reference from its first 50 samples per detector.
- `/observability/drift` adds `computed_at`, `detectors`, `enough_data`, sample counts and
  `reference_start`; `retrain_recommendations` stays but is always empty; the per-service blocks
  report `performance_drift: null` (D9). `/alerts` returns the alerts in 5.3 (`name` and `since`
  added; `timestamp` kept). `/health` derives its status from the alerts. Nothing is removed.
  Alerts no longer say "retrain".
- A detector scores only after a 50-sample reference and 30 more samples (80 after a reset).
- New Prometheus metrics: `webagent_alert_active`, `webagent_drift_detected`,
  `webagent_drift_enough_data`, `webagent_drift_current_samples`, `webagent_drift_samples_total`;
  `data_drift_score` is one series per detector (multiprocess `max`), 0 without enough data.
- The drift dashboard's "performance drift" gauge becomes "model failure share, last hour" (D9).
- New settings: `WEBAGENT_DRIFT_ENABLED`, `WEBAGENT_DRIFT_WINDOW`,
  `WEBAGENT_DRIFT_INTERVAL_SECONDS`, `WEBAGENT_TOKEN_ALERT_DAILY`; `TOGETHER_API_BASE` is now
  honoured by the Together client.
- Removed: `LLMService.generate_response`, `LLMService.get_embedding`,
  `DocumentProcessingObservability.process_document` (simulations nobody called), the detector
  classes, `get_all_drift_stats()`, `should_retrain_any_model()` and `_build_llm_without_tags`.
  `self_hosted_llm_service.py` moved to `app/services/examples/`.
- `scipy` is declared in `backend/requirements.txt` (it was already installed as a dependency).
- Kubernetes: on every deploy, `deploy.sh` re-applies the Grafana data source, replaces the
  dashboard ConfigMap with `backend/grafana/dashboards/`, adds the alert rules and restarts
  Grafana.

## 9. Risks and limits

- **Low traffic.** After a reset each detector needs a 50-sample reference and 30 current samples
  (80) before it scores. A research run makes about 8 model calls spread over several agents, so
  per-agent model drift needs roughly 30–80 research runs, depending on the agent; document drift
  needs 80 loads, which may take weeks. The detector table and `enough_data` make this visible
  instead of showing a misleading score.
- **What drift can show.** Speed and token drift catch a provider change, overload or a prompt
  change; they say nothing about answer quality.
- **Shared folder.** Fine for one replica (improvement plan, decision 5). More replicas would need a
  shared ReadWriteMany volume, as for results. `flock` works on local and block-storage volumes;
  not on every network filesystem.
- **Together credit.** The local check (spec task 15) needs a few real calls: well under $1 of the
  $25. The drift and the alerts are forced with the fake Together server, not real spend.
- **Embedding threshold.** The existing similarity threshold (0.3) is loose; it may need tuning once
  real centroids exist.
- **OpenAI calls** (the coding agents, when an OpenAI key is set) are not measured.

## 10. Open questions

1. A contact point for Grafana alerts (email or Slack) once someone should be told without opening
   Grafana. Not needed for 3.6.0.
