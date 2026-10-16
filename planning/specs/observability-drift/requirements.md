# Requirements: real drift and alerts (3.6.0)

Each requirement has an ID, a statement, acceptance criteria, and how it is checked. All are
"must" for 3.6.0 unless marked "should".

Terms:

- **Sample**: one number (or file type) recorded for one detector, with a timestamp.
- **Detector**: a named stream of samples compared over time, for example `llm_speed:supervisor`.
- **Group**: the detectors one reset covers: `llm` (model and embedding detectors) or `doc`.
- **Reference**: the first `WEBAGENT_DRIFT_WINDOW` (default 50) samples a detector receives after
  its group's last reset (for embeddings: the first aggregates reaching that many vectors), frozen
  once complete.
- **Current**: the detector's latest samples that are not part of the reference and are within the
  retention period, up to `WEBAGENT_DRIFT_WINDOW` (for embeddings: the latest aggregates, up to that
  many vectors).
- **Enough data**: a complete (frozen) reference and at least 30 current samples (vectors for
  embeddings). After a reset a detector therefore needs `WEBAGENT_DRIFT_WINDOW` + 30 samples
  (80 by default) before it scores.
- **Drift store**: the files under `data/drift/` on the backend's data volume.
- **Compute tick**: one recomputation of every detector and alert, every
  `WEBAGENT_DRIFT_INTERVAL_SECONDS` (default 60).
- **Call label**: who made a model call: an agent name, `judge`, `triage` or `other`.

## 1. Samples from real traffic

**OBS-001 Model calls (must).** Each model request attempt through the Together client or the
self-hosted model adds:
- `llm_speed:<label>`: seconds per 1,000 completion tokens, under the call's label (only for a
  successful reply with completion tokens);
- `llm_tokens:<label>`: total tokens of the request, under the call's label (only for a successful
  reply that reports usage);
- `llm_outcome` (not labelled): `ok`, `error` or `rate_limited` (an HTTP 429 is `rate_limited`,
  never `error`; for the self-hosted model the status comes from the HTTP error, if any).
- Accept: an agent's call is labelled with its agent name; the judge's with `judge`; the triage
  tie-breaker's with `triage`; anything else with `other`.
- Check: pytest with fake Together replies (200, 429, 500), one per label kind.

**OBS-002 Document loads (must).** Each call to `DocumentLoaderService.load_document` adds one
`doc_processing_time` (seconds), one `doc_size` (bytes), one `doc_type` (lower-case file extension)
and one `doc_outcome` (`ok` or `error`). A file refused before loading (unsupported type, 415) adds
nothing. Loading a directory adds one set per file.
- Check: pytest through `POST /api/v1/documents/upload` and `load_directory` with temp files.

**OBS-003 Embeddings (must; should if the 5 ms limit of OBS-014 can't be met).** Vectors computed
by the embeddings object `vectordb.get_embeddings()` returns are summed in memory; each time a
worker has 30, it appends one aggregate (sum, count, dimension, provider) for
`embeddings:<provider>`. The reference and current centroids are built from aggregates totalling
at least 50 vectors. A change of dimension (a new embeddings provider) starts a new reference.
- Check: pytest with a fake embeddings object.

**OBS-004 Privacy (must).** Nothing under `data/drift/` contains question text, history, document
text, file names, file paths or a single embedding vector.
- Check: pytest runs a fake task, a fake upload and 30 fake embeddings, all containing a marker
  string, and asserts the marker appears in no file; asserts every embedding record has
  `count ≥ 30`.

## 2. The shared store

**OBS-005 Same answer everywhere (must).** Drift results, alerts and health read the same state
from the drift store, so any worker returns the same answer, and a restart loses nothing but
samples from an in-memory embedding sum not yet written.
- Check: pytest with two store instances on one folder (two workers) and a new instance after
  writes (restart).

**OBS-006 Not enough data (must).** A detector without enough data reports
`enough_data: false`, no score and no drift, and raises no alert.
- Check: pytest with 49 reference samples (not frozen) and with a frozen reference plus 29 and 30
  current samples.

**OBS-007 Drift is detected (must).** For KS detectors (speed, tokens, processing time, size) and
the chi-square detector (type), drift means p < 0.05 (score `1 − p` > 0.95). For embeddings, drift
means cosine similarity of the centroids < 0.3 (score > 0.7). A shifted distribution is reported as
drift within one compute tick; samples from the same distribution are not (fixed random seed).
- Check: pytest with seeded samples for each kind.

**OBS-008 Frozen reference (must).** Once a detector has its reference, it is kept in the store's
state and survives the deletion of the sample files it came from.
- Check: pytest: reference collected, sample files aged past retention and deleted, reference and
  score unchanged.

**OBS-009 Reset (must).** `POST /api/v1/observability/llm/reset_baseline` and
`/doc/reset_baseline` clear the references of their group; the next samples form a new reference.
Paths, API key and response bodies are unchanged.
- Check: pytest.

**OBS-010 One worker computes (must).** Per tick, only the worker holding a non-blocking lock on
`data/drift/compute.lock` computes and writes the state; the other reads it. A drift or alert
transition (off → on) is logged once at WARNING (and once to MLflow for drift).
- Check: pytest with two compute loops on one folder: one state write per tick, one log line per
  transition.

**OBS-011 Retention (must).** Sample files older than `WEBAGENT_RESULTS_RETENTION_DAYS` are
deleted at startup and then at most once a day by the worker that computes.
- Check: pytest with fake dates.

## 3. Alerts and API

**OBS-012 Alerts (must).** Each compute tick evaluates:

| Alert | Fires when | Severity |
|---|---|---|
| `model_drift` | Any `llm_speed:*`, `llm_tokens:*` or `embeddings:*` detector reports drift | warning |
| `model_errors` | Of `llm_outcome` samples in the last hour, excluding `rate_limited`, more than 5% are `error`, with at least 20 | critical |
| `rate_limited` | More than 20% of `llm_outcome` samples in the last hour are `rate_limited`, with at least 20 | warning |
| `token_usage` | The sum of `llm_tokens:*` samples in the last 24 h exceeds `WEBAGENT_TOKEN_ALERT_DAILY` (0 = off, the default) | warning |
| `document_drift` | Any `doc_processing_time`, `doc_size` or `doc_type` detector reports drift | warning |
| `document_errors` | Of `doc_outcome` samples in the last 24 h, more than 10% are `error`, with at least 10 | critical |

An alert stays active while its condition holds and clears on the first tick it doesn't. No alert
message contains "retrain"; drift messages say to check for a provider, model or prompt change and
reset the baseline once understood.
- Check: pytest, one case per alert: fires, stays, clears.

**OBS-013 API (must).**
- `GET /observability/drift` keeps `global_drift_stats`, `llm_service_drift`, `doc_service_drift`
  and `retrain_recommendations` (now always an empty list, kept for existing clients) and adds
  `computed_at` and `detectors` (per detector: `group`, `kind`, `enough_data`, `score`, `drift`,
  `reference_count`, `current_count`, `reference_start`, `last_drift_at`).
- `GET /observability/alerts` returns the active alerts: `name`, `severity`, `service`,
  `message`, `since` (the old `severity`, `service`, `message`, `timestamp` keys stay; `timestamp`
  equals `since`).
- `GET /observability/health` derives `overall_status` from the alerts (critical → `unhealthy`,
  warning → `degraded`, none → `healthy`) and lists model figures by provider and call label.
- `GET /observability/metrics` keeps `model_metrics`, `document_metrics` and `timestamp`, filled
  from the drift store.
- All need the API key, as today.
- With `WEBAGENT_DRIFT_ENABLED=false`, no samples are recorded, the compute loop doesn't start, and
  the routes answer with no detectors, no alerts and `overall_status: healthy`, whatever an older
  `state.json` holds.
- Check: pytest contract tests (extends `test_observability_api.py`).

## 4. Metrics and dashboards

**OBS-014 Cost (must).** Recording samples adds under 5 ms per model call at p95, embeddings
included (else OBS-003 is dropped, per decision 2).
- Check: pytest timing on 1,000 recorded calls with a fake store folder.

**OBS-015 Prometheus metrics (must).** Each worker sets, from the drift state on every tick:
`webagent_alert_active{alert,severity}` (1 or 0, for every alert in OBS-012),
`webagent_drift_detected{detector}`, `webagent_drift_enough_data{detector}`,
`webagent_drift_current_samples{detector}` and `data_drift_score{component}` (the score with enough
data, 0 without), all with multiprocess mode `max`. Each recorded sample increments
`webagent_drift_samples_total{detector}` (by the vector count for an embedding aggregate).
- Check: pytest reads `/metrics` in multiprocess mode with two fake workers: one series per label.

**OBS-016 Drift dashboard (must).** The drift dashboard has an "Alerts" row at the top (active
alerts count; alerts over time; a detector table with enough data, drift, score and current
samples), and
its gauges turn red at the score that means drift: 0.95 for KS and chi-square detectors, 0.7 for
embeddings, and 0.05 for the model failure share. Every query uses exported metrics through the
data source with uid `prometheus`.
- Check: `test_grafana_dashboards.py` (extended); local stack check (OBS-020).

**OBS-017 Grafana alert rules (must).** `backend/grafana/provisioning/alerting/` provisions one
Grafana alert rule per alert in OBS-012, firing while `webagent_alert_active{alert="…"}` is 1, in
folder "WebAgent", with the alert's severity and service as labels. No contact point is
provisioned.
- Check: YAML test (one rule per alert, valid queries); Grafana API on the local stack lists them.

**OBS-018 Kubernetes Grafana (must).** On every deploy, `deploy.sh` re-applies the Grafana data
source (so the 3.5.3 uid and port fix reaches existing clusters), applies ConfigMaps built from
`backend/grafana/dashboards/*.json` and `backend/grafana/provisioning/alerting/*.yaml`, and
restarts Grafana (it reads alert-rule files only at startup); `grafana.yaml` mounts both
ConfigMaps and no longer carries its own dashboard copy.
- Check: test that runs `deploy.sh` with a fake `kubectl`: the data source is applied, both
  ConfigMaps contain every file, Grafana is restarted.

## 5. Cleanup

**OBS-019 No simulations (must).** `llm_service.py` and `document_ingestion_service.py` contain no
simulated generation, embedding or document processing; `self_hosted_llm_service.py` is not under
`app/services/` (it is in `app/services/examples/`); the module-level detector instances, the
detector classes and registry (`register_drift_detector`, `get_all_drift_stats`,
`should_retrain_any_model`), the ad-hoc `record_data_drift("llm_response", 0.8)` and the unused
`_build_llm_without_tags` are gone.
- Check: grep test; import scan passes.

## 6. Check on the local stack

**OBS-020 Seen working (must).** On the local stack (`npm run stack:up`):
- a few real model calls and one upload move the sample counts (real Together, well under $1);
- the Together client is then pointed at the dev-only fake server (`scripts/fake_together.py`,
  through `TOGETHER_API_BASE`, which the client now honours; see the backend design): after a
  reset, a slower fake raises `model_drift`, an error rate raises `model_errors`, a 429 rate raises
  `rate_limited`, and uploads with a different file-type mix raise `document_drift`;
- each alert is visible in `/observability/alerts`, on the drift dashboard and in Grafana's alert
  list, and clears when the change ends.
- Check: notes in the PR, with the commands used.

**OBS-021 Together base URL (must).** The Together client posts to
`<TOGETHER_API_BASE>/completions` (default `https://api.together.xyz/v1/completions`, today's
URL), so a test or a local check can point it at another server.
- Check: pytest that the client uses the configured URL.

## 7. Non-functional requirements

| ID | Requirement | Check |
|---|---|---|
| OBS-NFR-01 | All existing tests pass; new code has tests; CI is green on the PR before merge | CI |
| OBS-NFR-02 | No new service and no new runtime dependency. `scipy` (used by the scoring and already by ContextDefender, but installed only as another package's dependency) is declared in `backend/requirements.txt` at the version `constraints.txt` already pins (1.13.1) | diff of requirements |
| OBS-NFR-03 | Existing observability routes keep their paths, API key and fields; changes are additions (OBS-013) | contract tests |
| OBS-NFR-04 | New settings are in `.env.local.example` with their defaults | review |
| OBS-NFR-05 | New files stay under about 200 lines | review |
| OBS-NFR-06 | Every change of default or behaviour is in the CHANGELOG's upgrade notes | review |
| OBS-NFR-07 | Released through a PR to `main` with both CI checks green, then tagged `v3.6.0` | release |
