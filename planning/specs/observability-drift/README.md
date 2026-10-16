# Spec: real drift and alerts (3.6.0)

Status: implemented in 3.6.0 (2026-10-16); reviewed against the code for 3.6.1 (notes below) · Source plan: [`planning/observability-plan-3.6.0.md`](../../observability-plan-3.6.0.md) ·
Base: `main` at v3.5.3 (`cac70c6`) · 2026-10-15 · Release: 3.6.0, before the next shadow week

## What this work does

3.5.0 mounted the observability API and 3.5.3 made the Grafana dashboards and per-call model
metrics work, but nothing real reaches drift detection or alerts. The model and document
"services" that feed the detectors are simulations nothing calls, drift state lives in each
worker's memory, and the alert checks read keys that are never written.

3.6.0:

- feeds drift from real model calls and real document loads (numbers and file types only);
- keeps drift samples and alert state in one shared store on the data folder, computed by one
  worker at a time, so every worker gives the same answer and it survives restarts;
- computes alerts from that store and exports them as Prometheus metrics, shown on the drift
  dashboard and as Grafana alert rules;
- removes the simulations and moves the duplicate self-hosted client out of `app/services/`;
- fixes what the dashboards and alerts say.

It finishes [`platform-improvements`](../platform-improvements/README.md) requirement IMP-128
(task 4.4, "wire drift hooks from production LLM calls"), which 3.5.0 left undone.

## Documents

| File | What it holds |
|---|---|
| [`requirements.md`](requirements.md) | What must be true in 3.6.0: numbered requirements (`OBS-…`, `OBS-NFR-…`) with acceptance criteria and how each is checked |
| [`design-high-level.md`](design-high-level.md) | Components, flows (a model call, a document load, a compute tick, a reset, an alert), failure handling, data, deployment, decisions, limits |
| [`design-low-level-backend.md`](design-low-level-backend.md) | Modules, function signatures, file formats, detectors, alerts, API shapes, metrics, removals, tests |
| [`design-low-level-ops.md`](design-low-level-ops.md) | Grafana dashboard and alert-rule files, Compose, Kubernetes ConfigMaps, docs, release |
| [`tasks.md`](tasks.md) | Tasks in dependency order, each with the requirements it covers and the check that proves it |

Read in that order. Requirements say *what*; the designs say *how*; tasks say *in what order*.
When documents disagree, `requirements.md` wins, then the designs, then the plan.

## Scope

In scope: everything in the plan's sections 5–7.

Out of scope:

- Retraining or switching models automatically; answer-quality drift (no ground truth at run time).
- Notifications (email, Slack). Grafana can add a contact point later without code changes.
- OpenAI calls (the coding agents when `OPENAI_API_KEY` is set) are not measured, as in 3.5.3:
  only the Together client and the self-hosted model record per-call metrics.
- A time-series database or any new service (improvement plan, decision 3).
- The security guard and triage (their own specs).

## Decisions

All made with the plan on 2026-10-15.

1. **One shared store** for drift samples and alert state on the data folder (`data/drift/`).
2. **Embedding drift kept, as centroids of at least 30 vectors**; single vectors are never stored.
   Dropped instead if the tap costs more than 5 ms per call at p95.
3. **Reference = the first 50 samples after a reset**, per detector, then frozen. A fresh install
   counts as reset at its first sample. A detector scores once it has that reference and 30 more
   (current) samples.
4. **`self_hosted_llm_service.py` moves to `app/services/examples/`.**
5. **Alerts never say "retrain"**: WebAgent uses hosted models it can't retrain.
6. **`deploy.sh` generates the Kubernetes dashboard and alert-rule ConfigMaps from
   `backend/grafana/`.**
7. **Alerts show on a dashboard**: Prometheus metrics, an "Alerts" row on the drift dashboard, and
   Grafana alert rules.
8. **Release before the next shadow week.**
9. **Failures and rate limits feed alerts, not a drift detector.** (Made while writing this spec.)
   A failure share against a fixed limit is clearer than a weighted-change score and needs no
   invented accuracy figure. The plan's "performance drift" detector is replaced by the
   `model_errors` and `rate_limited` alerts, and the dashboard's performance gauge becomes a
   failure-share gauge.

## Findings that shaped the spec

Checked in the code on 2026-10-15; details in the plan's section 2.

1. `LLMService.generate_response` and `DocumentProcessingObservability.process_document` are
   simulations with no caller in `app`, demos, tests or scripts. Real document loads all go through
   `DocumentLoaderService.load_document` (upload API, `ingest_documents`, `batch_load_documents`,
   `scripts/batch_document_loader.py`).
2. The judge (`security/judge.py`) and the triage tie-breaker (`services/triage.py`) use
   `get_trusted_llm()`, the same Together client as the agents, so model samples must be labelled
   by caller or a change in the question mix would look like drift.
3. Drift scores mean different things: KS and chi-square give `1 − p-value` (drift above 0.95);
   embeddings give `1 − cosine similarity` (drift above 0.7). The 3.5.3 gauges used 0.05, 0.3
   and 0.15 as score thresholds, which is wrong.
4. `/alerts` reads `gpt-4_error_rate`, `gpt-4_total_tokens` and `document_processing_error_rate`,
   which nothing writes; `/health` filters for `gpt-4`. Neither can ever report a problem.
5. `deploy.sh` applies the monitoring manifests only when the `monitoring` namespace doesn't exist
   yet, so regenerated Grafana ConfigMaps must be applied outside that branch.
6. `data_drift_score` has no multiprocess mode (default `all`: one series per worker); the new
   gauges need `max` so `/metrics` shows one value.
7. `scipy` (used by the drift scoring and by ContextDefender) is installed only as another
   package's dependency; it isn't in `backend/requirements.txt`.
8. `_build_llm_without_tags` in `llm.py` has no caller.
9. `TOGETHER_API_BASE` is read in `llm.py` but never passed to the Together client, which always
   posts to `api.together.xyz`; honouring it makes the local check possible without real credit
   (OBS-021).

## Implementation notes (3.6.0)

Where the code differs from the designs, or adds to them:

- `drift_alerts` exposes `figures(samples, now)` and `evaluate(figures, detectors, token_limit)`
  instead of `evaluate(samples, detectors, now)`, so the figures are computed once and stored.
- The per-detector reference and scoring logic is in a helper module, `app/core/drift_reference.py`,
  to keep `drift_store.py` near the size limit (OBS-NFR-05).
- The detector classes left `app/core/drift_detection.py` as planned, but `demos/ml_observability_demo.py`
  used them, so they moved to `backend/demos/drift_detectors.py` for the demo only.
- The directory loader is `DocumentLoaderService.batch_load_documents` (the designs said
  `load_directory`; they now use the real name).
- `SelfHostedChatModel` declares its settings as model fields: building it failed before
  ("object has no field base_url"), found while adding `call_label`.
- Docker Compose passes `TOGETHER_API_BASE` and the four drift settings to the backend (the local
  check needs them); an empty `TOGETHER_API_BASE` means Together's address.
- `load_document` also observes `document_processing_latency_seconds`: the local check found the
  dashboard's document-time panel empty because uploads never fed it.
- Local check (task 15, 2026-10-16): a real model call and a real upload recorded samples;
  with `scripts/fake_together.py`, `model_drift`, `model_errors`, `rate_limited` and
  `document_drift` (from real uploads) fired in `/observability/alerts`, `/metrics` and Grafana's
  alert list, and cleared again. It also showed that the speed detector flags shifts of about 10%
  when latencies are steady (KS at p < 0.05 on 30–50 samples), so `model_drift` may fire often on
  real traffic: watch it in the shadow week.

Fixed in the review after the release (3.6.1, 2026-10-16), where the code or the checks fell
short of this spec:

- `SelfHostedChatModel._generate` returned an `AIMessage`, which LangChain rejects, and a reply
  without `usage` failed, so the self-hosted half of OBS-001 couldn't work; both fixed, with a test
  against a fake server (200 and 429).
- OBS-014 covers the embeddings tap, which wasn't timed; a test now times 1,000 embedding calls
  with 1,536-dimension vectors.
- An unreadable `state.json` was logged on every read; now once until a read succeeds (high-level
  design §4).
- `apply_gauges` now sets `data_drift_score` through `record_data_drift` (backend design §9).
- The Grafana alert rules link to `docs/RUNBOOK.md#drift-and-alerts`, but the heading was
  "Drift and alerts (3.6.0)" (a different anchor); renamed, with a test.
- The designs and the plan now match the code: the detector table's module, `base_url` read when
  the client is built, the samples read in `compute_once`, `figures` key names, the `/health` and
  `/metrics` provider (the configured default: samples don't record one), file sizes.

## Traceability

Every requirement names its check. Every task in `tasks.md` names the requirements it satisfies,
and the coverage table there shows each requirement has at least one task.
