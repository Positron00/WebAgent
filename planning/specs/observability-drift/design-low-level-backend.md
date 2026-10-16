# Low-level design: backend

Paths are relative to `backend/`. Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Module layout

| File | New? | Purpose | Approx. size |
|---|---|---|---|
| `app/core/drift_store.py` | new | `add`, `add_aggregate`, `reset`, `compute_once`, `load_state`, `compute_loop`, `apply_gauges`, `cleanup` | ~190 lines |
| `app/core/drift_alerts.py` | new | `figures(samples, now)` and `evaluate(figures, detectors, token_limit)`: the six alert rules | ~100 lines |
| `app/core/drift_reference.py` | new (in implementation) | Detector table, `update_detector` (reference freeze, current window, scoring), `public()`; split out so `drift_store.py` stays near 200 lines | ~125 lines |
| `app/core/drift_detection.py` | changed | Pure scoring functions; detector classes, module registry, `get_all_drift_stats`, `should_retrain_any_model` and MLflow calls removed | −700 / +80 |
| `demos/drift_detectors.py` | new (in implementation) | The old detector classes, kept for `demos/ml_observability_demo.py` only | moved |
| `app/services/drift_embeddings.py` | new | `DriftRecordingEmbeddings`: wraps an embeddings object, records aggregates | ~70 lines |
| `app/services/llm_call_metrics.py` | changed | `record(..., call_label=…)` also adds drift samples | +25 lines |
| `app/services/together_llm.py` | changed | `call_label` field; pass it to `record`; base URL from `TOGETHER_API_BASE` | +10 lines |
| `app/services/llm.py` | changed | `call_label` through `get_agent_llm`, `get_trusted_llm`, `_build_llm`; `SelfHostedChatModel.call_label` and status code; remove `record_data_drift("llm_response", 0.8)` and the unused `_build_llm_without_tags` | ±40 lines |
| `app/security/judge.py`, `app/services/triage.py` | changed | `get_trusted_llm(call_label="judge" / "triage")` | one line each |
| `app/services/document_loader.py` | changed | Record document samples in `load_document` | +20 lines |
| `app/services/vectordb.py` | changed | `get_embeddings()` returns the wrapped object | +3 lines |
| `app/services/llm_service.py` | changed | Keep `get_drift_status`, `reset_drift_baselines` over the store; remove simulations and detectors | −200 / +40 |
| `app/services/document_ingestion_service.py` | changed | `DocumentProcessingObservability` keeps status and reset over the store; remove `process_document` and module detectors | −90 / +25 |
| `app/services/self_hosted_llm_service.py` | moved | To `app/services/examples/self_hosted_llm_service.py`; the example's import updated | 0 |
| `app/api/routes/observability.py` | changed | Read the store; alerts and health from it | ±120 lines |
| `app/core/observability.py` | changed | `DATA_DRIFT_SCORE` gets `multiprocess_mode="max"`; new gauges and counter | +30 lines |
| `app/core/config.py` | changed | Four drift settings | +6 lines |
| `main.py` | changed | Sample cleanup at startup; start and stop the compute loop | +6 lines |
| `requirements.txt` | changed | Declare `scipy>=1.11` (pinned to 1.13.1 by `constraints.txt`; already installed) | +1 line |
| `scripts/fake_together.py` (repo root `scripts/`) | new | Dev-only fake Together completions server | ~80 lines |

## 2. Configuration (`app/core/config.py`)

```python
DRIFT_ENABLED: bool = os.getenv("WEBAGENT_DRIFT_ENABLED", "true").lower() == "true"
DRIFT_WINDOW: int = _env_number("WEBAGENT_DRIFT_WINDOW", 50, minimum=30)
DRIFT_INTERVAL_SECONDS: int = _env_number("WEBAGENT_DRIFT_INTERVAL_SECONDS", 60, minimum=5)
TOKEN_ALERT_DAILY: int = _env_number("WEBAGENT_TOKEN_ALERT_DAILY", 0, minimum=0)
```

`DRIFT_WINDOW` is the reference size and the most current samples kept; its minimum is
`MIN_SAMPLES` (30, in `drift_store.py`), the fewest current samples a detector scores with, so a
reference is never smaller than the current minimum.

## 3. Scoring (`app/core/drift_detection.py`)

The existing maths, as pure functions:

```python
DRIFT_P_VALUE = 0.05          # KS and chi-square: drift when p < 0.05
EMBEDDING_MIN_SIMILARITY = 0.3  # centroids: drift when cosine similarity < 0.3

def ks_score(reference: Sequence[float], current: Sequence[float]) -> tuple[float, bool]:
    """(1 - p, p < DRIFT_P_VALUE) from scipy.stats.ks_2samp; NaNs replaced by the median."""

def chi2_score(reference: Sequence[str], current: Sequence[str]) -> tuple[float, bool]:
    """(1 - p, p < DRIFT_P_VALUE) from chi2_contingency over the union of categories."""

def centroid_score(reference: np.ndarray, current: np.ndarray) -> tuple[float, bool]:
    """(1 - cos, cos < EMBEDDING_MIN_SIMILARITY); (0.0, False) if either is all zeros."""
```

The detector classes, `_drift_detectors`, `register_drift_detector`, `get_all_drift_stats`,
`should_retrain_any_model` and the MLflow calls inside the detectors are removed; nothing outside
`llm_service.py`, `document_ingestion_service.py` and `observability.py` uses them (checked with
grep).

## 4. Detectors

A static table in `drift_store.py` maps detector names to their kind and group. Names with a label
are matched by prefix.

| Detector | Kind | Group | Value |
|---|---|---|---|
| `llm_speed:<label>` | `ks` | `llm` | seconds per 1,000 completion tokens |
| `llm_tokens:<label>` | `ks` | `llm` | total tokens of the request |
| `embeddings:<provider>` | `centroid` | `llm` | aggregate (sum, count, dim) |
| `doc_processing_time` | `ks` | `doc` | seconds |
| `doc_size` | `ks` | `doc` | bytes |
| `doc_type` | `chi2` | `doc` | lower-case extension without the dot |

`llm_outcome` (`ok` / `error` / `rate_limited`) and `doc_outcome` (`ok` / `error`) are samples, not
detectors: only the alerts read them (decision 9).

## 5. Drift store (`app/core/drift_store.py`)

### 5.1 Files and records

```text
data/drift/samples-2026-10-15.jsonl
  {"ts": "2026-10-15T10:31:05.123", "d": "llm_speed:supervisor", "v": 4.81}
  {"ts": "...", "d": "llm_outcome", "v": "rate_limited"}
  {"ts": "...", "d": "embeddings:openai", "agg": {"sum": [...], "n": 30, "dim": 1536}}
data/drift/state.json
data/drift/compute.lock
```

Directory: `Path(settings.DATA_DIR) / "drift"` (so tests get `tmp_path` through the existing
`_isolate_triage` fixture, which sets `settings.DATA_DIR`). Number lines are under 512 bytes; an
aggregate line is larger (a 1,536-float sum is about 30 KB), so aggregates are written with one
`os.write` on an `O_APPEND` descriptor while holding a short `fcntl.flock` on a separate
`append.lock` (only aggregates take it; number lines don't need it).

### 5.2 API

```python
MIN_SAMPLES = 30

def add(detector: str, value: float | str) -> None
    """Append one sample; increment webagent_drift_samples_total by 1. Never raises.
    No-op when DRIFT_ENABLED is false."""

def add_aggregate(detector: str, vector_sum: list[float], count: int, dim: int) -> None
    """Append one embedding aggregate under append.lock; increment webagent_drift_samples_total
    by `count`. The caller only calls it with count >= MIN_SAMPLES. Never raises."""

def reset(group: str, now: datetime | None = None) -> None
    """Under a blocking lock on compute.lock: set state["resets"][group] = now and drop the
    frozen references of every detector in the group."""

def compute_once(now: datetime | None = None) -> dict
    """Under the caller's lock: read state and samples, freeze references, score detectors,
    evaluate alerts, log transitions, write state.json (temp file + os.replace). Returns the state."""

def load_state() -> dict
    """Read state.json; on error return the last good state (module cache) or an empty state."""

async def compute_loop(interval_sec: float | None = None) -> None
    """Forever: try a non-blocking flock on compute.lock; if held, run compute_once in a thread
    (it does file I/O and scipy); release; then apply_gauges(load_state()); sleep."""

def apply_gauges(state: dict) -> None
    """Set webagent_alert_active, webagent_drift_detected, webagent_drift_enough_data,
    webagent_drift_current_samples and data_drift_score from the state."""

def cleanup(retention_days: int, now: datetime | None = None) -> int
    """Delete samples-*.jsonl files older than the cutoff (by the date in the name). Called at
    startup and by compute_loop at most once a day."""
```

### 5.3 `state.json`

```json
{
  "computed_at": "2026-10-15T10:32:00",
  "resets": {"llm": "2026-10-15T09:00:00", "doc": null},
  "detectors": {
    "llm_speed:supervisor": {
      "group": "llm", "kind": "ks",
      "reference": [4.8, 5.1, "..."], "reference_start": "2026-10-15T09:03:11",
      "reference_count": 50, "current_count": 50, "enough_data": true,
      "score": 0.41, "drift": false, "last_drift_at": null
    },
    "embeddings:openai": {
      "group": "llm", "kind": "centroid",
      "reference_centroid": [0.01, "..."], "reference_dim": 1536, "reference_count": 60,
      "current_count": 90, "enough_data": true, "score": 0.12, "drift": false, "last_drift_at": null
    }
  },
  "alerts": [
    {"name": "rate_limited", "severity": "warning", "service": "llm",
     "message": "...", "since": "2026-10-15T10:20:00"}
  ],
  "figures": {
    "llm": {"window": "1h", "requests": 42, "errors": 1, "rate_limited": 3,
            "tokens_24h": 51234, "by_label": {"supervisor": {"requests": 6, "tokens_24h": 9100}}},
    "doc": {"window": "24h", "loads": 4, "errors": 0}
  },
  "last_cleanup": "2026-10-15T00:00:12"
}
```

### 5.4 `compute_once` step by step

1. Load state; for each group, `reset_at = state["resets"].get(group)` (absent = beginning of time,
   which is "reset at the first sample" for a fresh install).
2. Read sample files from `min(reset_at of detectors without a frozen reference, now − retention)`
   to now, in date order; skip bad lines.
3. For each detector seen in the samples or in the state:
   - **Reference not frozen:** its samples after `reset_at`; when there are
     `DRIFT_WINDOW` of them, freeze the first `DRIFT_WINDOW` (`ks`/`chi2`: the values; `centroid`:
     the summed centroid of the first aggregates reaching `DRIFT_WINDOW` vectors) and record
     `reference_start`.
   - **Current:** the latest `DRIFT_WINDOW` samples (or aggregates reaching that many vectors),
     excluding those in the reference.
   - **Embeddings:** if the current dimension differs from `reference_dim`, drop the reference
     (a provider change starts a new one).
   - With a frozen reference and at least `MIN_SAMPLES` current samples (vectors for
     embeddings), score with the kind's function; otherwise `enough_data` false, no score, no
     drift.
   - Set `last_drift_at` when `drift` turns true.
4. `figures = drift_alerts.figures(samples, now)` (step 5);
   `alerts = drift_alerts.evaluate(figures, detectors, settings.TOKEN_ALERT_DAILY)`; keep `since` for alerts already
   active; log WARNING once for each new alert and each new drift; one MLflow metric per new drift
   (`log_to_mlflow`, experiment `drift_detection`).
5. Compute `figures` (model requests, errors and rate limits over the last hour; tokens over the
   last 24 h, in total and per label; document loads and errors over the last 24 h).
6. If `last_cleanup` is more than 24 h ago, run `cleanup` and record the time.
7. Write the state; return it.

## 6. Alerts (`app/core/drift_alerts.py`)

```python
def figures(samples: list[Sample], now: datetime) -> dict      # {"llm": {...}, "doc": {...}}
def evaluate(figs: dict, detectors: dict, token_limit: int) -> list[dict]
# each alert: {"name", "severity", "service", "message"}; compute_once adds "since"
```

Rules exactly as OBS-012. Shares are computed over `ts > now - 1 h` (model) or `24 h` (documents and
tokens).

| Alert | Severity | Service | Message |
|---|---|---|---|
| `model_drift` | warning | `llm` | "Model behaviour changed ({detectors}). Check for a provider, model or prompt change, then reset the LLM baseline once it's understood." |
| `model_errors` | critical | `llm` | "{pct} of model requests failed in the last hour ({n} requests, rate limits excluded)." |
| `rate_limited` | warning | `llm` | "{pct} of model requests were rate-limited in the last hour. Together is busy or the account is near its limits." |
| `token_usage` | warning | `llm` | "{tokens} tokens used in the last 24 hours (alert set at {limit})." |
| `document_drift` | warning | `document_processing` | "Documents changed ({detectors}). Check what is being uploaded, then reset the document baseline once it's understood." |
| `document_errors` | critical | `document_processing` | "{pct} of document loads failed in the last 24 hours ({n} loads)." |

## 7. Feeding samples

### 7.1 Call labels (`llm.py`, `together_llm.py`)

```python
class TogetherWithUsage(Together):
    call_label: str = "other"
    base_url: str = f"{TOGETHER_API_BASE.rstrip('/')}/completions"   # OBS-021

class SelfHostedChatModel(BaseChatModel):
    call_label: str = "other"   # declared field, set by the builder
```

- `_build_llm(model_name=None, call_label="other")` passes the label to the Together wrapper's
  `_create_llm` (and to `SelfHostedChatModel`).
- `_build_llm_for_agent(agent_name, …)` calls `_build_llm(model, call_label=agent_name)`.
- `get_trusted_llm(model_name=None, call_label="other")`; `judge.py` passes `"judge"`,
  `triage.py` passes `"triage"`.
- `TOGETHER_API_BASE` moves from `llm.py` to `together_llm.py` (imported back by `llm.py`), so the
  client and the module constant agree.

### 7.2 `llm_call_metrics.record`

```python
def record(provider, model, seconds, ok, usage=None, *, call_label="other", status_code=None) -> None:
    # 3.5.3 Prometheus updates, unchanged …
    outcome = "ok" if ok else ("rate_limited" if status_code == 429 else "error")
    drift_store.add("llm_outcome", outcome)
    if ok and usage:
        completion = usage.get("completion_tokens") or 0
        total = usage.get("total_tokens") or 0
        if completion > 0:
            drift_store.add(f"llm_speed:{call_label}", seconds * 1000.0 / completion)
        if total > 0:
            drift_store.add(f"llm_tokens:{call_label}", total)
```

`together_llm._check_status` raises exceptions that carry the status, keeping the 3.5.3 types and
messages the retry logic and tests rely on: `TogetherServerError(Exception)` with `.status` for 429
and 5xx ("Together Server: Error 429 (rate limited): …", "Together Server: Error 503"), and
`TogetherRequestError(ValueError)` with `.status` for other 4xx ("Together received an invalid
payload: …"). The client passes `getattr(error, "status", None)` to `record` as `status_code`.
`SelfHostedChatModel` passes `error.response.status_code` for a `requests.HTTPError` and `None`
for other failures, with its `call_label`.

### 7.3 Document loads (`document_loader.py`)

`load_document` wraps its body:

```python
started = time.perf_counter(); ok = False
try:
    ...  # existing body
    ok = True
    return artifact_id
finally:
    _record_document_samples(file_path, time.perf_counter() - started, ok)
```

`_record_document_samples` adds `doc_processing_time`, `doc_size` (`file_path.stat().st_size`, if
the file still exists), `doc_type` (`file_path.suffix.lower().lstrip(".") or "none"`) and
`doc_outcome`. The `FileNotFoundError` and unsupported-type checks at the top stay before the
`try`, so a refused file records nothing (OBS-002).

### 7.4 Embeddings (`drift_embeddings.py`)

```python
class DriftRecordingEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, provider: str): ...
    def embed_documents(self, texts): vectors = self._inner.embed_documents(texts); self._observe(vectors); return vectors
    def embed_query(self, text): vector = self._inner.embed_query(text); self._observe([vector]); return vector
    async def aembed_documents(...), aembed_query(...): same, awaiting the inner calls
    def _observe(self, vectors): under a threading.Lock, add to self._sum (np.float64) and self._n;
        when self._n >= MIN_SAMPLES: drift_store.add_aggregate(f"embeddings:{provider}", self._sum.tolist(), self._n, dim); reset
```

`get_embeddings()` returns `DriftRecordingEmbeddings(inner, provider)` when `DRIFT_ENABLED`. The
wrapper is a LangChain `Embeddings`, so Chroma and the internal-research retrievers use it
unchanged. OBS-014 times `_observe`; if p95 exceeds 5 ms, `get_embeddings()` stops wrapping and
OBS-003 is dropped (decision 2).

## 8. API (`app/api/routes/observability.py`)

All routes keep the API key dependency (they are on `api_router`). With `DRIFT_ENABLED` false,
every route answers as if the state were empty (no detectors, no alerts, `healthy`), ignoring any
older `state.json` (OBS-013).

- `GET /drift` → `{"computed_at", "detectors": {name: public fields},
  "global_drift_stats": {name: public fields}` (same content, the old key),
  `"llm_service_drift": llm_service.get_drift_status()`,
  `"doc_service_drift": doc_ingestion_service.get_drift_status()`,
  `"retrain_recommendations": []}`. Public fields leave out the stored reference values and
  centroids.
- `LLMService.get_drift_status()` → `{"collecting_baseline": any llm detector without a frozen
  reference, "baseline_count": max reference_count, "latency_drift", "token_usage_drift",
  "embedding_drift"`: the worst (highest-score) detector of each kind or `null`,
  `"performance_drift": null` (decision 9), `"retrain_recommended": false}`. Same keys as 3.5.x.
- `DocumentProcessingObservability.get_drift_status()` → the same pattern for `doc_*`, with
  `"pipeline_adjustments_recommended": false`.
- `POST /llm/reset_baseline`, `POST /doc/reset_baseline` → `drift_store.reset("llm" | "doc")`;
  responses unchanged.
- `GET /alerts` → `[{"name", "severity", "service", "message", "since", "timestamp": since}]`
  from `load_state()["alerts"]`.
- `GET /health` → `overall_status` from the alerts; `alert_counts` by severity; `services.llm`
  and `services.document_processing` with `status` and figures from the last hour or day: requests,
  failures, rate limits, tokens per call label (computed in `compute_once` and stored under
  `state["figures"]`).
- `GET /metrics` → `model_metrics` (per provider and call label from `state["figures"]`),
  `document_metrics`, `system_metrics`, `timestamp`; `pipeline_metrics` and
  `data_quality_metrics` stay as empty objects.

## 9. Prometheus metrics (`app/core/observability.py`)

```python
DATA_DRIFT_SCORE = Gauge("data_drift_score", "...", ["component"], registry=WEBAGENT_REGISTRY,
                         multiprocess_mode="max")
ALERT_ACTIVE = Gauge("webagent_alert_active", "1 while an alert is active",
                     ["alert", "severity"], registry=WEBAGENT_REGISTRY, multiprocess_mode="max")
DRIFT_DETECTED = Gauge("webagent_drift_detected", "...", ["detector"], ..., multiprocess_mode="max")
DRIFT_ENOUGH_DATA = Gauge("webagent_drift_enough_data", "...", ["detector"], ..., multiprocess_mode="max")
DRIFT_CURRENT_SAMPLES = Gauge("webagent_drift_current_samples", "Current samples (vectors for embeddings)",
                              ["detector"], ..., multiprocess_mode="max")
DRIFT_SAMPLES = Counter("webagent_drift_samples_total", "...", ["detector"], registry=WEBAGENT_REGISTRY)
```

`apply_gauges` sets every alert in OBS-012 to 0 or 1 (so the series exist from the first tick, and
the dashboard shows 0 instead of "no data"). For a detector without enough data it sets
`data_drift_score` to 0 and `webagent_drift_enough_data` to 0: removing a labelled series isn't
reliable in multiprocess mode, and the detector table shows why the score is 0.
`record_data_drift` stays for the store's own use only.

## 10. Startup and shutdown (`main.py`)

In `_startup`, after the existing cleanup tasks:

```python
drift_store.cleanup(settings.RESULTS_RETENTION_DAYS)
if settings.DRIFT_ENABLED:
    app.state.drift_task = asyncio.create_task(drift_store.compute_loop())
```

`_shutdown` cancels `drift_task` with the other tasks. `compute_loop` itself runs
`cleanup(settings.RESULTS_RETENTION_DAYS)` in the worker holding the lock, at most once every 24
hours (the time of the last cleanup is kept in `state.json`), so the store's retention doesn't
depend on the results module.

## 11. Removals

| Removed | Why |
|---|---|
| `LLMService.generate_response`, `get_embedding`; the four module-level detectors and the `_lock` they used | Simulations with no caller |
| `DocumentProcessingObservability.process_document`; its module-level detectors | Simulation with no caller |
| Detector classes, `_drift_detectors`, `register_drift_detector`, `get_all_drift_stats`, `should_retrain_any_model` | Replaced by the store and pure scoring |
| `record_data_drift("llm_response", 0.8)` in `SelfHostedChatModel` | An invented score with a different meaning from every other detector |
| `self_hosted_llm_service.py` under `app/services/` | Moved to `app/services/examples/` (decision 4) |
| `_build_llm_without_tags` in `llm.py` | No caller (grep); it would otherwise need a call label |

## 12. Fake Together server (`scripts/fake_together.py`, dev only)

A standard-library HTTP server for the local check (OBS-020), never imported by the app:

```text
python scripts/fake_together.py --port 18200 --delay 0.5 --completion-tokens 80 \
    --error-rate 0.0 --rate-limit-rate 0.0
```

It answers `POST /v1/completions` with a Together-shaped body (`choices[0].text`, `usage`), sleeping
`--delay` seconds, and returns 500 or 429 at the given rates. The backend container reaches it with
`TOGETHER_API_BASE=http://host.docker.internal:18200/v1`.

## 13. Tests

| File | Covers |
|---|---|
| `tests/test_drift_scoring.py` (new) | `ks_score`, `chi2_score`, `centroid_score`: same vs. shifted distributions, seeded (OBS-007) |
| `tests/test_drift_store.py` (new) | add/compute; 29 vs 30 samples (OBS-006); reference freeze surviving retention (OBS-008); reset (OBS-009); two instances and a restart (OBS-005); two loops, one write per tick, one log per transition (OBS-010); retention (OBS-011); bad lines skipped |
| `tests/test_drift_alerts.py` (new) | Each alert fires, stays, clears; no "retrain" in any message (OBS-012) |
| `tests/test_drift_feeding.py` (new) | Model samples per label and outcome with fake Together replies 200/429/500 (OBS-001); document samples via upload and `batch_load_documents` (OBS-002); embeddings aggregates (OBS-003); timing (OBS-014); base URL (OBS-021) |
| `tests/test_drift_privacy.py` (new) | Marker text never in `data/drift/`; every aggregate has `n ≥ 30` (OBS-004) |
| `tests/test_observability_api.py` (extended) | Contract per route (OBS-013, OBS-NFR-03) |
| `tests/test_metrics_multiprocess.py` (extended) | New gauges, one series per label with two workers (OBS-015) |
| `tests/test_grafana_dashboards.py` (extended) | Alerts row, gauges and thresholds (OBS-016); alert-rule file (OBS-017) |
| `tests/test_drift_cleanup.py` (new) | No simulated code, removed registry and builder; `self_hosted_llm_service.py` moved (OBS-019) |
| `tests/test_deploy_script.py` (new; see the ops design) | Data source, ConfigMaps and Grafana restart with a fake `kubectl` (OBS-018) |
