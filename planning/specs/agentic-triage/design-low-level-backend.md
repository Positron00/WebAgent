# Low-level design: backend

Paths are relative to `backend/` unless they start with `src/` or `scripts/`. Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Module layout

| File | New? | Purpose | Approx. size |
|---|---|---|---|
| `app/services/triage.py` | new | `classify()`, rules, input building, threshold, fallback | ~170 lines |
| `app/services/triage_model.py` | new | Load encoder + classifier, `predict_proba`, warm-up, fake for tests | ~120 lines |
| `app/services/triage_collect.py` | new | Append collected rows, 30-day cleanup | ~80 lines |
| `app/services/routing.py` | new (phase 5) | `resolve_mode()`, `choose_route()` | ~60 lines |
| `app/api/endpoints/triage.py` | new | `POST /api/triage` | ~50 lines |
| `app/api/endpoints/task.py` | changed | `previous_messages`, shadow, `triage_override`, `mode` | +60 lines |
| `app/services/task_manager.py` | changed | Store `triage`/`route` on results; `run_direct()` (phase 5) | +60 lines |
| `app/agents/supervisor.py` | changed (phase 5 only; phase 0 needs no change) | Extract `answer_directly()` from `run()` | ±30 lines |
| `app/api/endpoints/frontend.py` | changed (phase 5) | `mode` on chat completions | +20 lines |
| `app/core/config_def.py`, `app/core/config.py` | changed | `TriageConfig`, settings, env overrides | +40 lines |
| `app/core/metrics.py` | changed | Four triage metrics | +30 lines |
| `app/core/startup_checks.py` | changed | `check_triage_files()` | +20 lines |
| `main.py` | changed | Mount `/api/triage`; warm up on startup | +10 lines |
| `scripts/train_triage.py`, `scripts/eval_triage.py`, `scripts/triage_data.py` | new | Training, release gate, shared row loading | ~150, ~150, ~60 lines |
| `data/triage/categories.json`, `train.jsonl`, `test.jsonl`, `README.md` | new | Curated data and how it was made | data |
| `models/triage/classifier.joblib`, `metadata.json` | new | Trained artifacts | < 100 KB |
| `config/{dev,uat,prod}.yaml` | changed | `triage:` section | small |
| `Dockerfile` | changed | Bake encoder; `WEB_CONCURRENCY=2` | small |
| `grafana/dashboards/webagent-backend-dashboard.json` | changed | Triage panel | small |

`triage.py` is split from `triage_model.py` so the decision logic can be tested with a fake model and never imports torch.

## 2. Configuration

`app/core/config_def.py`:

```python
class TriageConfig(BaseModel):
    enabled: bool = True
    mode: str = "act"                     # "shadow" | "act"; config.py turns any other YAML value into "act" with a warning
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # None: use metadata.json
    uncertain_band: float = Field(default=0.15, ge=0.0, le=0.5)       # |p - threshold| < band: "uncertain" metric
    llm_tiebreak: bool = False
    tiebreak_upper: float = Field(default=0.65, ge=0.0, le=1.0)
    collect: bool = False
    collect_retention_days: int = Field(default=30, gt=0)
    model_dir: str = "models/triage"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    short_query_words: int = Field(default=6, gt=0)
```

Add `triage: TriageConfig = Field(default_factory=TriageConfig)` to `AppConfig`.

The fields after `collect` are tuning values set only in YAML; they have no environment variable.

YAML (`config/dev.yaml` shown; uat/prod set `collect: false`, and `enabled: false` until a classifier version passes the release gate; phase 2 sets `mode: shadow` in dev, where collection is on):

```yaml
triage:
  enabled: true
  mode: act
  collect: true
  llm_tiebreak: false
```

Environment overrides, applied in `app/core/config.py` the same way as the `security:` section:

| Variable | Field | Parse |
|---|---|---|
| `WEBAGENT_TRIAGE_ENABLED` | `enabled` | bool (`true`/`false`, case-insensitive) |
| `WEBAGENT_TRIAGE_MODE` | `mode` | `shadow` \| `act`; other values: warning, keep YAML |
| `WEBAGENT_TRIAGE_THRESHOLD` | `threshold` | float in [0, 1]; else warning, ignore |
| `WEBAGENT_TRIAGE_LLM_TIEBREAK` | `llm_tiebreak` | bool |
| `WEBAGENT_TRIAGE_COLLECT` | `collect` | bool |

Expose as `settings.TRIAGE` (the `TriageConfig` object).

`mode` only controls shadow recording in `/api/task` (section 6.2). It never changes a route: `/api/triage` always answers, and `mode: auto` requests (phase 5) always act.

## 3. `app/services/triage.py`

### 3.1 Public types

```python
Route = Literal["direct", "agentic"]
DecidedBy = Literal["rule", "model", "fallback", "config", "mode", "override"]

@dataclass(frozen=True)
class TriageDecision:
    route: Route
    p_agentic: Optional[float]      # None for rule/config/fallback
    decided_by: DecidedBy
    reason: str                     # never contains the question text
    classifier_version: str         # "none" when no model was used
    latency_ms: float

    def to_dict(self) -> dict: ...  # JSON-safe, rounds p to 3 decimals
```

### 3.2 Public function

```python
def classify(
    query: str,
    previous_user_question: Optional[str] = None,
    has_image: bool = False,
) -> TriageDecision:
    """Decide the route for one question. Never raises."""
```

Steps (REQ-013 to REQ-018):

1. Start a timer.
2. If `not settings.TRIAGE.enabled`: return `agentic`, `config`, reason `"Triage disabled"`.
3. `rule = match_rule(query, has_image)`; if found, return it with `decided_by="rule"`.
4. `text = build_model_input(query, previous_user_question)`.
5. `model = triage_model.get_model()`; if `None` (not loaded): return `agentic`, `fallback`, reason `"Classifier unavailable"`.
6. `p = model.predict_proba(text)`; on any exception: log it, return `agentic`, `fallback`, reason `"Classifier error"`.
7. `threshold = settings.TRIAGE.threshold if settings.TRIAGE.threshold is not None else model.metadata.threshold` (a plain `or` would ignore an override of 0.0).
8. `route = "agentic" if p >= threshold else "direct"`.
9. If `llm_tiebreak` and `threshold <= p < tiebreak_upper`: `route = _llm_tiebreak(query, route)`. It asks the text LLM (`get_llm().invoke`, run on a 2-thread pool so it can be timed out) to reply `RESEARCH` or `DIRECT`, waits at most 3 s, and keeps `route` on an error, a timeout or any other reply.
10. `reason = model.nearest_category(text)` formatted as `"Looks like: <category label>"`.
11. Record metrics (passing the threshold used, for the uncertain count) and the log line (section 7). Return the decision.

The whole body sits in one `try/except Exception` that returns the fallback decision, so `classify` can't raise.

### 3.3 Rules

```python
GREETINGS = {"hi", "hello", "hey", "thanks", "thank you", "thx", "ok", "okay", "bye", "good morning", ...}
RESEARCH_PREFIX = re.compile(
    r"^\s*(please\s+)?(research|find sources|find studies|write a report|do a deep dive|"
    r"give me a literature review)\b",
    re.IGNORECASE,
)

def match_rule(query: str, has_image: bool) -> Optional[tuple[Route, str]]:
    if has_image:
        return "direct", "Has an image"
    normalized = re.sub(r"[^\w\s']", "", query).strip().lower()
    if not normalized or normalized in GREETINGS:
        return "direct", "Greeting or short acknowledgement"
    if RESEARCH_PREFIX.match(query):
        return "agentic", "Explicit research request"
    return None
```

Near-misses that must **not** match a rule, and go to the model instead (tested): "hi, compare the top CRM tools…", "thanks to whom was penicillin discovered?". Greetings match only the whole message, never a prefix.

One known false match, kept on purpose and documented in the test: "research shows coffee is healthy, true?" starts with "research" used as a noun, so it routes agentic. That costs a slow answer, not a wrong one. "investigate" is left out of the list because it often starts coding questions ("investigate why this test fails").

The image rule's reason is just "Has an image": in the web app the direct route sends the image to the model, but on the backend direct path (phase 5) it is ignored, so the reason mustn't promise more.

### 3.4 Model input

```python
def build_model_input(query: str, previous_user_question: Optional[str]) -> str:
    q = query.strip()[:2000]
    if previous_user_question and len(q.split()) < settings.TRIAGE.short_query_words:
        return f"{previous_user_question.strip()[:1000]} → {q}"
    return q
```

The training script uses this same function (imported), so training and serving inputs match. Curated follow-up rows store `text` already in the joined form, and the script passes them through unchanged.

## 4. `app/services/triage_model.py`

```python
@dataclass(frozen=True)
class TriageMetadata:
    version: str            # e.g. "triage-2026-10-a"
    encoder_name: str
    threshold: float
    data_hash: str          # sha256 of train.jsonl
    categories: dict[str, str]  # category id -> human label

class TriageModel(Protocol):
    metadata: TriageMetadata
    def predict_proba(self, text: str) -> float: ...
    def nearest_category(self, text: str) -> str: ...

class MiniLMTriageModel:
    """Frozen sentence-transformers encoder + sklearn LogisticRegression + category centroids."""

def load_model(model_dir: Path) -> Optional[TriageModel]: ...
def get_model() -> Optional[TriageModel]: ...   # cached per process
def set_model_for_tests(model: Optional[TriageModel]) -> None: ...
def warm_up() -> None: ...                     # load + embed one sentence; logs duration, or a warning if files are missing
def model_files_present(model_dir: Path) -> bool: ...
def encoder_files_present(encoder_name: Optional[str] = None) -> bool: ...  # base encoder in a cache; no torch
def encoder_available(model_dir: Optional[Path] = None) -> bool: ...        # what this classifier needs; no torch
```

`TriageMetadata` also has `encoder_path: Optional[str]`: the fine-tuned encoder folder relative to the model folder (`"encoder"`), or `None` for a frozen classifier that uses the base encoder by name.

`encoder_available` reads `metadata.json`: with `encoder_path`, it checks that folder has `config.json` and `model.safetensors`; otherwise it calls `encoder_files_present`, which looks for `models--sentence-transformers--all-MiniLM-L6-v2` under `SENTENCE_TRANSFORMERS_HOME`, `HF_HUB_CACHE`, and `HF_HOME/hub` (default `~/.cache/huggingface/hub`). The startup check uses `encoder_available`.

Details:

- `classifier.joblib` holds a dict: `{"classifier": LogisticRegression, "centroids": np.ndarray (n_categories, 384), "category_ids": list[str]}`. The classifier is fitted on the string labels `"agentic"` / `"direct"`, so `predict_proba` finds the agentic column by name. Centroids are the mean normalized embedding per category; `nearest_category` is the max cosine similarity.
- With `encoder_path`, the encoder loads from that local folder (`SentenceTransformer(str(model_dir / encoder_path), device="cpu")`); a missing folder logs an error and falls back. Without it, it loads by name with `local_files_only=<env != dev>`. `torch.set_num_threads(1)` is called once before loading.
- Embeddings use `normalize_embeddings=True`.
- Loading is lazy and guarded by a `threading.Lock`, so two requests at startup don't load twice. A failed load (`load_model` returns `None` or raises, e.g. a corrupt `classifier.joblib`) is remembered for 60 s before retrying, so a broken model doesn't add load time to every request. `get_model` and `warm_up` never raise.
- `metadata.encoder_name` must equal the configured encoder; a mismatch logs an error and `load_model` returns `None` (fallback).
- `classify` runs in FastAPI's threadpool (the endpoint is a plain `def`), so the CPU work (~10 ms measured) doesn't block the event loop.
- `FakeTriageModel(p: float | Callable[[str], float], category="test")` lives in `tests/fakes/triage.py` for unit tests.

## 5. `POST /api/triage` (`app/api/endpoints/triage.py`)

```python
class TriageRequest(BaseModel):
    query: str = Field(default="", max_length=8000)   # empty allowed: the empty-message rule answers it
    previous_user_question: Optional[str] = Field(default=None, max_length=8000)
    has_image: bool = False

class TriageResponse(BaseModel):
    route: Literal["direct", "agentic"]
    p_agentic: Optional[float]
    decided_by: Literal["rule", "model", "fallback", "config"]
    reason: str
    classifier_version: str

@router.post("", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    return TriageResponse(**classify(req.query, req.previous_user_question, req.has_image).to_dict())
```

Mounted in `main.py` like the task router: `app.include_router(triage_router, prefix="/api/triage", tags=["triage"], dependencies=[Depends(require_api_key)])`. The existing rate-limit middleware applies to it unchanged.

## 6. Changes to `/api/task`

### 6.1 Request (phases 0, 3, 5)

```python
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)

class TaskRequest(BaseModel):
    query: str
    image: Optional[str] = None
    agentic: Optional[bool] = None          # was bool = True; None means "not given" (phase 5)
    mode: Optional[Literal["off", "auto", "always"]] = None   # phase 5
    document_id: Optional[str] = None
    previous_messages: List[ChatTurn] = Field(default_factory=list, max_length=8)  # phase 0
    triage_override: bool = False           # phase 3
```

In phases 0–4, `agentic` keeps today's `bool = True`. It becomes `Optional[bool] = None` in phase 5, when `resolve_mode` handles `None` (section 6.3). Each field is added in the phase shown in its comment.

`create_task` adds `context["previous_messages"] = [t.model_dump() for t in req.previous_messages]`. `SupervisorAgent._with_history` already reads it; no prompt change.

### 6.2 Shadow mode and override (phases 2–3)

In `create_task`, before scheduling the workflow. `sent_to_workflow` is `req.agentic` in phases 2–4, and "the resolved mode is `always`" from phase 5:

```python
prev_q = last_user_question(req.previous_messages)
if req.triage_override:
    override = TriageDecision("agentic", None, "override", "Research in depth", "none", 0.0)
    record_triage_decision(override)          # counts TRIAGE_DECISIONS{override} and TRIAGE_OVERRIDES
    triage_collect.record("override", req.query, None, prev_q)
    context["triage"] = override.to_dict()
else:
    resolved = resolve_mode(req.mode, req.agentic)
    decision = await run_in_threadpool(choose_route, resolved, req.query, prev_q, req.image is not None)
    context["triage"] = decision.to_dict()    # the decision that picks the path
    if settings.TRIAGE.mode == "shadow" and resolved == "always":
        try:
            shadow = await run_in_threadpool(classify, req.query, prev_q, req.image is not None)
            triage_collect.record("shadow", req.query, shadow, prev_q)
            context["shadow_triage"] = {**shadow.to_dict(), "shadow": True}
        except Exception:
            logger.warning(...)               # a triage error never fails the task
```

In phases 2–4, before `mode` exists, `resolved == "always"` is simply `req.agentic`. `last_user_question` returns the content of the last `user` turn in `previous_messages`, or `None`. When the task finishes, `TaskManager` stores the routing fields (section 6.3) on the saved record and inside the polled `result`: `route`, `decided_by` and `p_agentic` come from `context["triage"]`, and `triage` is `context["shadow_triage"]` when present (so in shadow mode `triage.route` is what the classifier would have chosen while `route` is the path that ran), else `context["triage"]`.

### 6.3 Backend modes (phase 5)

`app/services/routing.py`:

```python
Mode = Literal["off", "auto", "always"]

def resolve_mode(mode: Optional[str], legacy_flag: Optional[bool]) -> Mode:
    if mode is not None:
        return mode
    if legacy_flag is not None:
        return "always" if legacy_flag else "off"
    return "off"        # decided: no mode and no flag means off, on every endpoint

def choose_route(mode: Mode, query, previous_user_question, has_image) -> TriageDecision:
    if mode == "off":    return TriageDecision("direct", None, "mode", "Off mode", "none", 0.0)
    if mode == "always": return TriageDecision("agentic", None, "mode", "Always mode", "none", 0.0)
    return classify(query, previous_user_question, has_image)
```

`create_task` then calls `task_manager.run_direct(...)` or `task_manager.run_workflow(...)`. `triage_override` wins over everything and routes agentic.

`TaskManager.run_direct(task_id, query, context)` (same signature as `run_workflow`; `context` holds `previous_messages`, `triage` and `had_image`):

- Calls `SupervisorAgent.answer_directly(query, previous_messages)`: the code moved out of the `agentic_toggle == False` branch of `SupervisorAgent.run`, using the existing text LLM and `_with_history`. `run` keeps calling it, so today's behavior is unchanged.
- **Text only (decided).** The image is never sent to the model. Image support would need a chat-style vision call (e.g. `ChatTogether` with image parts) and a model that accepts images; that is separate work.
- Saves a result with the same shape as the workflow's: `{"title": "Response", "content": ..., "type": "direct_response"}`, under the same `register_pending` / `result_store` flow, with metrics and the task-ID log tag.

Every saved result (both paths) gets top-level `route`, `decided_by`, `p_agentic` (when the model ran), `triage`, and `image_ignored: true` when the request had an image. The same fields are merged into the report that `GET /api/task/{id}` returns as `result`, so pollers see them without a contract change.

`POST /api/v1/chat/completions` (`frontend.py`): add `mode: Optional[...] = None`; change `agentic_toggle: bool = False` to `Optional[bool] = None`; resolve with the same `resolve_mode` and call `choose_route` in the threadpool. No mode and no flag means `off`, which matches its current default, so its callers see no change. This endpoint has no shadow recording.

## 7. Observability

`app/core/metrics.py`:

```python
TRIAGE_DECISIONS = Counter("webagent_triage_decisions_total", "Triage decisions", ["route", "decided_by"])
TRIAGE_LATENCY = Histogram("webagent_triage_latency_seconds", "Triage latency",
                           buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 3.0))
TRIAGE_OVERRIDES = Counter("webagent_triage_overrides_total", "Research-in-depth clicks")
TRIAGE_UNCERTAIN = Counter("webagent_triage_uncertain_total", "Model decisions near the threshold")
```

`record_triage_decision` (in `metrics.py`) does the counting. `classify` increments `TRIAGE_DECISIONS` and observes `TRIAGE_LATENCY` on every call, and `TRIAGE_UNCERTAIN` when `abs(p - threshold) < uncertain_band`. (This "uncertain" range is wider than the tie-break band `[threshold, tiebreak_upper)`, because it also counts near-misses just below the threshold.) `choose_route` counts `mode` decisions in `TRIAGE_DECISIONS`, and the override branch counts `override`; neither observes latency, because no classification ran and their 0 ms would hide `classify`'s real p95 (NFR-01).

Log line (logger `app.services.triage`, INFO):

```
triage route=direct p=0.104 by=model version=triage-2026-10-a latency_ms=48.2 reason="Looks like: coding help"
```

In dev, `query="<first 200 chars>"` is appended. The task ID tag comes from the existing `log_context`.

## 8. Collection (`app/services/triage_collect.py`)

```python
def record(kind: Literal["shadow", "override"], query: str,
           decision: Optional[TriageDecision], previous_user_question: Optional[str]) -> None
def cleanup(now: Optional[datetime] = None) -> int   # returns files deleted
async def cleanup_daily(interval_sec: float = 86400) -> None   # loops: sleep, cleanup
```

- Does nothing unless `settings.TRIAGE.collect`.
- Writes one JSON line to `data/triage/collected/<UTC date>.jsonl`, relative to the backend folder (`/app/data/triage/collected/` in a container, which is on the data volume; copy it out with `docker cp` for review): `{"ts", "kind", "text", "previous_user_question", "route", "p_agentic", "decided_by", "classifier_version"}`. `text` is cut to 2,000 characters and `previous_user_question` to 1,000. Within a worker, appends run under a `threading.Lock`. Across the 2 workers, the file is opened with `O_APPEND` and each line goes out in a single `os.write` call, which keeps lines whole on a local Linux file system. The reader (`load_rows`) skips any line that isn't valid JSON, with a warning, so a damaged line can't break a review.
- Errors are logged at warning and swallowed.
- `cleanup` deletes files whose date in the name is older than `collect_retention_days`; files with other names are left alone. Called from the startup hook and from `cleanup_daily`, an `asyncio` task started there and cancelled on shutdown.
- Tests: an autouse fixture in `tests/conftest.py` points `DATA_DIR` at a temporary folder, sets `mode: act` and `collect: false`, and stops the real classifier from loading (unless a test is marked `real_triage_model`), so the suite never writes questions into `backend/data/triage/collected/`.
- `.gitignore` gets `backend/data/triage/collected/`.

## 9. Startup

`main.py` startup hook (after `validate_production_settings`):

```python
@app.on_event("startup")
async def warm_up_triage() -> None:
    if settings.TRIAGE.enabled:
        await run_in_threadpool(triage_model.warm_up)
    # Overrides are collected even with triage disabled, so retention cleanup always runs.
    triage_collect.cleanup()
    app.state.triage_cleanup_task = asyncio.create_task(triage_collect.cleanup_daily())
```

A matching shutdown hook cancels `triage_cleanup_task`.

`app/core/startup_checks.py`:

```python
def check_triage_files() -> None:
    """In uat/prod with triage enabled, the classifier and encoder must be on disk."""
```

It checks `classifier.joblib`, `metadata.json`, that `metadata.json["test"]["passed"]` is `true` (`gate_passed()`, REQ-023), and `encoder_files_present()` (section 4). It doesn't load torch, so the check stays fast. Called from `__main__` and from `check_production_settings`.

Shell or container `WEBAGENT_*` variables win over `.env.local` and over the values the YAML loader exports to the environment (`app/core/config.py` saves them before loading files and restores them after), so `WEBAGENT_TRIAGE_ENABLED=false` on the command line really disables triage.

## 10. Training and evaluation scripts

Both live in `backend/scripts/` (next to the MLflow helper scripts) and run from `backend/`, so they can import `app.services.triage.build_model_input` and the training input matches serving. The plan lists them under the repo-root `scripts/`; `backend/scripts/` avoids path hacks.

`scripts/train_triage.py`:

```
python scripts/train_triage.py [--data data/triage/train.jsonl] [--out models/triage]
                               [--version triage-YYYY-MM-x] [--recall-target 0.95] [--seed 13]
                               [--finetune [--pairs-per-row 4] [--epochs 1]]
```

1. Load and validate rows (`load_rows` in `scripts/triage_data.py`, shared with the eval script) and the category labels (`load_categories`). `categories.json` is a list of `{"id", "label", "class": "direct" | "agentic" | "mixed", "hard_case": bool}`; `mixed` is for follow-ups, which can have either label.
2. Stratified 80/20 split of `train.jsonl` into fit and validation.
3. Embed `build_model_input(row["text"], None)` for each row (follow-up rows are already in the joined form; this applies the same 2,000-character cut as serving) with `normalize_embeddings=True`.
4. Fit `LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)`.
5. Threshold: the **highest** value whose validation research recall ≥ target, capped at 0.5. Candidates are the validation scores ≤ 0.5 plus 0.5 itself. (Any lower threshold also meets the recall target but sends more routine questions to research.)
6. Refit on all of `train.jsonl`; compute category centroids.
7. Write `classifier.joblib` and `metadata.json`:

```json
{"version": "triage-2026-10-a", "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
 "threshold": 0.35, "data_hash": "sha256:…", "trained_at": "2026-10-14T10:00:00Z",
 "categories": {"definition": "definition of a stable concept", "current_events": "current events"},
 "validation": {"research_recall": 0.97, "direct_rate": 0.86},
 "test": null}
```

`--finetune` (alias `--setfit`, REQ-024) fine-tunes the encoder before step 4, SetFit-style, using sentence-transformers only (no `setfit` package):

- Pairs come from the fit split only: for each row, `--pairs-per-row` (default 4) partners with the same label (score 1.0) and as many with the other label (score 0.0).
- `SentenceTransformerTrainer` with `CosineSimilarityLoss`, 1 epoch (`--epochs`), batch 32, learning rate 2e-5, warm-up 10%. It runs on the Apple GPU (MPS) or CUDA when there is one; about 1 minute for ~16,000 pairs on an Apple laptop.
- The encoder is saved to `<out>/encoder/` (about 88 MB); `metadata.json` gets `"encoder_path": "encoder"` and a `fine_tuning` block. The threshold is still picked on the validation split, which the encoder never saw.
- `eval_triage.py` loads the encoder from `encoder_path` when it is set.

`scripts/eval_triage.py`:

```
python scripts/eval_triage.py [--data data/triage/test.jsonl] [--model models/triage] [--check-data] [--write-metadata]
```

- Prints overall research recall and routine-direct rate, a confusion matrix per category, and the 20 worst errors.
- Exit code 0 only if both targets are met (REQ-023). `--write-metadata` stores the test metrics in `metadata.json["test"]`, with `"passed": true | false`.
- `--check-data` validates both files: schema, labels, categories known in `categories.json`, each label matching its category's class (except `mixed`), no duplicate `text` within a file or across train and test, no near-duplicates (embedding cosine similarity ≥ 0.9), and at least 20 train / 10 test rows per hard-case category (REQ-020, REQ-021). It prints every problem and exits 1 if there are any.

## 11. Docker and Kubernetes

`Dockerfile`, runtime stage:

- In the main `ENV` block: change `WEB_CONCURRENCY=4` to `WEB_CONCURRENCY=2` (decided).
- `models/triage/` (classifier, metadata and the fine-tuned `encoder/`) is copied by the existing `COPY . .`; `.dockerignore` excludes `data/*`, so the curated data stays out of the image. Build after `git lfs pull`, so the encoder weights are real files and not LFS pointers.
- The step after the copy only creates and chowns the writable folders (`/app/logs`, `/app/config`, `/app/data`). A recursive `chown -R`/`chmod -R /app` there stored a second copy of `/app`, encoder included, in its own layer.
- 3.1.0 and 3.1.1 instead baked the base encoder into `/opt/models` (`SENTENCE_TRANSFORMERS_HOME`). That step was removed with `triage-2026-10-c`; a frozen classifier version would need it back.

`HF_HOME` stays on `/app/data/hf-cache` for anything else. `kubernetes/deployment.yaml` already sets `WEB_CONCURRENCY=2`; its memory limit rose from 1 GiB to 1.5 GiB after task 2.9 measured 1.08 GiB with triage on. `docker-compose.yml` sets no worker count, so it picks up the new image default of 2.

## 12. Tests

| File | Covers |
|---|---|
| `tests/test_triage_rules.py` | Rule table and near-misses (REQ-013) |
| `tests/test_triage_config.py` | YAML values, each `WEBAGENT_TRIAGE_*` override, bad values ignored with a warning, a bad YAML `mode` becomes `act` (task 2.1) |
| `tests/test_triage_classify.py` | Threshold, short-query joining, fallback, config off, tie-breaker, reasons, determinism, metrics (mode and override decisions leave the latency histogram alone), a load error remembered for 60 s (REQ-014–018, REQ-080, NFR-07) |
| `tests/test_triage_endpoint.py` | Contract, empty query → rule, 401, 422 on over-long input, rate limit applies (REQ-010) |
| `tests/test_triage_collect.py` | On/off, path, no writes outside `collected/`, cleanup (REQ-030–032) |
| `tests/test_task_previous_messages.py` | Field limits, context, supervisor input (REQ-002) |
| `tests/test_task_shadow.py` | Shadow records and still runs; triage error doesn't fail the task; override counted (REQ-040, REQ-058) |
| `tests/test_task_modes.py` | `resolve_mode` table (no mode + no flag → `off`); both endpoints; direct path skips graph; `image_ignored`; route fields; `/api/triage` and `mode: auto` agree (REQ-012, REQ-070–072) |
| `tests/test_startup_checks.py` (extend) | Missing files fail in prod, pass in dev, pass when disabled (NFR-08) |
| `tests/test_triage_real_model.py` | Loads cached MiniLM + committed classifier; 6 clear cases; skipped if not cached |
| `tests/test_train_triage.py` | Training smoke test with a fake encoder and 20 rows; threshold choice; gate exit codes; `--check-data` problems |
