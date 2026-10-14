# Low-level design: backend

Paths are relative to `backend/`. Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Module layout

| File | New? | Purpose | Release | Approx. size |
|---|---|---|---|---|
| `app/services/run_supervisor.py` | new | Start a run as an `asyncio.Task`; watch deadline and marker; write the final record once | 3.2.7 (deadline), 3.3.0 (cancel) | ~120 lines |
| `app/services/task_budget.py` | new | Per-run token counter in a context variable; `TaskBudgetExceeded` | 3.3.0 | ~60 lines |
| `app/services/progress.py` | new | Node → step label; build and save `progress` | 3.3.0 | ~60 lines |
| `app/api/intake.py` | new | `GuardRefused`, `refusal_body`, `run_intake_guard` moved out of `task.py`, shared with `frontend.py` | 3.2.7 | ~90 lines |
| `app/services/task_manager.py` | changed | Use the supervisor; `astream`; budget and error codes; list from files | 3.2.7–3.3.0 | −40 / +60 |
| `app/services/result_store.py` | changed | `save_final` (3.2.7, for the run limit); marker files, `save_progress`, `list_results`, `delete_result`, `cleanup` (3.3.0) | 3.2.7, 3.3.0 | +80 lines |
| `app/api/endpoints/task.py` | changed | `query` limit; cancel route; `progress` in status | 3.2.7–3.3.0 | +30 lines |
| `app/api/endpoints/frontend.py` | changed | Intake guard; message limits | 3.2.7 | +15 lines |
| `app/api/endpoints/documents.py` | changed | UUID IDs; bounds; extension check before write | 3.2.7 | +20 lines |
| `app/api/endpoints/tasks.py` | changed | Paginated list from files | 3.3.0 | +15 lines |
| `app/core/security.py` | changed | `client_identifier`; remove dev keys and JWT | 3.2.7, 3.3.0 | −70 / +20 |
| `app/core/startup_checks.py` | changed | `check_dev_exposure`; run-limit vs stale-timeout warning | 3.2.7 | +30 lines |
| `app/core/config.py` | changed | Pass every guard field; new settings; remove Redis and burst | 3.2.7, 3.3.0 | ±30 lines |
| `app/services/llm.py` | changed | `apply_agent_settings` (temperature only when set) | 3.2.7 | +20 lines |
| `app/services/together_llm.py` | changed | aiohttp timeout; budget check and count | 3.2.7, 3.3.0 | +15 lines |
| `app/services/web_search.py` | changed | `wait_for` timeout; `SearchError` result | 3.2.7, 3.3.0 | +25 lines |
| `app/graph/workflows.py` | changed | `gather(return_exceptions=True)`; remove unreachable nodes | 3.2.7, 3.4.0 | ±25 lines |
| `app/agents/senior_research.py` | changed | Loop limit from config; `unknown` on a failed parse; prompt limit | 3.2.7 | ±25 lines |
| `app/agents/web_research.py`, `internal_research.py` | changed | Search and document caps; `search_errors`; no question text in logs | 3.2.7, 3.3.0 | small |
| other agents | changed | No question text in logs | 3.3.0 | one line each |
| `main.py` | changed | `lifespan`; one CORS setup; docs off in uat/prod; results cleanup loop | 3.2.7, 3.3.0 | ±40 lines |
| `app/core/config_def.py` | changed | Remove unused agent keys (3.2.7) and `RedisConfig` (3.3.0) | 3.2.7, 3.3.0 | small |
| `app/services/document_loader.py` | changed | `supported_extensions()` for the upload check | 3.2.7 | +5 lines |
| `app/core/log_context.py` | changed | `describe_text` | 3.3.0 | +5 lines |
| `app/api/router.py` | changed | `/health/detailed` behind the API key | 3.3.0 | small |
| `app/api/http_errors.py` | changed | `error_body` | 3.4.0 | +15 lines |
| `app/services/vectordb.py` | changed | Search in a thread; embedding provider | 3.4.0 | +20 lines |
| `app/agents/document_extraction_agent.py` | changed | LDA size limit and pass cap; remove `_route_to_team_manager` | 3.4.0 | ±20 lines |
| `config/{dev,uat,prod}.yaml` | changed | Dev host; remove unused keys; uat/prod `allow_headers` gain `X-API-Key` (3.2.7); remove `database.redis` (3.3.0) | 3.2.7, 3.3.0 | small |

## 2. Configuration

New settings are read once in `app/core/config.py` and exposed on `settings`.

| Variable | Setting | Type, default | Used by |
|---|---|---|---|
| `WEBAGENT_ALLOW_OPEN_DEV` | `ALLOW_OPEN_DEV` | bool, `false` | `check_dev_exposure` |
| `WEBAGENT_TASK_MAX_SECONDS` | `TASK_MAX_SECONDS` | int > 0, 420 | run supervisor |
| `WEBAGENT_SEARCH_TIMEOUT_SECONDS` | `SEARCH_TIMEOUT_SECONDS` | float > 0, 30 | `web_search.py` |
| `WEBAGENT_TASK_TOKEN_BUDGET` | `TASK_TOKEN_BUDGET` | int ≥ 0; 0 (no budget) until task 2.10 sets the default from measurement | `task_budget.py` |
| `WEBAGENT_RESULTS_RETENTION_DAYS` | `RESULTS_RETENTION_DAYS` | int > 0, 30 | `result_store.cleanup` |
| `WEBAGENT_LDA_MAX_CHARS` | `LDA_MAX_CHARS` | int > 0, 200000 | document extraction (3.4.0) |
| `WEBAGENT_EMBEDDINGS_PROVIDER` | `EMBEDDINGS_PROVIDER` | `openai` \| `local`, `openai` | `vectordb.py` (3.4.0) |

A bad value logs a warning and keeps the default, like the `WEBAGENT_TRIAGE_*` parsing.

Removed: `RATE_LIMIT_BURST_LIMIT` in 3.2.7 (task 1.16, unless that task implements it
instead), and `REDIS_*` with the YAML `database.redis` section in 3.3.0 (task 2.22).

## 3. Client ID for rate limits (`app/core/security.py`, IMP-012)

```python
CLIENT_HEADER = "x-webagent-client"

def _holds_api_key(request) -> bool:
    """The caller sent a valid X-API-Key (only servers, such as the web app, hold one)."""
    given = request.headers.get("x-api-key")
    return bool(
        settings.API_KEY_ENABLED and given
        and any(secrets.compare_digest(given, k) for k in _configured_api_keys())
    )

def client_identifier(request) -> str:
    """Who to rate-limit: the web app's session, else the nearest untrusted address."""
    peer = request.client.host if request.client else "unknown"
    via_proxy = peer in TRUSTED_PROXIES
    if via_proxy or _holds_api_key(request):
        client = request.headers.get(CLIENT_HEADER, "").strip()
        if client.startswith("s:") and 3 <= len(client) <= 66:
            return client
    if not via_proxy:
        return peer
    hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    for hop in reversed(hops):          # right-most first: the hop our proxy added
        if hop not in TRUSTED_PROXIES:
            return hop
    return peer
```

The rate-limit middleware runs before the `require_api_key` dependency, so it checks the
key itself with the same constant-time comparison. A wrong key still gets 401 from the
dependency afterwards; it just doesn't earn the session header.

Table test (`test_client_identifier.py`): untrusted peer, no key, any headers → peer;
untrusted peer with a valid key and `s:abc` → `s:abc` (the Vercel case); untrusted peer
with a wrong key and `s:abc` → peer; trusted peer with `s:abc` → `s:abc`; trusted peer with
`X-Forwarded-For: 1.1.1.1, 2.2.2.2` → `2.2.2.2`; trusted peer, header `s:` with 200
characters → falls through to `X-Forwarded-For`; nothing → peer.

## 4. Dev exposure (`app/core/startup_checks.py`, IMP-014)

```python
LOOPBACK = {"127.0.0.1", "::1", "localhost"}

def check_dev_exposure(host: str) -> None:
    """In dev with API keys off, refuse a host other machines can reach."""
    if settings.WEBAGENT_ENV != "dev" or settings.API_KEY_ENABLED or settings.ALLOW_OPEN_DEV:
        return
    if host not in LOOPBACK:
        raise SystemExit(
            f"Dev backend would listen on {host} with API keys off. "
            "Set api.host to 127.0.0.1, turn on API keys, or set WEBAGENT_ALLOW_OPEN_DEV=true."
        )
```

Called from the `__main__` block of `main.py` before `uvicorn.run`. The container path
(Dockerfile `CMD`, always `--host 0.0.0.0`) runs prod by default, and Compose ports bind to
`127.0.0.1` (ops design), so the container doesn't call it.

`config/dev.yaml`: `api.host: "127.0.0.1"`.

## 5. Shared intake guard (`app/api/intake.py`, IMP-020)

Move `GuardRefused`, `refusal_body` and the body of `_intake_guard` out of `task.py`,
with a signature that doesn't depend on `TaskRequest`:

```python
def run_intake_guard(
    query: str,
    history: List[Dict[str, str]],
    has_image: bool,
    consent_token: Optional[str],
    task_id: str,
) -> bool:
    """Checkpoint 1 on the user turn and history. Returns whether the user consented.
    Raises GuardRefused (409 consent, 403 decline/escalate) in act mode."""
```

`task.py` calls it as today. `frontend.py` calls it with the last user message, the
earlier messages and `consent_token=None`, **before** its `try:` block (so a refusal isn't
turned into a 500), and returns `JSONResponse(status_code=refused.status,
content=refused.body)`.

`FrontendMessage` gets the same limits as `ChatTurn` (IMP-022): `role:
Literal["user", "assistant", "system"]`, `content` at most 20,000 characters; `messages`
at most 9 items. `system` messages are dropped before building history.

## 6. Documents (`app/api/endpoints/documents.py`, IMP-021, IMP-022)

```python
def _artifact_dir(document_id: str) -> Path:
    """The artifact folder for a real document ID; 404 for anything else."""
    try:
        canonical = str(uuid.UUID(document_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(settings.ARTIFACTS_PATH) / canonical
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    return path
```

Used by `get_document` and `related_documents`. Bounds: `limit: int = Field(10, ge=1,
le=50)` on `DocumentSearchQuery`; `Query(1, ge=1, le=3)` for `depth` and
`Query(10, ge=1, le=50)` for `limit`. Upload: look up the suffix in the loader's
extension table (expose it as `DocumentLoaderService.supported_extensions()`) and answer
415 before `NamedTemporaryFile`.

`TaskRequest.query: str = Field(max_length=20000)` in `task.py`.

## 7. Guard config (`app/core/config.py`, IMP-023)

`_build_security_guard_settings()` starts from the YAML object and overrides only the
fields that have environment variables:

```python
return base.model_copy(update={
    "enabled": _triage_bool("WEBAGENT_GUARD_ENABLED", base.enabled),
    "input_mode": _mode("input_mode", "WEBAGENT_GUARD_INPUT_MODE", base.input_mode),
    "tool_mode": _mode("tool_mode", "WEBAGENT_GUARD_TOOL_MODE", base.tool_mode),
    "judge": _triage_bool("WEBAGENT_GUARD_JUDGE", base.judge),
    "collect": _triage_bool("WEBAGENT_GUARD_COLLECT", base.collect),
})
```

Every other field (`harm_judge`, `jailbreak_judge`, `prompt_guard_threshold`, retention,
directories) comes through unchanged. The test compares `settings.SECURITY_GUARD` with
the YAML for each environment, field by field.

## 8. Agent settings (IMP-030, IMP-031)

### 8.1 `app/services/llm.py`

```python
def apply_agent_settings(llm, agent_cfg) -> Any:
    """Apply an agent's YAML max_tokens and, only if the YAML sets it, temperature."""
    llm = apply_max_tokens(llm, getattr(agent_cfg, "max_tokens", None))
    if agent_cfg is not None and "temperature" in agent_cfg.model_fields_set:
        for target in (llm, getattr(llm, "llm", None)):   # TogetherWrapper and its client
            if target is not None and hasattr(target, "temperature"):
                try:
                    setattr(target, "temperature", agent_cfg.temperature)
                except (AttributeError, ValueError, TypeError):
                    logger.warning(f"Could not set temperature on {type(target).__name__}")
    return llm
```

`get_agent_llm` calls it instead of `apply_max_tokens`. `model_fields_set` holds only the
keys present in YAML, because `AgentsConfig` is built from the parsed file.

### 8.2 Agents

- **Senior Research.** `self.max_iterations = get_agents_config().senior_research.max_follow_up_questions`.
  The prompt's "(Maximum: 3)" becomes "(Maximum: {max_iterations})", filled when the
  chain is built. The two log lines that hard-code "/3" use the same value.
- **Web research.** Count `perform_web_search` calls in one pass; stop when the count
  reaches `agents.web_research.max_searches_per_task` (default 5; prod 10). Today a pass
  makes at most 3, so the cap changes nothing until someone lowers it.
- **Internal research.** `k = min(<current k>, agents.internal_research.max_documents)`
  at the two `retrieve_documents` calls (`internal_research.py:1117`, `:1138`). The cap
  is an upper bound, so prod's 50 doesn't raise today's 20.
- **Remove** any `AgentConfig` field still unused afterwards from `config_def.py` and all
  three YAML files. (`coding.allowed_modules` is used, by `coding_assistant.py:86`.)

`model_fields_set` was checked on the prod YAML: `senior_research` reports
`{model, temperature, max_follow_up_questions}` and `data_analysis` `{model, temperature}`,
so the "only when set" rule works as written.

### 8.3 Failed evaluation parse (`senior_research.py`)

`_parse_evaluation` returns, for both the "no block" and the "exception" branches:

```python
{"research_status": "unknown", "score": None, "requirements_met": [],
 "missing_information": [], "research_questions": [], "requires_additional_research": True}
```

`unknown` takes the "needs more research" branch until `reached_max_iterations`, then the
existing max-iterations branch finishes with the last synthesis. `verified_findings["score"]`
stores `None`, not 10.

## 9. Timeouts (IMP-040)

- `together_llm.py`: `ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT_SECONDS))`.
  The existing retry already treats `TimeoutError` as transient (`"Timeout"` marker,
  `llm.py:283-286`).
- `web_search.py`: both `asyncio.to_thread(self.tavily_client.search, …)` calls are wrapped in
  `asyncio.wait_for(..., timeout=settings.SEARCH_TIMEOUT_SECONDS)`. A timeout raises
  `SearchError("Search timed out after 30 s")` (3.2.7 logs it and returns `[]`; 3.3.0
  reports it, section 13).

## 10. One failed research agent (`app/graph/workflows.py`, IMP-043)

```python
results = await asyncio.gather(
    *(agent(state.model_copy(deep=True)) for agent in agents), return_exceptions=True
)
failures = [(name, r) for name, r in zip(names, results) if isinstance(r, BaseException)]
for name, exc in failures:
    if not isinstance(exc, Exception):        # GuardStop, TaskBudgetExceeded, CancelledError
        raise exc
    state.add_log(f"{name} failed: {type(exc).__name__}")
    state.context.setdefault("research_errors", []).append({"agent": name, "error": str(exc)[:300]})
ok = [r for r in results if not isinstance(r, BaseException)]
if not ok:
    raise RuntimeError("; ".join(f"{n}: {e}" for n, e in failures))
security_ledger.merge(state, ok)
# … existing merge of reports and history over `ok`
```

`names` is built next to `agents` ("web_research", "internal_research"). Errors that
derive from `BaseException` only (guard stops, the budget, cancellation) still end the run.

## 11. Run supervisor (`app/services/run_supervisor.py`, IMP-041, IMP-050)

```python
STOP_CODES = {"timeout", "cancelled", "budget"}
POLL_SECONDS = 2.0

class RunSupervisor:
    def __init__(self):
        self._runs: Dict[str, asyncio.Task] = {}

    def start(self, task_id: str, coro: Awaitable[None]) -> None:
        """Run `coro` as a task and watch it; returns at once."""
        run = asyncio.create_task(coro, name=f"run-{task_id}")
        self._runs[task_id] = run
        asyncio.create_task(self._watch(task_id, run), name=f"watch-{task_id}")

    async def _watch(self, task_id: str, run: asyncio.Task) -> None:
        deadline = time.monotonic() + settings.TASK_MAX_SECONDS
        try:
            while not run.done():
                await asyncio.wait({run}, timeout=POLL_SECONDS)
                if run.done():
                    break
                if result_store.cancel_requested(task_id):
                    await self._stop(task_id, run, "cancelled", "Research stopped.")
                elif time.monotonic() > deadline:
                    await self._stop(task_id, run, "timeout",
                                     "The research took too long and was stopped.")
        finally:
            self._runs.pop(task_id, None)

    async def _stop(self, task_id, run, code, message) -> None:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await run
        result_store.save_final(task_id, status="error", error_code=code, error=message)
```

- `TaskManager.run_workflow` and `run_direct` are no longer passed to `BackgroundTasks`.
  `task.py` and `frontend.py` call `supervisor.start(task_id, task_manager.run_workflow(...))`.
  One `RunSupervisor` per worker, created with the task manager.
- On cancel, `asyncio.CancelledError` passes through `run_workflow`'s `except Exception`
  (it is a `BaseException`), and its `finally` skips the save because the status is still
  `processing`. So the supervisor's `save_final` is the only write. Their normal final
  save changes from `result_store.save_result` to `save_final`.
- **`result_store.save_final(task_id, **record)`**: reads the current file; if its
  `error_code` is in `STOP_CODES`, it does nothing and returns `False`; otherwise it writes
  the record (merged with `task_id`, `query`, `created_at` from the file) atomically, as
  today. This is the "final write wins once" rule. Workers don't share memory, so the
  check reads the file; the window between read and write is a few milliseconds and the
  only race is a stop and a finish in that window, where either outcome is acceptable.
- **Cancel route** in `task.py`:

```python
@router.post("/{task_id}/cancel", status_code=202)
async def cancel_task(task_id: str):
    saved = result_store.load_result(task_id)
    if saved is None:
        raise HTTPException(404, "Task not found")
    if saved.get("status") != "processing":
        raise HTTPException(409, "Task already finished")
    result_store.request_cancel(task_id)       # touch <id>.cancel
    return {"status": "cancelling"}
```

- `startup_checks`: if `TASK_TIMEOUT_MINUTES * 60 <= TASK_MAX_SECONDS + 60`, log a warning
  (a stale-record check shorter than the run limit would report live runs as interrupted).

## 12. Progress (`app/services/progress.py`, `task_manager.py`, IMP-052)

`run_workflow` replaces `ainvoke` with:

```python
final_values = None
async for mode, chunk in workflow.astream(initial_state, stream_mode=["updates", "values"]):
    if mode == "values":
        final_values = chunk
    elif mode == "updates":
        node = next(iter(chunk), None)
        progress.save(task_id, node, _logs_of(final_values))
result = final_values
```

`progress.save` writes `progress: {step, logs, updated_at}` into the result file with
`status: "processing"`, through `result_store.save_progress`, which does nothing if the
file already holds a final status. `logs` is the last 5 entries of `state.logs`. The step
label comes from:

```python
STEP_LABELS = {
    "supervisor": "Planning the research",
    "research": "Web and internal research",
    "senior_research": "Checking the findings",
    "team_manager": "Writing the report",
    "data_analysis": "Analysing data",
    "coding_assistant": "Writing code",
    "coding_review": "Reviewing code",
    "document_extraction": "Reading the document",
}
```

An unknown node uses its name with underscores replaced. `GET /api/task/{id}` adds
`body["progress"] = saved["progress"]` when the status is `processing`, reading the file
whether or not this worker runs the task.

The pinned LangGraph is 0.2.76, whose `astream` accepts a list of stream modes
(`langgraph/pregel/__init__.py:1751`). If a future upgrade changes the tuple shape, the
test in section 20 fails first.

## 13. Search errors (`web_search.py`, `web_research.py`, IMP-055)

`perform_web_search` returns `List[dict]` as today, but on failure or timeout it raises
`SearchError` instead of returning `[]` after the retry. `web_research.py` catches it at each
of its three call sites, appends `{"query_chars": len(q), "error": str(e)}` to a local
`search_errors` list, logs it, and goes on. The report dict gains `search_errors` when the
list isn't empty, and the workflow log gains "Web search failed: <error>".

## 14. Token budget (`app/services/task_budget.py`, IMP-042)

```python
class TaskBudgetExceeded(BaseException):
    """Derives from BaseException so agents' `except Exception` can't swallow it."""

@dataclass
class _Counter:
    limit: int
    used: int = 0

_counter: ContextVar[Optional[_Counter]] = ContextVar("task_budget", default=None)

def start(limit: int) -> Token: ...           # set a new counter for this run
def check() -> None: ...                      # raise if used >= limit > 0
def add(tokens: int) -> None: ...             # used += tokens
def used() -> int: ...
```

- `TaskManager.run_workflow` and `run_direct` call `task_budget.start(settings.TASK_TOKEN_BUDGET)`
  next to `current_task_id.set`. Child tasks from `gather` and `to_thread` copy the context,
  and share the same `_Counter` object, so their tokens add up.
- `TogetherWithUsage._generate` and `_agenerate` call `task_budget.check()` before each
  request, and `task_budget.add(usage.get("total_tokens", 0))` in `_result`.
- The task manager catches `TaskBudgetExceeded` next to `GuardStop` and saves
  `error_code: "budget"` with the message from IMP-042. Every final record gets
  `tokens_used: task_budget.used()`.
- Measuring the default (task 2.10): `tokens_used` from 10 typical agentic runs in uat.

## 15. Privacy in logs (IMP-070)

A helper in `app/core/log_context.py`:

```python
def describe_text(text: Optional[str]) -> str:
    """Log-safe description of user text: its length, plus the text only in dev at DEBUG."""
    return f"{len(text or '')} chars"
```

Every agent line of the form `logger.info(f"... query: {state.query}")` becomes
`logger.info(f"... ({describe_text(state.query)})")`, with an extra
`logger.debug("Query: %s", state.query)` only when `settings.WEBAGENT_ENV == "dev"`. The
same applies to other lines that print search queries or history
(`web_research.py:127`, the supervisor, Senior Research, Team Manager, internal research,
data analysis, the coding agents, `web_search.py:106`, `:136`). The test in section 20 finds
any line missed.

## 16. Results: retention, delete, list (`result_store.py`, IMP-071, IMP-072)

```python
def request_cancel(task_id) -> bool          # create <id>.cancel
def cancel_requested(task_id) -> bool        # marker exists
def save_progress(task_id, progress) -> None # no-op once final
def save_final(task_id, **record) -> bool    # section 11
def delete_result(task_id) -> bool           # remove .json, .json.tmp and .cancel
def list_results(limit=20, offset=0) -> List[Dict]  # newest first by created_at
def cleanup(retention_days, now=None) -> int  # returns files removed
async def cleanup_daily() -> None            # cleanup at startup, then every 24 h
```

- `list_results` reads only the fields `TaskInfo` needs (`task_id`, `status`,
  `created_at`, `completed_at`, `query` replaced by `describe_text(query)`), so the list
  never returns question text.
- `cleanup` judges age by `created_at`, falling back to the file's modified time.
- `main.py` starts `cleanup_daily` next to the triage and security loops
  (`main.py:106-110`) and cancels it on shutdown.
- `TaskManager.delete_task` calls `delete_result`. `GET /api/v1/tasks` takes `limit:
  int = Query(20, ge=1, le=100)`, `offset: int = Query(0, ge=0)` and returns
  `list_results(...)`.

## 17. Hardening (IMP-063, IMP-065, IMP-024)

- `app/api/router.py`: move `/health/detailed` from `health_router` into a router with
  `Depends(require_api_key)`. `/health` and `/api/v1/health/` stay open.
- `main.py`: when `WEBAGENT_ENV` is `uat` or `prod`, create the app with `docs_url=None`,
  `redoc_url=None`, `openapi_url=None`.
- `app/core/security.py`: delete `DEV_API_KEYS`, `create_jwt`, `decode_jwt`,
  `validate_auth`; `_configured_api_keys` returns only `WEBAGENT_API_KEY`.
- One CORS setup: delete the `app.add_middleware(CORSMiddleware, …)` in `main.py:79-85`.
  Add `CORS_ALLOW_METHODS`, `CORS_ALLOW_HEADERS` and `CORS_ALLOW_CREDENTIALS` to `settings`
  from `CorsConfig` (`config_def.py:28-33`, which already has them), and use them in
  `setup_middlewares`. Prod and uat `allow_headers` (`prod.yaml:19`) lack `X-API-Key`; add
  it, so a browser-based API client still passes the preflight. The web app itself is not
  affected: its proxy calls the backend server to server.
- `lifespan`: move the bodies of the `@app.on_event("startup")` and `("shutdown")`
  functions in `main.py` and `core/metrics.py` into one `asynccontextmanager` passed to
  `FastAPI(lifespan=...)`.

## 18. Error envelope (IMP-082, 3.4.0)

`app/api/http_errors.py` gains `error_body(code, message, detail=None) -> dict` returning
`{"error": {"code": code, "message": message}}`, plus `"detail": detail` when given.
Exception handlers in `main.py` use it for `HTTPException` (code from status:
`not_found`, `unauthorized`, …; `detail` kept as today), validation errors (`detail`
kept as today), the rate limit (`rate_limited`), the size limit (`too_large`) and the
global handler (`internal_error`). No field that exists today is removed (IMP-NFR-06). Guard refusals keep their top-level body, which the web app
reads (`src/utils/apiClient.ts`, `isGuardRefusal`).

## 19. Cleanup (3.4.0) and observability (3.5.0)

- **IMP-080 (3.4.0):** delete only orphans and true duplicates, for example
  `services/lda_document_extractor.py` (unused; LDA is in `document_extraction_agent.py`).
  Do not delete the advanced observability stack.
- **IMP-128 (3.5.0):** repair imports on `drift_detection`, `grafana_integration`,
  `metrics_registry`, `mlflow_integration`, `llm_service`, `api/routes/observability`, and
  `self_hosted_llm_service`; merge `doc_ingestion_service` drift helpers into
  `document_ingestion_service.py`; mount the observability router on `/api/v1`; document
  how it relates to `/metrics` and LangSmith. Chat routing stays in `llm.py` only
  (IMP-086).
- **IMP-081:** remove `workflow.add_node("web_research", …)` and `("internal_research", …)`
  (`workflows.py:237-238`); remove `_route_to_team_manager` and Senior Research's
  `needs_analysis` branch (or wire it to `team_manager_router` with a test).
- **IMP-090:** `vectordb.search` wraps `similarity_search_with_score` in
  `asyncio.to_thread`.
- **IMP-091:** document extraction skips LDA above `LDA_MAX_CHARS`, caps `passes` at 5, and
  adds `"topics_skipped": "document too large"` to the result.
- **IMP-086:** add `get_embeddings()` to `vectordb.py` (today's `OpenAIEmbeddings(api_key=
  settings.OPENAI_API_KEY)`), use it in `VectorDBService.__init__` and in
  `internal_research.py:205`. A test walks `backend/app` with `ast` and checks the import
  rule in the requirement.
- **IMP-095:** `get_embeddings()` picks `OpenAIEmbeddings` or
  `langchain_huggingface.HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
  cache_folder="/opt/models")` by `EMBEDDINGS_PROVIDER`. `langchain-huggingface` is already
  a dependency (`requirements.txt:35`), so nothing new is installed (IMP-NFR-02).

## 20. Tests

| Test file | Covers |
|---|---|
| `test_client_identifier.py` | IMP-012 table |
| `test_dev_exposure.py` | IMP-014 cases |
| `test_chat_completions_guard.py` | IMP-020: act refuses like `/api/task`; shadow runs; 400 stays 400 |
| `test_documents_ids.py` | IMP-021, IMP-022 bounds and 415 |
| `test_guard_config_passthrough.py` | IMP-023, per environment |
| `test_agent_settings.py` | IMP-030: per YAML, each agent's loop limit, temperature (set and unset), caps |
| `test_senior_research.py` (extend) | IMP-031 three parse cases |
| `test_together_timeout.py`, `test_web_search_async.py` (extend) | IMP-040 |
| `test_research_partial_failure.py` | IMP-043: one fails, both fail, a `BaseException` still ends the run |
| `test_run_supervisor.py` | IMP-041 (1 s limit), IMP-050 (two `TaskManager`s, one folder), "final write wins once" |
| `test_task_progress.py` | IMP-052: fake 3-node graph, 3 updates, read from another instance; also asserts the `astream` tuple shape |
| `test_task_budget.py` | IMP-042: counter shared across `gather`; `except Exception` in a fake agent doesn't swallow it |
| `test_no_question_in_logs.py` | IMP-070: marker question, `WEBAGENT_ENV=prod`, full fake run, `caplog` |
| `test_result_store.py` (extend) | IMP-071, IMP-072 |
| `test_hardening.py` | IMP-063 (docs 404 in prod, detailed health 401), IMP-065 (dev keys rejected), IMP-024 (CORS preflight) |
| `test_error_envelope.py` | IMP-082 |
| `test_graph_reachable.py` | IMP-081 |

All use the existing offline fakes (`tests/fakes/`) and the network block in
`tests/conftest.py`.
