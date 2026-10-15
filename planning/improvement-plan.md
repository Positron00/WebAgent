# WebAgent Improvement Plan

Version reviewed: 3.2.6 (commit `741368a`).
Every file and line reference below was checked against the code on 2026-10-13.
Decisions in section 4 were made on 2026-10-13, and the plan reflects them.
Releases 3.2.7–3.5.0 are specified in
[`planning/specs/platform-improvements/`](specs/platform-improvements/README.md); when the
two disagree, the spec wins. **Current platform release: 3.5.1** (2026-10-15; 3.5.0 plus CI fixes).
File and line references below describe 3.2.6; what was done and what is still open is in the
spec's [implementation status](specs/platform-improvements/tasks.md#implementation-status-2026-10-14).

## 1. Summary

WebAgent works, and the parts that changed most recently (the agentic security guard,
triage, task status and the web app's agentic mode) have solid offline tests. The gaps
are mostly around them:

1. **Nothing runs automatically.** There is no CI, and the git hooks only run Git LFS.
   All 633 tests pass today, but only because someone runs them by hand.
2. **The web app is an open door to paid APIs.** `/api/chat`, `/api/task` and
   `/api/triage` accept anyone who can reach the site. The web app's own rate limiter
   never counts requests, and the backend's per-user limit can be dodged by sending a
   made-up `X-Forwarded-For` header.
3. **Runs have no overall time limit, no cost limit and no cancel button.** Single calls
   do time out, but late: 300 s for an async Together call (aiohttp's default, retried up
   to 3 times) and 100 s for a Tavily search. With up to 8 model calls and several
   searches, a slow upstream can keep a run going far past the web app's 8-minute wait.
   Users see "Thinking…" with no progress.
4. **Some settings in the YAML files do nothing.** The research-loop limit (prod says
   5; the code always uses 3), each agent's `temperature`, `max_documents`,
   `max_searches_per_task`, and three security guard settings are read nowhere.
5. **User questions are kept forever.** Every agent logs the question text, and result
   files never expire. (The backend runs as one replica with files on one volume; we
   decided to keep it that way, see section 4.)
6. **A large slice of the backend is demo-only or not wired into the running server**
   (especially the advanced observability stack). The spec keeps that stack for repair in
   **3.5.0** (IMP-128); **3.4.0** removes only true orphans. The README still describes
   several features that are really stubs (model gateway, artifact registry, state
   persistence).
7. **101 files are over the 200-line rule:** 59 in app code, 19 tests, and 23 demos,
   scripts and examples.

The security guard and triage still run in shadow mode. That is deliberate and tracked
in their own plans; this plan lists only the gaps around them.

## 2. How this was checked

- Ran every test suite: backend `pytest` (497 passed, 3 skipped, 55% line coverage),
  web app Jest (136 passed, 60% line coverage), `tsc --noEmit` (clean), `next lint` (clean).
- Four parallel read-only reviews: agents and workflow; API, security and config;
  web app; tests, deployment and repo hygiene.
- Then read the code behind every item, and ran the config at runtime to confirm
  the ignored settings. Claims that turned out weaker than first reported are in
  section 12.

**Priority:** **High** = can cost money, leak data or hang users now. **Med** = real bug
or risk with limited reach. **Low** = cleanup or hardening.
**Effort:** **S** = under half a day, **M** = 1–3 days, **L** = more than 3 days.

## 3. Top 10

| # | Item | Why | Effort |
|---|------|-----|--------|
| 1 | Add CI (T1) | Every later change needs a safety net | M |
| 2 | Make the web app rate limiter count requests (Q1) | It is a no-op today | S |
| 3 | Stop forwarding the caller's own `X-Forwarded-For` (S13) | Anyone can get a fresh backend rate limit on every request | S |
| 4 | Protect the web app's API routes (S2) | Anyone who reaches the site can spend Together and Tavily credit | M |
| 5 | Dev backend: listen on localhost only, or require a key (S1) | Same risk on any shared network | S |
| 6 | An overall run limit, plus shorter timeouts on async Together and Tavily calls (R10, Q3, Q4) | Slow calls add up to runs far longer than the web app waits, still spending | S |
| 7 | Apply the per-agent YAML settings (Q15) | Prod config says one thing, code does another | S |
| 8 | Fix two small reliability bugs: one failed research agent fails the run (Q5); web app retries after a timeout fail instantly (Q2) | Cheap fixes for failures users see | S |
| 9 | Cancel button and live progress for research runs (R1, R2) | Biggest user-facing gap | M |
| 10 | Remove orphan code, repair observability in 3.5.0, fix README claims (H1 narrowed, IMP-128, docs table in section 11) | Drops true duplicates in 3.4.0; integrates drift/LLM APIs in 3.5.0; corrects false claims | M |

## 4. Decisions

All five were chosen for the simplest option that stays reliable at the current scale
(a handful of users, one backend).

1. **Protecting the web app (S2): a shared access code.** Next.js middleware asks for
   the code once and keeps it in a signed, HTTP-only cookie. It works the same on
   Vercel, Docker and `next start`. *Not chosen:* Vercel password protection (depends
   on the Vercel plan and doesn't cover self-hosting); user accounts (only needed for
   per-person history or quotas). Also set monthly spending caps in the Together and
   Tavily dashboards as a backstop.
2. **Advanced observability (H1): superseded by spec IMP-128.** The platform-improvements
   spec keeps the stack and schedules **repair and integrate in 3.5.0** (drift APIs, LLM
   management surface, Grafana helpers). **3.4.0** removes only true duplicates and orphans
   (for example unused `lda_document_extractor.py`). Production metrics stay on `/metrics`
   and `app/core/observability.py`. The separate model-microservices framework in
   `backend/models/` is unchanged (see H10).
3. **Task state (C2): keep the JSON files, add retention.** They already survive
   restarts, are shared between workers and are tested. *Not chosen:* Redis, which
   adds a service that can fail. The only thing it would fix today is the per-worker
   rate limit, which just makes the real limit about double the setting.
4. **Keeping user questions (S14):**
   - **Logs:** don't write question text; log the task ID and the question's length.
   - **Result files:** delete after 30 days (`WEBAGENT_RESULTS_RETENTION_DAYS=30`).
   - **Guard and triage collections:** unchanged. They have their own
     `collect_retention_days`, and the shadow-week review needs them.
5. **Replicas: stay on one.** A research run mostly waits on Together, and the backend
   spaces Together calls 2 seconds apart in each process, so more replicas add API
   pressure, not speed. C6 and C7 are dropped; the docs will say one replica by design.

## 5. Phase 0: quick fixes

Each is a small, local change that should come with a test.

| ID | Pri | Problem | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| Q1 | High | Web app rate limiter never records a request, so `isRateLimited()` is always false | `src/app/api/chat/route.ts:30` only checks; `addRequest()` at `src/utils/rateLimiter.ts:37` is never called | Call `addRequest()` when a request is accepted, with one bucket per sign-in session: today the limiter is one bucket for the whole server (10 a minute, `src/config/chat.ts:153`), so a fix without per-session keys would let one user block everyone (spec IMP-011) | S |
| Q2 | Med | `fetchWithRetry` makes one `AbortController` and one timer for all attempts and clears the timer on the first failure. After a timeout every retry fails at once; after another failure, retries have no timeout | `src/utils/apiClient.ts:250-253`, `:270`, `:320` | New controller and timer inside the loop for each attempt | S |
| Q3 | Med | Async Together call falls back to aiohttp's default 300 s timeout (the sync call uses 120 s); with 3 attempts one call can take about 15 minutes | `backend/app/services/together_llm.py:102` vs `:81-83` | Pass `ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)` to `ClientSession` | S |
| Q4 | Low | Tavily search relies on the client's own 100 s timeout (`tavily/tavily.py:62`), twice with the retry at `:137` | `backend/app/services/web_search.py:108-115` | Wrap both calls in `asyncio.wait_for(...)` at about 30 s and log the timeout | S |
| Q5 | Med | `asyncio.gather` without `return_exceptions=True`: if one research agent raises, both results are lost | `backend/app/graph/workflows.py:170-172` | Collect exceptions, keep the successful report, record the failure in state | S |
| Q6 | Med | Document ID is joined onto the artifacts folder unchecked; `..` reaches the parent folder | `backend/app/api/endpoints/documents.py:188`, same pattern in `related` | Require a UUID, as `result_store.py:15-20` already does for task IDs | S |
| Q7 | Med | Missing input limits: the upload is written to disk before its type is checked; search `limit`, related `depth`/`limit` and the main task `query` have no maximum | `documents.py:100-103`, `:44`, `:235-236`; `task.py:33` (history turns are capped at 20,000 characters at `:29`, but the query isn't) | Allowlist extensions before writing; add `le=` bounds; `query: str = Field(max_length=20000)`; the same limits on `/api/v1/chat/completions` messages (at most 9, each at most 20,000 characters; spec IMP-022) | S |
| Q8 | Med | `harm_judge`, `jailbreak_judge` and `prompt_guard_threshold` in YAML are dropped when guard settings are rebuilt | `backend/app/core/config.py:317-326` | Pass every field through; add a config test | S |
| Q9 | Med | Senior Research: if the model's evaluation can't be parsed, the research is marked `complete` with score 10, so a bad parse ends the loop as a pass. The prompt also hard-codes "Maximum: 3" | `senior_research.py:207-226`, `:34` | Treat a failed parse as "unknown" and log it; put the real limit in the prompt | S |
| Q10 | Low | `npm run research` runs `scripts/run_webagent.js`, which doesn't exist | `package.json:24`; only `scripts/run_webagent.py` exists | Point at the Python script or remove the npm script | S |
| Q11 | Low | `fine_tune_service.py` imports a module that doesn't exist | `backend/models/services/fine_tune_service.py:18` (`models.schemas`) | Delete the service or add the module | S |
| Q12 | Low | CORS is added twice, both with `*` methods and headers, so prod YAML's tighter lists are ignored | `backend/app/core/middleware.py:108-114`; `backend/main.py:79-85` | Keep one CORS setup, driven by YAML; add `X-API-Key` to uat/prod `allow_headers` (`prod.yaml:19`), which would otherwise start blocking browser-based API clients | S |
| Q13 | Low | `RATE_LIMIT_BURST_LIMIT` is loaded but never used | `config.py:221`, `:386` | Implement or delete | S |
| Q14 | Low | FastAPI `on_event` is deprecated (shows as test warnings) | `backend/main.py:95,113`; `core/metrics.py:476` | Move to a `lifespan` handler | S |
| Q15 | Med | Per-agent YAML settings are ignored. Senior Research reads `settings.senior_research_config`, which doesn't exist (confirmed at runtime), so the loop limit is always 3 and prod's `5` does nothing. `get_agent_llm` reads only `model`, `provider` and `max_tokens`, so each agent's `temperature` is ignored and every Together agent runs at the global `llm.temperature` (0.7 dev, 0.5 uat, 0.3 prod) instead of its own 0.0–0.4. `max_documents` and `max_searches_per_task` are defined but never read | `senior_research.py:129-130`; `services/llm.py:640-650`, `:737-742`; `config/prod.yaml:68-97` | Read the limit from `get_agents_config().senior_research`; apply `temperature` only where the YAML sets it; use or delete the other keys; add a test that loads each YAML file and checks each agent's values. This changes answers, so compare a few runs before and after (spec IMP-NFR-07) | S |

## 6. Phase 1: security and privacy

| ID | Pri | Problem | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| S1 | High | Dev backend binds `0.0.0.0` with API keys off, so anyone on the network can run tasks on your keys | `backend/config/dev.yaml:8`; `api_key_enabled` is set only in `uat.yaml:130` and `prod.yaml:130` | Bind `127.0.0.1` in dev, and refuse to start on a non-loopback host without a key. Compose (which defaults to dev) also publishes 8000, 9090 and Grafana on every interface (`backend/docker-compose.yml:19,50,69`); bind them to `127.0.0.1` | S |
| S2 | High | Web app routes `/api/chat`, `/api/task` and `/api/triage` don't check the caller; the server adds the Together key and backend key itself | `src/app/api/chat/route.ts:27-154`; `src/app/api/task/backend.ts:15-32` | Decision 1: access-code middleware covering every page and `/api/*` route except `/api/health` and the sign-in route, plus spending caps in the Together and Tavily dashboards. The backend's per-user limit (after S13) is the real rate limit, because the web app's in-memory limiter can't work on Vercel | M |
| S3 | Med | `/api/v1/chat/completions` starts a workflow without the intake guard that `/api/task` runs | `backend/app/api/endpoints/frontend.py:47-93` vs `task.py:74-122` | Keep the endpoint (the triage spec's REQ-070 gives API clients a `mode` field on it) and call the same intake guard, moved into a shared module (spec decision 11) | S |
| S4 | Med | No security headers or CSP on the web app | `next.config.js` has no `headers()` | Add CSP, `frame-ancestors`, `nosniff`, `Referrer-Policy`, HSTS | S |
| S5 | Med | `/api/chat` forwards any `image` value; type is checked only in the browser, and size not at all | `route.ts:39`, `:126-130`; `src/components/Chat.tsx:74-85` | Server side: require `data:image/(png\|jpeg\|gif\|webp);base64,` and cap the length | S |
| S6 | Low | Source links use the URL as-is in `href`; a `javascript:` URL would run | `src/components/chat/SourceCanvas.tsx:110-111` | Allow only `http:` and `https:` | S |
| S7 | Med | Info leaks on open endpoints: `/api/ping` returns the key's length and the model; `/api/v1/health/detailed`, OpenAPI docs and `/metrics` need no key | `src/app/api/ping/route.ts:9-17`; `backend/app/api/router.py:11-12`; `backend/main.py:51-53` | Ping returns `{status: ok}`; the web app's `/api/health` (which must stay open for checks, and returns memory use and log statistics) returns only status and version; detailed health behind the key and docs off in uat/prod; `/metrics` reachable only inside the cluster | S |
| S8 | Low | Error panel's diagnostics button sends a real chat request | `src/components/chat/MessageList.tsx:507-511` | Call `/api/ping` instead | S |
| S9 | Low | Hard-coded `dev_admin_key` / `dev_user_key` (accepted only in dev); JWT helpers exist but no route uses them, while the README says JWT protects the API | `backend/app/core/security.py:24-36`, `:106-160`; `README.md:223` | Remove the static keys and the unused JWT code (access is by `X-API-Key`); fix the README | S |
| S10 | Low | Next image optimizer allows any HTTPS host | `next.config.js:6-11` | List the hosts actually used | S |
| S11 | Low | Optional LLM images run as root | `backend/app/services/Dockerfile.llm`, `Dockerfile.llm-mlflow` (no `USER`) | Add a non-root user, like the main image | S |
| S12 | Low | No `SECURITY.md` | repo root | Add contact, scope and response time | S |
| S13 | High | The web app copies the caller's own `X-Forwarded-For` header to the backend. The backend trusts the web app and rate-limits by the first address in that header, so a caller can send a new made-up address on every request. Vercel overwrites the header; `next start` and Docker don't | `src/app/api/task/backend.ts:8-13`, `:24-27`; `backend/app/core/security.py:76-84` | Stop copying the header. Rate-limit by sign-in session instead: the web app sends `X-WebAgent-Client: s:<session id>`, and the backend honours it only from a trusted proxy or a caller holding the API key (the Vercel case, whose addresses change). Otherwise take the right-most address that isn't a trusted proxy, not the first (spec decision 6, IMP-012) | S |
| S14 | Med | User questions are written to logs at INFO by every agent, and kept in `backend/data/results/` with no expiry | e.g. `supervisor.py:193`, `senior_research.py:261`, `web_research.py:127`; `result_store.py` | Decision 4: log the task ID and question length, not the text. Result-file retention is done in C2 | S |

Planned elsewhere, listed so they aren't forgotten: security guard `shadow` to `act`
(agentic-security task 4.6, after the shadow-week review), turning on the judge, the
output checkpoint on final answers, and triage `act` mode (agentic-triage task 3.11).

## 7. Phase 2: reliability and long runs

| ID | Pri | Problem | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| R1 | High | No way to cancel a run. The web app polls for up to 8 minutes; backend `DELETE` clears memory but the workflow keeps running | `src/utils/apiClient.ts:653-664`; `backend/app/api/endpoints/tasks.py:104-116`; `task_manager.py:422-432` | Run each workflow as an `asyncio.Task` instead of a `BackgroundTasks` job; add `POST /api/task/{id}/cancel`, which writes a marker file the running worker checks every 2 s (the cancel may reach the other worker); pass an `AbortController` into polling; add a Stop button | M |
| R2 | Med | Workflow logs appear only after the run ends | `MessageList.tsx:325-337`, `:474` | Return the current stage and recent log lines with the status; show them under "Thinking…" | M |
| R3 | Med | Any agentic failure silently falls back to plain chat, with only a line in the system prompt | `apiClient.ts:566-590` | Show a clear "research failed" state with a Retry button; make the fallback a choice | S |
| R4 | Med | Web search errors become an empty list, so "search broken" looks like "nothing found" | `web_search.py:119-149` | Return an error marker the agent logs and reports | S |
| R5 | Med | Vector search runs a sync search (and embedding call) on the event loop, blocking every other request | `backend/app/services/vectordb.py:153` | `await asyncio.to_thread(...)` | S |
| R6 | Med | Document extraction trains an LDA model on every request | `document_extraction_agent.py:252-261` | Cap input size and passes; cache, or make LDA optional | M |
| R7 | Low | Browser storage keeps the last N messages but doesn't handle "storage full"; saving then fails silently | `src/utils/storage.ts:51-58` | On `QuotaExceededError`, drop more old turns and tell the user | S |
| R8 | Low | Bare `except:` hides real errors | `coding_assistant.py:164,174,190,199`; `coding_review.py:327,345` | Catch specific exceptions and log them | S |
| R9 | Low | Jest uses `forceExit`, hiding timers left running by tests | `jest.config.js:50` | Clean up timers, then remove `forceExit` | S |
| R10 | High | No overall time limit on a run. `TASK_TIMEOUT_MINUTES` (30) only relabels an old task when someone reads it; the workflow keeps running and spending | `task_manager.py:24`, `:76-84` | A supervisor task cancels the run after `WEBAGENT_TASK_MAX_SECONDS` (420 s: twice a typical run and under the web app's 8-minute wait, so the user sees the backend's "took too long" message) and saves a clear timeout error | S |
| R11 | Med | No per-task token or cost limit; only per-call `max_tokens` | no budget logic in `backend/app` | Count the tokens Together reports per task (they all pass through `TogetherWithUsage._result`) and stop before the next call once over a budget set from measured runs. The error must derive from `BaseException`, like the guard's, so agents' `except Exception` can't swallow it | M |

## 8. Phase 3: state and data

The backend runs as one replica by design (decision 5;
`backend/kubernetes/deployment.yaml:10-18`: `replicas: 1`, `strategy: Recreate`, one
ReadWriteOnce volume), and task state stays in JSON files (decision 3).

| ID | Pri | Problem | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| C1 | Low | Rate limits are a dict in each worker, so with 2 workers the real limit is about double the setting (about 120 instead of 60 requests a minute per caller). Redis is installed and configured for this but never used | `backend/app/core/security.py:61`; `backend/Dockerfile:41`; `backend/requirements.txt:43`; `config.py:206-210` | Decision 3: keep per-worker limits and say so in `PRODUCTION_READINESS.md`; remove the unused `redis` package and settings (D2) | S |
| C2 | Med | Result files never expire; deleting a task leaves its file; the task list (`GET /api/v1/tasks`) shows only the answering worker's in-memory tasks and isn't paginated | `task_manager.py:41-43`, `:422-432`; `result_store.py`; `tasks.py:19-26` | Decision 4: delete result files older than `WEBAGENT_RESULTS_RETENTION_DAYS` (default 30) in a background loop like the existing triage and security cleanup (`backend/main.py:106-110`); delete the file when a task is deleted; build the list from the files and paginate it | S–M |
| C3 | Low | Graph database falls back to in-memory NetworkX, lost on restart | `backend/app/services/graphdb.py:48-62` | Persist it, or require Neo4j outside dev | M |
| C4 | Med | Embeddings always use OpenAI, so internal research needs an OpenAI key although chat uses Together | `vectordb.py:59-61` | Make the embedding provider configurable | M |
| C5 | Med | No backup of the `/app/data` volume, which holds the shadow-week collection | `deployment.yaml:186-198` | VolumeSnapshot or a backup CronJob; document the restore | S |
| C6 | — | No checkpointer, so a restart loses in-flight runs | `workflows.py:309` | **Dropped** (decision 5). State the limit in `PRODUCTION_READINESS.md` | S (docs) |
| C7 | — | More than one replica | — | **Dropped** (decision 5). Revisit only if one replica becomes the bottleneck | — |

## 9. Phase 4: tests and CI

| ID | Pri | Problem | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| T1 | High | No CI | no `.github/` folder | GitHub Actions: `npm ci`, `lint`, `check-types`, `test:coverage`; `pip install -r requirements.txt` (uses the constraints file); `pytest backend/tests`. Check out without Git LFS files (about 268 MB, which would use up GitHub's free LFS bandwidth in days); mark tests that need them `needs_lfs` and run those weekly with an LFS cache | M |
| T2 | Med | `lint-staged` is configured but no hook runs it | `package.json:73-78`; `.husky/` has only LFS hooks | Add `.husky/pre-commit` running `npx lint-staged` | S |
| T3 | Med | No Python lint or type check | no ruff or mypy config or script | Add `ruff check` now; `mypy` for `app/security` and `app/services` later | S, then M |
| T4 | Med | Weak coverage. Backend: `document_extraction_agent` 22%, `internal_research` 29%, `vectordb` 33%, `llm.py` 40%, `documents` endpoint 42%. Web app floors are 46% lines (`jest.config.js:32-38`); no tests for `rateLimiter`, `storage`, `SourceCanvas`, `ErrorBoundary`, `useDismiss` | coverage runs in section 2 | Test each Phase 0 and 1 fix first; in 3.4.0 raise the Jest floors by at least 5 points and add a backend coverage floor to CI | M |
| T5 | Low | Duplicate tests, and tests written as scripts | `test_langgraph.py` vs `test_langgraph_simple.py`; `test_model_manager.py` vs `_simplified.py`; `backend/tests/diagnostics/` vs `backend/diagnostics/tests/` (same two files); `__main__` blocks in 17 test files | Keep one of each; remove `__main__` blocks | S |
| T6 | Med | Red-team suite is a placeholder | `backend/tests/redteam/test_redteam_skeleton.py` | Add poisoned-search and forged-history cases before the guard moves to `act` | M |
| T7 | Med | No browser-to-backend test; the closest is a mocked graph run | `backend/tests/integration/test_workflow_e2e_scripted.py` | One Playwright run in Always mode against a backend with fake LLM and search | M |
| T8 | Low | No load test, although `PRODUCTION_READINESS.md:117` lists one | none found | Small k6 or Locust script for `/health` and `/api/task` with fakes | M |

## 10. Phase 5: code health

### Dead, duplicate and observability code

These modules have low or no test coverage. When this plan disagrees with
[`planning/specs/platform-improvements/`](specs/platform-improvements/README.md) on H1,
the spec wins (decision 2 → IMP-080 in 3.4.0, IMP-128 in 3.5.0).

| Module | Lines | Used by | Release |
|--------|------:|---------|---------|
| `core/grafana_integration.py` | 881 | demos; IMP-128 runbook | 3.5.0 repair |
| `core/drift_detection.py` | 844 | unmounted route until IMP-128 | 3.5.0 repair |
| `core/metrics_registry.py` | 586 | demos; IMP-128 | 3.5.0 repair |
| `core/mlflow_integration.py` | 406 | demos; IMP-128 | 3.5.0 repair |
| `services/self_hosted_llm_service.py` | 783 | Docker / examples; IMP-128 | 3.5.0 repair |
| `services/lda_document_extractor.py` | 618 | nothing (LDA is in the agent) | **3.4.0 delete** (IMP-080) |
| `services/document_ingestion_service.py` | 625 | **live:** `internal_research.py` | keep; gains doc drift in 3.5.0 |
| `services/doc_ingestion_service.py` | 238 | observability drift API | **3.5.0:** merged into `document_ingestion_service.py`, file deleted |
| `services/llm_service.py` | 233 | unmounted route until IMP-128 | 3.5.0 repair (observability only) |
| `api/routes/observability.py` | — | not mounted until IMP-128 | 3.5.0 mount on `/api/v1` |

- **H1 (IMP-080, 3.4.0; IMP-128, 3.5.0).** Delete only true orphans such as
  `lda_document_extractor.py` (full list in the backend design's 3.4.0 cleanup table).
  **Do not** delete the observability stack, `ml_observability_demo.py`, or
  `model_fine_tuning.py`. In **3.5.0**, repair imports, mount
  `api/routes/observability` behind the API key, merge document drift into
  `document_ingestion_service.py`, and remove `doc_ingestion_service.py`. Drop packages
  nothing else uses when safe (D1 / IMP-100–101). Keep `mlflow` for `task_tracking.py` and
  `mlflow_utils.py`. (M, split across 3.4.0 and 3.5.0)
- **H2 (IMP-086, 3.4.0).** Chat models are built only in `llm.py` (`together_llm.py` is
  imported only there). `llm_service.py` and `self_hosted_llm_service.py` stay for
  observability and self-hosted images after IMP-128; they must not duplicate chat routing.
  Embeddings today are built in two places (`vectordb.py:14` and `internal_research.py:205`);
  give them one `get_embeddings()` before the provider switch (IMP-095). (S)
- **H3.** Document extraction goes straight to the end of the graph (`workflows.py:267`),
  `_route_to_team_manager` only logs (`document_extraction_agent.py:635-644`), and
  Senior Research's `needs_analysis` flag (`senior_research.py:331-336`) is never read
  by a router. Wire them up or remove them. (S)
- **H4.** The graph registers `web_research` and `internal_research` nodes that no edge
  reaches (`workflows.py:237-238`); only the `research` node runs them. (S)

### Structure

- **H5.** Three config layers (`config.py`, `config_def.py`, `loadEnvYAML.py`) with
  different merge rules caused Q8 and Q15. Move to one typed loader and generate the
  `.env.local.example` list from it. (L)
- **H6.** 59 app-code files exceed 200 lines. Split each one the next time you change it,
  worst first: `internal_research.py` (1431: retrieval, reranking, ingestion),
  `llm.py` (1192: providers, wrappers, metrics), `apiClient.ts` (791: retry, direct
  chat, multi-agent, polling), `MessageInput.tsx` (776: voice, screenshot, menus,
  composer), `document_extraction_agent.py` (744), `MessageList.tsx` (544),
  `ChatContext.tsx` (488). (L, incremental)
- **H7.** Agent factories are inconsistent: some return one shared agent, some build a
  new one per call. Add one `AgentNode` type and typed report models. (M)
- **H8.** Two `ApiError` classes (`src/utils/errors.ts:1`, `src/utils/apiClient.ts:79`);
  two accessibility-settings types (`src/utils/storage.ts`, `src/types/storage.ts`). (S)
- **H9.** At least three error shapes reach clients: FastAPI's `{"detail": ...}`, the
  guard's top-level refusal body (`task.py:122`), and `{"error": ...}` from the rate
  limit, size limit and global handler (`security.py:272`, `:280`; `main.py:178-179`).
  Pick one. (M)
- **H10.** The model-microservices framework in `backend/models/` (`api/`, `registry/`,
  `services/`, `model_manager.py`, `run_manager.py`, `run_model_service.py`) is not
  imported by the app, but it has its own tests. Don't confuse it with
  `backend/models/triage/` and `backend/models/security/`, which hold the live encoders.
  Keep the framework for now; describe it in the README as an optional, separate tool,
  and fix its broken import (Q11). Decide later whether to delete it. (S)

### Repo clutter (all S)

- Delete `.babelrc.backup`, `babel.config.js.backup`, the local
  `backend/tests/test_langgraph.py.bak`, the empty `backend/api/` and `backend/src/`
  (only empty folders).
- Untrack `backend/diagnostics_results.json` and `backend/demos/demo_outputs/` (19 files).
- Move `test_docs/` to `backend/tests/fixtures/`.
- Remove web app dependencies that nothing in `src/` imports: `ai`, `axios`,
  `framer-motion`, `react-hook-form`, `clsx`, `tailwind-merge`, `zod`,
  `react-textarea-autosize`, `tailwindcss-animate`. Keep `@headlessui/react` if you use
  it for the consent dialog's focus trap (P4); otherwise remove it too.

## 11. Phase 6: dependencies, deployment and docs

### Dependencies and deployment

| ID | Pri | Problem | Fix | Effort |
|----|-----|---------|-----|--------|
| D1 | Med | `backend/requirements.txt` puts torch, chromadb, mlflow, gensim, neo4j, plotting libraries and dev tools (pytest, black) into the production image | Split into runtime, dev and demo files (dev also holds the libraries the dev-only local code sandbox needs); the image installs runtime only | M |
| D2 | Med | Old, replaced or unused packages: `py2neo` (`:89`), `python-jose` (`:63`), `PyPDF2` (`:82`), `platform-utils` (`:97`, unused), `redis` (`:43`, unused; decision 3) | Use the `neo4j` driver and `pypdf`; drop `python-jose` once S9 removes the JWT code; remove the unused ones | M |
| D3 | Med | Grafana 8.5.3 and Prometheus 2.36 are years old | Move to supported versions | S |
| D4 | Low | Compose image tag falls back to `latest`; Compose has no memory limits | Default to the package version; add limits | S |
| D5 | Low | `eslint-config-next` 14.0.4 vs `next` 14.2 | Align versions | S |

### Docs that don't match the code

| Doc | Claim | Reality |
|-----|-------|---------|
| `README.md:183` | "State management and persistence" | No checkpointer; only finished-task JSON is saved |
| `README.md:30` | "Artifact registry integration" | Local folder copy; `ARTIFACTS_REGISTRY_URL` is never read |
| `README.md:37-41`, `:289-303` | Model gateway is part of the platform | Not mounted; agents call Together directly. Reword as an optional, separate tool (H10) |
| `README.md:181` | "Parallel execution of compatible agents" | Only web and internal research run in parallel |
| `README.md:51` | "Up to 3 times" | True by accident: prod YAML says 5, but the code ignores it (Q15). Update after Q15 |
| `README.md:203-205` | Table and image extraction, hierarchical documents | Not found in the document code |
| `README.md:223` | JWT authentication | No route uses JWT; access is by `X-API-Key` |
| `README.md:4` | Last updated 2026-10-10 | Version 3.2.6 released 2026-10-13 |
| `PRODUCTION_READINESS.md:46` | Rolling updates | `strategy: Recreate` |
| `PRODUCTION_READINESS.md:16` | Kubernetes allows 1.5 GiB | Limit is 3 GiB, request 2 GiB |
| `PRODUCTION_READINESS.md:116-117` | End-to-end and load tests | Neither exists |
| `VERCEL_DEPLOYMENT.md:74` | The proxy forwards each user's `X-Forwarded-For`; all Vercel users share one budget | True today; after S13 (3.2.7) limits are per sign-in session on Vercel too. Update with S13 |

Add to `PRODUCTION_READINESS.md`, from the decisions: one replica by design, and why;
rate limits are per worker (about double the setting with 2 workers); a restart loses
runs in progress; result files are deleted after `WEBAGENT_RESULTS_RETENTION_DAYS`;
the web app needs an access code; set spending caps in the Together and Tavily
dashboards.

Also: archive `docs/VERSION_2_4_5.md` and `backend/docs/VERSION_2_4_{1,2}.md`; move
CHANGELOG entries older than 3.0 to `docs/CHANGELOG-archive.md` (the file is 133 KB);
add a short `CONTRIBUTING.md` and an ops `docs/RUNBOOK.md` (deploy, roll back, rotate
secrets, disk full, failed probes).

### Product gaps (later)

- **P1.** Stream direct answers instead of waiting for the full reply.
- **P2.** Several saved conversations, plus "export answer" and "export conversation".
- **P3.** Sources and code panel on phones (now `hidden lg:block`, `Chat.tsx:163`).
- **P4.** Focus trap in the consent dialog (`ConsentDialog.tsx`).

## 12. Review claims checked and downgraded

- **Calls with no timeout (Q3, Q4):** found while writing the spec. aiohttp defaults to
  300 s and the Tavily client sets 100 s, so neither waits forever. Q3 is now Medium and
  Q4 Low; the overall run limit (R10) stays High.

- **Document path traversal (Q6):** FastAPI path parameters can't contain `/`, so only
  `..` (one level up) is reachable. Medium, not High.
- **Unsafe links in answers (S6):** react-markdown 10 already removes `javascript:`
  links, and `MessageList.tsx` has no custom link renderer. Only `SourceCanvas` uses a
  raw `href`. Low.
- **Together rate-limit counter race** (`llm.py:256`): nothing awaits between the read
  and the write, so async callers can't race. Low; not listed.
- **Chat history wiped on page load** (`ChatContext.tsx:97-118`): the save effect writes
  `[]` once on mount, but the loaded messages are saved again on the next render.
  Low; fix with a "loaded" flag when the file is split.
- **Static dev API keys (S9):** accepted only when `WEBAGENT_ENV=dev`
  (`security.py:35`). Low.

## 13. What's already good (keep it)

- Backend tests run fully offline, block network access and never trace to LangSmith.
- The main Docker image runs as a non-root user with a health check; Kubernetes has
  startup, liveness and readiness probes and Prometheus alert rules.
- uat/prod refuse to start without a secret key, an API key, and (when code execution
  is on) the Docker sandbox.
- API keys are compared in constant time; task IDs are checked as UUIDs before touching
  disk; result files are written atomically; log files rotate.
- The Together key never reaches the browser.
- The toolbar menus and chat log have proper labels, Escape handling and live regions.

## 14. Release order

Every item in this plan is placed below.

| Release | Theme | Items |
|---------|-------|-------|
| 3.2.7 | Safety net, quick fixes, close the open doors | T1, T2, Q1–Q15, S1, S2, S3, S13, R10, Together and Tavily spending caps |
| 3.3.0 | Hardening, privacy and long runs | S4–S12, S14, C1, C2, C5, T3, R1–R4, R8 (with T3's lint rule), R11, docs table and `PRODUCTION_READINESS.md` additions in section 11 |
| 3.4.0 | Cleanup | H1 narrowed (orphans only), H3–H4, H8–H10, repo clutter, R5–R7, R9, C4, D1–D5, T4–T7, CONTRIBUTING, RUNBOOK, CHANGELOG archive |
| 3.5.0 | Observability | IMP-128: repair and integrate H1 stack (drift, `/api/v1/observability`, merge doc drift) |
| Later | Larger work | C3, H5–H7, T8, P1–P4; guard and triage `act` modes per their own plans |

C6 and C7 are dropped (decision 5); C6's limit is documented in 3.3.0.

**Done means:** a test that failed before the fix and passes after; CI green; a
CHANGELOG entry; and any doc that described the old behaviour updated.
