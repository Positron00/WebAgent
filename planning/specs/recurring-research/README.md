# Spec: recurring research with tracked summaries and email alerts

Status: v3, phase A implemented (3.11.0, 2026-10-18) · Source plan: [`planning/recurring-research-plan.md`](../../recurring-research-plan.md) · Base: `main` at v3.10.1 (`270a012`) · Releases: 3.11.0 (phase A: topics, scheduler, history, web app), 3.12.0 (phase B: change summaries, email). Both are deployed after shadow week 2.

## What this work does

Today a research task runs once, when someone asks. This spec adds **topics**: a question WebAgent researches again daily or weekly. Each run's report is kept with the sources that were new since the last runs. From phase B, a **change summary** says what changed, and confirmed subscribers can get an **email** about it.

A scheduled run is an ordinary `POST /api/task` in Always mode, sent by a small separate **scheduler** process. So the security guard, the time limit, the token budget, the search cap, the collection and the metrics apply to it as to any task. The scheduler has its own secret, never given to the web app. Each file has one writer: the backend writes definitions and the email ledger, and the scheduler writes schedule state, run history and usage. Spending is bounded by a monthly allowance that a run can't start without.

## Documents

| File | What it holds |
|---|---|
| [`requirements.md`](requirements.md) | REC-100 – REC-139 and RNFR-01 – RNFR-10, with acceptance criteria and checks |
| [`design-high-level.md`](design-high-level.md) | Components, flows, data layout, failure handling, deployment, security, limits |
| [`design-low-level-backend.md`](design-low-level-backend.md) | Modules, data models, signatures, endpoints, scheduler loop, schedule maths, metrics, config, startup checks, email, tests |
| [`design-low-level-webapp.md`](design-low-level-webapp.md) | Pages, components, proxies, middleware, the task proxy allow-list, tests |
| [`tasks.md`](tasks.md) | Tasks per phase in dependency order, with coverage |

Requirements say *what*; the designs say *how*; tasks say *in what order*. When documents disagree, `requirements.md` wins, then the designs, then the plan.

## Scope

**In scope:** the plan's §6–§9. That's topics, the scheduler, scheduled runs through the task API, run history, the source diff, change summaries, SMTP email with double opt-in and unsubscribe, the caps and the monthly allowance, the web pages, metrics and alerts, deployment, and the per-task search cap and usage accounting for every run.

**Out of scope:**
- user accounts and per-user privacy;
- hourly schedules;
- notification channels other than email;
- HTML email;
- paid or premium sources;
- monitoring a single page;
- changes to what the research agents do.

## Decisions (accepted 2026-10-18; plan §5 and §11)

| ID | Decision |
|---|---|
| D1 | The scheduler is a separate one-process service from the same image: a Compose service, and a sidecar in the backend pod on Kubernetes (the data volume is ReadWriteOnce) |
| D2 | Topics are instance-wide: anyone signed in can list, create, pause and delete them, until user accounts exist |
| D3 | Email goes over provider-neutral SMTP, with a `log` transport in dev and tests. **Open:** the provider and From address, needed before phase B is deployed |
| D4 | Emails carry a notice only while `security_guard.input_mode` isn't `act`; summary emails come only in act mode, enforced at startup |
| D5 | At most 10 active topics and 20 scheduled runs a day, daily at most. The monthly allowance for scheduled runs is 400 Tavily credits and $5 of Together |
| D6 | Topics notify on changes by default, with `always` as an option |
| D7 | Each topic keeps its last 20 runs; topics paused for 90 days are deleted |
| D8 | Build now; deploy after shadow week 2, with the scheduler disabled until then |
| D9 | `max_searches_per_task` becomes a real per-task cap for every run, set from shadow week 2's search counts |

## Findings from the code that shaped the spec

1. **The search cap is per pass.** `web_research.py`'s `capped_search` counts within one research pass, which makes at most 2 searches, so `max_searches_per_task` never binds. A run makes at most `max_follow_up_questions` passes (3 in dev and uat, 5 in prod), so it can make at most 6 or 10 searches, for at most 9 or 15 Tavily credits.
2. **Parallel research works on deep copies of the state** (`run_research_agents`, `state.model_copy(deep=True)`), and the merge copies back only reports, logs, history and the security ledger. A per-task counter therefore can't live in `state.context`. It goes in the run's context-variable counter in `task_budget`, whose mutable object survives `asyncio.gather` (REC-109).
3. **The web proxy forwards the browser's raw body and adds `X-API-Key`** (`src/app/api/task/route.ts`). The API key can't identify the scheduler, so the scheduler gets its own secret (REC-107).
4. **The backend runs two uvicorn workers** (`WEB_CONCURRENCY=2`), and each runs the daily cleanups. Its files are changed under `fcntl.flock`, reusing `app/core/drift_store.py`'s helper (REC-101).
5. **The token counter keeps only `total_tokens`** (`together_llm.py`), although the reply carries `prompt_tokens` and `completion_tokens`. Pricing needs both (REC-109).
6. **Saved results don't hold the supervisor's standalone question** (REC-110).
7. **Triage shadow collection records Always-mode questions** (`task.py`, `triage_collect.record("shadow", …)`), so scheduled runs must skip it (REC-108).
8. **Embedding drift observes every vector-store query** (`DriftRecordingEmbeddings`), so scheduled runs must skip it (REC-108).
9. **Importing `app.core.config` and `app.services.result_store` takes 45 MB and loads no ML library,** so the scheduler can stay small (RNFR-02).
10. **In shadow mode, checkpoint 1 keeps no per-item record of what it would have dropped** (`hop_guard.check_hop`: the effective outcome is `allow`, so `dropped` stays empty). Notice emails, which are sent exactly in shadow mode, need that record to leave out flagged sources' domains, so it's added as `security.shadow_flagged` (task B0, REC-135).
11. **The backend's request-size limit is 10 MB,** and image attachments travel in the task body, so the web proxy's allow-list parsing keeps that limit.

## Traceability

| Plan section | Requirements |
|---|---|
| 6.1 Topics | REC-100 – REC-103, REC-114, REC-115 |
| 6.2 Scheduler | REC-104 – REC-107, RNFR-01, RNFR-02 |
| 6.3 Backend changes | REC-107 – REC-110 |
| 6.4 History and summaries | REC-111, REC-112, REC-130, REC-131 |
| 6.5 Web app | REC-116 |
| 6.6 Email | REC-132 – REC-138 |
| 6.7 Security | REC-103, REC-107, REC-133 – REC-135, RNFR-06 |
| 6.8 Cost controls | REC-109, REC-113, REC-139, RNFR-04 |
| 6.9 Privacy | REC-114, REC-115, RNFR-03 |
| 6.10 Observability | REC-117, REC-139 |
| 6.11 Configuration, deployment | REC-118 |
| 8 Phases | `tasks.md` |
| 9 Tests | each requirement's check; `design-low-level-*.md` §Tests |

## Implementation notes (phase A, 3.11.0)

- **Built** by four parallel agents on a foundation of config, topic files, models and schedule maths, then integrated and tested together:
  - the topics API (A2, A8);
  - the task path (A3, A4);
  - the scheduler with history, holds, allowance and metrics (A6, A7, A8, A10);
  - the web app (A9).
- **Tests:** 1,172 backend and 412 web tests pass, up from 961 and 329.
- **End-to-end check** (2026-10-18), in containers: backend in uat with 2 workers, the scheduler, the repo's fake Together server, no Tavily key, so no credit was spent.
  - **The loop:** a topic created through the API, then Run now. The scheduler started the task with its fixed ID, the backend ran it (`origin: schedule`, `route: agentic`, `decided_by: schedule`, usage recorded), and the run was recorded as completed and `first`. A second run compared against it.
  - **Separation:** triage collected nothing, collection rows carried `topic_create` and `schedule`, and the topic metrics were exported.
  - **Delete:** removed the topic's folder and its runs' results; the topic answers 404.
  - **Web app:** the topics page and proxies worked against that backend. A scheduler field sent through the proxy was dropped, and `summarize` isn't reachable from the web.
- **Measured:** the scheduler uses 39 MiB of resident memory (target under 150 MB).
- **Found by the end-to-end check and fixed:**
  1. **A restart refused to start** while the killed scheduler's heartbeat was still fresh, which would make a crash loop after any crash or redeploy. It now waits for the heartbeat to go stale (REC-105).
  2. **Every research task was counted twice** in `webagent_tasks_total` and `webagent_task_duration_seconds`: the team manager also recorded a completion, with its own step's duration. This existed before 3.11.0; the new `origin` label made it visible. The team manager's calls are removed.
  3. **A deleted topic's tombstone kept its question, name and addresses** until the daily cleanup. The tombstone is now scrubbed at once (REC-114).
  4. **Scheduled results had no `route`;** they now record `agentic` and `schedule` (REC-108).
- **Searches without a key:** a search is counted when it's attempted, so a failed search (here, no Tavily key) also counts toward the cap and the credits. That's conservative for the allowance.
- **Still open:**
  - A0: setting the search cap from shadow week 2's counts. Until then the cap stays at today's values (5 in dev and uat, 10 in prod), which a run can't reach anyway (at most 2 searches per pass).
  - A11's trial week on the local stack, after shadow week 2.

