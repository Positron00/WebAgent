# Spec: platform improvements (3.2.7 – 3.4.0)

Status: draft v1 · Source plan: [`planning/improvement-plan.md`](../../improvement-plan.md) · Base: `main` at v3.2.6 (`741368a`) · 2026-10-14

## What this work does

The 3.2.6 review found gaps around the parts that work well. This spec turns the plan's
first three releases into requirements, designs and tasks:

- **3.2.7, safety net and open doors.** CI; an access code on the web app; rate limits
  that can't be dodged; the dev backend kept off the network; per-agent settings that
  actually apply; an overall time limit on runs; and the plan's quick fixes.
- **3.3.0, hardening, privacy and long runs.** Cancel and live progress for research
  runs; a per-task token budget; no question text in logs; result files that expire;
  security headers and smaller open endpoints; honest docs.
- **3.4.0, cleanup.** Dead code removed; dependencies split; more tests (coverage, red
  team, one browser-to-backend run); contributor and ops docs.

## Documents

| File | What it holds |
|---|---|
| [`requirements.md`](requirements.md) | What must be true at each release. Numbered requirements (`IMP-…`, `IMP-NFR-…`) with acceptance criteria and how each is checked |
| [`design-high-level.md`](design-high-level.md) | Components, request flows (sign-in, rate limiting, cancel, progress, limits), failure handling, data, deployment, key decisions, known limitations |
| [`design-low-level-backend.md`](design-low-level-backend.md) | Python modules, function signatures, config, endpoint changes, cleanup list |
| [`design-low-level-webapp.md`](design-low-level-webapp.md) | Middleware, sign-in route and page, proxy, rate limiter, `apiClient`, UI changes |
| [`design-low-level-ops.md`](design-low-level-ops.md) | CI workflow, git hooks, Python lint, Docker and Compose, dependency split, docs |
| [`tasks.md`](tasks.md) | Work in three phases (one per release). Each task lists the requirements it covers and the check that proves it |

Read in that order. Requirements say *what*; the designs say *how*; tasks say *in what
order*. When documents disagree, `requirements.md` wins, then the designs, then the plan.
Each requirement names the plan item it comes from (for example "plan Q1"), so the plan
and this spec can be read side by side.

## Scope

In scope: every plan item placed in releases 3.2.7, 3.3.0 and 3.4.0.

Out of scope (the plan's "Later" row and items owned by other specs):

- C3 (persistent graph database), H5 (one config loader), H6 (splitting large files
  beyond what tasks touch), H7 (typed agent interface), T8 (load test), P1–P4 (product
  features).
- C6 and C7 (checkpointer, more replicas): dropped by decision 5.
- Security guard and triage `act` mode, the judge, and the output checkpoint: owned by
  [`agentic-security`](../agentic-security/README.md) and
  [`agentic-triage`](../agentic-triage/README.md).
- Changes to what the agents research or how they write reports, apart from applying
  the settings the YAML files already contain (IMP-030).

## Decisions

Decisions 1–5 were made with the plan on 2026-10-13; 6–12 were made while writing this
spec.

1. **Web app access: a shared access code** in a signed, HTTP-only cookie. Not Vercel
   password protection (plan-dependent, Vercel only) and not user accounts.
2. **Unused observability code: delete it** (everything in the plan's H1 table except
   `document_ingestion_service.py`).
3. **Task state: keep JSON files**, add retention. No Redis.
4. **User questions:** not in logs; result files deleted after 30 days; guard and
   triage collections unchanged.
5. **One backend replica by design.** No checkpointer, no shared volume.
6. **Rate limits key on a sign-in session, not an IP address.** The access-code cookie
   carries a random session ID. The web app forwards it to the backend as
   `X-WebAgent-Client`, and both rate limiters key on it. The backend honours that header
   from a trusted proxy address or from a caller holding the API key, which covers the
   web app on Vercel (its addresses change). The web app stops copying the caller's
   `X-Forwarded-For`, which closes plan item S13 without guessing which proxy headers
   can be trusted.
7. **Fail closed in production.** A production web app with no access code configured
   answers 503 on every API route, unless `WEBAGENT_ACCESS_CODE_DISABLED=true` says
   another layer (for example a company proxy) does sign-in.
8. **Cancel through a marker file.** With 2 workers, a cancel request may reach the
   worker that isn't running the task. The request writes `results/<id>.cancel` on the
   shared data folder; the running worker checks for it every 2 seconds.
9. **Progress through LangGraph streaming.** `run_workflow` switches from `ainvoke` to
   `astream(stream_mode=["updates", "values"])` (supported by the pinned 0.2.76), and
   saves the current step and the last few log lines to the task's result file after
   each node. No new storage.
10. **Agent temperature applies only when the YAML sets it.** `AgentConfig` defaults
    `temperature` to 0.2, so applying it blindly would change agents that have no
    setting. Every Together agent has really been running at the global `llm.temperature`
    (0.7 in dev, 0.5 in uat, 0.3 in prod; `config/dev.yaml:39`) instead of its own 0.0–0.4,
    because `get_together_llm` passes only the global value (`services/llm.py:737-742`).
    Applying the agent values changes answers, so the change is checked with a
    before-and-after run on fixed questions (IMP-NFR-07).
11. **`/api/v1/chat/completions` stays** (the triage spec's REQ-070 gives it a `mode`
    field for API clients) **and gets the intake guard** that `/api/task` runs.
12. **CI checks out without Git LFS files.** The encoders and data total over 200 MB,
    which would use up GitHub's free LFS bandwidth within days. Tests that need real
    weights are marked and skipped in CI; a weekly job (or a manual run) covers them.

## Findings from the code that shaped the spec

These came from reading the code while writing the designs. The plan has been updated
where they change it.

1. **Timeouts exist but are long.** aiohttp's default is 300 s
   (`ClientTimeout(total=300)`) and the Tavily client sets 100 s on its own request. So
   a stuck call waits minutes, not forever, as the plan first said. The real risk is the
   sum: up to 8 model calls and several searches with no overall limit. The overall run
   limit (IMP-041) matters more than the per-call fixes (IMP-040).
2. **The web app's rate limiter is one bucket for the whole server** (10 requests a
   minute, `src/config/chat.ts:153`), not one per user. Fixing only the missing
   `addRequest()` call would let one person block everyone. Hence decision 6.
3. **Compose publishes 8000 (backend), 9090 (Prometheus) and 3001 (Grafana) on every
   network interface** (`backend/docker-compose.yml:19,50,69`), and Compose defaults to
   `WEBAGENT_ENV=dev`, where API keys are off. IMP-014 covers Compose as well as
   `python -m backend.main`.
4. **The web app's `/api/health` returns memory use, API metrics and log statistics**
   (`src/app/api/health/route.ts`). It must stay open for health checks, so it shrinks
   instead (IMP-063).
5. **Token counts already pass through one place**, `TogetherWithUsage._result`
   (`backend/app/services/together_llm.py:64-69`), so the per-task budget needs no
   changes in the agents. Calls through other providers (OpenAI for coding) aren't
   counted; see known limitations.
6. **Tasks run as FastAPI `BackgroundTasks`**, which give no handle to cancel. Runs
   move to `asyncio.create_task`, kept in the task manager, so a run can be cancelled
   and timed.
7. **The Python environment has a stale LangGraph 0.3.5 `dist-info`** next to the
   pinned 0.2.76 (0.2.76 is what loads). A clean CI install catches this kind of drift.

## Traceability

Every requirement names its plan item and its check. Every task in `tasks.md` names the
requirements it satisfies, and the coverage table at the end of `tasks.md` shows each
requirement has at least one task.
