# Changelog

All notable changes to the WebAgent Platform will be documented in this file.

## 3.13.0 - 2026-10-21

Tools for the shadow week, and a browser-to-backend test suite, which found three bugs.

### Added
- **Shadow-week report.** `python backend/scripts/shadow_week_report.py --since <date>` gathers everything the shadow week decides in one place. It covers checkpoint 1 and 2 scores against the thresholds with false-positive bounds, how often each shadow intent rule fires, the shadow classifier, searches per task, tokens per run with suggested budgets, triage, the Prometheus figures for the window, and a checklist. It prints counts and scores only, never the text of a question or a page. It suggests a value only when there are enough rows. The runbook's step 0 runs it.
- **End-to-end tests (Playwright).** `e2e/` drives Chromium against the built web app, the backend and the scheduler, using the fake Together server and a local SMTP sink. The tests cover sign-in, a research answer, **Track this**, **Run now** through the scheduler, the stored report, and email subscribe, confirm, notice and one-click unsubscribe. `scripts/e2e/stack.sh` starts and stops the stack, and the new `e2e` CI job runs the suite. See `docs/TESTING.md`.
- **`WEBAGENT_EMAIL_ENABLED`, `WEBAGENT_EMAIL_TRANSPORT`, `WEBAGENT_EMAIL_MODE`** override `notifications.email` without editing the YAML, like `WEBAGENT_SCHEDULER_ENABLED`. Compose passes them from `.env.local`. The startup checks still apply.

### Fixed
- **A topic's stored reports never opened.** The run record keeps the report as a Markdown string, as the spec says, but the web app read only an object form and showed "This run has no stored report." It now reads the string. A test uses the record exactly as the scheduler writes it.
- **The operator's default agentic mode didn't apply after first sign-in.** The app asked for `/api/settings` on the sign-in page, got 401, settled on Off and never asked again. Now it skips the sign-in and email-link pages, treats 401 as "not yet", and asks again on the next page.
- **Screen readers could announce "You are currently offline" at any time.** The banner was always in the page, hidden off-screen, with `role="alert"`. It's now rendered only while the app is offline.
- **ARCHITECTURE.md** describes collected security rows correctly: scores, decisions, URLs and tool arguments (search queries, code), with user text kept only as a hash outside dev.

## 3.12.1 - 2026-10-18

A check of 3.11.0 and 3.12.0's code against the recurring-research spec and plan.

### Fixed
- **Topic email couldn't be turned on in a deployment.** Neither the Compose file nor the Kubernetes manifest passed `WEBAGENT_PUBLIC_URL` or the `WEBAGENT_SMTP_*` settings to the backend, so with email enabled, the backend's startup check refused to start. Compose now passes them from `.env.local`. On Kubernetes they're optional keys in `webagent-secrets`, which `deploy.sh` fills from the environment. A test checks that every secret key the manifest reads is filled by `deploy.sh`.

### Changed
- **The spec and plan describe what was built:**
  - the scheduler lock waits out a dead holder instead of exiting, and the lock's fields are written down;
  - the run record's fields, the scheduler's error kinds, `webagent_scheduler_enabled`, and skips as `outcome="skipped"`;
  - the notify rules, reasons and deduplication, and no retry on permanent SMTP refusals;
  - address hashes in tokens, and the 998-character mail policy;
  - the web app's receipts and lookup-before-POST;
  - the current file lists.

## 3.12.0 - 2026-10-18

Recurring research, phase B: **change summaries** and **email** for topics (recurring-research spec, tasks B0–B7). Email is off until an operator sets it up (`docs/RUNBOOK.md`, "Recurring research"); while the security guard is in shadow mode, emails carry a notice only.

### Added
- **What changed:** after a run with a previous one, a model call compares the two reports, with both scored by checkpoint 1 as untrusted input. It writes up to 5 bullets citing sources, or "No material change". It shows on each run in the topic's timeline and counts toward the monthly allowance.
- **Email for topics:**
  - **Subscribing:** an optional address in Track this and on the topic page, up to 5 per topic, with double opt-in.
  - **Confirm and unsubscribe:** pages that change nothing until you press the button, because mail scanners open links. Plus RFC 8058 one-click unsubscribe.
  - **Sending:** plain-text messages over SMTP (STARTTLS or TLS), with caps per run, per address and per day. Notices are sent on changes, or on every run.
  - **Hold and deletion notices:** when a topic is held after failures, and 7 days before a long-paused topic is deleted.
- **Notice and summary modes:** a notice lists only the domains of new sources that checkpoint 1 didn't flag. The summary goes into emails only with `security_guard.input_mode: act`, enforced at startup.
- **`security.shadow_flagged`:** in shadow mode, checkpoint 1 now records the sources it would have dropped. Notices use it.
- **Metrics and alerts:** `webagent_email_sent_total{kind, outcome}`, and a "Scheduled budget 80%" alert rule.

### Security
- **Tokens are redacted in logs:** credential query parameters (`token=`) in the backend's request log, uvicorn's access log and httpx's log. Confirm and unsubscribe tokens carry a keyed hash of the address, not the address.

### Fixed (found by the end-to-end check, before release)
- **`List-Unsubscribe` was MIME-encoded on the wire,** so mail clients couldn't use one-click unsubscribe. Messages now use the SMTP line limit.

### Upgrade notes
- **Config:** `notifications.email` (`enabled`, `transport: log|smtp`, `mode: notice|summary`) and the `WEBAGENT_SMTP_*` and `WEBAGENT_PUBLIC_URL` settings. The backend refuses to start with SMTP enabled and no host, From address or public URL, and with `mode: summary` while `input_mode` isn't `act`.
- **Still to do** before real use: choose a provider and From address, set up SPF and DKIM, and check delivery with your own address.

## 3.11.0 - 2026-10-18

Recurring research, phase A: **topics** that WebAgent researches again daily or weekly, with each run's report and new sources kept (recurring-research spec, `planning/specs/recurring-research/`). Scheduling is **off** in every environment; turn it on after shadow week 2 (`docs/RUNBOOK.md`, "Recurring research"). Email notifications come in 3.12.0.

### Added
- **Topics:**
  - **Track this** under research answers, prefilled with the question the research used.
  - `/topics` lists them, with state, next and last run, Run now, pause and resume, and delete.
  - `/topics/[id]` shows the runs, newest first, with the new sources and the stored report.
  - The backend API is `/api/topics` (API key). A topic's question passes the security guard's checkpoint 1 when it's created.
- **Scheduler** (`python -m app.scheduler`; a `webagent-scheduler` Compose service and a `scheduler` sidecar on Kubernetes):
  - One small process (about 40 MB, no models) that starts due runs through `POST /api/task`, so the guard, the time limit, the token budget and the search cap all apply.
  - **Single runner:** a heartbeat lock, and a restart waits for a dead holder's heartbeat to expire.
  - **Crash safety:** fixed task IDs, so a crash never runs a slot twice.
  - **Catch-up:** missed slots aren't backfilled.
  - **Holds:** after 3 failures in a row.
- **Caps:**
  - 10 active topics, 20 scheduled runs a day, daily at most.
  - A monthly allowance for scheduled runs, 400 Tavily credits and $5 of Together. A run starts only if its worst case fits, so the allowance is never passed.
- **Scheduler credential:** `WEBAGENT_SCHEDULER_TOKEN`, held only by the backend and the scheduler. The API key alone can't send scheduler fields, because the web app's proxy adds it to browser requests.
- **Metrics:**
  - `webagent_topic_*` and `webagent_scheduler_*` from the scheduler's files.
  - A "Recurring research" dashboard row.
  - A "Scheduler down" alert rule.

### Changed
- **Usage on every task:** each task's result records `usage` (input and output tokens, searches, Tavily credits) and, for research runs, the supervisor's `standalone_query`.
- **A real per-task search cap:** `agents.web_research.max_searches_per_task` was counted per research pass, so it never applied. Its value will be set from shadow week 2's search counts; today's (5 in dev and uat, 10 in prod) is above what a run can make.
- **Origin labels:** `webagent_tasks_total` and `webagent_task_duration_seconds` have an `origin` label (`interactive` or `schedule`). Scheduled runs skip triage and its collection and embedding drift. Their results, collection rows and LangSmith runs carry `origin` and `topic_id`.
- **Web proxy allow-list:** the task proxy forwards only the fields the backend knows; invalid JSON is a 400 and an oversized body a 413. The proxy now also passes 204 responses through.
- **The chat's Markdown renderer** is one shared component, used for chat answers and stored reports (no raw HTML).
- **Config:** a `schedule:` and `notifications:` block, and `llm.price_input_per_million` and `llm.price_output_per_million`.

### Fixed
- **Every research task was counted twice** in `webagent_tasks_total`, and its duration recorded twice. The team manager agent also recorded a completion, with its own step's duration.
- **Validation errors with a custom message** (for example an unknown time zone) made the 422 handler fail with a 500.

### Upgrade notes
- Nothing runs on a schedule until `schedule.enabled` and `WEBAGENT_SCHEDULER_TOKEN` are set. Deploy with `WEBAGENT_SCHEDULER_TOKEN` exported to `deploy.sh` when you want scheduling; without it the sidecar idles.
- Task metric queries that filter on labels other than `status` should add `origin` or sum over it. The provided dashboards already sum.

## 3.10.1 - 2026-10-17

All documentation checked against the code and updated, with a few gaps that the check found fixed. The 3.10.0 shadow week keeps running on 3.10.0: nothing here changes what it measures.

### Documentation
- **README rewritten:**
  - a new system diagram (web app, backend, security guard, triage, stores, external services);
  - the research workflow as it's actually built (document extraction is its own branch; research loops through Senior Research; Coding Assistant is followed by Coding Review);
  - a **Using WebAgent** guide (the web app, loading documents, the HTTP API with `X-API-Key`, reading results);
  - **What WebAgent doesn't do:** no scheduled research, saved topics or email notifications; no per-user accounts or private documents;
  - a documentation map.
- **`backend/docs/ARCHITECTURE.md` rewritten.** It now has sections for the security guard (both checkpoints, outcomes, modes, the shadow intent classifier), triage, tasks, stores and observability. Claims the code doesn't support are gone: Axios, circuit breakers, result caching, JSON logs, Llama 3.3 / GPT-4 as the models, and a Kafka roadmap.
- **API reference:** adds the Documents API, `POST /api/guard/input`, and `/api/v1/tasks/{id}/result`, `/error` and `/create`.
- **Runbooks:** a "Local uat with Docker Compose" section, and the shadow-week runbook's one-machine setup uses Compose, so its PromQL queries work.
- **`backend/README.md`:** the correct workflow diagram, project layout and providers. The production server runs on uvicorn workers; gunicorn was never installed.
- **`PRODUCTION_READINESS.md`:** a guard checklist item. The "2.4.0" section claimed JWT authentication, password hashing and JSON logs, which don't exist; it's replaced by what's actually in place.
- **CONTRIBUTING:** what CI runs, and how to change dependencies safely.
- **LOCAL_DEVELOPMENT, the MLflow docs, the demos and Vercel guides:** updated to match.

### Fixed
- **`GET /api/v1/tasks/{id}` and `/error` answered 500** for a task another worker ran, or one from before a restart. They read only the worker's memory. Both now fall back to the saved record, as `/api/task/{id}` already did.
- **`scripts/batch_document_loader.py --registry` logged "Artifact published to registry" and did nothing.** There is no artifact registry; the option is removed.
- **The MLflow server and UI** need the full `mlflow` package, which 3.10.0 replaced with the client. `requirements-demos.txt` now adds it (2.x).

### Changed
- `passlib` and the direct `bcrypt` requirement are removed; nothing imports them. bcrypt stays installed for chromadb.

## 3.10.0 - 2026-10-17

A security refresh of the served package's dependencies, from a scan of the whole codebase. Nothing changes in what the agents do: ContextDefender, triage and the shadow classifier give identical scores on the new versions.

### Security
- **Web app on Next.js 15.5.28 and React 19** (from Next.js 14.2.25 and React 18). 14.2.25 had open advisories up to critical. They include unauthenticated remote code execution through the image optimizer, which the sign-in middleware leaves reachable at `/_next/image`, plus SSRF, cache poisoning and denial of service. `npm audit --omit=dev` now reports none. Next's bundled postcss is overridden to a fixed 8.5 release.
- **Backend dependencies refreshed:** the image's known advisories drop from 344 in 49 packages to 25 in 12. The fixed packages include those on the request path:
  - h11 0.14.0 → 0.16.0 (request smuggling);
  - starlette 0.46 → 1.8 and fastapi 0.115 → 0.143;
  - pypdf 5 → 6 and pillow 10 → 13 (both parse uploads);
  - aiohttp, urllib3, requests, jinja2, cryptography and torch 2.5 → 2.14.

  The remaining 25 need a new major version of LangChain, LangGraph, Chroma, transformers, sentence-transformers or nltk. None is reachable in WebAgent; each is explained in `backend/SECURITY_DEPENDENCIES.md`.
- **mlflow → mlflow-skinny 2.22.5:** the tracking client only. The full package brought a tracking server stack (flask, werkzeug, gitpython, sqlalchemy) that WebAgent never runs.
- **CI audits dependencies:** `pip-audit` on `backend/constraints.txt` and `npm audit --omit=dev --audit-level=high`. A new advisory fails the build.

### Changed
- **`backend/constraints.txt` is regenerated** from a clean `python:3.12-slim` resolve of `requirements-dev.txt`: 204 exact versions, replacing a 540-line freeze of a conda environment. The numeric stack the gated models were calibrated with stays as it was (numpy 1.26, pandas 2, scipy 1.13.1, scikit-learn 1.5.2).
- **CPU torch build in the image:** the Dockerfile installs torch from the CPU wheel index, as CI does. On amd64, PyPI's default would pull the CUDA libraries.
- Checkpoint 2 classifier data: the generated code rows are complete at 750. Generation was stopped there, since more code rows wouldn't help the gate; the next rebuild uses them.
- **ESLint 9 with the flat config:** `npm run lint` runs `eslint .`, since `next lint` is deprecated. The duplicate `.eslintrc.json` is removed.

### Fixed
- **Tracking failures no longer stop the server.** With `MLFLOW_TRACKING_URI` set, a tracking store that couldn't be used made `internal_research.py` raise on import and stopped the server. Now that worker logs a warning and turns tracking off.
- Next.js 15 changes:
  - the two task routes await their `params`;
  - the browser-compatibility warning is lazy-loaded from a client component;
  - `useDismiss` and `Tooltip` accept React 19's ref and element types.

### Upgrade notes
- **Local development:** update the Python environment with `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt` (the root file adds the constraints), and run `npm install`.
- The web app needs Node.js 18.18 or later (CI uses 20).
- Measured in Docker on the Mac: checkpoint 2 p95 is 25.6 ms (28.7 ms on 3.9.0), and two workers at rest use 1.16 GiB (1.20 GiB). The image is 4.0 GB (3.8 GB), mostly the newer torch.

## 3.9.0 - 2026-10-17

The checkpoint 2 classifier, retrained on a larger dataset, runs in shadow. It still fails the gate, so it never decides: its scores are recorded next to phase A's signals for the shadow week (checkpoint2-intent SEC-128, task B11).

### Added
- **Shadow-only classifier (SEC-128).** `scripts/eval_alignment_model.py --install-shadow` installs the best candidate that failed the gate with `"deployment": "shadow"` in its `metadata.json`. It scores every checkpoint 2 call and plan subgoal in parallel with phase A. Its scores go into collection rows (`shadow_classifier_alignment`, `shadow_classifier_risk`) and two new histograms (`webagent_guard_shadow_classifier_alignment{action}`, `webagent_guard_shadow_classifier_risk{action}`). They never reach Cedar, the thresholds or the judge band, `classifier_available` stays false, and a failing shadow model changes nothing.
- The retrained MiniLM-L6 cross-encoder (`ac-2026-10-17-1`, 88 MB, LFS) ships shadow-only in `backend/models/security/alignment/`. On the gate's test split, real aligned calls held are 0.94% (target ≤ 1%, passes). It fails the rest: off-intent recall 0.51, risk recall 0.44, look-alikes held 10%, calibration error 0.062 and 0.108, and p95 of 12.3 ms while the machine was busy (9.2 ms idle).

### Changed
- **Rebuilt classifier dataset:** 45,650 rows (39,291 committed; 6,359 local and git-ignored). New since 3.8.0:
  - Researchy Questions, sharing ASQ's quota evenly with HotpotQA.
  - xLAM and TraceSafe, now read with the Hugging Face token (local only).
  - 2,496 generated planner tasks.
  - The first 388 generated code rows (`webagent_gen_code.jsonl`), scored as the runtime scores them: the code itself, capped at 1,000 characters.

  223 generated tasks whose rewrite drifted from the question are excluded and listed by id.
- The startup check accepts a shadow-only model in uat and prod. An ungated model without the shadow label is still refused.

### Fixed
- The TraceSafe adapter guessed its field names and produced no rows. It now reads the real schema: the mutated trace's user query and its first agent turn.

### Upgrade notes
- Run `git lfs pull` before building the image; otherwise the model folder holds LFS pointer files. The worker then logs the load failure and keeps phase A's signals.
- Each worker loads the shadow model (about 90 MB more per worker). To turn it off, point `security_guard.alignment_model_dir` at an empty folder.

## 3.8.2 - 2026-10-17

### Security
- **The coding assistant bypassed the guard.** Its first attempt at every request was a direct HTTP call to Together with its own copy of the API key. That skipped checkpoint 1, the per-task token budget, the model-call metrics and drift samples, `TOGETHER_API_BASE` and retries (SEC-001). It is removed: every request goes through the guarded model from `services/llm.py`. A new test bans direct model-provider URLs in the agents.

### Fixed
- **The coding agents' guarded model was misconfigured.** With an OpenAI key set, `get_agent_llm("coding_assistant"/"coding_review")` built an OpenAI client but gave it the YAML's Together model name, which OpenAI rejects. The direct call above hid this. The coding agents now use OpenAI only with `provider: openai` in their config, as the agentic-security design says; otherwise they use the configured model (Together, as the direct call did).

### Upgrade notes
- With an OpenAI key set and no `provider: openai` in `agents.coding`, the coding agents now run on the configured Together model and count toward `WEBAGENT_TASK_TOKEN_BUDGET`. To keep OpenAI for them, set `provider: openai` and an OpenAI `model` in that section.

### Changed
- Checkpoint 2 classifier data: Researchy Questions (CDLA-Permissive-2.0, approved 2026-10-17) joins HotpotQA as an ASQ source, the quota split evenly; the next dataset build uses it.

## 3.8.1 - 2026-10-16

A check of 3.7.0 and 3.8.0's code against the checkpoint2-intent spec and plan.

### Fixed
- **Code calls were scored on different text at runtime than in training:** the classifier's training rows for code are the code itself (RedCode), but at runtime it would have read the coding assistant's explanation. A phase B classifier now scores the code (capped at 1,000 characters); phase A still scores the explanation (SEC-107). No model is installed, so nothing changes at runtime today.
- **Benign code rows could not be generated:** `scripts/generate_alignment_calls.py --code` now runs the coding assistant over coding questions (the code is never run) and the dataset builder turns its output into aligned `execute_code` rows, as SEC-122 requires.
- Loading a classifier set torch to one thread for the whole worker, which would also have slowed ContextDefender; it no longer does (the gate measures one thread through the environment).
- The startup error for an ungated classifier named a script flag that doesn't exist.

### Changed
- The spec's requirements, designs and tasks, the plan and the data README now match the code: the classifier's inputs, thresholds and judge band; the row schema (`groups`, `sources`, `cluster`); how each source is paired (look-alikes as self-pairs, PrivacyLens calls, RedCode intents, ASQ actions, saroku's conditions); the threshold rule; the helper modules; the 3.8.0 build's label shares.
- New tests: every intent rule is shadow-only, code calls are scored on the code, generated code rows, the generator's code mode.

## 3.8.0 - 2026-10-16

Phase B of the checkpoint 2 upgrade: everything needed to train, gate and run a small classifier for checkpoint 2's intent checks. **No model ships**: the first candidates failed the gate, so checkpoint 2 keeps 3.7.0's signals and nothing changes at runtime. Spec: `planning/specs/checkpoint2-intent/` (README, phase B notes).

### Upgrade notes
- **New startup check (uat and prod):** if `backend/models/security/alignment/` (`security_guard.alignment_model_dir`) holds a model whose `metadata.json` doesn't record a passed gate, the server refuses to start. The folder is absent in 3.8.0, so nothing changes.
- `signals.score_text` takes an optional `action`; context keys starting with `_` are no longer passed to Cedar.

### Added
- **Runtime** (`app/security/alignment_model.py`): loads a gated cross-encoder or siamese model, scores (intent, call) pairs for alignment and risk above intent, supplies its own thresholds, and sends `verify` decisions in its uncertain band to the judge. Without a gated model, phase A's signals are used.
- **Data** (`data/security/alignment/`, Git LFS): 34,953 (intent, call) rows from the approved sources, built by `scripts/fetch_alignment_sources.py` and `build_alignment_dataset.py` (`--check-data` passes). Rows from gated and no-train sources stay in the git-ignored `local/` folder, because the repository is public. The saroku dataset passed its quality check with conditions (`QUALITY.md`).
- **Training and gate:** `scripts/train_alignment_model.py` (cross-encoder or siamese, isotonic calibration, thresholds from a held-out fold), `eval_alignment_model.py` (the gate) and `eval_alignment_suites.py` (evaluation-only sets, counts only).

### Measured
- **Gate:** none of the three candidates passed. At 95% off-intent recall, the MiniLM-L6 cross-encoder (AUC 0.888) would hold 5.6% of real aligned calls against the gate's 1%; MiniLM-L12 held 7.4%; the siamese baseline held 32.4%. The next attempt needs reviewed real calls from the shadow week.
- **Evaluation-only sets with 3.7.0's signals** (shadow rules counted as acting): AgentDojo 74% held, OS-Harm misuse and injection 87%, AgentHarm 8%; AgentDAM over-holds legitimate web actions (89%).

## 3.7.0 - 2026-10-16

Checkpoint 2 now checks tool calls, and the subgoals agents write, against what the user actually asked, not only against the supervisor's own rewrite. Every new rule is **shadow-only**: it is recorded and never changes what an agent does, until a later release sets its threshold from shadow-week data. Plan: `planning/checkpoint2-upgrade-plan.md`; spec: `planning/specs/checkpoint2-intent/` (phase A).

### Upgrade notes
- **New shadow-only rules** (nine): `rewrite-drift-verify`, `subgoal-harm-decline`, `subgoal-off-intent-decline`, `call-risk-above-intent-decline`, `call-risk-above-intent-verify`, `off-intent-search-verify`, `private-egress-decline`, `undeclared-tool-verify`, `drift-escalate`. Their matches appear in `webagent_guard_shadow_policy_total{policy_id}` and in collection rows' `shadow_policy_ids`.
- **`untainted-search-misaligned` is replaced** by `off-intent-search-verify`, which compares a search with the user's own turns instead of the supervisor's rewrite, tainted or not. Shadow-week reviews that counted the old rule should count the new one.
- **The plan check** (`WEBAGENT_GUARD_PLAN_CHECK`, default true; `security_guard.plan_check` in YAML) decides the supervisor's rewrite and plan questions, and the senior research agent's follow-up questions, against the user's turns. In shadow mode it only records; with `tool_mode: act` its decisions also apply through the existing acting rules (for example `repeated-declines-escalate`).
- **New Prometheus metrics:** `webagent_guard_intent_alignment{action}`, `webagent_guard_risk_above_intent{action}`, `webagent_guard_egress_overlap`; plan decisions count in `webagent_guard_decisions_total` with `action="plan::rewrite"` / `"plan::subgoal"`.
- **Collection rows** (`data/security/collected/`) gain numeric fields (`intent_alignment`, `arg_harm`, `intent_harm`, `risk_above_intent`, `egress_overlap`, `declared`, `low_alignment_calls`, `signals_available`) and plan-check rows; outside dev they hold no intent or subgoal text.
- **Checkpoint 2 takes longer per call:** about 27 ms p95 in the container at 2 CPUs (was a single small encode), because ContextDefender's harm score now runs on each search, retrieval and code call, in parallel with the encoder. Measured on arm64 Docker on a Mac; the server measurement is still owed.

### Added
- **Intent record:** the user's turn and up to three previous user turns, fixed when a research task starts; never agent text (`app/security/intent.py`).
- **Signals per call:** alignment with the intent, harm of the arguments, risk above intent, private text copied into a web search (salted 8-word shingles of retrieved and uploaded text, kept only in memory), whether the plan declared the tool, and a per-task count of off-intent calls (`signals.py`, `egress.py`).
- **Judge prompt:** the checkpoint 2 judge (still off by default) sees the user's request, the agents' subgoals and the last three calls, each inside data markers, so text in a query can't instruct it.
- `scripts/eval_checkpoint2_injecagent.py --untainted`: with the intent rules acting, 98.6% of InjecAgent's direct-harm attacks in untainted tasks would be held at a search threshold of 30 and 95.9% at the shipped 25 (today 10%).
- `scripts/generate_alignment_calls.py` (benign planner calls for the phase B classifier; the 50-question estimate cost $0.05) and `scripts/bench_checkpoint2.py` (checkpoint 2 latency with the real models).

### Fixed
- Cedar reported matched policies in no fixed order, so `policy_ids` and `shadow_policy_ids` could differ between identical decisions; they are now sorted by their place in the policy file.

### Found
- In 2 of 50 benign test questions the supervisor's rewrite replaced the user's goal with an unrelated one (for example a dog-training question rewritten into one about AI chip makers). `rewrite-drift-verify` matched both. This is the kind of drift the new rules exist to catch; it is recorded in the spec.

## 3.6.1 - 2026-10-16

A check of the 3.6.0 code against its spec (`planning/specs/observability-drift/`) and plan.

### Fixed
- **The self-hosted model couldn't answer:** `SelfHostedChatModel` returned a bare message where LangChain expects a chat result, and a reply without token usage raised an error. Both are fixed, so its model calls now also record drift samples (with their call label, and a 429 as a rate limit)
- The Grafana alert rules' runbook link pointed to `docs/RUNBOOK.md#drift-and-alerts`, which didn't exist (the heading carried "(3.6.0)"); the heading is now "Drift and alerts"
- An unreadable drift `state.json` was logged at WARNING on every read; it's logged once until a read succeeds again

### Changed
- The spec, its designs and the plan now match the code (module of the detector table, file sizes, `/health` and `/metrics` provider, `figures` keys, how samples are read); the API reference describes `model_metrics` field by field
- New tests: the self-hosted model's samples (200 and 429), the cost of the embeddings tap with 1,536-dimension vectors, the runbook anchor, and logging an unreadable state once

## 3.6.0 - 2026-10-16

Drift and alerts now come from real traffic. Before, the observability API reported drift from simulated calls that nothing made, each worker kept its own numbers, and its alerts recommended retraining a hosted model we can't retrain. Plan: `planning/observability-plan-3.6.0.md`; spec: `planning/specs/observability-drift/`.

### Upgrade notes
- **Drift is computed from real traffic and shared by all workers.** Samples and state live in `/app/data/drift/` (on the data volume); sample files are deleted after `WEBAGENT_RESULTS_RETENTION_DAYS`. A fresh install takes its reference from the first 50 samples per detector, then needs 30 more before a detector scores (80 after a reset); until then the dashboard shows "not enough data" and a score of 0.
- `/observability/drift` adds `computed_at` and `detectors` (per detector: `group`, `kind`, `enough_data`, `score`, `drift`, `reference_count`, `current_count`, `reference_start`, `last_drift_at`). `retrain_recommendations` stays but is always `[]`; the per-service blocks report `performance_drift: null` / `doc_quality_drift: null` and never recommend retraining. `/alerts` adds `name` and `since` (`timestamp` kept, equal to `since`); severities are now `warning` and `critical`. `/health` takes its status from the alerts and adds `services`; `alert_counts` adds `critical` and `warning` (`high`/`medium`/`low` kept). `/metrics` fills `model_metrics` and `document_metrics` from the drift store. No field was removed.
- New Prometheus metrics: `webagent_alert_active{alert,severity}`, `webagent_drift_detected`, `webagent_drift_enough_data`, `webagent_drift_current_samples` (by `detector`) and `webagent_drift_samples_total`. `data_drift_score` is now one series per detector across workers (multiprocess `max`), 0 without enough data.
- The drift dashboard starts with an Alerts row; its gauges turn red at 0.95 (KS and chi-square scores) and 0.7 (embeddings) instead of 3.5.3's wrong 0.05 / 0.3 / 0.15; "performance drift" became "model failure share, last hour".
- Grafana provisions six alert rules (folder "WebAgent", `backend/grafana/provisioning/alerting/`). It sends no notifications until you add a contact point.
- **Kubernetes:** every `deploy.sh` run re-applies the Grafana data source, rebuilds the dashboard ConfigMap from `backend/grafana/dashboards/`, applies the alert-rule ConfigMap and restarts Grafana. The old inline dashboard in `monitoring/grafana.yaml` is gone.
- New settings (all optional): `WEBAGENT_DRIFT_ENABLED` (true), `WEBAGENT_DRIFT_WINDOW` (50, minimum 30), `WEBAGENT_DRIFT_INTERVAL_SECONDS` (60), `WEBAGENT_TOKEN_ALERT_DAILY` (0 = off). `TOGETHER_API_BASE` is now honoured by the Together client (dev only). Docker Compose passes all five to the backend.
- Removed: `LLMService.generate_response` and `get_embedding`, `DocumentProcessingObservability.process_document` (simulations nobody called), the drift detector classes and registry in `app/core/drift_detection.py` (`get_all_drift_stats`, `should_retrain_any_model`; the demo keeps a copy in `backend/demos/drift_detectors.py`), and `_build_llm_without_tags`. `app/services/self_hosted_llm_service.py` moved to `app/services/examples/`.
- `scipy` is declared in `backend/requirements.txt` (it was already installed).

### Added
- **Drift from real traffic** (`app/core/drift_store.py`): every model call adds its speed (seconds per 1,000 completion tokens), tokens and outcome under its call label (the agent's name, `judge`, `triage` or `other`); every document load adds its time, size, file type and outcome; embeddings are added as sums of at least 30 vectors. No text, prompt or single vector is stored. One worker per tick computes KS, chi-square and centroid scores against a frozen reference and writes `state.json`; every worker sets its gauges from it.
- **Six alerts:** `model_drift`, `model_errors` (over 5% of at least 20 requests in an hour, rate limits not counted), `rate_limited` (over 20%), `token_usage` (over `WEBAGENT_TOKEN_ALERT_DAILY` in 24 hours), `document_drift`, `document_errors` (over 10% of at least 10 loads in 24 hours). Each shows on the dashboard and in Grafana's alert list, and is logged once when it starts. `docs/RUNBOOK.md` "Drift and alerts" says what each means and what to check
- `scripts/fake_together.py`: a dev-only fake Together server (slow, failing or rate-limited on request) for checking drift and alerts without spending credit
- Uploads now feed `document_processing_latency_seconds`, so the dashboard's "Document processing time (p95)" panel has data

### Fixed
- The self-hosted model client failed to build ("object has no field base_url"): its settings weren't declared as model fields
- Together 4xx and 5xx errors now carry their HTTP status, so a 429 is counted as a rate limit and not as an error

### Changed
- Docs: RUNBOOK, API reference (observability fields), backend README (settings), production checklist (drift and alerts), `.env.local.example`; the platform-improvements spec closes task 4.4 (IMP-128)
- New tests: scoring, the store (reference freeze, reset, two workers, restart, retention), alerts, feeding from model calls, uploads and embeddings, privacy (no text in `data/drift/`), the API contract, multiprocess gauges, dashboards and alert rules, `deploy.sh`, and the removals

## 3.5.3 - 2026-10-15

Checking the Grafana 11.4 dashboards on the local stack (spec task 3.14) found that they had never worked there, and that model calls weren't being measured.

### Upgrade notes
- **Grafana data source uid is now `prometheus`** in both the Compose and Kubernetes provisioning. An existing Grafana volume keeps working (provisioning updates the data source by name); dashboards made by hand that point at the old generated uid need it changed.
- `backend/grafana/model-drift-dashboard.json` moved to `backend/grafana/dashboards/`.
- Model-call metrics change meaning: `ml_model_latency_seconds{task_type="chat"}` and `ml_model_predictions_total` now count real model requests. Before, they timed building the LLM client.

### Fixed
- **The local stack showed no dashboards:** Compose never mounted `backend/grafana/dashboards`, which the dashboard provider reads, so Grafana logged "Cannot read directory" every 10 seconds. It's mounted now, with the drift dashboard moved into it
- **The drift dashboard failed on every panel:** it pointed at a data source uid that didn't exist and queried metric names nothing exports (`data_drift`, `model_latency`, `model_token_usage_*`, `document_processing_time`). It now shows drift scores per detector (gauges turn red at each detector's threshold), model latency p95, model errors and tokens per minute, and document processing time, from metrics the backend exports
- **Model calls weren't measured:** `llm_requests_total`, `llm_request_duration_seconds` and `llm_tokens_total` were never written (their decorator wasn't used anywhere), so the backend dashboard's model panels were always empty; and the "model latency" that was recorded timed building the LLM client, observed twice per call. Each Together or self-hosted request now records all of them once (`app/services/llm_call_metrics.py`)
- **Together rate limits failed the request at once:** a 429 was reported as "invalid payload" and not retried, so a busy moment at Together failed a question. It's reported as a rate limit and retried with back-off, like 5xx errors
- **Kubernetes Grafana couldn't reach Prometheus:** its data source URL had no port, and the `prometheus-server` Service listens on 9090
- Compose no longer installs `grafana-piechart-panel`, an Angular plugin Grafana 11 refuses to load and no dashboard used

### Changed
- `PRODUCTION_READINESS.md` records the spending caps: Together $25 in total, Tavily Researcher plan (1,000 credits a month)
- `docs/RUNBOOK.md` describes the two dashboards and where their data comes from; the platform-improvements spec closes tasks 1.11 and 3.14 and notes how 1.18, 2.10 and 2.24 will be done
- New tests: dashboards query only exported metrics through the provisioned data source, each model call records each metric once, client set-up isn't counted as a call, and rate limits are retried

## 3.5.2 - 2026-10-15

### Fixed
- CI still failed after 3.5.1 (8 backend tests). The earlier key errors had hidden a second missing piece: the document agent loads NLTK's stopword list when it's built, the runner has no NLTK data, and the tests forbid downloading it. Both CI workflows now install the same NLTK data as the Docker image before the tests run. 3.5.1's note that CI was green was written before CI had run; it passes from this release

### Changed
- `CONTRIBUTING.md` and `docs/TESTING.md`: a fresh checkout needs the NLTK data once (`python -m nltk.downloader …`), and the tests supply placeholder provider keys when `.env.local` has none

## 3.5.1 - 2026-10-15

The first CI runs on GitHub failed in the backend job (31 tests); the web job passed. Every failure came from the runner having no `.env.local`, no Git LFS files and no downloaded models, which a developer machine has.

### Fixed
- 26 tests failed with "Unable to initialize any LLM provider": building an agent needs a provider key. The test setup now supplies placeholder Together, OpenAI and Tavily keys when none are set; they never reach a provider (the suite blocks outbound connections), and a real `.env.local` still wins
- 5 startup-check tests stopped at "ContextDefender base encoder is missing" before reaching what they test. The production-settings tests treat the encoder as present (the guard's file checks keep their own tests), and the command-line test runs with the guard off
- The local code sandbox could end a runaway script through its CPU limit before the wall-clock timeout, and then returned normally instead of raising a timeout (seen on Linux). A CPU-limit stop (`SIGXCPU`) is now reported as a timeout too, the same as the wall-clock one, and has its own test

### Changed
- Version 3.5.1 in the docs and manifests

## 3.5.0 - 2026-10-14

Advanced observability repair and integration (IMP-128), plus CSP-aligned web app connectivity and research log merging.

### Upgrade notes
- **`/api/v1/observability/*`** is available behind the API key (`metrics`, `drift`, `health`, `alerts`, LLM/doc baseline reset). Prometheus **`/metrics`** and LangSmith tracing are unchanged.
- Document drift for the observability API lives in **`document_ingestion_service.py`** (`doc_ingestion_service` alias); the standalone **`doc_ingestion_service.py`** module is removed.
- **Web app:** API calls use same-origin **`/api`** only (`clientConfig`); offline detection uses **`/api/ping`** only (no third-party connectivity probes under CSP).
- **Deprecated `POST /api/v1/chat/` and `POST /api/v1/tasks/create`** now run the intake guard (they can answer 403 or 409 like `/api/task`), reject questions over 20,000 characters (422), and ignore everything in `context` except `document_id`. Move to `POST /api/task`.
- **CHANGELOG** entries before 3.0 are in `docs/CHANGELOG-archive.md`.

### Added
- Observability bridge helpers on `app.core.observability` (`record_model_latency`, `get_all_metrics`, …)
- `backend/standalone/llm_server.py` as the self-hosted LLM Docker entrypoint
- `test_observability_api.py`; full-app import scan with no observability `xfail`

### Changed
- `api/routes/observability` mounted on the key-protected `/api/v1` router
- Production LLM usage records latency/tokens into the observability registries via `llm.py`
- JWT/PyJWT config and diagnostics leftovers removed (API-key auth only)
- `docs/RUNBOOK.md` advanced observability section updated for operators
- Platform-improvements specs and `planning/improvement-plan.md` aligned for IMP-128 vs IMP-080
- **Docs aligned with the code** (alignment review): the README no longer claims JWT, an artifact registry, table and image extraction, hierarchical documents, parallel execution of every agent or state persistence, and labels the model gateway optional; `backend/README.md` and `docs/api_reference.md` cover cancel, progress, `error_code`, `tokens_used`, task-list paging, the error object, 415 uploads and the observability API, and no longer mention dev keys; `PRODUCTION_READINESS.md` says one replica by design, `Recreate` updates and which tests exist; every version reference says 3.5.0; `.env.local.example` lists the token budget, result retention, embeddings provider, LDA size and rate-limit settings
- CHANGELOG entries before 3.0 moved to `docs/CHANGELOG-archive.md`; `VERSION_2_4_*.md` moved to `docs/archive/` (IMP-121)
- Spec status: `planning/specs/platform-improvements/tasks.md` lists the manual checks still open (branch protection, spending caps, uat comparisons and drills, token-budget measurement) and moves the Playwright test (IMP-113) to the next release. The shadow-week runbook targets 3.5.0 and records that the first week was stopped after 9 tasks

### Fixed
- Parallel research agents merge workflow **log lines** back into the run (`workflows.py`, IMP-043 follow-up)
- **Deprecated task routes:** `POST /api/v1/chat/` and `POST /api/v1/tasks/create` started research runs without the intake guard and passed the caller's `context` into the workflow, so a caller with the API key could claim `consented` or `approved_tools`; `/chat/` also ran outside the run supervisor (no time limit or cancel) and had no length limit. Both now take the `/api/task` path in Always mode; only `context.document_id` is kept (`test_legacy_task_routes.py`)
- **Logs:** internal research logged follow-up questions (derived from the user's question) at INFO, and a failed Tavily search logged its response body, which can echo the query; both now log a length outside dev. The log scan test covers more variable names
- **Browser console:** voice input logged every spoken transcript; the debug logging is removed, and a Jest check fails on console calls that pass user text
- **Self-hosted LLM images:** `llm_server.py` imported the backend package, which the images don't contain, so both failed at start; the MLflow image passed `"${MODEL_PATH}"` and the other settings literally (exec-form `CMD`); the two Dockerfiles assumed different build contexts. The server is standalone now, both images build with `backend/` as the context, and the build commands are in `README_SELF_HOSTED_LLM.md` (`test_llm_images.py`)
- **`neo4j`** was unpinned, so a fresh install took 6.x; it's pinned to 5.28.6 in `constraints.txt` and capped below 6
- Unused JWT settings (`WEBAGENT_SECURITY_ALGORITHM`, `WEBAGENT_SECURITY_TOKEN_EXPIRE_MINUTES`) removed from `config.py`
- **CSP:** `AppContext` no longer fetches `google.com` on ping failure; `clientConfig` no longer targets `NEXT_PUBLIC_VERCEL_URL` for `/api`

## 3.4.0 - 2026-10-14

Cleanup, reliability fixes, and contributor docs. Advanced observability code is kept for repair in 3.5.0 (IMP-128).

### Upgrade notes
- **API errors:** Most backend error JSON now includes `"error": {"code": "...", "message": "..."}`. `HTTPException` and validation responses still include `detail` when they did before. Rate limit (429), request too large (413), and the global 500 handler use the new shape instead of a plain `"error": "string"`.
- **Embeddings:** `WEBAGENT_EMBEDDINGS_PROVIDER` (`openai` or `local`) selects the embedding implementation in `vectordb.py`. Rebuild or re-index the vector store if you switch providers.
- **Python deps:** Root `requirements.txt` installs runtime plus dev tools via `backend/requirements-dev.txt`. Demo-only packages moved to `backend/requirements-demos.txt`. Removed: `py2neo` (now the `neo4j` driver), `PyPDF2` (now `pypdf`), `platform-utils`, `redis`, `python-jose` and `PyJWT`; code importing them must switch.

### Added
- Unified `ApiError` and storage quota handling in the web app (`QuotaExceededError` trims oldest chat turns and retries)
- `CONTRIBUTING.md`; expanded `docs/RUNBOOK.md` (deploy, rollback, rotation, disk full, probes)
- Backend: `error_body` handlers (`test_error_envelope.py`), graph reachability test, vector search off the event loop, LDA size/pass caps, red-team guard cases, requirements split, CI coverage floors
- `get_embeddings()` entry point and embedding-provider tests

### Changed
- Removed orphan `lda_document_extractor.py` and repo clutter (backup Babel configs, tracked demo outputs, duplicate tests)
- `py2neo` replaced with the `neo4j` driver; PDF extraction uses `pypdf`
- Grafana/Prometheus image versions in Compose and Kubernetes monitoring manifests
- Jest: higher coverage floors, `forceExit` removed
- Spec: IMP-080 narrowed to true duplicates; **IMP-128** schedules observability repair for **3.5.0**

### Fixed
- Unreachable LangGraph nodes removed; document extraction routing simplified
- Duplicate test modules merged; diagnostics tests live under `backend/tests/diagnostics/`

## 3.3.0 - 2026-10-14

Hardening, privacy in logs, cancellable long runs with live progress, token budgets, and ops manifests.

### Upgrade notes
- **Stop a run:** `POST /api/task/{task_id}/cancel` (and the web app **Stop** button). A cancelled run finishes with `error_code: cancelled`; progress and failure panels replace the old silent fallback to `/api/chat`.
- **Token budget:** `WEBAGENT_TASK_TOKEN_BUDGET` defaults to `0` (off). When set, Together usage for a run is counted across parallel agents; overages save `error_code: budget`. Every finished task record includes `tokens_used`.
- **Result retention:** Files under `results/` older than `WEBAGENT_RESULTS_RETENTION_DAYS` (default 30) are removed by a daily cleanup loop started with the server. `DELETE` on a task removes its on-disk record.
- **Logs in uat/prod:** User questions are logged only as lengths (`describe_text`), not full text. Dev still logs the query at DEBUG on agent entry.
- **Kubernetes:** `deploy.sh` applies `networkpolicy.yaml` (ingress from your nginx namespace and from `monitoring`). The Ingress returns 404 for `/metrics` on the public host. Optional `backup-cronjob.yaml` archives `/app/data` when snapshots are unavailable (see `docs/RUNBOOK.md`).
- **OpenAPI:** `/api/v1/docs`, `/api/v1/redoc` and `/api/v1/openapi.json` are off in uat and prod (`test_docs_off.py`). **`GET /api/v1/health/detailed` needs the API key**; `/health` and `/api/v1/health/` stay open.
- **Web app headers:** every page sends a Content Security Policy (`connect-src 'self'`: the browser talks only to the web app's own `/api` routes), `X-Frame-Options: DENY`, `nosniff`, a referrer policy and, in production, HSTS. Remote images load only from hosts in `WEBAGENT_IMAGE_HOSTS` (empty by default). Images sent to `/api/chat` must be PNG, JPEG, GIF or WebP data URLs of at most 5 MB.

### Added
- `result_store`: cancel markers, `save_progress`, `list_results`, `cleanup` / `cleanup_daily`; paginated `GET /api/v1/tasks`
- `progress.py` and LangGraph `astream(["updates", "values"])` so status replies include `progress`
- `task_budget.py` and usage accounting in `TogetherWithUsage`
- `SearchError` and `search_errors` on web research reports
- Web: `taskPolling.ts`, `useResearchRun`, `ResearchStatus`, `ResearchFailed`, cancel proxy route, CSP/image checks, `safeUrl`, `/api/ping` and `/api/health`
- `SECURITY.md`; backend `ruff` lint in CI; `test_hardening.py`, `test_task_progress.py`, `test_task_budget.py`, `test_no_question_in_logs.py`

### Changed
- Removed `DEV_API_KEYS`, JWT helpers, and the unused `redis` dependency
- Self-hosted LLM Dockerfiles run as non-root user `llm` (uid 10001)
- `PRODUCTION_READINESS.md` and the Kubernetes README note per-worker rate limits, retention, and backup options

### Fixed
- Detailed health (`/api/v1/health/detailed`) requires the API key; basic `/health` stays open

## 3.2.7 - 2026-10-14

First release of the platform-improvements spec (`planning/specs/platform-improvements/`): a safety net, the open doors closed, and limits on runs.

### Upgrade notes
- **A production web app needs `WEBAGENT_ACCESS_CODE` (12+ characters) and `WEBAGENT_SESSION_SECRET` (32+ characters)**, or every API route answers 503 `access_code_not_configured`. Set `WEBAGENT_ACCESS_CODE_DISABLED=true` only if another layer does sign-in. Development stays open without them, with a warning
- The dev backend listens on `127.0.0.1`, and `python -m backend.main` refuses another address in dev while API keys are off (`WEBAGENT_ALLOW_OPEN_DEV=true` overrides). Docker Compose publishes the backend, Prometheus and Grafana on `127.0.0.1` only
- Agents now use their own `temperature`, loop limit and search/document caps from `agents:` in the YAML. Before, every Together agent ran at the global `llm.temperature` (0.7 dev, 0.5 uat, 0.3 prod) and Senior Research always stopped after 3 rounds (prod asks for 5). Answers will differ
- A research run is stopped after 7 minutes (`WEBAGENT_TASK_MAX_SECONDS=420`) and reported as `error_code: timeout`; the web app shows "The research took too long and was stopped." instead of falling back to plain chat
- `/api/v1/chat/completions` runs the intake guard and rejects more than 9 messages, messages over 20,000 characters, and roles other than user, assistant and system. `POST /api/task` rejects questions over 20,000 characters
- `RATE_LIMIT_BURST_LIMIT` is gone (it was never applied)

### Added
- Web app sign-in: one shared access code, a signed HTTP-only session cookie (Web Crypto, no new dependency), `/login`, `/api/auth/login` (500 ms minimum answer time; 429 after 20 wrong codes in a minute) and `/api/auth/logout`. Middleware protects every page and API route except `/api/health`, sign-in and static files
- CI: GitHub Actions run lint, type check, Jest with coverage, the web build and the backend suite on every push and pull request to `main` and `release/*`, without Git LFS files; a weekly job runs the tests that need the LFS weights. A pre-commit hook runs `lint-staged`
- Run supervisor (`app/services/run_supervisor.py`): runs start as `asyncio` tasks instead of FastAPI background tasks, and the supervisor enforces the run limit. A stop is saved once and a late result can't overwrite it
- A test imports every backend module

### Fixed
- The web app's rate limiter never counted a request, and it was one bucket for the whole server. `/api/chat` now counts each request against the caller's sign-in session
- Anyone could get a fresh backend rate limit on every request by sending their own `X-Forwarded-For` through the web app. The web app no longer copies browser headers; it sends the session as `X-WebAgent-Client`, which the backend accepts only from trusted proxies or callers holding the API key (so it works on Vercel too). Other proxies' callers are taken from the right-most untrusted `X-Forwarded-For` address
- Web app retries after a timeout failed at once, and after another error ran with no timeout: every attempt now has its own timer
- Async Together calls waited up to 300 s (aiohttp's default) instead of 120 s; each Tavily search is limited to 30 s (`WEBAGENT_SEARCH_TIMEOUT_SECONDS`)
- One failing research agent no longer throws away the other's report; the failure is logged and kept in `research_errors`
- When Senior Research couldn't read the model's evaluation it counted as complete with score 10; it now counts as unknown and asks for more research until the loop limit
- Document IDs are checked as UUIDs (`..` reached the parent folder); search and related-document limits are bounded; an unsupported upload is refused with 415 before it is written to disk
- `harm_judge`, `jailbreak_judge` and `prompt_guard_threshold` in the YAML were dropped when the guard settings were built
- CORS was added twice with `*`, so the YAML lists were ignored; it's added once from the YAML, and uat/prod allow `X-API-Key`. Security headers and the request size limit were also added twice; the size limit now uses `WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT`
- `/api/v1/chat/completions` turned "No user message found" (400) into a 500
- `npm run research` pointed at a script that doesn't exist; `fine_tune_service.py` imported a module that doesn't exist
- Startup and shutdown use FastAPI's `lifespan` (no more `on_event` deprecation warnings)

### Changed
- Removed five unused `lda_*` keys from `config/dev.yaml`
- Docs: README sign-in setup, Vercel guide (sign-in variables, per-session limits), production checklist (run limit, sign-in, spending caps), rate-limit sections of the backend README, API reference and architecture

## 3.2.6 - 2026-10-13

### Fixed
- `deploy.sh -e uat` deployed prod settings: `deployment.yaml` hard-coded `WEBAGENT_ENV=prod`. The script now sets it from `-e` (`prod` or `uat`; `production` still means prod)
- The week's collected data would have been lost on any pod restart: `/app/data` was an `emptyDir`. It's now a 5 GiB persistent volume, and the Deployment uses `Recreate` (a ReadWriteOnce volume can't attach to two pods)
- Secrets went through `sed`, which broke on keys containing `/` or `&`. `deploy.sh` now creates `webagent-secrets` and Grafana's `grafana-credentials` with `kubectl create secret`; no key is written to a file
- Every backend metric would have been counted twice in Kubernetes: two Prometheus jobs scraped the same pod. The general pod job now skips the backend
- On clusters without the Prometheus Operator, `deploy.sh` stopped partway: `prometheus.yaml` included a `ServiceMonitor` and a `PrometheusRule`. They moved to `monitoring/prometheus-operator.yaml`, applied only when the Operator's CRDs exist, and the `ServiceMonitor` now matches the real Service and `/metrics`
- Internal research's reranker downloaded from Hugging Face on the first research task (the pod needed internet access, and it failed where the cache folder belonged to root). It's baked into the image (`HF_HUB_CACHE=/opt/hf-hub`), and `/app/data` in the image belongs to the app user
- `webagent_contextdefender_score` was registered but never recorded; each freshly scored item now adds its four axis scores
- The uat config said triage's shadow week left routing unchanged; that holds only for questions sent with Always. In Auto, triage routes for real and `/api/triage` records nothing. The comment and the runbook now say to keep the web app on Always for the week

### Changed
- `deploy.sh` takes the image repository (`-r`) and ingress host (`-H`), runs from any folder, stops on errors inside pipes, validates its inputs and leaves no rendered manifest behind. CPU limit 2 (request 1) so each worker scores on its own core; a startup probe allows up to 5 minutes for the encoders to load
- Removed `kubernetes/service.yaml`, `ingress.yaml`, `secrets.yaml` and `configmap.yaml`: unused, with another namespace and labels
- Shadow-week runbook section 1 covers running the week on one machine with Docker and in Kubernetes
- Web app headline: the chat's font (Inter, semibold, wide spacing) in soft grey instead of Playfair Display with a gradient; Playfair is no longer loaded
- Web app logo: greyscale and faded, without the dark square behind it, and it pulses while an answer is on its way (not with reduced motion)
- Web app toolbar: Agentic is a menu like Style and Focus, its button showing the current mode and its panel listing the three modes with their hints. Style and Focus show their current values on their buttons, coloured by value, instead of the coloured pills. All four menus close on Escape (focus returns to the button) or on a click outside
- Web app scrollbars are dark and thin, and the page declares a dark colour scheme

## 3.2.5 - 2026-10-13

Fixes from a simulated shadow week (3.2.4 in Docker with the uat settings and real model and search calls, three tasks at a time).

### Fixed
- Concurrent tasks in one worker failed with "Run ... is already active": MLflow's fluent API keeps one active run per thread and every task runs on the event loop's thread. Each task now gets its own run through `MlflowClient` (`app/core/task_tracking.py`), and tracking errors never fail a task
- Security scoring failed with "Already borrowed" when tasks scored at the same time: Hugging Face fast tokenizers aren't thread-safe. Each tokenizer now has a lock (`app/core/model_lock.py`), used by ContextDefender, search alignment and triage
- Every document extraction failed: NLTK 3.9's tokenizers need `punkt_tab`, which the image didn't install. The image and the NLTK helper now include it, and a test checks the image downloads every resource the code asks for
- Some plain questions with Agentic on ended with "Workflow ended without a report": the supervisor planned them as document extraction with every step off. A plan with nothing to do is now researched on the web
- Follow-up research searched for fragments such as "b" and "middle", and for the prompt's own placeholders ("question 1"): follow-up questions were split at every comma. Quoted list items are now kept whole and placeholders dropped
- Internal research no longer reads the task's state from the agent instance, which concurrent tasks overwrote

### Changed
- ContextDefender encodes all chunks of a text in one batch instead of one at a time: 4–7× faster on long pages, same scores
- Kubernetes memory limit 3 GiB, request 2 GiB (was 2 GiB / 1.5 GiB): two workers peaked at 1.81 GiB in the simulation
- Shadow-week runbook: deploy 3.2.5, the new memory targets, and what the simulation found (section 9)

## 3.2.4 - 2026-10-13

### Fixed
- `/metrics` showed one uvicorn worker's values at random (the image runs two): every worker now writes to `PROMETHEUS_MULTIPROC_DIR` (`/tmp/webagent_prometheus`, set in the image and emptied before the workers start) and `/metrics` sums them. Requests in progress count live workers; memory is one series per worker (`pid` label). Without the variable (dev, tests) nothing changes
- On a fresh data volume one worker still crashed at startup, creating MLflow's default experiment at the same moment as the other ("Yaml file ... exists"; uvicorn restarted it). The store is now prepared once by the startup checks, before the workers start, and the import-time lookup retries once
- `/metrics` counted its own scrapes in `http_requests_total`

## 3.2.3 - 2026-10-13

### Changed
- uat runs triage's pending shadow week alongside the security guard's: `triage.enabled: true`, `mode: shadow`, `collect: true` in `backend/config/uat.yaml`. With Agentic on, triage classifies each question and records what it would have done; routing doesn't change. Triage collection stores question text for its review (30-day retention), approved for this week
- Shadow-week runbook covers both: triage metrics, the mix of questions to send, triage's end-of-week review, and rolling either back

## 3.2.2 - 2026-10-13

Getting ready for the security guard's shadow week in uat: what it judged is now recorded and measurable, and there is a runbook for the week and for deciding on act mode.

### Added
- Collection rows say what was judged: checkpoint 1 rows carry the item's source, axis scores, pattern hits and model version (and the URL for web results; text only in dev); checkpoint 2 rows carry alignment, taint and pattern hits
- `webagent_guard_hop_seconds{agent}` and `webagent_guard_tool_seconds{action}`: the whole time each checkpoint adds, for the latency targets (checkpoint 1 < 150 ms p95 per hop, checkpoint 2 < 20 ms per tool call)
- `scripts/review_security_collection.py`: daily summary of collected decisions (counts only), a labelling sheet of untrusted items the guard would have acted on, and the real false-positive rate with a 95% upper bound from the labelled sheet
- Shadow-week runbook: `planning/specs/agentic-security/shadow-week-runbook.md` (deploy checklist, daily queries and targets, review, act-mode criteria, rollback)

### Changed
- uat collects the guard's decisions (`security_guard.collect: true`; 30-day retention); the guard stays in shadow mode and the judge stays off

### Fixed
- `webagent_guard_search_alignment` and `webagent_guard_shadow_policy_total` were registered outside the app's metrics registry, so `/metrics` never showed them

## 3.2.1 - 2026-10-13

Agentic security guard, in shadow mode: it checks every LLM call's context and every tool call, records what it would do, and blocks nothing until checkpoint 1 is switched to act mode. Spec: `planning/specs/agentic-security/`; plan: `planning/agentic-security-plan.md`.

### Added
- Checkpoint 1 (before every LLM call, including the web app's direct chat through `POST /api/guard/input`) and checkpoint 2 (before web search, retrieval, document loading and code execution), deciding through Cedar policies run in-process with `cedarpy`. Outcomes are allow, verify, consent, decline and escalate; consent and approval use single-use signed tokens
- ContextDefender `cd-2026-10-13-2`: scores context on four axes (injection, harm, sensitive data, jailbreak) with `all-MiniLM-L12-v2` fine-tuned on 35,026 rows from public datasets, 96-token windows and calibrated heads. It **passes the release gate**: injection recall 0.973 at 1.4% false positives on benign pages; harm and jailbreak recall 0.931 and 0.961 with at most 0.8% of ordinary questions declined or asked for consent; secret and personal-data recall 0.984 and 0.979
- Per-model policy thresholds chosen on held-out rows with an upper confidence bound and passed into the policy context; borderline harm turns go to OpenAI's moderation endpoint (`harm_judge: moderation`; the judge is off by default), and a refusal or content block from a chat-model judge counts as flagged
- Shadow-only policies (`@shadow("true")`): recorded, never applied. `untainted-search-misaligned` collects how far real searches drift from the task (`webagent_guard_search_alignment`) before any rule acts on it
- Fast patterns for more credential formats (Stripe, SendGrid, Twilio, Hugging Face, npm, PyPI, GitLab, JWT, bearer tokens, credentials in URLs), with placeholders such as `REDACTED` or `${API_KEY}` excluded
- Web app: guard call on `/api/chat` with a pattern fallback, consent dialog, security summary under agentic answers
- Data and evaluation scripts: `fetch_security_sources.py`, `build_security_dataset.py` (licences and attribution in `backend/data/security/README.md`), `train_context_defender.py` (`--finetune`, `--reselect-thresholds`), `eval_context_defender.py` (`--with-judge`), `eval_checkpoint2_injecagent.py`
- The ContextDefender dataset and encoder in Git LFS (`git lfs pull` after cloning)

### Changed
- Every agent LLM call goes through the guarded model factory; `get_trusted_llm()` is limited to the triage tie-breaker, the judges and diagnostics
- `security_guard:` in `backend/config/{dev,uat,prod}.yaml` with `WEBAGENT_GUARD_*` overrides; shadow mode everywhere. Startup requires a passing gate only in act mode, and the model's encoder files in any mode
- Backend image bakes the base `all-MiniLM-L6-v2` again (alignment checks use it); `.dockerignore` excludes local virtualenvs, which had added 1.6 GB to the image. Kubernetes memory budget 2 GiB for two workers
- New dependency: `cedarpy`

### Notes
- Llama Prompt Guard 2 was evaluated as a jailbreak judge and as a second opinion on injection; it isn't enabled or shipped (more consent prompts than jailbreaks caught, and no injection gain)
- Before act mode: confirm false positives on real pages from shadow mode; end to end with the harm judge, 1.23% of ordinary questions are declined (reported, not gated)

## 3.2.0 - 2026-10-11

### Added
- Classifier `triage-2026-10-c`: MiniLM fine-tuned SetFit-style on about 4x the training data (2,496 rows, from 600), with a new `general_knowledge` category. It passes the release gate (155/155 on `test.jsonl`) and gets 39/40 on the new `probe.jsonl`, a second check in a different style. Classification takes about 5 ms
- `train_triage.py --finetune` (alias `--setfit`): contrastive-pair fine-tuning with sentence-transformers' own trainer, no new dependency; the encoder is saved to `models/triage/encoder/`
- Git LFS for the encoder weights (`.gitattributes`); run `git lfs pull` after cloning

### Changed
- The server, eval script and startup check load the fine-tuned encoder from `encoder_path` in `metadata.json`
- Backend Docker image: no longer bakes the base encoder into `/opt/models`; the fine-tuned encoder comes in with the code. The writable-folder step no longer runs a recursive `chown`/`chmod`, which stored a second copy of `/app` in its own layer
- uat and prod still keep triage off until the shadow week and a human review of the training data
- Docs: data and backend READMEs, production checklist, `docs/TESTING.md`, plan (v7) and spec (v5) describe the fine-tuned encoder, `probe.jsonl` and Git LFS; version 3.2.0

### Fixed
- On a fresh data volume, one uvicorn worker died at startup ("Experiment 'rag_agent_performance' already exists") because both workers created the MLflow experiment at once; uvicorn restarted it, so it showed only as a traceback

## 3.1.1 - 2026-10-11

Fixes found by running 3.1.0 for real (real model and search keys, the Docker image, and the web app by hand) before merging.

### Fixed
- Backend direct answers (`mode: off`, or `auto` routed direct) included the model's reasoning ("assistant to=self … We need …"): the prompt now ends where the answer starts, and the reply goes through the reasoning strip shared with web research (`app/utils/reasoning.py`)
- The backend direct path described images it never received; the prompt now says an image was attached but can't be seen
- A plain question that the supervisor planned as document extraction, with no document attached, failed in seconds ("Document ID is required for extraction"); it is now researched
- Sending an image in the web app failed with "did not match any variant of untagged enum MessageContent": `/api/chat` now sends `image_url` as `{url}`, the format Together expects
- `npm run build` failed: `/api/settings` exported helper functions, which Next.js doesn't allow in a route file; they moved to `src/utils/defaultAgenticMode.ts`
- No tooltips on the Agentic mode toggle or the route label in some browsers: a `Tooltip` component replaces the native `title` attribute and shows at once on hover and keyboard focus; the route label is focusable

### Changed
- Kubernetes: memory limit 1.5 GiB (was 1 GiB) and request 1 GiB (was 512 MiB). Two workers measured at 955 MiB with triage off and 1.08 GiB with it on
- Measured and recorded: triage p95 25 ms through the web app, direct answers median 3.4 s (p95 5.1 s), encoder layer 91.7 MB, offline start of the image with triage on

## 3.1.0 - 2026-10-11

### Added
- Agentic triage: the Agentic button becomes Off / Auto / Always. In Auto, `POST /api/triage` (rules plus a local MiniLM + logistic-regression classifier, ~9 ms) picks a direct answer or the research agents; every answer shows "Answered directly" or "Deep research", and direct answers in Auto offer "Research this in depth"
- `mode` (`off` / `auto` / `always`) on `POST /api/task` and `POST /api/v1/chat/completions`; a backend direct path that answers without building the workflow (text only; `image_ignored: true` when a request had an image); `route`, `decided_by` and `p_agentic` on task results
- Agentic requests carry the last 8 messages (`previous_messages`), so follow-ups are researched in context
- Shadow mode, opt-in collection of questions for review (30-day retention), four `webagent_triage_*` metrics and two Grafana panels
- Curated data (`backend/data/triage/`), training and release-gate scripts (`backend/scripts/train_triage.py`, `eval_triage.py`), and classifier `triage-2026-10-a`
- `GET /api/settings` serves the default mode from `WEBAGENT_DEFAULT_AGENTIC_MODE` in `.env.local`, re-read without a restart

### Changed
- **`/api/task` with neither `mode` nor `agentic` now means `off`** (it used to run the workflow). The web app always sends `agentic: true`
- `POST /api/v1/chat/` and `POST /api/v1/tasks/create` are marked deprecated in OpenAPI
- Shell and container `WEBAGENT_*` variables win over `.env.local` and the YAML loader's exported values
- Docs: API reference (task modes, `/api/triage`), architecture, Vercel and production checklist cover triage; version 3.1.0
- Backend Docker image: `WEB_CONCURRENCY=2` (was 4) and the MiniLM encoder baked into `/opt/models`
- uat and prod keep triage off for now: `triage-2026-10-a` misses the release gate (research recall 0.931, target 0.95), and the startup check refuses a classifier that hasn't passed

### Fixed
- Direct answers no longer show invented sources (`generateMockSources` removed)
- `/api/v1/openapi.json` failed to build because `TaskInfo` put a Python type into its JSON schema

## 3.0.3 - 2026-10-10

### Added
- Per-agent output limit: `max_tokens` under an agent in `config/{env}.yaml` lowers that agent's limit; web research uses 4,096 in every environment
- A warning in the logs whenever a Together reply stops at the `max_tokens` limit, since the text is then cut off
- `npm run stack:up` / `npm run stack:down` start and stop the local Docker Compose stack (backend, Prometheus, Grafana) from the repo root, tagging the backend image with the `package.json` version
- `GRAFANA_ADMIN_PASSWORD`, `GRAFANA_URL` and `GRAFANA_SERVICE_ACCOUNT_TOKEN` in `.env.local.example`
- Tests that fail if the web research prompt reaches the model with unfilled placeholders or without its `ANALYSIS:` cue, plus tests for the reasoning cleanup, per-agent limits, zip-only NLTK data, the cut-off warning and repeated MLflow logging

### Changed
- Web research sends one filled-in prompt that tells the model to use only the search results and ends with `ANALYSIS:`; any reasoning a model writes before its answer is dropped
- The Docker image sets `MLFLOW_TRACKING_URI=file:///app/data/mlruns` (the root filesystem is read-only) and `GIT_PYTHON_REFRESH=quiet` (the image has no `git`)
- Each `log_to_mlflow` call gets its own run, nested under any active run
- Docker Compose reads the Grafana admin password and the backend's API keys from the repo-root `.env.local` (`--env-file .env.local`) instead of a separate `backend/.env`, so all secrets live in one file
- The Compose stack's Grafana listens on host port 3001 (`GRAFANA_HOST_PORT`), so it no longer collides with the Next.js dev server or a local Grafana on 3000
- Docs updated: backend README troubleshooting, testing guide, triage report

### Fixed
- Web research took about 3 minutes per round and could invent sources: it sent its prompt with unfilled `{query}`/`{search_results}` placeholders and ended on raw search results, so the model kept writing until the 16,384-token prod limit. The prompt is now filled in and ends with `ANALYSIS:`, reasoning the model writes before its answer is dropped, and web research is capped at 4,096 output tokens. A real run of the same question went from 479 s to 153–208 s, with research calls taking 18–28 s instead of 175 s
- MLflow logging failed in the container: it wrote to the read-only `/app/mlruns` (now `/app/data/mlruns`), and inside an active run a second call failed with "Changing param values is not allowed" (each call now gets a nested run)
- The NLTK check reported `wordnet` as missing when it was installed as a zip

## 3.0.2 - 2026-10-10

### Added
- `GET /health` at the server root for Docker and Kubernetes health checks: no API key, not rate-limited, not logged
- Prometheus metrics are also served at `/metrics`, the path Prometheus and the Kubernetes annotations scrape (`/api/v1/metrics` still works)
- `WEBAGENT_CODE_SANDBOX` chooses where model-written code runs: `local` (default; no API keys or other secrets in its environment, plus CPU, file-size and memory limits) or `docker` (a throwaway container with no network and a read-only filesystem, built from `backend/sandbox/`)
- Follow-up questions such as "what about Europe?" are researched as a complete question: the supervisor adds `standalone_query` to the research plan, and web and internal research search with it
- Task status is saved to `backend/data/results/` when a task is created, so any server worker can answer a poll; a task still processing after `WEBAGENT_TASK_TIMEOUT_MINUTES` (30) is reported as interrupted
- Per-user rate limits behind the web app: `/api/task` forwards the caller's address, which the backend trusts only from `WEBAGENT_TRUSTED_PROXIES`; a limited request now gets a `Retry-After` header
- `backend/.dockerignore`, so local data, logs and caches stay out of the image
- Tests for shared task status, health and metrics, rate limiting, the code sandbox, deployment settings, follow-up questions and the Tavily retry

### Changed
- Backend tests run offline: `conftest.py` blocks network connections other than localhost, turns LangSmith tracing off, and points the model registry at a temporary file (`WEBAGENT_TESTS_ALLOW_NETWORK=true` lifts the network block for debugging)
- Docker image: Python 3.12, installed with `backend/constraints.txt`, runs with `WEBAGENT_ENV=prod`, ships NLTK data, takes the version as `--build-arg APP_VERSION`, sets the worker count with `WEB_CONCURRENCY`, and checks health with Python (the slim image has no `curl`)
- Kubernetes: the pod runs with `WEBAGENT_ENV=prod` and gets `WEBAGENT_SECURITY_SECRET_KEY`, `WEBAGENT_API_KEY` and `TAVILY_API_KEY` from its secret; `/app/data` and `/app/logs` are writable volumes on the read-only filesystem; one replica until `/app/data` is shared; `deploy.sh` requires the new secrets and defaults to version 3.0.2
- Docker Compose reads secrets from `backend/.env`, keeps task data in a volume, and sets `WEBAGENT_ENV`
- The model registry's state file moved from `backend/models/registry/` to the git-ignored `backend/data/registry/` (`WEBAGENT_MODEL_REGISTRY_PATH`)
- NLTK data is checked once through `ensure_nltk_data`; missing data is downloaded only if `WEBAGENT_NLTK_AUTO_DOWNLOAD` is true (the default; false in the Docker image)
- Docs updated: README, backend README, local development, Kubernetes, Vercel, API reference, architecture, production checklist, testing and `.env.local.example`

### Fixed
- Containers started in dev mode (API keys not required, debug errors shown) because the Dockerfile and Kubernetes set `ENVIRONMENT` while the app reads `WEBAGENT_ENV`
- `WEBAGENT_ENV` set in the shell was ignored whenever `.env.local` existed; the shell or container value now wins
- `WEBAGENT_SECURITY_*` variables (API keys on/off, rate limits, request size) had no effect because the YAML config overrode them
- With several workers, a poll that reached another worker returned "Task not found" and the web app reported the task as failed
- Health checks called `/health`, which didn't exist, so containers were marked unhealthy and pods restarted
- The container couldn't start as its non-root user: data and log paths resolved to `/backend/...` inside the image, and the version read as 0.0.0
- Prometheus scraped `/metrics`, which didn't exist, so dashboards stayed empty
- Building the workflow downloaded NLTK data on every fresh start, and the document extractor downloaded NLTK data on every entity extraction
- The Tavily retry after a 400 error blocked the server
- All web app users shared one rate-limit budget, and the per-caller table never shrank
- Tests reached real services: the scripted end-to-end test ran the real web research agent, and test runs sent traces to LangSmith
- A test run rewrote the tracked `models_registry.json`
- `test_request_size_limiting` sent 1 MB against a 10 MB limit, so it was always skipped; it now sends one byte over the limit
- A log line reported ".env.local exists: False" by checking the wrong folder
- The Docker image couldn't start: `langchain-chroma` and `langchain-huggingface` are imported at load time but were missing from `requirements.txt` (they were only installed in the local conda environment)
- With several workers, a container with bad uat/prod settings kept restarting its workers forever instead of stopping; the image now runs the startup checks (`python -m app.core.startup_checks`) once before uvicorn and exits with a clear message
- Matplotlib warned on every run in the code sandbox because its config folder was read-only

### Security
- Model-written code no longer sees API keys or other secrets, and in uat/prod the server refuses to start with code execution on unless it runs in the container sandbox
- The Grafana admin password was removed from Docker Compose, the Kubernetes manifests and the docs; set `GRAFANA_ADMIN_PASSWORD`. The old password remains in git history, so change it wherever it was used

### Removed
- The Kubernetes ConfigMap that was mounted over `/app/config`: its format didn't match the app's YAML, and it hid `config/prod.yaml`
- Tracked generated files: `test_langgraph.py.bak2`/`.bak3`, the test-only `models_registry.json`, `backend/mlruns/0/meta.yaml` and `mlartifacts/` (the MLflow files stay on disk)
- The unused Flask app in `backend/api/` (the old Vercel placeholder for `/api/chat` and `/api/task`; Next.js routes serve both)

## 3.0.1 - 2026-10-10

### Added
- Every log line written during a task is tagged `[task <id>]`, including parallel agents and library logs
- Token counts for Together AI calls now reach LangSmith (`TogetherWithUsage` keeps the `usage` data Together returns)
- Tests for the agentic client, `ChatContext`, the `/api/chat` and `/api/task` routes, LLM provider selection, rate limiting, the coding agents, task-tagged logging, LangSmith setup and token usage
- Coverage commands: `npm run test:coverage`, `npm run test:backend`, `npm run coverage:backend`
- Complete `.env.local.example` template listing every setting, now tracked in git

### Changed
- LangSmith upgraded from 0.1.125 to 0.3.45
- Agents rely on LangChain's built-in tracing instead of attaching a second tracer
- Jest coverage minimum set just below current coverage (it was 70%, which was never met)
- Docs updated: setup, environment variables, testing, logs, saved results and tracing

### Fixed
- Traces were split between two LangSmith projects when the shell exported `LANGSMITH_PROJECT`, leaving steps stuck as "pending"; the app now sets both `LANGSMITH_*` and `LANGCHAIN_*` variables
- A failed agentic task fell back to plain chat silently instead of showing the fallback notice (missing `await`)
- Reports could contain only a title when the model's reply had no summary or analysis; the research synthesis is used instead
- `jest.setup.js` crashed in tests that use the Node environment

## 3.0.0 - 2026-10-10

### Added
- `/api/task` endpoint on the FastAPI backend, proxied by Next.js, so the web app's Agentic mode runs the multi-agent workflow
- API key authentication (`X-API-Key` / `WEBAGENT_API_KEY`), with startup checks that block insecure secrets in uat/prod
- Finished task results saved to `backend/data/results/<task_id>.json` and served after a restart
- Automatic retry (up to 3 attempts) for temporary Together AI errors such as 429 and 503
- Regression tests that run the real LangGraph workflow, plus tests for report parsing, saved results and retries

### Changed
- Default model is now `meta-models/Muse-Glimmer-30B` (set with `WEBAGENT_LLM_DEFAULT_MODEL`)
- Together AI rate limit reduced from 12s to 2s between calls and made configurable (`TOGETHER_RATE_LIMIT_SECONDS`)
- Chat UI restyled: sans-serif type, neutral dark colors, centered questions and left-aligned answers
- Generated files (caches, logs, artifacts, vector databases) are no longer tracked in git

### Fixed
- Agent reports were dropped between workflow steps, so research never reached the senior agent and could loop until the recursion limit
- The backend stopped responding while a task ran, because the rate limiter used a blocking sleep
- Workflow routing, prompt template errors and report building so a run finishes with a full report
- Team Manager and Supervisor now read JSON wrapped in code fences or followed by extra text, and no longer repeat the report summary
- Agentic mode sent the wrong request shape and silently fell back to plain chat
- The user's question disappeared from the chat when the answer arrived
- Frontend tests could not run from a clean install (missing Babel and Jest packages)

## Before 3.0

Entries for 2.x and earlier are in [docs/CHANGELOG-archive.md](docs/CHANGELOG-archive.md).
