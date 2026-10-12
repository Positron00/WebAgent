# Spec: agentic triage (Off / Auto / Always)

Status: draft v5, aligned with the implementation (see "Implementation status") · Source plan: [`planning/agentic-triage-plan.md`](../../agentic-triage-plan.md) (draft v7, aligned with this spec) · Base: `main` at v3.0.3 (`d408f60`) · 2026-10-11

## What this feature does

Today, with Agentic on, every chat question runs the multi-agent research workflow (153–208 s, up to 8 model calls), even "hi". This feature replaces the on/off Agentic toggle with three modes:

- **Off**: every question goes straight to the LLM.
- **Auto** (new): a small local classifier decides. Research questions go to the agentic workflow; routine questions go straight to the LLM.
- **Always**: every question goes to the agentic workflow.

Every answer shows which route answered it. In Auto, a direct answer offers "Research this in depth".

## Documents

| File | What it holds |
|---|---|
| [`requirements.md`](requirements.md) | What must be true when the feature ships. Numbered requirements (`REQ-…`, `NFR-…`) with acceptance criteria |
| [`design-high-level.md`](design-high-level.md) | Components, request flows for each mode, failure handling, deployment, main decisions, known limitations |
| [`design-low-level-backend.md`](design-low-level-backend.md) | Python modules, data types, endpoint schemas, config, metrics, training and eval scripts, Docker changes |
| [`design-low-level-webapp.md`](design-low-level-webapp.md) | TypeScript types, storage and migration, Next.js routes, `apiClient` routing, UI components |
| [`tasks.md`](tasks.md) | Work broken into phases and tasks. Each task lists the requirements it covers and the tests that prove it |

Read in that order. Requirements say *what*; the designs say *how*; tasks say *in what order*. When documents disagree, `requirements.md` wins, then the designs, then the plan.

## Scope

In scope: phases 0–5 of the plan (fix existing problems, dataset and classifier, backend service and shadow mode, three-way toggle, tuning, backend modes for API clients).

Out of scope:

- Phase 6, recurring research tasks. It gets its own spec. This spec only makes sure phase 5 gives it what it needs (a `mode` field on `POST /api/task`).
- Changes to the agentic workflow itself.
- Streaming responses.
- Sending images to the model from the backend (the backend direct path is text only).

## Decisions (2026-10-11)

1. **Backend direct path is text only.** Results set `image_ignored: true` when a request had an image.
2. **No mode and no old flag means `off`**, on both `POST /api/task` and `POST /api/v1/chat/completions`. The web app always sends `agentic: true`, so it is not affected.
3. **2 uvicorn workers.** Kubernetes already sets `WEB_CONCURRENCY=2`; the Dockerfile default (used by Compose) drops from 4 to 2. The Kubernetes memory limit rises from 1 GiB to 1.5 GiB, after the container measured 1.08 GiB with triage on.

## Findings from the code that shaped the spec

These came from checking the plan against the code; the plan has been updated to match.

1. **Follow-up history already reaches the supervisor.** `SupervisorAgent._with_history` reads `context["previous_messages"]` and folds the last 8 messages into `{query}`. Phase 0 needs only the new `TaskRequest` field and the client change; no new prompt variable, and `test_followup_query.py` stays as it is.
2. **The Hugging Face cache lives on the data volume.** The Dockerfile sets `HF_HOME=/app/data/hf-cache`, and `/app/data` is mounted in Compose and Kubernetes, so a model there would be hidden. 3.1.x baked the base encoder into `/opt/models`; since `triage-2026-10-c` the fine-tuned encoder ships in `models/triage/encoder/` with the code instead.
3. **"Backend unreachable" needs a marker.** A 503 could come from the Next.js proxy or the backend. The proxy adds `X-WebAgent-Proxy-Error: unreachable` so the client can tell them apart.
4. **The backend LLM wrapper is text only**, which led to decision 1.
5. **`/api/v1/chat/completions` defaults to non-agentic today**, which led to decision 2.
6. **Training scripts go in `backend/scripts/`**, so they import the same `build_model_input` the server uses.
7. **The Docker image excludes `data/*`** (`.dockerignore`), so curated data stays out of the image, while `models/triage/` is copied by the existing `COPY . .`.
8. **Shadow mode only sees questions sent with Agentic on**, so its data leans toward research. The shadow week uses a deliberate mix of questions, and the seed set supplies routine examples.

## Implementation status (2026-10-11, releases 3.1.0, 3.1.1 and 3.2.0, merged to `main`)

Code for phases 0, 1, 2, 3 and 5 is in place and matches these documents; backend pytest and Jest pass.

| Item | Status |
|---|---|
| Classifier `triage-2026-10-c` (release 3.2.0; tasks 1.8, 4.3) | Training data grown about 4x to 2,496 generated rows (new `general_knowledge` category), MiniLM fine-tuned SetFit-style (REQ-024). **Passes the gate**: 155/155 on `test.jsonl`, and 39/40 on `probe.jsonl`, a second check in a different style. Scores are almost always near 0 or 1, so the uncertain metric and tie-breaker see little. Earlier `triage-2026-10-a` (600 rows, frozen) failed with research recall 0.931. Details in `backend/data/triage/README.md` |
| Container with `triage-2026-10-c` | About 83 MB larger than 3.0.3 by layers (target ≤ 100 MB); the fine-tuned encoder is the only new content. Starts with no network, triage off and on; both workers load the encoder from `/app/models/triage/encoder` in ~155 ms and route as locally. Per-worker memory unchanged: ~595 MB off, ~677 MB on, within the 1.5 GiB limit |
| Consequence (updated) | The gate no longer blocks uat/prod; they keep `triage.enabled: false` until the shadow week and a human data review. The encoder ships in `models/triage/encoder/` (weights in Git LFS) and is copied into the image; the base-encoder bake step was removed |
| Data review (tasks 1.2–1.4) | Rows were written by AI agents (`source: "generated"`), not by hand; a person still has to review both files |
| Classify latency (task 1.10, NFR-01) | p50 8.9 ms, p95 9.6 ms on a laptop CPU (Apple Silicon), well under 100 ms. Through the Next.js proxy: p50 17 ms, p95 25 ms (target < 300 ms) |
| Direct-path timing (task 1.9, NFR-02) | `/api/chat`, 20 routine questions: median 3.4 s, p95 5.1 s, max 6.6 s. The < 10 s target stands. Triage routed all 20 direct |
| Container (task 2.9, NFR-06, NFR-09) | Measured with 3.1.0's baked base encoder: encoder layer 91.7 MB (target ≤ 100 MB). Starts with no network, triage off and on; with triage on, both workers load the encoder in ~0.2 s and classify the same as locally. Triage adds ~90 MB per worker (682 vs 592 MB resident). **Memory doesn't fit:** the container uses 1.08 GiB with triage on and 955 MiB with it off, against the old 1 GiB Kubernetes limit. **Decided:** the limit is now 1.5 GiB (request 1 GiB), which leaves 28% headroom with triage on |
| Follow-up history (task 0.5) | Real run: "what about Europe?" after a question on US EV sales produced a report on the European EV market in 2025 |
| Not done yet | 2.11 dashboard screenshot; manual browser check of the toggle and override; shadow week; 3.11 switch to `act`; 3.12 container run in Auto; phase 4 |

### Review fixes (2026-10-11)

A check of code against these documents found and fixed:

| Gap | Fix |
|---|---|
| "Research this in depth" answers were never shown: `MessageList` dropped an assistant message that followed another one | The reply is inserted under its own question, and the list shows a second answer to the same question |
| Off, Always and override decisions fed 0 ms into `webagent_triage_latency_seconds`, hiding the classifier's p95 (NFR-01) | Only real classifications are timed |
| A corrupt `classifier.joblib` raised out of `get_model`: no 60 s backoff, and the dev warm-up could stop startup | Load errors fall back and are retried at most once a minute; `warm_up` never raises |
| Collection cleanup ran only with triage enabled, though overrides are collected either way | Cleanup always runs at startup and daily |
| Grafana override rate divided by Off-mode answers, which can't be overridden | Denominator is direct decisions made by triage |
| An image with no text in Always was sent to the agents with an empty question | It goes direct (`decided_by: rule`) |
| A saved settings object with the old `agentic` key skipped the colour migration | The key is dropped and the migration still runs |
| A bad `triage.mode` in YAML wasn't caught; the invalid-value `/api/settings` reply could be cached | YAML falls back to `act` with a warning; both replies send `no-store` |
| Threshold "starts at 0.35" contradicted REQ-022 (training picks it; v-a uses 0.5) | Definition now points to REQ-022 everywhere |
| Missing tests: rate limit on `/api/triage`, prod with triage disabled, send disabled before the mode is known, image note | Added. Docs: `docs/TESTING.md`, `.env.local.example` (`WEBAGENT_TRIAGE_*`), backend README (gate check, measured latency) |

### Smoke-test fixes (2026-10-11, release 3.1.1)

Real runs with real model and search keys before merging found three bugs that the faked tests couldn't:

| Bug | Fix |
|---|---|
| The backend direct path (`answer_directly`) sent the bare question to the text-completion model, which continued it and wrote its reasoning into the answer ("assistant to=self … We need population of Canberra …") | A direct-answer prompt that ends in `Answer:`, and the reasoning-marker strip shared with web research (`app/utils/reasoning.py`) |
| With an image, the text-only direct path described a picture it never received | The prompt says an image was attached but can't be seen; the reply now says so |
| The supervisor model sometimes plans `requires_document_extraction` for a plain question; with no document the task failed in 7 s ("Document ID is required for extraction"). Predates triage; hit Always too | `without_missing_document`: with no `document_id`, the plan researches instead |

### Manual test fixes (2026-10-11, release 3.1.1)

| Problem | Fix |
|---|---|
| No tooltips on the mode toggle or route label: they used the native `title` attribute, which shows only after a delay, never on focus, and not in some embedded browsers | `Tooltip.tsx`, shown at once on hover and keyboard focus; the route label is focusable |
| Sending an image failed: "messages[10]: data did not match any variant of untagged enum MessageContent". `/api/chat` sent `image_url` as a string; Together expects `{url}`. Predates triage | Fixed format; a real call with a test image now returns the right colour |
| `next build` failed: `src/app/api/settings/route.ts` exported helper functions, which Next.js doesn't allow in a route file | Helpers moved to `src/utils/defaultAgenticMode.ts`; `npm run build` passes |

## Traceability

Every task in `tasks.md` names the requirements it satisfies. Every requirement names its verification (test, eval or manual check). The coverage table at the end of `tasks.md` shows that each requirement has at least one task.
