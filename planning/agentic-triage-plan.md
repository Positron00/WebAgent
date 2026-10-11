# Plan: triage chat questions before the agentic workflow

Status: draft v6, aligned with the spec in [`planning/specs/agentic-triage/`](specs/agentic-triage/README.md); implementation status is in the spec README (the first classifier version doesn't pass the gate yet, so uat/prod keep triage off) · Base: `main` at v3.0.3 (d408f60) · 2026-10-11

Decided:

- The Agentic toggle becomes **Off / Auto / Always**.
- Every answer shows which **route** answered it.
- The classifier is **our own**: MiniLM sentence embeddings with a small trained classifier on top, fine-tuned later only if the evaluation calls for it.
- Users whose Agentic toggle is on today move to **Auto**.
- New users start on **Off**. That default is a runtime setting: changing it doesn't require restarting `npm run dev` or `npm start`.
- "Research this in depth" appears on direct answers in **Auto** mode only.
- The backend gets the same three modes (**phase 5, in scope**), so API clients and scheduled jobs can use them. The next feature, **recurring research tasks** (phase 6), builds on that.
- On the backend, a request with **no mode and no old flag means `off`**, on every endpoint.
- The backend's direct path is **text only** for now; images are ignored there.
- The backend runs **2 uvicorn workers**, within the existing 1 GiB Kubernetes limit.

## Summary

Today, with Agentic on, every question goes through the multi-agent workflow, including "hi", "what does REST stand for?" and "make this paragraph shorter". In the 3.0.3 container runs, a research question took 153–208 s and up to 8 model calls.

The toggle gets three settings:

| Setting | Behavior |
|---|---|
| **Off** | Every question goes straight to the LLM (`/api/chat`). Same as today's Off |
| **Auto** (new) | A local classifier decides. Deep research goes to the agentic workflow; routine questions go straight to the LLM |
| **Always** | Every question goes to the agentic workflow. Same as today's On |

The classifier runs inside the backend in about 10 ms on CPU once warmed up (p95 9.6 ms measured on a laptop), with no model API call and no cost per question. Every answer shows "Answered directly" or "Deep research". In Auto mode, a direct answer also offers "Research this in depth".

## Goals and non-goals

Goals:

- In Auto mode, routine questions are answered in seconds instead of minutes.
- Questions that need research still reach the agentic workflow. A missed research question costs more than a slow routine one, so uncertain cases go to research.
- Users always see which route answered them, and can override it.
- Decisions are logged, counted and testable offline.

Non-goals:

- Changing the agentic workflow itself.
- Streaming responses (separate work).

## How routing works today

| Step | Where | What happens |
|---|---|---|
| Toggle | `src/components/chat/MessageInput.tsx`; stored as `agentic: boolean` in `src/utils/storage.ts` (default `false`) | The **Agentic** button flips the setting. `AppContext` saves the whole settings object on every load, so the default `false` is saved too, and a saved `false` can't be told apart from a user's choice |
| Route | `src/utils/apiClient.ts` → `sendChatMessage` | On: `sendMultiAgentRequest(query, image)` → `/api/task` (polls up to 8 min). Off: `/api/chat` |
| Agentic | `src/app/api/task/*` → FastAPI `POST /api/task` | LangGraph workflow: supervisor → research (1–3 rounds) → senior research → team manager |
| Direct | `src/app/api/chat/route.ts` | One Together AI chat call, with the prompt style, knowledge focus and citation settings, plus up to 50 earlier messages |
| Fallback | `sendChatMessage` | If the agentic request fails, the client retries through `/api/chat` and shows the "fallback mode" notice |

Measured on 3.0.3: agentic runs took 153–208 s, and the supervisor call alone took 6–10 s. The direct path hasn't been timed yet; measure it in phase 1.

## Problems to fix first

These are existing issues. Auto mode would make each one more visible, so they come first.

1. **Direct answers show made-up sources.** With "cite sources" on, `ChatContext.tsx` calls `generateMockSources`, which invents 3–5 sources with random domains (nature.com, arxiv.org, research.edu…) and titles built from the answer's own words. **Fix:** remove it. Direct answers show no source list; only agentic answers have real sources.
2. **Agentic requests drop the conversation.** `sendMultiAgentRequest` sends only the latest question, and `TaskRequest` has no field for earlier messages. The supervisor's `standalone_query` rewrite from 3.0.2 therefore never fires from the web app, so a follow-up such as "what about Europe?" is researched literally. **Fix:** send the last 8 messages as `previous_messages` to `/api/task` and pass them into the workflow context. The supervisor already reads `context["previous_messages"]` (`_with_history`), so its prompt doesn't change.
3. **Images are ignored by the agentic workflow.** `TaskRequest.image` is accepted but never used. **Fix:** in Auto mode, a question with an image always goes direct. In Always mode, show a note that images are only used by direct answers. An image sent with no text goes direct in every mode, since the agents would have nothing to research.

## Design

### Where triage runs

The **backend classifies** and the **web app routes**.

1. In Auto mode, `apiClient` calls `POST /api/triage` with the first 8,000 characters of the question. This is a new Next.js route that forwards to a new FastAPI endpoint, like `/api/task` does, keeping the API key on the server.
2. If the answer is `direct`, the client sends the conversation to `/api/chat`, exactly as in Off mode.
3. If the answer is `agentic`, the client calls `/api/task` with the question and the earlier messages. The supervisor rewrites follow-ups into standalone questions, as it already does.

Why the backend classifies:

- The model files, Python ML stack, metrics and offline tests already live there.
- The same classifier can serve other API clients later; see "API clients" below.

Why the web app routes:

- Routine answers keep the user's prompt settings and avoid task polling.

Off and Always skip the triage call entirely.

All triage rules and thresholds live in one Python module (`app/services/triage.py`), with model loading in `app/services/triage_model.py`. The web app never classifies, so `/api/triage` and the phase 5 backend modes can't disagree.

### API

`POST /api/triage` (FastAPI; requires `X-API-Key`; proxied by `src/app/api/triage/route.ts`)

Request:

```json
{"query": "Compare the top CRM tools in 2026 with pricing", "previous_user_question": null, "has_image": false}
```

`query` may be empty (up to 8,000 characters), so an image sent without text is answered by the rules.

Response:

```json
{"route": "agentic", "p_agentic": 0.92, "decided_by": "model", "reason": "Asks to compare current products", "classifier_version": "triage-2026-10-a"}
```

`decided_by` is one of:

- `rule`, `model` or `fallback` (classifier failed) from triage.
- `config` when `WEBAGENT_TRIAGE_ENABLED=false`.
- `mode` (Off or Always chose the route) and `override` ("Research this in depth"), which the web app and phase 5 use.

The same field name and values are used everywhere: on `/api/triage`, on task results and on stored messages. `reason` comes from the rule that fired, or from the nearest labeled category for model decisions, so the label's hover text says something useful.

**Status codes and what the web app does:**

| Outcome | Web app route |
|---|---|
| 200 | Use `route`. A classifier that can't load still returns 200 with `route: agentic`, `decided_by: fallback` |
| 200 with a body that isn't a valid triage reply | Agentic (`decided_by: fallback`) |
| 503 from the Next.js proxy (backend unreachable), marked by the header `X-WebAgent-Proxy-Error: unreachable` | Direct, with the existing fallback notice. The agentic request would fail too |
| Other 5xx, or no answer within 2 s (client timeout) | Agentic. The backend is up, so the workflow may still work |
| 429 (rate limited) | Show the existing rate-limit message; send nothing |
| 401 (bad server API key) | Direct, with the fallback notice, and log an error. It's a deployment bug, and `/api/task` would fail the same way |
| Other 4xx (e.g. 422) | Agentic (`decided_by: fallback`), and log a warning |
| The browser's request fails (offline) | Direct, with the fallback notice |

The header is needed because a 503 can also come from the backend itself; only the proxy sets it.

**Rate limits:** `/api/triage` counts toward the per-client limit (60 requests per minute by default) like any other route. It isn't exempt, because it runs model inference. One extra request per question is small next to task polling (from 1 request per second, with backoff), and direct answers make no polling requests at all, so Auto lowers the total.

### Classifier

#### Approach

| | LLM prompt | MiniLM embeddings + logistic regression (**start here**) | Fine-tuned MiniLM (SetFit or full) |
|---|---|---|---|
| Latency | 1–3 s (estimate; 6–10 s for a supervisor-sized call today) | ~50 ms on CPU (measured, see below) | ~50 ms on CPU |
| Cost per question | One API call | None | None |
| Same answer every time | No (sampling, model updates) | Yes | Yes |
| Training data needed | None | ~500–1,000 labeled questions | Same set; does better with fewer |
| Rewrites follow-ups | Yes | No (not needed: the supervisor does it) | No |
| Testable offline | Only with a fake | Yes | Yes |
| Effort | Lowest | Low: train in seconds, retrain any time | Medium: adds the `setfit` dependency and a training run |

**Recommendation:** start with frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions, 22M parameters) and a scikit-learn logistic regression on top. Move to SetFit fine-tuning of the same model only if the held-out evaluation misses the targets below. Keep an LLM tie-breaker as an option for the uncertain band, but leave it off by default.

#### Feasibility test (done, 2026-10-11)

Run offline in the `agents` environment. `all-MiniLM-L6-v2` is already in the local Hugging Face cache, and `sentence-transformers` 3.3.1 and `scikit-learn` 1.5.2 are installed.

- **Data:** 60 hand-written questions (30 research, 30 routine).
- **5-fold cross-validation:** research recall 30/30, routine routed direct 30/30.
- **Latency:** median 52 ms, max 89 ms per question (embed and classify, CPU, including warm-up).
- **Unseen probes:**

| Question | p(agentic) |
|---|---|
| "Compare the top CRM tools in 2026 with pricing" | 0.92 |
| "What's the weather like on Mars?" | 0.42 (uncertain) |
| "thanks" | 0.21 |
| "explain list comprehensions" | 0.10 |

**Caveat:** 60 examples written by one person are easy to separate. Research questions were long and mentioned "compare" or "2026"; routine ones were short. Real questions will be harder, especially short ones that need current facts ("Bitcoin price now?", "who won last night?"). The dataset must include them on purpose.

#### Decision logic

1. **Rules** (instant):
   - An image is attached: direct.
   - The message is empty, or the whole message is a greeting or thanks: direct.
   - The message starts (after an optional "please") with an explicit research request: "research", "find sources", "find studies", "write a report", "do a deep dive" or "give me a literature review": agentic. "investigate" is left out because it often starts coding questions.
2. **Model:** `p = p(agentic)`. The input is the current question. If it's short (under 6 words) and there's a previous user question, the input becomes `previous question → current question`, so "what about Europe?" inherits its context. The training data includes examples in this joined form, so the model sees the same input at training and serving time.
3. **Threshold:** `p ≥ threshold`: agentic; otherwise direct. Training picks the highest threshold, up to 0.5, that keeps 95% research recall on a validation split. The first estimate was 0.35; `triage-2026-10-a` landed on the 0.5 cap.

   Erring toward agentic is deliberate, to keep research recall high.
4. **Optional LLM tie-breaker** for the tie-break band `threshold ≤ p < 0.65`, off by default (`WEBAGENT_TRIAGE_LLM_TIEBREAK=false`). If the call fails or takes more than 3 s, the model's route stands.
5. **Fallback:** if the classifier can't load or errors, the route is agentic. If the backend can't be reached at all, the client goes direct and shows the existing fallback notice, since the agentic request would fail too. The status-code table under "API" is the full contract.

#### Training data

The classes are **agentic** (needs current or external information, several sources, comparison, citations, or multi-step research) and **direct** (everything else).

1. **Seed set:** about 150 hand-written questions across categories, listed in `backend/data/triage/categories.json`. These are training data.
   - Agentic: current events, product comparisons, market or industry analysis, scientific evidence, regulation, statistics, "with sources" requests.
   - Direct: greetings, definitions of stable concepts, how-to and coding help, math, rewriting or translating provided text, creative writing, follow-ups about the previous answer.
   - Hard cases: short time-sensitive questions, long routine questions, and research phrased casually.
2. **Expansion:** generate about 800 varied paraphrases with an LLM from the category list, then spot-check 10% by hand.
3. **Real questions:** from shadow mode (phase 2) and override clicks (phase 3), labeled by an LLM judge. Review every case where the judge and the classifier disagree. Shadow mode only sees questions sent with Agentic on, so its sample leans toward research; during the shadow week, use a deliberate mix of questions.
4. **Held-out test set:** about 150 questions, **written by hand separately** from the seed set (not paraphrases of it), stratified by category, with at least 10 per hard-case category. Never used for training or for picking the threshold, and checked for near-duplicates of training rows. This is the release gate.

**As built (2026-10-11):** the first version's 600 training and 155 test rows were written by AI agents instead of by hand (the test set by a separate agent that never saw the training set), across 14 categories, and still need a person's review. See `backend/data/triage/README.md`.

The curated data is stored as `backend/data/triage/train.jsonl` and `test.jsonl`, with `text`, `label`, `category`, `source`. It's versioned in git: it's small text, with no user data unless anonymized.

**Collected questions are user data.** They are kept apart from the curated set:

- Shadow decisions and overrides are written, with question text, to `backend/data/triage/collected/YYYY-MM-DD.jsonl`. That path is added to `.gitignore`. In a container the files are on the data volume (`/app/data/triage/collected/`); copy them out with `docker cp` for review.
- Collection is controlled by `WEBAGENT_TRIAGE_COLLECT`. It's on by default in dev and off in uat and prod until someone approves it there. This is separate from logs, which never carry question text outside dev.
- A question moves into `train.jsonl` only after a person reviews it and removes or rewrites names, emails, numbers and anything else personal. Collected questions never go into `test.jsonl`. `backend/scripts/train_triage.py` reads only the curated files.
- Collected files are deleted after 30 days. The cleanup runs at startup and daily even when triage is disabled, because overrides are still collected.

#### Model lifecycle

- **Training:** `backend/scripts/train_triage.py` embeds the training set, fits the classifier, picks the **highest** threshold (capped at 0.5) that still reaches 95% research recall on a validation split, and writes `backend/models/triage/classifier.joblib` (the classifier plus one centroid per category for the hover reasons; under 100 KB) and `metadata.json`. The metadata holds the version, encoder name, threshold, data hash, category labels, and validation and test metrics. The scripts live in `backend/scripts/` so they use the same input-building function as the server.
- **Evaluation gate:** `backend/scripts/eval_triage.py` reports recall, precision and a confusion matrix per category on `test.jsonl`, and exits non-zero if the targets are missed. `--check-data` validates both data files. A new classifier version ships only if it meets the targets.
- **Serving:** `backend/app/services/triage.py` loads the encoder and classifier once at startup and exposes `classify(query, previous_user_question, has_image) -> TriageDecision`.
  - Each uvicorn worker loads its own copy: about 100–150 MB of RAM per worker. The backend runs **2 workers**, about 0.3 GB in total: Kubernetes already sets `WEB_CONCURRENCY=2`, and the Dockerfile default (used by Compose) drops from 4 to 2. Measure that this fits the 1 GiB Kubernetes limit with 20% headroom.
  - Set `torch.set_num_threads(1)` in the service, so several workers don't compete for every CPU core.
  - Embed one warm-up sentence at startup, so the first user doesn't pay for model loading.
  - `sentence-transformers` 3.3.1, `torch` 2.5.1, `scikit-learn` 1.5.2 and `joblib` are already in `requirements.txt` and `constraints.txt`. No new dependency is needed until SetFit.
- **Startup checks:** in uat and prod, `app.core.startup_checks` fails if `classifier.joblib` or the encoder is missing, unless `WEBAGENT_TRIAGE_ENABLED=false`. This follows the sandbox check from 3.0.2. In dev, a missing model logs a warning, and triage falls back to agentic.
- **Docker:** download the encoder at build time into `/opt/models` and set `SENTENCE_TRANSFORMERS_HOME=/opt/models`, not into `/app/data` (where `HF_HOME` points today). That path is a mounted volume, which would hide files baked into the image. The image grows by about 90 MB. `models/triage/` is copied by the existing `COPY . .`, and `.dockerignore` already keeps `data/*` (the curated data) out. This must land before shadow mode runs in a container.
- **Fine-tuning (if needed):** `backend/scripts/train_triage.py --setfit` fine-tunes MiniLM with SetFit on the same data. The fine-tuned model is saved under `backend/models/triage/encoder/` (about 90 MB; store it with Git LFS or as a release asset, not in plain git).

### User experience

- **Toggle:** a three-way control labelled **Agentic: Off · Auto · Always**, with tooltips:
  - Off: "Answer directly, no research agents".
  - Auto: "Use research agents when the question needs them".
  - Always: "Always use research agents (slower)".
- **Stored setting:** `agenticMode: 'off' | 'auto' | 'always'` replaces `agentic: boolean`. It's saved **only when the user picks a mode**, so "no saved mode" still means "follow the default".
  - Today, `AppContext` saves the whole settings object on every load. The default would then be saved as if chosen, and later changes to the default would never reach that user. `agenticMode` is therefore kept out of that object, in its own storage key (`agentic_mode`), and written only by the toggle's handler.
  - Migration, once, on first load after the upgrade: `agentic: true` → `agenticMode: 'auto'`. `agentic: false` → no saved mode (follow the default). A saved `false` is almost always the old default being saved, not a choice, so these users follow the default, which starts as Off. The old `agentic` key is then removed, so the migration can't run twice.
- **Changes apply at once.** As today, switching the mode updates the app state and the browser's saved setting, and the very next message uses the new mode. No reload or restart is needed. A Jest test keeps it that way.
- **Default for new users: Off, set at runtime.** The default comes from `WEBAGENT_DEFAULT_AGENTIC_MODE` in `.env.local` (`off`, `auto` or `always`; default `off`). A new route, `GET /api/settings`, re-reads `.env.local` whenever the file changes (cached by modified time), instead of using a build-time `NEXT_PUBLIC_` variable. Editing the value applies on the next page load, with no restart of `npm run dev` or `npm start` and no rebuild. It only affects people with no saved mode; anyone who has picked a mode keeps it.
  - **Where `.env.local` doesn't exist** (Vercel, or a container with real environment variables), the route falls back to `process.env.WEBAGENT_DEFAULT_AGENTIC_MODE`. There, a change needs a redeploy or restart, as any environment variable does. This is documented in `.env.local.example`.
  - The route returns only `{"defaultAgenticMode": "off"}`. The value isn't secret, so the route needs no API key. An unknown value logs a warning and returns `off`. The route is `dynamic = 'force-dynamic'`, so Next.js never caches it at build time.
  - **First paint:** until the mode is known, the toggle shows Off and sending is disabled for that moment. Users with a saved mode skip the request; for others, if the route fails, the default is Off.
- **Route label on every answer:** "Answered directly" or "Deep research", stored on the message as `route: {route, decidedBy, reason}`. Hovering shows the reason, e.g. "Auto · Looks like: definition of a stable concept" or "Always mode". Workflow logs show for any answer whose route is agentic, instead of depending on the toggle as `MessageList.tsx` does today.
- **Override (Auto only):** direct answers show "Research this in depth". It re-sends that question, with the earlier messages, to `/api/task` with `"triage_override": true`, skipping triage. The research answer is inserted directly below the direct answer, under the same question, even if the chat has moved on; the direct answer stays.
  - The backend counts the flag (`webagent_triage_overrides_total`) and records the question when collection is on. No separate feedback endpoint is needed.
  - Questions with an image don't show the button, because the agentic workflow can't see the image (problem 3).
  - **How it differs from the toggle:** the toggle is a *mode* for future questions. The button acts on *one question already answered*. Without it, fixing one misrouted question means switching to Always, re-typing the question, and switching back to Auto.
  - **Training signal:** each click is the best label we get. It's counted, and when collection is on, the question is recorded as a candidate "agentic" example. It joins the training set after review, like other collected questions.
  - **Not shown in Off** (the user chose no agents) **or Always** (every answer is already researched).
- **Progress:** triage adds about 10 ms plus a network round trip, so the existing loading indicator is enough.

### Configuration

| Setting | Read by | Default | Purpose |
|---|---|---|---|
| `WEBAGENT_DEFAULT_AGENTIC_MODE` | Web app | `off` | Mode for users with no saved setting. Read at runtime by `GET /api/settings`, so changes apply on the next page load without a restart |
| `WEBAGENT_TRIAGE_ENABLED` | Backend | `true` | `false`: triage answers `agentic` with `decided_by: config`, so Auto behaves like Always |
| `WEBAGENT_TRIAGE_MODE` | Backend | `act` (`shadow` in dev during phase 2) | `shadow`: `/api/task` classifies every question sent to the workflow (`agentic: true`, later `mode: always`; not overrides) and records the decision, but always runs the workflow. It never changes a route: `/api/triage` always answers, and `mode: auto` always acts |
| `WEBAGENT_TRIAGE_THRESHOLD` | Backend | from `metadata.json` (0.5 for `triage-2026-10-a`) | Override the trained threshold |
| `WEBAGENT_TRIAGE_LLM_TIEBREAK` | Backend | `false` | Ask the LLM when p is in the uncertain band |
| `WEBAGENT_TRIAGE_COLLECT` | Backend | `true` in dev, `false` in uat and prod | Write shadow decisions and overrides, with question text, to `backend/data/triage/collected/` |

Backend settings also get a `triage:` section in `backend/config/{dev,uat,prod}.yaml`, with the environment variables overriding it, as the `security:` settings do. A few tuning values are YAML only, with no environment variable: the uncertain band (0.15), the tie-break upper bound (0.65), retention days (30), the model folder, the encoder name, and the short-question word count (6).

The backend image also sets `SENTENCE_TRANSFORMERS_HOME=/opt/models` and `WEB_CONCURRENCY=2`.

### Observability

- **Logs:** one line per decision: route, `p_agentic`, `decided_by`, classifier version and latency. The question text is logged only in dev.
- **Prometheus:** `webagent_triage_decisions_total{route, decided_by}`, `webagent_triage_latency_seconds`, `webagent_triage_overrides_total`, `webagent_triage_uncertain_total`. "Uncertain" means `|p − threshold| < 0.15`. That is wider than the tie-break band, because it also counts near-misses just below the threshold.
- **Grafana:** a panel for the route split, override rate and uncertain rate. A rising override rate means too many questions go direct. The override rate counts only direct decisions made by triage, not Off-mode answers, which can't be overridden.
- **Latency:** only real classifications feed `webagent_triage_latency_seconds`. Off, Always and override decisions are counted but not timed, so they can't hide the classifier's p95.

## API clients and recurring tasks

**What API clients are:** programs that call the FastAPI backend over HTTP themselves, rather than through the web app. Examples would be a script, a notebook, a Slack bot, another service, `curl`, or the recurring-task scheduler in phase 6. The backend offers these entry points today:

- `POST /api/task` (with `"agentic": true/false`)
- `POST /api/v1/chat/completions` (OpenAI-style, `agentic_toggle`)
- `POST /api/v1/chat/` and `POST /api/v1/tasks/create`

**Who uses them today:** only the web app. `scripts/run_webagent.py` and the demos import the workflow in-process, without HTTP.

**Why the backend needs modes too:** in phases 0–4, the *web app* decides the route. Every other client would get the full workflow for each question, including "hi". Phase 5 moves the same three modes into the backend, so any client can use them.

### Phase 5: backend modes

- `POST /api/task` and `POST /api/v1/chat/completions` accept `"mode": "off" | "auto" | "always"`.
  - The old `agentic` / `agentic_toggle` flags still work: `true` → `always`, `false` → `off`.
  - With neither `mode` nor the old flag, the default is **`off`** on both endpoints. `/api/v1/chat/completions` already defaults to non-agentic, so its callers see no change. `/api/task` changes from agentic to `off` in that case; the web app always sends `agentic: true`, so it isn't affected. The backend README notes the change.
  - If both `mode` and the old flag are given, `mode` wins.
  - `"triage_override": true` skips all of this and routes agentic.
  - `POST /api/v1/chat/` and `POST /api/v1/tasks/create` keep their current behavior. Nothing uses them; mark them deprecated in the API docs.
- **`off`:** one direct, text-only model call with the question and `previous_messages`. This is the supervisor's existing direct-answer path, moved into `SupervisorAgent.answer_directly` and run without building the graph.
- **`auto`:** the backend runs the same `triage.classify(...)` as `/api/triage`, then takes the `off` or `always` path.
- **`always`:** the full workflow, as today.
- **Response:** every task result includes `"route": "direct" | "agentic"`, `decided_by` (the values listed under "API"), `p_agentic` when the model ran, and `"image_ignored": true` when the request had an image. The polling contract is unchanged; direct tasks simply complete in seconds.
- **Images:** with `mode: auto`, an image routes direct by the same rule as in the web app. The backend direct path is **text only** for now, so the image is ignored there, as it is on the agentic path (problem 3). Sending images from the backend would need a chat-style vision call and a model that accepts images; that is separate work.
- **Web app:** keeps routing on its own side for interactive chat, so routine answers keep the user's prompt settings and avoid polling. Both sides share one classifier and one set of thresholds.
- **Effort:** about 1.5 days.

### Phase 6: recurring research tasks (outline; separate plan to follow)

The user sets up a question that runs on a schedule, for example "Every Monday at 8:00, research what changed in EU AI regulation". Each run produces a new report.

- **Schedule record:**
  - `id`, `query`, `mode` (default `always`, since a scheduled research question is research by definition; `auto` is allowed). The scheduler always sends `mode` explicitly, because an API request without it means `off`
  - `cron` (or simple presets: daily, weekly, monthly), `timezone`, `enabled`
  - `last_run_at`, `next_run_at`, and the task IDs of past runs
- **Storage:** `backend/data/schedules/` (JSON, git-ignored), following the result-store pattern. Move to SQLite if the count grows.
- **Scheduler process:** a **separate single process** (`python -m app.scheduler`; its own Compose service and Kubernetes Deployment with one replica). It must not run inside the API workers, because with 2–4 uvicorn workers each would fire the same schedule. It wakes every minute, finds due schedules, and starts each one through `POST /api/task` with its mode, like any other API client. That's why phase 5 comes first.
  - **Auth and limits:** it sends `WEBAGENT_API_KEY` like the Next.js proxy does, so it needs that secret in its Compose service and Kubernetes Deployment. Its requests come from one address, so they share one rate-limit bucket. The per-day guardrails below keep it well under 60 per minute, and it polls results less often than the web app (every 10 s).
  - **Shared storage:** it reads `backend/data/schedules/` and the API writes it, so both need the same volume. In Kubernetes, that means a `ReadWriteMany` volume or moving schedules to a database first.
  - **Missed runs:** if the scheduler was down when a run was due, it runs that schedule once on restart, not once for each missed time.
- **API:** `POST/GET/PATCH/DELETE /api/schedules` and `POST /api/schedules/{id}/run` (run now), all behind `X-API-Key`.
- **Results:** each run is a normal task, saved in `backend/data/results/`. The schedule page lists past runs. A later step can add **"what changed since the last run"**, a short diff summary produced by one model call that compares the new report with the previous one.
- **Notifications:** in-app first; email or Slack later.
- **Guardrails:**
  - A limit on runs per day per schedule, and on total scheduled runs per day.
  - Skip a run if the previous one is still processing.
  - Stop a schedule after 3 consecutive failures, and show why.
- **Dependencies:** `croniter` to read cron expressions; it's small and not installed yet. Avoid a full job queue (Celery or RQ) until volume needs one.
- **Effort:** about 3–4 days for schedules, the scheduler, the API and a basic page. "What changed" and notifications come on top.
- **Open questions for the phase 6 plan:** who owns a schedule (there are no user accounts yet; API keys are shared), where results are delivered, and the cost limits.

## Plan

Each phase ends with its tests passing and can ship on its own. The detailed task list, with estimates per task, is in [`specs/agentic-triage/tasks.md`](specs/agentic-triage/tasks.md); the efforts below are its sums, including tests and docs.

| Phase | Effort | Needs |
|---|---|---|
| 0. Fix existing problems | ~1 day | Nothing; can start now |
| 1. Dataset and classifier | ~3 days | Nothing; can run alongside phase 0 |
| 2. Backend service and shadow mode | ~4 days | Phase 0 (shadow mode needs `previous_messages`) and phase 1 |
| 3. Toggle, labels, act | ~3.5 days | Phase 2, plus about a week of shadow data reviewed and retrained on |
| 4. Tune | ongoing | Phase 3 |
| 5. Backend modes | ~1.5 days | Phase 2 (the triage service). Can come before or alongside phase 3 |
| 6. Recurring tasks | 3–4 days | Phase 5; separate plan |

Total for phases 0–3 and 5: about 13 working days, plus the shadow-mode week. Phase 6 adds 3–4 days.

**Phase 0: fix the problems above** (about 1 day)

- Remove `generateMockSources`, and update the `ChatContext` tests.
- Add `previous_messages` (the last 8 messages, as `{role, content}`) to `TaskRequest`, `sendMultiAgentRequest` and the workflow context. The supervisor already reads it (`_with_history`) and folds it into its single `{query}` variable, so the prompt and `test_followup_query.py` stay as they are. Add a Jest and a pytest test.
- Show the image note in agentic mode (problem 3).

**Phase 1: dataset and classifier** (about 3 days)

- Category list, seed set, LLM expansion, spot-check, and a separately hand-written held-out test set.
- `backend/scripts/train_triage.py`, `backend/scripts/eval_triage.py`, and `backend/models/triage/` artifacts. Meet the targets on `test.jsonl`.
- Time the direct `/api/chat` path for 20 routine questions (median and p95), and confirm or adjust the "< 10 s" target before phase 3.
- Measure classify latency and the memory of one loaded model.

**Phase 2: backend service and shadow mode** (about 4 days)

- `app/services/triage.py` and `triage_model.py`, FastAPI `POST /api/triage`, the Next.js route (with the unreachable header), settings, startup checks, metrics, collection and the Grafana panel.
- Bake the encoder into the Docker image and set 2 workers; measure memory.
- Turn on shadow mode in dev, where collection is on. The three-way toggle doesn't exist yet, so shadow mode runs **inside the backend**: `POST /api/task` classifies every question it receives with Agentic on, records the decision, and runs the workflow as usual. The web app doesn't change. In uat, shadow mode records only metrics unless collection there is approved.
- After about a week of a deliberate mix of questions, label the collected questions, review and anonymize them, add them to `train.jsonl`, retrain, and re-run the eval gate.

**Phase 3: three-way toggle, route labels, act** (about 3.5 days)

- `agenticMode` setting with migration, `GET /api/settings`, the three-way control and tooltips.
- `apiClient` routing for each mode, the route label on every answer, workflow logs shown by route, and the override button (counted and collected by the backend).
- Set `WEBAGENT_TRIAGE_MODE=act` wherever shadow was on. This stops shadow recording; the web app already chooses the routes.
- Container run in Auto mode, then docs, CHANGELOG and version bump.

**Phase 4: tune** (ongoing)

- Weekly: retrain with override and shadow data, re-run the eval gate, and release a new classifier version if it improves.
- Try SetFit fine-tuning if the frozen encoder plateaus below the targets.

**Phase 5: backend modes for API clients** (about 1.5 days). See "API clients and recurring tasks".

**Phase 6: recurring research tasks** (about 3–4 days, separate plan). Builds on phase 5.

## Testing

| Level | What |
|---|---|
| pytest (offline) | The rules table. Decision logic with a fake model, so it's fast and doesn't need torch. Threshold, fallback and `TRIAGE_ENABLED=false` cases. The `/api/triage` contract and auth, including an empty query. Shadow mode: `/api/task` records a decision and still runs the workflow. Collection writes only when `WEBAGENT_TRIAGE_COLLECT` is on, and never to the curated files. The `triage_override` flag is counted. Startup checks fail in prod without the model. `previous_messages` reaches the supervisor prompt. A test that loads the real cached MiniLM and checks a handful of clear cases; it's skipped if the model isn't in the cache |
| Classifier eval (manual, before each release) | `backend/scripts/eval_triage.py --check-data`, then the gate on `test.jsonl`; must meet the targets |
| Jest | Routing for each mode: Off → `/api/chat`; Always → `/api/task`; Auto → triage, then the right route. Every row of the status-code table, and the 8,000-character cut. A mode change applies to the very next message without a reload. Migration: `true` → `auto`, `false` → no saved mode, and it runs only once. The new-user default comes from `GET /api/settings`, isn't saved, and follows a changed default on the next load. `/api/settings` falls back to `process.env` and rejects unknown values. The route label on every answer, and the override button only in Auto and not on image questions. `previous_messages` is sent to `/api/task`. No mock sources |
| pytest (phase 5) | `/api/task` and `/api/v1/chat/completions` with each mode; the old flags still work; `mode` wins over an old flag; neither field means `off`; `route` and `decided_by` appear in results; `off` and `auto` → direct answers without building the graph; an image in `auto` → direct with `image_ignored: true`; `/api/triage` and `mode: auto` give the same decision |
| Container run | The 3.0.3 real-run setup with ten research and ten routine questions in Auto mode; record each route, its time and its LangSmith trace |

## Success criteria

- Held-out test set: **≥ 95% research recall** (research questions routed agentic) and **≥ 80% of routine questions routed direct**.
- Triage latency **< 100 ms at p95** in the backend, and < 300 ms end to end from the web app.
- In Auto mode, routine questions finish in **< 10 s** (to be confirmed by the phase 1 timing), against 150 s or more today. Research questions are no slower than in Always mode, apart from the triage call itself.
- Override rate **< 10%** of direct answers after phase 4.
- No mock sources anywhere, and all existing tests still pass.

## Risks

| Risk | Mitigation |
|---|---|
| Short, time-sensitive questions are classified as routine (short text looks routine) | Include them deliberately in the training data; a low threshold for agentic; the override button; an optional LLM tie-breaker |
| The training data doesn't match real questions | Shadow mode collects real questions before acting; weekly retraining with overrides |
| Shadow data leans toward research (it only sees questions sent with Agentic on) | A deliberate mix of questions during the shadow week; the seed set supplies routine examples; Auto decisions and overrides add real ones after phase 3 |
| The classifier drifts as usage changes | Version every classifier, gate it on the held-out set, and watch the override and uncertain rates in Grafana |
| The image grows by about 90 MB | Acceptable next to its 2.77 GB size today |
| A research question gets a shallow direct answer | Low threshold, the override button on every direct answer in Auto, and the override metric |
| Each worker loads its own encoder: more RAM and slower startup | About 100–150 MB per worker; 2 workers within the 1 GiB limit, measured; one torch thread per worker; warm up at startup |
| API callers that relied on `/api/task` being agentic by default get direct answers after phase 5 | Only the web app calls it today, and it sends `agentic: true`; documented in the backend README and CHANGELOG |
| Collected questions contain personal data | Off by default outside dev, git-ignored, 30-day deletion, human review before anything joins the curated set |
| A saved default stops users following later default changes | Save `agenticMode` only when the user picks it; migrate `false` to "no saved mode" |

## Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Where do users with Agentic on today go? | **Auto** |
| 2 | Default for new users | **Off**, set by `WEBAGENT_DEFAULT_AGENTIC_MODE` in `.env.local` and read at runtime, so changing it needs no restart |
| 3 | Where "Research this in depth" appears | **Auto only**. It acts on one answered question, which the toggle can't do, and its clicks are the best training labels. Drop it if you'd rather keep the UI minimal; the cost is lost labels and a 3-step workaround |
| 4 | Backend modes for API clients | **Yes, phase 5**, as the base for recurring research tasks (phase 6) |
| 5 | Should the backend direct path send images to the model? | **No, text only for now.** Results set `image_ignored: true` |
| 6 | What does an API request with neither `mode` nor the old flag mean? | **`off`**, on every endpoint |
| 7 | Memory with an encoder in every worker | **2 uvicorn workers**, keeping the 1 GiB Kubernetes limit |

Still open (for the phase 6 plan): schedule ownership without user accounts, where results are delivered, and limits on scheduled runs.

## Files likely to change

- Backend:
  - `app/services/triage.py`, `app/services/triage_model.py`, `app/services/triage_collect.py`, `app/api/endpoints/triage.py` (all new)
  - `main.py` (mount `/api/triage`, warm-up), `app/api/endpoints/task.py` (`previous_messages` in phase 0, shadow, override), `app/services/task_manager.py`, `app/core/metrics.py`, `app/core/startup_checks.py`, `app/core/config.py` and `config_def.py`, `config/*.yaml` (`triage:` section)
  - `models/triage/` (new), `data/triage/` (new: `categories.json`, `train.jsonl`, `test.jsonl`, `README.md`; `collected/` git-ignored)
  - `Dockerfile` (bake the encoder, `WEB_CONCURRENCY=2`), `grafana/dashboards/webagent-backend-dashboard.json`, tests. No new Python dependencies before SetFit. Kubernetes already runs 2 workers, so `deployment.yaml` stays unless the memory check fails
- Scripts: `backend/scripts/train_triage.py`, `backend/scripts/eval_triage.py`, `backend/scripts/triage_data.py` (all new)
- Web app:
  - `src/app/api/triage/route.ts` (new), `src/app/api/settings/route.ts` (new), `src/app/api/task/backend.ts` (unreachable header)
  - `src/utils/apiClient.ts`, `src/utils/routeQuestion.ts` (new), `src/utils/agenticMode.ts` (new), `src/utils/storage.ts`
  - `src/contexts/ChatContext.tsx`, `src/contexts/AppContext.tsx`
  - `src/components/chat/AgenticModeToggle.tsx` (new), `src/components/chat/RouteLabel.tsx` (new), `src/components/chat/MessageInput.tsx`, `src/components/chat/MessageList.tsx`
  - `src/types/routing.ts` (new), `src/types/chat.ts`, `src/types/api.ts`, `src/test-utils.tsx`, tests
- Backend (phase 5): `app/services/routing.py` (new), `app/api/endpoints/frontend.py` (OpenAI-style route), `app/agents/supervisor.py` (`answer_directly`, the direct answer without the graph)
- Backend (phase 6): `app/scheduler/` (new), `app/api/endpoints/schedules.py` (new), `docker-compose.yml` and Kubernetes (scheduler service), `requirements.txt` (`croniter`)
- Repo: `.gitignore` (`backend/data/triage/collected/`)
- Docs: README, backend README, `docs/TESTING.md`, `.env.local.example` (`WEBAGENT_DEFAULT_AGENTIC_MODE`, `WEBAGENT_TRIAGE_*`), CHANGELOG
