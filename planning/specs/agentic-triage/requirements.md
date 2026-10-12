# Requirements: agentic triage

Each requirement has an ID, a statement, acceptance criteria, and how it is verified. "Must" is required for release; "should" is wanted but can slip. Phase numbers match [`tasks.md`](tasks.md).

Terms used throughout:

- **Route**: which path answered a question. `direct` (one LLM call through `/api/chat`, or the backend's direct path) or `agentic` (the LangGraph workflow).
- **Mode**: the user's or caller's setting. `off`, `auto` or `always`.
- **Decided by**: why a route was chosen. One of `rule`, `model`, `fallback`, `config`, `mode`, `override`.
- **Research recall**: of the questions labelled agentic, the share routed agentic.
- **Threshold**: the `p_agentic` value at or above which the model routes agentic. Set by training (REQ-022): the highest value up to 0.5 that keeps 95% research recall on validation. The plan's first estimate was 0.35; `triage-2026-10-a` uses 0.5.
- **Tie-break band**: `threshold ≤ p_agentic < 0.65`, where the optional LLM tie-breaker may run.
- **Uncertain**: `|p_agentic − threshold| < 0.15`. Only used for the `webagent_triage_uncertain_total` metric.

## 1. Fix existing problems (phase 0)

**REQ-001 No invented sources (must).** Direct answers must not show a source list. `generateMockSources` is removed. Agentic answers keep the sources the workflow returns.
- Accept: with "cite sources" on and the route direct, the stored message has no `sources` field, and nothing renders a source card.
- Verify: Jest (`ChatContext` test).

**REQ-002 Agentic requests carry the conversation (must).** `POST /api/task` accepts `previous_messages`: a list of at most 8 `{role, content}` items, oldest first, not including the current question. The backend puts it in the workflow context as `previous_messages`, where the supervisor's existing `_with_history` already reads it. The supervisor prompt keeps its single `{query}` variable.
- Accept: a follow-up such as "what about Europe?" after a question about the US reaches the supervisor prompt with the earlier turns included.
- Accept: more than 8 items, or a role other than `user` / `assistant`, is rejected with 422.
- Verify: pytest (request model, context, supervisor prompt input); Jest (request body).

**REQ-003 Images and the agentic workflow (must).** The agentic workflow does not read images. In Auto, a question with an image routes direct (`decided_by: rule`), where `/api/chat` sends the image to the model. In Always (Agentic on, before phase 3), the UI shows a note that the image is only used by direct answers, and the question is still sent agentic. An image sent with no text goes direct in every mode (`decided_by: rule`, reason "Image without a question"), because the agents would have nothing to research.
- Verify: Jest; pytest (rule table).

## 2. Triage service (phases 1–2)

**REQ-010 Triage endpoint (must).** FastAPI exposes `POST /api/triage`, behind `X-API-Key` and the normal rate limit. Request: `query` (string, 0–8,000 chars; empty is allowed so the empty-message rule can answer), `previous_user_question` (string or null, ≤ 8,000 chars), `has_image` (bool). Response: `route`, `p_agentic` (0–1, or null when no model ran), `decided_by`, `reason`, `classifier_version`.
- Verify: pytest (contract, 401 without key, 422 on over-long input).

**REQ-011 Next.js proxy (must).** `POST /api/triage` in the web app forwards to the backend with the server's API key, like `/api/task`. The browser never sees the key. When the backend can't be reached, the proxy returns 503 with the header `X-WebAgent-Proxy-Error: unreachable`.
- Verify: Jest (route test).

**REQ-012 One source of truth (must).** All rules, thresholds and the model live in `backend/app/services/triage.py` (with model loading in `triage_model.py`). The web app never classifies. `/api/triage`, shadow mode and the phase 5 `auto` mode all call the same `classify` function.
- Verify: code review; in phase 5, pytest asserts `/api/triage` and `/api/task` with `mode: auto` return the same decision for the same input.

**REQ-013 Rules before the model (must).** These run first, in this order, and skip the model:
1. `has_image` is true: `direct`.
2. The text is empty, or the whole message is a greeting or thanks (after removing punctuation, trimming and lowercasing): `direct`.
3. The text starts (after an optional "please") with an explicit research request: "research", "find sources", "find studies", "write a report", "do a deep dive" or "give me a literature review": `agentic`.
- Verify: pytest table test, one row per rule and several near-misses.

**REQ-014 Model decision (must).** Otherwise the model computes `p_agentic`. If the question has fewer than 6 words and there is a previous user question, the model input is `"<previous> → <current>"`. The route is `agentic` when `p_agentic ≥ threshold`, else `direct`. The threshold comes from `metadata.json`, overridable by `WEBAGENT_TRIAGE_THRESHOLD`.
- Verify: pytest with a fake model.

**REQ-015 Fail toward research (must).** If the model can't load or raises, `classify` returns `agentic` with `decided_by: fallback`, and the endpoint answers 200. `classify` never raises to the caller.
- Verify: pytest.

**REQ-016 Kill switch (must).** `WEBAGENT_TRIAGE_ENABLED=false` makes `classify` return `agentic` with `decided_by: config` without loading the model. Auto then behaves like Always.
- Verify: pytest.

**REQ-017 Optional LLM tie-breaker (should).** When `WEBAGENT_TRIAGE_LLM_TIEBREAK=true` and `p_agentic` is in the tie-break band, one LLM call decides. Off by default. If the call fails or takes more than 3 s, keep the model's route.
- Verify: pytest with a fake LLM.

**REQ-018 Reasons (must).** `reason` is a short, human-readable string: the rule that fired (e.g. "Greeting or short acknowledgement"), the nearest labelled category for model decisions (e.g. "Looks like: definition of a stable concept"), or the fallback cause (e.g. "Classifier unavailable"). It never contains the question text.
- Verify: pytest.

## 3. Dataset, training and evaluation (phase 1, phase 4)

**REQ-020 Curated data (must).** `backend/data/triage/train.jsonl` and `test.jsonl`, one JSON object per line with `text`, `label` (`agentic` | `direct`), `category`, `source` (`seed` | `generated` | `collected`). Category ids and labels live in `backend/data/triage/categories.json`.
- The test set has about 150 rows, stratified by category, **written by hand separately from the seed set** (not paraphrases of training rows), and is never used for training or threshold picking.
- No test row is a near-duplicate of a training row (cosine similarity of embeddings ≥ 0.9).
- Both files are in git and contain no personal data.
- Verify: `scripts/eval_triage.py --check-data` (schema, labels, categories, exact and near duplicates across files, minimum counts per category).

**REQ-021 Hard cases are covered (must).** The data deliberately includes short time-sensitive questions ("bitcoin price now?"), long routine questions (for example, pasted text to rewrite), casually phrased research, and joined follow-ups (`previous → current`). Each of these categories has at least 20 training rows and 10 test rows.
- Verify: `--check-data`.

**REQ-022 Training script (must).** `scripts/train_triage.py` embeds the training set, fits logistic regression, and picks the **highest** threshold (capped at 0.5) that keeps research recall ≥ 95% on a validation split. It writes `backend/models/triage/classifier.joblib` (the classifier and per-category centroids, under 100 KB) and `metadata.json` (version, encoder name, threshold, data hash, category labels, validation and test metrics). Runs offline in under 2 minutes on a laptop CPU.
- Verify: pytest smoke test on a tiny dataset with a fake encoder.

**REQ-023 Release gate (must).** `scripts/eval_triage.py` prints recall, precision and a per-category confusion matrix on `test.jsonl`, and exits non-zero unless research recall ≥ 95% and routine-routed-direct ≥ 80%. A classifier version ships only if the gate passes.
- Verify: run before each release; result recorded in `metadata.json["test"]` (with `passed`). In uat and prod, `startup_checks` refuses a classifier whose `test.passed` isn't `true` (NFR-08).

**REQ-024 Fine-tuning path (should, phase 4; done 2026-10-11).** `train_triage.py --finetune` (alias `--setfit`) fine-tunes the encoder SetFit-style on the same data, using sentence-transformers only. The fine-tuned encoder is stored in `models/triage/encoder/` with its weights in Git LFS, not in plain git.
- Verify: pytest (pair building); gate and `probe.jsonl` on the trained version.

## 4. Collection and privacy (phase 2)

**REQ-030 Collection is opt-in outside dev (must).** When `WEBAGENT_TRIAGE_COLLECT` is true (default true in dev, false in uat and prod), shadow decisions and override clicks are appended, with question text, to `data/triage/collected/YYYY-MM-DD.jsonl` under the backend folder (`/app/data/triage/collected/` in a container, on the data volume). The repo path `backend/data/triage/collected/` is git-ignored. Nothing is written when collection is off.
- Verify: pytest.

**REQ-031 Curated files are never written by the server (must).** The server writes only under `collected/`. `train_triage.py` reads only `train.jsonl`, and `eval_triage.py` only `test.jsonl` (plus `train.jsonl` for `--check-data`).
- Verify: pytest asserts no writes outside `collected/`.

**REQ-032 Retention (must).** Collected files older than 30 days are deleted. The cleanup runs at startup and once a day, whether or not triage is enabled (overrides are collected either way).
- Verify: pytest with fake dates.

**REQ-033 Logs carry no question text outside dev (must).** One log line per decision with route, `p_agentic`, `decided_by`, version and latency. Question text only when `WEBAGENT_ENV=dev`.
- Verify: pytest with `caplog`.

## 5. Shadow mode (phase 2)

**REQ-040 Shadow mode (must).** With `WEBAGENT_TRIAGE_MODE=shadow`, `POST /api/task` classifies every request that is going to the workflow because the caller asked for it (`agentic: true`, or `mode: always` from phase 5), except overrides. It records the decision (metrics, logs, collection) and stores it on the task result as `triage` with `shadow: true`, and still runs the full workflow. A triage error never fails the task.
- `WEBAGENT_TRIAGE_MODE` controls only this recording. `POST /api/triage` always answers, and `mode: auto` (phase 5) always acts on the decision.
- Verify: pytest.

## 6. Web app (phase 3)

**REQ-050 Three-way toggle (must).** The Agentic button becomes a three-way control labelled Agentic: Off · Auto · Always, with these tooltips, shown on hover and keyboard focus: "Answer directly, no research agents", "Use research agents when the question needs them", "Always use research agents (slower)". It is keyboard accessible (arrow keys move between options, `aria-checked` on the selected one).
- Verify: Jest + manual.

**REQ-051 Stored mode (must).** The setting is `agenticMode: 'off' | 'auto' | 'always'`, saved in its own storage key (not inside the accessibility settings), and written only when the user picks a mode. "No saved mode" means "follow the server default".
- Verify: Jest.

**REQ-052 Migration (must).** On first load after upgrade: saved `agentic: true` → save `agenticMode: 'auto'`; `agentic: false` or missing → no saved mode. The old `agentic` key is removed from the saved settings, so the migration runs once.
- Verify: Jest (three cases, plus "runs only once").

**REQ-053 Runtime default (must).** `GET /api/settings` returns `{"defaultAgenticMode": "off" | "auto" | "always"}`. It reads `WEBAGENT_DEFAULT_AGENTIC_MODE` from `.env.local`, re-reading when the file's modified time changes; if the file doesn't exist, it reads `process.env`. Missing means `off`. An unknown value logs a warning and returns `off`. No API key is needed. The route is never cached at build time.
- Accept: editing `.env.local` changes the default on the next page load with no restart.
- Verify: Jest (file present, file changed, file missing, bad value).

**REQ-054 First paint (must).** Until the mode is known (saved mode read, or `/api/settings` answered), the toggle shows Off and sending is disabled. If the route fails, the default is Off.
- Verify: Jest.

**REQ-055 Changes apply at once (must).** Switching mode affects the very next message, with no reload.
- Verify: Jest.

**REQ-056 Routing per mode (must).** Off → `/api/chat`. Always → `/api/task`. Auto → `/api/triage`, then `/api/chat` or `/api/task` by its answer, following the status-code table in the high-level design. The client sends at most the first 8,000 characters of the question to triage.
- Verify: Jest, one test per table row.

**REQ-057 Route label (must).** Every new assistant message stores `route: {route, decidedBy, reason}`, and shows "Answered directly" or "Deep research". Hover shows the reason, e.g. "Auto · Looks like: definition of a stable concept" or "Always mode". Workflow logs show for any message whose route is agentic.
- Verify: Jest.

**REQ-058 Override (must).** In Auto, direct answers to questions without an image show "Research this in depth". Clicking it sends that question and the messages before it to `/api/task` with `triage_override: true`, skipping triage. The backend counts the click and, when collection is on, records the question. The research answer is inserted as a new message directly below the direct answer (after any earlier research answers to the same question), and shown under the same question; the direct answer stays. Not shown in Off or Always.
- Verify: Jest; pytest (flag counted and collected).

## 7. Backend modes for API clients (phase 5)

**REQ-070 Mode field (must).** `POST /api/task` and `POST /api/v1/chat/completions` accept `mode: "off" | "auto" | "always"`. Old flags still work (`agentic` / `agentic_toggle`: `true` → `always`, `false` → `off`). `mode` wins when both are given. **No mode and no flag means `off`** on both endpoints.
- `/api/v1/chat/completions` already defaults to non-agentic, so its callers see no change.
- `/api/task` changes from agentic to `off` when neither field is sent. The web app always sends `agentic: true`, so it is not affected.
- Verify: pytest, one test per combination.

**REQ-071 Off and Auto-direct skip the graph (must).** The direct path makes one text-only model call with the question and `previous_messages`. It does not build the LangGraph workflow. Images are not sent to the model on either path for now; when a request includes an image, the task result has `image_ignored: true`.
- Verify: pytest asserts the workflow factory is not called, and checks `image_ignored`.

**REQ-072 Route on results (must).** Every task result includes `route`, `decided_by` and, when the model ran, `p_agentic`. The polling contract is unchanged.
- Verify: pytest.

**REQ-073 Deprecations (should).** `POST /api/v1/chat/` and `POST /api/v1/tasks/create` are marked deprecated in OpenAPI; behavior is unchanged.

## 8. Observability (phase 2)

**REQ-080 Metrics (must).** Prometheus: `webagent_triage_decisions_total{route, decided_by}`, `webagent_triage_latency_seconds`, `webagent_triage_overrides_total`, `webagent_triage_uncertain_total` (model decisions that are uncertain, as defined above).
- Verify: pytest reads the registry.

**REQ-081 Dashboard (should).** A Grafana panel shows the route split, override rate and uncertain rate. The override rate is overrides divided by direct decisions made by triage (`decided_by` other than `mode`), since Off-mode answers can't be overridden.

## 9. Non-functional requirements

| ID | Requirement | Verify |
|---|---|---|
| NFR-01 | Triage latency < 100 ms at p95 inside the backend, < 300 ms end to end from the browser | Metric histogram; container run |
| NFR-02 | In Auto, routine questions finish in < 10 s (confirm the target with the phase 1 timing of `/api/chat`) | Container run, 10 routine questions |
| NFR-03 | In Auto, research questions are no slower than Always, except for the triage call | Container run, 10 research questions |
| NFR-04 | Held-out test: research recall ≥ 95%, routine routed direct ≥ 80% | `eval_triage.py` |
| NFR-05 | Override rate < 10% of direct answers after phase 4 | Grafana |
| NFR-06 | Each worker's extra memory for triage ≤ 150 MB. The backend runs **2 uvicorn workers** (image default and Kubernetes) and fits the Kubernetes memory limit (1.5 GiB) with 20% headroom | Measure with cgroup stats in the container |
| NFR-07 | Classifier is deterministic: same input and version → same output | pytest |
| NFR-08 | In uat and prod, the server refuses to start without the classifier and encoder, or with a classifier that hasn't passed the release gate, unless triage is disabled. In dev, it starts with a warning and triage falls back to agentic | pytest (`startup_checks`) |
| NFR-09 | No new Python dependency (fine-tuning uses sentence-transformers' own trainer). The image grows by ≤ 100 MB over 3.0.3 | Diff of `requirements.txt`; `docker image ls` |
| NFR-10 | All existing tests still pass; new code has unit tests | CI |
