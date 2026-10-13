# Requirements: security checkpoints

Each requirement has an ID, a statement, acceptance criteria, and how it is verified. "Must" is required for release; "should" is wanted but can slip. Phase numbers match [`tasks.md`](tasks.md).

Terms used throughout:

- **Hop**: one LLM call in a task or in the direct chat, together with the context it reads.
- **Context item**: one piece of variable text that goes into a prompt, with a **source** (`user_turn`, `history`, `agent_output`, `web_result`, `retrieved_doc`, `document`, `tool_output`, `image`, `unlabelled`). Template text written by us is not a context item.
- **Untrusted source**: `web_result`, `retrieved_doc`, `document`, `tool_output`, `unlabelled`.
- **Tool action**: `tool::web_search`, `tool::internal_retrieve`, `tool::document_extract`, `tool::execute_code`.
- **Outcome**: `allow`, `verify`, `consent`, `decline`, `escalate`. Severity: escalate > decline > consent > verify > allow.
- **Shadow mode**: a checkpoint scores, decides and records, but applies `allow`.
- **Tainted**: LLM output that may have been steered by untrusted content (SEC-038).

## 1. Groundwork (phase 0)

**SEC-001 One way to an LLM (must).** Every model an agent uses comes from `app/services/llm.py`. The direct `get_llm`, `get_llm_without_tags`, `ChatOpenAI(...)` and `openai` client uses in `internal_research.py`, `coding_assistant.py` and `coding_review.py` are replaced by factory calls that return guarded models.
- Accept: a test fails if any file under `app/agents/` imports `ChatOpenAI`, `openai`, or calls `get_llm`/`get_llm_without_tags` directly.
- Verify: pytest (static check of the source files).

**SEC-002 Trusted factory (must).** `get_trusted_llm()` returns an unguarded model for calls whose inputs are only trusted (the triage tie-breaker and the LLM judge). It is the only unguarded path, and each use is listed in the backend design.
- Verify: code review; the entry-point test allows `get_trusted_llm` only in `app/services/triage.py`, `app/security/judge.py` and `app/utils/diagnostics.py` (a fixed health-check prompt).

## 2. Decision point (phases 0–1)

**SEC-010 Cedar, in-process (must).** Decisions come from Cedar policies evaluated in the backend process through `cedarpy`. The decision point receives a request (principal, action, resource, context) and returns an outcome with the matching policy ids. It never receives raw context text.
- Verify: pytest (request has no text fields); `cedarpy` pinned in `requirements.txt` and `constraints.txt`.

**SEC-011 Five outcomes (must).** Non-allow rules are `forbid` policies annotated `@id("…")` and `@outcome("verify" | "consent" | "decline" | "escalate")`, plus one baseline `permit`. When Cedar denies, the outcome is the most severe `@outcome` among the matching policies.
- Accept: two matching policies with `decline` and `escalate` give `escalate`.
- Verify: policy tests.

**SEC-012 Policy ids by annotation (must).** The loader reads each policy's `@id` and `@outcome` and maps them to `cedarpy`'s positional ids (`policy0`, `policy1`…). Logs, metrics, results and tests use the `@id`. A policy without `@id`, or a `forbid` without `@outcome`, fails loading.

**SEC-012a Shadow-only policies (should).** A policy annotated `@shadow("true")` is evaluated on every request, and its matches are recorded (`Decision.shadow_policy_ids`, `webagent_guard_shadow_policy_total`, collection rows) but never change the outcome, whatever the checkpoint's mode. It is how a new rule is measured on real traffic before it acts. Another `@shadow` value fails loading.
- Verify: pytest (`policy_tests.yaml` rows with `expected_shadow`; bad annotation).
- Verify: pytest.

**SEC-013 Complete context (must).** Every request carries every context attribute in the schema, with defaults filled in by the decision point (scores 0, sets empty, `alignment` 100, `scores_available` true, `tainted` false, `consented` false, `has_image` false, `declined_actions_in_task` 0, `top_axis` "none"), so policies need no `has` checks.
- Verify: pytest (schema validation of built requests).

**SEC-014 Validated policies (must).** At load, policies are linted (every `forbid` has `@id` and `@outcome`; ids unique; only consent policies read `approved_tools` or `consented`; no consent policy applies to untrusted sources at `converse`), validated against `schema.cedarschema`, and the policy test table (`policy_tests.yaml`) is run. In uat and prod, a parse error, a validation error or a failing test stops startup; in dev, it logs an error and the guard runs in shadow mode.
- Verify: pytest (`startup_checks`).

## 3. ContextDefender (phase 2)

**SEC-020 Scores, not decisions (must).** ContextDefender returns, per context item, the calibrated probability that each of `injection`, `harm`, `sensitive_data` and `jailbreak` is present (labels `low` and `high` both count as present), each also as an integer 0–100 (`round(100 × P)`), the `top_axis` (`"none"` when every axis is 0), and per-chunk scores. It has no access to policies and returns no outcome.
- Verify: pytest (type and range).

**SEC-021 Fast patterns (must).** Before the classifier, a pattern set from `config/security/patterns.json` detects known injection phrases (`injection_phrase`), fake role or system markers (`role_marker`), hidden-text markers such as zero-width characters or invisible styling (`hidden_text`), and secret formats (`api_key`, `private_key`, `password`), reported as `pattern_hits`. Patterns run even when the classifier is down. No other values appear in `pattern_hits` at checkpoint 1.
- Verify: pytest table, one row per pattern plus near-misses.

**SEC-022 Routed features (must).** Natural-language text is embedded with `all-MiniLM-L6-v2`, frozen or fine-tuned for ContextDefender (with character n-grams added to the embedding by default); code, base64, URL-heavy or mostly non-dictionary text uses character n-gram TF-IDF alone. The route is recorded per chunk.
- Verify: pytest (router table).

**SEC-023 Long inputs (must).** Items longer than the window are chunked (MiniLM tokens; the window the model records in its artifacts, 96 tokens every 64 by default, 200 every 150 for older models), scored per chunk, and pooled per axis with the maximum. When a training row marks where an injection was inserted into a page (`span`), only the chunks containing it are labelled injection.
- Verify: pytest (an injection in the last chunk of a long page is found; its chunk index is reported).

**SEC-024 Calibrated heads (must).** Each head is a binary classifier (risk present or not) calibrated with Platt scaling: on 15% of the training rows held out from fitting when the encoder is fine-tuned, otherwise by 3-fold cross-validation (`CalibratedClassifierCV`, sigmoid). Expected calibration error ≤ 0.05 per head is a release criterion.
- Verify: `eval_context_defender.py`.

**SEC-025 threat_risk (must).** `threat_risk` = `round(100 × max over axes of weight[source][axis] × P(axis present))`, with weights from `config/security/source_weights.yaml` (start: 1.0 for untrusted sources, 0.6 for `user_turn` on `injection`, 0.8 for `history`).
- Verify: pytest.

**SEC-026 Cache (should).** Scores are cached per worker by SHA-256 of the item text and model version (LRU, about 10,000 entries).
- Verify: pytest (second call doesn't run the model).

**SEC-027 Curated data (must).** Training data lives in `backend/data/security/` (`train.jsonl`, `test.jsonl`, `probe.jsonl`), each row `{text, source, labels: {injection, harm, sensitive_data, jailbreak}, origin}` with labels `none|low|high`, plus `span` on rows with an inserted injection. The sets are built from public datasets by `scripts/fetch_security_sources.py` and `scripts/build_security_dataset.py`; adversarial text is never hand-written. Public sets are used only with a recorded, compatible licence (`data/security/README.md`). Splits are by near-duplicate cluster (cosine ≥ 0.9), so test and probe have no near-copies of training rows, and at least one source is held out of training for each of test and probe. Every realistic (axis, source) cell in `REQUIRED_CELLS` has positives.
- Verify: `eval_context_defender.py --check-data`.

**SEC-027a What counts as harm (must).** The `harm` axis covers requests for dangerous or prohibited help: criminal planning, violence, hate, harassment, weapons, self-harm, fraud, controlled substances, sexual content involving minors, threats, illegal activity and malware. Public data labelled unsafe for other reasons is relabelled: requests for someone's personal data become `sensitive_data`; profanity, controversial or political topics, unqualified advice and "needs caution" questions are benign look-alikes; other categories are left out (data README).
- Verify: pytest (`aegis_group`), `--check-data`.

**SEC-028 Release gate (must).** A ContextDefender version ships only if, on `test.jsonl`: injection recall ≥ 95% on untrusted sources at ≤ 2% false positives on benign pages; harm and jailbreak recall ≥ 90% on user turns, counting every turn sent to decline, consent or the judge (scores at or above the verify threshold), at ≤ 1% direct declines or consents of ordinary benign questions (user turns not from the look-alike sets in `LOOKALIKE_ORIGINS`; the look-alike rates and how often the judge is asked are reported, not gated); sensitive-data recall ≥ 95% for secrets and ≥ 85% for personal data at ≤ 2% false positives; ECE ≤ 0.05 per head. Benign rows are those with every axis `none`, from any dataset; a check with no rows in the test set fails. The result is written to `metadata.json["test"]` with `passed`.
- Verify: `eval_context_defender.py --write-metadata` (exit code 1 on failure).

**SEC-028a Per-model thresholds (must).** Each model version chooses, on held-out training rows, the lowest score per axis whose false-positive rate meets the gate's limit by the one-sided 95% Wilson upper bound: `injection_drop_at` on benign untrusted items (watch at half of it), `harm_decline_at` and `jailbreak_consent_at` on ordinary questions. (Until 2026-10-13 injection used the point estimate under 75% of the limit, while its held-out set was small.) Below those, `harm_verify_at` and `jailbreak_verify_at` are the highest scores that still reach 92% of the held-out harmful or jailbreak user turns. `scripts/train_context_defender.py --reselect-thresholds` re-chooses them for a trained model without retraining. They are stored in `metadata.json["thresholds"]`, passed into the checkpoint-1 policy context, and used by the policies and by the gate (other checks use 50). Without a model, the defaults are 40/20, 85 and 70, with verify from 50.
- Verify: pytest (selection; context; policy rows with non-default thresholds in `policy_tests.yaml`).

## 4. Checkpoint 1: pre-LLM (phase 3)

**SEC-030 Labelled items (must).** Agents build prompts from labelled context items; template text is never a context item and is never scored.
- Verify: pytest (scoring calls receive only item text; a template full of instructions doesn't raise `injection`).

**SEC-031 check_hop (must).** Before each LLM call, the agent calls `check_hop(agent, items)`. It runs patterns, ContextDefender and the decision point per item, and returns the items to use (flagged chunks removed), the hop's decisions and its taint. The agent formats its prompt from the returned items.
- Verify: pytest per agent (a flagged chunk is absent from the formatted prompt).

**SEC-032 Safety net (must).** The guarded model checks, through a context variable, that `check_hop` ran for the current call. If not, it scores the formatted prompt minus the registered template text as one `unlabelled` item, applies the outcome, logs a warning naming the agent and counts it (`webagent_guard_unlabelled_total{agent}`).
- Verify: pytest.

**SEC-033 User turn first (must).** `POST /api/task` runs checkpoint 1 on the user turn and `previous_messages` before creating the task. Consent, decline and escalate are answered at once (SEC-036), and no task is created.
- Verify: pytest.

**SEC-034 Guard endpoint (must).** `POST /api/guard/input` (FastAPI, `X-API-Key`, rate-limited) takes `{messages: [{role, content}], has_image, consent_token?}` and returns `{outcome, reason, policy_ids, consent_token?, decision_id}`. It runs checkpoint 1 on the last user message (`user_turn`) and the earlier ones (`history`).
- Verify: pytest (contract, auth, 422 on bad input).

**SEC-035 Images (must).** An attached image is recorded as an item with `source = "image"` and no scores. Policies see it in the context (`has_image`), and nothing blocks on it by default.
- Verify: pytest.

**SEC-035a Judge for borderline user turns (should).** A user turn (or history item, for harm) scoring from the model's verify threshold up to its decline or consent threshold matches `user-harm-verify` or `user-jailbreak-verify`. With `WEBAGENT_GUARD_JUDGE=true`, a judge is asked about the message alone: for harm, OpenAI's moderation endpoint by default (`harm_judge`; `llm` and `off` are the alternatives), flagging when its highest category score reaches the model's `harm_moderation_cutoff` (chosen on held-out user turns so that estimated end-to-end declines of ordinary questions stay under 75% of the harm limit; moderation's own `flagged` when no cut-off is set); for jailbreak, Meta's Llama Prompt Guard 2 run locally (`jailbreak_judge: prompt_guard`, flagged at `prompt_guard_threshold`, default 0.5; nothing leaves the server and it can't refuse) once its gated model is installed, else `get_trusted_llm()` with the message between markers, as data, answering `JAILBREAK` or `FINE` (`llm`, the current default). A chat-model reply refusing to judge, or a provider error naming a content policy, counts as flagged. Harmful → decline (403); jailbreak → consent (409), or allow if the user already consented; fine → allow without taint. A 3 s timeout, an error or any other answer leaves `verify` (continue, tainted), failing open as for other user-turn errors (SEC-061). Verdicts are cached by (text hash, axis). In shadow mode the judge's decision is recorded, not applied.
- Verify: pytest with a fake LLM (flag, clear, consented, timeout, unknown answer, judge off, shadow, cache).

**SEC-036 Outcomes at checkpoint 1 (must).**

| Outcome | User turn or history | Untrusted item |
|---|---|---|
| allow | Continue | Keep |
| verify | Harm or jailbreak verify policy with the judge on: the judge decides (SEC-035a). Otherwise continue; the task is tainted | Keep; the hop is tainted |
| consent | `409 {code: "consent_required", reason, consent_token}`; nothing runs | Not allowed: the policy lint rejects a consent policy that applies to untrusted sources (SEC-014) |
| decline | `403 {code: "declined", reason}`; no LLM call | Drop the flagged chunks (the whole item if over half its chunks are flagged); the hop is tainted |
| escalate | `403 {code: "escalated", reason}`; recorded for review | Drop the item; recorded for review; the task continues |

- Verify: pytest per row.

**SEC-037 Report notes (must).** Every dropped item or chunk is listed in the task's `security` block with its source, URL or title (no text), the policy id and a one-line reason.
- Verify: pytest.

**SEC-038 Taint (must).** A hop is tainted when it kept an untrusted item with outcome `verify`, or dropped any part of an untrusted item. A tainted hop's outputs are `agent_output` items marked tainted; any hop that reads a tainted item is tainted. A consent-confirmed user turn taints the task. Taint lasts until the task ends, and a tool action proposed by a tainted hop has `tainted = true` at checkpoint 2.
- Verify: pytest (taint crosses supervisor → web research → senior research → search).

**SEC-039 Parallel hops (must).** Security records from research agents that run in parallel are merged into the task state (union of hops, taint is OR, counters summed).
- Verify: pytest with two fake parallel agents.

## 5. Checkpoint 2: pre-tool (phase 1)

**SEC-040 Every call site (must).** `guard_tool(action, args, state)` runs before: the three `perform_web_search` calls in `web_research.py` (main, simplified retry, follow-up), both `retrieve` calls in `internal_research.py`, the document load in `document_extraction_agent.py`, and **each** `run_python_file` call in `coding_review.py`. A non-allow outcome means the call does not run.
- Verify: pytest (each site, with a fake tool that fails the test if called after decline).

**SEC-041 Search and retrieval checks (must).** Arguments are checked for secret formats (`api_key`, `private_key`, `password`), emails (`email`), phone numbers (`phone`), long base64/hex (`long_token`), URLs (`url`), `site:` operators (`site_operator`) and length > 300 characters (`query_too_long`), reported in `pattern_hits`. A match whose text appears in the **current user turn** is not reported.
- Verify: pytest table (user-typed email allowed; email from history reported).

**SEC-042 Document provenance (must).** `tool::document_extract` is allowed only for the `document_id` on the task request; any other ID gets `"document_provenance"` in `pattern_hits`, which the `document-provenance-decline` policy turns into decline.
- Verify: pytest.

**SEC-043 Code checks (must).** Before every execution attempt, the code's AST is checked and findings reported in `code_findings`: `network` (socket, http clients, urllib, requests…), `process` (subprocess, os.system, os.exec*), `filesystem` (writes outside the work folder, path traversal), `eval` (eval, exec, compile, `__import__`), `native` (ctypes, cffi), `env` (os.environ), `unparseable`. The context also carries `code_execution_enabled` and `sandbox`.
- Verify: pytest table.

**SEC-044 Alignment (must).** For search and retrieval, `alignment` = round(100 × cosine) between the arguments and the task text (the supervisor's `standalone_query`, else the user turn), with ContextDefender's encoder. Code and document actions get 100.

**SEC-044a Misaligned searches in untainted tasks (should).** Injections checkpoint 1 misses leave the task untainted, so `tainted-search-verify` doesn't apply; on InjecAgent such searches had alignment median 9–16 (`scripts/eval_checkpoint2_injecagent.py`). `untainted-search-misaligned` (verify below 30, untainted) runs shadow-only until real searches' alignment (`webagent_guard_search_alignment{tainted}`) shows how many legitimate follow-up searches it would hold; then it gets a measured threshold and acts.
- Verify: pytest (policy rows); shadow-mode metrics before enabling.
- Verify: pytest.

**SEC-045 LLM judge (should).** With `WEBAGENT_GUARD_JUDGE=true`, `verify` at checkpoint 2 asks `get_trusted_llm()` for `ALIGNED`/`NOT_ALIGNED` with only the task text and the proposed call. A 3 s timeout, an error or any other answer is `NOT_ALIGNED`. Results are cached per task by (action, arguments). With the judge off, `verify` becomes `decline`.
- Verify: pytest with a fake LLM.

**SEC-046 Outcomes at checkpoint 2 (must).** allow: run. verify: judge (SEC-045). consent: skip, record, offer approval (SEC-051). decline: skip, record, continue. escalate: stop the task with `status: failed` and `error_code: "escalated"`, and record for review.
- Verify: pytest per outcome.

**SEC-047 Decline counter (must).** `declined_actions_in_task` counts declined tool actions and declined user turns in the task. Dropped content items don't count.
- Verify: pytest (five dropped web results leave it at 0).

## 6. Tokens (phase 4)

**SEC-050 Consent token (must).** A `consent_required` answer carries a token: an HMAC-SHA256 (key from `WEBAGENT_SECURITY_SECRET_KEY`) over `{kind: "consent", text_hash: sha256(user turn), policy_id, exp}`, with a 15-minute expiry. Resending the same user turn with the token sets `consented = true` and taints the task. A token is single use.
- Verify: pytest (wrong text, expired, reused, tampered → rejected).

**SEC-051 Approval token (must).** A task result with a consent-skipped action carries `approvals: [{action, policy_id, approval_token}]`. The token is an HMAC over `{kind: "approval", query_hash, action, policy_id, exp}`. Re-running the task with the token adds the action to `approved_tools` for that run only. Single use.
- Verify: pytest.

**SEC-052 Approval never overrides decline (must).** `approved_tools` and `consented` are read only by consent policies. Policy validation fails if a `decline` or `escalate` policy reads them.
- Verify: policy lint test.

**SEC-053 Single use across workers (must).** Used token IDs are recorded in `data/security/tokens/` (one file per ID, created with `O_EXCL`), so any worker rejects a reused token. Files older than the expiry are cleaned up daily.
- Verify: pytest with two `TokenStore` instances on one folder.

## 7. Escalation, failures, modes (phases 1–4)

**SEC-060 Escalation queue (must).** Each escalation appends one JSON line to `data/security/escalations/YYYY-MM-DD.jsonl` (git-ignored, 30-day retention): time, task or chat ID, hop, action, policy ids, scores, tool and arguments, and text only in dev (SHA-256 elsewhere). It increments `webagent_guard_escalations_total`.
- Verify: pytest.

**SEC-061 Failure handling (must).**

| Failure | Checkpoint 1 | Checkpoint 2 |
|---|---|---|
| ContextDefender unavailable | Patterns run; `scores_available = false`; policies decide | Deterministic checks run |
| Decision point error | Drop untrusted items; allow user turns; error logged | Decline `execute_code`; allow search and retrieval; error logged |
| Judge timeout or error | User turn in a verify band: verify stands (continue, tainted) | `NOT_ALIGNED` |
| Token store unavailable | No consent | Not approved |

- Verify: pytest per row.

**SEC-062 Shadow and act (must).** `WEBAGENT_GUARD_INPUT_MODE` and `WEBAGENT_GUARD_TOOL_MODE` are `shadow` (default) or `act`. In shadow mode the outcome is computed, recorded (metrics, logs, `security` block with `shadow: true`, collection) and then treated as `allow`. `WEBAGENT_GUARD_ENABLED=false` skips both checkpoints entirely.
- Verify: pytest.

**SEC-063 Configuration (must).** A `security_guard:` section in `config/{dev,uat,prod}.yaml` with `enabled`, `input_mode`, `tool_mode`, `judge`, `collect`, `collect_retention_days`, `model_dir`, `policy_dir`; `WEBAGENT_GUARD_*` environment variables override it like the triage settings. Thresholds live only in policies.
- Verify: pytest (each override; bad values warn and keep YAML).

**SEC-064 Collection is opt-in outside dev (must).** With `WEBAGENT_GUARD_COLLECT=true` (default only in dev), shadow and non-allow decisions are written with text to `data/security/collected/YYYY-MM-DD.jsonl` (git-ignored, 30-day retention). Training scripts never read `collected/`.
- Verify: pytest.

**SEC-065 Startup checks (must).** In uat and prod with the guard enabled, startup fails if the ContextDefender model (once phase 3 ships), its encoder, the policies or the schema are missing or invalid, or the model hasn't passed its gate.
- Verify: pytest (`startup_checks`).

## 8. Observability (phases 1–4)

**SEC-080 Metrics (must).** `webagent_guard_decisions_total{checkpoint, action, outcome, shadow}`, `webagent_guard_latency_seconds{checkpoint}`, `webagent_contextdefender_score{axis, source}` (histogram of 0–100), `webagent_guard_judge_total{verdict}`, `webagent_guard_unlabelled_total{agent}`, `webagent_guard_escalations_total`, `webagent_guard_shadow_policy_total{policy_id, action}`, `webagent_guard_search_alignment{tainted}` (histogram of 0–100 for search and retrieval), `webagent_guard_hop_seconds{agent}` and `webagent_guard_tool_seconds{action}` (whole-checkpoint time, for SNFR-01).
- Verify: pytest reads the registry.

**SEC-081 No text in logs outside dev (must).** One log line per non-allow decision (checkpoint, action, outcome, policy ids, top axis and score, task ID); text only when `WEBAGENT_ENV=dev`.
- Verify: pytest with `caplog`.

**SEC-082 Security block (must).** Every task result has `security: {tainted, shadow, hops: [...], dropped: [...], skipped: [...], approvals: [...]}`; the direct chat response has `security: {outcome, reason}` when not allow.
- Verify: pytest.

**SEC-083 Trace metadata (should).** Decisions are attached to the LangSmith run of the hop as metadata (`guard.outcome`, `guard.policy_ids`).
- Verify: pytest with a fake tracer.

**SEC-084 Dashboard (should).** A Grafana panel shows decisions by checkpoint and outcome, latency p95 and escalations.

## 9. Web app (phases 3–4)

**SEC-090 Guarded direct chat (must).** `src/app/api/chat/route.ts` calls `POST /api/guard/input` server to server (with `X-API-Key` and the caller's forwarded address, 3 s timeout) before calling Together. Allow and verify continue; consent, decline and escalate return the backend's status and body to the client, and Together is not called.
- Verify: Jest (route test).

**SEC-091 Fallback (must).** If the guard is unreachable (network error, timeout or 503) or answers 401, 422 or another 5xx (also logged as an error), the route applies the `injection_phrase` and `role_marker` patterns from its copy of `patterns.json` to the user's message, refuses on a hit with `403 {code: "declined"}`, and otherwise continues with the existing fallback notice.
- Verify: Jest; a Jest test fails if the web app's copy of `patterns.json` differs from the backend's.

**SEC-092 Consent dialog (must).** On `409 consent_required` (from `/api/chat` or `/api/task`), the web app shows the reason with "Send anyway" and "Edit message". "Send anyway" resends the same message with the token; "Edit message" puts the text back in the input.
- Verify: Jest.

**SEC-093 Declined and escalated (must).** On `403 declined` or `escalated`, the user's message stays visible and the reason is shown in place of an answer; nothing is retried automatically.
- Verify: Jest.

**SEC-094 Security summary (must).** Under an agentic answer with a non-empty `security` block, a one-line summary ("2 web results left out · 1 search skipped"); hover or focus shows the reasons. Consent-skipped actions show "Approve and re-run", which re-sends the task with the approval tokens.
- Verify: Jest.

## 10. Non-functional requirements

| ID | Requirement | Verify |
|---|---|---|
| SNFR-01 | Checkpoint 1 adds < 150 ms p95 per hop (five web results); checkpoint 2 < 20 ms p95 per tool call without the judge | Metric histogram; container run |
| SNFR-02 | A decision takes < 2 ms p95 | pytest benchmark |
| SNFR-03 | Red-team suite: zero exfiltrating searches and zero unsafe code executed, and the tasks still finish | Red-team pytest |
| SNFR-04 | Benign regression: false-decline rate < 0.5% of tasks (triage training questions and 3.0.3 container-run questions) | Benign suite |
| SNFR-05 | Two workers with triage and ContextDefender fit the 3 GiB Kubernetes limit with 20% headroom (2 GiB until 3.2.5; two workers peaked at 1.81 GiB in the simulated shadow week) | cgroup stats in the container |
| SNFR-06 | Classifier and decisions are deterministic: same input, model and policies → same outcome | pytest |
| SNFR-07 | New dependencies: only `cedarpy` | `requirements.txt` diff |
| SNFR-08 | All tests run offline (fakes for LLMs, Tavily and the judge); existing tests still pass | CI |
| SNFR-09 | No raw context text leaves the backend in metrics or logs outside dev | SEC-081 tests; review |
