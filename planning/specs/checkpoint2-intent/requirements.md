# Requirements: checkpoint 2 checked against the user's intent

Each requirement has an ID, a statement, acceptance criteria and how it is checked. All are
"must" unless marked "should". IDs continue the agentic-security spec (`SEC-0xx`, `SNFR-01`–`09`):
`SEC-1xx` and `SNFR-1x` here. Phase A is release 3.7.0 (before the shadow week); phase B the
classifier (3.8.0); phase C switching rules from shadow to acting.

Terms:

- **Intent**: the user's own text for the task: the current turn and up to three previous turns
  with `role: user`. Never agent output.
- **Subgoal**: agent-written text that steers later actions: the supervisor's rewrite
  (`standalone_query`), its plan questions, and the senior research agent's follow-up
  `research_questions`.
- **Call**: a checkpoint 2 tool action with its arguments (`tool::web_search`,
  `tool::internal_retrieve`, `tool::document_extract`, `tool::execute_code`).
- **Private text**: internal retrieval results and the attached document's text, seen during the
  task.
- **Shadow rule**: a Cedar policy with `@shadow("true")`: its matches are recorded and never change
  the outcome.

## 1. Intent and subgoals (phase A)

**SEC-100 Intent record (must).** When the research workflow's state is built
(`TaskManager.run_workflow`), the security ledger gets `intent_text` (the current turn and up to
three previous `role: user` turns from `previous_messages`, newest last, capped at 2,000
characters); its base MiniLM-L6 embedding and `intent_harm` (ContextDefender's harm score of the
current turn) are computed on first use and cached in the same record, so starting a task loads no
model. It is never changed during the task and never built from agent output. The direct path (no
tools) has none.
- Check: pytest: assistant turns excluded; cap applied; unchanged after agents run; harm from
  ContextDefender (fake model).

**SEC-101 Rewrite check (must).** After the supervisor sets `research_plan`, the guard decides
`plan::rewrite` with `rewrite_alignment` (the rewrite against the intent; 100 when there is no
rewrite or it equals the turn). In act mode, a non-allow outcome empties `standalone_query`, so
searches use the user's turn; in shadow mode nothing changes.
- Check: pytest with a fake encoder: aligned rewrite allowed; drifted rewrite matched by the shadow
  rule; act mode falls back to the turn.

**SEC-102 Subgoal check (must).** Each supervisor plan question, and each senior-research
`research_questions` entry when that agent asks for another round (the only time they become
searches and retrievals), is decided as `plan::subgoal` with its `intent_alignment`,
`subgoal_harm` (ContextDefender harm), `intent_harm` and `subgoal_origin` (`plan` or `follow_up`)
before it is stored. In act mode a declined subgoal is removed (recorded in `security.skipped`
with its origin) and counts toward `declined_actions_in_task`; escalate stops the task as at
checkpoint 2 (SEC-046).
- Check: pytest: a harmful follow-up question under a benign intent is matched; removal and
  counting in act mode; nothing removed in shadow.

## 2. Per-call signals (phase A)

**SEC-103 Intent alignment (must).** Every search and retrieval call gets `intent_alignment` =
round(100 × cosine) between its arguments and the intent, from the same MiniLM-L6 encode of the
arguments that the existing `alignment` (against the task text) uses. Code calls get it from
their scored text (SEC-107); document extraction has no text to score (its ID is checked by
provenance, SEC-042) and keeps the default. The existing `alignment` field keeps its meaning.
- Check: pytest: one encode per call (counted with a fake encoder); both fields set.

**SEC-104 Risk above intent (must).** Every search, retrieval and code call gets `arg_harm`
(ContextDefender harm of its scored text, on the same `user_turn` route as `intent_harm`, so the two
come from one calibrated head) and `risk_above_intent` = max(0, `arg_harm` − `intent_harm`).
- Check: pytest table (benign intent with harmful arguments → high; harmful intent with the same
  arguments → low).

**SEC-105 Parallel scoring (must).** The arguments' MiniLM-L6 encode and ContextDefender's harm
score run concurrently on two threads (each model keeps its own lock), so checkpoint 2's time is
about the slower of the two, not their sum.
- Check: pytest with two fake models that each sleep 50 ms: the context is built in under 90 ms.

**SEC-106 Private-text egress (must).** After an allowed internal retrieval and after a document
is read, the guard stores salted SHA-256 hashes of every 8-word window (shingle) of that text in
the ledger, excluding windows that also occur in the intent. A web search gets `egress_overlap`:
the length of the longest run of its words whose 8-word windows all match stored hashes (0 when
none). Only hashes are kept, per task, in memory, at most 20,000 (the newest); they never reach
results, the API, logs or collection.
- Check: pytest: a search copying 8+ words of a retrieved passage scores ≥ 8; words the user typed
  don't count; the result file and `security` block contain no shingle data.

**SEC-107 Declared tools (must).** `declared` is true for search and retrieval when the plan set
`requires_web_search` or `requires_internal_knowledge` respectively, and for document extraction
when it set `requires_document_extraction`. For `execute_code` it is true when the intent asks for
code, a chart or a calculation (the `code_request` regex list in the new
`config/security/intent.json`, matched on the intent only); its scored text for SEC-103 and
SEC-104 is the coding assistant's `explanation`, passed to the guard only (never to the tool or a
log outside dev). With no `research_plan` (for example a document-only task), `declared` is true.
- Check: pytest table per tool.

**SEC-108 Drift counter (must).** `low_alignment_calls` counts this task's search and retrieval
calls with `intent_alignment` below `drift_align_at`; parallel branches sum it when they merge (SEC-115).
- Check: pytest.

## 3. Decisions (phase A)

**SEC-109 New policies, shadow-only (must).** `policies.cedar` gains, all with `@shadow("true")`:
`rewrite-drift-verify`, `subgoal-harm-decline`, `subgoal-off-intent-decline`,
`call-risk-above-intent-decline`, `call-risk-above-intent-verify`, `off-intent-search-verify`,
`private-egress-decline`, `undeclared-tool-verify` and `drift-escalate` (conditions in the backend
design §6). `off-intent-search-verify` replaces `untainted-search-misaligned` (SEC-044a). Each rule
has rows in `policy_tests.yaml`; the policy lint still rejects decline or escalate rules that read
`approved_tools` or `consented` (SEC-052).
- Check: the load-time policy table; pytest that every new rule is shadow-only.

**SEC-110 Safe defaults (must).** Every new context field has a default in `CONTEXT_DEFAULTS`
that matches no rule (`intent_alignment` and `rewrite_alignment` 100; `arg_harm`, `subgoal_harm`,
`intent_harm`, `risk_above_intent`, `egress_overlap`, `low_alignment_calls` 0; `declared` true;
`subgoal_origin` "plan"; `classifier_available` false), and `signals_available` (bool) says
whether they were computed. An
encoder or ContextDefender failure leaves the defaults, sets `signals_available` false, logs once,
and never declines by itself.
- Check: pytest with failing fakes.

**SEC-111 Judge prompt (must).** The checkpoint 2 judge prompt gives the intent (not the rewrite),
the task's subgoals and the last three checked calls (kept in the ledger, never in results), each
inside its own data markers in the style
checkpoint 1's prompts use (`<<<NAME … NAME>>>`), with an instruction that marked text is data;
marker strings inside the data are removed. Its timeout, cache
and failure meaning stay as SEC-045.
- Check: pytest with a fake LLM: the prompt contains the markers and the intent; an argument
  containing "ALIGNED" doesn't change the verdict parsing.

**SEC-112 Metrics (must).** `webagent_guard_intent_alignment{action}`,
`webagent_guard_risk_above_intent{action}` and `webagent_guard_egress_overlap` (histograms). Plan
decisions are counted in the existing `webagent_guard_decisions_total` with `checkpoint="2"` and
`action` = `plan::rewrite` / `plan::subgoal`; shadow rule matches keep counting in
`webagent_guard_shadow_policy_total`.
- Check: pytest reads the registry.

**SEC-113 Collection (must).** Checkpoint 2 and plan-check collection rows add the new numeric
fields and `subgoal_origin`. Plan-check rows carry the subgoal text only in dev; outside dev no row
holds intent text, and no row ever holds a shingle. Checkpoint 2 rows keep storing tool arguments
as they do today (the plan's §11 records that as a separate privacy question).
- Check: pytest in dev and uat modes.

**SEC-114 Configuration (must).** `security_guard.plan_check` (default true) turns SEC-101 and
SEC-102 on or off (`WEBAGENT_GUARD_PLAN_CHECK` overrides it); `security_guard.alignment_model_dir`
(default `models/security/alignment`) names the phase B model. New thresholds live in
`thresholds.py` (defaults) and the alignment model's metadata, never in YAML.
- Check: pytest (`test_security_guard_config.py`).

**SEC-115 Ledger merge (must).** `ledger.merge` unions shingle hashes and sums
`low_alignment_calls`, as it ORs taint and sums declines (SEC-039).
- Check: pytest.

## 4. The alignment classifier (phase B)

**SEC-120 Dataset (must).** `scripts/build_alignment_dataset.py` builds
`data/security/alignment/{train,test,probe}.jsonl` (Git LFS), plus git-ignored
`data/security/alignment/local/` files for gated and no-train sources (the repository is public;
SEC-121), of rows `{intent, call, action,
labels: {aligned: yes|no, risk: none|low|high}, origin, role}` from the sources and roles in the
model-and-data design §2 only. Splits follow ContextDefender's rules: near-duplicate clusters
(MiniLM cosine ≥ 0.9) and attack templates never cross splits; evaluation-only sources appear only
in `test` or `probe`, never in `train`. `--check-data` validates schema, duplicates, leakage, licence mixing and a minimum of 50
rows per label and action family. The data README records every source, licence and use.
- Check: `--check-data` passes; pytest of the builder on fixture rows.

**SEC-121 Source conditions (must).** The builder refuses to include: saroku rows until
`data/security/alignment/QUALITY.md` records its quality check as passed; Agent-SafetyBench cases
derived from AdvBench, AgentDojo or InjecAgent; ASQ rows whose source question set isn't on the
accepted list; any row from a no-train source (AgentHarm, SHADE-Arena, OS-Harm, TraceSafe) in
`train`; AgentDAM outside `test`; DeCompBench at all (skipped); and any row from a gated source
(xLAM, TraceSafe) or a no-train source in a committed file (they go only to `local/`). Gated
sources are read with the user's Hugging Face token from its standard location, never from the
repository.
- Check: pytest of the builder's filters.

**SEC-122 Generated benign calls (must).** `scripts/generate_alignment_calls.py` runs WebAgent's
supervisor planning (one model call per question) over benign questions from approved sets and
records the rewrite and plan questions as aligned calls; for coding questions it also runs the
coding assistant's generation (never executing the code) for benign code rows. It first runs 50
questions and reports
the tokens and cost; a full run stops before it would leave less than half of the credit that was
left when it started. Generated rows are labelled `origin: webagent-gen`.
- Check: pytest with a fake LLM (cost cap, stop rule); the 50-question report in the PR.

**SEC-123 Classifier (must).** A cross-encoder fine-tuned from MiniLM-L6 or MiniLM-L12 (chosen by
SEC-125 and SNFR-11) reads (intent, `"<action>: <scored text>"`, truncated to 128 tokens) and has
two calibrated heads: `aligned` (probability) and `risk` (none / low / high). It is stored in
`models/security/alignment/` with `metadata.json` (version `ac-YYYY-MM-DD-N`, data hash,
thresholds, gate results, encoder). Thresholds are chosen like ContextDefender's (upper confidence
bound of held-out false positives).
- Check: pytest loads a small fixture model; the gate (SEC-125).

**SEC-124 Baseline (must).** The same gate runs a "siamese" baseline: heads on
`[u, v, |u−v|, u·v]` of ContextDefender's encoder embeddings of both texts. The cross-encoder
ships only if it beats the baseline on SEC-125's recall at equal false positives; otherwise the
baseline ships.
- Check: the gate report lists both.

**SEC-125 Gate (must).** On `test` (and reported on `probe`): off-intent and injected-goal recall
≥ 0.95; aligned real calls held ≤ 1% (upper confidence bound); risk recall (to verify or decline)
≥ 0.90; look-alikes held ≤ 2%; expected calibration error ≤ 0.05 per head. Recorded in
`metadata.json["test"]`; a model that fails is not loaded in uat or prod.
- Check: `scripts/eval_alignment_model.py`; startup check (SEC-126).

**SEC-126 Runtime (must).** With a gated model present, `intent_alignment` and `risk_above_intent`
(and `subgoal_harm` for subgoals) come from the classifier instead of the phase A sources, and
`classifier_available` is true. Without it (missing, failed gate, load error) the phase A sources
are used and `classifier_available` is false. In uat and prod startup fails if the configured
model exists but hasn't passed its gate.
- Check: pytest both ways; startup check test.

**SEC-127 Judge band (should).** When the judge is on, a `verify` whose classifier `aligned`
probability lies in the model's uncertain band (`metadata.json`) goes to the judge; outside the
band the classifier's side decides (above: allow; below: decline).
- Check: pytest with a fake model and judge.

## 5. Acting and evaluation (phase C)

**SEC-130 Switching a rule to act (must).** A rule loses `@shadow("true")` only after (a) its
threshold is set from shadow data (or, in phase B, the gate) and (b) its shadow false-positive rate
on reviewed real tasks is within the rule's target: ≤ 1% of untainted searches for
`off-intent-search-verify`, ≤ 0.5% of tasks for each decline rule. Each switch is its own release
with its policy rows, test rows and the measurement in the PR. `tool_mode` must be `act` for any of
it to take effect.
- Check: review; policy tests.

**SEC-131 End-to-end evaluation (must).** Before any rule acts, and after each model change:
InjecAgent through checkpoint 2 holds ≥ 95% of direct-harm cases in **untainted** tasks (today
10%); the red-team suite (SNFR-03) passes with new plan-check and egress cases built from approved
data; the benign suite (SNFR-04) keeps false declines < 0.5% of tasks; AgentHarm, SHADE-Arena's
public split, OS-Harm, TraceSafe, AgentDAM and AgentDojo are run as evaluation only, results
reported with counts.
- Check: `scripts/eval_checkpoint2_injecagent.py --untainted`; `scripts/eval_alignment_suites.py`.

## 6. Non-functional requirements

| ID | Requirement | Check |
|---|---|---|
| SNFR-10 | Checkpoint 2 adds < 20 ms p95 per call without the judge (SNFR-01 unchanged), with SEC-105's parallel scoring; if the container on the target CPU exceeds it, < 30 ms for phase A only, back to < 20 ms with the classifier | `webagent_guard_tool_seconds`; container run, idle and with both workers busy |
| SNFR-11 | The classifier scores one pair in < 10 ms p95 on the target CPU, one thread | Benchmark in the gate report |
| SNFR-12 | Two workers with the classifier stay within 2.4 GiB (3 GiB limit, 20% headroom; SNFR-05) | cgroup stats in the container |
| SNFR-13 | Nothing new leaves the backend outside dev: no intent text, plan-check subgoal text or private text in logs, metrics or collection (checkpoint 2's existing argument storage is unchanged); private text only as salted hashes, in memory | pytest; review |
| SNFR-14 | Deterministic: same inputs, models and policies give the same outcome | pytest |
| SNFR-15 | All tests run offline (fake encoders, models, LLM and datasets); existing tests still pass | CI |
| SNFR-16 | No new runtime dependency: the classifier uses `torch` and `transformers` (already installed with `sentence-transformers`) and `scikit-learn` for calibration | `requirements.txt` diff |
| SNFR-17 | Data rules: public licensed data and our own benign generation only; no hand-written adversarial text; attack rows handled with counts only; every source in the data README | review |
| SNFR-18 | New files stay under about 250 lines | review |
| SNFR-19 | Each release through a PR with both CI checks green, then tagged | release |
