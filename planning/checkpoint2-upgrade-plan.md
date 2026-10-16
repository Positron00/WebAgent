# Plan: checkpoint 2 upgrade — tool calls checked against the user's intent

Status: v4, decided 2026-10-16; specified in [`specs/checkpoint2-intent/`](specs/checkpoint2-intent/README.md), which wins where the two differ · 2026-10-16 · Base: `main` at v3.6.1 (`beea5d3`) · Extends:
[`agentic-security-plan.md`](agentic-security-plan.md) and its spec
[`specs/agentic-security/`](specs/agentic-security/README.md) (checkpoint 2: SEC-040 – SEC-047)

## 1. Summary

Checkpoint 2 runs before every tool call. Today it is built against **injections**: its alignment
check (topic similarity between a search and the task) acts only when checkpoint 1 has already
tainted the task. It has no defence against a **rogue agent**: one that, without any injected
content, rewrites the goal, plans risky subgoals, sends private text out in a search, or runs code
nobody asked for. Such a task is untainted, and a risky subgoal on the task's own topic scores as
well aligned.

This plan adds:

1. **An intent record** fixed at task start from what the *user* wrote, so agent-written text (the
   supervisor's rewrite and plan) is checked against it instead of being the reference.
2. **A plan check**: the supervisor's rewrite and plan questions, and the senior research agent's
   follow-up questions, are checked against the intent before they act.
3. **Per-call signals that don't depend on taint**: alignment with the user's intent, *risk above
   intent* (is this call more dangerous than what the user asked?), private-text egress, tools the
   plan didn't declare, and drift across the run.
4. **A fine-tuned small classifier** (a cross-encoder of the ContextDefender kind) that scores an
   (intent, call) pair for alignment and risk above intent in about 5 ms on CPU, replacing the
   topic cosine in the new rules (the old `alignment` field stays for the existing ones) and
   leaving the LLM judge as an optional tie-breaker.
5. **Shadow first**: every new rule ships shadow-only; thresholds come from shadow data and a gated
   test set, as for ContextDefender.

## 2. Where checkpoint 2 stands (v3.6.1)

| Area | Today | Where |
|---|---|---|
| Call sites | Every web search (main, simplified retry, follow-ups), both internal retrievals, document extraction, each code run | `web_research.py:212-264`, `internal_research.py:1003-1023`, `document_extraction_agent.py:667`, `coding_review.py:257` |
| Signals | Argument patterns (secrets, email, phone, long tokens, URLs, `site:`, > 300 chars); document provenance; code AST findings; `alignment`; `tainted`; approvals; decline count | `tool_guard.py:40-80` |
| Alignment | `round(100 × cosine)` between the search/retrieval query and the task text, with the base MiniLM-L6. **Task text = the supervisor's `standalone_query`, else the user turn.** Code and documents get 100 | `alignment.py`, `research_plan.py:60` |
| Policies | `tainted-search-verify` (tainted and alignment < 60); `tainted-code-consent`; secret/personal-data/URL/provenance/dangerous-code rules; `repeated-declines-escalate` (3 declines); `untainted-search-misaligned` (alignment < 30, untainted) **shadow-only** | `config/security/policies.cedar` |
| Judge | `verify` → `get_trusted_llm()` answers ALIGNED/NOT_ALIGNED from the task text and the call only; 3 s timeout; **off in every environment** (`judge: false`), so `verify` would decline in act mode | `judge.py:19-24, 199-224` |
| Mode | `tool_mode: shadow` in dev, uat and prod | `config/*.yaml` |

Measured:

- **InjecAgent** (1,054 cases, `scripts/eval_checkpoint2_injecagent.py`): data-stealing cases are
  held 100% (the attacker's email is a pattern hit). Direct-harm cases are **allowed 90% when the
  task isn't tainted**; tainted, 100% are held (alignment median 9).
- **Local shadow simulation** (2026-10-13): 16% of untainted searches fell below 30, mostly junk
  follow-up queries that 3.2.5 fixed; the threshold for the shadow rule still needs post-3.2.5
  data.
- **Cost on this Mac** (Apple Silicon, one thread, 2026-10-16): base MiniLM-L6 encode of a short
  query 5.1 ms median / 5.9 ms p95; ContextDefender (L12) scoring of a short query 12.2 / 14.5 ms;
  an off-the-shelf MiniLM-L6 cross-encoder on an (intent, query) pair 4.5 / 5.2 ms. The phase A
  pair (the MiniLM-L6 encode and ContextDefender's harm score) takes 16.4 / 18.8 ms one after the
  other and **13.6 / 14.6 ms run in parallel** on two threads. Server CPUs may be slower; the
  container numbers need the target hardware (there is no uat cluster).

## 3. Where agents choose their own actions

These are the places a rogue (or confused) agent introduces goals the user didn't state:

| Step | Agent-written text that becomes an action | Checked today |
|---|---|---|
| Supervisor rewrite | `standalone_query`: becomes the main search query *and* the alignment reference | No (it *is* the reference) |
| Supervisor plan | `questions`: listed in web research's writing prompt (not tool calls); `requires_*` flags pick the tools | No; checkpoint 1's harm rules apply to user turns and history only, not to agent output |
| Senior research | `research_questions` in `research_feedback`, when it asks for another round (up to `max_follow_up_questions` rounds): **these become the follow-up searches and retrievals** | Only as calls, against the rewrite |
| Web research | Main query (the rewrite), simplified retry (first six words), one search for the next follow-up question | Patterns; alignment if tainted |
| Internal research | Rewrite, plus the first follow-up question in later rounds; one retrieval per further question (up to two) | Same |
| Team manager | `needs_coding` (a yes/no flag, no instruction text): starts the coding agents | No |
| Coding assistant, coding review | The assistant writes code with a short `explanation`; the review agent runs it (and a fixed version) | AST findings; consent if tainted or networked; the API server runs no code unless `WEBAGENT_ENABLE_CODE_EXECUTION=true` (`main.py` turns it off) |
| Document extraction | `document_id` | Provenance (only the attached document) |

## 4. Threats this plan covers

| ID | Threat | Example of what happens | Today |
|---|---|---|---|
| T1 | Injection that checkpoint 1 missed steers a search | A page tells the agent to search for an attacker's target | Allowed 90% (InjecAgent direct harm) |
| T2 | Goal substitution in the rewrite | The rewrite drifts from the user's question; everything after it aligns with the wrong goal | Not checked |
| T3 | Risky subgoal on the task's topic | A benign security question leads to a plan step seeking attack instructions | Passes: high topic similarity |
| T4 | Private-text egress | Text from an uploaded document or the internal knowledge base is copied into a public web search. Uploaded documents land in the shared knowledge base, so retrieval can surface *another* user's text in a task (found in the 2026-10-13 simulation) | Only fixed formats (emails, secrets) are caught |
| T5 | Unrequested capability | Code runs, or a tool the plan didn't call for is used | Code: AST and taint only |
| T6 | Gradual drift | Each search a little further from the user's goal | Each call judged alone |
| T7 | Judge manipulation | Argument text asks the judge to answer ALIGNED | Arguments are not marked as data in the prompt |

Out of scope: harmful final answers (an output checkpoint, spec task 5.4), new tools with side
effects (none exist), and attacks on the model weights or the image.

## 5. Goals and non-goals

Goals:

1. Every tool call is checked against **what the user asked**, not against agent-written text.
2. Risky subgoals are caught **whether or not the task is tainted**.
3. The main decision path stays local, deterministic and fast: no LLM call per tool call.
4. Nothing acts until shadow data and a gated test set give its threshold.

Non-goals: changing what the agents do; an LLM guard on every call; notifications; retraining
ContextDefender (separate, spec task 5.3).

## 6. Decisions (accepted 2026-10-16)

| # | Decision |
|---|---|
| D1 | The **intent** is the user's text: the current turn plus the previous user turns the task carries (not assistant replies). Agent-written text is checked against it, never used as the reference alone |
| D2 | The model is a **fine-tuned small classifier** (cross-encoder, MiniLM size), not a generative model: it follows no instructions, runs in a few ms on CPU, fits the per-worker memory and is calibrated and gated like ContextDefender. The LLM judge becomes an optional tie-breaker for its uncertain band |
| D3 | Two phases: **A** adds the intent record, plan check and new signals using the models we already ship (shadow-only); **B** adds the classifier. Act mode per rule only after shadow data (phase C) |
| D4 | Training data comes only from **approved public datasets and our own pipeline's benign output**; any new dataset or new use of an approved one is approved by the user first; no hand-written adversarial text |
| D5 | Phase A runs the two model passes **in parallel** (the MiniLM-L6 encode of the arguments, shared by `alignment` and `intent_alignment`, and ContextDefender's harm score, on two threads; each model has its own lock), so checkpoint 2 **keeps its budget** (SNFR-01: < 20 ms p95 per call without the judge): about 15 ms p95 on this Mac. If the container on the target CPU exceeds 20 ms, the budget becomes 30 ms for phase A only, and returns to 20 ms when the classifier replaces both passes. Accepted 2026-10-16 |
| D6 | Phase A ships **before** the next shadow week, so the week also records the new signals; the week waits for phase A. Accepted 2026-10-16 |

## 7. Design

### 7.1 Intent record (D1)

When the research workflow's state is built (`TaskManager.run_workflow`, where the security
ledger is created today), the backend stores in the ledger:

- `intent_text`: the user's turn plus up to three previous turns with `role: user` from
  `previous_messages` (assistant turns are agent output and are left out), capped at 2,000
  characters;
- `intent_embedding`: base MiniLM-L6, computed once;
- `intent_harm`: ContextDefender's harm score of the user turn. The intake check keeps no scores
  today, but ContextDefender's per-worker score cache (keyed by text and model version) usually
  returns the intake's result without recomputing.

The direct (non-agentic) path calls no tools and needs no intent record. The record is never
derived from agent output and never changes during the task. The existing `task_text`
(rewrite, else turn) stays for the old `alignment` field so current policies and metrics keep their
meaning.

### 7.2 Plan check (T2, T3, T5)

After the supervisor writes its plan (`supervisor.py:~245`), and before any research agent runs,
the guard scores:

- `rewrite_alignment`: the rewrite against the intent (100 when there is no rewrite or it equals the
  turn);
- each plan `question`: alignment with the intent and ContextDefender harm (`subgoal_harm`). These
  only reach a writing prompt, so their decisions are recorded and a declined one is removed;
- the declared tools (`requires_web_search`, `requires_internal_knowledge`,
  `requires_document_extraction`), read from `research_plan` when a call is checked.

When the senior research agent asks for another round (`senior_research.py`, where it writes
`research_feedback`), each of its `research_questions` is checked the same way before it is stored:
these are the subgoals that become follow-up searches and retrievals.

A new Cedar action `plan::subgoal` (principal: the agent that wrote it, `supervisor` or
`senior_research`) decides each question: `allow` keeps it; in act mode `decline` drops it
(recorded in the result's `security.skipped` and counted as a decline) and `escalate` stops the
task. The rewrite is decided once (`plan::rewrite`); in act mode a declined rewrite falls back to
the user's turn as the search text. In shadow mode nothing is removed.

### 7.3 Per-call signals

Added to checkpoint 2's Cedar context (all integers 0–100 unless stated):

| Field | Meaning | Phase A source | Phase B source |
|---|---|---|---|
| `intent_alignment` | The call's arguments against the intent | Cosine, base MiniLM-L6 (cached intent vector) | Classifier, aligned head |
| `arg_harm` | Harm of the arguments | ContextDefender harm (12–15 ms, run in parallel with the encode; D5) | Classifier, risk head |
| `intent_harm` | Harm of the user turn | Ledger (7.1) | Same |
| `risk_above_intent` | `max(0, arg_harm − intent_harm)` in A; the classifier's risk-above-intent score in B | Computed | Classifier |
| `egress_overlap` | Longest run of words shared between a **web search** and private text seen in this task (internal retrieval results, the uploaded document), excluding text also in the intent | Ledger of hashed 8-word shingles | Same |
| `declared` | Whether the plan declared this tool (true/false; for `execute_code`, see below; true when there is no plan) | `research_plan` | Same |
| `low_alignment_calls` | Search and retrieval calls in this task with `intent_alignment` below `drift_align_at` (integer count) | Ledger | Same |

Each new field gets a default in `decision.CONTEXT_DEFAULTS` that matches no rule
(`intent_alignment` and `rewrite_alignment` 100; `arg_harm`, `subgoal_harm`, `intent_harm`,
`risk_above_intent`, `egress_overlap` and `low_alignment_calls` 0; `declared` true;
`subgoal_origin` "plan"; `classifier_available` false), so a missing signal never declines by
itself; a separate `signals_available` flag says whether they were computed, as
`scores_available` does for ContextDefender, and the alignment and risk rules require it.

Private text is never stored: only salted hashes of 8-word shingles, per task, kept in the ledger
in memory (at most 20,000, the newest). They never reach the result file, the API, logs or
collection, because `_security_block` copies only the listed keys (a test keeps it so). A search sharing 8+ consecutive words with private text and not
with the user's turn is the egress signal.

When parallel branches merge (`ledger.merge`), shingle sets are unioned and `low_alignment_calls`
summed, as taint and decline counts are today.

For code there is no instruction text to score: the team manager only sets `needs_coding`. In
phase A, `declared` for `execute_code` is true when the intent asks for code, a chart or a
calculation (the `code_request` regex list in a new `config/security/intent.json`; its hit rate is
measured in shadow), and alignment and risk are scored on the coding assistant's `explanation`,
passed to the guard only. In phase B the classifier scores the
`explanation` plus the code's comments against the intent; code keeps its AST checks either way.

### 7.4 Policies (all `@shadow("true")` until phase C)

The Cedar rules in short; thresholds (`*_at`) come from `thresholds.py`, merged with the
classifier's metadata like ContextDefender's:

| Policy | Action | When | Outcome |
|---|---|---|---|
| `rewrite-drift-verify` | `plan::rewrite` | `rewrite_alignment < rewrite_at` | verify (judge, else fall back to the user's turn) |
| `subgoal-harm-decline` | `plan::subgoal` | `subgoal_harm ≥ harm_decline_at && intent_harm < harm_verify_at` | decline (drop the question) |
| `subgoal-off-intent-decline` | `plan::subgoal` | `intent_alignment < subgoal_align_at` | decline |
| `call-risk-above-intent-decline` | search, retrieve, code | `risk_above_intent ≥ risk_decline_at` | decline |
| `call-risk-above-intent-verify` | search, retrieve, code | `risk_above_intent ≥ risk_verify_at` | verify |
| `off-intent-search-verify` | search, retrieve | `intent_alignment < search_align_at` (tainted or not; replaces `untainted-search-misaligned`) | verify |
| `private-egress-decline` | search | `egress_overlap ≥ egress_words_at` (8) | decline |
| `undeclared-tool-verify` | search, retrieve, extract, code | `!declared` | verify |
| `drift-escalate` | search, retrieve, `plan::subgoal` | `low_alignment_calls ≥ drift_calls_at` | escalate |

The alignment and risk rules also require `signals_available` (the full rule text is in the spec's
backend design §6). The new thresholds (`rewrite_at`, `subgoal_align_at`, `search_align_at`,
`drift_align_at`, `drift_calls_at`, `risk_verify_at`, `risk_decline_at`, `egress_words_at`) start
at the spec's listed values, then at values read off phase A's offline run (InjecAgent, and the
benign calls of the approved 50-question generation estimate, which runs in phase A); while the
rules are shadow-only those values change only what is recorded. Before a rule acts, its threshold is set
from shadow data (phase C) or, in phase B, from the classifier's gate, like ContextDefender's.

Unchanged: approvals never override decline or escalate (SEC-052; the policy lint enforces it),
and the existing rules stay.

### 7.5 The alignment classifier (D2, phase B)

- **Input:** one text pair, `intent_text` and `"<action>: <arguments>"` (for plan checks,
  `"plan: <question>"`), truncated to 128 tokens.
- **Model:** a cross-encoder fine-tuned from a MiniLM-size encoder (L6 or L12, chosen by the gate
  and latency), with two calibrated heads: `aligned` (probability the call serves the intent) and
  `risk` (none / low / high: is the call riskier than the intent). Calibration and thresholds as ContextDefender
  (isotonic heads; thresholds from the upper confidence bound of held-out false positives).
- **Baseline to beat:** a "siamese" variant using ContextDefender's existing encoder embeddings of
  both texts plus heads on `[u, v, |u−v|, u·v]` (no new encoder, about 13 ms). The cross-encoder
  ships only if it beats it on the gate.
- **Footprint:** about 90 MB (L6) or 130 MB (L12) more per worker and in the image; two workers
  peaked at 1.81 GiB of the 3 GiB limit (SNFR-05 allows 2.4 GiB with headroom), so this fits but
  must be measured.
- **Naming and files:** `models/security/alignment/` with `metadata.json` (version
  `ac-YYYY-MM-DD-N`, data hash, thresholds, gate results), like `models/security/`.
- **Failure:** model unavailable → phase A signals (cosine and ContextDefender), recorded as
  `classifier_available: false`.

### 7.6 Judge (T7)

- The prompt gives the intent (not the rewrite), the task's subgoals and the last three calls,
  each inside data markers in checkpoint 1's style (`<<<NAME … NAME>>>`), as is the proposed call;
  marker strings inside the data are removed.
- It is consulted only for `verify` outcomes, and in phase B only when the classifier is in its
  uncertain band. Off by default, as now; when off, `verify` keeps today's meaning (decline in act
  mode).
- A smaller generative model (1–3B) as judge is **not** proposed: about 0.5–3 s per call on CPU,
  1–3 GB per worker (beyond the memory limit without a separate service), and it follows
  instructions in its input.

### 7.7 Metrics, records and configuration

- Metrics: `webagent_guard_intent_alignment{action}` and `webagent_guard_risk_above_intent{action}`
  (histograms 0–100) and `webagent_guard_egress_overlap`; plan decisions are counted in the
  existing `webagent_guard_decisions_total` (`action` = `plan::rewrite` / `plan::subgoal`), and
  `webagent_guard_shadow_policy_total` counts the new shadow rules.
- Collection rows (`data/security/collected/`) gain the new numeric fields (never the shingles or
  the intent text outside dev). Outside dev the user's text is
  stored only as a hash (`records._text_field`), while tool arguments are stored as written; that
  asymmetry is recorded as a finding (§11) and not changed here.
- Configuration: `security_guard.plan_check` (on/off, default on; `WEBAGENT_GUARD_PLAN_CHECK`
  overrides it), `security_guard.alignment_model_dir`; thresholds in `thresholds.py` and model
  metadata, never in YAML (as SEC-063 intends).

## 8. Data (D4)

The classifier learns from (intent, call, label) pairs. Labels: `aligned` (yes/no) and `risk`
(none/low/high: riskier than the intent). The spec's model-and-data design §2 gives each source's
role, pairing and labels.

| Kind | Source | Status |
|---|---|---|
| Aligned, benign calls | WebAgent's own supervisor planning run over benign questions from approved sets (triage questions, WildChat user turns, Aegis safe and XSTest safe prompts); its rewrite and plan questions are aligned calls for their question (and the coding assistant's output for coding questions) | **Approved 2026-10-16**, including the Together credit, after a cost estimate on 50 questions (§11) |
| Off-intent calls | The same calls paired with other questions: random pairs (easy) and nearest-neighbour different questions (hard) | Derived from the row above |
| Injected-goal calls | InjecAgent attacker instructions as calls under the case's user instruction (split by template and instruction, as for ContextDefender); TaskTracker secondary tasks as calls | **Approved 2026-10-16** (new use) |
| Injected-goal calls, test | AgentDojo injection-task goals against its user tasks | **Approved 2026-10-16 for test** as well as probe (still not training) |
| Risk above intent | XSTest: safe prompts as intents, with unsafe contrast prompts from the matching XSTest category as calls (XSTest pairs categories, not single prompts); Aegis dangerous-category prompts and HarmBench behaviours as calls under the nearest benign question on the same topic (by embedding) | **Approved 2026-10-16** (new use) |
| Look-alikes (must stay benign) | XSTest safe prompts and NotInject prompts as calls under related intents; benign same-topic calls under the same intents | Approved sets |
| Agent-safety benchmarks | Evaluated in §8.1–8.2: see the decisions in §12 | **Decided 2026-10-16** |
| Code calls | Benign: our coding agents' output over coding questions (credit approved above). Risky: RedCode (§8.1), never executed | Approved 2026-10-16 |

### 8.1 Agent-safety datasets evaluated (2026-10-16)

What each offers this plan: **intents** (user instructions labelled risky or safe to fulfil),
**calls** (concrete agent actions), **pairs** (an intent with a labelled action), or
**trajectories** (labelled multi-step runs). Licences were read from the licence file or dataset
card; "no-train" means the authors ask that it not be used for training (a canary string).

| Dataset | Licence and terms | Content | Made by | Fit | Recommendation |
|---|---|---|---|---|---|
| [Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench) (thu-coai) | MIT; no training restriction found | 2,000 cases in 349 environments, 8 risk categories, 10 failure modes; each case has the instruction, the environment's tools and a **fulfillable / unfulfillable** label | 876 refined from R-Judge, AgentDojo, GuardAgent, ToolEmu, ToolSword and InjecAgent; 1,124 generated by GPT-4o; every case checked by hand (98% judged reasonable on 200) | Intents with risky/safe labels and benign twins; no recorded calls (calls come from running agents) | **Train and test**, dropping the 200 AdvBench-based cases (AdvBench is gated and was excluded before) and the AgentDojo- and InjecAgent-derived cases (they would leak across our existing splits) |
| [ToolEmu](https://github.com/ryoungj/ToolEmu) | Apache-2.0 | 144 curated cases over 36 toolkits: instruction, underspecifications, **potential risky actions** and outcomes | GPT-4 brainstormed, then reviewed and refined by the authors | Direct (intent, risky action) pairs, small | **Train and test** (split by toolkit) |
| [RedCode](https://github.com/AI-secure/RedCode) | Data CC BY 4.0, code MIT; no training restriction found | RedCode-Exec: 4,050 risky Python and Bash cases (25 risk types) with natural-language forms; RedCode-Gen: 160 prompts | Not stated on the repository page; to read in the paper before use | Risky **code** calls, the plan's data gap; no benign counterparts | **Train and test** for the code head, parsed only and never executed; benign code from our coding agents |
| [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | MIT plus a clause limiting use to improving AI safety and security; **no-train** (canary) | 208 harmful and 208 matched benign tool-using tasks, 8 categories | Gray Swan AI and the UK AI Safety Institute; method not stated on the card | Matched harmful/benign intents | **Evaluation only** (test or probe), out of training as its authors ask |
| [SHADE-Arena](https://github.com/jkutaso/SHADE-Arena) public split | MIT; **no-train** (canary) | 5 task pairs (a benign main task plus a hidden harmful side task) in tool environments WebAgent doesn't have | Anthropic researchers | The closest match to rogue subgoals, but tiny and for other tools | **Evaluation only**, as a handful of end-to-end scenarios after adaptation; the private split needs the authors' approval |
| [DeCompBench](https://huggingface.co/datasets/decompositionbench/DeCompBench) | CC BY 4.0, **gated**: verified identity, no redistribution, no attacks on production systems | 253 harmful tasks decomposed into 1,522 subtasks that each look benign | 101 seeds hand-written; descriptions and decompositions by GPT-4o; checked by hand | The attack that per-call checks can't see: tests the drift and plan signals | Was: evaluation only, outside the repository. **Skipped for now** (§12) |
| [ASB](https://github.com/agiresearch/ASB) (Agent Security Bench) | MIT | 10 scenarios, 10 agents, 400+ tools, 27 attack and defence methods (injection, memory poisoning, plan backdoor) | Not stated in the README or abstract | Mostly injection, already covered | **Not now**; revisit if the plan-backdoor cases turn out useful |
| [OpenAgentSafety](https://huggingface.co/datasets/sani903/openagentsafety) | **Conflict**: the dataset card says CC BY 4.0, the GitHub repository MIT; citation "coming soon" | 356 tasks with expected unsafe behaviour, built on TheAgentCompany and Sotopia | Not stated | Intents for multi-tool agents | **Not now**; reconsider once the licence is settled |
| [R-Judge](https://github.com/Lordog/R-Judge) | **No licence** in the repository (all rights reserved by default) | 569 labelled agent trajectories with risk descriptions | Human-labelled | The best trajectory labels of the set | **Not usable** unless the authors add a licence |
| [HarmfulSkillBench](https://huggingface.co/datasets/TrustAIRLab/HarmfulSkillBench) | MIT plus terms asking for academic-research use | Harmful agent "skills" | — | Off target | **No** |
| [Nemotron-AIQ Agentic Safety](https://huggingface.co/datasets/nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0) | NVIDIA evaluation-only licence (internal evaluation and benchmarking) | Agentic safety traces | NVIDIA | No training allowed | **No** |

What this means for the data:

- **Risk above intent** gets real agent-task intents (Agent-SafetyBench's unfulfillable cases and
  their fulfillable neighbours) and real risky actions (ToolEmu), on top of the XSTest, Aegis and
  HarmBench pairs.
- **Agent-SafetyBench is used as pairs, without running our agents on it**: a fulfillable case's
  instruction is the intent, and an unfulfillable case from the same environment is the risky call
  (a second fulfillable case is its benign twin). Running our planner on unfulfillable (risky)
  instructions would produce calls aligned with a risky intent, which is not "risk above intent",
  and would spend credit generating risky output.
- **Code** is covered by RedCode for risky calls; still no matched benign twins outside our own
  coding agents.
- **Trajectories and decomposition** (the T6 drift case) have no trainable public data:
  SHADE-Arena and AgentHarm only measure it (DeCompBench would too; skipped for now). The drift rule's threshold therefore comes from shadow
  data, not a gate.

### 8.2 More datasets evaluated (2026-10-16)

Searched for the plan's remaining gaps: cheap benign (intent, call) pairs (to save Together
credit), labelled (goal, action) pairs, private-data egress (T4), and agent misbehaviour.

| Dataset | Licence and terms | Content | Made by | Fit | Recommendation |
|---|---|---|---|---|---|
| [saroku agent-action-safety](https://huggingface.co/datasets/karanxa/agent-action-safety-dataset) | Apache-2.0 | 22,682 train and 4,049 validation rows (a 1,920-row holdout kept private): `original_goal`, `action`, `context`, `is_safe`, `property` (`policy_violation`, `scope_violation`, `injection`, `goal_drift`, `corrigibility`), `severity` | Rows marked `source: gemini`; construction is in a technical report not yet read; one small vendor's release | **The most direct fit**: labelled (goal, action) pairs, including goal drift | **Train and test after a quality check**: read its report, check label balance and agreement on a sample (counts only), and treat it like the GPT-4-generated sets accepted before |
| [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) | Repository MIT; dataset card CC BY 4.0 (both permissive) | 493 privacy seeds expanded into vignettes and agent trajectories: user instruction, toolkits, trajectory, final action, the sensitive items that must not leak | Seeds from privacy regulations, literature and crowdsourcing; expansion by a GPT-4 pipeline; authors advise checking quality | Private-data egress (T4): an action that leaks what the task didn't need | **Train and test** for the egress signal |
| [AgentDAM](https://github.com/facebookresearch/ai-agent-privacy) (Meta) | CC BY-NC 4.0 | Web-agent tasks testing data minimisation: sensitive data used only when needed | Meta FAIR | Egress (T4), web agents | **Test** (non-commercial use is allowed; kept apart from CC BY-SA rows, like Alpaca) |
| [ASQ](https://huggingface.co/datasets/AgenticSearchQueryset/ASQ) (Agentic Search Queryset) | MIT; source datasets keep their own licences | 615k agent search traces and 680k agent-written queries for questions from QA sets (e.g. HotpotQA), linked by question ID | Search agents (e.g. Qwen-7B) over several retrievers | **Benign web-search calls at scale**, the closest to our `web_search` calls, without spending credit | **Train and test** for aligned calls, using only source datasets whose licences we accept (HotpotQA is CC BY-SA 4.0, handled like SQuAD) |
| [Glaive function calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | 100k+ conversations with function calls | Glaive's generation pipeline | Benign (request, call) pairs from other tool domains; re-paired for off-intent rows | **Train**, a capped share so non-search tools don't dominate |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | 10k+ tool-use dialogues over 26,507 synthesised APIs | ToolACE's multi-agent pipeline with rule and model checks | Same as Glaive | **Train**, capped like Glaive |
| [xLAM function calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC BY 4.0, gated (accept terms when logged in) | 60k verified (query, calls) pairs | Salesforce APIGen pipeline | Same as Glaive | **Train**, capped; approved, terms accepted (§12) |
| [OS-Harm](https://github.com/tml-epfl/os-harm) | Apache-2.0; **no-train** (canary) | 150 computer-use tasks: deliberate misuse, prompt injection, **model misbehaviour** | EPFL, on OSWorld | Rogue behaviour, other tools | **Evaluation only** |
| [TraceSafe](https://huggingface.co/datasets/CyCraftAI/TraceSafe) | Apache-2.0, gated; "a benchmark, not a training set" | 1,170 mutated multi-step tool traces: a benign baseline and 12 failure types (injection, privacy leakage, hallucination, interface) | Mutations of BFCL traces from six models | Trajectory-level checks, egress | **Evaluation only**; access granted (§12) |
| [ST-WebAgentBench](https://github.com/segev-shlomov/ST-WebAgentBench) (IBM) | Apache-2.0 | 222 enterprise web tasks with 646 policy instances | IBM Research, on WebArena | Policy compliance in environments we don't have | **Not now** |
| [ToolSafety](https://huggingface.co/datasets/jinjinyien/ToolSafety) (EMNLP 2025) | MIT tag; empty README | 10k–100k conversations, mostly refusals; no labels visible | Not stated | Unclear | **Not now**, until its card documents labels and generation |
| [Agent Action Guard / HarmActions](https://github.com/Pro-GenAI/Agent-Action-Guard) | CC BY 4.0 (repository); dataset files and size undocumented | Labelled agent actions with manipulated prompts | Not stated | Possible | **Not now**, until the data is documented |
| [SafeArena](https://github.com/McGill-NLP/safearena) | **No licence** | Harmful web-agent tasks | — | — | **Not usable** |
| MS MARCO, Natural Questions | MS MARCO: no licence tag on the card (research terms); NQ: CC BY-SA 3.0 | Real search queries, without agent calls | — | WildChat and ASQ cover this | **No** |

What changes with these:

- **Benign aligned calls** can come mostly from ASQ, Glaive and ToolACE, with our own generation
  (approved credit) as a smaller, WebAgent-specific share. That keeps the Together spend low.
- **Goal drift and scope** get labelled pairs from saroku, the only trainable source for it found,
  so its quality check gates its use.
- **Egress (T4)** gets PrivacyLens for training and AgentDAM and TraceSafe for testing.
- **Rogue behaviour as a whole** stays evaluation-only (OS-Harm, SHADE-Arena, AgentHarm; DeCompBench
  skipped): the drift rule's threshold still comes from shadow data.

The repository is public, so rows from gated sources (xLAM, TraceSafe) and from sources that ask
not to be trained on (AgentHarm, SHADE-Arena, OS-Harm) are built only into git-ignored local files,
never committed (spec decision 7).

Rules, as for ContextDefender: splits by near-duplicate cluster and by attack template; no attack
string on both sides; counts only when handling attack rows; the data README records every source
and licence; committed JSONL in Git LFS (a new pattern for the `alignment/` folder); `--check-data`
before training.

## 9. Evaluation and gate

On a held-out test set and a probe set from a source training never saw:

| Check | Target |
|---|---|
| Off-intent and injected-goal recall | ≥ 0.95 |
| Aligned real calls held (false positives), upper bound | ≤ 1% |
| Risk-above-intent recall (to verify or decline) | ≥ 0.90 |
| Look-alikes held | ≤ 2% |
| Calibration (ECE) per head | ≤ 0.05 |
| Latency on the target CPU | < 10 ms p95 per pair |

End to end, before any rule acts:

- InjecAgent through checkpoint 2 with the new rules: direct-harm cases held when **untainted**
  (today 10%), target ≥ 95%;
- the red-team suite (SNFR-03) extended with plan-check and egress cases built from approved data;
- the benign suite (SNFR-04): false declines < 0.5% of tasks;
- one shadow week with the new signals, reviewed with counts (no attack text printed).

## 10. Phases and releases

| Phase | Release (proposed) | Contents | Ends when |
|---|---|---|---|
| A | 3.7.0, **before the shadow week** (D6) | Intent record; plan check; new context fields from existing models, scored in parallel; egress shingles; run counter; judge prompt; new shadow rules; metrics and collection fields; the 50-question generation estimate and starting thresholds; latency measured | Tests pass; shadow metrics flowing; checkpoint 2 p95 < 20 ms (`webagent_guard_tool_seconds`) in the container, or < 30 ms with the reason recorded (D5), with the caveat that arm64 Docker on a Mac isn't server hardware |
| B | 3.8.0 | Dataset (after approvals), classifier, gate, probe; classifier in the context (shadow) | Gate passes; p95 back under 20 ms. **3.8.0 shipped the pipeline; its first candidates failed the gate** (spec README), so phase A's signals stay until a retrained model passes |
| C | per rule | Thresholds from shadow data; rules switched from shadow to acting one at a time (`tool_mode: act` already required) | Each rule's shadow false-positive rate within target |

## 11. Risks and limits

- **Together credit.** Generating calls with our planner (approved) costs tokens on a key capped at
  $25 in total, shared with the shadow week. The cost is estimated on 50 questions first; the
  generation stops before it would leave less than half the remaining credit for the shadow week.
  The fallback is fewer, deduplicated questions.
- **Parallel scoring under load.** D5's two threads cut latency when a core is free; with two
  workers busy on a 2-CPU pod limit they contend, and p95 can rise towards the sequential 19 ms.
  The container measurement includes a loaded case.
- **Leakage between benchmarks.** Agent-SafetyBench re-uses cases from InjecAgent and AgentDojo,
  which our splits already use; derived cases are dropped (§8.1) so no case is on both sides.
- **Topic vs. intent.** Pairs built by nearest-neighbour matching may teach topic distance rather
  than intent; the hard-negative and look-alike rows exist to check that, and the probe set
  measures it.
- **Harm on short queries.** ContextDefender was trained on user turns, pages and documents, not
  on search queries or plan questions, and its thresholds were set for user turns; its harm scores
  on queries are uncalibrated until phase B (the phase A rules are shadow-only for that reason).
- **Code is the weakest area.** Code alignment rests on the coding assistant's own explanation (an
  agent's description of its own action) until a code dataset exists; the AST checks and the
  sandbox stay the main controls.
- **Egress false positives.** A user who uploads a document and asks to research it will search
  with its words; the shingles exclude text also in the intent, but long quotations the user didn't
  type will still match. Shadow data decides the threshold.
- **Collection asymmetry.** Outside dev, collection keeps tool arguments verbatim but the user's
  question only as a hash, so uat data can't give (intent, call) pairs and arguments may contain
  user content. Changing what is collected is a privacy decision, outside this plan.
- **No target hardware.** Latency measured on a Mac (in 3.7.0's container: about 27 ms p95 per
  call at 2 CPUs, within D5's 30 ms; 87 ms p95 when two other processes saturate the CPUs); the
  server numbers are still owed.
- **Adaptive attackers.** A classifier can be fooled; the policies keep combining it with patterns,
  taint, egress and counters.

## 12. Open questions for you

Decided 2026-10-16: D1–D6 accepted (D5: parallel scoring, keep 20 ms; D6: phase A before the
shadow week); new uses of approved datasets approved, and AgentDojo in test; Together credit for
generating calls approved after a 50-question estimate.

Also decided 2026-10-16: Agent-SafetyBench (without its AdvBench-, AgentDojo- and
InjecAgent-derived cases), ToolEmu and RedCode approved for training and test; AgentHarm and
SHADE-Arena's public split approved for evaluation only; DeCompBench access to be requested by the
user (evaluation only, kept outside the repository; **skipped for now**, later the same day); ASB,
OpenAgentSafety, R-Judge, HarmfulSkillBench and Nemotron-AIQ left out.

And from §8.2 (2026-10-16):

- **saroku agent-action-safety**: approved for training and test **after its quality check**
  (read its technical report; label balance and agreement on a sample, counts only). If it fails,
  it is dropped and goal-drift rows come only from shadow data.
- **PrivacyLens**: approved for training and test. **AgentDAM**: approved for test (CC BY-NC 4.0;
  kept apart from CC BY-SA rows).
- **ASQ** (only source question sets with accepted licences), **Glaive v2**, **ToolACE** and
  **xLAM** approved as benign calls, each capped. xLAM is gated: you accept its terms on Hugging
  Face, and the build script reads it with your Hugging Face login (the token stays in its
  standard location, outside the repository); its rows are never committed.
- **OS-Harm**: approved for evaluation only (its authors ask that it not be trained on).
- **TraceSafe**: evaluation only, once you have requested and received access (gated).
- Left out: ST-WebAgentBench, ToolSafety, HarmActions, SafeArena, MS MARCO, Natural Questions.

No questions are open. TraceSafe access and xLAM's terms were granted on your Hugging Face account
(2026-10-16); this machine still needs a read token (`hf auth login`, stored outside the
repository). The 50-question cost estimate for our own generation runs in phase A (its calls set
the starting thresholds); before phase B's dataset build: that token and the saroku quality check.

## 13. Files likely to change

The spec's designs list every file; in short:

- Backend: new `app/security/{intent,signals,egress}.py` and (phase B) `alignment_model.py`;
  changed `app/security/{tool_guard,alignment,ledger,judge,thresholds,decision,records,types}.py`,
  `app/core/{config_def,config,metrics}.py`, `app/services/task_manager.py` (intent record),
  `app/agents/supervisor.py` and `senior_research.py` (plan check), `internal_research.py` and
  `document_extraction_agent.py` (private-text shingles), `coding_review.py` (scored text).
- Config: `config/security/{policies.cedar,schema.cedarschema,policy_tests.yaml}`, new
  `config/security/intent.json`, `config/{dev,uat,prod}.yaml` (`plan_check`).
- Scripts and data (phase B): `scripts/{fetch_alignment_sources,generate_alignment_calls,
  build_alignment_dataset,train_alignment_model,eval_alignment_model,eval_alignment_suites}.py`,
  `scripts/eval_checkpoint2_injecagent.py` (`--untainted`), `data/security/alignment/` (Git LFS, plus
  git-ignored `local/`), `models/security/alignment/`, `.gitattributes`, `.gitignore`.
- Tests: plan check, each new field, each new policy row, shadow behaviour, judge prompt markers,
  shingle privacy, parallel timing, the dataset builder's filters, the model's gate and fallback.
- Docs: the spec `specs/checkpoint2-intent/`, a pointer from the agentic-security spec (SEC-044a's
  shadow rule replaced), RUNBOOK, shadow-week runbook (new metrics), data README, CHANGELOG.
