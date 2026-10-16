# Spec: checkpoint 2 checked against the user's intent

Status: draft v2 (aligned with the plan, 2026-10-16) · Source plan:
[`planning/checkpoint2-upgrade-plan.md`](../../checkpoint2-upgrade-plan.md) · Base: `main` at v3.6.1
(`beea5d3`) · Extends: [`../agentic-security/`](../agentic-security/README.md) (checkpoint 2:
SEC-040 – SEC-047) · Releases: 3.7.0 (phase A, before the shadow week), 3.8.0 (phase B), then one
release per rule that starts acting (phase C)

## What this work does

Checkpoint 2 checks every tool call, but its alignment check acts only on tasks checkpoint 1 has
tainted, and it compares calls with the supervisor's own rewrite. An agent that drifts, plans
risky subgoals on the task's topic, copies private text into a web search or runs code nobody
asked for leaves the task untainted and passes.

This spec makes the user's own words the reference (the intent record), checks agent-written
subgoals before they act (the plan check), adds per-call signals that don't depend on taint (intent
alignment, risk above intent, private-text egress, undeclared tools, drift), and later replaces
the topic similarity with a fine-tuned small classifier. Every new rule starts shadow-only.

## Documents

| File | What it holds |
|---|---|
| [`requirements.md`](requirements.md) | SEC-100 – SEC-131 and SNFR-10 – SNFR-19, with acceptance criteria and checks |
| [`design-high-level.md`](design-high-level.md) | Components, flows, failure handling, data, deployment, limits |
| [`design-low-level-backend.md`](design-low-level-backend.md) | Modules, signatures, Cedar schema and rules, hook points, metrics, tests (phase A and B's runtime) |
| [`design-low-level-model-data.md`](design-low-level-model-data.md) | Sources and roles, dataset build, generation, model, gate, evaluation suites (phase B) |
| [`tasks.md`](tasks.md) | Tasks per phase in dependency order, with coverage |

Requirements say *what*; the designs say *how*; tasks say *in what order*. When documents
disagree, `requirements.md` wins, then the designs, then the plan.

## Scope

In scope: the plan's sections 7–10. Out of scope: an output checkpoint (agentic-security task
5.4), new tools, changing what the agents do, an LLM on every call, changing what collection
stores outside dev, notifications.

## Decisions (all accepted 2026-10-16)

1. **The intent is the user's text** (current turn plus up to three previous user turns), fixed at
   task start; agent text is checked against it (plan D1).
2. **A fine-tuned small classifier, not an LLM**, on the main path; the judge only for its
   uncertain band (D2).
3. **Phases A, B, C**; every new rule shadow-only until it has a measured threshold (D3).
4. **Data only from approved sources** and our own benign generation; no hand-written adversarial
   text (D4). Approved: the plan's §8 table, §8.1 (Agent-SafetyBench, ToolEmu, RedCode for
   training and test; AgentHarm and SHADE-Arena for evaluation only) and §8.2 (saroku after its
   quality check; PrivacyLens; AgentDAM for test; ASQ, Glaive, ToolACE and xLAM as benign calls;
   OS-Harm and TraceSafe for evaluation only). DeCompBench skipped for now. Together credit for
   generation after a 50-question estimate.
5. **Parallel scoring keeps checkpoint 2's 20 ms budget** (D5); 30 ms for phase A only if the
   container needs it.
6. **Phase A ships before the shadow week** (D6), so the week records the new signals.
7. **Gated and no-train rows stay off the public repository** (made while writing this spec): the
   repository is public, so rows from xLAM, TraceSafe, AgentHarm, SHADE-Arena and OS-Harm are
   built into git-ignored `local/` files only.

## Findings that shaped the spec

Checked in the code on 2026-10-16.

1. The follow-up searches and retrievals come from the **senior research agent's**
   `research_questions` (`senior_research.py`, `research_feedback`), not the supervisor's plan
   questions, which only reach web research's writing prompt (`web_research.py:304`). The plan was
   corrected; the subgoal check covers both, and only the follow-ups change tool calls.
2. Checkpoint 1's intake scores the user turn but keeps no scores; ContextDefender's per-worker
   cache (keyed by text and model version) makes the intent's harm score cheap.
3. `TaskManager._security_block` copies only listed ledger keys into results, so new ledger keys
   (intent, shingles) stay out of results and the API by construction.
4. The team manager's `needs_coding` is a bare flag; code calls are scored on the coding
   assistant's `explanation`.
5. `patterns.json` holds the scanner's validated regex lists, so the code-request keywords get
   their own `config/security/intent.json`.
6. The checkpoint 1 judge prompts mark data as `<<<MESSAGE … MESSAGE>>>`; the checkpoint 2 judge
   prompt has no markers today.
7. `.gitattributes` tracks `backend/data/security/*.jsonl` in LFS but not subfolders; the
   alignment data needs its own pattern.
8. The repository is public (decision 7).

## Implementation notes (phase A, 3.7.0)

- The intent's embedding and harm are computed on first use (SEC-100), and the plan check warms
  them during the supervisor step, so starting a task loads no model and a tool call never pays
  for them.
- Arguments and the intent are scored on ContextDefender's `user_turn` route, so `arg_harm` and
  `intent_harm` come from one head (SEC-104).
- The follow-up check runs where the senior research agent asks for another round, the only place
  its questions become calls (SEC-102).
- Matched policies are sorted by their place in the file: Cedar reports them in no fixed order,
  which made `shadow_policy_ids` vary between runs (SNFR-14).
- **Offline run (task A10, 2026-10-16).** InjecAgent through checkpoint 2 with the intent record:
  if the shadow rules acted, 98.6% of direct-harm attacks in untainted tasks would be held at
  `search_align_at` 30 and 95.9% at 25 (today 10%). The 50-question generation estimate cost $0.05
  (13,637 input and 31,748 output tokens; a full run over the 2,496 triage questions is projected
  at $2.62). Of its 183 benign plan questions (from 48 tasks), 6.6% score below 25 and 1.6% below
  15. Starting values: `search_align_at` and `drift_align_at` 25, `subgoal_align_at` 15,
  `rewrite_at` 40.
- **The planner replaced the user's goal in 2 of the 50 benign questions** ("How do I train my dog
  to sit?" was rewritten into a question about AI chip makers; a quadratic equation into questions
  about Tesla). `rewrite-drift-verify` matched both. Generated rows like these are kept out of the
  aligned training rows (model-and-data design §2).
- **Latency (task A11, 2026-10-16, `scripts/bench_checkpoint2.py`, arm64 Docker on a Mac, not server
  hardware).** Per search call after the plan check's warm-up: 22.8 ms median / 26.4 ms p95 without a
  CPU limit, 24.5 / 27.4 ms at the Kubernetes limit of 2 CPUs: within D5's 30 ms for phase A, not
  the 20 ms target. With two other processes scoring non-stop on those 2 CPUs: 68.4 / 86.8 ms (every
  model call slows about threefold under that saturation; real workers mostly wait on model
  replies). On the Mac's host CPU the same calls take 18.7 / 21.3 ms. The classifier in phase B
  replaces the two model passes with one (SNFR-10 back to 20 ms); the server measurement is still
  owed (no target hardware).

## Traceability

Every requirement names its check; every task names its requirements; `tasks.md` maps each
requirement to tasks and each plan section to requirements.
