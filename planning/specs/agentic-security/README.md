# Spec: security checkpoints (pre-LLM and pre-tool)

Status: phases 0–4 implemented, running in shadow mode; ContextDefender `cd-2026-10-13-2` **passed the release gate** (2026-10-13), so checkpoint 1 may go to act mode when the operators choose (`input_mode: act`; not switched yet) · Release 3.2.1 · Source plan: [`planning/agentic-security-plan.md`](../../agentic-security-plan.md) (v5) · 2026-10-13

Next: checkpoint 2 checked against the user's intent (rogue subgoals, risk above intent, egress) is specified in [`../checkpoint2-intent/`](../checkpoint2-intent/README.md) (SEC-100 onwards; its `off-intent-search-verify` will replace SEC-044a's shadow rule).

## ContextDefender status (2026-10-13)

`cd-2026-10-13-2`: fine-tuned from **`all-MiniLM-L12-v2`** (12 layers) on the same 35,026 rows as `-13-1` (data README), 96-token windows. Thresholds (SEC-028a), all chosen with the upper confidence bound of the held-out false-positive rate: injection drop 57 (watch 28); harm verify 20, decline 98; jailbreak verify 26, consent 96. Injection used a point estimate with a margin until this version; it gave 36 for this model, whose test false positives sat exactly at the 2% limit (the second time that rule landed on the limit), and the held-out part now has 821 benign untrusted rows, enough for the bound. That rule change was made after seeing that result.

Gate on `test.jsonl` (6,784 rows; `metadata.json["test"]`):

| Check | `-13-2` (L12) | `-13-1` (L6) | Target |
|---|---|---|---|
| Injection recall, untrusted sources | **0.973** ✓ | 0.939 | ≥ 0.95 |
| Injection false positives, benign pages | 0.014 ✓ | 0.013 ✓ | ≤ 0.02 |
| Harm recall to decline or judge | 0.931 ✓ | 0.917 ✓ | ≥ 0.90 |
| Harm direct declines of ordinary questions | 0.008 ✓ | 0.008 ✓ | ≤ 0.01 |
| Jailbreak recall to consent or judge | 0.961 ✓ | 0.969 ✓ | ≥ 0.90 |
| Jailbreak direct consents of ordinary questions | 0.001 ✓ | 0.004 ✓ | ≤ 0.01 |
| Secret / personal-data recall | 0.984 ✓ / 0.979 ✓ | 0.984 / 0.979 | ≥ 0.95 / ≥ 0.85 |
| Sensitive-data false positives | 0.006 ✓ | 0.009 ✓ | ≤ 0.02 |
| ECE injection / harm / sensitive / jailbreak | 0.003 / 0.036 / 0.007 / 0.022 ✓ | ≤ 0.035 | ≤ 0.05 |

Encoder comparison on the same data and test set (fine-tuned the same way; injection recall / false positives on all benign untrusted text by threshold; CPU per page on one thread):

| Encoder | at 40 | at 60 | at 73 | ms per page (median / p95) | Size |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (`-13-1`) | 0.955 / 2.2% | 0.942 / 1.4% | 0.929 / 1.2% | 70 / 156 | 91 MB |
| **all-MiniLM-L12-v2** (`-13-2`) | 0.977 / 1.7% | 0.971 / 1.4% | 0.965 / 1.1% | 120 / 273 | 133 MB |
| BAAI/bge-small-en-v1.5 | 0.977 / 2.2% | 0.972 / 1.4% | 0.966 / 1.2% | 121 / 261 | 133 MB |

L12 and BGE-small both pass; L12 has slightly fewer false positives at low thresholds. The image (3.4 GB) loads it offline in prod mode, peak memory 574 MB per process (about 40 MB more).

**End to end with the harm judge** (339 moderation calls, no errors): harm recall 0.76, ordinary questions declined 1.23% (reported, not gated; the judge adds a few to the 0.8% declined directly), look-alikes 7.7%.

**Before act mode.** The gate measures constructed data. Real pages from shadow mode (the injection and alignment metrics, reviewed collected rows) should confirm the false-positive rate before `input_mode: act`, and the end-to-end harm declines (1.23%) are worth watching.

**Checkpoint 2 on InjecAgent** (`scripts/eval_checkpoint2_injecagent.py`, all 1,054 cases, local): WebAgent has none of InjecAgent's target tools, so the nearest action is a web search carrying the attacker's instruction. Data-stealing cases (544) are held for verification 100% of the time even when checkpoint 1 missed the injection, because the attacker's email address is a pattern hit. Direct-harm cases (510) are **allowed 90% of the time when the task isn't tainted**; tainted, 100% are held, because their alignment with the user's task is low (median 9; 99.6% below 40). The alignment check only applies to tainted tasks, so an injection checkpoint 1 misses also escapes it. Applying it without taint (for example verify below 30) would close that, but how often real follow-up searches fall that low hasn't been measured: the local logs hold no search queries, so that needs shadow-mode data.

**Moderation category cut-off (tried 2026-10-12, not kept).** Judging on moderation's highest category score with a cut-off chosen on held-out user turns (0.2) gave harm recall 0.757, ordinary questions declined 1.40% and look-alikes 12.6%: a different point on the same trade-off, further from the 1% limit, so `-8` (moderation's `flagged`) stays. The code remains (`--choose-moderation-cutoff`; unset by default).

**Why recall stops near 0.7–0.75.** On held-out harmful user turns, moderation scores the dangerous Aegis categories high (missed below 0.2: weapons 0/16, self-harm 0/5, violence 3/44, criminal planning 9/105, HarmBench 2/21) and the softer ones low (PII/privacy 29/34, Needs Caution 14/18, unauthorized advice 7/7, political/misinformation 5/5, immoral/unethical 2/2, hate 16/43). Most of the missed recall is in categories a research assistant arguably shouldn't decline at all; which Aegis categories count as `harm` is a labelling decision not yet made.

**What this shows.** Sending more to moderation cut wrongful declines to about a quarter, but moderation's `flagged` misses about a quarter of the harmful requests in the data: its categories are narrower than the Aegis "unsafe" labels most harm rows carry. Injection recall (0.88) and secret recall (0.73) are still short; the jailbreak judge (chat model) hasn't been measured, pending Prompt Guard access.

**Gate fix (2026-10-11).** Until this version the gate's false-positive checks only counted benign rows whose `origin` started with `benign`, and secret/personal recall only rows starting with `secret`/`personal`; the public-data origins (`wildchat:assistant`, `generated:secret`, `nemotron:pii`…) matched none, and an empty group passed. Those checks now select by label, and a check with no rows fails. The figures recorded earlier for `cd-2026-10-11-4` (false positives 0.000, secret recall 1.000) were not measured.

**Where it fails.** Harm false positives come mostly from benign prompts built to look harmful: XSTest safe (42%), JBB benign (46%) and Aegis safe (36%); ordinary questions are lower (WildChat 7.4%, triage questions 4.7%) but still over 1%. Jailbreak false positives are mostly TrustAIRLab regular prompts (39%), which are role-play system prompts. Secret recall depends on the classifier alone here; at runtime the fast patterns also catch the common key formats.

Next: hard-negative weighting for the contrast sets, per-axis thresholds in the policies instead of one score of 50, and reviewed shadow-mode data (task 5.3).

## What this feature does

WebAgent sends untrusted text (the user's question, client-supplied history, web pages, documents, code output) to LLMs on every step of a task, and the LLMs' output decides what the tools do. This feature puts two checkpoints around every LLM call:

- **Checkpoint 1 (pre-LLM)** checks every context item before it reaches a model, on every hop, including the web app's direct chat. A flagged web page chunk is dropped; a risky user message is declined, sent back for consent, or escalated.
- **Checkpoint 2 (pre-tool)** checks every tool action the LLM proposes (web search, internal retrieval, document extraction, code execution) before it runs.

Both ask the same **decision point** (Cedar policies, in-process). A local classifier, **ContextDefender**, scores context on four axes (injection, harm, sensitive data, jailbreak); it fills in the policy context and never decides. Outcomes are allow, verify, consent, decline and escalate.

## Documents

| File | What it holds |
|---|---|
| [`requirements.md`](requirements.md) | What must be true when the feature ships. Numbered requirements (`SEC-…`, `SNFR-…`) with acceptance criteria and verification |
| [`design-high-level.md`](design-high-level.md) | Components, sources and trust, flows for each checkpoint, outcomes, taint, tokens, failure handling, deployment, key decisions |
| [`design-low-level-backend.md`](design-low-level-backend.md) | Python modules, types, the LLM factory change, `check_hop`, tool guards, the decision point and policies, ContextDefender, endpoints, config, metrics, scripts |
| [`design-low-level-webapp.md`](design-low-level-webapp.md) | The `/api/chat` guard call and fallback, `apiClient` handling of consent/decline/escalate, the consent dialog, the security summary under answers |
| [`shadow-week-runbook.md`](shadow-week-runbook.md) | Deploying 3.8.0 to uat with the guard and triage in shadow mode, what to watch, the review, and the criteria for act mode |
| [`tasks.md`](tasks.md) | Work in phases, each task with the requirements it covers, dependencies, done criteria and estimates |

Read in that order. Requirements say *what*; the designs say *how*; tasks say *in what order*. When documents disagree, `requirements.md` wins, then the designs, then the plan.

IDs are local to this spec: `SEC-0xx` here is unrelated to `REQ-0xx` in the triage spec.

## Scope

In scope: phases 0–4 of the plan (groundwork, checkpoint 2, ContextDefender, checkpoint 1, act) and the tuning loop of phase 5.

Out of scope:

- An output checkpoint on the final answer (later; same engine).
- Scoring images (recorded as `source = "image"`, no scores).
- User accounts and per-user permissions.
- Changing the code sandbox, API keys or rate limits.

## Decisions (2026-10-11)

1. **Cedar** is the decision point, run in-process through `cedarpy`.
2. The web app's direct chat **calls the backend** for checkpoint 1; when the backend is unreachable, Next.js runs the fast patterns and fails open with the existing fallback notice.
3. The Kubernetes memory limit rises to **2 GiB** (request 1.5 GiB) before checkpoint 1 ships. Raised again to **3 GiB** (request 2 GiB) in 3.2.5: two workers peaked at 1.81 GiB in the simulated shadow week.
4. Consent in the middle of a task: **skip the action and offer "Approve and re-run"**; pausing tasks comes later.
5. Public datasets **only where the licence allows**, each recorded in the data README.

Open:

6. **Image size.** Re-adding the base MiniLM for ContextDefender grows the image about 90 MB more, about 175 MB over 3.0.3, past the triage spec's NFR-09 budget (100 MB). Recommendation: raise that budget to 200 MB for both models now, and revisit one shared encoder (a frozen-encoder triage version) after phase 2.

### Judge for borderline user turns (2026-10-11)

- **Harm:** OpenAI's moderation endpoint (`harm_judge: moderation`, needs `OPENAI_API_KEY`). It is built to receive harmful text, so it doesn't refuse; a chat model asked to judge such text may refuse, and the user turn's text then also goes to OpenAI, not only the chat provider.
- **Jailbreak:** the trusted chat model for now. The Prompt Guard backend is built and tested, access was granted on 2026-10-12, and both sizes were measured locally against `cd-2026-10-12-5` (verify band 41–85; cut-off chosen on held-out turns: the highest with the best held-out recall, 0.9). 22M: 283 MB, 95 ms per message on one CPU thread; 86M: 1.1 GB (over the GitHub LFS allowance and the 2 GiB worker limit), 185 ms, and no more accurate. On the test set, 22M against no judge (consent only at 85 and above): jailbreaks caught 0.889 → 0.926 (+5 of 135), ordinary questions asked for consent 0.69% → 1.66% (+7, all WildChat user turns, over the 1% limit), look-alikes 3.6% → 7.2%. It isn't enabled or shipped: the extra jailbreaks don't justify the extra consent prompts. The models are not in the repository.
- **Prompt Guard as a second opinion on injection** (measured 2026-10-12, not used): on untrusted test text (3,320 rows) Prompt Guard 22M alone finds 12.5% of injections at 0.5 with 2.5% false positives on benign pages (6.6% / 1.3% at 0.99); combined with ContextDefender (Prompt Guard deciding in a band below the drop threshold, or either one dropping, cut-offs chosen on 1,688 held-out rows) recall moves from 0.914 to 0.915. It is built for explicit prompt attacks, not task-like instructions inside data, which is the remaining gap. Meta's **Llama Prompt Guard 2** (22M or 86M, text only) is the planned local replacement. Licence checked 2026-10-11: Llama 4 Community License — commercial use allowed below 700 million monthly active users; shipping it in the image means including the licence text and a "Notice" file ("Llama 4 is licensed under the Llama 4 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved."), showing "Built with Llama" in the product documentation or UI, following the Acceptable Use Policy, and naming any fine-tuned version "Llama…". The EU restriction in the use policy covers multimodal models only. The models are gated with manual approval on Hugging Face, so access has to be requested from a logged-in account before they can be downloaded.

## Findings from the code that shaped the spec

1. **There is no single LLM entry point.** Agents use `get_agent_llm`, `get_llm` (internal research's retry fallback, the triage tie-breaker), `get_llm_without_tags` (coding agents), `ChatOpenAI` built in the agent, and the `openai` client directly (coding assistant); `app/utils/diagnostics.py` also calls `get_llm()` for its health check. Phase 0 routes all of them through `services/llm.py`, with `get_trusted_llm()` for the tie-breaker, the judge and diagnostics.
2. **Prompts are LangChain templates formatted with variable text** (for example `WEB_RESEARCH_PROMPT` with `{search_results}`). Scoring a formatted prompt would score our own instructions; checkpoint 1 therefore scores the variables, at prompt-building time.
3. **Research agents run in parallel on deep copies of the state** (`run_research_agents`), so security records written by parallel hops must be merged back.
4. **Search queries come from three places in `web_research.py`**: the main query (the supervisor's `standalone_query`), the simplified-query retry, and the follow-up query from senior research feedback. Retrieval queries in `internal_research.py` also mix in the feedback questions.
5. **`document_id` comes from the task request** (`task_manager` copies it into `state.inputs`), not from an LLM.
6. **Code execution is off by default on the server** (`WEBAGENT_ENABLE_CODE_EXECUTION`), and `coding_review.py` may run code several times while fixing it.
7. **`previous_messages` and the direct chat's history come from the client**, including "assistant" turns, so they can be forged.
8. **`cedarpy` names policies by position** (`policy0`, `policy1`…) and ignores `@id`; the decision point maps annotations itself. Checked with `cedarpy` 4.8.7: the plan's 10 policies pass 14 cases at about 0.2 ms per decision; the full starting set of 15 policies (backend design, section 6.4) validates against the schema and passes 21 cases.

## Traceability

Every task in `tasks.md` names the requirements it satisfies. Every requirement names its verification. The coverage table at the end of `tasks.md` shows that each requirement has at least one task.
