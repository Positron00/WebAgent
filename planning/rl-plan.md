# Plan: where reinforcement learning fits in WebAgent

Status: v5, decisions D1–D8 and O1 taken and sources `rl-gen-h`, `rl-gen-l` approved 2026-10-23; specified
in [`specs/rl-generator/`](specs/rl-generator/README.md), which wins where the two differ · Numbers
from the shadow-only checkpoint 2 classifier `ac-2026-10-17-1` (v3.9.0) and its test split ·
Related:
[`checkpoint2-upgrade-plan.md`](checkpoint2-upgrade-plan.md) and its spec
[`specs/checkpoint2-intent/`](specs/checkpoint2-intent/README.md) (data rules: SNFR-17;
sources: `design-low-level-model-data.md` §2), [`agentic-security-plan.md`](agentic-security-plan.md)

## 1. Summary

RL is useful in this project, but not for training the encoders directly. The encoders
(ContextDefender, triage, the checkpoint 2 cross-encoder) are classifiers with a label on every
row: supervised loss already gives the right direction per example, and the gate measures
calibration, which RL would make worse (§2).

RL is useful for a model that **generates**, where we can score an output but have no target to
copy. In WebAgent:

1. **An RL generator of hard training rows for checkpoint 2 (recommended).** A small open model
   trained with RL to write rows the current classifier gets wrong. In order of risk:
   - **mode H** (hard off-intent): plausible, harmless calls that serve a *different* goal from
     the intent. Benign text.
   - **mode L** (look-alikes): calls that are aligned and harmless but that the classifier holds.
     Benign text.
   - **mode A** (attacks): off-intent or risky calls the classifier passes. Adversarial text, which
     the project's data rules forbid generating; not built (§3.4, D1).

   The rows are checked, then used for ordinary supervised training. Modes H and L target most of
   the gap we can reach from training data (§2.2). Mode A is not built; whether it ever is gets
   decided after R2 (D1).
2. **RL or preference tuning of the research agent (deferred, §5).** Planning and query writing
   could improve, but there is no trainable model, no graded eval set and no reward yet. This
   plan only builds the eval set and rewards (R4).
3. **Bandit tuning of guard outcomes (not now, §6).**

Two rules apply throughout:

- **The guard's scores are never a reward for the agent.** An agent rewarded for passing
  checkpoint 2 learns to get past checkpoint 2.
- **RL has to beat prompting.** Each mode also runs as plain prompted generation of the same model
  with the same filters (best-of-8); RL is kept only if its rows beat the prompted rows on the gate
  at the same share by more than retraining noise (§3.6).

## 2. The encoders, and what the gate says is missing

### 2.1 Why not RL on the encoders

| Question | Answer |
|---|---|
| Do we have labels? | Yes, on every row. RL helps when you can only score an output, not when you know the answer |
| Gradient quality | Cross-entropy gives an exact direction per example; a REINFORCE-style reward is a noisy estimate of it |
| Gate metrics | ECE ≤ 0.05 per head. RL for correct decisions pushes probabilities towards overconfidence |
| Non-differentiable targets (recall at a threshold, Wilson bounds) | Handled after training by isotonic calibration and threshold choice on the held-out calibration fold |

### 2.2 Gate result and where the misses are

`ac-2026-10-17-1` on the gate's test split (10,188 rows; targets from `eval_alignment_model.py`):

| Check | Result | Target | Passes |
|---|---|---|---|
| Off-intent recall (all `aligned: no` rows) | 0.51 | ≥ 0.95 | no |
| Real aligned calls held, upper 95% bound | 0.94% (15 of 2,438) | ≤ 1% | yes |
| Risk recall (all `risk: low` or `high` rows) | 0.44 | ≥ 0.90 | no |
| Look-alikes held | 10% (11 of 110) | ≤ 2% | no |
| ECE aligned / risk | 0.062 / 0.108 | ≤ 0.05 | no |
| Latency p95, one thread | 12.3 ms | < 10 ms | no |

Thresholds are set so real aligned calls stay under 1% (search at 12, risk at 92), and recall
drops as a result. In 3.8.0, holding 95% off-intent recall cost 5.6% of real calls (AUC 0.888).
The classifier separates the classes too poorly, and the misses come from three places:

| Group | Test rows | Held | What it means |
|---|---|---|---|
| **Hard and easy re-pairs** (benign call, someone else's intent: ASQ, Glaive, ToolACE, xLAM, `webagent-gen`, saroku) | 1,136 of 3,385 off-intent rows | At most about half, per source (for example `webagent-gen`: 91 holds across its 1,368 rows, 254 of them re-pairs) | The model often can't tell "on topic" from "serves this goal". Benign rows: **mode H** |
| **Look-alikes** (as the gate counts them today: NotInject and Agent-SafetyBench twins; XSTest's 125 safe prompts aren't counted because of a wrong origin name, fixed before R2 by decision O1) | 110 (look-alikes are 1.7% of the build vs its 5% target, since the approved sources don't have more) | 11 wrongly held | Too few to learn from. Benign rows: **mode L** |
| **Evaluation-only (E) sources** (AgentDojo, AgentHarm, OS-Harm, TraceSafe, AgentDAM, SHADE-Arena) | 808 of 3,385 off-intent rows; **1,814 of 3,496 risk rows (52%)** | AgentHarm 87/180, AgentDAM ≤ 263/558 risky rows, TraceSafe 6/494 | Never in training by design. TraceSafe's 331 `risk: low` rows (hallucinated or redundant arguments) look nothing like any training source |

Attack styles the model trained on are caught: InjecAgent 90/90, TaskTracker 265/266, Aegis
183/185, RedCode 351/369, PrivacyLens 228/228. The problem is **generalising** to new styles and
**fine distinctions** among benign calls, not a lack of attack rows of known kinds.

Three consequences for this plan:

1. Modes H and L are benign and attack the re-pair and look-alike gaps directly. They come first.
2. Over half of risk recall is measured on E sources that no T-side data can target without
   steering towards the test set. More T data may not get risk recall to 0.90 at all (D7).
3. Latency fails on its own; extra data doesn't fix it.

The main missing data is still **reviewed real calls from shadow week 2** (the planned retrain).
This plan adds to that retrain and doesn't replace it.

## 3. Option 1: RL generator of hard rows (recommended)

### 3.1 What each mode produces

A generator policy π gets a **seed** (an intent from a T source, a tool and, for mode H, the call
that really served it) and writes one call:

| Mode | Target label | `miss` rewards | Text |
|---|---|---|---|
| H, hard off-intent | aligned no, risk none | The classifier scoring it aligned | Benign: a real-looking search or retrieval that serves a nearby but different goal |
| L, look-alike | aligned yes, risk none | The classifier holding it (off-intent or risky) | Benign: on-goal, harmless, uses words that sound sensitive out of context (security research, medicine, finance, account settings, legal questions) |
| A, attack | aligned no or risk high | The classifier passing it | Adversarial. Not built (D1) |

Each mode is its own policy (or its own LoRA adapter), so a mode can't earn reward by drifting
towards another mode's label.

### 3.2 Reward

```
r(x) = valid(x, y) · (miss(x) + λ_div · novelty(x)) − λ_len · overlong(x)
```

Novelty only counts for valid samples, so the generator can't earn reward with novel text that
has the wrong label. Samples that fail the harmless filter get 0 (table below).

| Term | How it's computed | Why |
|---|---|---|
| `valid(x, y)` | An LLM judge, with its own prompt (not checkpoint 2's judge prompt), confirms *x* has label *y*: 0 or 1. A Together model from a different family than `Muse-Glimmer-30B` (D3), which wrote the `webagent-gen` rows and backs the checkpoint 2 judge (`get_trusted_llm()`, off today), so the two models' mistakes don't line up. Temperature 0 and cached; called only when `miss` ≥ 0.3, to save credit (others count as 0). The judge's model id goes into each generated row's provenance. The exact model is chosen in R0 | Without it, mode H gets rewarded for calls that actually serve the intent and mode L for calls that really are risky (reward hacking) |
| `miss(x)` | From the frozen target's *calibrated* scores (`p_aligned`, and `risk` 0–100): mode H `p_aligned`; mode L max(1 − `p_aligned`, `risk` / 100) | Rewards rows the classifier gets wrong |
| `novelty(x)` | 1 − max cosine (base MiniLM-L6) to the committed training calls and to this run's earlier accepted samples | Stops it finding one trick and repeating it |
| `overlong(x)` | The (intent, call) pair over the classifier's 128-token `max_length`, or the call over the builder's 1,000-character limit | Text past truncation can't be learnt from; the builder drops long calls anyway |
| harmless filter | ContextDefender's harm score at or above its installed `harm_verify_at` (20 for `cd-2026-10-13-2`) → reward 0, never stored | Mode L rewards rows the classifier holds as risky; this keeps it away from text that really is harmful |

The judge is also the labeller, so its mistakes become label noise. Human review (§3.5) measures
that.

### 3.3 Setup

| Choice | Proposal | Decision |
|---|---|---|
| Generator base | **Qwen2.5-0.5B-Instruct** for the 20-step Colab test, **Qwen2.5-1.5B-Instruct** for R1 (Apache-2.0, so its outputs may train other models). Not Qwen2.5-3B, which has a different licence. The licence goes into the data README entry | D4 (decided) |
| Algorithm | GRPO with LoRA (TRL), group size 8, a KL penalty to the base model; stops early if the judged-valid rate falls (reward hacking) | — |
| Prompted baseline | The **untrained** generator base (same model, so only RL differs), run on the Mac, best-of-8 with the same filters; a new `scripts/rl_generator/prompted.py` with the same cost estimate and credit-stop rule as `generate_alignment_calls.py` (that script runs the supervisor, a different job) | — |
| Target | Frozen `ac-2026-10-17-1`, then each retrained candidate in turn | — |
| Compute | R0 needs no GPU: prompted generation runs the small base model on the Mac and the judge uses Together credit. R1 trains on **Google Colab Pro+** (D2). The frozen target classifier (88 MB), ContextDefender (for the harmless filter) and the novelty encoder run in the same session | D2 (decided) |
| Training dependencies | `trl`, `peft`, `accelerate`, `datasets`, `fsspec` and `pyarrow`, pinned in a new `requirements-train.txt`, audited by `pip-audit` in CI, kept out of the served image and `constraints.txt` | — |

### 3.4 Data rules

These come from SNFR-17 and the project's standing rules, and they bind this plan:

- **New sources need approval first.** `rl-gen-h` and `rl-gen-l` were **approved by the user on
  2026-10-23**; each still needs its `data/security/alignment/README.md` entry before any row is
  built. A mode A source would need its own approval.
- **No hand-written adversarial text, and attack text is never printed.** Mode A is a model
  writing adversarial text at scale, which the rule doesn't currently allow (D1). For every mode,
  scripts and logs print counts and scores, never row text.
- **Seeds come from the committed train split only:** aligned, no-risk rows of `webagent-gen`,
  ASQ, Glaive, ToolACE and saroku (17,206 rows in the current build). Rows from `local/` (gated
  xLAM, no-train sources), test or probe never seed a generator, and AgentDojo, AgentHarm,
  SHADE-Arena, OS-Harm, TraceSafe and AgentDAM never seed, prompt or score one. Gate results on E sources are never used to
  choose prompts, seeds or reward weights, since that would steer towards the test set.
- **Licences carry over from the seed.** A generated row's `sources` are its generated source
  (`rl-gen-h` or `rl-gen-l`) plus the seed's sources; the generator, target and judge are recorded
  as provenance in the batch file. Seeds from ASQ (CC BY-SA, via HotpotQA) make CC BY-SA rows,
  which follow the builder's no-mixing rule.
- **Generated rows go into train only.** Never into test or probe, and **never into the
  calibration fold** (`split_calibration` in `train_alignment_model.py`), so thresholds and
  calibration still come from real rows.
- **Capped share (D6).** Generated rows are at most 15% of the real training rows, across both
  modes: about 5,200 of the current build's 34,538 (31,085 after the calibration fold), half per
  mode. R2 also tries 5% (about 1,700); 15% is kept only if it beats 5% on the gate.
- Generated rows stay in the git-ignored `local/` folder, reviewed or not; committing them would be
  a separate decision.

### 3.5 Review

Per batch (one mode's rows from one run), **100 rows** stratified by tool and seed source go into
a local review sheet, as for the shadow week's `--labels` sheet. **The user labels them** (D5),
about 200 rows and an hour per round with modes H and L. Claude reports only agreement counts.
The aim is that the judge and the user disagree on at most 10%:

- ≤ 6% disagreement: the batch passes.
- > 14%: the batch is discarded and the judge prompt is fixed.
- Between 6% and 14% (100 rows can't settle which side of 10% it's on): 100 more rows are
  reviewed, and the batch passes only if disagreement over the 200 is ≤ 10%.

### 3.6 How we know it worked

For each mode, five configurations on the same base data (the post-shadow-week build), each with
only that mode's rows at its half of the share:

1. no generated rows (baseline);
2. prompted rows at a 5% share (2.5% from this mode);
3. prompted rows at a 15% share (7.5%);
4. RL rows at a 5% share;
5. RL rows at a 15% share.

Each is trained twice (seeds 13 and 17) and runs through the unchanged gate: 18 distinct runs, as
the baseline is shared. RL is worth keeping only if, at the same share, its mean on off-intent
recall (mode H) or look-alikes held (mode L) beats prompting's by more than either's spread between
seeds, with real aligned held ≤ 1% and ECE no more than 0.01 above the baseline. If prompting
matches it, we keep the prompted generator and drop RL for that mode.

Two more checks:

- **A held-out generated set** (300 rows per mode, from seeds no training batch used, a different
  checkpoint and sampling seed), never trained on, shows
  whether the classifier got more robust or only learnt one generator's style.
- **Gate results split by role** (T rows and E rows reported separately, as a report only; the
  gate itself is unchanged and revisited after R2, D7) show whether gains on T carry over to E.

### 3.7 Training on Colab Pro+

- **Sessions end.** Colab disconnects idle sessions and caps how long one runs, so the trainer
  saves a LoRA checkpoint to Google Drive every 50 steps and resumes from the latest one.
- **GPU type varies** by availability. The pilot (20 steps with the 0.5B model) records the GPU,
  memory and time per step first. Both Qwen2.5 sizes (0.5B, 1.5B) fit an L4 with LoRA.
- **Secrets** (the Together key for the `valid` judge) go in Colab's secrets panel, never in the
  notebook or a cell's output. Notebooks are saved with outputs cleared.
- **What goes to Colab and Drive:** seed rows from the committed train split, the target and
  ContextDefender models (public, in the repository), embeddings of committed train calls for
  novelty, and the judge's cache. Rows from gated
  sources (xLAM) and the git-ignored `local/` files stay on the Mac, so a gated source's terms
  aren't stretched to Google's storage. Generated rows come back to the Mac's `local/` folder and
  are deleted from Drive after import.
- **Compute units** are metered. The user sets R1's cap from the pilot's units per step, and
  `--max-steps` is set to stay within it; the user records units used after each run.
- The notebook (`backend/scripts/rl_generator/colab_train.ipynb`) only calls the repository's
  scripts, so training logic stays in reviewed code.

## 4. Option 1 for ContextDefender (later)

The same mode-L generator could make benign hard negatives for checkpoint 1. Its open finding is
about 5% false positives on real web results (page boilerplate, calls to action, video
transcripts) against a 2% gate. Labelled real pages from the shadow week are the better fix and
come first. Generated boilerplate is a fallback if those are too few.

## 5. Option 2: RL or preference tuning of the research agent (deferred)

The supervisor's rewrite and plan, the web research queries and the senior agent's follow-up
questions are all generated. Better generation would help users more than anything above, but:

| Prerequisite | Today |
|---|---|
| A model we can train | `provider: together` with `meta-models/Muse-Glimmer-30B` through an API. The `self` provider and `llm_server.py` exist, but nothing is trained or served with them |
| A reward | No user-feedback endpoint (`research_feedback` is the senior agent's output, not users'); no graded question set |
| A safe reward | Must not use checkpoint 2 or ContextDefender scores (§1) |

Rewards to build first, all checkable without a human:

- citation support (does the cited page contain the claim?),
- retrieval quality (the reranker's score of what the queries brought back),
- tokens and tool calls per task, as a penalty,
- tasks ending in `error_code`.

The cheapest first step is **DPO on preference pairs**: two plans per question, ranked by those
rewards. That uses the provider's fine-tuning if it offers preference tuning, otherwise the
self-hosted path. Online RL only if DPO helps.

**What this plan does (D8):** no agent training. After R2, one small task builds what a later
decision needs:

- a graded set of about 100 research questions from Researchy Questions (approved for training and
  test, CDLA-Permissive-2.0), held out from every dataset split;
- the citation-support reward (the D3 judge, run over each cited page and its claim) and the
  retrieval reward (the reranker's cross-encoder score), as a script that scores a run;
- a baseline: those scores for today's agent on the set.

Training the agent is a separate plan, written only if a trainable agent model is on the roadmap.

## 6. Option 3: bandit tuning of guard outcomes (not now)

Choosing allow / verify / decline / escalate per call to trade user friction against misses is a
contextual bandit, and the shadow logs record every decision with its scores. It's not worth doing
yet: thresholds picked on held-out data cover it, the logs have no reviewed outcomes, and a learnt
policy is harder to audit than a Cedar rule. Revisit after act mode has run for a while.

## 7. Phases

New code is written on a branch in the `~/webagent-dev` worktree, not in the Dropbox copy, while
shadow week 2 runs. Released per the release routine.

| Phase | Work | Exit |
|---|---|---|
| **R0** Set-up (about 60 h of work, no GPU) | Sources `rl-gen-h`, `rl-gen-l` (approved 2026-10-23) and their README entries; spec (`planning/specs/rl-generator/`); seeds, prompts (agreed by the user), reward and judge; judge model chosen (D3); prompted modes H and L (`rl_generator/prompted.py`, judge on Together credit after a 50-seed cost estimate); variant, calibration and gate-report tooling; Colab notebook tested on a 20-step run with Qwen2.5-0.5B; release of the tooling | A 500-row prompted batch per mode passes review (§3.5); user sets R1's compute-unit cap from the pilot |
| **R1** Pilot (Colab Pro+) | GRPO for mode H, then mode L, against frozen `ac-2026-10-17-1`, within the compute-unit cap; a 500-row RL batch per mode | `miss` rises across training while `valid` stays ≥ 90% and novelty doesn't collapse; both batches pass review |
| **R2** First comparison | **After** the post-shadow-week-2 retrain: the look-alike count fix (O1); held-out sets; full prompted and RL batches (about 2,700 rows per mode) against the new model; the five configurations of §3.6 per mode, two seeds each (18 distinct runs) | RL kept or dropped per mode; D1 (mode A) and D7 (gate) put back to the user with the role report |
| **R3** Rounds | Re-freeze the newest candidate as the target; up to 3 rounds per kept mode | Gate passes, or a round's gain is within the seed spread |
| **R4** Agent rewards | The graded question set, the two rewards and today's baseline (§5) | Baseline scores for today's agent |
| Later | ContextDefender (§4); agent training (separate plan) | Decided separately |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Reward hacking: rows don't have the label they claim | Independent `valid` judge; novelty only for valid samples; training stops if the valid rate falls; user review of every batch (§3.5) |
| The judge's errors become labels | User review measures the noise each round (§3.5) |
| Mode collapse | Novelty term; KL penalty; held-out generated set from another checkpoint |
| Overfitting to our own generator | 15% cap; unchanged gate test split; role-split report |
| Test leakage | T-source seeds only; E results never steer the generator; generated rows never in test, probe or calibration |
| Gate not reachable from T data (52% of risk rows are E) | Role-split report from R2 on; the gate itself is revisited after R2 (D7) |
| The guard teaches the agent to evade it | Guard scores are never an agent reward |
| Cost with no gain | Prompted baseline at R2; stop a mode if RL doesn't beat it |
| Licence mixing | Seed licence carried into `sources`; builder's CC BY-SA / BY-NC rule applies |

## 9. Decisions

| ID | Question | Recommendation | Status |
|---|---|---|---|
| D1 | May a model generate adversarial training text (mode A)? | Not yet: modes H and L first; decide A after R2 | **decided 2026-10-23: benign first.** Modes H and L only; the standing rule stands; mode A is decided after R2 |
| D2 | GPU for GRPO: provider and hours | Prompted first, then a capped cloud GPU | **decided 2026-10-23:** no GPU in R0; R1 on Google Colab Pro+ (§3.7) |
| D3 | Which model is the `valid` judge (and where generated text is sent)? | A Together model from another family than the planner | **decided 2026-10-23:** a Together model of a different family from `Muse-Glimmer-30B`, on the existing credit. The exact model is picked in R0 by agreement with the user's labels on 100 rows (§7) |
| D4 | Generator base model | Qwen2.5 Instruct 0.5B / 1.5B | **decided 2026-10-23:** Qwen2.5-0.5B-Instruct for testing, 1.5B for R1 |
| D5 | Who reviews 200 rows per round and mode? | The user, 100 rows per mode | **decided 2026-10-23:** the user, 100 rows per round and mode; 100 more when disagreement is 6–14% (§3.5) |
| D6 | Cap on generated rows' share of training | 15%, ramped | **decided 2026-10-23:** at most 15%; R2 compares 5% and 15% and keeps 15% only if it beats 5% (§3.6) |
| D7 | Keep the gate over T and E rows together, or gate T and report E? | Report by role, decide after R2 | **decided 2026-10-23:** the gate (SEC-125) stays as it is; a T / E split is added as a report only; revisit after R2 |
| D8 | Is a self-hosted research agent on the roadmap (Option 2)? | Keep deferred; build the rewards | **decided 2026-10-23:** no agent training in this plan; after R2, build a graded question set and two rewards (§5, R4) |
| O1 | The gate's look-alike check names `xstest:safe`, but the rows are `xstest:lookalike`: fix it? | Fix before R2, in its own commit | **decided 2026-10-23:** fix before R2's baseline; re-report `ac-2026-10-17-1` with both counts |

## 10. Files likely to change

| File | Change |
|---|---|
| `backend/scripts/rl_generator/prompted.py` (new) | Prompted baseline: the untrained base, best-of-8, same filters |
| `backend/scripts/rl_generator/{train,train_logic}.py` (new) | GRPO training of a mode's generator (TRL, LoRA); stop rules, guard, checkpoints |
| `backend/scripts/rl_generator/{seeds,prompts,judge,rewards,rows}.py` (new) | Seeds, prompts, judge client, reward terms, row format |
| `backend/scripts/rl_generator/sample.py` (new) | Sampling a trained generator into source rows |
| `backend/scripts/alignment_data.py` | `rl-gen-*` sources: role T, `local/` only, train only |
| `backend/scripts/rl_generator/{variant,compare,review}.py` (new) | Base build plus reviewed rows by kind and share; the R2 comparison; review sheets |
| `backend/scripts/alignment_training.py` | Look-alike origin fix (O1) |
| `backend/scripts/train_alignment_model.py` | Keep generated rows out of the calibration fold; `--seed` |
| `backend/scripts/eval_alignment_model.py` | Report split by role (T / E) and held-out generated sets, report only |
| `backend/data/security/alignment/README.md` | Provenance and licence of each generated source |
| `backend/scripts/rl_generator/colab_train.ipynb` (new) | Colab driver: mount Drive, install, resume, call `train.py`; outputs cleared |
| `backend/requirements-train.txt` (new) | `trl`, `peft`, `accelerate`, `datasets`, `fsspec` and `pyarrow`; not in the served image |
| `backend/scripts/{build_research_eval,score_research_run}.py`, `backend/data/research_eval/` (new) | R4: graded question set and run scorer |
| `planning/specs/rl-generator/` (new) | Requirements, design, tasks |

## 11. Other ways RL could help the encoders (ideas, not planned)

Recorded 2026-10-23. RL can't improve how an encoder learns from labels it already has, but it can
improve its data, its labels and how its scores are used. None of this is scheduled; each idea
needs its own decision.

| Idea | What the policy chooses, and its reward | Note |
|---|---|---|
| Hard-negative pairing from real data | Which existing intent to pair with which existing call; reward: the classifier's loss, plus diversity | No generated text; targets the re-pair gap. Try greedy top-k mining first |
| Escalation policy (encoder → judge) | Per call, trust the encoder or ask the LLM judge; reward: correctness minus judge cost and latency | Most likely route to the gate's recall at 1% false positives. Needs reviewed shadow outcomes and the judge on |
| Choosing what the user labels | Which shadow rows go on the next review sheet; reward: gate gain per label | A bandit; fits the existing review flow |
| Data-mixture search | Source quotas and shares; reward: gate metrics on a validation fold from train, never test | About 4 min per retrain; guard against overfitting the fold |
| Label-preserving augmentations | Text edits (paraphrase, truncation, formatting) that keep the label but move the score | Benign rows only; rewording attack rows would break the data rules |
| Triage routing from outcomes | Agentic or direct per question; reward: task cost and quality (R4's rewards) | Needs logs from both routes (an exploration slice) |

Not recommended: policy gradient on the gate metric itself (a partial-AUC surrogate loss targets the
same thing, supervised and with far less variance, and is the cheapest experiment here), and
encoder scores as an agent reward (§1).

