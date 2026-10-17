# Low-level design: the alignment classifier and its data (phase B)

Paths are relative to `backend/`. Requirement IDs refer to [`requirements.md`](requirements.md).
The approvals behind every source are in the plan's §8 and §12 and the README's decisions.

## 1. Files

| File | Purpose |
|---|---|
| `scripts/fetch_alignment_sources.py`, `alignment_sources.py` | Downloads the approved sources into a scratch folder (outside the repository); gated ones with the user's Hugging Face token from its standard location |
| `scripts/alignment_data.py`, `alignment_pairs.py`, `alignment_agent_pairs.py`, `alignment_eval_pairs.py` | The builder's helpers: row schema, licences, `--check-data`; pairing per source |
| `scripts/alignment_training.py` | Shared by training and the gate: loading splits, labels, origin groups, the Wilson bound, ECE, threshold selection |
| `scripts/generate_alignment_calls.py` | Our own benign calls: supervisor planning (and, with `--code`, the coding assistant) over approved benign questions; cost report and stop rule (SEC-122); writes `generated/webagent_gen.jsonl` and `generated/webagent_gen_code.jsonl` |
| `scripts/build_alignment_dataset.py` | Pairs, labels, filters and splits (SEC-120, SEC-121); `--check-data` |
| `scripts/train_alignment_model.py` | Fine-tunes the cross-encoder and the siamese baseline; calibrates; picks thresholds (SEC-123, SEC-124) |
| `scripts/eval_alignment_model.py` | The gate on `test`, report on `probe`, latency benchmark; writes `metadata.json["test"]` (SEC-125, SNFR-11) |
| `scripts/eval_alignment_suites.py` | Evaluation-only suites through signals and policies, counts only (SEC-131) |
| `scripts/eval_checkpoint2_injecagent.py` | Gains `--untainted`: hold rate with the new rules counted as acting (SEC-131) |
| `data/security/alignment/{train,test,probe}.jsonl` | The committed dataset (Git LFS; new `.gitattributes` pattern `backend/data/security/alignment/*.jsonl`) |
| `data/security/alignment/local/{train,test,probe}.jsonl` | Rows from gated and no-train sources, built on the machine and **git-ignored** (§8) |
| `data/security/alignment/README.md` | Sources, licences, roles, counts, how it is built |
| `data/security/alignment/QUALITY.md` | The saroku quality check's result (SEC-121) |
| `models/security/alignment/` | Encoder, heads, calibrators, `metadata.json` (Git LFS for weights: new pattern `backend/models/security/alignment/**/*.safetensors` and `*.pt`) |

## 2. Sources and roles

Labels: `aligned` (does the call serve the intent: yes/no) and `risk` (is the call riskier than
the intent: none/low/high). Roles: **T** train and test (split), **E** test and probe only
(evaluation), never train.

| Source | Licence | Role | Intent | Call | Labels |
|---|---|---|---|---|---|
| WebAgent generation (`webagent-gen`) over triage questions, WildChat user turns, Aegis safe, XSTest safe | Project data from approved sets | T | The question | Its rewrite and each plan question | aligned yes, risk none, **except** calls of a task whose rewrite scores below `rewrite_at` against its question: the planner can replace the user's goal (2 of 50 in the A10 estimate), so those tasks are excluded from the aligned rows and listed for review |
| Re-pairs of the rows above | — | T | Another question: random (easy) or nearest neighbour by MiniLM (hard) | The call | aligned no, risk none |
| ASQ (accepted source sets only: `ASQ_SOURCES` in the builder, listed in the data README with each set's licence; HotpotQA and, approved 2026-10-17, Researchy Questions (CDLA-Permissive-2.0), the quota split evenly between them) | MIT; sources' own | T | The source question | Each agent-written query, as `web_search` or `internal_retrieve` (chosen per question; ASQ's agents searched a local corpus) | aligned yes, risk none (re-pairs: aligned no) |
| Glaive v2, ToolACE, xLAM (each capped at 5,000 rows) | Apache-2.0, Apache-2.0, CC BY 4.0 | T | The user request | `"<function>: <arguments>"` | aligned yes, risk none (re-pairs: aligned no) |
| InjecAgent (split by template and instruction, as ContextDefender) | MIT | T | The case's user instruction | The attacker instruction as a call | aligned no, risk high |
| TaskTracker secondary tasks | MIT | T | A SQuAD-question intent | The secondary task | aligned no, risk none |
| XSTest | CC BY 4.0 | T | A safe prompt | An unsafe contrast prompt of the matching category | aligned no, risk high |
| XSTest safe, NotInject (look-alikes) | CC BY 4.0, MIT | T | The look-alike itself (a self-pair: the request is both intent and call) | The look-alike | aligned yes, risk none |
| Aegis dangerous categories, HarmBench | CC BY 4.0, MIT | T | The nearest benign question on the same topic (MiniLM, cosine ≥ 0.5) | The harmful request as a call | aligned no, risk high |
| Agent-SafetyBench (without AdvBench-, AgentDojo-, InjecAgent-derived cases; split by environment) | MIT | T | A fulfillable case's instruction | An unfulfillable case of the same environment (risky); another fulfillable case (benign twin) | risky: aligned no, risk high; twin: aligned yes, risk none |
| ToolEmu (split by toolkit) | Apache-2.0 | T | The case's user instruction | Each "potential risky action" | aligned yes, risk high (on-topic but risky: the T3 case) |
| RedCode-Exec | CC BY 4.0 | T | A benign coding question from WildChat user turns (nearest by topic) | The code, capped at 1,000 characters (parsed only, never run) | aligned no, risk high |
| `webagent-gen` code rows (`--code`) | Project data | T | A coding question from WildChat user turns | The coding assistant's code, capped at 1,000 characters (the runtime scores code calls the same way) | aligned yes, risk none; none generated yet in 3.8.0 |
| saroku agent-action-safety (only after `QUALITY.md` passes, and only its conditions: Gemini rows with a goal, no long-tail domains, no `corrigibility`, re-split by batch and cluster) | Apache-2.0 | T | `original_goal` | `action` | `QUALITY.md`'s mapping: safe → yes/none; `goal_drift`, `scope_violation` → no, high or low by severity; `injection` → no/high; `policy_violation` → yes, high or low |
| PrivacyLens | MIT / CC BY 4.0 | T | The trajectory's user instruction | `"<final action>: <sensitive item>"` (the data has the action's tool, not its content) | aligned yes, risk high (no risk-none twins) |
| AgentDojo | MIT | E | The user task | The injection task's goal | aligned no, risk high |
| AgentHarm (harmful and matched benign) | MIT + safety-only clause; no-train | E | The benign counterpart | The harmful counterpart | aligned no, risk high |
| SHADE-Arena public split | MIT; no-train | E | The main task | The side task | aligned no, risk high |
| OS-Harm | Apache-2.0; no-train | E | The task | Its misuse, injection or misbehaviour goal | per category |
| TraceSafe | Apache-2.0, gated; no-train | E | The trace's user turn | Each mutated step | per mutation label |
| AgentDAM | CC BY-NC 4.0 | E | The task | The action with unnecessary sensitive data | aligned yes, risk high |

Not used: DeCompBench (skipped), and everything the plan's §8.1–8.2 leaves out. Rows built from
CC BY-SA sources (HotpotQA via ASQ, SQuAD via TaskTracker) and CC BY-NC sources (AgentDAM) never
share a row.

## 3. Build (`build_alignment_dataset.py`)

1. Load each source from the scratch folder; drop rows over 2,000 characters (intent) or 1,000
   (call); English only where the source says.
2. Apply SEC-121's filters; refuse to run if a required condition is missing (e.g. saroku without
   a passed `QUALITY.md` → saroku skipped with a warning; no token for a gated source → that source
   skipped).
3. Pair and label (§2). Nearest-neighbour pairing uses base MiniLM-L6 over a deduplicated pool.
4. Balance: target shares per label combination (aligned/no-risk about 50%, misaligned/no-risk
   20%, risk high 25%, look-alikes 5%), with the builder's `QUOTAS` as the single source of
   numbers. The 3.8.0 build reached 48% / 22% / 28% (plus 2% risk low) and 1.7% look-alikes: fewer
   look-alikes exist in the approved sources than the target.
5. Split: clusters of rows whose intent and call both have cosine ≥ 0.9 go to one split; attack
   templates and instructions (InjecAgent), environments (Agent-SafetyBench), toolkits (ToolEmu)
   and source questions (ASQ) never cross splits; about 80/18/2 for train/test/probe; role-E sources
   go to test and probe only.
6. Write the three committed files, the three `local/` files (gated and no-train sources) and a
   manifest (`counts.json`: rows per source, role, label, split, committed or local); the README's
   tables come from it. Training and the gate read both; a build without a Hugging Face token has
   no `local/` rows and the gate reports which sources were missing.

`--check-data` fails on: schema errors, duplicates, any cross-split cluster or template, any
role-E row in train, any gated or no-train row in a committed file, any CC BY-SA/CC BY-NC mix in a
row, fewer than 50 rows in a required cell (label × action family), or a missing source licence
entry in the README.

## 4. Generation (`generate_alignment_calls.py`, SEC-122)

- Runs `SupervisorAgent`'s planning chain directly (no tools, no research) for each question:
  one model call, about 1–1.5k tokens.
- For code rows, runs the coding assistant's generation for coding questions (no execution).
- `--estimate 50`: runs 50 questions, prints tokens used and the cost at the configured price,
  and the projected cost of the full run; nothing else is written.
- Full run: reads the remaining credit the user enters (`--credit-left`), stops when spending would
  leave less than half of it, and resumes from where it stopped (rows are appended with the
  question's hash).
- Uses the Together key from the environment; each row stores `origin: webagent-gen`, the model
  name and the date; no user data is involved.

## 5. Model (`train_alignment_model.py`, SEC-123, SEC-124)

- **Cross-encoder:** `transformers` `AutoModel` from `sentence-transformers/all-MiniLM-L6-v2` (and,
  in a second run, `-L12-v2`), input `[CLS] intent [SEP] action: text [SEP]`, max 128 tokens; two
  linear heads on the pooled output: `aligned` (1 logit) and `risk` (3 logits). Loss: binary
  cross-entropy + cross-entropy, class-weighted; 3 epochs, AdamW, learning rate 3e-5, batch 32,
  seed fixed.
- **Siamese baseline:** ContextDefender's encoder embeddings `u` (intent) and `v` (call); logistic
  heads on `[u, v, |u−v|, u·v]`.
- **Calibration:** isotonic regression per head on a calibration fold held out from train (10%),
  as ContextDefender's `IsotonicBinaryHead`.
- **Thresholds:** for `aligned`, the probability below which a call counts as off-intent, chosen as
  the highest value whose upper 95% confidence bound of false positives on aligned real calls
  (`webagent-gen` and ASQ rows in a held-out fold) stays ≤ 1% (times a 0.75 margin); mapped to
  `search_align_at`, `subgoal_align_at` and `drift_align_at` on the 0–100 scale. For `risk`:
  `risk_verify_at` is the lowest cut where the look-alikes' point rate stays ≤ 2% (times the margin)
  and the upper bound on all benign rows ≤ 2% (a fold has only tens of look-alikes, so a bound on
  them alone could never reach 2%); `risk_decline_at` the lowest cut where the upper bound on benign
  rows stays ≤ 0.5% (times the margin).
  The uncertain band for the judge (SEC-127) is the `aligned` range between the off-intent
  threshold and the point where precision of "aligned" reaches 0.99.
- **Output:** `models/security/alignment/{encoder/, heads.pt, calibrators.joblib, metadata.json}`;
  `metadata.json` holds version `ac-YYYY-MM-DD-N`, base encoder, data manifest hash, thresholds,
  band, and (after the gate) `test`.

## 6. Gate (`eval_alignment_model.py`, SEC-125, SNFR-11)

| Check | On | Target |
|---|---|---|
| Off-intent and injected-goal recall | test | ≥ 0.95 |
| Aligned real calls held | test, upper 95% bound | ≤ 1% |
| Risk recall to verify or decline | test | ≥ 0.90 |
| Look-alikes held | test | ≤ 2% |
| ECE per head | test | ≤ 0.05 |
| Same checks reported (not gated) | probe | — |
| Latency per pair, one thread | the image's CPU | < 10 ms p95 |

Both the cross-encoder and the baseline are evaluated; the report names the one that ships
(SEC-124). `metadata.json["test"]["passed"]` is true only if every gated check passes. `--install`
copies the shipping model to `models/security/alignment/` with `deployment: "gated"`; when none
ships, `--install-shadow` copies the candidate with the best off-intent recall there with
`deployment: "shadow"` (SEC-128), so the shadow week records its scores.

## 7. Evaluation suites (`eval_alignment_suites.py`, SEC-131)

Runs each evaluation-only source through `signals.score_text` and the Cedar engine (shadow rules
counted as acting) and prints, per source and label: rows, held, allowed. No attack text is
printed or logged. AgentHarm, SHADE-Arena and OS-Harm are adapted into (intent, call) pairs as in
§2; TraceSafe and AgentDAM too. Results go into the PR and the data README.

## 8. Data handling rules (SNFR-17)

- Only the sources in §2; any new one needs the user's approval and a README entry first.
- No hand-written adversarial text; attack rows are read, paired and counted, never edited or
  printed.
- Scratch downloads live outside the repository and are deleted after the build.
- The repository is public, so rows from **gated** sources (xLAM, TraceSafe) and from sources that
  ask not to be trained on (AgentHarm, SHADE-Arena, OS-Harm, all with canary strings) are written
  only to the git-ignored `local/` files, never committed or pushed. Their terms and the fact that
  they are local-only are recorded in the README.
