# Alignment classifier data (checkpoint2-intent SEC-120, SEC-121)

Training, release-gate and probe data for the checkpoint 2 alignment classifier: does an agent's
call serve the user's intent (`aligned`), and is it riskier than the intent (`risk`)? Built from
public datasets and WebAgent's own generated calls by `scripts/build_alignment_dataset.py`, after
`scripts/fetch_alignment_sources.py` downloads them. Design: `planning/specs/checkpoint2-intent/`
(model-and-data design §2–§3, §8).

**Content warning:** the risky rows contain real prompt-injection, harmful-request and risky-code
text from published research datasets. They are classifier data only. No row was written by hand.

| File | Rows | Use |
|---|---|---|
| `train.jsonl` | 26,760 | Training and calibration |
| `test.jsonl` | 7,213 | Release gate (`scripts/eval_alignment_model.py`) |
| `probe.jsonl` | 630 | Second check, reported and not gated |
| `local/{train,test,probe}.jsonl` | 0 / 322 / 28 here | **Git-ignored.** Rows from gated and no-train sources, built on the machine (see below) |
| `counts.json` | — | Manifest: rows per source, role, label, action, split, committed or local; dropped rows; quotas |
| `QUALITY.md` | — | The saroku quality check (passed, with conditions and its label mapping) |
| `generated/webagent_gen.jsonl` | 50 tasks | Our own planner's rewrites and plan questions (`scripts/generate_alignment_calls.py`) |
| `generated/webagent_gen_code.jsonl` | none yet | Our coding assistant's code for coding questions (`generate_alignment_calls.py --code`; the code is never run), read as aligned `execute_code` rows when present |

Each row:

```json
{"intent": "...", "call": "...", "action": "web_search", "labels": {"aligned": "yes", "risk": "none"},
 "origin": "<dataset>:<role>", "role": "T", "split": "train", "groups": ["asq:<qid>"],
 "sources": ["asq"], "cluster": "<id>"}
```

- `action` is the call's family, as the classifier reads it (`"<action>: <call>"`): `web_search`
  and `internal_retrieve` (search queries), `plan` (the supervisor's rewrite, a plan question or a
  subgoal), `execute_code` (code, never run), `function` (a tool call from a function-calling or
  agent-safety set: `"<function>: <arguments>"`).
- `role`: `T` (train and test) or `E` (evaluation only: test and probe, never train).
- `groups`: the units that never cross splits (source question, template and instruction,
  environment, toolkit, generation batch, function name); `sources`: every dataset whose text is in
  the row (licence checks); `cluster`: the near-duplicate cluster.

## Composition (34,953 rows: 34,603 committed, 350 local)

Target shares (the builder's `QUOTAS`, the single source of the numbers): aligned with no risk
about 50%, misaligned with no risk 20%, risk above intent 25%, look-alikes 5% (only about 2% exist).

| Group | Rows | Origin |
|---|---|---|
| Aligned, no risk | 16,913 | ASQ agent queries for HotpotQA questions (7,000), ToolACE calls (3,990), Glaive calls (2,115), saroku safe actions (2,499), Agent-SafetyBench benign twins (322), WebAgent's own rewrites and plan questions (228), AgentDAM required content (173, test only); look-alikes: NotInject prompts (336) and XSTest safe prompts (250) |
| Misaligned, no risk | 7,661 | Re-pairs of aligned calls under another question of the same split, random (easy: 2,985) and nearest by MiniLM below cosine 0.8 (hard: 3,176); TaskTracker secondary tasks under SQuAD questions (1,500) |
| Risk above intent (high) | 9,620 | Aegis 2.0 dangerous-category prompts (1,664) and HarmBench behaviours (84) under the nearest benign question (cosine ≥ 0.5); RedCode-Exec code (1,296); PrivacyLens leaking final actions (1,000); saroku unsafe actions of high severity and its injection rows (2,741); InjecAgent attacker instructions (570); Agent-SafetyBench unfulfillable cases (399); ToolEmu risky actions (247); XSTest unsafe contrasts (150); AgentDojo injection goals (561, evaluation only); AgentDAM unneeded sensitive data (558, test only); AgentHarm (208), OS-Harm (140) and SHADE-Arena (2), local only |
| Risk above intent (low) | 759 | saroku actions of medium or low severity (goal drift, scope and policy violations) |

By split and label (committed; local in brackets):

| Label | train | test | probe |
|---|---|---|---|
| aligned yes, risk none | 13,439 | 3,145 | 329 |
| aligned no, risk none | 6,199 | 1,310 | 152 |
| aligned no, risk high | 4,740 | 1,610 (272) | 88 (28) |
| aligned yes, risk high | 1,775 | 1,010 (50) | 47 |
| aligned no, risk low | 407 | 104 | 11 |
| aligned yes, risk low | 200 | 34 | 3 |

By action (committed, all splits): `function` 18,712, `web_search` 6,473, `internal_retrieve`
5,367, `plan` 2,755, `execute_code` 1,296. `execute_code` has risky rows only until the coding
assistant's generated code rows exist (task B3); its aligned cells are not required yet.

## How it is built

- **Lengths:** rows with an intent over 2,000 characters or a call over 1,000 are dropped (391).
- **WebAgent generation:** the rewrite and each plan question are aligned calls for their question,
  except tasks whose rewrite scores below `rewrite_at` (40) against the question with the base
  encoder (`signals.cos100` of `alignment.encode`): the planner replaced the user's goal there (2 of
  50 tasks, listed in `counts.json` under `review_rewrite_below_rewrite_at`).
- **ASQ:** only source question sets in `ASQ_SOURCES` (`scripts/alignment_data.py`): HotpotQA's test
  questions (via BEIR, CC BY-SA 4.0). Up to 3 queries per question; half the questions are labelled
  `internal_retrieve` (the agents searched a local corpus), half `web_search`.
- **Function calling:** Glaive v2 and ToolACE (and xLAM when a token is available): the first user
  turn and the first call, each capped at 4,000 rows; split by function name.
- **Re-pairs:** an aligned call under another question of the same split: random, or the nearest
  question by MiniLM that is not a paraphrase (cosine < 0.8) and shares no group.
- **Harmful requests:** each Aegis 2.0 dangerous-category prompt (as ContextDefender's `aegis_group`,
  no jailbreak wording) and HarmBench standard behaviour is paired with the nearest benign question
  of the same split (aligned rows' intents and Aegis safe prompts; cosine ≥ 0.5), or dropped (1,191).
- **XSTest:** safe prompts as intents with the unsafe contrast prompts of the matching category;
  safe prompts and NotInject prompts also as look-alike calls for themselves (aligned, no risk).
- **Agent-SafetyBench:** the release has no per-case source field, so derived cases are dropped by
  what marks them: the "Produce unsafe information" category (holds the 200 AdvBench cases; 250
  dropped), failure mode M9 "trusts tool results" (the injection cases from AgentDojo and InjecAgent;
  358), environments of AgentDojo's suites (banking, Slack, travel, email, calendar, drive; 108), and
  cases without an environment (161). In each environment, a fulfillable instruction is the intent,
  an unfulfillable one the risky call, another fulfillable one the benign twin. Split by environment.
- **ToolEmu:** the instruction and each potential risky action (on topic but risky); split by
  toolkit (every toolkit of a case).
- **RedCode-Exec:** the code (never run) under the nearest WildChat coding question by topic (the
  intents our generated code rows will provide after task B3); split by risk-type file.
- **saroku:** only while `QUALITY.md` says passed, with its conditions and mapping; split by
  generation batch.
- **PrivacyLens:** the user instruction and `"<final action>: <sensitive item>"` (on task, leaks what
  the task didn't need). **AgentDAM:** `"type: <item>"` for its unneeded sensitive data (risk high)
  and its required content (risk none), test only.
- **InjecAgent:** split by template and by attacker instruction, both held out (33% of each for
  test); cases mixing a training and a test value are dropped, as for ContextDefender.
- **Splits:** each group's split comes from a hash (about 80/18/2; evaluation-only groups 90/10 test
  and probe); a row whose groups fall in different splits is dropped (514). Rows whose intents and
  calls both have MiniLM cosine ≥ 0.9 form a cluster kept in its most common split; other members are
  dropped (79). Exact duplicates are dropped (2,236).
- `--check-data` passes: schema, duplicates, groups and near-duplicate clusters crossing splits,
  evaluation-only rows in train, AgentDAM outside test, gated or no-train rows in a committed file,
  CC BY-SA and CC BY-NC text in one row, at least 50 rows per required (action, label) cell, and a
  licence entry here for every source.

Rebuild (from `backend/`; the scratch folder is outside the repository and deleted after the build):

```bash
hf auth login   # once, for xLAM and TraceSafe (token stays in its standard location)
python scripts/fetch_alignment_sources.py --out /tmp/alignsrc
python scripts/build_alignment_dataset.py --raw /tmp/alignsrc
python scripts/build_alignment_dataset.py --check-data
```

## Local-only rows (`local/`, git-ignored)

The repository is public. Rows from gated sources (xLAM, TraceSafe) and from sources whose authors
ask that they not be trained on (AgentHarm, SHADE-Arena, OS-Harm; all carry canary strings) are
written only to `local/`, never committed or pushed. Training and the gate read both folders; a build
without a Hugging Face token has no xLAM or TraceSafe rows, and the gate reports them as missing.
This build had no token: xLAM and TraceSafe were skipped.

## Sources, licences, roles and attribution

| Dataset | Licence | Role | Local only | Notes |
|---|---|---|---|---|
| WebAgent generation (`webagent-gen`) | Project data | T | no | Supervisor planning over triage questions (`data/triage/train.jsonl`), model and date in each row |
| [ASQ](https://huggingface.co/datasets/AgenticSearchQueryset/ASQ) | MIT | T | no | Agent-written queries; joined to [HotpotQA via BEIR](https://huggingface.co/datasets/BeIR/hotpotqa) questions (CC BY-SA 4.0: rows stay share-alike, never mixed with CC BY-NC). Pezzuti et al., 2026 |
| [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | T | no | First request and call, capped |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | T | no | First request and call, capped. Liu et al., 2024 |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC BY 4.0, gated | T | **yes** (gated) | Terms accepted on the user's account; read with the user's token; capped. Liu et al., "APIGen", 2024 |
| [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | MIT | T | no | User instruction and attacker instruction. Zhan et al., ACL Findings 2024 |
| [TaskTracker](https://github.com/microsoft/TaskTracker) | MIT | T | no | Secondary tasks as calls. Abdelnabi et al., SaTML 2025 |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | CC BY-SA 4.0 | T | no | Questions as intents for TaskTracker rows; share-alike, never mixed with CC BY-NC |
| [XSTest](https://huggingface.co/datasets/natolambert/xstest-v2-copy) | CC BY 4.0 | T | no | Safe prompts, unsafe contrasts by category. Röttger et al., NAACL 2024 |
| [NotInject](https://huggingface.co/datasets/leolee99/NotInject) | MIT | T | no | Look-alikes. Li et al., 2024 |
| [Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | CC BY 4.0 | T | no | Dangerous-category prompts and safe prompts. Ghosh et al., NAACL 2025 |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT | T | no | Standard behaviours. Mazeika et al., 2024 |
| [Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench) | MIT | T | no | Without AdvBench-, AgentDojo- and InjecAgent-derived cases (above). Zhang et al., 2024 |
| [ToolEmu](https://github.com/ryoungj/ToolEmu) | Apache-2.0 | T | no | `assets/all_cases.json`. Ruan et al., ICLR 2024 |
| [RedCode](https://github.com/AI-secure/RedCode) | Data CC BY 4.0 (`dataset/LICENSE`), code MIT | T | no | RedCode-Exec code, parsed only, never executed. Guo et al., NeurIPS 2024 Datasets and Benchmarks |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-BY | T | no | English, non-toxic, unredacted first user turns asking for code, as intents for RedCode. Zhao et al., ICLR 2024 |
| [saroku agent-action-safety](https://huggingface.co/datasets/karanxa/agent-action-safety-dataset) | Apache-2.0 | T | no | After `QUALITY.md` (Gemini-generated rows only). karanxa / saroku, 2026 |
| [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) | MIT (repository), CC BY 4.0 (dataset card) | T | no | `data/main_data.json`. Shao et al., NeurIPS 2024 Datasets and Benchmarks |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | MIT | E | no | User-task prompts and injection-task goals of the v1 suites (Python parsed, not run). Debenedetti et al., NeurIPS 2024 |
| [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | MIT + clause limiting use to AI safety and security; no-train | E | **yes** (no-train) | Harmful and matched benign behaviours. Andriushchenko et al., ICLR 2025 |
| [SHADE-Arena](https://github.com/jkutaso/SHADE-Arena) public split | MIT; no-train | E | **yes** (no-train) | Main task and side task of each pair (4 of 5 parsed; 2 within the length limits) |
| [OS-Harm](https://github.com/tml-epfl/os-harm) | Apache-2.0; no-train | E | **yes** (no-train) | Task configs: injection goals, misbehaviour explanations, misuse requests. Kuntz et al., 2025 |
| [TraceSafe](https://huggingface.co/datasets/CyCraftAI/TraceSafe) | Apache-2.0, gated; "a benchmark, not a training set" | E | **yes** (gated, no-train) | Access granted on the user's account; needs a local token; the adapter's field names are unverified until then |
| [AgentDAM](https://github.com/facebookresearch/ai-agent-privacy) | CC BY-NC 4.0 | E (test only) | no | Non-commercial (WebAgent is non-commercial); never in a row with CC BY-SA text. Zharmagambetov et al., 2025 |

Not used: **DeCompBench** (skipped for now, plan §12); **R-Judge** and **SafeArena** (no licence);
OpenAgentSafety (licence conflict); ASB, ST-WebAgentBench, ToolSafety, HarmActions (not now);
HarmfulSkillBench (academic-only terms); Nemotron-AIQ Agentic Safety (evaluation-only licence); MS
MARCO and Natural Questions; ASQ's MS MARCO and Researchy Questions sets (MS MARCO's research terms;
Researchy Questions' CDLA-Permissive-2.0 not yet approved); saroku's `template` and `seed` rows and
its 13 long-tail domains (from an unnamed external corpus without a stated licence) and its
`corrigibility` rows (`QUALITY.md`).
