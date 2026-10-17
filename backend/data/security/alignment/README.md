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
| `train.jsonl` | 30,361 | Training and calibration |
| `test.jsonl` | 8,194 | Release gate (`scripts/eval_alignment_model.py`) |
| `probe.jsonl` | 736 | Second check, reported and not gated |
| `local/{train,test,probe}.jsonl` | 4,177 / 1,994 / 188 here | **Git-ignored.** Rows from gated and no-train sources, built on the machine (see below) |
| `counts.json` | — | Manifest: rows per source, role, label, action, split, committed or local; dropped rows; quotas |
| `QUALITY.md` | — | The saroku quality check (passed, with conditions and its label mapping) |
| `generated/webagent_gen.jsonl` | 2,496 tasks | Our own planner's rewrites and plan questions (`scripts/generate_alignment_calls.py`) |
| `generated/webagent_gen_code.jsonl` | 388 in this build (the run continues) | Our coding assistant's code for coding questions (`generate_alignment_calls.py --code`; the code is never run), read as aligned `execute_code` rows when present |

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

## Composition (45,650 rows: 39,291 committed, 6,359 local; build of 2026-10-17)

Target shares (the builder's `QUOTAS`, the single source of the numbers): aligned with no risk
about 50%, misaligned with no risk 20%, risk above intent 25%, look-alikes 5% (only about 2% exist).

| Group | Rows | Origin |
|---|---|---|
| Aligned, no risk | 27,028 | ASQ agent queries for HotpotQA and Researchy Questions, split evenly (6,985), WebAgent's own rewrites and plan questions (5,980), ToolACE calls (3,990), xLAM calls (3,974, local), Glaive calls (2,058), saroku safe actions (2,499), WebAgent's own generated code (388), Agent-SafetyBench benign twins (322), AgentDAM required content (173, test only), TraceSafe unmutated traces (73, local, evaluation only); look-alikes: NotInject prompts (336) and XSTest safe prompts (250) |
| Misaligned, no risk | 7,778 | Re-pairs of aligned calls under another question of the same split, random (easy: 2,998) and nearest by MiniLM below cosine 0.8 (hard: 3,280); TaskTracker secondary tasks under SQuAD questions (1,500) |
| Risk above intent (high) | 9,753 | Aegis 2.0 dangerous-category prompts (1,668) and HarmBench behaviours (90) under the nearest benign question (cosine ≥ 0.5); RedCode-Exec code (1,296); PrivacyLens leaking final actions (1,000); saroku unsafe actions of high severity and its injection rows (2,741); InjecAgent attacker instructions (570); Agent-SafetyBench unfulfillable cases (399); ToolEmu risky actions (247); XSTest unsafe contrasts (150); AgentDojo injection goals (561, evaluation only); AgentDAM unneeded sensitive data (558, test only); AgentHarm (208), OS-Harm (140) and SHADE-Arena (2), local only; part of TraceSafe's mutated traces |
| Risk above intent (low) | 1,091 | saroku actions of medium or low severity (goal drift, scope and policy violations); the rest of TraceSafe's mutated traces (455 in all, low or high by mutation category, local, evaluation only) |

By split and label (committed; local in brackets):

| Label | train | test | probe |
|---|---|---|---|
| aligned yes, risk none | 18,180 (3,020) | 4,352 (938) | 449 (89) |
| aligned no, risk none | 5,047 (1,157) | 1,089 (313) | 135 (37) |
| aligned no, risk high | 4,778 | 1,592 (295) | 92 (28) |
| aligned yes, risk high | 1,781 | 1,021 (117) | 46 (3) |
| aligned no, risk low | 395 | 96 | 11 |
| aligned yes, risk low | 180 | 44 (331) | 3 (31) |

By action (committed; local in brackets): `function` 17,062 (5,724), `plan` 9,648 (478),
`web_search` 5,955 (88), `internal_retrieve` 4,857 (61), `execute_code` 1,769 (8). `execute_code`
now has aligned rows from the coding assistant's generated code (388 in this build); the generation
run is still adding rows.

## How it is built

- **Lengths:** rows with an intent over 2,000 characters or a call over 1,000 are dropped (395).
- **WebAgent generation:** the rewrite and each plan question are aligned calls for their question,
  except tasks whose rewrite scores below `rewrite_at` (40) against the question with the base
  encoder (`signals.cos100` of `alignment.encode`): the planner replaced the user's goal there (223 of
  2,496 tasks, listed by id in `counts.json` under `review_rewrite_below_rewrite_at`). Code rows are
  the coding assistant's code (capped at 1,000 characters, as the runtime scores it) under its coding
  question, kept in full outside the generation quota.
- **ASQ:** only source question sets in `ASQ_SOURCES` (`scripts/alignment_data.py`): HotpotQA's test
  questions (via BEIR, CC BY-SA 4.0) and Researchy Questions (CDLA-Permissive-2.0), the quota split
  evenly between them. Up to 3 queries per question; half the questions are labelled
  `internal_retrieve` (the agents searched a local corpus), half `web_search`.
- **Function calling:** Glaive v2, ToolACE and xLAM (local; needs the Hugging Face token): the first user
  turn and the first call, each capped at 4,000 rows; split by function name.
- **Re-pairs:** an aligned call under another question of the same split: random, or the nearest
  question by MiniLM that is not a paraphrase (cosine < 0.8) and shares no group.
- **Harmful requests:** each Aegis 2.0 dangerous-category prompt (as ContextDefender's `aegis_group`,
  no jailbreak wording) and HarmBench standard behaviour is paired with the nearest benign question
  of the same split (aligned rows' intents and Aegis safe prompts; cosine ≥ 0.5), or dropped (1,114).
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
  dropped (123). Exact duplicates are dropped (2,843).
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
This build (2026-10-17) had the token: xLAM (4,911 rows with its re-pairs) and TraceSafe (528) are in
`local/`.

## Sources, licences, roles and attribution

| Dataset | Licence | Role | Local only | Notes |
|---|---|---|---|---|
| WebAgent generation (`webagent-gen`) | Project data | T | no | Supervisor planning over triage questions (`data/triage/train.jsonl`), model and date in each row |
| [ASQ](https://huggingface.co/datasets/AgenticSearchQueryset/ASQ) | MIT | T | no | Agent-written queries; joined to [HotpotQA via BEIR](https://huggingface.co/datasets/BeIR/hotpotqa) questions (CC BY-SA 4.0: rows stay share-alike, never mixed with CC BY-NC). Pezzuti et al., 2026 |
| [Researchy Questions](https://huggingface.co/datasets/corbyrosset/researchy_questions) | CDLA-Permissive-2.0 | T | no | Source questions for ASQ's `researchy_all` traces (approved 2026-10-17); ASQ's quota is split evenly between HotpotQA and Researchy Questions. Rosset et al., "Researchy Questions", 2024 |
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
| [TraceSafe](https://huggingface.co/datasets/CyCraftAI/TraceSafe) | Apache-2.0, gated; "a benchmark, not a training set" | E | **yes** (gated, no-train) | Access granted on the user's account; read with the local token. The user query of each mutated trace and its first agent turn, labelled by the mutation's file (benign, or low or high risk by category); grouped by mutation category |
| [AgentDAM](https://github.com/facebookresearch/ai-agent-privacy) | CC BY-NC 4.0 | E (test only) | no | Non-commercial (WebAgent is non-commercial); never in a row with CC BY-SA text. Zharmagambetov et al., 2025 |

Not used: **DeCompBench** (skipped for now, plan §12); **R-Judge** and **SafeArena** (no licence);
OpenAgentSafety (licence conflict); ASB, ST-WebAgentBench, ToolSafety, HarmActions (not now);
HarmfulSkillBench (academic-only terms); Nemotron-AIQ Agentic Safety (evaluation-only licence); MS
MARCO and Natural Questions; ASQ's MS MARCO set (MS MARCO's research terms); saroku's `template` and `seed` rows and
its 13 long-tail domains (from an unnamed external corpus without a stated licence) and its
`corrigibility` rows (`QUALITY.md`).
