# Spec: an RL generator of hard training rows for checkpoint 2

Status: v3, R0 tooling committed on `release/3.14.0` (pushed, not merged or released), 2026-10-23 ·
Source plan:
[`planning/rl-plan.md`](../../rl-plan.md) (v5) · Extends:
[`../checkpoint2-intent/`](../checkpoint2-intent/README.md) (the classifier: SEC-120 – SEC-128;
data rules: SNFR-17) · Work happens in the `~/webagent-dev` worktree while shadow week 2 runs ·
Releases: the R0 tooling (G11); the look-alike fix (G14, its own commit, released with the
post-shadow-week retrain); R4 (G21). R1–R3 are runs, recorded here; a model that passes the gate
ships through checkpoint2-intent's install and release steps

## What this work does

The checkpoint 2 classifier (`ac-2026-10-17-1`) fails its gate. It catches the attack styles it was
trained on, but it often can't tell a call that is on the intent's topic from one that serves the
intent, and it has too few look-alikes to learn from. Both gaps are in **benign** rows.

This spec builds a small generator (Qwen2.5 Instruct) that writes benign rows the current classifier
gets wrong, in two modes:

- **H**: harmless calls that serve a nearby but different goal (label: aligned no, risk none);
- **L**: harmless calls that serve the goal but sound risky (label: aligned yes, risk none).

The generator is trained with GRPO, rewarded when an independent LLM judge confirms the label *and*
the frozen classifier gets the row wrong. Its rows, after review by the user, become ordinary
supervised training rows. The same rows are also made by prompting the untrained model, and RL is
kept only for a mode where it beats prompting on the gate.

It also builds, but doesn't train against, an evaluation set and rewards for the research agent
(R4).

## Documents

| File | What it holds |
|---|---|
| [`requirements.md`](requirements.md) | SEC-200 – SEC-218, SEC-220 – SEC-222 and SNFR-20 – SNFR-28, with checks |
| [`design-high-level.md`](design-high-level.md) | Components, flows, where data goes, failure handling, limits |
| [`design-low-level-generator.md`](design-low-level-generator.md) | Seeds, prompts, rewards, judge, prompted generation, GRPO trainer, Colab, review |
| [`design-low-level-data-eval.md`](design-low-level-data-eval.md) | Sources, variants, training and gate changes, the R2 comparison and rounds, agent rewards (R4) |
| [`tasks.md`](tasks.md) | Tasks per phase in dependency order, with coverage |

Requirements say *what*; the designs say *how*; tasks say *in what order*. When documents
disagree, `requirements.md` wins, then the designs, then the plan.

## Scope

In scope: the plan's §3 (modes H and L, including §3.7 Colab), §5's R4 task, §7's phases R0–R4,
and the look-alike fix (O1).

Out of scope:

- mode A (attack rows), and any generated adversarial text (D1);
- RL on any encoder; training the research agent (D8);
- changing the gate's checks (SEC-125; D7), changing anything the server runs;
- ContextDefender rows (the plan's §4, later).

## Decisions

All taken by the user on 2026-10-23 (plan §9):

1. **D1 Benign first.** Only modes H and L. The standing rule (no generated or hand-written
   adversarial training text) is unchanged. Mode A is decided after R2.
2. **D2 Compute.** No GPU in R0; R1 trains on Google Colab Pro+.
3. **D3 Judge.** A Together model from a different family than `meta-models/Muse-Glimmer-30B`,
   on the existing credit; the exact model is picked in R0 by agreement with the user's labels.
4. **D4 Generator.** Qwen2.5-0.5B-Instruct for tests, Qwen2.5-1.5B-Instruct for R1 (both
   Apache-2.0).
5. **D5 Review.** The user labels 100 rows per round and mode; 100 more when disagreement is
   between 6% and 14%.
6. **D6 Share.** Generated rows are at most 15% of training rows; R2 compares 5% and 15%.
7. **D7 Gate.** Unchanged; a report split by role (T / E) is added; revisited after R2.
8. **D8 Agent.** No agent training; R4 builds a graded set of 100 questions and two rewards.
9. **Sources approved.** `rl-gen-h` and `rl-gen-l` as new training sources (SNFR-17), approved
   2026-10-23. Their data README entries come with task G1 (SEC-200).
10. **O1 Look-alike count.** Fix the gate's look-alike origin before R2 (SEC-218, task G14).

Made while writing this spec (the plan follows them):

11. **The prompted baseline uses the untrained generator** on the Mac, not a Together model, so
    only RL differs between the two kinds of rows. The judge is the only Together cost.
12. **Generated rows stay in `local/`**, git-ignored, for this whole spec. Committing them would
    be a separate decision.
13. **Seeds come from the committed train split only**, and a generated row joins its seed's
    groups, so a generated row can never share an intent with a test or probe row.
14. **A harmless filter.** Mode L rewards rows the classifier holds as risky, which could pull
    the generator towards text that really is harmful. ContextDefender's harm score (at the
    installed model's threshold, 20) and the judge both have to clear a row, otherwise its reward
    is 0 and it's never stored (SEC-205).
15. **Novelty only for valid samples** (SEC-207): the plan's first reward added novelty to every
    sample, which would pay for novel text with the wrong label.
16. **Share per mode.** The share counts both modes; each mode's part is half. R2 compares each
    mode alone at its half (2.5% and 7.5%), the amount it would have in the combined data
    (SEC-213, SEC-217).
17. **Pilot-sized batches before R2.** R0 and R1 make 500-row batches per mode (enough to
    review and measure). R2's full batches (about 2,700 rows per mode) are made against the
    post-shadow-week model, since batches must match the base build's data hash.

## Findings that shaped the spec

| Finding | Where | Effect |
|---|---|---|
| The gate's look-alike check counts NotInject and Agent-SafetyBench twins (110 rows), not XSTest's safe prompts: `LOOKALIKE_ORIGINS` has `xstest:safe`, but the rows are `xstest:lookalike` | `scripts/alignment_training.py:22`; model-data design §2 says XSTest safe prompts are look-alikes | SEC-218 (open item O1, decided) |
| `origin_matches` matches by prefix, so a new origin starting `asq` or `webagent-gen` would count as a real call, and one starting `lookalike` as a look-alike | `alignment_training.py:45-47` | Generated origins are `rl-gen-h:*` and `rl-gen-l:*` (SEC-200) |
| `split_calibration` takes a random 10% of all train rows, and thresholds come from that fold's real calls and look-alikes | `train_alignment_model.py:51-58` | Generated rows are kept out of it (SEC-214) |
| 52% of the gate's risk-recall rows are evaluation-only sources | test split counts (plan §2.2) | Role report (SEC-215); D7 revisited after R2 |
| The installed ContextDefender's `harm_verify_at` is 20 (`models/security/metadata.json`), not the code default of 50 | `app/security/thresholds.py` `merged()` | SEC-205 reads `context_defender.current_thresholds()` |
| `check_data(readme=...)` checks each source's `SOURCES` title is in the data README | `alignment_data.py:155-159` | SEC-200's README entries are enforced; variants are checked with the base's README |
| 34,538 real train rows today (31,085 after the calibration fold); 17,206 seed candidates (ASQ 5,512, `webagent-gen` 4,809, ToolACE 3,209, saroku 2,013, Glaive 1,663) | current build | Share and seed numbers in the designs |
| `generate_alignment_calls.py` runs the supervisor; its cost estimate and stop rule are reusable, the rest isn't | `generate_alignment_calls.py` | New `rl_generator/prompted.py` imports the helpers |

## Implementation notes (3.14.0, unreleased)

Built in the `~/webagent-dev` worktree and committed on `release/3.14.0`. Tests: the full backend suite passes
(1,400 passed, 4 skipped); the generator tests also pass with TRL 1.14.2 in a scratch venv (123).
R4 (G18–G21: `build_research_eval.py`, `score_research_run.py`) isn't built yet; it follows R2.

What was checked with real models on the Mac (fake judge, so Together was never called):

| Check | Result |
|---|---|
| Prompted smoke run (G4): Qwen2.5-0.5B, 5 seeds | 40 samples; 7 removed by the harmless filter; 10 judged; 5 rows written, all valid; 3.6 s per seed |
| Seeds (G2): `seeds.py --n 12000`, mode H | 7,200 train and 4,800 sample seeds; every origin under a third, every action over 10% |
| Novelty bank | 25,476 unique committed train calls, 19.6 MB, data hash `8e9fdd8db77d99c8` |
| GRPO (G8): Qwen2.5-0.5B on CPU, 2 steps, then resume to 3, then `sample.py --n 3` | Checkpoints, metrics and `best_step` written; resume continued at step 2; 3 valid rows; about 160–200 s per CPU step |
| `none` variant (G10), trained with `--seed 13` | Gate numbers identical to `ac-2026-10-17-1` (off-intent 0.5078, risk 0.4431, look-alikes 0.1, real held 15 / 2,438, ECE 0.0619 / 0.1075); about 4 min on MPS. By role: T off-intent 0.554, risk 0.641; E off-intent 0.361, risk 0.260 |

Where the code fills gaps the designs left open. These notes take precedence over the designs where
they differ; `requirements.md` has been updated to match:

- **Judge.** The cache key includes mode and action. The error stop needs at least 10 calls in its
  window. `PRICES` starts empty, so `--price-input` and `--price-output` are required until G6.
  The cost formula and token totals are the same as `generate_alignment_calls.py`'s, read directly
  (`app.core.observability`), because importing that script sets guard and drift settings for the
  whole process. In judge choice, a candidate's failed call is counted as `failed`, not as a
  rejection.
- **Files and flags.** Seed and batch file names use the mode in lower case (`seeds_h.jsonl`).
  `prompted.py --resume` continues the file's last batch; without it, each run starts a new
  batch. Row provenance adds `seed_origin`, which review stratifies by.
- **Trainer.**
  - Checkpoint rotation keeps the newest 4 plus the guard's fallback (the latest checkpoint at or
    before step − 200). HF's own limit would have deleted the fallback.
  - The guard fires only on a full 200-step window. "Miss rises" means a positive slope of
    `miss_mean`.
  - Scorers run on CPU only, because the app's models are CPU-only. Training uses fp32 on CPU,
    bf16 or fp16 on CUDA, and never MPS.
  - `top_p` stays at TRL's default of 1.0 during training; sampling uses 0.95.
  - TRL counts completions per device, so the batch is set as 2 prompts × 8 = 16.
- **Calibration fold.** Train rows keep the old shuffled order, followed by the generated rows. The
  design's sketch would have reordered training, so it was not followed.
- **Rounds.** `variant.py --batches` takes only the named batch ids, so an R3 round's batches
  replace the previous round's (data-eval design §7).
- **R2 decision.** RL is kept for a mode if it beats prompting at either share; 15% vs 5% is then
  judged within the winning kind. The real-aligned limit applies to every seed of the challenger.
- **Training environment.** It needs `pyarrow==25.0.1`, because pyarrow 26 refuses numpy below 2.
  `fsspec` is 2026.6.0 there.

## Open items

| ID | Item | Recommendation | Status |
|---|---|---|---|
| O1 | Fix `LOOKALIKE_ORIGINS` (`xstest:safe` → `xstest:lookalike`) so the gate counts the 125 XSTest look-alikes, as the data spec says? It changes SEC-125's look-alike number for every model | Fix it before R2's baseline retrain, in its own commit, and re-report `ac-2026-10-17-1` with both counts | **decided 2026-10-23:** fix before R2 (SEC-218, task G14) |
| O2 | Together credit left, for the judge's stop rule | Needed before G7's full runs | open (user) |
| O3 | Colab compute-unit cap for R1 | Set from G9's pilot numbers, before G12 | open (user) |
| O4 | Tavily credits for R4's baseline run (about 100 questions × 4 searches) | Approve with R4 | open (user) |
