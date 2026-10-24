# Low-level design: data, training, gate and comparison

Paths are relative to `backend/`. Requirement IDs refer to [`requirements.md`](requirements.md).
`LOCAL` is `data/security/alignment/local/` (git-ignored). The builder, training and gate are
described in [`../checkpoint2-intent/design-low-level-model-data.md`](../checkpoint2-intent/design-low-level-model-data.md).

## 1. Files

| File | Change | Requirement |
|---|---|---|
| `scripts/alignment_data.py` | Two sources; `GENERATED`; a rule for generated rows | SEC-200, SEC-213 |
| `data/security/alignment/README.md` | Entries for `rl-gen-h`, `rl-gen-l` | SEC-200 |
| `scripts/rl_generator/variant.py` (new) | A dataset folder = the base build + reviewed generated rows | SEC-213 |
| `scripts/train_alignment_model.py` | `split_calibration` without generated rows; `--seed` | SEC-214, SEC-217 |
| `scripts/eval_alignment_model.py` | `by_role`; `--extra` | SEC-215, SEC-216 |
| `scripts/rl_generator/compare.py` (new) | Runs and tabulates R2's configurations and R3's rounds | SEC-217 |
| `scripts/alignment_training.py` | `LOOKALIKE_ORIGINS`: `xstest:lookalike` in place of `xstest:safe` | SEC-218 |
| `data/research_eval/questions.jsonl`, `scripts/build_research_eval.py` (new) | The graded set | SEC-220 |
| `scripts/score_research_run.py` (new) | Agent rewards | SEC-221, SEC-222 |

`build_alignment_dataset.py` doesn't change: the base build is made as today, and generated rows
are added afterwards by `variant.py`. That keeps the base rows and their splits identical across
every configuration in R2.

## 2. Sources (`alignment_data.py`, SEC-200)

```python
SOURCES["rl-gen-h"] = ("project data (Qwen2.5 output, Apache-2.0)", "T", True, "WebAgent RL generator, mode H")
SOURCES["rl-gen-l"] = ("project data (Qwen2.5 output, Apache-2.0)", "T", True, "WebAgent RL generator, mode L")
GENERATED = {"rl-gen-h", "rl-gen-l"}
```

- The third field (`True`) makes `is_local` true, so a generated row can only be in `local/`
  files: the existing committed-file rule then covers it.
- `rule_errors` adds: a row with a `GENERATED` source outside `train` → "generated row outside
  train".
- `schema_errors` already requires the origin's dataset in `sources`, and every source in
  `SOURCES`; generated rows meet both (`sources = ["rl-gen-h", *seed sources]`).
- The share-alike and non-commercial checks apply unchanged through the seed's sources.
- `check_data(readme=...)` already requires each source in the data README, so the README entries
  are enforced.
- `alignment_training.REAL_CALL_ORIGINS` and `LOOKALIKE_ORIGINS` are unchanged; a test asserts
  `rl-gen-*` origins match neither (README finding on prefix matching).

README entry (both sources, each with its `SOURCES` title, which `check_data` looks for): what the
generator is, the two prompts' purpose (not their text), the licence, "approved by the user
2026-10-23", local-only, train-only, review rule, and counts per batch (filled when built).

## 3. Variants (`rl_generator/variant.py`, SEC-213)

```
python scripts/rl_generator/variant.py --base data/security/alignment --kind rl --share 0.15 \
    --modes H,L --out LOCAL/variants/rl-15
```

Steps:

1. Copy the base folder's `{train,test,probe}.jsonl` and `local/{train,test,probe}.jsonl` into
   `--out` (and `--out/local/`). `--out` must be under `LOCAL`.
2. Read `LOCAL/generated/rl_review.json`; keep batches of `--kind` (only those named by `--batches`,
   when given) with verdict `pass` whose rows'
   `data_hash` equals the base's `data_hash`. Held-out batches (`<mode>-heldout-*`, in
   `rl_gen_<mode>_heldout.jsonl`) are refused.
3. **Leakage drops**, counted per reason:
   - the row's seed `groups` aren't all in the base `train` split;
   - the row's call, embedded with base MiniLM-L6, has cosine ≥ 0.9 (`NEAR_DUPLICATE`) to any
     base test or probe call;
   - an exact (intent, call, action) duplicate of any base row or an earlier generated row.
4. **Cap:** each mode named in `--modes` gets at most `round(share / 2 × real train rows)`
   (real train rows = the base's `train` plus `local/train`: 34,538 today, so 864 rows per mode at
   5% and 2,590 at 15%). One mode alone still gets only its half, so a single-mode variant has
   what that mode would have in the combined data. If a mode has fewer rows, the other mode
   doesn't take its place, and `variant.json` records the shortfall. Rows are chosen uniformly at
   random (seed 13), not by `miss`, so the cap doesn't bias towards extremes.
5. Strip `meta` (provenance stays in the batch files), set `split: "train"`, `cluster:
   "gen-<seed_id>"`, append to `--out/local/train.jsonl`.
6. Run `check_data(--out, embed, <base>/README.md)`; any error removes `--out` and exits non-zero.
7. Write `--out/variant.json`: base data hash, kind, share, modes, batches used, counts kept and
   dropped per reason. Printed as counts.

`--kind none --share 0` makes the baseline variant (a copy of the base), so every R2
configuration is trained from a variant folder the same way.

## 4. Training (`train_alignment_model.py`, SEC-214, SEC-217)

```python
def split_calibration(rows, share=0.1, seed=SEED):
    real = [i for i, r in enumerate(rows) if alignment_data.dataset_of(r["origin"]) not in GENERATED]
    rng = random.Random(seed); rng.shuffle(real)
    calib_idx = set(real[:int(len(real) * share)])
    return ([r for i, r in enumerate(rows) if i not in calib_idx],
            [rows[i] for i in sorted(calib_idx)])
```

- The fold is 10% of **real** rows, so its size and makeup don't change with the share.
- `--seed` (default 13, today's `SEED`) sets the calibration split, shuffling, torch and numpy.
  R2 trains every configuration with seeds 13 and 17.
- `metadata.json` gains `train_seed` and, when `--data` has a `variant.json`, `variant` (its
  contents).

## 5. Gate report (`eval_alignment_model.py`, SEC-215, SEC-216)

- `report()` is called once on all rows (as today) and once each on `role == "T"` and
  `role == "E"` rows; the two extra results go into `out["by_role"] = {"T": ..., "E": ...}`
  without `held_by_source`. `checks` and `passed` read only the all-rows metrics, so SEC-125 is
  unchanged.
- `--extra <file>` (repeatable): rows of a held-out generated set. Reported as
  `out["extra"][<name>] = {"rows", "held", "recall" (mode H) or "held_share" (mode L)}`. Refuses a
  file whose rows aren't all `rl-gen-*`.
- The printed report gains a line per role and per extra set, counts and rates only.

## 6. Comparison (`rl_generator/compare.py`, SEC-217)

```
python scripts/rl_generator/compare.py --base data/security/alignment --modes H,L --out LOCAL/compare/r2 \
    [--dry-run | --time-one]          # print the plan / train and gate one run and project the total
python scripts/rl_generator/compare.py --round --modes H --previous <variant> --new <variant> \
    --out LOCAL/compare/r3-h-1        # R3 (§7)
```

For each mode separately (`--modes` names only that mode, so effects don't mix; the mode gets
half the share, §3):

| Config | Variant | Seeds |
|---|---|---|
| none | `--kind none --share 0` | 13, 17 |
| p5 | `--kind prompted --share 0.05` | 13, 17 |
| p15 | `--kind prompted --share 0.15` | 13, 17 |
| r5 | `--kind rl --share 0.05` | 13, 17 |
| r15 | `--kind rl --share 0.15` | 13, 17 |

That is 20 training runs: 5 configurations × 2 modes × 2 seeds. `none` is shared by both modes, so
there are 18 distinct runs. Each run is a cross-encoder from MiniLM-L6 with today's settings, then
`eval_alignment_model.py --extra <both held-out sets>`. Runs resume: a configuration whose
`metadata.json` exists is skipped. G16 times one run first, so the user can see the total time
before starting. The base is the post-shadow-week build with SEC-218 applied, so every run uses the
same look-alike count.

The table, per mode, has a row per configuration: the mean and spread of the mode's metric,
off-intent recall, risk recall, look-alikes held, real aligned held (upper bound), ECE aligned and
risk, the held-out set's metric, and `by_role` T and E off-intent and risk recall.

**Decision rule** (from SEC-217), per mode:

- `spread(c)` is the absolute difference between a configuration's two seeds.
- RL beats prompting at a share if `mean(r) − mean(p) > max(spread(r), spread(p))` on the mode's
  metric (for look-alikes held, lower is better), with real aligned held ≤ 1% and both ECEs no
  higher than `none`'s mean plus 0.01.
- 15% beats 5% the same way, within the winning kind.
- The README records the table and the decision. The decision on mode A (D1) and the gate (D7)
  are then put to the user with the role report.

## 7. Rounds (R3)

For each mode R2 kept (its winning kind and share):

1. **Target:** the seed-13 model of the previous round's winning variant (R2's for round 1). It
   isn't installed; it's only the `miss` target.
2. **Seeds:** new seeds from the same base build, with `exclude` = every seed id used so far.
3. **Batches:** for `rl`, a new adapter trained against the new target (§7 of the generator
   design), then sampled; for `prompted`, a new prompted batch against the new target. Reviewed.
4. **Variant:** `variant.py --batches <the round's batch ids>` at the kept share, so the round's
   batches replace the previous round's (the cap
   doesn't grow).
5. **Compare:** `compare.py --round` trains `none`, the previous variant and the new one, two seeds
   each, and applies the decision rule between the new and the previous variant.

A mode stops when the new variant doesn't beat the previous one, or after 3 rounds. If a base
rebuild happens between rounds (new real data), the round restarts from R2's comparison on the new
build.

## 8. Agent rewards (R4: SEC-220 – SEC-222)

### 8.1 Graded set (`build_research_eval.py`)

- Draws 100 questions, seed 13, from Researchy Questions in a fresh `fetch_alignment_sources.py`
  scratch folder (`--raw`; scratch folders are deleted after use). It drops any question within
  cosine 0.9 (base MiniLM-L6) of an intent in the alignment splits, a question in the triage data
  (`data/triage/*.jsonl`) or one in the shadow feed log (`--exclude
  ~/webagent-shadow-feed/fed.jsonl`).
- Writes `data/research_eval/questions.jsonl` (`id`, `text`, `source: "researchy-questions"`,
  `licence: "CDLA-Permissive-2.0"`) and a README with source and licence. Committed (the licence
  allows it).

### 8.2 Run scorer (`score_research_run.py`)

```
python scripts/score_research_run.py --questions data/research_eval/questions.jsonl \
    --judge-model <D3 model> --out LOCAL/research_eval/baseline.json [--limit 5]
```

- Runs `TaskManager`'s research workflow for each question in dev settings, sequentially, with
  the guard in its configured mode.
- **Citation support:** for each cited source in the report, the claim sentence and the cited
  page's text from the run's results (no new fetch) go to the D3 judge: "does the page support the
  claim?" → JSON boolean. Score = supported / cited. G19 first checks what the run's results keep;
  where only a search snippet is kept, the snippet is used and the report gives the share of
  claims judged on snippets.
- **Retrieval quality:** each query's retrieved documents scored by the scorer with the same
  cross-encoder `RAGReranker` uses (web and internal results alike); the mean per task.
- **Cost:** tokens (`tokens_used`) and tool calls per task; `error_code` per task.
- Never reads checkpoint 1 or 2 scores (SEC-221).
- The report holds ids and numbers per question plus means; no question, page or claim text.
- `--limit 5` first, for a cost and Tavily-credit estimate; the full run only after O4.

## 9. Tests (offline)

| Test | Covers |
|---|---|
| `test_alignment_data_generated.py` | Sources; local-only; train-only rule; share-alike via seed; no prefix match with real or look-alike origins (SEC-200) |
| `test_rl_generator_variant.py` | Pass batches only; stale data hash refused; leakage drops; half-share cap per mode and shortfall; held-out refused; base rows unchanged (SEC-213) |
| `test_train_alignment_calibration.py` | No generated row in the fold; fold size from real rows; seed (SEC-214) |
| `test_eval_alignment_role.py` | `by_role`; checks unchanged; `--extra` (SEC-215, SEC-216) |
| `test_rl_generator_compare.py` | Decision rule on fixed numbers (both directions of the metric, ECE limit); resume; `--round` (SEC-217) |
| `test_alignment_training_lookalike.py` | `xstest:lookalike` rows count as look-alikes; `rl-gen-*` rows don't (SEC-218, SEC-200) |
| `test_research_eval.py` | Overlap exclusion; scorer with fake agent, judge and reranker; no guard scores read; no text in the report (SEC-220, SEC-221) |
