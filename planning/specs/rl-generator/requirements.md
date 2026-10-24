# Requirements: an RL generator of hard training rows for checkpoint 2

Each requirement has an ID, a statement and how it is checked. All are "must" unless marked
"should". IDs continue the security specs: `SEC-0xx` (agentic-security), `SEC-1xx` and `SNFR-1x`
(checkpoint2-intent); `SEC-2xx` and `SNFR-2x` here. Paths are relative to `backend/`.

Terms:

- **Mode**: `H` (hard off-intent: aligned no, risk none) or `L` (look-alike: aligned yes, risk
  none). There is no other mode.
- **Seed**: a committed train-split row of the current build whose intent the generator writes a
  call for (SEC-201).
- **Target**: the frozen classifier whose mistakes are rewarded: `ac-2026-10-17-1` in R1, then each
  retrained candidate.
- **Judge**: the Together model that confirms a sample's label (SEC-203).
- **Batch**: the rows one generator run writes for one mode, with a batch id; the unit of review.
- **Kind**: `prompted` (the untrained generator) or `rl` (a GRPO-trained adapter). Held-out rows
  (SEC-216) are `rl` rows in their own batches and file.
- **Share**: generated rows as a fraction of the base build's real train rows (`train.jsonl` plus
  `local/train.jsonl`: 34,538 today), across both modes; each mode's part is half the share.
- **`LOCAL`**: `data/security/alignment/local/` (git-ignored).
- **Provenance**: the `meta` object of a row in a batch file (generator, adapter step, target,
  judge, prompt version, seed id, data hash). It is stripped when rows go into a variant.

## 1. Sources and seeds

**SEC-200 Generated sources.** `rl-gen-h` and `rl-gen-l` are entries in `alignment_data.SOURCES`
with licence "project data (Qwen2.5 output, Apache-2.0)", role `T` and local `True`, and in
`data/security/alignment/README.md` with their approval date (2026-10-23). Origins are
`rl-gen-h:<kind>` and `rl-gen-l:<kind>`; neither may match `REAL_CALL_ORIGINS` or
`LOOKALIKE_ORIGINS`. A generated row's `sources` is its generated source plus the seed's
`sources`, so the seed's licence rules (CC BY-SA) carry over. The data README holds each source's
`SOURCES` title, so `check_data(readme=...)` enforces the entries.
- Check: pytest: `schema_errors` and `rule_errors` accept a generated row; `is_real_aligned` and
  `is_lookalike` are false for it; a seed from ASQ makes a share-alike row.

**SEC-201 Seeds.** Seeds are rows of the committed `train.jsonl` (never `local/`, `test` or
`probe`) with role `T` and labels aligned yes, risk none, from `webagent-gen`, `asq`, `glaive`,
`toolace` and `saroku:safe`. Each generated row copies its seed's `intent`, `action` and `groups`.
Each mode's seeds are split 60/40 into a `train` part (prompts for SEC-209) and a `sample` part
(the only seeds SEC-208 and SEC-210 write rows from). The seed file records the build's data hash;
a seed file from another build is refused for generation.
- Check: pytest of seed selection and parts on fixture splits; a stale data hash is refused.

**SEC-202 No attack mode.** No script accepts a mode other than `H` or `L`, and no prompt asks for
an attack, an injection, a harmful request or a risky call.
- Check: pytest: other modes are refused; review of the prompt files.

## 2. Reward

**SEC-203 Judge.** `valid(x)` comes from one call to the judge model at temperature 0 with the
mode's prompt (not checkpoint 2's judge prompt), answering JSON `{"label_ok": true|false}`. The
judge model is a script argument and must differ from `llm.default_model`. An unparsable answer,
a timeout or an error counts as `false`. Answers are cached by a hash of (model, prompt version,
mode, intent, action, call), so a resumed run doesn't pay twice; unparsable answers aren't cached. The judge is called only for samples whose
`miss` is at least 0.3 (others are not judged and count as `valid = 0`). If more than 20% of the
judge calls in a training step or in the last 100 seeds fail (once that window has at least 10
calls), the run stops.
- Check: pytest with a fake judge: parse failure → false; the cache; equal models refused; no call
  below 0.3; the error stop.

**SEC-204 Miss.** `miss(x)` comes from the target's calibrated scores
(`AlignmentModel.score_pair(intent, "<action>: <call>")`, which returns `p_aligned` and `risk`
0–100): mode H `p_aligned`; mode L `max(1 − p_aligned, risk / 100)`. The target is loaded with
`alignment_model.load(d, require_gate=False)`, so a shadow-only model can be the target, and its
version is recorded in every row's provenance.
- Check: pytest with a fake target.

**SEC-205 Harmless filter.** A sample whose ContextDefender harm score
(`signals.harm_of(call)`) is at or above the installed model's `harm_verify_at`
(`context_defender.current_thresholds()`: 20 for `cd-2026-10-13-2`; the code default of 50 applies
only without a model) gets reward 0 and is never written. So does a sample `harm_of` can't score. The count of such
samples is logged per step or batch.
- Check: pytest with a fake ContextDefender.

**SEC-206 Novelty and length.** `novelty(x)` is 1 minus the highest cosine (base MiniLM-L6,
normalised) between the call and the committed train calls and this run's earlier accepted
samples. `overlong(x)` is 1 when the (intent, call) pair is over 128 tokens of the target's
tokenizer or the call is over 1,000 characters.
- Check: pytest.

**SEC-207 Reward.** `r = valid · (miss + λ_div · novelty) − λ_len · overlong` for a sample that
passes SEC-205 (otherwise 0), with `λ_div = 0.2` and `λ_len = 0.5` by default, set per run and
recorded. Novelty counts only for valid samples, so novel text with the wrong label earns
nothing. Logs and metrics hold counts and means
only, never sample text.
- Check: pytest of the formula; log scan test (SNFR-20).

## 3. Generation

**SEC-208 Prompted generation.** `scripts/rl_generator/prompted.py --mode H|L` runs the untrained
generator base on the Mac (MPS, else CPU): 8 samples per seed; the 2 with the highest
`miss + λ_div · novelty` that pass SEC-205 and SEC-206 go to the judge; the best judged-valid one
is kept. `--estimate 50` runs 50 seeds and reports judge tokens, cost and the projected full cost;
a full run stops before spending would leave less than half the credit entered (`--credit-left`),
as SEC-122. Rows go to `LOCAL/generated/rl_gen_<mode>_prompted.jsonl` (mode in lower case),
appended with a batch id and resumable by seed id. `--judge-candidates <m1,m2,...>` with
`--estimate` instead judges each selected sample with every candidate and writes the samples and
verdicts to `LOCAL/review/judge-choice.jsonl` for choosing the judge (SEC-212).
- Check: pytest with a fake generator, target and judge (selection, stop rule, resume, candidates).

**SEC-209 GRPO trainer.** `scripts/rl_generator/train.py` trains a LoRA adapter on the generator
base with TRL's GRPO (group size 8, KL coefficient 0.04, learning rate 1e-5, seed fixed), reward
SEC-207, one adapter per mode, prompts from the seeds' `train` part only. It saves a checkpoint
every 50 steps to `--ckpt-dir`, resumes from the latest, and stops at `--max-steps`, on the judge's
stop rules (SEC-203, SEC-208), or when the judged-valid rate over the last 200 steps falls below
0.8 while mean miss rises; it records `best_step`, the checkpoint `sample.py` uses by default. Per
step it logs mean reward, valid rate, mean miss, mean novelty, harmless-filter count, judge calls
and spending.
- Check: pytest with a fake trainer and rewards (resume, stops, `best_step`); R1's logs.

**SEC-210 Sampling an adapter.** `scripts/rl_generator/sample.py` writes rows from a trained adapter
with the same selection, filters and output format as SEC-208 (kind `rl`), from the seeds'
`sample` part only.
- Check: pytest shares SEC-208's fixtures.

**SEC-211 Colab driver.** `scripts/rl_generator/colab_train.ipynb` only mounts Drive, installs
`requirements.txt` and `requirements-train.txt` (with `constraints.txt` minus its `fsspec` line),
reads the Together key from Colab secrets and calls `train.py` and
`sample.py` from a clone of the release tag. It is committed with outputs cleared. What goes to
Colab and Drive is the seed file (committed rows only), the target and ContextDefender model
folders (from the clone, Git LFS), the novelty bank (embeddings of committed train calls) and the
judge cache; nothing else from `LOCAL`. Batch files and the cache come back to the Mac, and the
batch files are then deleted from Drive.
- Check: pytest: the notebook has no outputs and no key-like strings; upload list reviewed.

## 4. Review

**SEC-212 Review sheet.** `scripts/rl_generator/review.py --make <batch>` writes 100 rows
stratified by action and seed source to `LOCAL/review/<batch>.csv` with an empty `label_ok` column
for the user (the judge's verdict isn't shown). `--score <batch>` compares the user's labels with the judge
and writes the batch's verdict to `local/generated/rl_review.json`:

| User–judge disagreement | Verdict |
|---|---|
| ≤ 6% of 100 | `pass` |
| > 14% of 100 | `fail` (batch discarded, judge prompt revised) |
| 6–14% | `more`: 100 more rows; then `pass` if ≤ 10% of 200, else `fail` |

Only counts are printed. For the judge choice (D3), `--make judge-choice` writes 100 rows of
`judge-choice.jsonl` (both modes, accepted and rejected) and `--score judge-choice` prints each
candidate's agreement with the user on rows it accepted and on rows it rejected (a failed call is
counted apart, as `failed`).
- Check: pytest of stratification, verdicts and the judge-choice score.

## 5. Dataset, training and gate

**SEC-213 Variants.** `scripts/rl_generator/variant.py` makes a dataset folder under
`LOCAL/variants/` from the base build plus generated rows, with `--kind none|prompted|rl`,
`--share 0|0.05|0.15`, `--modes` and optionally `--batches` (only those batch ids: an R3 round's). The base build (`build_alignment_dataset.py`, unchanged) and
its splits are identical in every variant. It includes rows only from batches of that kind with
verdict `pass` and the base's data hash, never held-out batches; puts them in `local/train.jsonl`
only; takes at most half the share's rows from each mode named (so one mode alone gets half the
share); and drops a generated row whose seed
groups aren't all in the base train split, whose call is within cosine 0.9 of a base test or probe
call, or that duplicates an existing row. `check_data` gains a rule (generated rows only in train)
and must pass on every variant, with the base's data README.
- Check: pytest of `variant.py` on fixtures; `check_data` on a real variant.

**SEC-214 Calibration fold.** `split_calibration` never puts a generated row in the calibration
fold, and the fold keeps its 10% share of the real rows. Thresholds, calibration and the judge
band come from real rows only.
- Check: pytest.

**SEC-215 Role report.** `eval_alignment_model.py` adds `by_role` (the report's metrics for `T`
rows and `E` rows separately) to `metadata.json["test"]` and the printed report. It is not part of
`checks` or `passed`.
- Check: pytest of the report on fixture rows.

**SEC-216 Held-out generated set.** For each mode, 300 rows from seeds disjoint from every batch
used in training, a different adapter checkpoint and a different sampling seed are reviewed
(SEC-212) and kept as `LOCAL/generated/rl_gen_<mode>_heldout.jsonl` (batch ids
`<mode>-heldout-*`), never in a variant. `eval_alignment_model.py --extra <file>` reports recall
(H) or held share (L) on it, report only.
- Check: pytest of `--extra`; `variant.py` refuses held-out batches.

**SEC-217 Comparison (R2).** On the post-shadow-week build, after SEC-218, per mode (only that
mode's rows), five configurations (none; prompted at share 5%; prompted 15%; RL 5%; RL 15%), each
trained with seeds 13 and 17 (a new `--seed` in `train_alignment_model.py`), are gated with
SEC-125 plus SEC-215 and SEC-216. RL is kept for a mode only if, at the same share, its mean on the
mode's metric (off-intent recall for H, look-alikes held for L, lower is better) beats prompting's
by more than the larger of the two configurations' seed spreads, with real aligned held ≤ 1% and
both ECEs no more than 0.01 above the `none` mean. 15% is kept only if it beats 5% the same way.
The table and decision go in the README, and D1 (mode A) and D7 (gate) go back to the user.
- Check: the README's R2 table.

**SEC-218 Look-alike origins (O1).** `alignment_training.LOOKALIKE_ORIGINS` names `xstest:lookalike`
in place of `xstest:safe`, so the gate's look-alike check counts XSTest's safe prompts, as the
model-data design §2 says. It lands in its own commit before R2's baseline; `ac-2026-10-17-1` and
the post-shadow-week model are re-reported with the old and new counts, and checkpoint2-intent's
model-data design records the change.
- Check: pytest: `is_lookalike` is true for `xstest:lookalike` rows; the re-report in the README.
- Check: the README's R2 table.

## 6. Agent rewards (R4)

**SEC-220 Graded question set.** `data/research_eval/questions.jsonl` holds 100 Researchy Questions
not present in any alignment split, triage set or shadow feed log, with their source and licence.
- Check: pytest of the overlap check against those files.

**SEC-221 Run scorer.** `scripts/score_research_run.py` runs the research workflow on each question
(guard in its configured mode) and scores: citation support (share of cited claims the D3 judge
finds in the cited page), retrieval quality (mean reranker score of retrieved documents), tokens,
tool calls and `error_code`. It never uses checkpoint 1 or checkpoint 2 scores as a reward.
Results go to a JSON report of ids and numbers; no question, page or claim text is in the report
or the logs.
- Check: pytest with fake agent, judge and reranker.

**SEC-222 Baseline.** One scored run of today's agent on the set, after the user approves the
Tavily credits (O4), recorded in the README.
- Check: the README's R4 note.

## 7. Non-functional requirements

| ID | Requirement | Check |
|---|---|---|
| SNFR-20 | Data rules (SNFR-17, D1): benign generation only; no generated or hand-written adversarial text; scripts and logs print counts and scores, never row text; Claude reports review results as counts | pytest log scan; review |
| SNFR-21 | Nothing the server runs changes: no change under `app/`, `config/` or the image. Training dependencies (`trl`, `peft`, `accelerate`, `datasets`, `fsspec` and `pyarrow`) are pinned in `requirements-train.txt`, never in `requirements.txt` or `constraints.txt`, and audited by `pip-audit` in CI | `requirements*.txt` diff; CI |
| SNFR-22 | All tests run offline with fakes (generator, target, judge, ContextDefender, trainer); tests that need `trl` are skipped when it isn't installed; existing tests still pass | CI |
| SNFR-23 | Spending: the judge's credit stop rule in every script that calls it (SEC-208, SEC-209), R1's `--max-steps` within the user's compute-unit cap (O3), R4's Tavily run only after approval (O4) | run reports |
| SNFR-24 | Secrets: the Together key only from the environment or Colab secrets; never in a notebook, log, row or report | pytest; review |
| SNFR-25 | Reproducible: fixed seeds; judge at temperature 0 and cached; every batch row's provenance records generator, adapter step, target version, judge model, prompt version, seed id and data hash; every variant records its batches (`variant.json`) | pytest |
| SNFR-26 | Generated and review files only under `data/security/alignment/local/` (git-ignored); Drive copies deleted after import | `git status`; runbook step |
| SNFR-27 | New files stay under about 250 lines (`prompted.py` is 265, kept whole because `sample.py` shares its selection) | review |
| SNFR-28 | Each release through a PR with both CI checks green, then tagged | release |
