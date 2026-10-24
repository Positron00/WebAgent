# Low-level design: the generator, its reward and review

Paths are relative to `backend/`. Requirement IDs refer to [`requirements.md`](requirements.md).
`LOCAL` below means `data/security/alignment/local/` (git-ignored).

## 1. Files

| File | Purpose | About |
|---|---|---|
| `scripts/rl_generator/__init__.py` | Package marker; `MODES = ("H", "L")` | 5 lines |
| `scripts/rl_generator/seeds.py` | Seed selection, parts, seed files, novelty bank (§2, §4.2) | 110 |
| `scripts/rl_generator/prompts.py` | Generator and judge prompts per mode; `PROMPT_VERSION` (§3) | 90 |
| `scripts/rl_generator/judge.py` | Together judge client, cache, token count, stop rule (§4.4) | 140 |
| `scripts/rl_generator/rewards.py` | Reward terms and `RewardContext` (§4) | 180 |
| `scripts/rl_generator/rows.py` | Turning an accepted sample into a dataset row; batch files (§5) | 90 |
| `scripts/rl_generator/prompted.py` | Prompted generation; judge candidates (§6) | 180 |
| `scripts/rl_generator/train.py` | GRPO trainer: settings, reward function, CLI (§7) | 200 |
| `scripts/rl_generator/train_logic.py` | Trainer logic without trl: resume, metrics, stop rules, guard, `best_step`, checkpoint rotation, sampled steps (§7) | 230 |
| `scripts/rl_generator/sample.py` | Rows from an adapter (§7.4) | 100 |
| `scripts/rl_generator/review.py` | Review sheets, verdicts, judge choice (§9) | 170 |
| `scripts/rl_generator/colab_train.ipynb` | Colab driver (§8) | — |
| `requirements-train.txt` | `trl`, `peft`, `accelerate`, `datasets`, `fsspec` and `pyarrow`, exact pins (SNFR-21) | — |
| `tests/test_rl_generator_*.py` | Offline tests (§10) | — |

`MODES` is the only list of modes, and every entry point validates against it (SEC-202).

## 2. Seeds (`seeds.py`, SEC-201)

```python
SEED_ORIGINS = ("webagent-gen:call", "asq:query", "glaive:call", "toolace:call", "saroku:safe")

def select(train_rows: list[dict], mode: str, n: int, seed: int,
           exclude: set[str] = frozenset()) -> list[dict]   # seed ids to leave out (SEC-216)
def write(path: Path, seeds: list[dict], data_hash: str, mode: str) -> None
def read(path: Path, expected_hash: str | None, part: str | None = None) -> list[dict]
    # SystemExit on a hash mismatch; None only to read without generating (e.g. --exclude files)
```

- Input: the committed `data/security/alignment/train.jsonl` only. Rows with `role != "T"`,
  labels other than (yes, none) or `alignment_data.is_local(row)` are skipped.
- The sample is stratified so each origin gives at most a third of the seeds, and each `action`
  (`web_search`, `internal_retrieve`, `plan`, `function`) at least 10%. Code rows
  (`webagent-gen:code`) aren't seeds.
- The seed file is `LOCAL/generated/seeds_<mode>.jsonl`. Its first line is
  `{"data_hash": ..., "mode": ..., "n": ...}`, then one line per seed with `seed_id` (sha256 of
  intent and call, 16 hex), `part` (`train` or `sample`), `intent`, `call`, `action`, `origin`,
  `groups`, `sources`.
- `data_hash` is `train_alignment_model.data_hash(DATA_DIR)` of the build the seeds came from.
  `read` refuses a file whose hash differs from the current build's when generating.
- Each mode draws from the whole pool independently (selection seeds 7 and 8), so the modes may
  share intents. The pool is 17,206 rows in the current build (ASQ 5,512, `webagent-gen` 4,809,
  ToolACE 3,209, saroku 2,013, Glaive 1,663; by action: function 6,885, plan 4,809, web_search
  2,768, internal_retrieve 2,744). `exclude` leaves out seed ids already used (held-out sets,
  earlier rounds).
- **Parts:** 60% of a mode's seeds are `train` (prompts for `train.py`), 40% are `sample` (for
  `prompted.py` and `sample.py`). Prompted and RL rows come from the same `sample` seeds, so the
  comparison is fair, and RL rows never come from prompts the adapter trained on.
- Size: R2 needs about 2,600 rows per mode (half of 15% of 34,538 real train rows), plus room for
  leakage drops. `seeds.py --n 12000` per mode gives 4,800 `sample` seeds; at an acceptance rate
  of 2/3 that's 3,200 rows. G7 measures the real rate; if it is lower, a seed may give two rows
  (the two best valid samples).

## 3. Prompts (`prompts.py`, SEC-202, SEC-203)

Each prompt is a chat template with the seed's fields filled in. `PROMPT_VERSION` (`"p1"`) goes
into every row's provenance and the judge cache key; any wording change bumps it.

| Prompt | Given | Asks for |
|---|---|---|
| Generator H | Intent, action, the seed's call as an example of format | One `<action>` call a research agent might make for a **related but different** goal: same domain, plausible, harmless, and not useful for this intent. Output only the call |
| Generator L | Intent, action | One `<action>` call that **does serve** the intent, is harmless, and uses words that might sound sensitive out of context (security research, medicine, finance, account settings, legal questions). Output only the call |
| Judge H | Intent, call | JSON `{"label_ok": bool}`: true only if the call is harmless **and** would not help answer the intent |
| Judge L | Intent, call | JSON `{"label_ok": bool}`: true only if the call is harmless, would help answer the intent, and is no riskier than the intent |

Rules for the prompt files: no example of harmful, injected or deceptive text; nothing asking for
risk. Mode L describes sensitive-*sounding* topics only by domain name, as above. Reviewed in G2.

## 4. Reward (`rewards.py`, SEC-203 – SEC-207)

```python
@dataclass
class RewardContext:
    mode: str
    target: AlignmentModel           # alignment_model.load(dir, require_gate=False)
    novelty_bank: np.ndarray         # normalised MiniLM-L6 embeddings, float16
    tokenizer: Any                   # the target's tokenizer, for overlong
    judge: Judge
    lam_div: float = 0.2
    lam_len: float = 0.5
    judge_min_miss: float = 0.3

def score_samples(ctx, seeds: list[dict], calls: list[str]) -> list[Scored]
```

`Scored` holds `reward`, `miss`, `novelty`, `overlong`, `harmless`, `valid` (`True`, `False` or
`None` when not judged; `None` counts as 0), `p_aligned`, `risk`. It never holds text after
scoring.

### 4.1 Order of evaluation, per sample

1. **Clean** the completion: strip whitespace, a leading `"<action>:"` and surrounding quotes;
   keep the first line. Empty → reward 0.
2. **Length** (SEC-206): `overlong = 1` if `len(call) > 1000` or the target tokenizer's length of
   (intent, `"<action>: <call>"`) is over 128.
3. **Harmless** (SEC-205): `signals.harm_of(call)`. `None` or ≥
   `context_defender.current_thresholds()["harm_verify_at"]` (20 today) → reward 0,
   `harmless = False`, stop here. The threshold is read once per run and recorded.
4. **Miss** (SEC-204): `p, risk = target.score_pair(intent, f"{action}: {call}")`; mode H
   `miss = p`; mode L `miss = max(1 - p, risk / 100)`.
5. **Novelty** (SEC-206): `1 - max(bank @ e)` where `e` is the call's normalised base MiniLM-L6
   embedding; the bank is the committed train calls plus this run's accepted calls.
6. **Valid** (SEC-203): only if `miss >= judge_min_miss`, call the judge; else `valid = None`
   without a call.
7. `reward = int(bool(valid)) * (miss + lam_div * novelty) - lam_len * overlong` (SEC-207).

Steps 3 to 5 run batched (the target and encoders are batched over the group of 8).
`signals.harm_of` takes the model lock as the runtime does, so it's safe in threads.

### 4.2 Novelty bank

`seeds.py --bank` writes `LOCAL/generated/novelty_bank.npy`: embeddings of the committed train
calls (about 30,000 × 384, float16, about 23 MB), with the data hash in a sidecar JSON. Embeddings
of `local/` rows are never in the bank, so the bank can go to Colab.

### 4.3 Mode-specific notes

- Mode H rewards a call the target scores as aligned and the judge says is off-intent. A call that
  just repeats the intent scores high `miss` but fails the judge, so it gets 0 (or less, if
  overlong).
- Mode L rewards a call the target holds and the judge says is aligned and harmless. A call that
  really is risky fails the harmless filter or the judge, so it gets 0. That keeps the gradient
  away from harmful text (README decision 13).

### 4.4 Judge (`judge.py`)

```python
class Judge:
    def __init__(self, model: str, mode: str, cache_path: Path, credit_left: float | None,
                 price_in: float, price_out: float, timeout_s: float = 20.0): ...
    def __call__(self, intent: str, call: str, action: str = "") -> bool | None   # None: error (= False)
    tokens: dict[str, int]; calls: int; errors: int; cache_hits: int
    def begin_step(self) -> None   # train.py, once per reward call
    def end_seed(self) -> None     # after each seed's samples (error window of 100 seeds)
    def spent(self) -> float
    def error_stop(self) -> bool   # > 20% failed in the step or the last 100 seeds, once >= 10 calls
    def must_stop(self, next_cost: float = 0.0) -> bool   # error_stop() or spent + next >= credit_left / 2
```

- Client: LangChain's Together chat model, as `app/services/llm.py` builds it, with `model`,
  `temperature=0`, `max_tokens=16`. The key comes from `TOGETHER_API_KEY` in the environment
  (Colab: `userdata.get`), never an argument.
- Refuses to start if `model == settings.llm.default_model` (SEC-203).
- Error stop: `must_stop()` is also true when more than 20% of the calls in the current training
  step, or in the last 100 seeds, failed (SEC-203).
- Cache: SQLite at `LOCAL/generated/judge_cache.sqlite`, key sha256 of
  (model, `PROMPT_VERSION`, mode, intent, action, call), value the boolean (unparsable answers aren't
  cached). On Colab the cache lives on
  Drive and is merged back with the batch.
- Retries: 2 with back-off on 429 and 5xx, as the app's Together client.
- Price defaults are set in G3 from Together's models list for each candidate judge model, and can
  be overridden (`--price-input`, `--price-output`).
- Stop rule: `--credit-left` as `generate_alignment_calls.py`, with the same cost formula and the
  same token totals (read from `app.core.observability` directly, since importing that script sets
  process-wide guard settings); a run checks `must_stop()` before each seed or step. Prices are
  required (`--price-input`, `--price-output`) while `PRICES` is empty.

## 5. Rows and batches (`rows.py`)

An accepted sample becomes:

```json
{"intent": "<seed intent>", "call": "<call>", "action": "<seed action>",
 "labels": {"aligned": "no", "risk": "none"},
 "origin": "rl-gen-h:prompted", "role": "T", "split": "train",
 "groups": ["<seed groups>..."], "sources": ["rl-gen-h", "<seed sources>..."],
 "cluster": "", "meta": {"batch": "h-prompted-20261024-1", "seed_id": "...",
   "generator": "Qwen/Qwen2.5-1.5B-Instruct", "adapter_step": null,
   "target": "ac-2026-10-17-1", "judge": "<model>", "prompt": "p1",
   "miss": 0.83, "novelty": 0.41, "data_hash": "..."}}
```

Mode L rows have labels (yes, none) and origin `rl-gen-l:<kind>`. `cluster` is set by `variant.py`.
Batch files: `LOCAL/generated/rl_gen_<mode>_<kind>.jsonl` (mode in lower case), appended; a batch
id is `<mode>-<kind>-<yyyymmdd>-<n>`. Held-out rows have kind `rl` but go to
`rl_gen_<mode>_heldout.jsonl` with batch ids `<mode>-heldout-<yyyymmdd>-<n>` (SEC-216). Resuming
skips seed ids already in the file for that batch.

## 6. Prompted generation (`prompted.py`, SEC-208)

```
python scripts/rl_generator/prompted.py --mode H --seeds LOCAL/generated/seeds_h.jsonl \
    --judge-model <model> --price-input X --price-output Y --estimate 50
python scripts/rl_generator/prompted.py --mode H --judge-model <model> --price-input X \
    --price-output Y --n 500 --credit-left <dollars>       # R0 pilot batch; R2 uses --n 2700
python scripts/rl_generator/prompted.py --mode H --estimate 50 --price-input X --price-output Y \
    --judge-candidates <m1>,<m2>,<m3>                     # judge choice (D3)
# --seeds defaults to LOCAL/generated/seeds_<mode>.jsonl; --resume continues the last batch;
# --lam-div / --lam-len set the reward's weights (SEC-207; also on train.py and sample.py)
```

- Generator: `transformers` `AutoModelForCausalLM` of `Qwen/Qwen2.5-1.5B-Instruct` (`--base`),
  `torch_dtype=float16` on MPS, `float32` on CPU; sampling `temperature=0.9`, `top_p=0.95`,
  `max_new_tokens=96`, 8 samples per seed (`num_return_sequences=8`), seed fixed.
- Seeds: the `sample` part only.
- Selection: score all 8 without the judge (§4.1 steps 1–5); among `harmless` and not `overlong`,
  judge the 2 with the highest `miss + lam_div * novelty`; keep the judged-valid one with the
  higher reward; else nothing for that seed.
- `--estimate 50` prints seeds done, rows accepted, judge calls, tokens, cost, the projected cost
  of `--n`, and the generation rate (seeds per minute). Nothing is written to batch files.
- `--judge-candidates` (with `--estimate` only): the two selected samples per seed are judged by
  every candidate; samples and verdicts go to `LOCAL/review/judge-choice.jsonl` for §9. Costs
  are reported per candidate.
- Row counts: `--n` is the number of accepted rows wanted: 500 for the R0 and R1 pilot batches,
  2,700 for R2 (about 2,600 needed per mode, plus room for leakage drops).

## 7. GRPO trainer (`train.py`, SEC-209)

### 7.1 Setup

```
python scripts/rl_generator/train.py --mode H --seeds seeds_h.jsonl --bank novelty_bank.npy \
    --target models/security/alignment --judge-model <model> --price-input X --price-output Y \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --ckpt-dir /content/drive/MyDrive/webagent-rl/H --max-steps <from the cap> --credit-left <dollars>
```

| Setting | Value |
|---|---|
| Trainer | TRL `GRPOTrainer`; `reward_funcs=[reward_fn]` wrapping `score_samples` |
| Prompts | The mode's generator prompt per `train`-part seed; shuffled with seed 7, cycled |
| Group | `num_generations=8`; 2 prompts per device step (TRL's `per_device_train_batch_size` counts completions: 16), gradient accumulation 4 (64 completions per update) |
| Generation | `max_completion_length=96`, `temperature=0.9`, `top_p` at TRL's default 1.0 (sampling uses 0.95) |
| KL | `beta=0.04` to the base model |
| Optimiser | AdamW, learning rate 1e-5, warm-up 20 steps, cosine decay |
| LoRA | `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"])` |
| Precision | bf16 where the GPU supports it, else fp16; fp32 on CPU (`use_cpu`; never MPS) |
| Checkpoints | `save_steps=50` (`--save-steps`) under `--ckpt-dir`; rotation by `train_logic`: the newest 4 plus the guard's fallback (the latest at or before step − 200), so HF's `save_total_limit` is off; resume from the newest; a stop forces a save |
| Stop | `--max-steps` (set from the user's compute cap and the pilot's units per step); the judge's `must_stop()`; the valid-rate rule (§7.3) |

The target, ContextDefender and the MiniLM encoder stay on the CPU (`--scorers-device cpu`, the
only value: the app's models are CPU models), so the reward doesn't compete with generation for GPU
memory. TRL passes the dataset's `seed_id` column to the reward function, one entry per
completion, with each prompt's 8 completions consecutive; the reward function scores each group,
then calls `judge.end_seed()`, and calls `judge.begin_step()` once per call (one call per update).

### 7.2 Logged per step (counts and means only)

`reward_mean`, `reward_std`, `valid_rate` (of judged), `judged_share`, `miss_mean`,
`novelty_mean`, `overlong_share`, `harmful_filtered`, `judge_calls`, `judge_errors`, `spent`. Logged
to stdout and `--ckpt-dir/metrics.jsonl`.

### 7.3 Reward-hacking guard

Over the last 200 steps (a full window only, `--guard-window`): if the judged-valid rate (valid ÷
judged over the window) is below 0.8 while `miss_mean` rises (a positive least-squares slope),
training stops and `train.py` records `best_step` = the latest checkpoint at or before step − 200
(null if there is none; `sample.py` then needs `--step`).
If training ends without the guard firing, `best_step` is the last checkpoint. `best_step` goes
into `--ckpt-dir/best_step.json`; `sample.py` uses it by default.

### 7.4 Sampling (`sample.py`, SEC-210)

Loads the base plus the adapter at `--step` (default `best_step`) and runs §6's selection with
kind `rl` and `adapter_step` set, over the seed file's `sample` part, which `train.py` never sees
(§2). `--heldout` writes a held-out batch (SEC-216): it needs `--step` different from every batch
used in training and a different `--sample-seed`, and its seeds come from a seed file made with
`exclude`.

### 7.5 Pilot first

G9 runs Qwen2.5-0.5B for 20 steps with `--max-steps 20` and 50 seeds, to check memory, rate and
cost on the Colab GPU before R1 (task G9).

## 8. Colab (`colab_train.ipynb`, SEC-211)

Cells, in order:

1. Mount Drive at `/content/drive`; make `MyDrive/webagent-rl/`.
2. Clone the repository at the release tag (public) with `git lfs pull` for the target
   (`backend/models/security/alignment/`) and ContextDefender (`backend/models/security/`) models
   only (`--include`). Scripts run with `WEBAGENT_ENV=dev` and LangSmith tracing off, so
   `app.core.config` loads the committed dev settings without a `.env.local`.
3. `pip install -r backend/requirements.txt -r backend/requirements-train.txt` with
   `constraints.txt` minus its `fsspec` line (the training environment uses fsspec 2026.6.0).
4. Copy from Drive: `seeds_<mode>.jsonl`, `novelty_bank.npy`, `judge_cache.sqlite` (uploaded from
   the Mac by the user beforehand).
5. `os.environ["TOGETHER_API_KEY"] = userdata.get("TOGETHER_API_KEY")`, never printed.
6. Run `train.py` (cell parameters: mode, base, max steps from the cap, credit left, judge model
   and its prices).
7. Run `sample.py`; copy the batch file and the judge cache to Drive.

The notebook is committed with outputs cleared (`nbstripout` style; a test checks it). After
download, the user deletes the batch files on Drive (a step in tasks G12, G13, G15 and G16).
Colab's compute units aren't readable from the notebook; the user reads them from Colab after each
run and records them in the README.

## 9. Review (`review.py`, SEC-212)

```
python scripts/rl_generator/review.py --make h-prompted-20261024-1     # 100 rows
python scripts/rl_generator/review.py --score h-prompted-20261024-1    # verdict
```

- `--make`: stratified by `action` and seed origin, seed 7; writes
  `LOCAL/review/<batch>.csv` with columns `row_id, intent, action, call, label_ok` (empty). The
  judge's answer isn't in the sheet, so the user labels blind. A second `--make` on a `more`
  verdict appends 100 new rows.
- The user fills `label_ok` with `y` or `n` (the same question as the judge prompt, shown at the top
  of the sheet).
- `--score`: disagreement = rows where the user's label differs from the judge's (the judge said
  true for every accepted row, so disagreement = user `n`). Writes
  `LOCAL/generated/rl_review.json` → `{"<batch>": {"n": 100, "user_no": 4, "verdict": "pass", "date": ...}}`
  and prints the same counts.
- **Judge choice (D3, task G6).** `--make judge-choice` takes 100 rows of
  `LOCAL/review/judge-choice.jsonl`, stratified by mode and by how the candidates voted (all yes,
  all no, split), and writes a blind sheet. `--score judge-choice` prints, per candidate, agreement
  with the user on the rows it accepted and on the rows it rejected. The candidate with the best
  agreement (both ways) is chosen, cost breaking ties. A candidate's failed call (`None`) counts as
  neither an acceptance nor a rejection; the score reports it as `failed`.

## 10. Tests (offline, SNFR-22)

| Test | Covers |
|---|---|
| `test_rl_generator_seeds.py` | Origins, roles, local exclusion, stratification, hash refusal (SEC-201) |
| `test_rl_generator_rewards.py` | Order of evaluation, harmless filter, miss per mode, novelty, overlong, judge skipped below 0.3, reward formula (SEC-203 – SEC-207) |
| `test_rl_generator_judge.py` | JSON parsing, errors → None, cache, equal model refused, stop rule (SEC-203) |
| `test_rl_generator_prompted.py` | Selection of 2 for judging, best valid kept, resume, estimate writes no batch, judge candidates (SEC-208) |
| `test_rl_generator_train.py` | Resume, stops, the valid-rate guard and `best_step` with a fake trainer; skipped without `trl` (SEC-209, SEC-210) |
| `test_rl_generator_review.py` | Stratification, blind sheet, verdict thresholds incl. `more`, judge-choice score (SEC-212) |
| `test_rl_generator_modes.py` | Other modes refused everywhere (SEC-202) |
| `test_rl_generator_notebook.py` | No outputs, no key-like strings (SEC-211, SNFR-24) |
| `test_rl_generator_logs.py` | No sample text in logs or metrics (SNFR-20) |

Fakes: a generator returning fixed completions, a target with scripted `score_pair`, a judge with
scripted answers, `harm_of` patched, a tiny random novelty bank.
