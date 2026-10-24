#!/usr/bin/env python3
"""GRPO training of the generator's LoRA adapter for one mode (rl-generator SEC-209; design §7).

    python scripts/rl_generator/train.py --mode H --base Qwen/Qwen2.5-1.5B-Instruct \
        --seeds LOCAL/generated/seeds_h.jsonl --bank LOCAL/generated/novelty_bank.npy \
        --target models/security/alignment --judge-model <m> --price-input X --price-output Y \
        --ckpt-dir <dir> --max-steps <from the cap> --credit-left <dollars>

TRL's GRPOTrainer with the mode's generator prompt per `train`-part seed (shuffled with seed 7,
cycled) and a reward that wraps `rewards.score_samples`. Checkpoints every 50 steps under
`--ckpt-dir`, resumes from the newest; stops at `--max-steps`, on the judge's stop rules or on the
valid-rate guard; writes `best_step.json` and per-step metrics (`metrics.jsonl`, numbers only).
trl and peft are imported only when training starts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_generator  # noqa: E402
from rl_generator import check_mode, rewards, seeds  # noqa: E402
from rl_generator import train_logic as tl  # noqa: E402
from rl_generator.prompted import BASE, TARGET, load_target  # noqa: E402
from rl_generator.prompts import generator_messages  # noqa: E402

rl_generator.bootstrap()

NUM_GENERATIONS = 8
PROMPTS_PER_DEVICE_STEP = 2
GRAD_ACCUM = 4
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "task_type": "CAUSAL_LM"}


def precision(cuda: bool, bf16_ok: bool) -> str:
    """bf16 where the GPU supports it, else fp16; fp32 on the CPU (the 0.5B test run)."""
    return ("bf16" if bf16_ok else "fp16") if cuda else "fp32"


def grpo_settings(args: argparse.Namespace, prec: str) -> dict:
    """GRPOConfig keyword arguments (§7.1). TRL counts completions per device batch, so 2 prompts
    x 8 generations = 16; with accumulation 4 that is 64 completions per update. Checkpoint
    rotation is ours (`train_logic.to_delete`: newest 4 plus the guard's fallback)."""
    return {"output_dir": str(args.ckpt_dir), "max_steps": int(args.max_steps), "seed": int(args.seed),
            "num_generations": NUM_GENERATIONS, "per_device_train_batch_size": PROMPTS_PER_DEVICE_STEP * NUM_GENERATIONS,
            "gradient_accumulation_steps": GRAD_ACCUM, "max_completion_length": 96, "temperature": 0.9,
            "beta": 0.04, "learning_rate": 1e-5, "warmup_steps": 20, "lr_scheduler_type": "cosine",
            "optim": "adamw_torch", "save_strategy": "steps", "save_steps": int(args.save_steps),
            "save_total_limit": None, "bf16": prec == "bf16", "fp16": prec == "fp16", "use_cpu": prec == "fp32",
            "shuffle_dataset": False, "remove_unused_columns": False, "log_completions": False,
            "report_to": "none", "logging_steps": 1, "disable_tqdm": True}


def prompt_rows(seed_rows: List[dict], mode: str, shuffle_seed: int = 7) -> List[dict]:
    """One conversational prompt per `train` seed, with its seed id as an extra column (TRL passes
    extra columns to the reward function, one entry per completion)."""
    out = [{"prompt": generator_messages(mode, s), "seed_id": s["seed_id"]} for s in seed_rows]
    random.Random(shuffle_seed).shuffle(out)
    return out


def completion_text(completion) -> str:
    """A completion as TRL passes it: a string, or for chat prompts a list of messages."""
    if isinstance(completion, str):
        return completion
    for message in reversed(list(completion or [])):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def make_reward_fn(ctx: rewards.RewardContext, seeds_by_id: Dict[str, dict], controller: tl.Controller):
    """reward_fn(prompts, completions, seed_id, **kw) -> one reward per completion. Completions of
    one prompt are consecutive; each group is scored together and closed with `judge.end_seed()`.
    Judged-valid samples join the run's novelty bank."""
    def reward_fn(prompts, completions, seed_id, **kwargs) -> List[float]:
        judge = ctx.judge
        judge.begin_step()
        calls0, errors0 = getattr(judge, "calls", 0), getattr(judge, "errors", 0)
        filtered0 = ctx.harmful_filtered
        scored: List[rewards.Scored] = []
        i = 0
        while i < len(completions):
            j = i
            while j < len(completions) and seed_id[j] == seed_id[i]:
                j += 1
            group = [seeds_by_id[seed_id[i]]] * (j - i)
            part = rewards.score_samples(ctx, group, [completion_text(c) for c in completions[i:j]])
            judge.end_seed()
            for s in part:
                if s.valid:
                    ctx.accept(s)
            scored += part
            i = j
        controller.record(tl.step_metrics(scored, ctx.harmful_filtered - filtered0,
                                          getattr(judge, "calls", 0) - calls0, getattr(judge, "errors", 0) - errors0,
                                          judge.spent()))
        return [float(s.reward) for s in scored]

    reward_fn.__name__ = "rl_gen_reward"
    return reward_fn


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, help="H or L")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--seeds", type=Path, default=None, help="default LOCAL/generated/seeds_<mode>.jsonl")
    ap.add_argument("--bank", type=Path, default=seeds.BANK)
    ap.add_argument("--target", type=Path, default=TARGET)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--price-input", type=float, default=None)
    ap.add_argument("--price-output", type=float, default=None)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--credit-left", type=float, required=True)
    ap.add_argument("--lam-div", type=float, default=0.2)
    ap.add_argument("--lam-len", type=float, default=0.5)
    ap.add_argument("--scorers-device", default="cpu", choices=["cpu"],
                    help="the target and ContextDefender are CPU models; kept for the notebook's calls")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-steps", type=int, default=tl.SAVE_STEPS)
    ap.add_argument("--guard-window", type=int, default=tl.GUARD_WINDOW)
    args = ap.parse_args(argv)
    args.mode = check_mode(args.mode)
    if args.max_steps < 1 or args.save_steps < 1:
        ap.error("--max-steps and --save-steps must be positive")
    return args


def run(args: argparse.Namespace, judge_factory=None, log=print) -> dict:
    """The CLI's work; `judge_factory(model, mode)` is the hook for tests and smoke runs."""
    from rl_generator.judge import Judge

    started = time.time()
    dhash = seeds.current_hash()
    seed_rows = seeds.read(args.seeds or seeds.seed_path(args.mode), dhash, part="train")
    if not seed_rows:
        raise SystemExit("the seed file has no train part")
    target = load_target(args.target)
    factory = judge_factory or (lambda m, mode: Judge(m, mode, credit_left=args.credit_left,
                                                      price_in=args.price_input, price_out=args.price_output))
    judge = factory(args.judge_model, args.mode)
    ctx = rewards.RewardContext(args.mode, target, seeds.read_bank(args.bank, dhash), target.tokenizer, judge,
                                lam_div=args.lam_div, lam_len=args.lam_len)
    resume = tl.latest_checkpoint(args.ckpt_dir)
    resumed_at = int(resume.name.split("-")[1]) if resume else None
    controller = tl.Controller(args.ckpt_dir, judge, tl.load_metrics(args.ckpt_dir, resumed_at),
                               window=args.guard_window, log=log)
    log(json.dumps({"mode": args.mode, "train_seeds": len(seed_rows), "resume_step": resumed_at,
                    "max_steps": args.max_steps, "reward": ctx.record()}))
    trainer = build_trainer(args, ctx, seed_rows, controller)
    try:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    finally:
        judge_report = judge.report() if hasattr(judge, "report") else {}
        if hasattr(judge, "close"):
            judge.close()
    out = controller.finish(trainer.state.global_step)
    out.update({"judge": judge_report, "seconds": round(time.time() - started, 1)})
    log(json.dumps(out))
    return out


def build_trainer(args: argparse.Namespace, ctx: rewards.RewardContext, seed_rows: List[dict], controller):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    cuda = torch.cuda.is_available()
    prec = precision(cuda, cuda and torch.cuda.is_bf16_supported())
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[prec]
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    return GRPOTrainer(model=model, reward_funcs=[make_reward_fn(ctx, {s["seed_id"]: s for s in seed_rows}, controller)],
                       args=GRPOConfig(**grpo_settings(args, prec)),
                       train_dataset=Dataset.from_list(prompt_rows(seed_rows, args.mode)),
                       processing_class=tok, peft_config=LoraConfig(**LORA),
                       callbacks=[tl.make_callback(controller)])


def main(argv: Optional[List[str]] = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
