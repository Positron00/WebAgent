#!/usr/bin/env python3
"""Rows from a trained adapter (rl-generator SEC-210, SEC-216; design §7.4).

    python scripts/rl_generator/sample.py --mode H --seeds ... --bank ... --target ... \
        --judge-model <m> --price-input X --price-output Y --ckpt-dir <dir> --n 500 --credit-left <dollars>
    ... --heldout --step 300 --sample-seed 11      # held-out batch (SEC-216)

Loads the base plus the adapter at `--step` (default `best_step` from `<ckpt-dir>/best_step.json`)
and runs prompted.py's selection, filters and output with kind `rl` and `adapter_step` set, over
the seed file's `sample` part only (train.py never sees it). Steps used for training batches are
recorded in `<ckpt-dir>/sampled_steps.json`; `--heldout` needs a step none of them used, a
`--sample-seed` other than 7 and seeds no existing batch of the mode used, and writes
`rl_gen_<mode>_heldout.jsonl` with `<mode>-heldout-*` batch ids. Prints counts only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_generator  # noqa: E402
from rl_generator import GENERATED_DIR, check_mode, rewards, rows, seeds  # noqa: E402
from rl_generator import train_logic as tl  # noqa: E402
from rl_generator.prompted import (BASE, TARGET, Provenance, device, load_generator, load_target,  # noqa: E402
                                   make_generate, select_and_write)

rl_generator.bootstrap()

DEFAULT_SAMPLE_SEED = 7


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
    ap.add_argument("--step", type=int, default=None, help="default best_step from best_step.json")
    ap.add_argument("--n", type=int, default=500, help="accepted rows wanted")
    ap.add_argument("--credit-left", type=float, required=True)
    ap.add_argument("--heldout", action="store_true", help="write a held-out batch (SEC-216)")
    ap.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    ap.add_argument("--lam-div", type=float, default=0.2)
    ap.add_argument("--lam-len", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true", help="continue the file's last batch of this kind")
    ap.add_argument("--out-dir", type=Path, default=GENERATED_DIR)
    args = ap.parse_args(argv)
    args.mode = check_mode(args.mode)
    if args.heldout and args.step is None:
        ap.error("--heldout needs --step (a checkpoint no training batch used)")
    if args.heldout and args.sample_seed == DEFAULT_SAMPLE_SEED:
        ap.error(f"--heldout needs a --sample-seed other than {DEFAULT_SAMPLE_SEED}")
    return args


def choose_step(args: argparse.Namespace) -> int:
    """`--step`, else best_step; the checkpoint must exist and, for a held-out batch, must differ
    from every step used for training batches (and a training batch may not use a held-out step)."""
    step = args.step if args.step is not None else tl.read_best_step(args.ckpt_dir)
    if not tl.checkpoint_path(args.ckpt_dir, step).is_dir():
        raise SystemExit(f"no checkpoint at step {step} in --ckpt-dir")
    used = tl.read_sampled(args.ckpt_dir)
    if args.heldout and step in used["rl"]:
        raise SystemExit(f"step {step} was used for a training batch; a held-out batch needs another step")
    if not args.heldout and step in used["heldout"]:
        raise SystemExit(f"step {step} was used for a held-out batch; pick another step for training rows")
    return step


def heldout_overlap(mode: str, seed_rows: List[dict], out_dir: Path) -> int:
    """Seeds of `seed_rows` already used by a prompted or rl batch of the mode (must be 0)."""
    used = set()
    for kind in ("prompted", "rl"):
        used |= {r["meta"]["seed_id"] for r in rows.read_rows(rows.batch_path(mode, kind, out_dir=out_dir))}
    return sum(s["seed_id"] in used for s in seed_rows)


def load_adapter(base: str, path: Path, dev: str):
    from peft import PeftModel

    model, tok = load_generator(base, dev)
    return PeftModel.from_pretrained(model, str(path)).to(dev).eval(), tok


def run(args: argparse.Namespace, judge_factory=None, generate=None, log=print) -> dict:
    """The CLI's work; `judge_factory(model, mode)` and `generate` are hooks for tests and smoke runs."""
    from rl_generator.judge import Judge

    step = choose_step(args)
    dhash = seeds.current_hash()
    seed_rows = seeds.read(args.seeds or seeds.seed_path(args.mode), dhash, part="sample")
    if args.heldout and heldout_overlap(args.mode, seed_rows, args.out_dir):
        raise SystemExit("held-out seeds overlap existing batches; make the seed file with --exclude")
    target = load_target(args.target)
    factory = judge_factory or (lambda m, mode: Judge(m, mode, credit_left=args.credit_left,
                                                      price_in=args.price_input, price_out=args.price_output))
    judge = factory(args.judge_model, args.mode)
    ctx = rewards.RewardContext(args.mode, target, seeds.read_bank(args.bank, dhash), target.tokenizer, judge,
                                lam_div=args.lam_div, lam_len=args.lam_len)
    if generate is None:
        dev = device()
        model, tok = load_adapter(args.base, tl.checkpoint_path(args.ckpt_dir, step), dev)
        generate = make_generate(model, tok, dev, seed=args.sample_seed)
    prov = Provenance("rl", args.base, target.version, dhash, adapter_step=step, judge=args.judge_model,
                      out=rows.batch_path(args.mode, "rl", heldout=args.heldout, out_dir=args.out_dir))
    prefix = f"{args.mode.lower()}-{'heldout' if args.heldout else 'rl'}-"
    last = [b for b in rows.batches(prov.out) if b.startswith(prefix)]
    if args.resume and not last:
        raise SystemExit("--resume: no batch to continue")
    prov.batch = last[-1] if args.resume else rows.new_batch_id(prov.out, args.mode, "rl", heldout=args.heldout)
    tl.record_sampled(args.ckpt_dir, step, args.heldout)
    report = select_and_write(ctx, seed_rows, generate, prov, n=args.n, credit_left=args.credit_left, log=log)
    if hasattr(judge, "close"):
        judge.close()
    return {**report, "adapter_step": step, "heldout": bool(args.heldout), "sample_seed": args.sample_seed}


def main(argv: Optional[List[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
