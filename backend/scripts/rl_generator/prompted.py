#!/usr/bin/env python3
"""Prompted rows from the untrained generator (rl-generator SEC-208; design §6).

Per `sample`-part seed: 8 samples; score all without the judge (§4.1 steps 1–5); among harmless,
not-overlong samples judge the 2 with the highest `miss + lam_div * novelty`; keep the judged-valid
one with the higher reward, else nothing. `select_and_write` is shared with sample.py.

    python scripts/rl_generator/prompted.py --mode H --judge-model <m> --price-input X --price-output Y --estimate 50
    python scripts/rl_generator/prompted.py --mode H --judge-model <m> ... --n 500 --credit-left <dollars>
    python scripts/rl_generator/prompted.py --mode H --estimate 50 --judge-candidates <m1>,<m2> ...

A full run stops before spending would leave less than half of `--credit-left`, or on the judge's
error rule, and resumes by seed id (`--resume`). Prints counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_generator  # noqa: E402
from rl_generator import BACKEND, GENERATED_DIR, REVIEW_DIR, check_mode, rewards, rows, seeds  # noqa: E402
from rl_generator.prompts import PROMPT_VERSION, generator_messages  # noqa: E402

rl_generator.bootstrap()

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET = BACKEND / "models" / "security" / "alignment"
SAMPLES, JUDGED = 8, 2
CHOICE = REVIEW_DIR / "judge-choice.jsonl"
Generate = Callable[[List[list]], List[List[str]]]


@dataclass
class Provenance:
    kind: str                 # prompted | rl
    generator: str
    target: str
    data_hash: str
    adapter_step: Optional[int] = None
    judge: str = ""
    batch: Optional[str] = None   # None: write no batch (estimate)
    out: Optional[Path] = None


def pick(ctx: rewards.RewardContext, seed: dict, completions: Sequence[str], judges: Sequence = ()):
    """(call, Scored) of the kept sample or (None, None), the judged (call, Scored) pairs and the
    scores of all samples. `judges`: extra candidate judges for the judge choice (ctx.judge unused)."""
    group = [seed] * len(completions)
    scored = rewards.score_samples(ctx, group, list(completions), judge=False)
    texts = [rewards.clean(c, seed["action"]) for c in completions]
    ranked = sorted((i for i, s in enumerate(scored) if s.judgeable() and s.miss >= ctx.judge_min_miss),
                    key=lambda i: scored[i].selection_score(ctx.lam_div), reverse=True)[:JUDGED]
    judged, verdicts = [], []
    for i in ranked:
        if judges:
            verdicts.append({j.model: j(seed["intent"], texts[i], seed["action"]) for j in judges})
        else:
            rewards.apply_judge(ctx, seed, texts[i], scored[i])
        judged.append((texts[i], scored[i]))
    valid = [(t, s) for t, s in judged if s.valid]
    best = max(valid, key=lambda ts: ts[1].reward) if valid else (None, None)
    return best, judged, verdicts, scored


def _stop(judges: Sequence, credit_left: Optional[float], next_cost: float) -> Optional[str]:
    if any(j.error_stop() for j in judges):
        return "judge errors over 20%"
    if credit_left is not None and sum(j.spent() for j in judges) + next_cost >= credit_left / 2:
        return "credit rule"
    return None


def select_and_write(ctx: rewards.RewardContext, seed_rows: List[dict], generate: Generate, prov: Provenance, *,
                     n: int, credit_left: Optional[float] = None, candidates: Sequence = (),
                     choice_out: Optional[Path] = None, log: Callable[[str], None] = print) -> dict:
    """Runs the selection over `seed_rows` until `n` rows are accepted (or, with no batch, over all
    of them), appending accepted rows to `prov.out` under `prov.batch`. Returns counts only."""
    judges = list(candidates) or ([ctx.judge] if ctx.judge is not None else [])
    done = rows.done_seed_ids(prov.out, prov.batch) if prov.batch and prov.out else set()
    accepted = len(done)
    counts = {"seeds": 0, "samples": 0, "empty": 0, "harmful_filtered": 0, "overlong": 0, "judged": 0,
              "valid": 0, "rows": 0, "resumed_rows": accepted}
    per_candidate = {j.model: 0 for j in candidates}
    started, stopped = time.time(), None
    for seed in seed_rows:
        if prov.batch and accepted >= n:
            break
        if seed["seed_id"] in done:
            continue
        per_seed = sum(j.spent() for j in judges) / counts["seeds"] if counts["seeds"] else 0.0
        stopped = _stop(judges, credit_left, per_seed)
        if stopped:
            log(f"Stopping ({stopped}) after {counts['seeds']} seeds")
            break
        filtered_before = ctx.harmful_filtered
        completions = generate([generator_messages(ctx.mode, seed)])[0]
        (call, best), judged, verdicts, scored = pick(ctx, seed, completions, candidates)
        counts["seeds"] += 1
        counts["samples"] += len(scored)
        counts["empty"] += sum(s.empty for s in scored)
        counts["harmful_filtered"] += ctx.harmful_filtered - filtered_before
        counts["overlong"] += sum(bool(s.overlong) for s in scored)
        counts["judged"] += len(judged)
        counts["valid"] += sum(bool(s.valid) for _, s in judged)
        for j in judges:
            j.end_seed()
        if candidates and choice_out is not None:
            _write_choice(choice_out, ctx.mode, seed, judged, verdicts)
            for v in verdicts:
                for model, ok in v.items():
                    per_candidate[model] += ok is True
        if call is None:
            continue
        ctx.accept(best)
        accepted += 1
        counts["rows"] += 1
        if prov.batch and prov.out:
            rows.append(prov.out, [rows.make_row(
                seed, call, ctx.mode, prov.kind, batch=prov.batch, generator=prov.generator,
                adapter_step=prov.adapter_step, target=prov.target, judge=prov.judge, prompt=PROMPT_VERSION,
                miss=best.miss, novelty=best.novelty, data_hash=prov.data_hash)])
    minutes = max(time.time() - started, 1e-9) / 60
    spent = sum(j.spent() for j in judges)
    report = {**counts, "stopped": stopped, "batch": prov.batch, "seeds_per_minute": round(counts["seeds"] / minutes, 2),
              "judges": [j.report() for j in judges], "cost_usd": round(spent, 4), "reward": ctx.record()}
    if candidates:
        report["accepted_by_candidate"] = per_candidate
    if counts["rows"]:
        report["projected_cost_for_n_usd"] = round(spent / counts["rows"] * n, 2)
    return report


def _write_choice(path: Path, mode: str, seed: dict, judged, verdicts) -> None:
    out = []
    for (call, s), v in zip(judged, verdicts):
        rid = hashlib.sha256(f"{mode}\n{seed['seed_id']}\n{call}".encode("utf-8")).hexdigest()[:12]
        out.append({"row_id": rid, "mode": mode, "seed_id": seed["seed_id"], "seed_origin": seed.get("origin", ""),
                    "intent": seed["intent"], "action": seed["action"], "call": call,
                    "miss": round(s.miss, 4), "novelty": round(s.novelty, 4), "verdicts": v})
    rows.append(path, out)


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_generator(base: str, dev: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    dtype = torch.float16 if dev == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(dev).eval()
    return model, tok


def make_generate(model, tok, dev: str, seed: int = 7, samples: int = SAMPLES,
                  temperature: float = 0.9, top_p: float = 0.95, max_new_tokens: int = 96) -> Generate:
    """A `generate(prompts) -> [[completion] * samples]` over chat messages (SEC-208; reused by sample.py)."""
    import torch

    torch.manual_seed(seed)

    def generate(prompts: List[list]) -> List[List[str]]:
        out = []
        for messages in prompts:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(dev)
            with torch.no_grad():
                ids = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                                     max_new_tokens=max_new_tokens, num_return_sequences=samples,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            out.append(tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return out

    return generate


def load_target(path: Path):
    from app.security import alignment_model

    target = alignment_model.load(path, require_gate=False)
    if target is None:
        raise SystemExit("target model missing or incomplete (Git LFS pointer?)")
    return target


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, help="H or L")
    ap.add_argument("--seeds", type=Path, default=None, help="default LOCAL/generated/seeds_<mode>.jsonl")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-candidates", default=None, help="comma-separated models; with --estimate only")
    ap.add_argument("--estimate", type=int, default=0, help="run this many seeds, write no batch")
    ap.add_argument("--n", type=int, default=500, help="accepted rows wanted")
    ap.add_argument("--credit-left", type=float, default=None)
    ap.add_argument("--price-input", type=float, default=None)
    ap.add_argument("--price-output", type=float, default=None)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--target", type=Path, default=TARGET)
    ap.add_argument("--bank", type=Path, default=seeds.BANK)
    ap.add_argument("--lam-div", type=float, default=0.2)
    ap.add_argument("--lam-len", type=float, default=0.5)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true", help="continue the file's last batch of this kind")
    ap.add_argument("--out-dir", type=Path, default=GENERATED_DIR)
    ap.add_argument("--choice-out", type=Path, default=CHOICE)
    args = ap.parse_args(argv)
    args.mode = check_mode(args.mode)
    args.candidates = [m for m in (args.judge_candidates or "").split(",") if m]
    if args.candidates and not args.estimate:
        ap.error("--judge-candidates needs --estimate")
    if not args.candidates and not args.judge_model:
        ap.error("--judge-model is required")
    if not args.estimate and args.credit_left is None:
        ap.error("a full run needs --credit-left (or use --estimate N)")
    return args


def run(args: argparse.Namespace, judge_factory=None, generate: Optional[Generate] = None, log=print) -> dict:
    """The CLI's work; `judge_factory(model, mode)` and `generate` are hooks for tests and smoke runs."""
    from rl_generator.judge import Judge

    dhash = seeds.current_hash()
    seed_rows = seeds.read(args.seeds or seeds.seed_path(args.mode), dhash, part="sample")
    if args.estimate:
        seed_rows = seed_rows[:args.estimate]
    target = load_target(args.target)
    factory = judge_factory or (lambda m, mode: Judge(m, mode, credit_left=args.credit_left,
                                                      price_in=args.price_input, price_out=args.price_output))
    candidates = [factory(m, args.mode) for m in args.candidates]
    judge = None if candidates else factory(args.judge_model, args.mode)
    ctx = rewards.RewardContext(args.mode, target, seeds.read_bank(args.bank, dhash), target.tokenizer, judge,
                                lam_div=args.lam_div, lam_len=args.lam_len)
    if generate is None:
        dev = device()
        model, tok = load_generator(args.base, dev)
        generate = make_generate(model, tok, dev, seed=args.sample_seed)
    prov = Provenance("prompted", args.base, target.version, dhash, judge=args.judge_model or "")
    if not args.estimate:
        prov.out = rows.batch_path(args.mode, "prompted", out_dir=args.out_dir)
        last = [b for b in rows.batches(prov.out) if b.startswith(f"{args.mode.lower()}-prompted-")]
        if args.resume and not last:
            raise SystemExit("--resume: no batch to continue")
        prov.batch = last[-1] if args.resume else rows.new_batch_id(prov.out, args.mode, "prompted")
    return select_and_write(ctx, seed_rows, generate, prov, n=args.n, credit_left=args.credit_left,
                            candidates=candidates, choice_out=args.choice_out if candidates else None, log=log)


def main(argv: Optional[List[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
