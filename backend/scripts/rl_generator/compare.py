#!/usr/bin/env python3
"""R2's comparison and R3's rounds: variants, two seeds each, gate, table, decision (rl-generator SEC-217).

    python scripts/rl_generator/compare.py --base data/security/alignment --modes H,L \
        --out data/security/alignment/local/compare/r2 [--dry-run | --time-one]
    python scripts/rl_generator/compare.py --round --modes H --previous <variant> --new <variant> \
        --out data/security/alignment/local/compare/r3-h-1
    (build --new with variant.py --batches <the round's batch ids>, at the kept share)

Per mode (only that mode's rows): none, p5, p15, r5, r15, each trained with seeds 13 and 17 (`none`
is shared, so 18 runs for two modes), gated with `--extra` on the held-out sets. Resumable: a run whose
metadata.json has a gate result is not redone. Counts and numbers only (SNFR-20).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_generator  # noqa: E402

rl_generator.bootstrap()

SEEDS = (13, 17)
CONFIGS = {"p5": ("prompted", 0.05), "p15": ("prompted", 0.15), "r5": ("rl", 0.05), "r15": ("rl", 0.15)}
METRIC = {"H": ("off_intent_recall", True), "L": ("lookalikes_held", False)}  # (key, higher is better)
REAL_HELD_MAX, ECE_SLACK = 0.01, 0.01
COLUMNS = ("metric", "off_intent_recall", "risk_recall", "lookalikes_held", "real_aligned_held_upper",
           "ece_aligned", "ece_risk", "heldout", "T_off", "T_risk", "E_off", "E_risk")


# ---- pure decision logic (tests feed fixed numbers) ----

def metrics(test: dict, mode: str) -> Dict[str, Optional[float]]:
    """The numbers one gated run contributes to the table of a mode."""
    role = test.get("by_role") or {}
    held = (test.get("extra") or {}).get(f"rl_gen_{mode.lower()}_heldout") or {}
    return {"metric": test.get(METRIC[mode][0]),
            **{k: test.get(k) for k in COLUMNS[1:7]},
            "heldout": held.get("recall" if mode == "H" else "held_share"),
            **{f"{r}_{short}": (role.get(r) or {}).get(key) for r in ("T", "E")
               for short, key in (("off", "off_intent_recall"), ("risk", "risk_recall"))}}


def summarise(per_seed: Sequence[Dict[str, Optional[float]]]) -> Dict[str, dict]:
    """{column: {"mean", "spread", "max"}}; spread = |seed 13 - seed 17| (max - min for other counts)."""
    out = {}
    for col in COLUMNS:
        vals = [m[col] for m in per_seed if m.get(col) is not None]
        out[col] = {"mean": round(statistics.fmean(vals), 4), "spread": round(max(vals) - min(vals), 4),
                    "max": max(vals)} if vals else {"mean": None, "spread": None, "max": None}
    return out


def beats(new: dict, old: dict, none: dict, mode: str) -> dict:
    """SEC-217: `new` beats `old` on the mode's metric by more than the larger seed spread, with real
    aligned held (upper bound, every seed) <= 1% and both ECEs no more than 0.01 above none's mean."""
    higher = METRIC[mode][1]
    a, b = new["metric"], old["metric"]
    if a["mean"] is None or b["mean"] is None:
        return {"wins": False, "diff": None, "noise": None, "reason": "missing metric"}
    diff = round((a["mean"] - b["mean"]) if higher else (b["mean"] - a["mean"]), 4)
    noise = max(a["spread"], b["spread"])
    reasons = []
    if not diff > noise:
        reasons.append("difference within seed noise")
    if new["real_aligned_held_upper"]["max"] is None or new["real_aligned_held_upper"]["max"] > REAL_HELD_MAX:
        reasons.append("real aligned held upper bound above 1%")
    for ece in ("ece_aligned", "ece_risk"):
        if new[ece]["mean"] is None or none[ece]["mean"] is None or new[ece]["mean"] > round(none[ece]["mean"] + ECE_SLACK, 6):
            reasons.append(f"{ece} above none + {ECE_SLACK}")
    return {"wins": not reasons, "diff": diff, "noise": noise, "reason": "; ".join(reasons) or "wins"}


def decide(summary: Dict[str, dict], mode: str) -> dict:
    """R2's decision for one mode from {"none", "p5", "p15", "r5", "r15"} summaries: RL is kept if it
    beats prompting at the same share (at either share); 15% is kept only if it beats 5% in that kind."""
    none = summary["none"]
    at_share = {s: beats(summary[f"r{s}"], summary[f"p{s}"], none, mode) for s in ("5", "15")}
    kind = "rl" if any(v["wins"] for v in at_share.values()) else "prompted"
    k = kind[0]
    share = beats(summary[f"{k}15"], summary[f"{k}5"], none, mode)
    return {"rl_vs_prompted": at_share, "kind": kind, "15_vs_5": share, "share": 0.15 if share["wins"] else 0.05,
            "keep": f"{k}{15 if share['wins'] else 5}"}


def decide_round(summary: Dict[str, dict], mode: str) -> dict:
    """R3: the new round's variant is kept only if it beats the previous one; else the mode stops."""
    result = beats(summary["new"], summary["previous"], summary["none"], mode)
    return {"new_vs_previous": result, "keep": "new" if result["wins"] else "previous", "stop": not result["wins"]}


def table(summary: Dict[str, dict], mode: str) -> str:
    head = ["config", f"{METRIC[mode][0]} (mean ± spread)", *COLUMNS[1:]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for config, s in summary.items():
        m = s["metric"]
        lines.append("| " + " | ".join([config, f"{m['mean']} ± {m['spread']}",
                                        *(str(s[c]["mean"]) for c in COLUMNS[1:])]) + " |")
    return "\n".join(lines)


# ---- runs ----

def plan(args) -> Dict[str, dict]:
    """{config name: {"variant", "args" (variant.py) or None, "modes"}}; run dirs are <out>/runs/<name>-s<seed>."""
    out, modes = args.out, args.modes
    configs = {"none": {"variant": out / "variants" / "none", "args": ["--kind", "none", "--share", "0",
                                                                         "--modes", ",".join(modes)]}}
    if args.round:
        configs["previous"] = {"variant": args.previous.resolve(), "args": None}
        configs["new"] = {"variant": args.new.resolve(), "args": None}
        return configs
    for mode in modes:
        for name, (kind, share) in CONFIGS.items():
            configs[f"{mode.lower()}-{name}"] = {"variant": out / "variants" / f"{mode.lower()}-{name}",
                                                 "args": ["--kind", kind, "--share", str(share), "--modes", mode]}
    return configs


def run(cmd: List[str], dry: bool, ok_codes: Sequence[int] = (0,)) -> None:
    print(("would run: " if dry else "running: ") + " ".join(Path(c).name if "/" in c else c for c in cmd), flush=True)
    if not dry and subprocess.run(cmd).returncode not in ok_codes:
        raise SystemExit(f"{Path(cmd[1]).name} failed")


def gated(run_dir: Path) -> Optional[dict]:
    meta = run_dir / "metadata.json"
    if not meta.is_file():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("test")


def execute(args, configs: Dict[str, dict], dry: bool, limit: Optional[int] = None) -> int:
    """Builds missing variants, trains and gates missing runs; returns the number of trainings done."""
    scripts, py = rl_generator.SCRIPTS, sys.executable
    extras = [p for m in args.modes if (p := args.generated / f"rl_gen_{m.lower()}_heldout.jsonl").is_file()]
    trained = 0
    for name, c in configs.items():
        if limit is not None and trained >= limit:
            return trained
        if c["args"] is not None and not (c["variant"] / "variant.json").is_file():
            run([py, str(scripts / "rl_generator" / "variant.py"), "--base", str(args.base), *c["args"],
                 "--out", str(c["variant"]), "--generated", str(args.generated)], dry)
        for seed in args.seeds:
            run_dir = args.out / "runs" / f"{name}-s{seed}"
            if limit is not None and trained >= limit:
                return trained
            if not (run_dir / "metadata.json").is_file():
                run([py, str(scripts / "train_alignment_model.py"), "--kind", "cross_encoder", "--data",
                     str(c["variant"]), "--seed", str(seed), "--out", str(run_dir)], dry)
                trained += 1
            if gated(run_dir) is None:  # the gate exits 1 when the run doesn't pass: still a result
                gate = [py, str(scripts / "eval_alignment_model.py"), "--model", str(run_dir), "--data", str(c["variant"])]
                run(gate + [x for p in extras for x in ("--extra", str(p))], dry, ok_codes=(0, 1))
    return trained


def summaries(args, configs: Dict[str, dict], mode: str) -> Dict[str, dict]:
    names = [n for n in configs if n in ("none", "previous", "new") or n.startswith(mode.lower() + "-")]
    out = {}
    for name in names:
        tests = [gated(args.out / "runs" / f"{name}-s{s}") for s in args.seeds]
        if any(t is None for t in tests):
            raise SystemExit(f"run {name} has no gate result yet")
        out[name.split("-", 1)[-1] if name.startswith(mode.lower() + "-") else name] = summarise(
            [metrics(t, mode) for t in tests])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=rl_generator.DATA_DIR)
    ap.add_argument("--modes", default="H,L")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generated", type=Path, default=rl_generator.GENERATED_DIR)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--time-one", action="store_true", help="train and gate one run, print the projected total")
    ap.add_argument("--round", action="store_true", help="R3: none, --previous and --new variants")
    ap.add_argument("--previous", type=Path)
    ap.add_argument("--new", type=Path)
    args = ap.parse_args(argv)
    args.modes = [rl_generator.check_mode(m) for m in args.modes.split(",") if m]
    args.seeds = [int(s) for s in args.seeds.split(",") if s]
    args.out = args.out.resolve()
    if args.round and (not args.previous or not args.new or len(args.modes) != 1):
        ap.error("--round takes one mode, --previous and --new")
    configs = plan(args)
    total = len(configs) * len(args.seeds)
    if args.dry_run:
        print(f"{len(configs)} configurations x {len(args.seeds)} seeds = {total} runs")
        execute(args, configs, dry=True)
        return 0
    if args.time_one:
        started = time.monotonic()
        done = execute(args, configs, dry=False, limit=1)
        took = time.monotonic() - started
        print(json.dumps({"trained": done, "seconds": round(took), "runs": total,
                          "projected_hours": round(took * total / 3600, 1) if done else None}))
        return 0
    execute(args, configs, dry=False)
    report = {}
    for mode in args.modes:
        summary = summaries(args, configs, mode)
        report[mode] = {"summary": summary,
                        "decision": decide_round(summary, mode) if args.round else decide(summary, mode)}
        print(f"\nmode {mode}\n{table(summary, mode)}\n{json.dumps(report[mode]['decision'], indent=1)}")
    (args.out / "compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
