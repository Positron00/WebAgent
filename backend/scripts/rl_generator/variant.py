#!/usr/bin/env python3
"""A dataset folder = the base build + reviewed generated rows (rl-generator SEC-213).

    python scripts/rl_generator/variant.py --base data/security/alignment --kind rl --share 0.15 \
        --modes H,L --out data/security/alignment/local/variants/rl-15

Copies the base's `{train,test,probe}.jsonl` and `local/` files, then appends to `--out/local/train.jsonl`
the rows of `pass` batches of `--kind` made from this build (data hash), never held-out batches,
after the leakage drops, at most round(share / 2 x real train rows) per mode, chosen at random (seed
13). `--batches` limits it to the named batch ids (an R3 round replaces the previous round's rows
with its own). `check_data` must pass on the result, else `--out` is removed. `--kind none --share 0`
is a plain copy (the baseline). Counts only (SNFR-20).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_generator  # noqa: E402

rl_generator.bootstrap()

from alignment_data import NEAR_DUPLICATE, SPLITS, check_data, read_jsonl  # noqa: E402

PICK_SEED = 13
REVIEW_FILE = "rl_review.json"


def batch_file(generated: Path, mode: str, kind: str) -> Path:
    return generated / f"rl_gen_{mode.lower()}_{kind}.jsonl"


def is_heldout(batch: str) -> bool:
    return "-heldout-" in f"{batch}-"


def base_hash(base: Path) -> str:
    from train_alignment_model import data_hash  # lazy: the training script imports sklearn

    return data_hash(base)


def copy_base(base: Path, out: Path) -> None:
    (out / "local").mkdir(parents=True, exist_ok=True)
    for folder in ("", "local"):
        for split in SPLITS:
            src, dst = base / folder / f"{split}.jsonl", out / folder / f"{split}.jsonl"
            if src.is_file():
                shutil.copyfile(src, dst)
            elif dst.exists():
                dst.unlink()


def candidates(generated: Path, mode: str, kind: str, review: dict, data_hash: str, dropped: Counter,
               only: Optional[set] = None):
    """(rows, batches) of `pass` batches of this mode and kind made from this build (only the batch
    ids in `only`, when given)."""
    rows, batches = [], set()
    for row in read_jsonl(batch_file(generated, mode, kind)):
        meta = row.get("meta") or {}
        batch = str(meta.get("batch", ""))
        if is_heldout(batch):
            raise SystemExit(f"held-out batch in {batch_file(generated, mode, kind).name}: refused (SEC-216)")
        if row.get("origin") != f"{rl_generator.SOURCE[mode]}:{kind}":
            raise SystemExit(f"row of another mode or kind in {batch_file(generated, mode, kind).name}")
        if only is not None and batch not in only:
            dropped["batch not selected"] += 1
        elif (review.get(batch) or {}).get("verdict") != "pass":
            dropped["not passed"] += 1
        elif meta.get("data_hash") != data_hash:
            dropped["stale data hash"] += 1
        else:
            rows.append(row)
            batches.add(batch)
    return rows, sorted(batches)


def leakage_drops(rows: List[dict], base_rows: Dict[str, List[dict]], seen: set,
                  embed: Callable[[List[str]], object], dropped: Counter) -> List[dict]:
    """SEC-213: seed groups all in base train; call not near a base test or probe call; no duplicate."""
    import numpy as np

    train_groups = {g for r in base_rows["train"] for g in r.get("groups") or []}
    inside = [set(r.get("groups") or []) <= train_groups for r in rows]
    dropped["seed groups not all in base train"] += inside.count(False)
    rows = [r for r, ok in zip(rows, inside) if ok]
    held_calls = [r["call"] for s in ("test", "probe") for r in base_rows[s]]
    if rows and held_calls:
        sims = np.asarray(embed([r["call"] for r in rows])) @ np.asarray(embed(held_calls)).T
        near = sims.max(axis=1) >= NEAR_DUPLICATE
        dropped["near a test or probe call"] += int(near.sum())
        rows = [r for r, n in zip(rows, near) if not n]
    kept = []
    for r in rows:
        key = (r["intent"], r["call"], r["action"])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept


def finished(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k != "meta"}
    out["split"], out["cluster"] = "train", f"gen-{(row.get('meta') or {}).get('seed_id', '')}"
    return out


def build(base: Path, out: Path, kind: str, share: float, modes: Sequence[str], *,
          generated: Path = rl_generator.GENERATED_DIR, local: Path = rl_generator.LOCAL,
          embed: Optional[Callable[[List[str]], object]] = None, batches: Optional[Sequence[str]] = None) -> dict:
    base, out = base.resolve(), out.resolve()
    if not out.is_relative_to(local.resolve()) or out == local.resolve():
        raise SystemExit(f"--out must be a folder under {local}")
    if kind not in ("none", *rl_generator.KINDS) or (kind == "none") != (share == 0) or not 0 <= share <= 1:
        raise SystemExit("--kind none goes with --share 0 only; prompted or rl need a share in (0, 1]")
    modes = [rl_generator.check_mode(m) for m in modes]
    only = set(batches) if batches else None
    if only and any(is_heldout(b) for b in only):
        raise SystemExit("--batches names a held-out batch: refused (SEC-216)")
    data_hash = base_hash(base)
    base_rows = {s: read_jsonl(base / f"{s}.jsonl") + read_jsonl(base / "local" / f"{s}.jsonl") for s in SPLITS}
    real_train = len(base_rows["train"])
    cap = round(share / 2 * real_train)
    review = json.loads((generated / REVIEW_FILE).read_text(encoding="utf-8")) if (generated / REVIEW_FILE).is_file() else {}
    seen = {(r["intent"], r["call"], r["action"]) for rows in base_rows.values() for r in rows}
    if embed is None and kind != "none":
        from build_alignment_dataset import Embedder

        embed = Embedder()
    copy_base(base, out)
    info = {"base_data_hash": data_hash, "kind": kind, "share": share, "modes": modes, "real_train_rows": real_train,
            "cap_per_mode": cap, "selected_batches": sorted(only) if only else None, "batches": {}, "candidates": {}, "kept": {}, "dropped": {}, "shortfall": {}}
    added: List[dict] = []
    for mode in modes:
        dropped: Counter = Counter()
        rows, batches = ([], []) if kind == "none" else candidates(generated, mode, kind, review, data_hash, dropped, only)
        info["candidates"][mode] = len(rows)
        rows = leakage_drops(rows, base_rows, seen, embed, dropped) if rows else rows
        if len(rows) > cap:
            picked = set(random.Random(PICK_SEED).sample(range(len(rows)), cap))
            rows = [r for i, r in enumerate(rows) if i in picked]
        info["batches"][mode], info["kept"][mode] = batches, len(rows)
        info["dropped"][mode] = {k: v for k, v in sorted(dropped.items()) if v}
        info["shortfall"][mode] = max(cap - len(rows), 0)
        added += [finished(r) for r in rows]
    if added:
        with (out / "local" / "train.jsonl").open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in added)
    if embed is None:
        from build_alignment_dataset import Embedder

        embed = Embedder()
    errors = check_data(out, embed, base / "README.md")
    if errors:
        shutil.rmtree(out)
        print(json.dumps({"check_data": "failed", "errors": errors}, indent=1))
        raise SystemExit(f"check_data failed on the variant; {out.name} removed")
    (out / "variant.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=rl_generator.DATA_DIR)
    ap.add_argument("--kind", choices=("none", *rl_generator.KINDS), required=True)
    ap.add_argument("--share", type=float, required=True)
    ap.add_argument("--modes", default="H,L")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generated", type=Path, default=rl_generator.GENERATED_DIR)
    ap.add_argument("--batches", default="", help="comma-separated batch ids to use (default: all passed)")
    ap.add_argument("--local", type=Path, default=rl_generator.LOCAL, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    info = build(args.base, args.out, args.kind, args.share, [m for m in args.modes.split(",") if m],
                 generated=args.generated, local=args.local,
                 batches=[b for b in args.batches.split(",") if b] or None)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
