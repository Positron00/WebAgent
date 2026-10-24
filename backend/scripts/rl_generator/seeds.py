#!/usr/bin/env python3
"""Seeds for the generator, their parts, and the novelty bank (rl-generator SEC-201, SEC-206; design §2).

Seeds are committed train rows (never `local/`, test or probe) of role T, labels (yes, none), from
the five seed origins. Each mode draws independently (selection seeds 7 for H, 8 for L); 60% of a
mode's seeds are the `train` part (prompts for train.py), 40% the `sample` part (prompted.py and
sample.py). The seed file records the build's data hash and is refused for another build.

    python scripts/rl_generator/seeds.py --mode H --n 12000      # LOCAL/generated/seeds_h.jsonl
    python scripts/rl_generator/seeds.py --bank                  # LOCAL/generated/novelty_bank.npy

Prints counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_generator  # noqa: E402
from rl_generator import DATA_DIR, GENERATED_DIR, check_mode  # noqa: E402

rl_generator.bootstrap()

SEED_ORIGINS = ("webagent-gen:call", "asq:query", "glaive:call", "toolace:call", "saroku:safe")
SELECTION_SEED = {"H": 7, "L": 8}
TRAIN_SHARE = 0.6
MAX_ORIGIN_SHARE = 1 / 3
MIN_ACTION_SHARE = 0.1
SEED_ACTIONS = ("web_search", "internal_retrieve", "plan", "function")
BANK = GENERATED_DIR / "novelty_bank.npy"


def seed_id(intent: str, call: str) -> str:
    return hashlib.sha256(f"{intent}\n{call}".encode("utf-8")).hexdigest()[:16]


def seed_path(mode: str, out_dir: Path = GENERATED_DIR) -> Path:
    return out_dir / f"seeds_{check_mode(mode).lower()}.jsonl"


def eligible(row: dict) -> bool:
    import alignment_data

    return (row.get("role") == "T" and row.get("labels") == {"aligned": "yes", "risk": "none"}
            and row.get("origin") in SEED_ORIGINS and row.get("split", "train") == "train"
            and not alignment_data.is_local(row))


def _record(row: dict) -> dict:
    return {"seed_id": seed_id(row["intent"], row["call"]), "intent": row["intent"], "call": row["call"],
            "action": row["action"], "origin": row["origin"], "groups": list(row["groups"]),
            "sources": list(row["sources"])}


def select(train_rows: List[dict], mode: str, n: int, seed: Optional[int] = None,
           exclude: Iterable[str] = frozenset()) -> List[dict]:
    """`n` seeds stratified so each origin gives at most a third and each action at least 10%
    (as far as the pool allows), split 60/40 into `train` and `sample` parts. Deterministic."""
    mode = check_mode(mode)
    rng = random.Random(SELECTION_SEED[mode] if seed is None else seed)
    skip, pool, seen = set(exclude), [], set()
    for row in train_rows:
        if not eligible(row):
            continue
        rec = _record(row)
        if rec["seed_id"] in skip or rec["seed_id"] in seen:
            continue
        seen.add(rec["seed_id"])
        pool.append(rec)
    rng.shuffle(pool)
    n = min(n, len(pool))
    origin_cap = max(1, math.ceil(n * MAX_ORIGIN_SHARE))
    by_origin: Counter = Counter()
    chosen: List[dict] = []
    taken = set()

    def take(rec: dict) -> bool:
        if len(chosen) >= n or rec["seed_id"] in taken or by_origin[rec["origin"]] >= origin_cap:
            return False
        chosen.append(rec)
        taken.add(rec["seed_id"])
        by_origin[rec["origin"]] += 1
        return True

    by_action: Dict[str, List[dict]] = defaultdict(list)
    for rec in pool:
        by_action[rec["action"]].append(rec)
    action_min = math.ceil(n * MIN_ACTION_SHARE)
    for action in SEED_ACTIONS:
        got = 0
        for rec in by_action.get(action, []):
            if got >= action_min:
                break
            got += take(rec)
    for rec in pool:
        if len(chosen) >= n:
            break
        take(rec)
    rng.shuffle(chosen)
    cut = round(len(chosen) * TRAIN_SHARE)
    return [{**rec, "part": "train" if i < cut else "sample"} for i, rec in enumerate(chosen)]


def write(path: Path, seeds: List[dict], data_hash: str, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"data_hash": data_hash, "mode": check_mode(mode), "n": len(seeds)}
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for rec in seeds:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_header(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def read(path: Path, expected_hash: Optional[str], part: Optional[str] = None) -> List[dict]:
    """The seeds of `path` (optionally one part); SystemExit when its data hash isn't `expected_hash`
    (pass None only to read a file without generating from it)."""
    header = read_header(path)
    if expected_hash is not None and header.get("data_hash") != expected_hash:
        raise SystemExit("seed file is from another build (data hash differs); make new seeds")
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    seeds = [json.loads(line) for line in lines if line.strip()]
    return [s for s in seeds if part is None or s["part"] == part]


def current_hash(data_dir: Path = DATA_DIR) -> str:
    from train_alignment_model import data_hash

    return data_hash(data_dir)


def base_embed(texts: List[str]):
    """Normalised base MiniLM-L6 embeddings (SEC-206), float32."""
    import numpy as np
    from app.core.model_lock import lock_for
    from app.security.encoder import get_base_encoder

    encoder = get_base_encoder()[0]
    if encoder is None:
        raise SystemExit("the base MiniLM encoder is not available")
    with lock_for(encoder):
        out = encoder.encode(list(texts), normalize_embeddings=True, batch_size=256,
                             show_progress_bar=len(texts) > 10000)
    return np.asarray(out, dtype=np.float32)


def write_bank(train_rows: List[dict], out: Path, data_hash: str,
               embed: Callable[[List[str]], "object"] = base_embed) -> int:
    """Embeddings of the committed train calls (never `local/` rows), float16, with a sidecar JSON."""
    import alignment_data
    import numpy as np

    calls = sorted({r["call"] for r in train_rows if not alignment_data.is_local(r)})
    bank = np.asarray(embed(calls), dtype=np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, bank)
    out.with_suffix(".json").write_text(json.dumps({"data_hash": data_hash, "n": len(calls),
                                                    "encoder": "all-MiniLM-L6-v2"}) + "\n")
    return len(calls)


def read_bank(path: Path, expected_hash: str):
    import numpy as np

    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("data_hash") != expected_hash:
        raise SystemExit("novelty bank is from another build (data hash differs); run seeds.py --bank")
    return np.load(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", help="H or L")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="seed files whose ids to leave out")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bank", action="store_true", help="write the novelty bank instead of seeds")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    import alignment_data

    rows = alignment_data.read_jsonl(args.data_dir / "train.jsonl")
    dhash = current_hash(args.data_dir)
    if args.bank:
        n = write_bank(rows, args.out or BANK, dhash)
        print(json.dumps({"bank_calls": n, "data_hash": dhash}))
        return 0
    mode = check_mode(args.mode or "")
    exclude = {s["seed_id"] for f in args.exclude for s in read(f, None)}
    seeds = select(rows, mode, args.n, exclude=exclude)
    out = args.out or seed_path(mode)
    write(out, seeds, dhash, mode)
    print(json.dumps({"mode": mode, "seeds": len(seeds), "parts": dict(Counter(s["part"] for s in seeds)),
                      "by_origin": dict(Counter(s["origin"] for s in seeds)),
                      "by_action": dict(Counter(s["action"] for s in seeds)), "data_hash": dhash}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
