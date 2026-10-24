"""Dataset rows from accepted samples, and batch files (rl-generator SEC-208, SEC-216, SNFR-25; design §5).

A row copies its seed's intent, action and groups, takes the mode's labels and generated source
(plus the seed's sources, so licence rules carry over), and records its provenance in `meta`
(stripped when it goes into a variant). Batch files are `LOCAL/generated/rl_gen_<mode>_<kind>.jsonl`
(held-out: `rl_gen_<mode>_heldout.jsonl`), appended; batch ids are `<mode>-<kind>-<yyyymmdd>-<n>`
(`<mode>-heldout-...`), mode in lower case. A resumed batch skips seed ids it already has.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Set

from rl_generator import GENERATED_DIR, KINDS, LABELS, SOURCE, check_mode

HELDOUT = "heldout"
_BATCH = re.compile(r"^(h|l)-(prompted|rl|heldout)-(\d{8})-(\d+)$")


def _kind(kind: str, heldout: bool) -> str:
    if kind not in KINDS:
        raise SystemExit(f"unknown kind {kind!r}: only {', '.join(KINDS)}")
    if heldout and kind != "rl":
        raise SystemExit("held-out rows are rl rows")
    return HELDOUT if heldout else kind


def batch_path(mode: str, kind: str, heldout: bool = False, out_dir: Path = GENERATED_DIR) -> Path:
    return out_dir / f"rl_gen_{check_mode(mode).lower()}_{_kind(kind, heldout)}.jsonl"


def batch_id(mode: str, kind: str, day: str, n: int, heldout: bool = False) -> str:
    return f"{check_mode(mode).lower()}-{_kind(kind, heldout)}-{day}-{int(n)}"


def parse_batch(batch: str) -> Optional[dict]:
    """{"mode", "kind", "heldout", "day", "n"} of a batch id; None if it isn't one."""
    m = _BATCH.match(batch or "")
    if not m:
        return None
    heldout = m.group(2) == HELDOUT
    return {"mode": m.group(1).upper(), "kind": "rl" if heldout else m.group(2), "heldout": heldout,
            "day": m.group(3), "n": int(m.group(4))}


def read_rows(path: Path, batch: Optional[str] = None) -> List[dict]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [r for r in rows if batch is None or r.get("meta", {}).get("batch") == batch]


def batches(path: Path) -> List[str]:
    """Batch ids in a file, in first-seen order."""
    seen: List[str] = []
    for r in read_rows(path):
        b = r.get("meta", {}).get("batch")
        if b and b not in seen:
            seen.append(b)
    return seen


def new_batch_id(path: Path, mode: str, kind: str, heldout: bool = False, day: Optional[str] = None) -> str:
    day = day or date.today().strftime("%Y%m%d")
    prefix = batch_id(mode, kind, day, 0, heldout).rsplit("-", 1)[0] + "-"
    taken = [int(b.rsplit("-", 1)[1]) for b in batches(path) if b.startswith(prefix)]
    return batch_id(mode, kind, day, max(taken, default=0) + 1, heldout)


def done_seed_ids(path: Path, batch: str) -> Set[str]:
    return {r["meta"]["seed_id"] for r in read_rows(path, batch)}


def make_row(seed: dict, call: str, mode: str, kind: str, *, batch: str, generator: str,
             adapter_step: Optional[int], target: str, judge: str, prompt: str, miss: float,
             novelty: float, data_hash: str) -> dict:
    mode = check_mode(mode)
    src = SOURCE[mode]
    return {"intent": seed["intent"], "call": call, "action": seed["action"], "labels": dict(LABELS[mode]),
            "origin": f"{src}:{kind}", "role": "T", "split": "train", "groups": list(seed["groups"]),
            "sources": [src, *[s for s in seed["sources"] if s != src]], "cluster": "",
            "meta": {"batch": batch, "seed_id": seed["seed_id"], "seed_origin": seed.get("origin", ""),
                     "generator": generator, "adapter_step": adapter_step, "target": target, "judge": judge,
                     "prompt": prompt, "miss": round(float(miss), 4), "novelty": round(float(novelty), 4),
                     "data_hash": data_hash}}


def append(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
