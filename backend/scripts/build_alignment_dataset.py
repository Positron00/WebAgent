#!/usr/bin/env python3
"""Build the alignment classifier's train/test/probe JSONL (checkpoint2-intent SEC-120, SEC-121).

Run from backend/ after fetch_alignment_sources.py:
    python scripts/build_alignment_dataset.py --raw <scratch dir> [--out data/security/alignment] [--seed 13]
    python scripts/build_alignment_dataset.py --check-data [--out data/security/alignment]

Pairs, labels, filters and splits as in the model-and-data design §2–§3. Rows from gated and no-train
sources go only to <out>/local/ (git-ignored; the repository is public). QUOTAS below is the single
source of the numbers. Nothing here prints example text.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
for p in (str(BACKEND), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import alignment_agent_pairs as agent  # noqa: E402
import alignment_eval_pairs as evals  # noqa: E402
import alignment_pairs as pairs  # noqa: E402
from alignment_data import ASQ_SOURCES, NEAR_DUPLICATE, SPLITS, check_data, counts, is_local  # noqa: E402
from build_security_dataset import JAILBREAK_HINTS, aegis_group  # noqa: E402

os.environ.setdefault("TQDM_DISABLE", "1")  # no progress bars in the build log
OUT = BACKEND / "data" / "security" / "alignment"
# Target rows per group (before splitting). Shares: aligned with no risk ~50%, misaligned with no risk
# ~20%, risk above intent ~25%, look-alikes ~5% (fewer exist: about 2%). Function-calling sets capped.
QUOTAS = {
    # aligned, no risk
    "webagent_gen": 2000, "asq": 7000, "glaive": 4000, "toolace": 4000, "xlam": 4000, "saroku_safe": 2500,
    "asb_twin": 400,
    # misaligned, no risk
    "repair_easy": 3000, "repair_hard": 3500, "tasktracker": 1500,
    # risk above intent (saroku_unsafe also holds its low-risk rows)
    "injecagent": 900, "xstest_contrast": 200, "aegis": 1800, "harmbench": 200, "asb_risky": 700,
    "toolemu": 300, "redcode": 1300, "saroku_unsafe": 3500, "privacylens": 1000,
    # look-alikes (aligned, no risk)
    "xstest_lookalike": 250, "notinject": 339,
    # evaluation only (test and probe)
    "agentdojo": 600,
}


class Embedder:
    """Base MiniLM-L6 (`app.security.encoder.get_base_encoder`), normalised, cached per text."""

    def __init__(self):
        from app.security.encoder import get_base_encoder

        self.encoder, _ = get_base_encoder()
        if self.encoder is None:
            raise SystemExit("base MiniLM encoder unavailable")
        self.cache: Dict[str, object] = {}

    def __call__(self, texts: List[str]):
        import numpy as np

        todo = sorted({t for t in texts if t not in self.cache})
        for i in range(0, len(todo), 4096):
            chunk = todo[i:i + 4096]
            vecs = self.encoder.encode(chunk, normalize_embeddings=True, batch_size=128)
            self.cache.update(zip(chunk, np.asarray(vecs, dtype=np.float32)))
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 384), dtype=np.float32)


def rewrite_score(rewrite: str, question: str) -> int:
    """The plan check's rewrite alignment (0-100) with the real base encoder."""
    from app.security import alignment, signals

    score = signals.cos100(alignment.encode(rewrite), alignment.encode(question))
    return 100 if score is None else score


def build(raw: Path, seed: int, embed) -> tuple:
    from app.security.thresholds import DEFAULT_THRESHOLDS

    ctx, q = pairs.Ctx(raw, seed, embed), QUOTAS
    gen, review = pairs.gen_rows(ctx, pairs.read_gen(OUT / "generated" / "webagent_gen.jsonl"), rewrite_score,
                                 DEFAULT_THRESHOLDS["rewrite_at"])
    aligned = (ctx.take(gen, q["webagent_gen"]) + pairs.asq_rows(ctx, ctx.read("asq"), ASQ_SOURCES, q["asq"])
               + sum((pairs.function_rows(ctx, n, ctx.read(n), q[n]) for n in ("glaive", "toolace", "xlam")), []))
    saroku: List[dict] = []
    if agent.saroku_passed(OUT / "QUALITY.md"):
        saroku = agent.saroku_rows(ctx, q["saroku_safe"], q["saroku_unsafe"])
    else:
        print(json.dumps({"warning": "saroku skipped: QUALITY.md has no passed verdict"}), file=sys.stderr)
    aligned += [r for r in saroku if r["labels"] == {"aligned": "yes", "risk": "none"}]
    rows = aligned + [r for r in saroku if r["labels"] != {"aligned": "yes", "risk": "none"}]
    rows += agent.repair_rows(ctx, aligned, q["repair_easy"], q["repair_hard"])
    rows += pairs.injecagent_rows(ctx, q["injecagent"]) + pairs.tasktracker_rows(ctx, q["tasktracker"])
    rows += pairs.xstest_rows(ctx, q["xstest_contrast"], q["xstest_lookalike"], q["notinject"])
    pool = benign_pool(ctx, aligned)
    aegis = [{"text": r["text"], "source": "aegis"} for r in ctx.read("aegis") if r.get("label") == "unsafe"
             and len(r["text"]) < 2000 and not JAILBREAK_HINTS.search(r["text"]) and aegis_group(r.get("categories")) == "harm"]
    rows += pairs.harm_rows(ctx, aegis, pool, q["aegis"])
    rows += pairs.harm_rows(ctx, [{"text": r["text"], "source": "harmbench"} for r in ctx.read("harmbench")], pool, q["harmbench"])
    rows += agent.asb_rows(ctx, q["asb_risky"], q["asb_twin"]) + agent.toolemu_rows(ctx, q["toolemu"])
    rows += agent.redcode_rows(ctx, q["redcode"]) + agent.privacylens_rows(ctx, q["privacylens"])
    rows += (evals.agentdojo_rows(ctx, q["agentdojo"]) + evals.agentharm_rows(ctx) + evals.shade_rows(ctx)
             + evals.osharm_rows(ctx) + evals.tracesafe_rows(ctx) + evals.agentdam_rows(ctx))
    rows = dedupe(ctx, [r for r in rows if r])
    assign_clusters(ctx, rows, embed)
    splits: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if not r.pop("_drop", False):
            splits[r["split"]].append(r)
    for s in splits.values():
        ctx.rng.shuffle(s)
    return splits, {"dropped": dict(ctx.dropped), "review_rewrite_below_rewrite_at": review}


def benign_pool(ctx: pairs.Ctx, aligned: List[dict]) -> List[dict]:
    """Benign questions that can be an intent for a harmful call: aligned rows' intents (committed
    sources only) and Aegis 2.0 safe prompts."""
    seen, pool = set(), []
    for r in aligned:
        if not is_local(r) and r["intent"] not in seen:
            seen.add(r["intent"])
            pool.append({"text": r["intent"], "groups": r["groups"], "sources": r["sources"], "split": r["split"]})
    for r in ctx.read("aegis"):
        t = r["text"]
        if r.get("label") == "safe" and len(t) < 2000 and not JAILBREAK_HINTS.search(t) and t not in seen:
            seen.add(t)
            pool.append(pairs.item(ctx, t, [f"aegis-safe:{pairs.h(t)}"], ["aegis"]))
    return pool


def dedupe(ctx: pairs.Ctx, rows: List[dict]) -> List[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r["intent"], r["call"], r["action"])
        if key in seen:
            ctx.dropped["duplicate"] += 1
            continue
        seen.add(key)
        out.append(r)
    return out


def assign_clusters(ctx: pairs.Ctx, rows: List[dict], embed) -> None:
    """Rows whose intents and calls both have cosine >= 0.9 form one cluster, kept in one split: the
    cluster's most common split; members of other splits are dropped, never moved (their groups decide
    their split). Each row gets its cluster id."""
    import numpy as np

    ei, ec = embed([r["intent"] for r in rows]), embed([r["call"] for r in rows])
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(rows), 1024):
        sims = ec[start:start + 1024] @ ec.T
        for bi, j in zip(*np.nonzero(sims >= NEAR_DUPLICATE)):
            i = start + int(bi)
            if i < j and float(ei[i] @ ei[j]) >= NEAR_DUPLICATE:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[rj] = ri
    members: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    for root, idx in members.items():
        votes = Counter(rows[i]["split"] for i in idx)
        split = max(SPLITS, key=lambda s: (votes[s], s == "test"))
        cid = pairs.h(rows[min(idx)]["intent"] + "\n" + rows[min(idx)]["call"])
        for i in idx:
            rows[i]["cluster"] = cid
            if rows[i]["split"] != split:
                rows[i]["_drop"] = True
                ctx.dropped["near_duplicate_in_other_split"] += 1


def write(out: Path, splits: Dict[str, List[dict]], info: dict) -> dict:
    (out / "local").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = splits.get(split, [])
        for folder, keep in ((out, False), (out / "local", True)):
            with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    if is_local(r) == keep:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {**counts(splits), **info, "quotas": QUOTAS}
    (out / "counts.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, help="scratch folder from fetch_alignment_sources.py")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--check-data", action="store_true", help="validate the built files and exit")
    args = ap.parse_args()
    embed = Embedder()
    if args.check_data:
        errors = check_data(args.out, embed, args.out / "README.md")
        print(json.dumps({"check_data": "failed" if errors else "passed", "errors": errors}, indent=1))
        return 1 if errors else 0
    if args.raw is None:
        ap.error("--raw is required to build")
    splits, info = build(args.raw, args.seed, embed)
    manifest = write(args.out, splits, info)
    print(json.dumps({"by_split": manifest["by_split"], "dropped": info["dropped"],
                      "review": len(info["review_rewrite_below_rewrite_at"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
