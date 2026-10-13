#!/usr/bin/env python3
"""Summarise the security guard's collected decisions for the shadow-week review (counts only).

Run from backend/:
    python scripts/review_security_collection.py [--dir data/security/collected] [--since 2026-10-14]
        [--sheet review.csv]            # untrusted items the guard would have acted on, for labelling
        [--labels review_labelled.csv]  # the same sheet with `label` filled in: injection / benign

The summary never contains text: sources, outcomes, policy ids, scores, alignment. The sheet holds
web-result URLs (public pages, so a reviewer can open them) and scores, never page text.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BACKEND = Path(__file__).resolve().parents[1]
UNTRUSTED = {"web_result", "retrieved_doc", "document", "tool_output", "unlabelled"}
ACTING = {"decline", "escalate", "consent"}
SHEET_FIELDS = ["ts", "task_id", "source", "ref", "outcome", "policy_ids", "injection", "harm",
                "sensitive_data", "jailbreak", "pattern_hits", "label"]


def load_rows(directory: Path, since: Optional[str]) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        if since and path.stem < since:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def percentiles(values: List[float], points=(10, 25, 50, 75, 90, 99)) -> Dict[str, float]:
    if not values:
        return {}
    v = sorted(values)
    return {f"p{p}": v[min(len(v) - 1, int(p / 100 * len(v)))] for p in points}


def wilson_upper(hits: int, n: int, z: float = 1.645) -> Optional[float]:
    if n == 0:
        return None
    p = hits / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(min(1.0, (centre + spread) / (1 + z * z / n)), 4)


def summarise(rows: Iterable[dict]) -> Dict[str, Any]:
    rows = list(rows)
    out: Dict[str, Any] = {"rows": len(rows), "days": sorted({r.get("ts", "")[:10] for r in rows}),
                           "tasks": len({r.get("task_id") for r in rows})}
    cp1 = [r for r in rows if r.get("checkpoint") == 1 and r.get("item")]
    by_source: Dict[str, Counter] = defaultdict(Counter)
    inj: Dict[str, List[float]] = defaultdict(list)
    policies: Counter = Counter()
    for r in cp1:
        src = r["item"]["source"]
        by_source[src]["items"] += 1
        by_source[src][r["outcome"]] += 1
        if src in UNTRUSTED:
            inj[src].append(r["item"]["axes"].get("injection", 0))
        for p in r.get("policy_ids", []):
            if p != "baseline-permit":
                policies[p] += 1
    out["checkpoint1"] = {
        "items_by_source": {s: dict(c) for s, c in sorted(by_source.items())},
        "would_act_rate_by_source": {s: round(sum(c[o] for o in ACTING) / c["items"], 4)
                                     for s, c in sorted(by_source.items()) if c["items"]},
        "untrusted_injection_score_percentiles": {s: percentiles(v) for s, v in sorted(inj.items())},
        "policies_matched": dict(policies.most_common()),
        "scores_unavailable": sum(1 for r in cp1 if not r["item"].get("scores_available", True)),
    }
    cp2 = [r for r in rows if r.get("checkpoint") == 2]
    by_action: Dict[str, Counter] = defaultdict(Counter)
    align_untainted: List[float] = []
    shadow: Counter = Counter()
    for r in cp2:
        by_action[r.get("action", "?")][r["outcome"]] += 1
        ctx = r.get("context") or {}
        if r.get("action") in ("tool::web_search", "tool::internal_retrieve") and not ctx.get("tainted") \
                and ctx.get("alignment") is not None:
            align_untainted.append(ctx["alignment"])
        for p in r.get("shadow_policy_ids", []):
            shadow[p] += 1
    out["checkpoint2"] = {
        "outcomes_by_action": {a: dict(c) for a, c in sorted(by_action.items())},
        "untainted_search_alignment": {
            "n": len(align_untainted), **percentiles(align_untainted),
            **{f"share_below_{c}": round(sum(a < c for a in align_untainted) / len(align_untainted), 4)
               for c in (10, 20, 30, 40, 50, 60) if align_untainted},
        },
        "shadow_policy_matches": dict(shadow),
    }
    return out


def sheet_rows(rows: Iterable[dict]) -> List[dict]:
    """Untrusted items the guard would have acted on (or watched), for a reviewer to label."""
    out = []
    for r in rows:
        item = r.get("item")
        if r.get("checkpoint") != 1 or not item or item["source"] not in UNTRUSTED:
            continue
        if r["outcome"] == "allow":
            continue
        axes = item.get("axes", {})
        out.append({"ts": r.get("ts", "")[:19], "task_id": r.get("task_id"), "source": item["source"],
                    "ref": item.get("ref", ""), "outcome": r["outcome"], "policy_ids": ";".join(r.get("policy_ids", [])),
                    "injection": axes.get("injection"), "harm": axes.get("harm"),
                    "sensitive_data": axes.get("sensitive_data"), "jailbreak": axes.get("jailbreak"),
                    "pattern_hits": ";".join(item.get("pattern_hits", [])), "label": ""})
    return out


def labelled_estimate(labelled: List[dict], rows: List[dict]) -> Dict[str, Any]:
    """False-positive estimate from the labelled sheet: benign items the guard would have dropped,
    over all untrusted items it judged (the denominator comes from the collection)."""
    judged = sum(1 for r in rows if r.get("checkpoint") == 1 and r.get("item")
                 and r["item"]["source"] in UNTRUSTED)
    labels = Counter((r.get("label") or "").strip().lower() for r in labelled)
    dropped_benign = sum(1 for r in labelled if (r.get("label") or "").strip().lower() == "benign"
                         and r.get("outcome") in ACTING)
    return {
        "untrusted_items_judged": judged,
        "sheet_labels": dict(labels),
        "benign_items_that_would_be_dropped": dropped_benign,
        "false_positive_rate": round(dropped_benign / judged, 4) if judged else None,
        "false_positive_rate_upper_95": wilson_upper(dropped_benign, judged),
        "unlabelled_rows": labels.get("", 0),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=BACKEND / "data" / "security" / "collected")
    ap.add_argument("--since", default=None, help="first day to include (YYYY-MM-DD)")
    ap.add_argument("--sheet", type=Path, default=None, help="write the labelling sheet (CSV)")
    ap.add_argument("--labels", type=Path, default=None, help="labelled sheet to estimate false positives")
    args = ap.parse_args(argv)
    rows = load_rows(args.dir, args.since)
    report = summarise(rows)
    if args.sheet:
        sheet = sheet_rows(rows)
        with args.sheet.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
            writer.writeheader()
            writer.writerows(sheet)
        report["sheet"] = {"path": str(args.sheet), "rows": len(sheet)}
    if args.labels:
        with args.labels.open(encoding="utf-8") as f:
            report["labelled"] = labelled_estimate(list(csv.DictReader(f)), rows)
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
