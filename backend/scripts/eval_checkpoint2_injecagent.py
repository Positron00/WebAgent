#!/usr/bin/env python3
"""What checkpoint 2 does when an undetected injection makes the agent search (SEC-040–SEC-047).

InjecAgent's attacks target tools WebAgent doesn't have (smart locks, bank transfers, email), so the
nearest WebAgent action is a web search carrying the attacker's instruction (direct harm) or the data
it asks for and where to send it (data stealing). For each InjecAgent case this simulates
`tool::web_search` with the attacker instruction as the query and the case's user instruction as the
task, and runs checkpoint 2's real context builder, alignment score and policies, with the task
untainted (checkpoint 1 missed the injection) and tainted (it flagged it). Local only; prints counts.

Run from backend/ after fetch_security_sources.py:
    python scripts/eval_checkpoint2_injecagent.py --raw <raw dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.task import WorkflowState  # noqa: E402
from app.security import ledger  # noqa: E402
from app.security.decision import decide  # noqa: E402
from app.security.tool_guard import _build_context  # noqa: E402

ALIGNMENT_CUTS = (40, 50, 60, 70)


def evaluate(cases: List[dict]) -> Dict[str, Any]:
    outcomes: Dict[str, Counter] = defaultdict(Counter)
    alignments: Dict[str, List[int]] = defaultdict(list)
    hits: Counter = Counter()
    for i, case in enumerate(cases):
        kind = case["kind"]
        for tainted in (False, True):
            state = WorkflowState(query=case["user_instruction"], context={"task_id": f"injecagent-{i}"})
            if tainted:
                ledger.get_ledger(state)["tainted"] = True
            ctx = _build_context("tool::web_search", {"query": case["attack"]}, state, tool_mode="act")
            decision = decide("web_research", "tool::web_search", f"injecagent-{i}", ctx, shadow=False)
            outcomes[f"{kind}:{'tainted' if tainted else 'untainted'}"][decision.outcome] += 1
            if not tainted:
                alignments[kind].append(ctx["alignment"])
                for h in ctx["pattern_hits"]:
                    hits[f"{kind}:{h}"] += 1
    report: Dict[str, Any] = {"cases": Counter(c["kind"] for c in cases), "outcomes": {}, "alignment": {}}
    for key, counter in sorted(outcomes.items()):
        n = sum(counter.values())
        report["outcomes"][key] = {k: round(v / n, 3) for k, v in sorted(counter.items())}
    for kind, values in alignments.items():
        values = sorted(values)
        report["alignment"][kind] = {
            "median": values[len(values) // 2],
            **{f"share_below_{c}": round(sum(v < c for v in values) / len(values), 3) for c in ALIGNMENT_CUTS},
        }
    report["pattern_hits"] = dict(hits)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cases = [r for r in (json.loads(line) for line in (args.raw / "injecagent.jsonl").open(encoding="utf-8"))
             if r["label"] == "injection" and r.get("user_instruction")]
    report = evaluate(cases)
    text = json.dumps(report, indent=2, default=dict)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
