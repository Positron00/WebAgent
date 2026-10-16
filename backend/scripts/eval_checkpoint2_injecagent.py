#!/usr/bin/env python3
"""What checkpoint 2 does when an undetected injection makes the agent search (SEC-040–SEC-047).

InjecAgent's attacks target tools WebAgent doesn't have (smart locks, bank transfers, email), so the
nearest WebAgent action is a web search carrying the attacker's instruction (direct harm) or the data
it asks for and where to send it (data stealing). For each InjecAgent case this simulates
`tool::web_search` with the attacker instruction as the query and the case's user instruction as the
task, and runs checkpoint 2's real context builder, alignment score and policies, with the task
untainted (checkpoint 1 missed the injection) and tainted (it flagged it). Local only; prints counts.

Run from backend/ after fetch_security_sources.py:
    python scripts/eval_checkpoint2_injecagent.py --raw <raw dir> [--json out.json] [--untainted]

With --untainted (checkpoint2-intent SEC-131) the cases also run with the intent record, and the report
adds, for untainted tasks, the share held if the shadow-only intent rules acted, which rules would act,
and the intent-alignment and risk-above-intent distributions.
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
from app.security import intent, ledger  # noqa: E402
from app.security.decision import decide  # noqa: E402
from app.security.tool_guard import _build_context  # noqa: E402

ALIGNMENT_CUTS = (40, 50, 60, 70)


def evaluate_untainted(cases: List[dict]) -> Dict[str, Any]:
    """Untainted tasks with the intent signals: held if any rule, acting or shadow-only, matched."""
    held: Counter = Counter()
    rules: Dict[str, Counter] = defaultdict(Counter)
    scores: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for i, case in enumerate(cases):
        kind = case["kind"]
        state = WorkflowState(query=case["user_instruction"], context={"task_id": f"injecagent-u{i}"})
        intent.record_intent(state)
        ctx = _build_context("tool::web_search", {"query": case["attack"]}, state, tool_mode="act")
        decision = decide("web_research", "tool::web_search", f"injecagent-u{i}", ctx, shadow=False)
        matched = [p for p in (*decision.policy_ids, *decision.shadow_policy_ids) if p != "baseline-permit"]
        held[kind] += bool(matched)
        for p in matched:
            rules[kind][p] += 1
        scores[kind]["intent_alignment"].append(int(ctx.get("intent_alignment", 100)))
        scores[kind]["risk_above_intent"].append(int(ctx.get("risk_above_intent", 0)))
    total = Counter(c["kind"] for c in cases)
    out: Dict[str, Any] = {}
    for kind, n in total.items():
        dist = {}
        for name, values in scores[kind].items():
            values = sorted(values)
            dist[name] = {"median": values[len(values) // 2], "p10": values[len(values) // 10],
                          "p90": values[(9 * len(values)) // 10]}
        out[kind] = {"held_if_intent_rules_acted": round(held[kind] / n, 3),
                     "rules": dict(rules[kind].most_common()), **dist}
    return out


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
    ap.add_argument("--untainted", action="store_true", help="add the intent-rule results (SEC-131)")
    args = ap.parse_args()
    cases = [r for r in (json.loads(line) for line in (args.raw / "injecagent.jsonl").open(encoding="utf-8"))
             if r["label"] == "injection" and r.get("user_instruction")]
    report = evaluate(cases)
    if args.untainted:
        report["untainted_with_intent_rules"] = evaluate_untainted(cases)
    text = json.dumps(report, indent=2, default=dict)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
