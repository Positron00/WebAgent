#!/usr/bin/env python3
"""Evaluation-only suites through checkpoint 2's intent signals and rules (checkpoint2-intent SEC-131).

Runs the role-E rows of the alignment test and probe sets (AgentDojo, AgentHarm, SHADE-Arena, OS-Harm,
TraceSafe, AgentDAM; the no-train and gated ones live in the git-ignored local/ files) through the
signals checkpoint 2 uses now (the classifier when a gated one is installed, else phase A's encoder and
ContextDefender) and the Cedar engine, with the shadow-only rules counted as acting. Prints counts per
source and label only, never row text.

    python scripts/eval_alignment_suites.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("WEBAGENT_GUARD_COLLECT", "false")  # evaluation rows are not collection data
BACKEND = Path(__file__).resolve().parents[1]
for p in (BACKEND, BACKEND / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from alignment_training import DATA_DIR, load_split  # noqa: E402

ACTIONS = {"web_search": "tool::web_search", "internal_retrieve": "tool::internal_retrieve",
           "execute_code": "tool::execute_code", "function": "tool::web_search"}


def held(row: dict) -> tuple:
    """(held, signals source) for one row: any acting or shadow-only rule matched."""
    from app.models.task import WorkflowState
    from app.security import intent, signals
    from app.security.decision import decide
    from app.security.tool_guard import _build_context

    state = WorkflowState(query=row["intent"], context={"task_id": "suite", "research_plan": {}})
    intent.record_intent(state)
    if row.get("action") == "plan":
        sig = signals.score_text(state, row["call"], action="plan")
        ctx = {"source": "plan", "subgoal_origin": "follow_up", "signals_available": sig.available,
               "intent_alignment": sig.intent_alignment, "subgoal_harm": signals.call_risk(state, sig)
               if sig.classifier else sig.harm, "intent_harm": signals.intent_harm(state)}
        decision = decide("senior_research", "plan::subgoal", "suite", ctx, shadow=False)
        source = "classifier" if sig.classifier else "phase_a"
    else:
        action = ACTIONS.get(row.get("action"), "tool::web_search")
        args = {"code": row["call"]} if action == "tool::execute_code" else {"query": row["call"]}
        ctx = _build_context(action, args, state, tool_mode="act", scored_text=row["call"])
        decision = decide("agent", action, "suite", ctx, shadow=False)
        source = "classifier" if ctx.get("classifier_available") else "phase_a"
    matched = [p for p in (*decision.policy_ids, *decision.shadow_policy_ids) if p != "baseline-permit"]
    return bool(matched), source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    import logging

    logging.disable(logging.INFO)
    counts = defaultdict(lambda: {"rows": 0, "held": 0})
    sources = set()
    for split in ("test", "probe"):
        for row in load_split(split, args.data):
            if row.get("role") != "E":
                continue
            is_held, source = held(row)
            sources.add(source)
            key = f"{row['origin'].split(':')[0]} | aligned={row['labels']['aligned']} risk={row['labels']['risk']}"
            counts[key]["rows"] += 1
            counts[key]["held"] += int(is_held)
    report = {"signals": sorted(sources), "suites": {k: {**v, "held_share": round(v["held"] / v["rows"], 3)}
                                                     for k, v in sorted(counts.items())}}
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
