#!/usr/bin/env python3
"""Checkpoint 2's time per call with the real models (checkpoint2-intent SNFR-10, task A11).

Runs `guard_tool` (context with the intent signals, Cedar decision, metrics, collection) for search
calls under realistic intents, on one thread, and reports median and p95. With --busy N, N other
processes keep scoring text with the same models at the same time, as busy workers would.

    python scripts/bench_checkpoint2.py --calls 300 [--busy 2]

Run it where the numbers matter: in the backend container, on the target CPU.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("WEBAGENT_GUARD_COLLECT", "false")  # benchmark calls are not collection data
BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

INTENTS = [
    "compare postgres and mysql for a write-heavy app",
    "what changed in the latest python release",
    "how do heat pumps work in cold climates",
    "summarise recent research on sleep and memory",
]
QUERIES = [
    "postgres vs mysql write performance benchmarks",
    "python 3.13 release notes new features",
    "heat pump efficiency below freezing",
    "sleep memory consolidation studies 2025",
    "best hiking trails in the alps",
]


def _burn(stop_at: float) -> None:
    from app.security import signals
    from app.models.task import WorkflowState

    state = WorkflowState(query=INTENTS[0], context={"task_id": "bench-busy"})
    i = 0
    while time.time() < stop_at:
        signals.score_text(state, f"{QUERIES[i % len(QUERIES)]} {i}")
        i += 1


async def _run(calls: int, per_task: int = 3) -> tuple:
    """Times calls in tasks of `per_task` calls each, after the plan check's warm-up of the intent
    (as in a real task); the first call of a task without that warm-up is timed separately."""
    import logging

    logging.disable(logging.INFO)
    from app.models.task import WorkflowState
    from app.security import alignment, intent, signals
    from app.security.tool_guard import guard_tool

    warm, cold = [], []
    for t in range((calls + 10) // per_task + 1):
        state = WorkflowState(query=f"{INTENTS[t % len(INTENTS)]} ({t})", context={
            "task_id": f"bench-{t}", "research_plan": {"requires_web_search": True}})
        intent.record_intent(state)
        if t % 2:  # odd tasks: as after the plan check, the intent is already scored
            signals.intent_embedding(state)
            signals.intent_harm(state)
            alignment.task_embedding(state)
        for j in range(per_task):
            query = f"{QUERIES[(t + j) % len(QUERIES)]} {t}-{j}"  # unique text: no score cache hits
            started = time.perf_counter()
            await guard_tool("tool::web_search", {"query": query}, state, principal="web_research")
            ms = (time.perf_counter() - started) * 1000
            if t < 4:
                continue  # the first tasks load and warm the models
            (cold if (j == 0 and t % 2 == 0) else warm).append(ms)
    return warm, cold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=int, default=300)
    ap.add_argument("--busy", type=int, default=0, help="other processes scoring at the same time")
    args = ap.parse_args()
    busy = []
    if args.busy:
        ctx = mp.get_context("spawn")
        stop_at = time.time() + 600
        busy = [ctx.Process(target=_burn, args=(stop_at,), daemon=True) for _ in range(args.busy)]
        for p in busy:
            p.start()
        time.sleep(20)  # let them load their models and start scoring
    warm, cold = asyncio.run(_run(args.calls))
    for p in busy:
        p.terminate()
    warm, cold = sorted(warm), sorted(cold)
    report = {"calls": len(warm), "busy_processes": args.busy, "cpus": os.cpu_count(),
              "median_ms": round(statistics.median(warm), 1), "p95_ms": round(warm[int(0.95 * len(warm))], 1),
              "first_call_unwarmed": {"calls": len(cold), "median_ms": round(statistics.median(cold), 1),
                                      "p95_ms": round(cold[int(0.95 * len(cold))], 1)}}
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
