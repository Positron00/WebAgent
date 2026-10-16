"""Pairing for the evaluation-only (role E) sources (model-and-data design §2): test and probe only.

AgentHarm, SHADE-Arena, OS-Harm and TraceSafe rows are local-only (`local/`, git-ignored): the
repository is public and these are gated or ask not to be trained on. AgentDAM (CC BY-NC) is test
only. Nothing here prints text.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from alignment_pairs import Ctx, h

# TraceSafe mutation files -> (aligned, risk). Benign traces are the baseline; injected steps don't
# serve the user; leaks serve the task but expose data; interface and hallucination faults are on
# task with a lower risk.
TRACESAFE_LABELS = {
    "benign": ("yes", "none"),
    "PromptInjectionIn": ("no", "high"), "PromptInjectionOut": ("no", "high"),
    "UserInfoLeak": ("yes", "high"), "ApiKeyLeak": ("yes", "high"), "DataLeak": ("yes", "high"),
}


def agentdojo_rows(ctx: Ctx, n: int) -> List[dict]:
    """The user task as the intent and an injection task's goal as the call (aligned no, risk high)."""
    tasks: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in ctx.read("agentdojo_tasks"):
        tasks[r["suite"]][r["kind"]].append(r)
    out = []
    for suite, by in tasks.items():
        for goal in by["injection"]:
            for user in by["user"]:
                out.append(ctx.row(user["text"], goal["text"], "function", "no", "high", "agentdojo:injection_goal",
                                   [f"agentdojo:{suite}:{goal['task']}"], ["agentdojo"]))
    return ctx.take(out, n)


def agentharm_rows(ctx: Ctx) -> List[dict]:
    """The benign counterpart as the intent and the harmful one as the call (aligned no, risk high)."""
    return ctx.take([ctx.row(r["benign"], r["harmful"], "plan", "no", "high", "agentharm:harmful",
                             [f"agentharm:{r['group']}"], ["agentharm"]) for r in ctx.read("agentharm")], 10**9)


def shade_rows(ctx: Ctx) -> List[dict]:
    """The main task as the intent and the hidden side task as the call (aligned no, risk high)."""
    return ctx.take([ctx.row(r["main"], r["side"], "plan", "no", "high", "shade-arena:side_task",
                             [f"shade:{r['task']}"], ["shade-arena"]) for r in ctx.read("shade")], 10**9)


def osharm_rows(ctx: Ctx) -> List[dict]:
    """Per category: injection goals under the task (no, high); the misbehaviour a task invites, on
    topic but risky (yes, high); misuse requests under a benign task of the same app (no, high)."""
    rows = ctx.read("osharm")
    benign: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] != "misuse":
            benign[r["app"]].append(r)
    out = []
    for r in rows:
        key = f"osharm:{r['app']}"
        if r["category"] == "injection":
            out += [ctx.row(r["instruction"], g, "plan", "no", "high", "os-harm:injection", [key], ["os-harm"]) for g in r["goals"]]
        elif r["category"] == "misbehavior":
            out.append(ctx.row(r["instruction"], r["explanation"], "plan", "yes", "high", "os-harm:misbehavior", [key], ["os-harm"]))
        elif benign.get(r["app"]):
            task = benign[r["app"]][int(h(r["id"]), 16) % len(benign[r["app"]])]
            out.append(ctx.row(task["instruction"], r["instruction"], "plan", "no", "high", "os-harm:misuse", [key], ["os-harm"]))
    return [r for r in out if r]


def tracesafe_rows(ctx: Ctx) -> List[dict]:
    """The trace's user turn and the (mutated) step, labelled per mutation file; other mutations
    (argument and interface faults) are on task with a low risk."""
    out = []
    for r in ctx.read("tracesafe"):
        if not r.get("user") or not r.get("step"):
            ctx.dropped["tracesafe_unparsed"] += 1
            continue
        aligned, risk = TRACESAFE_LABELS.get(r["label"], ("yes", "low"))
        out.append(ctx.row(r["user"], r["step"], "function", aligned, risk, f"tracesafe:{r['label']}",
                           [f"tracesafe:{h(r['user'])}"], ["tracesafe"]))
    return [r for r in out if r]


def agentdam_rows(ctx: Ctx) -> List[dict]:
    """The task, and a typed action carrying data the task didn't need (yes, high) or the content it
    must include (yes, none). Test only (CC BY-NC 4.0)."""
    out = []
    for r in ctx.read("agentdam"):
        key = [f"agentdam:{r['uid']}"]
        out += [ctx.row(r["intent"], f"type: {s}", "function", "yes", "high", "agentdam:unneeded_data", key, ["agentdam"], split="test")
                for s in r["sensitive"]]
        out += [ctx.row(r["intent"], f"type: {s}", "function", "yes", "none", "agentdam:required", key, ["agentdam"], split="test")
                for s in r["required"]]
    return [r for r in out if r]
