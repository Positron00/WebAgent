"""Pairing for agent-safety sources and the off-intent re-pairs (model-and-data design §2), role T.

Source filters required by SEC-121 live here as plain functions so the tests can check them.
Nothing here prints text.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from alignment_pairs import Ctx, h

# Agent-SafetyBench cases derived from AdvBench, AgentDojo or InjecAgent (SEC-121). The release has no
# per-case source field, so derived cases are dropped by what marks them: AdvBench's 200 cases are the
# content-only "Produce unsafe information" category (all 250 dropped); injection cases (AgentDojo,
# InjecAgent) are failure mode M9, "trusts tool results without validation" (all dropped); and any
# case in an AgentDojo suite environment (banking, Slack, travel, workspace: email, calendar, drive).
ASB_ADVBENCH_RISK = "Produce unsafe information"
ASB_INJECTION_MODE = 9
ASB_AGENTDOJO_ENVS = {"Bank", "Banking", "Slack", "Travel", "Calendar", "Email", "CloudDrive", "Drive", "Workspace"}
# saroku (QUALITY.md): only Gemini rows with a goal, outside the 13 long-tail domains of an unnamed
# external corpus; corrigibility not used.
SAROKU_LONG_TAIL = {"git", "kubernetes", "secrets", "ops", "iam", "storage", "deploy", "capacity", "cache",
                    "feature_flags", "support", "data_platform", "injection"}


def asb_keep(case: dict) -> Tuple[bool, str]:
    if any(ASB_ADVBENCH_RISK in r for r in case.get("risks") or []):
        return False, "advbench"
    if not case.get("envs"):
        return False, "no_environment"
    if ASB_INJECTION_MODE in (case.get("failure_modes") or []):
        return False, "injection_derived"
    if set(case["envs"]) & ASB_AGENTDOJO_ENVS:
        return False, "agentdojo_environment"
    return True, "kept"


def saroku_passed(quality_md: Path) -> bool:
    text = quality_md.read_text(encoding="utf-8") if quality_md.is_file() else ""
    return re.search(r"^\*\*Verdict: PASSED\*\*", text, re.M) is not None


def saroku_labels(row: dict) -> Optional[Tuple[str, str]]:
    """QUALITY.md's mapping to (aligned, risk); None for rows that aren't used."""
    if not (row.get("goal") or "").strip() or row.get("source") != "gemini" or row.get("domain") in SAROKU_LONG_TAIL:
        return None
    if row.get("is_safe"):
        return "yes", "none"
    sev = "high" if row.get("severity") == "high" else "low"
    return {"goal_drift": ("no", sev), "scope_violation": ("no", sev), "injection": ("no", "high"),
            "policy_violation": ("yes", sev)}.get(row.get("property") or "")


def asb_rows(ctx: Ctx, n_risky: int, n_twin: int) -> List[dict]:
    """A fulfillable case's instruction as the intent; an unfulfillable case of the same environment is
    the risky call (aligned no, risk high), another fulfillable case its benign twin (yes, none)."""
    envs: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for case in ctx.read("agent_safetybench"):
        ok, why = asb_keep(case)
        if not ok:
            ctx.dropped[f"agent_safetybench_{why}"] += 1
        else:
            envs[case["envs"][0]][case["fulfillable"]].append(case["instruction"])
    risky, twin = [], []
    for env, by in envs.items():
        good, bad, key = by.get(1, []), by.get(0, []), [f"asb:env:{env}"]
        for i, call in enumerate(bad if good else []):
            risky.append(ctx.row(good[i % len(good)], call, "plan", "no", "high", "agent-safetybench:unfulfillable", key, ["agent-safetybench"]))
        for i, call in enumerate(good if len(good) > 1 else []):
            twin.append(ctx.row(good[i - 1], call, "plan", "yes", "none", "agent-safetybench:twin", key, ["agent-safetybench"]))
    return ctx.take(risky, n_risky) + ctx.take(twin, n_twin)


def toolemu_rows(ctx: Ctx, n: int) -> List[dict]:
    """The case's instruction and each potential risky action: on topic but risky (yes, high)."""
    out = [ctx.row(c["instruction"], a, "function", "yes", "high", "toolemu:risky_action",
                   [f"toolemu:tk:{t}" for t in c["toolkits"]], ["toolemu"])
           for c in ctx.read("toolemu") for a in c["risky_actions"]]
    return ctx.take(out, n)


def redcode_rows(ctx: Ctx, n: int) -> List[dict]:
    """RedCode-Exec code (parsed only, never run) under the nearest benign coding question by topic
    (WildChat coding questions until our generated code rows exist); aligned no, risk high."""
    cases = ctx.take([c for c in ctx.read("redcode") if len(c["code"]) <= 1000], n)
    pool = [{"text": q["question"], "groups": [f"wildchat:{h(q['question'])}"], "sources": ["wildchat"],
             "split": ctx.split_of(f"wildchat:{h(q['question'])}")} for q in ctx.read("wildchat_code")]
    splits = [ctx.split_of(f"redcode:{c['file']}") for c in cases]
    near = ctx.nearest([c["summary"] or c["code"] for c in cases], splits, pool)
    out = []
    for c, idx in zip(cases, near):
        if idx:
            p = pool[idx[0]]
            out.append(ctx.row(p["text"], c["code"], "execute_code", "no", "high", "redcode:exec",
                               [f"redcode:{c['file']}", *p["groups"]], ["redcode", "wildchat"]))
    return [r for r in out if r]


def saroku_rows(ctx: Ctx, n_safe: int, n_unsafe: int) -> List[dict]:
    safe, unsafe = [], []
    for r in ctx.read("saroku"):
        lab = saroku_labels(r)
        if lab is None:
            ctx.dropped["saroku_not_used"] += 1
            continue
        row = ctx.row(r["goal"], r["action"], "function", *lab, f"saroku:{r.get('property') or 'safe'}",
                      [f"saroku:batch:{r['batch']}"], ["saroku"])
        (safe if lab == ("yes", "none") else unsafe).append(row)
    return ctx.take(safe, n_safe) + ctx.take(unsafe, n_unsafe)


def privacylens_rows(ctx: Ctx, n: int) -> List[dict]:
    """The user instruction, and the final action carrying a sensitive item: on task, but it leaks
    what the task didn't need (aligned yes, risk high)."""
    out = [ctx.row(c["instruction"], f"{c['final_action']}: {item}", "function", "yes", "high", "privacylens:leak",
                   [f"privacylens:{c['name']}"], ["privacylens"])
           for c in ctx.read("privacylens") for item in c["items"]]
    return ctx.take(out, n)


def repair_rows(ctx: Ctx, aligned: List[dict], n_easy: int, n_hard: int) -> List[dict]:
    """Off-intent calls: an aligned call under another question of the same split, random (easy) or the
    nearest by MiniLM that isn't a paraphrase (cosine < 0.8) and shares no group (hard); aligned no."""
    by_split: Dict[str, List[dict]] = defaultdict(list)
    for r in aligned:
        by_split[r["split"]].append(r)
    easy = []
    for r in ctx.take(list(aligned), n_easy):
        other = by_split[r["split"]][ctx.rng.randrange(len(by_split[r["split"]]))]
        if not set(other["groups"]) & set(r["groups"]):
            easy.append(_repair(ctx, other, r, "repair_easy"))
    calls = ctx.take(list(aligned), n_hard)
    intents = {}
    for r in aligned:
        intents.setdefault(r["intent"], r)
    pool = [{"text": t, "split": r["split"], "row": r} for t, r in intents.items()]
    near = ctx.nearest([r["intent"] for r in calls], [r["split"] for r in calls], pool, k=10, max_cos=0.8)
    hard = []
    for r, idx in zip(calls, near):
        other = next((pool[i]["row"] for i in idx if not set(pool[i]["row"]["groups"]) & set(r["groups"])), None)
        if other is not None:
            hard.append(_repair(ctx, other, r, "repair_hard"))
    return [r for r in easy + hard if r]


def _repair(ctx: Ctx, intent_row: dict, call_row: dict, kind: str) -> Optional[dict]:
    src = call_row["origin"].split(":", 1)[0]
    return ctx.row(intent_row["intent"], call_row["call"], call_row["action"], "no", "none", f"{src}:{kind}",
                   intent_row["groups"] + call_row["groups"], intent_row["sources"] + call_row["sources"])
