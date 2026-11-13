#!/usr/bin/env python3
"""One report for the end of a shadow week: every number the week has to decide (counts only).

Run from backend/, on a copy of the week's data folder (or the folder itself, read-only):

    python scripts/shadow_week_report.py --data ~/webagent-uat-data-3.10.0 [--since 2026-10-17]
        [--labels review_labelled.csv]      # checkpoint 1's labelled sheet (review_security_collection.py)
        [--prometheus http://127.0.0.1:9090] [--json report.json]

Sections, and the decision each one feeds:
1. Volume: is there enough to decide anything (shadow-week runbook section 5: about 2,000 untrusted items)?
2. Checkpoint 1: would-be outcomes per source and, with labels, the real false-positive rate and its
   95% upper bound (act mode needs <= 2%).
3. Checkpoint 2 intent rules: per rule, how often it would act on real traffic at today's threshold,
   against its SEC-130 target, and the threshold the week's data would suggest.
4. The shadow-only intent classifier next to phase A's signals.
5. Searches per task (sets `max_searches_per_task`, recurring-research task A0), tokens per research
   run (sets `schedule.token_budget_default` and `WEBAGENT_TASK_TOKEN_BUDGET`), and run times.
6. Triage: what it would have routed (Agentic = Always questions only).
7. The act-mode checklist (runbook section 5), with what is met, not met, or not measured here.

Real traffic is treated as benign until reviewed: a rule's "would act" count is its false-positive
count only after the reviewer confirms none of those tasks was an attack.

It never prints text: no question, search query, page, document or tool output, only numbers,
policy ids, sources and task counts.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

from review_security_collection import (  # noqa: E402
    UNTRUSTED,
    labelled_estimate,
    load_rows,
    percentiles,
    summarise,
    wilson_upper,
)

DEFAULT_THRESHOLDS = {   # mirrors app/security/thresholds.py DEFAULT_THRESHOLDS (checked by a test)
    "harm_verify_at": 50, "harm_decline_at": 85,
    "rewrite_at": 40, "subgoal_align_at": 15, "search_align_at": 25, "drift_align_at": 25,
    "drift_calls_at": 3, "risk_verify_at": 30, "risk_decline_at": 60, "egress_words_at": 8,
}
UNTRUSTED_TARGET = 2000          # runbook section 5
MIN_FOR_SUGGESTION = 100         # fewer rows than this: report the values, suggest nothing
FP_UPPER_TARGET = 0.02           # runbook section 5, the release gate's limit
SEARCH_RULE_TARGET = 0.01        # SEC-130: off-intent-search-verify, share of untainted searches
DECLINE_RULE_TARGET = 0.005      # SEC-130: each decline rule, share of tasks
SEARCH_ACTIONS = ("tool::web_search", "tool::internal_retrieve")
CALL_ACTIONS = ("tool::web_search", "tool::internal_retrieve", "tool::execute_code")


# --- loading ------------------------------------------------------------------------------------

def load_jsonl(pattern: str, since: Optional[str]) -> List[dict]:
    out = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if since and str(row.get("ts", ""))[:10] < since:
                continue
            out.append(row)
    return out


def load_results(folder: Path, since: Optional[str]) -> List[dict]:
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if since and str(row.get("created_at", ""))[:10] < since:
            continue
        out.append(row)
    return out


# --- helpers ------------------------------------------------------------------------------------

def share(hits: int, n: int) -> Optional[float]:
    return round(hits / n, 4) if n else None


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """The value at quantile q (0-1), nearest rank; None for no values."""
    vals = sorted(values)
    if not vals:
        return None
    rank = max(1, math.ceil(q * len(vals)))
    return vals[rank - 1]


def status(value: Optional[float], target: float, n: int, low_n: int = 30) -> str:
    if value is None:
        return "no data"
    if n < low_n:
        return f"too few ({n}) to decide"
    return "within target" if value <= target else "over target"


def ctx(row: dict) -> Dict[str, Any]:
    return row.get("context") or {}


# --- checkpoint 2 intent rules ------------------------------------------------------------------

RULES = [
    # id, actions, would-act test on the context, denominator ("calls", "untainted_searches", "tasks"), target
    ("off-intent-search-verify", SEARCH_ACTIONS,
     lambda c, t: c.get("signals_available") and _num(c, "intent_alignment") < t["search_align_at"],
     "untainted_searches", SEARCH_RULE_TARGET),
    ("call-risk-above-intent-verify", CALL_ACTIONS,
     lambda c, t: c.get("signals_available") and _num(c, "risk_above_intent", 0) >= t["risk_verify_at"],
     "calls", None),
    ("call-risk-above-intent-decline", CALL_ACTIONS,
     lambda c, t: c.get("signals_available") and _num(c, "risk_above_intent", 0) >= t["risk_decline_at"],
     "tasks", DECLINE_RULE_TARGET),
    ("private-egress-decline", ("tool::web_search",),
     lambda c, t: _num(c, "egress_overlap", 0) >= t["egress_words_at"], "tasks", DECLINE_RULE_TARGET),
    ("undeclared-tool-verify", SEARCH_ACTIONS + ("tool::document_extract", "tool::execute_code"),
     lambda c, t: c.get("declared") is False, "calls", None),
    ("drift-escalate", SEARCH_ACTIONS + ("plan::subgoal",),
     lambda c, t: _num(c, "low_alignment_calls", 0) >= t["drift_calls_at"], "tasks", DECLINE_RULE_TARGET),
    ("rewrite-drift-verify", ("plan::rewrite",),
     lambda c, t: c.get("signals_available") and _num(c, "rewrite_alignment") < t["rewrite_at"], "calls", None),
    ("subgoal-off-intent-decline", ("plan::subgoal",),
     lambda c, t: c.get("signals_available") and _num(c, "intent_alignment") < t["subgoal_align_at"],
     "tasks", DECLINE_RULE_TARGET),
    ("subgoal-harm-decline", ("plan::subgoal",),
     lambda c, t: c.get("signals_available") and _num(c, "subgoal_harm", 0) >= t["harm_decline_at"]
     and _num(c, "intent_harm", 0) < t["harm_verify_at"], "tasks", DECLINE_RULE_TARGET),
]


def _num(c: Dict[str, Any], key: str, missing: float = 100.0) -> float:
    value = c.get(key)
    return float(value) if isinstance(value, (int, float)) else missing


def intent_rules(rows: List[dict], thresholds: Dict[str, int]) -> Dict[str, Any]:
    cp2 = [r for r in rows if r.get("checkpoint") == 2]
    tasks = {r.get("task_id") for r in cp2}
    untainted_searches = [r for r in cp2 if r.get("action") in SEARCH_ACTIONS and not ctx(r).get("tainted")]
    out: Dict[str, Any] = {"tasks_with_checkpoint2_rows": len(tasks), "untainted_searches": len(untainted_searches),
                           "rules": {}}
    for rule_id, actions, test, denominator, target in RULES:
        rows_for = [r for r in cp2 if r.get("action") in actions]
        hits = [r for r in rows_for if test(ctx(r), thresholds)]
        recorded = sum(1 for r in rows_for if rule_id in (r.get("shadow_policy_ids") or []))
        if denominator == "untainted_searches":
            hit_n = sum(1 for r in hits if not ctx(r).get("tainted"))
            n = len(untainted_searches)
        elif denominator == "tasks":
            hit_n = len({r.get("task_id") for r in hits})
            n = len(tasks)
        else:
            hit_n, n = len(hits), len(rows_for)
        rate = share(hit_n, n)
        entry = {"rows_checked": len(rows_for), "would_act": hit_n, "per": denominator, "of": n, "rate": rate,
                 "upper_95": wilson_upper(hit_n, n) if n else None,
                 "recorded_shadow_matches": recorded}
        if target is not None:
            entry.update(target=target, status=status(entry["upper_95"], target, n))
        out["rules"][rule_id] = entry
    out["suggested_thresholds"] = suggest_thresholds(cp2, untainted_searches)
    return out


def suggest_thresholds(cp2: List[dict], untainted_searches: List[dict]) -> Dict[str, Any]:
    """Where each threshold would sit for the rule to act on at most its target share of the week's
    (assumed benign) traffic. For "below" rules that's a low quantile, for "at or above" a high one."""
    def values(rows, key):
        return [float(ctx(r)[key]) for r in rows if isinstance(ctx(r).get(key), (int, float))]

    sub = [r for r in cp2 if r.get("action") == "plan::subgoal"]
    rew = [r for r in cp2 if r.get("action") == "plan::rewrite"]
    calls = [r for r in cp2 if r.get("action") in CALL_ACTIONS]
    search_align = values(untainted_searches, "intent_alignment")
    enough = len(search_align) >= MIN_FOR_SUGGESTION
    out = {
        "search_align_at": {"n": len(search_align), "now": DEFAULT_THRESHOLDS["search_align_at"],
                            "at_most_1pct_below": quantile(search_align, SEARCH_RULE_TARGET) if enough else None,
                            **percentiles(search_align)},
        "subgoal_align_at": {"n": len(values(sub, "intent_alignment")), "now": DEFAULT_THRESHOLDS["subgoal_align_at"],
                             "lowest": min(values(sub, "intent_alignment"), default=None),
                             **percentiles(values(sub, "intent_alignment"))},
        "rewrite_at": {"n": len(values(rew, "rewrite_alignment")), "now": DEFAULT_THRESHOLDS["rewrite_at"],
                       "lowest": min(values(rew, "rewrite_alignment"), default=None)},
        "risk_verify_at": {"n": len(values(calls, "risk_above_intent")), "now": DEFAULT_THRESHOLDS["risk_verify_at"],
                           "highest": max(values(calls, "risk_above_intent"), default=None),
                           **percentiles(values(calls, "risk_above_intent"))},
        "egress_words_at": {"n": len(values(calls, "egress_overlap")), "now": DEFAULT_THRESHOLDS["egress_words_at"],
                            "highest": max(values(calls, "egress_overlap"), default=None)},
    }
    return out


def shadow_classifier(rows: List[dict]) -> Dict[str, Any]:
    """The shadow-only classifier (3.9.0) next to phase A's MiniLM alignment, on the same rows."""
    pairs = []
    for r in rows:
        c = ctx(r)
        if r.get("checkpoint") == 2 and isinstance(c.get("shadow_classifier_alignment"), (int, float)) \
                and isinstance(c.get("intent_alignment"), (int, float)):
            pairs.append((float(c["intent_alignment"]), float(c["shadow_classifier_alignment"])))
    out: Dict[str, Any] = {"rows_with_both": len(pairs)}
    if len(pairs) >= 3:
        xs, ys = zip(*pairs)
        out["phase_a_alignment"] = percentiles(list(xs))
        out["classifier_alignment"] = percentiles(list(ys))
        out["pearson"] = _pearson(xs, ys)
        cut = DEFAULT_THRESHOLDS["search_align_at"]
        out["below_search_threshold"] = {"phase_a": sum(x < cut for x in xs), "classifier": sum(y < cut for y in ys),
                                         "both": sum(x < cut and y < cut for x, y in pairs)}
    return out


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 3)


# --- runs ---------------------------------------------------------------------------------------

def searches_per_task(rows: List[dict], current_cap: int = 5) -> Dict[str, Any]:
    """Checkpoint 2 checks each web search before it runs, so its rows count searches per task."""
    per_task = Counter(r.get("task_id") for r in rows if r.get("action") == "tool::web_search")
    research_tasks = {r.get("task_id") for r in rows if r.get("checkpoint") == 2}
    counts = [per_task.get(t, 0) for t in research_tasks]
    p95 = quantile(counts, 0.95)
    return {"research_tasks": len(counts), "per_task": percentiles([float(c) for c in counts]) if counts else {},
            "max": max(counts, default=None), "p95": p95,
            "suggested_max_searches_per_task": (int(p95) + 1) if p95 is not None and len(counts) >= 20 else None,
            "rule": "at least 95% of the week's research tasks under the cap (recurring-research A0)",
            "current_value_uat": current_cap}


def run_usage(results: List[dict]) -> Dict[str, Any]:
    research = [r for r in results if r.get("route") != "direct" and r.get("status") == "completed"]
    tokens = [float(r["tokens_used"]) for r in research if isinstance(r.get("tokens_used"), (int, float))]
    seconds = [float(r["duration_seconds"]) for r in research if isinstance(r.get("duration_seconds"), (int, float))]
    p90 = quantile(tokens, 0.9)
    return {
        "tasks": len(results), "research_completed": len(research),
        "failed": sum(1 for r in results if r.get("status") in ("failed", "error")),
        "tokens_per_research_run": {"n": len(tokens), **percentiles(tokens), "max": max(tokens, default=None)},
        "suggested_token_budget_default": int(math.ceil(1.5 * p90 / 1000) * 1000) if p90 and len(tokens) >= 20 else None,
        "suggested_task_token_budget": int(math.ceil(2 * max(tokens) / 1000) * 1000) if len(tokens) >= 20 else None,
        "seconds_per_research_run": {"n": len(seconds), **percentiles(seconds), "max": max(seconds, default=None),
                                     "limit": 420},
    }


def triage_summary(rows: List[dict]) -> Dict[str, Any]:
    p = [float(r["p_agentic"]) for r in rows if isinstance(r.get("p_agentic"), (int, float))]
    return {"questions": len(rows),
            "would_route": dict(Counter(r.get("route") for r in rows)),
            "decided_by": dict(Counter(r.get("decided_by") for r in rows)),
            "uncertain_0.35_to_0.65": sum(0.35 <= x <= 0.65 for x in p),
            "classifier_versions": sorted({str(r.get("classifier_version")) for r in rows})}


# --- Prometheus (optional) ----------------------------------------------------------------------

PROM_QUERIES = {   # {r} is the week's range, so earlier traffic (e.g. a dev stack on the same Prometheus) is left out
    "checkpoint1_p95_seconds": "histogram_quantile(0.95, sum by (le) (increase(webagent_guard_hop_seconds_bucket[{r}])))",
    "checkpoint2_p95_seconds": "histogram_quantile(0.95, sum by (le) (increase(webagent_guard_tool_seconds_bucket[{r}])))",
    "unlabelled_llm_calls": "sum(increase(webagent_guard_unlabelled_total[{r}]))",
    "peak_rss_bytes_per_worker": 'max(max_over_time(memory_usage_bytes{{type="rss"}}[{r}]))',
}
# Labelled counters have no series until their first increment, so an empty answer means 0
ZERO_WHEN_EMPTY = {"unlabelled_llm_calls"}


def prom_range(since: Optional[str], now: Optional[datetime] = None) -> str:
    """The PromQL range from `since` (a date) to now, in hours (at least 1 h); 7d without `since`."""
    if not since:
        return "7d"
    start = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    hours = max(1, math.ceil(((now or datetime.now(timezone.utc)) - start).total_seconds() / 3600))
    return f"{hours}h"


def prometheus(url: Optional[str], since: Optional[str] = None) -> Dict[str, Any]:
    if not url:
        return {"checked": False}
    out: Dict[str, Any] = {"checked": True, "range": prom_range(since)}
    for name, template in PROM_QUERIES.items():
        query = template.format(r=out["range"])
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/api/v1/query?{urllib.parse.urlencode({'query': query})}",
                                        timeout=10) as reply:
                data = json.load(reply)
            result = data.get("data", {}).get("result", [])
            out[name] = round(float(result[0]["value"][1]), 4) if result else (0.0 if name in ZERO_WHEN_EMPTY else None)
        except Exception as exc:  # noqa: BLE001 - optional; report, don't fail
            out[name] = None
            out.setdefault("errors", []).append(f"{name}: {type(exc).__name__}")
    return out


def checklist(volume_items: int, fp: Optional[Dict[str, Any]], cp1: Dict[str, Any], prom: Dict[str, Any]) -> List[Dict[str, str]]:
    def item(name, ok: Optional[bool], detail):
        return {"check": name, "status": "met" if ok else ("not met" if ok is False else "not measured"), "detail": detail}

    upper = (fp or {}).get("false_positive_rate_upper_95")
    p1, p2 = prom.get("checkpoint1_p95_seconds"), prom.get("checkpoint2_p95_seconds")
    rss = prom.get("peak_rss_bytes_per_worker")
    missing = "no data in Prometheus for the week" if prom.get("checked") else "needs --prometheus"
    return [
        item("about 2,000 untrusted items judged", volume_items >= UNTRUSTED_TARGET, f"{volume_items} judged"),
        item("false-positive upper bound <= 2% (labelled)", None if upper is None else upper <= FP_UPPER_TARGET,
             "label the sheet first (--labels)" if upper is None else f"upper 95% bound {upper:.4f}"),
        item("checkpoint 1 p95 < 150 ms", None if p1 is None else p1 < 0.15, missing if p1 is None else f"{p1:.3f} s"),
        item("checkpoint 2 p95 < 20 ms", None if p2 is None else p2 < 0.02, missing if p2 is None else f"{p2:.4f} s"),
        # Per-worker RSS counts the shared, memory-mapped model files once per worker, so 2 x RSS
        # overstates the pod; the pod's own figure (docker stats, kubectl top) decides this check.
        item("memory <= 2.4 GiB per pod (two workers)", None,
             "measure the pod with `docker stats` or `kubectl top pod`" + (
                 "" if rss is None else f"; peak RSS of one worker {rss / 2**30:.2f} GiB (includes shared model pages)")),
        item("no unlabelled LLM calls", None if prom.get("unlabelled_llm_calls") is None else prom["unlabelled_llm_calls"] == 0,
             missing if prom.get("unlabelled_llm_calls") is None else f"{prom['unlabelled_llm_calls']:.0f}"),
        item("scores_unavailable near 0", cp1.get("scores_unavailable", 0) == 0, f"{cp1.get('scores_unavailable', 0)} items"),
        item("declined / consent user turns read and judged reasonable", None,
             "manual: their counts are in checkpoint 1's outcomes; text isn't collected outside dev"),
    ]


# --- report -------------------------------------------------------------------------------------

def build(data: Path, since: Optional[str], labels: Optional[Path], prom_url: Optional[str]) -> Dict[str, Any]:
    rows = load_rows(data / "security" / "collected", since)
    results = load_results(data / "results", since)
    triage = load_jsonl(str(data / "triage" / "collected" / "*.jsonl"), since)
    base = summarise(rows)
    judged = sum(1 for r in rows if r.get("checkpoint") == 1 and r.get("item") and r["item"].get("source") in UNTRUSTED)
    fp = None
    if labels:
        with open(labels, newline="", encoding="utf-8") as handle:
            fp = labelled_estimate(list(csv.DictReader(handle)), rows)
    prom = prometheus(prom_url, since)
    return {
        "volume": {"days": base["days"], "tasks_in_collection": base["tasks"], "rows": base["rows"],
                   "untrusted_items_judged": judged, "target": UNTRUSTED_TARGET,
                   "share_of_target": share(judged, UNTRUSTED_TARGET)},
        "checkpoint1": {**base["checkpoint1"], "labelled_estimate": fp},
        "checkpoint2_intent": intent_rules(rows, DEFAULT_THRESHOLDS),
        "shadow_classifier": shadow_classifier(rows),
        "searches": searches_per_task(rows),
        "runs": run_usage(results),
        "triage": triage_summary(triage),
        "prometheus": prom,
        "act_mode_checklist": checklist(judged, fp, base["checkpoint1"], prom),
    }


def markdown(report: Dict[str, Any]) -> str:
    v, s, r = report["volume"], report["searches"], report["runs"]
    lines = ["# Shadow week report", "",
             f"Days: {', '.join(v['days']) or 'none'} · tasks: {v['tasks_in_collection']} · rows: {v['rows']}", "",
             "## 1. Volume", "",
             f"Untrusted items judged: **{v['untrusted_items_judged']}** of about {v['target']} needed "
             f"({(v['share_of_target'] or 0) * 100:.0f}%).", "",
             "## 2. Checkpoint 1", "", "| Source | Items | Would act (share) |", "|---|---|---|"]
    cp1 = report["checkpoint1"]
    for src, counts in cp1["items_by_source"].items():
        lines.append(f"| {src} | {counts.get('items', 0)} | {cp1['would_act_rate_by_source'].get(src, 0)} |")
    est = cp1.get("labelled_estimate")
    lines += ["", f"Policies matched: {cp1['policies_matched'] or 'none'}. Scores unavailable: {cp1['scores_unavailable']}.",
              "", ("False positives: label the sheet (`review_security_collection.py --sheet`), then pass `--labels`."
                   if not est else f"False positives: {est['benign_items_that_would_be_dropped']} of "
                   f"{est['untrusted_items_judged']} (upper 95% bound {est['false_positive_rate_upper_95']})."),
              "", "## 3. Checkpoint 2 intent rules (shadow-only)", "",
              "| Rule | Would act | Per | Rate | Upper 95% | Target | Status |", "|---|---|---|---|---|---|---|"]
    for rule_id, e in report["checkpoint2_intent"]["rules"].items():
        lines.append(f"| {rule_id} | {e['would_act']} | {e['per']} ({e['of']}) | {e['rate']} | {e['upper_95']} | "
                     f"{e.get('target', '—')} | {e.get('status', 'reported')} |")
    sug = report["checkpoint2_intent"]["suggested_thresholds"]
    lines += ["", "Thresholds from the week's data (real traffic assumed benign until reviewed):", "",
              (f"- `search_align_at` now {sug['search_align_at']['now']}: at most 1% of untainted searches below "
               f"{sug['search_align_at']['at_most_1pct_below']} (n={sug['search_align_at']['n']})."
               if sug['search_align_at']['at_most_1pct_below'] is not None else
               f"- `search_align_at` now {sug['search_align_at']['now']}: too few untainted searches to suggest one "
               f"(n={sug['search_align_at']['n']}, need {MIN_FOR_SUGGESTION}); median {sug['search_align_at'].get('p50')}."),
              f"- `subgoal_align_at` now {sug['subgoal_align_at']['now']}: lowest subgoal alignment "
              f"{sug['subgoal_align_at']['lowest']} (n={sug['subgoal_align_at']['n']}).",
              f"- `rewrite_at` now {sug['rewrite_at']['now']}: lowest rewrite alignment {sug['rewrite_at']['lowest']} "
              f"(n={sug['rewrite_at']['n']}).",
              f"- `risk_verify_at` now {sug['risk_verify_at']['now']}: highest risk above intent "
              f"{sug['risk_verify_at']['highest']} (n={sug['risk_verify_at']['n']}).",
              f"- `egress_words_at` now {sug['egress_words_at']['now']}: highest overlap {sug['egress_words_at']['highest']}.",
              "", "## 4. Shadow-only classifier", ""]
    sc = report["shadow_classifier"]
    lines.append(f"Rows with both scores: {sc['rows_with_both']}." + (
        f" Correlation with phase A: {sc.get('pearson')}. Below the search threshold: {sc.get('below_search_threshold')}."
        if sc["rows_with_both"] >= 3 else ""))
    lines += ["", "## 5. Searches, tokens and run times", "",
              f"- Searches per research task: p95 {s['p95']}, max {s['max']} over {s['research_tasks']} tasks. " +
              (f"Suggested `max_searches_per_task`: **{s['suggested_max_searches_per_task']}** (now {s['current_value_uat']} in uat)."
               if s['suggested_max_searches_per_task'] is not None else "Too few tasks to suggest a cap (need 20)."),
              f"- Tokens per research run: median {r['tokens_per_research_run'].get('p50')}, p90 "
              f"{r['tokens_per_research_run'].get('p90')}, max {r['tokens_per_research_run'].get('max')} "
              f"(n={r['tokens_per_research_run']['n']}). " + (
                  f"Suggested `schedule.token_budget_default` **{r['suggested_token_budget_default']}**; "
                  f"`WEBAGENT_TASK_TOKEN_BUDGET` **{r['suggested_task_token_budget']}** (2x the max)."
                  if r['suggested_token_budget_default'] else "Too few runs to suggest budgets (need 20)."),
              f"- Seconds per research run: median {r['seconds_per_research_run'].get('p50')}, max "
              f"{r['seconds_per_research_run'].get('max')} (limit 420). Failed tasks: {r['failed']} of {r['tasks']}.",
              "", "## 6. Triage (Agentic = Always questions)", ""]
    t = report["triage"]
    lines += [f"{t['questions']} questions; would route {t['would_route']}; decided by {t['decided_by']}; "
              f"uncertain {t['uncertain_0.35_to_0.65']}.", "", "## 7. Act-mode checklist (checkpoint 1)", "",
              "| Check | Status | Detail |", "|---|---|---|"]
    for c in report["act_mode_checklist"]:
        lines.append(f"| {c['check']} | {c['status']} | {c['detail']} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="the week's data folder (holds security/, results/, triage/)")
    ap.add_argument("--since", default=None, help="first day to include (YYYY-MM-DD)")
    ap.add_argument("--labels", type=Path, default=None, help="checkpoint 1's labelled sheet")
    ap.add_argument("--prometheus", default=None, help="Prometheus base URL for latency, memory and unlabelled calls")
    ap.add_argument("--json", type=Path, default=None, help="also write the full report as JSON")
    args = ap.parse_args(argv)
    report = build(args.data.expanduser(), args.since, args.labels, args.prometheus)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    sys.stdout.write(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
