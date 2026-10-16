"""Alignment dataset rules shared by the builder, its --check-data and tests (SEC-120, SEC-121).

Rows: {"intent", "call", "action", "labels": {"aligned": yes|no, "risk": none|low|high},
"origin": "<dataset>:<role>", "role": "T"|"E", "split", "groups", "sources", "cluster"}.
`sources` names every dataset whose text is in the row (licence checks); `groups` are the units that
never cross splits (source question, template, environment, toolkit, batch). Nothing here prints text.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

ACTIONS = ("web_search", "internal_retrieve", "execute_code", "plan", "function")
ALIGNED = ("yes", "no")
RISKS = ("none", "low", "high")
SPLITS = ("train", "test", "probe")
MAX_INTENT, MAX_CALL = 2000, 1000
NEAR_DUPLICATE = 0.9

# ASQ source question sets we accept, with their licences (SEC-121). MS MARCO (research terms, no
# licence tag) was left out in the plan (§8.2); Researchy Questions (CDLA-Permissive-2.0) is not on
# the approved list. Anything not here is skipped.
ASQ_SOURCES = {"hotpotqa-test": {"licence": "CC BY-SA 4.0", "questions": ("BeIR/hotpotqa", "queries/queries-00000-of-00001.parquet")}}

# dataset -> (licence, role, local-only, README name). Local-only: gated (xLAM, TraceSafe) or the
# authors ask that it not be trained on (AgentHarm, SHADE-Arena, OS-Harm); the repository is public.
SOURCES: Dict[str, tuple] = {
    "webagent-gen": ("Project data", "T", False, "WebAgent generation"),
    "asq": ("MIT; HotpotQA CC BY-SA 4.0", "T", False, "ASQ"),
    "glaive": ("Apache-2.0", "T", False, "glaive-function-calling-v2"),
    "toolace": ("Apache-2.0", "T", False, "ToolACE"),
    "xlam": ("CC BY 4.0, gated", "T", True, "xlam-function-calling-60k"),
    "injecagent": ("MIT", "T", False, "InjecAgent"),
    "tasktracker": ("MIT", "T", False, "TaskTracker"),
    "squad": ("CC BY-SA 4.0", "T", False, "SQuAD"),
    "xstest": ("CC BY 4.0", "T", False, "XSTest"),
    "notinject": ("MIT", "T", False, "NotInject"),
    "aegis": ("CC BY 4.0", "T", False, "Aegis"),
    "harmbench": ("MIT", "T", False, "HarmBench"),
    "agent-safetybench": ("MIT", "T", False, "Agent-SafetyBench"),
    "toolemu": ("Apache-2.0", "T", False, "ToolEmu"),
    "redcode": ("CC BY 4.0", "T", False, "RedCode"),
    "wildchat": ("ODC-BY", "T", False, "WildChat"),
    "saroku": ("Apache-2.0", "T", False, "saroku"),
    "privacylens": ("MIT / CC BY 4.0", "T", False, "PrivacyLens"),
    "agentdojo": ("MIT", "E", False, "AgentDojo"),
    "agentharm": ("MIT + safety-only clause; no-train", "E", True, "AgentHarm"),
    "shade-arena": ("MIT; no-train", "E", True, "SHADE-Arena"),
    "os-harm": ("Apache-2.0; no-train", "E", True, "OS-Harm"),
    "tracesafe": ("Apache-2.0, gated; no-train", "E", True, "TraceSafe"),
    "agentdam": ("CC BY-NC 4.0", "E", False, "AgentDAM"),
}
SHARE_ALIKE = {"asq", "squad"}  # CC BY-SA text
NON_COMMERCIAL = {"agentdam"}  # CC BY-NC text
TEST_ONLY = {"agentdam"}
# (aligned, risk) cells per action family that must have at least MIN_CELL rows in the committed
# data. execute_code's aligned cells wait for the coding assistant's generated rows (task B3).
REQUIRED_CELLS = {
    "web_search": (("yes", "none"), ("no", "none"), ("no", "high")),
    "internal_retrieve": (("yes", "none"), ("no", "none"), ("no", "high")),
    "plan": (("yes", "none"), ("no", "none"), ("no", "high")),
    "function": (("yes", "none"), ("no", "none"), ("no", "high"), ("yes", "high"), ("no", "low")),
    "execute_code": (("no", "high"),),
}
MIN_CELL = 50


def dataset_of(origin: str) -> str:
    return origin.split(":", 1)[0]


def is_local(row: dict) -> bool:
    return any(SOURCES[s][2] for s in row["sources"])


def schema_errors(row: dict) -> List[str]:
    errs = []
    for key in ("intent", "call", "origin"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errs.append(f"empty {key}")
    if row.get("action") not in ACTIONS:
        errs.append("bad action")
    labels = row.get("labels") or {}
    if labels.get("aligned") not in ALIGNED or labels.get("risk") not in RISKS:
        errs.append("bad labels")
    if row.get("role") not in ("T", "E") or row.get("split") not in SPLITS:
        errs.append("bad role or split")
    srcs = row.get("sources") or []
    if not srcs or any(s not in SOURCES for s in srcs) or dataset_of(row.get("origin", "")) not in srcs:
        errs.append("bad sources")
    elif row["role"] != ("E" if any(SOURCES[s][1] == "E" for s in srcs) else "T"):
        errs.append("role does not match sources")
    if len(row.get("intent") or "") > MAX_INTENT or len(row.get("call") or "") > MAX_CALL:
        errs.append("too long")
    if not isinstance(row.get("groups"), list) or not row["groups"]:
        errs.append("no groups")
    return errs


def rule_errors(row: dict, committed: bool) -> List[str]:
    """SEC-121 rules for one row in a committed (True) or local/ (False) file."""
    errs, srcs = [], set(row["sources"])
    if row["role"] == "E" and row["split"] == "train":
        errs.append("evaluation-only row in train")
    if srcs & TEST_ONLY and row["split"] != "test":
        errs.append("test-only source outside test")
    if committed and is_local(row):
        errs.append("gated or no-train row in a committed file")
    if srcs & SHARE_ALIKE and srcs & NON_COMMERCIAL:
        errs.append("CC BY-SA and CC BY-NC text in one row")
    return errs


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.is_file() else []


def check_data(root: Path, embed: Optional[Callable[[List[str]], "object"]] = None,
               readme: Optional[Path] = None) -> List[str]:
    """--check-data: every failure as a short message (counts only, never row text)."""
    errors: Counter = Counter()
    rows: List[dict] = []
    for committed, folder in ((True, root), (False, root / "local")):
        for split in SPLITS:
            for row in read_jsonl(folder / f"{split}.jsonl"):
                for e in schema_errors(row) or rule_errors(row, committed):
                    errors[f"{folder.name}/{split}: {e}"] += 1
                if row.get("split") != split:
                    errors[f"{folder.name}/{split}: row says split {row.get('split')}"] += 1
                row["_committed"] = committed
                rows.append(row)
    keys = Counter((r.get("intent"), r.get("call"), r.get("action")) for r in rows)
    if sum(v - 1 for v in keys.values()):
        errors["duplicate (intent, call, action) rows"] = sum(v - 1 for v in keys.values())
    group_splits: Dict[str, set] = defaultdict(set)
    for r in rows:
        for g in r.get("groups") or []:
            group_splits[g].add(r.get("split"))
    crossing = sum(1 for s in group_splits.values() if len(s) > 1)
    if crossing:
        errors["groups (question, template, environment, toolkit) in more than one split"] = crossing
    cells = Counter((r["action"], r["labels"]["aligned"], r["labels"]["risk"]) for r in rows if r["_committed"])
    for action, wanted in REQUIRED_CELLS.items():
        for aligned, risk in wanted:
            if cells[(action, aligned, risk)] < MIN_CELL:
                errors[f"cell {action} aligned={aligned} risk={risk}: {cells[(action, aligned, risk)]} rows < {MIN_CELL}"] += 1
    if readme is not None:
        text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        for name, (_, _, _, title) in SOURCES.items():
            if title not in text:
                errors[f"README has no licence entry for {name}"] += 1
    if embed is not None and rows:
        n = cross_split_near_duplicates(rows, embed)
        if n:
            errors["near-duplicate (intent and call cosine >= 0.9) pairs across splits"] = n
    return [f"{k} ({v})" for k, v in sorted(errors.items())]


def cross_split_near_duplicates(rows: List[dict], embed: Callable[[List[str]], "object"]) -> int:
    """Pairs of rows in different splits whose intents and calls both have cosine >= 0.9."""
    import numpy as np

    ei, ec = embed([r["intent"] for r in rows]), embed([r["call"] for r in rows])
    split = np.array([SPLITS.index(r["split"]) for r in rows])
    n = 0
    for start in range(0, len(rows), 1024):
        sims = ec[start:start + 1024] @ ec.T
        for bi, j in zip(*np.nonzero(sims >= NEAR_DUPLICATE)):
            i = start + int(bi)
            if i < j and split[i] != split[j] and float(ei[i] @ ei[j]) >= NEAR_DUPLICATE:
                n += 1
    return n


def counts(splits: Dict[str, Iterable[dict]]) -> dict:
    """The manifest: rows per source, role, label, action, split, committed or local."""
    out: dict = {"by_split": {}, "by_origin": Counter(), "by_label": Counter(), "by_action_label": Counter()}
    for split, rows in splits.items():
        rows = list(rows)
        place = Counter("local" if is_local(r) else "committed" for r in rows)
        out["by_split"][split] = {"rows": len(rows), **place,
                                  "role": dict(Counter(r["role"] for r in rows))}
        for r in rows:
            where = "local" if is_local(r) else "committed"
            lab = f"aligned={r['labels']['aligned']},risk={r['labels']['risk']}"
            out["by_origin"][f"{r['origin']}|{split}|{where}"] += 1
            out["by_label"][f"{lab}|{split}|{where}"] += 1
            out["by_action_label"][f"{r['action']}|{lab}|{where}"] += 1
    for k in ("by_origin", "by_label", "by_action_label"):
        out[k] = dict(sorted(out[k].items()))
    return out
