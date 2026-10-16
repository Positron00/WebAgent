"""Pairing and labelling for build_alignment_dataset.py (model-and-data design §2), role-T sources.

Every row is made by `Ctx.row`, which applies the length limits and gives the row the split of its
groups (source question, template, environment, toolkit, batch); a row whose groups fall in different
splits is dropped and counted, so no group ever crosses splits. Partners (another question, the
nearest benign question) are always chosen from the same split. Nothing here prints text.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from alignment_data import MAX_CALL, MAX_INTENT, SOURCES


# Percent of groups held out for test, by group prefix (default 18). InjecAgent rows need both their
# template and their instruction held out, so at 18% only ~3% of cases would reach test.
TEST_SHARE = {"injecagent:": 33}


def h(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:16]


class Ctx:
    def __init__(self, raw: Path, seed: int, embed: Callable[[List[str]], np.ndarray]):
        self.raw, self.seed, self.embed = raw, seed, embed
        self.rng = random.Random(seed)
        self.dropped: Counter = Counter()

    def read(self, name: str) -> List[dict]:
        path = self.raw / f"{name}.jsonl"
        return [json.loads(line) for line in path.open(encoding="utf-8")] if path.is_file() else []

    def split_of(self, key: str, role: str = "T") -> str:
        v = int(hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest(), 16) % 100
        if role == "E":  # evaluation-only: test and probe, never train
            return "probe" if v < 10 else "test"
        test = next((share for prefix, share in TEST_SHARE.items() if key.startswith(prefix)), 18)
        return "probe" if v < 2 else "test" if v < 2 + test else "train"

    def pick(self, key: str, options: Sequence[str]) -> str:
        return options[int(h(f"{self.seed}:{key}"), 16) % len(options)]

    def row(self, intent: str, call: str, action: str, aligned: str, risk: str, origin: str,
            groups: Sequence[str], sources: Sequence[str], split: Optional[str] = None) -> Optional[dict]:
        intent, call = (intent or "").strip(), (call or "").strip()
        if not intent or not call:
            self.dropped["empty"] += 1
            return None
        if len(intent) > MAX_INTENT or len(call) > MAX_CALL:
            self.dropped["too_long"] += 1
            return None
        srcs = sorted(set(sources))
        role = "E" if any(SOURCES[s][1] == "E" for s in srcs) else "T"
        splits = {split} if split else {self.split_of(g, role) for g in groups}
        if len(splits) != 1:
            self.dropped["groups_in_different_splits"] += 1
            return None
        return {"intent": intent, "call": call, "action": action, "labels": {"aligned": aligned, "risk": risk},
                "origin": origin, "role": role, "split": splits.pop(), "groups": sorted(set(groups)), "sources": srcs}

    def take(self, rows: List[dict], n: int) -> List[dict]:
        rows = [r for r in rows if r is not None]
        self.rng.shuffle(rows)
        return rows[:n]

    def nearest(self, queries: List[str], q_splits: List[str], pool: List[dict], k: int = 1,
                min_cos: float = -1.0, max_cos: float = 2.0) -> List[List[int]]:
        """Per query, up to k pool indices in the same split, most similar first, min_cos <= cos < max_cos."""
        if not queries or not pool:
            return [[] for _ in queries]
        qv, pv = self.embed(queries), self.embed([p["text"] for p in pool])
        by_split: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(pool):
            by_split[p["split"]].append(i)
        out: List[List[int]] = []
        for qi, split in enumerate(q_splits):
            idx = by_split.get(split, [])
            if not idx:
                out.append([])
                continue
            sims = pv[idx] @ qv[qi]
            order = np.argsort(-sims)
            out.append([idx[j] for j in order if min_cos <= sims[j] < max_cos][:k])
        return out


def item(ctx: Ctx, text: str, groups: List[str], sources: List[str], **extra) -> dict:
    """A pool entry (a question that can be an intent): text, groups, sources and its split."""
    return {"text": text, "groups": groups, "sources": sources, "split": ctx.split_of(groups[0]), **extra}


def read_gen(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.is_file() else []


def gen_rows(ctx: Ctx, rows: List[dict], rewrite_score: Callable[[str, str], int], rewrite_at: int):
    """webagent-gen: the rewrite and each plan question are aligned calls for their question, except
    tasks whose rewrite scores below rewrite_at against the question (the planner replaced the goal)."""
    out, review = [], []
    for r in rows:
        q, rewrite = r["question"], (r.get("rewrite") or "").strip()
        if rewrite and rewrite_score(rewrite, q) < rewrite_at:
            review.append(r["question_hash"])
            continue
        for call in [rewrite] + list(r.get("questions") or []):
            out.append(ctx.row(q, call, "plan", "yes", "none", "webagent-gen:call", [f"gen:{r['question_hash']}"], ["webagent-gen"]))
    return [r for r in out if r], review


def asq_rows(ctx: Ctx, rows: List[dict], accepted: Dict[str, dict], n: int, per_question: int = 3) -> List[dict]:
    out = []
    for r in rows:
        if r.get("set") not in accepted:
            ctx.dropped["asq_set_not_accepted"] += 1
            continue
        action = ctx.pick(r["qid"], ("web_search", "internal_retrieve"))  # ASQ agents searched a local corpus
        for q in r["queries"][:per_question]:
            out.append(ctx.row(r["question"], q, action, "yes", "none", "asq:query", [f"asq:{r['qid']}"], ["asq"]))
    return ctx.take(out, n)


def function_rows(ctx: Ctx, name: str, rows: List[dict], n: int) -> List[dict]:
    """Glaive, ToolACE, xLAM: the user request and "<function>: <arguments>"; split by function name."""
    out = [ctx.row(r["request"], f"{r['function']}: {r['arguments']}", "function", "yes", "none", f"{name}:call",
                   [f"{name}:fn:{r['function']}"], [name]) for r in rows]
    return ctx.take(out, n)


def injecagent_rows(ctx: Ctx, n: int) -> List[dict]:
    """The case's user instruction and the attacker instruction as a call; split by template and instruction."""
    seen, out = set(), []
    for r in ctx.read("injecagent"):
        if r.get("label") != "injection" or not r.get("template") or (r["user_instruction"], r["attack"]) in seen:
            continue
        seen.add((r["user_instruction"], r["attack"]))
        out.append(ctx.row(r["user_instruction"], r["attack"], "function", "no", "high", "injecagent:attack",
                           [f"injecagent:tpl:{r['template']}", f"injecagent:atk:{h(r['attack'])}"], ["injecagent"]))
    return ctx.take(out, n)


def tasktracker_rows(ctx: Ctx, n: int) -> List[dict]:
    """A SQuAD question as the intent, a TaskTracker secondary task as the call (aligned no, risk none)."""
    squad: Dict[str, List[dict]] = defaultdict(list)
    for q in ctx.read("squad_questions"):
        squad[ctx.split_of(f"squad:{q['title']}")].append(q)
    out = []
    for r in ctx.read("tasktracker"):
        key = f"tasktracker:{h(r['text'])}"
        pool = squad.get(ctx.split_of(key))
        if pool:
            q = pool[ctx.rng.randrange(len(pool))]
            out.append(ctx.row(q["question"], r["text"], ctx.pick(key, ("plan", "web_search")), "no", "none",
                               "tasktracker:task", [key, f"squad:{q['title']}"], ["tasktracker", "squad"]))
    return ctx.take(out, n)


def xstest_rows(ctx: Ctx, n_contrast: int, n_lookalike: int, n_notinject: int) -> List[dict]:
    """XSTest: a safe prompt as the intent and an unsafe contrast prompt of the matching category as the
    call (aligned no, risk high); safe prompts and NotInject prompts as look-alike calls for themselves."""
    rows = ctx.read("xstest")
    safe: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        if not str(r["type"]).startswith("contrast_"):
            safe[r["type"]].append(r["text"])
    contrast, look = [], []
    for r in rows:
        t = str(r["type"])
        if t.startswith("contrast_") and safe.get(t[len("contrast_"):]):
            base = t[len("contrast_"):]
            intent = safe[base][ctx.rng.randrange(len(safe[base]))]
            contrast.append(ctx.row(intent, r["text"], ctx.pick(r["text"], ("plan", "web_search")), "no", "high",
                                    "xstest:contrast", [f"xstest:type:{base}"], ["xstest"]))
        elif not t.startswith("contrast_"):
            look.append(ctx.row(r["text"], r["text"], ctx.pick(r["text"], ("plan", "web_search")), "yes", "none",
                                "xstest:lookalike", [f"xstest:type:{t}"], ["xstest"]))
    notinject = [ctx.row(r["text"], r["text"], ctx.pick(r["text"], ("plan", "web_search")), "yes", "none",
                         "notinject:lookalike", [f"notinject:{h(r['text'])}"], ["notinject"]) for r in ctx.read("notinject")]
    return ctx.take(contrast, n_contrast) + ctx.take(look, n_lookalike) + ctx.take(notinject, n_notinject)


def harm_rows(ctx: Ctx, harmful: List[dict], pool: List[dict], n: int, min_cos: float = 0.5) -> List[dict]:
    """Aegis dangerous-category prompts and HarmBench behaviours as calls under the nearest benign
    question on the same topic (MiniLM cosine >= min_cos, same split); aligned no, risk high."""
    harmful = ctx.take(harmful, n * 2)
    splits = [ctx.split_of(f"{x['source']}:{h(x['text'])}") for x in harmful]
    near = ctx.nearest([x["text"] for x in harmful], splits, pool, min_cos=min_cos)
    out = []
    for x, idx in zip(harmful, near):
        if not idx:
            ctx.dropped["harm_no_benign_neighbour"] += 1
            continue
        p = pool[idx[0]]
        out.append(ctx.row(p["text"], x["text"], ctx.pick(x["text"], ("web_search", "plan", "internal_retrieve")), "no", "high",
                           f"{x['source']}:harmful", [f"{x['source']}:{h(x['text'])}", *p["groups"]], [x["source"], *p["sources"]]))
    return ctx.take(out, n)
