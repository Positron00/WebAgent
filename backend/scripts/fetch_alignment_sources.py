#!/usr/bin/env python3
"""Download the approved alignment sources into a scratch folder (counts only; SEC-120, SEC-121).

Run from backend/:
    python scripts/fetch_alignment_sources.py --out <scratch dir> [--only saroku asq ...] [--cap 20000]

Each source is written to <out>/<name>.jsonl with only the fields the builder uses. Gated sources
(xLAM, TraceSafe) are read with the Hugging Face token from its standard location (`hf auth login`),
never from the repository; without one they are skipped with a message. Licences and roles are in
data/security/alignment/README.md. Nothing here prints example text.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
for p in (str(BACKEND), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import alignment_sources as gh  # noqa: E402
from alignment_data import ASQ_SOURCES  # noqa: E402  (accepted ASQ source sets and licences)
import fetch_security_sources as sec  # noqa: E402

DEFAULT_OUT = Path("/private/tmp/claude-501/-Users-danc-Library-CloudStorage-Dropbox-MachineLearning-Repos-WebAgent/"
                   "5135da8e-c349-41ba-b909-83ca3bcb587e/scratchpad/alignsrc")
SEED = 13
GATED = {"xlam", "tracesafe"}


def hf_token() -> Optional[str]:
    """The user's token from huggingface_hub's standard location (HF_TOKEN or `hf auth login`)."""
    from huggingface_hub import get_token

    return get_token()


def hf_file(repo: str, path: str, token: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset", token=token)


def sample(rows: list, cap: int) -> list:
    """Deterministic sample of at most ``cap`` rows."""
    if len(rows) <= cap:
        return rows
    idx = sorted(random.Random(SEED).sample(range(len(rows)), cap))
    return [rows[i] for i in idx]


def saroku_rows(cap: int) -> Iterator[dict]:
    """karanxa/agent-action-safety-dataset (Apache-2.0): the fields the quality check and mapping use."""
    for split in ("train", "val"):
        for line in open(hf_file("karanxa/agent-action-safety-dataset", f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            yield {"id": r["id"], "goal": r.get("original_goal") or "", "action": r["action"], "is_safe": r["is_safe"],
                   "property": r.get("property"), "severity": r.get("severity"), "batch": r.get("batch_id"),
                   "domain": r.get("domain"), "source": r.get("source"), "split": split}


def asq_rows(cap: int) -> Iterator[dict]:
    """AgenticSearchQueryset/ASQ (MIT): agent-written queries per question id, joined to the source
    question for the accepted sets only."""
    import pandas as pd
    from huggingface_hub import HfApi

    files = [f.path for f in HfApi().list_repo_tree("AgenticSearchQueryset/ASQ", repo_type="dataset", recursive=True)
             if getattr(f, "path", "").endswith("/iter_queries.tsv")]
    per_set = max(1, cap // max(len(ASQ_SOURCES), 1))  # the quota is shared evenly between sets
    for name, meta in ASQ_SOURCES.items():
        n = 0
        repo, paths = meta["questions"]
        text: Dict[str, str] = {}
        for path in paths:
            local = hf_file(repo, path)
            if path.endswith(".parquet"):
                questions = pd.read_parquet(local)
                text.update(zip(questions["_id"].astype(str), questions["text"]))
            else:
                with open(local, encoding="utf-8") as fh:
                    for line in fh:
                        r = json.loads(line)
                        text[str(r["id"])] = r["question"]
        queries: Dict[str, list] = {}
        for path in sorted(f for f in files if f.split("/")[1] == name):
            df = pd.read_csv(hf_file("AgenticSearchQueryset/ASQ", path), sep="\t", dtype=str, quoting=csv.QUOTE_NONE)
            for qid, q in zip(df["qid"], df["llm_query"]):
                q = (q or "").strip() if isinstance(q, str) else ""
                if q and qid in text and q not in queries.setdefault(qid, []):
                    queries[qid].append(q)
        for qid in sample(sorted(queries), per_set - n):
            yield {"qid": qid, "set": name, "licence": meta["licence"], "question": text[qid], "queries": queries[qid][:10]}
            n += 1


def glaive_rows(cap: int) -> Iterator[dict]:
    """glaiveai/glaive-function-calling-v2 (Apache-2.0): the first user turn and the first call."""
    call = re.compile(r"<functioncall>\s*\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*'(.*?)'\s*\}\s*<\|endoftext\|>", re.S)
    rows = []
    for r in json.load(open(hf_file("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json"), encoding="utf-8")):
        chat = r.get("chat") or ""
        m, user = call.search(chat), re.search(r"USER:\s*(.*?)\s*ASSISTANT:", chat, re.S)
        if m and user and m.start() > user.start():
            rows.append({"request": user.group(1).strip(), "function": m.group(1), "arguments": m.group(2).strip()})
    yield from sample(rows, cap)


def toolace_rows(cap: int) -> Iterator[dict]:
    """Team-ACE/ToolACE (Apache-2.0): the first user turn and the first assistant call `[f(a=b)]`."""
    rows = []
    for r in json.load(open(hf_file("Team-ACE/ToolACE", "data.json"), encoding="utf-8")):
        conv = r.get("conversations") or []
        if len(conv) >= 2 and conv[0].get("from") == "user" and conv[1].get("from") == "assistant":
            m = re.fullmatch(r"\[\s*([A-Za-z_][\w.\- ]*)\((.*)\)\s*\]", conv[1]["value"].strip(), re.S)
            if m:
                rows.append({"request": conv[0]["value"].strip(), "function": m.group(1).strip(), "arguments": m.group(2).strip()})
    yield from sample(rows, cap)


def xlam_rows(cap: int, token: str) -> Iterator[dict]:
    """Salesforce/xlam-function-calling-60k (CC BY 4.0, gated): query and its first call."""
    rows = []
    for r in json.load(open(hf_file("Salesforce/xlam-function-calling-60k", "xlam_function_calling_60k.json", token), encoding="utf-8")):
        answers = json.loads(r["answers"]) if isinstance(r.get("answers"), str) else r.get("answers") or []
        if answers and answers[0].get("name"):
            rows.append({"request": r["query"], "function": answers[0]["name"],
                         "arguments": json.dumps(answers[0].get("arguments") or {}, ensure_ascii=False)})
    yield from sample(rows, cap)


def tracesafe_rows(cap: int, token: str) -> Iterator[dict]:
    """CyCraftAI/TraceSafe (Apache-2.0, gated; evaluation only): per mutation file, the trace's user
    turn and the mutated step. The adapter reads common field names; rows it can't read are counted."""
    from huggingface_hub import HfApi

    files = [s.rfilename for s in HfApi().dataset_info("CyCraftAI/TraceSafe", token=token).siblings if s.rfilename.endswith(".jsonl")]
    for path in sorted(files):
        label = path.split("_", 2)[-1].rsplit(".", 1)[0]
        for line in open(hf_file("CyCraftAI/TraceSafe", path, token), encoding="utf-8"):
            r = json.loads(line)
            msgs = r.get("messages") or r.get("conversation") or r.get("trace") or []
            user = next((m.get("content") for m in msgs if isinstance(m, dict) and m.get("role") == "user"), None)
            step = r.get("mutated_step") or r.get("step") or r.get("tool_call")
            yield {"file": path, "label": label, "user": user if isinstance(user, str) else None,
                   "step": json.dumps(step, ensure_ascii=False) if step is not None else None}


def agentharm_rows(cap: int) -> Iterator[dict]:
    """ai-safety-institute/AgentHarm (MIT + safety-only clause; no-train): harmful and benign
    behaviours matched by id."""
    for split in ("test_public", "validation"):
        harmful = json.load(open(hf_file("ai-safety-institute/AgentHarm", f"benchmark/harmful_behaviors_{split}.json")))["behaviors"]
        benign = {b["id"]: b for b in json.load(open(hf_file("ai-safety-institute/AgentHarm", f"benchmark/benign_behaviors_{split}.json")))["behaviors"]}
        for h in harmful:
            if h["id"] in benign:
                yield {"id": f"{split}:{h['id']}", "group": h.get("id_original"), "category": h.get("category"),
                       "harmful": h["prompt"], "benign": benign[h["id"]]["prompt"]}


def squad_question_rows(cap: int) -> Iterator[dict]:
    """SQuAD 2.0 questions (CC BY-SA 4.0), as intents for TaskTracker's secondary tasks."""
    rows = []
    for name in ("train-v2.0.json", "dev-v2.0.json"):
        for article in json.loads(gh.fetch_url(sec.SQUAD.format(name=name)).decode("utf-8"))["data"]:
            for para in article["paragraphs"]:
                rows += [{"question": q["question"].strip(), "title": article.get("title")} for q in para["qas"][:1]]
    yield from sample(rows, cap)


def wildchat_code_rows(cap: int) -> Iterator[dict]:
    """allenai/WildChat-1M (ODC-BY): English, non-toxic, unredacted first user turns that ask for
    code (intent.json's patterns), as benign coding questions for RedCode's risky calls."""
    from app.security.intent import code_requested

    n = 0
    for r in sec.hf_rows("allenai/WildChat-1M", split="train", streaming=True):
        if r.get("toxic") or r.get("redacted") or r.get("language") != "English":
            continue
        user = next((t["content"] for t in r.get("conversation") or [] if t.get("role") == "user" and t.get("content")), "")
        if 20 <= len(user) <= 600 and code_requested(user):
            yield {"question": user.strip()}
            n += 1
            if n >= min(cap, 3000):
                return


def security_rows(name: str) -> Callable[[int], Iterator[dict]]:
    """Sets already used by ContextDefender, fetched the same way as fetch_security_sources.py."""
    def rows(cap: int) -> Iterator[dict]:
        if name == "xstest":
            yield from ({"text": r["prompt"], "type": r["type"]} for r in sec.hf_rows("natolambert/xstest-v2-copy", split="prompts"))
        elif name == "harmbench":
            for r in csv.DictReader(io.StringIO(gh.fetch_url(sec.HARMBENCH_CSV).decode("utf-8"))):
                if r.get("FunctionalCategory") == "standard":
                    yield {"text": r["Behavior"], "category": r.get("SemanticCategory")}
        elif name == "aegis":
            for r in sec.hf_rows("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train"):
                prompt = (r.get("prompt") or "").strip()
                if prompt and prompt != "REDACTED":
                    yield {"text": prompt, "label": r.get("prompt_label"), "categories": r.get("violated_categories")}
        else:
            yield from {"injecagent": sec.injecagent_rows, "tasktracker": sec.tasktracker_rows,
                        "notinject": sec.notinject_rows}[name]()
    return rows


SOURCES: Dict[str, Callable] = {
    "saroku": saroku_rows, "asq": asq_rows, "glaive": glaive_rows, "toolace": toolace_rows,
    "xlam": xlam_rows, "tracesafe": tracesafe_rows, "agentharm": agentharm_rows,
    "agent_safetybench": lambda cap: gh.agent_safetybench_rows(), "toolemu": lambda cap: gh.toolemu_rows(),
    "redcode": lambda cap: gh.redcode_rows(), "privacylens": lambda cap: gh.privacylens_rows(),
    "agentdam": lambda cap: gh.agentdam_rows(), "shade": lambda cap: gh.shade_rows(),
    "osharm": lambda cap: gh.osharm_rows(), "agentdojo_tasks": lambda cap: gh.agentdojo_task_rows(),
    "squad_questions": squad_question_rows, "wildchat_code": wildchat_code_rows,
    **{n: security_rows(n) for n in ("xstest", "harmbench", "aegis", "injecagent", "tasktracker", "notinject")},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="scratch folder outside the repository")
    ap.add_argument("--only", nargs="*", default=None, help=f"fetch only these: {' '.join(SOURCES)}")
    ap.add_argument("--cap", type=int, default=20000, help="rows kept per large source (deterministic sample)")
    args = ap.parse_args()
    if BACKEND.parent in args.out.resolve().parents:
        ap.error("--out must be outside the repository")
    args.out.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    counts: Dict[str, object] = {}
    for name, fn in SOURCES.items():
        if args.only is not None and name not in args.only:
            continue
        if name in GATED and not token:
            counts[name] = "skipped: gated, no Hugging Face token (run `hf auth login`)"
            continue
        try:
            rows = fn(args.cap, token) if name in GATED else fn(args.cap)
            part = args.out / f"{name}.jsonl.part"
            counts[name] = sec.write_jsonl(part, rows)
            part.replace(args.out / f"{name}.jsonl")  # a failed fetch never leaves a half file in place
        except Exception as exc:  # noqa: BLE001 - one failed source never stops the run
            counts[name] = f"failed: {type(exc).__name__}: {str(exc)[:160]}"
        print(json.dumps({name: counts[name]}), flush=True)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
