#!/usr/bin/env python3
"""Download the public source datasets for ContextDefender into a raw folder (counts only).

Run from backend/:
    python scripts/fetch_security_sources.py --out <raw dir> [--wildchat 12000] [--llmail 6000] [--nemotron 2500]

Each source's licence is recorded in data/security/README.md. Large sets are streamed and sampled.
Nothing here prints example text.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

HARMBENCH_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/"
    "harmbench_behaviors_text_all.csv"
)
INJECAGENT = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/{name}"
AGENTDOJO = "https://raw.githubusercontent.com/ethz-spylab/agentdojo/main/src/agentdojo/{path}"
AGENTDOJO_SUITES = ("banking", "slack", "travel", "workspace")
BIPIA_BASE = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/"
BIPIA_ATTACKS = {
    "text_attack_train": BIPIA_BASE + "text_attack_train.json",
    "text_attack_test": BIPIA_BASE + "text_attack_test.json",
    # Code attacks (added 2026-10-12); BIPIA's code contexts (Stack Exchange, CC BY-SA) are not used.
    "code_attack_train": BIPIA_BASE + "code_attack_train.json",
    "code_attack_test": BIPIA_BASE + "code_attack_test.json",
}
TASKTRACKER = "https://raw.githubusercontent.com/microsoft/TaskTracker/main/task_tracker/dataset_creation/datasets/{name}"
SQUAD = "https://rajpurkar.github.io/SQuAD-explorer/dataset/{name}"
SEP_URL = ("https://raw.githubusercontent.com/egozverev/Should-It-Be-Executed-Or-Processed/main/"
           "SEP_dataset/SEP_dataset.json")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def hf_rows(name: str, config=None, split="train", streaming=False) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split, streaming=streaming)
    yield from ds


def fetch_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _locate(haystack: str, needle: str):
    i = haystack.find(needle)
    return [i, i + len(needle)] if i >= 0 and needle else None


def nvidia_rows() -> Iterator[dict]:
    """nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 (CC BY 4.0): the record holding the
    injection, as JSON tool output with the injection's span, and the same record without it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1", "train.jsonl", repo_type="dataset")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        inj = r["injection"]["injection_text"]
        holder = _holder(r["environment"], inj)
        if holder is None:
            continue
        text = json.dumps(holder, indent=1, ensure_ascii=False)
        needle = json.dumps(inj, ensure_ascii=False)[1:-1]
        span = _locate(text, needle)
        if span is None:
            continue
        clean = {k: (v.replace(inj, "").strip() if isinstance(v, str) else v) for k, v in holder.items()}
        group = f"nvidia:{r['id']}"
        yield {"text": text, "span": span, "label": "injection", "group": group, "domain": r["domain"]}
        yield {"text": json.dumps(clean, indent=1, ensure_ascii=False), "span": None, "label": "benign",
               "group": group, "domain": r["domain"]}


def _holder(env, inj: str):
    """The innermost record (dict) with a string field containing the injection text."""
    found = None
    stack = [env]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if any(isinstance(v, str) and inj in v for v in node.values()):
                found = node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def injecagent_rows() -> Iterator[dict]:
    """InjecAgent (MIT): tool responses containing an attacker instruction, and the same response
    with the instruction removed."""
    for name in ("test_cases_dh_base.json", "test_cases_ds_base.json"):
        for i, case in enumerate(json.loads(fetch_url(INJECAGENT.format(name=name)).decode("utf-8"))):
            text, inj = case["Tool Response"], case["Attacker Instruction"]
            span = _locate(text, inj)
            if span is None:
                continue
            group = f"injecagent:{name}:{i}"
            meta = {"group": group, "template": case.get("User Tool"), "attack": inj,
                    "attack_type": case.get("Attack Type"), "user_instruction": case.get("User Instruction"),
                    "attacker_tools": case.get("Attacker Tools"), "kind": name.split("_")[2]}
            yield {"text": text, "span": span, "label": "injection", **meta}
            yield {"text": (text[:span[0]] + text[span[1]:]).strip(), "span": None, "label": "benign", **meta}


def _ast_strings(tree, ast) -> Dict[str, str]:
    """Module- and class-level string constants (NAME = "..." / f"..." resolvable from earlier ones)."""
    consts: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _ast_eval(node.value, consts, ast)
            if value is not None:
                consts[node.targets[0].id] = value
    return consts


def _ast_eval(node, consts: Dict[str, str], ast):
    """Evaluate a string literal, f-string, + or implicit concatenation from known constants; else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                inner = v.value
                key = inner.id if isinstance(inner, ast.Name) else inner.attr if isinstance(inner, ast.Attribute) else None
                if key not in consts:
                    return None
                out.append(consts[key])
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _ast_eval(node.left, consts, ast), _ast_eval(node.right, consts, ast)
        return a + b if a is not None and b is not None else None
    return None


def agentdojo_rows() -> Iterator[dict]:
    """AgentDojo (MIT): environment data with its "important instructions" attack (goals from the
    suites' injection tasks) at each injection point, and the same data with the default benign value.
    The Python sources are parsed, never executed."""
    import ast
    import re

    import yaml

    attack_src = fetch_url(AGENTDOJO.format(path="attacks/important_instructions_attacks.py")).decode("utf-8")
    attack_consts = _ast_strings(ast.parse(attack_src), ast)
    template = next((v for k, v in attack_consts.items() if "{goal}" in v), None)
    if template is None:
        return
    for suite in AGENTDOJO_SUITES:
        base = f"data/suites/{suite}"
        env = fetch_url(AGENTDOJO.format(path=f"{base}/environment.yaml")).decode("utf-8")
        if suite == "workspace":
            for inc in ("calendar", "cloud_drive", "inbox"):
                env += "\n" + fetch_url(AGENTDOJO.format(path=f"{base}/include/{inc}.yaml")).decode("utf-8")
        defaults = yaml.safe_load(fetch_url(AGENTDOJO.format(path=f"{base}/injection_vectors.yaml")).decode("utf-8"))
        defaults = {k: (v.get("default") if isinstance(v, dict) else v) for k, v in defaults.items()}
        tasks_src = fetch_url(AGENTDOJO.format(path=f"default_suites/v1/{suite}/injection_tasks.py")).decode("utf-8")
        tree = ast.parse(tasks_src)
        goals = []
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            consts = _ast_strings(tree, ast)
            consts.update(_ast_strings(cls, ast))
            goal = consts.get("GOAL")
            if goal:
                goals.append(goal)
        for vi, vector in enumerate(defaults):
            marker = "{" + vector + "}"
            pos = env.find(marker)
            if pos < 0:
                continue
            for gi, goal in enumerate(goals):
                attack = template.format(goal=goal, user="the user", model="the AI assistant", tools="")
                window = env[max(0, pos - 700): pos + 700]
                for other, value in defaults.items():
                    if other != vector:
                        window = window.replace("{" + other + "}", str(value or ""))
                before, after = window.split(marker, 1)
                group = f"agentdojo:{suite}:{vector}"
                yield {"text": before + attack + after, "span": [len(before), len(before) + len(attack)],
                       "label": "injection", "group": group, "suite": suite}
                if gi == 0:
                    yield {"text": before + str(defaults[vector] or "") + after, "span": None, "label": "benign",
                           "group": group, "suite": suite}


PROMPT_GUARD_REPOS = {"22M": "meta-llama/Llama-Prompt-Guard-2-22M", "86M": "meta-llama/Llama-Prompt-Guard-2-86M"}
PROMPT_GUARD_NOTICE = ("Llama 4 is licensed under the Llama 4 Community License, "
                       "Copyright © Meta Platforms, Inc. All Rights Reserved.\n")


def _hf_token() -> Optional[str]:
    """HF token from the environment or the repo's .env.local (HuggingFace_API_Key / HF_TOKEN); never printed."""
    import os
    import re

    for name in ("HF_TOKEN", "HuggingFace_API_Key"):
        if os.environ.get(name):
            return os.environ[name]
    env_file = Path(__file__).resolve().parents[2] / ".env.local"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:HF_TOKEN|HuggingFace_API_Key)\s*=\s*['\"]?([^'\"\s#]+)", line)
            if m:
                return m.group(1)
    return None


def fetch_prompt_guard(size: str, out: Path) -> int:
    """Download Llama Prompt Guard 2 (gated; the account must have been granted access) with its
    licence and use policy, and write the NOTICE the Llama 4 Community License requires."""
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download

    # Download to a temporary folder and copy files in only when complete: a gated or interrupted
    # download must not leave a half-installed model (and files are overwritten in place, never the folder).
    with tempfile.TemporaryDirectory() as tmp:
        try:
            snapshot_download(PROMPT_GUARD_REPOS[size], token=_hf_token(), local_dir=tmp,
                              allow_patterns=["*.json", "*.safetensors", "*.model", "LICENSE*", "USE_POLICY*", "README.md"])
        except Exception as exc:
            raise SystemExit(f"Prompt Guard download failed ({type(exc).__name__}); the Hugging Face account "
                             "needs access to the gated repository") from None
        src = Path(tmp)
        (src / "NOTICE").write_text(PROMPT_GUARD_NOTICE, encoding="utf-8")
        missing = [f for f in ("config.json", "LICENSE", "NOTICE") if not (src / f).is_file()]
        if missing or not any(src.glob("*.safetensors")):
            raise SystemExit(f"Prompt Guard download incomplete (missing {missing or 'weights'})")
        out.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copyfile(f, out / f.name)
    print(json.dumps({"prompt_guard": PROMPT_GUARD_REPOS[size], "dir": str(out)}))
    return 0


def tasktracker_rows() -> Iterator[dict]:
    """TaskTracker's secondary-task lists (MIT): its own train and test injection lists, trivia
    questions and translated tasks, one instruction per line."""
    for name, split in (("saved_injections.txt", "train"), ("saved_injections_test.txt", "test"),
                        ("trivia_pairs.txt", None), ("translated_secondary_tasks.txt", None)):
        for line in fetch_url(TASKTRACKER.format(name=name)).decode("utf-8").splitlines():
            if line.strip():
                yield {"text": line.strip(), "split": split, "kind": name.split(".")[0]}


def squad_rows(limit: int = 6000) -> Iterator[dict]:
    """SQuAD 2.0 paragraphs (CC BY-SA 4.0), unique, as TaskTracker-style carriers; never combined with
    CC BY-NC text (build_security_dataset.py pairs them only with TaskTracker's MIT lists)."""
    seen = set()
    for name in ("train-v2.0.json", "dev-v2.0.json"):
        for article in json.loads(fetch_url(SQUAD.format(name=name)).decode("utf-8"))["data"]:
            for para in article["paragraphs"]:
                ctx = para["context"].strip()
                if 300 <= len(ctx) <= 2400 and ctx not in seen:
                    seen.add(ctx)
                    yield {"text": ctx, "title": article.get("title")}
                    if len(seen) >= limit:
                        return


def alpaca_rows(limit: int = 6000) -> Iterator[dict]:
    """tatsu-lab/alpaca (CC BY-NC 4.0; generated with text-davinci-003): instructions without an input,
    used as TaskTracker-style injected tasks in WildChat pages only (never in CC BY-SA text)."""
    n = 0
    for r in hf_rows("tatsu-lab/alpaca"):
        if not (r.get("input") or "").strip() and 15 <= len(r["instruction"]) <= 300:
            yield {"text": r["instruction"].strip()}
            n += 1
            if n >= limit:
                return


def cosmopedia_rows(per_config: int = 3000) -> Iterator[dict]:
    """HuggingFaceTB/cosmopedia (Apache-2.0; synthetic, generated with Mixtral-8x7B-Instruct): how-to
    articles (`wikihow`, written from wikiHow titles) and web articles, as benign instruction-rich pages."""
    for config in ("wikihow", "web_samples_v2"):
        n = 0
        for r in hf_rows("HuggingFaceTB/cosmopedia", config, streaming=True):
            text = (r.get("text") or "").strip()
            if 300 <= len(text) <= 6000:
                yield {"text": text, "config": config}
                n += 1
                if n >= per_config:
                    break


def oasst_rows(limit: int = 4000) -> Iterator[dict]:
    """OpenAssistant/oasst2 (Apache-2.0): English assistant replies, as benign instruction-rich text."""
    n = 0
    for r in hf_rows("OpenAssistant/oasst2", split="train", streaming=True):
        text = (r.get("text") or "").strip()
        if r.get("role") == "assistant" and r.get("lang") == "en" and 300 <= len(text) <= 4000:
            yield {"text": text}
            n += 1
            if n >= limit:
                return


def bipia_attack_rows() -> Iterator[dict]:
    """BIPIA (MIT) text and code attacks, with their train/test split names."""
    for name, url in BIPIA_ATTACKS.items():
        for category, attacks in json.loads(fetch_url(url).decode("utf-8")).items():
            for attack in attacks:
                yield {"text": attack, "category": category, "split": name}


def sep_rows() -> Iterator[dict]:
    """SEP (MIT): task data with a probe instruction added at its start or end (span exact), and the
    same data without it; `probe` is the probe id, so splits can hold whole probes out."""
    for i, e in enumerate(json.loads(fetch_url(SEP_URL).decode("utf-8"))):
        clean, instructed = e["prompt_clean"], e["prompt_instructed"]
        if instructed.startswith(clean):
            span = [len(clean), len(instructed)]
        elif instructed.endswith(clean):
            span = [0, len(instructed) - len(clean)]
        else:
            continue
        a, b = span
        while a < b and instructed[a].isspace():
            a += 1
        while b > a and instructed[b - 1].isspace():
            b -= 1
        if b - a < 8:
            continue
        meta = {"group": f"sep:{i}", "probe": int(e["info"]["appended_task_id"]),
                "insistent": bool(e["info"].get("is_insistent"))}
        yield {"text": instructed, "span": [a, b], "label": "injection", **meta}
        yield {"text": clean, "span": None, "label": "benign", **meta}


def bipia_email_rows() -> Iterator[dict]:
    """BIPIA's EmailQA contexts (from OpenAI Evals, MIT), as carriers for BIPIA's own attacks."""
    for split in ("train", "test"):
        for line in fetch_url(BIPIA_BASE + f"email/{split}.jsonl").decode("utf-8").splitlines():
            if line.strip():
                yield {"text": json.loads(line)["context"], "split": split}


def notinject_rows() -> Iterator[dict]:
    """leolee99/NotInject (MIT): benign prompts full of injection trigger words (over-defence test)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    for sub in ("one", "two", "three"):
        path = hf_hub_download("leolee99/NotInject", f"data/NotInject_{sub}-00000-of-00001.parquet", repo_type="dataset")
        for r in pd.read_parquet(path).to_dict("records"):
            yield {"text": r["prompt"], "category": r.get("category"), "subset": sub}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="raw data folder (required unless --prompt-guard)")
    ap.add_argument("--wildchat", type=int, default=12000, help="English, non-toxic conversations to keep")
    ap.add_argument("--llmail", type=int, default=6000, help="unique email bodies to keep")
    ap.add_argument("--nemotron", type=int, default=2500, help="synthetic PII documents to keep")
    ap.add_argument("--prompt-guard", choices=sorted(PROMPT_GUARD_REPOS), default=None,
                    help="only download Llama Prompt Guard 2 into backend/models/security/prompt_guard")
    ap.add_argument("--only", nargs="*", default=None, help="fetch only these sources (e.g. nvidia_ipi injecagent agentdojo)")
    args = ap.parse_args()
    if args.prompt_guard:
        return fetch_prompt_guard(args.prompt_guard, Path(__file__).resolve().parents[1] / "models" / "security" / "prompt_guard")
    if args.out is None:
        ap.error("--out is required")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    counts = {}

    # Indirect injection in agent tool data (approved 2026-10-12): NVIDIA for training, InjecAgent for
    # test only, AgentDojo for the probe set.
    for name, rows_fn in (("nvidia_ipi", nvidia_rows), ("injecagent", injecagent_rows), ("agentdojo", agentdojo_rows),
                          ("sep", sep_rows), ("bipia_email", bipia_email_rows), ("notinject", notinject_rows),
                          ("bipia", bipia_attack_rows), ("tasktracker", tasktracker_rows), ("squad", squad_rows),
                          ("alpaca", alpaca_rows), ("cosmopedia", cosmopedia_rows), ("oasst", oasst_rows)):
        if args.only is None or name in args.only:
            counts[name] = write_jsonl(out / f"{name}.jsonl", rows_fn())
    if args.only is not None:
        print(json.dumps(counts, indent=2))
        return 0

    # Direct injection / benign user prompts
    rows = []
    for split in ("train", "test"):
        rows += [{"text": r["text"], "label": int(r["label"]), "split": split} for r in hf_rows("deepset/prompt-injections", split=split)]
    counts["deepset"] = write_jsonl(out / "deepset.jsonl", rows)

    rows = []
    for split in ("train", "validation", "test"):
        rows += [{"text": r["text"], "label": int(r["label"]), "split": split} for r in hf_rows("S-Labs/prompt-injection-dataset", split=split)]
    counts["slabs"] = write_jsonl(out / "slabs.jsonl", rows)

    # Jailbreak and regular prompts
    rows = []
    for cfg in ("jailbreak_2023_05_07", "jailbreak_2023_12_25", "regular_2023_05_07", "regular_2023_12_25"):
        rows += [{"text": r["prompt"], "jailbreak": bool(r["jailbreak"]), "config": cfg} for r in hf_rows("TrustAIRLab/in-the-wild-jailbreak-prompts", cfg)]
    counts["trustairlab"] = write_jsonl(out / "trustairlab.jsonl", rows)

    # Harmful / benign behaviours
    rows = [{"text": r["Goal"], "label": "harmful", "category": r.get("Category")} for r in hf_rows("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")]
    rows += [{"text": r["Goal"], "label": "benign", "category": r.get("Category")} for r in hf_rows("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")]
    counts["jbb"] = write_jsonl(out / "jbb.jsonl", rows)

    rows = [{"text": r["prompt"], "type": r["type"]} for r in hf_rows("natolambert/xstest-v2-copy", split="prompts")]
    counts["xstest"] = write_jsonl(out / "xstest.jsonl", rows)

    reader = csv.DictReader(io.StringIO(fetch_url(HARMBENCH_CSV).decode("utf-8")))
    rows = [{"text": r["Behavior"], "category": r.get("SemanticCategory"), "functional": r.get("FunctionalCategory")}
            for r in reader if r.get("FunctionalCategory") == "standard"]
    counts["harmbench"] = write_jsonl(out / "harmbench.jsonl", rows)

    # Indirect injection
    rows = [{"text": r["text"], "label": r.get("label"), "category": r.get("category")} for r in hf_rows("treycsa/realistic-prompt-injections")]
    counts["treycsa"] = write_jsonl(out / "treycsa.jsonl", rows)

    counts["bipia"] = write_jsonl(out / "bipia.jsonl", bipia_attack_rows())

    seen, rows = set(), []
    for r in hf_rows("microsoft/llmail-inject-challenge", split="Phase1", streaming=True):
        body = (r.get("body") or "").strip()
        key = body.lower()
        if len(body) < 40 or key in seen:
            continue
        seen.add(key)
        rows.append({"text": body, "subject": r.get("subject"), "scenario": r.get("scenario")})
        if len(rows) >= args.llmail:
            break
    counts["llmail"] = write_jsonl(out / "llmail.jsonl", rows)

    # Harm / benign prompts with human safety labels (Aegis 2.0, CC BY 4.0)
    rows = []
    for r in hf_rows("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train"):
        prompt = (r.get("prompt") or "").strip()
        if not prompt or prompt == "REDACTED":
            continue
        rows.append({"text": prompt, "label": r.get("prompt_label"), "categories": r.get("violated_categories")})
    counts["aegis"] = write_jsonl(out / "aegis.jsonl", rows)

    # Sensitive data (synthetic PII documents)
    rows = []
    for r in hf_rows("nvidia/Nemotron-PII", split="train", streaming=True):
        spans = r.get("spans") or []
        if isinstance(spans, str):
            import ast

            try:
                spans = ast.literal_eval(spans or "[]")  # stored as a Python-literal string
            except (ValueError, SyntaxError):
                spans = []
        rows.append({"text": r["text"], "labels": sorted({s.get("label") for s in spans if isinstance(s, dict)}),
                     "document_type": r.get("document_type")})
        if len(rows) >= args.nemotron:
            break
    counts["nemotron"] = write_jsonl(out / "nemotron.jsonl", rows)

    # Benign real chat turns (English, flagged non-toxic, not redacted)
    rows = []
    for r in hf_rows("allenai/WildChat-1M", split="train", streaming=True):
        if r.get("toxic") or r.get("redacted") or r.get("language") != "English":
            continue
        conv = r.get("conversation") or []
        user = [t["content"] for t in conv if t.get("role") == "user" and t.get("content")]
        assistant = [t["content"] for t in conv if t.get("role") == "assistant" and t.get("content")]
        if not user:
            continue
        rows.append({"user": user[:3], "assistant": assistant[:2]})
        if len(rows) >= args.wildchat:
            break
    counts["wildchat"] = write_jsonl(out / "wildchat.jsonl", rows)

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
