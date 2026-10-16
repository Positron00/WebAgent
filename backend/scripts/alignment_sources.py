"""Source adapters for fetch_alignment_sources.py: each yields normalised rows with only the fields
the alignment builder uses (checkpoint2-intent model-and-data design §2). Nothing here prints text.

Agent-task benchmarks from GitHub (Agent-SafetyBench, ToolEmu, RedCode, PrivacyLens, AgentDAM,
SHADE-Arena, OS-Harm, AgentDojo tasks). Python sources are parsed with `ast`, never executed.
"""
from __future__ import annotations

import ast
import json
import textwrap
import urllib.request
from typing import Dict, Iterator, List, Optional

GH_RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
GH_TREE = "https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"


def fetch_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def gh_json(repo: str, ref: str, path: str):
    return json.loads(fetch_url(GH_RAW.format(repo=repo, ref=ref, path=path)).decode("utf-8"))


def gh_paths(repo: str, ref: str, prefix: str) -> List[str]:
    tree = json.loads(fetch_url(GH_TREE.format(repo=repo, ref=ref)).decode("utf-8"))["tree"]
    return sorted(t["path"] for t in tree if t["type"] == "blob" and t["path"].startswith(prefix))


def agent_safetybench_rows() -> Iterator[dict]:
    """thu-coai/Agent-SafetyBench (MIT): every case with what the builder's source filter needs
    (risk categories, failure modes, environment names); the builder drops derived cases."""
    for case in gh_json("thu-coai/Agent-SafetyBench", "main", "data/released_data.json"):
        envs = [e.get("name") for e in case.get("environments") or [] if e.get("name")]
        modes = sorted({int(m) for m in str(case.get("failure_modes") or "").split(",") if m.strip().isdigit()})
        yield {"id": case["id"], "instruction": case.get("instruction") or "", "envs": envs,
               "fulfillable": int(case.get("fulfillable", 0)), "risks": case.get("risks") or [],
               "failure_modes": modes, "has_dialog": "dialog" in case}


def toolemu_rows() -> Iterator[dict]:
    """ryoungj/ToolEmu (Apache-2.0) assets/all_cases.json: instruction, toolkits, risky actions."""
    for case in gh_json("ryoungj/ToolEmu", "main", "assets/all_cases.json"):
        yield {"name": case.get("name"), "instruction": case.get("User Instruction") or "",
               "toolkits": case.get("Toolkits") or [], "risky_actions": case.get("Potential Risky Actions") or []}


def redcode_rows() -> Iterator[dict]:
    """AI-secure/RedCode (dataset CC BY 4.0) RedCode-Exec cases: code and its summary (never run)."""
    for path in gh_paths("AI-secure/RedCode", "main", "dataset/RedCode-Exec/"):
        if not path.endswith(".json"):
            continue
        name = path.rsplit("/", 1)[-1].split("_")[0]  # e.g. index10: one risk type per file
        kind = path.split("/")[2].split("2text")[0]  # py or bash
        for case in gh_json("AI-secure/RedCode", "main", path):
            if (case.get("Code") or "").strip():
                yield {"file": f"{kind}:{name}", "index": case.get("Index"), "language": case.get("Language"),
                       "code": case["Code"], "summary": case.get("Text_summary") or ""}


def privacylens_rows() -> Iterator[dict]:
    """SALT-NLP/PrivacyLens (MIT; card CC BY 4.0) data/main_data.json: instruction, final action
    (a tool name) and the sensitive items that must not leak."""
    for case in gh_json("SALT-NLP/PrivacyLens", "main", "data/main_data.json"):
        t = case.get("trajectory") or {}
        yield {"name": case.get("name"), "instruction": t.get("user_instruction") or "",
               "final_action": t.get("final_action") or "", "items": t.get("sensitive_info_items") or [],
               "toolkits": t.get("toolkits") or []}


def agentdam_rows() -> Iterator[dict]:
    """facebookresearch/ai-agent-privacy AgentDAM (CC BY-NC 4.0): web tasks, the sensitive data the
    task doesn't need and the content it must include. Test only."""
    for site in ("gitlab", "reddit", "shopping"):
        for task in gh_json("facebookresearch/ai-agent-privacy", "main", f"agentdam/data/wa_format/{site}_privacy.json"):
            required = [c for p in (task.get("eval") or {}).get("program_html") or []
                        for c in ((p.get("required_contents") or {}).get("must_include") or [])]
            yield {"uid": f"{site}:{task.get('task_id')}", "site": site, "intent": task.get("intent") or "",
                   "sensitive": task.get("sensitive_data") or [], "required": required}


def _ast_strings(node, shallow: bool = False) -> Dict[str, str]:
    """String constants assigned in ``node`` (only its direct body when ``shallow``)."""
    consts: Dict[str, str] = {}
    for n in (node.body if shallow else ast.walk(node)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = _ast_eval(n.value, consts)
            if v is not None:
                consts[n.targets[0].id] = v
    return consts


def _ast_eval(node, consts: Dict[str, str]) -> Optional[str]:
    """A string literal, f-string, + or textwrap.dedent(...)/.strip() of them, from known names."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
                continue
            inner = v.value if isinstance(v, ast.FormattedValue) else None
            key = inner.id if isinstance(inner, ast.Name) else inner.attr if isinstance(inner, ast.Attribute) else None
            if key not in consts:
                return None
            out.append(consts[key])
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _ast_eval(node.left, consts), _ast_eval(node.right, consts)
        return a + b if a is not None and b is not None else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "strip":
            v = _ast_eval(node.func.value, consts)
            return v.strip() if v is not None else None
        if node.func.attr == "dedent" and node.args:
            v = _ast_eval(node.args[0], consts)
            return textwrap.dedent(v) if v is not None else None
    return None


def _property_strings(tree) -> Dict[str, str]:
    """Return values of methods/properties that evaluate to strings, resolved in two passes."""
    consts = _ast_strings(tree)
    for n in ast.walk(tree):  # self.name = "..." in __init__
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute):
            v = _ast_eval(n.value, consts)
            if v is not None:
                consts[n.targets[0].attr] = v
    for _ in range(2):
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for ret in [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]:
                    v = _ast_eval(ret.value, consts)
                    if v is not None:
                        consts[fn.name] = v
    return consts


def _ast_eval_in(cls, name: str, consts: Dict[str, str]) -> bool:
    """True when the class itself assigns ``name`` (possibly from an f-string of known names)."""
    for n in cls.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            v = _ast_eval(n.value, consts)
            if v is not None:
                consts[name] = v
                return True
    return False


def shade_rows() -> Iterator[dict]:
    """jkutaso/SHADE-Arena public split (MIT; no-train): each task pair's main task and side task."""
    repo = "jkutaso/SHADE-Arena"
    for path in gh_paths(repo, "master", "task_pairs/"):
        if not path.endswith("/task_definition.py"):
            continue
        tree = ast.parse(fetch_url(GH_RAW.format(repo=repo, ref="master", path=path)).decode("utf-8"))
        props = _property_strings(tree)
        main, side = props.get("user_query"), props.get("secondary_task")
        if main and side:
            yield {"task": path.split("/")[1], "main": main.strip(), "side": side.strip()}


def osharm_rows() -> Iterator[dict]:
    """tml-epfl/os-harm (Apache-2.0; no-train) task configs: instruction plus injection goals
    (injection) or the misbehaviour explanation (misbehavior); misuse tasks as they are."""
    repo = "tml-epfl/os-harm"
    for category in ("injection", "misbehavior", "misuse"):
        index = gh_json(repo, "main", f"evaluation_examples/test_{category}.json")
        for app, ids in index.items():
            for task_id in ids:
                cfg = gh_json(repo, "main", f"evaluation_examples/examples/{app}/{task_id}.json")
                goals = [g for inj in cfg.get("injections") or [] for g in inj.get("goals") or []]
                yield {"id": task_id, "category": category, "app": app, "instruction": cfg.get("instruction") or "",
                       "goals": goals, "explanation": cfg.get("explanation") or ""}


def agentdojo_task_rows() -> Iterator[dict]:
    """ethz-spylab/agentdojo (MIT) v1 suites: user-task prompts and injection-task goals."""
    repo = "ethz-spylab/agentdojo"
    for suite in ("banking", "slack", "travel", "workspace"):
        for kind, attr in (("user", "PROMPT"), ("injection", "GOAL")):
            src = fetch_url(GH_RAW.format(repo=repo, ref="main",
                                          path=f"src/agentdojo/default_suites/v1/{suite}/{kind}_tasks.py")).decode("utf-8")
            tree = ast.parse(src)
            module = _ast_strings(tree, shallow=True)
            for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                own = _ast_strings(cls, shallow=True)
                consts = {**module, **_ast_strings(cls), **own}
                if attr in own or (attr in consts and _ast_eval_in(cls, attr, consts)):
                    yield {"suite": suite, "kind": kind, "task": cls.name, "text": consts[attr]}
