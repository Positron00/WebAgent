"""AST findings for generated code (checkpoint 2)."""
from __future__ import annotations

import ast
from typing import FrozenSet, Optional, Set

_NETWORK_MODULES = frozenset(
    {"socket", "http", "urllib", "requests", "httpx", "aiohttp", "ftplib", "smtplib", "paramiko"}
)
_PROCESS_MODULES = frozenset({"subprocess", "pty"})
_EVAL_NAMES = frozenset({"eval", "exec", "compile", "__import__"})
_NATIVE_MODULES = frozenset({"ctypes", "cffi"})


def _module_root(name: str) -> str:
    return name.split(".")[0]


def _is_write_mode(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        mode = node.value
        return any(ch in mode for ch in "wax+")
    return False


def _path_risky(path_node: ast.AST) -> bool:
    if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
        p = path_node.value
        return p.startswith("/") or ".." in p
    return False


def _call_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
    return None


def find_findings(code: str, work_folder: Optional[str] = None) -> FrozenSet[str]:
    findings: Set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset({"unparseable"})

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _module_root(alias.name)
                if root in _NETWORK_MODULES:
                    findings.add("network")
                if root in _PROCESS_MODULES:
                    findings.add("process")
                if root in _NATIVE_MODULES:
                    findings.add("native")
                if root == "importlib":
                    findings.add("eval")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = _module_root(node.module)
            if root in _NETWORK_MODULES:
                findings.add("network")
            if root in _PROCESS_MODULES:
                findings.add("process")
            if root in _NATIVE_MODULES:
                findings.add("native")
            if root == "importlib":
                findings.add("eval")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _EVAL_NAMES:
                findings.add("eval")
            if name == "os.system" or name == "os.popen" or (
                name and name.startswith("os.exec")
            ) or (name and name.startswith("os.spawn")):
                findings.add("process")
            if name in ("os.environ", "os.getenv"):
                findings.add("env")
            if name == "shutil.rmtree" or name == "os.remove":
                findings.add("filesystem")
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                mode_node = node.args[1] if len(node.args) > 1 else None
                for kw in node.keywords:
                    if kw.arg == "mode":
                        mode_node = kw.value
                path_node = node.args[0] if node.args else None
                if mode_node and path_node and _is_write_mode(mode_node) and _path_risky(path_node):
                    findings.add("filesystem")

    return frozenset(findings)


findings = find_findings
