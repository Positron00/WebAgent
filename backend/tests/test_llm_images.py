"""The optional self-hosted LLM images build and start (IMP-067, IMP-128 task 4.5).

Found in the 3.5.0 alignment review: llm_server.py imported the backend package, which the
images don't contain; the MLflow image passed "${MODEL_PATH}" literally (exec-form CMD); and
the two Dockerfiles assumed different build contexts. Checked statically: building the
images needs a multi-gigabyte PyTorch base.
"""
import ast
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
SERVER = BACKEND / "standalone" / "llm_server.py"
DOCKERFILES = [BACKEND / "app" / "services" / name for name in ("Dockerfile.llm", "Dockerfile.llm-mlflow")]


def test_server_runs_without_the_backend_package():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "app" not in imported


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_copied_files_exist_in_the_backend_context(dockerfile):
    text = dockerfile.read_text(encoding="utf-8")
    assert "-t webagent/" in text and text.rstrip().count(" backend") >= 1  # build command documented
    for source in re.findall(r"^COPY\s+(?:--\S+\s+)?(\S+)\s+\S+", text, re.M):
        assert (BACKEND / source).exists(), f"{dockerfile.name}: COPY {source} not found under backend/"


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_command_expands_settings_and_runs_as_non_root(dockerfile):
    text = dockerfile.read_text(encoding="utf-8")
    cmd = text[text.rindex("\nCMD"):]
    exec_form = cmd.lstrip("\nCMD").lstrip().startswith("[")
    # Exec form never expands ${...} unless it runs a shell itself.
    assert not (exec_form and "${" in cmd and '"sh", "-c"' not in cmd)
    users = re.findall(r"^USER\s+(\S+)", text, re.M)
    assert users and users[-1] not in ("root", "0")
