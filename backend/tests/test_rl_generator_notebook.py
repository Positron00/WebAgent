"""The Colab driver is committed clean (rl-generator SEC-211, SNFR-24): no outputs, no secrets,
and it only calls the repository's scripts."""
import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "scripts" / "rl_generator" / "colab_train.ipynb"
KEY_LIKE = re.compile(r"(sk-[A-Za-z0-9]{16,}|tgp_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{16,}|[A-Fa-f0-9]{40,})")


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def test_no_outputs_or_execution_counts():
    for c in cells():
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_no_key_like_strings():
    assert not KEY_LIKE.search(NOTEBOOK.read_text())
    src = "".join("".join(c["source"]) for c in cells())
    assert 'userdata.get("TOGETHER_API_KEY")' in src


def test_calls_only_repository_scripts():
    src = "".join("".join(c["source"]) for c in cells() if c["cell_type"] == "code")
    assert "scripts/rl_generator/train.py" in src and "scripts/rl_generator/sample.py" in src
    assert "GRPOTrainer" not in src and "import trl" not in src
    assert "local/generated" in src and "print(os.environ" not in src
