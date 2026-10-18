"""RNFR-02: the scheduler process loads no ML library, no agents and no security guard."""
import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "transformers", "sentence_transformers", "chromadb", "langchain", "langchain_core",
         "langchain_community", "langgraph", "app.agents", "app.security", "app.services.task_manager",
         "fastapi")

SCRIPT = """
import json, sys
import app.scheduler
import app.services.topic_runs
heavy = %r
loaded = sorted(m for m in sys.modules if any(m == h or m.startswith(h + ".") for h in heavy))
print(json.dumps(loaded))
""" % (HEAVY,)


def test_importing_the_scheduler_loads_nothing_heavy():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=BACKEND, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"the scheduler imported {loaded}"


def test_the_health_flag_runs_as_a_module():
    done = subprocess.run([sys.executable, "-m", "app.scheduler", "--health"], cwd=BACKEND,
                          capture_output=True, text=True, timeout=120,
                          env={**__import__("os").environ, "WEBAGENT_DATA_DIR": str(BACKEND / "tests" / "no-such-dir")})
    assert done.returncode == 1
    assert done.stdout.strip().splitlines()[-1].startswith("unhealthy")
