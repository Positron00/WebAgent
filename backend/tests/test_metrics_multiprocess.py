"""With several uvicorn workers, /metrics must sum every worker's values (PROMETHEUS_MULTIPROC_DIR)."""
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

WORKER = """
from app.core import metrics
metrics.GUARD_HOP_SECONDS.labels(agent="web_research").observe(0.05)
metrics.GUARD_DECISIONS.labels(checkpoint="1", action="converse", outcome="allow", shadow="true").inc()
"""

SCRAPE = """
from app.core import metrics
text = metrics.metrics_payload().decode()
hop = [l for l in text.splitlines() if l.startswith('webagent_guard_hop_seconds_count{agent="web_research"}')]
dec = [l for l in text.splitlines() if l.startswith("webagent_guard_decisions_total{") and 'outcome="allow"' in l]
print(hop[0].split()[-1], dec[0].split()[-1])
"""


def _run(code, env):
    return subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True, check=True)


def test_metrics_from_two_workers_are_summed(tmp_path):
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path), "PYTHONPATH": str(BACKEND), "WEBAGENT_ENV": "dev"}
    _run(WORKER, env)  # worker 1
    _run(WORKER, env)  # worker 2 (another process)
    out = _run(SCRAPE, env).stdout.strip().splitlines()[-1]
    assert out.split() == ["2.0", "2.0"]


def test_single_process_uses_the_app_registry(monkeypatch):
    from app.core import metrics

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    metrics.GUARD_HOP_SECONDS.labels(agent="single").observe(0.01)
    assert b'webagent_guard_hop_seconds_count{agent="single"}' in metrics.metrics_payload()
