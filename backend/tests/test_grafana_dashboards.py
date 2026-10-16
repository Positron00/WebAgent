"""Grafana dashboards query metrics the backend really exports, through the provisioned data source.

Found in 3.5.2: Compose never mounted the dashboards, the drift dashboard pointed at a data source
uid that didn't exist and queried metric names nothing exports (`data_drift`, `model_latency`),
and each model call observed its latency twice.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

BACKEND = Path(__file__).resolve().parents[1]
DASHBOARDS = sorted((BACKEND / "grafana" / "dashboards").glob("*.json"))
DATASOURCE_UID = "prometheus"
# Words in PromQL that aren't metric names.
PROMQL_WORDS = {
    "sum", "rate", "max", "min", "avg", "by", "without", "histogram_quantile", "vector", "or",
    "and", "on", "ignoring", "increase", "irate", "count", "le", "offset",
}


def _panels(dashboard):
    for panel in dashboard.get("panels", []):
        yield panel
        yield from panel.get("panels", [])


def _exported_metric_names():
    from app.core.metrics import WEBAGENT_REGISTRY
    import app.core.observability  # noqa: F401  (registers its metrics on WEBAGENT_REGISTRY)

    names = set()
    for metric in WEBAGENT_REGISTRY.collect():
        names.add(metric.name)
        names.update(sample.name for sample in metric.samples)
        for suffix in ("_total", "_bucket", "_count", "_sum", "_created"):
            names.add(metric.name + suffix)
    return names


def test_compose_mounts_the_dashboards_and_fixes_the_datasource_uid():
    compose = yaml.safe_load((BACKEND / "docker-compose.yml").read_text())
    volumes = compose["services"]["grafana"]["volumes"]
    assert any(v.startswith("./grafana/dashboards:/var/lib/grafana/dashboards") for v in volumes)
    datasources = yaml.safe_load((BACKEND / "grafana/provisioning/datasources/prometheus.yaml").read_text())
    assert datasources["datasources"][0]["uid"] == DATASOURCE_UID


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboard_queries_exported_metrics_through_the_provisioned_datasource(path):
    raw = json.loads(path.read_text())
    dashboard = raw.get("dashboard", raw)
    exported = _exported_metric_names()
    for panel in _panels(dashboard):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if not expr:
                continue
            source = target.get("datasource") or panel.get("datasource")
            if isinstance(source, dict):
                assert source.get("uid") == DATASOURCE_UID, f"{panel.get('title')}: data source {source}"
            # Metric names are identifiers followed by a selector, a range or a closing ")".
            for name in re.findall(r"\b([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?=\{|\[)", expr):
                if name in PROMQL_WORDS:
                    continue
                assert name in exported, f"{path.name} / {panel.get('title')}: {name} is not exported"


def test_one_model_call_records_each_metric_once():
    from app.core.metrics import WEBAGENT_REGISTRY
    from app.services import llm_call_metrics

    def value(name, labels):
        return WEBAGENT_REGISTRY.get_sample_value(name, labels) or 0

    model = {"provider": "test-provider", "model": "m"}
    before = (
        value("ml_model_latency_seconds_count", {"model_name": "test-provider", "task_type": "chat"}),
        value("llm_requests_total", {**model, "status": "success"}),
        value("llm_tokens_total", {**model, "type": "prompt"}),
    )
    llm_call_metrics.record("test-provider", "m", 0.5, ok=True, usage={"prompt_tokens": 7, "completion_tokens": 3})
    after = (
        value("ml_model_latency_seconds_count", {"model_name": "test-provider", "task_type": "chat"}),
        value("llm_requests_total", {**model, "status": "success"}),
        value("llm_tokens_total", {**model, "type": "prompt"}),
    )
    assert after == (before[0] + 1, before[1] + 1, before[2] + 7)


def test_client_set_up_is_not_counted_as_a_model_call():
    # LLMMetrics also wraps building the client; that must not look like a chat call.
    from app.core.metrics import WEBAGENT_REGISTRY
    from app.services.llm import LLMMetrics

    labels = {"model_name": "setup-only", "task_type": "chat"}
    LLMMetrics().record_request("setup-only", True, tokens=0, latency=0.5)
    assert not WEBAGENT_REGISTRY.get_sample_value("ml_model_latency_seconds_count", labels)
