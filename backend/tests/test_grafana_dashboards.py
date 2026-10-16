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


def _metric_names(expr):
    """Every metric name in a PromQL expression, with or without a selector."""
    expr = re.sub(r'"[^"]*"', "", expr)  # strings
    expr = re.sub(r"\{[^}]*\}|\[[^\]]*\]", " ", expr)  # selectors and ranges
    expr = re.sub(r"\b(by|without|on|ignoring)\s*\([^)]*\)", " ", expr)  # label lists
    names = re.findall(r"\b([a-zA-Z_:][a-zA-Z0-9_:]*)\b(?!\s*\()", expr)  # not a function call
    return [n for n in names if n not in PROMQL_WORDS]


def test_metric_names_finds_bare_names_and_skips_labels_and_functions():
    expr = 'max by (alert) (label_replace(webagent_alert_active, "a", "$1", "b", "(.*)")) or vector(0)'
    assert _metric_names(expr) == ["webagent_alert_active"]
    assert _metric_names('sum(rate(x_total{job="j"}[5m])) by (le) / y') == ["x_total", "y"]


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
            for name in _metric_names(expr):
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


# 3.6.0: alerts row, gauge thresholds (OBS-016) and Grafana alert rules (OBS-017).
DRIFT_DASHBOARD = BACKEND / "grafana" / "dashboards" / "model-drift-dashboard.json"
ALERT_RULES = BACKEND / "grafana" / "provisioning" / "alerting" / "webagent-alerts.yaml"
ALERTS = {
    "model_drift": ("warning", "llm"),
    "model_errors": ("critical", "llm"),
    "rate_limited": ("warning", "llm"),
    "token_usage": ("warning", "llm"),
    "document_drift": ("warning", "document_processing"),
    "document_errors": ("critical", "document_processing"),
}


def _drift_dashboard():
    raw = json.loads(DRIFT_DASHBOARD.read_text())
    return raw.get("dashboard", raw)


def test_drift_dashboard_starts_with_the_alerts_row():
    panels = sorted(_drift_dashboard()["panels"], key=lambda p: (p["gridPos"]["y"], p["gridPos"]["x"]))
    rows = [p for p in panels if p["type"] == "row"]
    assert rows[0]["title"] == "Alerts" and rows[0]["gridPos"]["y"] == 0
    in_first_row = [p for p in panels if rows[0]["gridPos"]["y"] < p["gridPos"]["y"] < rows[1]["gridPos"]["y"]]
    assert {(p["title"], p["type"]) for p in in_first_row} == {
        ("Active alerts", "stat"), ("Alerts over time", "state-timeline"), ("Detectors", "table"),
    }
    exprs = " ".join(t["expr"] for p in in_first_row for t in p["targets"])
    for metric in ("webagent_alert_active", "webagent_drift_detected", "webagent_drift_enough_data",
                   "webagent_drift_current_samples", "data_drift_score"):
        assert metric in exprs
    for panel in _drift_dashboard()["panels"]:
        if panel["type"] != "row":
            assert panel["datasource"] == {"type": "prometheus", "uid": DATASOURCE_UID}
        for target in panel.get("targets", []):
            assert target["datasource"] == {"type": "prometheus", "uid": DATASOURCE_UID}


@pytest.mark.parametrize("title, red_at", [
    ("Model speed drift", 0.95), ("Embedding drift", 0.7), ("Model failure share, last hour", 0.05),
])
def test_drift_gauges_turn_red_where_the_score_means_drift(title, red_at):
    gauges = {p["title"]: p for p in _drift_dashboard()["panels"] if p["type"] == "gauge"}
    gauge = gauges[title]
    steps = gauge["fieldConfig"]["defaults"]["thresholds"]["steps"]
    assert [s["value"] for s in steps if s["color"] == "red"] == [red_at]
    assert gauge.get("description")
    assert not any("performance drift" in t.lower() for t in gauges)


def test_alert_rules_provision_one_rule_per_alert():
    config = yaml.safe_load(ALERT_RULES.read_text())
    assert config["apiVersion"] == 1
    assert not config.get("contactPoints") and not config.get("policies")
    [group] = config["groups"]
    assert (group["name"], group["folder"], group["interval"]) == ("webagent", "WebAgent", "1m")
    rules = group["rules"]
    assert len({r["uid"] for r in rules}) == len(rules)
    by_alert = {}
    for rule in rules:
        query = next(d for d in rule["data"] if d["refId"] == "A")
        name = re.fullmatch(r'max\(webagent_alert_active\{alert="(\w+)"\}\) or vector\(0\)',
                            query["model"]["expr"]).group(1)
        assert name not in by_alert, f"two rules for {name}"
        by_alert[name] = rule
        assert query["datasourceUid"] == DATASOURCE_UID and query["model"]["instant"] is True
        threshold = next(d for d in rule["data"] if d["refId"] == rule["condition"])
        assert threshold["datasourceUid"] == "__expr__"
        assert threshold["model"]["type"] == "threshold" and threshold["model"]["expression"] == "A"
        assert threshold["model"]["conditions"][0]["evaluator"] == {"type": "gt", "params": [0]}
        assert (rule["noDataState"], rule["execErrState"], rule["for"]) == ("OK", "Error", "0s")
        assert rule["annotations"]["runbook"] == "docs/RUNBOOK.md#drift-and-alerts"
        assert "retrain" not in rule["annotations"]["summary"].lower()
        assert (rule["labels"]["severity"], rule["labels"]["service"]) == ALERTS[name]
    assert set(by_alert) == set(ALERTS)
    # The annotation's anchor must exist: GitHub makes "#drift-and-alerts" from this exact heading.
    runbook = (BACKEND.parent / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "\n## Drift and alerts\n" in runbook
