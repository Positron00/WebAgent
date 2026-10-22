"""The shadow-week report (scripts/shadow_week_report.py): numbers only, never text."""
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "scripts"))

import shadow_week_report as report  # noqa: E402

SECRET = "secret question text that must never be printed"


def _write(data: Path, n_tasks=3, searches_per_task=2, tokens=20000):
    collected = data / "security" / "collected"
    collected.mkdir(parents=True)
    rows = []
    for t in range(n_tasks):
        tid = f"task-{t}"
        rows.append({"ts": "2026-10-20T10:00:00", "checkpoint": 1, "action": "converse", "task_id": tid,
                     "outcome": "decline" if t == 0 else "allow", "policy_ids": ["untrusted-injection-drop"] if t == 0 else [],
                     "item": {"source": "web_result", "ref": "https://example.org/p", "axes": {"injection": 80 if t == 0 else 5},
                              "scores_available": True}, "args": {}})
        for s in range(searches_per_task):
            rows.append({"ts": "2026-10-20T10:00:01", "checkpoint": 2, "action": "tool::web_search", "task_id": tid,
                         "outcome": "allow", "shadow_policy_ids": ["off-intent-search-verify"] if s == 0 else [],
                         "args": {"query": SECRET},
                         "context": {"signals_available": True, "tainted": False, "intent_alignment": 10 if s == 0 else 60,
                                     "risk_above_intent": 0, "egress_overlap": 0, "declared": True,
                                     "shadow_classifier_alignment": 20 if s == 0 else 70}})
        rows.append({"ts": "2026-10-20T10:00:02", "checkpoint": 2, "action": "plan::subgoal", "task_id": tid, "outcome": "allow",
                     "context": {"signals_available": True, "intent_alignment": 50, "subgoal_harm": 0, "intent_harm": 0,
                                 "shadow_classifier_alignment": 55}})
    (collected / "2026-10-20.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    results = data / "results"
    results.mkdir()
    for t in range(n_tasks):
        (results / f"r{t}.json").write_text(json.dumps({"task_id": f"task-{t}", "status": "completed", "route": "agentic",
                                                         "tokens_used": tokens + t, "duration_seconds": 100 + t,
                                                         "created_at": "2026-10-20T10:00:00", "query": SECRET}))
    triage = data / "triage" / "collected"
    triage.mkdir(parents=True)
    (triage / "2026-10-20.jsonl").write_text(json.dumps({"ts": "2026-10-20T10:00:00", "route": "agentic", "decided_by": "model",
                                                          "p_agentic": 0.9, "text": SECRET}) + "\n")


def test_report_numbers_and_no_text(tmp_path, capsys):
    _write(tmp_path)
    assert report.main(["--data", str(tmp_path), "--json", str(tmp_path / "out.json")]) == 0
    out = capsys.readouterr().out
    saved = (tmp_path / "out.json").read_text()
    assert SECRET not in out and SECRET not in saved
    data = json.loads(saved)
    rule = data["checkpoint2_intent"]["rules"]["off-intent-search-verify"]
    assert (rule["would_act"], rule["of"], rule["recorded_shadow_matches"]) == (3, 6, 3)
    assert data["searches"]["max"] == 2 and data["runs"]["research_completed"] == 3
    assert data["checkpoint1"]["items_by_source"]["web_result"]["items"] == 3
    assert data["shadow_classifier"]["rows_with_both"] == 9


def test_small_samples_suggest_nothing(tmp_path):
    _write(tmp_path)
    data = report.build(tmp_path, None, None, None)
    assert data["searches"]["suggested_max_searches_per_task"] is None
    assert data["runs"]["suggested_token_budget_default"] is None
    assert data["checkpoint2_intent"]["suggested_thresholds"]["search_align_at"]["at_most_1pct_below"] is None
    assert "too few" in data["checkpoint2_intent"]["rules"]["off-intent-search-verify"]["status"]


def test_enough_tasks_give_suggestions(tmp_path):
    _write(tmp_path, n_tasks=25, searches_per_task=3, tokens=30000)
    data = report.build(tmp_path, None, None, None)
    assert data["searches"]["suggested_max_searches_per_task"] == 4
    assert data["runs"]["suggested_token_budget_default"] == 46000   # ceil(1.5 x p90 / 1000) x 1000


def test_thresholds_mirror_the_guard():
    from app.security.thresholds import DEFAULT_THRESHOLDS

    for key, value in report.DEFAULT_THRESHOLDS.items():
        assert DEFAULT_THRESHOLDS[key] == value, key


def test_every_shadow_rule_is_reported():
    policies = (BACKEND / "config" / "security" / "policies.cedar").read_text()
    shadow_rules = [line.split('"')[1] for line in policies.splitlines() if "@shadow(\"true\")" in line]
    assert set(shadow_rules) == {rule_id for rule_id, *_ in report.RULES}


@pytest.mark.parametrize("since,expected", [(None, "7d"), ("2026-10-17", "96h")])
def test_prometheus_range_covers_the_week_only(since, expected):
    from datetime import datetime, timezone

    assert report.prom_range(since, now=datetime(2026, 10, 21, tzinfo=timezone.utc)) == expected
