"""Gate report by role and on held-out generated sets (rl-generator SEC-215, SEC-216). Offline: a fake model."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_alignment_model as ev  # noqa: E402

TH = {"search_align_at": 50, "subgoal_align_at": 50, "drift_align_at": 50, "risk_verify_at": 70, "risk_decline_at": 90}


def _row(i, aligned="yes", risk="none", origin="webagent-gen:T", role="T"):
    return {"intent": f"placeholder-intent {i}", "call": f"placeholder-call {i} {'off' if aligned == 'no' else 'on'} {risk}",
            "action": "web_search", "labels": {"aligned": aligned, "risk": risk}, "origin": origin, "role": role,
            "split": "test", "groups": [f"g{i}"], "sources": [origin.split(":")[0]]}


def _rows():
    return ([_row(i) for i in range(40)] + [_row(100 + i, "no", origin="injecagent:T") for i in range(20)]
            + [_row(200 + i, "no", "high", "agentdojo:E", "E") for i in range(30)]
            + [_row(300 + i, "yes", origin="agentdojo:E", role="E") for i in range(10)])


class FakeModel:
    """Off-intent calls score low on alignment; the E rows' risky calls are only half caught."""

    def score_pair(self, intent, call):
        if " off " in call:
            n = int(intent.split()[-1])
            return 0.1, (95 if "high" in call and n % 2 else 10)
        return 0.9, 5


def _score(rows):
    return ev.score(FakeModel(), rows)


def test_by_role_reports_each_role_alone():
    rows = _rows()
    p, risk = _score(rows)
    out = ev.by_role(rows, p, risk, TH)
    assert set(out) == {"T", "E"}
    assert out["T"]["rows"] == 60 and out["E"]["rows"] == 40
    assert "held_by_source" not in out["T"] and "held_by_source" not in out["E"]
    assert out["T"]["risk_recall"] is None and out["E"]["risk_recall"] == 0.5
    t_only = [r for r in rows if r["role"] == "T"]
    expected = ev.report(t_only, *_score(t_only), TH)
    expected.pop("held_by_source")
    assert out["T"] == expected
    assert ev.by_role(t_only, *_score(t_only), TH)["E"] is None


def test_checks_read_only_the_all_rows_metrics():
    rows = _rows()
    p, risk = _score(rows)
    test = ev.report(rows, p, risk, TH)
    before = ev.passes(test, {"p95_ms": 5.0})
    test["by_role"] = ev.by_role(rows, p, risk, TH)
    test["by_role"]["T"]["risk_recall"] = 0.0
    assert ev.passes(test, {"p95_ms": 5.0}) == before


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_extra_reports_recall_and_held_share(tmp_path, monkeypatch):
    from app.security import alignment_model

    data, model = tmp_path / "data", tmp_path / "model"
    data.mkdir()
    model.mkdir()
    _write(data / "test.jsonl", _rows())
    (model / "metadata.json").write_text(json.dumps({"kind": "cross_encoder", "version": "v", "thresholds": TH}))
    monkeypatch.setattr(alignment_model, "load", lambda *a, **k: FakeModel())
    h = _write(tmp_path / "rl_gen_h_heldout.jsonl",
               [_row(i, "no", origin="rl-gen-h:rl") for i in range(9)] + [_row(9, origin="rl-gen-h:rl")])
    lo = _write(tmp_path / "rl_gen_l_heldout.jsonl",
                [_row(i, origin="rl-gen-l:rl") for i in range(3)] + [_row(5, "no", origin="rl-gen-l:rl")])
    result = ev.evaluate(model, data, [h, lo])
    assert result["test"]["extra"] == {"rl_gen_h_heldout": {"rows": 10, "held": 9, "recall": 0.9},
                                       "rl_gen_l_heldout": {"rows": 4, "held": 1, "held_share": 0.25}}
    assert set(result["test"]["by_role"]) == {"T", "E"}
    assert "extra" not in result["checks"] and "by_role" not in result["checks"]
    lines = ev.summary_lines("model", result["test"])
    assert "role T" in lines and "extra rl_gen_h_heldout: rows 10, held 9, recall 0.9" in lines
    assert "placeholder" not in lines


@pytest.mark.parametrize("origins", [["webagent-gen:T"], ["rl-gen-h:rl", "injecagent:T"], ["rl-gen-h:rl", "rl-gen-l:rl"], []])
def test_extra_refuses_other_rows(tmp_path, origins):
    path = _write(tmp_path / "set.jsonl", [_row(i, "no", origin=o) for i, o in enumerate(origins)])
    with pytest.raises(SystemExit):
        ev.load_extra(path)


def test_extra_mode_from_source(tmp_path):
    mode, rows = ev.load_extra(_write(tmp_path / "s.jsonl", [_row(0, origin="rl-gen-l:rl")]))
    assert mode == "L" and len(rows) == 1
    rep = ev.extra_report("L", rows, np.array([0.9]), np.array([80]), TH)
    assert rep == {"rows": 1, "held": 1, "held_share": 1.0}
