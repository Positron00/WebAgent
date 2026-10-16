"""Thresholds and gate report for the alignment classifier (checkpoint2-intent B5, B6), synthetic scores."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import alignment_training as at  # noqa: E402
import eval_alignment_model as ev  # noqa: E402


def _row(aligned, risk="none", origin="webagent-gen:T"):
    return {"intent": "an intent", "call": "a call", "action": "web_search",
            "labels": {"aligned": aligned, "risk": risk}, "origin": origin}


def _fold(n_real=400, n_off=200, n_risky=100, n_look=150):
    rows = ([_row("yes") for _ in range(n_real)] + [_row("no", origin="injecagent:T") for _ in range(n_off)]
            + [_row("no", "high", "aegis:T") for _ in range(n_risky)]
            + [_row("yes", "none", "xstest:safe") for _ in range(n_look)])
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(0.6, 1.0, n_real), rng.uniform(0.0, 0.2, n_off),
                        rng.uniform(0.0, 0.3, n_risky), rng.uniform(0.7, 1.0, n_look)])
    risk = np.concatenate([rng.integers(0, 10, n_real), rng.integers(0, 20, n_off),
                           rng.integers(70, 100, n_risky), rng.integers(0, 25, n_look)])
    return rows, p, risk


def test_thresholds_separate_well_separated_scores():
    rows, p, risk = _fold()
    chosen = at.choose_thresholds(rows, p, risk)
    th = chosen["thresholds"]
    assert 20 <= th["search_align_at"] <= 60
    assert th["risk_verify_at"] <= th["risk_decline_at"] <= 70
    lo, hi = chosen["uncertain_band"]
    assert lo <= hi


def test_report_and_checks():
    rows, p, risk = _fold()
    th = at.choose_thresholds(rows, p, risk)["thresholds"]
    rep = ev.report(rows, p / 1.0, risk, th)
    assert rep["off_intent_recall"] >= 0.95 and rep["real_aligned_held"] == 0
    assert rep["risk_recall"] == 1.0 and rep["lookalikes_held"] <= 0.02
    checks = ev.passes(rep, {"p95_ms": 5.0})
    # Uniform synthetic scores aren't calibrated, so only the ECE checks may fail here.
    assert all(v for k, v in checks.items() if not k.startswith("ece_"))
    assert 0.0 <= rep["ece_aligned"] <= 1.0
    assert not ev.passes(rep, {"p95_ms": 50.0})["latency_p95_ms"]


def test_wilson_upper_bound():
    assert at.wilson_upper(0, 0) == 1.0
    assert 0.0 < at.wilson_upper(0, 500) < 0.01
    assert at.wilson_upper(10, 100) > 0.1


def test_call_text_and_origins():
    assert at.call_text({"action": "plan", "call": "x"}) == "plan: x"
    assert at.is_real_aligned(_row("yes", origin="asq:T")) and not at.is_real_aligned(_row("yes", origin="glaive:T"))
    assert at.is_lookalike(_row("yes", origin="notinject:T"))


def _write_split(d, name, n):
    rows = []
    for i in range(n):
        if i % 3 == 0:
            rows.append({"intent": "weather forecast paris", "call": "paris weather tomorrow", "action": "web_search",
                         "labels": {"aligned": "yes", "risk": "none"}, "origin": "webagent-gen:T"})
        elif i % 3 == 1:
            rows.append({"intent": "weather forecast paris", "call": "build device", "action": "web_search",
                         "labels": {"aligned": "no", "risk": "high"}, "origin": "fixture:T"})
        else:
            rows.append({"intent": "weather forecast paris", "call": "search the web", "action": "plan",
                         "labels": {"aligned": "no", "risk": "none"}, "origin": "fixture:T"})
    (d / f"{name}.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))


def test_train_then_gate_round_trip(tmp_path):
    """The trainer's output loads through the runtime loader and gets a gate report (tiny model)."""
    import json
    import types

    import train_alignment_model as tr
    from tests.test_alignment_model import write_fixture

    from app.security import alignment_model

    base = write_fixture(tmp_path / "base")
    data = tmp_path / "data"
    data.mkdir()
    for name, n in (("train", 90), ("test", 30), ("probe", 9)):
        _write_split(data, name, n)
    out = tmp_path / "candidate"
    args = types.SimpleNamespace(kind="cross_encoder", encoder=str(base / "encoder"), out=out, epochs=1,
                                 batch=16, lr=1e-3, device="cpu", data=data)
    rows = at.load_split("train", data)
    tr.train_cross_encoder(args, *tr.split_calibration(rows))
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["kind"] == "cross_encoder" and set(meta["thresholds"]) >= {"search_align_at", "risk_verify_at"}
    assert alignment_model.load(out) is None  # not gated yet
    result = ev.evaluate(out, data)
    assert result["test"]["rows"] == 30 and "passed" in result and result["latency"]["p95_ms"] > 0


def test_suites_count_role_e_rows_only(tmp_path, monkeypatch):
    import json
    import eval_alignment_suites as su

    rows = [{"intent": "weather forecast paris", "call": "placeholder call", "action": "web_search",
             "labels": {"aligned": "no", "risk": "high"}, "origin": "agentdojo:injection_goal", "role": "E"},
            {"intent": "weather forecast paris", "call": "placeholder subgoal", "action": "plan",
             "labels": {"aligned": "no", "risk": "high"}, "origin": "os-harm:misuse", "role": "E"},
            {"intent": "x", "call": "y", "action": "web_search", "labels": {"aligned": "yes", "risk": "none"},
             "origin": "asq:query", "role": "T"}]
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["eval_alignment_suites.py", "--data", str(tmp_path), "--json", str(out)])
    assert su.main() == 0
    report = json.loads(out.read_text())
    assert sum(v["rows"] for v in report["suites"].values()) == 2
