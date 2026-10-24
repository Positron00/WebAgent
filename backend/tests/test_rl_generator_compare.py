"""compare.py: decision rule on fixed numbers, resume, --dry-run, --round (rl-generator SEC-217). Offline: no training."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rl_generator import compare  # noqa: E402


def _test(metric_key, value, upper=0.005, ece_a=0.05, ece_r=0.06):
    base = {"off_intent_recall": 0.6, "risk_recall": 0.5, "lookalikes_held": 0.1, "real_aligned_held_upper": upper,
            "ece_aligned": ece_a, "ece_risk": ece_r,
            "by_role": {"T": {"off_intent_recall": 0.7, "risk_recall": 0.6}, "E": None},
            "extra": {"rl_gen_h_heldout": {"rows": 300, "held": 150, "recall": 0.5},
                      "rl_gen_l_heldout": {"rows": 300, "held": 30, "held_share": 0.1}}}
    base[metric_key] = value
    return base


def _summary(mode, values, **kw):
    key = compare.METRIC[mode][0]
    return compare.summarise([compare.metrics(_test(key, v, **kw), mode) for v in values])


def test_metrics_and_summary():
    m = compare.metrics(_test("off_intent_recall", 0.7), "H")
    assert m["metric"] == 0.7 and m["heldout"] == 0.5 and m["T_off"] == 0.7 and m["E_risk"] is None
    assert compare.metrics(_test("lookalikes_held", 0.04), "L")["heldout"] == 0.1
    s = _summary("H", [0.6, 0.64])
    assert s["metric"] == {"mean": 0.62, "spread": 0.04, "max": 0.64} and s["E_off"]["mean"] is None


def test_mode_h_higher_is_better():
    none, p, r = _summary("H", [0.5, 0.52]), _summary("H", [0.60, 0.62]), _summary("H", [0.70, 0.71])
    assert compare.beats(r, p, none, "H")["wins"] and not compare.beats(p, r, none, "H")["wins"]
    close = _summary("H", [0.61, 0.65])  # mean diff 0.02 <= spread 0.04
    assert compare.beats(close, p, none, "H")["reason"] == "difference within seed noise"


def test_mode_l_lower_is_better():
    none, p, r = _summary("L", [0.10, 0.10]), _summary("L", [0.08, 0.07]), _summary("L", [0.03, 0.04])
    assert compare.beats(r, p, none, "L")["wins"] and not compare.beats(p, r, none, "L")["wins"]


def test_ece_and_real_aligned_limits():
    none = _summary("H", [0.5, 0.5], ece_a=0.05, ece_r=0.06)
    p = _summary("H", [0.6, 0.6])
    assert compare.beats(_summary("H", [0.8, 0.8], ece_a=0.06, ece_r=0.07), p, none, "H")["wins"]  # exactly +0.01
    assert "ece_aligned" in compare.beats(_summary("H", [0.8, 0.8], ece_a=0.0601), p, none, "H")["reason"]
    assert "ece_risk" in compare.beats(_summary("H", [0.8, 0.8], ece_r=0.08), p, none, "H")["reason"]
    worse = compare.summarise([compare.metrics(_test("off_intent_recall", 0.8, upper=u), "H") for u in (0.005, 0.012)])
    assert "real aligned" in compare.beats(worse, p, none, "H")["reason"]


def test_decide_kind_then_share():
    s = {"none": _summary("H", [0.5, 0.5]), "p5": _summary("H", [0.6, 0.6]), "p15": _summary("H", [0.62, 0.62]),
         "r5": _summary("H", [0.7, 0.71]), "r15": _summary("H", [0.8, 0.81])}
    d = compare.decide(s, "H")
    assert d["kind"] == "rl" and d["share"] == 0.15 and d["keep"] == "r15"
    s["r5"], s["r15"] = _summary("H", [0.6, 0.62]), _summary("H", [0.6, 0.6])
    s["p15"] = _summary("H", [0.6, 0.63])  # 0.015 better than p5, within its 0.03 spread
    d = compare.decide(s, "H")
    assert d["kind"] == "prompted" and d["keep"] == "p5" and not d["rl_vs_prompted"]["5"]["wins"]
    s["p15"] = _summary("H", [0.7, 0.7])
    assert compare.decide(s, "H")["keep"] == "p15"


def test_decide_round():
    s = {"none": _summary("L", [0.1, 0.1]), "previous": _summary("L", [0.05, 0.05]), "new": _summary("L", [0.02, 0.03])}
    assert compare.decide_round(s, "L") == {**compare.decide_round(s, "L"), "keep": "new", "stop": False}
    s["new"] = _summary("L", [0.05, 0.06])
    assert compare.decide_round(s, "L")["stop"] is True


def _fake_runner(calls, numbers):
    """Stands in for the subprocesses: variant.json, metadata.json, then the gate's test block."""
    def run(cmd, dry, ok_codes=(0,)):
        calls.append([Path(c).name if "/" in c else c for c in cmd])
        if dry:
            return
        script, out = Path(cmd[1]).name, Path(cmd[cmd.index("--out" if "--out" in cmd else "--model") + 1])
        out.mkdir(parents=True, exist_ok=True)
        if script == "variant.py":
            (out / "variant.json").write_text("{}")
        elif script == "train_alignment_model.py":
            (out / "metadata.json").write_text(json.dumps({"train_seed": int(cmd[cmd.index("--seed") + 1])}))
        else:
            meta = json.loads((out / "metadata.json").read_text())
            name, seed = out.name.rsplit("-s", 1)
            meta["test"] = _test("off_intent_recall", numbers(name, int(seed)))
            (out / "metadata.json").write_text(json.dumps(meta))
    return run


def test_dry_run_plan(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(compare, "run", _fake_runner(calls, None))
    assert compare.main(["--modes", "H,L", "--out", str(tmp_path / "r2"), "--dry-run"]) == 0
    assert "9 configurations x 2 seeds = 18 runs" in capsys.readouterr().out
    assert sum(c[1] == "train_alignment_model.py" for c in calls) == 18
    assert sum(c[1] == "variant.py" for c in calls) == 9
    assert all(c[0] == Path(sys.executable).name for c in calls)
    assert not (tmp_path / "r2").exists()


def test_full_run_resumes_and_decides(tmp_path, monkeypatch, capsys):
    score = {"none": 0.5, "h-p5": 0.6, "h-p15": 0.62, "h-r5": 0.7, "h-r15": 0.8}
    calls = []
    monkeypatch.setattr(compare, "run", _fake_runner(calls, lambda n, s: score[n] + (0.01 if s == 17 else 0)))
    out = tmp_path / "r2"
    assert compare.main(["--modes", "H", "--out", str(out)]) == 0
    assert sum(c[1] == "train_alignment_model.py" for c in calls) == 10
    report = json.loads((out / "compare.json").read_text())
    assert report["H"]["decision"]["keep"] == "r15"
    assert set(report["H"]["summary"]) == {"none", "p5", "p15", "r5", "r15"}
    assert "| r15 | 0.805 ± 0.01 |" in capsys.readouterr().out
    calls.clear()
    assert compare.main(["--modes", "H", "--out", str(out)]) == 0
    assert calls == []  # every run already gated


def test_time_one_trains_one(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(compare, "run", _fake_runner(calls, lambda n, s: 0.5))
    assert compare.main(["--modes", "H", "--out", str(tmp_path / "r2"), "--time-one"]) == 0
    assert sum(c[1] == "train_alignment_model.py" for c in calls) == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["runs"] == 10


def test_round(tmp_path, monkeypatch):
    prev, new = tmp_path / "prev", tmp_path / "new"
    score = {"none": 0.5, "previous": 0.7, "new": 0.75}
    calls = []
    monkeypatch.setattr(compare, "run", _fake_runner(calls, lambda n, s: score[n]))
    out = tmp_path / "r3"
    assert compare.main(["--round", "--modes", "H", "--previous", str(prev), "--new", str(new), "--out", str(out)]) == 0
    assert sum(c[1] == "variant.py" for c in calls) == 1  # only none is built; round variants are given
    assert sum(c[1] == "train_alignment_model.py" for c in calls) == 6
    assert json.loads((out / "compare.json").read_text())["H"]["decision"]["keep"] == "new"
    with pytest.raises(SystemExit):
        compare.main(["--round", "--modes", "H,L", "--previous", str(prev), "--new", str(new), "--out", str(out)])
