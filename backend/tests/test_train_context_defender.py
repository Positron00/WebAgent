"""train_context_defender.py / eval_context_defender.py smoke tests."""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from tests.fakes.triage import HashEncoder

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


_SIGNAL = {"injection": "qzvtokenalpha", "harm": "qzvtokenbravo", "sensitive_data": "qzvtokencharlie", "jailbreak": "qzvtokendelta"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_cd = _load("train_context_defender")
eval_cd = _load("eval_context_defender")
security_data = _load("security_data")


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(text, source, axis=None, origin="seed"):
    labels = {a: "none" for a in security_data.AXES}
    if axis:
        labels[axis] = "high"
        # A neutral, learnable signal per axis for this smoke test (not production data).
        text = f"{_SIGNAL[axis]} {text}"
    return {"text": text, "source": source, "labels": labels, "origin": origin}


def _tiny_dataset(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    rows = []
    for source in security_data.CONTEXT_SOURCES:
        for j in range(3):
            rows.append(_row(f"benign {source} {j}", source, origin=f"benign_{source}"))
        for axis in security_data.AXES:
            rows.append(_row(f"flag {axis}", source, axis=axis, origin=f"synthetic_{axis}"))
    _write(train, rows)
    test = tmp_path / "test.jsonl"
    test_rows = []
    for i in range(12):
        test_rows.append(_row(f"EVAL benign web gardening compost {i}", "web_result", origin="benign_web_result"))
        test_rows.append(_row(f"EVAL inj {i}", "web_result", axis="injection", origin="synthetic_injection"))
        test_rows.append(_row(f"EVAL harm {i}", "user_turn", axis="harm", origin="synthetic_harm"))
        test_rows.append(_row(f"EVAL jail {i}", "user_turn", axis="jailbreak", origin="synthetic_jailbreak"))
        test_rows.append(_row(f"EVAL secret {i}", "document", axis="sensitive_data", origin="secret_api"))
        test_rows.append(_row(f"EVAL personal {i}", "user_turn", axis="sensitive_data", origin="personal_contact"))
        test_rows.append(_row(f"EVAL geography quiz continents rivers {i}", "user_turn", origin="benign_user_turn"))
    _write(test, test_rows)
    probe = tmp_path / "probe.jsonl"
    _write(
        probe,
        [_row(f"probe {s}", s, origin=f"benign_probe_{s}") for s in security_data.CONTEXT_SOURCES],
    )
    return train, test, probe


def test_train_writes_artifacts(tmp_path):
    train, _, _ = _tiny_dataset(tmp_path)
    out = tmp_path / "model"
    meta = train_cd.train(train, out, "test-v", encoder=HashEncoder(), tfidf_only=False)
    bundle = joblib.load(out / "heads.joblib")
    assert "heads" in bundle and "injection" in bundle["heads"]
    saved = json.loads((out / "metadata.json").read_text())
    assert saved["version"] == meta["version"]
    assert saved["data_hash"].startswith("sha256:")


def _scorer(correct: bool):
    """Item scores that match the labels (oracle) or always say "no risk" (blind)."""

    def predict(_model, rows):
        scores = np.zeros((len(rows), len(security_data.AXES)))
        probs = {}
        for j, axis in enumerate(security_data.AXES):
            truth = np.array([1 if r["labels"][axis] == "high" else 0 for r in rows])
            p = truth.astype(float) if correct else np.zeros(len(rows))
            scores[:, j] = 100 * p
            probs[axis] = (p, truth)
        return scores, probs

    return predict


def test_gate_exit_code(tmp_path, monkeypatch):
    # The gate logic is tested with fixed scorers; a toy model on a few dozen rows can't be calibrated.
    train, test, probe = _tiny_dataset(tmp_path)
    out = tmp_path / "model"
    train_cd.train(train, out, "test-v", encoder=HashEncoder(), tfidf_only=True)
    args = ["--data", str(test), "--train", str(train), "--probe", str(probe), "--model", str(out), "--write-metadata"]

    monkeypatch.setattr(eval_cd, "predict_rows", _scorer(correct=True))
    assert eval_cd.main(args) == 0
    assert json.loads((out / "metadata.json").read_text())["test"]["passed"] is True

    monkeypatch.setattr(eval_cd, "predict_rows", _scorer(correct=False))
    assert eval_cd.main(args) == 1
    assert json.loads((out / "metadata.json").read_text())["test"]["passed"] is False


def test_train_records_chunking_and_features(tmp_path, monkeypatch):
    import app.security.context_defender as cd

    # train() sets the module's window size; restore it for the tests that follow.
    monkeypatch.setattr(cd, "CHUNK_SIZE", cd.CHUNK_SIZE)
    monkeypatch.setattr(cd, "CHUNK_STRIDE", cd.CHUNK_STRIDE)
    train, _, _ = _tiny_dataset(tmp_path)
    out = tmp_path / "model"
    train_cd.train(train, out, "test-v", encoder=HashEncoder(), chunk_size=96, chunk_stride=64,
                   combine_features=True, c=4.0)
    bundle = joblib.load(out / "heads.joblib")
    assert bundle["chunking"] == {"size": 96, "stride": 64}
    assert bundle["combine_features"] is True and bundle["c"] == 4.0


def test_check_data_flags_duplicates(tmp_path):
    train, test, probe = _tiny_dataset(tmp_path)
    shared = _row("shared", "user_turn")
    _write(train, [shared])
    _write(test, [shared])
    problems = eval_cd.check_data(train, test, probe, encoder=HashEncoder())
    assert any("also in train" in p for p in problems)


def test_calibration_split_keeps_rows_sharing_an_inserted_attack_together():
    attack = "shared inserted text " * 3
    rows = []
    for i in range(40):
        page = f"benign page number {i} " * 5
        cut = len(page) // 2
        text = page[:cut] + attack + page[cut:]
        rows.append({"text": text, "span": [cut, cut + len(attack)]})
    fit, cal = train_cd.split_for_calibration(rows, seed=13)
    assert not fit or not cal  # all 40 on one side
    plain = [{"text": f"question {i}"} for i in range(400)]
    fit, cal = train_cd.split_for_calibration(plain, seed=13)
    assert 0.08 < len(cal) / len(plain) < 0.22


def _report_for(rows, scorer, tmp_path, monkeypatch):
    test = tmp_path / "gate_test.jsonl"
    _write(test, rows)
    monkeypatch.setattr(eval_cd, "load_runtime_model", lambda _d: None)
    monkeypatch.setattr(eval_cd, "predict_rows", scorer)
    return eval_cd.evaluate(tmp_path, test)


def _gate_rows():
    """Real-style origins (`<dataset>:<role>`), as build_security_dataset.py writes them."""
    rows = []
    for i in range(10):
        rows.append(_row(f"page {i}", "web_result", origin="wildchat:assistant"))
        rows.append(_row(f"question {i}", "user_turn", origin="wildchat:user"))
        rows.append(_row(f"look-alike {i}", "user_turn", origin="xstest:safe"))
        rows.append(_row(f"inj {i}", "web_result", axis="injection", origin="slabs_deepset:embedded"))
        rows.append(_row(f"harm {i}", "user_turn", axis="harm", origin="aegis:unsafe"))
        rows.append(_row(f"jail {i}", "user_turn", axis="jailbreak", origin="trustairlab:jailbreak"))
        rows.append(_row(f"key {i}", "document", axis="sensitive_data", origin="generated:secret"))
        rows.append(_row(f"pii {i}", "document", axis="sensitive_data", origin="nemotron:pii"))
    return rows


def test_gate_counts_false_positives_on_benign_rows_from_any_dataset(tmp_path, monkeypatch):
    def flags_everything(_model, rows):
        scores = np.full((len(rows), len(security_data.AXES)), 100.0)
        probs = {a: (np.ones(len(rows)), np.array([1 if r["labels"][a] == "high" else 0 for r in rows]))
                 for a in security_data.AXES}
        return scores, probs

    report = _report_for(_gate_rows(), flags_everything, tmp_path, monkeypatch)
    assert report["injection_fp_benign_pages"] == 1.0
    assert report["harm_fp_ordinary_questions"] == 1.0
    assert report["harm_fp_lookalike_direct"] == 1.0  # reported separately, not gated
    assert report["secret_recall"] == 1.0 and report["personal_recall"] == 1.0
    assert eval_cd.passes(report) is False


def test_gate_fails_a_check_with_no_rows(tmp_path, monkeypatch):
    rows = [r for r in _gate_rows() if r["origin"] != "nemotron:pii"]
    report = _report_for(rows, _scorer(correct=True), tmp_path, monkeypatch)
    assert report["personal_recall"] is None
    assert eval_cd.passes(report) is False


def test_hard_negatives_weigh_more_only_on_the_axis_they_imitate():
    row = {"origin": "xstest:safe"}
    assert train_cd.chunk_weight(row, "harm") == train_cd.HARD_NEGATIVE_WEIGHT
    assert train_cd.chunk_weight(row, "injection") == 1.0
    assert train_cd.chunk_weight({"origin": "wildchat:user"}, "harm") == 1.0


def test_thresholds_are_the_lowest_scores_meeting_the_false_positive_limit():
    rows, harm = [], []
    for i in range(200):  # ordinary questions scoring 0..99
        rows.append(_row(f"q{i}", "user_turn", origin="wildchat:user"))
        harm.append((i % 100) / 100)
    for i in range(100):  # look-alikes scoring 0.99: left to the judge, so they don't move the decline score
        rows.append(_row(f"l{i}", "user_turn", origin="xstest:safe"))
        harm.append(0.99)
    for i in range(50):  # harmful requests: 46 at 0.95, 4 at 0.30
        rows.append(_row(f"h{i}", "user_turn", axis="harm", origin="aegis:unsafe"))
        harm.append(0.95 if i < 46 else 0.30)
    n = len(rows)
    scores = {a: np.zeros(n) for a in security_data.AXES}
    scores["harm"] = np.array(harm)
    chosen, report = train_cd.select_thresholds(rows, scores)
    # Ordinary questions are 0..99 twice each, so at t the false-positive rate is (100 - t) / 100;
    # nothing in 50..98 gets under 0.75%, so the decline score is the range's upper bound.
    assert chosen["harm_decline_at"] == 98
    # Verify: the highest score still reaching 92% of the harmful rows (46/50 at 95).
    assert chosen["harm_verify_at"] == 95
    assert report["harm_verify_at"]["recall"] == 0.92
    assert report["harm_verify_at"]["judge_rate_ordinary"] == 0.05  # 10 of 200 ordinary score >= 95
    assert chosen["injection_watch_at"] == chosen["injection_drop_at"] // 2


def test_judged_harm_report_declines_only_what_the_judge_flags():
    rows = [
        _row("ordinary", "user_turn", origin="wildchat:user"),      # in band, judge clears
        _row("look-alike", "user_turn", origin="xstest:safe"),      # in band, judge flags (wrongly)
        _row("harmful a", "user_turn", axis="harm", origin="aegis:unsafe"),  # above decline
        _row("harmful b", "user_turn", axis="harm", origin="aegis:unsafe"),  # in band, judge errors
    ]
    scores = np.zeros((4, len(security_data.AXES)))
    h = security_data.AXES.index("harm")
    scores[:, h] = [40, 40, 90, 40]
    th = {"harm_verify_at": 30, "harm_decline_at": 80}
    verdicts = {rows[0]["text"]: False, rows[1]["text"]: True, rows[3]["text"]: None}
    report = eval_cd.judged_harm_report(rows, scores, th, lambda text: verdicts[text])
    assert report == {"harm_recall": 0.5, "declined_ordinary_questions": 0.0,
                      "declined_lookalike_questions": 1.0, "judge_calls": 3, "judge_errors": 1}


def test_wilson_upper_bound():
    assert train_cd.wilson_upper(0, 0) == 1.0
    assert 0.015 < train_cd.wilson_upper(0, 169) < 0.017  # zero false positives in 169 still allows ~1.6%
    assert train_cd.wilson_upper(0, 1000) < 0.003


def test_acting_thresholds_use_the_confidence_bound():
    rows, harm, inj = [], [], []
    for i in range(169):  # ordinary questions, all scoring low: zero false positives at any score >= 50
        rows.append(_row(f"q{i}", "user_turn", origin="wildchat:user"))
        harm.append(0.1)
        inj.append(0.0)
    for i in range(169):  # benign pages, all scoring low
        rows.append(_row(f"p{i}", "web_result", origin="wildchat:assistant"))
        harm.append(0.0)
        inj.append(0.1)
    n = len(rows)
    scores = {a: np.zeros(n) for a in security_data.AXES}
    scores["harm"], scores["injection"] = np.array(harm), np.array(inj)
    chosen, report = train_cd.select_thresholds(rows, scores)
    # 0/169 has an upper bound of ~1.6% > 1%: the sample can't show the limit is met, so harm
    # declines only at the top of its range and the rest goes to the judge.
    assert chosen["harm_decline_at"] == 98
    assert report["harm_decline_at"]["rule"] == "upper confidence bound"
    # 0/169 benign pages: an upper bound of ~1.6% is under injection's 2% limit, so the lowest score passes.
    assert chosen["injection_drop_at"] == 30
    assert report["injection_drop_at"]["rule"] == "upper confidence bound"


def test_moderation_cutoff_keeps_estimated_ordinary_declines_under_the_limit():
    rows, harm_ints, tops = [], [], []
    for i in range(100):  # ordinary: 10% in the verify band; moderation scores spread 0..0.99
        rows.append(_row(f"q{i}", "user_turn", origin="wildchat:user"))
        harm_ints.append(50 if i < 10 else 5)
        tops.append(i / 100)
    for i in range(20):  # harmful, all in the band, moderation scores 0.3
        rows.append(_row(f"h{i}", "user_turn", axis="harm", origin="aegis:unsafe"))
        harm_ints.append(50)
        tops.append(0.3)
    th = {"harm_verify_at": 30, "harm_decline_at": 98}
    cutoff, report = train_cd.choose_moderation_cutoff(rows, harm_ints, tops, th)
    # A 0.75% budget over a 10% judge rate allows flagging 7.5% of ordinary questions, i.e. a cut-off
    # above 0.92; the grid stops at 0.9 (flags 10%), so no cut-off qualifies.
    assert cutoff is None and report["harm_recall_end_to_end"] == 0.0
    tops = [t / 10 for t in tops[:100]] + tops[100:]  # ordinary now 0..0.099: 0.1 is the first to qualify
    cutoff, report = train_cd.choose_moderation_cutoff(rows, harm_ints, tops, th)
    assert cutoff == 0.1 and report["harm_recall_end_to_end"] == 1.0


def test_aegis_categories_map_to_webagent_labels():
    build = _load("build_security_dataset")
    assert build.aegis_group("Violence, Profanity") == "harm"
    assert build.aegis_group("PII/Privacy, Needs Caution") == "privacy"
    assert build.aegis_group("Needs Caution, Political/Misinformation/Conspiracy") == "soft"
    assert build.aegis_group("Sexual") is None and build.aegis_group("") is None
    assert build.aegis_group("Profanity, Copyright/Trademark/Plagiarism") is None  # partly ambiguous


def test_judged_jailbreak_report_asks_consent_only_when_flagged():
    rows = [
        _row("role play", "user_turn", origin="wildchat:user"),
        _row("persona", "user_turn", axis="jailbreak", origin="trustairlab:jailbreak"),
    ]
    scores = np.zeros((2, len(security_data.AXES)))
    scores[:, security_data.AXES.index("jailbreak")] = [40, 40]
    th = {"jailbreak_verify_at": 30, "jailbreak_consent_at": 90}
    verdicts = {rows[0]["text"]: False, rows[1]["text"]: True}
    report = eval_cd.judged_jailbreak_report(rows, scores, th, lambda text: verdicts[text])
    assert report["jailbreak_recall"] == 1.0 and report["consent_asked_ordinary_questions"] == 0.0
    assert report["judge_calls"] == 2
