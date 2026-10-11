"""train_triage.py / eval_triage.py with a fake encoder (REQ-022, REQ-023, REQ-020)."""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from tests.fakes.triage import HashEncoder

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_triage = _load("train_triage")
eval_triage = _load("eval_triage")

CATEGORIES = [
    {"id": "coding_help", "label": "coding help", "class": "direct", "hard_case": False},
    {"id": "current_events", "label": "current events", "class": "agentic", "hard_case": False},
]
DIRECT_WORDS = ["python", "loop", "function", "variable", "syntax", "list", "class", "string", "error", "array"]
AGENTIC_WORDS = ["news", "today", "latest", "election", "market", "price", "announced", "week", "update", "report"]


def _rows(words, label, category, n, offset=0):
    return [
        {"text": f"{words[i % 10]} {words[(i + 3) % 10]} question {i + offset}", "label": label,
         "category": category, "source": "generated"}
        for i in range(n)
    ]


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _dataset(tmp_path: Path):
    (tmp_path / "categories.json").write_text(json.dumps(CATEGORIES))
    train = tmp_path / "train.jsonl"
    _write(train, _rows(DIRECT_WORDS, "direct", "coding_help", 10) + _rows(AGENTIC_WORDS, "agentic", "current_events", 10))
    return train


def test_train_writes_artifacts(tmp_path):
    train = _dataset(tmp_path)
    out = tmp_path / "model"
    meta = train_triage.train(train, out, "test-v", "fake-encoder", 0.95, 13, encoder=HashEncoder())
    bundle = joblib.load(out / "classifier.joblib")
    assert set(bundle["classifier"].classes_) == {"agentic", "direct"}
    assert bundle["centroids"].shape[0] == len(bundle["category_ids"]) == 2
    saved = json.loads((out / "metadata.json").read_text())
    assert saved == meta
    assert saved["categories"] == {"coding_help": "coding help", "current_events": "current events"}
    assert 0.0 <= saved["threshold"] <= 0.5
    assert saved["data_hash"].startswith("sha256:")
    assert set(saved["validation"]) == {"research_recall", "direct_rate"}
    assert (out / "classifier.joblib").stat().st_size < 100_000


def test_pick_threshold_is_highest_meeting_recall():
    y = np.array([1, 1, 1, 0, 0])
    proba = np.array([0.45, 0.30, 0.20, 0.10, 0.05])
    assert train_triage.pick_threshold(y, proba, 1.0) == 0.20
    assert train_triage.pick_threshold(y, np.array([0.9, 0.8, 0.7, 0.1, 0.05]), 1.0) == 0.5


def test_gate_exit_code_follows_targets(tmp_path, monkeypatch):
    train = _dataset(tmp_path)
    out = tmp_path / "model"
    train_triage.train(train, out, "test-v", "fake-encoder", 0.95, 13, encoder=HashEncoder())
    good = tmp_path / "test_good.jsonl"
    _write(good, _rows(DIRECT_WORDS, "direct", "coding_help", 5, 100) + _rows(AGENTIC_WORDS, "agentic", "current_events", 5, 100))
    monkeypatch.setattr(eval_triage, "_encoder", lambda name: HashEncoder())
    assert eval_triage.main(["--data", str(good), "--model", str(out), "--write-metadata"]) == 0
    assert json.loads((out / "metadata.json").read_text())["test"]["passed"] is True

    flipped = [dict(r, label="agentic" if r["label"] == "direct" else "direct")
               for r in json.loads("[" + ",".join(good.read_text().splitlines()) + "]")]
    bad = tmp_path / "test_bad.jsonl"
    _write(bad, flipped)
    assert eval_triage.main(["--data", str(bad), "--model", str(out)]) == 1


def test_check_data_finds_problems(tmp_path):
    cats = CATEGORIES + [{"id": "followup", "label": "follow-up", "class": "mixed", "hard_case": True}]
    (tmp_path / "categories.json").write_text(json.dumps(cats))
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    shared = {"text": "python loop question", "label": "direct", "category": "coding_help", "source": "generated"}
    _write(train, [shared, {"text": "news today", "label": "direct", "category": "current_events", "source": "seed"}])
    _write(test, [shared, {"text": "x", "label": "direct", "category": "unknown", "source": "seed"}])
    problems = eval_triage.check_data(train, test, tmp_path / "categories.json", encoder=HashEncoder())
    text = "\n".join(problems)
    assert "does not match category current_events" in text
    assert "unknown category" in text
    assert "test row also in train" in text
    assert "near-duplicate" in text
    assert "hard case followup" in text
