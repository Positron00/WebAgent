"""Review sheets, verdicts and the judge-choice score (rl-generator SEC-212).

Offline; neutral placeholder rows only.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rl_generator import review, rows  # noqa: E402

BATCH = "h-prompted-20261024-1"


@pytest.mark.parametrize("n,no,expected", [
    (100, 0, "pass"), (100, 6, "pass"), (100, 7, "more"), (100, 14, "more"), (100, 15, "fail"),
    (200, 20, "pass"), (200, 21, "fail")])
def test_verdict_table(n, no, expected):
    assert review.verdict(n, no) == expected


def make_batch(gen_dir, n=300):
    seed = {"intent": "placeholder intent", "groups": ["g"], "sources": ["asq"]}
    out = []
    for i in range(n):
        action = "web_search" if i % 3 else "function"
        origin = "asq:query" if i % 3 else "glaive:call"
        out.append(rows.make_row({**seed, "seed_id": f"s{i}", "action": action, "origin": origin},
                                 f"placeholder call {i}", "H", "prompted", batch=BATCH, generator="g",
                                 adapter_step=None, target="t", judge="j", prompt="p1", miss=0.5,
                                 novelty=0.5, data_hash="h"))
    rows.append(rows.batch_path("H", "prompted", out_dir=gen_dir), out)


def label(sheet: Path, no_count: int):
    text = sheet.read_text()
    comments = [line for line in text.splitlines(keepends=True) if line.startswith("#")]
    items = review.read_sheet(sheet)
    for i, item in enumerate(items):
        item["label_ok"] = "n" if i < no_count else "y"
    with sheet.open("w", newline="") as handle:
        handle.writelines(comments)
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def test_stratified_keeps_shares():
    items = [{"k": "a"}] * 300 + [{"k": "b"}] * 100
    picked = review.stratified(items, lambda i: (i["k"],), 100)
    assert Counter(i["k"] for i in picked) == {"a": 75, "b": 25}


def test_blind_sheet_is_stratified(tmp_path):
    gen, rev = tmp_path / "gen", tmp_path / "rev"
    make_batch(gen)
    assert review.make(BATCH, gen, rev)["sheet_rows"] == 100
    sheet = rev / f"{BATCH}.csv"
    assert sheet.read_text().startswith("# ")
    items = review.read_sheet(sheet)
    assert list(items[0]) == ["row_id", "intent", "action", "call", "label_ok"]
    assert all(i["label_ok"] == "" for i in items)
    assert Counter(i["action"] for i in items) == {"web_search": 67, "function": 33}
    with pytest.raises(SystemExit):
        review.make(BATCH, gen, rev)   # no second sheet without a `more` verdict


def test_score_more_then_pass(tmp_path):
    gen, rev = tmp_path / "gen", tmp_path / "rev"
    make_batch(gen)
    review.make(BATCH, gen, rev)
    sheet = rev / f"{BATCH}.csv"
    with pytest.raises(SystemExit):
        review.score(BATCH, gen, rev)   # unlabelled rows
    label(sheet, 10)
    assert review.score(BATCH, gen, rev)["verdict"] == "more"
    assert review.make(BATCH, gen, rev) == {"batch": BATCH, "sheet_rows": 200, "added": 100}
    items = review.read_sheet(sheet)
    assert len({i["row_id"] for i in items}) == 200
    label(sheet, 18)
    result = review.score(BATCH, gen, rev)
    assert (result["n"], result["user_no"], result["verdict"]) == (200, 18, "pass")
    saved = json.loads((gen / "rl_review.json").read_text())[BATCH]
    assert saved["verdict"] == "pass" and saved["n"] == 200


def test_judge_choice(tmp_path):
    rev = tmp_path / "rev"
    items = []
    for i in range(40):
        verdicts = {"model-a": True, "model-b": i % 2 == 0}
        items.append({"row_id": f"r{i}", "mode": "H" if i < 20 else "L", "seed_id": f"s{i}", "intent": "placeholder",
                      "action": "plan", "call": f"placeholder {i}", "verdicts": verdicts})
    rows.append(rev / "judge-choice.jsonl", items)
    assert review.make("judge-choice", tmp_path, rev)["sheet_rows"] == 40
    sheet = rev / "judge-choice.csv"
    assert "verdicts" not in sheet.read_text() and "model-a" not in sheet.read_text()
    labelled = review.read_sheet(sheet)
    by_id = {i["row_id"]: i for i in items}
    for item in labelled:
        item["label_ok"] = "y" if by_id[item["row_id"]]["verdicts"]["model-b"] else "n"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labelled[0]))
        writer.writeheader()
        writer.writerows(labelled)
    result = review.score("judge-choice", tmp_path, rev)["candidates"]
    assert result["model-b"] == {"accepted": 20, "accepted_agree": 20, "rejected": 20, "rejected_agree": 20,
                                 "failed": 0}
    assert result["model-a"] == {"accepted": 40, "accepted_agree": 20, "rejected": 0, "rejected_agree": 0,
                                 "failed": 0}


def test_judge_choice_failed_calls_are_not_rejections(tmp_path):
    rev = tmp_path / "rev"
    items = [{"row_id": f"r{i}", "mode": "H", "seed_id": f"s{i}", "intent": "placeholder", "action": "plan",
              "call": f"placeholder {i}", "verdicts": {"model-a": True, "model-b": None if i < 4 else False}}
             for i in range(10)]
    rows.append(rev / "judge-choice.jsonl", items)
    assert review._vote(items[0]) == "all_yes" and review._vote(items[5]) == "split"
    assert review._vote({"verdicts": {"model-a": None}}) == "failed"
    review.make("judge-choice", tmp_path, rev)
    sheet = rev / "judge-choice.csv"
    labelled = review.read_sheet(sheet)
    for item in labelled:
        item["label_ok"] = "y"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labelled[0]))
        writer.writeheader()
        writer.writerows(labelled)
    result = review.score("judge-choice", tmp_path, rev)["candidates"]
    assert result["model-b"] == {"accepted": 0, "accepted_agree": 0, "rejected": 6, "rejected_agree": 0,
                                 "failed": 4}


def test_cli_prints_counts_only(tmp_path, capsys):
    gen, rev = tmp_path / "gen", tmp_path / "rev"
    make_batch(gen, 50)
    review.main(["--make", BATCH, "--generated-dir", str(gen), "--review-dir", str(rev)])
    assert "placeholder" not in capsys.readouterr().out
