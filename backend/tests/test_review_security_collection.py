"""scripts/review_security_collection.py: shadow-week summary, labelling sheet, FP estimate."""
import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_security_collection.py"
spec = importlib.util.spec_from_file_location("review", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def _cp1(source, outcome, injection, ref=None, policy="baseline-permit"):
    item = {"source": source, "axes": {"injection": injection, "harm": 0, "sensitive_data": 0, "jailbreak": 0},
            "pattern_hits": [], "scores_available": True}
    if ref:
        item["ref"] = ref
    return {"ts": "2026-10-14T10:00:00", "checkpoint": 1, "action": "converse", "outcome": outcome,
            "policy_ids": [policy], "task_id": "t", "item": item}


def _cp2(alignment, tainted=False, shadow=()):
    return {"ts": "2026-10-14T10:00:01", "checkpoint": 2, "action": "tool::web_search", "outcome": "allow",
            "policy_ids": ["baseline-permit"], "shadow_policy_ids": list(shadow), "task_id": "t",
            "context": {"alignment": alignment, "tainted": tainted}}


def _write(tmp_path, rows):
    d = tmp_path / "collected"
    d.mkdir()
    (d / "2026-10-14.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def test_summary_sheet_and_false_positive_estimate(tmp_path, capsys):
    rows = [_cp1("web_result", "allow", 5, "https://a") for _ in range(97)]
    rows += [_cp1("web_result", "decline", 80, f"https://drop{i}", "untrusted-injection-drop") for i in range(3)]
    rows += [_cp1("user_turn", "allow", 1)]
    rows += [_cp2(80), _cp2(15, shadow=["untainted-search-misaligned"]), _cp2(10, tainted=True)]
    d = _write(tmp_path, rows)
    sheet = tmp_path / "sheet.csv"
    assert review.main(["--dir", str(d), "--sheet", str(sheet)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint1"]["items_by_source"]["web_result"] == {"items": 100, "allow": 97, "decline": 3}
    assert report["checkpoint1"]["would_act_rate_by_source"]["web_result"] == 0.03
    align = report["checkpoint2"]["untainted_search_alignment"]
    assert align["n"] == 2 and align["share_below_30"] == 0.5   # the tainted search is left out
    assert report["checkpoint2"]["shadow_policy_matches"] == {"untainted-search-misaligned": 1}
    assert report["sheet"]["rows"] == 3

    with sheet.open() as f:
        labelled = list(csv.DictReader(f))
    labelled[0]["label"] = "benign"
    labelled[1]["label"] = "injection"
    labelled[2]["label"] = "injection"
    lab = tmp_path / "labelled.csv"
    with lab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review.SHEET_FIELDS)
        w.writeheader()
        w.writerows(labelled)
    assert review.main(["--dir", str(d), "--labels", str(lab)]) == 0
    est = json.loads(capsys.readouterr().out)["labelled"]
    assert est["untrusted_items_judged"] == 100 and est["benign_items_that_would_be_dropped"] == 1
    assert est["false_positive_rate"] == 0.01 and 0.01 < est["false_positive_rate_upper_95"] < 0.05


def test_summary_and_sheet_contain_no_text(tmp_path, capsys):
    row = _cp1("web_result", "decline", 90, "https://x", "untrusted-injection-drop")
    row["item"]["text"] = "dev-only page text"
    d = _write(tmp_path, [row])
    sheet = tmp_path / "s.csv"
    review.main(["--dir", str(d), "--sheet", str(sheet)])
    assert "dev-only page text" not in capsys.readouterr().out
    assert "dev-only page text" not in sheet.read_text()
