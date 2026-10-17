"""Alignment dataset builder: source filters, split rules and --check-data (SEC-120, SEC-121).

Offline: a fake embedder and neutral placeholder rows (no dataset text, no encoder download).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import alignment_agent_pairs as agent  # noqa: E402
import alignment_data as data  # noqa: E402
import alignment_pairs as pairs  # noqa: E402
import build_alignment_dataset as build  # noqa: E402
import fetch_alignment_sources as fetch  # noqa: E402


def fake_embed(texts):
    """Same text -> same unit vector; different texts -> nearly orthogonal."""
    out = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32
        v = np.random.default_rng(seed).normal(size=384)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


@pytest.fixture
def ctx(tmp_path):
    return pairs.Ctx(tmp_path, 13, fake_embed)


def write_raw(path: Path, name: str, rows):
    (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_agent_safetybench_drops_derived_cases():
    base = {"risks": ["Lead to property loss"], "envs": ["Shop"], "failure_modes": [5]}
    assert agent.asb_keep(base) == (True, "kept")
    assert agent.asb_keep({**base, "risks": ["Produce unsafe information / misinformation"]})[1] == "advbench"
    assert agent.asb_keep({**base, "envs": []})[1] == "no_environment"
    assert agent.asb_keep({**base, "failure_modes": [2, 9]})[1] == "injection_derived"
    assert agent.asb_keep({**base, "envs": ["Shop", "Slack"]})[1] == "agentdojo_environment"


def test_saroku_needs_a_passed_quality_check(tmp_path):
    q = tmp_path / "QUALITY.md"
    assert not agent.saroku_passed(q)
    q.write_text("# check\n\n**Verdict: FAILED**\n")
    assert not agent.saroku_passed(q)
    q.write_text("# check\n\n**Verdict: PASSED** (with conditions)\n")
    assert agent.saroku_passed(q)


def test_committed_quality_file_has_a_verdict():
    assert agent.saroku_passed(build.OUT / "QUALITY.md")


@pytest.mark.parametrize("row, want", [
    ({"is_safe": True}, ("yes", "none")),
    ({"is_safe": False, "property": "goal_drift", "severity": "high"}, ("no", "high")),
    ({"is_safe": False, "property": "scope_violation", "severity": "medium"}, ("no", "low")),
    ({"is_safe": False, "property": "injection", "severity": "medium"}, ("no", "high")),
    ({"is_safe": False, "property": "policy_violation", "severity": "low"}, ("yes", "low")),
    ({"is_safe": False, "property": "corrigibility", "severity": "high"}, None),
    ({"is_safe": True, "source": "template"}, None),
    ({"is_safe": True, "domain": "kubernetes"}, None),
    ({"is_safe": True, "goal": ""}, None),
])
def test_saroku_mapping_and_filters(row, want):
    full = {"goal": "goal one", "source": "gemini", "domain": "email", **row}
    assert agent.saroku_labels(full) == want


def test_row_limits_and_role(ctx):
    assert ctx.row("q" * 2001, "call 1", "plan", "yes", "none", "asq:query", ["asq:1"], ["asq"]) is None
    assert ctx.row("question one", "c" * 1001, "plan", "yes", "none", "asq:query", ["asq:1"], ["asq"]) is None
    r = ctx.row("question one", "risky call 1", "function", "no", "high", "agentdojo:injection_goal", ["dojo:1"], ["agentdojo"])
    assert r["role"] == "E" and r["split"] in ("test", "probe")
    forced = ctx.row("question one", "call 1", "function", "yes", "high", "agentdam:unneeded_data", ["dam:1"], ["agentdam"], split="test")
    assert forced["split"] == "test"


def test_evaluation_only_groups_never_reach_train(ctx):
    assert {ctx.split_of(f"g{i}", "E") for i in range(500)} == {"test", "probe"}
    shares = [ctx.split_of(f"g{i}") for i in range(2000)]
    assert 0.74 < shares.count("train") / 2000 < 0.86 and "probe" in shares


def test_row_whose_groups_fall_in_different_splits_is_dropped(ctx):
    a = next(f"a{i}" for i in range(100) if ctx.split_of(f"a{i}") == "train")
    b = next(f"b{i}" for i in range(100) if ctx.split_of(f"b{i}") == "test")
    assert ctx.row("question one", "call 1", "function", "no", "high", "injecagent:attack", [a, b], ["injecagent"]) is None
    assert ctx.dropped["groups_in_different_splits"] == 1


def test_asq_uses_accepted_source_sets_only(ctx):
    rows = [{"qid": "1", "set": "hotpotqa-test", "question": "question one", "queries": ["query 1"]},
            {"qid": "2", "set": "msmarco-dev", "question": "question two", "queries": ["query 2"]}]
    out = pairs.asq_rows(ctx, rows, data.ASQ_SOURCES, 10)
    assert [r["call"] for r in out] == ["query 1"] and ctx.dropped["asq_set_not_accepted"] == 1
    assert out[0]["action"] in ("web_search", "internal_retrieve")


def test_generated_task_with_replaced_goal_is_excluded(ctx):
    rows = [{"question_hash": "h1", "question": "question one", "rewrite": "rewrite one", "questions": ["sub 1"]},
            {"question_hash": "h2", "question": "question two", "rewrite": "rewrite two", "questions": ["sub 2"]}]
    score = lambda rewrite, q: 10 if rewrite == "rewrite two" else 80  # noqa: E731
    out, review = pairs.gen_rows(ctx, rows, score, 40)
    assert review == ["h2"] and {r["call"] for r in out} == {"rewrite one", "sub 1"}
    assert all(r["labels"] == {"aligned": "yes", "risk": "none"} for r in out)


def test_repairs_stay_in_split_and_label_off_intent(ctx):
    aligned = [ctx.row(f"question {i}", f"call {i}", "web_search", "yes", "none", "asq:query", [f"asq:{i}"], ["asq"])
               for i in range(40)]
    out = agent.repair_rows(ctx, aligned, 20, 20)
    assert out and all(r["labels"] == {"aligned": "no", "risk": "none"} for r in out)
    by_call = {r["call"]: r for r in aligned}
    assert all(r["split"] == by_call[r["call"]]["split"] and r["intent"] != by_call[r["call"]]["intent"] for r in out)


def _row(intent, call, split="train", sources=("asq",), origin="asq:query", role="T", **kw):
    return {"intent": intent, "call": call, "action": "web_search", "labels": {"aligned": "yes", "risk": "none"},
            "origin": origin, "role": role, "split": split, "groups": [f"g:{intent}"], "sources": list(sources), **kw}


def test_rule_errors():
    assert "evaluation-only row in train" in data.rule_errors(
        _row("q1", "c1", sources=["agentdojo"], origin="agentdojo:injection_goal", role="E"), True)
    assert "gated or no-train row in a committed file" in data.rule_errors(
        _row("q1", "c1", split="test", sources=["agentharm"], origin="agentharm:harmful", role="E"), True)
    assert data.rule_errors(_row("q1", "c1", split="test", sources=["agentharm"], origin="agentharm:harmful", role="E"), False) == []
    assert "CC BY-SA and CC BY-NC text in one row" in data.rule_errors(
        _row("q1", "c1", split="test", sources=["asq", "agentdam"], origin="agentdam:required", role="E"), False)
    assert "test-only source outside test" in data.rule_errors(
        _row("q1", "c1", split="probe", sources=["agentdam"], origin="agentdam:required", role="E"), True)
    assert data.schema_errors(_row("q1", "c1", sources=["agentdojo"], origin="agentdojo:x")) == ["role does not match sources"]


def test_near_duplicates_keep_one_split(ctx):
    rows = [_row("question one", "call 1", "train"), _row("question one", "call 1", "test"), _row("question two", "call 2", "test")]
    rows[1]["action"] = "plan"
    build.assign_clusters(ctx, rows, fake_embed)
    assert sum(1 for r in rows if r.get("_drop")) == 1 and rows[2].get("_drop") is None
    assert rows[0]["cluster"] == rows[1]["cluster"] != rows[2]["cluster"]


def test_local_rows_are_written_only_to_local(tmp_path):
    splits = {"test": [_row("q1", "c1", "test"), _row("q2", "risky call 1", "test", sources=["agentharm"],
                                                         origin="agentharm:harmful", role="E")]}
    build.write(tmp_path, splits, {})
    assert [json.loads(line)["intent"] for line in (tmp_path / "test.jsonl").open()] == ["q1"]
    assert [json.loads(line)["intent"] for line in (tmp_path / "local" / "test.jsonl").open()] == ["q2"]
    assert json.loads((tmp_path / "counts.json").read_text())["by_split"]["test"] == {
        "rows": 2, "committed": 1, "local": 1, "role": {"T": 1, "E": 1}}


def test_check_data_finds_leaks_and_passes_clean_data(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "REQUIRED_CELLS", {"web_search": (("yes", "none"),)})
    monkeypatch.setattr(data, "MIN_CELL", 1)
    readme = tmp_path / "README.md"
    readme.write_text(" ".join(t for *_, t in data.SOURCES.values()))
    clean = {"train": [_row("q1", "c1")], "test": [_row("q2", "c2", "test")]}
    build.write(tmp_path, clean, {})
    assert data.check_data(tmp_path, fake_embed, readme) == []
    leaky = {"train": [_row("q1", "c1"), _row("q1", "c1")],
             "test": [_row("q3", "c3", "test", groups=["g:q1"]), _row("q4", "c4", "test", sources=["xlam"], origin="xlam:call")]}
    for split, rows in leaky.items():
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    errors = " ".join(data.check_data(tmp_path, fake_embed, readme))
    assert "duplicate" in errors and "more than one split" in errors and "committed file" in errors
    readme.write_text("nothing here")
    assert "README has no licence entry" in " ".join(data.check_data(tmp_path, None, readme))


def test_gated_sources_are_skipped_without_a_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "hf_token", lambda: None)
    monkeypatch.setattr(fetch, "SOURCES", {"xlam": lambda cap, token: iter([]), "toolemu": lambda cap: iter([{"name": "n1"}])})
    monkeypatch.setattr(sys, "argv", ["fetch", "--out", str(tmp_path)])
    assert fetch.main() == 0
    out = capsys.readouterr().out
    assert "skipped: gated" in out and (tmp_path / "toolemu.jsonl").is_file() and not (tmp_path / "xlam.jsonl").exists()


def test_fetch_refuses_an_output_folder_inside_the_repository(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch", "--out", str(SCRIPTS / "scratch")])
    with pytest.raises(SystemExit):
        fetch.main()


def test_generated_code_rows_are_aligned_execute_code_calls(ctx):
    rows = [{"question_hash": "h1", "question": "placeholder coding question", "code": "x = 1\n" * 400,
             "explanation": "placeholder", "kind": "code"},
            {"question_hash": "h2", "question": "another question", "code": "", "kind": "code"}]
    out = pairs.gen_code_rows(ctx, rows)
    assert len(out) == 1
    row = out[0]
    assert row["action"] == "execute_code" and row["labels"] == {"aligned": "yes", "risk": "none"}
    assert row["origin"] == "webagent-gen:code" and len(row["call"]) <= pairs.CODE_CALL_CHARS


def test_asq_quota_is_split_evenly_between_accepted_sets(ctx):
    accepted = {"set-a": {"licence": "x"}, "set-b": {"licence": "y"}}
    rows = [{"qid": f"{s}{i}", "set": s, "question": f"placeholder question {s} {i}",
             "queries": [f"placeholder query {s} {i} {j}" for j in range(3)]}
            for s in ("set-a", "set-b") for i in range(40)]
    rows.append({"qid": "z", "set": "not-accepted", "question": "q", "queries": ["x"]})
    out = pairs.asq_rows(ctx, rows, accepted, 60)
    origins = [r["groups"][0].split(":")[1] for r in out]
    assert origins.count("set-a") == origins.count("set-b") == 30
