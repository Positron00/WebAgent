"""Cedar decision point: load, lint, schema, policy tests, severity (SEC-010–SEC-014)."""
import time
from pathlib import Path

import cedarpy
import pytest

from app.security import decision
from app.security.decision import PolicyLoadError, build_context, load_policy_engine, split_policies


@pytest.fixture(autouse=True)
def fresh_engine():
    decision._ENGINE = None
    yield
    decision._ENGINE = None


def test_policies_load_validate_and_offline_tests_pass():
    engine = load_policy_engine()
    # 15 starting + two user-turn verify (SEC-035a) + nine shadow-only intent rules (checkpoint2-intent
    # SEC-109; off-intent-search-verify replaced SEC-044a's untainted-search-misaligned)
    assert len(engine.annotations) == 26
    assert engine.annotations["policy0"].policy_id == "baseline-permit"


def test_split_policies_respects_strings():
    text = 'permit (principal, action, resource); @id("x") forbid (principal, action, resource) when { context.source == "a;b" };'
    parts = split_policies(text)
    assert len(parts) == 2
    assert 'source == "a;b"' in parts[1]


def test_build_context_fills_defaults():
    ctx = build_context({"source": "user_turn"})
    assert ctx["alignment"] == 100
    assert ctx["pattern_hits"] == []
    assert ctx["declined_actions_in_task"] == 0


def test_severity_decline_beats_verify():
    d = decision.decide(
        "web_research",
        "converse",
        "task-1",
        {"source": "web_result", "injection": 45, "scores_available": False},
    )
    assert d.outcome == "decline"


def test_consent_policy_lint_rejects_decline_reading_consent_fields():
    bad = """permit (principal, action, resource);
@id("bad") @outcome("decline")
forbid (principal, action, resource) when { context.consented };"""
    statements = split_policies(bad)
    annotations = decision._parse_annotations(statements)
    with pytest.raises(PolicyLoadError):
        decision.lint_policies(statements, annotations)


def test_decision_latency_budget():
    started = time.perf_counter()
    for _ in range(50):
        decision.decide("web_research", "converse", "task-1", {"source": "user_turn"})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms / 50 < 50  # generous; spec target ~2 ms


def test_shadow_only_policy_is_recorded_but_never_applied(tmp_path):
    from app.security.decision import load_policy_engine

    engine = load_policy_engine()
    untainted = engine.decide("web_research", "tool::web_search", "task-1",
                              {"source": "tool", "intent_alignment": 12})
    assert untainted.outcome == "allow"
    assert untainted.shadow_policy_ids == ("off-intent-search-verify",)
    tainted = engine.decide("web_research", "tool::web_search", "task-1",
                            {"source": "tool", "alignment": 12, "intent_alignment": 80, "tainted": True})
    assert tainted.outcome == "verify" and tainted.shadow_policy_ids == ()


def test_invalid_shadow_annotation_fails_loading(tmp_path):
    import shutil

    import pytest

    from app.security.decision import PolicyLoadError, load_policy_engine

    src = Path(__file__).resolve().parents[1] / "config" / "security"
    for f in src.iterdir():
        shutil.copy(f, tmp_path / f.name)
    text = (tmp_path / "policies.cedar").read_text().replace('@shadow("true")', '@shadow("maybe")')
    (tmp_path / "policies.cedar").write_text(text)
    with pytest.raises(PolicyLoadError, match="invalid @shadow"):
        load_policy_engine(tmp_path)
