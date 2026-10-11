"""classify(): threshold, input joining, fallback, kill switch, reasons, metrics, logs, tie-breaker."""
import logging
import time

import pytest

from app.core import metrics
from app.core.config import settings
from app.services import triage, triage_model
from app.services.triage import build_model_input, classify
from tests.fakes.triage import FakeTriageModel


@pytest.fixture(autouse=True)
def fake_model():
    triage_model.set_model_for_tests(FakeTriageModel(p=0.1))
    yield


def _set_triage(monkeypatch, **changes):
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update=changes))


def _counter(metric, **labels) -> float:
    return metrics.WEBAGENT_REGISTRY.get_sample_value(metric, labels) or 0.0


def test_build_model_input_joins_short_followup():
    assert build_model_input("what about Europe?", "compare US healthcare") == (
        "compare US healthcare → what about Europe?"
    )


def test_build_model_input_keeps_long_question_alone():
    long_q = "how does the healthcare system in Europe compare on cost"
    assert build_model_input(long_q, "compare US healthcare") == long_q


def test_build_model_input_truncates():
    text = build_model_input("x" * 5000, None)
    assert len(text) == 2000


def test_low_score_routes_direct():
    d = classify("explain list comprehensions")
    assert d.route == "direct"
    assert d.decided_by == "model"


def test_high_score_routes_agentic():
    triage_model.set_model_for_tests(FakeTriageModel(p=0.9))
    assert classify("compare top CRM tools with pricing").route == "agentic"


def test_threshold_from_metadata_and_env_override(monkeypatch):
    triage_model.set_model_for_tests(FakeTriageModel(p=0.3, threshold=0.35))
    assert classify("some ordinary question here").route == "direct"
    _set_triage(monkeypatch, threshold=0.0)
    assert classify("some ordinary question here").route == "agentic"


def test_fallback_when_no_model():
    triage_model.set_model_for_tests(None)
    d = classify("hello world test query here")
    assert (d.route, d.decided_by, d.reason) == ("agentic", "fallback", "Classifier unavailable")


def test_fallback_when_model_raises():
    def boom(_text):
        raise RuntimeError("broken")

    triage_model.set_model_for_tests(FakeTriageModel(p=boom))
    d = classify("a question that reaches the model")
    assert (d.route, d.decided_by, d.reason) == ("agentic", "fallback", "Classifier error")


def test_kill_switch_skips_rules_and_model(monkeypatch):
    _set_triage(monkeypatch, enabled=False)
    d = classify("hi")
    assert (d.route, d.decided_by) == ("agentic", "config")


def test_reason_names_category_and_never_the_question():
    triage_model.set_model_for_tests(FakeTriageModel(p=0.1, category="coding help"))
    query = "why does my unique-zebra-function fail"
    d = classify(query)
    assert d.reason == "Looks like: coding help"
    assert query not in d.reason


def test_same_input_same_output():
    assert classify("explain closures").to_dict()["route"] == classify("explain closures").to_dict()["route"]


def test_metrics_count_decisions_and_uncertain():
    triage_model.set_model_for_tests(FakeTriageModel(p=0.3, threshold=0.35))
    before = _counter("webagent_triage_decisions_total", route="direct", decided_by="model")
    uncertain_before = _counter("webagent_triage_uncertain_total")
    classify("a borderline question for the model")
    assert _counter("webagent_triage_decisions_total", route="direct", decided_by="model") == before + 1
    assert _counter("webagent_triage_uncertain_total") == uncertain_before + 1


def test_question_text_logged_only_in_dev(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.services.triage")
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "prod")
    classify("secret question text")
    assert "secret question text" not in caplog.text
    caplog.clear()
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "dev")
    classify("secret question text")
    assert "secret question text" in caplog.text


def test_tiebreak_can_flip_a_borderline_decision(monkeypatch):
    _set_triage(monkeypatch, llm_tiebreak=True)
    triage_model.set_model_for_tests(FakeTriageModel(p=0.4, threshold=0.35))
    monkeypatch.setattr(triage, "_ask_llm", lambda prompt: "DIRECT")
    assert classify("is this borderline or not").route == "direct"


def test_tiebreak_not_used_outside_band(monkeypatch):
    _set_triage(monkeypatch, llm_tiebreak=True)
    triage_model.set_model_for_tests(FakeTriageModel(p=0.9, threshold=0.35))
    monkeypatch.setattr(triage, "_ask_llm", lambda prompt: "DIRECT")
    assert classify("clearly research question").route == "agentic"


def test_tiebreak_error_or_timeout_keeps_model_route(monkeypatch):
    _set_triage(monkeypatch, llm_tiebreak=True)
    triage_model.set_model_for_tests(FakeTriageModel(p=0.4, threshold=0.35))
    monkeypatch.setattr(triage, "TIEBREAK_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(triage, "_ask_llm", lambda prompt: time.sleep(0.5) or "DIRECT")
    assert classify("is this borderline or not").route == "agentic"

    def fail(prompt):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(triage, "_ask_llm", fail)
    assert classify("is this borderline or not").route == "agentic"


def test_mode_and_override_decisions_leave_latency_alone():
    from app.services.routing import choose_route
    from app.services.triage import TriageDecision

    count = lambda: _counter("webagent_triage_latency_seconds_count")  # noqa: E731
    before = count()
    choose_route("off", "q", None, False)
    choose_route("always", "q", None, False)
    metrics.record_triage_decision(TriageDecision("agentic", None, "override", "r", "none", 0.0))
    assert count() == before
    classify("a question for the model")
    assert count() == before + 1


def test_load_error_is_remembered_and_never_raises(monkeypatch):
    calls = []

    def broken_load(*_args, **_kwargs):
        calls.append(1)
        raise ValueError("corrupt classifier.joblib")

    triage_model.set_model_for_tests(None)
    monkeypatch.setattr(triage_model, "load_model", broken_load)
    assert triage_model.get_model() is None
    assert triage_model.get_model() is None
    assert len(calls) == 1  # the 60 s backoff stops a reload on every request
    triage_model.warm_up()  # must not raise, or the server wouldn't start in dev
    decision = classify("a question that needs the model")
    assert (decision.route, decision.decided_by) == ("agentic", "fallback")
