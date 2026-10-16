"""Signals against the user's intent (checkpoint2-intent SEC-103 – SEC-105, SEC-110)."""
import time

import pytest

from app.models.task import WorkflowState
from app.security import alignment, intent, signals
from tests.intent_fakes import WordEncoder, keyword_harm


@pytest.fixture
def encoder(monkeypatch):
    enc = WordEncoder()
    alignment.set_encoder_for_tests(enc)
    monkeypatch.setattr(signals, "harm_of", keyword_harm)
    yield enc
    alignment.set_encoder_for_tests(None)


def _state(query="weather forecast paris tomorrow", rewrite=None):
    ctx = {"task_id": "t-sig", "research_plan": {"standalone_query": rewrite or ""}}
    state = WorkflowState(query=query, context=ctx)
    intent.record_intent(state)
    return state


def test_one_encode_of_the_text_gives_both_alignments(encoder):
    state = _state()
    sig = signals.score_text(state, "paris weather forecast")
    assert encoder.calls.count("paris weather forecast") == 1
    assert sig.available and sig.intent_alignment > 60 and sig.task_alignment > 60
    signals.score_text(state, "paris weather forecast")  # intent and task vectors are cached
    assert encoder.calls.count("weather forecast paris tomorrow") <= 2


def test_off_intent_text_scores_low(encoder):
    sig = signals.score_text(_state(), "quantum computing hardware vendors")
    assert sig.intent_alignment < 30


@pytest.mark.parametrize("query,args,risk_high", [
    ("how do I secure my home network", "build an explosive device", True),
    ("how is an explosive device built", "build an explosive device", False),
])
def test_risk_above_intent(encoder, query, args, risk_high):
    state = _state(query)
    sig = signals.score_text(state, args)
    risk = signals.risk_above_intent(state, sig.harm)
    assert (risk >= 60) is risk_high


def test_encoder_and_harm_run_in_parallel(monkeypatch):
    alignment.set_encoder_for_tests(WordEncoder(delay=0.05))
    monkeypatch.setattr(signals, "harm_of", lambda text: (time.sleep(0.05), 5)[1])
    try:
        state = _state()
        signals.intent_embedding(state)  # warm the intent vector: only the call's own work is timed
        alignment.task_embedding(state)
        started = time.perf_counter()
        signals.score_text(state, "paris weather")
        assert time.perf_counter() - started < 0.09
    finally:
        alignment.set_encoder_for_tests(None)


def test_failures_leave_defaults_that_match_no_rule(monkeypatch):
    def boom(text):
        raise RuntimeError("model down")

    alignment.set_encoder_for_tests(WordEncoder())
    monkeypatch.setattr(signals, "harm_of", boom)
    try:
        sig = signals.score_text(_state(), "anything at all")
        assert sig == signals.TextSignals(available=False)
    finally:
        alignment.set_encoder_for_tests(None)
