"""Regression tests for gaps found when checking the security implementation against the spec."""
import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.models.task import WorkflowState
from app.security import context_defender, hop_guard, ledger, patterns, records, tokens, tool_guard
from app.security.errors import GuardDeclined, GuardStop
from app.security.guarded_llm import (
    GuardedModel,
    _strip_template,
    clear_hop,
    current_hop_agent,
    mark_hop,
    register_template,
)
from app.security.types import ContextItem, Decision


def _guard(monkeypatch, **update):
    monkeypatch.setattr(settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(update={"enabled": True, **update}))


# --- guard stops are not swallowed by agents' broad handlers (SEC-046) ---

def test_guard_stop_is_not_caught_by_except_exception():
    decision = Decision("escalate", ("p",), "r", False, 0.0)
    with pytest.raises(GuardStop):
        try:
            raise GuardStop(decision)
        except Exception:  # what agents do
            pytest.fail("GuardStop must not be an Exception")
    assert not issubclass(GuardDeclined, Exception)


# --- tokens (SEC-050–SEC-053) ---

def test_token_round_trip_binding_expiry_reuse_and_tamper(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    binding = tokens.consent_binding("q", [])
    t = tokens.issue("consent", binding, "p1")
    assert tokens.redeem(t, "approval", binding) is None          # wrong kind
    assert tokens.redeem(t, "consent", tokens.consent_binding("other", [])) is None  # wrong binding
    payload = tokens.redeem(t, "consent", binding)
    assert payload and payload["policy_id"] == "p1"
    assert tokens.redeem(t, "consent", binding) is None            # single use (shared claim folder)
    body, sig = tokens.issue("consent", binding, "p1").split(".")
    assert tokens.redeem(body + "." + sig[::-1], "consent", binding) is None   # tampered
    expired = tokens.issue("consent", binding, "p1", ttl_s=-1)
    assert tokens.redeem(expired, "consent", binding) is None


def test_approval_token_carries_the_signed_action(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    b = {"query_hash": tokens.query_hash("q")}
    payload = tokens.redeem(tokens.issue("approval", b, "p", action="tool::execute_code"), "approval", b)
    assert payload["action"] == "tool::execute_code"


def test_unavailable_token_store_means_no_consent(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(settings, "DATA_DIR", str(blocker))  # a file, so the claim folder can't be created
    b = tokens.consent_binding("q", [])
    assert tokens.redeem(tokens.issue("consent", b, "p"), "consent", b) is None


def test_token_claims_are_cleaned_up_after_their_lifetime(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    claims = tmp_path / "security" / "tokens"
    claims.mkdir(parents=True)
    old, new = claims / "old-jti", claims / "new-jti"
    old.write_text(""); new.write_text("")
    past = time.time() - 2 * 3600
    os.utime(old, (past, past))
    assert records.cleanup() >= 1
    assert not old.exists() and new.exists()


def test_escalation_record_without_args(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    records.record_escalation(hop="intake", action="converse",
                              decision=Decision("escalate", ("p",), "r", False, 0.0), state_context={})
    assert list((tmp_path / "security" / "escalations").glob("*.jsonl"))


# --- patterns (SEC-021, SEC-041) ---

def test_user_turn_exception_is_per_string_not_per_kind():
    assert "email" not in patterns.scan_args("contact bob@x.com", user_turn="email bob@x.com")
    assert "email" in patterns.scan_args("contact eve@evil.example", user_turn="email bob@x.com")


def test_long_query_is_still_scanned_for_secrets():
    hits = patterns.scan_args(("word " * 70) + "sk-proj-" + "a" * 40, user_turn="")
    assert {"query_too_long", "api_key"} <= hits


@pytest.mark.parametrize("key", ["sk-proj-" + "a1" * 20, "sk-live-" + "b2" * 20, "tvly-" + "c3" * 10,
                                 "lsv2_" + "d4" * 12, "ghp_" + "e5" * 18])
def test_modern_key_formats(key):
    assert "api_key" in patterns.scan(f"my key is {key}")


# --- safety net (SEC-032, SEC-083) ---

def test_template_with_named_placeholders_is_stripped():
    register_template("review_test_agent", "You are an agent. Answer {query} using {documents}. Be brief.")
    stripped = _strip_template("review_test_agent", "You are an agent. Answer WHAT using DOCS. Be brief.")
    assert "You are an agent" not in stripped and "Be brief" not in stripped


def test_hop_marker_is_per_context():
    async def agent(name):
        mark_hop(name)
        await asyncio.sleep(0)
        return current_hop_agent()

    async def run():
        return await asyncio.gather(agent("a1"), agent("a2"))

    assert asyncio.run(run()) == ["a1", "a2"]
    clear_hop()


def test_safety_net_does_not_block_in_shadow_mode(monkeypatch):
    _guard(monkeypatch, input_mode="shadow")
    monkeypatch.setattr(hop_guard, "decide",
                        lambda *a, **k: Decision("decline", ("p",), "r", bool(k.get("shadow")), 0.0))

    class _Inner:
        def invoke(self, input, config=None, **kwargs):
            return config

    clear_hop()
    assert GuardedModel(_Inner(), agent="shadow_agent").invoke("unlabelled text here") is not None


def test_guard_metadata_reaches_the_model_run(monkeypatch):
    class _Inner:
        def invoke(self, input, config=None, **kwargs):
            return config

    mark_hop("meta_agent", [Decision("verify", ("untrusted-injection-watch",), "r", False, 0.0)])
    config = GuardedModel(_Inner(), agent="meta_agent").invoke("prompt")
    assert config["metadata"]["guard.outcome"] == "verify"


# --- checkpoint 1 failure handling and chunks (SEC-036, SEC-061) ---

def test_policy_error_fails_open_for_user_turns_closed_for_untrusted(monkeypatch):
    _guard(monkeypatch, input_mode="act")
    monkeypatch.setattr(hop_guard, "decide",
                        lambda *a, **k: Decision("decline", (), "Policy evaluation failed.", False, 0.0, error=True))
    state = WorkflowState(query="hello", context={"task_id": "t-err"})
    hop = hop_guard.check_hop("web_research", [ContextItem("user_turn", "hello"),
                                               ContextItem("web_result", "page", ref="u")], state)
    assert [i.source for i in hop.items] == ["user_turn"]


def test_removing_chunks_does_not_repeat_overlapping_text():
    words = [f"w{i}" for i in range(500)]
    text = " ".join(words)
    kept = context_defender.text_without_chunks(text, [1], None).split()
    assert len(kept) == len(set(kept))  # no duplicated overlap
    a, b = context_defender.windows(500)[1]
    assert all(f"w{i}" not in kept for i in range(a, b))


# --- checkpoint 2 failure handling and approvals (SEC-051, SEC-061) ---

def test_policy_error_fails_closed_for_code_open_for_search(monkeypatch):
    _guard(monkeypatch, tool_mode="act")
    monkeypatch.setattr(tool_guard, "decide",
                        lambda *a, **k: Decision("decline", (), "Policy evaluation failed.", False, 0.0, error=True))
    state = WorkflowState(query="q", context={"task_id": "t2"})
    assert asyncio.run(tool_guard.guard_tool("tool::web_search", {"query": "q"}, state)) is True
    assert asyncio.run(tool_guard.guard_tool("tool::execute_code", {"code": "print(1)"}, state)) is False


def test_consent_skip_issues_an_approval_token(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    _guard(monkeypatch, tool_mode="act")
    monkeypatch.setattr(tool_guard, "decide",
                        lambda *a, **k: Decision("consent", ("tainted-code-consent",), "needs approval", False, 0.0))
    state = WorkflowState(query="plot it", context={"task_id": "t3"})
    assert asyncio.run(tool_guard.guard_tool("tool::execute_code", {"code": "print(1)"}, state)) is False
    approval = ledger.get_ledger(state)["approvals"][0]
    payload = tokens.redeem(approval["approval_token"], "approval", {"query_hash": tokens.query_hash("plot it")})
    assert payload["action"] == "tool::execute_code"
