"""Who the rate limiter counts: web app session, nearest untrusted hop, or the peer (IMP-012)."""
from types import SimpleNamespace

import pytest

from app.core import security
from app.core.config import settings

KEY = "server-only-test-key"
TRUSTED = "127.0.0.1"
UNTRUSTED = "198.51.100.4"


def _request(peer, headers=None):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers or {})


@pytest.fixture(autouse=True)
def _api_keys_on(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setenv("WEBAGENT_API_KEY", KEY)


CASES = [
    ("untrusted peer, no key, any headers",
     UNTRUSTED, {"x-webagent-client": "s:abc", "x-forwarded-for": "1.2.3.4"}, UNTRUSTED),
    ("untrusted peer with a valid key (the web app on Vercel)",
     UNTRUSTED, {"x-api-key": KEY, "x-webagent-client": "s:abc"}, "s:abc"),
    ("untrusted peer with a wrong key",
     UNTRUSTED, {"x-api-key": "wrong", "x-webagent-client": "s:abc"}, UNTRUSTED),
    ("trusted peer with a session header",
     TRUSTED, {"x-webagent-client": "s:abc"}, "s:abc"),
    ("trusted peer, right-most untrusted forwarded hop",
     TRUSTED, {"x-forwarded-for": "1.1.1.1, 2.2.2.2"}, "2.2.2.2"),
    ("trusted peer, trusted hops are skipped",
     TRUSTED, {"x-forwarded-for": "1.1.1.1, 127.0.0.1"}, "1.1.1.1"),
    ("trusted peer, overlong session header falls through",
     TRUSTED, {"x-webagent-client": "s:" + "a" * 198, "x-forwarded-for": "2.2.2.2"}, "2.2.2.2"),
    ("trusted peer, header without the s: prefix is ignored",
     TRUSTED, {"x-webagent-client": "abc", "x-forwarded-for": "2.2.2.2"}, "2.2.2.2"),
    ("trusted peer, empty session id is ignored",
     TRUSTED, {"x-webagent-client": "s:"}, TRUSTED),
    ("browser-sent forwarded header through the web app keeps the session",
     TRUSTED, {"x-webagent-client": "s:abc", "x-forwarded-for": "1.2.3.4"}, "s:abc"),
    ("nothing", TRUSTED, {}, TRUSTED),
]


@pytest.mark.parametrize("name,peer,headers,expected", CASES, ids=[c[0] for c in CASES])
def test_client_identifier(name, peer, headers, expected):
    assert security.client_identifier(_request(peer, headers)) == expected


def test_valid_key_earns_nothing_when_api_keys_are_off(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    request = _request(UNTRUSTED, {"x-api-key": KEY, "x-webagent-client": "s:abc"})

    assert security.client_identifier(request) == UNTRUSTED


def test_missing_client_counts_as_unknown():
    request = SimpleNamespace(client=None, headers={"x-webagent-client": "s:abc"})

    assert security.client_identifier(request) == "unknown"


def test_two_vercel_sessions_get_separate_budgets(monkeypatch):
    monkeypatch.setattr(security, "rate_limit_data", {})
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 1)

    def session(name):
        return _request(UNTRUSTED, {"x-api-key": KEY, "x-webagent-client": name})

    assert security.apply_rate_limit(security.client_identifier(session("s:one")))
    assert security.apply_rate_limit(security.client_identifier(session("s:two")))
    assert not security.apply_rate_limit(security.client_identifier(session("s:one")))
