"""The Together judge: parsing, errors, cache, model refusal, prices and stop rules (rl-generator SEC-203).

Offline: a fake client; no request leaves the process.
"""
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rl_generator import judge as judge_mod  # noqa: E402


class FakeClient:
    def __init__(self, answers):
        self.answers, self.prompts, self.tokens = list(answers), [], {"prompt": 0, "completion": 0}

    def invoke(self, prompt):
        self.prompts.append(prompt)
        self.tokens["prompt"] += 100
        self.tokens["completion"] += 10
        answer = self.answers.pop(0) if self.answers else '{"label_ok": true}'
        if isinstance(answer, Exception):
            raise answer
        return answer


def make(tmp_path, answers=(), **kw):
    client = FakeClient(answers)
    kw.setdefault("price_in", 1.0)
    kw.setdefault("price_out", 2.0)
    j = judge_mod.Judge("judge-model", "H", tmp_path / "cache.sqlite", client=client,
                        tokens_fn=lambda: dict(client.tokens), default_model="app-model", **kw)
    return j, client


@pytest.mark.parametrize("text,expected", [
    ('{"label_ok": true}', True), (' {"label_ok": false} trailing', False), ('{"label_ok": "yes"}', None),
    ("no json", None), ('{"label_ok": tru', None), ("", None)])
def test_parse(text, expected):
    assert judge_mod.parse(text) is expected


def test_answers_errors_and_tokens(tmp_path):
    j, client = make(tmp_path, ['{"label_ok": false}', "garbled", TimeoutError("timed out")])
    assert j("placeholder intent", "placeholder a", "plan") is False
    assert j("placeholder intent", "placeholder b", "plan") is None
    assert j("placeholder intent", "placeholder c", "plan") is None
    assert (j.calls, j.errors) == (3, 2)
    assert j.tokens == {"prompt": 300, "completion": 30}
    assert j.spent() == pytest.approx(300 / 1e6 * 1.0 + 30 / 1e6 * 2.0)
    assert "placeholder a" in client.prompts[0] and "label_ok" in client.prompts[0]


def test_cache_saves_a_second_call(tmp_path):
    j, client = make(tmp_path, ['{"label_ok": true}', "garbled"])
    assert j("placeholder intent", "placeholder a") is True
    assert j("placeholder intent", "placeholder a") is True
    assert len(client.prompts) == 1 and j.cache_hits == 1
    j.close()
    j2, client2 = make(tmp_path)
    assert j2("placeholder intent", "placeholder a") is True and not client2.prompts
    other_mode = judge_mod.Judge("judge-model", "L", tmp_path / "cache.sqlite", client=client2,
                                 tokens_fn=lambda: {}, default_model="app-model", price_in=1, price_out=1)
    assert other_mode.key("i", "", "c") != j2.key("i", "", "c")


def test_errors_are_not_cached(tmp_path):
    j, client = make(tmp_path, ["garbled", '{"label_ok": true}'])
    assert j("placeholder intent", "placeholder a") is None
    assert j("placeholder intent", "placeholder a") is True
    assert len(client.prompts) == 2


def test_equal_model_refused(tmp_path):
    with pytest.raises(SystemExit):
        judge_mod.Judge("same", "H", tmp_path / "c.sqlite", client=FakeClient([]), default_model="same",
                        price_in=1, price_out=1)


def test_default_model_from_settings(tmp_path):
    from app.core.config import settings

    with pytest.raises(SystemExit):
        judge_mod.Judge(settings.LLM_DEFAULT_MODEL, "H", tmp_path / "c.sqlite", client=FakeClient([]),
                        price_in=1, price_out=1)


def test_prices_required_unless_known(monkeypatch):
    assert judge_mod.PRICES == {}
    with pytest.raises(SystemExit):
        judge_mod.prices_for("m", None, None)
    assert judge_mod.prices_for("m", 0.5, 1.5) == (0.5, 1.5)
    monkeypatch.setitem(judge_mod.PRICES, "m", (0.2, 0.4))
    assert judge_mod.prices_for("m", None, 1.0) == (0.2, 1.0)


def test_key_only_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        judge_mod.Judge("judge-model", "H", tmp_path / "c.sqlite", default_model="app-model",
                        price_in=1, price_out=1)


def test_credit_stop(tmp_path):
    j, _ = make(tmp_path, credit_left=0.001, price_in=5.0, price_out=5.0)
    assert not j.must_stop()
    j("placeholder intent", "placeholder a")   # 110 tokens at $5/M = $0.00055 >= half of $0.001
    assert j.must_stop()
    j2, _ = make(tmp_path / "b", credit_left=1.0)
    assert not j2.must_stop() and j2.must_stop(next_cost=0.5)


def test_error_stop_per_step(tmp_path):
    j, _ = make(tmp_path, ["garbled"] * 3)
    for i in range(10):
        j("placeholder intent", f"placeholder {i}")
    assert j.errors == 3 and j.must_stop()
    j.begin_step()
    assert not j.error_stop()


def test_error_stop_over_last_100_seeds(tmp_path):
    j, _ = make(tmp_path, ["garbled", '{"label_ok": true}'] * 3)
    for i in range(6):   # 6 calls, 3 errors: below the minimum number of calls
        j("placeholder intent", f"placeholder {i}")
        j.end_seed()
    j.begin_step()
    assert not j.error_stop()
    for i in range(6, 12):
        j("placeholder intent", f"placeholder {i}")
        j.end_seed()
    j.begin_step()
    assert j.error_stop()   # 12 calls in the window, 3 errors = 25%
    for i in range(12, 112):
        j("placeholder intent", f"placeholder {i}")
        j.end_seed()
    j.begin_step()
    assert not j.error_stop()   # the failing seeds left the window


def test_real_token_counter_does_not_import_the_generation_script():
    """generate_alignment_calls sets process-wide guard settings on import; the judge must not load it."""
    import importlib

    sys.modules.pop("generate_alignment_calls", None)
    judge_mod = importlib.import_module("rl_generator.judge")
    tokens = judge_mod._default_tokens()
    assert set(tokens) == {"prompt", "completion"}
    assert "generate_alignment_calls" not in sys.modules
