"""Phase B intent-alignment classifier (checkpoint2-intent SEC-123 – SEC-127), all offline (SNFR-15)."""
import asyncio
import json

import joblib
import numpy as np
import pytest
import torch
from sklearn.isotonic import IsotonicRegression

from app.core import startup_checks
from app.core.config import settings
from app.models.task import WorkflowState
from app.security import alignment, alignment_model, intent, signals, tool_guard
from app.security.types import Decision
from tests.intent_fakes import WordEncoder, guard_on, keyword_harm

WORDS = "weather forecast paris tomorrow rain build an explosive device search for the of web_search plan".split()


def _identity():
    return IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.0, 1.0])


def write_fixture(d, *, passed=True, kind="cross_encoder", **meta_extra):
    """A tiny random BERT cross-encoder in the trainer's format; no download."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    enc = d / "encoder"
    enc.mkdir(parents=True)
    vocab = enc / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS, ":"]), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(str(enc))
    torch.manual_seed(0)
    config = BertConfig(vocab_size=200, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=128)
    BertModel(config).save_pretrained(str(enc))
    torch.save({"aligned": torch.nn.Linear(16, 1).state_dict(), "risk": torch.nn.Linear(16, 3).state_dict(),
                "hidden": 16, "pooling": "mean", "max_length": 128}, d / "heads.pt")
    joblib.dump({"aligned": _identity(), "risk": _identity()}, d / "calibrators.joblib")
    meta = {"kind": kind, "version": "ac-2026-10-16-1", "base_encoder": "tiny-bert", "max_length": 128,
            "thresholds": {"search_align_at": 40}, "uncertain_band": [0.3, 0.7], "data_hash": "sha256:x",
            "test": {"passed": passed}, **meta_extra}
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


class FakeModel(alignment_model.AlignmentModel):
    def __init__(self, p=0.9, risk=10, band=(0.3, 0.7), thresholds=None, fail=False):
        super().__init__("cross_encoder", {"version": "ac-test", "uncertain_band": list(band),
                                           "thresholds": thresholds or {}}, {})
        self.p, self.risk, self.fail, self.pairs = p, risk, fail, []

    def score_pair(self, intent_text, call_text):
        self.pairs.append((intent_text, call_text))
        if self.fail:
            raise RuntimeError("broken")
        return self.p, self.risk


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    alignment.set_encoder_for_tests(WordEncoder())
    harm_calls = []
    monkeypatch.setattr(signals, "harm_of", lambda text: harm_calls.append(text) or keyword_harm(text))
    yield harm_calls
    alignment.set_encoder_for_tests(None)
    alignment_model.set_model_for_tests(None)


def _state(query="weather forecast paris tomorrow"):
    state = WorkflowState(query=query, context={"task_id": "t-ac", "research_plan": {
        "requires_web_search": True, "requires_internal_knowledge": False, "standalone_query": ""}})
    intent.record_intent(state)
    return state


def _rows(tmp_path):
    return [json.loads(line) for f in sorted(tmp_path.rglob("collected/*.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_load_scores_a_gated_model_deterministically(tmp_path):
    model = alignment_model.load(write_fixture(tmp_path / "m"))
    assert model is not None and model.kind == "cross_encoder" and model.version == "ac-2026-10-16-1"
    first = model.score_pair("weather forecast paris", "web_search: paris rain tomorrow")
    assert 0.0 <= first[0] <= 1.0 and isinstance(first[1], int) and 0 <= first[1] <= 100
    assert all(model.score_pair("weather forecast paris", "web_search: paris rain tomorrow") == first
               for _ in range(3))  # SNFR-14
    assert model.uncertain_band() == (0.3, 0.7) and model.thresholds() == {"search_align_at": 40}


def test_ungated_missing_or_incomplete_models_are_not_used(tmp_path):
    ungated = write_fixture(tmp_path / "u", passed=False)
    assert alignment_model.load(ungated) is None
    assert alignment_model.load(ungated, require_gate=False) is not None  # the gate script's path
    assert alignment_model.load(tmp_path / "missing") is None
    (ungated / "calibrators.joblib").unlink()
    assert alignment_model.load(ungated, require_gate=False) is None
    assert alignment_model.load(write_fixture(tmp_path / "k", kind="other")) is None


def test_get_model_uses_the_configured_dir_and_backs_off(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path, alignment_model_dir=str(tmp_path / "none"))
    alignment_model.set_model_for_tests(None)
    assert alignment_model.get_model() is None and alignment_model.current_thresholds() == {}
    write_fixture(tmp_path / "none")
    assert alignment_model.get_model() is None  # within the failure backoff
    alignment_model.set_model_for_tests(None)
    assert alignment_model.get_model() is not None


def test_siamese_baseline_loads_and_scores(monkeypatch, tmp_path):
    from sklearn.linear_model import LogisticRegression

    d = tmp_path / "s"
    d.mkdir()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 256))
    joblib.dump({"aligned": LogisticRegression().fit(x, np.arange(30) % 2),
                 "risk": LogisticRegression().fit(x, np.arange(30) % 3)}, d / "heads.joblib")
    joblib.dump({"aligned": _identity(), "risk": _identity()}, d / "calibrators.joblib")
    (d / "metadata.json").write_text(json.dumps({"kind": "siamese", "version": "ac-2026-10-16-2",
                                                 "test": {"passed": True}}), encoding="utf-8")
    monkeypatch.setattr(alignment_model, "_defender_encoder", WordEncoder)
    model = alignment_model.load(d)
    p, risk = model.score_pair("paris weather", "web_search: paris rain")
    assert model.kind == "siamese" and 0 <= p <= 1 and 0 <= risk <= 100
    assert model.score_pair("paris weather", "web_search: paris rain") == (p, risk)


def test_siamese_features_and_embed():
    u, v = np.ones((1, 4)), np.zeros((1, 4))
    assert alignment_model.siamese_features(u, v).shape == (1, 16)
    emb = alignment_model.siamese_embed(["paris weather", "explosive"], WordEncoder())
    assert emb.shape == (2, 64) and np.allclose(np.linalg.norm(emb, axis=1), 1.0)


def test_signals_use_the_classifier_and_skip_context_defender(fakes):
    state = _state()
    signals.intent_harm(state)  # the intent's own harm is scored once per task, as in phase A
    fakes.clear()
    model = FakeModel(p=0.234, risk=77)
    alignment_model.set_model_for_tests(model)
    sig = signals.score_text(state, "build an explosive device", action="web_search")
    assert fakes == []  # no ContextDefender call per text (SNFR-10)
    assert model.pairs == [("weather forecast paris tomorrow", "web_search: build an explosive device")]
    assert (sig.intent_alignment, sig.risk, sig.harm, sig.classifier) == (23, 77, 77, True)
    assert sig.task_alignment < 30 and signals.call_risk(state, sig) == 77


def test_classifier_failure_falls_back_to_phase_a(fakes):
    alignment_model.set_model_for_tests(FakeModel(fail=True))
    sig = signals.score_text(_state(), "build an explosive device", action="web_search")
    assert sig.classifier is False and sig.risk is None and sig.harm == 90 and sig.available


def test_collection_row_has_classifier_available(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path)
    alignment_model.set_model_for_tests(FakeModel(p=0.12, risk=65))
    assert asyncio.run(tool_guard.guard_tool("tool::web_search", {"query": "build an explosive"}, _state()))
    ctx = _rows(tmp_path)[-1]["context"]
    assert ctx["classifier_available"] is True and ctx["intent_alignment"] == 12
    assert ctx["risk_above_intent"] == 65 and ctx["arg_harm"] == 65
    alignment_model.set_model_for_tests(None)
    monkeypatch.setattr(alignment_model, "_failed_until", float("inf"))  # no model on disk either
    asyncio.run(tool_guard.guard_tool("tool::web_search", {"query": "paris weather"}, _state()))
    assert _rows(tmp_path)[-1]["context"]["classifier_available"] is False


def test_plan_subgoal_harm_comes_from_risk(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path)
    model = FakeModel(p=0.8, risk=44)
    alignment_model.set_model_for_tests(model)
    intent._decide_text(_state(), "plan::subgoal", "paris rain", "plan", "supervisor")
    ctx = _rows(tmp_path)[-1]["context"]
    assert ctx["subgoal_harm"] == 44 and ctx["classifier_available"] is True
    assert model.pairs[-1][1] == "plan: paris rain"


def test_model_thresholds_override_and_are_clamped(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path)
    alignment_model.set_model_for_tests(FakeModel(p=0.6, thresholds={
        "search_align_at": 70, "risk_verify_at": 90, "risk_decline_at": 50, "harm_decline_at": 10}))
    ctx = tool_guard._build_context("tool::web_search", {"query": "paris rain"}, _state(), tool_mode="shadow")
    assert ctx["search_align_at"] == 70 and ctx["risk_verify_at"] == 50 and ctx["risk_decline_at"] == 50
    assert "harm_decline_at" not in ctx  # checkpoint-1 keys stay ContextDefender's
    assert ctx["_p_aligned"] == 0.6


def _verify_with(monkeypatch, tmp_path, p, policy="off-intent-search-verify"):
    guard_on(monkeypatch, tmp_path, tool_mode="act", judge=True)
    alignment_model.set_model_for_tests(FakeModel(p=p, band=(0.3, 0.7)))
    asked = []
    monkeypatch.setattr(tool_guard.guard_judge, "is_aligned", lambda *a: asked.append(a) or True)
    monkeypatch.setattr(tool_guard, "decide", lambda *a, **k: Decision("verify", (policy,), "check", False, 0.0))
    ran = asyncio.run(tool_guard.guard_tool("tool::web_search", {"query": "paris rain"}, _state()))
    return ran, len(asked)


@pytest.mark.parametrize("p, ran, judged", [(0.5, True, 1), (0.9, True, 0), (0.1, False, 0)])
def test_judge_band(monkeypatch, tmp_path, p, ran, judged):
    assert _verify_with(monkeypatch, tmp_path, p) == (ran, judged)


def test_other_verifies_still_go_to_the_judge(monkeypatch, tmp_path):
    assert _verify_with(monkeypatch, tmp_path, 0.95, policy="untrusted-code-verify") == (True, 1)


def _uat_guard(monkeypatch, tmp_path, model_dir):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "uat")
    monkeypatch.setattr(settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(
        update={"enabled": True, "alignment_model_dir": str(model_dir)}))
    monkeypatch.setattr(startup_checks, "defender_model_files_present", lambda *a: False)


def test_startup_refuses_an_ungated_model_in_uat(monkeypatch, tmp_path):
    _uat_guard(monkeypatch, tmp_path, tmp_path / "m")
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "metadata.json").write_text(json.dumps({"test": {"passed": False}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="alignment classifier"):
        startup_checks.check_security_guard_files()


@pytest.mark.parametrize("passed", [None, True])
def test_startup_accepts_a_missing_or_gated_model(monkeypatch, tmp_path, passed):
    _uat_guard(monkeypatch, tmp_path, tmp_path / "m")
    if passed:
        (tmp_path / "m").mkdir()
        (tmp_path / "m" / "metadata.json").write_text(json.dumps({"test": {"passed": True}}), encoding="utf-8")
    startup_checks.check_security_guard_files()


def test_code_calls_are_scored_on_the_code_itself(monkeypatch, tmp_path):
    """Code rows in training are the code itself; the runtime scores the same, capped (not the explanation)."""
    guard_on(monkeypatch, tmp_path)
    model = FakeModel()
    alignment_model.set_model_for_tests(model)
    code = "print('placeholder')\n" * 100
    asyncio.run(tool_guard.guard_tool("tool::execute_code", {"code": code}, _state("plot a chart"),
                                      scored_text="a placeholder explanation"))
    intent_text, call = model.pairs[-1]
    assert call.startswith("execute_code: print('placeholder')") and "explanation" not in call
    assert len(call) <= len("execute_code: ") + tool_guard.CODE_CALL_CHARS
