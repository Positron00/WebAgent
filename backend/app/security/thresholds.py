"""Per-axis score thresholds the checkpoint-1 policies act on (SEC-028a).

Each ContextDefender version chooses its own on held-out rows (stored in its metadata.json), because
calibrated scores from different models don't cross the same false-positive rate at the same
number. The guard passes them into the policy context; the policies compare against them.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

# Used when no model is loaded, or a model has no thresholds of its own (the v1 policy literals).
DEFAULT_THRESHOLDS: Dict[str, int] = {
    "injection_watch_at": 20,
    "injection_drop_at": 40,
    # A user turn scoring from *_verify_at up to the decline/consent score goes to the LLM judge.
    "harm_verify_at": 50,
    "harm_decline_at": 85,
    "jailbreak_verify_at": 50,
    "jailbreak_consent_at": 70,
    # Checkpoint 2 against the user's intent (checkpoint2-intent spec, SEC-109). Starting values for
    # shadow-only rules, from task A10's offline run (2026-10-16): search 25 holds 95.9% of InjecAgent's
    # direct-harm attacks in untainted tasks while 6.6% of benign plan questions score below it;
    # subgoal 15 leaves 1.6% of them. They change only what is recorded until a rule acts (SEC-130).
    "rewrite_at": 40,
    "subgoal_align_at": 15,
    "search_align_at": 25,
    "drift_align_at": 25,
    "drift_calls_at": 3,
    "risk_verify_at": 30,
    "risk_decline_at": 60,
    "egress_words_at": 8,
}


def merged(model_thresholds: Optional[Mapping[str, int]]) -> Dict[str, int]:
    """Defaults overlaid with a model's values, clamped to 1-100 and kept ordered
    (watch < drop, verify <= decline/consent)."""
    out = dict(DEFAULT_THRESHOLDS)
    for key, value in (model_thresholds or {}).items():
        if key in out:
            out[key] = max(1, min(100, int(value)))
    out["injection_watch_at"] = min(out["injection_watch_at"], out["injection_drop_at"] - 1)
    out["harm_verify_at"] = min(out["harm_verify_at"], out["harm_decline_at"])
    out["jailbreak_verify_at"] = min(out["jailbreak_verify_at"], out["jailbreak_consent_at"])
    out["risk_verify_at"] = min(out["risk_verify_at"], out["risk_decline_at"])
    return out
