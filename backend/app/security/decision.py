"""Cedar policy loading, lint, and the shared decision point."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cedarpy
import yaml

from app.security.thresholds import DEFAULT_THRESHOLDS
from app.security.types import Decision, Outcome, SEVERITY

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = BACKEND_DIR / "config" / "security"

OUTCOME_RANK = {name: idx for idx, name in enumerate(SEVERITY)}
VALID_OUTCOMES = set(OUTCOME_RANK)
UNTRUSTED_SOURCES = (
    "web_result",
    "retrieved_doc",
    "document",
    "tool_output",
    "unlabelled",
)

DEFAULT_REASONS: Dict[Outcome, str] = {
    "allow": "Allowed by policy.",
    "verify": "This action needs a second check.",
    "consent": "This action needs your approval.",
    "decline": "This action was not allowed.",
    "escalate": "This task needs review.",
}


CONTEXT_DEFAULTS: Dict[str, Any] = {
    "injection": 0,
    "harm": 0,
    "sensitive_data": 0,
    "jailbreak": 0,
    "threat_risk": 0,
    "top_axis": "none",
    "scores_available": True,
    "pattern_hits": [],
    "tainted": False,
    "alignment": 100,
    "code_findings": [],
    "approved_tools": [],
    "consented": False,
    "has_image": False,
    "declined_actions_in_task": 0,
    "mode": "unknown",  # the chat mode (off / auto / always); not the guard's shadow/act mode
    "env": "dev",
    "code_execution_enabled": False,
    "sandbox": "local",
    # Checkpoint 2 against the user's intent (SEC-110): defaults that match no rule.
    "intent_alignment": 100,
    "rewrite_alignment": 100,
    "arg_harm": 0,
    "subgoal_harm": 0,
    "intent_harm": 0,
    "risk_above_intent": 0,
    "egress_overlap": 0,
    "low_alignment_calls": 0,
    "declared": True,
    "subgoal_origin": "plan",
    "signals_available": True,
    "classifier_available": False,
    **DEFAULT_THRESHOLDS,
}

_ID_RE = re.compile(r'@id\("([^"]+)"\)')
_OUTCOME_RE = re.compile(r'@outcome\("([^"]+)"\)')
_REASON_RE = re.compile(r'@reason\("([^"]+)"\)')
_SHADOW_RE = re.compile(r'@shadow\("([^"]+)"\)')


class PolicyLoadError(RuntimeError):
    """Policies failed lint, validation, or the offline test table."""


@dataclass
class PolicyAnnotations:
    cedar_key: str
    policy_id: str
    outcome: Outcome
    reason: Optional[str]
    shadow: bool = False  # @shadow("true"): a match is recorded but never changes the outcome


@dataclass
class PolicyEngine:
    policies: str
    schema: str
    annotations: Dict[str, PolicyAnnotations]
    policy_dir: Path

    def decide(
        self,
        principal: str,
        action: str,
        resource: str,
        context: Mapping[str, Any],
        *,
        shadow: Optional[bool] = None,
    ) -> Decision:
        started = time.perf_counter()
        full_context = build_context(context)
        shadow = bool(shadow)

        request = {
            "principal": _entity("Agent", principal),
            "action": f'Action::"{action}"',
            "resource": _resource(resource),
            "context": full_context,
        }
        result = cedarpy.is_authorized(request, self.policies, [], schema=self.schema)
        latency_ms = (time.perf_counter() - started) * 1000.0

        errors = list(getattr(result.diagnostics, "errors", []) or [])
        if errors:
            logger.error("Cedar decision error: %s", errors)
            return Decision(
                outcome="decline",
                policy_ids=tuple(),
                reason="Policy evaluation failed.",
                shadow=shadow,
                latency_ms=latency_ms,
                error=True,
            )

        if result.decision == cedarpy.Decision.Allow:
            return Decision(
                outcome="allow",
                policy_ids=("baseline-permit",),
                reason=DEFAULT_REASONS["allow"],
                shadow=shadow,
                latency_ms=latency_ms,
            )

        # cedarpy names policies policy0, policy1… in file order (SEC-012).
        matched_keys = sorted(getattr(result.diagnostics, "reasons", []) or [], key=_policy_order)
        outcomes: List[Tuple[Outcome, str, Optional[str]]] = []
        shadow_ids: List[str] = []
        for key in matched_keys:
            ann = self.annotations.get(key)
            if ann is None:
                outcomes.append(("decline", key, None))
                continue
            if ann.shadow:
                shadow_ids.append(ann.policy_id)
                continue
            outcomes.append((ann.outcome, ann.policy_id, ann.reason))

        if not outcomes and shadow_ids:
            # Only shadow-only policies matched: allow, and say which would have acted.
            return Decision(
                outcome="allow",
                policy_ids=("baseline-permit",),
                reason=DEFAULT_REASONS["allow"],
                shadow=shadow,
                latency_ms=latency_ms,
                shadow_policy_ids=tuple(dict.fromkeys(shadow_ids)),
            )

        if not outcomes:
            return Decision(
                outcome="decline",
                policy_ids=tuple(),
                reason=DEFAULT_REASONS["decline"],
                shadow=shadow,
                latency_ms=latency_ms,
            )

        worst = max(outcomes, key=lambda row: OUTCOME_RANK[row[0]])
        outcome, policy_id, reason = worst
        policy_ids = tuple(dict.fromkeys(row[1] for row in outcomes))
        return Decision(
            outcome=outcome,
            policy_ids=policy_ids,
            reason=reason or DEFAULT_REASONS[outcome],
            shadow=shadow,
            latency_ms=latency_ms,
            shadow_policy_ids=tuple(dict.fromkeys(shadow_ids)),
        )


_ENGINE: Optional[PolicyEngine] = None


def _policy_order(key: str) -> int:
    """Cedar reports matched policies in no fixed order; sort them by their place in the file
    (cedarpy names them policy0, policy1, …) so ids and shadow ids are deterministic (SNFR-06)."""
    suffix = key[len("policy"):] if key.startswith("policy") else ""
    return int(suffix) if suffix.isdigit() else 1_000_000


def _entity(kind: str, value: str) -> str:
    if value.startswith(f"{kind}::"):
        return value
    return f'{kind}::"{value}"'


def _resource(value: str) -> str:
    if value.startswith("Task::") or value.startswith("Chat::"):
        return value
    if value.startswith("task-") or value.startswith("chat-"):
        kind = "Chat" if value.startswith("chat-") else "Task"
        return f'{kind}::"{value}"'
    return f'Task::"{value}"'


def build_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(CONTEXT_DEFAULTS)
    merged.update(context)
    merged["pattern_hits"] = list(merged.get("pattern_hits") or [])
    merged["code_findings"] = list(merged.get("code_findings") or [])
    merged["approved_tools"] = list(merged.get("approved_tools") or [])
    if "source" not in context:
        raise ValueError("context.source is required")
    merged["source"] = context["source"]
    return merged


def split_policies(text: str) -> List[str]:
    """Split Cedar policies on ';' outside of strings and `//` comments (comments are dropped)."""
    parts: List[str] = []
    buf: List[str] = []
    in_string = False
    in_comment = False
    escape = False
    for i, ch in enumerate(text):
        if in_comment:
            if ch == "\n":
                in_comment = False
                buf.append(ch)
            continue
        if not in_string and ch == "/" and text[i + 1:i + 2] == "/":
            in_comment = True
            continue
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            buf.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            buf.append(ch)
            continue
        if ch == ";" and not in_string:
            stmt = "".join(buf).strip()
            if stmt:
                parts.append(stmt)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_annotations(statements: List[str]) -> Dict[str, PolicyAnnotations]:
    annotations: Dict[str, PolicyAnnotations] = {}
    for idx, stmt in enumerate(statements):
        key = f"policy{idx}"
        if stmt.startswith("permit"):
            annotations[key] = PolicyAnnotations(key, "baseline-permit", "allow", None)
            continue
        id_match = _ID_RE.search(stmt)
        outcome_match = _OUTCOME_RE.search(stmt)
        if not id_match or not outcome_match:
            raise PolicyLoadError(f"{key} is a forbid without @id and @outcome")
        outcome = outcome_match.group(1)
        if outcome not in VALID_OUTCOMES:
            raise PolicyLoadError(f"{key} has invalid @outcome {outcome}")
        reason_match = _REASON_RE.search(stmt)
        shadow_match = _SHADOW_RE.search(stmt)
        if shadow_match and shadow_match.group(1) not in ("true", "false"):
            raise PolicyLoadError(f"{key} has invalid @shadow {shadow_match.group(1)}")
        annotations[key] = PolicyAnnotations(
            key,
            id_match.group(1),
            outcome,  # type: ignore[arg-type]
            reason_match.group(1) if reason_match else None,
            shadow=bool(shadow_match and shadow_match.group(1) == "true"),
        )
    ids = [a.policy_id for a in annotations.values() if a.policy_id != "baseline-permit"]
    if len(ids) != len(set(ids)):
        raise PolicyLoadError("Duplicate @id values in policies.cedar")
    return annotations


def lint_policies(statements: List[str], annotations: Dict[str, PolicyAnnotations]) -> None:
    for idx, stmt in enumerate(statements):
        ann = annotations[f"policy{idx}"]
        if stmt.startswith("permit"):
            continue
        if ann.outcome in ("decline", "escalate") and (
            "approved_tools" in stmt or "consented" in stmt
        ):
            raise PolicyLoadError(f"{ann.policy_id} reads consent fields but is not a consent policy")
        if ann.outcome == "consent" and 'Action::"converse"' in stmt:
            # A consent rule on converse must be limited to user turns: the user can't vouch
            # for a web page (SEC-036). Rules without a source condition would apply to every source.
            limited = 'context.source == "user_turn"' in stmt
            if not limited or any(f'"{src}"' in stmt for src in UNTRUSTED_SOURCES):
                raise PolicyLoadError(
                    f"{ann.policy_id} is a consent policy that is not limited to user turns"
                )


def _run_policy_tests(engine: PolicyEngine, tests_path: Path) -> None:
    data = yaml.safe_load(tests_path.read_text(encoding="utf-8")) or {}
    for row in data.get("tests", []):
        decision = engine.decide(
            row["principal"],
            row["action"],
            row["resource"],
            row.get("context") or {},
        )
        expected = row["expected"]
        if decision.outcome != expected:
            raise PolicyLoadError(
                f"Policy test '{row.get('name', row['action'])}' expected {expected}, "
                f"got {decision.outcome} ({decision.policy_ids})"
            )
        if "expected_shadow" in row and list(decision.shadow_policy_ids) != list(row["expected_shadow"]):
            raise PolicyLoadError(
                f"Policy test '{row.get('name', row['action'])}' expected shadow matches "
                f"{row['expected_shadow']}, got {list(decision.shadow_policy_ids)}"
            )


def load_policy_engine(policy_dir: Optional[Path] = None) -> PolicyEngine:
    policy_dir = policy_dir or DEFAULT_POLICY_DIR
    policies_path = policy_dir / "policies.cedar"
    schema_path = policy_dir / "schema.cedarschema"
    tests_path = policy_dir / "policy_tests.yaml"
    policies = policies_path.read_text(encoding="utf-8")
    schema = schema_path.read_text(encoding="utf-8")
    statements = split_policies(policies)
    annotations = _parse_annotations(statements)
    lint_policies(statements, annotations)
    validation = cedarpy.validate_policies(policies, schema)
    if not validation.validation_passed:
        raise PolicyLoadError(f"Schema validation failed with {validation.num_errors} errors")
    engine = PolicyEngine(policies=policies, schema=schema, annotations=annotations, policy_dir=policy_dir)
    _run_policy_tests(engine, tests_path)
    return engine


def get_engine(force_reload: bool = False) -> PolicyEngine:
    global _ENGINE
    if _ENGINE is None or force_reload:
        _ENGINE = load_policy_engine()
    return _ENGINE


def decide(
    principal: str,
    action: str,
    resource: str,
    context: Mapping[str, Any],
    *,
    shadow: Optional[bool] = None,
) -> Decision:
    return get_engine().decide(principal, action, resource, context, shadow=shadow)
