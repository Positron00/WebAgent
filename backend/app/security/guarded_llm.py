"""Guarded LLM wrapper: safety net when ``check_hop`` was not called (SEC-032)."""
from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langchain.schema.runnable import Runnable

from app.core.config import settings
from app.core.metrics import GUARD_UNLABELLED
from app.security.errors import GuardDeclined
from app.security.types import SEVERITY, ContextItem

logger = logging.getLogger(__name__)

_TEMPLATES: Dict[str, List[str]] = {}


def register_template(agent: str, template_text: str) -> None:
    """Register an agent's prompt template (its static text is never scored). An agent may have several."""
    if template_text and template_text not in _TEMPLATES.setdefault(agent, []):
        _TEMPLATES[agent].append(template_text)


_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _strip_template(agent: str, prompt: str) -> str:
    """Remove the registered template's static text (split on `{name}` placeholders)."""
    text = prompt
    for template in _TEMPLATES.get(agent, []):
        for part in _PLACEHOLDER.split(template):
            part = part.strip()
            if part and part in text:
                text = text.replace(part, "", 1)
    return text.strip()


# Per asyncio task / thread context, so parallel agents and concurrent tasks don't share it.
_CURRENT_HOP: ContextVar[Optional[str]] = ContextVar("guard_current_hop", default=None)
_HOP_META: ContextVar[Optional[Dict[str, Any]]] = ContextVar("guard_hop_meta", default=None)


def mark_hop(agent: str, decisions: Optional[List[Any]] = None) -> None:
    """Record that check_hop ran for `agent`; the next model call in this context skips the net."""
    _CURRENT_HOP.set(agent)
    if decisions:
        worst = max(decisions, key=lambda d: SEVERITY.index(d.outcome))
        ids = [pid for d in decisions for pid in d.policy_ids if pid != "baseline-permit"]
        _HOP_META.set({"guard.outcome": worst.outcome, "guard.policy_ids": list(dict.fromkeys(ids))})
    else:
        _HOP_META.set(None)


def clear_hop() -> None:
    _CURRENT_HOP.set(None)
    _HOP_META.set(None)


def current_hop_agent() -> Optional[str]:
    return _CURRENT_HOP.get()


class GuardedModel(Runnable):
    """LangChain runnable that delegates to an inner model after the safety net."""

    def __init__(self, inner: Any, *, agent: str):
        self.inner = inner
        self.agent = agent

    @staticmethod
    def _merge_config(config: Optional[Dict[str, Any]], meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Attach the guard's decision to the model run, so LangSmith shows it (SEC-083)."""
        if not meta:
            return config
        cfg = dict(config or {})
        cfg["metadata"] = {**dict(cfg.get("metadata") or {}), **meta}
        return cfg

    def _run_safety_net(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Return run metadata; raise when the net declines the call (act mode only)."""
        from app.security.hop_guard import check_hop

        if current_hop_agent() == self.agent:
            return _HOP_META.get()

        variable = _strip_template(self.agent, prompt)
        if not variable:
            return None

        GUARD_UNLABELLED.labels(agent=self.agent).inc()
        logger.warning("Guard safety net: %s invoked without check_hop", self.agent)
        hop = check_hop(self.agent, [ContextItem("unlabelled", variable)], state=None)
        if not hop.decisions:
            return None
        decision = max(hop.decisions, key=lambda d: SEVERITY.index(d.outcome))
        if not decision.shadow and not decision.error and decision.outcome in ("decline", "escalate", "consent"):
            raise GuardDeclined(decision)
        return {"guard.outcome": decision.outcome, "guard.policy_ids": list(decision.policy_ids)}

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        try:
            meta = self._run_safety_net(input if isinstance(input, str) else str(input))
            return self.inner.invoke(input, config=self._merge_config(config, meta), **kwargs)
        finally:
            clear_hop()

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        try:
            meta = self._run_safety_net(input if isinstance(input, str) else str(input))
            merged = self._merge_config(config, meta)
            if hasattr(self.inner, "ainvoke"):
                return await self.inner.ainvoke(input, config=merged, **kwargs)
            return self.inner.invoke(input, config=merged, **kwargs)
        finally:
            clear_hop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
