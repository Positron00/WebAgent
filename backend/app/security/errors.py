"""Exceptions raised by the security guard.

Both derive from ``BaseException`` on purpose: agents wrap their work in ``except Exception`` and
would otherwise swallow a guard stop and carry on. Only the task manager and the API endpoints
catch them (SEC-046, SEC-036).
"""
from __future__ import annotations

from app.security.types import Decision


class GuardStop(BaseException):
    """A checkpoint stopped the request or task (consent, decline or escalate on a user turn,
    escalate on a tool action)."""

    def __init__(self, decision: Decision):
        self.decision = decision
        super().__init__(decision.reason)


class GuardDeclined(BaseException):
    """The safety net declined a model call whose prompt it scored as one unlabelled item."""

    def __init__(self, decision: Decision):
        self.decision = decision
        super().__init__(decision.reason)
