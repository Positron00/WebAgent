"""Per-run token counter and budget (IMP-042).

`start` sets a fresh counter at the beginning of a run. Child tasks from `asyncio.gather`
and `asyncio.to_thread` copy the context, so they share the same counter object and their
tokens add up. Outside a run there is no counter: `check` and `add` do nothing.
"""
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Optional

BUDGET_MESSAGE = "The research used its token budget and was stopped."


class TaskBudgetExceeded(BaseException):
    """Derives from BaseException so agents' `except Exception` can't swallow it."""

    def __init__(self, used: int, limit: int):
        super().__init__(f"Token budget reached: {used} of {limit} tokens used")
        self.used = used
        self.limit = limit


@dataclass
class _Counter:
    limit: int
    used: int = 0


_counter: ContextVar[Optional[_Counter]] = ContextVar("task_budget", default=None)


def start(limit: int) -> Token:
    """Set a new counter for this run. A limit of 0 (or less) means no budget."""
    return _counter.set(_Counter(limit=max(0, int(limit or 0))))


def reset(token: Token) -> None:
    _counter.reset(token)


def check() -> None:
    """Raise TaskBudgetExceeded if this run has used its budget."""
    counter = _counter.get()
    if counter is not None and counter.limit > 0 and counter.used >= counter.limit:
        raise TaskBudgetExceeded(counter.used, counter.limit)


def add(tokens: int) -> None:
    counter = _counter.get()
    if counter is not None and isinstance(tokens, int) and tokens > 0:
        counter.used += tokens


def used() -> int:
    counter = _counter.get()
    return counter.used if counter is not None else 0
