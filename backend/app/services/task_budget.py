"""Per-run token counter, budget, search cap and usage (IMP-042, REC-109).

`start` sets a fresh counter at the beginning of a run. Child tasks from `asyncio.gather`
and `asyncio.to_thread` copy the context, so they share the same counter object and their
tokens and searches add up. Parallel research works on deep copies of the state, which is why
the counter lives here and not in `state.context`. Outside a run there is no counter: `check`,
`add` and `charge_search` do nothing (and searches are not capped).
"""
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Dict, Optional

BUDGET_MESSAGE = "The research used its token budget and was stopped."

# Tavily credits per search, by search depth.
_CREDITS_BY_DEPTH = {"basic": 1, "advanced": 2}


class TaskBudgetExceeded(BaseException):
    """Derives from BaseException so agents' `except Exception` can't swallow it."""

    def __init__(self, used: int, limit: int):
        super().__init__(f"Token budget reached: {used} of {limit} tokens used")
        self.used = used
        self.limit = limit


@dataclass
class _Counter:
    limit: int
    used: int = 0  # total tokens
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    credits: int = 0
    search_cap: int = 0  # 0 = no cap


_counter: ContextVar[Optional[_Counter]] = ContextVar("task_budget", default=None)


def start(limit: int, search_cap: int = 0) -> Token:
    """Set a new counter for this run. A limit (or search cap) of 0 or less means none."""
    return _counter.set(_Counter(limit=max(0, int(limit or 0)), search_cap=max(0, int(search_cap or 0))))


def reset(token: Token) -> None:
    _counter.reset(token)


def check() -> None:
    """Raise TaskBudgetExceeded if this run has used its budget."""
    counter = _counter.get()
    if counter is not None and counter.limit > 0 and counter.used >= counter.limit:
        raise TaskBudgetExceeded(counter.used, counter.limit)


def _positive_int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def add(tokens: int, input_tokens: int = 0, output_tokens: int = 0) -> None:
    counter = _counter.get()
    if counter is None:
        return
    counter.used += _positive_int(tokens)
    counter.input_tokens += _positive_int(input_tokens)
    counter.output_tokens += _positive_int(output_tokens)


def charge_search(depth: str = "basic") -> bool:
    """Count one search before it is made; False (nothing counted) when the run's cap is reached.

    A search costs 1 Tavily credit at basic depth and 2 at advanced.
    """
    counter = _counter.get()
    if counter is None:
        return True
    if counter.search_cap > 0 and counter.searches >= counter.search_cap:
        return False
    counter.searches += 1
    counter.credits += _CREDITS_BY_DEPTH.get(str(depth or "basic").lower(), 1)
    return True


def search_cap() -> int:
    counter = _counter.get()
    return counter.search_cap if counter is not None else 0


def used() -> int:
    counter = _counter.get()
    return counter.used if counter is not None else 0


def current() -> Optional[_Counter]:
    """This run's counter object (None outside a run), for reading its usage from outside the run."""
    return _counter.get()


def usage(counter: Optional[_Counter] = None) -> Dict[str, int]:
    """This run's usage (or `counter`'s): {input_tokens, output_tokens, searches, tavily_credits}."""
    counter = counter if counter is not None else _counter.get()
    if counter is None:
        return {"input_tokens": 0, "output_tokens": 0, "searches": 0, "tavily_credits": 0}
    return {
        "input_tokens": counter.input_tokens,
        "output_tokens": counter.output_tokens,
        "searches": counter.searches,
        "tavily_credits": counter.credits,
    }
