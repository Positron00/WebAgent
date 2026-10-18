"""Which run is executing, for code that must treat scheduled runs differently (REC-108).

`topic_run` holds the topic ID while a scheduled (topic) run executes, and is None otherwise.
Tiny and dependency-free so drift and security records can import it cheaply.
"""
from contextvars import ContextVar
from typing import Optional

topic_run: ContextVar[Optional[str]] = ContextVar("topic_run", default=None)


def current_origin() -> str:
    """"schedule" during a topic run, else "interactive"."""
    return "schedule" if topic_run.get() else "interactive"
