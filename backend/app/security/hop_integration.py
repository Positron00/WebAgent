"""Helpers for agent prompt building after checkpoint 1."""
from __future__ import annotations

from typing import Awaitable, Callable, List, Optional, TypeVar

from app.models.task import WorkflowState
from app.security.hop_guard import acheck_hop, task_items
from app.security.types import ContextItem, HopResult

T = TypeVar("T")


async def run_with_hop(
    agent: str,
    state: WorkflowState,
    variable_items: List[ContextItem],
    build_prompt: Callable[[HopResult], str],
    invoke: Callable[[str], Awaitable[T]],
) -> tuple[T, HopResult]:
    hop = await acheck_hop(agent, task_items(state) + variable_items, state)
    prompt = build_prompt(hop)
    result = await invoke(prompt)
    return result, hop


def web_result_items(results: List[dict]) -> List[ContextItem]:
    items: List[ContextItem] = []
    for row in results:
        items.append(
            ContextItem(
                "web_result",
                str(row.get("content") or ""),
                ref=str(row.get("url") or row.get("title") or ""),
            )
        )
    return items


def format_web_items(items: List[ContextItem]) -> str:
    if not items:
        return "No relevant search results found."
    formatted = ""
    for i, item in enumerate(items, 1):
        formatted += f"SOURCE {i}:\n"
        formatted += f"URL: {item.ref or 'No URL'}\n"
        formatted += f"Content: {item.text}\n\n"
    return formatted
