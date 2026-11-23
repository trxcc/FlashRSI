"""Decode Responses SSE framing, completion usage, and reasoning deltas.

This module processes one response stream and holds no client state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx


async def sse_events(response: httpx.Response) -> AsyncIterator[str]:
    """SSE framing equivalent to eventsource_stream, across arbitrary chunks."""
    data = []
    first = True
    async for line in response.aiter_lines():
        if first:
            line, first = line.removeprefix("\ufeff"), False
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value.removeprefix(" "))


def usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    # sse/responses.rs::From<ResponseCompletedUsage> for TokenUsage.
    if usage is None:
        return {}
    result = {
        key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    result.update(
        cached_input_tokens=details["cached_tokens"] if details is not None else 0,
        cache_write_input_tokens=details.get("cache_write_tokens", 0)
        if details is not None
        else 0,
        reasoning_output_tokens=output_details["reasoning_tokens"]
        if output_details is not None
        else 0,
    )
    if any(
        type(value) is not int or not -(2**63) <= value < 2**63
        for value in result.values()
    ):
        raise ValueError("token usage fields must be 64-bit integers")
    if usage.get("codex_rollout_budget_units") is not None:
        result["codex_rollout_budget_units"] = usage["codex_rollout_budget_units"]
    return result


def remember_reasoning(
    event: Mapping[str, Any], reasoning: dict[int | None, str]
) -> None:
    kind = event["type"]
    index = event.get("output_index")
    if kind == "response.reasoning_text.delta":
        reasoning[index] = reasoning.get(index, "") + (event.get("delta") or "")
    elif kind == "response.reasoning_text.done" and event.get("text"):
        reasoning[index] = event["text"]


def with_reasoning_content(item: Any, accumulated: str | None) -> Any:
    if (
        not isinstance(item, dict)
        or item.get("type") != "reasoning"
        or item.get("content")
    ):
        return item
    trace = item.get("reasoning_content") or item.get("reasoning_text") or accumulated
    if isinstance(trace, str):
        return {**item, "content": [{"type": "reasoning_text", "text": trace}]}
    return item
