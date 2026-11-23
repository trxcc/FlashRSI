"""Run sequential program proposals and delegate file tasks to Pi.

The host keeps the provider credential and accounts for each proposal made
within the shared episode deadline.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from flashrsi.agents.pi_agent import Pi
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.integrations.sandbox_runtime import (
    MODEL_CONNECT_TIMEOUT,
    MODEL_IDLE_TIMEOUT,
    PUBLIC_ORIGIN,
    key,
)

EpisodeRunner = Callable[[Agent, Task], Awaitable[Trajectory]]


class ProgramSearchTarget(Agent):
    name = "program_search"
    description = "Program proposals with immutable parent and a bounded native oracle."
    _container_dependencies = Pi._container_dependencies
    instruction_file = "prompt/instructions.md"

    @property
    def _env_runner(self) -> EpisodeRunner | None:
        return self._bound_runner

    @_env_runner.setter
    def _env_runner(self, runner: EpisodeRunner | None) -> None:
        # Agent.rollout calls the bound integration for the entire search.
        self._bound_runner = (
            None if runner is None else partial(self._run_episode, runner)
        )

    async def _run_episode(
        self, runner: EpisodeRunner, target: Agent, task: Task
    ) -> Trajectory:
        if _EPISODE_STEPS.get() is not None:
            return await runner(target, task)
        steps = []
        token = _EPISODE_STEPS.set(steps)
        try:
            async with self._episode_deadline():
                trajectory = await runner(target, task)
        finally:
            _EPISODE_STEPS.reset(token)
        if not steps:
            return trajectory
        return replace(
            trajectory,
            usage={
                **trajectory.usage,
                "model_calls": [step["usage"] for step in steps],
            },
            messages=(*steps, *trajectory.messages),
        )

    @asynccontextmanager
    async def _episode_deadline(self) -> AsyncIterator[None]:
        deadline = asyncio.timeout(_seconds(self.config.get("timeout", 900), "timeout"))
        try:
            async with deadline:
                yield
        except TimeoutError:
            if not deadline.expired():
                raise
            raise InfraError("program_search_episode_timeout") from None

    async def _run(self, prompt: str, working_dir: Path | None = None) -> Trajectory:
        config = copy.deepcopy(dict(self.config))
        config.pop("source_dir", None)
        config["name"] = "pi"
        instructions = (self.files_dir / self.instruction_file).read_text()
        return await Pi(config).run(instructions + "\n\n" + prompt, working_dir)

    async def propose(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        instructions = self.snapshot.files[self.instruction_file].content.decode()
        steps = _EPISODE_STEPS.get()
        if steps is not None:
            return await self._propose(instructions, prompt, steps)
        async with self._episode_deadline():
            return await self._propose(instructions, prompt, [])

    async def _propose(
        self,
        instructions: str,
        prompt: str,
        steps: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        llm = self.config.get("llm", {})
        cap = llm.get("max_tokens")
        if cap is None:
            cap = 65536
        if type(cap) is not int or cap <= 0:
            raise ValueError("ProgramSearch llm.max_tokens must be a positive integer")
        timeout = _seconds(
            llm.get(
                "request_timeout",
                self.config.get("request_timeout", MODEL_IDLE_TIMEOUT),
            ),
            "request_timeout",
        )
        credential = key()
        payload = {
            "model": "deepseek-flash",
            "reasoning_effort": "high",
            "thinking": {"type": "enabled"},
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": cap,
            "messages": [
                {
                    "role": "system",
                    "content": "Return one JSON object and no Markdown.",
                },
                {"role": "user", "content": instructions + "\n\n" + prompt},
            ],
        }
        step = {
            "type": "ProgramSearchProposal",
            "step": len(steps) + 1,
            "max_tokens": cap,
            "request_timeout": timeout,
            "connect_timeout": MODEL_CONNECT_TIMEOUT,
            "finish_reason": None,
            "usage": {},
        }
        steps.append(step)
        try:
            text = await _stream_proposal(payload, credential, timeout, step)
        except InfraError as exc:
            step["status"] = exc.reason
            raise
        if credential in text:
            raise InfraError("credential in response suppressed")
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError
        except ValueError:
            step["status"] = "invalid_json"
            raise ValueError("ProgramSearch proposal must be a JSON object") from None
        step["status"] = "finished"
        return value, step["usage"]


_EPISODE_STEPS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "program_search_steps", default=None
)


def _seconds(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"ProgramSearch {field} must be a finite positive number")
    return value


def _token_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the provider counts and expose consistent cache and reasoning totals."""
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    result = {field: usage[field] for field in fields if type(usage.get(field)) is int}
    for nested, field in (
        ("completion_tokens_details", "reasoning_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    ):
        breakdown = usage.get(nested)
        if isinstance(breakdown, dict) and type(breakdown.get(field)) is int:
            result[nested] = {field: breakdown[field]}
            result.setdefault(field, breakdown[field])
    if "prompt_cache_hit_tokens" in result:
        result.setdefault("cached_tokens", result["prompt_cache_hit_tokens"])
    elif "cached_tokens" in result:
        result["prompt_cache_hit_tokens"] = result["cached_tokens"]
    return result


async def _stream_proposal(
    payload: dict[str, Any],
    credential: str,
    timeout: float,
    step: dict[str, Any],
) -> str:
    pieces = []
    terminal = False
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=MODEL_CONNECT_TIMEOUT),
            trust_env=False,
        ) as client:
            async with client.stream(
                "POST",
                PUBLIC_ORIGIN + "/v1/chat/completions",
                headers={"Authorization": "Bearer " + credential},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    if not value:
                        continue
                    if value == "[DONE]":
                        terminal = True
                        break
                    event = json.loads(value)
                    if "error" in event:
                        raise InfraError("incomplete_stream")
                    if event.get("usage"):
                        step["usage"].update(_token_usage(event["usage"]))
                    for choice in event.get("choices", ()):
                        if choice.get("index", 0) != 0:
                            continue
                        finish = choice.get("finish_reason")
                        if finish:
                            if finish not in {
                                "stop",
                                "length",
                                "content_filter",
                                "tool_calls",
                                "function_call",
                            }:
                                raise InfraError("incomplete_stream")
                            step["finish_reason"] = finish
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            pieces.append(content)
        text = "".join(pieces)
    except httpx.TimeoutException as exc:
        reason = (
            "connection"
            if isinstance(exc, httpx.ConnectTimeout)
            else "program_search_stream_timeout"
        )
        raise InfraError(reason) from None
    except (httpx.TransportError, AttributeError, TypeError, ValueError):
        raise InfraError("incomplete_stream") from None
    finish = step["finish_reason"]
    if finish == "length":
        raise InfraError("program_search_output_length")
    if not terminal or not finish:
        raise InfraError("incomplete_stream")
    if finish != "stop":
        step["status"] = "incomplete_proposal"
        raise ValueError("ProgramSearch proposal did not finish with a JSON result")
    return text
