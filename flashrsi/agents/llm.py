"""A stateless text Worker using the OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from flashrsi._records import _redact, _secrets
from flashrsi.core import Agent, InfraError, Trajectory
from flashrsi.core.infra import infra_reason


def _completion_options(llm: dict[str, Any], model: str, base_url: str) -> None:
    cap = llm.pop("max_tokens", None)
    thinking = llm.pop("thinking", None)
    if thinking is not None:
        llm["extra_body"] = {**(llm.get("extra_body") or {}), "thinking": thinking}
    if llm.get("max_completion_tokens") is None:
        llm.pop("max_completion_tokens", None)
        token_field = "max_tokens"
        if base_url.rstrip("/") == "https://api.openai.com/v1" and model.startswith(
            ("gpt-5", "o1", "o3", "o4")
        ):
            token_field = "max_completion_tokens"
        if cap is not None:
            llm[token_field] = cap
        elif model == "deepseek-flash":
            llm[token_field] = 65536
        else:
            llm[token_field] = 8192


class LLM(Agent):
    name = "llm"
    description = "Generate text without versioned files or file-editing tools."

    def __init__(self, config=None):
        super().__init__(config)
        # Direct callers may omit the role, but this adapter cannot edit files.
        self.config.type = "llm"

    async def _run(self, prompt: str, working_dir: Path | None = None) -> Trajectory:
        if working_dir is not None:
            raise NotImplementedError(
                "LLM Worker generates text; the algorithm applies file edits"
            )
        from openai import AsyncOpenAI

        config = self.config
        llm = json.loads(json.dumps(config.llm, default=dict))
        model = llm.pop("model_name")
        api_key = llm.pop("api_key", None)
        base_url = llm.pop("base_url", None) or "https://api.openai.com/v1"
        request_timeout = llm.pop("request_timeout", config.get("request_timeout", 900))
        _completion_options(llm, model, base_url)
        secrets = sorted(
            set(_secrets(config) + _secrets(os.environ)), key=len, reverse=True
        )
        messages = [{"type": "RunInput", "prompt": prompt}]
        request_messages = []
        if config.get("_worker_system_prompt"):
            request_messages.append(
                {"role": "system", "content": config["_worker_system_prompt"]}
            )
        request_messages.append({"role": "user", "content": prompt})
        usage = {}
        try:
            if not api_key:
                raise ValueError("LLM Worker requires llm.api_key in config")
            async with AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=httpx.Timeout(request_timeout, connect=20),
                max_retries=config.get("max_retries", 1),
            ) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=request_messages,
                    **llm,
                )
            messages.append(
                {"type": "ChatCompletion", **response.model_dump(mode="json")}
            )
            usage = response.usage.model_dump(mode="json") if response.usage else {}
            output = response.choices[0].message.content if response.choices else None
            if not isinstance(output, str) or not output.strip():
                raise ValueError("LLM Worker returned no text")
            if response.choices[0].finish_reason != "stop":
                raise ValueError(
                    f"LLM Worker did not finish: {response.choices[0].finish_reason}"
                )
        except Exception as exc:
            if reason := infra_reason(exc):
                messages.append({"type": "InfraError", "reason": reason})
                exc = InfraError(reason)
            return Trajectory(
                status="error",
                messages=_redact(messages, secrets),
                usage=_redact(usage, secrets),
                error_message=_redact(str(exc), secrets),
            )
        return Trajectory(
            output=_redact(output, secrets),
            messages=_redact(messages, secrets),
            usage=_redact(usage, secrets),
        )


__all__ = ["LLM"]
