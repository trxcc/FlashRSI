"""HTTP Responses protocol port of Codex rust-v0.147.0 (be6e8eac).

The request is built from core/src/client.rs::build_responses_request rather
than OpenAI SDK defaults. Stream interpretation follows codex-api/src/sse/
responses.rs; request retry and stream retry are distinct layers upstream.
WebSockets and authenticated-account recovery are not ported.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
from uuid import uuid4

import httpx

from .history import canonical_item
from .models import Model, validate_model_settings
from .response_request import apply_output_options
from .response_request import lite_tools as _lite_tools
from .response_stream import (
    remember_reasoning,
    with_reasoning_content,
)
from .response_stream import (
    sse_events as _sse_events,
)
from .response_stream import (
    usage as _usage,
)


class ResponseError(RuntimeError):
    def __init__(
        self,
        message,
        *,
        code="stream",
        retryable=True,
        delay=None,
        partial_items=None,
        usage=None,
        status=None,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.delay = delay
        self.partial_items = partial_items or []
        self.usage = usage or {}
        self.status = status

    @property
    def infra_reason(self):
        if self.status in {402, 429, 500, 502, 503, 504}:
            return f"http_{self.status}"
        if self.code == "rate_limit_exceeded":
            return "rate_limit"
        if self.code in {"server_is_overloaded", "slow_down"}:
            return "http_503"
        if self.code in {
            "stream",
            "server_error",
            "api_error",
            "internal_server_error",
        }:
            return "incomplete_stream"
        return None


class ResponsesClient:
    def __init__(self, llm: dict, model: Model, settings: dict, thread_id: str, emit):
        validate_model_settings(settings)
        validate_model_settings({"model_reasoning_effort": llm.get("reasoning_effort")})
        self.llm, self.model, self.settings = (
            copy.deepcopy(llm),
            model,
            copy.deepcopy(settings),
        )
        self.thread_id, self.emit = thread_id, emit
        self.provider = settings.get("model_providers", {}).get(
            settings.get("model_provider"), {}
        )
        self.base_url = (
            llm.get("base_url")
            or self.provider.get("base_url")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.url = self.base_url + "/responses"
        self.deepseek = bool(llm.get("_deepseek_responses"))
        request_retries = llm.get("max_retries")
        if request_retries is None:
            request_retries = self.provider.get("request_max_retries", 4)
        self.request_retries = min(request_retries, 100)
        self.stream_retries = min(self.provider.get("stream_max_retries", 5), 100)
        self.idle_timeout = llm.get("request_timeout")
        if self.idle_timeout is None:
            self.idle_timeout = (
                llm.get("stream_idle_timeout_ms")
                or self.provider.get("stream_idle_timeout_ms", 900000)
            ) / 1000
        self.turn_state = None
        self._http = None
        self._installation_id = str(uuid4())

    async def __aenter__(self):
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self.idle_timeout, connect=20), follow_redirects=True
        )
        return self

    async def __aexit__(self, *exc):
        await self._http.aclose()
        self._http = None

    def request(self, items: list, tools: list, client_metadata: dict | None = None):
        """core::ModelClientSession::{build_reasoning,build_responses_request}."""
        info, settings = self.model.info, self.settings
        lite = info["use_responses_lite"]
        reasoning = self._reasoning(lite)
        input_items = self._request_items(items, lite)
        payload = {
            "model": info["slug"],
            "input": input_items,
            "tools": copy.deepcopy(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": info["supports_parallel_tool_calls"] and not lite,
            "reasoning": reasoning,
            "store": self.provider.get("name", "").lower() == "azure"
            or any(
                marker in self.base_url.lower()
                for marker in (
                    "openai.azure.",
                    "cognitiveservices.azure.",
                    "aoai.azure.",
                    "azure-api.",
                    "azurefd.",
                    "windows.net/openai",
                )
            ),
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": settings.get("_session_id", self.thread_id),
            "client_metadata": client_metadata
            if client_metadata is not None
            else {
                "x-codex-installation-id": self._installation_id,
                "session_id": self.thread_id,
                "thread_id": self.thread_id,
                "x-codex-window-id": self.thread_id + ":0",
            },
        }
        self._apply_request_options(payload, input_items, tools, lite)
        return payload

    def _reasoning(self, lite: bool) -> dict:
        info, settings = self.model.info, self.settings
        reasoning = {}
        effort = self.llm.get("reasoning_effort")
        if effort is None:
            effort = settings.get("model_reasoning_effort")
        if effort is None:
            effort = info.get("default_reasoning_level")
        if effort is not None:
            reasoning["effort"] = "max" if effort == "ultra" else effort
        summary = settings.get("model_reasoning_summary")
        if summary is None:
            summary = info["default_reasoning_summary"]
        if info["supports_reasoning_summary_parameter"] and summary != "none":
            reasoning["summary"] = summary
        if lite:
            reasoning["context"] = "all_turns"
        return reasoning

    def _request_items(self, items: list, lite: bool) -> list:
        # Non-OpenAI providers must not receive internal metadata or encrypted
        # function args; this is separate from reasoning.encrypted_content.
        input_items = copy.deepcopy(items)
        for item in input_items:
            identity = item.get("id")
            if identity is not None and not re.fullmatch(
                r"[^_]+_.+", identity, re.DOTALL
            ):
                item.pop("id", None)
            if not self.is_openai:
                item.pop("internal_chat_message_metadata_passthrough", None)
                if item.get("type") == "function_call":
                    item.pop("encrypted_function_args", None)
            if lite:
                for field in ("content", "output"):
                    if isinstance(item.get(field), list):
                        for part in item[field]:
                            if part.get("type") == "input_image":
                                part.pop("detail", None)
        return input_items

    def _apply_request_options(
        self, payload: dict, input_items: list, tools: list, lite: bool
    ) -> None:
        info, settings = self.model.info, self.settings
        reasoning = payload["reasoning"]
        output_tokens = self.llm.get("max_output_tokens") or self.llm.get("max_tokens")
        if output_tokens is not None:
            payload["max_output_tokens"] = output_tokens
        if self.deepseek:
            thinking = self.llm.get("thinking")
            if thinking is None:
                thinking = (self.llm.get("extra_body") or {}).get("thinking")
            mode = thinking.get("type") if isinstance(thinking, dict) else thinking
            if mode is False or mode in ("disabled", "off", "none"):
                payload["reasoning"] = {"effort": "none"}
            else:
                payload["reasoning"] = {"effort": reasoning.get("effort", "high")}
            if output_tokens is None and info["slug"] == "deepseek-flash":
                payload["max_output_tokens"] = 65536
            for item in input_items:
                if item["type"] == "reasoning":
                    item.pop("summary", None)
                    item.pop("encrypted_content", None)
        if self.model.instructions:
            payload["instructions"] = self.model.instructions
        if lite:
            prefix = [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": _lite_tools(tools)
                    if settings.get("_provider_capabilities", {}).get(
                        "namespace_tools", True
                    )
                    else copy.deepcopy(tools),
                }
            ]
            if self.model.instructions:
                prefix.append(
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {"type": "input_text", "text": self.model.instructions}
                        ],
                    }
                )
            payload["input"] = prefix + input_items
            payload.pop("instructions", None)
            payload.pop("tools", None)
        apply_output_options(
            payload, settings, info, reasoning, is_openai=self.is_openai
        )

    @property
    def is_openai(self):
        # ModelProviderInfo::is_openai compares the provider's display name.
        return self.provider.get("name") == "OpenAI" or (
            self.settings.get("model_provider") == "openai" and not self.provider
        )

    def _headers(self, metadata):
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "originator": "codex_exec",
            "session-id": metadata.get("session_id", self.thread_id),
            "thread-id": self.thread_id,
            "x-client-request-id": self.thread_id,
        }
        headers.update(self.provider.get("http_headers", {}))
        headers.update(self.llm.get("extra_headers", {}))
        if self.llm.get("api_key"):
            headers["authorization"] = "Bearer " + self.llm["api_key"]
        for key in (
            "x-codex-window-id",
            "x-codex-turn-metadata",
            "x-codex-parent-thread-id",
            "x-openai-subagent",
        ):
            if key in metadata:
                headers[key] = metadata[key]
        if self.turn_state is not None:
            headers["x-codex-turn-state"] = self.turn_state
        if self.model.info["use_responses_lite"]:
            headers["x-openai-internal-codex-responses-lite"] = "true"
        return headers

    async def compact_response(self, items, tools, metadata=None):
        """codex-api/endpoint/compact.rs: unary remote compaction, HTTP retries only."""
        if not items:
            return []
        if self._http is None:
            raise RuntimeError("ResponsesClient must be used with async with")
        full = self.request(items, tools, metadata)
        fields = {
            "model",
            "input",
            "instructions",
            "tools",
            "parallel_tool_calls",
            "reasoning",
            "service_tier",
            "prompt_cache_key",
            "text",
        }
        payload = {key: value for key, value in full.items() if key in fields}
        # API-key authentication does not send the account-only tier hint.
        payload.pop("service_tier", None)
        headers = self._headers(full["client_metadata"])
        headers["accept"] = "application/json"
        for attempt in range(self.request_retries + 1):
            try:
                response = await self._http.post(
                    self.base_url + "/responses/compact",
                    json=payload,
                    headers=headers,
                    params=self.provider.get("query_params"),
                    timeout=self.idle_timeout * 4,
                )
                if not 200 <= response.status_code < 300:
                    if response.status_code >= 500 and attempt < self.request_retries:
                        await asyncio.sleep(_backoff(attempt + 1))
                        continue
                    raise _http_error(response.status_code, response.text)
                self.turn_state = self.turn_state or response.headers.get(
                    "x-codex-turn-state"
                )
                try:
                    value = response.json()
                    output = value["output"]
                    if not isinstance(output, list):
                        raise ValueError("output must be an array")
                    parsed = []
                    for item in output:
                        if not isinstance(item, dict) or not isinstance(
                            item.get("type"), str
                        ):
                            raise ValueError("invalid response item")
                        projected = canonical_item(item, strict=True)
                        if projected is not None:
                            parsed.append(projected)
                        # ResponseItem::Other accepts unknown variants; the
                        # history manager subsequently excludes them.
                    return parsed
                except (KeyError, TypeError, ValueError) as error:
                    raise ResponseError(
                        "failed to parse CompactHistoryResponse: " + str(error),
                        retryable=False,
                    ) from error
            except httpx.TransportError as error:
                if attempt >= self.request_retries:
                    raise ResponseError(str(error)) from error
                await asyncio.sleep(_backoff(attempt + 1))

    async def response(
        self,
        items: list,
        tools: list,
        client_metadata: dict | None = None,
        *,
        retry_stream: bool = True,
    ) -> dict:
        if self._http is None:
            raise RuntimeError("ResponsesClient must be used with async with")
        payload = self.request(items, tools, client_metadata)
        headers = self._headers(payload["client_metadata"])
        # core/src/responses_retry.rs. A completed output item has already been
        # acted on upstream; callers must resolve a partial stream explicitly
        # instead of replaying tool effects from an unchanged request history.
        stream_retries = self.stream_retries if retry_stream else 0
        for stream_attempt in range(stream_retries + 1):
            if self.turn_state is not None:
                headers["x-codex-turn-state"] = self.turn_state
            try:
                return await self._request_with_retry(payload, headers)
            except ResponseError as error:
                if (
                    error.partial_items
                    or not error.retryable
                    or stream_attempt >= stream_retries
                ):
                    raise
                attempt = stream_attempt + 1
                self.emit(
                    {
                        "type": "error",
                        "message": f"Reconnecting... {attempt}/{self.stream_retries}",
                        "will_retry": True,
                    }
                )
                await asyncio.sleep(
                    error.delay if error.delay is not None else _backoff(attempt)
                )

    async def _request_with_retry(self, payload, headers):
        # DeepSeek rate limits are transient. Keep other providers' existing
        # request retry policy.
        for attempt in range(self.request_retries + 1):
            try:
                async with self._http.stream(
                    "POST",
                    self.url,
                    json=payload,
                    headers=headers,
                    params=self.provider.get("query_params"),
                ) as response:
                    if not 200 <= response.status_code < 300:
                        body = (await response.aread()).decode(errors="replace")
                        if (
                            response.status_code >= 500
                            or (self.deepseek and response.status_code == 429)
                        ) and attempt < self.request_retries:
                            await asyncio.sleep(_backoff(attempt + 1))
                            continue
                        error = _http_error(response.status_code, body)
                        if self.deepseek and error.infra_reason:
                            # The request layer has used its retry budget.
                            error.retryable = False
                        raise error
                    self.turn_state = self.turn_state or response.headers.get(
                        "x-codex-turn-state"
                    )
                    if "x-reasoning-included" in response.headers:
                        self.emit({"type": "server_reasoning_included"})
                    return await self._reduce(response)
            except httpx.TransportError as error:
                if attempt >= self.request_retries:
                    raise ResponseError(str(error)) from error
                await asyncio.sleep(_backoff(attempt + 1))

    async def _reduce(self, response):
        items, pending_error = [], None
        reasoning_text = {}
        stream = _sse_events(response).__aiter__()
        while True:
            try:
                data = await asyncio.wait_for(anext(stream), self.idle_timeout)
            except StopAsyncIteration:
                error = pending_error or ResponseError(
                    "stream closed before response.completed"
                )
                error.partial_items = items
                raise error
            except TimeoutError as exc:
                raise ResponseError(
                    "idle timeout waiting for SSE", partial_items=items
                ) from exc
            except httpx.TransportError as exc:
                raise ResponseError(str(exc), partial_items=items) from exc
            try:
                event = json.loads(data)
            except (ValueError, UnicodeError):
                # The upstream decoder logs malformed/unknown events and keeps
                # consuming; [DONE] is not Responses completion.
                continue
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                continue
            kind = event["type"]
            if self.deepseek:
                remember_reasoning(event, reasoning_text)
            if kind == "response.output_item.done":
                item = event.get("item")
                if self.deepseek:
                    item = with_reasoning_content(
                        item, reasoning_text.get(event.get("output_index"))
                    )
                item = canonical_item(item)
                if item is None:
                    continue
                event["item"] = item
                items.append(item)
            self.emit(event)
            if kind == "response.output_item.done":
                continue
            elif kind == "response.failed":
                pending_error = _stream_error(
                    (event.get("response") or {}).get("error") or {}
                )
            elif kind == "response.incomplete":
                reason = (
                    (event.get("response") or {}).get("incomplete_details") or {}
                ).get("reason", "unknown")
                pending_error = ResponseError(
                    "Incomplete response returned, reason: " + str(reason),
                    code=str(reason) if self.deepseek else "stream",
                    retryable=not self.deepseek,
                )
            elif kind == "response.completed" and event.get("response") is not None:
                completed = event["response"]
                try:
                    if not isinstance(completed["id"], str):
                        raise ValueError("response id must be a string")
                    if (
                        completed.get("end_turn") is not None
                        and type(completed["end_turn"]) is not bool
                    ):
                        raise ValueError("end_turn must be a boolean")
                    usage = _usage(completed.get("usage"))
                except (KeyError, ValueError, TypeError) as exc:
                    pending_error = ResponseError(
                        "failed to parse ResponseCompleted: " + str(exc)
                    )
                    continue
                return {
                    "items": items,
                    "usage": usage,
                    "response_id": completed["id"],
                    "end_turn": completed.get("end_turn"),
                }


def _stream_error(error):
    code = error.get("code") or "stream"
    message = error.get("message") or "response.failed event received"
    fatal = {
        "context_length_exceeded",
        "insufficient_quota",
        "usage_not_included",
        "cyber_policy",
        "invalid_prompt",
        "bio_policy",
        "server_is_overloaded",
        "slow_down",
    }
    if code == "cyber_policy" and not str(error.get("message") or "").strip():
        message = "This request has been flagged for possible cybersecurity risk."
    delay = None
    if code == "rate_limit_exceeded":
        match = re.search(
            r"try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)", message, re.IGNORECASE
        )
        if match:
            delay = float(match[1]) / (1000 if match[2].lower() == "ms" else 1)
    return ResponseError(message, code=code, retryable=code not in fatal, delay=delay)


def _http_error(status, body):
    try:
        error = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    code = error.get("code", "") if isinstance(error, dict) else ""
    fatal = 400 <= status < 500 or (
        status == 503 and code in {"server_is_overloaded", "slow_down"}
    )
    return ResponseError(
        f"HTTP {status}: {body}",
        code=code or f"http_{status}",
        retryable=not fatal,
        status=status,
    )


def _backoff(attempt):
    # Both upstream HTTP and core retry loops use 200ms * 2^(n-1), ±10%.
    return int(200 * 2 ** max(0, attempt - 1) * random.uniform(0.9, 1.1)) / 1000
