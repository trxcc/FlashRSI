"""Retryable execution failures that must never be interpreted as task scores."""

from __future__ import annotations

import asyncio
import os
import random
import re

_STATUS_CODES = {402, 429, 500, 502, 503, 504}
_STATUS_TEXT = re.compile(
    r"\b(?:HTTP(?:/\d(?:\.\d)?)?(?: error)?|status(?: code)?|(?:API )?error(?: code)?)"
    r"\s*[:=]?\s*(402|429|500|502|503|504)\b", re.I
)
_MESSAGES = {
    "rate_limit": ("ratelimiterror", "rate limit reached", "rate limit exceeded"),
    "incomplete_stream": (
        "incomplete stream", "stream ended without", "stream ended before", "missing terminal stop",
        "inference has not started", "waiting for generation to start timed out",
        "chat completions returned no choices", "response ended prematurely",
    ),
    "connection": (
        "connection reset", "connection refused", "connection dropped", "econnreset",
        "server disconnected", "remoteprotocolerror", "apiconnectionerror",
        "apitimeouterror", "remote end closed connection",
    ),
    "docker_daemon": (
        "cannot connect to the docker daemon", "error during connect:",
        "docker daemon is not running", "docker daemon unavailable",
        "docker daemon disconnected", "context deadline exceeded",
    ),
}


class InfraError(RuntimeError):
    """A transient failure before an authoritative task result was obtained."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"InfraError: {reason}")


def infra_reason(failure: BaseException | str | None) -> str | None:
    if isinstance(failure, InfraError):
        return failure.reason
    status = getattr(failure, "status_code", None) or getattr(failure, "status", None)
    if not isinstance(status, int):
        status = getattr(getattr(failure, "response", None), "status_code", None)
    if isinstance(status, int) and status in _STATUS_CODES:
        return f"http_{status}"
    if isinstance(failure, ConnectionError) or any(
        cls.__name__ == "TransportError" and cls.__module__.split(".")[0] in {"httpx", "httpx2"}
        for cls in type(failure).__mro__
    ):
        return "connection"
    if type(failure).__name__ in {"APIConnectionError", "APITimeoutError"}:
        return "connection"
    message = str(failure or "").lower()
    marker = re.search(r"\binfraerror: (http_\d{3}|[a-z_]+)", message)
    if marker:
        return marker[1]
    match = _STATUS_TEXT.search(message) or re.match(r"(402|429|500|502|503|504):", message)
    if match:
        return f"http_{match[1]}"
    for reason, fragments in _MESSAGES.items():
        if any(fragment in message for fragment in fragments):
            return reason
    return None


def trajectory_infra_reason(trajectory) -> str | None:
    if trajectory.status != "error":
        return None
    for event in trajectory.messages:
        if event.get("type") == "InfraError":
            return str(event.get("reason") or "infrastructure")
        if event.get("type") == "HarborVerifier" and event.get("infra_error"):
            return str(event["infra_error"])
    return infra_reason(trajectory.error_message)


def env_integer(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        value = minimum - 1
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


async def retry(operation, retries: int):
    for attempt in range(retries + 1):
        try:
            result = await operation()
            reason = trajectory_infra_reason(result)
        except Exception as exc:
            reason = infra_reason(exc)
            if reason is None:
                raise
        else:
            if reason is None:
                return result
        if reason == "http_402" or attempt == retries:
            raise InfraError(reason) from None
        await asyncio.sleep(random.uniform(0.5, 1.0) * (2 ** min(attempt, 5)))
