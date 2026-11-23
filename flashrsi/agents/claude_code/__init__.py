"""Run Claude Code using the selected versioned extensions.

The optional SDK is loaded on execution. Each session owns its configuration,
task state, delegated turns, and redacted trajectory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from flashrsi._records import _redact, _secrets
from flashrsi.agents.claude_code._mcp import evidence as _mcp_evidence
from flashrsi.core import Agent, InfraError, Trajectory
from flashrsi.core.infra import infra_reason

if TYPE_CHECKING:
    import claude_agent_sdk

_LOGGER = logging.getLogger(__name__)
_PROVIDER_ENV = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_GATEWAY",
)


class ClaudeCode(Agent):
    """Use a fresh SDK session for each prompt or file-editing task."""

    name = "claude_code"
    description = "Claude Code with versioned native extensions and delegated tasks."

    _container_dependencies = ("claude-agent-sdk",)
    instruction_file = "CLAUDE.md"

    def _check_container(self) -> None:
        super()._check_container()
        if self.config.get("cli_path"):
            raise ValueError("Container execution installs its own CLI; omit cli_path")
        if self.config["llm"].get("auth", "api_key") == "login":
            raise ValueError(
                "Claude Code containers cannot use llm.auth: login; "
                "configure llm.api_key, llm.auth_token, or llm.oauth_token"
            )
        _authentication_env(self.config["llm"])

    async def _run(self, prompt: str, working_dir: Path | None = None) -> Trajectory:
        from claude_agent_sdk import ClaudeSDKClient

        config = self.config
        try:
            env = _session_environment(config, self.files_dir)
        except ValueError as exc:
            return Trajectory(status="error", error_message=str(exc))
        max_turns = config.get("max_turns")
        if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
            raise ValueError("Claude Code max_turns must be a positive integer or None")
        secrets = sorted(
            set(_secrets(os.environ) + _secrets(env) + _secrets(config)),
            key=len,
            reverse=True,
        )
        run_input = {
            "type": "RunInput",
            "prompt": prompt,
            "snapshot_id": self.snapshot.id,
        }
        session = _Session(messages=[run_input], secrets=secrets)
        try:
            inputs = _session_inputs(self.files_dir, config)
        except (OSError, UnicodeError, ValueError) as exc:
            return Trajectory(
                status="error",
                messages=_redact(session.messages, secrets),
                error_message=_redact(str(exc), secrets),
            )

        with TemporaryDirectory(prefix="flashrsi-claude-code-") as temporary:
            runtime = Path(temporary).resolve()
            _prepare_runtime(config, env, runtime)
            task_dir = working_dir or runtime / "work"
            task_dir.mkdir(exist_ok=True)
            options = _session_options(
                config, self.files_dir, task_dir, inputs, env, session
            )
            run_input["system_prompt"] = options.system_prompt
            await session.run(ClaudeSDKClient(options=options), prompt, config)
        return session.trajectory()


@dataclass(slots=True)
class _Session:
    messages: list[dict[str, Any]]
    secrets: list[str]
    result: claude_agent_sdk.ResultMessage | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    stderr_tail: str = ""

    def stderr(self, line: str) -> None:
        redacted = _redact(line, self.secrets)
        _LOGGER.debug("%s", redacted)
        self.stderr_tail = (self.stderr_tail + redacted + "\n")[-8192:]

    async def run(
        self,
        client: claude_agent_sdk.ClaudeSDKClient,
        prompt: str,
        config: Mapping[str, Any],
    ) -> None:
        from claude_agent_sdk import ResultMessage, SystemMessage

        timeout = config.get("timeout", 900)
        try:
            try:
                grace = config.get("completion_grace", 0.5)
                async with asyncio.timeout(timeout):
                    await client.connect()
                    await client.query(prompt)
                    async with aclosing(
                        _messages(client, completion_grace=grace)
                    ) as stream:
                        async for message in stream:
                            self.messages.append(_record(message))
                            if isinstance(message, SystemMessage):
                                self.error = _extension_error(message)
                            if isinstance(message, ResultMessage):
                                self.result = message
                                _add_usage(self.usage, message)
                                self.error = _result_error(message)
                            if self.error:
                                break
                if self.result is not None and not self.error:
                    self.messages.append(await _mcp_evidence(client, self.messages))
            finally:
                # Cleanup has its own deadline and must outlive caller cancellation.
                await _disconnect(client)
        except TimeoutError:
            self.error = f"Claude Code timed out after {timeout:g}s"
        except Exception as exc:
            reason = infra_reason(exc)
            self.error = (
                str(InfraError(reason)) if reason else str(exc) or type(exc).__name__
            )

    def trajectory(self) -> Trajectory:
        error = self.error
        if self.result is None:
            error = error or "Claude Code ended without a result"
        reason = _infra_reason(error) or (
            _infra_reason(self.stderr_tail) if error else None
        )
        if reason:
            self.messages.append({"type": "InfraError", "reason": reason})
            error = str(InfraError(reason))
        elif error and self.stderr_tail:
            error += "\nClaude Code stderr (tail):\n" + self.stderr_tail.rstrip()
        output = ""
        if self.result is not None and not error:
            output = self.result.result or ""
        return Trajectory(
            status="error" if error else "finished",
            output=_redact(output, self.secrets),
            messages=_redact(self.messages, self.secrets),
            usage=_redact(self.usage, self.secrets),
            error_message=_redact(error, self.secrets),
        )


def _session_environment(config: Mapping[str, Any], files: Path) -> dict[str, str]:
    llm = config["llm"]
    env = _authentication_env(llm, request_timeout=config.get("request_timeout", 900))
    env.update(
        CLAUDE_PLUGIN_ROOT=str(files),
        CLAUDE_CODE_DISABLE_AUTO_MEMORY="1",
        CLAUDE_CODE_SKIP_PROMPT_HISTORY="",
    )
    if config.get("_suite_model_relay", False):
        env["IS_SANDBOX"] = "1"
        for family in ("OPUS", "SONNET", "HAIKU"):
            env[f"ANTHROPIC_DEFAULT_{family}_MODEL"] = llm["model_name"]
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = llm["model_name"]
    return env


def _session_inputs(
    files: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], str | None]:
    try:
        instructions = (files / "CLAUDE.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        instructions = ""
    if config.get("_worker_system_prompt"):
        instructions += "\n\n" + config["_worker_system_prompt"]
    system_prompt = {"type": "preset", "preset": "claude_code"}
    if instructions.strip():
        system_prompt["append"] = instructions
    return system_prompt, _settings(files / "settings.json")


def _prepare_runtime(
    config: Mapping[str, Any], env: dict[str, str], runtime: Path
) -> None:
    state = str(runtime / "state")
    env["CLAUDE_CONFIG_DIR"] = state
    # Only explicit login mode may consult the user's native credential store.
    env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = state
    if config["llm"].get("auth", "api_key") == "login":
        env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = os.environ.get(
            "CLAUDE_SECURESTORAGE_CONFIG_DIR",
            os.environ.get("CLAUDE_CONFIG_DIR", ""),
        )
    temp = runtime / "tmp"
    temp.mkdir(mode=0o700)
    env["CLAUDE_CODE_TMPDIR"] = str(temp)
    if config.get("_suite_model_relay", False):
        isolated_home = runtime / "home"
        isolated_home.mkdir(mode=0o700)
        env["HOME"] = str(isolated_home)


def _session_options(
    config: Mapping[str, Any],
    files: Path,
    task_dir: Path,
    inputs: tuple[dict[str, Any], str | None],
    env: dict[str, str],
    session: _Session,
) -> claude_agent_sdk.ClaudeAgentOptions:
    from claude_agent_sdk import ClaudeAgentOptions

    llm = config["llm"]
    permission = (
        "bypassPermissions" if config.get("_suite_model_relay") else "acceptEdits"
    )
    mcp = files / ".mcp.json"
    cli_path = config.get("cli_path")
    system_prompt, settings = inputs
    # Task-local settings cannot replace the selected harness's extensions.
    return ClaudeAgentOptions(
        model=llm["model_name"],
        system_prompt=system_prompt,
        plugins=[{"type": "local", "path": str(files)}],
        skills="all",
        setting_sources=[],
        strict_mcp_config=True,
        mcp_servers=str(mcp) if mcp.is_file() else {},
        settings=settings,
        forward_subagent_text=True,
        include_hook_events=True,
        cwd=task_dir,
        env=env,
        permission_mode=config.get("permission_mode", permission),
        allowed_tools=config.get("allowed_tools", ["Read", "Glob", "Grep"]),
        disallowed_tools=config.get("disallowed_tools", []),
        tools=config.get("tools"),
        max_turns=config.get("max_turns"),
        max_budget_usd=config.get("max_budget_usd"),
        # Tool output and long reasoning can exceed the SDK's 1 MiB default.
        max_buffer_size=config.get("max_buffer_size", 32 * 1024 * 1024),
        effort=_effort(llm),
        thinking=_thinking(llm),
        cli_path=Path(cli_path).expanduser().resolve() if cli_path else None,
        stderr=session.stderr,
    )


def _thinking(llm: Mapping[str, Any]) -> dict[str, Any] | None:
    value = llm.get("thinking")
    if value is None:
        value = (llm.get("extra_body") or {}).get("thinking")
    if isinstance(value, Mapping):
        if value.get("type") == "enabled" and "budget_tokens" not in value:
            return {"type": "adaptive"}
        return dict(value)
    if value is False or value in ("off", "none", "disabled"):
        return {"type": "disabled"}
    if value is True or value in ("enabled", "adaptive"):
        return {"type": "adaptive"}
    if (llm.get("reasoning_effort") or llm.get("effort")) in ("off", "none"):
        return {"type": "disabled"}
    return None


def _effort(llm: Mapping[str, Any]) -> str | None:
    value = llm.get("reasoning_effort") or llm.get("effort")
    aliases = {"off": None, "none": None, "minimal": "low", "ultra": "max"}
    return aliases.get(value, value)


def _authentication_env(
    llm: Mapping[str, Any], *, request_timeout: float = 900
) -> dict[str, str]:
    """Map explicit connection settings to the native CLI's environment."""
    auth = llm.get("auth", "api_key")
    if auth not in ("api_key", "login"):
        raise ValueError("Claude Code llm.auth must be 'api_key' or 'login'")
    credentials = {
        "api_key": "ANTHROPIC_API_KEY",
        "auth_token": "ANTHROPIC_AUTH_TOKEN",
        "oauth_token": "CLAUDE_CODE_OAUTH_TOKEN",
    }
    provided = [name for name in credentials if llm.get(name)]
    if auth == "login" and provided:
        raise ValueError("Claude Code llm.auth: login cannot include API credentials")
    if auth == "api_key" and len(provided) != 1:
        raise ValueError(
            "Claude Code requires exactly one of llm.api_key, llm.auth_token, "
            "or llm.oauth_token; use llm.auth: login for native login"
        )
    # Empty values suppress inherited credentials and cloud-provider routing.
    env = dict.fromkeys((*credentials.values(), *_PROVIDER_ENV), "")
    base_url = llm.get("base_url") or "https://api.anthropic.com"
    if urlsplit(base_url).hostname == "api.deepseek.com":
        base_url = "https://api.deepseek.com/anthropic"
    env["ANTHROPIC_BASE_URL"] = base_url
    request_timeout = llm.get("request_timeout", request_timeout)
    if (
        type(request_timeout) not in (int, float)
        or not math.isfinite(request_timeout)
        or request_timeout < 0.001
    ):
        raise ValueError("Claude Code request_timeout must be at least 0.001 seconds")
    request_timeout_ms = int(request_timeout * 1000)
    env.update(
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        CLAUDE_CODE_GZIP_REQUEST_BODIES="0",
        CLAUDE_CODE_ALWAYS_ENABLE_EFFORT="1",
    )
    output_tokens = 65536 if llm.get("model_name") == "deepseek-flash" else None
    idle_variable = "CLAUDE_STREAM_IDLE_TIMEOUT_MS"
    for setting, variable, default, minimum in (
        ("api_timeout_ms", "API_TIMEOUT_MS", request_timeout_ms, 1),
        ("stream_idle_timeout_ms", idle_variable, request_timeout_ms, 1),
        ("max_tokens", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", output_tokens, 1),
        ("max_retries", "CLAUDE_CODE_MAX_RETRIES", None, 0),
    ):
        value = llm.get(setting, default)
        if value is not None:
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"Claude Code llm.{setting} must be an integer >= {minimum}"
                )
            env[variable] = str(value)
    for name in provided:
        if not isinstance(llm[name], str) or not llm[name].strip():
            raise ValueError(f"Claude Code llm.{name} must be a non-empty string")
        env[credentials[name]] = llm[name]
    return env


def _settings(path: Path) -> str | None:
    """Validate native settings and reserve runtime credentials and state."""
    if not path.is_file():
        return None
    try:
        settings = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise ValueError(f"Invalid {path.name}: {exc}") from None
    if not isinstance(settings, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    environment = settings.get("env", {})
    if not isinstance(environment, dict) or any(
        not isinstance(value, str) for value in environment.values()
    ):
        raise ValueError(f"{path.name}.env must map names to strings")
    reserved = {
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_SECURESTORAGE_CONFIG_DIR",
        "CLAUDE_PLUGIN_ROOT",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
        "CLAUDE_CODE_TMPDIR",
        *_PROVIDER_ENV,
    }
    for name in environment:
        if name.upper() in reserved or name.upper().endswith(
            ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            raise ValueError(
                f"{path.name}.env cannot set runtime field {name}; "
                "use Agent runtime config"
            )
    return str(path)


def _infra_reason(error: str | None) -> str | None:
    # The CLI and SDK report transport failures as text on some versions.
    marker = "InfraError: "
    if isinstance(error, str) and error.startswith(marker):
        return error.removeprefix(marker)
    return infra_reason(error)


def _extension_error(message: claude_agent_sdk.SystemMessage) -> str | None:
    if message.subtype != "init":
        return None
    failures = [
        json.dumps(item, ensure_ascii=False)
        for item in message.data.get("plugin_errors", ())
    ]
    failures.extend(
        f"MCP server {server['name']}: {server['status']}"
        for server in message.data.get("mcp_servers", ())
        if server["status"] in {"failed", "needs-auth"}
    )
    if failures:
        return "Claude Code extensions failed to load: " + "; ".join(failures)
    return None


def _result_error(result: claude_agent_sdk.ResultMessage) -> str | None:
    terminal = result.terminal_reason
    if result.deferred_tool_use is not None or terminal == "tool_deferred":
        return "Claude Code deferred a tool; this run cannot resume deferred work"
    if terminal == "aborted_streaming":
        return str(InfraError("incomplete_stream"))
    if terminal == "aborted_tools":
        return "Claude Code ended with aborted_tools"
    if result.is_error or result.subtype != "success":
        return (
            "; ".join(result.errors or [])
            or result.result
            or f"Claude Code ended with {result.subtype}"
        )
    return None


@dataclass(slots=True)
class _DelegatedTurns:
    active: set[str] = field(default_factory=set)
    delegated: set[str] = field(default_factory=set)
    background: set[str] = field(default_factory=set)
    notifications: set[str] = field(default_factory=set)
    handling: set[str] = field(default_factory=set)
    in_turn: bool = True
    result: claude_agent_sdk.ResultMessage | None = None

    @property
    def settled(self) -> bool:
        return self.result is not None and not (
            self.active or self.notifications or self.in_turn
        )

    def record(self, message: claude_agent_sdk.Message) -> bool:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TaskNotificationMessage,
            TaskStartedMessage,
            TaskUpdatedMessage,
        )

        if isinstance(message, TaskStartedMessage):
            if message.task_type in {"local_agent", "local_workflow"}:
                self.delegated.add(message.task_id)
                self.active.add(message.task_id)
                if message.data.get("is_backgrounded") is not False:
                    self.background.add(message.task_id)
        elif isinstance(message, TaskUpdatedMessage):
            if message.status in {"completed", "failed", "stopped", "killed"}:
                self.active.discard(message.task_id)
        elif isinstance(message, TaskNotificationMessage):
            self.active.discard(message.task_id)
            if message.task_id in self.background and message.status in {
                "completed",
                "failed",
            }:
                self.notifications.add(message.task_id)
        elif (
            isinstance(message, SystemMessage)
            and message.subtype == "background_tasks_changed"
        ):
            self.background.update(
                task["task_id"]
                for task in message.data.get("tasks", ())
                if task["task_id"] in self.delegated
            )
        elif (
            isinstance(message, SystemMessage)
            and message.subtype == "init"
            or isinstance(message, AssistantMessage)
            and message.parent_tool_use_id is None
            and not self.in_turn
        ):
            # New notifications arriving during this turn belong to a later one.
            self.in_turn = True
            self.handling = self.notifications.copy()
        elif isinstance(message, ResultMessage):
            self.result = message
            self.in_turn = False
            self.notifications.difference_update(self.handling)
            self.handling.clear()
            return message.is_error or message.subtype != "success"
        return False


async def _messages(
    client: claude_agent_sdk.ClaudeSDKClient,
    *,
    completion_grace: float,
) -> AsyncIterator[claude_agent_sdk.Message]:
    """Drain finite delegated work and its parent continuation before closing."""
    from claude_agent_sdk import ClaudeSDKError

    turns = _DelegatedTurns()
    async with aclosing(client.receive_messages()) as stream:
        while True:
            settled = turns.settled
            if settled and not turns.delegated:
                return
            try:
                if settled:
                    try:
                        async with asyncio.timeout(completion_grace):
                            message = await anext(stream)
                    except TimeoutError:
                        return
                else:
                    message = await anext(stream)
            except StopAsyncIteration:
                if not settled:
                    raise ClaudeSDKError(
                        "Claude Code stream ended before delegated work and its parent response completed"
                    ) from None
                return
            failed = turns.record(message)
            yield message
            if failed:
                return


def _add_usage(usage: dict[str, Any], result: claude_agent_sdk.ResultMessage) -> None:
    """Sum turn counters; native model totals and cost are session cumulative."""
    _sum_counters(usage, result.usage or {})
    usage["num_turns"] = usage.get("num_turns", 0) + result.num_turns
    usage["duration_ms"] = usage.get("duration_ms", 0) + result.duration_ms
    usage.update(
        model_usage=result.model_usage,
        total_cost_usd=result.total_cost_usd,
        duration_api_ms=result.duration_api_ms,
        session_id=result.session_id,
    )


def _sum_counters(total: dict[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, dict):
            _sum_counters(total.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


async def _disconnect(client: claude_agent_sdk.ClaudeSDKClient) -> None:
    cleanup = asyncio.create_task(client.disconnect())
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError


def _record(value: Any) -> Any:
    """Keep SDK message and content-block types along with their native fields."""
    if is_dataclass(value):
        record = {"type": type(value).__name__}
        for attribute in fields(value):
            record[attribute.name] = _record(getattr(value, attribute.name))
        return record
    if isinstance(value, dict):
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    return value
