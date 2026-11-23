"""Verify offline request, token, and container timeout contracts."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from email.message import Message
from functools import partial
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import flashrsi
from flashrsi.agents.claude_code import _authentication_env
from flashrsi.agents.miniagent.runtime import llm as mini
from flashrsi.agents.openclaw.runtime.openclaw_native import providers as claw_provider
from flashrsi.agents.openclaw.runtime.openclaw_native.policy import ToolPolicy
from flashrsi.agents.openclaw.runtime.openclaw_native.tools import ToolRunner
from flashrsi.agents.pi_agent import Pi
from flashrsi.agents.pi_agent.runtime.pi_native.anthropic_messages import (
    build_anthropic_payload,
)
from flashrsi.agents.pi_agent.runtime.pi_native.compaction import model_limits, settings
from flashrsi.agents.pi_agent.runtime.pi_native.openai_provider import build_payload
from flashrsi.agents.pi_agent.runtime.pi_native.tools import execute_tool
from flashrsi.integrations import _model_proxy, sandbox_runtime

CHECKOUT = Path(__file__).resolve().parents[1]
assert Path(flashrsi.__file__).resolve().parent == CHECKOUT / "flashrsi"


def transport(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )


def sse(events):
    return ": keep-alive\n\n" + "".join(
        f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n"
        for event in events
    )


def assert_timeout(request, expected=900):
    assert request.extensions["timeout"] == {
        "connect": 20,
        "read": expected,
        "write": expected,
        "pool": expected,
    }


@pytest.mark.parametrize("cap", [None, 65536, 128000])
def test_pi_decoding_and_compaction(cap, tmp_path):
    llm = {"model_name": "deepseek-flash", "provider_name": "deepseek"}
    if cap is not None:
        llm["max_tokens"] = cap
    config = {"llm": llm}
    messages = [{"role": "user", "content": "test"}]
    chat, _ = build_payload(config, messages, "openai-completions", "high")
    anthropic, _, _ = build_anthropic_payload(config, messages, "high")
    assert chat["max_tokens"] == anthropic["max_tokens"] == (cap or 65536)
    assert chat["thinking"] == anthropic["thinking"] == {"type": "enabled"}
    assert anthropic["output_config"] == {"effort": "high"}
    assert model_limits(config) == (1000000, 65536)
    assert settings(config, tmp_path) == {
        "enabled": True,
        "reserveTokens": 16384,
        "keepRecentTokens": 20000,
    }


def test_pi_stock_remaining_context_clamp():
    messages = [{"role": "user", "content": "a" * 7000}]
    config = {
        "llm": {"model_name": "deepseek/deepseek-flash", "max_tokens": 128000},
        "context_window": 10000,
    }
    payload, _ = build_payload(config, messages, "openai-completions", "high")
    assert payload["max_tokens"] == 10000 - 2000 - 4096
    exhausted, _ = build_payload(
        {**config, "context_window": 2000}, messages, "openai-completions", "high"
    )
    assert exhausted["max_tokens"] == 1


def _pi_provider_reply(request_timeout, request):
    assert_timeout(request, request_timeout or 900)
    assert json.loads(request.content)["max_tokens"] == 65536
    return httpx.Response(
        200,
        text=sse(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "done"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                "[DONE]",
            ]
        ),
        headers={"Content-Type": "text/event-stream"},
    )


@pytest.mark.parametrize("request_timeout", [None, 1200])
def test_pi_native_adapter_contacts_provider_with_idle_and_connect_timeouts(
    monkeypatch, request_timeout
):
    handler = partial(_pi_provider_reply, request_timeout)

    transport(monkeypatch, handler)
    llm = {
        "model_name": "deepseek-flash",
        "provider_name": "deepseek",
        "api_key": "fixture-only",
        "base_url": "https://fixture.invalid/v1",
    }
    cfg = {"llm": llm, "tools": [], "max_turns": 1}
    if request_timeout is not None:
        cfg["request_timeout"] = request_timeout
    trace = asyncio.run(Pi(cfg).run("test"))
    assert trace.output == "done", trace.error_message


def _pi_catalog_reply(request):
    assert json.loads(request.content)["max_completion_tokens"] == 128000
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def test_pi_uses_catalog_limit_for_other_thinking_models(monkeypatch):
    transport(monkeypatch, _pi_catalog_reply)
    llm = {
        "model_name": "openai/gpt-5",
        "api_key": "fixture-only",
        "model_api": "openai-completions",
        "base_url": "https://fixture.invalid/v1",
    }
    trace = asyncio.run(Pi({"llm": llm, "tools": [], "max_turns": 1}).run("test"))
    assert trace.output == "done", trace.error_message


def test_pi_bash_uses_stock_timeout_and_tail_truncation(tmp_path):
    assert not execute_tool(
        "bash", {"command": "true", "timeout": 601}, tmp_path
    ).is_error
    assert (
        "maximum is 2147483.647"
        in execute_tool(
            "bash",
            {"command": "false", "timeout": 2147483.648},
            tmp_path,
        ).content
    )
    command = "echo first-error; seq 15000; echo last-error; exit 7"
    result = execute_tool("bash", {"command": command}, tmp_path)
    spill = Path(result.details["fullOutputPath"])
    try:
        assert result.is_error and result.structured_content["exit_code"] == 7
        assert result.details["truncation"]["truncated"]
        assert "first-error" not in result.content and "last-error" in result.content
        assert spill.read_text().endswith("last-error\n")
    finally:
        spill.unlink()
    timed = execute_tool(
        "bash", {"command": "echo before; exec sleep 10", "timeout": 0.05}, tmp_path
    )
    assert timed.is_error and timed.content.startswith("before\n")
    assert "Command timed out after 0.05 seconds" in timed.content


async def _run_openclaw_shell_fixture(shell):
    output = await shell.execute(
        "exec",
        {
            "command": "echo first; head -c 40000 /dev/zero | tr '\\000' x; echo last",
        },
    )
    assert output["ok"] and len(output["output"]) > 30000
    assert not output["truncated"] and output["output"].rstrip().endswith("last")
    result = await shell.execute("exec", {"command": "exec sleep 10", "timeout": 0.05})
    assert result["error_type"] == "timeout"


def test_openclaw_exec_uses_stock_timeout_and_output_limit(tmp_path):
    shell = ToolRunner(tmp_path, tmp_path, ToolPolicy({}, tmp_path))

    run = partial(_run_openclaw_shell_fixture, shell)

    asyncio.run(run())


def _openclaw_reply(timeout, request):
    assert_timeout(request, timeout or 900)
    return httpx.Response(
        200,
        text=sse(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                "[DONE]",
            ]
        ),
    )


@pytest.mark.parametrize("timeout", [None, 1200])
def test_openclaw_wire_timeout(monkeypatch, timeout):
    handler = partial(_openclaw_reply, timeout)

    transport(monkeypatch, handler)
    config = {
        "llm": {
            "model_name": "fixture",
            "api_key": "fixture-only",
            "base_url": "https://fixture.invalid/v1",
        }
    }
    if timeout is not None:
        config["request_timeout"] = timeout
    outcome = asyncio.run(
        claw_provider.complete(
            config,
            [{"role": "user", "content": "test"}],
            [],
        )
    )
    assert outcome.text == "ok"


def test_claude_timeouts_and_output_environment():
    llm = {"api_key": "fixture-only"}
    env = _authentication_env(llm)
    assert env["API_TIMEOUT_MS"] == env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "900000"
    assert not any(key.startswith("BASH_") for key in env)
    assert (
        _authentication_env({**llm, "model_name": "deepseek-flash"})[
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
        ]
        == "65536"
    )
    env = _authentication_env(
        {**llm, "max_tokens": 49152, "api_timeout_ms": 1000000}, request_timeout=1200
    )
    assert env["API_TIMEOUT_MS"] == "1000000"
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "1200000"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "49152"


def _miniagent_reply(expected_timeout, expected_cap, request):
    assert_timeout(request, expected_timeout)
    assert json.loads(request.content)["max_tokens"] == expected_cap
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _miniagent_mock_client(real_client, handler, **kw):
    timeout = kw["timeout"]
    return real_client(
        http_client=httpx.AsyncClient(
            timeout=timeout,
            transport=httpx.MockTransport(handler),
        ),
        **kw,
    )


async def _run_miniagent_fixture():
    try:
        reply = await mini.chat([{"role": "user", "content": "test"}])
        assert reply.choices[0].message.content == "ok"
    finally:
        for item in mini._clients.values():
            await item.close()


@pytest.mark.parametrize(
    "config,expected_cap,expected_timeout",
    [
        ({"thinking": {"type": "enabled"}}, 65536, 900),
        (
            {
                "extra_body": {"thinking": {"type": "enabled"}},
                "max_tokens": 49152,
                "request_timeout": 1200,
            },
            49152,
            1200,
        ),
        ({"thinking": {"type": "disabled"}, "reasoning_effort": "high"}, 65536, 900),
    ],
)
def test_miniagent_cap_and_transport(
    monkeypatch, config, expected_cap, expected_timeout
):
    import openai

    real_client = openai.AsyncOpenAI

    handler = partial(_miniagent_reply, expected_timeout, expected_cap)

    client = partial(_miniagent_mock_client, real_client, handler)

    monkeypatch.setattr(mini, "AsyncOpenAI", client)
    monkeypatch.setattr(mini, "_clients", {})
    try:
        mini.configure(
            {
                "model_name": "deepseek-flash",
                "api_key": "fixture-only",
                "stream": False,
                **config,
            }
        )

        run = _run_miniagent_fixture

        asyncio.run(run())
    finally:
        mini.configure({})


def _miniagent_sse_reply(request):
    assert_timeout(request)
    assert request.url.path == "/anthropic/v1/messages"
    payload = json.loads(request.content)
    assert payload["max_tokens"] == 65536 and payload["stream"]
    return httpx.Response(
        200,
        text=sse(
            [
                {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "ok"},
                },
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
                {"type": "message_stop"},
            ]
        ),
    )


def test_miniagent_messages_sse_uses_default_thinking_cap(monkeypatch):
    transport(monkeypatch, _miniagent_sse_reply)
    try:
        mini.configure(
            {
                "model_name": "deepseek-flash",
                "api_key": "fixture-only",
                "base_url": "https://fixture.invalid/anthropic",
                "thinking": {"type": "enabled"},
            }
        )
        trace = asyncio.run(mini.chat([{"role": "user", "content": "test"}]))
        assert trace.choices[0].message.content == "ok"
    finally:
        mini.configure({})


def _relay_reply(requests, request):
    assert_timeout(request)
    requests.append(request.url.path)
    return httpx.Response(200, text="event: message_stop\n\n")


def _relay_mock_client(real_client, upstream, **kw):
    return real_client(transport=httpx.MockTransport(upstream), **kw)


def test_uds_relay_uses_connect_and_idle_timeouts(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only")
    requests = []
    real_client = httpx.Client

    upstream = partial(_relay_reply, requests)

    client = partial(_relay_mock_client, real_client, upstream)

    sock = str(tmp_path / "model.sock")
    monkeypatch.setattr(sandbox_runtime.httpx, "Client", client)
    with sandbox_runtime.relay(sock, "deepseek-flash", unix=True) as relay:
        assert relay.idle_timeout == 900
        with real_client(transport=httpx.HTTPTransport(uds=sock)) as guest:
            response = guest.post(
                "http://localhost/anthropic/v1/messages",
                json={"model": "deepseek-flash"},
            )
    assert response.status_code == 200 and response.text == "event: message_stop\n\n"
    assert requests == ["/anthropic/v1/messages"]


class _IdleTimeoutSocket:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def settimeout(self, timeout):
        assert timeout == 900


def _proxy_dial(upstream, address, timeout):
    assert address == ("model.invalid", 443) and timeout == 20
    return upstream


def _select_proxy_peers(handler, upstream, readers, writers, exceptional, timeout):
    assert readers == [handler.connection, upstream] and timeout == 900
    return [], [], []


def test_container_proxy_connect_and_no_progress_deadlines(monkeypatch):
    upstream = _IdleTimeoutSocket()
    handler = object.__new__(_model_proxy._ProxyHandler)
    handler.allowed = frozenset({("https", "model.invalid", 443)})
    handler.path = "model.invalid:443"
    handler.connection = object()
    handler.send_response_only = (
        lambda status, _: None if status == 200 else pytest.fail(status)
    )
    handler.end_headers = lambda: None

    dial = partial(_proxy_dial, upstream)

    select = partial(_select_proxy_peers, handler, upstream)

    monkeypatch.setattr(_model_proxy.socket, "create_connection", dial)
    monkeypatch.setattr(_model_proxy.select, "select", select)
    handler.do_CONNECT()


class _IdleTimeoutHTTPConnection:
    def __init__(self, host, port, *, timeout):
        assert (host, port, timeout) == ("model.invalid", 80, 20)
        self.sock = SimpleNamespace(settimeout=self.settimeout)
        self.idle = None

    def connect(self):
        pass

    def settimeout(self, timeout):
        self.idle = timeout

    def request(self, method, path, **_):
        assert self.idle == 900 and method == "POST" and path == "/v1/chat/completions"

    def getresponse(self):
        return SimpleNamespace(
            status=200, headers=Message(), read1=BytesIO(b"body").read1
        )

    def close(self):
        pass


def test_container_proxy_plain_http_switches_to_idle_timeout(monkeypatch):
    handler = object.__new__(_model_proxy._ProxyHandler)
    handler.allowed = frozenset({("http", "model.invalid", 80)})
    handler.path = "http://model.invalid/v1/chat/completions"
    handler.headers = Message()
    handler.command = "POST"
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    handler.send_response_only = (
        lambda status: None if status == 200 else pytest.fail(status)
    )
    handler.send_header = lambda *_: None
    handler.end_headers = lambda: None

    monkeypatch.setattr(
        _model_proxy.http.client, "HTTPConnection", _IdleTimeoutHTTPConnection
    )
    handler._forward_http()
    assert handler.wfile.getvalue() == b"body"


class _FakeSolverProcess:
    returncode = 0

    async def communicate(self):
        return b'{"output":"mock container"}', b""


@contextmanager
def _fixture_relay(_address, model, *, unix, request_timeout):
    assert model == "deepseek-flash" and unix and request_timeout == 900
    yield


async def _fixture_docker(expected, *command, **_):
    assert command[:2] == ("docker", "run") and "--network=none" in command
    flags = ("--cpus", "--memory", "--pids-limit", "--shm-size")
    values = {
        flag: command[command.index(flag) + 1] for flag in flags if flag in command
    }
    assert values == expected
    assert (
        f"type=bind,src={CHECKOUT / 'flashrsi'},dst=/native/flashrsi,readonly"
        in command
    )

    return _FakeSolverProcess()


@pytest.mark.parametrize(
    "container,kwargs,expected",
    [
        ({}, {}, {"--memory": "4g", "--pids-limit": "192"}),
        (
            {"cpus": 2, "memory": "3g", "pids": 128, "shm_size": "256m"},
            {"cpus": 1.5, "pids": 180},
            {
                "--cpus": "1.5",
                "--memory": "3g",
                "--pids-limit": "180",
                "--shm-size": "256m",
            },
        ),
    ],
)
def test_solver_resource_flags_compile_in_worktree(
    monkeypatch, tmp_path, container, kwargs, expected
):
    relay = _fixture_relay

    docker = partial(_fixture_docker, expected)

    monkeypatch.setattr(sandbox_runtime, "relay", relay)
    monkeypatch.setattr(sandbox_runtime.asyncio, "create_subprocess_exec", docker)
    agent = Pi(
        {
            "llm": {"model_name": "deepseek-flash", "provider_name": "deepseek"},
            "container": container,
        }
    )
    trajectory, _ = asyncio.run(
        sandbox_runtime.solver(agent, "fixture", tmp_path, image="fixture", **kwargs)
    )
    assert trajectory.output == "mock container"
