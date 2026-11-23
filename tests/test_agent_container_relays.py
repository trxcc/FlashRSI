"""Verify Claude Code and Codex model relays with scripted responses."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest

import flashrsi
from flashrsi.agents.claude_code import ClaudeCode, _authentication_env
from flashrsi.agents.codex import Codex, _settings
from flashrsi.agents.codex.python_core import client as codex_client
from flashrsi.agents.codex.python_core.history import History
from flashrsi.agents.codex.python_core.models import Model
from flashrsi.benchmarks._suite import resolve_root
from flashrsi.core.infra import trajectory_infra_reason
from flashrsi.integrations import _model_egress, _model_proxy, sandbox_runtime
from flashrsi.integrations._container_payload import _setup_script, _stage
from flashrsi.integrations._docker import _docker

CHECKOUT = Path(__file__).resolve().parents[1]
assert Path(flashrsi.__file__).resolve().parent == CHECKOUT / "flashrsi"

FIXTURE_KEY = "fixture-provider-credential"
PROBE = (
    "Write probe.txt with Bash, then read it with another tool and report the content."
)


def config(**llm):
    return {
        "timeout": 40,
        "llm": {
            "model_name": "deepseek-flash",
            "base_url": "https://api.deepseek.com/anthropic",
            "api_key": FIXTURE_KEY,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            **llm,
        },
    }


@contextmanager
def server(handler, address="127.0.0.1"):
    with ThreadingHTTPServer((address, 0), handler) as httpd:
        thread = Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            yield f"http://{address}:{httpd.server_port}"
        finally:
            httpd.shutdown()
            thread.join()


@pytest.fixture
def docker_image():
    image = os.environ.get("FLASHRSI_SUITE_DOCKER_IMAGE")
    if image:
        return image
    reference = "reference=flashrsi-workspace:*"
    command = ["docker", "image", "ls", "--filter", reference, "--quiet"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    images = set(result.stdout.splitlines())
    if len(images) != 1:
        pytest.fail(
            "set FLASHRSI_SUITE_DOCKER_IMAGE to select an installed smoke image"
        )
    return images.pop()


@pytest.mark.parametrize("agent_type,suffix", [(Codex, ""), (ClaudeCode, "/anthropic")])
def test_shared_role_config_and_placeholder(agent_type, suffix, tmp_path):
    values = config(
        max_tokens=32768, request_timeout=812, extra_headers={"Cookie": FIXTURE_KEY}
    )
    agent = agent_type(values)
    routed = sandbox_runtime.local_config(agent, "http://127.0.0.1:1234")
    assert FIXTURE_KEY not in json.dumps(routed)
    assert routed["llm"]["base_url"] == "http://127.0.0.1:1234" + suffix
    assert routed["llm"]["request_timeout"] == 812
    assert agent.config.llm.api_key == FIXTURE_KEY
    assert routed.get("max_turns") is None
    if agent_type is Codex:
        settings, env, _ = _settings(routed, tmp_path, tmp_path)
        assert (
            settings["model_providers.flashrsi.base_url"] == routed["llm"]["base_url"]
        )
        assert env["FLASHRSI_CODEX_API_KEY"] == "flashrsi-local-relay"
        assert routed["sandbox"] == "danger-full-access"
    else:
        env = _authentication_env(routed["llm"])
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-relay-placeholder"
        assert env["API_TIMEOUT_MS"] == "812000"
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert not any(name.startswith("BASH_") for name in env)
        assert routed["permission_mode"] == "bypassPermissions"


def test_direct_provider_endpoints_and_cli_defaults(tmp_path):
    assert (
        _authentication_env(config(base_url="https://api.deepseek.com/v1")["llm"])[
            "ANTHROPIC_BASE_URL"
        ]
        == "https://api.deepseek.com/anthropic"
    )
    env = _authentication_env(config()["llm"])
    assert env["API_TIMEOUT_MS"] == "900000"
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "900000"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "65536"
    settings, _, _ = _settings(config(), tmp_path, tmp_path)
    assert settings["model_providers.flashrsi.base_url"] == "https://api.deepseek.com"
    for turns in (None, 200):
        _settings({**config(), "max_turns": turns}, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="max_turns"):
        _settings({**config(), "max_turns": 0}, tmp_path, tmp_path)


def sse(*events):
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )


def completed():
    return {"type": "response.completed", "response": {"id": "resp_fixture"}}


def done(item, index=0):
    return {"type": "response.output_item.done", "item": item, "output_index": index}


def _reasoning_replay_reply(requests, events, item, call, request):
    assert request.url.path == "/responses"
    assert request.extensions["timeout"]["read"] == 900
    assert request.extensions["timeout"]["connect"] == 20
    requests.append(json.loads(request.content))
    body = (
        sse(*events, done(item), done(call, 1), completed())
        if len(requests) == 1
        else sse(completed())
    )
    return httpx.Response(200, text=body)


async def _request_two_codex_turns():
    model = Model("deepseek-flash", {})
    history = History(model)
    llm = {
        **config(base_url="https://api.deepseek.com")["llm"],
        "_deepseek_responses": True,
    }
    async with codex_client.ResponsesClient(
        llm, model, {}, "thread", lambda _: None
    ) as client:
        turn = await client.response([], [])
        for output in turn["items"]:
            history.append(output)
        history.append(
            {"type": "function_call_output", "call_id": "probe", "output": "done"}
        )
        await client.response(history.input_items(), [])


@pytest.mark.parametrize(
    "carrier", ["content", "reasoning_text", "reasoning_content", "delta"]
)
def test_codex_replays_reasoning_without_summary(carrier, monkeypatch):
    requests = []
    item = {"type": "reasoning", "id": "rs_fixture"}
    events = []
    if carrier == "content":
        item[carrier] = [{"type": "reasoning_text", "text": "fixture reasoning"}]
    elif carrier == "delta":
        events.append(
            {
                "type": "response.reasoning_text.delta",
                "output_index": 0,
                "delta": "fixture reasoning",
            }
        )
    else:
        item[carrier] = "fixture reasoning"
    call = {
        "type": "function_call",
        "name": "shell_command",
        "call_id": "probe",
        "arguments": "{}",
    }

    handle = partial(_reasoning_replay_reply, requests, events, item, call)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )

    asyncio.run(_request_two_codex_turns())
    assert requests[0]["reasoning"] == {"effort": "high"}
    assert requests[0]["max_output_tokens"] == 65536
    assert [item["type"] for item in requests[1]["input"]] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    reasoning = requests[1]["input"][0]
    assert reasoning["content"] == [
        {"type": "reasoning_text", "text": "fixture reasoning"}
    ]
    assert "summary" not in reasoning


class _TruncatedUpstream(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(b"3\r\nabc\r\n")
        self.close_connection = True

    def log_message(self, *_args):
        pass


def test_sidecar_preserves_truncated_http_body():
    with server(_TruncatedUpstream) as upstream:
        origin = urlsplit(upstream)
        allowed = {("http", origin.hostname, origin.port)}
        with server(_model_proxy._handler(allowed)) as proxy:
            with httpx.Client(proxy=proxy, trust_env=False) as client:
                with pytest.raises(httpx.RemoteProtocolError):
                    client.post(upstream + "/responses", json={})
                response = client.post("http://unexpected.invalid/responses", json={})
                assert response.status_code == 403


def _shell_call(call_id, command, index=0):
    return done(
        {
            "type": "function_call",
            "name": "shell_command",
            "call_id": call_id,
            "arguments": json.dumps({"command": command}),
        },
        index,
    )


def _codex_steps(request, turn, *, network_none):
    if turn == 1:
        reasoning = {"type": "reasoning_text", "text": "fixture reasoning"}
        item = {"type": "reasoning", "id": "rs_fixture", "content": [reasoning]}
        command = script(network_none=network_none)
        return sse(done(item), _shell_call("write", command, index=1), completed())
    if turn == 2:
        assert any(item.get("type") == "reasoning" for item in request["input"])
        return sse(_shell_call("read", "cat probe.txt"), completed())
    assert any(
        "relay-fixture" in item.get("output", "")
        for item in request["input"]
        if item.get("type") == "function_call_output" and item["call_id"] == "read"
    )
    content = {"type": "output_text", "text": "relay-fixture"}
    output = {"type": "message", "role": "assistant", "content": [content]}
    return sse(done(output), completed())


def _claude_steps(request, turn, *, network_none):
    if turn == 1:
        block = {"type": "tool_use", "id": "toolu_write", "name": "Bash", "input": {}}
        delta = {
            "type": "input_json_delta",
            "partial_json": json.dumps(
                {
                    "command": script(network_none=network_none),
                    "description": "Write the probe file",
                }
            ),
        }
    elif turn == 2:
        assert "toolu_write" in json.dumps(request["messages"])
        block = {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {}}
        delta = {
            "type": "input_json_delta",
            "partial_json": json.dumps({"file_path": "/task/probe.txt"}),
        }
    else:
        assert "toolu_read" in json.dumps(request["messages"])
        assert "relay-fixture" in json.dumps(request["messages"])
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": "relay-fixture"}
    return sse(
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{turn}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "deepseek-flash",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 6, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": block},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use" if turn < 3 else "end_turn",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    )


def script(*, network_none=False):
    network_check = 'test "$(ls /sys/class/net)" = lo && ' if network_none else ""
    return network_check + (
        'test -z "${DEEPSEEK_API_KEY:-}" && '
        "case \"${ANTHROPIC_API_KEY:-}\" in ''|sk-ant-relay-placeholder) ;; *) exit 1;; esac && "
        "case \"${FLASHRSI_CODEX_API_KEY:-}\" in ''|flashrsi-local-relay) ;; *) exit 1;; esac && "
        "python3 -c 'import os; assert not any("
        'value == "fixture-provider-credential" for value in os.environ.values())\' && '
        "printf relay-fixture > probe.txt"
    )


class _FixtureHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, requests, errors, network_none=False):
        self.requests = requests
        self.errors = errors
        self.network_none = network_none
        super().__init__(*args)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        path = self.path.split("?", 1)[0]
        code = 200
        try:
            assert body["model"] == "deepseek-flash"
            if path == "/responses":
                assert self.headers["Authorization"] == "Bearer " + FIXTURE_KEY
                assert body["reasoning"] == {"effort": "high"}
                assert body["max_output_tokens"] == 65536
            else:
                assert self.headers["x-api-key"] == FIXTURE_KEY
                assert body["thinking"]["type"] in {"enabled", "adaptive"}
                assert body["output_config"]["effort"] == "high"
                assert body["max_tokens"] == 65536
            self.requests.append((path, body))
            assert path in {"/responses", "/anthropic/v1/messages"}
            steps = _codex_steps if path == "/responses" else _claude_steps
            response = steps(body, len(self.requests), network_none=self.network_none)
        except Exception as exc:
            self.errors.append(repr(exc))
            response, code = "fixture rejected request", 400
        self.send_response(code)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, *_args):
        pass


@pytest.mark.skipif(
    os.environ.get("FLASHRSI_SUITE_DOCKER_SMOKE") != "1",
    reason="requires the local Docker smoke environment",
)
@pytest.mark.parametrize("agent_type", [Codex, ClaudeCode])
def test_docker_native_multi_turn(agent_type, tmp_path, monkeypatch, docker_image):
    requests, errors = [], []
    monkeypatch.setenv("DEEPSEEK_API_KEY", FIXTURE_KEY)
    with server(
        partial(_FixtureHandler, requests=requests, errors=errors, network_none=True)
    ) as url:
        monkeypatch.setattr(sandbox_runtime, "PUBLIC_ORIGIN", url)
        work = tmp_path / "work"
        work.mkdir(mode=0o755)
        trace, container = asyncio.run(
            sandbox_runtime.solver(
                agent_type(config()),
                PROBE,
                work,
                image=docker_image,
            )
        )
        assert (
            subprocess.run(
                ["docker", "inspect", container], capture_output=True
            ).returncode
            != 0
        )
    assert not errors
    assert trace.status == "finished", trace.error_message
    assert trajectory_infra_reason(trace) is None
    assert (work / "probe.txt").read_text() == "relay-fixture"
    assert trace.output == "relay-fixture"
    assert len(requests) == 3


async def _run_sidecar_fixture(
    agent_type, source, tmp_path, docker_image, _start_proxy, _PROXY_ENV, errors
):
    gateway = await sandbox_runtime.gateway()
    name = "flashrsi-sidecar-fixture-" + uuid4().hex[:10]
    network, proxy = name + "-internal", name + "-proxy"
    agent = agent_type({**config(), "container_runtime_kit": str(source)})
    root = "/opt/flashrsi-lite"
    with sandbox_runtime.relay((gateway, 0), "deepseek-flash", unix=False) as relay:
        token = sandbox_runtime.RELAY.set(relay)
        try:
            routed = sandbox_runtime.route_config(agent, dict(agent.config))
            _stage(tmp_path, agent, routed, PROBE)
            assert (tmp_path / "kit/site-packages.tar.gz").is_file()
            await _docker(
                "network",
                "create",
                "--internal",
                "--opt=com.docker.network.bridge.gateway_mode_ipv4=isolated",
                network,
                timeout=30,
            )
            await _docker(
                "run",
                "-d",
                "--name",
                name,
                "--network",
                network,
                "--user",
                "root",
                "--entrypoint",
                "/bin/sh",
                docker_image,
                "-c",
                "sleep 180",
                timeout=30,
            )
            await _docker("exec", name, "mkdir", "-p", root, "/task", timeout=30)
            await _docker("cp", f"{tmp_path}/.", f"{name}:{root}", timeout=60)
            await _docker("exec", name, "/bin/sh", "-c", _setup_script(), timeout=60)
            await _start_proxy(proxy, network, _model_egress._origins(routed), 40)
            env = [
                part
                for key, value in _PROXY_ENV.items()
                for part in ("-e", f"{key}={value}")
            ]
            result = await _docker(
                "exec",
                *env,
                "-w",
                "/task",
                name,
                f"{root}/venv/bin/python",
                "-I",
                "-c",
                f"import sys, runpy; sys.path.insert(0, {root + '/package'!r}); "
                "import flashrsi.agents; "
                "runpy.run_module('flashrsi.integrations._container', "
                "run_name='__main__', alter_sys=True)",
                f"{root}/request.json",
                "/tmp/trajectory.json",
                timeout=65,
                check=False,
            )
            data = await _docker(
                "exec", name, "cat", "/tmp/trajectory.json", timeout=20, check=False
            )
            assert data.returncode == 0, result.stderr
            trace = json.loads(data.stdout)
            assert not errors
            assert trace["status"] == "finished", trace["error_message"]
            assert trace["output"] == "relay-fixture"
            file = await _docker("exec", name, "cat", "/task/probe.txt", timeout=20)
            assert file.stdout == "relay-fixture"
        finally:
            await _docker("rm", "-f", name, proxy, timeout=20, check=False)
            await _docker("network", "rm", network, timeout=20, check=False)
            sandbox_runtime.RELAY.reset(token)


@pytest.mark.skipif(
    os.environ.get("FLASHRSI_SUITE_DOCKER_SMOKE") != "1",
    reason="requires the local Docker smoke environment",
)
@pytest.mark.parametrize("agent_type", [Codex, ClaudeCode])
def test_docker_internal_sidecar(agent_type, tmp_path, monkeypatch, docker_image):
    pytest.importorskip("flashrsi.integrations.harbor_suite")
    from flashrsi.integrations.swe_bench import _PROXY_ENV, _start_proxy

    source = resolve_root({}, "harbor_track") / "runtime_kit"
    if not (source / ".ready").is_file():
        pytest.skip("suite runtime kit is unavailable")
    requests, errors = [], []
    monkeypatch.setenv("DEEPSEEK_API_KEY", FIXTURE_KEY)
    with server(partial(_FixtureHandler, requests=requests, errors=errors)) as url:
        monkeypatch.setattr(sandbox_runtime, "PUBLIC_ORIGIN", url)

        asyncio.run(
            _run_sidecar_fixture(
                agent_type,
                source,
                tmp_path,
                docker_image,
                _start_proxy,
                _PROXY_ENV,
                errors,
            )
        )
    assert len(requests) == 3


class _StoppingClaudeClient:
    def __init__(self, options, *, options_seen):
        options_seen.append(options)

    async def connect(self):
        raise RuntimeError("fixture stops before any model request")

    async def disconnect(self):
        pass


@pytest.mark.parametrize("turns", [None, 200])
def test_claude_sdk_episode_options(turns, monkeypatch, tmp_path):
    import claude_agent_sdk

    options_seen = []

    monkeypatch.setattr(
        claude_agent_sdk,
        "ClaudeSDKClient",
        partial(_StoppingClaudeClient, options_seen=options_seen),
    )
    values = {**config(), "max_turns": turns}
    routed = sandbox_runtime.local_config(ClaudeCode(values), "http://127.0.0.1:1234")
    trace = asyncio.run(ClaudeCode(routed).run("test", tmp_path))
    assert trace.error_message == "fixture stops before any model request"
    assert len(options_seen) == 1
    options = options_seen[0]
    assert options.max_turns == turns
    assert options.permission_mode == "bypassPermissions"
    assert options.thinking == {"type": "adaptive"}
    assert options.effort == "high"
    assert options.env["HOME"] != os.environ.get("HOME")
    assert (
        Path(options.env["HOME"]).parent
        == Path(options.env["CLAUDE_CONFIG_DIR"]).parent
    )
    assert not Path(options.env["HOME"]).exists()


def test_codex_honors_explicit_turn_budget(tmp_path):
    requests, errors = [], []
    with server(partial(_FixtureHandler, requests=requests, errors=errors)) as url:
        agent = Codex(
            {
                **config(base_url=url),
                "_suite_model_relay": True,
                "sandbox": "danger-full-access",
                "max_turns": 1,
            }
        )
        trace = asyncio.run(agent.run(PROBE, tmp_path))
    assert not errors
    assert len(requests) == 1
    assert trace.status == "error"
    assert "max_turns (1)" in trace.error_message
    assert trajectory_infra_reason(trace) is None
