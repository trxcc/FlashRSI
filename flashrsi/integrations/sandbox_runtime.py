"""Run native harnesses in isolated containers through a host-side model relay.

The host owns Docker and the DeepSeek credential. Guest configurations carry
a route and a placeholder, while the relay authorizes only the owned solver.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socketserver
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import httpx

from flashrsi._records import _redact, _redact_config, _secrets
from flashrsi.core import Agent, InfraError, Trajectory
from flashrsi.integrations._container_payload import _plain
from flashrsi.integrations._docker import _docker
from flashrsi.integrations.sandbox_assets.runtime_kit import (
    extend_runtime_kit as _extend_runtime_kit,
)

REPO = Path(__file__).resolve().parents[2]
SCRATCH = REPO / ".git/sandbox-runtime"
_LABEL_KEY = "org.flashrsi.sandbox"
LABEL = f"{_LABEL_KEY}=native"
RELAY = ContextVar("suite_model_relay", default=None)
ROUTES = {
    "/v1/chat/completions": "/v1/chat/completions",
    "/responses": "/responses",
    "/v1/responses": "/responses",
    "/anthropic/v1/messages": "/anthropic/v1/messages",
}
CLAUDE_GUEST_CLI = "/native/claude"
PUBLIC_ORIGIN = "https://api.deepseek.com"
RELAY_PLACEHOLDER = "flashrsi-local-relay"
MODEL_IDLE_TIMEOUT = 900
MODEL_CONNECT_TIMEOUT = 20


def scratch() -> tempfile.TemporaryDirectory:
    if (REPO / ".git").is_file():
        return tempfile.TemporaryDirectory(prefix="flashrsi-episode-")
    SCRATCH.mkdir(mode=0o700, parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="ep-", dir=SCRATCH)


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def key() -> str:
    value = os.getenv("DEEPSEEK_API_KEY")
    if not value:
        raise InfraError("DEEPSEEK_API_KEY is required for suite model execution")
    return value


class _Forward(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.idle_timeout)

    def log_message(self, *_args):
        pass

    def do_POST(self):
        if (
            self.server.allowed is not None
            and self.client_address[0] not in self.server.allowed
        ):
            self.send_error(403)
            return
        route, separator, query = self.path.partition("?")
        if route not in ROUTES:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 32 * 1024 * 1024:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            payload = json.loads(body)
            if payload.get("model") != self.server.model:
                self.send_error(422)
                return
        except (OSError, ValueError, AttributeError):
            self.send_error(400)
            return
        if route == "/anthropic/v1/messages" and self.server.anthropic_options:
            payload.update(self.server.anthropic_options)
            body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        for header in ("Anthropic-Version", "Anthropic-Beta", "Accept", "User-Agent"):
            if header in self.headers:
                headers[header] = self.headers[header]
        if route.startswith("/anthropic/"):
            headers["x-api-key"] = self.server.credential
        else:
            headers["Authorization"] = "Bearer " + self.server.credential
        sent = False
        try:
            target = PUBLIC_ORIGIN + ROUTES[route] + separator + query
            with self.server.client.stream(
                "POST",
                target,
                content=body,
                headers=headers,
                timeout=httpx.Timeout(
                    self.server.idle_timeout, connect=MODEL_CONNECT_TIMEOUT
                ),
            ) as upstream:
                self.send_response(upstream.status_code)
                self.send_header(
                    "Content-Type",
                    upstream.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Connection", "close")
                self.send_header("Transfer-Encoding", "chunked")
                self.close_connection = True
                if upstream.headers.get("Retry-After"):
                    self.send_header("Retry-After", upstream.headers["Retry-After"])
                self.end_headers()
                sent = True
                for chunk in upstream.iter_bytes():
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
        except (OSError, httpx.HTTPError):
            if not sent:
                try:
                    self.send_error(502, "Upstream transport interrupted")
                except OSError:
                    pass


@contextmanager
def relay(
    address: str | tuple[str, int],
    model: str,
    *,
    unix: bool,
    request_timeout: float = MODEL_IDLE_TIMEOUT,
) -> Iterator[Any]:
    credential = key()
    server = (_UnixServer if unix else ThreadingHTTPServer)(address, _Forward)
    server.credential = credential
    server.model = model
    server.allowed = None if unix else set()
    server.idle_timeout = request_timeout
    server.anthropic_options = {}
    server.client = httpx.Client(
        timeout=httpx.Timeout(request_timeout, connect=MODEL_CONNECT_TIMEOUT),
        trust_env=False,
    )
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        with ExitStack() as resources:
            server.resources = resources
            yield server
    finally:
        server.shutdown()
        server.server_close()
        server.client.close()
        thread.join()


def local_config(agent: Agent, endpoint: str) -> dict[str, Any]:
    return _local_config(agent.name, agent.config, endpoint)


def _local_config(
    agent_name: str, config: Mapping[str, Any], endpoint: str
) -> dict[str, Any]:
    config = _redact_config(config)
    llm = config["llm"]
    for field in ("api_key", "auth_token", "oauth_token", "extra_headers", "headers"):
        llm.pop(field, None)
    llm.pop("side_api_key", None)
    llm.pop("side_base_url", None)
    llm["api_key"] = (
        "sk-ant-relay-placeholder" if agent_name == "claude_code" else RELAY_PLACEHOLDER
    )
    suffix = {"claude_code": "/anthropic", "codex": ""}.get(agent_name, "/v1")
    llm["base_url"] = endpoint.rstrip("/") + suffix
    if agent_name in {"claude_code", "codex"}:
        config["_suite_model_relay"] = True
        llm["auth"] = "api_key"
    if agent_name == "claude_code":
        config.pop("cli_path", None)
        config.setdefault("permission_mode", "bypassPermissions")
    if agent_name == "codex":
        # The container already owns the filesystem and network isolation.
        config.setdefault("sandbox", "danger-full-access")
    config.setdefault("request_timeout", MODEL_IDLE_TIMEOUT)
    return config


def _anthropic_options(agent: Agent, config: Mapping[str, Any]) -> dict[str, Any]:
    if agent.name != "claude_code":
        return {}
    from flashrsi.agents.claude_code import _effort, _thinking

    llm = config["llm"]
    fields = {}
    thinking = _thinking(llm)
    if thinking is not None:
        fields["thinking"] = thinking
    effort = _effort(llm)
    if effort is not None:
        fields["output_config"] = {"effort": effort}
    max_tokens = llm.get("max_tokens")
    if max_tokens is None and llm["model_name"] == "deepseek-flash":
        max_tokens = 65536
    if max_tokens is not None:
        fields["max_tokens"] = max_tokens
    return fields


def _claude_cli() -> Path:
    package = find_spec("claude_agent_sdk")
    if package is None or package.origin is None:
        raise InfraError("suite requires the installed Claude Agent SDK")
    binary = Path(package.origin).parent / "_bundled/claude"
    if not os.access(binary, os.X_OK):
        raise InfraError("suite requires the Claude Agent SDK's bundled executable")
    return binary.resolve()


def route_config(agent: Agent, config: dict[str, Any]) -> dict[str, Any]:
    server = RELAY.get()
    if server is None:
        return config
    host, port = server.server_address
    routed = _local_config(agent.name, config, f"http://{host}:{port}")
    server.anthropic_options = _anthropic_options(agent, routed)
    server.idle_timeout = routed["llm"].get(
        "request_timeout", routed["request_timeout"]
    )
    if agent.name in {"claude_code", "codex"} and (
        kit := routed.get("container_runtime_kit")
    ):
        routed["container_runtime_kit"] = str(
            _extend_runtime_kit(
                Path(kit), agent._container_dependencies, server.resources
            )
        )
    config.clear()
    config.update(routed)
    return config


async def gateway() -> str:
    result = await _docker(
        "network", "inspect", "bridge", "--format", "{{json .IPAM.Config}}", timeout=10
    )
    configs = json.loads(result.stdout)
    return next(item["Gateway"] for item in configs if ":" not in item["Gateway"])


async def admit(container: str, *, server: Any = None) -> None:
    server = RELAY.get() if server is None else server
    if server is None:
        return
    result = await _docker(
        "inspect",
        "--format",
        "{{json .NetworkSettings.Networks}}",
        container,
        timeout=10,
    )
    addresses = [
        entry["IPAddress"]
        for entry in json.loads(result.stdout).values()
        if entry.get("IPAddress")
    ]
    if not addresses:
        raise InfraError("suite owned solver has no model relay address")
    server.allowed.update(addresses)


async def relay_harbor(environment: Any) -> bool:
    if RELAY.get() is None:
        return False
    from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

    name = _sanitize_docker_compose_project_name(environment.session_id)
    result = await _docker(
        "ps",
        "-q",
        "--filter",
        f"label=com.docker.compose.project={name}",
        "--filter",
        "label=com.docker.compose.service=main",
        timeout=15,
    )
    containers = result.stdout.split()
    if len(containers) != 1:
        raise InfraError("suite could not identify the owned Harbor solver")
    await admit(containers[0])
    return True


def docker_resource_args(
    *,
    cpus: float | None = None,
    memory: str | int | None = None,
    pids: int | None = None,
    shm_size: str | int | None = None,
) -> list[str]:
    args = []
    for flag, value in (
        ("--cpus", cpus),
        ("--memory", memory),
        ("--pids-limit", pids),
        ("--shm-size", shm_size),
    ):
        if value is not None:
            args.extend((flag, str(value)))
    return args


@dataclass(frozen=True, slots=True)
class _ManagedPython:
    launcher: Path
    environment: Path
    root: Path
    launcher_root: Path


def _managed_python() -> _ManagedPython:
    launcher = Path(sys.executable)
    python_root = launcher.resolve().parents[1]
    launcher_root = (
        (launcher.parent / launcher.readlink()).parents[1]
        if launcher.is_symlink()
        else python_root
    )
    git_dir = REPO / ".git"
    if git_dir.is_file():
        git_dir = Path(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(REPO),
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-common-dir",
                ],
                text=True,
            ).strip()
        )
    if not python_root.is_relative_to(git_dir):
        raise InfraError("the sandbox requires its installed managed Python runtime")
    return _ManagedPython(launcher, Path(sys.prefix), python_root, launcher_root)


def _solver_mounts(
    bundle: Path,
    workspace: Path,
    workdir: str,
    python: _ManagedPython,
    *,
    claude: bool,
) -> list[tuple[Path, str, bool]]:
    mounts = [
        (bundle, "/native", True),
        (REPO / "flashrsi", "/native/flashrsi", True),
        (python.environment, str(python.environment), True),
        (python.root, str(python.root), True),
        (workspace, workdir, False),
    ]
    if claude:
        (bundle / "claude").touch()
        mounts.append((_claude_cli(), CLAUDE_GUEST_CLI, True))
    if python.launcher_root != python.root:
        mounts.append((python.root, str(python.launcher_root), True))
    return mounts


def _docker_command(
    name: str,
    image: str,
    workdir: str,
    resources: list[str],
    launcher: Path,
    mounts: list[tuple[Path, str, bool]],
) -> list[str]:
    owner = f"{os.getuid()}:{os.getgid()}"
    args = [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--init",
        "--name",
        name,
        "--label",
        LABEL,
        "--network=none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        *resources,
        "--user",
        owner,
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,mode=1777",
        "-w",
        workdir,
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPATH=/native",
    ]
    for source, dest, readonly in mounts:
        mount = f"type=bind,src={source},dst={dest}"
        if readonly:
            mount += ",readonly"
        args.extend(("--mount", mount))
    return [
        *args,
        "--entrypoint",
        str(launcher),
        image,
        "-m",
        "flashrsi.integrations.sandbox_assets.solver_guest",
    ]


async def _run_solver(command: list[str], name: str, timeout: float) -> Trajectory:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    finally:
        if process.returncode is None:
            await asyncio.shield(remove(name))
            if process.returncode is None:
                process.kill()
                await process.wait()
    if len(output) > 24 * 1024 * 1024:
        raise InfraError("native solver trajectory exceeded its transport budget")
    try:
        value = json.loads(output)
        return Trajectory(**_redact(value, _secrets(os.environ)))
    except (ValueError, TypeError):
        errors = re.findall(
            rb"(?m)^([A-Za-z][A-Za-z0-9_]{0,60}(?:Error|Exception)):", stderr
        )
        detail = f", {errors[-1].decode('ascii')}" if errors else ""
        raise InfraError(
            f"native solver failed before returning a trajectory (exit {process.returncode}{detail})"
        ) from None


async def solver(
    agent: Agent,
    prompt: str,
    workspace: Path,
    *,
    image: str,
    workdir: str = "/task",
    cpus: float | None = None,
    memory: str | int | None = None,
    pids: int | None = None,
    shm_size: str | int | None = None,
) -> tuple[Trajectory, str]:
    """Run the native harness with optional agent.container or per-call limits."""
    resource_config = agent.config.get("container", {})
    resources = docker_resource_args(
        cpus=resource_config.get("cpus") if cpus is None else cpus,
        memory=resource_config.get("memory", "4g") if memory is None else memory,
        pids=resource_config.get("pids", 192) if pids is None else pids,
        shm_size=resource_config.get("shm_size") if shm_size is None else shm_size,
    )
    if not workspace.is_dir():
        raise ValueError("the sandbox solver requires a real workspace")
    name = f"flashrsi-native-solver-{uuid4().hex[:12]}"
    python = _managed_python()
    claude = agent.name == "claude_code"
    with scratch() as temporary:
        bundle = Path(temporary)
        # Mount only the importable package. The Git checkout may contain private data.
        (bundle / "flashrsi").mkdir()
        agent.snapshot.save(bundle / "snapshot.json")
        config = local_config(agent, "http://127.0.0.1:28451")
        request = {
            "agent": f"{type(agent).__module__}:{type(agent).__qualname__}",
            "config": _plain(config),
            "prompt": prompt,
        }
        (bundle / "request.json").write_text(json.dumps(request), encoding="utf-8")
        llm = config["llm"]
        request_timeout = llm.get("request_timeout", config["request_timeout"])
        with relay(
            str(bundle / "relay.sock"),
            llm["model_name"],
            unix=True,
            request_timeout=request_timeout,
        ) as server:
            if claude:
                server.anthropic_options = _anthropic_options(agent, config)
            mounts = _solver_mounts(bundle, workspace, workdir, python, claude=claude)
            command = _docker_command(
                name, image, workdir, resources, python.launcher, mounts
            )
            trajectory = await _run_solver(
                command, name, agent.config.get("timeout", 900) + 45
            )
            return trajectory, name


async def remove(name: str) -> None:
    selector = '{{index .Config.Labels "' + _LABEL_KEY + '"}}'
    inspected = await _docker(
        "inspect", "--format", selector, name, timeout=15, check=False
    )
    if inspected.returncode == 0 and inspected.stdout.strip() == "native":
        await _docker("rm", "-f", name, timeout=30)


def verify_workspace(root: Path) -> None:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise InfraError("solver produced a symlink or special workspace member")
        if path.is_file():
            total += path.stat().st_size
    if total > 128 * 1024 * 1024:
        raise InfraError("solver workspace exceeds the artifact size budget")
