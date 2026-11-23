"""Bind reviewed Harbor and SWE tasks and share their container runtime support.

Native shells stage private per-task payloads and use read-only runtime mounts.
Harbor dependencies stay lazy so the SWE backend can run independently.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import yaml

from flashrsi._records import _redact, _secrets
from flashrsi.benchmarks._suite import DEFAULT_AGENT_SHELLS, OPTIONAL_AGENT_SHELLS
from flashrsi.benchmarks.harbor_suite import HarborSuite, bind_track
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.core.infra import infra_reason, trajectory_infra_reason
from flashrsi.integrations._container_payload import _ROOT, _plain
from flashrsi.integrations._docker import _docker

if TYPE_CHECKING:
    from harbor.models.trial.result import TrialResult
    from harbor.trial.trial import Trial

_CONTAINER_SHELLS = (*DEFAULT_AGENT_SHELLS, *OPTIONAL_AGENT_SHELLS)
_RUNTIME = "/opt/flashrsi-runtime"
_RELAY_MOUNT = "/opt/flashrsi-relay"
_RELAY_KEY = "flashrsi-local-relay"
_ANTHROPIC_RELAY_KEY = "sk-ant-relay-placeholder"


def bind(agent: Agent, benchmark: HarborSuite) -> None:
    from flashrsi.integrations.sandbox_runtime import RELAY, gateway, relay

    benchmark.load()
    runners = {}
    for name, backend in benchmark._backends.items():
        bind_track(agent, backend, name)
        runners[name] = agent._env_runner

    async def run(target: Agent, task: Task) -> Trajectory:
        benchmark.check_task(task)
        host = await gateway()
        with relay((host, 0), target.config.llm.model_name, unix=False) as server:
            token = RELAY.set(server)
            try:
                return await runners[benchmark._routes[task.task_id]](target, task)
            finally:
                RELAY.reset(token)

    agent._env_runner = run
    if (benchmark.root / "runtime_kit/.ready").is_file():
        agent.config["container_runtime_kit"] = str(benchmark.root / "runtime_kit")



def _host_runtime() -> tuple[str, dict[str, Path]]:
    """Return the managed venv launcher and narrowly scoped read-only binds."""
    python = Path(sys.executable)
    environment = Path(sys.prefix)
    python_root = python.resolve().parents[1]
    mounts = {str(environment): environment, str(python_root): python_root}
    if python.is_symlink():
        # The venv links via the bootstrap's version-independent Python path.
        launcher_root = Path(os.path.abspath(python.parent / python.readlink())).parents[1]
        mounts[str(launcher_root)] = python_root
    mounts[f"{_RUNTIME}/flashrsi"] = Path(__file__).resolve().parents[1]
    codex = Path.home() / ".local/lib/codex/current/bin/codex"
    if codex.is_file():
        mounts[f"{_RUNTIME}/bin/codex"] = codex.resolve()
    return str(python), mounts


def _shell_turns(agent: Agent, config: dict[str, Any], default: int | None = None) -> None:
    name = getattr(agent, "name", None)
    turns = config.get("max_turns", default)
    if turns is not None and (type(turns) is not int or turns < 1):
        raise ValueError("agent.max_turns must be a positive integer or null")
    if turns is None and name in {"codex", "miniagent", "openclaw"}:
        config.pop("max_turns", None)
    elif name in _CONTAINER_SHELLS or "max_turns" in config:
        config["max_turns"] = turns


def _relay_shell_env(endpoint: str) -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": endpoint + "/anthropic",
        "ANTHROPIC_API_KEY": _ANTHROPIC_RELAY_KEY,
        "OPENAI_BASE_URL": endpoint + "/v1",
        "OPENAI_API_KEY": _RELAY_KEY,
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_GZIP_REQUEST_BODIES": "0",
        "CLAUDE_CODE_ALWAYS_ENABLE_EFFORT": "1",
    }


def _stage_native(directory: Path, agent: Agent, config: Mapping[str, Any], prompt: str) -> None:
    # Only mutable inputs are copied. Imports and the CLI live on read-only binds.
    secrets = set(_secrets(agent.config) + _secrets(config) + _secrets(os.environ))
    secrets.difference_update({_RELAY_KEY, _ANTHROPIC_RELAY_KEY})
    config = _redact(_plain(config), sorted(secrets, key=len, reverse=True))
    config["llm"]["api_key"] = _ANTHROPIC_RELAY_KEY if agent.name == "claude_code" else _RELAY_KEY
    cls = type(agent)
    if not config.get("source_dir"):
        config["source_dir"] = str(Path(sys.modules[cls.__module__].__file__).resolve().parent)
    agent.snapshot.save(directory / "snapshot.json")
    request = directory / "request.json"
    request.write_text(json.dumps({
        "agent_class": f"{cls.__module__}:{cls.__qualname__}",
        "config_yaml": yaml.safe_dump(config),
        "snapshot_path": f"{_ROOT}/snapshot.json",
        "prompt": prompt,
    }), encoding="utf-8")
    request.chmod(0o600)


def _container_python(python: str, destination: str, *, native: bool = False) -> list[str]:
    package = _RUNTIME if native else _ROOT + "/package"
    return [
        python, "-I", "-B", "-c",
        f"import sys; sys.path.insert(0, {package!r}); "
        "import runpy; runpy.run_module('flashrsi.integrations._container', "
        "run_name='__main__', alter_sys=True)",
        f"{_ROOT}/request.json", destination,
    ]


# This bridge stays in the Agent process and speaks only to the mounted relay.
# No Docker route, extra internet allowance or repository directory is needed.
_RELAY_BRIDGE = """\
import os
import socket
import socketserver
import subprocess
import sys
from threading import Thread

class Bridge(socketserver.StreamRequestHandler):
    rbufsize = 0

    def handle(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect("/opt/flashrsi-relay/relay.sock")
            request = self.rfile.readline()
            if request.startswith((b"POST /responses ", b"POST /responses?")):
                request = request.replace(b"POST /responses", b"POST /v1/responses", 1)
            upstream.sendall(request)

            def pipe(source, destination):
                try:
                    while block := source.recv(65536):
                        destination.sendall(block)
                except OSError:
                    pass
                finally:
                    try:
                        destination.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass

            Thread(target=pipe, args=(self.request, upstream), daemon=True).start()
            pipe(upstream, self.request)

for key in list(os.environ):
    if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
        os.environ.pop(key)
with socketserver.ThreadingTCPServer(("127.0.0.1", 28451), Bridge) as server:
    Thread(target=server.serve_forever, daemon=True).start()
    raise SystemExit(subprocess.call(sys.argv[1:]))
"""


# Route only the local solver API paths through the host relay.
_REVERSE_PROXY = """\
import json
import sys
from functools import partial
from http.server import ThreadingHTTPServer
from runpy import run_path

allowed = {tuple(item) for item in json.loads(sys.argv[1])}
relay = sys.argv[2]
routes = {
    "/anthropic/v1/messages": "/anthropic/v1/messages",
    "/v1/chat/completions": "/v1/chat/completions",
    "/responses": "/v1/responses",
    "/v1/responses": "/v1/responses",
}

class RelayProxy(run_path("/model_proxy.py")["_ProxyHandler"]):
    def do_POST(self):
        path, separator, query = self.path.partition("?")
        if path not in routes:
            self.send_error(404)
            return
        self.path = relay + routes[path] + separator + query
        super().do_POST()

    def do_CONNECT(self):
        self.send_error(403)

    do_GET = do_CONNECT

handler = partial(RelayProxy, allowed=frozenset(allowed))
with ThreadingHTTPServer(("model-egress", 8080), handler) as server:
    server.serve_forever()
"""


def _read_log(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def _archive_trial(
    trial: Trial, directory: str | Path, metadata: Mapping[str, Any], secrets: Sequence[str],
) -> dict[str, Any]:
    """Keep native outputs before temporary cleanup, without staging credentials."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    saved = Path(mkdtemp(prefix="trial-", dir=directory))
    record = {"directory": str(saved), "files": []}
    try:
        # Never copy the trial's config/lock or our payload: those can contain
        # solver and verifier credentials. Harbor already collected these outputs.
        for source in (
            trial.paths.artifacts_dir,
            trial.paths.verifier_dir,
            trial.paths.agent_dir,
            trial.paths.result_path,
        ):
            if source.is_symlink() or source.is_junction():
                raise ValueError(f"Refusing linked artifact: {source}")
            if not source.exists():
                continue
            paths = sorted(source.rglob("*")) if source.is_dir() else [source]
            for path in paths:
                if path.is_symlink() or path.is_junction():
                    raise ValueError(f"Refusing linked artifact: {path}")
                if not path.is_file():
                    continue
                relative = path.relative_to(trial.paths.trial_dir)
                destination = saved / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = path.read_bytes()
                if path.name == "model.patch" and any(
                    secret.encode() in data for secret in secrets
                ):
                    raise ValueError(
                        "Refusing to archive a patch containing configured credentials"
                    )
                # Deliverables (including patches) must remain reproducible.
                # Redact native logs, not the files that the verifier evaluated.
                if source != trial.paths.artifacts_dir and path.name != "model.patch":
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                    else:
                        if "\0" not in text:
                            data = _redact(text, secrets).encode("utf-8")
                destination.write_bytes(data)
                record["files"].append(relative.as_posix())
    except Exception as exc:
        record["error"] = str(exc)
        raise RuntimeError(f"Could not archive outputs to {saved}: {exc}") from exc
    finally:
        # Also identify partial archives when a trial fails or is cancelled.
        pending = saved / ".manifest.json"
        pending.write_text(
            json.dumps(_redact({**metadata, **record}, secrets), ensure_ascii=False),
            encoding="utf-8",
        )
        pending.replace(saved / "manifest.json")
    return record


def _trial_artifacts(
    trial: Trial, options: Mapping[str, Any], event: dict[str, Any], secrets: Sequence[str], *, archive: Callable,
) -> None:
    if options.get("trials_dir"):
        directory = trial.paths.trial_dir.resolve()
        event["trial_directory"] = str(directory)
        event["files"] = {
            str(file.relative_to(directory)): hashlib.sha256(file.read_bytes()).hexdigest()
            for folder in (trial.paths.artifacts_dir, trial.paths.verifier_dir)
            for file in sorted(folder.rglob("*"))
            if file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(directory)
        }
    event["artifacts"] = archive(
        trial, options.get("artifacts_dir", "runs/artifacts/harbor"), event, secrets,
    )


async def _cleanup_owned(
    containers: Sequence[str], network: str | None, grace: float, *, docker: Callable = _docker,
) -> None:
    async def remove(*args, missing):
        result = await docker(*args, timeout=grace, check=False)
        if result.returncode and not (
            missing in result.stderr
            or (args[0] == "network" and f"network {network} not found" in result.stderr)
        ):
            raise InfraError(f"Docker {args[0]} cleanup failed: {result.stderr[-2000:]}")

    async def cleanup():
        async with asyncio.timeout(grace):
            removed = await asyncio.gather(
                *(remove("rm", "--force", name, missing="No such container") for name in containers),
                return_exceptions=True,
            )
            if network is not None:
                await remove("network", "rm", network, missing="No such network")
            for result in removed:
                if isinstance(result, BaseException):
                    raise result

    pending = asyncio.create_task(cleanup())
    try:
        await asyncio.shield(pending)
    except asyncio.CancelledError:
        await asyncio.gather(pending, return_exceptions=True)
        raise



def _stage_runtime(
    root: Path, resources: ExitStack, agent: Agent, config: Mapping[str, Any],
) -> tuple[str, list[Path]]:
    from flashrsi.integrations.sandbox_runtime import _anthropic_options, relay

    python, mounts = _host_runtime()
    socket_dir = root / "relay"
    socket_dir.mkdir(mode=0o700)
    relay_path = socket_dir / "relay.sock"
    relay_options = {"unix": True}
    if config.get("request_timeout") is not None:
        relay_options["request_timeout"] = config["request_timeout"]
    server = resources.enter_context(relay(str(relay_path), config["llm"]["model_name"], **relay_options))
    server.anthropic_options = _anthropic_options(agent, config)
    # The local socket works even when the trial disables networking.
    socket_dir.chmod(0o711)
    relay_path.chmod(0o666)
    return python, [_runtime_compose(root, {**mounts, _RELAY_MOUNT: socket_dir})]


def _runtime_compose(root: Path, mounts: Mapping[str, Path]) -> Path:
    path = root / "flashrsi-runtime.json"
    volumes = [
        {"type": "bind", "source": str(source), "target": target,
         "read_only": True, "bind": {"create_host_path": False}}
        for target, source in mounts.items()
    ]
    path.write_text(json.dumps({"services": {"main": {"volumes": volumes}}}))
    return path



def _read_trial_result(trial: Trial, result: TrialResult, trajectory: Trajectory, event: dict[str, Any], snapshot_id: str) -> Trajectory:
    output = trial.paths.agent_dir / "flashrsi.json"
    if output.is_file():
        values = json.loads(output.read_text())
        values.pop("task", None)
        trajectory = Trajectory(**values)
        if not any(
            message.get("type") == "RunConfig"
            and message.get("snapshot_id") == snapshot_id
            for message in trajectory.messages
        ):
            raise ValueError(
                "Container result does not identify the requested Snapshot"
            )
    elif result.exception_info is None:
        raise ValueError("Harbor agent ended without a FlashRSI trajectory")
    event["trial_id"] = str(result.id)
    event["rewards"] = (
        result.verifier_result.rewards
        if result.verifier_result is not None
        else None
    )
    event["stdout"] = _read_log(trial.paths.test_stdout_path)
    event["stderr"] = _read_log(trial.paths.test_stderr_path)
    if result.exception_info is not None:
        event["exception_type"] = result.exception_info.exception_type
        error = (
            result.exception_info.exception_message
            or result.exception_info.exception_type
        )
        event["error"] = error
        reason = trajectory_infra_reason(trajectory) or infra_reason(error)
        if reason is not None:
            event["infra_error"] = reason
        elif event["exception_type"] != "AgentTimeoutError" and not (
            event["exception_type"] == "NonZeroAgentExitCodeError"
            and trajectory.status == "error"
        ):
            event["infra_error"] = error
        trajectory = replace(
            trajectory,
            status="error",
            error_message=trajectory.error_message or error,
        )
    elif trajectory.status == "finished" and event["rewards"] is None:
        raise ValueError("Harbor verifier ended without rewards")
    elif reason := trajectory_infra_reason(trajectory):
        event["infra_error"] = reason
    return trajectory


def _redacted_trajectory(trajectory: Trajectory, secrets: list[str], event: dict[str, Any] | None = None) -> Trajectory:
    messages = trajectory.messages if event is None else (*trajectory.messages, event)
    return replace(
        trajectory,
        output=_redact(trajectory.output, secrets),
        messages=_redact(messages, secrets),
        usage=_redact(trajectory.usage, secrets),
        error_message=_redact(trajectory.error_message, secrets),
    )



_PROXY_ENV = {
    name: value
    for key, value in (
        ("HTTP_PROXY", "http://model-egress:8080"),
        ("HTTPS_PROXY", "http://model-egress:8080"),
        ("ALL_PROXY", "http://model-egress:8080"),
        ("NO_PROXY", ""),
    )
    for name in (key, key.lower())
}
_CLEAR_PROXIES = tuple(
    f"--env={name}="
    for name in (*_PROXY_ENV, "FTP_PROXY", "ftp_proxy")
)
_KEEPALIVE = "while :; do sleep 3600; done"


async def _reset(name: str, base: str, timeout: float, *, docker: Callable) -> None:
    # Official images can retain a later repair commit. Keep the baseline and
    # ancestors, and remove later refs, reflogs, objects and setup files.
    await docker(
        "exec", name, "/bin/bash", "-c",
        "set -e\ncd /testbed\n"
        f"git reset --hard {shlex.quote(base)}\n"
        "git clean -fd\ngit checkout --detach\n"
        "git for-each-ref --format='delete %(refname)' | git update-ref --stdin\n"
        "git config --local --remove-section remote.origin || true\n"
        'rm -f -- "$(git rev-parse --git-path ORIG_HEAD)" "$(git rev-parse --git-path FETCH_HEAD)"\n'
        "git reflog expire --expire=now --all\ngit gc --prune=now\n"
        "rm -f -- /root/setup_repo.sh /root/setup_env.sh /root/install.sh "
        "/eval.sh /patch.diff /test.patch /tmp/patch.diff\n",
        timeout=timeout,
    )


async def _create_task_container(
    name: str, image: str, network: str, resources: Sequence[str], timeout: float, *,
    mounts: Mapping[str, Path] | None = None, docker: Callable = _docker,
) -> None:
    binds = tuple(
        f"--mount=type=bind,src={source},dst={target},readonly"
        for target, source in (mounts or {}).items()
    )
    await docker(
        "create", "--pull=never", "--name", name, "--init", "--user", "root",
        f"--network={network}", "--cap-drop=NET_RAW", "--security-opt=no-new-privileges",
        *resources, *binds, *_CLEAR_PROXIES, "--workdir", "/testbed", "--entrypoint", "/bin/sh",
        "--", image, "-c", _KEEPALIVE, timeout=timeout,
    )
    await docker("start", name, timeout=timeout)


async def _prepare_checkout(name: str, base: str, timeout: float, *, docker: Callable = _docker) -> None:
    await docker(
        "exec", name, "git", "config", "--global", "--add", "safe.directory",
        "/testbed", timeout=timeout,
    )
    await _reset(name, base, timeout, docker=docker)


async def _check_solver_network(name: str, network: str, timeout: float, *, docker: Callable = _docker) -> None:
    async def attached():
        result = await docker(
            "inspect", "--format", "{{json .NetworkSettings.Networks}}", name, timeout=timeout
        )
        return set(json.loads(result.stdout))

    networks = await attached()
    if "bridge" in networks:
        await docker("network", "disconnect", "bridge", name, timeout=timeout)
        networks = await attached()
    internal = await docker("network", "inspect", "--format", "{{.Internal}}", network, timeout=timeout)
    if networks != {network} or internal.stdout.strip() != "true":
        raise InfraError("Solver must be attached only to its internal model network")


async def _check_solver_routes(name: str, timeout: float, *, python: str, docker: Callable = _docker) -> None:
    await docker(
        "exec", name, python, "-I", "-B", "-c",
        "from pathlib import Path, PurePosixPath\n"
        "import socket, time\n"
        "v4 = Path('/proc/net/route').read_text().splitlines()[1:]\n"
        "v6_file = Path('/proc/net/ipv6_route')\n"
        "v6 = v6_file.read_text().splitlines() if v6_file.exists() else []\n"
        "if any(r.split()[1] == '00000000' for r in v4) or any(\n"
        "    r.split()[:2] == ['0'*32, '00'] and r.split()[-1] != 'lo' for r in v6\n"
        "): raise SystemExit('Solver has a default network route')\n"
        "for _ in range(30):\n"
        " try:\n"
        "  with socket.create_connection(('model-egress', 8080), timeout=1): break\n"
        " except OSError: time.sleep(.2)\n"
        "else: raise SystemExit('Model proxy did not start')\n",
        timeout=timeout,
    )



async def _host_process(*args: str, cwd: Path, timeout: float) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"},
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
    return process.returncode, stdout, stderr


async def _patch_paths(
    path: Path, timeout: float,
) -> set[tuple[str, ...]] | None:
    paths = set()
    # Numstat names the destination of a rename. Reverse mode also obtains
    # its source, without interpreting quoted paths or diff content ourselves.
    for flags in ((), ("--reverse",)):
        code, output, stderr = await _host_process(
            "git", "apply", *flags, "--numstat", "-z", "--", path.name,
            cwd=path.parent, timeout=timeout,
        )
        if code:
            if any(
                message in stderr
                for message in (b"corrupt patch", b"patch fragment", b"No valid patches", b"git diff header")
            ):
                return None
            raise InfraError(f"Git patch inspection exited with {code}")
        for item in output.split(b"\0")[:-1]:
            raw = item.split(b"\t", 2)[2]
            path_parts = PurePosixPath(os.fsdecode(raw)).parts
            paths.add(tuple(part.casefold() for part in path_parts))
    return paths
