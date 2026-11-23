"""Collect SWE patches in an isolated solver and run the official evaluator.

Solver setup and execution have independent clocks and verifier resources.
FLASHRSI_SWE_SOLVER_* and FLASHRSI_SWE_VERIFIER_* supply host defaults.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import shlex
import shutil
import sys
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from flashrsi._records import _config_record, _redact, _secrets
from flashrsi.benchmarks.harbor_suite import _protected_path as _protected_path
from flashrsi.benchmarks.swe_bench import SWEBench, _checksum
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.core.infra import trajectory_infra_reason
from flashrsi.integrations import _model_egress, _swe_grade
from flashrsi.integrations._container import _agent_class
from flashrsi.integrations._container_payload import _ROOT, _setup_script, _stage
from flashrsi.integrations._docker import _docker
from flashrsi.integrations.harbor_suite import (
    _CLEAR_PROXIES,
    _CONTAINER_SHELLS,
    _REVERSE_PROXY,
    _RUNTIME,
    _check_solver_network,
    _check_solver_routes,
    _cleanup_owned,
    _container_python,
    _create_task_container,
    _host_process,
    _host_runtime,
    _patch_paths,
    _prepare_checkout,
    _redacted_trajectory,
    _relay_shell_env,
    _shell_turns,
    _stage_native,
)
from flashrsi.integrations.harbor_suite import (
    _PROXY_ENV as _PROXY_ENV,
)

_SIDECAR = "http://model-egress:8080"
_PROXY_IMAGE = "python:3.12-slim"
_TEST_LOCATION = "/tmp/flashrsi-evaluation"
_MAX_TEST_OUTPUT = 32 * 1024 * 1024


@asynccontextmanager
async def _solver_relay(agent, native):
    from flashrsi.integrations.sandbox_runtime import RELAY, gateway, relay

    active = RELAY.get()
    if not native or active is not None:
        yield active
        return
    host = await gateway()
    with relay((host, 0), agent.config["llm"]["model_name"], unix=False) as server:
        token = RELAY.set(server)
        try:
            yield server
        finally:
            RELAY.reset(token)


def _seconds(options, key, default, env, *, section):
    value = options.get(key)
    if value is None:
        raw = os.getenv(env)
        try:
            value = default if raw is None else float(raw)
        except ValueError:
            value = None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{section}.{key} ({env}) must be a positive finite number")
    return value


def _resource_flags(options, phase):
    prefix = f"FLASHRSI_SWE_{phase.upper()}_"
    flags = []
    for key, flag, default in (
        ("cpus", "cpus", 2),
        ("memory", "memory", "4g"),
        ("pids_limit", "pids-limit", 512),
        ("shm_size", "shm-size", "64m"),
    ):
        value = options.get(key, os.getenv(prefix + key.upper(), default))
        if isinstance(value, bool) or not isinstance(value, (int, float, str)) or not str(value).strip():
            raise ValueError(f"Invalid SWE {phase} resource {key}")
        flags.append(f"--{flag}={value}")
    # The Docker daemon must support writable-layer quotas.
    if storage := options.get("storage", os.getenv(prefix + "STORAGE", "20g")):
        flags.append(f"--storage-opt=size={storage}")
    return tuple(flags)


def _evaluation_options(config):
    clocks = {
        key: _seconds(config, key, default, f"FLASHRSI_SWE_VERIFIER_{env}", section="benchmark")
        for key, default, env in (
            ("setup_timeout", 600, "SETUP_TIMEOUT"),
            ("evaluation_timeout", 1800, "TEST_TIMEOUT"),
            ("summary_timeout", 60, "SUMMARY_TIMEOUT"),
            ("cleanup_grace", 60, "CLEANUP_GRACE"),
        )
    }
    return {
        **clocks,
        "resources": _resource_flags(config.get("verifier_resources", {}), "verifier"),
    }


def bind(agent: Agent, benchmark: SWEBench | None = None) -> None:
    """Bind full evaluation with a Benchmark, or retain explicit patch collection."""
    agent._check_container()
    agent.snapshot
    suite_track = getattr(benchmark, "_track_split", None) is not None
    if not suite_track:
        _model_egress._origins(agent.config)
    cls = type(agent)
    if _agent_class(f"{cls.__module__}:{cls.__qualname__}") is not cls:
        raise ValueError("SWE execution requires an importable Agent class")
    if shutil.which("docker") is None:
        raise InfraError("SWE execution requires Docker CLI and a Linux container daemon")
    if benchmark is None:
        agent._env_runner = _solve
        return
    try:
        _swe_grade.check()
    except ValueError as exc:
        raise InfraError(str(exc)) from None
    options = _evaluation_options(benchmark.config)

    async def run(target: Agent, task: Task) -> Trajectory:
        if benchmark._tasks_by_id.get(task.task_id) != task:
            raise ValueError("SWE evaluation requires a loaded benchmark Task")
        row = benchmark._rows_by_id[task.task_id]
        if _checksum(row) != task.metadata["task_checksum"]:
            raise InfraError(f"{task.task_id}: SWE data changed after loading")
        if row.get("image_assets"):
            raise InfraError(f"{task.task_id}: this SWE integration does not support image_assets")
        result = await _solve(target, task, suite_track=suite_track)
        if result.status == "finished":
            result = await _evaluate(target, task, result, row, options)
        return result

    agent._env_runner = run


def _infra(task, phase, exc, secrets):
    detail = str(exc) or ("deadline exceeded" if isinstance(exc, TimeoutError) else type(exc).__name__)
    return InfraError(_redact(f"{task.task_id}: SWE {phase}: {detail}", secrets))


async def _candidate_exec(name, command, *, timeout, env=None):
    """Separate Docker transport failures from the command's own exit status."""
    marker = "\\nflashrsi-exec-status=%s\\n"
    wrapper = f"(\n{command}\n)\nstatus=$?\nprintf '{marker}' \"$status\""
    result = await _docker(
        "exec", *(f"--env={key}={value}" for key, value in (env or {}).items()),
        name, "/bin/bash", "-c", wrapper, timeout=timeout,
    )
    _, separator, status = result.stdout.rstrip("\n").rpartition("\nflashrsi-exec-status=")
    if not separator or not status.isdecimal():
        raise InfraError("Docker exec did not return the command status")
    status = int(status)
    if status in (125, 126, 127):
        raise InfraError(f"SWE container command exited with {status}: {result.stderr[-2000:]}")
    return status


async def _start_proxy(proxy, network, origins, timeout, *, relay_url=None):
    from flashrsi.integrations.sandbox_runtime import admit

    source = Path(_model_egress.__file__).with_name("_model_proxy.py")
    arguments = ["-u", "-c", source.read_text(encoding="utf-8"), json.dumps(origins)]
    binds = ()
    if relay_url is not None:
        binds = (f"--mount=type=bind,src={source},dst=/model_proxy.py,readonly",)
        arguments = ["-u", "-c", _REVERSE_PROXY, json.dumps(origins), relay_url]
    await _docker(
        "create", "--pull=never", "--name", proxy, "--network=bridge",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
        "--sysctl", "net.ipv4.ip_forward=0", "--sysctl", "net.ipv6.conf.all.forwarding=0",
        "--pids-limit=128", "--memory=128m", *_CLEAR_PROXIES, "--entrypoint", "python3",
        *binds, "--", _PROXY_IMAGE, *arguments, timeout=timeout,
    )
    await _docker("network", "connect", "--alias", "model-egress", network, proxy, timeout=timeout)
    await _docker("start", proxy, timeout=timeout)
    await admit(proxy)


async def _run_solver(name, timeout, *, python=f"{_ROOT}/venv/bin/python", native=False):
    command = shlex.join(_container_python(python, f"{_ROOT}/trajectory.json", native=native))
    env = _relay_shell_env(_SIDECAR) if native else _PROXY_ENV
    return await _candidate_exec(
        name, "set -e\nsource /opt/miniconda3/bin/activate testbed\n"
        + (f"export PATH={_RUNTIME}/bin:$PATH\n" if native else "") + "exec " + command,
        timeout=timeout, env=env,
    )


async def _collect(name, task, record, image, timeout, status, secrets):
    # Read in the container to avoid copying an Agent-controlled file or link.
    raw = await _docker("exec", name, "cat", f"{_ROOT}/trajectory.json", timeout=timeout)
    values = json.loads(raw.stdout)
    values.pop("task", None)
    trajectory = Trajectory(**values)
    if reason := trajectory_infra_reason(trajectory):
        raise InfraError(reason)
    if status:
        error = trajectory.error_message or "missing Agent error detail"
        if trajectory.status == "error" and error.startswith((
            "Pi timed out", "Claude Code timed out", "Codex timed out", "MiniAgent timed out", "OpenClaw timed out",
        )):
            return trajectory
        raise InfraError(f"SWE Agent exited with {status}: {error}")

    await _docker("exec", name, "git", "-C", "/testbed", "add", "-N", "--", ".", timeout=timeout)
    patch = (
        await _docker(
            "exec", name, "git", "--no-replace-objects", "-c", "core.quotePath=true",
            "-C", "/testbed", "diff",
            "--no-ext-diff", "--no-textconv", "--binary", "--full-index", "--no-renames",
            "--src-prefix=a/", "--dst-prefix=b/", task.metadata["base_commit"], "--", ".",
            timeout=timeout,
        )
    ).stdout
    if _redact(patch, secrets) != patch:
        raise InfraError("SWE patch contains configured credentials")
    event = {
        "type": "SWEBenchPatch",
        "task_id": task.task_id,
        "task_checksum": task.metadata["task_checksum"],
        "snapshot_id": record["snapshot_id"],
        "base_commit": task.metadata["base_commit"],
        "image_id": image,
        "model_patch": patch,
    }
    return replace(
        trajectory,
        messages=(
            record,
            *(m for m in trajectory.messages
              if m.get("type") not in ("RunConfig", "SWEBenchPatch", "SWEBenchVerifier")),
            event,
        ),
    )


async def _solve(agent: Agent, task: Task, *, suite_track: bool = False) -> Trajectory:
    """Run an isolated Agent with an offline, trusted setup phase."""
    native = suite_track and getattr(agent, "name", None) in _CONTAINER_SHELLS
    async with _solver_relay(agent, native) as server:
        return await _solve_in_container(agent, task, suite_track=suite_track, native=native, server=server)


@dataclass(frozen=True, slots=True)
class _SolverPlan:
    config: dict[str, Any]
    native: bool
    python: str
    mounts: dict[str, Path]
    origins: list[tuple[str, str, int]]
    relay_url: str | None
    timeout: float
    setup: float
    grace: float
    resources: tuple[str, ...]


@dataclass(slots=True)
class _SolverContainer:
    name: str
    network: str
    proxy: str
    phase: str = "solver setup"
    network_started: bool = False


def _plan_solver(agent: Agent, *, suite_track: bool, native: bool, server: Any) -> _SolverPlan:
    from flashrsi.integrations.sandbox_runtime import route_config

    config = copy.deepcopy(agent.config)
    if native:
        # The managed runtime already contains these dependencies. The relay
        # can configure transport without staging or extending an offline kit.
        config.pop("container_runtime_kit", None)
    config = route_config(agent, config)
    origins = _model_egress._origins(config)
    python, mounts = f"{_ROOT}/venv/bin/python", {}
    relay_url = None
    if native:
        python, mounts = _host_runtime()
        host, port = server.server_address
        relay_url = f"http://{host}:{port}"
        origins = [("http", host, port)]
        config["_suite_model_relay"] = True
        llm = config["llm"]
        llm["base_url"] = _SIDECAR + ("/anthropic" if agent.name == "claude_code" else "/v1")
        llm.pop("side_base_url", None)
        llm.pop("side_api_key", None)
    _shell_turns(agent, config)
    options = config.get("swe_bench", {})
    legacy_timeout = 3600 if suite_track else config.get("timeout", 3600)
    timeout = _seconds(options, "timeout", legacy_timeout, "FLASHRSI_SWE_SOLVER_TIMEOUT", section="agent.swe_bench")
    setup = _seconds(options, "setup_timeout", 600, "FLASHRSI_SWE_SOLVER_SETUP_TIMEOUT", section="agent.swe_bench")
    grace = _seconds(options, "cleanup_grace", 60, "FLASHRSI_SWE_SOLVER_CLEANUP_GRACE", section="agent.swe_bench")
    resources = _resource_flags(options.get("resources", {}), "solver")
    config["timeout"] = timeout

    return _SolverPlan(config, native, python, mounts, origins, relay_url, timeout, setup, grace, resources)


async def _setup_solver(container: _SolverContainer, task: Task, plan: _SolverPlan, payload: Path) -> str:
    name, network, proxy = container.name, container.network, container.proxy
    setup = plan.setup
    python, mounts, native = plan.python, plan.mounts, plan.native
    resources, origins, relay_url = plan.resources, plan.origins, plan.relay_url
    await _docker(
        "network", "create", "--internal",
        "--opt=com.docker.network.bridge.gateway_mode_ipv4=isolated",
        "--opt=com.docker.network.bridge.gateway_mode_ipv6=isolated",
        network, timeout=setup,
    )
    container.phase = f"solver creation (stage image {task.env} on the Docker daemon)"
    await _create_task_container(name, task.env, network, resources, setup, mounts=mounts, docker=_docker)
    image = (await _docker("inspect", "--format", "{{.Image}}", name, timeout=setup)).stdout.strip()
    if not image:
        raise InfraError("Docker did not report the Solver image ID")
    container.phase = "solver network isolation"
    await _check_solver_network(name, network, setup, docker=_docker)
    container.phase = "solver setup"
    await _prepare_checkout(name, task.metadata["base_commit"], setup, docker=_docker)
    await _docker(
        "exec", name, "/bin/bash", "-c",
        "set -e\nsource /opt/miniconda3/bin/activate testbed",
        timeout=setup,
    )
    await _docker("exec", name, "mkdir", "-p", _ROOT, timeout=setup)
    await _docker("cp", str(payload) + os.sep + ".", f"{name}:{_ROOT}", timeout=setup)
    if native:
        await _docker("exec", name, python, "-I", "-B", "--version", timeout=setup)
    elif (payload / "kit/python312.tar.gz").is_file():
        await _docker("exec", name, "/bin/sh", "-c", _setup_script(), timeout=setup)
    else:
        await _docker(
            "exec", name, "/bin/sh", "-c",
            f"test -x {shlex.quote(_ROOT + '/venv/bin/python')} || "
            " { echo 'Stage a compatible agent.container_runtime_kit or preinstall "
            "the FlashRSI runtime in the SWE image' >&2; exit 1; }",
            timeout=setup,
        )
    container.phase = f"model proxy setup (stage image {_PROXY_IMAGE} on the Docker daemon)"
    await _start_proxy(proxy, network, origins, setup, relay_url=relay_url)
    await _check_solver_network(name, network, setup, docker=_docker)
    await _check_solver_routes(name, setup, python=python, docker=_docker)
    return image


async def _solve_in_container(
    agent: Agent, task: Task, *, suite_track: bool, native: bool, server: Any,
) -> Trajectory:
    plan = _plan_solver(agent, suite_track=suite_track, native=native, server=server)
    config, setup, timeout, grace, python = plan.config, plan.setup, plan.timeout, plan.grace, plan.python
    record = {"type": "RunConfig", "config": _config_record(config), "snapshot_id": agent.snapshot.id}
    secrets = sorted(
        set(_secrets(agent.config) + _secrets(config) + _secrets(os.environ)),
        key=len, reverse=True,
    )
    name = f"flashrsi-swe-{uuid4().hex}"
    container = _SolverContainer(name, name + "-internal", name + "-proxy")
    with TemporaryDirectory(prefix="flashrsi-swe-") as temporary:
        payload = Path(temporary) / "payload"
        payload.mkdir(mode=0o700)
        try:
            async with asyncio.timeout(setup):
                stage = _stage_native if native else _stage
                stage(payload, agent, config, task.question)
                container.network_started = True
                image = await _setup_solver(container, task, plan, payload)
            container.phase = "solver execution"
            try:
                status = await _run_solver(name, timeout, python=python, native=native)
            except TimeoutError:
                # Confirm the daemon and Solver are still reachable before scoring
                # this as a candidate episode timeout.
                await _docker("exec", name, "true", timeout=grace)
                trajectory = Trajectory(
                    status="error", messages=(record,),
                    error_message=f"SWE solver episode timed out after {timeout:g}s",
                )
            else:
                container.phase = "solver result collection"
                async with asyncio.timeout(grace):
                    trajectory = await _collect(name, task, record, image, grace, status, secrets)
        except Exception as exc:
            raise _infra(task, container.phase, exc, secrets) from None
        finally:
            if container.network_started:
                try:
                    await _cleanup_owned((name, container.proxy), container.network, grace, docker=_docker)
                except Exception as exc:
                    raise _infra(task, "solver cleanup", exc, secrets) from None
    return _redacted_trajectory(trajectory, secrets)


async def _official(action, root, timeout):
    code, stdout, stderr = await _host_process(
        sys.executable, "-I", "-X", "utf8", str(Path(_swe_grade.__file__)),
        action, str(root), cwd=root, timeout=timeout,
    )
    if code:
        # The private helper emits only exception types.
        raise InfraError(stderr.decode(errors="replace").strip() or "Official SWE evaluator failed")
    return json.loads(stdout)


def _verdict_event(agent, task, trajectory, image, *, official=False, **verdict):
    return replace(
        trajectory,
        messages=(*trajectory.messages, {
            "type": "SWEBenchVerifier",
            "task_id": task.task_id,
            "task_checksum": task.metadata["task_checksum"],
            "snapshot_id": agent.snapshot.id,
            "image_id": image,
            "evaluator_revision": _swe_grade.REVISION,
            "official_evaluation_performed": official,
            **verdict,
        }),
    )


async def _apply_candidate(name, commands, timeout):
    applied = False
    # Inspect and apply with the same Git patch parser. Alternate patch
    # programs can interpret headers differently and bypass the path check.
    git_commands = [command for command in commands if command.startswith("git apply ")]
    if not git_commands:
        raise InfraError("Official evaluator supplied no Git apply command")
    for index, command in enumerate(git_commands):
        if index:
            await _docker(
                "exec", name, "/bin/bash", "-c",
                "cd /testbed && git reset --hard HEAD && git clean -fd",
                timeout=timeout,
            )
        status = await _candidate_exec(
            name, f"{command} {_TEST_LOCATION}/model.patch", timeout=timeout,
        )
        if status == 0:
            applied = True
            break
    return applied


async def _prepare_evaluation(
    root: Path, patch: str, row: Mapping[str, Any], setup: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not patch.strip():
        return None, {"outcome": "empty_candidate_patch"}
    model_patch = root / "model.patch"
    model_patch.write_text(patch, encoding="utf-8", newline="\n")
    paths = await _patch_paths(model_patch, setup)
    if not paths:
        return None, {"outcome": "invalid_candidate_patch"}
    test_patch = root / "test.patch"
    test_patch.write_text(row["test_patch"], encoding="utf-8", newline="\n")
    trusted = await _patch_paths(test_patch, setup)
    if not trusted:
        raise InfraError("Cannot read protected paths from the trusted SWE test patch")
    if any(_protected_path(path, trusted) for path in paths):
        return None, {
            "outcome": "rejected_candidate_patch",
            "rejected_reason": "modified_trusted_test_or_harness_files",
        }

    fields = (
        "instance_id", "image", "repo", "version", "eval_script",
        "log_parser", "eval_type", "FAIL_TO_PASS", "PASS_TO_PASS",
    )
    (root / "request.json").write_text(
        json.dumps({"row": {key: row[key] for key in fields}}), encoding="utf-8",
    )
    return await _official("prepare", root, setup), {}


async def _start_verifier(name: str, image: str, task: Task, root: Path, options: Mapping[str, Any]) -> None:
    setup = options["setup_timeout"]
    await _create_task_container(name, image, "none", options["resources"], setup, docker=_docker)
    await _prepare_checkout(name, task.metadata["base_commit"], setup, docker=_docker)
    inputs = root / "inputs"
    inputs.mkdir()
    for filename in ("eval.sh", "model.patch"):
        shutil.copyfile(root / filename, inputs / filename)
    await _docker("exec", name, "mkdir", "-p", _TEST_LOCATION, timeout=setup)
    await _docker(
        "cp", str(inputs) + os.sep + ".", f"{name}:{_TEST_LOCATION}", timeout=setup,
    )


async def _summarize_verdict(name: str, root: Path, timeout: float) -> dict[str, Any]:
    async with asyncio.timeout(timeout):
        size = await _docker("exec", name, "wc", "-c", f"{_TEST_LOCATION}/test_output.txt", timeout=timeout)
        if int(size.stdout.split()[0]) > _MAX_TEST_OUTPUT:
            raise InfraError("SWE test output exceeds 32 MiB")
        output = await _docker(
            "exec", name, "cat", f"{_TEST_LOCATION}/test_output.txt",
            timeout=timeout, binary=True,
        )
        (root / "test_output.txt").write_text(output.stdout.decode("utf-8", errors="replace"), encoding="utf-8")
        verdict = await _official("grade", root, timeout)
        if (
            type(verdict.get("resolved")) is not bool
            or verdict.get("patch_successfully_applied") is not True
        ):
            raise InfraError("Invalid official SWE evaluator verdict")
        return verdict


async def _evaluate(
    agent: Agent, task: Task, trajectory: Trajectory, row: Mapping[str, Any], options: Mapping[str, Any],
) -> Trajectory:
    setup, timeout = options["setup_timeout"], options["evaluation_timeout"]
    summary, grace = options["summary_timeout"], options["cleanup_grace"]
    event = trajectory.messages[-1]
    patch, image = event["model_patch"], event["image_id"]
    name = f"flashrsi-swe-grade-{uuid4().hex}"
    secrets = sorted(set(_secrets(agent.config) + _secrets(os.environ)), key=len, reverse=True)
    with TemporaryDirectory(prefix="flashrsi-swe-grade-") as temporary:
        root = Path(temporary)
        phase = "verifier setup"
        creating = False

        def unresolved(reason: str, **metadata: Any) -> Trajectory:
            return _verdict_event(
                agent, task, trajectory, image, resolved=False,
                patch_successfully_applied=False, outcome=reason, **metadata,
            )

        try:
            async with asyncio.timeout(setup):
                prepared, rejection = await _prepare_evaluation(root, patch, row, setup)
                if prepared is None:
                    return _verdict_event(
                        agent, task, trajectory, image, resolved=False, patch_successfully_applied=False,
                        **rejection,
                    )
                creating = True
                await _start_verifier(name, image, task, root, options)
                if not await _apply_candidate(name, prepared["apply_commands"], setup):
                    return unresolved("candidate_patch_did_not_apply")

            phase = "verifier test execution"
            try:
                await _candidate_exec(
                    name, f"ulimit -f 32768; exec /bin/bash {_TEST_LOCATION}/eval.sh "
                    f"> {_TEST_LOCATION}/test_output.txt 2>&1",
                    timeout=timeout,
                )
            except TimeoutError:
                await _docker("exec", name, "true", timeout=grace)
                return _verdict_event(
                    agent, task, trajectory, image, official=True,
                    resolved=False, patch_successfully_applied=True,
                    outcome="official_test_timeout",
                )

            phase = "verifier summary"
            verdict = await _summarize_verdict(name, root, summary)
            return _verdict_event(
                agent, task, trajectory, image, official=True,
                outcome="tests_passed" if verdict["resolved"] else "tests_failed", **verdict,
            )
        except Exception as exc:
            raise _infra(task, phase, exc, secrets) from None
        finally:
            if creating:
                try:
                    await _cleanup_owned((name,), None, grace, docker=_docker)
                except Exception as exc:
                    raise _infra(task, "verifier cleanup", exc, secrets) from None
