"""Run a FlashRSI snapshot inside a native Harbor trial.

Harbor execution requires the optional Harbor extra and Docker Compose.
The host controls trial setup and collects verifier results and artifacts.
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
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

if sys.version_info < (3, 12):
    raise ImportError("Harbor execution requires Python 3.12 or newer")

try:
    from harbor.agents.base import BaseAgent
    from harbor.agents.installed.base import NonZeroAgentExitCodeError
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.task import Task as HarborTask
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_task_verifier_mode,
    )
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )
    from harbor.models.trial.paths import EnvironmentPaths
    from harbor.models.trial.result import TrialResult
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.trial import Trial
except ImportError as exc:
    raise ImportError("Install flashrsi-lite[harbor] for environment tasks") from exc

from flashrsi._records import _config_record, _redact, _secrets
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.core.infra import infra_reason, trajectory_infra_reason
from flashrsi.integrations._container import _agent_class
from flashrsi.integrations._container_payload import _plain, _setup_script
from flashrsi.integrations._container_payload import _stage as _stage_payload
from flashrsi.integrations._docker import _docker
from flashrsi.integrations._harbor_remote import _RemoteDockerEnvironment
from flashrsi.integrations.harbor_suite import (
    _CONTAINER_SHELLS,
    _RELAY_BRIDGE,
    _RUNTIME,
    _archive_trial,
    _container_python,
    _read_trial_result,
    _redacted_trajectory,
    _relay_shell_env,
    _shell_turns,
    _stage_native,
    _stage_runtime,
    _trial_artifacts,
)
from flashrsi.integrations.harbor_suite import (
    _RELAY_MOUNT as _RELAY_MOUNT,
)

_ROOT = "/opt/flashrsi-lite"
_RELAY_ENDPOINT = "http://127.0.0.1:28451"
_CLEANUP_FAILURES = ContextVar("harbor_cleanup_failures", default=None)


def bind(agent: Agent) -> None:
    """Connect an Agent to Harbor; CLI runs do this in the runner.

    The adapter must be importable in the container. Built-ins travel with the
    current FlashRSI source; external adapters must include their distribution
    in _container_dependencies. No task or model is executed here.
    """
    agent._check_container()
    from flashrsi.integrations._terminal_integrity import validate

    validate(agent.config.get("harbor", {}).get("integrity_review"))
    agent.snapshot  # Container execution requires a versioned harness.
    cls = type(agent)
    if _agent_class(f"{cls.__module__}:{cls.__qualname__}") is not cls:
        raise ValueError("Container execution requires an importable Agent class")
    agent._env_runner = _run


async def _load_task(task: Task, strict: bool | None) -> tuple[HarborTask, str]:
    try:
        native = HarborTask(Path(task.env))
    except (OSError, ValueError) as exc:
        raise InfraError(
            f"{task.task_id}: cannot load Harbor assets at {task.env}: {exc}. "
            "Stage the reviewed Harbor task directory and task.toml."
        ) from None
    if strict:
        await _preflight_image(task, native)
    if native.config.steps or native.config.environment.os.value != "linux":
        raise ValueError("This integration supports single-step Linux Harbor tasks")
    if native.instruction != task.question:
        raise ValueError("Harbor task instruction changed after benchmark.load()")
    checksum = native.checksum
    if task.metadata.get("task_checksum", checksum) != checksum:
        raise ValueError("Harbor task files changed after benchmark.load()")

    return native, checksum


@dataclass(frozen=True, slots=True)
class _TrialPlan:
    config: dict[str, Any]
    options: dict[str, Any]
    strict: bool | None
    mounted_runtime: bool
    shared_verifier: bool
    environment_import: str | None
    installer: str
    multiplier: float
    fallback_timeout: float | None
    setup_timeout: float
    grace: float
    native_build: float
    build_timeout: float
    outer_timeout: float | None


def _plan_trial(agent: Agent, native: HarborTask, strict: bool | None) -> _TrialPlan:
    from flashrsi.integrations._terminal_integrity import validate
    from flashrsi.integrations.sandbox_runtime import RELAY, local_config, route_config

    config = copy.deepcopy(agent.config)
    options = config.get("harbor", {})
    validate(options.get("integrity_review"))
    remote_docker = options.get("remote_docker", False)
    if not isinstance(remote_docker, bool):
        raise ValueError("harbor.remote_docker must be a boolean")
    mounted_runtime = (
        not remote_docker
        and (strict is not None or RELAY.get() is not None)
        and getattr(agent, "name", None) in _CONTAINER_SHELLS
    )
    if mounted_runtime:
        config = local_config(agent, _RELAY_ENDPOINT)
        config["_suite_model_relay"] = True
        config["llm"].pop("side_base_url", None)
        config["llm"].pop("side_api_key", None)
    else:
        config = route_config(agent, config)
    _shell_turns(agent, config)
    shared_verifier = resolve_task_verifier_mode(native.config) == VerifierEnvironmentMode.SHARED
    environment_import = None if mounted_runtime else _docker_environment_import(native, remote_docker)
    install_source = options.get("uv_install_source", "astral")
    installers = {
        "astral": "https://astral.sh/uv/0.11.4/install.sh",
        "github": "https://github.com/astral-sh/uv/releases/download/0.11.4/uv-installer.sh",
    }
    if not isinstance(install_source, str) or install_source not in installers:
        raise ValueError("harbor.uv_install_source must be astral or github")
    multiplier = _positive(options.get("timeout_multiplier", 1.0), "harbor.timeout_multiplier")
    native_timeout = native.config.agent.timeout_sec
    # Legacy tasks without an explicit deadline can use the configured agent timeout.
    fallback_timeout = config.get("timeout") if native_timeout is None else None
    agent_timeout = native_timeout if native_timeout is not None else fallback_timeout
    setup_timeout = _positive(options.get("setup_timeout", 360), "harbor.setup_timeout")
    grace = _positive(options.get("cleanup_grace", 60), "harbor.cleanup_grace")
    native_build = _positive(
        getattr(native.config.environment, "build_timeout_sec", 600), "harbor native build timeout",
    )
    build_timeout = _positive(options.get("build_timeout", native_build), "harbor.build_timeout")
    outer_timeout = None
    if agent_timeout is not None:
        timeout = _positive(
            _positive(agent_timeout, "harbor agent timeout") * multiplier,
            "harbor effective agent timeout",
        )
        config["timeout"] = timeout
        verifier_timeout = getattr(getattr(native.config, "verifier", None), "timeout_sec", 600)
        environments = 1 if shared_verifier else 2
        outer_timeout = timeout + setup_timeout + verifier_timeout + environments * (build_timeout + grace) + grace
    return _TrialPlan(
        config, options, strict, mounted_runtime, shared_verifier, environment_import,
        installers[install_source], multiplier, fallback_timeout, setup_timeout, grace,
        native_build, build_timeout, outer_timeout,
    )


def _trial_config(
    task: Task, plan: _TrialPlan, root: Path, payload: Path, python: str | None, compose: list[Path],
    verifier_env: dict[str, str] | None, agent_allowed_hosts: list[str] | None, secrets: list[str],
) -> TrialConfig:
    environment_env = _plain(plan.options.get("environment_env", {}))
    if plan.mounted_runtime:
        environment_env = _redact(environment_env, secrets)
    trials_dir = plan.options.get("trials_dir")
    environment_class = "_TimedRemoteDockerEnvironment" if plan.environment_import else "_TimedDockerEnvironment"
    return TrialConfig(
        task=TaskConfig(path=Path(task.env)),
        trials_dir=Path(trials_dir).expanduser().resolve() if trials_dir else root / "trials",
        timeout_multiplier=plan.multiplier,
        verifier_timeout_multiplier=1.0,
        agent_setup_timeout_multiplier=1.0,
        environment_build_timeout_multiplier=plan.build_timeout / plan.native_build,
        agent=AgentConfig(
            import_path="flashrsi.integrations.harbor:_SnapshotAgent",
            model_name=plan.config["llm"]["model_name"],
            override_timeout_sec=plan.fallback_timeout,
            override_setup_timeout_sec=plan.setup_timeout,
            extra_allowed_hosts=agent_allowed_hosts or [],
            kwargs={"payload_dir": str(payload), "uv_installer_url": plan.installer, "runtime_python": python},
        ),
        environment=EnvironmentConfig(
            type="docker", delete=True, env=environment_env, extra_docker_compose=compose,
            import_path=f"flashrsi.integrations.harbor:{environment_class}",
            kwargs={"cleanup_grace": plan.grace, "require_local_images": bool(plan.strict)},
        ),
        verifier=VerifierConfig(env=verifier_env or {}),
    )


async def _execute_trial(trial: Trial, plan: _TrialPlan) -> TrialResult:
    if plan.shared_verifier:
        async def prepare_verifier(_event):
            await _prepare_shared_verifier(trial.agent_environment)

        trial.add_hook(TrialEvent.VERIFICATION_START, prepare_verifier)
    cleanup_failures = []
    token = _CLEANUP_FAILURES.set(cleanup_failures)
    try:
        async with asyncio.timeout(plan.outer_timeout):
            result = await trial.run()
        if cleanup_failures:
            raise InfraError(f"Harbor environment cleanup failed: {cleanup_failures[0]}")
        return result
    finally:
        _CLEANUP_FAILURES.reset(token)


async def _run(
    agent: Agent, task: Task, *, verifier_env: dict[str, str] | None = None,
    agent_allowed_hosts: list[str] | None = None,
) -> Trajectory:
    """The caller's rollout has already pinned its Snapshot and configuration."""
    strict = task.metadata.get("suite_require_local_image")
    if task.env is None:
        raise ValueError("Harbor execution requires task.env")
    if shutil.which("docker") is None:
        raise InfraError(f"{task.task_id}: Harbor requires Docker with the Compose plugin")
    agent._check_container()
    native, checksum = await _load_task(task, strict)
    snapshot = agent.snapshot
    plan = _plan_trial(agent, native, strict)
    secrets = sorted(
        set(_secrets(agent.config) + _secrets(plan.config) + _secrets(os.environ) + _secrets(verifier_env)),
        key=len, reverse=True,
    )
    trajectory = Trajectory(messages=({
        "type": "RunConfig", "config": _config_record(plan.config), "snapshot_id": snapshot.id,
    },))
    event = {
        "type": "HarborVerifier",
        "verifier_environment_mode": "shared" if plan.shared_verifier else "separate",
        "host_log_transport": "copy" if plan.environment_import else "mount",
        "task_id": task.task_id,
        "task_checksum": checksum,
        "snapshot_id": snapshot.id,
    }
    with TemporaryDirectory(prefix="flashrsi-harbor-") as temporary, ExitStack() as resources:
        root = Path(temporary)
        payload = root / "payload"
        payload.mkdir(mode=0o700)
        trial = None
        try:
            python, compose = None, []
            if plan.mounted_runtime:
                python, compose = _stage_runtime(root, resources, agent, plan.config)
            if plan.mounted_runtime:
                _stage(payload, agent, plan.config, task.question, native=True)
            else:
                _stage(payload, agent, plan.config, task.question)
            trial = await Trial.create(_trial_config(
                task, plan, root, payload, python, compose, verifier_env, agent_allowed_hosts, secrets,
            ))
            result = await _execute_trial(trial, plan)
            trajectory = _read_trial_result(trial, result, trajectory, event, snapshot.id)
        except Exception as exc:
            error = str(exc) or (
                "Harbor trial exceeded its outer deadline" if isinstance(exc, TimeoutError)
                else type(exc).__name__
            )
            event.update(error=error, infra_error=infra_reason(exc) or error)
            trajectory = replace(trajectory, status="error", error_message=error)
        finally:
            if trial is not None:
                try:
                    _trial_artifacts(trial, plan.options, event, secrets, archive=_archive_trial)
                except Exception as exc:
                    event["artifact_error"] = str(exc)
                    event.setdefault("infra_error", "Harbor artifact archive failed")
                    trajectory = replace(
                        trajectory, status="error", error_message=f"Harbor artifact archive failed: {exc}",
                    )
        return await _finish_trial(task, native, plan, trajectory, event, secrets)


async def _finish_trial(
    task: Task, native: HarborTask, plan: _TrialPlan, trajectory: Trajectory,
    event: dict[str, Any], secrets: list[str],
) -> Trajectory:
    if plan.strict is not None and event.get("infra_error"):
        detail = _redact(event.get("error", event["infra_error"]), secrets)
        image = getattr(native.config.environment, "docker_image", None)
        assets = f"{task.env}/environment" + (f" and image {image}" if image else "")
        raise InfraError(
            f"{task.task_id}: Harbor execution failed: {detail}. "
            f"Check staged assets at {assets} and the Docker daemon."
        ) from None
    review_options = plan.options.get("integrity_review")
    if review_options is not None:
        from flashrsi.integrations._terminal_integrity import review

        event = _redact(event, secrets)
        await review(task, _redacted_trajectory(trajectory, secrets), event, review_options)
    return _redacted_trajectory(trajectory, secrets, event)


async def _preflight_image(task: Task, native: HarborTask) -> None:
    image = native.config.environment.docker_image
    if image:
        try:
            await _docker("image", "inspect", "--format", "{{.Id}}", image, timeout=30)
        except Exception as exc:
            # Docker inspection can include image environment data. Never echo it.
            reason = infra_reason(exc) or "image is unavailable locally"
            raise InfraError(
                f"{task.task_id}: Harbor image {image}: {reason}. "
                "Stage the pinned image on the active Docker daemon (pull or load it)."
            ) from None


class _TimedCleanup:
    def __init__(self, *args, cleanup_grace, require_local_images, **kwargs):
        super().__init__(*args, **kwargs)
        self._cleanup_grace = cleanup_grace
        self._require_local_images = require_local_images
        self._cleanup_failure = None
        self._stopping = False

    async def _run_docker_compose_command(self, args, *options, **kwargs):
        if self._require_local_images and args[0] == "up":
            args = ["up", "--pull=never", *args[1:]]
        try:
            return await super()._run_docker_compose_command(args, *options, **kwargs)
        except Exception as exc:
            if self._stopping and args[0] in {"down", "stop"}:
                self._cleanup_failure = exc
            raise

    async def stop(self, delete):
        self._cleanup_failure = None
        self._stopping = True
        try:
            async with asyncio.timeout(self._cleanup_grace):
                await super().stop(delete=delete)
            if self._cleanup_failure is not None:
                raise self._cleanup_failure
        except Exception as exc:
            # Harbor logs separate-verifier stop failures without setting
            # exception_info. Capture them in this trial's execution context.
            if (failures := _CLEANUP_FAILURES.get()) is not None:
                failures.append(str(exc) or type(exc).__name__)
            raise
        finally:
            self._stopping = False


class _TimedDockerEnvironment(_TimedCleanup, DockerEnvironment):
    pass


class _TimedRemoteDockerEnvironment(_TimedCleanup, _RemoteDockerEnvironment):
    pass


def _docker_environment_import(native: HarborTask, remote_docker: bool = False) -> str | None:
    shared = resolve_task_verifier_mode(native.config) == VerifierEnvironmentMode.SHARED
    if remote_docker or (
        not (native.paths.environment_dir / "docker-compose.yaml").exists()
        and (shared or not (native.paths.tests_dir / "docker-compose.yaml").exists())
    ):
        return "flashrsi.integrations._harbor_remote:_RemoteDockerEnvironment"
    return None


async def _prepare_shared_verifier(environment):
    """Reset reserved scoring paths before Harbor uploads the native tests."""
    paths = EnvironmentPaths.for_os(environment.os)
    result = await environment.exec("test ! -L /logs", user="root")
    if result.return_code:
        raise RuntimeError(
            "Refusing a linked Harbor logs directory before verification"
        )
    for path, chmod in ((paths.verifier_dir, True), (paths.tests_dir, False)):
        result = await environment.empty_dirs([path], chmod=chmod)
        if result is None or result.return_code:
            raise RuntimeError("Cannot clear Harbor verifier inputs and rewards")


def _positive(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"agent.{field} must be a positive finite number")
    return value


def _stage(directory: Path, agent: Agent, config: dict[str, Any], prompt: str, *, native: bool = False) -> None:
    config = copy.deepcopy(config)
    config.get("harbor", {}).pop("integrity_review", None)
    config.get("harbor", {}).pop("artifacts_dir", None)
    config.get("harbor", {}).pop("trials_dir", None)
    # Keep evaluation rules outside the candidate's mutable snapshot. Public
    # benchmark repositories expose hidden tests and oracle answers online.
    prompt += (
        "\n\nEvaluation integrity rules: Solve the task independently. Do not "
        "search for, download, read, or use benchmark-specific reference solutions, "
        "hidden verifier tests, expected answers, or other agents' task solutions "
        "from benchmark repositories, mirrors, datasets, or discussion threads. "
        "Do not modify the verifier or forge reward files. Internet access for "
        "ordinary documentation, dependencies, and resources requested by the task "
        "is permitted. Tests and source files explicitly supplied as task inputs "
        "may be inspected; this does not permit fetching hidden benchmark files."
    )
    stage = _stage_native if native else _stage_payload
    stage(directory, agent, config, prompt)


class _SnapshotAgent(BaseAgent):
    """Harbor adapter for our existing, independently versioned harness adapter."""

    def __init__(
        self,
        *args,
        payload_dir,
        uv_installer_url="https://astral.sh/uv/0.11.4/install.sh",
        working_dir=None,
        runtime_python=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.payload_dir = Path(payload_dir)
        self.uv_installer_url = uv_installer_url
        self.working_dir = working_dir
        self.runtime_python = runtime_python

    @staticmethod
    def name() -> str:
        return "flashrsi-lite"

    def version(self) -> str:
        return version("flashrsi-lite")

    async def setup(self, environment) -> None:
        await environment.upload_dir(self.payload_dir, _ROOT)
        # uv installs the requested Python independently of a task's Python.
        # Minimal Debian/Ubuntu task images may not include a downloader yet.
        command = (
            shlex.join([self.runtime_python, "-I", "-B", "--version"])
            if self.runtime_python else _setup_script(self.uv_installer_url)
        )
        result = await environment.exec(command)
        if result.return_code:
            raise RuntimeError(
                f"Container dependency setup failed: {result.stderr or result.stdout}"
            )

    async def run(self, instruction, environment, context) -> None:
        native = self.runtime_python is not None
        python = self.runtime_python or f"{_ROOT}/venv/bin/python"
        args = _container_python(python, str(self.environment_logs_dir / "flashrsi.json"), native=native)
        command = shlex.join(args)
        if native:
            environment_vars = _relay_shell_env(_RELAY_ENDPOINT)
            command = (
                f"export PATH={_RUNTIME}/bin:$PATH; "
                + shlex.join(["env", *(f"{key}={value}" for key, value in environment_vars.items()),
                              python, "-I", "-B", "-c", _RELAY_BRIDGE, *args])
            )
        else:
            from flashrsi.integrations.sandbox_runtime import relay_harbor

            if await relay_harbor(environment):
                command = ("env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY "
                           "-u http_proxy -u https_proxy -u all_proxy " + command)
        command = getattr(self, "_command_prefix", "") + command
        # Isolate bridge imports without changing the task's cwd or PYTHONPATH;
        # native tools inherit the image's environment. The entry point removes
        # request.json before starting the harness.
        result = await environment.exec(
            command, **({"cwd": self.working_dir} if self.working_dir else {})
        )
        if not environment.capabilities.mounted and await environment.is_file(
            str(self.environment_logs_dir / "flashrsi.json")
        ):
            await environment.download_file(
                str(self.environment_logs_dir / "flashrsi.json"),
                self.logs_dir / "flashrsi.json",
            )
        context.metadata = {"flashrsi_trajectory": "flashrsi.json"}
        if result.return_code:
            trace_path = self.logs_dir / "flashrsi.json"
            if native and trace_path.is_file():
                payload = json.loads(trace_path.read_text())
                payload.pop("task", None)
                if reason := trajectory_infra_reason(Trajectory(**payload)):
                    raise InfraError(reason)
            # Harbor continues the verifier for this exception, preserving task
            # feedback even when the harness has failed or reached its timeout.
            raise NonZeroAgentExitCodeError(
                f"Container agent exited with {result.return_code}: "
                f"{result.stderr or result.stdout or ''}"
            )
