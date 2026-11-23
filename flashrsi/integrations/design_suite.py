"""Bind design tasks and provide shared host-side sandbox utilities.

Only the host dispatches proposals and reads trusted verifier receipts.
FLASHRSI_ENG_IMAGE and FLASHRSI_ADMET_IMAGE can pin preloaded offline images.
Container environments and authenticated worker requests use explicit inputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from flashrsi._records import _redact, _secrets
from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.benchmarks._suite import read_json
from flashrsi.benchmarks.design_suite import (
    TDCADMET,
    EngDesignOpen,
    FrontierEng,
    KernelBench,
)
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.core.infra import trajectory_infra_reason
from flashrsi.integrations import sandbox_runtime
from flashrsi.integrations.sandbox_runtime import LABEL
from flashrsi.integrations.workspace_suite import episode_config

ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8"}
ENG_RESOURCES = {"cpus": 4, "memory": "8g", "pids_limit": 256, "shm_size": "1g"}
PROMPT_WINDOW = 96_000
Resources = Mapping[str, int | float | str]
Mounts = Sequence[tuple[Path, str]]
Verdict = tuple[Trajectory, dict[str, Any]]
DesignBenchmark = EngDesignOpen | FrontierEng | TDCADMET | KernelBench


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def suite_image(repository: str, variable: str) -> str:
    if image := os.getenv(variable):
        return image
    command = ["docker", "image", "ls", "--format", "{{.Tag}}", repository]
    try:
        with subprocess.Popen(command, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
            try:
                listing, _ = process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
        if process.returncode:
            raise InfraError(f"could not list preloaded {repository} images")
    except FileNotFoundError:
        return f"{repository}:suite"
    except (OSError, subprocess.SubprocessError) as exc:
        raise InfraError("could not resolve the preloaded suite image") from exc
    tags = set(listing.splitlines()) - {"<none>"}
    if "suite" in tags or not tags:
        return f"{repository}:suite"
    if len(tags) == 1:
        return f"{repository}:{next(iter(tags))}"
    raise InfraError(f"set {variable} to a preloaded {repository} image")


def engineering_image() -> str:
    return suite_image("flashrsi-eng", "FLASHRSI_ENG_IMAGE")


def admet_image() -> str:
    return suite_image("flashrsi-mol", "FLASHRSI_ADMET_IMAGE")


def finite_number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def tail_bytes(text: str, size: int) -> str:
    return text.encode("utf-8", errors="replace")[-size:].decode("utf-8", errors="ignore") if size else ""


class CandidateExecutionError(ValueError):
    """A candidate exited abnormally inside an otherwise working sandbox."""


def scratch() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="flashrsi-design-")


def smoke_mode(config: Mapping[str, Any]) -> bool:
    return config.get("smoke", False)


def oracle_budget(config: Mapping[str, Any], default: int, maximum: int, *, smoke_default: int = 1) -> int:
    if smoke_mode(config):
        budget = config.get("smoke_oracle_budget", config.get("oracle_budget", smoke_default))
    else:
        budget = config.get("oracle_budget", default)
    if type(budget) is not int or not 1 <= budget <= maximum:
        raise ValueError(f"oracle_budget must be an integer from 1 to {maximum}")
    return budget


def container_limits(resources: Resources) -> list[str]:
    cpus, memory = resources["cpus"], resources["memory"]
    pids_limit, shm_size = resources["pids_limit"], resources["shm_size"]
    if (
        type(cpus) not in (int, float)
        or not math.isfinite(cpus)
        or cpus <= 0
        or type(pids_limit) is not int
        or pids_limit <= 0
        or any(not isinstance(size, str) or not re.fullmatch(r"[1-9][0-9]*[kmg]", size) for size in (memory, shm_size))
    ):
        raise ValueError("invalid container resource limits")
    return ["--cpus", str(cpus), "--memory", memory, "--pids-limit", str(pids_limit), "--shm-size", shm_size]


def docker_args(
    image: str,
    name: str,
    mounts: Mounts = (),
    interactive: bool = False,
    *,
    resources: Resources | None = None,
    writable: Mounts = (),
    gpu_device: str | None = None,
    workdir: str = "/tmp",
    env: Sequence[str] = (),
    entrypoint: str = "python",
) -> list[str]:
    limits = container_limits(resources or {"cpus": 2, "memory": "3g", "pids_limit": 128, "shm_size": "64m"})
    args = [
        "docker", "run", "--pull=never", "--rm", "--name", name, "--label", LABEL,
        "--init", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", *limits, "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=512m", "-w", workdir,
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
        "-e", "MPLCONFIGDIR=/tmp/matplotlib", "-e", "HOME=/tmp",
    ]
    if interactive:
        args.append("-i")
    if gpu_device:
        args.extend(("--gpus", f"device={gpu_device}"))
    for source, destination in mounts:
        args.extend(("--mount", f"type=bind,source={source.resolve()},target={destination},readonly"))
    for source, destination in writable:
        args.extend(("--mount", f"type=bind,source={source.resolve()},target={destination}"))
    for value in env:
        args.extend(("-e", value))
    command = [*args, "--entrypoint", entrypoint, image]
    return [*command, "-I", "-B"] if entrypoint == "python" else command


def cleanup(name: str) -> None:
    key, value = LABEL.split("=", 1)
    label_lookup = "{{index .Config.Labels " + json.dumps(key) + "}}"
    for _ in range(12):
        r = subprocess.run(["docker", "container", "inspect", "--format", label_lookup, name], capture_output=True, timeout=20, env=ENV)
        if r.returncode and (b"No such container" in r.stderr or b"No such object" in r.stderr):
            return
        if not r.returncode:
            if r.stdout.strip() != value.encode():
                raise InfraError("refusing to remove a container outside the suite run")
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20, env=ENV)
        time.sleep(0.3)
    raise RuntimeError("container cleanup could not be confirmed")


def docker_json(
    image: str,
    command: Sequence[str],
    mounts: Mounts = (),
    timeout: float = 75,
    stdin: str | None = None,
    *,
    candidate: bool = False,
    **container: Any,
) -> tuple[Any, dict[str, Any]]:
    name = "flashrsi-design-" + uuid.uuid4().hex[:14]
    start = time.monotonic()
    try:
        r = subprocess.run(
            [*docker_args(image, name, mounts, interactive=stdin is not None, **container), *command],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=ENV,
        )
        if r.returncode != 0:
            errors = re.findall(r"(?m)^([A-Za-z][A-Za-z0-9_]{0,60}(?:Error|Exception)):", r.stderr)
            if "libgomp" in r.stderr and "Thread creation failed" in r.stderr:
                errors.append("OpenMPThreadError")
            reason = f" ({errors[-1]})" if errors else ""
            failure = CandidateExecutionError if candidate and r.returncode not in (125, 126, 127, 137) else InfraError
            raise failure(f"isolated container failed exit={r.returncode}{reason}")
        try:
            response = json.loads(r.stdout)
        except ValueError:
            raise InfraError("isolated verifier returned invalid JSON") from None
    finally:
        cleanup(name)
    receipt = {"container": name, "cleaned": True, "seconds": round(time.monotonic() - start, 3), "network": "none"}
    if container.get("resources") is not None:
        receipt["resources"] = dict(container["resources"])
    return response, receipt


def _read_candidate(workspace: Path, filename: str) -> str:
    sandbox_runtime.verify_workspace(workspace)
    output = workspace / filename
    if not output.is_file() or output.stat().st_size > 512 * 1024:
        raise ValueError("candidate file absent or exceeds size limit")
    return output.read_text()


def python_environment_mounts(python: str, environment: Path) -> list[tuple[Path, str]]:
    python = Path(python)
    root = python.resolve().parents[1]
    mounts = [(environment, str(environment))]
    if root != environment:
        mounts.append((root, str(root)))
    if python.is_symlink():
        launcher_root = (python.parent / python.readlink()).parents[1]
        if launcher_root not in (root, environment):
            mounts.append((root, str(launcher_root)))
    return mounts


async def solver(agent: Agent, prompt: str, workspace: Path, *, image: str, resources: Resources | None = None) -> tuple[Trajectory, str]:
    """Run the solver with the shared relay and explicit container limits."""
    options = resources or ENG_RESOURCES
    container_limits(options)
    return await sandbox_runtime.solver(
        agent,
        prompt,
        workspace,
        image=image,
        cpus=options["cpus"],
        memory=options["memory"],
        pids=options["pids_limit"],
        shm_size=options["shm_size"],
    )


async def artifact_proposal(
    agent: Agent, task: Task, workspace: Path, *, image: str, filename: str, prompt: str | None = None, resources: Resources | None = None
) -> tuple[str, Trajectory, str]:
    contract = (prompt or task.question) + f"\nSave the final answer to {filename} in the task workspace."
    trajectory, name = await solver(agent, contract, workspace, image=image, resources=resources)
    if reason := trajectory_infra_reason(trajectory):
        raise InfraError(reason)
    value = _read_candidate(workspace, filename)
    return value, trajectory, name


def search_prompt(
    spec: str, parent: str, metrics: object, feedback: Sequence[str], *, instruction: str = "Return a JSON object with key code."
) -> str:
    header = "\n\nTask specification\n"
    best = "\n\nBest parent program and metrics\n"
    recent = "\n\nRecent attempts and failures\n"
    ending = "\n\n" + instruction
    capacity = PROMPT_WINDOW - sum(map(len, (header, best, recent, ending)))
    metrics_text = json.dumps(metrics, ensure_ascii=False, allow_nan=False)[:2048] + "\n"
    feedback_text = "\n".join(feedback)[-13_000:]
    available = capacity - len(metrics_text) - len(feedback_text)
    spec_limit = min(len(spec), available - min(len(parent), 32000))
    parent_limit = available - spec_limit
    return header + spec[:spec_limit] + best + metrics_text + parent[:parent_limit] + recent + feedback_text + ending


def cpu_sandbox_command(mounts: Mounts, work: Path, cwd: Path, environment: Mapping[str, str]) -> list[str]:
    bwrap = shutil.which("bwrap", path=ENV["PATH"])
    if not bwrap:
        raise InfraError("staged Frontier CPU execution needs bubblewrap or a configured CPU image")
    args = [
        bwrap,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--chmod",
        "1777",
        "/tmp",
        "--tmpfs",
        "/dev/shm",
    ]
    for system in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/localtime", "/etc/alternatives"):
        args.extend(("--ro-bind-try", system, system))
    for source, destination in mounts:
        args.extend(("--ro-bind", str(source.resolve()), destination))
    for variable, value in environment.items():
        args.extend(("--setenv", variable, value))
    return [*args, "--bind", str(work), "/work", "--chdir", str(Path("/work") / cwd), "--", "/bin/bash"]


def stage_readonly_assets(source: Path, work: Path, guest: Path, writable: set[Path], excluded: set[Path], relative: Path = Path()) -> None:
    for asset in source.iterdir():
        child = relative / asset.name
        if child in excluded:
            continue
        link = work / child
        if child in writable:
            if asset.is_symlink() or not asset.is_dir():
                raise InfraError("invalid candidate destination parent")
            link.mkdir()
            stage_readonly_assets(asset, work, guest, writable, excluded, child)
        else:
            link.symlink_to(guest / child)


def run_sandbox(args: Sequence[str], timeout: int) -> tuple[int, str]:
    with tempfile.TemporaryFile() as stderr:
        result = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=stderr, timeout=timeout, env=ENV)
        stderr.seek(max(0, stderr.tell() - 4096))
        diagnostic = tail_bytes(stderr.read(4096).decode(errors="replace"), 4096)
    return result.returncode, diagnostic


def read_sandbox_metrics(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise InfraError("Frontier evaluator did not emit a bounded metrics.json")
    try:
        report = read_json(path)
    except ValueError:
        raise InfraError("Frontier evaluator emitted malformed metrics.json") from None
    if not isinstance(report, dict):
        raise InfraError("Frontier evaluator emitted malformed metrics.json")
    return report


async def propose_code(
    agent: Agent, task: Task, prompt: str, *, workspace: Path | None = None, json_hint: bool = False
) -> tuple[object, dict[str, Any]]:
    if isinstance(agent, ProgramSearchTarget):
        instruction = '\nUse JSON key "code".' if json_hint else ""
        proposal, usage = await agent.propose(prompt + instruction)
        return proposal.get("code"), usage
    if workspace is not None:
        return await _propose_code_artifact(agent, task, prompt, workspace)
    with scratch() as tmp:
        return await _propose_code_artifact(agent, task, prompt, Path(tmp))


async def _propose_code_artifact(agent: Agent, task: Task, prompt: str, workspace: Path) -> tuple[str, dict[str, Any]]:
    code, trajectory, _ = await artifact_proposal(agent, task, workspace, image=engineering_image(), filename="candidate.py", prompt=prompt)
    return code, dict(trajectory.usage)


def receipt_signature(receipt: Mapping[str, Any], token: str) -> str:
    body = json.dumps(
        {key: value for key, value in receipt.items() if key != "signature"},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hmac.new(token.encode(), body.encode(), hashlib.sha256).hexdigest()


async def _rpc_request(endpoint: str, route: str, payload: dict[str, Any], timeout: float, token: str | None) -> Any:
    headers = {"Idempotency-Key": payload["job_id"]}
    if token:
        headers["Authorization"] = "Bearer " + token
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(timeout, 20)), trust_env=False) as client:
            async with client.stream("POST", endpoint.rstrip("/") + route, headers=headers, json=payload) as response:
                if response.status_code == 403:
                    raise PermissionError("GPU worker denied the task outside its open manifest")
                response.raise_for_status()
                raw = bytearray()
                async for block in response.aiter_bytes():
                    raw.extend(block)
                    if len(raw) > 1024 * 1024:
                        raise InfraError("GPU worker receipt exceeded its transport limit")
    return json.loads(raw)


def worker_rpc(endpoint: str, route: str, payload: dict[str, Any], timeout: float, token: str | None = None) -> dict[str, Any]:
    try:
        url = urlsplit(endpoint)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError
    except (TypeError, ValueError):
        raise InfraError("invalid GPU worker URL") from None
    try:
        receipt = asyncio.run(_rpc_request(endpoint, route, payload, timeout, token))
        if not isinstance(receipt, dict):
            raise ValueError
        if token:
            signature = receipt.get("signature")
            if not isinstance(signature, str) or not hmac.compare_digest(signature, receipt_signature(receipt, token)):
                raise InfraError("GPU worker receipt authentication failed")
        return receipt
    except PermissionError:
        raise
    except (httpx.HTTPError, OSError, ValueError, TypeError):
        raise InfraError("GPU worker RPC failed or exceeded its deadline") from None


@dataclass(frozen=True, slots=True)
class _DesignExecution:
    benchmark: DesignBenchmark
    execute: Callable[[Agent, Any, Task], Awaitable[Verdict]]

    async def run(self, target: Agent, task: Task) -> Trajectory:
        self.benchmark.check_task(task)
        try:
            cap = getattr(self.benchmark, "proposal_max_tokens", None)
            with episode_config(target, max_tokens=cap):
                trajectory, verdict = await self.execute(target, self.benchmark, task)
        except (ValueError, SyntaxError):
            trajectory, verdict = Trajectory(), {"score": 0.0, "candidate_valid": False}
        event = {"type": "SuiteVerifier", "task_id": task.task_id, **verdict}
        metrics = dict(trajectory.metrics)
        for field in ("normalized_score", "raw_mean", "raw_sample_sd", "tdc_mean", "tdc_std"):
            if field in verdict:
                metrics[field] = verdict[field]
        return replace(trajectory, metrics=metrics, messages=(*trajectory.messages, _redact(event, _secrets(os.environ))))


async def engdesign(agent: Agent, benchmark: EngDesignOpen, task: Task) -> Verdict:
    from flashrsi.integrations.engdesign import engdesign as execute

    return await execute(agent, benchmark, task)


async def frontier(agent: Agent, benchmark: FrontierEng, task: Task) -> Verdict:
    from flashrsi.integrations.frontier_eng import frontier as execute

    return await execute(agent, benchmark, task)


async def admet(agent: Agent, benchmark: TDCADMET, task: Task) -> Verdict:
    from flashrsi.integrations.admet import admet as execute

    return await execute(agent, benchmark, task)


async def kernel(agent: Agent, benchmark: KernelBench, task: Task) -> Verdict:
    from flashrsi.integrations.kernelbench import kernel as execute

    return await execute(agent, benchmark, task)


def bind(agent: Agent, benchmark: DesignBenchmark) -> None:
    execute = {EngDesignOpen: engdesign, FrontierEng: frontier, TDCADMET: admet, KernelBench: kernel}[type(benchmark)]
    agent._env_runner = _DesignExecution(benchmark, execute).run
