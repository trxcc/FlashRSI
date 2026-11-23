"""Exercise Harbor and SWE production clocks and frozen manifests.

Test Docker and evaluator behavior with local, credential-free fixtures.
The host-side tests do not launch benchmark containers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from importlib.util import find_spec
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flashrsi.benchmarks import _suite, harbor_suite
from flashrsi.benchmarks.swe_bench import SWEBench, _task_from_row
from flashrsi.core import InfraError, Task, Trajectory
from flashrsi.integrations import harbor_suite as harbor_execution
from flashrsi.integrations import sandbox_runtime as runtime
from flashrsi.integrations import swe_bench as swe

if find_spec("harbor"):
    from flashrsi.integrations import harbor
else:
    harbor = None


def _agent(**options):
    return SimpleNamespace(
        config={
            "llm": {"model_name": "fixture", "base_url": "https://model.example/v1"},
            **options,
        },
        snapshot=SimpleNamespace(id="fixture-snapshot"),
        _check_container=lambda: None,
    )


def _git_patch(path):
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
    )


def _swe_row(identifier="fixture-0001"):
    return {
        "instance_id": identifier, "problem_statement": "Repair the example.",
        "image": "fixture:staged", "base_commit": "1" * 40,
        "environment_setup_commit": "2" * 40, "version": "1", "repo": "example/fixture",
        "patch": _git_patch("src/example.py"), "test_patch": _git_patch("tests/test_protected.py"),
        "eval_script": "#!/bin/bash\nexit 0\n", "log_parser": "pytest", "eval_type": "pass_and_fail",
        "FAIL_TO_PASS": ["tests/test_protected.py::test_example"], "PASS_TO_PASS": [],
    }


def _harbor_task(directory, timeout, environment="", verifier=113):
    (directory / "environment").mkdir(parents=True)
    (directory / "tests").mkdir()
    (directory / "instruction.md").write_text("Do the example.")
    (directory / "environment/Dockerfile").write_text("FROM scratch\n")
    (directory / "tests/test.sh").write_text("#!/bin/sh\n")
    (directory / "task.toml").write_text(
        f"[agent]\ntimeout_sec = {timeout}\n[verifier]\ntimeout_sec = {verifier}\n"
        f"[environment]\n{environment}"
    )


@pytest.mark.skipif(harbor is None, reason="Harbor is an optional dependency")
@pytest.mark.parametrize("native,multiplier", [(600, 1), (28800, 1), (28800, 1.25), (0.75, 0.5)])
@pytest.mark.parametrize("legacy_resources", [False, True])
def test_harbor_native_deadlines_and_resources(tmp_path, monkeypatch, native, multiplier, legacy_resources):
    from harbor.models.trial.paths import TrialPaths

    directory = tmp_path / "task"
    sizes = 'memory = "4G"\nstorage = "10G"\n' if legacy_resources else "memory_mb = 4096\nstorage_mb = 10240\n"
    _harbor_task(directory, native, "cpus = 4\ngpus = 1\nallow_internet = false\n" + sizes)
    task = Task("synthetic", "Do the example.", env=str(directory))
    options = {"artifacts_dir": str(tmp_path / "saved")}
    if multiplier != 1:
        options["timeout_multiplier"] = multiplier
    agent = _agent(timeout=900, harbor=options)
    staged = []
    budgets = []
    real_timeout = asyncio.timeout
    monkeypatch.setattr(harbor.asyncio, "timeout", lambda value: budgets.append(value) or real_timeout(value))
    monkeypatch.setattr(harbor, "_stage", lambda _root, _agent, config, _prompt: staged.append(config))

    async def create(config):
        native_task = harbor.HarborTask(directory)
        environment = native_task.config.environment
        assert environment.cpus == 4 and environment.gpus == 1
        assert environment.memory_mb == 4096 and environment.storage_mb == 10240
        assert environment.network_mode.value == "no-network"
        assert config.environment.override_cpus is None and config.environment.override_memory_mb is None
        assert config.environment.override_storage_mb is None and config.environment.override_gpus is None
        assert config.agent.override_timeout_sec is None
        assert config.verifier.override_timeout_sec is None
        trial = SimpleNamespace(
            config=config, task=native_task, paths=TrialPaths(config.trials_dir / "fixture"),
        )
        trial._resolve_timeout_sec = harbor.Trial._resolve_timeout_sec.__get__(trial)
        assert harbor.Trial._compute_agent_timeout_sec(trial) == native * multiplier
        assert harbor.Trial._compute_agent_setup_timeout_sec(trial) == 360
        assert harbor.Trial._compute_environment_build_timeout_sec(trial) == 600
        assert harbor.Trial._compute_verifier_timeout_sec(trial) == 113
        assert config.environment.kwargs["cleanup_grace"] == 60
        trial.paths.mkdir()

        async def run():
            (trial.paths.agent_dir / "flashrsi.json").write_text(
                json.dumps({"messages": [{"type": "RunConfig", "snapshot_id": agent.snapshot.id}]})
            )
            return SimpleNamespace(
                id="fixture", exception_info=None, verifier_result=SimpleNamespace(rewards={"reward": 1}),
            )

        trial.run = run
        trial.add_hook = lambda *_: None
        return trial

    monkeypatch.setattr(harbor.Trial, "create", create)
    result = asyncio.run(harbor._run(agent, task))
    assert result.status == "finished", result.error_message
    assert staged[0]["timeout"] == native * multiplier
    assert budgets[0] > native * multiplier + 360 + 600 + 113 + 60


@pytest.mark.skipif(harbor is None, reason="Harbor is an optional dependency")
@pytest.mark.parametrize("failure", ["image", "build", "setup", "verifier", "agent_timeout"])
def test_harbor_errors_have_task_and_staging_context(tmp_path, monkeypatch, failure):
    directory = tmp_path / "task"
    _harbor_task(directory, 600, 'docker_image = "fixture:needs-staging"\n')
    task = Task("missing-example", "Do the example.", env=str(directory),
                metadata={"suite_require_local_image": True})
    agent = _agent(harbor={"artifacts_dir": str(tmp_path / "saved")})
    docker = AsyncMock(side_effect=RuntimeError("No such image") if failure == "image" else None)
    monkeypatch.setattr(harbor, "_docker", docker)
    monkeypatch.setattr(harbor, "_stage", lambda *_: None)
    if failure == "build":
        monkeypatch.setattr(harbor.Trial, "create", AsyncMock(side_effect=RuntimeError("build context missing")))
    else:
        from harbor.models.trial.paths import TrialPaths

        async def create(config):
            paths = TrialPaths(config.trials_dir / "failure")
            paths.mkdir()

            async def run():
                return SimpleNamespace(
                    id="fixture", verifier_result=None,
                    exception_info=SimpleNamespace(
                        exception_type={
                            "setup": "AgentSetupTimeoutError", "verifier": "VerifierTimeoutError",
                            "agent_timeout": "AgentTimeoutError",
                        }[failure],
                        exception_message=f"{failure} exceeded the phase budget",
                    ),
                )

            return SimpleNamespace(paths=paths, run=run, add_hook=lambda *_: None)

        monkeypatch.setattr(harbor.Trial, "create", create)
    if failure == "agent_timeout":
        row = asyncio.run(harbor._run(agent, task))
        assert row.status == "error" and "infra_error" not in row.messages[-1]
    else:
        with pytest.raises(InfraError, match=r"(?is)missing-example.*fixture:needs-staging.*stag|missing-example.*stag.*fixture:needs-staging"):
            asyncio.run(harbor._run(agent, task))
    docker.assert_awaited_once_with("image", "inspect", "--format", "{{.Id}}", "fixture:needs-staging", timeout=30)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def manifest_root(tmp_path):
    lock = {"files": {}}
    review = {}
    for name in ("tb21_train", "tb21_val", "swegym_lite_train"):
        spec = harbor_suite.SPLITS[name]
        rows, review[name] = [], {}
        for index in range(spec.count):
            task_id = f"{name}-{index:03d}"
            if spec.family == "harbor":
                path = tmp_path / "tasks" / spec.payload / task_id
                _harbor_task(path, 600)
                hashes = {item.relative_to(path).as_posix(): _digest(item)
                          for item in path.rglob("*") if item.is_file()}
                review[name][task_id] = {"sha256": hashes}
                rows.append({
                    "task_id": task_id, "instruction_sha256": hashes["instruction.md"],
                    "verifier_sha256": hashes["tests/test.sh"], "harbor_ref": {},
                })
            else:
                path = tmp_path / "prepared" / spec.payload / f"{task_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                row = _swe_row(task_id)
                path.write_text(json.dumps(row))
                review[name][task_id] = {"sha256": _digest(path)}
                rows.append({key: row[key] for key in ("instance_id", "base_commit", "repo", "image")})
        path = tmp_path / spec.filename
        path.write_text(json.dumps({
            "schema_version": 1, "id_field": "task_id" if spec.family == "harbor" else "instance_id",
            "split": spec.role, "count": spec.count, "tasks": rows,
        }))
        lock["files"][spec.filename] = {"sha256": _digest(path)}
    path = tmp_path / "reviewed_payloads.json"
    path.write_text(json.dumps(review))
    lock["reviewed_payloads_sha256"] = _digest(path)
    (tmp_path / "manifest_checksums.json").write_text(json.dumps(lock))
    return tmp_path


@pytest.mark.parametrize("name,counts,smoke_id", [
    ("tb21_clean", (56, 32), "tb21_train-000"),
    ("swegym_lite", (230, 0), "swegym_lite_train-000"),
])
def test_default_loads_entire_manifest_and_smoke_uses_explicit_ids(manifest_root, name, counts, smoke_id):
    if name == "tb21_clean":
        pytest.importorskip("harbor")
    config = {"name": name, "root": str(manifest_root), "smoke_task_ids": [smoke_id]}
    benchmark = harbor_suite.HarborSuite(config)
    assert benchmark.smoke_whitelist_only is False
    benchmark.load()
    assert (len(benchmark.train), len(benchmark.dev)) == counts
    smoke = harbor_suite.HarborSuite({**config, "smoke_whitelist_only": True})
    smoke.load()
    assert [task.task_id for task in smoke.train] == [smoke_id]
    assert smoke.dev == ()


def test_missing_task_aborts_full_load_without_breaking_smoke(manifest_root):
    pytest.importorskip("harbor")
    missing = manifest_root / "tasks/tb21/tb21_train-001/task.toml"
    missing.unlink()
    config = {"name": "tb21_clean", "root": str(manifest_root), "smoke_task_ids": ["tb21_train-000"]}
    with pytest.raises(InfraError) as error:
        harbor_suite.HarborSuite(config).load()
    assert str(missing) in str(error.value) and "Stage the pinned" in str(error.value)
    smoke = harbor_suite.HarborSuite({**config, "smoke_whitelist_only": True})
    assert len(smoke.load()["train"]) == 1


def test_wiring_passes_swe_options_to_native_backend(manifest_root):
    config = {
        "name": "swegym_lite", "root": str(manifest_root),
        "setup_timeout": 610, "evaluation_timeout": 1900, "cleanup_grace": 70,
        "summary_timeout": 80, "verifier_resources": {"memory": "6g"},
    }
    benchmark = harbor_suite.HarborSuite(config)
    benchmark.load()
    options = swe._evaluation_options(benchmark._backends["swegym_lite_train"].config)
    assert options == {
        "setup_timeout": 610, "evaluation_timeout": 1900, "cleanup_grace": 70,
        "summary_timeout": 80, "resources": (
            "--cpus=2", "--memory=6g", "--pids-limit=512", "--shm-size=64m", "--storage-opt=size=20g",
        ),
    }


def test_suite_binding_dispatches_only_reviewed_tasks(manifest_root, monkeypatch):
    benchmark = harbor_suite.HarborSuite({
        "name": "swegym_lite", "root": manifest_root, "task_ids": ["swegym_lite_train-000"],
    })
    task = benchmark.load()["train"][0]
    agent = SimpleNamespace(config=SimpleNamespace(llm=SimpleNamespace(model_name="fixture")))
    server = object()
    gateway = AsyncMock(return_value="127.0.0.1")

    @contextmanager
    def relay(address, model, *, unix):
        assert address == ("127.0.0.1", 0) and model == "fixture" and unix is False
        yield server

    async def dispatch(target, received):
        assert target is agent and received == task
        assert runtime.RELAY.get() is server
        return Trajectory(output="dispatched")

    def bind_swe(target, backend):
        assert backend is benchmark._backends["swegym_lite_train"]
        target._env_runner = dispatch

    monkeypatch.setattr(swe, "bind", bind_swe)
    monkeypatch.setattr(runtime, "gateway", gateway)
    monkeypatch.setattr(runtime, "relay", relay)
    harbor_execution.bind(agent, benchmark)
    assert asyncio.run(agent._env_runner(agent, task)).output == "dispatched"
    with pytest.raises(PermissionError):
        asyncio.run(agent._env_runner(agent, replace(task, question="unreviewed")))
    assert runtime.RELAY.get() is None
    gateway.assert_awaited_once()


def test_swe_root_override_and_payload_tamper_detection(manifest_root, monkeypatch):
    split = "swegym_lite_train"
    task_id = "swegym_lite_train-000"
    monkeypatch.setenv("FLASHRSI_SUITE_HARBOR_TRACK_ROOT", str(manifest_root))
    assert harbor_suite.load_manifest(split)["count"] == 230
    track = harbor_suite.open_split(split, [task_id])
    assert [task.task_id for task in track.train] == [task_id]
    payload = manifest_root / "prepared" / split / f"{task_id}.json"
    payload.write_bytes(payload.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SWE payload integrity mismatch"):
        harbor_suite.open_split(split, [task_id])


@pytest.mark.parametrize("name", sorted(_suite.SEALED))
def test_every_sealed_split_stays_locked_without_touching_assets(tmp_path, name):
    with pytest.raises(PermissionError):
        harbor_suite.HarborSuite({"name": name, "root": str(tmp_path)})
    if name in harbor_suite.SPLITS:
        with pytest.raises(PermissionError):
            harbor_suite.load_manifest(name, root=tmp_path)
        with pytest.raises(PermissionError):
            harbor_suite.open_split(name, [], root=tmp_path)


def _docker_result(args, stdout="", stderr="", code=0):
    if "flashrsi-exec-status=" in " ".join(args) and code == 0 and not stdout:
        stdout = "\nflashrsi-exec-status=0\n"
    return CompletedProcess(args, code, stdout, stderr)


def _solver_docker(calls, state, break_phase):
    async def docker(*args, timeout, check=True, binary=False):
        calls.append((args, timeout))
        if args[:2] == ("network", "create"):
            state["network"] = args[-1]
            if break_phase == "daemon":
                raise RuntimeError("Cannot connect to the Docker daemon")
        if args[0] == "create" and break_phase == "missing_image":
            raise RuntimeError("No such image")
        if args[0] == "inspect":
            if args[2] == "{{.Image}}":
                return _docker_result(args, "sha256:fixture")
            networks = {state["network"]: {}}
            if break_phase == "bridge":
                networks["bridge"] = {}
            return _docker_result(args, json.dumps(networks))
        if args[:2] == ("network", "inspect"):
            return _docker_result(args, "true")
        if args[:3] == ("network", "disconnect", "bridge"):
            assert check
            raise RuntimeError("bridge disconnect denied")
        if args[0] == "start" and args[-1].endswith("-proxy") and break_phase == "proxy":
            raise RuntimeError("proxy failed to start")
        if args[0] == "exec":
            joined = " ".join(args)
            if "Solver has a default network route" in joined and break_phase == "route":
                raise RuntimeError("Solver has a default network route")
            if "git reset --hard" in joined and break_phase == "setup":
                raise RuntimeError("base commit missing")
            if "runpy.run_module" in joined:
                if break_phase == "agent_exec":
                    raise RuntimeError("Error response from daemon: container is not running")
                if break_phase == "episode_timeout":
                    raise TimeoutError
            if "cat" in args:
                return _docker_result(args, '{"status": "finished"}')
            if "diff" in args:
                return _docker_result(args, _git_patch("src/fix.py"))
        return _docker_result(args)
    return docker


@pytest.mark.parametrize("break_phase", [None, "bridge", "route", "proxy", "daemon", "missing_image", "setup", "agent_exec", "episode_timeout"])
def test_swe_solver_offline_start_and_infra_classification(monkeypatch, break_phase):
    calls = []
    staged = []
    task = _task_from_row(_swe_row(), "train")
    agent = _agent(timeout=900)
    state = {"network": None}

    docker = _solver_docker(calls, state, break_phase)
    monkeypatch.setattr(swe, "_docker", docker)
    monkeypatch.setattr(swe, "_stage", lambda _path, _agent, config, _question: staged.append(config))
    if break_phase not in (None, "episode_timeout"):
        with pytest.raises(InfraError, match="fixture-0001"):
            asyncio.run(swe._solve(agent, task, suite_track=True))
        if break_phase != "agent_exec":
            assert all("runpy.run_module" not in " ".join(args) for args, _ in calls)
        return

    trajectory = asyncio.run(swe._solve(agent, task, suite_track=True))
    assert staged[0]["timeout"] == 3600
    solver = next(args for args, _ in calls if args[0] == "create")
    assert f"--network={state['network']}" in solver
    assert {"--cpus=2", "--memory=4g", "--pids-limit=512", "--shm-size=64m", "--storage-opt=size=20g"} <= set(solver)
    positions = {name: next(i for i, (args, _) in enumerate(calls) if name in " ".join(args))
                 for name in ("{{.Internal}}", "Solver has a default network route", "runpy.run_module")}
    assert positions["{{.Internal}}"] < positions["Solver has a default network route"] < positions["runpy.run_module"]
    assert calls[positions["runpy.run_module"]][1] == 3600
    if break_phase == "episode_timeout":
        assert trajectory.status == "error" and "episode timed out" in trajectory.error_message
    else:
        assert trajectory.messages[-1]["type"] == "SWEBenchPatch"


@pytest.fixture
def evaluation(monkeypatch):
    row = _swe_row()
    task = _task_from_row(row, "train")
    agent = _agent()
    calls = []
    state = {"test": None, "apply": None, "prepare": None, "cleanup": None, "resolved": True}

    async def docker(*args, timeout, check=True, binary=False):
        calls.append((args, timeout))
        joined = " ".join(args)
        if args[0] == "create" and state["prepare"]:
            raise RuntimeError(state["prepare"])
        if "git apply --verbose" in joined and state["apply"]:
            if state["apply"] != "patch does not apply":
                raise RuntimeError(state["apply"])
            return _docker_result(args, stdout="\nflashrsi-exec-status=1\n", stderr=state["apply"])
        if "exec /bin/bash " in joined:
            if state["test"] == "timeout":
                raise TimeoutError
            if state["test"]:
                if state["test"] != "test suite failed":
                    raise RuntimeError(state["test"])
                return _docker_result(args, stdout="\nflashrsi-exec-status=1\n", stderr=state["test"])
        if "wc" in args:
            return _docker_result(args, "5\n")
        if "cat" in args:
            return _docker_result(args, b"logs\n")
        if args[0] == "rm" and state["cleanup"]:
            return _docker_result(args, stderr=state["cleanup"], code=1)
        return _docker_result(args)

    async def official(action, root, timeout):
        calls.append(((action,), timeout))
        if action == "prepare":
            (root / "eval.sh").write_text("#!/bin/bash\n")
            return {"apply_commands": ["git apply --verbose"]}
        return {"resolved": state["resolved"], "patch_successfully_applied": True}

    monkeypatch.setattr(swe, "_docker", docker)
    monkeypatch.setattr(swe, "_official", official)
    benchmark = SWEBench({"split": {"train": 1, "dev": 0, "test": 0}})
    monkeypatch.setattr(benchmark, "_iter_tasks", lambda: iter([task]))
    benchmark.load()

    def execute(patch):
        trajectory = Trajectory(task=task, messages=({"model_patch": patch, "image_id": "sha256:fixture"},))
        result = asyncio.run(swe._evaluate(agent, task, trajectory, row, swe._evaluation_options({})))
        return benchmark.grade((result,))[0]

    return execute, calls, state


@pytest.mark.parametrize("path", [
    ".git/hooks/pre-commit", "run_tests.sh", "eval.sh", "swebench/harness/grading.py",
    "pip.conf", ".condarc", "pytest.ini", "tests/unit/conftest.py",
    "tests/test_protected.py", "tests",
])
def test_rejects_protected_paths_before_creating_verifier(evaluation, path):
    execute, calls, _ = evaluation
    result = execute(_git_patch(path))
    assert result.metrics["resolved"] == 0
    assert result.messages[-1]["rejected_reason"] == "modified_trusted_test_or_harness_files"
    assert not calls


def test_renaming_protected_file_also_rejected(evaluation):
    execute, calls, _ = evaluation
    patch = (
        "diff --git a/tests/test_protected.py b/src/renamed.py\n"
        "similarity index 100%\nrename from tests/test_protected.py\nrename to src/renamed.py\n"
    )
    assert execute(patch).messages[-1]["rejected_reason"] == "modified_trusted_test_or_harness_files"
    assert not calls


@pytest.mark.parametrize("outcome", ["resolved", "unresolved", "apply", "timeout", "empty", "invalid"])
def test_verifier_independent_clocks_and_candidate_outcomes(evaluation, outcome):
    execute, calls, state = evaluation
    if outcome == "unresolved":
        state["resolved"] = False
        state["test"] = "test suite failed"
    if outcome == "timeout":
        state["test"] = "timeout"
    if outcome == "apply":
        state["apply"] = "patch does not apply"
    patch = {"empty": "", "invalid": "not a patch"}.get(outcome, _git_patch("src/fix.py"))
    result = execute(patch)
    assert result.metrics["resolved"] == (1 if outcome == "resolved" else 0)
    if outcome in ("empty", "invalid"):
        assert not calls
        return
    create, deadline = next((args, limit) for args, limit in calls if args[0] == "create")
    assert deadline == 600 and "--network=none" in create
    assert {"--cpus=2", "--memory=4g", "--pids-limit=512", "--shm-size=64m", "--storage-opt=size=20g"} <= set(create)
    assert next(limit for args, limit in calls if args[0] == "rm") == 60
    assert (("prepare",), 600) in calls
    if outcome == "apply":
        assert result.messages[-1]["outcome"] == "candidate_patch_did_not_apply"
    else:
        assert next(limit for args, limit in calls if "exec /bin/bash " in " ".join(args)) == 1800
        if outcome == "timeout":
            assert result.messages[-1]["outcome"] == "official_test_timeout"
        else:
            assert (("grade",), 60) in calls


@pytest.mark.parametrize("phase,diagnostic", [
    ("prepare", "No such image"), ("apply", "OCI runtime exec failed"),
    ("test", "Error response from daemon: container is not running"),
    ("cleanup", "Cannot connect to the Docker daemon"),
])
def test_swe_verifier_infra_does_not_score(evaluation, phase, diagnostic):
    execute, _, state = evaluation
    state[phase] = diagnostic
    with pytest.raises(InfraError, match=diagnostic):
        execute(_git_patch("src/fix.py"))


def test_resource_and_timeout_overrides(monkeypatch):
    monkeypatch.setenv("FLASHRSI_SWE_VERIFIER_TEST_TIMEOUT", "2150")
    monkeypatch.setenv("FLASHRSI_SWE_VERIFIER_SHM_SIZE", "128m")
    monkeypatch.setenv("FLASHRSI_SWE_SOLVER_CPUS", "3")
    verifier = swe._evaluation_options({})
    assert verifier["evaluation_timeout"] == 2150
    assert "--shm-size=128m" in verifier["resources"]
    assert swe._resource_flags({"memory": "6g", "pids_limit": 300, "storage": "20g"}, "solver") == (
        "--cpus=3", "--memory=6g", "--pids-limit=300", "--shm-size=64m", "--storage-opt=size=20g",
    )
