"""Verify official benchmark clocks, resources, and native shell dispatch.

Synthetic fixtures exercise host-side wiring without provider credentials.
The tests cover both supported container shell adapters.
"""

from __future__ import annotations

import asyncio
import json
import runpy
import shlex
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import CompletedProcess
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml

from flashrsi.agents.claude_code import ClaudeCode
from flashrsi.agents.codex import Codex
from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.benchmarks import _suite
from flashrsi.benchmarks import harbor_suite as terminal
from flashrsi.benchmarks.design_suite import (
    TDC_METRICS,
    TDCADMET,
    EngDesignOpen,
    FrontierEng,
    KernelBench,
)
from flashrsi.benchmarks.workspace_suite import (
    WorkspaceBenchmark,
    WorkspaceSuite,
    summarize_harvey,
)
from flashrsi.core import InfraError, Task, Trajectory
from flashrsi.integrations import admet, design_suite, engdesign, kernelbench
from flashrsi.integrations import sandbox_runtime as runtime
from flashrsi.integrations import swe_bench as swe
from flashrsi.integrations import workspace_suite as workspace
from flashrsi.integrations.harbor_suite import (
    _ANTHROPIC_RELAY_KEY,
    _RELAY_KEY,
    _REVERSE_PROXY,
    _RUNTIME,
    _create_task_container,
    _host_runtime,
)

SHELLS = (ClaudeCode, Codex)
DUMMY_SECRET = "fixture-key-that-must-stay-on-the-host"


def make_agent(shell, **config):
    return shell({
        "llm": {
            "model_name": "deepseek-flash",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": DUMMY_SECRET,
            "reasoning_effort": "high",
        },
        **config,
    })


def snapshot_request(directory):
    request_path = Path(directory) / "request.json"
    assert request_path.stat().st_mode & 0o777 == 0o600
    body = request_path.read_text()
    assert DUMMY_SECRET not in body
    request = json.loads(body)
    return request["agent_class"], yaml.safe_load(request["config_yaml"])


def test_managed_runtime_paths_are_minimal_and_find_bundled_tools():
    pytest.importorskip("claude_agent_sdk")
    python, mounts = _host_runtime()
    assert Path(python) == Path(sys.executable)
    assert mounts[sys.prefix] == Path(sys.prefix)
    assert Path(sys.executable).resolve().parents[1] in mounts.values()
    assert mounts[f"{_RUNTIME}/flashrsi"] == Path(swe.__file__).resolve().parents[1]
    import claude_agent_sdk
    launcher = Path(claude_agent_sdk.__file__).parent / "_bundled/claude"
    assert launcher.is_file()
    assert launcher.is_relative_to(Path(sys.prefix))
    stock_codex = Path.home() / ".local/lib/codex/current/bin/codex"
    if stock_codex.is_file():
        assert mounts[f"{_RUNTIME}/bin/codex"] == stock_codex.resolve()
    assert all(source.name not in (".git", ".env", "FlashRSI") for source in mounts.values())


def _harbor_trial_runner(harbor, config, paths, task, agent, commands):
    async def exec_command(command):
        commands.append(command)
        return SimpleNamespace(return_code=0)

    async def run():
        adapter = harbor._SnapshotAgent(logs_dir=paths.agent_dir, **config.agent.kwargs)
        environment = SimpleNamespace(
            upload_dir=AsyncMock(), exec=exec_command,
            capabilities=SimpleNamespace(mounted=True),
        )
        await adapter.setup(environment)
        context = SimpleNamespace()
        await adapter.run(task.question, environment, context)
        environment.upload_dir.assert_awaited_once()
        assert context.metadata == {"flashrsi_trajectory": "flashrsi.json"}
        (paths.agent_dir / "flashrsi.json").write_text(json.dumps({
            "messages": [{"type": "RunConfig", "snapshot_id": agent.snapshot.id}],
        }))
        return SimpleNamespace(id="fixture", exception_info=None,
                               verifier_result=SimpleNamespace(rewards={"reward": 1}))
    return run


@pytest.mark.parametrize("shell", SHELLS, ids=lambda cls: cls.name)
@pytest.mark.parametrize("options,turns", [({}, None), ({"max_turns": 7}, 7)])
def test_harbor_native_mounts_clocks_and_launch(tmp_path, monkeypatch, shell, options, turns):
    pytest.importorskip("harbor")
    from harbor.models.trial.paths import TrialPaths

    from flashrsi.integrations import harbor

    path = tmp_path / "tiny-open-task"
    (path / "tests").mkdir(parents=True)
    (path / "environment").mkdir()
    (path / "instruction.md").write_text("Use this open fixture.")
    (path / "environment/Dockerfile").write_text("FROM scratch\n")
    (path / "tests/test.sh").write_text("#!/bin/sh\n")
    (path / "task.toml").write_text(
        "[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 113\n"
        "[environment]\nallow_internet = false\n"
    )
    task = Task("open-fixture", "Use this open fixture.", env=str(path),
                metadata={"suite_require_local_image": False})
    agent = make_agent(shell, timeout=28, harbor={"artifacts_dir": str(tmp_path / "saved")}, **options)
    commands = []
    mounts_seen = {}

    @contextmanager
    def relay(path, model, *, unix, **kwargs):
        assert unix and model == "deepseek-flash"
        Path(path).touch()
        yield SimpleNamespace()

    async def create(cfg):
        native = harbor.HarborTask(path)
        paths = TrialPaths(cfg.trials_dir / "fixture")
        paths.mkdir()
        trial = SimpleNamespace(config=cfg, task=native, paths=paths, add_hook=lambda *_: None)
        trial._resolve_timeout_sec = harbor.Trial._resolve_timeout_sec.__get__(trial)
        assert cfg.timeout_multiplier == cfg.verifier_timeout_multiplier == 1.0
        assert cfg.agent.override_timeout_sec is None
        assert cfg.verifier.override_timeout_sec is None
        assert harbor.Trial._compute_agent_timeout_sec(trial) == 900
        assert harbor.Trial._compute_verifier_timeout_sec(trial) == 113
        volumes = json.loads(cfg.environment.extra_docker_compose[0].read_text())
        for mount in volumes["services"]["main"]["volumes"]:
            assert mount["read_only"] is True
            mounts_seen[mount["target"]] = mount["source"]
        assert sys.prefix in mounts_seen and f"{_RUNTIME}/flashrsi" in mounts_seen
        assert (Path(mounts_seen[harbor._RELAY_MOUNT]) / "relay.sock").is_file()
        identifier, config = snapshot_request(cfg.agent.kwargs["payload_dir"])
        assert identifier.endswith(":" + shell.__name__)
        assert config.get("max_turns") == turns
        assert config["timeout"] == 900
        assert config["llm"]["base_url"].startswith(harbor._RELAY_ENDPOINT)
        assert config["llm"]["base_url"].endswith("/anthropic") == (shell is ClaudeCode)
        assert "max_tokens" not in config["llm"]

        trial.run = _harbor_trial_runner(harbor, cfg, paths, task, agent, commands)
        return trial

    monkeypatch.setattr(harbor.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(runtime, "relay", relay)
    monkeypatch.setattr(harbor.Trial, "create", create)
    result = asyncio.run(harbor._run(agent, task))
    assert result.status == "finished", result.error_message
    assert shlex.split(commands[0]) == [sys.executable, "-I", "-B", "--version"]
    command = commands[1]
    assert f"ANTHROPIC_BASE_URL={harbor._RELAY_ENDPOINT}/anthropic" in command
    assert f"OPENAI_BASE_URL={harbor._RELAY_ENDPOINT}/v1" in command
    assert "runpy.run_module" in command and "relay.sock" in command
    assert DUMMY_SECRET not in command
    assert agent.config["timeout"] == 28


@pytest.mark.parametrize("name,role,expected", [
    ("tb21_train", "train", 1),
    ("tb21_val", "val", 5),
    ("tb41_sealed", "sealed", 5),
    ("tbs01_sealed", "sealed", 3),
])
def test_harbor_attempt_defaults_and_override(name, role, expected):
    assert terminal.SPLITS[name].role == role
    assert terminal._attempts(name, {}) == expected
    assert terminal._attempts(name, {"smoke": True}) == 1
    assert terminal._attempts(name, {"n_attempts": 1}) == 1
    assert terminal.HarborSuite({"name": "tb21_clean"}).metadata["n_attempts"] == 5
    assert terminal.HarborSuite({"name": "tb21_clean", "smoke_whitelist_only": True}).metadata["n_attempts"] == 1


def docker_result(args, output="", *, status=0):
    return CompletedProcess(args, status, output, "")


@pytest.mark.parametrize("shell", SHELLS, ids=lambda cls: cls.name)
def test_swe_native_solver_readonly_runtime_sidecar_and_clocks(monkeypatch, shell):
    agent = make_agent(shell, timeout=28)
    task = Task("open-swe-fixture", "Repair the example.", env="fixture:staged",
                metadata={"base_commit": "1" * 40, "task_checksum": "fixture"})
    calls = []
    seen = {}
    proxy = SimpleNamespace(server_address=("172.17.0.1", 32100))

    async def docker(*args, timeout, check=True):
        calls.append((args, timeout))
        if args[:2] == ("network", "create"):
            assert "--internal" in args
            seen["network"] = args[-1]
        if args[:2] == ("network", "inspect"):
            return docker_result(args, "true")
        if args[:3] == ("inspect", "--format", "{{.Image}}"):
            return docker_result(args, "sha256:fixture")
        if args[:3] == ("inspect", "--format", "{{json .NetworkSettings.Networks}}"):
            return docker_result(args, json.dumps({seen["network"]: {}}))
        if args[0] == "cp":
            seen["request"] = snapshot_request(args[1])
        if "trajectory.json" in args[-1] and "cat" in args:
            return docker_result(args, '{"status":"finished"}')
        if "runpy.run_module" in args[-1]:
            return docker_result(args, "\nflashrsi-exec-status=0\n")
        return docker_result(args)

    monkeypatch.setattr(swe, "_docker", docker)
    admit = AsyncMock()
    monkeypatch.setattr(runtime, "admit", admit)
    token = runtime.RELAY.set(proxy)
    try:
        result = asyncio.run(swe._solve(agent, task, suite_track=True))
    finally:
        runtime.RELAY.reset(token)

    assert result.messages[-1]["type"] == "SWEBenchPatch"
    identifier, config = seen["request"]
    assert identifier.endswith(":" + shell.__name__)
    assert config["timeout"] == 3600 and config.get("max_turns") is None
    assert "max_tokens" not in config["llm"]
    suffix = "/anthropic" if shell is ClaudeCode else "/v1"
    assert config["llm"]["base_url"] == swe._SIDECAR + suffix
    create, setup = next((args, timeout) for args, timeout in calls if args[0] == "create")
    assert setup == 600 and f"--network={seen['network']}" in create
    assert {"--cpus=2", "--memory=4g", "--pids-limit=512", "--shm-size=64m"} <= set(create)
    binds = [arg for arg in create if arg.startswith("--mount")]
    assert all(bind.endswith(",readonly") for bind in binds)
    assert any(f"dst={sys.prefix}," in bind for bind in binds)
    assert any(f"dst={_RUNTIME}/flashrsi," in bind for bind in binds)
    position = next(i for i, (args, _) in enumerate(calls) if "Solver has a default network route" in args[-1])
    execute = next(i for i, (args, _) in enumerate(calls) if "runpy.run_module" in args[-1])
    assert position < execute
    args, timeout = calls[execute]
    assert timeout == 3600
    assert f"--env=ANTHROPIC_BASE_URL={swe._SIDECAR}/anthropic" in args
    assert f"--env=OPENAI_BASE_URL={swe._SIDECAR}/v1" in args
    assert f"--env=ANTHROPIC_API_KEY={_ANTHROPIC_RELAY_KEY}" in args
    assert f"--env=OPENAI_API_KEY={_RELAY_KEY}" in args
    assert not any(DUMMY_SECRET in arg for call, _ in calls for arg in call)
    admit.assert_awaited_once()
    assert {timeout for args, timeout in calls if args[0] == "rm"} == {60}


@pytest.mark.parametrize("shell", SHELLS, ids=lambda cls: cls.name)
def test_swe_native_infrastructure_errors_propagate_redacted(shell, monkeypatch):
    agent = make_agent(shell)
    task = Task("missing-image-fixture", "Open fixture", env="missing:fixture")
    server = SimpleNamespace(server_address=("172.17.0.1", 32100))

    async def docker(*args, timeout, check=True):
        if args[0] == "create":
            raise RuntimeError(f"missing image, details {DUMMY_SECRET}")
        return docker_result(args)

    monkeypatch.setattr(swe, "_docker", docker)
    token = runtime.RELAY.set(server)
    try:
        with pytest.raises(InfraError, match="missing-image-fixture") as exc:
            asyncio.run(swe._solve(agent, task, suite_track=True))
        assert DUMMY_SECRET not in str(exc.value)
    finally:
        runtime.RELAY.reset(token)


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        self.server.requests.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


@contextmanager
def http_server(handler, *, requests=None):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        server.requests = requests
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", server.server_port
        finally:
            server.shutdown()
            thread.join()


def test_swe_sidecar_forwards_only_solver_api_paths(monkeypatch):
    import http.server

    from flashrsi.integrations import _model_proxy

    seen = []
    handlers = []

    def capture_handler(_address, handler):
        return control(handler)

    @contextmanager
    def control(handler):
        handlers.append(handler)
        yield SimpleNamespace(serve_forever=lambda: None)

    with http_server(Upstream, requests=seen) as (relay, port):
        allowed = [["http", "127.0.0.1", port]]
        with monkeypatch.context() as patch:
            patch.setattr(http.server, "ThreadingHTTPServer", capture_handler)
            patch.setattr(runpy, "run_path", lambda _: {"_ProxyHandler": _model_proxy._ProxyHandler})
            patch.setattr(sys, "argv", ["-c", json.dumps(allowed), relay])
            exec(_REVERSE_PROXY, {})
        handler = handlers.pop()
        with http_server(handler) as (sidecar, _):
            routes = (
                ("/anthropic/v1/messages", "/anthropic/v1/messages"),
                ("/v1/chat/completions", "/v1/chat/completions"),
                ("/v1/responses", "/v1/responses"),
                ("/responses?stream=true", "/v1/responses?stream=true"),
            )
            for path, forwarded in routes:
                with urlopen(Request(sidecar + path, b"{}"), timeout=2) as result:
                    assert result.read() == b"ok"
                assert seen[-1] == (forwarded, b"{}")
            with pytest.raises(HTTPError, match="404"):
                urlopen(Request(sidecar + "/arbitrary", b"{}"), timeout=2)
            with pytest.raises(HTTPError, match="403"):
                urlopen(Request(sidecar + "/", method="CONNECT"), timeout=2)
            assert len(seen) == len(routes)


def test_swe_verifier_offline_and_protected_patch_rejection():
    options = swe._evaluation_options({})
    assert (options["setup_timeout"], options["evaluation_timeout"], options["cleanup_grace"]) == (600, 1800, 60)
    docker = AsyncMock()
    asyncio.run(_create_task_container("fixture", "fixture", "none", options["resources"], 600, docker=docker))
    assert "--network=none" in docker.await_args_list[0].args
    assert not any("--mount" in arg for arg in docker.await_args_list[0].args)
    test = ("tests", "test_protected.py")
    assert swe._protected_path(test, {test})
    assert swe._protected_path((".git", "config"), set())
    assert swe._protected_path(("harness", "verify.py"), set())
    assert not swe._protected_path(("src", "fix.py"), set())


@pytest.mark.parametrize("shell", SHELLS, ids=lambda cls: cls.name)
@pytest.mark.parametrize("profile,split,turn_override,expected", [
    ("default", "clawgym_train", {}, (660, None)),
    ("long", "clawgym_val", {}, (3000, None)),
    ("default", "harvey_val", {}, (3600, 200)),
    ("default", "harvey_train", {"harvey_max_turns": 250}, (3600, 250)),
    ("default", "harvey_val", {"harvey_max_turns": None}, (3600, None)),
])
def test_workspace_shells_in_both_families(tmp_path, monkeypatch, shell, profile, split, turn_override, expected):
    (tmp_path / "input").mkdir()
    (tmp_path / "input/readme").write_text("allowed input")
    (tmp_path / "grader").mkdir()
    bench = WorkspaceSuite({"name": "workspace_dualgate", "profile": profile, **turn_override})
    monkeypatch.setattr(bench, "backend", lambda _: SimpleNamespace(name=split, lookup=lambda _: ({}, tmp_path)))
    check = workspace.WorkspaceExecutor(bench)
    monkeypatch.setattr(check, "_claw", AsyncMock(return_value={"score": 1.0}))
    monkeypatch.setattr(check, "_harvey", AsyncMock(return_value={"score": 1.0}))
    agent = make_agent(shell, timeout=28)

    async def launch(target, prompt, folder, **kwargs):
        assert target is agent
        assert (target.config["timeout"], target.config.get("max_turns")) == expected
        assert "max_tokens" not in target.config["llm"]
        assert kwargs["image"] == workspace.IMAGE and kwargs["workdir"] == "/workspace"
        assert {item.name for item in folder.iterdir()} == {"readme"}
        return Trajectory(), "fixture"

    monkeypatch.setattr(runtime, "solver", launch)
    monkeypatch.setattr(workspace, "_workspace_image", AsyncMock(return_value=workspace.IMAGE))
    result = asyncio.run(check.run(agent, Task("fixture", "Open workspace task")))
    assert result.messages[-1]["solver_timeout"] == expected[0]
    assert agent.config["timeout"] == 28 and "max_turns" not in agent.config


def test_workspace_judge_modes_and_weighted_metrics():
    claw = WorkspaceSuite({"name": "clawgym"})
    harvey = WorkspaceSuite({"name": "harvey_workspace"})
    assert claw.metadata["clawgym"] == {
        "solver_timeout": 660, "long_episode_timeout": 3000, "checker_timeout": 120,
        "judge_timeout": 180, "max_turns": None,
    }
    assert harvey.settings.mode_for("harvey_val") == "per_criterion"
    assert harvey.settings.mode_for("harvey_train") == "batched"
    assert harvey.primary_metric == WorkspaceBenchmark("harvey_val").primary_metric == "micro_avg"
    assert summarize_harvey([
        {"passed_criteria": 1, "total_criteria": 1},
        {"passed_criteria": 1, "total_criteria": 3},
    ]) == {"passed_criteria": 2, "total_criteria": 4, "micro_avg": .5,
           "macro_avg": pytest.approx(2 / 3), "all_pass": .5}


@pytest.mark.parametrize("shell", SHELLS, ids=lambda cls: cls.name)
@pytest.mark.parametrize("overrides,expected", [
    ({}, (5400, 420, 250)),
    ({"max_turns": None}, (5400, 420, None)),
    ({"solver_timeout": 9000, "grader_timeout": 500, "max_turns": 200}, (9000, 500, 200)),
])
def test_engdesign_native_solver_and_grader(tmp_path, monkeypatch, shell, overrides, expected):
    benchmark = EngDesignOpen({"name": "engdesign_open", "root": tmp_path, **overrides})
    benchmark.manifest_data = {"commit": "fixture", "files": {"fixture": {}}}
    monkeypatch.setattr(benchmark, "directory", lambda _: tmp_path)
    monkeypatch.setattr(engdesign.subprocess, "check_output", lambda *_: b"")
    monkeypatch.setattr(engdesign.subprocess, "run", lambda *_, **__: None)
    agent = make_agent(shell)

    async def launch(target, prompt, work_path, **kwargs):
        assert target is agent
        assert kwargs["image"].split(":")[0] == "flashrsi-eng"
        assert "candidate.json" in prompt
        assert (target.config["timeout"], target.config.get("max_turns")) == (expected[0], expected[2])
        assert "max_tokens" not in target.config["llm"]
        (work_path / "candidate.json").write_text("{}")
        return Trajectory(), "fixture-solver"

    def evaluate(image, _command, mounts, timeout, **kwargs):
        assert image.split(":")[0] == "flashrsi-eng" and timeout == expected[1]
        assert mounts[0][1] == "/grader" and kwargs["stdin"] == "{}"
        return {"score": .5}, "fixture-grader"

    monkeypatch.setattr(runtime, "solver", launch)
    monkeypatch.setattr(engdesign, "docker_json", evaluate)
    _, verdict = asyncio.run(engdesign.engdesign(agent, benchmark, Task("fixture", "Open design fixture")))
    assert verdict["score"] == .5
    assert benchmark.metadata["val_repeats"] == 4
    assert benchmark.metadata["min_val_gain"] == 0.020


def test_frontier_per_task_v1_evaluator_timeouts(tmp_path):
    checkout = tmp_path / "Frontier-Engineering"
    batch_path = checkout / "frontier_eval/conf/batch/v1.yaml"
    batch_path.parent.mkdir(parents=True)
    rows = {}
    entries = []
    for timeout in (300, 600, 1200):
        name = f"open-{timeout}"
        (checkout / "benchmarks" / name).mkdir(parents=True)
        rows[name] = {"path": name, "split": "val", "task_spec": "Open frontier test"}
        entries.append({"label": name, "overrides": [f"algorithm.oe.evaluator.timeout={timeout}"]})
    batch_path.write_text(yaml.safe_dump({"tasks": entries}))
    manifest = tmp_path / "fixture.json"
    manifest.write_text(json.dumps({"tasks": rows}))
    benchmark = FrontierEng({
        "name": "frontier_eng", "root": tmp_path, "mode": "real_tasks", "frontier_manifest": manifest,
    })
    benchmark.load()
    assert {task.metadata["timeout_s"] for task in benchmark.dev} == {300, 600, 1200}
    assert benchmark.metadata["oracle_budget"] == 100 and benchmark.metadata["max_tokens"] == 32768


@pytest.mark.parametrize("cls,name,execute,budget,cap", [
    (FrontierEng, "frontier_eng", "frontier", 100, 32768),
    (TDCADMET, "tdc_admet", "admet", 20, 65536),
    (KernelBench, "kernelbench", "kernel", 10, 65536),
])
@pytest.mark.parametrize("override", [None, 4096])
def test_program_search_gets_benchmark_model_cap(cls, name, execute, budget, cap, override, monkeypatch):
    benchmark = cls({"name": name})
    task = Task("open-fixture", "Propose an open candidate.", env=name)
    benchmark.set_tasks((task,), ())
    llm = {"model_name": "deepseek-flash"}
    if override is not None:
        llm["max_tokens"] = override
    agent = ProgramSearchTarget({"llm": llm})

    async def run(target, active, received):
        assert active is benchmark and received == task
        assert target.config["llm"]["max_tokens"] == (override or cap)
        return Trajectory(), {"score": 1.0}

    monkeypatch.setattr(design_suite, execute, run)
    design_suite.bind(agent, benchmark)
    trajectory = asyncio.run(agent._env_runner(agent, task))
    assert trajectory.messages[-1]["score"] == 1.0
    assert benchmark.metadata["oracle_budget"] == budget and benchmark.metadata["max_tokens"] == cap
    assert agent.config["llm"].get("max_tokens") == override


def test_tdc_scaffold_seeds_and_metric_mapping():
    benchmark = TDCADMET({"name": "tdc_admet"})
    assert benchmark.metadata["split_method"] == "scaffold"
    assert admet.admet_seeds({}) == benchmark.metadata["seeds"] == (1, 2, 3, 4, 5)
    assert admet.admet_seeds({"smoke": True}) == (1,)
    expected = {
        ("MAE", "minimize"): (
            "caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az", "ld50_zhu",
        ),
        ("AUROC", "maximize"): (
            "hia_hou", "pgp_broccatelli", "bioavailability_ma", "bbb_martins",
            "cyp3a4_substrate_carbonmangels", "herg", "ames", "dili",
        ),
        ("AUPRC", "maximize"): (
            "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
            "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith",
        ),
        ("Spearman", "maximize"): (
            "vdss_lombardo", "half_life_obach", "clearance_hepatocyte_az", "clearance_microsome_az",
        ),
    }
    assert len(TDC_METRICS) == 22
    assert TDC_METRICS == {key: metric for metric, endpoints in expected.items() for key in endpoints}


def test_kernel_default_uses_ten_oracle_slots(monkeypatch):
    benchmark = KernelBench({"name": "kernelbench", "gpu_queue": "http://invalid.local",
                             "device_uuid": "GPU-fixture", "require_real_cuda": False})
    task = Task("open-kernel", "Implement a synthetic kernel.", env="kernelbench")
    benchmark.set_tasks((task,), ())
    agent = ProgramSearchTarget()
    monkeypatch.setattr(agent, "propose", AsyncMock(return_value=(
        {"code": "class ModelNew:\n    def forward(self, x):\n        return x\n"}, {},
    )))
    def submit(_queue, **job):
        return {"job_id": job["job_id"], "score": .5}
    monkeypatch.setattr(kernelbench.GPUTimingClient, "submit", submit)
    _, result = asyncio.run(kernelbench.kernel(agent, benchmark, task))
    assert result["budget"] == result["gpu_calls"] == agent.propose.await_count == 10
    assert kernelbench.oracle_budget({"smoke": True}, 10, 100) == 1
    assert (kernelbench.GPU_COMPILE_TIMEOUT, kernelbench.GPU_RUN_TIMEOUT) == (180, 180)
    assert (kernelbench.GPU_CORRECT_TRIALS, kernelbench.GPU_PERF_TRIALS) == (5, 100)
    assert kernelbench.GPU_TOLERANCES == {"fp32": 1e-4, "fp16": 1e-2, "bf16": 1e-2}
    assert (kernelbench.GPU_WARMUP, kernelbench.GPU_L2_BYTES) == (3, 256 * 1024 * 1024)


@pytest.mark.parametrize("name", sorted(_suite.SEALED))
def test_sealed_guard_checks_before_asset_access(tmp_path, name):
    from flashrsi.runner import _BENCHMARKS

    for options in ({}, {"allow_sealed_test": True}):
        with pytest.raises(PermissionError):
            _BENCHMARKS[name]({"name": name, "root": tmp_path, **options})
        if name in terminal.SPLITS:
            with pytest.raises(PermissionError):
                terminal.open_split(name, ["untrusted"], root=tmp_path, **options)


@pytest.mark.parametrize("name", sorted(_suite.ACTIVE))
@pytest.mark.parametrize("split", ["test", "sealed"])
def test_active_entry_points_reject_sealed_split(tmp_path, name, split):
    from flashrsi.runner import _BENCHMARKS

    with pytest.raises(PermissionError):
        _BENCHMARKS[name]({"name": name, "root": tmp_path, "split": split})


def test_sealed_endpoint_alias_denies_before_access(tmp_path):
    with pytest.raises(PermissionError):
        TDCADMET({"name": "tdc_admet_sealed", "root": tmp_path})
    with pytest.raises(PermissionError):
        TDCADMET({"name": "tdc_admet", "root": tmp_path}).dataset(Task("herg", "closed"))


@pytest.mark.parametrize("name,benchmark", [
    ("tb21_clean", terminal.HarborSuite), ("swegym_lite", terminal.HarborSuite),
    ("clawgym", WorkspaceSuite), ("harvey_workspace", WorkspaceSuite),
    ("workspace_dualgate", WorkspaceSuite), ("engdesign_open", EngDesignOpen),
])
def test_interactive_benchmarks_expose_two_primary_shells(name, benchmark):
    assert benchmark({"name": name}).metadata["agent_shells"] == ("claude_code", "codex")
    assert "pi" in _suite.OPTIONAL_AGENT_SHELLS
