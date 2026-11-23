"""Verify suite registration, sealed guards, and native dispatch."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import httpx
import pytest

from flashrsi import _startup, runner
from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.algorithms.promotion import WorkspaceGate
from flashrsi.benchmarks import _suite
from flashrsi.config import load_config
from flashrsi.core import InfraError, Task, Trajectory

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs/suite").glob("*.yaml"))
ACTIVE = [p for p in CONFIGS if not p.stem.endswith("_sealed")]
SEALED = [p for p in CONFIGS if p.stem.endswith("_sealed")]
ASSETS = _suite.default_envs_root().is_dir()

TOKENS = {
    "prompt_tokens": 101,
    "completion_tokens": 202,
    "completion_tokens_details": {"reasoning_tokens": 190},
    "prompt_cache_hit_tokens": 80,
}


def test_suite_root_precedence_and_checkout_fallback(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    module = checkout / "flashrsi" / "benchmarks" / "_suite.py"
    monkeypatch.setattr(_suite, "__file__", str(module))
    monkeypatch.delenv("FLASHRSI_SUITE_ENVS_ROOT", raising=False)
    track_var = "FLASHRSI_SUITE_WORKSPACE_TRACK_ROOT"
    monkeypatch.delenv(track_var, raising=False)

    assert _suite.default_envs_root() == checkout / "artifacts" / "envs"
    assets = tmp_path / "envs"
    assets.mkdir()
    assert _suite.resolve_root({}, "workspace_track") == assets / "workspace_track"

    suite_override = tmp_path / "custom_suite"
    monkeypatch.setenv("FLASHRSI_SUITE_ENVS_ROOT", str(suite_override))
    assert (
        _suite.resolve_root({}, "workspace_track") == suite_override / "workspace_track"
    )

    track_override = tmp_path / "custom_track"
    monkeypatch.setenv(track_var, str(track_override))
    assert _suite.resolve_root({}, "workspace_track") == track_override
    explicit = tmp_path / "explicit"
    assert _suite.resolve_root({"root": explicit}, "workspace_track") == explicit


def _require_family(name):
    if name in {"tb21_clean", "swegym_lite"}:
        module = "harbor_suite"
    elif name in {"clawgym", "harvey_workspace", "workspace_dualgate"}:
        module = "workspace_suite"
    else:
        module = "design_suite"
    pytest.importorskip(f"flashrsi.integrations.{module}")


def test_native_installed_import_has_no_peer_sys_path():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import flashrsi
from flashrsi import runner
assert Path(flashrsi.__file__).resolve().parent == Path(sys.argv[1]) / 'flashrsi'
from flashrsi.benchmarks import _suite
assert _suite.NAMES <= runner._BENCHMARKS.keys()
assert {'program_search', 'pi_agent'} <= runner._AGENTS.keys()
assert {'two_gate_workspace', 'sequential_memory'} <= runner._ALGORITHMS.keys()
assert not any(name == 'adapters' or name.startswith('adapters.') for name in sys.modules)
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(REPO)], cwd=REPO, env=env, check=True
    )
    for path in (REPO / "flashrsi").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                assert all(name.name.split(".")[0] != "adapters" for name in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "adapters"


async def _forbid_preflight(*_args):
    pytest.fail("sealed config reached startup")


@pytest.mark.parametrize("path", SEALED, ids=lambda path: path.stem)
def test_sealed_denial_precedes_assets_credentials_and_docker(path, monkeypatch):
    forbidden = _forbid_preflight

    monkeypatch.setattr(_startup, "_preflight", forbidden)
    cfg = load_config(path)
    assert cfg.benchmark.allow_sealed_test is False
    cfg.benchmark.root = "/nonexistent-sealed-assets"
    with pytest.raises(PermissionError, match="Sealed suite"):
        runner._BENCHMARKS[cfg.benchmark.name](cfg.benchmark)
    with pytest.raises(PermissionError, match="Sealed suite"):
        asyncio.run(runner._prepare_target(cfg))


@pytest.mark.skipif(not ASSETS, reason="suite frozen assets unavailable")
@pytest.mark.parametrize("path", ACTIVE, ids=lambda path: path.stem)
def test_every_active_config_instantiates_runner_algorithm(path):
    config = load_config(path)
    _require_family(config.benchmark.name)
    algorithm = asyncio.run(runner._prepare(config, smoke=True))
    assert algorithm.target._env_runner is not None
    assert algorithm.target._require_native_env
    benchmark = algorithm.benchmark
    assert len(benchmark.train) + len(benchmark.dev) > 0
    assert not benchmark.test
    assert all(task.env for task in (*benchmark.train, *benchmark.dev))


@pytest.mark.skipif(not ASSETS, reason="suite frozen assets unavailable")
@pytest.mark.parametrize(
    "harness", ["pi_agent", "claude_code", "miniagent", "openclaw", "program_search"]
)
@pytest.mark.parametrize(
    "filename",
    [
        "tb21_clean",
        "swegym_lite",
        "clawgym",
        "harvey_workspace",
        "workspace_dualgate",
        "engdesign_open",
        "frontier_eng_dev",
        "tdc_admet",
        "kernelbench_train",
    ],
)
def test_every_family_binds_each_native_harness(filename, harness):
    cfg = load_config(
        REPO / "configs/suite" / f"{filename}.yaml",
        overrides={"agent": {"name": harness}},
    )
    _require_family(cfg.benchmark.name)
    target, benchmark = asyncio.run(runner._prepare_target(cfg, smoke=True))
    benchmark.load()
    assert target._env_runner is not None
    assert all(task.env for task in (*benchmark.train, *benchmark.dev))
    with pytest.raises(ValueError, match="task.env"):
        asyncio.run(target.rollout([Task("invalid", "must not run on the host")]))


def test_sealed_overrides_fail_closed():
    for options in (
        {"split": "test"},
        {"allow_sealed_test": True},
        {"split": "sealed"},
    ):
        with pytest.raises(PermissionError):
            runner._BENCHMARKS["tdc_admet"]({"name": "tdc_admet", **options})


def test_workspace_gate_and_budget_are_explicit():
    assert WorkspaceGate().accepts((0.6, 0.4), (0.6, 0.5), 100, 110)
    assert not WorkspaceGate(True).accepts((0.6, 0.4), (0.6, 0.5), 100, 110)
    assert not WorkspaceGate(max_growth_ratio=0.05).accepts(
        (0.6, 0.4), (0.7, 0.5), 100, 110
    )


def test_kernel_guard_accepts_torch_only_program():
    kernelbench = pytest.importorskip("flashrsi.integrations.kernelbench")
    assert kernelbench.kernel_guard(
        "import torch\ndef f(a,b): return torch.matmul(a,b)\n"
    )


def test_frontier_oracle_consumes_attempts_and_rejects_third(tmp_path, monkeypatch):
    frontier_eng = pytest.importorskip("flashrsi.integrations.frontier_eng")

    monkeypatch.setattr(
        frontier_eng,
        "docker_json",
        lambda *_args, **_kwargs: ((0.42, 0.28, 0.18, 0.12), {"cleaned": True}),
    )
    proxy = frontier_eng.setup_oracle(tmp_path, 2)
    first = proxy.submit_candidate(frontier_eng.SEED, submission_id="first")
    assert first["valid"] and first["call_count"] == 1
    assert first == proxy.submit_candidate(frontier_eng.SEED, submission_id="first")
    assert (
        proxy.submit_candidate("bad()", submission_id="invalid")["primary_score"] == 0
    )
    with pytest.raises(RuntimeError, match="oracle budget exhausted"):
        proxy.submit_candidate(frontier_eng.SEED, submission_id="third")
    assert proxy.count() == 2


def test_native_transport_keeps_terminal_tool_events_and_usage():
    from flashrsi.integrations.sandbox_assets.solver_guest import transport_record

    record = transport_record(
        Trajectory(
            messages=(
                {"type": "message_update", "partial": "large"},
                {"type": "tool_execution_end", "toolName": "bash"},
                {"type": "agent_end", "messages": [{"role": "assistant"}]},
            ),
            usage={"input": 10, "cacheRead": 9},
        )
    )
    assert record["messages"] == [
        {"type": "tool_execution_end", "toolName": "bash"},
        {"type": "agent_end"},
    ]
    assert record["usage"] == {"input": 10, "cacheRead": 9}


async def _record_kernel_proposal(prompts, program, prompt):
    prompts.append(prompt)
    return {"code": program}, {}


def _record_kernel_job(jobs, **kwargs):
    jobs.append(kwargs)
    return {"job_id": kwargs["job_id"], "score": 0.25 * len(jobs)}


def test_kernel_program_search_uses_multiple_isolated_queue_receipts(monkeypatch):
    from types import SimpleNamespace

    kernelbench = pytest.importorskip("flashrsi.integrations.kernelbench")

    prompts, jobs = [], []
    agent = ProgramSearchTarget(
        {"llm": {"model_name": "fixture", "api_key": "fixture"}}
    )
    program = "def kernel(x):\n    return x\n"

    proposal = partial(_record_kernel_proposal, prompts, program)

    submit = partial(_record_kernel_job, jobs)

    monkeypatch.setattr(agent, "propose", proposal)
    monkeypatch.setattr(kernelbench.GPUTimingClient, "submit", submit)
    benchmark = SimpleNamespace(
        config={
            "gpu_queue": "http://localhost:1234",
            "device_uuid": "GPU-fixture",
            "oracle_budget": 2,
        },
        _tasks={"fixture": None},
    )
    _, report = asyncio.run(
        kernelbench.kernel(agent, benchmark, Task("fixture", "Implement a kernel"))
    )
    assert report["gpu_calls"] == 2 and report["score"] == 0.5
    assert "Best scalar score: 0.25" in prompts[1]
    assert jobs[0]["job_id"] != jobs[1]["job_id"]


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


def proposal_events(text='{"code":"candidate"}', finish="stop"):
    return [
        {"choices": [{"delta": {"content": text}, "finish_reason": finish}]},
        {"choices": [], "usage": TOKENS},
        "[DONE]",
    ]


def _proposal_replay_reply(requests, request_timeout, cap, request):
    requests.append(request)
    assert request.extensions["timeout"] == {
        "connect": 20,
        "read": request_timeout,
        "write": request_timeout,
        "pool": request_timeout,
    }
    payload = json.loads(request.content)
    assert payload["max_tokens"] == cap and payload["stream"]
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["model"] == "deepseek-flash"
    assert payload["reasoning_effort"] == "high"
    assert payload["thinking"] == {"type": "enabled"}
    assert request.url.path == "/v1/chat/completions"
    return httpx.Response(200, text=sse(proposal_events()))


async def _run_two_proposals(target, _task):
    for _ in range(2):
        candidate, usage = await target.propose("candidate please")
        assert candidate == {"code": "candidate"}
        assert usage == {**TOKENS, "cached_tokens": 80, "reasoning_tokens": 190}
    return Trajectory(messages=[{"type": "SuiteVerifier", "score": 0.8}])


@pytest.mark.parametrize("override", [False, True])
def test_program_search_records_wire_cap_and_streaming_usage(monkeypatch, override):
    requests = []
    cap = 32768 if override else 65536
    request_timeout = 1200 if override else 900

    handler = partial(_proposal_replay_reply, requests, request_timeout, cap)

    transport(monkeypatch, handler)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only")
    cfg = (
        {"llm": {"max_tokens": cap}, "request_timeout": request_timeout}
        if override
        else {}
    )
    agent = ProgramSearchTarget(cfg)

    runner = _run_two_proposals

    agent._env_runner = runner
    [trace] = asyncio.run(agent.rollout([Task("synthetic", "fixture", env="fixture")]))
    assert len(requests) == 2
    assert [step["step"] for step in trace.messages[:-1]] == [1, 2]
    for step in trace.messages[:-1]:
        assert step["finish_reason"] == "stop"
        assert step["status"] == "finished"
        assert step["max_tokens"] == cap
    assert trace.usage["model_calls"] == [step["usage"] for step in trace.messages[:-1]]


@pytest.mark.parametrize(
    "events,reason",
    [
        (proposal_events(finish="length"), "program_search_output_length"),
        (proposal_events()[:-1], "incomplete_stream"),
        (proposal_events(finish=None), "incomplete_stream"),
        (["[DONE]"], "incomplete_stream"),
        (["{"], "incomplete_stream"),
        ([{"error": {"message": "stream ended"}}], "incomplete_stream"),
    ],
)
def test_program_search_truncation_is_never_scored(monkeypatch, events, reason):
    transport(monkeypatch, lambda _: httpx.Response(200, text=sse(events)))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only")
    with pytest.raises(InfraError, match=reason):
        asyncio.run(ProgramSearchTarget().propose("fixture"))


async def _invalid_frontier_proposal(target, _benchmark, _task):
    await target.propose("fixture")
    pytest.fail("invalid candidate reached the scorer")


@pytest.mark.parametrize("body", ["{", "[1, 2]"])
def test_program_search_invalid_json_is_scored_with_usage(monkeypatch, body):
    design = pytest.importorskip("flashrsi.benchmarks.design_suite")
    integration = pytest.importorskip("flashrsi.integrations.design_suite")
    frontier_eng = pytest.importorskip("flashrsi.integrations.frontier_eng")

    transport(
        monkeypatch, lambda _: httpx.Response(200, text=sse(proposal_events(body)))
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only")
    agent = ProgramSearchTarget()
    benchmark = object.__new__(design.FrontierEng)
    monkeypatch.setattr(benchmark, "check_task", lambda _: None)

    synthetic_frontier = _invalid_frontier_proposal

    monkeypatch.setattr(frontier_eng, "frontier", synthetic_frontier)
    if hasattr(integration, "frontier"):
        monkeypatch.setattr(integration, "frontier", synthetic_frontier)
    integration.bind(agent, benchmark)
    [trace] = asyncio.run(agent.rollout([Task("synthetic", "fixture", env="fixture")]))
    assert trace.messages[-1]["score"] == 0.0
    assert trace.messages[-1]["candidate_valid"] is False
    assert trace.messages[0]["status"] == "invalid_json"
    assert trace.usage["model_calls"][0]["completion_tokens"] == 202


def test_program_search_invalid_kernel_code_scores_zero(monkeypatch):
    design = pytest.importorskip("flashrsi.benchmarks.design_suite")
    integration = pytest.importorskip("flashrsi.integrations.design_suite")
    kernelbench = pytest.importorskip("flashrsi.integrations.kernelbench")

    transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            text=sse(
                proposal_events(
                    '{"code": "def not valid Python"}',
                )
            ),
        ),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only")
    benchmark = object.__new__(design.KernelBench)
    benchmark.config = {
        "gpu_queue": "http://localhost:1234",
        "device_uuid": "GPU-fixture",
        "oracle_budget": 2,
    }
    benchmark._tasks = {"synthetic": None}
    monkeypatch.setattr(benchmark, "check_task", lambda _: None)
    monkeypatch.setattr(
        kernelbench.GPUTimingClient,
        "submit",
        lambda *_args, **_kw: pytest.fail("invalid program reached the queue"),
    )
    agent = ProgramSearchTarget()
    integration.bind(agent, benchmark)
    [trace] = asyncio.run(agent.rollout([Task("synthetic", "fixture", env="fixture")]))
    assert trace.messages[-1]["score"] == 0
    assert trace.messages[-1]["gpu_calls"] == 0
    assert all(
        receipt["score"] == 0 for receipt in trace.messages[-1]["trial_receipts"]
    )
    assert len(trace.usage["model_calls"]) == 2


def _valid_proposal_reply(_request):
    return httpx.Response(200, text=sse(proposal_events()))


async def _overdue_scoring(calls, target, _task):
    for _ in range(3):
        await target.propose("fixture")
        calls.append("proposal")
        await asyncio.sleep(0.15)
    pytest.fail("deadline did not stop the iterative search")


def test_program_search_episode_deadline_includes_steps_and_scoring(monkeypatch):
    handler = _valid_proposal_reply

    transport(monkeypatch, handler)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only")
    monkeypatch.setenv("FLASHRSI_INFRA_RETRIES", "0")
    agent = ProgramSearchTarget({"timeout": 0.2, "request_timeout": 1200})
    calls = []

    runner = partial(_overdue_scoring, calls)

    agent._env_runner = runner
    with pytest.raises(InfraError, match="program_search_episode_timeout"):
        asyncio.run(agent.rollout([Task("synthetic", "fixture", env="fixture")]))
    assert calls == ["proposal", "proposal"]


def _retry_proposal_reply(attempts, request):
    attempts.append(request)
    if len(attempts) == 1:
        raise httpx.ReadTimeout("fixture idle timeout")
    return httpx.Response(200, text=sse(proposal_events()))


async def _score_completed_proposal(target, _task):
    await target.propose("fixture")
    return Trajectory(messages=[{"type": "SuiteVerifier", "score": 1.0}])


def test_program_search_transient_failure_retries_without_scoring(monkeypatch):
    attempts = []
    handler = partial(_retry_proposal_reply, attempts)

    runner = _score_completed_proposal

    transport(monkeypatch, handler)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only")
    monkeypatch.setenv("FLASHRSI_INFRA_RETRIES", "1")
    from flashrsi.core import infra

    monkeypatch.setattr(infra.random, "uniform", lambda *_: 0)
    agent = ProgramSearchTarget()
    agent._env_runner = runner
    [trace] = asyncio.run(agent.rollout([Task("synthetic", "fixture", env="fixture")]))
    assert len(attempts) == 2
    assert trace.messages[-1]["score"] == 1.0
    assert trace.messages[0]["step"] == 1
