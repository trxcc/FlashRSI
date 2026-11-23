"""Verify official suite profiles and fail-closed invocation settings."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from flashrsi import _startup, cli, runner
from flashrsi.agents.llm import LLM
from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.config import load_config
from flashrsi.core import Task, Trajectory

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs/suite"
SEALED = sorted(path.stem for path in CONFIGS.glob("*_sealed.yaml"))
ACTIVE = {
    "tb21_clean": ("claude_code", None, None, None, "ace", 4, 5),
    "swegym_lite": ("claude_code", 3600, None, None, "ace", 4, 0),
    "engdesign_open": ("claude_code", 5400, 250, None, "ace", 4, 4),
    "tdc_admet": (
        "program_search",
        240120,
        None,
        None,
        "sequential_memory",
        None,
        None,
    ),
    "frontier_eng_dev": (
        "program_search",
        213500,
        None,
        32768,
        "sequential_memory",
        None,
        None,
    ),
    "kernelbench_train": ("program_search", 12640, None, None, "ace", 1, 3),
    "clawgym": ("claude_code", 660, None, None, "ace", 4, 0),
    "harvey_workspace": ("claude_code", 3600, 200, None, "ace", 4, 0),
    "workspace_dualgate": (
        "claude_code",
        3600,
        200,
        None,
        "two_gate_workspace",
        None,
        2,
    ),
}
INTERACTIVE = {name for name, values in ACTIVE.items() if values[0] == "claude_code"}
ACTIVE.update({name + "_codex": ("codex", *ACTIVE[name][1:]) for name in INTERACTIVE})


def test_inventory_contains_ten_locked_configs_and_fifteen_active_configs():
    assert len(SEALED) == 10
    assert {path.stem for path in CONFIGS.glob("*.yaml")} == set(ACTIVE) | set(SEALED)


@pytest.mark.parametrize("smoke", [False, True], ids=["production", "smoke"])
@pytest.mark.parametrize("name", ACTIVE)
def test_active_configuration_profiles(name, smoke):
    source = load_config(CONFIGS / f"{name}.yaml")
    config = runner._execution_config(source, smoke=smoke)
    harness, timeout, turns, cap, algo, batch, repeats = ACTIVE[name]
    benchmark_name = name.removesuffix("_codex")
    assert source.run.get("smoke", False) is False
    assert config.benchmark.allow_sealed_test is False
    assert config.agent.name == harness and config.agent.get("max_turns") == turns
    if not smoke or benchmark_name not in {"tdc_admet", "frontier_eng_dev"}:
        assert config.agent.get("timeout") == timeout
    assert config.agent.llm.model_name == "deepseek-flash"
    assert config.agent.llm.thinking == {"type": "enabled"}
    assert config.agent.llm.reasoning_effort == "high"
    assert config.agent.request_timeout == 900
    assert config.agent.llm.get("max_tokens") == cap
    assert (
        Path(config.run.output_dir)
        == REPO / "runs" / ("suite_smoke" if smoke else "suite") / name
    )
    assert config.algorithm.name == algo
    assert config.algorithm.worker.llm.get("max_tokens") is None
    assert config.algorithm.get("batch_size") == batch
    if algo == "ace":
        assert config.algorithm.num_epochs == 1
        assert config.algorithm.max_reflection_rounds == 1
        assert config.algorithm.get("max_steps") == (1 if smoke else None)
    assert config.algorithm.get("val_repeats") == (
        1
        if smoke and benchmark_name == "workspace_dualgate"
        else 0
        if smoke and algo == "ace"
        else repeats
    )
    if smoke:
        assert config.run.smoke and config.run.limit == config.run.iterations == 1
    _assert_track_profile(config, benchmark_name, smoke)


def _assert_track_profile(config, benchmark_name, smoke):
    if benchmark_name in {"tb21_clean", "swegym_lite"}:
        assert config.benchmark.smoke_whitelist_only is smoke
        assert len(config.benchmark.smoke_task_ids) == 1
    if benchmark_name == "tb21_clean":
        assert config.benchmark.n_attempts == (1 if smoke else 5)
        assert config.agent.get("harbor", {}).get("timeout_multiplier", 1.0) == 1.0
    if benchmark_name == "swegym_lite":
        assert config.agent.swe_bench.timeout == 3600
        assert config.benchmark.evaluation_timeout == 1800
        assert config.benchmark.verifier_resources.pids_limit == 512
    if benchmark_name == "engdesign_open":
        assert config.benchmark.grader_timeout == 420
        assert config.benchmark.solver_resources == {
            "cpus": 4,
            "memory": "8g",
            "pids_limit": 256,
            "shm_size": "1g",
        }
    if benchmark_name == "tdc_admet":
        assert config.benchmark.seeds == ([1] if smoke else [1, 2, 3, 4, 5])
        assert config.benchmark.oracle_budget == (2 if smoke else 20)
        assert (
            config.benchmark.fit_timeout == 900
            and config.benchmark.predict_timeout == 120
        )
        assert config.benchmark.resources == {
            "cpus": 4,
            "memory": "8g",
            "pids_limit": 256,
            "shm_size": "512m",
        }
        assert config.benchmark.smoke_task_ids == ["hia_hou"]
    if benchmark_name == "frontier_eng_dev":
        assert config.benchmark.mode == ("synthetic_dev" if smoke else "real_tasks")
        assert config.benchmark.oracle_budget == (2 if smoke else 100)
    if benchmark_name == "kernelbench_train":
        assert config.benchmark.split == "auto"
        assert config.benchmark.oracle_budget == (1 if smoke else 10)
        assert config.benchmark.require_real_cuda and config.algorithm.eval_steps == 5
    if benchmark_name in {"clawgym", "harvey_workspace", "workspace_dualgate"}:
        assert config.benchmark.train_judge_mode == "batched"
        assert config.benchmark.val_judge_mode == "per_criterion"
        assert config.benchmark.val_judge.judge_model == "gemini-3.5-flash"
        assert config.benchmark.judge_max_tokens == 16384
        assert (
            config.benchmark.cpus,
            config.benchmark.memory,
            config.benchmark.solver_pids,
            config.benchmark.pids,
            config.benchmark.shm_size,
        ) == (2, "4g", 192, 256, "256m")
        assert config.benchmark.claw_solver_timeout == 660
        assert config.benchmark.harvey_solver_timeout == 3600


async def _forbid_sealed_preflight(*_args, **_kwargs):
    pytest.fail("sealed execution crossed the runner's preflight boundary")


@pytest.mark.parametrize(
    "smoke,override", [(False, None), (True, None), (True, "codex")]
)
@pytest.mark.parametrize("name", SEALED)
def test_sealed_cli_and_runner_fail_before_credentials_models_or_containers(
    name, smoke, override, monkeypatch, capsys
):
    forbidden = _forbid_sealed_preflight

    monkeypatch.setattr(_startup, "_preflight", forbidden)
    full = CONFIGS / f"{name}.yaml"
    config = load_config(
        full, overrides={"agent": {"name": override}} if override else None
    )
    assert config.benchmark.allow_sealed_test is False
    assert config.agent.request_timeout == 900
    with pytest.raises(PermissionError, match="Sealed suite split is locked"):
        asyncio.run(runner._prepare_target(config, smoke=smoke))

    argv = ["run", "--config", str(full)]
    if smoke:
        argv.append("--smoke")
    if override:
        argv.append(f"--agent.name={override}")
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert (
        f"PermissionError: Sealed suite split is locked: {name}"
        in capsys.readouterr().err
    )


class _PreflightChecked(Exception):
    pass


async def _assert_native_preflight(smoke, config):
    assert config.benchmark.smoke_whitelist_only is smoke
    raise _PreflightChecked()


@pytest.mark.parametrize("name", ["tb21_clean", "swegym_lite"])
@pytest.mark.parametrize("smoke", [True, False])
def test_runner_profiles_reach_native_harbor_before_startup(name, smoke, monkeypatch):
    preflight = partial(_assert_native_preflight, smoke)

    monkeypatch.setattr(_startup, "_preflight", preflight)
    with pytest.raises(_PreflightChecked):
        asyncio.run(
            runner._prepare_target(load_config(CONFIGS / f"{name}.yaml"), smoke=smoke)
        )


async def _assert_smoke_dispatch(expected, config, limit):
    resolved = runner._execution_config(config, limit=limit)
    assert resolved.run.smoke and resolved.run.limit == 2
    assert Path(resolved.run.output_dir) == expected
    assert resolved.benchmark.smoke_task_ids == ["caco2_wang"]
    return expected, {"primary_metric": "reward", "tasks": []}


def test_explicit_smoke_ids_stay_open_and_cli_run_config_dispatches(
    monkeypatch, tmp_path
):
    path = CONFIGS / "tdc_admet.yaml"
    expected = tmp_path / "picked"

    smoke = partial(_assert_smoke_dispatch, expected)

    monkeypatch.setattr(runner, "_smoke", smoke)
    assert (
        cli.main(
            [
                "run",
                "-c",
                str(path),
                "--run.smoke=true",
                "--limit",
                "2",
                "--output-dir",
                str(expected),
                "--benchmark.task_ids=[caco2_wang]",
            ]
        )
        == 0
    )
    config = load_config(path, overrides={"benchmark": {"split": "val"}})
    with pytest.raises(PermissionError, match="restricted to Train"):
        runner._execution_config(config, smoke=True)


async def _original_rollout(target, task, seen, agent, value):
    assert agent is target and value is task
    seen.append("original")
    return Trajectory(output="done")


async def _dispatch_verifier(seen, dispatch, agent, value):
    seen.append("verifier")
    return await dispatch(agent, value)


def test_program_search_nested_harbor_binding_keeps_original_dispatch():
    target = ProgramSearchTarget()
    task = Task("fixture", "fixture", env="fixture")
    seen = []

    original = partial(_original_rollout, target, task, seen)

    target._env_runner = original
    dispatch = target._env_runner

    verifier = partial(_dispatch_verifier, seen, dispatch)

    target._env_runner = verifier
    [row] = asyncio.run(target.rollout([task]))
    assert row.output == "done"
    assert seen == ["verifier", "original"]


class _KernelTrial(list):
    score = 0.0


async def _kernel_validation_receipt(calls, speeds, snapshot, _tasks, *, label):
    calls.append((snapshot.id, label))
    return _KernelTrial(
        Trajectory(
            messages=[
                {
                    "trial_receipts": [
                        {"speedup": speed, "correct": speed > 0},
                    ]
                }
            ]
        )
        for speed in speeds[snapshot.id]
    )


def test_kernel_ace_requires_correctness_non_regression_and_speed_gain():
    from flashrsi.algorithms.ace import ACE

    algorithm = object.__new__(ACE)
    algorithm.config = SimpleNamespace(val_repeats=3, min_val_gain=0.0)
    algorithm.benchmark = SimpleNamespace(name="kernelbench", dev=["fixture"])
    algorithm._record = lambda *_: None
    algorithm._best_score = float("-inf")
    algorithm._best_correctness = None
    snapshots = [SimpleNamespace(id=i) for i in range(4)]
    speeds = ([1.0, 1.0], [10.0, 0.0], [1.0, 1.0], [2.0, 1.0])
    calls = []

    evaluate = partial(_kernel_validation_receipt, calls, speeds)

    algorithm._evaluate = evaluate
    for index, expected in enumerate((0, 0, 0, 3)):
        asyncio.run(algorithm._validate(snapshots[index]))
        assert algorithm._best.id == expected
    assert len(calls) == 12


async def _assert_interactive_shell(shell, config):
    from flashrsi.agents import ClaudeCode, Codex, Pi

    assert config.agent.name == shell
    config.agent.llm.api_key = "fixture-local-placeholder"
    selected = {"claude_code": ClaudeCode, "codex": Codex, "pi_agent": Pi}[shell]
    assert isinstance(runner._make_agent(selected, config.agent, "agent"), selected)
    return SimpleNamespace(benchmark=SimpleNamespace(train=[], dev=[], test=[]))


@pytest.mark.parametrize("name", sorted(INTERACTIVE))
@pytest.mark.parametrize("shell", ["claude_code", "codex", "pi_agent"])
def test_interactive_dotted_shell_overrides(name, shell, monkeypatch):
    checked = partial(_assert_interactive_shell, shell)

    monkeypatch.setattr(runner, "_prepare", checked)
    assert (
        cli.main(
            [
                "run",
                "-c",
                str(CONFIGS / f"{name}.yaml"),
                "--check-only",
                f"--agent.name={shell}",
            ]
        )
        == 0
    )


def test_summary_pools_criteria_and_rolls_up_raw_and_wrapped_model_calls(tmp_path):
    pytest.importorskip("flashrsi.benchmarks.workspace_suite")
    from flashrsi._run_summary import _write_run_summary

    tasks = [
        Trajectory(metrics={"passed_criteria": 1, "total_criteria": 1, "micro_avg": 1}),
        Trajectory(
            metrics={"passed_criteria": 1, "total_criteria": 3, "micro_avg": 1 / 3}
        ),
    ]
    metrics = runner._aggregate_metrics(tasks)
    assert metrics["micro_avg"] == 0.5 and metrics["passed_criteria"] == 2
    assert metrics["total_criteria"] == 4
    folder = tmp_path / "trajectories"
    folder.mkdir()
    (folder / "fixture.json").write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "status": "finished",
                        "usage": {
                            "model_calls": [
                                {"prompt_tokens": 11, "completion_tokens": 2},
                                {
                                    "call_id": "fixture",
                                    "usage": {
                                        "prompt_tokens": 12,
                                        "completion_tokens": 3,
                                    },
                                },
                            ],
                        },
                    }
                ]
            }
        )
    )
    for snapshot in ("selected", "discarded"):
        (folder / f"{snapshot}-validation.json").write_text(
            json.dumps(
                {
                    "snapshot_id": snapshot,
                    "trajectories": [
                        {
                            "task": {"metadata": {"split": "harvey_val"}},
                            "metrics": dict(row.metrics),
                            "status": "finished",
                        }
                        for row in tasks
                    ],
                }
            )
        )
    validation = runner._harvey_validation_metrics(tmp_path, "selected")
    assert validation["micro_avg"] == 0.5 and validation["total_criteria"] == 4
    _write_run_summary(
        tmp_path,
        result={
            "status": "finished",
            "metrics": metrics,
            "validation_metrics": validation,
        },
        emit=False,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["result"]["metrics"]["micro_avg"] == 0.5
    assert summary["result"]["validation_metrics"]["micro_avg"] == 0.5
    assert summary["usage"]["total"]["input_tokens"] == 23
    assert summary["usage"]["total"]["output_tokens"] == 5


def _configured_worker_reply(override, request):
    assert request.extensions["timeout"]["connect"] == 20
    assert request.extensions["timeout"]["read"] == 900
    payload = json.loads(request.content)
    assert payload["max_tokens"] == (override or 65536)
    assert payload["thinking"] == {"type": "enabled"}
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ]
        },
    )


def _worker_client(reply, original, **kwargs):
    http = httpx.AsyncClient(
        timeout=kwargs["timeout"], transport=httpx.MockTransport(reply)
    )
    return original(http_client=http, **kwargs)


@pytest.mark.parametrize("override", [None, 32768])
def test_worker_uses_deepseek_budget_and_request_timeout(override, monkeypatch):
    original = openai.AsyncOpenAI

    reply = partial(_configured_worker_reply, override)

    client = partial(_worker_client, reply, original)

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    values = {
        "llm": {
            "model_name": "deepseek-flash",
            "api_key": "fixture-provider-credential",
            "base_url": "https://api.deepseek.com/v1",
            "max_tokens": override,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
    }
    assert asyncio.run(LLM(values).ask("test")) == "ok"


def _worker_reply(override, request):
    assert request.extensions["timeout"] == {
        "connect": 20,
        "read": 900,
        "write": 900,
        "pool": 900,
    }
    payload = json.loads(request.content)
    assert payload["max_tokens"] == (override or 65536)
    assert payload.get("stream") is not True
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


def _worker_mock_client(real_client, handler, **kw):
    assert kw["max_retries"] == 1
    return real_client(
        http_client=httpx.AsyncClient(
            timeout=kw["timeout"],
            transport=httpx.MockTransport(handler),
        ),
        **kw,
    )


@pytest.mark.parametrize("override", [None, 20000])
def test_worker_nonstreaming_cap_and_idle_timeout(monkeypatch, override):
    real_client = openai.AsyncOpenAI

    handler = partial(_worker_reply, override)

    client = partial(_worker_mock_client, real_client, handler)

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    llm = {"model_name": "deepseek-flash", "api_key": "fixture-only"}
    if override is not None:
        llm["max_tokens"] = override
    assert asyncio.run(LLM({"llm": llm}).ask("test")) == "ok"
