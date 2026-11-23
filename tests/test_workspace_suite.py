"""Exercise workspace clocks, isolation, judging, and validation.

Local fixtures stand in for model calls and Docker.
Both ClawGym and Harvey use the workspace benchmark contracts.
"""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError

from flashrsi.algorithms.promotion import (
    TDC_SPEARMAN_VAL,
    SequentialMemory,
    TwoGateWorkspacePromotion,
    WorkspaceGate,
    normalized_endpoint_score,
)
from flashrsi.benchmarks._suite import SEALED as SEALED_SUITES
from flashrsi.benchmarks._suite import guard
from flashrsi.benchmarks.workspace_suite import (
    JUDGE_USAGE_KEYS,
    SEALED,
    SOLVER_TIMEOUTS,
    WorkspaceBenchmark,
    WorkspaceSettings,
    WorkspaceSuite,
    summarize_harvey,
)
from flashrsi.core import InfraError, Task, Trajectory
from flashrsi.integrations import workspace_suite as ws
from flashrsi.integrations.sandbox_assets.extract_deliverables import extract


def executor(**config):
    return ws.WorkspaceExecutor(SimpleNamespace(settings=WorkspaceSettings(**config)))


def run(coroutine):
    return asyncio.run(coroutine)


def rubric(cid, *targets):
    return {"id": cid, "title": "Fixture", "match_criteria": "Check chosen evidence",
            "deliverables": list(targets)}


def harbor_fixture(tmp_path, criteria, names):
    base = tmp_path / "base"
    (base / "grader").mkdir(parents=True)
    (base / "grader/task.json").write_text(json.dumps({"instructions": "Fixture", "criteria": criteria}))
    result = tmp_path / "result"
    result.mkdir()
    for name in names:
        file = result / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"Fixture evidence in {name}")
    row = {"deliverables": names, "locked_files": {}, "criteria_count": len(criteria)}
    return row, base, result


def judge_response(ids, valid=True):
    return {
        "model": "fixture",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "criteria": [{"id": cid, "pass": cid != "fail", "reason": "fixture evidence"}
                         for cid in ids] if valid else []})}}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 12,
                  "completion_tokens_details": {"reasoning_tokens": 3},
                  "prompt_tokens_details": {"cached_tokens": 8}},
    }


class Client:
    def __init__(self, clients, requests, **kwargs):
        self.requests = requests
        clients.append({k: v for k, v in kwargs.items() if k != "api_key"})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        pass

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        data = json.loads(kwargs["messages"][1]["content"])
        response = judge_response([item["id"] for item in data["criteria"]])
        return SimpleNamespace(model_dump=lambda **_: response)


@pytest.fixture
def fake_judge(monkeypatch):
    clients = []
    requests = []
    monkeypatch.setenv("FIXTURE_CREDENTIAL", "local-fixture")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    monkeypatch.setattr(ws, "AsyncOpenAI", partial(Client, clients, requests))
    return clients, requests


def test_settings_defaults_and_unchanged_sealed_guards():
    settings = WorkspaceSettings()
    assert SOLVER_TIMEOUTS == {"clawgym": 660, "harvey": 3600, "jobbench": 3600, "gdpval": 3600}
    assert (settings.claw_solver_timeout, settings.harvey_solver_timeout) == (660, 3600)
    assert (settings.claw_checker_timeout, settings.extractor_timeout,
            settings.judge_timeout, settings.judge_phase_timeout) == (120, 120, 720, 3600)
    assert settings.resources() == {"cpus": 2, "memory": "4g", "pids": 256, "shm_size": "256m"}
    assert settings.mode_for("harvey_train") == "batched"
    assert settings.mode_for("harvey_val") == "per_criterion"
    assert settings.batch_size == 24 and settings.judge_max_tokens == 16384
    assert WorkspaceSuite({"name": "workspace_dualgate", "cpus": 3}).settings.cpus == 3
    for name in SEALED_SUITES:
        for override in (False, True):
            with pytest.raises(PermissionError):
                guard({"name": name, "allow_sealed_test": override})
    for name in SEALED:
        for override in (False, True):
            with pytest.raises(PermissionError):
                WorkspaceBenchmark(name, allow_sealed_test=override)


@pytest.mark.parametrize("options", [{"judge_concurrency": 9}, {"memory": "-4g"}, {"pids": True},
                                     {"judge_timeout": 0}, {"val_judge_mode": "invalid"}])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        WorkspaceSettings(**options)


class Agent:
    snapshot = object()
    _execution = ContextVar("test_episode", default=None)

    @property
    def config(self):
        return self._execution.get().config if self._execution.get() else {"timeout": 900}


def test_parallel_family_clocks_and_input_only_workspace(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for directory in ("input", "grader"):
        (source / directory).mkdir(parents=True)
        (source / directory / f"{directory}.txt").write_text(directory)
    tasks = [Task("a", "task", metadata={"split": "clawgym_train"}),
             Task("b", "task", metadata={"split": "harvey_val"})]

    agent = Agent()
    seen = []

    async def solver(agent, prompt, workspace, *, image, workdir, cpus, memory, pids, shm_size):
        assert image == ws.IMAGE and workdir == "/workspace"
        assert (cpus, memory, pids, shm_size) == (2, "4g", 192, "256m")
        assert sorted(file.name for file in workspace.iterdir()) == ["input.txt"]
        first = agent.config["timeout"]
        await asyncio.sleep(0)
        assert agent.config["timeout"] == first
        seen.append(first)
        return Trajectory(), "fixture-container"

    monkeypatch.setattr(ws, "_workspace_image", AsyncMock(return_value=ws.IMAGE))
    monkeypatch.setattr(ws.runtime, "solver", solver)
    benchmark = SimpleNamespace(settings=WorkspaceSettings(), backend=lambda task: SimpleNamespace(
        name=task.metadata["split"], lookup=lambda _: ({}, source)))
    harness = ws.WorkspaceExecutor(benchmark)
    monkeypatch.setattr(harness, "_claw", AsyncMock(return_value={"score": 1.0}))
    monkeypatch.setattr(harness, "_harvey", AsyncMock(return_value={"score": 1.0}))

    async def both():
        return await asyncio.gather(*(harness.run(agent, task) for task in tasks))

    trajectories = run(both())
    assert seen == [660, 3600]
    assert [t.messages[-1]["solver_timeout"] for t in trajectories] == seen
    assert agent.config["timeout"] == 900


def test_solver_wall_clock_expiry_cancels_before_grading(tmp_path, monkeypatch):
    (tmp_path / "input").mkdir()
    agent = SimpleNamespace(snapshot=object(), config={"timeout": 900},
                            _execution=ContextVar("fixture_task", default=None))
    backend = SimpleNamespace(name="clawgym_train", lookup=lambda _: ({}, tmp_path))
    benchmark = SimpleNamespace(settings=WorkspaceSettings(claw_solver_timeout=1),
                                backend=lambda _: backend)
    harness = ws.WorkspaceExecutor(benchmark)
    stopped = []

    async def hung(*_, **__):
        try:
            await asyncio.sleep(30)
        finally:
            stopped.append(True)

    monkeypatch.setattr(ws, "solve_workspace", hung)
    checker = AsyncMock()
    monkeypatch.setattr(harness, "_claw", checker)
    with pytest.raises(InfraError, match="solver transport deadline"):
        run(harness.run(agent, Task("fixture", "fixture")))
    assert stopped == [True] and agent._execution.get() is None
    checker.assert_not_awaited()


@pytest.mark.parametrize("images, expected", [
    (f"{ws.IMAGE} sha256:current\nflashrsi-workspace:older sha256:other", "sha256:current"),
    ("flashrsi-workspace:staged sha256:legacy", "sha256:legacy"),
    ("flashrsi-workspace:a sha256:a\nflashrsi-workspace:b sha256:b", None),
    ("", None),
])
def test_workspace_image_selection(monkeypatch, images, expected):
    command = AsyncMock(return_value=(0, images, ""))
    monkeypatch.setattr(ws, "cmd", command)
    if expected is None:
        with pytest.raises(InfraError, match="Stage the workspace image"):
            run(ws._workspace_image())
    else:
        assert run(ws._workspace_image()) == expected
    assert command.await_args.args[:3] == ("docker", "image", "ls")


def test_solver_delegates_to_shared_runtime_with_workspace_flags(tmp_path, monkeypatch):
    settings = WorkspaceSettings(cpus=3, solver_pids=144, memory="3g", shm_size="128m")
    launch = AsyncMock(return_value=(Trajectory(output="done"), "fixture"))
    monkeypatch.setattr(ws, "_workspace_image", AsyncMock(return_value=ws.IMAGE))
    monkeypatch.setattr(ws.runtime, "solver", launch)
    agent = object()

    trace, name = run(ws.solve_workspace(
        agent, "test", tmp_path, resources=settings.resources(solver=True)
    ))
    launch.assert_awaited_once_with(
        agent, "test", tmp_path, image=ws.IMAGE, workdir="/workspace",
        cpus=3, memory="3g", pids=144, shm_size="128m",
    )
    assert trace.output == "done" and name == "fixture"

def test_checker_extractor_flags_timeouts_and_cleanup(tmp_path, monkeypatch):
    row, base, result = harbor_fixture(tmp_path, [], ["output/report.txt"])
    row.update(checker_paths=["reward/test.py"], hook="python reward/test.py .", hook_lang="bash")
    calls = []

    async def command(*args, **kwargs):
        calls.append((args, kwargs))
        if args[1:3] == ("image", "ls"):
            return 0, f"{ws.IMAGE} sha256:fixture", ""
        if args[1] == "exec":
            if "extract.py" in " ".join(args):
                return 0, json.dumps(extract(result, ["output/report.txt"])), ""
            return 0, '{"reward": 0.5}', "fixture traceback"
        return 0, "", ""

    monkeypatch.setattr(ws, "cmd", command)
    harness = executor()
    assert run(harness._claw(row, base, result))["checker_stderr"] == "fixture traceback"
    assert run(harness._extract(result, ["output/report.txt"]))[0]["text"].startswith("Fixture")
    for args, _ in (call for call in calls if call[0][1] == "run"):
        assert args[args.index("--network") + 1] == "none"
        assert args[args.index("--cpus") + 1] == "2"
        assert args[args.index("--memory") + 1] == "4g"
        assert args[args.index("--pids-limit") + 1] == "256"
        assert args[args.index("--shm-size") + 1] == "256m"
    assert [kw["timeout"] for args, kw in calls if args[1] == "exec"] == [120, 120]
    assert sum(args[1] == "rm" for args, _ in calls) == 2

    async def timeout(*args, **kwargs):
        if args[1] == "exec":
            raise InfraError("fixture Docker exec timed out")
        return await command(*args, **kwargs)

    monkeypatch.setattr(ws, "cmd", timeout)
    with pytest.raises(InfraError, match="timed out"):
        run(harness._claw(row, base, result))
    assert sum(args[1] == "rm" for args, _ in calls) == 3


def test_extractor_only_reads_selected_files(tmp_path):
    (tmp_path / "report.csv").write_text("x,y\n1,2")
    (tmp_path / "extra.txt").write_text("unchanged")
    assert extract(tmp_path, ["report.csv"])[0]["text"] == "x,y\n1,2"
    (tmp_path / "alias.csv").symlink_to(tmp_path / "report.csv")
    with pytest.raises(ValueError, match="invalid artifact"):
        extract(tmp_path, ["alias.csv"])
    with pytest.raises(ValueError, match="distinct"):
        extract(tmp_path, ["report.csv", "report.csv"])


def test_harvey_val_only_sends_target_artifact_and_records_stats(tmp_path, monkeypatch, fake_judge):
    clients, requests = fake_judge
    names = ["output/a.txt", "output/b.txt", "one/duplicate.txt", "two/duplicate.txt"]
    items = [rubric("pass", "a.txt"), rubric("fail", "b.txt"),
             rubric("amb", "duplicate.txt"), rubric("missing", "missing.txt"),
             rubric("final", "final_response")]
    row, base, result = harbor_fixture(tmp_path, items, names)
    (result / "input-copy.txt").write_text("read-only evidence")
    row["locked_files"]["input/input-copy.txt"] = {"sha256": ws.sha256(result / "input-copy.txt")}
    row["deliverables"].append("input-copy.txt")
    harness = executor(judge_api_key_env="FIXTURE_CREDENTIAL", judge_temperature=0)
    monkeypatch.setattr(harness, "_extract", AsyncMock(side_effect=lambda base, paths: extract(base, paths)))
    receipt = run(harness._harvey(row, base, result, "Response content", "harvey_val"))
    contents = [json.loads(call["messages"][1]["content"]) for call in requests]
    assert [len(data["criteria"]) for data in contents] == [1, 1, 1]
    assert [[f["path"] for f in d["deliverables"]] for d in contents] == [
        ["output/a.txt"], ["output/b.txt"], []]
    assert ["final_response" in d for d in contents] == [False, False, True]
    assert all(c["max_tokens"] == 16384 and c["reasoning_effort"] == "high"
               and c["extra_body"] == {"thinking": {"type": "enabled"}}
               and "temperature" not in c for c in requests)
    assert clients[0]["base_url"] == "https://api.deepseek.com/v1"
    assert clients[0]["timeout"].connect == 20 and clients[0]["timeout"].read == 720
    assert receipt["judge_mode"] == "per_criterion"
    assert receipt["ambiguous_deliverables"]["duplicate.txt"] == ["one/duplicate.txt", "two/duplicate.txt"]
    assert receipt["verdicts"][2]["ambiguous_basenames"] == ["duplicate.txt"]
    assert "input-copy.txt" not in [a["path"] for a in receipt["artifacts"]]
    assert receipt["passed_criteria"] == 2 and receipt["total_criteria"] == 5
    assert receipt["macro_score"] == 0.4 and receipt["all_pass"] == 0
    assert tuple(receipt["judge_usage"][key] for key in JUDGE_USAGE_KEYS) == (63, 36, 9, 24)

    benchmark = WorkspaceBenchmark("harvey_val")
    task = Task("fixture", "fixture", env="workspace_track")
    benchmark._tasks = {task.task_id: task}
    benchmark._splits = {}
    graded, = benchmark.grade((Trajectory(task=task, messages=({
        **receipt, "type": "SuiteVerifier", "task_id": task.task_id, "split": "harvey_val",
    },)),))
    assert graded.metrics["passed_criteria"] == 2 and graded.metrics["all_pass"] == 0
    assert graded.usage["judge_usage"] == receipt["judge_usage"]


def test_harvey_train_batches_target_groups_and_judge_override(tmp_path, monkeypatch, fake_judge):
    clients, requests = fake_judge
    items = [rubric(str(i), "a.txt") for i in range(25)] + [rubric("last", "b.txt")]
    row, base, result = harbor_fixture(tmp_path, items, ["output/a.txt", "output/b.txt"])
    harness = executor(judge_api_key_env="FIXTURE_CREDENTIAL", judge_model="fixture-independent",
                       judge_base_url="https://judge.example/v1", judge_max_tokens=2048, judge_temperature=0)
    monkeypatch.setattr(harness, "_extract", AsyncMock(side_effect=lambda base, paths: extract(base, paths)))
    receipt = run(harness._harvey(row, base, result, "Response", "harvey_train"))
    documents = [json.loads(r["messages"][1]["content"]) for r in requests]
    assert [len(item["criteria"]) for item in documents] == [24, 1, 1]
    assert [item["deliverables"][0]["path"] for item in documents] == ["output/a.txt", "output/a.txt", "output/b.txt"]
    assert all(r["max_tokens"] == 2048 and r["temperature"] == 0 and "extra_body" not in r for r in requests)
    assert receipt["all_pass"] == 1 and receipt["judge_mode"] == "batched"
    assert clients[0]["base_url"] == "https://judge.example/v1"
    assert WorkspaceSettings(judge_mode="batched").mode_for("harvey_val") == "batched"
    assert WorkspaceSettings(train_judge_mode="per_criterion").mode_for("harvey_train") == "per_criterion"


def test_harvey_val_can_have_independent_provider(monkeypatch, fake_judge, tmp_path):
    clients, requests = fake_judge
    benchmark = WorkspaceSuite({
        "name": "workspace_dualgate", "judge_api_key_env": "FIXTURE_CREDENTIAL",
        "val_judge": {"judge_model": "gemini-independent", "judge_base_url": "https://judge.example/v1",
                      "judge_api_key_env": "FIXTURE_VAL", "judge_temperature": 0},
    })
    monkeypatch.setenv("FIXTURE_VAL", "local-val-fixture")
    row, base, result = harbor_fixture(tmp_path, [rubric("a", "a.txt")], ["output/a.txt"])
    harness = ws.WorkspaceExecutor(benchmark)
    monkeypatch.setattr(harness, "_extract", AsyncMock(side_effect=lambda base, paths: extract(base, paths)))
    assert run(harness._harvey(row, base, result, "", "harvey_train"))["judge_mode"] == "batched"
    assert run(harness._harvey(row, base, result, "", "harvey_val"))["judge_mode"] == "per_criterion"
    assert [call["model"] for call in requests] == ["deepseek-flash", "gemini-independent"]
    assert [client["base_url"] for client in clients] == ["https://api.deepseek.com/v1", "https://judge.example/v1"]
    assert "extra_body" not in requests[1] and requests[1]["temperature"] == 0


def test_harvey_judge_concurrency_is_shared(monkeypatch, fake_judge, tmp_path):
    row, base, result = harbor_fixture(tmp_path, [rubric(str(i), "a.txt") for i in range(8)], ["output/a.txt"])
    harness = executor(judge_api_key_env="FIXTURE_CREDENTIAL", judge_concurrency=2)
    monkeypatch.setattr(harness, "_extract", AsyncMock(side_effect=lambda base, paths: extract(base, paths)))
    active = peak = 0

    async def judge(client, settings, messages):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0)
        active -= 1
        data = json.loads(messages[1]["content"])
        return judge_response([item["id"] for item in data["criteria"]]), []

    monkeypatch.setattr(ws, "request_model", judge)

    async def concurrent_grading():
        return await asyncio.gather(*(harness._harvey(row, base, result, "", "harvey_val") for _ in range(2)))

    assert all(r["all_pass"] == 1 for r in run(concurrent_grading()))
    assert peak == 2


@pytest.mark.parametrize("failure", ["429", "503", "stream-drop", "timeout"])
def test_judge_transient_failures_retry_then_raise(monkeypatch, failure):
    fixture_request = httpx.Request("POST", "https://judge.example")
    failures = {
        "429": APIStatusError("fixture", response=httpx.Response(429, request=fixture_request), body=None),
        "503": APIStatusError("fixture", response=httpx.Response(503, request=fixture_request), body=None),
        "stream-drop": httpx.RemoteProtocolError("fixture"),
        "timeout": TimeoutError(),
    }
    create = AsyncMock(side_effect=failures[failure])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    backoff = AsyncMock()
    monkeypatch.setattr(ws.asyncio, "sleep", backoff)
    with pytest.raises(InfraError, match="request failed"):
        run(ws.request_model(client, WorkspaceSettings(), []))
    assert create.await_count == 3
    assert [call.args[0] for call in backoff.await_args_list] == [1, 2]


def test_judge_phase_cancels_pending_and_does_not_emit_zero(monkeypatch, fake_judge, tmp_path):
    row, base, result = harbor_fixture(tmp_path, [rubric("pending", "a.txt")], ["output/a.txt"])
    harness = executor(judge_api_key_env="FIXTURE_CREDENTIAL")
    monkeypatch.setattr(harness, "_extract", AsyncMock(side_effect=lambda base, paths: extract(base, paths)))
    harness.val_settings = replace(harness.val_settings, judge_phase_timeout=1)
    cancelled = []

    async def waiting(*_args):
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(ws, "request_model", waiting)
    with pytest.raises(InfraError, match="phase deadline"):
        run(harness._harvey(row, base, result, "", "harvey_val"))
    assert cancelled == [True]


def test_empty_rubric_and_bad_response_are_distinct(monkeypatch, fake_judge, tmp_path):
    row, base, result = harbor_fixture(tmp_path, [], [])
    harness = executor(judge_api_key_env="FIXTURE_CREDENTIAL")
    receipt = run(harness._harvey(row, base, result, "", "harvey_val"))
    assert (receipt["passed_criteria"], receipt["total_criteria"], receipt["all_pass"]) == (0, 0, 0)
    (base / "grader/task.json").write_text(json.dumps({"criteria": [rubric("a")], "instructions": "Fixture"}))
    row["criteria_count"] = 1
    failure = AsyncMock(return_value=(judge_response(["a"], valid=False), []))
    monkeypatch.setattr(ws, "request_model", failure)
    with pytest.raises(InfraError, match="malformed JSON twice"):
        run(harness._harvey(row, base, result, "response", "harvey_val"))
    assert failure.await_count == 2


def task_row(split, *, reward=0, passed=None, total=None, diagnostics=None):
    task = Task(f"{split}-fixture", "must not enter worker evidence", metadata={"split": split})
    metrics = {"reward": reward}
    if passed is not None:
        metrics.update(passed_criteria=passed, total_criteria=total)
    return Trajectory(task=task, metrics=metrics, messages=(diagnostics or {},))


def test_pooled_harvey_repeats_differ_from_task_macro():
    assert summarize_harvey([{"passed_criteria": 1, "total_criteria": 1},
                            {"passed_criteria": 1, "total_criteria": 9},
                            {"passed_criteria": 0, "total_criteria": 0}]) == {
        "passed_criteria": 2, "total_criteria": 10, "micro_avg": 0.2,
        "macro_avg": pytest.approx((1 + 1 / 9) / 3), "all_pass": pytest.approx(1 / 3),
    }
    scores = TwoGateWorkspacePromotion.scores([
        task_row("clawgym_val", reward=0.4), task_row("clawgym_val", reward=0.6),
        task_row("harvey_val", passed=1, total=1), task_row("harvey_val", passed=1, total=9),
    ])
    assert scores["claw_score"] == 0.5 and scores["harvey_micro_avg"] == 0.2
    assert scores["harvey_macro_avg"] == pytest.approx((1 + 1 / 9) / 2)


def test_workspace_thresholds_and_writable_growth():
    gate = WorkspaceGate()
    assert gate.accepts((0.5, 0.4), (0.5, 0.404), 100, 110)
    assert not gate.accepts((0.5, 0.4), (0.5, 0.403), 100, 110)
    assert not gate.accepts((0.5, 0.4), (0.499, 0.405), 100, 110)
    assert not gate.accepts((0.5, 0.4), (0.5, 0.405), 100, 111)
    assert gate.accepts((0.5, 0.4), (0.5, 0.405), 100, 0)
    assert gate.growth_ratio(0, 1) == 1
    snap = SimpleNamespace(files={"p": SimpleNamespace(component="prompt", content=b"abc"),
                                  "k": SimpleNamespace(component="skill", content=b"x" * 100)})
    assert TwoGateWorkspacePromotion.size(snap) == 3


def bare_algorithm(cls, configs=None):
    algorithm = object.__new__(cls)
    algorithm.config = SimpleNamespace(**{**cls._defaults, **(configs or {})})
    return algorithm


def test_workspace_evolution_repeats_worker_budget_and_telemetry():
    algo = bare_algorithm(TwoGateWorkspacePromotion)
    algo.benchmark = SimpleNamespace(train=["train"], dev=["val"], config={})
    entry = SimpleNamespace(content=b"x" * 120, component="prompt")
    seed = SimpleNamespace(files={"prompt": entry})
    trial = SimpleNamespace(files={"prompt": replace_value(entry, b"x" * 125)})
    algo.target = SimpleNamespace(snapshot=seed)
    prompts = []

    async def ask(prompt):
        prompts.append(prompt)
        return "candidate"

    algo.worker = SimpleNamespace(config={"llm": {}}, ask=ask)
    # Core Config provides both mapping and attribute access.
    from flashrsi.config import Config
    algo.config = Config(vars(algo.config))
    algo._check()
    algo._instruction_path = lambda: "prompt"
    algo._candidate = lambda *_: trial
    algo._promote = lambda _seed, incumbent, **_: incumbent
    records, calls = [], []
    algo._record = lambda *args: records.append(args)

    async def evaluate(snapshot, tasks, label):
        calls.append(label)
        if tasks == ["train"]:
            return [
                task_row("clawgym_train", diagnostics={"checker_stderr": "TRACE " + "x" * 10000}),
                task_row("harvey_train", passed=0, total=1, diagnostics={
                    "verdicts": [{"pass": False, "reason": "MISSING CITATION " + "y" * 10000}],
                }),
            ]
        return [task_row("clawgym_val", reward=0.5),
                task_row("harvey_val", passed=21 if snapshot is trial else 20, total=100)]

    algo._evaluate = evaluate
    assert run(algo.evolve()) is trial
    assert calls == ["baseline-val-1", "baseline-val-2", "workspace-train", "workspace-val-1", "workspace-val-2"]
    assert algo.worker.config["llm"].get("max_tokens") is None
    cl, hv = prompts[0].split("\nClawGym diagnostics:\n")[1].split("\nHarvey LAB diagnostics:\n")
    assert len(cl) <= 8000 and len(hv) <= 8000 and "TRACE " in cl and "MISSING CITATION " in hv
    assert "must not enter worker evidence" not in prompts[0]
    telemetry = records[0][1]
    assert (telemetry["harvey_macro_avg"], telemetry["harvey_all_pass"], telemetry["claw_score"]) == (0.21, 0.0, 0.5)
    assert telemetry["growth_ratio"] == 5 / 120
    calls.clear()
    assert run(algo.evolve(smoke=True)) is trial
    assert calls == ["baseline-val-1", "workspace-train", "workspace-val-1"]


def replace_value(entry, value):
    return SimpleNamespace(content=value, component=entry.component)


@pytest.mark.parametrize("payload,expected", [
    ({"metric": "Spearman", "normalized_score": 0.8, "raw_mean": -1}, 0.8),
    ({"metric_name": "Spearman", "raw_mean": -0.4}, 0.3),
    ({"metadata": {"direction_normalized_score": 0.89}}, 0.89),
    ({"mae": 3}, 0.25), ({"auroc": 0.7}, 0.7), ({"auprc": 0.6}, 0.6),
    ({"metric":"Spearman", "candidate_valid": False, "normalized_score": 0.0}, 0.0),
])
def test_direction_normalized_receipt(payload, expected):
    assert normalized_endpoint_score(Trajectory(messages=(payload,))) == pytest.approx(expected)


@pytest.mark.parametrize("payload", [{"mae": 0.1}, {"metric": "MAE", "normalized_score": 0.9},
                                      {"spearman": 1.1}, {"normalized_score": float("nan")}])
def test_tdc_gate_rejects_wrong_metric_or_invalid_receipt(payload):
    row = Trajectory(task=Task("half_life_obach", "fixture"), messages=(payload,))
    with pytest.raises(InfraError):
        normalized_endpoint_score(row, expected_metric="spearman")


@pytest.mark.parametrize("trial_scores,selected", [((0.9, 0.7, 0.7), True),
                                                     ((0.9, 0.69, 0.7), False),
                                                     ((0.7, 0.7, 0.7), False)])
def test_sequential_memory_requires_non_regression_on_each_spearman_endpoint(trial_scores, selected):
    algo = bare_algorithm(SequentialMemory)
    dev = tuple(Task(identifier, "fixture") for identifier in sorted(TDC_SPEARMAN_VAL))
    train = Task("dev-fixture", "fixture")
    algo.benchmark = SimpleNamespace(name="tdc_admet", dev=dev, train=[train])
    files = {f"memory/{kind}_memory.jsonl": SimpleNamespace(content=b"")
             for kind in ("solution", "refinement", "execution")}
    seed = SimpleNamespace(id="seed", files=files)
    trial = SimpleNamespace(id="trial")
    algo.target = SimpleNamespace(snapshot=seed)
    algo._candidate = lambda *_: trial
    decisions = {}
    algo._promote = lambda _seed, incumbent, **kw: decisions.update(kw) or incumbent

    async def evaluate(snapshot, tasks, label):
        if tasks == [train]:
            return [Trajectory(task=train, messages=({"type": "fixture", "score": 0.5},))]
        scores = (0.7,) * len(tasks) if snapshot is seed else trial_scores
        return [Trajectory(task=task, messages=({"metric": "Spearman", "normalized_score": score},))
                for task, score in zip(tasks, scores, strict=True)]

    algo._evaluate = evaluate
    assert run(algo.evolve()) is (trial if selected else seed)
    assert decisions["candidate_endpoints"] == dict(zip(sorted(TDC_SPEARMAN_VAL), trial_scores))
    algo.benchmark.dev = dev[:2]
    with pytest.raises(ValueError, match="all three"):
        run(algo.evolve())
