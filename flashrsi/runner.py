"""Prepare benchmark-bound agents and run search or frozen evaluation.

The host owns model roles, run directories, and score summaries. Native suite
assets and integration dependencies are resolved only for the selected run.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flashrsi._records import _trajectory_record, _write_record
from flashrsi._run_summary import _finish_summary
from flashrsi._suite_runner import (
    HARBOR_NAMES,
    SUITE_BENCHMARKS,
    _aggregate_metrics,
    _execution_config,
    _harvey_validation_metrics,
    _smoke_tasks,
    bind_suite,
    suite_target_config,
)
from flashrsi.agents import (
    LLM,
    ClaudeCode,
    Codex,
    Hermes,
    JiuwenSwarm,
    MiniAgent,
    OpenClaw,
    OpenCode,
    Pi,
)
from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.algorithms import (
    ACE,
    ADAS,
    AHE,
    GEPA,
    OPRO,
    RRSI,
    Arbor,
    DarwinX,
    EvoHarness,
    EvoPrompt,
    MetaHarness,
    ModularRSI,
    SkillAdam,
    SkillOpt,
)
from flashrsi.algorithms.promotion import SequentialMemory, TwoGateWorkspacePromotion
from flashrsi.benchmarks import (
    AIME,
    BBEH,
    BFCL,
    HLE,
    MATH,
    NQ,
    ALFWorld,
    AppWorldBenchmark,
    BrowseComp,
    DeepSearchQA,
    DeepSWE,
    HarveyLAB,
    HotpotQA,
    JobBench,
    LiveMathBench,
    MuSiQue,
    OfficeQAPro,
    SearchQA,
    SpreadsheetBench2,
    SuperGPQA,
    SWEBench,
    SWEBenchPro,
    Tau2Benchmark,
    Tau3Banking,
    TerminalBench,
    TravelPlanner,
    TriviaQA,
    TwoWikiMultiHopQA,
    WebShop,
    WideSearch,
    WorkBench,
    _suite,
)
from flashrsi.config import Config, prepare_run_dir
from flashrsi.core import Agent, Algorithm, Benchmark, Snapshot, Trajectory
from flashrsi.integrations.sandbox_runtime import RELAY_PLACEHOLDER

_AGENTS = {
    "program_search": ProgramSearchTarget,
    "pi_agent": Pi,
    "jiuwenswarm": JiuwenSwarm,
    "miniagent": MiniAgent,
    "claude_code": ClaudeCode,
    "codex": Codex,
    "pi": Pi,
    "hermes": Hermes,
    "openclaw": OpenClaw,
    "opencode": OpenCode,
}
_BENCHMARKS = {
    **SUITE_BENCHMARKS,
    "webshop": WebShop,
    "alfworld": ALFWorld,
    "tau2": Tau2Benchmark,
    "aime": AIME,
    "bbeh": BBEH,
    "supergpqa": SuperGPQA,
    "twowiki_multihopqa": TwoWikiMultiHopQA,
    "travelplanner": TravelPlanner,
    "harvey_lab": HarveyLAB,
    "bfcl": BFCL,
    "browsecomp": BrowseComp,
    "deepsearchqa": DeepSearchQA,
    "deepswe": DeepSWE,
    "widesearch": WideSearch,
    "workbench": WorkBench,
    "appworld": AppWorldBenchmark,
    "searchqa": SearchQA,
    "hotpotqa": HotpotQA,
    "musique": MuSiQue,
    "triviaqa": TriviaQA,
    "nq": NQ,
    "math": MATH,
    "livemathbench": LiveMathBench,
    "hle": HLE,
    "officeqa_pro": OfficeQAPro,
    "swe_bench": SWEBench,
    "swe_bench_pro": SWEBenchPro,
    "terminal_bench": TerminalBench,
    "jobbench": JobBench,
    "spreadsheetbench2": SpreadsheetBench2,
    "tau3_banking": Tau3Banking,
}
_ALGORITHMS = {
    "two_gate_workspace": TwoGateWorkspacePromotion,
    "sequential_memory": SequentialMemory,
    "arbor": Arbor,
    "adas": ADAS,
    "darwinx": DarwinX,
    "evo_harness": EvoHarness,
    "meta_harness": MetaHarness,
    "modular_rsi": ModularRSI,
    "rrsi": RRSI,
    "ace": ACE,
    "ahe": AHE,
    "skilladam": SkillAdam,
    "skillopt": SkillOpt,
    "opro": OPRO,
    "gepa": GEPA,
    "evoprompt": EvoPrompt,
}
_BENCHMARK_BINDINGS = (
    (ALFWorld, "alfworld"),
    (WebShop, "webshop"),
    (WorkBench, "workbench"),
    (TravelPlanner, "travelplanner"),
    (HarveyLAB, "harvey_lab"),
    (WideSearch, "widesearch"),
    (DeepSearchQA, "deepsearchqa"),
    (BrowseComp, "browsecomp"),
    (LiveMathBench, "livemathbench"),
    (HLE, "hle"),
    (OfficeQAPro, "officeqa_pro"),
    (SWEBench, "swe_bench"),
    (DeepSWE, "deepswe"),
    (JobBench, "jobbench"),
    (SWEBenchPro, "swe_bench_pro"),
)
_LOGGER = logging.getLogger(__name__)


def _make_agent(cls: type[Agent], section: Config, label: str) -> Agent:
    from flashrsi._startup import _role

    dependencies = {
        MiniAgent: (("openai", "requests"), "all"),
        ClaudeCode: (("claude_agent_sdk",), "all"),
        Codex: (("httpx", "PIL", "tree_sitter", "tree_sitter_bash"), "all"),
        Hermes: (("httpx",), "all"),
        OpenClaw: (("httpx",), "all"),
        OpenCode: (("httpx",), "all"),
        LLM: (("openai",), "all"),
    }
    if cls is Pi:
        from flashrsi.agents.pi_agent import _executable

        _executable(section)
    for module in dependencies.get(cls, ((), ""))[0]:
        if importlib.util.find_spec(module) is None:
            raise ValueError(
                f"Missing {module}; install flashrsi-lite[{dependencies[cls][1]}]"
            )
    if cls in dependencies or cls in (JiuwenSwarm, Pi):
        _role(section, label, cls.name)
    return cls(section)


async def _prepare_target(
    config: Config,
    *,
    snapshot: Snapshot | None = None,
    smoke: bool | None = None,
) -> tuple[Agent, Benchmark]:
    """Share benchmark bindings between search and frozen-Snapshot evaluation."""
    config = _execution_config(config, smoke=smoke)
    from flashrsi._startup import _preflight
    from flashrsi.benchmarks import _prepared

    await _preflight(config)
    native = config.benchmark.name in _suite.ACTIVE
    section = suite_target_config(config.agent, native=native)
    target = _make_agent(_lookup(_AGENTS, section.name, "agent"), section, "agent")
    if snapshot is not None:
        try:
            target.update(snapshot)
        except ValueError as exc:
            raise ValueError(
                f"Cannot evaluate this Snapshot: {exc}. Use the original Agent type and "
                "root source tree (agent.source_dir); evaluation does not rebase snapshots."
            ) from exc
    prepared = _prepared.selected(config.benchmark)
    if prepared:
        config = copy.deepcopy(config)
        config.benchmark = _prepared.prepare(config.benchmark)
    selected = target.snapshot.id
    benchmark = _lookup(_BENCHMARKS, config.benchmark.name, "benchmark")(
        config.benchmark
    )
    if native:
        bind_suite(target, benchmark)
    _bind_benchmark(target, benchmark, config.benchmark)
    if target.snapshot.id != selected:
        raise ValueError(
            f"{benchmark.name} integration replaced the selected harness Snapshot; "
            "task environments and tools must be provided outside Snapshot files"
        )
    if prepared:
        benchmark.load()
        benchmark.metadata = MappingProxyType(
            {**benchmark.metadata, "prepared": config.benchmark["_prepared_source"]}
        )
    return target, benchmark


def _bind_benchmark(target: Agent, benchmark: Benchmark, config: Config) -> None:
    from flashrsi._startup import _supplied_context

    # These integrations bind the same target/benchmark pair. Keep imports
    # local to the selected benchmark to avoid loading other optional stacks.
    for benchmark_type, module in _BENCHMARK_BINDINGS:
        if isinstance(benchmark, benchmark_type):
            importlib.import_module(f"flashrsi.integrations.{module}").bind(
                target, benchmark
            )
    if _supplied_context(config):
        from flashrsi.integrations.supplied_context import bind

        bind(target)
    if isinstance(benchmark, HotpotQA) and benchmark.mode == "fullwiki":
        from flashrsi.integrations.hotpotqa import bind

        bind(target, benchmark)
    if isinstance(benchmark, MATH):
        benchmark._check()
    if isinstance(benchmark, (BFCL, AppWorldBenchmark)):
        module = "bfcl" if isinstance(benchmark, BFCL) else "appworld"
        importlib.import_module(f"flashrsi.integrations.{module}").bind(target, config)
    if isinstance(benchmark, (TerminalBench, SpreadsheetBench2)):
        from flashrsi.integrations.harbor import bind

        bind(target)
    if isinstance(benchmark, (Tau2Benchmark, Tau3Banking)):
        from flashrsi.integrations.tau2 import bind

        bind(
            target,
            config.get("user_simulator"),
            task_timeout=config.get("task_timeout"),
            judge=config.get("judge"),
        )


async def _prepare(config: Config, *, smoke: bool | None = None) -> Algorithm:
    """Initialize search, including its independent Worker and algorithm checks."""
    config = _execution_config(config, smoke=smoke)
    algorithm_type = _lookup(_ALGORITHMS, config.algorithm.name, "algorithm")
    worker_config = config.algorithm.worker
    _suite.guard(config.benchmark)
    if config.benchmark.name in _suite.ACTIVE:
        worker_config = copy.deepcopy(worker_config)
        worker_config.llm.api_key = os.getenv("DEEPSEEK_API_KEY") or RELAY_PLACEHOLDER
    if worker_config.type == "agent" and worker_config.name in {
        "jiuwenswarm",
        "program_search",
    }:
        raise ValueError("Selected agent is available as a Target only")
    worker_type = (
        LLM
        if worker_config.type == "llm"
        else _lookup(_AGENTS, worker_config.name, "worker")
    )
    target, benchmark = await _prepare_target(config)
    worker = _make_agent(worker_type, worker_config, "algorithm.worker")
    algorithm = algorithm_type(
        config.algorithm, target=target, worker=worker, benchmark=benchmark
    )
    return algorithm


async def _run(config: Config) -> tuple[Path, dict[str, Any]]:
    config = _execution_config(config)
    if config.run.get("smoke", False):
        return await _smoke(config)
    algorithm = await _prepare(config)
    worker, benchmark = algorithm.worker, algorithm.benchmark
    resolved = copy.deepcopy(config)
    resolved.algorithm = algorithm.config
    run_dir = prepare_run_dir(resolved)
    _artifact_paths(algorithm.target, benchmark, run_dir)
    algorithm._run_dir = run_dir
    _LOGGER.info("Experiment: %s", run_dir)
    try:
        if worker.component_rules:
            worker.snapshot.save(run_dir / "snapshots" / "worker.json")
        final = await algorithm.evolve()
        results = await algorithm._evaluate(final, benchmark.test, label="test")
        metrics = _aggregate_metrics(results)
        summary = {
            "status": "finished",
            "snapshot_id": final.id,
            "primary_metric": benchmark.primary_metric,
            "test_score": metrics.get(benchmark.primary_metric),
            "test_tasks": len(results),
            "test_metrics": metrics,
            "benchmark": dict(benchmark.metadata),
        }
        if config.benchmark.name in {"harvey_workspace", "workspace_dualgate"}:
            validation = _harvey_validation_metrics(run_dir, final.id)
            if validation:
                summary["validation_metrics"] = validation
        _write_record(run_dir / "result.json", summary, config)
    except BaseException as exc:
        _write_record(
            run_dir / "error.json",
            {"status": "error", "error": str(exc) or type(exc).__name__},
            config,
        )
        raise
    finally:
        _finish_summary(run_dir)
    return run_dir, summary


def _lookup(choices, name, role):
    try:
        return choices[name]
    except KeyError:
        raise ValueError(
            f"Unsupported {role}: {name}; available: {', '.join(choices)}"
        ) from None


def _artifact_paths(target: Agent, benchmark: Benchmark, run_dir: Path) -> None:
    if isinstance(benchmark, HarveyLAB) and benchmark._artifacts_dir is None:
        benchmark._artifacts_dir = run_dir / "artifacts" / "harvey_lab"
    if benchmark.name in HARBOR_NAMES or isinstance(
        benchmark, (TerminalBench, SpreadsheetBench2, DeepSWE, JobBench, SWEBenchPro)
    ):
        target.config.setdefault("harbor", {}).setdefault(
            "artifacts_dir", str(run_dir / "artifacts" / "harbor")
        )


async def _prepare_evaluation(config: Config) -> tuple[Agent, Benchmark]:
    snapshot = Snapshot.load(Path(config.run.snapshot))
    target, benchmark = await _prepare_target(config, snapshot=snapshot)
    target.set_scope({name: "r" for name in target.component_rules})
    benchmark.load()
    if not benchmark.test:
        raise ValueError(
            "Evaluation requires a non-empty benchmark.test; configure explicit test roles"
        )
    if (
        type(target).rollout is Agent.rollout
        and target._env_runner is None
        and any(t.env is not None for t in benchmark.test)
    ):
        raise ValueError(
            "Evaluation target needs environment-aware rollout for these tasks"
        )
    return target, benchmark


async def _evaluate(config: Config) -> tuple[Path, dict[str, Any]]:
    """Evaluate a saved resource version without constructing a Worker or Algorithm."""
    target, benchmark = await _prepare_evaluation(config)
    snapshot = target.snapshot
    run_dir = prepare_run_dir(config)
    _artifact_paths(target, benchmark, run_dir)
    source = {
        "snapshot_path": str(Path(config.run.snapshot).resolve()),
        "snapshot_id": snapshot.id,
        "root_id": snapshot.root_id,
        "parent_id": snapshot.parent_id,
        "agent_type": snapshot.agent_type,
    }
    try:
        snapshot.save(run_dir / "snapshots" / f"{snapshot.id}.json")
        _write_record(
            run_dir / "evaluation.json",
            {"source": source, "benchmark": dict(benchmark.metadata), "split": "test"},
            config,
        )
        results = await _grade_evaluation(config, target, benchmark, run_dir)
        metrics = _aggregate_metrics(results)
        summary = {
            "status": "finished",
            "mode": "evaluate",
            "source": source,
            "snapshot_id": snapshot.id,
            "primary_metric": benchmark.primary_metric,
            "test_score": metrics[benchmark.primary_metric],
            "test_tasks": len(results),
            "test_metrics": metrics,
            "benchmark": dict(benchmark.metadata),
        }
        _write_record(run_dir / "result.json", summary, config)
    except BaseException as exc:
        _write_record(
            run_dir / "error.json",
            {
                "status": "error",
                "mode": "evaluate",
                "source": source,
                "error": str(exc) or type(exc).__name__,
            },
            config,
        )
        raise
    finally:
        _finish_summary(run_dir)
    return run_dir, summary


async def _grade_evaluation(
    config: Config,
    target: Agent,
    benchmark: Benchmark,
    run_dir: Path,
) -> tuple[Trajectory, ...]:
    snapshot = target.snapshot
    trajectories = await target.rollout(benchmark.test)
    # Preserve completed requests even when grading fails.
    _write_record(
        run_dir / "trajectories" / "000001-rollout.json",
        {
            "snapshot_id": snapshot.id,
            "trajectories": [_trajectory_record(t) for t in trajectories],
        },
        config,
    )
    results = benchmark.grade(trajectories)
    if tuple(t.task for t in results) != tuple(benchmark.test):
        raise ValueError("rollout/grade must preserve the requested tasks and order")
    record = {
        "snapshot_id": snapshot.id,
        "trajectories": [_trajectory_record(t) for t in results],
    }
    _write_record(run_dir / "trajectories" / "000002-test.json", record, config)
    # The scored record supersedes its raw copy for usage accounting.
    (run_dir / "trajectories" / "000001-rollout.json").unlink()
    return results


async def _smoke(
    config: Config, limit: int | None = None
) -> tuple[Path, dict[str, Any]]:
    """Grade an explicit train/dev slice through the ordinary target bindings."""
    config = _execution_config(config, smoke=True, limit=limit)
    target, benchmark = await _prepare_target(config)
    benchmark.load()
    tasks = _smoke_tasks(config, benchmark)
    run_dir = prepare_run_dir(config)
    _artifact_paths(target, benchmark, run_dir)
    try:
        target.snapshot.save(run_dir / "snapshots" / f"{target.snapshot.id}.json")
        trajectories = benchmark.grade(await target.rollout(tasks))
        _write_record(
            run_dir / "trajectories/000001-smoke.json",
            {"trajectories": [_trajectory_record(item) for item in trajectories]},
            config,
        )
        summary = {
            "status": "finished",
            "mode": "smoke",
            "snapshot_id": target.snapshot.id,
            "primary_metric": benchmark.primary_metric,
            "benchmark": dict(benchmark.metadata),
            "metrics": _aggregate_metrics(trajectories),
            "tasks": [
                {
                    "task_id": row.task.task_id,
                    "status": row.status,
                    "metrics": dict(row.metrics),
                }
                for row in trajectories
            ],
        }
        _write_record(run_dir / "result.json", summary, config)
        return run_dir, summary
    except BaseException as exc:
        _write_record(
            run_dir / "error.json",
            {"status": "error", "type": type(exc).__name__, "message": str(exc)},
            config,
        )
        raise
    finally:
        _finish_summary(run_dir)
