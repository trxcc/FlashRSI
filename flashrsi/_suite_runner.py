"""Register native suite benchmarks and configure their execution profiles.

Each family is imported lazily so legacy launch and sealed-split guards can
operate before the optional benchmark implementations are available.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from flashrsi.benchmarks import _suite
from flashrsi.config import Config
from flashrsi.core import Agent, Benchmark, Task, Trajectory
from flashrsi.integrations.sandbox_runtime import RELAY_PLACEHOLDER

HARBOR_NAMES = frozenset(
    {
        "tb21_clean",
        "swegym_lite",
        "tb41_sealed",
        "tbs01_sealed",
        "swebench_verified_sealed",
    }
)
_WORKSPACE_NAMES = frozenset(
    {
        "clawgym",
        "harvey_workspace",
        "workspace_dualgate",
        "clawgym_bench_sealed",
        "harvey_lab_sealed",
        "jobbench_sealed",
        "gdpval_sealed",
    }
)


def _build_suite_benchmark(config: Config) -> Benchmark:
    _suite.guard(config)
    name = config["name"]
    if name in HARBOR_NAMES:
        from flashrsi.benchmarks.harbor_suite import HarborSuite

        return HarborSuite(config)
    if name in _WORKSPACE_NAMES:
        from flashrsi.benchmarks.workspace_suite import WorkspaceSuite

        return WorkspaceSuite(config)
    from flashrsi.benchmarks.design_suite import (
        TDCADMET,
        EngDesignOpen,
        FrontierEng,
        KernelBench,
    )

    families = {
        "engdesign_open": EngDesignOpen,
        "frontier_eng": FrontierEng,
        "tdc_admet": TDCADMET,
        "kernelbench": KernelBench,
    }
    return families[name](config)


SUITE_BENCHMARKS = dict.fromkeys(sorted(_suite.NAMES), _build_suite_benchmark)


def bind_suite(target: Agent, benchmark: Benchmark) -> None:
    if benchmark.name in HARBOR_NAMES:
        from flashrsi.integrations.harbor_suite import bind

        bind(target, benchmark)
    elif benchmark.name in _WORKSPACE_NAMES:
        from flashrsi.integrations.workspace_suite import WorkspaceExecutor

        target._env_runner = WorkspaceExecutor(benchmark).run
    else:
        from flashrsi.integrations.design_suite import bind

        bind(target, benchmark)
    target._require_native_env = True


def suite_target_config(section: Config, *, native: bool) -> Config:
    if not native and section.name != "pi_agent":
        return section
    section = copy.deepcopy(section)
    if section.name == "pi_agent":
        section.name = "pi"
    if native:
        if (
            section.name in {"openclaw", "miniagent"}
            and section.get("max_turns") is None
        ):
            section.pop("max_turns", None)
        section.llm.api_key = RELAY_PLACEHOLDER
        section.llm.setdefault("provider_name", "deepseek")
        for name in ("auth_token", "oauth_token"):
            section.llm.pop(name, None)
    return section


def _execution_config(
    config: Config,
    *,
    smoke: bool | None = None,
    limit: int | None = None,
) -> Config:
    """Resolve an invocation profile before creating a benchmark or client."""
    _suite.guard(config.benchmark)
    enabled = config.run.get("smoke", False) if smoke is None else smoke
    if type(enabled) is not bool:
        raise ValueError("run.smoke must be a boolean")
    if not enabled:
        if config.run.get("smoke"):
            config = copy.deepcopy(config)
            config.run.smoke = False
        return config
    if (
        config.run.get("mode", "run") != "run"
        or config.benchmark.name not in _suite.ACTIVE
    ):
        raise ValueError("--smoke is available for native suite runs")
    if config.benchmark.get("split") == "val":
        raise PermissionError(
            "suite smoke is restricted to Train and open development tasks"
        )
    config = copy.deepcopy(config)
    run, benchmark, algorithm = config.run, config.benchmark, config.algorithm
    run.smoke = True
    run.limit = run.get("limit", 1) if limit is None else limit
    if type(run.limit) is not int or run.limit < 1:
        raise ValueError("smoke limit must be a positive integer")
    run.iterations = 1
    output = Path(run.output_dir)
    if output.parent.name == "suite" and output.parent.parent.name == "runs":
        run.output_dir = str(output.parent.with_name("suite_smoke") / output.name)

    benchmark.smoke = True
    for field in ("smoke_oracle_budget", "smoke_seeds"):
        benchmark.pop(field, None)
    if "task_ids" in benchmark:
        benchmark.smoke_task_ids = benchmark.task_ids
    if benchmark.name in {"tb21_clean", "swegym_lite"}:
        benchmark.smoke_whitelist_only = True
    if benchmark.name == "tb21_clean":
        benchmark.n_attempts = 1
    if benchmark.name == "tdc_admet":
        benchmark.seeds = [1]
        benchmark.oracle_budget = 2
        config.agent.timeout = min(config.agent.get("timeout", 7800), 7800)
    elif benchmark.name == "frontier_eng":
        benchmark.mode = "synthetic_dev"
        benchmark.oracle_budget = 2
        config.agent.timeout = min(config.agent.get("timeout", 1800), 1800)
    elif benchmark.name == "kernelbench":
        benchmark.oracle_budget = 1
    elif benchmark.name == "workspace_dualgate":
        algorithm.val_repeats = 1
        algorithm.rounds = 1
        algorithm.smoke = True

    if algorithm.name == "ace":
        algorithm.num_epochs = 1
        algorithm.max_steps = 1
        algorithm.val_repeats = 0
        # The smoke invocation grades the selected open task without evolution.
        algorithm.mode = "online"
    return config


def _aggregate_metrics(results: Sequence[Trajectory]) -> dict[str, float]:
    """Average shared task metrics and pool Harvey criterion counts."""
    if not results:
        return {}
    metrics = {
        key: mean(item.metrics[key] for item in results)
        for key in results[0].metrics
        if all(key in item.metrics for item in results)
    }
    harvey = [item.metrics for item in results if "total_criteria" in item.metrics]
    if harvey:
        from flashrsi.benchmarks.workspace_suite import summarize_harvey

        metrics.update(summarize_harvey(harvey))
    return metrics


def _harvey_validation_metrics(run_dir: Path, snapshot_id: str) -> dict[str, float]:
    """Pool the selected Snapshot's recorded per-criterion Val attempts."""
    rows = []
    folder = run_dir / "trajectories"
    for pattern in (
        "*-validation*.json",
        "*-baseline-val-*.json",
        "*-workspace-val-*.json",
    ):
        for path in sorted(folder.glob(pattern)):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("snapshot_id") == snapshot_id:
                rows.extend(
                    row["metrics"]
                    for row in report["trajectories"]
                    if row["task"]["metadata"].get("split") == "harvey_val"
                )
    if not rows:
        return {}
    from flashrsi.benchmarks.workspace_suite import summarize_harvey

    return summarize_harvey(rows)


def _smoke_tasks(config: Config, benchmark: Benchmark) -> tuple[Task, ...]:
    active = benchmark.dev if benchmark.name == "frontier_eng" else benchmark.train
    smoke_ids = config.benchmark.get("smoke_task_ids")
    if smoke_ids is not None:
        by_id = {task.task_id: task for task in active}
        if (
            not isinstance(smoke_ids, list)
            or not smoke_ids
            or any(
                not isinstance(task_id, str) or task_id not in by_id
                for task_id in smoke_ids
            )
            or len(set(smoke_ids)) != len(smoke_ids)
        ):
            raise PermissionError("smoke_task_ids must select distinct active tasks")
        active = tuple(by_id[task_id] for task_id in smoke_ids)
    tasks = active[: config.run.limit]
    if not tasks or any(task.env is None for task in tasks):
        raise PermissionError("no reviewed active suite tasks available for smoke")
    return tasks
