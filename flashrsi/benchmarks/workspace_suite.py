"""Load frozen ClawGym and Harvey Workspace manifests and grade host receipts.

The trusted manifests and checker inputs stay on the host. Asset roots follow
suite configuration and FLASHRSI_SUITE_WORKSPACE_TRACK_ROOT.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from statistics import mean
from typing import Any, TypeGuard

from flashrsi.benchmarks._suite import (
    DEFAULT_AGENT_SHELLS,
    SuiteBenchmark,
    confined,
    read_json,
    resolve_root,
    sha256,
)
from flashrsi.core import Benchmark, InfraError, Task, Trajectory

_TRACK_DIRECTORY = "workspace_track"
TRAIN = {"clawgym_train", "harvey_train"}
DEVELOPMENT = {"clawgym_val", "harvey_val"}
SEALED = {"clawgym_bench", "harvey_test", "jobbench", "gdpval"}

# Planning defaults for the two OOD suites do not enable their runners.
SOLVER_TIMEOUTS = {"clawgym": 660, "harvey": 3600, "jobbench": 3600, "gdpval": 3600}
JUDGE_PROVIDER_FIELDS = frozenset({
    "judge_model", "judge_base_url", "judge_api_key_env", "judge_thinking",
    "judge_reasoning_effort", "judge_temperature",
})
JUDGE_USAGE_KEYS = (
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "prompt_cache_hit_tokens",
)


_POSITIVE_INTEGER_SETTINGS = (
    "claw_solver_timeout", "long_episode_timeout", "harvey_solver_timeout",
    "claw_checker_timeout", "claw_judge_timeout", "extractor_timeout", "judge_timeout",
    "judge_phase_timeout", "cpus", "pids", "solver_pids", "batch_size", "judge_max_tokens",
    "judge_concurrency",
)
_JUDGE_MODES = ("per_criterion", "batched")


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_positive_integer(name: str, value: object, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not _is_integer(value) or value < 1:
        suffix = " or null" if nullable else ""
        raise ValueError(f"benchmark.{name} must be a positive integer{suffix}")


def _validate_docker_size(name: str, value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*[bkmgBKMG]?", value):
        raise ValueError(f"benchmark.{name} must be a positive Docker size")


def _validate_judge_provider(settings: WorkspaceSettings) -> None:
    if not isinstance(settings.judge_model, str) or not settings.judge_model.strip():
        raise ValueError("benchmark.judge_model must name a model")
    if not isinstance(settings.judge_api_key_env, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", settings.judge_api_key_env
    ):
        raise ValueError("benchmark.judge_api_key_env must name an environment variable")
    thinking = settings.judge_thinking
    if thinking is not None and (
        not isinstance(thinking, Mapping)
        or set(thinking) != {"type"}
        or thinking["type"] not in ("enabled", "disabled")
    ):
        raise ValueError("benchmark.judge_thinking must select enabled or disabled")
    temperature = settings.judge_temperature
    if temperature is not None and (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise ValueError("benchmark.judge_temperature must be in [0, 2]")


@dataclass(frozen=True, slots=True)
class WorkspaceSettings:
    profile: str = "default"
    claw_solver_timeout: int = SOLVER_TIMEOUTS["clawgym"]
    long_episode_timeout: int = 3000
    claw_max_turns: int | None = None
    claw_judge_timeout: int = 180
    harvey_max_turns: int | None = 200
    harvey_solver_timeout: int = SOLVER_TIMEOUTS["harvey"]
    claw_checker_timeout: int = 120
    extractor_timeout: int = 120
    judge_timeout: int = 720
    judge_phase_timeout: int = 3600
    cpus: int = 2
    memory: str = "4g"
    pids: int = 256
    solver_pids: int = 192
    shm_size: str = "256m"
    train_judge_mode: str = "batched"
    val_judge_mode: str = "per_criterion"
    judge_mode: str | None = None
    batch_size: int = 24
    judge_max_tokens: int = 16384
    judge_concurrency: int = 8
    judge_model: str = "deepseek-flash"
    judge_base_url: str | None = None
    judge_api_key_env: str = "DEEPSEEK_API_KEY"
    judge_thinking: Mapping[str, str] | None = None
    judge_reasoning_effort: str | None = None
    judge_temperature: float | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WorkspaceSettings:
        return cls(**{field.name: config[field.name] for field in fields(cls)
                      if field.name in config})

    def __post_init__(self) -> None:
        if self.profile not in ("default", "long"):
            raise ValueError("benchmark.profile must be default or long")
        for name in _POSITIVE_INTEGER_SETTINGS:
            _validate_positive_integer(name, getattr(self, name))
        _validate_positive_integer("claw_max_turns", self.claw_max_turns, nullable=True)
        _validate_positive_integer("harvey_max_turns", self.harvey_max_turns, nullable=True)
        if self.judge_concurrency > 8:
            raise ValueError("benchmark.judge_concurrency must be <= 8")
        _validate_docker_size("memory", self.memory)
        _validate_docker_size("shm_size", self.shm_size)
        if (
            self.train_judge_mode not in _JUDGE_MODES
            or self.val_judge_mode not in _JUDGE_MODES
            or self.judge_mode not in (*_JUDGE_MODES, None)
        ):
            raise ValueError("Harvey judge mode must be per_criterion or batched")
        _validate_judge_provider(self)

    @property
    def claw_timeout(self) -> int:
        return self.long_episode_timeout if self.profile == "long" else self.claw_solver_timeout

    def mode_for(self, split: str) -> str:
        if split not in {"harvey_train", "harvey_val"}:
            raise PermissionError("Harvey grading requires an active manifest split")
        return self.judge_mode or (
            self.train_judge_mode if split in TRAIN else self.val_judge_mode
        )

    def resources(self, *, solver: bool = False) -> dict[str, int | str]:
        result = {key: getattr(self, key) for key in ("cpus", "memory", "pids", "shm_size")}
        if solver:
            result["pids"] = self.solver_pids
        return result


def harvey_counts(event: Mapping[str, Any]) -> tuple[int, int]:
    passed = event.get("passed_criteria")
    total = event.get("total_criteria")
    if not _is_integer(passed) or not _is_integer(total) or not 0 <= passed <= total:
        raise InfraError("Harvey verifier criterion counts are missing or invalid")
    return passed, total


def summarize_harvey(records: Iterable[Mapping[str, Any]]) -> dict[str, int | float]:
    """Pool counts across tasks and repeats. Empty rubrics contribute zero task score."""
    counts = [harvey_counts(record) for record in records]
    if not counts:
        raise ValueError("Harvey aggregation requires at least one result")
    passed = sum(p for p, _ in counts)
    total = sum(t for _, t in counts)
    return {
        "passed_criteria": passed,
        "total_criteria": total,
        "micro_avg": passed / total if total else 0.0,
        "macro_avg": mean(p / t if t else 0.0 for p, t in counts),
        "all_pass": mean(float(t > 0 and p == t) for p, t in counts),
    }


def _grade_harvey(event: Mapping[str, Any], score: float) -> tuple[dict[str, float | int], dict[str, int]]:
    passed, total = harvey_counts(event)
    fraction = passed / total if total else 0.0
    all_pass = float(total > 0 and passed == total)
    if score != fraction or event.get("macro_score") != fraction or event.get("all_pass") != all_pass:
        raise InfraError("Harvey verifier scores disagree with criterion counts")
    accounting = event.get("judge_usage")
    if (
        not isinstance(accounting, Mapping)
        or any(not _is_integer(accounting.get(key)) or accounting[key] < 0
               for key in JUDGE_USAGE_KEYS)
        or event.get("judge_mode") not in _JUDGE_MODES
    ):
        raise InfraError("Harvey verifier accounting is invalid")
    return {
        "passed_criteria": passed,
        "total_criteria": total,
        "micro_avg": fraction,
        "macro_avg": fraction,
        "macro_score": fraction,
        "all_pass": all_pass,
        "harvey_rubric_score": fraction,
    }, dict(accounting)


class WorkspaceBenchmark(Benchmark):
    description = "Manifest-backed workspace tasks with host-side validation"

    def __init__(self, name: str, *, allow_sealed_test: bool = False, root: Path | None = None) -> None:
        super().__init__()
        if name not in TRAIN | DEVELOPMENT | SEALED:
            raise ValueError("unknown workspace split")
        if name in SEALED or allow_sealed_test is not False:
            raise PermissionError("sealed Workspace execution is disabled")
        self.name = name
        self.root = root if root is not None else resolve_root({}, _TRACK_DIRECTORY)
        self.primary_metric = "clawgym_hard_score" if name.startswith("clawgym") else "micro_avg"
        self._records: dict[str, dict[str, Any]] = {}
        self.metadata = {"split": name, "sealed": False}

    def load(self) -> Mapping[str, tuple[Task, ...]]:
        if self._splits is None:
            manifest_dir = self.root / "manifests"
            summary = read_json(manifest_dir / "manifest_summary.json")
            item = summary["splits"][self.name]
            path = manifest_dir / f"{self.name}.locked.json"
            if sha256(path) != item["manifest_sha256"]:
                raise InfraError("locked manifest digest mismatch")
            data = read_json(path)
            if (
                data["sealed"] != (self.name in SEALED)
                or data["count"] != item["count"]
                or [row["task_id"] for row in data["tasks"]] != item["ids"]
            ):
                raise InfraError("split manifest no longer matches summary")
            self._records = {row["task_id"]: row for row in data["tasks"]}
            tasks = tuple(
                Task(task_id=row["task_id"], question=row["prompt"], env=_TRACK_DIRECTORY,
                     metadata={"split": self.name})
                for row in data["tasks"]
            )
            self._tasks = {task.task_id: task for task in tasks}
            role = "train" if self.name in TRAIN else "dev"
            self._splits = {key: tasks if key == role else () for key in ("train", "dev", "test")}
        return self._splits

    def lookup(self, task: Task) -> tuple[dict[str, Any], Path]:
        self.load()
        if self._tasks.get(task.task_id) != task:
            raise ValueError("task rejected by split lock")
        row = self._records[task.task_id]
        path = confined(self.root, row["path"])
        locked_files = row["locked_files"]
        for relative, spec in locked_files.items():
            file = confined(path, relative)
            if not file.is_file() or file.stat().st_size != spec["bytes"] or sha256(file) != spec["sha256"]:
                raise InfraError(f"locked workspace file mismatch: {relative}")
        for scope in ("input", "grader"):
            actual = {
                file.relative_to(path).as_posix() for file in (path / scope).rglob("*")
                if file.is_file() or file.is_symlink()
            }
            if actual - locked_files.keys():
                raise InfraError("unlocked task files")
        return row, path

    def grade(self, trajs: tuple[Trajectory, ...]) -> tuple[Trajectory, ...]:
        graded = []
        self.load()
        for trajectory in trajs:
            task = trajectory.task
            if task is None or self._tasks.get(task.task_id) != task:
                raise ValueError("trajectory is outside the loaded workspace manifest")
            event = trajectory.messages[-1] if trajectory.messages else {}
            if (
                event.get("type") != "SuiteVerifier"
                or event.get("task_id") != task.task_id
                or event.get("split") != self.name
            ):
                raise InfraError("host verifier event is missing")
            score = event.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise InfraError("invalid verifier score")
            metrics = {**trajectory.metrics, self.primary_metric: float(score)}
            usage = trajectory.usage
            if self.name.startswith("harvey"):
                harvey_metrics, accounting = _grade_harvey(event, score)
                metrics.update(harvey_metrics)
                usage = {**usage, "judge_usage": accounting}
            graded.append(replace(trajectory, metrics=metrics, usage=usage))
        return tuple(graded)


class WorkspaceSuite(SuiteBenchmark):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config, _TRACK_DIRECTORY)
        self.settings = WorkspaceSettings.from_config(config)
        if self.name == "harvey_workspace":
            self.primary_metric = "micro_avg"
        self.metadata["agent_shells"] = DEFAULT_AGENT_SHELLS
        self.metadata["clawgym"] = {
            "solver_timeout": self.settings.claw_timeout,
            "long_episode_timeout": self.settings.long_episode_timeout,
            "checker_timeout": self.settings.claw_checker_timeout,
            "judge_timeout": self.settings.claw_judge_timeout,
            "max_turns": self.settings.claw_max_turns,
        }
        self.metadata["harvey"] = {
            "solver_timeout": self.settings.harvey_solver_timeout,
            "max_turns": self.settings.harvey_max_turns,
            "train_judge_mode": self.settings.mode_for("harvey_train"),
            "val_judge_mode": self.settings.mode_for("harvey_val"),
            "primary_metric": "micro_avg",
        }
        val_judge = config.get("val_judge", {})
        if not isinstance(val_judge, Mapping) or set(val_judge) - JUDGE_PROVIDER_FIELDS:
            raise ValueError("benchmark.val_judge only accepts judge provider settings")
        self.val_settings = replace(self.settings, **val_judge) if val_judge else self.settings
        self.backends: dict[str, WorkspaceBenchmark] = {}
        self._routes: dict[str, str] = {}

    def load(self) -> Mapping[str, tuple[Task, ...]]:
        if self._splits is not None:
            return self._splits
        tracks = {
            "clawgym": ("clawgym",),
            "harvey_workspace": ("harvey",),
            "workspace_dualgate": ("clawgym", "harvey"),
        }[self.name]
        groups: dict[str, list[Task]] = {"train": [], "dev": []}
        counts = {"train": 0, "dev": 0}
        backends = {}
        routes = {}
        for track in tracks:
            for role, suffix in (("train", "train"), ("dev", "val")):
                split = f"{track}_{suffix}"
                backend = WorkspaceBenchmark(split, root=self.root)
                backend.load()
                tasks = getattr(backend, role)
                counts[role] += len(tasks)
                selected = set(self.selected(role, [task.task_id for task in tasks]))
                for task in tasks:
                    if task.task_id in selected:
                        groups[role].append(task)
                        routes[task.task_id] = split
                backends[split] = backend
        splits = self.set_tasks(groups["train"], groups["dev"])
        self.backends, self._routes = backends, routes
        self.metadata["manifest_counts"] = counts
        return splits

    def backend(self, task: Task) -> WorkspaceBenchmark:
        self.check_task(task)
        return self.backends[self._routes[task.task_id]]

    def grade(self, trajectories: tuple[Trajectory, ...]) -> tuple[Trajectory, ...]:
        graded = []
        for trajectory in trajectories:
            backend = self.backend(trajectory.task)
            item = backend.grade((trajectory,))[0]
            graded.append(replace(item, metrics={
                **item.metrics, "reward": item.metrics[backend.primary_metric]
            }))
        return tuple(graded)
