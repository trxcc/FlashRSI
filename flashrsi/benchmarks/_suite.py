"""Shared contracts for benchmark suites with reviewed local assets.

Benchmark metadata and split guards are available on the host without importing
the individual benchmark integrations. Asset roots can be configured through
FLASHRSI_SUITE_ENVS_ROOT or a per-track FLASHRSI_SUITE_<TRACK>_ROOT.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flashrsi.core import Benchmark, InfraError, Task, Trajectory

ACTIVE = frozenset(
    {
        "tb21_clean",
        "swegym_lite",
        "clawgym",
        "harvey_workspace",
        "workspace_dualgate",
        "engdesign_open",
        "frontier_eng",
        "tdc_admet",
        "kernelbench",
    }
)
SEALED = frozenset(
    {
        "tb41_sealed",
        "tbs01_sealed",
        "swebench_verified_sealed",
        "clawgym_bench_sealed",
        "harvey_lab_sealed",
        "jobbench_sealed",
        "gdpval_sealed",
        "frontier_eng_sealed",
        "tdc_toxicity_sealed",
        "kernelbench_sealed",
    }
)
NAMES = ACTIVE | SEALED
DEFAULT_AGENT_SHELLS = ("claude_code", "codex")
OPTIONAL_AGENT_SHELLS = ("pi", "pi_agent", "miniagent", "openclaw")


def default_envs_root() -> Path:
    """Locate prepared assets for an installed suite or a checkout."""
    if configured := os.environ.get("FLASHRSI_SUITE_ENVS_ROOT"):
        return Path(configured).expanduser().resolve()
    module = Path(__file__).resolve()
    suite_envs = module.parents[3] / "envs"
    if suite_envs.is_dir():
        return suite_envs
    return module.parents[2] / "artifacts" / "envs"


def resolve_root(config: Mapping[str, Any], directory: str) -> Path:
    """Honor the explicit track root before suite-wide defaults."""
    override = config.get("root") or os.getenv(
        f"FLASHRSI_SUITE_{directory.upper()}_ROOT"
    )
    root = Path(override) if override else default_envs_root() / directory
    return root.expanduser().resolve()


def guard(config: Mapping[str, Any]) -> None:
    name = config["name"]
    if name not in NAMES:
        return
    split = config.get("split", "auto")
    if config.get("allow_sealed_test", False) is not False:
        raise PermissionError("suite sealed execution is disabled")
    if name in SEALED or split in ("test", "sealed"):
        raise PermissionError(f"Sealed suite split is locked: {name}")
    if split not in ("auto", "train", "dev", "val"):
        raise ValueError("suite benchmark.split must be auto, train, dev or val")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def confined(root: Path, path: Path | str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("unconfined suite asset path")
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("unconfined suite asset path")
    if any(
        part.is_symlink()
        for part in (candidate, *candidate.parents)
        if root in part.parents
    ):
        raise ValueError("unconfined suite asset path")
    return candidate


class SuiteBenchmark(Benchmark):
    """Load approved split tasks and grade their trusted verifier receipts."""

    primary_metric = "reward"

    def __init__(self, config: Mapping[str, Any], directory: str) -> None:
        guard(config)
        super().__init__()
        self.config = config
        self.name = config["name"]
        self.root = resolve_root(config, directory)
        self.metadata: dict[str, Any] = {"name": self.name, "sealed_locked": True}
        self._tasks: dict[str, Task] = {}

    def selected(self, role: str, identifiers: Iterable[str]) -> list[str]:
        configured = self.config.get("split", "auto")
        if configured not in ("auto", role) and not (
            configured == "val" and role == "dev"
        ):
            return []
        wanted = self.config.get("task_ids")
        if wanted is None:
            return list(identifiers)
        if (
            not isinstance(wanted, (list, tuple))
            or not wanted
            or any(not isinstance(item, str) for item in wanted)
        ):
            raise ValueError("benchmark.task_ids must be a nonempty list of IDs")
        return [identifier for identifier in identifiers if identifier in wanted]

    def set_tasks(
        self, train: Sequence[Task], dev: Sequence[Task]
    ) -> Mapping[str, tuple[Task, ...]]:
        all_tasks = (*train, *dev)
        if any(task.env is None for task in all_tasks):
            raise ValueError("native suite tasks require a Docker environment")
        tasks = {task.task_id: task for task in all_tasks}
        if len(tasks) != len(all_tasks):
            raise ValueError("duplicate suite task ID across splits")
        wanted = self.config.get("task_ids")
        if wanted is not None and set(wanted) != tasks.keys():
            raise PermissionError(
                "requested task lacks an active, reviewed execution payload"
            )
        self._tasks = tasks
        self._splits = MappingProxyType(
            {"train": tuple(train), "dev": tuple(dev), "test": ()}
        )
        self.metadata["executable_counts"] = {"train": len(train), "dev": len(dev)}
        return self._splits

    def check_task(self, task: Task | None) -> None:
        if task is None or task.env is None or self._tasks.get(task.task_id) != task:
            raise PermissionError("task is outside the active suite manifest")

    def grade(self, trajectories: tuple[Trajectory, ...]) -> tuple[Trajectory, ...]:
        graded = []
        for trajectory in trajectories:
            task = trajectory.task
            self.check_task(task)
            assert task is not None
            event = trajectory.messages[-1] if trajectory.messages else {}
            if (
                event.get("type") != "SuiteVerifier"
                or event.get("task_id") != task.task_id
            ):
                raise InfraError("trusted suite verifier result missing")
            score = event.get("score")
            if type(score) not in (int, float) or not 0 <= score <= 1:
                raise InfraError("invalid suite verifier score")
            graded.append(
                replace(
                    trajectory, metrics={**trajectory.metrics, "reward": float(score)}
                )
            )
        return tuple(graded)
