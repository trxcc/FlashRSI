"""Load reviewed Harbor and SWE splits for the benchmark suite.

Checksums tie the manifest and each executable payload to the review index.
Harbor is loaded only when building native Harbor tasks. Asset roots follow
the suite configuration and FLASHRSI_SUITE_HARBOR_TRACK_ROOT.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flashrsi.benchmarks._harbor import _grade, _task_from_directory
from flashrsi.benchmarks._suite import (
    DEFAULT_AGENT_SHELLS,
    SuiteBenchmark,
    read_json,
    resolve_root,
    sha256,
)
from flashrsi.benchmarks.swe_bench import SWEBench, _task_from_row
from flashrsi.core import Agent, Benchmark, InfraError, Task, Trajectory

_TRACK_DIRECTORY = "harbor_track"
_ROLES = {"train": "train", "val": "dev", "sealed": "test"}
_SWE_OPTIONS = ("setup_timeout", "evaluation_timeout", "summary_timeout", "cleanup_grace", "verifier_resources")

HARBOR_ATTEMPTS = MappingProxyType({
    "tb21_train": 1, "tb21_val": 5, "tb41_sealed": 5, "tbs01_sealed": 3,
})


def _attempts(split: str, options: Mapping[str, Any]) -> int:
    default = 1 if options.get("smoke") or options.get("smoke_whitelist_only") else HARBOR_ATTEMPTS[split]
    attempts = options.get("n_attempts", default)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("benchmark.n_attempts must be a positive integer")
    return attempts


@dataclass(frozen=True, slots=True)
class Split:
    filename: str
    count: int
    role: str
    family: str
    payload: str


SPLITS = MappingProxyType({
    "tb21_train": Split("tb21_train_56.json", 56, "train", "harbor", "tb21"),
    "tb21_val": Split("tb21_val_32.json", 32, "val", "harbor", "tb21"),
    "tb41_sealed": Split("tb41_sealed_66.json", 66, "sealed", "harbor", "tb41"),
    "tbs01_sealed": Split("tbs01_sealed_70.json", 70, "sealed", "harbor", "tbs01"),
    "swebench_verified_sealed": Split("swebench_verified_sealed_500.json", 500, "sealed", "swe", "swebench_verified_sealed"),
    "swegym_lite_train": Split("swegym_lite_train_230.json", 230, "train", "swe", "swegym_lite_train"),
})



_TRUSTED_NAMES = {
    ".git", ".gitconfig", ".gitmodules", ".gitattributes", ".githooks",
    "conftest.py", "eval.sh", "run_tests.sh", "setup_env.sh", "setup_repo.sh",
    "run-tests.sh", "run_test.sh", "test.patch", "patch.diff",
}
_ROOT_CONFIG = {
    ".condarc", ".coveragerc", ".env", ".envrc", ".pytest.ini", "environment.yml",
    "environment.yaml", "pip.conf", "pyproject.toml", "pytest.ini", "setup.cfg",
    "setup.py", "sitecustomize.py", "tox.ini", "usercustomize.py", "harness", "hooks",
}


def _protected_path(path: tuple[str, ...], trusted: set[tuple[str, ...]]) -> bool:
    return (
        not path
        or path[0] == "/"
        or ".." in path
        or bool(_TRUSTED_NAMES.intersection(path))
        or path[0] in _ROOT_CONFIG
        or path[:2] == (".github", "workflows")
        or any(
            a == "swebench" and b in {"harness", "run_evaluation.py", "grading.py"}
            for a, b in zip(path, path[1:])
        )
        or any(path[:len(test)] == test or test[:len(path)] == path for test in trusted)
    )


def _split(name: str, *, allow_sealed_test: bool = False) -> Split:
    split = SPLITS[name]
    if split.role == "sealed" or allow_sealed_test is not False:
        raise PermissionError(f"Sealed suite split is locked: {name}")
    return split


def _read_manifest(spec: Split, root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    values = (root / spec.filename).read_bytes()
    if hashlib.sha256(values).hexdigest() != lock["files"][spec.filename]["sha256"]:
        raise ValueError(f"Manifest SHA-256 mismatch: {spec.filename}")
    data = json.loads(values)
    identity = "task_id" if spec.family == "harbor" else "instance_id"
    if data["schema_version"] != 1 or data["id_field"] != identity or data["split"] != spec.role:
        raise ValueError(f"Invalid manifest schema or role: {spec.filename}")
    ids = [row[identity] for row in data["tasks"]]
    if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value)
               and value not in {".", ".."} for value in ids):
        raise ValueError(f"Invalid identifier: {spec.filename}")
    if data["count"] != len(ids) or len(ids) != spec.count or ids != sorted(set(ids)):
        raise ValueError(f"Wrong count, order or duplicate ID: {spec.filename}")
    return data


def load_manifest(name: str, *, allow_sealed_test: bool = False, root: Path | None = None) -> dict[str, Any]:
    spec = _split(name, allow_sealed_test=allow_sealed_test)
    root = root if root is not None else resolve_root({}, _TRACK_DIRECTORY)
    return _read_manifest(spec, root, read_json(root / "manifest_checksums.json"))


def _review_index(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    values = (root / "reviewed_payloads.json").read_bytes()
    if hashlib.sha256(values).hexdigest() != lock["reviewed_payloads_sha256"]:
        raise ValueError("Review index SHA-256 mismatch")
    return json.loads(values)


def _picked(manifest: Mapping[str, Any], ids: Iterable[str]) -> list[dict[str, Any]]:
    selected = tuple(ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Use one or more distinct task IDs")
    known = {row[manifest["id_field"]]: row for row in manifest["tasks"]}
    return [known[task_id] for task_id in selected]


def _require_payload(path: Path, spec: Split, task_id: str, review: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise InfraError(
            f"{task_id}: missing {path}. Stage the pinned {spec.payload} execution "
            "payload at that path and update the checksum-locked review index."
        )
    if task_id not in review:
        raise PermissionError(f"{task_id}: {path} lacks a checksum-reviewed execution payload")



def _harbor_task(
    root: Path, spec: Split, row: Mapping[str, Any], review: Mapping[str, Any],
) -> Task:
    task_id = row["task_id"]
    directory = root / "tasks" / spec.payload / task_id
    _require_payload(directory / "task.toml", spec, task_id, review)
    hashes = review[task_id]["sha256"]
    expected = {"instruction.md": row["instruction_sha256"], "tests/test.sh": row["verifier_sha256"]}
    if any(expected[file] != hashes[file] for file in expected):
        raise ValueError(f"Staging provenance mismatch: {task_id}")
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Task outside assigned root: {task_id}")
    actual = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in task payload: {task_id}")
        if path.is_file():
            actual[path.relative_to(directory).as_posix()] = sha256(path)
    missing = hashes.keys() - actual.keys()
    if missing:
        raise InfraError(
            f"{task_id}: missing Harbor asset {directory / min(missing)}. "
            f"Stage the checksum-reviewed payload under {directory}."
        )
    if hashes != actual:
        raise ValueError(f"Task payload integrity mismatch: {task_id}")
    try:
        task = _task_from_directory(directory, row["harbor_ref"], label=_TRACK_DIRECTORY)
    except FileNotFoundError as exc:
        raise InfraError(
            f"{task_id}: {exc} Stage the reviewed Harbor assets under {directory}."
        ) from None
    if task.task_id.rsplit("/", 1)[-1] != task_id:
        raise ValueError(f"Native Harbor task ID mismatch: {task_id}")
    return replace(task, task_id=task_id)


class _HarborSplitTrack(Benchmark):
    primary_metric = "reward"
    name = _TRACK_DIRECTORY
    description = "Pinned Harbor tasks with trusted native verifier rewards."

    def __init__(
        self, spec: Split, root: Path, rows: Iterable[Mapping[str, Any]],
        review: Mapping[str, Any], *, preflight_images: bool = False, n_attempts: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.root = root
        self._rows = tuple(rows)
        self._review = review
        self._preflight_images = preflight_images
        self.metadata = {"n_attempts": n_attempts, "agent_shells": DEFAULT_AGENT_SHELLS}
        self._tasks_by_id: dict[str, Task] = {}

    def load(self) -> Mapping[str, tuple[Task, ...]]:
        if self._splits is None:
            tasks = []
            for row in self._rows:
                task = _harbor_task(self.root, self.spec, row, self._review)
                metadata = {
                    **task.metadata,
                    "suite_require_local_image": self._preflight_images,
                    "n_attempts": self.metadata["n_attempts"],
                }
                tasks.append(replace(task, metadata=metadata))
            role = _ROLES[self.spec.role]
            self._splits = MappingProxyType({
                key: tuple(tasks) if key == role else () for key in ("train", "dev", "test")
            })
            self._tasks_by_id = {task.task_id: task for task in tasks}
        return self._splits

    def grade(self, trajectories: tuple[Trajectory, ...]) -> tuple[Trajectory, ...]:
        return _grade(trajectories, self._tasks_by_id, self.name)


class _SWESplitTrack(SWEBench):
    def __init__(
        self, spec: Split, root: Path, rows: Iterable[Mapping[str, Any]],
        review: Mapping[str, Any], options: Mapping[str, Any],
    ) -> None:
        role = _ROLES[spec.role]
        super().__init__({
            **options,
            "split": {key: int(key == role) for key in ("train", "dev", "test")},
        })
        self.spec = spec
        self.root = root
        self._rows = tuple(rows)
        self._review = review

    def _iter_tasks(self) -> Iterable[Task]:
        for identity in self._rows:
            task_id = identity["instance_id"]
            path = self.root / "prepared" / self.spec.payload / f"{task_id}.json"
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError(f"SWE payload outside assigned root: {task_id}")
            _require_payload(path, self.spec, task_id, self._review)
            values = path.read_bytes()
            if hashlib.sha256(values).hexdigest() != self._review[task_id]["sha256"]:
                raise ValueError(f"SWE payload integrity mismatch: {task_id}")
            row = json.loads(values)
            for key in ("instance_id", "repo", "base_commit", "image"):
                if key in identity and row[key] != identity[key]:
                    raise ValueError(f"SWE {key} mismatch: {task_id}")
            task = _task_from_row(row, self.spec.role)
            self._rows_by_id[task_id] = row
            yield task


def open_split(
    name: str, ids: Iterable[str], *, allow_sealed_test: bool = False, root: Path | None = None,
) -> Benchmark:
    """Build a native benchmark from reviewed, checksum-verified payloads."""
    spec = _split(name, allow_sealed_test=allow_sealed_test)
    root = root if root is not None else resolve_root({}, _TRACK_DIRECTORY)
    lock = read_json(root / "manifest_checksums.json")
    rows = _picked(_read_manifest(spec, root, lock), ids)
    review = _review_index(root, lock).get(name, {})
    return _open_selection(name, spec, root, rows, review)


def _open_selection(
    name: str, spec: Split, root: Path, rows: Iterable[Mapping[str, Any]],
    review: Mapping[str, Any], *, preflight_images: bool = False,
    options: Mapping[str, Any] | None = None,
) -> _HarborSplitTrack | _SWESplitTrack:
    options = options or {}
    benchmark: _HarborSplitTrack | _SWESplitTrack
    if spec.family == "harbor":
        benchmark = _HarborSplitTrack(
            spec, root, rows, review, preflight_images=preflight_images,
            n_attempts=_attempts(name, options),
        )
    else:
        benchmark = _SWESplitTrack(spec, root, rows, review, options)
    benchmark.load()
    benchmark._track_split = name
    return benchmark


def _smoke_whitelist(config: Mapping[str, Any]) -> set[str]:
    ids = config.get("smoke_task_ids")
    if (
        not isinstance(ids, (list, tuple)) or not ids
        or any(not isinstance(task_id, str) for task_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("benchmark.smoke_task_ids must be a nonempty list of distinct IDs")
    return set(ids)


def bind_track(
    agent: Agent, benchmark: Benchmark, name: str, *, allow_sealed_test: bool = False,
) -> None:
    spec = _split(name, allow_sealed_test=allow_sealed_test)
    if getattr(benchmark, "_track_split", None) != name:
        raise PermissionError("Benchmark was not constructed for the requested split")
    if spec.family == "harbor":
        from flashrsi.integrations.harbor import bind
        bind(agent)
    else:
        from flashrsi.integrations.swe_bench import bind
        bind(agent, benchmark)
    dispatch = agent._env_runner
    tasks = {task.task_id: task for group in benchmark.load().values() for task in group}

    async def run(target: Agent, task: Task) -> Trajectory:
        if tasks.get(task.task_id) != task:
            raise PermissionError("Task is outside the manifest-verified execution selection")
        return await dispatch(target, task)

    agent._env_runner = run


class HarborSuite(SuiteBenchmark):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config, _TRACK_DIRECTORY)
        self.primary_metric = "resolved" if self.name == "swegym_lite" else "reward"
        self._backends: dict[str, _HarborSplitTrack | _SWESplitTrack] = {}
        self._routes: dict[str, str] = {}
        self.metadata["agent_shells"] = DEFAULT_AGENT_SHELLS
        if self.name == "tb21_clean":
            self.metadata["n_attempts"] = _attempts("tb21_val", config)
            self.metadata["train_n_attempts"] = _attempts("tb21_train", config)
        self.smoke_whitelist_only = config.get("smoke_whitelist_only", False)
        if type(self.smoke_whitelist_only) is not bool:
            raise ValueError("benchmark.smoke_whitelist_only must be a boolean")


    def load(self) -> Mapping[str, tuple[Task, ...]]:
        if self._splits is not None:
            return self._splits
        roles = (
            {"train": "tb21_train", "dev": "tb21_val"}
            if self.name == "tb21_clean"
            else {"train": "swegym_lite_train"}
        )
        lock = read_json(self.root / "manifest_checksums.json")
        review = _review_index(self.root, lock)
        groups: dict[str, list[Task]] = {"train": [], "dev": []}
        backends = {}
        routes = {}
        counts = {}
        whitelist = _smoke_whitelist(self.config) if self.smoke_whitelist_only else None
        swe_options = {key: self.config[key] for key in _SWE_OPTIONS if key in self.config}
        for role, name in roles.items():
            spec = _split(name)
            manifest = _read_manifest(spec, self.root, lock)
            counts[role] = manifest["count"]
            ids = [row[manifest["id_field"]] for row in manifest["tasks"]]
            selected = self.selected(role, ids)
            reviewed = review.get(name, {})
            if whitelist is not None:
                selected = [task_id for task_id in selected if task_id in whitelist]
                for task_id in selected:
                    if task_id not in reviewed:
                        raise PermissionError(f"{name}/{task_id} lacks a reviewed smoke payload")
            if not selected:
                continue
            backend = _open_selection(
                name, spec, self.root, _picked(manifest, selected), reviewed,
                preflight_images=not self.smoke_whitelist_only,
                options=self.config if spec.family == "harbor" else swe_options,
            )
            backends[name] = backend
            for task in getattr(backend, role):
                groups[role].append(task)
                routes[task.task_id] = name
        if whitelist is not None and whitelist != routes.keys():
            missing = ", ".join(sorted(whitelist - routes.keys()))
            raise PermissionError(f"Smoke whitelist IDs outside the active selection: {missing}")
        splits = self.set_tasks(groups["train"], groups["dev"])
        self._backends, self._routes = backends, routes
        self.metadata["manifest_counts"] = counts
        return splits

    def grade(self, trajectories: tuple[Trajectory, ...]) -> tuple[Trajectory, ...]:
        graded = []
        for trajectory in trajectories:
            self.check_task(trajectory.task)
            backend = self._backends[self._routes[trajectory.task.task_id]]
            graded.extend(backend.grade((trajectory,)))
        return tuple(graded)
