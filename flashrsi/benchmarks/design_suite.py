"""Load open engineering, ADMET, and accelerator benchmark tasks.

This module verifies pinned assets and selects splits on the host. It does
not start a solver or verifier. FLASHRSI_SUITE_ENVS_ROOT and
FLASHRSI_SUITE_ENG_AND_MOL_ROOT select the staged assets via the suite base.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from flashrsi.benchmarks._suite import (
    DEFAULT_AGENT_SHELLS,
    SuiteBenchmark,
    confined,
    read_json,
    sha256,
)
from flashrsi.core import InfraError, Task

FRONTIER_DEV = "dev_water_allocate"
FRONTIER_PROMPT = (
    "Synthetic engineering task: choose four nonnegative allocation fractions that sum to "
    "one and track nominal demands [0.42, 0.28, 0.18, 0.12]. "
    "Return a Python function def solve(): return (four numeric literals). "
    "No filesystem, imports or expressions."
)
RECIPE_PROMPT = (
    "Fit an existing, public ADMET dataset using a fixed Morgan fingerprint and four descriptors. "
    "Choose a JSON recipe object with exactly model, radius, trees, leaf. "
    "model: ridge, extra_trees, random_forest, lightgbm or xgboost. "
    "radius: 2 or 3; trees: integer 16 to 128; leaf: integer 1 to 12."
)


TDC_METRICS = MappingProxyType(
    {
        "caco2_wang": ("MAE", "minimize"),
        "hia_hou": ("AUROC", "maximize"),
        "pgp_broccatelli": ("AUROC", "maximize"),
        "bioavailability_ma": ("AUROC", "maximize"),
        "lipophilicity_astrazeneca": ("MAE", "minimize"),
        "solubility_aqsoldb": ("MAE", "minimize"),
        "bbb_martins": ("AUROC", "maximize"),
        "ppbr_az": ("MAE", "minimize"),
        "vdss_lombardo": ("Spearman", "maximize"),
        "cyp2c9_substrate_carbonmangels": ("AUPRC", "maximize"),
        "cyp2d6_substrate_carbonmangels": ("AUPRC", "maximize"),
        "cyp3a4_substrate_carbonmangels": ("AUROC", "maximize"),
        "cyp2c9_veith": ("AUPRC", "maximize"),
        "cyp2d6_veith": ("AUPRC", "maximize"),
        "cyp3a4_veith": ("AUPRC", "maximize"),
        "half_life_obach": ("Spearman", "maximize"),
        "clearance_hepatocyte_az": ("Spearman", "maximize"),
        "clearance_microsome_az": ("Spearman", "maximize"),
        "herg": ("AUROC", "maximize"),
        "ames": ("AUROC", "maximize"),
        "dili": ("AUROC", "maximize"),
        "ld50_zhu": ("MAE", "minimize"),
    }
)
TDC_SEALED = frozenset({"herg", "ames", "dili", "ld50_zhu", "ld50"})
FRONTIER_TIMEOUTS = frozenset({300, 600, 1200})
FRONTIER_ORACLE_BUDGET = 100
FRONTIER_MAX_TOKENS = 32768
DEEPSEEK_HIGH_MAX_TOKENS = 65536
TDC_SEEDS = (1, 2, 3, 4, 5)
TDC_ORACLE_BUDGET = 20
KERNEL_ORACLE_BUDGET = 10
FRONTIER_ENVS = {
    "driver": "frontier-eval-driver",
    "main": "frontier-v1-main",
    "summit": "frontier-v1-summit",
    "sustaindc": "frontier-v1-sustaindc",
    "kernel": "frontier-v1-kernel",
}


class _DesignBenchmark(SuiteBenchmark):
    def __init__(self, config: Mapping[str, Any]) -> None:
        if config["name"].endswith("_sealed"):
            raise PermissionError(f"Sealed suite split is locked: {config['name']}")
        super().__init__(config, "eng_and_mol")

    def load(self) -> Mapping[str, tuple[Task, ...]]:
        if self._splits is not None:
            return self._splits
        ids = self.manifest()
        self.metadata["manifest_counts"] = {key: len(value) for key, value in ids.items()}
        selected = {}
        for role, names in ids.items():
            selected[role] = [self.task(name, role) for name in self.selected(role, names)]
        return self.set_tasks(selected["train"], selected["dev"])


@dataclass(frozen=True, slots=True)
class EngDesignSettings:
    solver_timeout: int = 5400
    grader_timeout: int = 420
    max_turns: int | None = 250
    val_repeats: int = 4
    min_val_gain: float = 0.020

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> EngDesignSettings:
        settings = cls(**{field.name: config[field.name] for field in fields(cls) if field.name in config})
        for name in ("solver_timeout", "grader_timeout", "val_repeats"):
            positive_integer(f"EngDesign {name}", getattr(settings, name))
        if settings.max_turns is not None:
            positive_integer("EngDesign max_turns", settings.max_turns)
        gain = settings.min_val_gain
        if isinstance(gain, bool) or not isinstance(gain, (float, int)) or not 0 <= gain <= 1:
            raise ValueError("EngDesign min_val_gain must be in [0, 1]")
        return settings


def positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class EngDesignOpen(_DesignBenchmark):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.settings = EngDesignSettings.from_config(config)
        self.metadata.update(asdict(self.settings), agent_shells=DEFAULT_AGENT_SHELLS)

    def manifest(self) -> dict[str, list[str]]:
        self.manifest_data = read_json(self.root / "engdesign_manifest.json")
        return {"train": self.manifest_data["train"], "dev": self.manifest_data["val"]}

    def directory(self, name: str) -> Path:
        if name not in self.manifest_data["files"]:
            raise PermissionError("EngDesign task outside prepared open manifest")
        base = confined(self.root / "EngDesign/EngDesign-Open", name)
        for filename, expected in self.manifest_data["files"][name].items():
            if sha256(confined(base, filename)) != expected:
                raise InfraError("EngDesign pinned asset digest mismatch")
        return base

    def task(self, name: str, role: str) -> Task:
        if role not in ("train", "dev", "val"):
            raise PermissionError("sealed EngDesign split")
        base = self.directory(name)
        native = (base / "LLM_prompt.txt").read_text()
        if name == "XG_13":
            output = "Return JSON with exactly three numeric keys: Kp, Ki, Kd."
        else:
            schema = _output_schema(base / "output_structure.py")
            output = "Return the outer response as a JSON object. Schema: " + schema
        return Task(name, native + "\n\n" + output, env="engdesign_open", metadata={"split": role})


def _output_schema(path: Path) -> str:
    if not path.is_file():
        return ""
    classes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef):
            names = [
                f"{field.target.id}: {ast.unparse(field.annotation)}"
                for field in node.body
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
            ]
            if names:
                classes.append(f"{node.name}: {', '.join(names)}")
    return "; ".join(classes)


class FrontierEng(_DesignBenchmark):
    proposal_max_tokens = FRONTIER_MAX_TOKENS

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.metadata.update(
            oracle_budget=config.get("oracle_budget", FRONTIER_ORACLE_BUDGET),
            max_tokens=self.proposal_max_tokens,
            evaluator_timeouts=tuple(sorted(FRONTIER_TIMEOUTS)),
        )

    @property
    def mode(self) -> str:
        return self.config.get("mode", "synthetic_dev")

    @property
    def repository(self) -> Path:
        return confined(self.root, "Frontier-Engineering")

    def manifest(self) -> dict[str, list[str]]:
        frozen = self.root / "frontier_manifest.json"
        if self.mode == "synthetic_dev":
            self.metadata["staged_official_counts"] = dict(read_json(frozen)["counts"])
            return {"train": [], "dev": [FRONTIER_DEV]}
        if self.mode != "real_tasks":
            raise ValueError("Frontier mode must be synthetic_dev or real_tasks")
        path = self.config.get("frontier_manifest")
        if not path:
            raise PermissionError("real Frontier requires an explicit open task manifest")
        official = read_json(frozen) if frozen.is_file() else {"tasks": {}}
        self.metadata["staged_official_counts"] = dict(official.get("counts", {}))
        sealed = {name: row for name, row in official["tasks"].items() if row["split"] in ("sealed", "test")}
        self._sealed_paths = {confined(self.repository / "benchmarks", row["path"]).resolve() for row in sealed.values()}
        manifest_path = Path(path).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = confined(self.root, manifest_path)
        self.manifest_data = read_json(manifest_path)["tasks"]
        selected = {"train": [], "dev": []}
        self._open_rows = {}
        for name, row in self.manifest_data.items():
            split = row.get("split")
            if name in sealed or split in ("sealed", "test"):
                continue
            if split not in ("train", "dev", "val"):
                raise ValueError("real Frontier tasks must have an explicit open split")
            base = self._checked_directory(row["path"])
            self._open_rows[name] = (base, row)
            selected["train" if split == "train" else "dev"].append(name)
        return selected

    def _checked_directory(self, relative: str) -> Path:
        base = confined(self.repository / "benchmarks", relative).resolve()
        if any(base.is_relative_to(path) or path.is_relative_to(base) for path in self._sealed_paths):
            raise PermissionError("Frontier task intersects a sealed task directory")
        return base

    def _batch_settings(self, relative: str, directory: Path) -> dict[str, Any]:
        overrides = {}
        batch = self.repository / "frontier_eval/conf/batch/v1.yaml"
        if batch.is_file():
            data = yaml.safe_load(batch.read_text(encoding="utf-8"))
            for entry in data["tasks"]:
                values = dict(item.split("=", 1) for item in entry.get("overrides", ()) if "=" in item)
                if values.get("task.benchmark", entry.get("label")) == relative:
                    configured = values.get("algorithm.oe.evaluator.timeout")
                    if configured is not None:
                        overrides["timeout_s"] = int(configured)
                    runtime = values.get("task.runtime.python_path", "").removeprefix("uv-env:")
                    overrides["env"] = runtime or values.get("task.runtime.env_name", "driver")
                    break
        local = directory / "v1.yaml"
        if local.is_file() and not local.is_symlink():
            data = yaml.safe_load(local.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("Frontier task v1.yaml must be a mapping")
            overrides.update(data)
        return overrides

    def task(self, name: str, role: str) -> Task:
        if role not in ("train", "dev", "val"):
            raise PermissionError("sealed Frontier split")
        if self.mode == "synthetic_dev" and name == FRONTIER_DEV:
            return Task(name, FRONTIER_PROMPT, env="frontier_eng_dev", metadata={"split": role})
        if name not in getattr(self, "_open_rows", {}):
            raise PermissionError("Frontier task outside approved open manifest")
        directory, row = self._open_rows[name]
        settings = {**self._batch_settings(row["path"], directory), **row}
        timeout = settings.get("timeout_s", settings.get("evaluator_timeout", 300))
        if type(timeout) is not int or timeout not in FRONTIER_TIMEOUTS:
            raise ValueError("Frontier evaluate.py timeout must be 300, 600 or 1200 seconds")
        runtime = settings.get("env", "driver")
        runtime = FRONTIER_ENVS.get(runtime, runtime)
        if runtime not in FRONTIER_ENVS.values():
            raise ValueError("unknown Frontier runtime environment")
        queue = settings.get("queue", "cpu")
        tags = settings.get("tags", ())
        if queue not in ("cpu", "gpu") or not isinstance(tags, (list, tuple)):
            raise ValueError("invalid Frontier queue or tags")
        if runtime == FRONTIER_ENVS["kernel"] or "gpu" in tags:
            queue = "gpu"
        spec = settings.get("task_spec")
        if spec is None:
            prompt_file = settings.get("prompt_file", "README.md")
            spec = confined(directory, prompt_file).read_text(encoding="utf-8")
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("Frontier task requires a nonempty public task spec")
        self._open_rows[name] = (directory, settings)
        return Task(name, spec, env="frontier_eng_real", metadata={"split": role, "queue": queue, "runtime": runtime, "timeout_s": timeout})

    def execution_assets(self, task: Task) -> tuple[Path, dict[str, Any]]:
        self.check_task(task)
        directory, settings = self._open_rows[task.task_id]
        if directory != self._checked_directory(settings["path"]):
            raise PermissionError("Frontier task directory changed")
        return directory, settings


class TDCADMET(_DesignBenchmark):
    proposal_max_tokens = DEEPSEEK_HIGH_MAX_TOKENS

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.metadata.update(
            split_method="scaffold",
            seeds=tuple(config.get("seeds", TDC_SEEDS)),
            oracle_budget=config.get("oracle_budget", TDC_ORACLE_BUDGET),
            max_tokens=self.proposal_max_tokens,
        )

    def manifest(self) -> dict[str, list[str]]:
        self.manifest_data = read_json(self.root / "tdc_admet/manifest.json")
        split = self.manifest_data["splits"]
        active = (*split["dev"], *split["val"])
        if any(name.casefold() in TDC_SEALED for name in active):
            raise PermissionError("sealed ADMET endpoint in active manifest")
        return {"train": split["dev"], "dev": split["val"]}

    def task(self, name: str, role: str) -> Task:
        if name.casefold() in TDC_SEALED or role not in ("train", "dev", "val"):
            raise PermissionError("sealed ADMET endpoint")
        metric, direction = TDC_METRICS[name]
        return Task(
            name,
            f"Endpoint: {name}. Metric: {metric} ({direction}).\n" + RECIPE_PROMPT,
            env="tdc_admet",
            metadata={"split": role, "metric": metric, "direction": direction},
        )

    def dataset(self, task: Task) -> Path:
        if task is not None and task.task_id.casefold() in TDC_SEALED:
            raise PermissionError("sealed ADMET endpoint")
        self.check_task(task)
        path = confined(self.root / "tdc_admet/private_scorer", f"{task.task_id}/train_val.csv")
        expected = self.manifest_data["csv_sha256"][f"admet_group/{task.task_id}/train_val.csv"]
        if sha256(path) != expected:
            raise InfraError("active ADMET source digest mismatch")
        return path


class KernelBench(_DesignBenchmark):
    proposal_max_tokens = DEEPSEEK_HIGH_MAX_TOKENS

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.metadata.update(oracle_budget=config.get("oracle_budget", KERNEL_ORACLE_BUDGET), max_tokens=self.proposal_max_tokens)

    def manifest(self) -> dict[str, list[str]]:
        self.manifest_data = read_json(self.root / "KernelBench/manifest.json")
        split = self.manifest_data["splits"]
        return {"train": split["train"], "dev": split["val"]}

    def reference(self, name: str) -> Path:
        if not hasattr(self, "manifest_data"):
            self.manifest()
        split = self.manifest_data["splits"]
        if name in split["test"] or name not in (*split["train"], *split["val"]):
            raise PermissionError("KernelBench task outside frozen open splits")
        ref = confined(self.root / "KernelBench/KernelBench", name)
        if sha256(ref) != self.manifest_data["files"][name]:
            raise InfraError("KernelBench pinned reference digest mismatch")
        return ref

    def task(self, name: str, role: str) -> Task:
        if role not in ("train", "dev", "val"):
            raise PermissionError("sealed KernelBench split")
        ref = self.reference(name)
        prompt = (
            "Implement the same interface using a custom Triton GPU kernel. "
            "Submit candidate.py containing ModelNew and no timing code.\n\n" + ref.read_text(encoding="utf-8")
        )
        return Task(name, prompt, env="kernelbench", metadata={"split": role})
