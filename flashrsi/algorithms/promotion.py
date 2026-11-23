"""Promotion policies for workspace and sequential search benchmarks.

These algorithms operate on host-side verifier receipts. The workspace scorer
is loaded only when pooling criterion results from that benchmark family.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from flashrsi.core import Algorithm, InfraError, Snapshot, Task, Trajectory


@dataclass(frozen=True, slots=True)
class WorkspaceGate:
    strict_claw_gain: bool = False
    max_growth_ratio: float | None = 0.10
    min_harvey_gain: float = 0.004
    claw_tol: float = 0.0

    @staticmethod
    def growth_ratio(parent_size: int, trial_size: int) -> float:
        return (trial_size - parent_size) / max(parent_size, 1)

    def accepts(
        self,
        parent: tuple[float, float],
        trial: tuple[float, float],
        parent_size: int,
        trial_size: int,
    ) -> bool:
        parent_claw, parent_harvey = parent
        trial_claw, trial_harvey = trial
        if not all(
            math.isfinite(value) and 0 <= value <= 1 for value in (*parent, *trial)
        ):
            raise ValueError("Workspace gate scores must be finite and within [0,1]")
        if self.max_growth_ratio is not None:
            if not math.isfinite(self.max_growth_ratio) or self.max_growth_ratio < 0:
                raise ValueError("max_growth_ratio must be nonnegative and finite")
            if self.growth_ratio(parent_size, trial_size) > self.max_growth_ratio:
                return False
        claw_ok = (
            trial_claw > parent_claw
            if self.strict_claw_gain
            else trial_claw >= parent_claw - self.claw_tol
        )
        return claw_ok and trial_harvey >= parent_harvey + self.min_harvey_gain


class TwoGateWorkspacePromotion(Algorithm):
    name = "two_gate_workspace"
    description = (
        "Promote on Claw hard non-regression, Harvey micro gain and bounded growth."
    )
    evolve_scope = {"prompt"}
    _defaults = {
        "rounds": 1,
        "strict_claw_gain": False,
        "max_growth_ratio": 0.10,
        "min_harvey_gain": 0.004,
        "claw_tol": 0.0,
        "val_repeats": 2,
        "smoke_val_repeats": 1,
        "smoke": False,
        "claw_evidence_chars": 8000,
        "harvey_evidence_chars": 8000,
        "worker_max_tokens": None,
    }
    _ranges = {
        "rounds": (int, 1),
        "strict_claw_gain": bool,
        "max_growth_ratio": (float | None, 0),
        "min_harvey_gain": (float, 0),
        "claw_tol": (float, 0),
        "val_repeats": (int, 1),
        "smoke_val_repeats": (int, 1),
        "smoke": bool,
        "claw_evidence_chars": (int, 1),
        "harvey_evidence_chars": (int, 1),
        "worker_max_tokens": (int | None, 1),
    }

    def _check(self) -> None:
        if any(
            self.config[name] > 8000
            for name in ("claw_evidence_chars", "harvey_evidence_chars")
        ):
            raise ValueError(
                "Workspace evidence may contain at most 8,000 chars per family"
            )
        if self.config.min_harvey_gain > 1 or self.config.claw_tol > 1:
            raise ValueError("Workspace score thresholds must be within [0,1]")
        # Both the original and the updated llm Worker use the active llm payload.
        if self.config.worker_max_tokens is not None:
            self.worker.config["llm"]["max_tokens"] = self.config.worker_max_tokens

    @staticmethod
    def scores(results: Iterable[Trajectory]) -> dict[str, float]:
        claw = []
        harvey = []
        for row in results:
            family = _task(row).metadata["split"]
            if family.startswith("clawgym"):
                claw.append(row.metrics["reward"])
            elif family.startswith("harvey"):
                harvey.append(row.metrics)
        if not claw or not harvey:
            raise ValueError("both Workspace validation families are required")
        from flashrsi.benchmarks.workspace_suite import summarize_harvey

        summary = summarize_harvey(harvey)
        return {
            "claw_score": mean(claw),
            "harvey_micro_avg": summary["micro_avg"],
            "harvey_macro_avg": summary["macro_avg"],
            "harvey_all_pass": summary["all_pass"],
            "passed_criteria": summary["passed_criteria"],
            "total_criteria": summary["total_criteria"],
        }

    @classmethod
    def size(cls, snapshot: Snapshot) -> int:
        return sum(
            len(file.content)
            for file in snapshot.files.values()
            if file.component in cls.evolve_scope
        )

    def evidence(self, results: Iterable[Trajectory], family: str, budget: int) -> str:
        matching = [
            row for row in results if _task(row).metadata["split"].startswith(family)
        ]
        ordered = sorted(matching, key=lambda row: row.metrics["reward"])[:8]
        if not ordered:
            return ""
        # Budget the weakest attempts separately so one long traceback cannot
        # consume the space reserved for other attempts or for the other family.
        part = max(0, (budget - len(ordered) + 1) // len(ordered))
        lines = []
        for row in ordered:
            line = json.dumps(_diagnostic(row, family), ensure_ascii=False)
            if part and len(line) > part:
                line = line[: part - 1] + "…"
            lines.append(line if part else "")
        return "\n".join(lines)[:budget]

    async def validation(
        self,
        snapshot: Snapshot,
        label: str,
        *,
        smoke: bool | None = None,
    ) -> dict[str, float]:
        benchmark_config = getattr(self.benchmark, "config", {})
        smoke = (
            (self.config.smoke or benchmark_config.get("smoke", False))
            if smoke is None
            else smoke
        )
        repeats = self.config.smoke_val_repeats if smoke else self.config.val_repeats
        # Retain all trials, even when the task ID repeats. Results.per_task
        # intentionally contains only one row per ID and would drop repeats.
        results = []
        for trial in range(repeats):
            results.extend(
                await self._evaluate(
                    snapshot, self.benchmark.dev, label=f"{label}-{trial + 1}"
                )
            )
        return self.scores(results)

    async def evolve(self, *, smoke: bool | None = None) -> Snapshot:
        seed = current = self.target.snapshot
        path = self._instruction_path()
        gate = WorkspaceGate(
            strict_claw_gain=self.config.strict_claw_gain,
            max_growth_ratio=self.config.max_growth_ratio,
            min_harvey_gain=self.config.min_harvey_gain,
            claw_tol=self.config.claw_tol,
        )
        current_scores = await self.validation(current, "baseline-val", smoke=smoke)
        for _ in range(self.config.rounds):
            train = await self._evaluate(
                current, self.benchmark.train, label="workspace-train"
            )
            claw = self.evidence(train, "clawgym", self.config.claw_evidence_chars)
            harvey = self.evidence(train, "harvey", self.config.harvey_evidence_chars)
            old = current.files[path].content.decode() if path in current.files else ""
            response = await self.worker.ask(
                "Write a replacement reusable agent instruction. Return only that instruction.\n"
                f"Current:\n{old}\nClawGym diagnostics:\n{claw}\nHarvey LAB diagnostics:\n{harvey}"
            )
            candidate = self._candidate(current, {path: response.strip() + "\n"})
            values = await self.validation(candidate, "workspace-val", smoke=smoke)
            parent = (current_scores["claw_score"], current_scores["harvey_micro_avg"])
            trial = (values["claw_score"], values["harvey_micro_avg"])
            parent_size, trial_size = self.size(current), self.size(candidate)
            accepted = gate.accepts(parent, trial, parent_size, trial_size)
            self._record(
                "workspace-gate",
                {
                    "parent": current_scores,
                    "candidate": values,
                    **values,
                    "growth_ratio": gate.growth_ratio(parent_size, trial_size),
                    "accepted": accepted,
                },
            )
            if accepted:
                current, current_scores = candidate, values
        return self._promote(seed, current, scores=current_scores)


TDC_SPEARMAN_VAL = frozenset(
    {
        "half_life_obach",
        "clearance_hepatocyte_az",
        "clearance_microsome_az",
    }
)
TDC_RAW_METRICS = {"spearman", "mae", "auroc", "auprc"}


def _task(row: Trajectory) -> Task:
    if row.task is None:
        raise InfraError("promotion requires a task-bound verifier result")
    return row.task


def _diagnostic(row: Trajectory, family: str) -> dict[str, Any]:
    event = row.messages[-1]
    if family == "clawgym":
        details = (
            event.get("checker_stderr"),
            event.get("checker_stdout"),
            event.get("checker_error"),
            row.error_message,
        )
        return {
            "score": row.metrics["reward"],
            "checker_exit": event.get("checker_exit"),
            "diagnostic": "\n".join(str(detail) for detail in details if detail),
        }
    return {
        "score": row.metrics["reward"],
        "criteria_failed": [
            item["reason"]
            for item in event.get("verdicts", ())
            if item["pass"] is False
        ],
    }


def _receipt_field(sources: Sequence[Mapping[str, Any]], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            if (value := source.get(key)) is not None:
                return value
    return None


def _metric_name(sources: Sequence[Mapping[str, Any]]) -> str | None:
    name = _receipt_field(sources, "metric_name", "metric")
    if isinstance(name, str):
        return name.lower()
    if name is None:
        return next(
            (
                key
                for key in sorted(TDC_RAW_METRICS)
                if _receipt_field(sources, key) is not None
            ),
            None,
        )
    raise InfraError("invalid TDC validation metric name")


def _numeric_score(value: Any, error: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise InfraError(error)
    return float(value)


def normalized_endpoint_score(
    row: Trajectory, *, expected_metric: str | None = None
) -> float:
    """Read a trusted normalized receipt or normalize the unrounded raw metric."""
    event = row.messages[-1] if row.messages else {}
    candidates = (
        event,
        event.get("metadata"),
        row.metrics,
        getattr(row, "metadata", None),
    )
    sources = tuple(source for source in candidates if isinstance(source, Mapping))
    name = _metric_name(sources)
    if expected_metric is not None and name is not None and name != expected_metric:
        raise InfraError(
            f"TDC validation requires {expected_metric} for {_task(row).task_id}"
        )
    if _receipt_field(sources, "candidate_valid", "valid") is False:
        return 0.0
    normalized = _receipt_field(
        sources,
        "direction_normalized_score",
        "normalized_score",
        "direction_normalized",
        "normalized_reward",
    )
    if normalized is not None:
        message = "invalid normalized TDC validation score"
        score = _numeric_score(normalized, message)
        if not 0 <= score <= 1:
            raise InfraError(message)
        return score
    if name is None or name not in TDC_RAW_METRICS:
        raise InfraError("missing TDC validation metric")
    raw = _receipt_field(
        sources,
        name,
        "raw_metric",
        "raw_score",
        "raw_mean",
        "metric_value",
        "metric_mean",
        "research_mean",
    )
    score = _numeric_score(raw, "missing or invalid raw TDC validation metric")
    if name == "mae" and score >= 0:
        return 1 / (1 + score)
    if name == "spearman" and -1 <= score <= 1:
        return (score + 1) / 2
    if name in {"auroc", "auprc"} and 0 <= score <= 1:
        return score
    raise InfraError("raw TDC validation metric is outside its valid range")


def _memory_updates(
    current: Snapshot, task: Task, event: Mapping[str, Any]
) -> dict[str, str]:
    recipe = event.get("recipe")
    entries = {
        "solution": {"score": event.get("score"), "recipe": recipe},
        "refinement": {"parent": current.id, "oracle_calls": event.get("oracle_calls")},
        "execution": {"verifier": event["type"], "budget": event.get("budget")},
    }
    updates = {}
    for kind, details in entries.items():
        path = f"memory/{kind}_memory.jsonl"
        state = current.files[path].content.decode()
        updates[path] = (
            state
            + json.dumps({"task_id": task.task_id, **details}, sort_keys=True)
            + "\n"
        )
    if recipe:
        updates["prompt/workflow.json"] = json.dumps(recipe, sort_keys=True) + "\n"
    return updates


class SequentialMemory(Algorithm):
    name = "sequential_memory"
    description = "Commit public endpoint receipts in order and gate the accumulated memory on validation."
    evolve_scope = {"prompt", "memory"}
    observation_scope = {"skill"}
    _defaults = {}

    async def validation(
        self,
        snapshot: Snapshot,
        tasks: Sequence[Task],
        label: str,
    ) -> tuple[float | None, dict[str, float]]:
        if not tasks:
            return None, {}
        results = await self._evaluate(snapshot, tasks, label=label)
        if self.benchmark.name != "tdc_admet":
            return results.score, {}
        scores = {
            _task(row).task_id: normalized_endpoint_score(
                row, expected_metric="spearman"
            )
            for row in results
        }
        return mean(scores.values()), scores

    async def evolve(self) -> Snapshot:
        seed = current = self.target.snapshot
        train = self.benchmark.train
        val = self.benchmark.dev
        # Frontier's synthetic development pool serves as its search pool.
        if not train and self.benchmark.name == "frontier_eng":
            train, val = val, ()
        if not train:
            raise ValueError("sequential_memory requires active endpoint tasks")
        is_tdc = self.benchmark.name == "tdc_admet"
        if is_tdc and {task.task_id for task in val} != TDC_SPEARMAN_VAL:
            raise ValueError(
                "TDC SequentialMemory requires all three frozen Spearman Val endpoints"
            )
        base_score, base_endpoints = await self.validation(
            seed, val, "sequential-baseline-val"
        )
        for task in train:
            row = (await self._evaluate(current, [task], label="sequential-step"))[0]
            current = self._candidate(
                current, _memory_updates(current, task, row.messages[-1])
            )
        val_score, val_endpoints = await self.validation(
            current, val, "sequential-candidate-val"
        )
        if is_tdc:
            if val_score is None or base_score is None:
                raise InfraError("missing TDC validation score for promotion")
            accepted = val_score > base_score and all(
                val_endpoints[key] >= score for key, score in base_endpoints.items()
            )
        else:
            accepted = val_score is None or (
                base_score is not None and val_score >= base_score
            )
        return self._promote(
            seed,
            current if accepted else seed,
            baseline_validation=base_score,
            candidate_validation=val_score,
            baseline_endpoints=base_endpoints,
            candidate_endpoints=val_endpoints,
        )
