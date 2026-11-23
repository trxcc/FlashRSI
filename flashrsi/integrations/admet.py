"""Evaluate ADMET recipes on isolated scaffold train and validation folds.

Fitting sees training labels, prediction sees validation features, and the
host scorer retains validation labels. Scientific execution dependencies live
inside the offline image selected via FLASHRSI_ADMET_IMAGE.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any

import numpy as np

from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.benchmarks.design_suite import (
    TDC_METRICS,
    TDC_ORACLE_BUDGET,
    TDC_SEEDS,
    TDCADMET,
    positive_integer,
)
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.integrations.design_suite import (
    ENG_RESOURCES,
    CandidateExecutionError,
    Resources,
    Verdict,
    admet_image,
    artifact_proposal,
    container_limits,
    docker_json,
    oracle_budget,
    scratch,
    smoke_mode,
)

MODELS = {"ridge", "extra_trees", "random_forest", "lightgbm", "xgboost"}
MOL_RESOURCES = {**ENG_RESOURCES, "shm_size": "512m"}
MOL_FIT_TIMEOUT = 900
MOL_PREDICT_TIMEOUT = 120
Folds = Mapping[int, tuple[Path, dict[str, float]]]


MOL_SCAFFOLD = """
import csv
import json
import random
import sys
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
groups = {}
with open("/train_val.csv", newline="") as stream:
    for index, row in enumerate(csv.DictReader(stream)):
        mol = Chem.MolFromSmiles(row["Drug"])
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            groups.setdefault(scaffold, set()).add(index)

# PyTDC c310c35 get_train_valid_split uses fractions (0.875, 0.125, 0).
# Its two shuffled buckets depend on the integer rounding remainder.
count = sum(map(len, groups.values()))
train_size, valid_size = int(count * 0.875), int(count * 0.125)
large, small = [], []
for group in groups.values():
    if len(group) > min(valid_size, count - train_size - valid_size) / 2:
        large.append(group)
    else:
        small.append(group)
folds = {}
for seed in json.loads(sys.argv[1]):
    rng = random.Random(seed)
    first, last = large.copy(), small.copy()
    rng.shuffle(first)
    rng.shuffle(last)
    train, valid = [], []
    for group in first + last:
        (train if len(train) + len(group) <= train_size else valid).extend(group)
    folds[str(seed)] = [train, valid]
print(json.dumps(folds))
"""

MOL_FEATURES = """
import csv
import json
import pickle
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

def features(smiles, generator):
    mol = Chem.MolFromSmiles(smiles)
    return np.r_[generator.GetFingerprintAsNumPy(mol), Descriptors.MolWt(mol) / 100,
                 Descriptors.MolLogP(mol), Descriptors.TPSA(mol) / 100,
                 Descriptors.NumRotatableBonds(mol)]

def prepare(rows, radius):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=512)
    return np.asarray([features(row["smiles"], generator) for row in rows], dtype=np.float32)
"""

MOL_FIT = (
    MOL_FEATURES
    + """
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

cfg = json.loads(sys.argv[1])
binary, seed = cfg["binary"], cfg["seed"]
with open("/train.csv", newline="") as stream:
    rows = list(csv.DictReader(stream))
X = prepare(rows, cfg["radius"])
y = np.asarray([float(row["y"]) for row in rows])
common = dict(n_estimators=cfg["trees"], n_jobs=1, random_state=seed)
model = cfg["model"]
if model == "ridge":
    estimator = LogisticRegression(C=1 / cfg["leaf"], random_state=seed) if binary else Ridge(alpha=cfg["leaf"])
elif model == "extra_trees":
    cls = ExtraTreesClassifier if binary else ExtraTreesRegressor
    estimator = cls(min_samples_leaf=cfg["leaf"], **common)
elif model == "random_forest":
    cls = RandomForestClassifier if binary else RandomForestRegressor
    estimator = cls(min_samples_leaf=cfg["leaf"], **common)
elif model == "lightgbm":
    cls = LGBMClassifier if binary else LGBMRegressor
    estimator = cls(min_child_samples=cfg["leaf"], verbosity=-1, **common)
else:
    cls = XGBClassifier if binary else XGBRegressor
    estimator = cls(max_depth=4, min_child_weight=cfg["leaf"], tree_method="hist", **common)
estimator.fit(X, y)
with open("/model/estimator.pkl", "wb") as stream:
    pickle.dump(estimator, stream)
print("{}")
"""
)

MOL_PREDICT = (
    MOL_FEATURES
    + """
cfg = json.loads(sys.argv[1])
with open("/val_features.csv", newline="") as stream:
    rows = list(csv.DictReader(stream))
with open("/model/estimator.pkl", "rb") as stream:
    estimator = pickle.load(stream)
X = prepare(rows, cfg["radius"])
if cfg["binary"]:
    positive = list(estimator.classes_).index(1)
    values = estimator.predict_proba(X)[:, positive]
else:
    values = estimator.predict(X)
print(json.dumps({row["id"]: float(value) for row, value in zip(rows, values)}, allow_nan=False))
"""
)


def mol_recipe(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"model", "radius", "trees", "leaf"}:
        raise ValueError("invalid recipe format")
    if value["model"] not in MODELS or value["radius"] not in (2, 3):
        raise ValueError("invalid recipe choice")
    for field, minimum, maximum in (("radius", 2, 3), ("trees", 16, 128), ("leaf", 1, 12)):
        if type(value[field]) is not int or not minimum <= value[field] <= maximum:
            raise ValueError("invalid recipe range")
    return value


SEED = "def solve():\n    return (0.25, 0.25, 0.25, 0.25)\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows({k: row[k] for k in names} for row in rows)


def admet_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    if smoke_mode(config):
        seeds = config.get("smoke_seeds", config.get("seeds", (1,)))
    else:
        seeds = config.get("seeds", TDC_SEEDS)
    if (
        not isinstance(seeds, (tuple, list))
        or not seeds
        or any(type(seed) is not int or seed not in TDC_SEEDS for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("ADMET seeds must be distinct integers from 1 to 5")
    return tuple(seeds)


def admet_resources(config: Mapping[str, Any]) -> Resources:
    resources = config.get("resources", {})
    if not isinstance(resources, Mapping) or set(resources) - MOL_RESOURCES.keys():
        raise ValueError("ADMET resources must contain supported resource names")
    resolved = {**MOL_RESOURCES, **resources}
    container_limits(resolved)
    return resolved


def admet_timeout(config: Mapping[str, Any], field: str, default: int) -> int:
    return positive_integer(f"ADMET {field}", config.get(field, default))


def prepare_admet_folds(benchmark: TDCADMET, task: Task, folder: Path, seeds: Sequence[int]) -> Folds:
    source = benchmark.dataset(task)
    folds, _ = docker_json(
        admet_image(),
        ["-c", MOL_SCAFFOLD, json.dumps(seeds)],
        ((source, "/train_val.csv"),),
        timeout=admet_timeout(benchmark.config, "predict_timeout", MOL_PREDICT_TIMEOUT),
        resources=admet_resources(benchmark.config),
    )
    with source.open(newline="") as stream:
        rows = [{"id": str(i), "smiles": row["Drug"], "y": float(row["Y"])} for i, row in enumerate(csv.DictReader(stream))]
    prepared = {}
    for seed in seeds:
        train_ids, valid_ids = folds[str(seed)]
        if (
            not train_ids
            or not valid_ids
            or len(set(train_ids + valid_ids)) != len(train_ids) + len(valid_ids)
            or any(type(i) is not int or not 0 <= i < len(rows) for i in train_ids + valid_ids)
        ):
            raise InfraError("offline ADMET splitter returned an invalid scaffold fold")
        fold = folder / f"seed-{seed}"
        fold.mkdir()
        train, valid = ([rows[i] for i in ids] for ids in (train_ids, valid_ids))
        _write_csv(fold / "train.csv", train, ("id", "smiles", "y"))
        _write_csv(fold / "val_features.csv", valid, ("id", "smiles"))
        prepared[seed] = fold, {row["id"]: row["y"] for row in valid}
    return prepared


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    boundaries = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, len(values)]
    ranks = np.empty(len(values), dtype=float)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return ranks


def admet_metric(metric: str, labels: Mapping[str, float], predictions: object) -> float:
    if (
        not isinstance(predictions, dict)
        or predictions.keys() != labels.keys()
        or any(type(value) not in (int, float) for value in predictions.values())
    ):
        raise ValueError("ADMET predictions must match validation row IDs")
    y = np.asarray(list(labels.values()), dtype=float)
    p = np.asarray([predictions[row] for row in labels], dtype=float)
    if not len(y) or not np.isfinite(y).all():
        raise InfraError("ADMET validation labels are missing or nonfinite")
    if not np.isfinite(p).all():
        raise ValueError("ADMET predictions must be finite")
    if metric == "MAE":
        value = float(np.mean(np.abs(p - y)))
    elif metric == "Spearman":
        p_rank, y_rank = _ranks(p), _ranks(y)
        if np.ptp(p_rank) == 0 or np.ptp(y_rank) == 0:
            raise ValueError("Spearman is undefined for a constant rank vector")
        value = float(np.clip(np.corrcoef(p_rank, y_rank)[0, 1], -1.0, 1.0))
    elif metric in ("AUROC", "AUPRC"):
        if ((p < 0) | (p > 1)).any():
            raise ValueError("ADMET classification requires positive-class probabilities in [0,1]")
        if set(y) != {0.0, 1.0}:
            raise InfraError("ADMET classification fold requires both binary classes")
        positives = int(y.sum())
        if metric == "AUROC":
            rank_sum = _ranks(p)[y == 1].sum()
            value = float((rank_sum - positives * (positives + 1) / 2) / (positives * (len(y) - positives)))
        else:
            order = np.argsort(-p, kind="stable")
            sorted_prob = p[order]
            ends = np.r_[np.flatnonzero(sorted_prob[:-1] != sorted_prob[1:]), len(p) - 1]
            tp = np.cumsum(y[order])[ends]
            value = float(np.dot(np.diff(np.r_[0, tp]) / positives, tp / (ends + 1)))
    else:
        raise ValueError("unknown ADMET metric")
    if not math.isfinite(value):
        raise ValueError("ADMET metric is nonfinite")
    return value


def aggregate_admet(metric: str, direction: str, trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw = [row["raw_metric"] for row in trials]
    rounded = [round(value, 3) for value in raw]
    mean = fmean(raw)
    if metric == "MAE":
        normalized = 1 / (1 + mean)
    elif metric == "Spearman":
        normalized = (mean + 1) / 2
    else:
        normalized = mean
    normalized = min(1.0, max(0.0, normalized))
    return {
        "score": normalized,
        "normalized_score": normalized,
        "metric": metric,
        "direction": direction,
        "tdc_mean": round(fmean(rounded), 3),
        "tdc_std": round(pstdev(rounded), 3),
        "raw_mean": mean,
        "raw_sample_sd": stdev(raw) if len(raw) > 1 else 0.0,
        "seed_metrics": [{**row, "tdc_metric": rounded[i]} for i, row in enumerate(trials)],
        "fold": "PyTDC scaffold train/valid",
    }


def score_admet(benchmark: TDCADMET, task: Task, recipe: dict[str, Any], folder: Path, *, folds: Folds | None = None) -> dict[str, Any]:
    metric, direction = TDC_METRICS[task.task_id]
    if folds is None:
        folds = prepare_admet_folds(benchmark, task, folder, admet_seeds(benchmark.config))
    config = getattr(benchmark, "config", {})
    resources = admet_resources(config)
    fit_timeout = admet_timeout(config, "fit_timeout", MOL_FIT_TIMEOUT)
    predict_timeout = admet_timeout(config, "predict_timeout", MOL_PREDICT_TIMEOUT)
    image = admet_image()
    results = []
    for seed, (fold, labels) in folds.items():
        cfg = json.dumps({**recipe, "seed": seed, "binary": metric in ("AUROC", "AUPRC")})
        with tempfile.TemporaryDirectory(dir=folder, prefix="model-") as model:
            model = Path(model)
            _, fit = docker_json(
                image,
                ["-c", MOL_FIT, cfg],
                ((fold / "train.csv", "/train.csv"),),
                fit_timeout,
                writable=((model, "/model"),),
                resources=resources,
                candidate=True,
            )
            pred, predict = docker_json(
                image,
                ["-c", MOL_PREDICT, cfg],
                ((fold / "val_features.csv", "/val_features.csv"), (model, "/model")),
                predict_timeout,
                resources=resources,
                candidate=True,
            )
        results.append(
            {
                "seed": seed,
                "raw_metric": admet_metric(metric, labels, pred),
                "validation_rows": len(labels),
                "fit_container": fit,
                "predict_container": predict,
                **({"probability_domain_validated": True} if metric in ("AUROC", "AUPRC") else {}),
            }
        )
    result = aggregate_admet(metric, direction, results)
    if metric == "MAE":
        result["mae"] = result["raw_mean"]
    return {**result, "recipe": recipe}


async def admet(agent: Agent, benchmark: TDCADMET, task: Task) -> Verdict:
    budget = oracle_budget(benchmark.config, TDC_ORACLE_BUDGET, TDC_ORACLE_BUDGET)
    seeds = admet_seeds(benchmark.config)
    metric, direction = TDC_METRICS[task.task_id]
    best = None
    slots, usage = [], []
    program_search = isinstance(agent, ProgramSearchTarget)
    if program_search:
        current = agent.snapshot.files["prompt/workflow.json"].content.decode()
        memory = agent.snapshot.files["memory/solution_memory.jsonl"].content.decode()
    else:
        current, memory = "", ""
    with scratch() as tmp:
        folder = Path(tmp)
        folds = await asyncio.to_thread(prepare_admet_folds, benchmark, task, folder, seeds)
        for slot in range(1, budget + 1):
            attempts = []
            for attempt in (1, 2):
                prompt = (
                    task.question
                    + "\nCurrent recipe: "
                    + current
                    + "\nPast public scores: "
                    + memory[-4000:]
                    + f"\nWorkflow slot {slot}/{budget}, attempt {attempt}/2."
                )
                if attempts:
                    prompt += "\nRepair the previous recipe. Failure: " + attempts[-1]["error"]
                try:
                    if program_search:
                        candidate, tokens = await agent.propose(prompt)
                    else:
                        with tempfile.TemporaryDirectory(dir=folder) as public:
                            raw, trace, _ = await artifact_proposal(
                                agent, task, Path(public), image=admet_image(), filename="recipe.json", prompt=prompt
                            )
                        tokens, candidate = dict(trace.usage), json.loads(raw)
                    usage.append(tokens)
                    recipe = mol_recipe(candidate)
                except (SyntaxError, ValueError) as exc:
                    attempts.append({"attempt": attempt, "valid": False, "error": type(exc).__name__})
                    continue
                try:
                    result = await asyncio.to_thread(score_admet, benchmark, task, recipe, folder, folds=folds)
                except CandidateExecutionError as exc:
                    attempts.append({"attempt": attempt, "recipe": recipe, "valid": False, "error": str(exc)})
                    continue
                except (ValueError, subprocess.TimeoutExpired) as exc:
                    attempts.append(
                        {
                            "attempt": attempt,
                            "recipe": recipe,
                            "valid": False,
                            "error": "candidate_timeout" if isinstance(exc, subprocess.TimeoutExpired) else str(exc),
                        }
                    )
                    break
                attempts.append({"attempt": attempt, "valid": True, **result})
                if best is None or result["score"] > best["score"]:
                    best, current = result, json.dumps(recipe, sort_keys=True)
                break
            slots.append({"slot": slot, "attempts": attempts})
    verdict = best or {"score": 0.0, "normalized_score": 0.0, "metric": metric, "direction": direction}
    return Trajectory(usage={"model_calls": usage}), {
        **verdict,
        "candidate_valid": best is not None,
        "seeds": list(seeds),
        "budget": budget,
        "oracle_calls": len(slots),
        "trial_receipts": slots,
    }
