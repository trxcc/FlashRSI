"""Verify engineering, ADMET, and GPU worker contracts.

Tests use host fixtures, staged offline Docker images and a loopback CPU GPU
worker. FLASHRSI_SUITE_ENVS_ROOT or FLASHRSI_SUITE_ENG_AND_MOL_ROOT enables
the checks against prepared assets.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
import secrets
import select
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import repeat
from pathlib import Path
from statistics import mean, pstdev, stdev
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.benchmarks import _suite
from flashrsi.benchmarks import design_suite as benchmarks
from flashrsi.core import InfraError, Task, Trajectory
from flashrsi.integrations import (
    admet,
    design_suite,
    engdesign,
    frontier_eng,
    gpu_worker,
    kernelbench,
)
from scripts import gpu_worker_daemon as daemon

STAGED = _suite.resolve_root({}, "eng_and_mol")
RECIPE = {"model": "ridge", "radius": 2, "trees": 16, "leaf": 2}
REFERENCE = """import random
class Model:
    def __call__(self, x):
        return x * 2
def get_inputs():
    return [random.random()]
def get_init_inputs():
    return []
"""
DRY_CANDIDATE = """class ModelNew:
    def forward(self, x):
        return x * 2
"""


async def propose_fixture(prompt, *, answer, prompts):
    prompts.append(prompt)
    result = answer(len(prompts)) if callable(answer) else answer
    if isinstance(result, BaseException):
        raise result
    return result, {"fixture_calls": 1}


def make_agent(answer):
    agent = ProgramSearchTarget({"llm": {"model_name": "fixture", "api_key": "fixture"}})
    prompts = []
    agent.propose = partial(propose_fixture, answer=answer, prompts=prompts)
    return agent, prompts


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


@pytest.mark.parametrize("name", sorted(_suite.SEALED))
@pytest.mark.parametrize("allow", [False, True])
def test_sealed_suite_names_are_guarded(name, allow):
    with pytest.raises(PermissionError):
        _suite.guard({"name": name, "allow_sealed_test": allow})
    loader = {"frontier_eng_sealed": benchmarks.FrontierEng, "tdc_toxicity_sealed": benchmarks.TDCADMET,
              "kernelbench_sealed": benchmarks.KernelBench}.get(name)
    if loader:
        with pytest.raises(PermissionError):
            loader({"name": name, "allow_sealed_test": allow})


def test_admet_metric_table_and_sealed_aliases():
    expected = {
        "MAE": "caco2_wang lipophilicity_astrazeneca solubility_aqsoldb ppbr_az ld50_zhu",
        "Spearman": "vdss_lombardo half_life_obach clearance_hepatocyte_az clearance_microsome_az",
        "AUROC": "hia_hou pgp_broccatelli bioavailability_ma bbb_martins "
                 "cyp3a4_substrate_carbonmangels herg ames dili",
        "AUPRC": "cyp2c9_substrate_carbonmangels cyp2d6_substrate_carbonmangels "
                 "cyp2c9_veith cyp2d6_veith cyp3a4_veith",
    }
    assert benchmarks.TDC_METRICS == {
        name: (metric, "minimize" if metric == "MAE" else "maximize")
        for metric, names in expected.items() for name in names.split()
    }
    for name in ("hERG", "AMES", "DILI", "LD50", "ld50_zhu"):
        with pytest.raises(PermissionError):
            benchmarks.TDCADMET({"name": "tdc_admet"}).task(name, "train")


@pytest.mark.parametrize("metric,expected", [("AUROC", 0.875), ("AUPRC", 5 / 6)])
def test_admet_classifier_probability_domain_and_ties(metric, expected):
    truth = {"a": 0, "b": 1, "c": 0, "d": 1}
    values = {"a": .1, "b": .5, "c": .5, "d": .9}
    assert admet.admet_metric(metric, truth, values) == pytest.approx(expected)
    for bad in (-.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            admet.admet_metric(metric, truth, {**values, "a": bad})
    with pytest.raises(ValueError, match="row IDs"):
        admet.admet_metric(metric, truth, {"a": 0.5})


def test_admet_seed_aggregation_and_spearman():
    labels = {"0": 4.0, "1": 2.0, "2": 6.0}
    assert admet.admet_metric("Spearman", labels, {"0": 2, "1": 3, "2": 1}) == pytest.approx(-1)
    assert admet.admet_metric("MAE", labels, {"0": 4, "1": 3, "2": 4}) == 1
    with pytest.raises(ValueError, match="undefined"):
        admet.admet_metric("Spearman", labels, dict.fromkeys(labels, 0.5))
    values = [-.10077, .34872, -.012345, .1789, .32876]
    rows = [{"raw_metric": value, "seed": seed} for seed, value in enumerate(values, 1)]
    result = admet.aggregate_admet("Spearman", "maximize", rows)
    rounded = [round(value, 3) for value in values]
    assert result["tdc_mean"] == round(mean(rounded), 3)
    assert result["tdc_std"] == round(pstdev(rounded), 3)
    assert result["raw_mean"] == pytest.approx(mean(values))
    assert result["raw_sample_sd"] == pytest.approx(stdev(values))
    assert result["normalized_score"] == pytest.approx((mean(values) + 1) / 2)
    assert [row["tdc_metric"] for row in result["seed_metrics"]] == rounded
    single = admet.aggregate_admet("MAE", "minimize", [{"raw_metric": 3.33333}])
    assert single["normalized_score"] == pytest.approx(1 / 4.33333)
    assert single["tdc_std"] == single["raw_sample_sd"] == 0


def test_real_admet_csv_five_seed_offline(tmp_path):
    if not (STAGED / "tdc_admet/private_scorer/solubility_aqsoldb/train_val.csv").is_file() or not shutil.which("docker"):
        pytest.skip("staged ADMET CSV and Docker required")
    benchmark = benchmarks.TDCADMET({"name": "tdc_admet", "task_ids": ["solubility_aqsoldb"]})
    task = benchmark.load()["train"][0]
    result = admet.score_admet(benchmark, task, RECIPE, tmp_path)
    assert [row["seed"] for row in result["seed_metrics"]] == [1, 2, 3, 4, 5]
    seeds = result["seed_metrics"]
    assert all(row["fit_container"]["cleaned"] and row["predict_container"]["cleaned"] for row in seeds)
    raw = [row["raw_metric"] for row in seeds]
    assert result["raw_mean"] == pytest.approx(mean(raw))
    assert result["raw_sample_sd"] == pytest.approx(stdev(raw))
    assert result["tdc_mean"] == round(mean(round(value, 3) for value in raw), 3)
    features = {(tmp_path / f"seed-{seed}/val_features.csv").read_text() for seed in range(1, 6)}
    assert len(features) == 5 and all(text.startswith("id,smiles\n") for text in features)


def test_admet_fit_predict_deadlines_and_mounts(tmp_path, monkeypatch):
    labels = {"a": 0, "b": 1}
    evaluate = Mock(side_effect=[({}, {"cleaned": True}), ({"a": .1, "b": .8}, {"cleaned": True})])
    monkeypatch.setenv("FLASHRSI_ADMET_IMAGE", "fixture")
    monkeypatch.setattr(admet, "docker_json", evaluate)
    task = Task("hia_hou", "fixture")
    result = admet.score_admet(None, task, RECIPE, tmp_path, folds={1: (tmp_path, labels)})
    assert result["metric"] == "AUROC" and result["score"] == 1
    fit, predict = evaluate.call_args_list
    assert fit.args[3] == 900 and [dst for _, dst in fit.args[2]] == ["/train.csv"]
    config = json.loads(fit.args[1][-1])
    assert config["seed"] == 1 and config["binary"] is True
    assert fit.kwargs["resources"] == {"cpus": 4, "memory": "8g", "pids_limit": 256, "shm_size": "512m"}
    assert predict.args[3] == 120 and [dst for _, dst in predict.args[2]] == ["/val_features.csv", "/model"]
    assert "predict_proba" in predict.args[1][1] and "writable" not in predict.kwargs
    args = design_suite.docker_args("fixture", "fixture", resources=predict.kwargs["resources"])
    assert args[args.index("--network") + 1] == "none"


def score_probe(*_args, folds, seen):
    seen.append(list(folds))
    if len(seen) <= 2:
        raise design_suite.CandidateExecutionError("candidate runtime")
    if len(seen) == 3:
        raise ValueError("invalid score")
    return {"score": .75, "normalized_score": .75, "recipe": RECIPE}


def test_admet_counts_only_one_crash_repair_and_cli_smoke(monkeypatch):
    task = Task("half_life_obach", "fixture")
    agent, prompts = make_agent(lambda n: SyntaxError() if n == 1 else RECIPE)
    seen = []
    monkeypatch.setattr(admet, "prepare_admet_folds", lambda *args: dict.fromkeys(args[-1]))
    monkeypatch.setattr(admet, "score_admet", partial(score_probe, seen=seen))
    _, verdict = asyncio.run(admet.admet(agent, SimpleNamespace(config={}), task))
    assert verdict["budget"] == verdict["oracle_calls"] == 20
    counts = [len(slot["attempts"]) for slot in verdict["trial_receipts"]]
    assert counts[:3] == [2, 2, 1] and counts[3:] == [1] * 17
    assert "Repair" in prompts[1] and "Repair" in prompts[3]
    assert seen == [[1, 2, 3, 4, 5]] * 21
    assert verdict["normalized_score"] == .75
    monkeypatch.setattr(sys, "argv", ["flashrsi", "run", "--smoke"])
    assert admet.admet_seeds({}) == (1, 2, 3, 4, 5)
    assert admet.admet_seeds({"smoke": True, "smoke_seeds": [2]}) == (2,)
    assert design_suite.oracle_budget({"smoke": True, "oracle_budget": 2}, 20, 20) == 2
    assert admet.admet_seeds({"smoke": True}) == (1,)


async def solver_probe(_agent, _prompt, workspace, *, image, resources, called):
    called["resources"] = resources
    (workspace / "candidate.json").write_text("{}")
    return Trajectory(), "solver"


def test_engdesign_defaults_and_overrides_reach_solver_and_grader(monkeypatch, tmp_path):
    task = Task("fixture", "produce JSON")
    manifest = {"commit": "fixture", "files": {"fixture": {}}}
    benchmark = SimpleNamespace(config={}, directory=lambda _: tmp_path, root=tmp_path, manifest_data=manifest)
    monkeypatch.setenv("FLASHRSI_ENG_IMAGE", "fixture")
    monkeypatch.setattr(engdesign.subprocess, "check_output", lambda *args: b"")
    monkeypatch.setattr(engdesign.subprocess, "run", lambda *args, **kwargs: None)
    called = {}
    monkeypatch.setattr(design_suite, "solver", partial(solver_probe, called=called))
    grader = Mock(return_value=({"score": .5}, "grader"))
    monkeypatch.setattr(engdesign, "docker_json", grader)
    _, result = asyncio.run(engdesign.engdesign(None, benchmark, task))
    assert result["score"] == .5 and grader.call_args.args[3] == 420
    assert called["resources"] == design_suite.ENG_RESOURCES
    benchmark.config = {"grader_timeout": 500, "solver_resources": {"cpus": 5}}
    asyncio.run(engdesign.engdesign(None, benchmark, task))
    assert grader.call_args.args[3] == 500 and called["resources"] == {**design_suite.ENG_RESOURCES, "cpus": 5}
    arguments = design_suite.docker_args("fixture", "fixture", resources=design_suite.ENG_RESOURCES)
    for option, expected in (("--cpus", "4"), ("--memory", "8g"), ("--pids-limit", "256"), ("--shm-size", "1g")):
        assert arguments[arguments.index(option) + 1] == expected


def open_frontier(tmp_path, configs=()):
    checkout = tmp_path / "Frontier-Engineering"
    specs = {
        "cpu": {"path": "Open/CPU", "split": "train", "task_spec": "Open CPU task"},
        "summit": {"path": "Open/Summit", "split": "dev", "task_spec": "Open research task"},
        "gpu": {"path": "Open/GPU", "split": "val", "task_spec": "Open GPU task"},
        "closed": {"path": "Closed/Test", "split": "sealed"},
    }
    frozen = {"counts": {"cpu": 42, "gpu_queue": 5}, "tasks": {"closed": specs["closed"]}}
    write_json(tmp_path / "frontier_manifest.json", frozen)
    for key in ("cpu", "summit", "gpu"):
        (checkout / "benchmarks" / specs[key]["path"]).mkdir(parents=True)
    batch = checkout / "frontier_eval/conf/batch/v1.yaml"
    batch.parent.mkdir(parents=True)
    batch.write_text("""tasks:
  - label: Open/CPU
    overrides: [task.runtime.env_name=frontier-v1-main]
  - label: Open/Summit
    overrides: [task.runtime.python_path=uv-env:frontier-v1-summit, algorithm.oe.evaluator.timeout=600]
""")
    (checkout / "benchmarks/Open/GPU/v1.yaml").write_text("env: kernel\ntimeout_s: 1200\n")
    write_json(tmp_path / "open.json", {"tasks": specs})
    config = {"name": "frontier_eng", "root": tmp_path, "mode": "real_tasks", "frontier_manifest": "open.json",
              "run_id": "fixture", "oracle_ledger": tmp_path / "ledger.sqlite", **dict(configs)}
    return benchmarks.FrontierEng(config), specs


def test_frontier_metadata_routing_and_sealed_alias(tmp_path):
    benchmark, specs = open_frontier(tmp_path)
    splits = benchmark.load()
    assert not splits["test"]
    expected = {"cpu": ("frontier-v1-main", "cpu", 300), "summit": ("frontier-v1-summit", "cpu", 600),
                "gpu": ("frontier-v1-kernel", "gpu", 1200)}
    for task in (*splits["train"], *splits["dev"]):
        assert tuple(task.metadata[key] for key in ("runtime", "queue", "timeout_s")) == expected[task.task_id]
    with pytest.raises(PermissionError):
        benchmark.task("closed", "dev")
    write_json(tmp_path / "open.json", {"tasks": {"alias": {**specs["closed"], "split": "train"}}})
    with pytest.raises(PermissionError, match="intersects a sealed"):
        benchmarks.FrontierEng(benchmark.config).load()
    with pytest.raises(PermissionError, match="explicit open"):
        benchmarks.FrontierEng({**benchmark.config, "frontier_manifest": None}).load()
    synthetic = benchmarks.FrontierEng({"name": "frontier_eng", "root": str(tmp_path)})
    assert synthetic.load()["dev"][0].task_id == "dev_water_allocate"


def test_frontier_atomic_budget_and_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = frontier_eng.FrontierLedger(path, "same-run", "same-task", 100)
    for _ in range(20):
        assert ledger.reserve("program") is not None
    with ThreadPoolExecutor(max_workers=12) as executor:
        slots = list(executor.map(ledger.reserve, repeat("program", 100)))
    assert sorted(slot for slot in slots if slot is not None) == list(range(21, 101))
    restarted = frontier_eng.FrontierLedger(path, "same-run", "same-task", 100)
    assert len(restarted.history()) == 100 and restarted.reserve("restart") is None
    assert frontier_eng.FrontierLedger(path, "new-run", "same-task", 100).reserve("x") == 1
    assert frontier_eng.FrontierLedger(path, "same-run", "new-task", 100).reserve("x") == 1
    with pytest.raises(ValueError, match="changed"):
        frontier_eng.FrontierLedger(path, "same-run", "same-task", 99)


def test_frontier_resume_and_prompt(tmp_path, monkeypatch):
    benchmark, _ = open_frontier(tmp_path, {"oracle_budget": 3})
    task = benchmark.load()["train"][0]
    agent, prompts = make_agent({"code": "x = 1"})
    ledger = frontier_eng.FrontierLedger(benchmark.config["oracle_ledger"], "fixture", "cpu", 3)
    ledger.finish(ledger.reserve("x = 0"), frontier_eng.frontier_result({"combined_score": .8}, {}))
    assert ledger.reserve("x = -1") == 2
    evaluate = Mock(return_value={"combined_score": .6, "valid": True})
    monkeypatch.setattr(frontier_eng, "execute_frontier", evaluate)
    _, result = asyncio.run(frontier_eng.frontier(agent, benchmark, task))
    evaluate.assert_called_once_with(benchmark, task, "x = 1")
    assert result["oracle_calls"] == 3
    assert "x = 0" in prompts[0] and "interrupted after reservation" in prompts[0]
    asyncio.run(frontier_eng.frontier(agent, benchmark, task))
    assert len(prompts) == 1
    prompt = design_suite.search_prompt("task" * 100_000, "parent" * 100_000, {"score": .1}, ["recent" * 100_000])
    assert len(prompt) <= 96_000 and all(needle in prompt for needle in ("tasktask", "parentparent", "recentrecent"))


@pytest.mark.parametrize("name,loader,assets,answer", [
    ("engdesign_open", benchmarks.EngDesignOpen, "engdesign_manifest.json", {"Kp": 1, "Ki": 100, "Kd": 1}),
    ("frontier_eng", benchmarks.FrontierEng, "frontier_manifest.json", {"code": frontier_eng.SEED}),
])
def test_staged_design_rollout(name, loader, assets, answer):
    if not (STAGED / assets).is_file() or not shutil.which("docker"):
        pytest.skip("staged design assets and Docker required")
    task_id = "XG_13" if loader is benchmarks.EngDesignOpen else "dev_water_allocate"
    benchmark = loader({"name": name, "root": STAGED, "task_ids": [task_id], "smoke": True})
    splits = benchmark.load()
    agent, _ = make_agent(answer)
    design_suite.bind(agent, benchmark)
    graded = benchmark.grade(asyncio.run(agent.rollout(splits["train"] + splits["dev"])))
    event = graded[0].messages[-1]
    assert len(graded) == 1 and event["type"] == "SuiteVerifier" and not splits["test"]
    assert 0 <= graded[0].metrics["reward"] <= 1
    assert "verifier_container" in event if loader is benchmarks.EngDesignOpen else event["rejected_over_budget"]


def test_frontier_live_cpu_evaluate_py(tmp_path):
    image = design_suite.engineering_image()
    if not shutil.which("docker") or subprocess.call(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL):
        pytest.skip("staged engineering image required")
    benchmark, _ = open_frontier(tmp_path, {"frontier_images": {"cpu": image}})
    path = tmp_path / "Frontier-Engineering/benchmarks/Open/CPU"
    (path / "evaluate.py").write_text("""import json, runpy, sys
data = runpy.run_path(sys.argv[1])
with open(sys.argv[3], 'w') as stream:
    json.dump({"combined_score": data["score"], "valid": 1}, stream)
""")
    task = benchmark.load()["train"][0]
    assert frontier_eng.execute_frontier(benchmark, task, "score = 0.75") == {"combined_score": .75, "valid": 1}


def frontier_rpc_probe(endpoint, route, payload, timeout, _token, *, calls):
    calls.append((endpoint, route, payload, timeout))
    return {**payload, "backend": "cuda", "result": {"combined_score": .9, "valid": True}}


def test_frontier_gpu_dispatch_and_missing_worker(tmp_path, monkeypatch):
    benchmark, _ = open_frontier(tmp_path, {"oracle_budget": 1})
    task = benchmark.load()["dev"][-1]
    agent, prompts = make_agent({"code": "x = 1"})
    monkeypatch.delenv("FLASHRSI_GPU_QUEUE", raising=False)
    with pytest.raises(InfraError, match="GPU worker URL"):
        asyncio.run(frontier_eng.frontier(agent, benchmark, task))
    assert not prompts
    benchmark.config["gpu_queue"] = "http://127.0.0.1:1"
    rpc_calls = []
    monkeypatch.setattr(frontier_eng, "worker_rpc", partial(frontier_rpc_probe, calls=rpc_calls))
    _, result = asyncio.run(frontier_eng.frontier(agent, benchmark, task))
    assert result["oracle_calls"] == 1
    assert rpc_calls[0][1] == "/v1/frontier-jobs" and rpc_calls[0][2]["timeout_s"] == 1200
    assert rpc_calls[0][3] == 1230


@pytest.fixture
def kernel_assets(tmp_path):
    directory = tmp_path / "KernelBench/KernelBench"
    train = [f"level1/train_{i}.py" for i in range(30)]
    val = [f"level1/val_{i}.py" for i in range(11)]
    sealed = [f"level1/sealed_{i}.py" for i in range(24)]
    (directory / "level1").mkdir(parents=True)
    for identifier in train + val:
        (directory / identifier).write_text(REFERENCE)
    write_json(tmp_path / "KernelBench/manifest.json", {
        "splits": {"train": train, "val": val, "test": sealed},
        "files": dict.fromkeys(train + val, hashlib.sha256(REFERENCE.encode()).hexdigest()),
    })
    return tmp_path, train[0], sealed[0]


def test_kernel_auto_and_budget(kernel_assets):
    root, _, sealed = kernel_assets
    benchmark = benchmarks.KernelBench({"name": "kernelbench", "root": str(root), "split": "auto"})
    splits = benchmark.load()
    assert {key: len(tasks) for key, tasks in splits.items()} == {"train": 30, "dev": 11, "test": 0}
    with pytest.raises(PermissionError):
        benchmark.reference(sealed)
    with pytest.raises(PermissionError):
        benchmarks.KernelBench({**benchmark.config, "task_ids": [sealed]}).load()
    if (STAGED / "KernelBench/manifest.json").is_file():
        staged = benchmarks.KernelBench({"name": "kernelbench", "root": STAGED}).load()
        assert {role: len(tasks) for role, tasks in staged.items()} == {"train": 30, "dev": 11, "test": 0}
    assert design_suite.oracle_budget({}, benchmarks.KERNEL_ORACLE_BUDGET, 100) == 10
    assert design_suite.oracle_budget({"smoke": True, "smoke_oracle_budget": 2}, benchmarks.KERNEL_ORACLE_BUDGET, 100) == 2


@pytest.mark.parametrize("source", [
    "import torch\nclass ModelNew(torch.nn.Linear): pass",
    "import torch\nx = torch.cuda.Event",
    "import torch as t\ndef f(x): return t.matmul(x, x)",
    "cache = {}\ndef f(x): return x",
    "class ModelNew:\n def forward(self, x): self.saved = x; return x",
    "class ModelNew:\n def forward(self, x): return self.saved.copy_(x)",
])
def test_kernel_anti_hacking(source):
    assert kernelbench.kernel_guard(source)


def test_kernel_gpu_boundaries_and_utf8():
    assert kernelbench.kernel_fast(True, 2.0) == {"fast_0": True, "fast_1": True, "fast_2": True}
    assert kernelbench.kernel_fast(True, 1.0) == {"fast_0": True, "fast_1": False, "fast_2": False}
    assert all(not value for value in kernelbench.kernel_fast(False, 100).values())
    assert len(design_suite.tail_bytes("φ" * 6000, 8192).encode()) == 8192
    assert len(design_suite.tail_bytes("界" * 6000, 4096).encode()) <= 4096


@pytest.fixture
def live_worker(kernel_assets, tmp_path):
    root, identifier, sealed = kernel_assets
    script = Path(daemon.__file__)
    token = secrets.token_urlsafe(24)
    command = [sys.executable, str(script), "--dry-run-cpu", "--port", "0", "--device", "GPU-test",
               "--auth-token", token, "--asset-root", str(root), "--state-dir", str(tmp_path / "worker")]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            assert select.select([process.stdout], [], [], 15)[0], "GPU test worker failed to become ready"
            match = re.search(r":(\d+) backend=dry_run_cpu", process.stdout.readline())
            assert match, "GPU test worker exited before listen"
            yield f"http://127.0.0.1:{match[1]}", token, identifier, sealed
        finally:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


def test_gpu_daemon_live_http_and_sealed_rejection(live_worker, monkeypatch):
    endpoint, token, identifier, sealed = live_worker
    client = kernelbench.GPUTimingClient(endpoint, "GPU-test", [identifier, sealed], token=token,
                                    dtype="bf16", require_real_cuda=False)
    args = {"task_id": identifier, "source": DRY_CANDIDATE, "snapshot_id": "snapshot", "job_id": "good"}
    received = client.submit(**args)
    assert received["backend"] == "dry_run_cpu" and received["correct"]
    assert received["correctness_trials"] == [True] * 5
    assert len(received["baseline_samples_ms"]) == len(received["candidate_samples_ms"]) == 100
    assert received["warmup"] == 3 and received["l2_flush_bytes"] == 2**28
    assert received["atol"] == received["rtol"] == 1e-2
    assert client.submit(**args) == received
    production = kernelbench.GPUTimingClient(endpoint, "GPU-test", [identifier], token=token, dtype="bf16")
    with monkeypatch.context() as context:
        # Route the signed CPU receipt through production backend validation.
        context.setattr(kernelbench, "kernel_has_triton_launch", lambda tree: True)
        with pytest.raises(InfraError, match="backend"):
            production.submit(**args)
    with pytest.raises(PermissionError):
        client.submit(**{**args, "task_id": sealed, "job_id": "sealed"})
    with pytest.raises(ValueError, match="AST"):
        client.submit(**{**args, "source": "import os", "job_id": "bad-ast"})
    with pytest.raises(InfraError):
        kernelbench.GPUTimingClient(endpoint, "GPU-test", [identifier], require_real_cuda=False).submit(**args)


@pytest.mark.parametrize("field,value", [
    ("num_correct_trials", 4), ("num_perf_trials", 99), ("warmup", 4),
    ("l2_flush_bytes", 2**28 - 1), ("atol", 0.1), ("rtol", 0.1),
    ("compile_timeout_s", 181), ("run_timeout_s", 181), ("static_ast_pass", False),
    ("backend", "host"), ("candidate_sha256", "tampered"), ("correctness_seeds", [1] * 5),
    ("correctness_trials", [True] * 4), ("speedup", 10000), ("score", -1),
    ("baseline_samples_ms", [float("nan")] * 100), ("fast_0", False),
    ("compile_stderr", "x" * 8193), ("timing_summary", "x" * 4097), ("fast_1", None), ("fast_2", None),
])
def test_gpu_rejects_tampered_receipts(field, value):
    payload = dict.fromkeys(kernelbench.GPU_BINDINGS, "fixture")
    payload["dtype"] = "fp32"
    receipt = {
        **payload, "backend": "dry_run_cpu", "hardware": {}, "static_ast_pass": True,
        "num_correct_trials": 5, "num_perf_trials": 100, "warmup": 3, "l2_flush_bytes": 2**28,
        "compile_timeout_s": 180, "run_timeout_s": 180, "atol": 1e-4, "rtol": 1e-4,
        "compile_stderr": "", "timing_summary": "{}", "correctness_seeds": [1, 2, 3, 4, 5],
        "correctness_trials": [True] * 5, "correct": True, "compiled": True, "failure": None,
        "baseline_ms": 2, "candidate_ms": 1, "baseline_samples_ms": [2] * 100, "candidate_samples_ms": [1] * 100,
        "speedup": 2, "score": 2 / 3, "fast_0": True, "fast_1": True, "fast_2": True,
    }
    assert kernelbench.validate_kernel_receipt(receipt, payload, require_real_cuda=False) is receipt
    with pytest.raises(InfraError):
        kernelbench.validate_kernel_receipt({**receipt, field: value}, payload, require_real_cuda=False)


class SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        time.sleep(.25)


def test_real_rpc_deadline():
    with ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler) as server:
        server.daemon_threads = True
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(InfraError, match="deadline"):
                kernelbench.worker_rpc(f"http://127.0.0.1:{server.server_port}", "/", {"job_id": "x"}, .02)
        finally:
            server.shutdown()
            thread.join()

def kernel_submit_probe(*, calls, **job):
    calls.append(job)
    return {"job_id": job["job_id"], "score": .8, "correct": True,
            "compile_stderr": "a" * 8182 + "compile-end", "timing_summary": "correct and fast"}


def test_kernel_feedback(monkeypatch):
    agent, prompts = make_agent({"code": DRY_CANDIDATE})
    calls = []
    monkeypatch.setattr(kernelbench.GPUTimingClient, "submit", partial(kernel_submit_probe, calls=calls))
    fake = SimpleNamespace(config={"gpu_queue": "http://example.invalid", "device_uuid": "GPU-fixture"},
                           _tasks={"fixture": None})
    _, report = asyncio.run(kernelbench.kernel(agent, fake, Task("fixture", "Implement fixture")))
    assert report["budget"] == report["gpu_calls"] == len(calls) == benchmarks.KERNEL_ORACLE_BUDGET
    assert len({call["job_id"] for call in calls}) == benchmarks.KERNEL_ORACLE_BUDGET
    assert "compile-end" in prompts[1] and "correct and fast" in prompts[1]


def test_production_triton_guard():
    source = """import torch
import triton
import triton.language as tl
@triton.jit
def add(X, Y, N: tl.constexpr):
    offset = tl.arange(0, N)
    tl.store(Y + offset, tl.load(X + offset) + 1)
class ModelNew(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        add[(1,)](x, out, 256)
        return out
"""
    assert not kernelbench.kernel_guard(source)
    assert kernelbench.kernel_has_triton_launch(ast.parse(source))
    assert kernelbench.kernel_guard("import torch\ndef f():\n nn = torch.nn\n return nn.Linear(2, 2)")
    assert not kernelbench.kernel_has_triton_launch(ast.parse(DRY_CANDIDATE))


def test_gpu_compile_failure_and_incorrect_live_receipts(live_worker):
    endpoint, token, identifier, _ = live_worker
    client = kernelbench.GPUTimingClient(endpoint, "GPU-test", [identifier], token=token,
                                    dtype="fp16", require_real_cuda=False)
    common = {"task_id": identifier, "snapshot_id": "snapshot"}
    code = 'class ModelNew:\n def forward(self, x):\n  raise RuntimeError("' + "φ" * 10000 + '")'
    failed = client.submit(**common, source=code, job_id="compile-failure")
    assert failed["failure"] == "compile_error" and not failed["compiled"] and not failed["fast_0"]
    assert len(failed["compile_stderr"].encode()) >= 8190
    assert len(failed["compile_stderr"].encode()) <= 8192
    assert failed["compile_stderr"].endswith("φ\n")
    assert json.loads(failed["timing_summary"])["failure"] == "compile_error"
    wrong = client.submit(**common, source=DRY_CANDIDATE.replace("x * 2", "x + 100"), job_id="wrong")
    assert wrong["failure"] == "incorrect" and wrong["compiled"]
    assert wrong["correctness_trials"] == [False] * 5
    assert not wrong["baseline_samples_ms"] and wrong["score"] == 0.0


def take_lease(path):
    with gpu_worker.gpu_lease(path, .05):
        return True


def test_gpu_worker_lease_serializes_jobs(tmp_path):
    path = tmp_path / "same-device.lock"
    with ThreadPoolExecutor(max_workers=1) as pool:
        with gpu_worker.gpu_lease(path, .1):
            with pytest.raises(gpu_worker.ProtocolError, match="queue wait"):
                pool.submit(take_lease, path).result(timeout=2)
        assert pool.submit(take_lease, path).result(timeout=2)


def test_gpu_worker_reaps_phase_timeout(tmp_path):
    worker = object.__new__(daemon.Worker)
    worker.dry = True
    (tmp_path / "reference.py").write_text(REFERENCE)
    (tmp_path / "candidate.py").write_text("class ModelNew:\n def forward(self, x):\n  while True: pass\n")
    cfg = {"backend": "dry_run_cpu", "seeds": [1, 2, 3, 4, 5], "dtype": "fp32", "tolerance": 1e-4}
    for phase in ("compile", "run"):
        receipt, _, failure = worker.run_phase(phase, tmp_path, cfg, .15)
        assert receipt is None and failure == phase + "_timeout"
