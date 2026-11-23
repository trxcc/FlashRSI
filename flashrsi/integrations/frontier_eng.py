"""Run Frontier-Eng program search and evaluate approved open tasks.

The host owns oracle ledgers and launches isolated CPU or GPU evaluators.
FLASHRSI_ENG_IMAGE selects the solver and synthetic evaluator image.
FLASHRSI_GPU_QUEUE and FLASHRSI_GPU_AUTH_TOKEN configure remote GPU dispatch.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flashrsi._records import _redact, _secrets
from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.benchmarks._suite import confined, read_json
from flashrsi.benchmarks.design_suite import FRONTIER_ORACLE_BUDGET, FrontierEng
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.integrations.design_suite import (
    ENG_RESOURCES,
    ENV,
    Verdict,
    cleanup,
    cpu_sandbox_command,
    docker_args,
    docker_json,
    engineering_image,
    finite_number,
    oracle_budget,
    propose_code,
    python_environment_mounts,
    read_sandbox_metrics,
    run_sandbox,
    scratch,
    search_prompt,
    sha,
    stage_readonly_assets,
    tail_bytes,
)
from flashrsi.integrations.kernelbench import worker_rpc, worker_token

SEED = "def solve():\n    return (0.25, 0.25, 0.25, 0.25)\n"
SYNTH_RUNNER = """
import json, os, runpy
assert not os.path.exists('/evaluate.py') and not os.path.exists('/private')
assert not os.environ.get('DEEPSEEK_API_KEY')
assert os.listdir('/public') == ['candidate.py']
print(json.dumps(runpy.run_path('/public/candidate.py')['solve'](),allow_nan=False))
"""


def safe_program(code: str) -> tuple[float, ...] | list[float]:
    if not isinstance(code, str) or len(code) > 1000:
        raise ValueError("invalid candidate length")
    body = ast.parse(code).body
    if len(body) != 1 or type(body[0]) is not ast.FunctionDef:
        raise ValueError("single function required")
    function = body[0]
    if (function.name != "solve" or ast.unparse(function.args) or function.decorator_list
            or function.returns or getattr(function, "type_params", ()) or len(function.body) != 1
            or type(function.body[0]) is not ast.Return):
        raise ValueError("unexpected function surface")
    vector = ast.literal_eval(function.body[0].value)
    if type(vector) not in (tuple, list) or len(vector) != 4 or not all(map(finite_number, vector)):
        raise ValueError("literal vector required")
    return vector


class OracleProxy:
    """Local CLI grader and durable, at-most-once submitted candidate ledger."""

    def __init__(self, ledger: Path, grader: Path, budget: int) -> None:
        self.ledger = ledger
        self.grader = grader
        self.budget = budget
        with sqlite3.connect(ledger) as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY, digest TEXT, result TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS budget(n INTEGER NOT NULL CHECK(n>0))")
            if db.execute("SELECT COUNT(*) FROM budget").fetchone()[0] == 0:
                db.execute("INSERT INTO budget VALUES(?)", (budget,))
            assert db.execute("SELECT n FROM budget").fetchone()[0] == budget

    def submit_candidate(self, code: str, *, submission_id: str) -> dict[str, Any]:
        digest = sha(code.encode())
        with sqlite3.connect(self.ledger) as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT digest,result FROM calls WHERE id=?", (submission_id,)).fetchone()
            if old:
                if old[0] != digest or old[1] is None:
                    raise ValueError("changed or incomplete replay")
                return json.loads(old[1])
            count = db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            if count >= self.budget:
                raise RuntimeError("oracle budget exhausted")
            db.execute("INSERT INTO calls VALUES(?,?,NULL)", (submission_id, digest))
        try:
            safe_program(code)
            with tempfile.TemporaryDirectory(dir=self.ledger.parent) as folder:
                candidate = Path(folder) / "candidate.py"
                candidate.write_text(code)
                os.chmod(candidate, 0o444)
                vector, clean = docker_json(engineering_image(), ["-c", SYNTH_RUNNER], ((Path(folder), "/public"),), timeout=35)
            graded = subprocess.run(
                [sys.executable, "-I", "-B", str(self.grader)], input=json.dumps(vector),
                text=True, capture_output=True, timeout=15, check=True, env=ENV,
            )
            result = {**json.loads(graded.stdout), "stderr_summary": "", "container": clean}
        except (ValueError, SyntaxError) as exc:
            result = {"valid": False, "primary_score": 0.0, "metrics": {}, "stderr_summary": type(exc).__name__}
        result.update(candidate_sha256=digest, call_count=count + 1)
        with sqlite3.connect(self.ledger) as db:
            db.execute("UPDATE calls SET result=? WHERE id=?", (json.dumps(result, allow_nan=False), submission_id))
        return result

    def count(self) -> int:
        with sqlite3.connect(self.ledger) as db:
            return db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]


def setup_oracle(run_dir: Path, budget: int = 2) -> OracleProxy:
    private = run_dir / "private_scorer"
    private.mkdir(mode=0o700)
    grader = private / "evaluate.py"
    grader.write_text("""import json,math,sys
x=json.load(sys.stdin)
valid=len(x)==4 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in x) and abs(sum(x)-1)<1e-6
targets=((.415,.285,.175,.125),(.43,.275,.18,.115),(.418,.29,.177,.115))
loss=sum(sum((a-b)**2 for a,b in zip(x,t)) for t in targets)/len(targets) if valid else None
print(json.dumps({'valid':valid,'primary_score':1/(1+loss) if valid else 0.0,'metrics':{'mean_squared_tracking_loss':loss} if valid else {}}))
""")
    grader.chmod(0o600)
    return OracleProxy(run_dir / "oracle.sqlite", grader, budget)


Attempt = tuple[int, str, dict[str, Any] | None]


class FrontierLedger:
    """Commit a reservation before dispatch and never re-dispatch an uncertain call."""

    def __init__(self, path: Path | str, run_id: str, task_id: str, budget: int) -> None:
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 256:
            raise ValueError("real Frontier requires a stable benchmark.run_id")
        if type(budget) is not int or not 1 <= budget <= FRONTIER_ORACLE_BUDGET:
            raise ValueError(f"Frontier ledger budget must be an integer from 1 to {FRONTIER_ORACLE_BUDGET}")
        self.path, self.key, self.budget = Path(path), (run_id, task_id), budget
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS frontier_runs (
                    run_id TEXT NOT NULL, task_id TEXT NOT NULL, budget INTEGER NOT NULL,
                    PRIMARY KEY (run_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS frontier_attempts (
                    run_id TEXT NOT NULL, task_id TEXT NOT NULL, slot INTEGER NOT NULL,
                    source TEXT NOT NULL, result TEXT,
                    PRIMARY KEY (run_id, task_id, slot)
                );
            """)
            db.execute("INSERT OR IGNORE INTO frontier_runs VALUES (?, ?, ?)", (*self.key, budget))
            recorded = db.execute("SELECT budget FROM frontier_runs WHERE run_id=? AND task_id=?", self.key).fetchone()
            if recorded[0] != budget:
                raise ValueError("Frontier run budget changed across restart")

    def history(self) -> list[Attempt]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT slot, source, result FROM frontier_attempts WHERE run_id=? AND task_id=? ORDER BY slot", self.key
            ).fetchall()
        return [(slot, source, json.loads(result) if result else None) for slot, source, result in rows]

    def reserve(self, source: str) -> int | None:
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT COUNT(*) FROM frontier_attempts WHERE run_id=? AND task_id=?", self.key).fetchone()
            if row[0] >= self.budget:
                return None
            slot = row[0] + 1
            db.execute("INSERT INTO frontier_attempts VALUES (?, ?, ?, ?, NULL)", (*self.key, slot, source))
            return slot

    def finish(self, slot: int, result: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as db:
            changed = db.execute(
                "UPDATE frontier_attempts SET result=? WHERE run_id=? AND task_id=? AND slot=? AND result IS NULL",
                (json.dumps(result, allow_nan=False), *self.key, slot),
            ).rowcount
            if changed != 1:
                raise InfraError("Frontier ledger reservation is missing or already finalized")


def frontier_incumbent(history: Sequence[Attempt]) -> tuple[str, dict[str, Any] | None]:
    return max(
        ((source, receipt) for _, source, receipt in history if receipt and receipt["valid"]),
        key=lambda item: item[1]["rank_score"],
        default=("", None),
    )


def frontier_result(report: object, settings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise InfraError("Frontier evaluator returned malformed metrics")
    metrics = report.get("metrics", report)
    if not isinstance(metrics, dict):
        raise InfraError("Frontier evaluator returned malformed metrics")
    validity = report.get("valid", metrics.get("valid", True))
    if type(validity) not in (bool, int, float) or validity not in (0, 1):
        raise InfraError("Frontier evaluator returned an invalid feasibility flag")
    metric_key = settings.get("metric", "combined_score")
    direction = settings.get("direction", "maximize")
    if direction not in ("maximize", "minimize"):
        raise ValueError("Frontier metric direction must be maximize or minimize")
    raw = metrics.get(metric_key, report.get("primary_score"))
    if validity and not finite_number(raw):
        raise InfraError("Frontier evaluator omitted its finite primary metric")
    rank = (float(raw) if direction == "maximize" else -float(raw)) if validity else 0.0
    score = 0.5 + math.atan(rank) / math.pi if validity else 0.0
    return {
        "valid": bool(validity), "primary_score": score, "rank_score": rank,
        "raw_metric": float(raw) if finite_number(raw) else None, "direction": direction,
        "metrics": {k: v for k, v in list(metrics.items())[:40] if finite_number(v)},
        "failure": tail_bytes(str(report.get("failure", "")), 4096),
    }


def frontier_script(directory: Path, settings: Mapping[str, Any], guest: str, interpreter: str) -> tuple[str, Path]:
    metadata = directory / "frontier_eval"
    command = settings.get("eval_command")
    if command is None and (metadata / "eval_command.txt").is_file():
        command = confined(directory, "frontier_eval/eval_command.txt").read_text().strip()
    if command is None:
        evaluator = next(
            (file for file in ("verification/evaluate.py", "verification/evaluator.py", "evaluate.py") if (directory / file).is_file()),
            None,
        )
        if evaluator is None:
            raise InfraError("Frontier task has no evaluate.py or pinned evaluation command")
        command = "{python} {benchmark_source}/" + evaluator + " {candidate} --metrics-out {benchmark}/metrics.json"
    path = settings.get("candidate_destination")
    for name in ("candidate_destination", "initial_program"):
        if path is None and (metadata / f"{name}.txt").is_file():
            path = confined(directory, f"frontier_eval/{name}.txt").read_text().strip()
    path = Path(path or "candidate.py")
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("invalid Frontier candidate destination")
    variables = {
        "python": interpreter, "candidate": "/work/" + path.as_posix(),
        "benchmark": "/work", "sandbox": "/work", "repo_root": "/frontier",
        "benchmark_source": guest, "benchmark_id": settings["path"],
    }
    formatted = {key: shlex.quote(value) for key, value in variables.items()}
    formatted.update({key + "_raw": value for key, value in variables.items()})
    try:
        command = command.format_map(formatted)
    except (ValueError, KeyError):
        raise InfraError("unsupported placeholder in Frontier evaluation command") from None
    return command, path


def frontier_diagnostic(work: Path, stderr: str) -> str:
    artifacts = work / "artifacts.json"
    if artifacts.is_symlink() or not artifacts.is_file() or artifacts.stat().st_size > 512 * 1024:
        return stderr
    try:
        value = read_json(artifacts)
    except ValueError:
        return stderr
    if not isinstance(value, dict):
        return stderr
    for key in ("failure_summary", "error_message", "candidate_validation"):
        if value.get(key):
            diagnostic = value[key] if isinstance(value[key], str) else json.dumps(value[key], ensure_ascii=False)
            return tail_bytes(diagnostic, 4096)
    return stderr


def _evaluation_mounts(benchmark: FrontierEng, directory: Path, settings: Mapping[str, Any], guest: str) -> list[tuple[Path, str]]:
    repository = benchmark.repository
    mounts = [(directory, guest)]
    if (repository / "frontier_eval").is_dir():
        mounts.append((confined(repository, "frontier_eval"), "/frontier/frontier_eval"))
    shared = "_shared/candidate_sandbox.py"
    if (repository / "benchmarks" / shared).is_file():
        mounts.append((benchmark._checked_directory(shared), "/frontier/benchmarks/" + shared))
    for relative in settings.get("shared_paths", ()):
        if Path(relative).parts[:1] == ("_shared",):
            raise PermissionError("Frontier shared evaluator files require individual review")
        mounts.append((benchmark._checked_directory(relative), "/frontier/benchmarks/" + relative))
    return mounts


def _evaluation_cwd(directory: Path, settings: Mapping[str, Any]) -> Path:
    cwd = settings.get("eval_cwd")
    metadata = "frontier_eval/eval_cwd.txt"
    if cwd is None and (directory / metadata).is_file():
        cwd = confined(directory, metadata).read_text().strip()
    cwd = Path(cwd or ".")
    if cwd.is_absolute() or ".." in cwd.parts:
        raise ValueError("invalid Frontier evaluator working directory")
    return cwd


def _evaluate(args: list[str], work: Path, timeout: int) -> dict[str, Any]:
    try:
        status, diagnostic = run_sandbox(args, timeout)
    except subprocess.TimeoutExpired:
        return {"valid": False, "failure": f"evaluate.py exceeded {timeout}s"}
    if diagnostic.startswith("bwrap:"):
        raise InfraError("Frontier CPU sandbox could not establish isolation")
    if status in (125, 126, 127, 137):
        raise InfraError(f"Frontier isolated evaluator exited with infrastructure status {status}")
    if status:
        return {"valid": False, "failure": diagnostic or f"evaluator exit {status}"}
    report = read_sandbox_metrics(work / "metrics.json")
    if report.get("valid") == 0:
        report["failure"] = frontier_diagnostic(work, diagnostic)
        failure = report["failure"].lower()
        if "candidate isolation requires bubblewrap" in failure or "no permissions to create a new namespace" in failure:
            raise InfraError("Frontier runtime cannot establish the upstream candidate sandbox")
    return report


def execute_frontier(benchmark: FrontierEng, task: Task, source: str, *, gpu_device: str | None = None) -> dict[str, Any]:
    directory, settings = benchmark.execution_assets(task)
    runtime, timeout = task.metadata["runtime"], task.metadata["timeout_s"]
    routes = benchmark.config.get("frontier_images", {})
    image = settings.get("image") or routes.get(runtime) or routes.get(task.metadata["queue"])
    guest = "/frontier/benchmarks/" + settings["path"]
    mounts = _evaluation_mounts(benchmark, directory, settings, guest)
    interpreter = "python"
    runtime_env = {
        "FRONTIER_EVAL_EVALUATOR_TIMEOUT_S": str(timeout),
        "FRONTIER_ENGINEERING_ROOT": "/frontier",
        "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    }
    if not image:
        if gpu_device:
            raise InfraError("Frontier GPU evaluation requires a configured offline image")
        venv = benchmark.repository / ".venvs" / runtime
        python = venv / "bin/python"
        if not python.is_file():
            raise InfraError(f"missing Frontier {runtime} environment or configured container image")
        interpreter = str(python)
        mounts[:0] = python_environment_mounts(interpreter, venv)
        runtime_env["PATH"] = str(python.parent) + ":" + ENV["PATH"]
    command, destination = frontier_script(directory, settings, guest, interpreter)
    cwd = _evaluation_cwd(directory, settings)
    with scratch() as tmp:
        work = Path(tmp)
        writable = {*destination.parents, *cwd.parents, cwd}
        reserved = {destination, Path("metrics.json"), Path("artifacts.json")}
        stage_readonly_assets(directory, work, Path(guest), writable, reserved)
        submitted = work / destination
        submitted.parent.mkdir(parents=True, exist_ok=True)
        submitted.write_text(source, encoding="utf-8")
        (work / cwd).mkdir(parents=True, exist_ok=True)
        if not image:
            args = cpu_sandbox_command(mounts, work, cwd, runtime_env)
            return _evaluate([*args, "-c", command], work, timeout)
        name = "flashrsi-frontier-" + uuid.uuid4().hex[:14]
        args = docker_args(
            image, name, mounts, resources=ENG_RESOURCES, writable=((work, "/work"),),
            gpu_device=gpu_device, workdir=str(Path("/work") / cwd),
            env=[f"{key}={value}" for key, value in runtime_env.items()], entrypoint="/bin/bash",
        )
        try:
            return _evaluate([*args, "-c", command], work, timeout)
        finally:
            cleanup(name)


async def _frontier_gpu(benchmark: FrontierEng, task: Task, source: str, endpoint: str, run_id: str, slot: int) -> dict[str, Any]:
    payload = {
        "job_id": sha(json.dumps([run_id, task.task_id, slot]).encode()),
        "task_id": task.task_id, "source": source, "candidate_sha256": sha(source.encode()),
        "timeout_s": task.metadata["timeout_s"],
    }
    response = await asyncio.to_thread(
        worker_rpc, endpoint, "/v1/frontier-jobs", payload, task.metadata["timeout_s"] + 30, worker_token(benchmark.config)
    )
    bindings = ("job_id", "task_id", "candidate_sha256", "timeout_s")
    if any(response.get(key) != payload[key] for key in bindings):
        raise InfraError("Frontier GPU worker receipt binding mismatch")
    if response.get("backend") != "cuda":
        raise InfraError("Frontier GPU worker must report CUDA execution")
    return response.get("result")


def _feedback(history: Sequence[Attempt]) -> list[str]:
    records = []
    for slot, _, receipt in history[-12:]:
        text = "interrupted after reservation"
        if receipt:
            text = json.dumps(
                {"valid": receipt["valid"], "metric": receipt.get("raw_metric"), "failure": receipt.get("failure")}, ensure_ascii=False
            )[:768]
        records.append(f"{slot}: {text}")
    return records


async def frontier_real(agent: Agent, benchmark: FrontierEng, task: Task) -> Verdict:
    budget = oracle_budget(benchmark.config, FRONTIER_ORACLE_BUDGET, FRONTIER_ORACLE_BUDGET)
    endpoint = benchmark.config.get("gpu_queue") or os.getenv("FLASHRSI_GPU_QUEUE")
    gpu = task.metadata["queue"] == "gpu"
    if gpu and not endpoint:
        raise InfraError("Frontier GPU task requires a configured GPU worker URL")
    path = (
        benchmark.config.get("oracle_ledger")
        or Path(os.getenv("XDG_STATE_HOME") or Path.home() / ".local/state") / "flashrsi/frontier-oracle.sqlite"
    )
    run_id = benchmark.config.get("run_id")
    ledger = FrontierLedger(path, run_id, task.task_id, budget)
    _, settings = benchmark.execution_assets(task)
    usage = []
    while len(history := ledger.history()) < budget:
        parent, incumbent = frontier_incumbent(history)
        prompt = search_prompt(task.question, parent, incumbent, _feedback(history))
        try:
            code, tokens = await propose_code(agent, task, prompt)
            usage.append(tokens)
        except (ValueError, SyntaxError):
            code = None
        if not isinstance(code, str) or not code.strip() or len(code.encode()) > 512 * 1024 or _redact(code, _secrets(os.environ)) != code:
            code = ""
        slot = ledger.reserve(code)
        if slot is None:
            break
        try:
            if code:
                ast.parse(code)
                if gpu:
                    report = await _frontier_gpu(benchmark, task, code, endpoint, run_id, slot)
                else:
                    report = await asyncio.to_thread(execute_frontier, benchmark, task, code)
            else:
                report = {"valid": False, "failure": "missing or invalid Python source"}
            result = frontier_result(report, settings)
        except SyntaxError:
            result = frontier_result({"valid": False, "failure": "invalid Python syntax"}, settings)
        except InfraError as exc:
            ledger.finish(slot, frontier_result({"valid": False, "failure": str(exc)}, settings))
            raise
        ledger.finish(slot, _redact(result, _secrets(os.environ)))
    history = ledger.history()
    _, best = frontier_incumbent(history)
    return Trajectory(usage={"model_calls": usage}), {
        "score": best["primary_score"] if best else 0.0, "budget": budget, "oracle_calls": len(history),
        "trial_receipts": [
            {"slot": slot, **(receipt or {"valid": False, "failure": "interrupted after reservation"})} for slot, _, receipt in history
        ],
    }


async def frontier(agent: Agent, benchmark: FrontierEng, task: Task) -> Verdict:
    if benchmark.config.get("mode", "synthetic_dev") == "real_tasks":
        return await frontier_real(agent, benchmark, task)
    budget = oracle_budget(benchmark.config, FRONTIER_ORACLE_BUDGET, FRONTIER_ORACLE_BUDGET, smoke_default=2)
    with scratch() as tmp:
        folder = Path(tmp)
        oracle = setup_oracle(folder, budget)
        parent, best = SEED, 0.0
        trials, usage = [], []
        for index in range(budget):
            prompt = task.question + f"\nPrior code:\n{parent}\nBest scalar score: {best}"
            workspace = None
            if not isinstance(agent, ProgramSearchTarget):
                workspace = folder / f"attempt-{index}"
                workspace.mkdir()
            code, tokens = await propose_code(agent, task, prompt, workspace=workspace, json_hint=True)
            usage.append(tokens)
            code = code if isinstance(code, str) else ""
            result = await asyncio.to_thread(oracle.submit_candidate, code, submission_id=f"step-{index + 1}")
            if result["valid"] and result["primary_score"] >= best:
                parent, best = code, result["primary_score"]
            trials.append(result)
        try:
            oracle.submit_candidate(SEED, submission_id="over-budget-probe")
        except RuntimeError as exc:
            if str(exc) != "oracle budget exhausted":
                raise
        else:
            raise InfraError("oracle exceeded its declared budget")
    return Trajectory(usage={"model_calls": usage}), {
        "score": best, "oracle_calls": len(trials), "budget": budget,
        "rejected_over_budget": True, "trial_receipts": trials,
    }
