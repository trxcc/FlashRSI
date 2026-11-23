"""Submit KernelBench candidates to the exclusive GPU timing worker.

The host checks candidate syntax and binds authenticated timing receipts to
the submitted job. CUDA and Triton execute in the worker's offline sandbox.
FLASHRSI_GPU_QUEUE, FLASHRSI_GPU_UUID and FLASHRSI_GPU_AUTH_TOKEN select it.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
import uuid
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import Any

from flashrsi.benchmarks.design_suite import KERNEL_ORACLE_BUDGET, KernelBench
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.integrations.design_suite import (
    CandidateExecutionError,
    Verdict,
    finite_number,
    oracle_budget,
    propose_code,
    search_prompt,
    sha,
    tail_bytes,
    worker_rpc,
)

GPU_COMPILE_TIMEOUT = 180
GPU_RUN_TIMEOUT = 180
GPU_RPC_TIMEOUT = 680
GPU_CORRECT_TRIALS = 5
GPU_PERF_TRIALS = 100
GPU_WARMUP = 3
GPU_L2_BYTES = 256 * 1024 * 1024
GPU_TOLERANCES = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 1e-2}
COMPILE_STDERR_BYTES = 8192
TIMING_SUMMARY_BYTES = 4096
GPU_BINDINGS = ("job_id", "task_id", "snapshot_id", "candidate_sha256", "device_uuid", "lease_mode", "timing_protocol", "dtype")


class KernelASTGuard(ast.NodeVisitor):
    """Restrict candidate code to a Python wrapper and custom Triton kernels."""

    BANNED = frozenset(
        """
    atexit breakpoint cache cached_property cdll classmethod compile ctypes cublas cublaslt delattr dir
    elapsed_time eval event exec getattr globals help input load_inline locals lru_cache open print setattr
    staticmethod type vars
    """.split()
    )
    IMPORTS = {"torch", "triton", "triton.language"}
    ROOT_ATTRIBUTES = {
        "torch": frozenset(
            """
        Tensor bfloat16 bool empty empty_like empty_strided float16 float32 float64 int32 int64 nn rand
        randn zeros zeros_like
        """.split()
        ),
        "torch.nn": {"Module", "Parameter"},
        "triton": {"jit", "language", "cdiv", "next_power_of_2"},
        "triton.language": frozenset(
            """
        abs arange argmax argmin associative_scan bfloat16 broadcast broadcast_to cat cdiv ceil clamp
        constexpr cos cumprod cumsum debug_barrier div_rn dot dot_scaled erf exp exp2 expand_dims float16
        float32 float64 floor fma full gather histogram int1 int16 int32 int64 int8 join load log log2
        make_block_ptr max maximum min minimum multiple_of num_programs permute pointer_type program_id
        range ravel reduce reshape rsqrt sigmoid sin softmax sort split sqrt static_assert static_range
        store sum swizzle2d trans uint16 uint32 uint64 uint8 view where xor_sum zeros zeros_like
        """.split()
        ),
    }
    STATE_WRITES = {"copy_", "set_", "resize_", "append", "extend", "update", "add", "clear", "pop"}

    def __init__(self) -> None:
        self.errors: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.attribute_bases: set[ast.expr] = set()
        self.initializing = False

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            if item.name not in self.IMPORTS:
                self.errors.add("import outside allowlist")
            self.aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.add("from-import refused")

    def root_name(self, node: ast.AST) -> str:
        base = ast.unparse(node)
        first, _, rest = base.partition(".")
        return ".".join(part for part in (self.aliases.get(first, first), rest) if part)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.lower() in self.BANNED or node.attr.startswith("__"):
            self.errors.add("disallowed attribute")
        root = self.root_name(node.value)
        if self.root_name(node) in self.ROOT_ATTRIBUTES and node not in self.attribute_bases:
            self.errors.add("module aliases outside import declarations")
        if root in self.ROOT_ATTRIBUTES and node.attr not in self.ROOT_ATTRIBUTES[root]:
            self.errors.add("operation outside wrapper allowlist")
        if node.attr in self.STATE_WRITES and any(isinstance(n, ast.Name) and n.id == "self" for n in ast.walk(node.value)):
            self.errors.add("persistent instance state/cache")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.lower() in self.BANNED or node.id.startswith("__"):
            self.errors.add("disallowed name")
        if node.id in self.aliases:
            if isinstance(node.ctx, ast.Store):
                self.errors.add("import binding mutation")
            if isinstance(node.ctx, ast.Load) and node not in self.attribute_bases:
                self.errors.add("module aliases outside import declarations")

    def visit_Global(self, node: ast.Global | ast.Nonlocal) -> None:
        self.errors.add("global mutation/state")

    visit_Nonlocal = visit_Global

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.decorator_list or node.keywords or any(ast.unparse(base) != "torch.nn.Module" for base in node.bases):
            self.errors.add("unsupported model class")
        if any(isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for child in node.body):
            self.errors.add("class state/cache")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            root = self.aliases.get(node.func.value.id)
            if root == "torch" and node.func.attr in {"rand", "randn"}:
                # Initialization randomness is permitted only in ModelNew.__init__.
                if not self.initializing:
                    self.errors.add("randomized wrapper output")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"matmul", "mm", "bmm", "einsum", "addmm", "conv2d", "linear"}:
                self.errors.add("passthrough tensor operation")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name.startswith("__") and node.name != "__init__":
            self.errors.add("unsupported special method")
        if node.args.defaults or any(node.args.kw_defaults):
            self.errors.add("default-argument state")
        if any(ast.unparse(item) != "triton.jit" for item in node.decorator_list):
            self.errors.add("unsupported decorator")
        initializing = self.initializing
        self.initializing = node.name == "__init__"
        for child in ast.walk(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                targets = child.targets if isinstance(child, (ast.Assign, ast.Delete)) else [child.target]
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        if not (self.initializing and isinstance(target.value, ast.Name) and target.value.id == "self"):
                            self.errors.add("persistent attribute mutation/cache")
                    if isinstance(target, ast.Subscript) and any(
                        isinstance(item, ast.Name) and item.id == "self" for item in ast.walk(target)
                    ):
                        self.errors.add("persistent instance state/cache")
        self.generic_visit(node)
        self.initializing = initializing

    visit_AsyncFunctionDef = visit_FunctionDef


def kernel_guard(source: str) -> list[str]:
    tree = ast.parse(source)
    guard = KernelASTGuard()
    guard.attribute_bases = {node.value for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not isinstance(node.value, ast.Constant) or not all(isinstance(target, ast.Name) for target in targets):
                guard.errors.add("module state/cache")
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                guard.errors.add("module side effect")
    if "cublas" in source.lower():
        guard.errors.add("external math library")
    guard.visit(tree)
    return sorted(guard.errors)


def kernel_has_triton_launch(tree: ast.AST) -> bool:
    kernels = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and any(ast.unparse(item) == "triton.jit" for item in node.decorator_list)
    }
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in kernels
        for node in ast.walk(tree)
    )


def worker_token(config: Mapping[str, Any]) -> str | None:
    return config.get("gpu_auth_token") or os.getenv("FLASHRSI_GPU_AUTH_TOKEN")


def kernel_fast(correct: bool, speedup: float) -> dict[str, bool]:
    return {"fast_0": correct, "fast_1": correct and speedup > 1.0, "fast_2": correct and speedup >= 2.0}


def kernel_score(correct: bool, speedup: float) -> float:
    return speedup / (1 + speedup) if correct else 0.0


def _validate_hardware(hardware: Any, device: str, cuda: bool) -> None:
    if not isinstance(hardware, dict):
        raise InfraError("invalid GPU timing receipt field hardware")
    if cuda:
        fields = ("gpu_name", "driver_version", "cuda_version", "graphics_clock_mhz", "memory_clock_mhz")
        if not all(isinstance(hardware.get(key), str) and hardware[key] for key in fields):
            raise InfraError("invalid GPU timing receipt field CUDA telemetry")
        if hardware.get("device_uuid") != device or type(hardware.get("clocks_locked")) is not bool:
            raise InfraError("invalid GPU timing receipt device or clock telemetry")


def _validate_correctness(receipt: dict[str, Any], cuda: bool) -> bool:
    seeds = receipt.get("correctness_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != GPU_CORRECT_TRIALS
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != GPU_CORRECT_TRIALS
    ):
        raise InfraError("invalid GPU timing receipt field correctness_seeds")
    compiled, correct = receipt.get("compiled"), receipt.get("correct")
    if type(compiled) is not bool or type(correct) is not bool:
        raise InfraError("invalid GPU timing receipt fields compiled/correct")
    if cuda and receipt.get("l2_flushed") is not correct:
        raise InfraError("invalid GPU timing receipt field l2_flushed")
    failure = receipt.get("failure")
    outcomes = receipt.get("correctness_trials")
    if not isinstance(outcomes, list) or any(type(outcome) is not bool for outcome in outcomes):
        raise InfraError("invalid GPU timing receipt field correctness_trials")
    if correct or failure == "incorrect":
        if not compiled or len(outcomes) != GPU_CORRECT_TRIALS or all(outcomes) != correct:
            raise InfraError("invalid GPU timing receipt correctness results")
        if failure != (None if correct else "incorrect"):
            raise InfraError("invalid GPU timing receipt field failure")
    else:
        expected = {"run_error", "run_timeout"} if compiled else {"compile_error", "compile_timeout"}
        if not isinstance(failure, str) or failure not in expected or outcomes:
            raise InfraError("invalid GPU timing receipt field failure")
    return correct


def _validate_samples(receipt: dict[str, Any], prefix: str, correct: bool) -> float:
    samples, latency = receipt.get(prefix + "_samples_ms"), receipt.get(prefix + "_ms")
    if not correct:
        if samples != [] or latency is not None:
            raise InfraError(f"invalid GPU timing receipt field {prefix} failed timing")
        return 0.0
    if (
        not isinstance(samples, list)
        or len(samples) != GPU_PERF_TRIALS
        or any(not finite_number(sample) or not 0 < sample <= GPU_RUN_TIMEOUT * 1000 for sample in samples)
    ):
        raise InfraError(f"invalid GPU timing receipt field {prefix} samples")
    mean = fmean(samples)
    if not finite_number(latency) or not math.isclose(latency, mean, rel_tol=1e-9):
        raise InfraError(f"invalid GPU timing receipt field {prefix} mean")
    return mean


def validate_kernel_receipt(receipt: Any, payload: Mapping[str, Any], *, require_real_cuda: bool = True) -> dict[str, Any]:
    if not isinstance(receipt, dict) or any(receipt.get(key) != payload[key] for key in GPU_BINDINGS):
        raise InfraError("invalid GPU timing receipt object or job binding")
    backend = receipt.get("backend")
    cuda = backend == "cuda"
    if not cuda and (require_real_cuda or backend != "dry_run_cpu"):
        raise InfraError("invalid GPU timing receipt field backend")
    protocol = {
        "num_correct_trials": GPU_CORRECT_TRIALS,
        "num_perf_trials": GPU_PERF_TRIALS,
        "warmup": GPU_WARMUP,
        "compile_timeout_s": GPU_COMPILE_TIMEOUT,
        "run_timeout_s": GPU_RUN_TIMEOUT,
    }
    for key, expected in protocol.items():
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            raise InfraError(f"invalid GPU timing receipt field {key}")
    l2_bytes = receipt.get("l2_flush_bytes")
    if type(l2_bytes) is not int or l2_bytes < GPU_L2_BYTES:
        raise InfraError("invalid GPU timing receipt field l2_flush_bytes")
    threshold = GPU_TOLERANCES.get(payload["dtype"])
    if threshold is None or any(receipt.get(key) != threshold for key in ("atol", "rtol")):
        raise InfraError("invalid GPU timing receipt dtype tolerance")
    if receipt.get("static_ast_pass") is not True:
        raise InfraError("invalid GPU timing receipt field static_ast_pass")
    for key, size in (("compile_stderr", COMPILE_STDERR_BYTES), ("timing_summary", TIMING_SUMMARY_BYTES)):
        value = receipt.get(key)
        if not isinstance(value, str) or value != tail_bytes(value, size):
            raise InfraError(f"invalid GPU timing receipt field {key}")
    _validate_hardware(receipt.get("hardware"), payload["device_uuid"], cuda)
    correct = _validate_correctness(receipt, cuda)
    baseline = _validate_samples(receipt, "baseline", correct)
    candidate = _validate_samples(receipt, "candidate", correct)
    ratio = baseline / candidate if correct else 0.0
    speedup = receipt.get("speedup")
    if not finite_number(speedup) or speedup < 0 or not math.isclose(speedup, ratio, rel_tol=1e-9):
        raise InfraError("invalid GPU timing receipt speedup ratio")
    # Derive boundary-sensitive flags from the measurements.
    for key, expected in kernel_fast(correct, ratio).items():
        if receipt.get(key) is not expected:
            raise InfraError(f"invalid GPU timing receipt field {key}")
    score = receipt.get("score")
    if not finite_number(score) or not math.isclose(score, kernel_score(correct, ratio), rel_tol=1e-9):
        raise InfraError("invalid GPU timing receipt field score")
    return receipt


def timing_receipt(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    hardware: dict[str, Any],
    compiled: Any,
    result: dict[str, Any] | None,
    failure: str | None,
    compile_stderr: str,
    runtime_stderr: str,
) -> dict[str, Any]:
    trials = result["correctness_trials"] if result else []
    correct = len(trials) == GPU_CORRECT_TRIALS and all(trials)
    samples = {key: result[key] if result else [] for key in ("baseline_samples_ms", "candidate_samples_ms")}
    baseline = fmean(samples["baseline_samples_ms"]) if correct else None
    candidate = fmean(samples["candidate_samples_ms"]) if correct else None
    speedup = baseline / candidate if correct else 0.0
    failure = failure or (None if correct else "incorrect")
    details = {"correct_trials": sum(trials), "baseline_ms": baseline, "candidate_ms": candidate, "speedup": speedup, "failure": failure}
    for limit in (3000, 1000, 512, 0):
        details["diagnostic"] = tail_bytes(runtime_stderr, limit) if limit else ""
        summary = json.dumps(details, ensure_ascii=False, allow_nan=False)
        if len(summary.encode()) <= TIMING_SUMMARY_BYTES:
            break
    return {
        **{field: payload[field] for field in GPU_BINDINGS},
        "backend": config["backend"],
        "hardware": hardware,
        "static_ast_pass": True,
        "l2_flushed": config["backend"] == "cuda" and correct,
        "num_correct_trials": GPU_CORRECT_TRIALS,
        "num_perf_trials": GPU_PERF_TRIALS,
        "warmup": GPU_WARMUP,
        "l2_flush_bytes": config["l2_flush_bytes"],
        "compile_timeout_s": GPU_COMPILE_TIMEOUT,
        "run_timeout_s": GPU_RUN_TIMEOUT,
        "atol": config["tolerance"],
        "rtol": config["tolerance"],
        "compile_stderr": compile_stderr,
        "timing_summary": summary,
        "correctness_seeds": config["seeds"],
        "correctness_trials": trials,
        "correct": correct,
        "compiled": bool(compiled),
        "failure": failure,
        **samples,
        "baseline_ms": baseline,
        "candidate_ms": candidate,
        "speedup": speedup,
        **kernel_fast(correct, speedup),
        "score": kernel_score(correct, speedup),
    }


class GPUTimingClient:
    """Submit an active task and verify the complete worker receipt."""

    def __init__(
        self,
        endpoint: str,
        device_uuid: str,
        task_ids: Iterable[str],
        *,
        token: str | None = None,
        dtype: str = "fp32",
        require_real_cuda: bool = True,
    ) -> None:
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", device_uuid):
            raise ValueError("physical UUID required")
        if dtype not in GPU_TOLERANCES:
            raise ValueError("GPU dtype must be fp32, fp16 or bf16")
        self.endpoint, self.device_uuid, self.token = endpoint, device_uuid, token
        self.dtype, self.require_real_cuda = dtype, require_real_cuda
        self.task_ids = frozenset(task_ids)

    def submit(self, *, task_id: str, snapshot_id: str, source: str, job_id: str) -> dict[str, Any]:
        if task_id not in self.task_ids:
            raise PermissionError("kernel task outside active split")
        if kernel_guard(source):
            raise CandidateExecutionError("kernel failed CPU AST preflight")
        if self.require_real_cuda and not kernel_has_triton_launch(ast.parse(source)):
            raise CandidateExecutionError("kernel must launch a custom Triton kernel")
        payload = {
            "task_id": task_id,
            "snapshot_id": snapshot_id,
            "job_id": job_id,
            "source": source,
            "candidate_sha256": sha(source.encode()),
            "device_uuid": self.device_uuid,
            "lease_mode": "exclusive",
            "timing_protocol": "locked-clock-flush-l2",
            "dtype": self.dtype,
        }
        receipt = worker_rpc(self.endpoint, "/v1/timing-jobs", payload, GPU_RPC_TIMEOUT, self.token)
        return validate_kernel_receipt(receipt, payload, require_real_cuda=self.require_real_cuda)


async def kernel(agent: Agent, benchmark: KernelBench, task: Task) -> Verdict:
    endpoint = benchmark.config.get("gpu_queue") or os.getenv("FLASHRSI_GPU_QUEUE")
    device = benchmark.config.get("device_uuid") or os.getenv("FLASHRSI_GPU_UUID")
    if not endpoint or not device:
        raise InfraError("no configured GPU timing queue and physical device UUID")
    budget = oracle_budget(benchmark.config, KERNEL_ORACLE_BUDGET, 100)
    queue = GPUTimingClient(
        endpoint,
        device,
        benchmark._tasks,
        token=worker_token(benchmark.config),
        dtype=benchmark.config.get("dtype", "fp32"),
        require_real_cuda=benchmark.config.get("require_real_cuda") is not False,
    )
    best = 0.0
    parent = None
    receipts = []
    usage = []
    for _ in range(budget):
        feedback = []
        if receipts:
            prior = receipts[-1]
            feedback = [
                "Compile stderr tail:\n" + prior.get("compile_stderr", ""),
                "Timing/correctness summary:\n" + prior.get("timing_summary", ""),
                "Static preflight: " + ", ".join(prior.get("cpu_guard", [])),
            ]
        prompt = search_prompt(task.question, parent or "", f"Best scalar score: {best}", feedback)
        source, tokens = await propose_code(agent, task, prompt, json_hint=True)
        usage.append(tokens)
        try:
            errors = kernel_guard(source) if isinstance(source, str) else ["candidate must be Python text"]
        except SyntaxError:
            errors = ["invalid Python syntax"]
        if errors:
            receipts.append({"score": 0.0, "cpu_guard": errors})
            continue
        try:
            receipt = await asyncio.to_thread(
                queue.submit, task_id=task.task_id, snapshot_id=agent.snapshot.id, source=source, job_id=uuid.uuid4().hex
            )
        except CandidateExecutionError as exc:
            receipts.append({"score": 0.0, "cpu_guard": [str(exc)]})
            continue
        except PermissionError:
            raise
        except (OSError, ValueError) as exc:
            raise InfraError(f"GPU timing queue request failed ({type(exc).__name__})") from None
        score = receipt.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise InfraError("GPU timing receipt lacks a valid score")
        receipts.append({"device_uuid": device, **receipt})
        if receipt.get("correct", True) and (parent is None or score > best):
            parent, best = source, score
    return Trajectory(usage={"model_calls": usage}), {
        "score": best,
        "budget": budget,
        "gpu_calls": sum("job_id" in receipt for receipt in receipts),
        "trial_receipts": receipts,
    }
