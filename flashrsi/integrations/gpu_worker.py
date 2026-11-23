"""Execute and supervise single-device KernelBench worker phases.

The child entry point runs as a standalone file with a stripped environment.
It loads PyTorch only for CUDA trials. The RPC process owns credentials and
selects physical device visibility through the caller's container command.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import fcntl
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

SAFE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_cpu(expected: Any, actual: Any, tolerance: float) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, (tuple, list)):
        return len(expected) == len(actual) and all(compare_cpu(a, b, tolerance) for a, b in zip(expected, actual))
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(compare_cpu(expected[key], actual[key], tolerance) for key in expected)
    if type(expected) in (int, float):
        return math.isfinite(expected) and math.isfinite(actual) and abs(expected - actual) <= (tolerance + tolerance * abs(expected))
    return expected == actual


def compare_cuda(torch: ModuleType, expected: Any, actual: Any, tolerance: float) -> bool:
    if isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and actual.device == expected.device
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and bool(torch.isfinite(actual).all())
            and bool(torch.isfinite(expected).all())
            and torch.allclose(expected, actual, atol=tolerance, rtol=tolerance)
        )
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, (tuple, list)):
        return len(expected) == len(actual) and all(compare_cuda(torch, a, b, tolerance) for a, b in zip(expected, actual))
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(compare_cuda(torch, expected[key], actual[key], tolerance) for key in expected)
    return compare_cpu(expected, actual, tolerance)


def transform_tensors(torch: ModuleType, value: Any, tensor_fn: Callable[[Any], Any]) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    if isinstance(value, tuple):
        return tuple(transform_tensors(torch, item, tensor_fn) for item in value)
    if isinstance(value, list):
        return [transform_tensors(torch, item, tensor_fn) for item in value]
    if isinstance(value, dict):
        return {key: transform_tensors(torch, item, tensor_fn) for key, item in value.items()}
    return value


class _KernelTrials:
    def __init__(self, reference: ModuleType, candidate: ModuleType, torch: ModuleType | None, dtype: Any) -> None:
        self.reference = reference
        self.candidate = candidate
        self.torch = torch
        self.dtype = dtype

    def seed(self, seed: int) -> None:
        random.seed(seed)
        if self.torch is not None:
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)

    def inputs(self, seed: int) -> Any:
        self.seed(seed)
        data = self.reference.get_inputs()
        if self.torch is None:
            return data
        return transform_tensors(self.torch, data, _DeviceTensor(self.dtype))

    def clone(self, value: Any) -> Any:
        return copy.deepcopy(value) if self.torch is None else transform_tensors(self.torch, value, _clone_tensor)

    def models(self, seed: int) -> tuple[Callable[..., Any], Callable[..., Any]]:
        self.seed(seed)
        inputs = self.reference.get_init_inputs()
        self.seed(seed)
        baseline = self.reference.Model(*self.clone(inputs))
        self.seed(seed)
        optimized = self.candidate.ModelNew(*self.clone(inputs))
        if self.torch is None:
            return baseline, optimized.forward
        baseline = baseline.to(device="cuda", dtype=self.dtype).eval()
        optimized = optimized.to(device="cuda", dtype=self.dtype).eval()
        # Compatible state entries share the same reference weights.
        original, proposed = baseline.state_dict(), optimized.state_dict()
        matching = {key: val for key, val in original.items() if key in proposed and val.shape == proposed[key].shape}
        optimized.load_state_dict(matching, strict=False)
        return baseline, optimized

    def correctness(self, config: Mapping[str, Any]) -> list[bool]:
        baseline, optimized = self.models(config["seeds"][0])
        outcomes = []
        for seed in config["seeds"]:
            inputs = self.inputs(seed)
            baseline_inputs, candidate_inputs = self.clone(inputs), self.clone(inputs)
            expected = baseline(*baseline_inputs)
            self.seed(seed)
            actual = optimized(*candidate_inputs)
            tolerance = config["tolerance"]
            if self.torch is None:
                matched = compare_cpu(expected, actual, tolerance) and compare_cpu(baseline_inputs, candidate_inputs, tolerance)
            else:
                matched = compare_cuda(self.torch, expected, actual, tolerance) and compare_cuda(
                    self.torch, baseline_inputs, candidate_inputs, tolerance
                )
            outcomes.append(bool(matched))
        return outcomes


class _DeviceTensor:
    def __init__(self, dtype: Any) -> None:
        self.dtype = dtype

    def __call__(self, tensor: Any) -> Any:
        return tensor.to(device="cuda", dtype=self.dtype if tensor.is_floating_point() else tensor.dtype)


def _clone_tensor(tensor: Any) -> Any:
    return tensor.detach().clone()


class _CudaTimer:
    def __init__(self, torch: ModuleType, flush_bytes: int) -> None:
        self.cuda = torch.cuda
        self.l2 = torch.empty(flush_bytes, dtype=torch.uint8, device="cuda")
        self.start = self.cuda.Event(enable_timing=True)
        self.end = self.cuda.Event(enable_timing=True)

    def __call__(self, model: Callable[..., Any], arguments: Any, index: int) -> float:
        self.cuda.synchronize()
        self.l2.fill_(index % 255 + 1)
        self.cuda.synchronize()
        self.start.record()
        output = model(*arguments)
        self.end.record()
        self.cuda.synchronize()
        del output
        return self.start.elapsed_time(self.end)


def _measure_cpu(model: Callable[..., Any], arguments: Any, _index: int) -> float:
    start = time.perf_counter_ns()
    output = model(*arguments)
    elapsed = max(1, time.perf_counter_ns() - start) / 1_000_000
    del output
    return elapsed


def _timing(trials: _KernelTrials, config: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    seed = config["seeds"][0]
    baseline, optimized = trials.models(seed)
    inputs = trials.inputs(seed)
    arguments = trials.clone(inputs), trials.clone(inputs)
    models = baseline, optimized
    samples: tuple[list[float], list[float]] = ([], [])
    measure = _CudaTimer(trials.torch, config["l2_flush_bytes"]) if trials.torch is not None else _measure_cpu
    for _ in range(config["warmup"]):
        baseline(*arguments[0])
        optimized(*arguments[1])
    for index in range(config["perf_trials"]):
        # Alternate execution order to reduce device-temperature bias.
        order = (1, 0) if index % 2 else (0, 1)
        for selected in order:
            samples[selected].append(measure(models[selected], arguments[selected], index))
    return samples


def _cuda_backend(dtype: str) -> tuple[ModuleType, Any]:
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("worker must see exactly one CUDA device")
        torch.cuda.set_device(0)
        precision = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    except (ImportError, OSError, RuntimeError):
        raise SystemExit(78) from None
    return torch, precision


def run_child(phase: str, reference: Path, candidate: Path, config: Mapping[str, Any]) -> None:
    torch, dtype = _cuda_backend(config["dtype"]) if config["backend"] == "cuda" else (None, None)
    # Candidate and reference stdout stay outside the child result protocol.
    with contextlib.redirect_stdout(sys.stderr), torch.no_grad() if torch is not None else contextlib.nullcontext():
        for path in (reference, candidate):
            compile(path.read_bytes(), str(path), "exec")
        original = load_module("kernelbench_reference", reference)
        proposed = load_module("kernelbench_candidate", candidate)
        trials = _KernelTrials(original, proposed, torch, dtype)
        if phase == "compile":
            seed = config["seeds"][0]
            _, optimized = trials.models(seed)
            optimized(*trials.inputs(seed))
            if torch is not None:
                torch.cuda.synchronize()
            result = {"compiled": True}
        else:
            outcomes = trials.correctness(config)
            baseline, candidate_samples = _timing(trials, config) if all(outcomes) else ([], [])
            result = {
                "correctness_trials": outcomes,
                "baseline_samples_ms": baseline,
                "candidate_samples_ms": candidate_samples,
                "cuda_version": str(torch.version.cuda) if torch else None,
            }
    print(json.dumps(result, allow_nan=False))


class ProtocolError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def smi(device: str, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", device, *arguments], env=SAFE_ENV, text=True, capture_output=True, timeout=15, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise ProtocolError(503, "NVIDIA device or clock query failed") from None


def hardware_info(device: str) -> dict[str, str]:
    fields = ("uuid", "name", "driver_version", "clocks.gr", "clocks.mem", "temperature.gpu")
    raw = smi(device, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits")
    rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(fields) or not rows[0][0].startswith("GPU-"):
        raise ProtocolError(503, "NVIDIA worker requires one physical GPU UUID")
    return dict(zip(("device_uuid", "gpu_name", "driver_version", "graphics_clock_mhz", "memory_clock_mhz", "temperature_c"), rows[0]))


@contextlib.contextmanager
def gpu_lease(path: Path, timeout: float) -> Iterator[None]:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ProtocolError(503, "worker GPU queue wait exceeded its deadline") from None
                time.sleep(0.05)
        yield


def run_phase(
    command: Sequence[str], env: Mapping[str, str], deadline: float, phase: str, diagnostic_bytes: int
) -> tuple[dict[str, Any] | None, str, str | None]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            status = process.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            status = None
        stderr.seek(max(0, stderr.tell() - diagnostic_bytes))
        data = stderr.read().decode(errors="replace").encode()
        diagnostic = data[-diagnostic_bytes:].decode(errors="ignore")
        if status in (78, 125, 126, 127, 137):
            raise ProtocolError(503, f"GPU worker execution infrastructure exit {status}")
        if status is None:
            return None, diagnostic, phase + "_timeout"
        if status != 0:
            return None, diagnostic, phase + "_error"
        stdout.seek(0)
        raw = stdout.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ProtocolError(503, "GPU subprocess exceeded its result limit")
    try:
        result = json.loads(raw)
    except ValueError:
        raise ProtocolError(503, "GPU subprocess returned malformed data") from None
    if not isinstance(result, dict):
        raise ProtocolError(503, "GPU subprocess returned malformed data")
    return result, diagnostic, None


if __name__ == "__main__":
    phase, reference_file, candidate_file, config_json = sys.argv[1:]
    run_child(phase, Path(reference_file), Path(candidate_file), json.loads(config_json))
